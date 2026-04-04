"""
Preprocessing transforms and pipeline utilities for specscout.

This module defines small, composable transforms that operate on extracted
time-frequency arrays and return transformed arrays suitable for downstream
analysis, plotting, or detection.

Transform composition is performed via `PreprocessPipeline`, which provides
ordered execution, metadata tracking, and optional DataSpace validation.

Current responsibilities
------------------------
- Basic intensity transforms such as linear -> dB via `safe_db`
- Polarization transforms:
  - dual-pol autocorrelations -> Stokes I
  - linear products -> full Stokes (I, Q, U, V)
- `PreprocessPipeline`: a lightweight, introspectable transform chain with
  optional DataSpace tracking and JSON-serializable step metadata

Design notes
------------
All transforms in this module follow the standard specscout callable
interface:

    transform(frame: np.ndarray, meta: FrameMeta) -> np.ndarray

where `frame` is typically an extracted `(T, F)` or `(T, F, C)` block and
`meta` provides contextual metadata about the extracted interval.

This module intentionally focuses on small, reusable transform primitives.
Background estimation, PCA modeling, and ROI detection live elsewhere in the
package (`outlier.py`, `rolling.py`, `roi_search.py`).

Notes
-----
- Pipelines are treated as immutable: methods such as `add()` and
  `with_metadata()` return new pipeline objects.
- Pipeline serialization stores step configs and metadata, but not executable
  Python callables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

import numpy as np

from .dataset import FrameMeta

Transform = Callable[[np.ndarray, FrameMeta], np.ndarray]
DataSpace = Literal["linear", "db"]


# -----------------------------------------------------------------------------
# Basic transforms
# -----------------------------------------------------------------------------


def safe_db(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """
    Convert linear power-like values to dB safely: ``10 * log10(x)``.

    This helper is designed for arrays that may contain NaNs and zeros.

    Parameters
    ----------
    x
        Input array. NaNs are preserved. Finite values are floored to `floor`
        before applying log10, to avoid ``-inf`` from zeros.
    floor
        Minimum finite value used before log conversion.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as `x`, dtype float (numpy chooses),
        containing dB values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        x2 = np.array(x, copy=True)
        finite = np.isfinite(x2)
        x2[finite] = np.maximum(x2[finite], floor)
        return 10.0 * np.log10(x2)


def make_safe_db_transform(
    *,
    floor: float = 1e-12,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a transform that converts linear-valued frames to dB using `safe_db`.

    This is useful as an explicit preprocessing step when you want:
    - dB-valued frames with no further transformation
    - clean DataSpace tracking ("linear" -> "db")
    - consistent NaN behavior (NaNs remain NaNs)

    Parameters
    ----------
    floor
        Floor applied before log10 to avoid log10(0). Passed to `core.safe_db`.
    dtype
        Output dtype.

    Returns
    -------
    Transform
        Callable with signature ``transform(frame, meta) -> frame_db``.
    """
    if floor <= 0:
        raise ValueError("floor must be positive.")

    out_dtype = np.dtype(dtype)

    def transform(frame: np.ndarray, _meta: FrameMeta) -> np.ndarray:
        db = safe_db(frame, floor=floor)
        return np.asarray(db, dtype=out_dtype)

    return transform


def make_stokes_i_transform(
    *,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a transform that converts dual-linear autocorrelation inputs
    ``(XX, YY)`` into Stokes I.

    Parameters
    ----------
    dtype
        Output dtype.

    Returns
    -------
    Transform
        Callable with signature ``transform(frame_tfc, meta) -> frame_tf``.

    Notes
    -----
    Expected input shape is ``(T, F, 2)`` ordered as ``(XX, YY)``.

    Convention::

        I = 0.5 * (XX + YY)
    """
    out_dtype = np.dtype(dtype)

    def transform(frame_tfc: np.ndarray, _meta: FrameMeta) -> np.ndarray:
        x = np.asarray(frame_tfc)

        if x.ndim != 3 or x.shape[2] != 2:
            raise ValueError("Stokes I transform expects input with shape (T, F, 2) ordered as (XX, YY).")

        xx = x[:, :, 0]
        yy = x[:, :, 1]
        I = 0.5 * (xx + yy)

        return np.asarray(I, dtype=out_dtype)

    return transform


def make_stokes_iquv_transform(
    *,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a transform that converts linear polarization products into
    full Stokes ``(I, Q, U, V)``.

    Parameters
    ----------
    dtype
        Output dtype.

    Returns
    -------
    Transform
        Callable with signature ``transform(frame_tfc, meta) -> frame_tfc_out``.

    Notes
    -----
    Expected input shape is ``(T, F, 4)`` ordered as::

        (XX, YY, XY_mag, XY_phase)

    Output shape is ``(T, F, 4)`` ordered as::

        (I, Q, U, V)

    Convention::

        I = 0.5 * (XX + YY)
        Q = 0.5 * (XX - YY)
        U = Re(XY)
        V = Im(XY)

    where::

        XY = XY_mag * exp(1j * XY_phase)
    """
    out_dtype = np.dtype(dtype)

    def transform(frame_tfc: np.ndarray, _meta: FrameMeta) -> np.ndarray:
        x = np.asarray(frame_tfc)

        if x.ndim != 3 or x.shape[2] != 4:
            raise ValueError("Stokes IQUV transform expects input with shape (T, F, 4) ordered as (XX, YY, XY_mag, XY_phase).")

        xx = x[:, :, 0]
        yy = x[:, :, 1]
        xy_mag = x[:, :, 2]
        xy_phase = x[:, :, 3]

        xy = xy_mag * np.exp(1j * xy_phase)

        I = 0.5 * (xx + yy)
        Q = 0.5 * (xx - yy)
        U = np.real(xy)
        V = np.imag(xy)

        out = np.stack([I, Q, U, V], axis=2)
        return np.asarray(out, dtype=out_dtype)

    return transform


# -----------------------------------------------------------------------------
# Pipeline abstraction
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStep:
    """
    A single preprocessing step in a `PreprocessPipeline`.

    A step couples:
    - an executable transform (callable)
    - a JSON-serializable config for reproducibility
    - optional DataSpace constraints for chaining validation

    Parameters
    ----------
    name
        Human-readable identifier for the step.
    transform
        Callable with signature ``transform(frame, meta) -> frame``.
    config
        JSON-serializable configuration for the step.
    in_space
        Expected DataSpace of the input.
    out_space
        DataSpace produced by this step.
    """

    name: str
    transform: Transform
    config: dict[str, Any]
    in_space: Optional[DataSpace] = None
    out_space: Optional[DataSpace] = None


class PreprocessPipeline:
    """
    Lightweight, immutable preprocessing pipeline.

    Provides:
    - Ordered execution of transforms
    - Optional DataSpace validation
    - JSON-serializable configuration
    """

    def __init__(
        self,
        steps: Optional[Sequence[PipelineStep]] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
        input_space: DataSpace = "linear",
    ) -> None:
        self._steps = list(steps) if steps is not None else []
        self._metadata = dict(metadata) if metadata is not None else {}
        self._input_space = input_space

    @property
    def steps(self) -> tuple[PipelineStep, ...]:
        return tuple(self._steps)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def input_space(self) -> DataSpace:
        return self._input_space

    @property
    def output_space(self) -> DataSpace:
        if not self._steps:
            return self._input_space
        last = self._steps[-1]
        return last.out_space if last.out_space is not None else self._input_space

    def _current_space(self) -> DataSpace:
        if not self._steps:
            return self._input_space
        s = self._steps[-1].out_space
        return s if s is not None else self._input_space

    def add_step(
        self,
        name: str,
        transform: Transform,
        config: Optional[dict[str, Any]] = None,
        *,
        in_space: Optional[DataSpace] = None,
        out_space: Optional[DataSpace] = None,
        validate_spaces: bool = True,
    ) -> "PreprocessPipeline":
        if not name:
            raise ValueError("step name must be non-empty")

        config = {} if config is None else config

        if validate_spaces and in_space is not None:
            current = self._current_space()
            if in_space != current:
                raise ValueError(f"Cannot add step {name!r}: in_space={in_space!r} != current space {current!r}")

        step = PipelineStep(
            name=name,
            transform=transform,
            config=dict(config),
            in_space=in_space,
            out_space=out_space,
        )

        return PreprocessPipeline(
            [*self._steps, step],
            metadata=self._metadata,
            input_space=self._input_space,
        )

    def add(self, step: PipelineStep, *, validate_spaces: bool = True) -> "PreprocessPipeline":
        return self.add_step(
            step.name,
            step.transform,
            step.config,
            in_space=step.in_space,
            out_space=step.out_space,
            validate_spaces=validate_spaces,
        )

    def extend(
        self,
        steps: Iterable[PipelineStep],
        *,
        validate_spaces: bool = True,
    ) -> "PreprocessPipeline":
        pipe = self
        for s in steps:
            pipe = pipe.add(s, validate_spaces=validate_spaces)
        return pipe

    def __call__(self, frame: np.ndarray, meta: FrameMeta) -> np.ndarray:
        out = frame
        for step in self._steps:
            out = step.transform(out, meta)
        return out

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("########################")
        lines.append("#  PreprocessPipeline  #")
        lines.append("########################\n")
        lines.append(f"input_space:  {self._input_space}")
        lines.append(f"output_space: {self.output_space}\n")

        if self._metadata:
            lines.append(f"metadata: {self._metadata}")

        lines.append(f"n_steps: {len(self._steps)}")

        for i, s in enumerate(self._steps):
            lines.append(f"[{i}] {s.name} in={s.in_space or '?'} out={s.out_space or '?'} config={s.config}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self._metadata),
            "input_space": self._input_space,
            "steps": [
                {
                    "name": s.name,
                    "config": dict(s.config),
                    "in_space": s.in_space,
                    "out_space": s.out_space,
                }
                for s in self._steps
            ],
        }

    def save_json(self, path: str | Path, *, indent: int = 2) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n")
        return out


# -----------------------------------------------------------------------------
# Step builders
# -----------------------------------------------------------------------------


def step_safe_db(**kwargs) -> PipelineStep:
    return PipelineStep(
        name="safe_db",
        transform=make_safe_db_transform(**kwargs),
        config={"floor": kwargs.get("floor", 1e-12)},
        in_space="linear",
        out_space="db",
    )


def step_stokes_i(**kwargs) -> PipelineStep:
    return PipelineStep(
        name="stokes_i",
        transform=make_stokes_i_transform(**kwargs),
        config={},
        in_space="linear",
        out_space="linear",
    )


def step_stokes_iquv(**kwargs) -> PipelineStep:
    return PipelineStep(
        name="stokes_iquv",
        transform=make_stokes_iquv_transform(**kwargs),
        config={},
        in_space="linear",
        out_space="linear",
    )
