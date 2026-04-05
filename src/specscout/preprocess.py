"""
Preprocessing transforms and pipeline utilities for specscout.

This module defines small, composable transforms that operate on extracted
time-frequency arrays and return transformed arrays suitable for downstream
analysis, plotting, or detection.

Transform composition is performed via `PreprocessPipeline`, which provides:

- ordered execution of transforms
- pipeline metadata tracking
- optional DataSpace validation
- lightweight semantic tracking of the transformed data product

Current responsibilities
------------------------
- Basic intensity transforms such as linear -> dB via `safe_db`
- Polarization transforms:
  - dual-pol autocorrelations -> Stokes I
  - linear products -> full Stokes (I, Q, U, V)
- `PreprocessPipeline`: a lightweight, introspectable transform chain with
  optional DataSpace validation and JSON-serializable step metadata
- `DataDesc`: a minimal semantic description of the current data product,
  including channel names and data space

Design notes
------------
All transforms in this module follow the standard specscout callable
interface:

    transform(frame: np.ndarray, meta: FrameMeta) -> np.ndarray

where `frame` is typically an extracted `(T, F)` or `(T, F, C)` block and
`meta` provides contextual metadata about the extracted interval.

The pipeline itself tracks a lightweight `DataDesc` object. This allows
specscout to preserve information such as channel names across transforms:

- raw `(pol00, pol11)` -> `stokes_I`
- raw `(pol00, pol11, pol01_mag, pol01_phase)` -> full Stokes
- linear -> dB

This metadata is useful for plotting, provenance, and validation, while
keeping the transform API itself simple and backward-compatible.

Notes
-----
- Pipelines are treated as immutable: methods such as `add()`,
  `with_metadata()`, and `with_input_desc()` return new pipeline objects.
- Pipeline serialization stores step configs, metadata, and data descriptors,
  but not executable Python callables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .dataset import FrameMeta

import warnings

Transform = Callable[[np.ndarray, "FrameMeta"], np.ndarray]
DataSpace = Literal["linear", "db"]


_DB_UNSAFE_CHANNEL_KEYS = (
    "phase",
    "stokes_q",
    "stokes_u",
    "stokes_v",
)


# -----------------------------------------------------------------------------
# Lightweight semantic descriptor
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataDesc:
    """
    Minimal semantic description of a time-frequency product.

    Parameters
    ----------
    channel_names
        Optional channel / product names. Examples:
        - ``("pol00",)`` for a single raw product
        - ``("pol00", "pol11")`` for dual-pol autocorrelations
        - ``("stokes_I",)`` after Stokes-I conversion
        - ``("stokes_I", "stokes_Q", "stokes_U", "stokes_V")`` for full Stokes
    space
        Semantic data space, e.g. ``"linear"`` or ``"db"``.
    """

    channel_names: tuple[str, ...] | None = None
    space: DataSpace = "linear"

    @property
    def n_channels(self) -> int | None:
        """Number of named channels, or None if unknown."""
        return None if self.channel_names is None else len(self.channel_names)

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a JSON-compatible dict."""
        return {
            "channel_names": None if self.channel_names is None else list(self.channel_names),
            "space": self.space,
        }


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
    if floor <= 0:
        raise ValueError("floor must be positive.")

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

    Parameters
    ----------
    floor
        Floor applied before log10 to avoid log10(0).
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
    ``(pol00, pol11)`` into Stokes I.

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
    Expected input shape is ``(T, F, 2)`` ordered as ``(pol00, pol11)``.

    Convention::

        I = 0.5 * (pol00 + pol11)
    """
    out_dtype = np.dtype(dtype)

    def transform(frame_tfc: np.ndarray, _meta: FrameMeta) -> np.ndarray:
        x = np.asarray(frame_tfc)

        if x.ndim != 3 or x.shape[2] != 2:
            raise ValueError("Stokes I transform expects input with shape (T, F, 2) ordered as (pol00, pol11).")

        pol00 = x[:, :, 0]
        pol11 = x[:, :, 1]
        I = 0.5 * (pol00 + pol11)

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

        (pol00, pol11, pol01_mag, pol01_phase)

    Output shape is ``(T, F, 4)`` ordered as::

        (stokes_I, stokes_Q, stokes_U, stokes_V)

    Convention::

        I = 0.5 * (pol00 + pol11)
        Q = 0.5 * (pol00 - pol11)
        U = Re(pol01)
        V = Im(pol01)

    where::

        pol01 = pol01_mag * exp(1j * pol01_phase)
    """
    out_dtype = np.dtype(dtype)

    def transform(frame_tfc: np.ndarray, _meta: FrameMeta) -> np.ndarray:
        x = np.asarray(frame_tfc)

        if x.ndim != 3 or x.shape[2] != 4:
            raise ValueError(
                "Stokes IQUV transform expects input with shape (T, F, 4) ordered as (pol00, pol11, pol01_mag, pol01_phase)."
            )

        pol00 = x[:, :, 0]
        pol11 = x[:, :, 1]
        pol01_mag = x[:, :, 2]
        pol01_phase = x[:, :, 3]

        pol01 = pol01_mag * np.exp(1j * pol01_phase)

        I = 0.5 * (pol00 + pol11)
        Q = 0.5 * (pol00 - pol11)
        U = np.real(pol01)
        V = np.imag(pol01)

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
    - Lightweight semantic tracking via `DataDesc`
    - JSON-serializable configuration
    """

    def __init__(
        self,
        steps: Optional[Sequence[PipelineStep]] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
        input_space: DataSpace = "linear",
        input_desc: DataDesc | None = None,
    ) -> None:
        self._steps = list(steps) if steps is not None else []
        self._metadata = dict(metadata) if metadata is not None else {}

        if input_desc is None:
            input_desc = DataDesc(channel_names=None, space=input_space)
        else:
            if input_desc.space != input_space:
                raise ValueError(f"input_desc.space={input_desc.space!r} does not match input_space={input_space!r}.")

        self._input_desc = input_desc

    @property
    def steps(self) -> tuple[PipelineStep, ...]:
        """Immutable view of pipeline steps."""
        return tuple(self._steps)

    @property
    def metadata(self) -> dict[str, Any]:
        """Pipeline-level metadata dict (copied on access)."""
        return dict(self._metadata)

    @property
    def input_desc(self) -> DataDesc:
        """Input semantic descriptor."""
        return self._input_desc

    @property
    def output_desc(self) -> DataDesc:
        """Output semantic descriptor after all steps."""
        desc = self._input_desc
        for step in self._steps:
            desc = self._apply_step_desc(desc, step)
        return desc

    @property
    def input_space(self) -> DataSpace:
        """Declared semantic space of pipeline inputs."""
        return self._input_desc.space

    @property
    def output_space(self) -> DataSpace:
        """Declared semantic space of pipeline outputs."""
        return self.output_desc.space

    @property
    def input_channel_names(self) -> tuple[str, ...] | None:
        """Named input channels, if known."""
        return self._input_desc.channel_names

    @property
    def output_channel_names(self) -> tuple[str, ...] | None:
        """Named output channels, if known."""
        return self.output_desc.channel_names

    def _current_desc(self) -> DataDesc:
        if not self._steps:
            return self._input_desc
        return self.output_desc

    def _apply_step_desc(self, desc: DataDesc, step: PipelineStep) -> DataDesc:
        out_space = step.out_space if step.out_space is not None else desc.space

        out_names = step.config.get("channel_order_out", None)
        if out_names is not None:
            out_names_t = tuple(str(x) for x in out_names)
        else:
            out_names_t = desc.channel_names

        return DataDesc(channel_names=out_names_t, space=out_space)

    def _validate_step_against_desc(self, desc: DataDesc, step: PipelineStep) -> None:
        expected_space = step.in_space
        if expected_space is not None and expected_space != desc.space:
            raise ValueError(
                f"Cannot apply step {step.name!r}: in_space={expected_space!r} does not match current space {desc.space!r}."
            )

        expected_names = step.config.get("channel_order_in", None)
        if expected_names is not None and desc.channel_names is not None:
            expected_t = tuple(str(x) for x in expected_names)
            if desc.channel_names != expected_t:
                raise ValueError(
                    f"Cannot apply step {step.name!r}: expected channels {expected_t!r}, got {desc.channel_names!r}."
                )

    @staticmethod
    def _infer_array_n_channels(frame: np.ndarray) -> int:
        """
        Infer number of channels from an array shape.

        Returns
        -------
        int
            - 1 for ``(T, F)``
            - C for ``(T, F, C)``
        """
        x = np.asarray(frame)
        if x.ndim == 2:
            return 1
        if x.ndim == 3:
            return int(x.shape[2])
        raise ValueError(f"Expected frame with shape (T, F) or (T, F, C), got {x.shape}.")

    @classmethod
    def _validate_frame_against_desc(cls, frame: np.ndarray, desc: DataDesc) -> None:
        """
        Validate an array shape against a known channel description.
        """
        if desc.channel_names is None:
            return

        n_expected = len(desc.channel_names)
        n_found = cls._infer_array_n_channels(frame)
        if n_found != n_expected:
            raise ValueError(
                f"Frame shape is inconsistent with declared channel_names "
                f"{desc.channel_names!r}: expected {n_expected} channel(s), "
                f"found {n_found}."
            )

    def with_metadata(self, **metadata: Any) -> "PreprocessPipeline":
        """
        Return a new pipeline with updated pipeline-level metadata.
        """
        md = dict(self._metadata)
        md.update(metadata)
        return PreprocessPipeline(
            self._steps,
            metadata=md,
            input_space=self._input_desc.space,
            input_desc=self._input_desc,
        )

    def with_input_desc(self, desc: DataDesc) -> "PreprocessPipeline":
        """
        Return a new pipeline with an updated input descriptor.
        """
        current = self._input_desc
        if current.space != desc.space:
            raise ValueError(
                f"Input descriptor space {desc.space!r} does not match current pipeline input space {current.space!r}."
            )

        return PreprocessPipeline(
            self._steps,
            metadata=self._metadata,
            input_space=desc.space,
            input_desc=desc,
        )

    def with_input_channels(self, *channel_names: str) -> "PreprocessPipeline":
        """
        Return a new pipeline with named input channels.

        Examples
        --------
        >>> pipe = (
        ...     PreprocessPipeline(input_space="linear")
        ...     .with_input_channels("pol00", "pol11")
        ...     .add(step_stokes_i())
        ...     .add(step_safe_db())
        ... )
        """
        desc = replace(self._input_desc, channel_names=tuple(str(x) for x in channel_names))
        return self.with_input_desc(desc)

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
        """
        Return a new pipeline with one step appended.
        """
        if not name:
            raise ValueError("step name must be non-empty")

        config = {} if config is None else dict(config)

        step = PipelineStep(
            name=name,
            transform=transform,
            config=config,
            in_space=in_space,
            out_space=out_space,
        )

        if validate_spaces:
            self._validate_step_against_desc(self._current_desc(), step)

        return PreprocessPipeline(
            [*self._steps, step],
            metadata=self._metadata,
            input_space=self._input_desc.space,
            input_desc=self._input_desc,
        )

    def add(self, step: PipelineStep, *, validate_spaces: bool = True) -> "PreprocessPipeline":
        """Convenience wrapper to add a pre-built `PipelineStep`."""
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
        """Append multiple steps, returning a new pipeline."""
        pipe = self
        for s in steps:
            pipe = pipe.add(s, validate_spaces=validate_spaces)
        return pipe

    def _maybe_warn_for_step(
        self,
        step: PipelineStep,
        desc: DataDesc,
    ) -> None:
        if step.name != "safe_db":
            return

        if desc.channel_names is None:
            return

        bad = []
        for name in desc.channel_names:
            name_l = str(name).lower()
            if "phase" in name_l or name_l == "stokes_q" or name_l == "stokes_u" or name_l == "stokes_v":
                bad.append(str(name))

        if bad:
            warnings.warn(
                f"safe_db is being applied to non power-like channels {tuple(bad)}; this is usually not physically meaningful.",
                RuntimeWarning,
                stacklevel=3,
            )

    def __call__(self, frame: np.ndarray, meta: FrameMeta) -> np.ndarray:
        """
        Apply all steps in order.

        Notes
        -----
        The pipeline tracks the evolving `DataDesc` internally while applying
        transforms. The returned object is still just the transformed array,
        preserving the existing specscout transform API.
        """
        out = np.asarray(frame)
        desc = self._input_desc

        self._validate_frame_against_desc(out, desc)

        for step in self._steps:
            self._validate_step_against_desc(desc, step)
            self._maybe_warn_for_step(step, desc)
            out = step.transform(out, meta)
            desc = self._apply_step_desc(desc, step)
            self._validate_frame_against_desc(out, desc)

        return out

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the pipeline."""
        lines: list[str] = []
        lines.append("########################")
        lines.append("#  PreprocessPipeline  #")
        lines.append("########################\n")
        lines.append(f"input_space:    {self.input_space}")
        lines.append(f"output_space:   {self.output_space}")
        lines.append(f"input_channels: {self.input_channel_names}")
        lines.append(f"output_channels:{self.output_channel_names}\n")

        if self._metadata:
            lines.append(f"metadata: {self._metadata}")

        lines.append(f"n_steps: {len(self._steps)}")

        for i, s in enumerate(self._steps):
            cin = s.config.get("channel_order_in", "?")
            cout = s.config.get("channel_order_out", "?")
            lines.append(
                f"[{i}] {s.name} "
                f"in_space={s.in_space or '?'} "
                f"out_space={s.out_space or '?'} "
                f"in_channels={cin} "
                f"out_channels={cout} "
                f"config={s.config}"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline configuration to a JSON-compatible dict."""
        return {
            "metadata": dict(self._metadata),
            "input_desc": self._input_desc.to_dict(),
            "output_desc": self.output_desc.to_dict(),
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
        """Save `to_dict()` as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n")
        return out


# -----------------------------------------------------------------------------
# Step builders
# -----------------------------------------------------------------------------


def step_safe_db(
    *,
    floor: float = 1e-12,
    dtype: np.dtype = np.float32,
    name: str = "safe_db",
    in_space: DataSpace = "linear",
    out_space: DataSpace = "db",
) -> PipelineStep:
    """
    Build a PipelineStep that converts linear-valued frames to dB via `safe_db`.

    Notes
    -----
    This step is intended for non-negative, power-like quantities. If the
    pipeline input descriptor indicates channels such as ``pol01_phase`` or
    signed Stokes products (``stokes_Q``, ``stokes_U``, ``stokes_V``), the
    pipeline should emit a warning when this step is applied.
    """
    t = make_safe_db_transform(floor=floor, dtype=dtype)
    cfg = {
        "floor": float(floor),
        "dtype": str(np.dtype(dtype)),
    }
    return PipelineStep(
        name=name,
        transform=t,
        config=cfg,
        in_space=in_space,
        out_space=out_space,
    )


def step_stokes_i(
    *,
    dtype: np.dtype = np.float32,
    name: str = "stokes_i",
    in_space: DataSpace = "linear",
    out_space: DataSpace = "linear",
) -> PipelineStep:
    """
    Build a PipelineStep that converts ``(pol00, pol11)`` into ``stokes_I``.
    """
    t = make_stokes_i_transform(dtype=dtype)
    cfg = {
        "dtype": str(np.dtype(dtype)),
        "channel_order_in": ["pol00", "pol11"],
        "channel_order_out": ["stokes_I"],
        "convention": "stokes_I = 0.5 * (pol00 + pol11)",
    }
    return PipelineStep(
        name=name,
        transform=t,
        config=cfg,
        in_space=in_space,
        out_space=out_space,
    )


def step_stokes_iquv(
    *,
    dtype: np.dtype = np.float32,
    name: str = "stokes_iquv",
    in_space: DataSpace = "linear",
    out_space: DataSpace = "linear",
) -> PipelineStep:
    """
    Build a PipelineStep that converts raw ALBATROS linear products into
    full Stokes ``(stokes_I, stokes_Q, stokes_U, stokes_V)``.
    """
    t = make_stokes_iquv_transform(dtype=dtype)
    cfg = {
        "dtype": str(np.dtype(dtype)),
        "channel_order_in": ["pol00", "pol11", "pol01_mag", "pol01_phase"],
        "channel_order_out": [
            "stokes_I",
            "stokes_Q",
            "stokes_U",
            "stokes_V",
        ],
        "convention": {
            "stokes_I": "0.5 * (pol00 + pol11)",
            "stokes_Q": "0.5 * (pol00 - pol11)",
            "stokes_U": "real(pol01_mag * exp(1j * pol01_phase))",
            "stokes_V": "imag(pol01_mag * exp(1j * pol01_phase))",
        },
    }
    return PipelineStep(
        name=name,
        transform=t,
        config=cfg,
        in_space=in_space,
        out_space=out_space,
    )
