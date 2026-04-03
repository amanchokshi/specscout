"""
Preprocessing transforms for specscout time–frequency data.

This module defines *pure(ish)* transformations that operate on extracted frames
(time–frequency arrays) and return transformed arrays suitable for inspection
or machine learning.

Core responsibilities
---------------------
- Defining "context" windows (longer time blocks) used to estimate statistics.
- Bandpass subtraction (per-frequency median removal).
- Robust whitening (per-frequency scaling using MAD).
- Optional soft clipping/compression for display or ML stability.
- Small transform-composition utilities.
- PreprocessPipeline: a named, introspectable transform chain with DataSpace tracking.

Design notes
------------
Many transforms need to estimate statistics over a wider window than the frame
being displayed/consumed (e.g., a 20-minute frame but a 4-hour baseline window).
To support this without loading large cubes into memory, transforms may read
only the required context slice from the Zarr array each time they are called.

All transforms in this module follow the signature used by the scrubber:

    transform(frame_tf: np.ndarray, meta: FrameMeta) -> np.ndarray
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

import numpy as np
import zarr

from .core import FrameMeta, clamp_int, safe_db, seconds_to_samples
from .patches import open_cube

Transform = Callable[[np.ndarray, FrameMeta], np.ndarray]
DataSpace = Literal["linear", "db", "z", "compressed"]


@dataclass(frozen=True)
class ContextSpec:
    """
    Specification for a time context window used to estimate statistics.

    Parameters
    ----------
    baseline_seconds
        Total duration of the context window in seconds. This is converted to a
        number of samples using the cube cadence `dt_s`.
    mode
        Placement of the context window relative to the frame start index:

        - ``"center"``: center the context on the frame start index.
        - ``"donut"``: like "center" but excludes a gap around the frame.
          This is useful when you want the baseline to be insensitive to a
          transient event in the frame.

    gap_seconds
        Only used when ``mode="donut"``. Size of the excluded gap around the
        frame window, in seconds. If None, defaults to the frame length
        (i.e., exclude exactly the frame window).
    """

    baseline_seconds: float
    mode: str = "center"
    gap_seconds: Optional[float] = None


def read_context_tf(
    cube: zarr.Array,
    *,
    chan: int,
    t_start_idx: int,
    window_n: int,
    dt_s: float,
    baseline_seconds: float,
    mode: str = "center",
    gap_seconds: Optional[float] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Read a time–frequency context block from a Zarr cube.

    This function reads a (time, frequency) block from the cube to support robust
    estimation of per-frequency statistics (median bandpass, MAD scale) that are
    applied to a smaller "frame" window.

    Parameters
    ----------
    cube
        Zarr array containing the full data cube with shape ``(nt, nfreq, nchan)``.
    chan
        Index of the cube channel to extract along the last axis.
    t_start_idx
        Start sample index of the frame (display window / ML patch).
    window_n
        Length of the frame in samples. Used for centering and for default donut gap.
    dt_s
        Cadence in seconds per sample (used to convert seconds to samples).
    baseline_seconds
        Requested duration of the context window in seconds. The resulting sample
        count is forced to be >= ``window_n``.
    mode
        One of ``{"center", "donut"}``:

        - ``"center"``: returns one contiguous time block centered on the *frame center*.
        - ``"donut"``: returns two blocks (left + right) surrounding the frame, excluding
          a gap around the frame to reduce contamination from events in-window.
    gap_seconds
        Only used when ``mode="donut"``. Duration of the excluded gap around the frame.
        If None, defaults to the frame length (exclude the frame window itself).
    dtype
        NumPy dtype to cast the returned array to.

    Returns
    -------
    numpy.ndarray
        Context array of shape ``(T_ctx, nfreq)``. Near dataset edges the returned
        context may be shorter than requested. If the window collapses entirely,
        an empty array of shape ``(0, nfreq)`` filled with NaNs is returned.

    Notes
    -----
    - Reads only the required slices from the Zarr store (no full-cube loads).
    - Never raises due to boundary effects; windows are clipped to data bounds.
    """
    nt, nfreq, nchan = cube.shape
    if not (0 <= chan < nchan):
        raise ValueError(f"chan={chan} out of bounds for cube with nchan={nchan}.")

    baseline_n = seconds_to_samples(dt_s, baseline_seconds)
    baseline_n = max(baseline_n, window_n)

    if mode not in {"center", "donut"}:
        raise ValueError("mode must be one of {'center','donut'}")

    def _read(i0: int, i1: int) -> np.ndarray:
        i0c = clamp_int(i0, 0, nt)
        i1c = clamp_int(i1, 0, nt)
        if i1c <= i0c:
            return np.full((0, nfreq), np.nan, dtype=dtype)
        return np.asarray(cube[i0c:i1c, :, chan], dtype=dtype)

    if mode == "center":
        w_half = window_n // 2
        b_half = baseline_n // 2
        return _read(t_start_idx + w_half - b_half, t_start_idx + w_half + b_half)

    # donut
    if gap_seconds is None:
        gap_n = window_n
    else:
        gap_n = max(seconds_to_samples(dt_s, gap_seconds), 0)

    b_half = baseline_n // 2
    g_half = gap_n // 2

    left = _read(t_start_idx - g_half - b_half, t_start_idx - g_half)
    right = _read(
        t_start_idx + window_n + g_half,
        t_start_idx + window_n + g_half + b_half,
    )

    if left.size == 0 and right.size == 0:
        return np.full((0, nfreq), np.nan, dtype=dtype)
    if left.size == 0:
        return right
    if right.size == 0:
        return left
    return np.concatenate([left, right], axis=0)


def make_safe_db_transform(
    *,
    floor: float = 1e-12,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a transform that converts linear-valued frames to dB using `safe_db`.

    This is useful as an explicit, named preprocessing step when you want:
    - dB-valued frames without any bandpass subtraction or whitening
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
        Output has the same shape as input.
    """
    if floor <= 0:
        raise ValueError("floor must be positive.")

    out_dtype = np.dtype(dtype)

    def transform(frame: np.ndarray, _meta: FrameMeta) -> np.ndarray:
        db = safe_db(frame, floor=floor)
        return np.asarray(db, dtype=out_dtype)

    return transform


def median_bandpass_db(context_db: np.ndarray) -> np.ndarray:
    """
    Compute a robust per-frequency median bandpass in dB space.

    Parameters
    ----------
    context_db
        Context array in dB with shape ``(T_ctx, F)``.

    Returns
    -------
    numpy.ndarray
        Per-frequency median bandpass of shape ``(F,)``.
    """
    return np.nanmedian(context_db, axis=0)


def mad_db(context_db: np.ndarray, median_f: np.ndarray) -> np.ndarray:
    """
    Compute a robust per-frequency scale estimate (scaled MAD) in dB space.

    The Median Absolute Deviation (MAD) is defined as::

        MAD = median(|x - median(x)|)

    This function returns ``1.4826 * MAD`` which is approximately an unbiased
    estimator of the standard deviation for Gaussian noise.

    Parameters
    ----------
    context_db
        Context array in dB with shape ``(T_ctx, F)``.
    median_f
        Per-frequency median bandpass of shape ``(F,)``.

    Returns
    -------
    numpy.ndarray
        Robust per-frequency scale of shape ``(F,)`` in dB units.
    """
    resid = context_db - median_f[None, :]
    mad = np.nanmedian(np.abs(resid), axis=0)
    return 1.4826 * mad


# -----------------------------------------------------------------------------
# Core transforms (cube-backed). These are the "source of truth".
# -----------------------------------------------------------------------------


def make_bandpass_subtractor(
    cube: zarr.Array,
    *,
    chan: int,
    ctx: ContextSpec,
    eps: float = 1e-12,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a bandpass-subtraction transform (cube-backed).

    This transform reads a context window from ``cube`` on each call, estimates a
    robust per-frequency median bandpass in dB, and subtracts it from the frame.

    Parameters
    ----------
    cube
        Zarr cube array of shape ``(nt, nfreq, nchan)``.
    chan
        Cube channel index to process.
    ctx
        Context window specification used to estimate the bandpass.
    eps
        Floor applied before converting to dB to avoid ``log10(0)``.
    dtype
        Output dtype.

    Returns
    -------
    Transform
        Callable with signature ``(frame_tf, meta) -> frame_db_resid``.

    DataSpace
    ---------
    - Assumes input is ``"linear"``.
    - Produces output in ``"db"`` (dB residual space).

    Notes
    -----
    If you want a Zarr-path convenience for pipelines, prefer
    ``step_bandpass_subtractor_from_zarr`` which *opens the store once* and
    returns a fully-specified ``PipelineStep`` (transform + config + DataSpace).
    """

    def transform(frame_tf: np.ndarray, meta: FrameMeta) -> np.ndarray:
        window_n = int(frame_tf.shape[0])
        context_tf = read_context_tf(
            cube,
            chan=chan,
            t_start_idx=meta.t_start_idx,
            window_n=window_n,
            dt_s=meta.dt_s,
            baseline_seconds=ctx.baseline_seconds,
            mode=ctx.mode,
            gap_seconds=ctx.gap_seconds,
            dtype=dtype,
        )

        frame_db = safe_db(frame_tf, floor=eps).astype(dtype, copy=False)
        context_db = safe_db(context_tf, floor=eps).astype(dtype, copy=False)

        bp = median_bandpass_db(context_db)
        return (frame_db - bp[None, :]).astype(dtype, copy=False)

    return transform


def make_mad_whitener(
    cube: zarr.Array,
    *,
    chan: int,
    ctx: ContextSpec,
    eps: float = 1e-12,
    min_scale: float = 0.1,
    alpha: float = 0.0,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a MAD-whitening transform (cube-backed).

    This transform reads a context window from ``cube`` on each call, estimates a
    per-frequency median bandpass and robust scale (scaled MAD) in dB space, and
    produces a z-score-like whitened output::

        z(t, f) = (db(t, f) - median_f) / scale_f

    Parameters
    ----------
    cube
        Zarr cube array of shape ``(nt, nfreq, nchan)``.
    chan
        Cube channel index to whiten.
    ctx
        Context window specification used to estimate bandpass and scale.
    eps
        Floor applied before converting to dB to avoid ``log10(0)``.
    min_scale
        Minimum allowed per-frequency scale (in dB) to avoid exploding dead channels.
    alpha
        Scale softening factor in [0, 1]. When > 0, blends per-frequency scale with a
        global scale to reduce over-downweighting of very noisy channels.
    dtype
        Output dtype.

    Returns
    -------
    Transform
        Callable with signature ``(frame_tf, meta) -> z``.

    DataSpace
    ---------
    - Assumes input is ``"linear"``.
    - Produces output in ``"z"`` (unitless whitened space).

    Notes
    -----
    If you want a Zarr-path convenience for pipelines, prefer
    ``step_mad_whitener_from_zarr`` which *opens the store once* and returns a
    fully-specified ``PipelineStep`` (transform + config + DataSpace).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    def transform(frame_tf: np.ndarray, meta: FrameMeta) -> np.ndarray:
        window_n = int(frame_tf.shape[0])

        context_tf = read_context_tf(
            cube,
            chan=chan,
            t_start_idx=meta.t_start_idx,
            window_n=window_n,
            dt_s=meta.dt_s,
            baseline_seconds=ctx.baseline_seconds,
            mode=ctx.mode,
            gap_seconds=ctx.gap_seconds,
            dtype=dtype,
        )

        frame_db = safe_db(frame_tf, floor=eps).astype(dtype, copy=False)
        context_db = safe_db(context_tf, floor=eps).astype(dtype, copy=False)

        bp = median_bandpass_db(context_db)
        scale = mad_db(context_db, bp)
        scale = np.maximum(scale, min_scale)

        if alpha > 0:
            global_scale = float(np.nanmedian(scale))
            scale = (1.0 - alpha) * scale + alpha * global_scale

        z = (frame_db - bp[None, :]) / scale[None, :]
        return z.astype(dtype, copy=False)

    return transform


def make_mad_whitener_multi(
    cube: zarr.Array,
    *,
    chans: Iterable[int],
    ctx: ContextSpec,
    eps: float = 1e-12,
    min_scale: float = 0.1,
    alpha: float = 0.0,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a multi-channel MAD-whitening transform (cube-backed).

    This is the channel-stacking analogue of ``make_mad_whitener``. It applies the
    same whitening logic independently to each requested cube channel and returns
    an output tensor with the same shape as the input.

    Parameters
    ----------
    cube
        Zarr cube array of shape ``(nt, nfreq, nchan)``.
    chans
        Iterable of cube channel indices to whiten (corresponds to the last axis).
    ctx
        Context window specification for estimating bandpass and scale.
    eps
        Floor applied before converting to dB.
    min_scale
        Minimum allowed per-frequency scale in dB.
    alpha
        Scale softening factor in [0, 1] (see ``make_mad_whitener``).
    dtype
        Output dtype.

    Returns
    -------
    Transform
        Callable with signature ``(frame_tfc, meta) -> out_tfc`` where both arrays
        have shape ``(T, F, C)``.

    DataSpace
    ---------
    - Assumes input is ``"linear"``.
    - Produces output in ``"z"`` for each channel.

    Notes
    -----
    If you want a Zarr-path convenience for pipelines, prefer
    ``step_mad_whitener_multi_from_zarr`` which *opens the store once* and returns
    a fully-specified ``PipelineStep`` (transform + config + DataSpace).
    """
    chans_t = tuple(int(c) for c in chans)
    if len(chans_t) == 0:
        raise ValueError("chans must be non-empty.")

    whiteners = [
        make_mad_whitener(
            cube,
            chan=c,
            ctx=ctx,
            eps=eps,
            min_scale=min_scale,
            alpha=alpha,
            dtype=dtype,
        )
        for c in chans_t
    ]

    def transform(frame_tfc: np.ndarray, meta: FrameMeta) -> np.ndarray:
        if frame_tfc.ndim != 3:
            raise ValueError("Expected frame_tfc with shape (T, F, C).")
        if frame_tfc.shape[2] != len(chans_t):
            raise ValueError("frame_tfc channel dimension does not match `chans`.")

        out = np.empty_like(frame_tfc, dtype=dtype)
        for j, w in enumerate(whiteners):
            out[:, :, j] = w(frame_tfc[:, :, j], meta)
        return out

    return transform


def make_softclip_transform(
    *,
    kind: str = "tanh",
    clip: float = 7.0,
    alpha: float = 0.7,
    dtype: np.dtype = np.float32,
) -> Transform:
    """
    Create a soft clipping / dynamic-range compression transform.

    This is typically applied after whitening (e.g., after ``make_mad_whitener``)
    to limit extreme values while keeping the transform differentiable.

    Supported kinds
    ---------------
    - ``"tanh"``: ``y = clip * tanh(x / clip)``
    - ``"asinh"``: ``y = clip * asinh(alpha * x) / asinh(alpha * clip)``

    Parameters
    ----------
    kind
        Either ``"tanh"`` or ``"asinh"``.
    clip
        Soft clip scale. Values with magnitude much larger than ``clip`` saturate.
    alpha
        Shape parameter for the "asinh" variant. Ignored for "tanh".
    dtype
        Output dtype.

    Returns
    -------
    Transform
        Callable with signature ``(frame, meta) -> frame``.

    DataSpace
    ---------
    - Expects a unitless input (commonly ``"z"``).
    - Output is still unitless; callers often label it ``"compressed"`` to
      distinguish it from raw z-scores.

    Notes
    -----
    For pipeline usage, prefer ``step_softclip`` which returns a ``PipelineStep``
    with config + DataSpace attached.
    """
    kind_l = kind.lower()
    if kind_l not in {"tanh", "asinh"}:
        raise ValueError('kind must be "tanh" or "asinh".')
    if clip <= 0:
        raise ValueError("clip must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    def transform(frame: np.ndarray, _meta: FrameMeta) -> np.ndarray:
        x = np.asarray(frame, dtype=dtype)
        if kind_l == "tanh":
            y = clip * np.tanh(x / clip)
        else:
            denom = np.arcsinh(alpha * clip)
            y = clip * np.arcsinh(alpha * x) / denom
        return y.astype(dtype, copy=False)

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
        Callable with signature ``transform(frame_tfc, meta) -> frame_tf`` where
        the input must have shape ``(T, F, 2)`` with channels ordered
        ``(XX, YY)``.

    Notes
    -----
    This transform assumes the convention:

        I = 0.5 * (XX + YY)

    and operates entirely in linear space.
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
        Callable with signature ``transform(frame_tfc, meta) -> frame_tfc_out`` where
        the input must have shape ``(T, F, 4)`` with channels ordered as:

        - ``XX``
        - ``YY``
        - ``XY_mag``
        - ``XY_phase``

        and the output has shape ``(T, F, 4)`` with channels ordered as:

        - ``I``
        - ``Q``
        - ``U``
        - ``V``

    Notes
    -----
    The complex cross term is reconstructed as::

        XY = XY_mag * exp(1j * XY_phase)

    and the Stokes convention used is::

        I = 0.5 * (XX + YY)
        Q = 0.5 * (XX - YY)
        U = Re(XY)
        V = Im(XY)

    This transform operates entirely in linear space.
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


def compose(*transforms: Transform) -> Transform:
    """
    Compose multiple transforms into a single transform.

    Parameters
    ----------
    *transforms
        Sequence of callables with signature ``(frame, meta) -> frame``.

    Returns
    -------
    Transform
        Callable that applies each input transform in order.
    """

    def composed(frame: np.ndarray, meta: FrameMeta) -> np.ndarray:
        out = frame
        for t in transforms:
            out = t(out, meta)
        return out

    return composed


# -----------------------------------------------------------------------------
# Pipeline wrapper (with DataSpace tracking)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStep:
    """
    A single preprocessing step in a `PreprocessPipeline`.

    A step couples:
    - an executable transform (callable),
    - a JSON-serializable config for reproducibility,
    - optional DataSpace constraints for chaining validation.

    Parameters
    ----------
    name
        Human-readable identifier for the step, e.g. "mad_whiten" or "softclip".
    transform
        Callable with signature ``transform(frame, meta) -> frame``.
    config
        JSON-serializable configuration for the step. Stored for logging and for
        reconstructing pipelines when used with an external registry of builders.
    in_space
        Expected DataSpace of the input to this step. If provided, the pipeline can
        validate that the step is inserted at a compatible point.
    out_space
        DataSpace produced by this step. If provided, the pipeline can report an
        accurate `output_space` for downstream labeling/validation.
    """

    name: str
    transform: Transform
    config: dict[str, Any]
    in_space: Optional[DataSpace] = None
    out_space: Optional[DataSpace] = None


class PreprocessPipeline:
    """
    A lightweight, named preprocessing chain with reproducibility metadata.

    The pipeline conforms to the standard specscout transform API::

        pipeline(frame: np.ndarray, meta: FrameMeta) -> np.ndarray

    This class exists to keep transform experiments tidy:
    - Ordered steps with human-readable names
    - Step configs stored in a JSON-friendly form
    - Optional DataSpace tracking + validation for correct chaining

    Notes
    -----
    - Callables/closures are not serialized by `to_dict()` / `save_json()`.
      Those methods store configs + DataSpace only.
    - For Zarr-backed transforms, prefer the `step_*_from_zarr` helpers in this module.
      They open the store once and return a `PipelineStep` with DataSpace + config.
    """

    def __init__(
        self,
        steps: Optional[Sequence[PipelineStep]] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
        input_space: DataSpace = "linear",
    ) -> None:
        self._steps: list[PipelineStep] = list(steps) if steps is not None else []
        self._metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self._input_space: DataSpace = input_space

    @property
    def steps(self) -> tuple[PipelineStep, ...]:
        """Pipeline steps in execution order (immutable view)."""
        return tuple(self._steps)

    @property
    def metadata(self) -> dict[str, Any]:
        """Pipeline-level metadata dict (copied on access)."""
        return dict(self._metadata)

    @property
    def input_space(self) -> DataSpace:
        """Declared DataSpace of inputs to the pipeline."""
        return self._input_space

    @property
    def output_space(self) -> DataSpace:
        """
        Best-effort DataSpace of pipeline outputs.

        If the last step declares `out_space`, that is used; otherwise the pipeline
        falls back to `input_space`.
        """
        if not self._steps:
            return self._input_space
        last = self._steps[-1]
        return last.out_space if last.out_space is not None else self._input_space

    def with_metadata(self, **metadata: Any) -> "PreprocessPipeline":
        """
        Return a new pipeline with updated pipeline-level metadata.

        This is useful for recording provenance such as zarr_path, date ranges,
        selection criteria, etc.
        """
        md = dict(self._metadata)
        md.update(metadata)
        return PreprocessPipeline(self._steps, metadata=md, input_space=self._input_space)

    def with_input_space(self, input_space: DataSpace) -> "PreprocessPipeline":
        """
        Return a new pipeline with a different declared `input_space`.

        Example: if you know your dataset is already in dB, you can set
        ``input_space="db"`` and validate compatible steps.
        """
        return PreprocessPipeline(self._steps, metadata=self._metadata, input_space=input_space)

    def _current_space(self) -> DataSpace:
        """Internal: current DataSpace after the last step (or input_space if empty)."""
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
        """
        Append one step, returning a new pipeline (pipelines are treated as immutable).

        Parameters
        ----------
        name
            Step name used for summaries and serialization.
        transform
            Callable with signature ``(frame, meta) -> frame``.
        config
            JSON-serializable configuration dict for this step.
        in_space
            Expected input DataSpace for this step. If provided (and
            ``validate_spaces=True``), the pipeline enforces that it matches the current
            pipeline space at insertion time.
        out_space
            Output DataSpace produced by this step.
        validate_spaces
            If True, enforce `in_space` compatibility checks when `in_space` is provided.

        Returns
        -------
        PreprocessPipeline
            New pipeline with the step appended.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("step name must be a non-empty string.")
        if config is None:
            config = {}

        if validate_spaces and in_space is not None:
            current = self._current_space()
            if in_space != current:
                raise ValueError(
                    f"Cannot add step {name!r}: in_space={in_space!r} does not match "
                    f"current pipeline space {current!r}. If this is intentional, set "
                    f"validate_spaces=False."
                )

        step = PipelineStep(
            name=name,
            transform=transform,
            config=dict(config),
            in_space=in_space,
            out_space=out_space,
        )
        return PreprocessPipeline([*self._steps, step], metadata=self._metadata, input_space=self._input_space)

    def add(self, step: PipelineStep, *, validate_spaces: bool = True) -> "PreprocessPipeline":
        """
        Convenience: add a pre-built `PipelineStep`.

        This is the intended way to build pipelines when using the `step_*` helpers::

            pipe = PreprocessPipeline().add(step_mad_whitener_from_zarr(...)).add(step_softclip(...))

        Parameters
        ----------
        step
            A `PipelineStep` produced by this module (often by a `step_*` helper).
        validate_spaces
            If True, enforce DataSpace chaining when the step declares `in_space`.

        Returns
        -------
        PreprocessPipeline
            New pipeline with the step appended.
        """
        return self.add_step(
            step.name,
            step.transform,
            step.config,
            in_space=step.in_space,
            out_space=step.out_space,
            validate_spaces=validate_spaces,
        )

    def extend(self, steps: Iterable[PipelineStep], *, validate_spaces: bool = True) -> "PreprocessPipeline":
        """
        Append multiple steps, returning a new pipeline.

        Parameters
        ----------
        steps
            Iterable of `PipelineStep` objects.
        validate_spaces
            If True, enforce DataSpace chaining for each step when `in_space` is provided.
        """
        pipe: PreprocessPipeline = self
        for s in steps:
            pipe = pipe.add(s, validate_spaces=validate_spaces)
        return pipe

    def __call__(self, frame: np.ndarray, meta: FrameMeta) -> np.ndarray:
        """Apply all steps in order."""
        out = frame
        for step in self._steps:
            out = step.transform(out, meta)
        return out

    def compose(self, *transforms: Transform) -> Transform:
        """
        Convenience: treat the pipeline as the first transform, then apply extras.

        Equivalent to: ``compose(self, *transforms)``.
        """

        def _composed(frame: np.ndarray, meta: FrameMeta) -> np.ndarray:
            out = self(frame, meta)
            for t in transforms:
                out = t(out, meta)
            return out

        return _composed

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the pipeline."""
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
            lines.append(
                f"[{i}] {s.name} "
                f"in={s.in_space if s.in_space is not None else '?'} "
                f"out={s.out_space if s.out_space is not None else '?'} "
                f"config={s.config}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the pipeline configuration to a JSON-compatible dict.

        Notes
        -----
        This does not serialize executable code. Only metadata + DataSpace + per-step
        configs are stored.
        """
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
        """
        Save `to_dict()` as JSON for reproducibility/logging.

        Parameters
        ----------
        path
            Output JSON file path.
        indent
            JSON indentation.

        Returns
        -------
        pathlib.Path
            Resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n")
        return out


# -----------------------------------------------------------------------------
# Step-builder helpers (Zarr-backed). These replace the old make_*_from_zarr layer.
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

    Parameters
    ----------
    floor
        Floor applied before log10 to avoid log10(0).
    dtype
        Output dtype.
    name
        Step name stored in the pipeline.
    in_space, out_space
        DataSpace annotations. Defaults are ("linear" -> "db").

    Returns
    -------
    PipelineStep
        A step suitable for `PreprocessPipeline.add(...)`.
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
    Build a `PipelineStep` that converts dual-linear autocorrelation inputs
    ``(XX, YY)`` into Stokes I.

    Parameters
    ----------
    dtype
        Output dtype.
    name
        Step name stored in the pipeline.
    in_space, out_space
        DataSpace annotations. Both default to ``"linear"`` since this transform
        operates and returns linear-valued products.

    Returns
    -------
    PipelineStep
        A step suitable for `PreprocessPipeline.add(...)`.

    Notes
    -----
    Expected input shape is ``(T, F, 2)`` with channels ordered as ``(XX, YY)``.
    The convention used is::

        I = 0.5 * (XX + YY)
    """
    t = make_stokes_i_transform(dtype=dtype)
    cfg = {
        "dtype": str(np.dtype(dtype)),
        "channel_order_in": ["XX", "YY"],
        "channel_order_out": ["I"],
        "convention": "I = 0.5 * (XX + YY)",
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
    Build a `PipelineStep` that converts linear polarization products into
    full Stokes ``(I, Q, U, V)``.

    Parameters
    ----------
    dtype
        Output dtype.
    name
        Step name stored in the pipeline.
    in_space, out_space
        DataSpace annotations. Both default to ``"linear"`` since this transform
        operates and returns linear-valued products.

    Returns
    -------
    PipelineStep
        A step suitable for `PreprocessPipeline.add(...)`.

    Notes
    -----
    Expected input shape is ``(T, F, 4)`` with channels ordered as::

        (XX, YY, XY_mag, XY_phase)

    Output shape is ``(T, F, 4)`` with channels ordered as::

        (I, Q, U, V)

    The convention used is::

        I = 0.5 * (XX + YY)
        Q = 0.5 * (XX - YY)
        U = Re(XY)
        V = Im(XY)

    where::

        XY = XY_mag * exp(1j * XY_phase)
    """
    t = make_stokes_iquv_transform(dtype=dtype)
    cfg = {
        "dtype": str(np.dtype(dtype)),
        "channel_order_in": ["XX", "YY", "XY_mag", "XY_phase"],
        "channel_order_out": ["I", "Q", "U", "V"],
        "convention": {
            "I": "0.5 * (XX + YY)",
            "Q": "0.5 * (XX - YY)",
            "U": "real(XY_mag * exp(1j * XY_phase))",
            "V": "imag(XY_mag * exp(1j * XY_phase))",
        },
    }
    return PipelineStep(
        name=name,
        transform=t,
        config=cfg,
        in_space=in_space,
        out_space=out_space,
    )


def step_bandpass_subtractor_from_zarr(
    zarr_path: str,
    *,
    chan: int,
    ctx: ContextSpec,
    eps: float = 1e-12,
    dtype: np.dtype = np.float32,
    name: str = "bandpass_subtract",
) -> PipelineStep:
    """
    Build a Zarr-backed bandpass subtraction `PipelineStep`.

    This helper is the recommended "Zarr path" entry point for bandpass subtraction.
    It **replaces** the older pattern of having both:

    - `make_bandpass_subtractor_from_zarr(...)` (path -> Transform), and
    - a separate step wrapper that attached config/DataSpace.

    Instead, this function does all of the following in one place:
    - Opens the Zarr store once via `patches.open_cube`
    - Builds the cube-backed transform via `make_bandpass_subtractor`
    - Attaches a JSON-serializable config (including `zarr_path`)
    - Declares DataSpace for chaining/validation in `PreprocessPipeline`

    Parameters
    ----------
    zarr_path
        Path to a specscout ``.zarr`` directory.
    chan
        Cube channel index to process.
    ctx
        Context window specification used to estimate the bandpass.
    eps
        Floor applied before converting to dB.
    dtype
        Output dtype used by the returned transform.
    name
        Name for the resulting `PipelineStep` (shows up in summaries/config dumps).

    Returns
    -------
    PipelineStep
        Step with `transform(frame, meta) -> frame` plus config and DataSpace tags.

    DataSpace
    ---------
    in_space="linear", out_space="db"
    """
    cube, _attrs, _time_axis = open_cube(zarr_path)
    t = make_bandpass_subtractor(cube, chan=chan, ctx=ctx, eps=eps, dtype=dtype)
    cfg = {
        "zarr_path": zarr_path,
        "chan": int(chan),
        "ctx": {
            "baseline_seconds": float(ctx.baseline_seconds),
            "mode": ctx.mode,
            "gap_seconds": ctx.gap_seconds,
        },
        "eps": float(eps),
        "dtype": str(np.dtype(dtype)),
    }
    return PipelineStep(name=name, transform=t, config=cfg, in_space="linear", out_space="db")


def step_mad_whitener_from_zarr(
    zarr_path: str,
    *,
    chan: int,
    ctx: ContextSpec,
    eps: float = 1e-12,
    min_scale: float = 0.1,
    alpha: float = 0.0,
    dtype: np.dtype = np.float32,
    name: str = "mad_whiten",
) -> PipelineStep:
    """
    Build a Zarr-backed MAD-whitening `PipelineStep`.

    This helper is the recommended "Zarr path" entry point for whitening.
    It **replaces** the older pattern of having both:

    - `make_mad_whitener_from_zarr(...)` (path -> Transform), and
    - a separate step wrapper that attached config/DataSpace.

    Instead, this function:
    - Opens the Zarr store once via `patches.open_cube`
    - Builds the cube-backed transform via `make_mad_whitener`
    - Attaches a JSON-serializable config (including `zarr_path`)
    - Declares DataSpace for pipeline chaining/validation

    Parameters
    ----------
    zarr_path
        Path to a specscout ``.zarr`` directory.
    chan
        Cube channel index to whiten.
    ctx
        Context window specification used to estimate bandpass/scale.
    eps
        Floor applied before converting to dB.
    min_scale
        Minimum allowed per-frequency scale (in dB).
    alpha
        Scale softening factor in [0, 1] (see `make_mad_whitener`).
    dtype
        Output dtype used by the returned transform.
    name
        Name for the resulting `PipelineStep`.

    Returns
    -------
    PipelineStep
        Step with `transform(frame, meta) -> frame` plus config and DataSpace tags.

    DataSpace
    ---------
    in_space="linear", out_space="z"
    """
    cube, _attrs, _time_axis = open_cube(zarr_path)
    t = make_mad_whitener(
        cube,
        chan=chan,
        ctx=ctx,
        eps=eps,
        min_scale=min_scale,
        alpha=alpha,
        dtype=dtype,
    )
    cfg = {
        "zarr_path": zarr_path,
        "chan": int(chan),
        "ctx": {
            "baseline_seconds": float(ctx.baseline_seconds),
            "mode": ctx.mode,
            "gap_seconds": ctx.gap_seconds,
        },
        "eps": float(eps),
        "min_scale": float(min_scale),
        "alpha": float(alpha),
        "dtype": str(np.dtype(dtype)),
    }
    return PipelineStep(name=name, transform=t, config=cfg, in_space="linear", out_space="z")


def step_mad_whitener_multi_from_zarr(
    zarr_path: str,
    *,
    chans: Iterable[int],
    ctx: ContextSpec,
    eps: float = 1e-12,
    min_scale: float = 0.1,
    alpha: float = 0.0,
    dtype: np.dtype = np.float32,
    name: str = "mad_whiten_multi",
) -> PipelineStep:
    """
    Build a Zarr-backed multi-channel MAD-whitening `PipelineStep`.

    This helper is the recommended "Zarr path" entry point for producing an
    ML-ready tensor with a channel dimension (e.g., stacking pol00/pol11/pol01mag).
    It **replaces** the older pattern of having both:

    - `make_mad_whitener_multi_from_zarr(...)` (path -> Transform), and
    - a separate step wrapper that attached config/DataSpace.

    Instead, this function:
    - Opens the Zarr store once via `patches.open_cube`
    - Builds the cube-backed transform via `make_mad_whitener_multi`
    - Attaches a JSON-serializable config (including `zarr_path`)
    - Declares DataSpace for pipeline chaining/validation

    Parameters
    ----------
    zarr_path
        Path to a specscout ``.zarr`` directory.
    chans
        Iterable of cube channel indices to whiten (corresponds to the last axis).
    ctx
        Context window specification used to estimate bandpass/scale.
    eps
        Floor applied before converting to dB.
    min_scale
        Minimum allowed per-frequency scale (in dB).
    alpha
        Scale softening factor in [0, 1] (see `make_mad_whitener`).
    dtype
        Output dtype used by the returned transform.
    name
        Name for the resulting `PipelineStep`.

    Returns
    -------
    PipelineStep
        Step with `transform(frame, meta) -> frame` plus config and DataSpace tags.

    DataSpace
    ---------
    in_space="linear", out_space="z"
    """
    chans_t = tuple(int(c) for c in chans)
    cube, _attrs, _time_axis = open_cube(zarr_path)
    t = make_mad_whitener_multi(
        cube,
        chans=chans_t,
        ctx=ctx,
        eps=eps,
        min_scale=min_scale,
        alpha=alpha,
        dtype=dtype,
    )
    cfg = {
        "zarr_path": zarr_path,
        "chans": list(chans_t),
        "ctx": {
            "baseline_seconds": float(ctx.baseline_seconds),
            "mode": ctx.mode,
            "gap_seconds": ctx.gap_seconds,
        },
        "eps": float(eps),
        "min_scale": float(min_scale),
        "alpha": float(alpha),
        "dtype": str(np.dtype(dtype)),
    }
    return PipelineStep(name=name, transform=t, config=cfg, in_space="linear", out_space="z")


def step_softclip(
    *,
    kind: str = "tanh",
    clip: float = 7.0,
    alpha: float = 0.7,
    dtype: np.dtype = np.float32,
    name: str = "softclip",
    in_space: DataSpace = "z",
    out_space: DataSpace = "compressed",
) -> PipelineStep:
    """
    Build a soft-clipping / compression `PipelineStep`.

    This helper is the recommended way to add soft clipping to a `PreprocessPipeline`
    because it automatically stores a JSON-friendly config and declares DataSpace for
    chaining/validation.

    Parameters
    ----------
    kind
        Either ``"tanh"`` or ``"asinh"``.
    clip
        Soft clip scale. Values with magnitude much larger than ``clip`` saturate.
    alpha
        Shape parameter for the "asinh" variant. Ignored for "tanh".
    dtype
        Output dtype used by the returned transform.
    name
        Name for the resulting `PipelineStep`.
    in_space
        Expected DataSpace of the input to this step. Defaults to ``"z"`` because
        soft clipping is commonly applied to whitened outputs.
    out_space
        DataSpace declared for the output. Defaults to ``"compressed"`` to distinguish
        it from raw z-scores. If you prefer to treat the output as still "z", pass
        ``out_space="z"``.

    Returns
    -------
    PipelineStep
        Step with `transform(frame, meta) -> frame` plus config and DataSpace tags.
    """
    t = make_softclip_transform(kind=kind, clip=clip, alpha=alpha, dtype=dtype)
    cfg = {
        "kind": str(kind),
        "clip": float(clip),
        "alpha": float(alpha),
        "dtype": str(np.dtype(dtype)),
    }
    return PipelineStep(name=name, transform=t, config=cfg, in_space=in_space, out_space=out_space)
