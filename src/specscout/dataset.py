"""
Dataset utilities for turning specscout Zarr cubes into rolling-window examples.

This module defines the dataset-layer abstractions used throughout specscout:

- `FrameMeta`: metadata for one extracted frame
- `FramePlan`: a validated sliding-window plan over a cube time range
- `DatasetPlan`: a compact description of dataset iteration and slicing
- `SpecscoutDataset`: a lazy, indexable sequence of rolling-window examples

Design goals
------------
- Centralize frame-planning logic in one place
- Reuse low-level Zarr reading via `specscout.patches.read_patch`
- Apply preprocessing consistently through `PreprocessPipeline`
- Provide both dataset-indexed access and direct loading by `t_start_idx`

Notes
-----
The dataset itself does not perform unit conversions or feature engineering.
Such transformations should be supplied via the optional preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import numpy as np

from .core import (
    channel_names_from_indices,
    clamp_int,
    freq_axis_from_attrs,
    parse_utc,
    seconds_to_samples,
    time_index,
)
from .patches import PatchSpec, open_cube, read_patch
from .preprocess import DataDesc

if TYPE_CHECKING:
    from .preprocess import DataDesc, PreprocessPipeline

Array = np.ndarray
MaybeChans = int | tuple[int, ...]


@dataclass(frozen=True)
class FrameMeta:
    """
    Minimal metadata describing a single extracted frame/patch.

    Parameters
    ----------
    t_start_idx
        Start sample index (time axis) in the cube.
    t_end_idx
        End sample index (exclusive) in the cube.
    frame_idx
        Frame number within the requested sequence (0..n_frames-1), or -1 if
        the patch is not associated with a dataset index.
    start_time_utc
        UTC datetime corresponding to `t_start_idx`.
    dt_s
        Cadence in seconds per sample.
    """

    t_start_idx: int
    t_end_idx: int
    frame_idx: int
    start_time_utc: datetime
    dt_s: float


@dataclass(frozen=True)
class FramePlan:
    """
    Plan for sliding-window frames over a time range in a cube.

    Attributes
    ----------
    i_start
        First valid start index (samples).
    i_stop
        Exclusive stop bound used to compute valid windows (samples).
    window_n
        Window length (samples).
    step_n
        Step length (samples).
    n_frames
        Number of frames in the plan.
    dt_s
        Cadence in seconds per sample.
    t0_unix_s
        Unix seconds corresponding to sample index 0.
    """

    i_start: int
    i_stop: int
    window_n: int
    step_n: int
    n_frames: int
    dt_s: float
    t0_unix_s: float


def plan_frames(
    *,
    nt: int,
    t0_unix_s: float,
    dt_s: float,
    start_utc: str,
    stop_utc: str,
    window_seconds: float,
    step_seconds: float,
) -> FramePlan:
    """
    Build a `FramePlan` from UTC strings and sliding-window durations.

    Parameters
    ----------
    nt
        Total number of time samples in the cube.
    t0_unix_s
        Unix seconds corresponding to sample index 0.
    dt_s
        Cadence in seconds per sample.
    start_utc, stop_utc
        UTC strings in format ``YYYYmmdd_HHMMSS``.
    window_seconds, step_seconds
        Window length and step size in seconds.

    Returns
    -------
    FramePlan
        Plan describing valid frame start indices and counts.
    """
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive.")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    window_n = seconds_to_samples(dt_s, window_seconds, min_n=1)
    step_n = seconds_to_samples(dt_s, step_seconds, min_n=1)

    start_dt = parse_utc(start_utc)
    stop_dt = parse_utc(stop_utc)
    if stop_dt < start_dt:
        raise ValueError("stop_utc must be >= start_utc.")

    i_start = clamp_int(time_index(start_dt.timestamp(), t0_unix_s, dt_s), 0, nt)
    i_stop = clamp_int(time_index(stop_dt.timestamp(), t0_unix_s, dt_s), 0, nt)

    last_start = i_stop - window_n
    if last_start < i_start:
        raise ValueError("Requested range is shorter than one window.")

    n_frames = (last_start - i_start) // step_n + 1

    return FramePlan(
        i_start=i_start,
        i_stop=i_stop,
        window_n=window_n,
        step_n=step_n,
        n_frames=n_frames,
        dt_s=float(dt_s),
        t0_unix_s=float(t0_unix_s),
    )


@dataclass(frozen=True)
class DatasetPlan:
    """
    Compact description of how a dataset iterates over a cube.

    Attributes
    ----------
    frame_plan
        Underlying frame plan (indices + window/step in samples).
    f_start
        First frequency channel index (inclusive).
    f_stop
        Stop frequency channel index (exclusive).
    chans
        Cube channels used by this dataset (e.g. ``(0,)``, ``(0, 1)``).
    """

    frame_plan: FramePlan
    f_start: int
    f_stop: int
    chans: tuple[int, ...]


class SpecscoutDataset(Sequence):
    """
    Lazy, indexable dataset of rolling-window patches from a specscout Zarr cube.

    Parameters
    ----------
    zarr_path
        Path to the specscout Zarr store directory.
    start_utc, stop_utc
        UTC timestamps (``YYYYmmdd_HHMMSS``) defining the dataset time span.
        Windows are generated such that `[t_start, t_start + window)` stays
        in-bounds.
    window_seconds, step_seconds
        Window length and step size in seconds.
    chans
        Cube channel indices to include. If an int, it is treated as a 1-tuple.
        - one channel -> samples have shape ``(T, F)``
        - multiple channels -> samples have shape ``(T, F, C)``
    f_start, f_stop
        Optional frequency slicing in channel indices (Python slice semantics).
        Defaults to full band.
    pipe
        Optional preprocessing pipeline applied to each patch after loading.
        The pipeline must conform to the standard specscout API:
        ``pipe(x, meta) -> x``.
    dtype
        Dtype to cast loaded patches to.
    return_meta
        If True, `__getitem__` returns ``(x, meta)``. If False, returns only
        ``x``.

    Notes
    -----
    - The Zarr store is opened once in the constructor.
    - Each `__getitem__` reads only the required patch from disk.
    - The dataset assumes the cube values are in linear space on read; if your
      pipeline expects something else, set the pipeline's `input_space`
      accordingly or include a step such as `step_safe_db`.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        *,
        start_utc: str,
        stop_utc: str,
        window_seconds: float,
        step_seconds: float,
        chans: MaybeChans = (0,),
        f_start: int = 0,
        f_stop: Optional[int] = None,
        pipe: Optional[PreprocessPipeline] = None,
        dtype: np.dtype = np.float32,
        return_meta: bool = True,
    ) -> None:
        cube, attrs, time_axis = open_cube(zarr_path)
        self._cube = cube
        self._attrs = attrs
        self._time_axis = time_axis

        nt, nfreq, nchan_total = cube.shape
        self._dt_s = float(attrs["dt_seconds"])
        self._t0_unix_s = float(attrs["t0_unix_seconds"])

        if isinstance(chans, int):
            chans = (chans,)
        chans = tuple(int(c) for c in chans)
        if len(chans) == 0:
            raise ValueError("chans must contain at least one channel index.")
        if any((c < 0 or c >= nchan_total) for c in chans):
            raise ValueError(f"chans out of bounds for cube with nchan={nchan_total}.")

        if f_stop is None:
            f_stop = nfreq
        if not (0 <= f_start < f_stop <= nfreq):
            raise ValueError("Invalid frequency slice f_start/f_stop.")

        frame_plan = plan_frames(
            nt=nt,
            t0_unix_s=self._t0_unix_s,
            dt_s=self._dt_s,
            start_utc=start_utc,
            stop_utc=stop_utc,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )

        self._spec = PatchSpec(
            window_n=frame_plan.window_n,
            step_n=frame_plan.step_n,
            f_start=int(f_start),
            f_stop=int(f_stop),
            chans=chans,
        )

        self._plan = DatasetPlan(
            frame_plan=frame_plan,
            f_start=int(f_start),
            f_stop=int(f_stop),
            chans=chans,
        )

        self._pipe = pipe
        self._dtype = np.dtype(dtype)
        self._return_meta = bool(return_meta)

    @property
    def attrs(self) -> dict[str, object]:
        """
        Zarr group attributes as a plain dict.
        """
        return dict(self._attrs)

    @property
    def plan(self) -> DatasetPlan:
        """
        Stable description of dataset iteration and cube slicing.
        """
        return self._plan

    @property
    def dt_s(self) -> float:
        """
        Cadence in seconds per sample.
        """
        return self._dt_s

    @property
    def pipe(self) -> Optional[PreprocessPipeline]:
        """
        Preprocessing pipeline applied to each example, if any.
        """
        return self._pipe

    def freq_axis(self) -> tuple[np.ndarray, str]:
        """
        Return the frequency axis and its label for this dataset slice.

        Returns
        -------
        freqs, x_label
            `freqs` has length `f_stop - f_start`.
        """
        _nt, nfreq_total, _nchan = self._cube.shape
        freqs, x_label = freq_axis_from_attrs(self._attrs, nfreq_total)
        return freqs[self._plan.f_start : self._plan.f_stop], x_label

    def __len__(self) -> int:
        """
        Number of examples/windows in the dataset.
        """
        return int(self._plan.frame_plan.n_frames)

    def _frame_start_idx(self, idx: int) -> int:
        """
        Map dataset index -> cube start sample index.

        Raises
        ------
        IndexError
            If `idx` is out of range.
        """
        fp = self._plan.frame_plan
        if idx < 0 or idx >= fp.n_frames:
            raise IndexError("dataset index out of range")
        return int(fp.i_start + idx * fp.step_n)

    def _meta_for(self, idx: int) -> FrameMeta:
        """
        Construct `FrameMeta` for a planned dataset frame without reading data.
        """
        fp = self._plan.frame_plan
        t_start_idx = self._frame_start_idx(idx)
        start_time = self._time_axis.start_time_utc(t_start_idx)
        return FrameMeta(
            t_start_idx=t_start_idx,
            t_end_idx=t_start_idx + fp.window_n,
            frame_idx=idx,
            start_time_utc=start_time,
            dt_s=fp.dt_s,
        )

    def load_by_t_start_idx(
        self,
        t_start_idx: int,
        *,
        frame_idx: int = -1,
        apply_pipe: bool = True,
        return_meta: Optional[bool] = None,
        dtype: Optional[np.dtype] = None,
    ):
        """
        Load a patch by an explicit cube start index.

        This is the preferred entry point for visualization and outlier tooling
        where you have a `FrameMeta` (or just a start index) but not necessarily
        a dataset index.

        Parameters
        ----------
        t_start_idx
            Cube start sample index for the patch/window.
        frame_idx
            Frame index to store in the returned `FrameMeta`. Use -1 when the
            patch is not associated with a dataset index.
        apply_pipe
            If True and a pipeline exists, apply it to the loaded data.
        return_meta
            Overrides the dataset's `return_meta` for this call.
            If None, uses the dataset default.
        dtype
            Optional dtype override for the read (and thus the pipeline input).

        Returns
        -------
        (x, meta) or x
            If metadata is requested, returns `(x, meta)`; otherwise returns `x`.

        Notes
        -----
        - This does not require `t_start_idx` to correspond to one of the
          planned dataset frames.
        - Bounds behavior is determined by `specscout.patches.read_patch`.
        """
        t_start_idx_i = int(t_start_idx)
        fp = self._plan.frame_plan
        use_dtype = self._dtype if dtype is None else np.dtype(dtype)

        patch = read_patch(
            self._cube,
            self._time_axis,
            self._spec,
            t_start_idx=t_start_idx_i,
            dtype=use_dtype,
        )
        x = patch.data

        meta = FrameMeta(
            t_start_idx=t_start_idx_i,
            t_end_idx=t_start_idx_i + fp.window_n,
            frame_idx=int(frame_idx),
            start_time_utc=patch.start_time_utc,
            dt_s=fp.dt_s,
        )

        if apply_pipe and self._pipe is not None:
            channel_names = channel_names = channel_names_from_indices(self._plan.chans)

            pipe = self._pipe.with_input_desc(
                DataDesc(
                    channel_names=channel_names,
                    space=self._pipe.input_space,
                )
            )

            x = pipe(x, meta)

        return_meta_flag = self._return_meta if return_meta is None else bool(return_meta)

        if return_meta_flag:
            return x, meta
        return x

    def __getitem__(self, idx: int):
        """
        Load one dataset example by dataset index.

        Returns
        -------
        (x, meta) or x
            If `return_meta=True`, returns `(x, meta)`. Otherwise returns `x`.
        """
        idx_i = int(idx)
        t_start_idx = self._frame_start_idx(idx_i)

        return self.load_by_t_start_idx(
            t_start_idx,
            frame_idx=idx_i,
            apply_pipe=True,
            return_meta=self._return_meta,
            dtype=self._dtype,
        )

    def iter(self) -> Iterator[tuple[Array, FrameMeta]]:
        """
        Iterate through all examples sequentially.

        Yields
        ------
        x, meta
            Each example plus metadata (always yields metadata regardless of
            `return_meta`).
        """
        for i in range(len(self)):
            if self._return_meta:
                x, meta = self[i]
            else:
                x = self[i]
                meta = self._meta_for(i)
            yield x, meta

    def random_sample(
        self,
        *,
        rng: Optional[np.random.Generator] = None,
        return_meta: Optional[bool] = None,
    ):
        """
        Draw a random example from the dataset.

        Parameters
        ----------
        rng
            NumPy random generator. If None, uses `np.random.default_rng()`.
        return_meta
            Overrides the dataset's default `return_meta` for this call.
            If None, uses the dataset default.

        Returns
        -------
        x or (x, meta)
            A random example (and metadata if requested).
        """
        if rng is None:
            rng = np.random.default_rng()

        i = int(rng.integers(0, len(self)))

        if return_meta is None:
            return self[i]

        if return_meta:
            if self._return_meta:
                return self[i]
            x = self[i]
            return x, self._meta_for(i)

        if self._return_meta:
            x, _ = self[i]
            return x
        return self[i]

    def to_numpy_batch(
        self,
        indices: Sequence[int],
        *,
        return_meta: bool = False,
        dtype: Optional[np.dtype] = None,
    ):
        """
        Load a batch of examples into a single NumPy array.

        Parameters
        ----------
        indices
            Sequence of dataset indices to load.
        return_meta
            If True, also return a list of `FrameMeta` objects.
        dtype
            If provided, cast the final stacked batch to this dtype.

        Returns
        -------
        batch
            Array of shape `(B, T, F)` or `(B, T, F, C)`.
        metas
            Only returned if `return_meta=True`.
        """
        xs: list[np.ndarray] = []
        metas: list[FrameMeta] = []

        for i in indices:
            i_i = int(i)
            if self._return_meta:
                x, meta = self[i_i]
            else:
                x = self[i_i]
                meta = self._meta_for(i_i)

            xs.append(np.asarray(x))
            if return_meta:
                metas.append(meta)

        batch = np.stack(xs, axis=0)
        if dtype is not None:
            batch = np.asarray(batch, dtype=dtype)

        if return_meta:
            return batch, metas
        return batch
