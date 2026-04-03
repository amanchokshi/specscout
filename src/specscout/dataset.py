"""
Dataset utilities for turning specscout Zarr cubes into ML-ready examples.

Design goals
------------
- No duplicated Zarr indexing/slicing logic:
  reading is delegated to `specscout.patches.read_patch`.
- No duplicated seconds->samples conversion:
  we compute a single `FramePlan` using `specscout.core.plan_frames`, then reuse
  `plan.window_n` and `plan.step_n` everywhere.
- Canonical preprocessing is supplied via a `PreprocessPipeline` (`pipe`), which
  mirrors `viz.py` usage and ensures consistent patterns across the project.
- Provide a fast, explicit "load by t_start_idx" entry point so visualization
  and outlier tooling does not need to reach into dataset internals.

The main entry point is `SpecscoutDataset`, which is a lazy, indexable sequence.
Each item yields `(x, meta)` by default, where:
- `x` is a NumPy array of shape (T, F) for one channel, or (T, F, C) for multiple
  cube channels.
- `meta` is a `specscout.core.FrameMeta`.

Units / "data space"
--------------------
The dataset itself does not apply unit conversions (e.g., safe dB). If you want
dB, z-scores, or compressed outputs, encode that as steps in `pipe`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from .core import FrameMeta, FramePlan, freq_axis_from_attrs, parse_utc, plan_frames
from .patches import PatchSpec, open_cube, read_patch
from .preprocess import PreprocessPipeline

Array = np.ndarray
MaybeChans = Union[int, Tuple[int, ...]]


@dataclass(frozen=True)
class DatasetPlan:
    """
    A compact description of how a dataset iterates over a cube.

    This is a thin wrapper around `specscout.core.FramePlan`, included so the
    dataset can expose a stable, dataset-specific plan object if you want to
    extend it later (e.g. stratified sampling, masks, etc.).

    Attributes
    ----------
    frame_plan
        The underlying `FramePlan` (indices + window/step in samples).
    f_start
        First frequency channel index (inclusive).
    f_stop
        Stop frequency channel index (exclusive).
    chans
        Cube channels used by this dataset (e.g. (0,), (0,1,2)).
    """

    frame_plan: FramePlan
    f_start: int
    f_stop: int
    chans: tuple[int, ...]


class SpecscoutDataset(Sequence[tuple[Array, FrameMeta]]):
    """
    Lazy, indexable dataset of rolling-window patches from a specscout Zarr cube.

    Parameters
    ----------
    zarr_path
        Path to the specscout Zarr store directory.
    start_utc, stop_utc
        UTC timestamps (``YYYYmmdd_HHMMSS``) defining the dataset time span.
        Windows are generated such that `[t_start, t_start + window)` stays in-bounds.
    window_seconds, step_seconds
        Window length and step size in seconds. These are converted once to samples
        via `specscout.core.plan_frames`.
    chans
        Cube channel indices to include. If an int, it is treated as a 1-tuple.
        - one channel -> samples have shape (T, F)
        - multiple channels -> samples have shape (T, F, C)
    f_start, f_stop
        Optional frequency slicing in *channel indices* (Python slice semantics).
        Defaults to full band.
    pipe
        Optional preprocessing pipeline applied to each patch after loading.
        The pipeline must conform to the standard specscout API:
            `pipe(x, meta) -> x`
        and is expected to encapsulate all conversions/whitening/clipping, etc.
    dtype
        dtype to cast loaded patches to (default float32).
    return_meta
        If True, `__getitem__` returns `(x, meta)`. If False, returns `x`.

    Notes
    -----
    - The Zarr store is opened once in the constructor.
    - Each `__getitem__` reads only the required patch from disk.
    - The dataset assumes the cube values are in "linear" space on read; if your
      pipeline expects something else, set the pipeline's `input_space` accordingly
      (or include a step like `step_safe_db` in the pipeline).
    """

    def __init__(
        self,
        zarr_path: str,
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

        # Normalize chans
        if isinstance(chans, int):
            chans = (chans,)
        chans = tuple(int(c) for c in chans)
        if len(chans) == 0:
            raise ValueError("chans must contain at least one channel index.")
        if any((c < 0 or c >= nchan_total) for c in chans):
            raise ValueError(f"chans out of bounds for cube with nchan={nchan_total}.")

        # Normalize freq slice
        if f_stop is None:
            f_stop = nfreq
        if not (0 <= f_start < f_stop <= nfreq):
            raise ValueError("Invalid frequency slice f_start/f_stop.")

        # Compute *once*: indices + window_n + step_n
        frame_plan = plan_frames(
            nt=nt,
            t0_unix_s=self._t0_unix_s,
            dt_s=self._dt_s,
            start_utc=start_utc,
            stop_utc=stop_utc,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            parse_utc=parse_utc,
        )

        # PatchSpec uses samples; reuse plan.window_n/plan.step_n (no duplication)
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
    def attrs(self) -> dict:
        """
        Zarr group attributes as a plain dict.

        Returns
        -------
        dict
            A shallow copy of the Zarr attributes for this store.
        """
        return dict(self._attrs)

    @property
    def plan(self) -> DatasetPlan:
        """
        The dataset plan (frame indices, window/step in samples, and channel selection).

        Returns
        -------
        DatasetPlan
            Stable description of dataset iteration and cube slicing.
        """
        return self._plan

    @property
    def dt_s(self) -> float:
        """
        Cadence in seconds per sample.

        Returns
        -------
        float
            The data cadence (seconds per time sample).
        """
        return self._dt_s

    @property
    def pipe(self) -> Optional[PreprocessPipeline]:
        """
        The preprocessing pipeline applied to each example (if any).

        Notes
        -----
        This is the canonical preprocessing hook across the project. The dataset
        itself does not perform unit conversions; all such logic should live in
        the pipeline steps.
        """
        return self._pipe

    def freq_axis(self) -> tuple[np.ndarray, str]:
        """
        Return the frequency axis and its label for this dataset.

        Returns
        -------
        freqs, x_label
            freqs
                Frequency axis array of shape (nfreq_total,).
            x_label
                "Frequency (MHz)" if df_mhz is present, else "Frequency channel".
        """
        _nt, nfreq, _nchan = self._cube.shape
        return freq_axis_from_attrs(self._attrs, nfreq)

    def __len__(self) -> int:
        """
        Number of examples/windows in the dataset.

        Returns
        -------
        int
            Total number of frames in the planned time range.
        """
        return int(self._plan.frame_plan.n_frames)

    def _frame_start_idx(self, idx: int) -> int:
        """
        Map dataset index -> cube start sample index.

        Parameters
        ----------
        idx
            Dataset index.

        Returns
        -------
        int
            Start sample index in the underlying cube.

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
        Construct FrameMeta without reading any data.

        Parameters
        ----------
        idx
            Dataset index (0 <= idx < len(self)).

        Returns
        -------
        FrameMeta
            Metadata for that frame. The UTC start time is computed from the
            dataset time axis.
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
        Load a patch by an explicit cube start index (`t_start_idx`).

        This is the preferred entry point for "out-of-band" visualization and
        outlier tooling, where you have a `FrameMeta` (or just a start index)
        but not necessarily the dataset index.

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
            If None, uses the dataset dtype.

        Returns
        -------
        (x, meta) or x
            If meta is requested, returns `(x, meta)`; otherwise returns `x`.

        Notes
        -----
        - This does not require `t_start_idx` to correspond to one of the planned
          dataset frames; it simply reads `window_n` samples starting there.
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
            x = self._pipe(x, meta)

        if return_meta is None:
            return_meta_flag = self._return_meta
        else:
            return_meta_flag = bool(return_meta)

        if return_meta_flag:
            return x, meta
        return x

    def __getitem__(self, idx: int):
        """
        Load one dataset example by dataset index.

        Parameters
        ----------
        idx
            Dataset index.

        Returns
        -------
        (x, meta) or x
            If `return_meta=True`, returns `(x, meta)`. Otherwise returns `x`.

        Notes
        -----
        `x` has shape:
        - (T, F) if one cube channel was requested
        - (T, F, C) if multiple cube channels were requested

        Preprocessing
        -------------
        If `pipe` is provided, it is applied after loading the patch and creating
        `FrameMeta`. The dataset itself does not do any unit conversions.
        """
        idx_i = int(idx)
        t_start_idx = self._frame_start_idx(idx_i)
        fp = self._plan.frame_plan

        patch = read_patch(
            self._cube,
            self._time_axis,
            self._spec,
            t_start_idx=t_start_idx,
            dtype=self._dtype,
        )
        x = patch.data

        meta = FrameMeta(
            t_start_idx=t_start_idx,
            t_end_idx=t_start_idx + fp.window_n,
            frame_idx=idx_i,
            start_time_utc=patch.start_time_utc,
            dt_s=fp.dt_s,
        )

        if self._pipe is not None:
            x = self._pipe(x, meta)

        if self._return_meta:
            return x, meta
        return x

    def iter(self) -> Iterator[tuple[Array, FrameMeta]]:
        """
        Iterate through all examples sequentially.

        Yields
        ------
        x, meta
            Each example plus metadata (always yields meta regardless of
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

        # return_meta=False
        if self._return_meta:
            return self[i][0]
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
            If True, also return a list of `FrameMeta` objects (one per sample).
        dtype
            If provided, cast the final stacked batch to this dtype.

        Returns
        -------
        batch
            Array of shape (B, T, F) or (B, T, F, C).
        metas
            Only returned if `return_meta=True`. List of length B.

        Notes
        -----
        Any preprocessing is applied per-example via `pipe` before stacking.
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
