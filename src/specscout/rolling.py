"""
Rolling / chunked outlier analysis utilities for specscout.

This module provides the orchestration layer that makes quiet-PCA outlier
scoring scale to long time spans without loading an entire season into memory.

It provides two main pieces:

1. `RingBuffer`
   A fixed-capacity circular buffer storing frames and their metadata in
   chronological order.

2. `RollingPCARunner`
   A streaming driver that:
   - iterates through a `SpecscoutDataset` sequentially
   - maintains a ring buffer covering the most recent rolling context
   - refits a quiet-only PCA background model on each scoring step
   - scores the next `score_hours` worth of frames using residual-based metrics

Centered context
----------------
This runner assumes a *centered* context window. For example, with a 24-hour
context and 1-hour scoring chunks, scoring a chunk beginning at time `H`
requires data from approximately `H - 12 h` to `H + 12 h`.

In a forward streaming pass this naturally introduces a lag:
the runner can only emit results for time `H` once it has read far enough into
the future to fill the centered context.

Design notes
------------
- The dataset supplied to this module should typically already include padding
  on both sides of the requested analysis interval. Use `padded_utc_range()`
  to construct that padded range.
- This module assumes `ds.iter()` yields frames in chronological order.
- Overlapping scored windows are intentionally unsupported:
  `score_hours` must be less than or equal to `stride_hours`.

Typical usage
-------------
Build a dataset covering the analysis range plus centered-context padding:

    ds = SpecscoutDataset(
        zarr_path,
        start_utc=A_minus_half_ctx,
        stop_utc=B_plus_half_ctx,
        ...,
        pipe=pipe_safe_db,
        return_meta=True,
    )

Then run:

    qs = QuietSelector(method="p99", quiet_fraction=0.3, freq_mask=rfi_mask)
    bg = RollingPCABackground(k=128, center=True, freq_mask=rfi_mask)

    runner = RollingPCARunner(
        ds=ds,
        quiet_selector=qs,
        background=bg,
        context_hours=24,
        stride_hours=1,
        score_hours=1,
        gap_hours=0,
        n_quiet=None,
        k_pca=16,
        score_kwargs={
            "method": "topk_sum",
            "topk": 2048,
            "positive_only": True,
        },
    )

    for res in runner.run(analysis_start_utc=A, analysis_stop_utc=B):
        ...
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .core import parse_utc
from .dataset import FrameMeta, SpecscoutDataset
from .outlier import QuietSelector, RollingPCABackground

Array = np.ndarray


# -----------------------------------------------------------------------------
# Small time helpers
# -----------------------------------------------------------------------------


def _floor_to_hour(t: datetime) -> datetime:
    """Floor a datetime to the start of its hour."""
    return t.replace(minute=0, second=0, microsecond=0)


def _ceil_to_hour(t: datetime) -> datetime:
    """Ceil a datetime to the start of the next hour if not already aligned."""
    f = _floor_to_hour(t)
    return f if f == t else (f + timedelta(hours=1))


def _iter_hour_starts(
    start: datetime,
    stop: datetime,
    *,
    stride_hours: float,
) -> Iterator[datetime]:
    """
    Yield scoring-window start times in ``[start, stop)``.

    Starts are aligned to the next hour boundary at or after `start`, then
    stepped by `stride_hours`.
    """
    if stride_hours <= 0:
        raise ValueError("stride_hours must be positive.")

    step = timedelta(seconds=float(stride_hours) * 3600.0)
    h = _ceil_to_hour(start)

    while h < stop:
        yield h
        h = h + step


def _normalize_mask(mask: Array | None) -> Array | None:
    """Return mask as a 1D bool array, or None."""
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError(f"freq_mask must be 1D, got shape {m.shape}.")
    return m


def _masks_equal(a: Array | None, b: Array | None) -> bool:
    """True if both None, or both arrays with identical values."""
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    return aa.shape == bb.shape and bool(np.all(aa == bb))


# -----------------------------------------------------------------------------
# Ring buffer
# -----------------------------------------------------------------------------


@dataclass
class RingBuffer:
    """
    Fixed-capacity ring buffer for frames + metadata.

    Parameters
    ----------
    capacity
        Maximum number of frames stored.
    dtype
        Optional storage dtype. If None, preserve the dtype of the first
        appended frame.
    """

    capacity: int
    dtype: np.dtype | None = np.float32

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive.")
        self._frames: Array | None = None
        self._metas: list[Any | None] = [None] * self.capacity
        self._head: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        """True if the buffer has reached capacity."""
        return self._size == self.capacity

    def clear(self) -> None:
        """Reset the buffer contents."""
        self._frames = None
        self._metas = [None] * self.capacity
        self._head = 0
        self._size = 0

    def append(self, frame: Array, meta: Any) -> None:
        """
        Append one frame and its metadata to the buffer.
        """
        frame = np.asarray(frame)

        if self._frames is None:
            item_shape = frame.shape
            dt = np.dtype(frame.dtype) if self.dtype is None else np.dtype(self.dtype)
            self._frames = np.empty((self.capacity, *item_shape), dtype=dt)

        if self._frames is None:
            raise RuntimeError("Internal allocation failed (_frames is None).")

        if frame.shape != self._frames.shape[1:]:
            raise ValueError(f"All frames must share the same shape. Got {frame.shape}, expected {self._frames.shape[1:]}.")

        self._frames[self._head] = frame
        self._metas[self._head] = meta

        self._head = (self._head + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def extend(self, frames: Array, metas: Sequence[Any]) -> None:
        """
        Append multiple frames and metadata objects.
        """
        frames = np.asarray(frames)
        if frames.ndim < 2:
            raise ValueError("frames must have shape (N, ...).")

        n = frames.shape[0]
        if len(metas) != n:
            raise ValueError(f"metas must have length {n}, got {len(metas)}.")

        for i in range(n):
            self.append(frames[i], metas[i])

    def _chronological_indices(self) -> Array:
        """
        Return the internal storage indices corresponding to chronological order.
        """
        if self._size == 0:
            return np.array([], dtype=int)

        start = (self._head - self._size) % self.capacity
        if start + self._size <= self.capacity:
            return np.arange(start, start + self._size, dtype=int)

        first = np.arange(start, self.capacity, dtype=int)
        second = np.arange(0, (start + self._size) % self.capacity, dtype=int)
        return np.concatenate([first, second])

    def get_all(self) -> tuple[Array, list[Any]]:
        """
        Return all buffered frames and metadata in chronological order.
        """
        if self._size == 0:
            if self._frames is not None:
                return self._frames[:0], []
            return np.empty((0,), dtype=np.float32), []

        if self._frames is None:
            raise RuntimeError("Buffer has metadata but no allocated frame array.")

        idx = self._chronological_indices()
        frames = self._frames[idx]
        metas = [self._metas[i] for i in idx]
        return frames, metas  # type: ignore[return-value]

    def select_time_range(
        self,
        *,
        start: datetime,
        stop: datetime,
        time_getter: Callable[[Any], datetime] | None = None,
    ) -> tuple[Array, list[Any]]:
        """
        Select buffered entries with timestamps in ``[start, stop)``.

        Parameters
        ----------
        start, stop
            Time bounds.
        time_getter
            Optional callable extracting a datetime from each metadata object.
            Defaults to `meta.start_time_utc`.

        Returns
        -------
        frames, metas
            Chronologically ordered subset.
        """
        if time_getter is None:

            def time_getter(m: Any) -> datetime:
                return m.start_time_utc

        frames, metas = self.get_all()
        if len(metas) == 0:
            return frames, metas

        keep: list[int] = []
        for i, m in enumerate(metas):
            t = time_getter(m)
            if start <= t < stop:
                keep.append(i)

        if len(keep) == 0:
            return frames[:0], []

        keep_idx = np.asarray(keep, dtype=int)
        return frames[keep_idx], [metas[i] for i in keep]


# -----------------------------------------------------------------------------
# Results container
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HourlyResult:
    """
    Results for one scored chunk.

    Parameters
    ----------
    hour_start, hour_stop
        Scored interval.
    context_start, context_stop
        Centered context interval used to fit the quiet PCA background.
    n_context
        Number of frames available in the full context window.
    n_quiet
        Number of frames actually used to fit PCA after quiet selection and
        finite-data filtering.
    scores
        Per-frame outlier scores for the scored interval.
    metas
        Metadata objects corresponding to `scores`.
    rank_idx
        Descending score order for the scored interval.
    """

    hour_start: datetime
    hour_stop: datetime
    context_start: datetime
    context_stop: datetime
    n_context: int
    n_quiet: int
    scores: Array
    metas: list[FrameMeta]
    rank_idx: Array


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


@dataclass
class RollingPCARunner:
    """
    Streaming runner for centered rolling quiet-PCA outlier scoring.

    Parameters
    ----------
    ds
        Input dataset. Must yield frames in chronological order.
    quiet_selector
        Quiet-frame ranking object.
    background
        PCA background model prototype.
    context_hours
        Width of the centered PCA-fit context window.
    stride_hours
        Step between successive scored windows.
    score_hours
        Width of each scored window.
    gap_hours
        Optional "donut" exclusion around the scored interval when selecting
        quiet frames for PCA fitting.
    n_quiet
        Optional fixed number of quiet frames to use. If None, uses
        `quiet_selector.quiet_fraction`.
    k_pca
        Number of PCA modes used during reconstruction and scoring.
    score_kwargs
        Metric configuration passed to `RollingPCABackground.score()`.
    freq_mask
        Optional explicit frequency mask. If provided, it takes precedence over
        masks already attached to `quiet_selector` or `background`.
    store_masked
        If True, apply the effective mask before storing frames in the ring
        buffer. This reduces memory and ensures downstream PCA/scoring sees
        already-masked frames.

    Notes
    -----
    - This runner assumes non-overlapping scored windows, so
      `score_hours <= stride_hours` is required.
    - Centered context means results are emitted with a lag of roughly
      `context_hours / 2` in a forward streaming pass.
    - The runner keeps internal copies of the quiet selector and background
      model so user-supplied objects are not mutated.
    """

    ds: SpecscoutDataset
    quiet_selector: QuietSelector
    background: RollingPCABackground

    context_hours: float = 24.0
    stride_hours: float = 1.0
    score_hours: float = 1.0
    gap_hours: float = 0.0

    n_quiet: int | None = None
    k_pca: int = 16
    score_kwargs: dict[str, Any] | None = None

    freq_mask: Array | None = None
    store_masked: bool = True

    _buffer: RingBuffer | None = None
    _qs: QuietSelector | None = None
    _bg: RollingPCABackground | None = None
    _storage_mask: Array | None = None

    def __post_init__(self) -> None:
        """
        Establish a single effective mask and create internal QS/BG copies.

        Mask precedence
        ---------------
        `runner.freq_mask` > `quiet_selector.freq_mask` > `background.freq_mask`

        Behavior
        --------
        - If both QuietSelector and RollingPCABackground specify masks and they
          differ, a ValueError is raised.
        - If `store_masked=True`, frames are masked before storage and the
          internal QS/BG copies operate with `freq_mask=None`.
        - If `store_masked=False`, frames are stored full-band and the internal
          QS/BG copies use the effective mask.
        """
        qs_mask = _normalize_mask(self.quiet_selector.freq_mask)
        bg_mask = _normalize_mask(self.background.freq_mask)
        run_mask = _normalize_mask(self.freq_mask)

        if not _masks_equal(qs_mask, bg_mask) and (qs_mask is not None and bg_mask is not None):
            raise ValueError(
                "QuietSelector.freq_mask and RollingPCABackground.freq_mask are both "
                "set but do not match. Provide only one mask or ensure they are "
                "identical."
            )

        effective = run_mask
        if effective is None:
            effective = qs_mask
        if effective is None:
            effective = bg_mask

        self._storage_mask = effective

        if self.store_masked:
            self._qs = replace(self.quiet_selector, freq_mask=None)
            self._bg = replace(
                self.background,
                freq_mask=None,
                mu_=None,
                Vt_=None,
                S_=None,
            )
        else:
            self._qs = replace(self.quiet_selector, freq_mask=effective)
            self._bg = replace(
                self.background,
                freq_mask=effective,
                mu_=None,
                Vt_=None,
                S_=None,
            )

    def _frames_per_hour(self) -> int:
        """
        Estimate the number of dataset frames per hour.

        Uses ceiling so context-buffer capacity is never underestimated.
        """
        fp = self.ds.plan.frame_plan
        step_seconds = float(fp.step_n) * float(self.ds.dt_s)
        if step_seconds <= 0:
            raise ValueError("Dataset step size must be positive.")
        return int(np.ceil(3600.0 / step_seconds))

    def _ensure_buffer(self) -> RingBuffer:
        """
        Allocate the ring buffer lazily using the configured context width.
        """
        if self._buffer is not None:
            return self._buffer

        frames_per_hour = self._frames_per_hour()
        cap = int(np.ceil(self.context_hours * frames_per_hour))
        self._buffer = RingBuffer(capacity=cap, dtype=np.float32)
        return self._buffer

    def _apply_mask_for_storage(self, frame: Array) -> Array:
        """
        Apply the effective storage mask if `store_masked=True`.

        Parameters
        ----------
        frame
            Frame of shape `(T, F)` or `(T, F, C)`.

        Returns
        -------
        frame_out
            Possibly masked frame.
        """
        if not self.store_masked:
            return np.asarray(frame)

        m = self._storage_mask
        if m is None:
            return np.asarray(frame)

        m = np.asarray(m, dtype=bool)
        frame = np.asarray(frame)

        if frame.ndim == 2:
            if m.shape[0] != frame.shape[1]:
                raise ValueError("freq_mask shape does not match frame F dimension.")
            return frame[:, m]

        if frame.ndim == 3:
            if m.shape[0] != frame.shape[1]:
                raise ValueError("freq_mask shape does not match frame F dimension.")
            return frame[:, m, :]

        raise ValueError("Expected frame shape (T, F) or (T, F, C).")

    def _select_quiet_frames(
        self,
        ctx_frames_fit: Array,
        *,
        qs: QuietSelector,
        bg: RollingPCABackground,
    ) -> tuple[Array | None, int]:
        """
        Select quiet frames that are fit-ready for PCA.

        This helper intentionally extends `QuietSelector` behavior with
        runner-specific logic:

        - optional fixed `n_quiet`
        - finite-score filtering
        - final feature-space finite-data validation for PCA fitting

        Parameters
        ----------
        ctx_frames_fit
            Candidate context frames after any donut exclusion.
        qs
            Internal quiet selector.
        bg
            Internal background model.

        Returns
        -------
        quiet_frames, n_quiet
            `quiet_frames` is None if no usable PCA training set exists.
        """
        quiet_scores = qs.scores(ctx_frames_fit)

        valid_quiet = np.isfinite(quiet_scores)
        if not np.any(valid_quiet):
            return None, 0

        ctx_frames_quiet = ctx_frames_fit[valid_quiet]
        quiet_scores_valid = quiet_scores[valid_quiet]

        order = np.argsort(quiet_scores_valid)

        if self.n_quiet is not None:
            nq = int(max(1, min(int(self.n_quiet), order.size)))
        else:
            qf = float(qs.quiet_fraction)
            nq = int(max(1, np.floor(qf * order.size)))

        quiet_frames = ctx_frames_quiet[order[:nq]]

        Xq = bg._prep(quiet_frames)
        ok_fit = np.isfinite(Xq).all(axis=1)
        if not np.any(ok_fit):
            return None, 0

        quiet_frames = quiet_frames[ok_fit]
        return quiet_frames, int(quiet_frames.shape[0])

    def n_steps(self, analysis_start_utc: str, analysis_stop_utc: str) -> int:
        """
        Return the number of scoring windows produced by `run()`.

        Parameters
        ----------
        analysis_start_utc, analysis_stop_utc
            UTC strings in ``YYYYmmdd_HHMMSS`` format.

        Returns
        -------
        int
            Number of scored windows.
        """
        start_dt = parse_utc(analysis_start_utc)
        stop_dt = parse_utc(analysis_stop_utc)

        if not (start_dt < stop_dt):
            raise ValueError("analysis_start_utc must be < analysis_stop_utc.")

        return sum(
            1
            for _ in _iter_hour_starts(
                start_dt,
                stop_dt,
                stride_hours=self.stride_hours,
            )
        )

    def run(
        self,
        *,
        analysis_start_utc: str,
        analysis_stop_utc: str,
    ) -> Iterator[HourlyResult]:
        """
        Run centered rolling quiet-PCA scoring over an analysis interval.

        Parameters
        ----------
        analysis_start_utc, analysis_stop_utc
            Requested scored interval in ``YYYYmmdd_HHMMSS`` format.

        Yields
        ------
        HourlyResult
            One result object per scored window.

        Notes
        -----
        - The dataset should usually cover the analysis interval plus padding on
          both sides for centered context.
        - Results are only emitted once enough future data has been seen to fill
          the centered context window.
        - Scored windows containing no frames are skipped.
        - Context windows with no usable quiet training frames are skipped.
        - Overlapping scored windows are intentionally unsupported.
        """
        start_dt = parse_utc(analysis_start_utc)
        stop_dt = parse_utc(analysis_stop_utc)
        if not (start_dt < stop_dt):
            raise ValueError("analysis_start_utc must be < analysis_stop_utc.")

        if self._qs is None or self._bg is None:
            raise RuntimeError("Runner not initialized correctly (missing internal quiet selector or background model).")

        qs = self._qs
        bg = self._bg

        half_ctx = timedelta(seconds=float(self.context_hours) * 3600.0 / 2.0)
        score_td = timedelta(seconds=float(self.score_hours) * 3600.0)
        gap_td = timedelta(seconds=float(self.gap_hours) * 3600.0)

        if self.context_hours <= 0 or self.score_hours <= 0 or self.stride_hours <= 0:
            raise ValueError("context_hours, score_hours, and stride_hours must be positive.")
        if self.score_hours > self.stride_hours:
            raise ValueError(
                "score_hours must be <= stride_hours because overlapping scored windows are not currently supported."
            )

        buf = self._ensure_buffer()
        score_kwargs = dict(self.score_kwargs or {})
        k_pca = int(self.k_pca)

        hour_starts = list(
            _iter_hour_starts(
                start_dt,
                stop_dt,
                stride_hours=self.stride_hours,
            )
        )
        if len(hour_starts) == 0:
            return iter(())

        next_h_idx = 0

        for frame, meta in self.ds.iter():
            frame_store = self._apply_mask_for_storage(frame)
            buf.append(frame_store.astype(np.float32, copy=False), meta)

            current_t = meta.start_time_utc

            while next_h_idx < len(hour_starts):
                h = hour_starts[next_h_idx]
                if current_t < (h + half_ctx):
                    break

                hour_start = h
                hour_stop = h + score_td
                ctx_start = h - half_ctx
                ctx_stop = h + half_ctx

                ctx_frames, ctx_metas_any = buf.select_time_range(
                    start=ctx_start,
                    stop=ctx_stop,
                )
                ctx_metas: list[FrameMeta] = [m for m in ctx_metas_any]  # type: ignore

                if ctx_frames.shape[0] == 0:
                    next_h_idx += 1
                    continue

                if self.gap_hours > 0:
                    half_gap = gap_td / 2.0
                    excl_start = hour_start - half_gap
                    excl_stop = hour_stop + half_gap

                    keep_idx: list[int] = []
                    for i_m, m in enumerate(ctx_metas):
                        t = m.start_time_utc
                        if not (excl_start <= t < excl_stop):
                            keep_idx.append(i_m)

                    if len(keep_idx) == 0:
                        next_h_idx += 1
                        continue

                    keep_idx_arr = np.asarray(keep_idx, dtype=int)
                    ctx_frames_fit = ctx_frames[keep_idx_arr]
                else:
                    ctx_frames_fit = ctx_frames

                quiet_frames, nq = self._select_quiet_frames(
                    ctx_frames_fit,
                    qs=qs,
                    bg=bg,
                )
                if quiet_frames is None:
                    next_h_idx += 1
                    continue

                bg.fit(quiet_frames)

                hour_frames, hour_metas_any = buf.select_time_range(
                    start=hour_start,
                    stop=hour_stop,
                )
                hour_metas: list[FrameMeta] = [m for m in hour_metas_any]  # type: ignore
                if hour_frames.shape[0] == 0:
                    next_h_idx += 1
                    continue

                scores = np.asarray(
                    bg.score(
                        hour_frames,
                        k_pca=k_pca,
                        metric_kwargs=score_kwargs,
                    )
                )
                rank_idx = np.argsort(scores)[::-1]

                yield HourlyResult(
                    hour_start=hour_start,
                    hour_stop=hour_stop,
                    context_start=ctx_start,
                    context_stop=ctx_stop,
                    n_context=int(ctx_frames.shape[0]),
                    n_quiet=int(nq),
                    scores=scores,
                    metas=hour_metas,
                    rank_idx=rank_idx,
                )

                next_h_idx += 1

            if next_h_idx >= len(hour_starts):
                break


# -----------------------------------------------------------------------------
# Helper: make padded UTC strings for building ds
# -----------------------------------------------------------------------------


def padded_utc_range(
    *,
    analysis_start_utc: str,
    analysis_stop_utc: str,
    context_hours: float,
) -> tuple[str, str]:
    """
    Expand an analysis interval by half the centered context on both sides.

    Parameters
    ----------
    analysis_start_utc, analysis_stop_utc
        Requested scored interval in ``YYYYmmdd_HHMMSS`` format.
    context_hours
        Width of the centered context window.

    Returns
    -------
    ds_start_utc, ds_stop_utc
        Padded UTC strings suitable for constructing a dataset that can support
        centered-context scoring over the requested interval.
    """
    start_dt = parse_utc(analysis_start_utc)
    stop_dt = parse_utc(analysis_stop_utc)
    if not (start_dt < stop_dt):
        raise ValueError("analysis_start_utc must be < analysis_stop_utc.")

    half_ctx = timedelta(seconds=float(context_hours) * 3600.0 / 2.0)
    ds_start = start_dt - half_ctx
    ds_stop = stop_dt + half_ctx

    return (
        ds_start.strftime("%Y%m%d_%H%M%S"),
        ds_stop.strftime("%Y%m%d_%H%M%S"),
    )
