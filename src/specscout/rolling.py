"""
Rolling / chunked outlier analysis utilities for specscout.

This module is the "glue" that makes quiet-PCA outlier scoring scale to years
of data without loading everything into memory.

It provides two main pieces:

1) RingBuffer
   A small, generic, fixed-capacity circular buffer that stores:
   - frames (NumPy arrays)
   - FrameMeta objects (or any metadata objects)
   in chronological order.

   The buffer is deliberately generic: you can set capacity to represent
   12h, 24h, 72h, etc., and you can store *masked* frames to reduce memory.

2) RollingPCARunner
   A streaming driver that:
   - iterates through a SpecscoutDataset sequentially,
   - maintains a ring buffer containing the most recent context window,
   - (once enough "future" is available for a centered context) refits a
     quiet-only PCA background model every stride_hours,
   - scores the next score_hours worth of frames using residual-based metrics,
     returning a per-hour ranking of "interesting" frames.

Centered context (important!)
-----------------------------
A centered context window requires data from both before and after the hour
being scored (e.g. 12h on either side for a 24h context). In a purely streaming
pass, this means scoring hour H can only occur once you've read data up to
H + context_hours/2.

This module implements that behavior naturally:
- you stream forward,
- after a warm-up of context_hours/2, the runner yields hourly results with
  a lag of context_hours/2.

Typical usage
-------------
You usually build the dataset to cover the analysis window PLUS padding for
centered context:

    # Want scores for [A, B)
    # Need ds to cover [A - half_ctx, B + half_ctx)
    ds = SpecscoutDataset(
        zarr_path,
        start_utc=A_minus_half_ctx,
        stop_utc=B_plus_half_ctx,
        ...
        pipe=pipe_safe_db,
        return_meta=True,
    )

Then run:

    qs = QuietSelector(method="p99", quiet_fraction=0.7, freq_mask=rfi_mask)
    bg = RollingPCABackground(k=128, center=True, freq_mask=rfi_mask)

    runner = RollingPCARunner(
        ds=ds,
        quiet_selector=qs,
        background=bg,
        context_hours=24,
        stride_hours=1,
        score_hours=1,
        gap_hours=0,
        n_quiet=None,          # or set an int for fixed compute
        k_pca=16,              # modes used for reconstruction during scoring
        score_kwargs=dict(method="topk_sum", topk=2048, positive_only=True),
    )

    for res in runner.run(analysis_start_utc=A, analysis_stop_utc=B):
        # res.scores, res.metas, res.rank_idx ...
        pass

Notes
-----
- This module assumes ds.iter() yields frames in chronological order.
- For memory efficiency, we recommend:
  - applying freq_mask before storing frames in the ring buffer
  - fitting/scoring only on unmasked channels (as you decided)
- Refitting randomized PCA once per hour over ~288 frames is usually fine.
  If it becomes too slow, we can optimize next:
  - fixed n_quiet
  - caching the flattened masked features
  - incremental/online PCA updates
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .core import FrameMeta, parse_utc
from .dataset import SpecscoutDataset
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
    Yield hour-aligned start times between [start, stop), stepping by stride_hours.
    """
    if stride_hours <= 0:
        raise ValueError("stride_hours must be positive.")
    step = timedelta(seconds=float(stride_hours) * 3600.0)

    h = _ceil_to_hour(start)
    while h < stop:
        yield h
        h = h + step


def _normalize_mask(mask: Optional[Array]) -> Optional[Array]:
    """Return mask as a 1D bool array, or None."""
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError(f"freq_mask must be 1D, got shape {m.shape}.")
    return m


def _masks_equal(a: Optional[Array], b: Optional[Array]) -> bool:
    """True if both None, or both arrays with identical values."""
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    # both not None
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
    """

    capacity: int
    dtype: Optional[np.dtype] = np.float32

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive.")
        self._frames: Optional[Array] = None
        self._metas: List[Optional[Any]] = [None] * self.capacity
        self._head: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def clear(self) -> None:
        self._frames = None
        self._metas = [None] * self.capacity
        self._head = 0
        self._size = 0

    def append(self, frame: Array, meta: Any) -> None:
        frame = np.asarray(frame)
        if self._frames is None:
            item_shape = frame.shape
            dt = np.dtype(frame.dtype) if self.dtype is None else np.dtype(self.dtype)
            self._frames = np.empty((self.capacity, *item_shape), dtype=dt)

        if self._frames is None:
            raise RuntimeError("Internal allocation failed (frames is None).")

        if frame.shape != self._frames.shape[1:]:
            raise ValueError(f"All frames must share the same shape. Got {frame.shape}, expected {self._frames.shape[1:]}.")

        self._frames[self._head] = frame
        self._metas[self._head] = meta

        self._head = (self._head + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def extend(self, frames: Array, metas: Sequence[Any]) -> None:
        frames = np.asarray(frames)
        if frames.ndim < 2:
            raise ValueError("frames must have shape (N, ...)")

        n = frames.shape[0]
        if len(metas) != n:
            raise ValueError(f"metas must have length {n}, got {len(metas)}")

        for i in range(n):
            self.append(frames[i], metas[i])

    def _chronological_indices(self) -> Array:
        if self._size == 0:
            return np.array([], dtype=int)

        start = (self._head - self._size) % self.capacity
        if start + self._size <= self.capacity:
            return np.arange(start, start + self._size, dtype=int)

        first = np.arange(start, self.capacity, dtype=int)
        second = np.arange(0, (start + self._size) % self.capacity, dtype=int)
        return np.concatenate([first, second])

    def get_all(self) -> Tuple[Array, List[Any]]:
        if self._size == 0:
            return np.empty((0,), dtype=np.float32), []

        if self._frames is None:
            raise RuntimeError("Buffer has metas but no frames array allocated.")

        idx = self._chronological_indices()
        frames = self._frames[idx]
        metas = [self._metas[i] for i in idx]
        return frames, metas  # type: ignore[return-value]

    def select_time_range(
        self,
        *,
        start: datetime,
        stop: datetime,
        time_getter: Optional[Any] = None,
    ) -> Tuple[Array, List[Any]]:
        if time_getter is None:

            def time_getter(m: Any) -> datetime:
                return m.start_time_utc

        frames, metas = self.get_all()
        if len(metas) == 0:
            return frames, metas

        keep: List[int] = []
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
    Results for one scored chunk (typically 1 hour).
    """

    hour_start: datetime
    hour_stop: datetime
    context_start: datetime
    context_stop: datetime
    n_context: int
    n_quiet: int
    scores: Array
    metas: List[FrameMeta]
    rank_idx: Array


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


@dataclass
class RollingPCARunner:
    """
    Streaming runner for centered rolling quiet-PCA outlier scoring.
    """

    ds: SpecscoutDataset
    quiet_selector: QuietSelector
    background: RollingPCABackground

    context_hours: float = 24.0
    stride_hours: float = 1.0
    score_hours: float = 1.0
    gap_hours: float = 0.0

    n_quiet: Optional[int] = None
    k_pca: int = 16
    score_kwargs: Optional[Dict[str, Any]] = None

    # NEW:
    freq_mask: Optional[Array] = None
    store_masked: bool = True

    # internal
    _buffer: Optional[RingBuffer] = None
    _qs: Optional[QuietSelector] = None
    _bg: Optional[RollingPCABackground] = None
    _storage_mask: Optional[Array] = None

    def __post_init__(self) -> None:
        """
        Establish a single effective mask and ensure QS/BG/storage are consistent.

        Rules
        -----
        - effective_mask precedence:
            runner.freq_mask > quiet_selector.freq_mask > background.freq_mask > None
        - If QS and BG both specify masks, they must match (else raise).
        - If store_masked=True:
            - frames are masked before storage using effective_mask
            - QS and BG are internally used with freq_mask=None (frames already masked)
        - If store_masked=False:
            - frames are stored full-band
            - QS and BG are internally used with freq_mask=effective_mask
        """
        qs_mask = _normalize_mask(self.quiet_selector.freq_mask)
        bg_mask = _normalize_mask(self.background.freq_mask)
        run_mask = _normalize_mask(self.freq_mask)

        if not _masks_equal(qs_mask, bg_mask) and (qs_mask is not None and bg_mask is not None):
            raise ValueError(
                "QuietSelector.freq_mask and RollingPCABackground.freq_mask are both set "
                "but do not match. Provide only one mask or ensure they are identical."
            )

        # precedence
        effective = run_mask
        if effective is None:
            effective = qs_mask
        if effective is None:
            effective = bg_mask
        self._storage_mask = effective

        # Build internal consistent copies used by run()
        if self.store_masked:
            # storage masked => downstream sees already-masked frames
            self._qs = replace(self.quiet_selector, freq_mask=None)
            # for background we can safely mutate or keep a reference; best is to mutate copy-like:
            self._bg = self.background
            self._bg.freq_mask = None
        else:
            # storage full-band => QS/BG both apply mask themselves (if any)
            self._qs = replace(self.quiet_selector, freq_mask=effective)
            self._bg = self.background
            self._bg.freq_mask = effective

    def _frames_per_hour(self) -> int:
        fp = self.ds.plan.frame_plan
        step_seconds = float(fp.step_n) * float(self.ds.dt_s)
        if step_seconds <= 0:
            raise ValueError("Dataset step_seconds must be positive.")
        return int(round(3600.0 / step_seconds))

    def _ensure_buffer(self) -> RingBuffer:
        if self._buffer is not None:
            return self._buffer
        frames_per_hour = self._frames_per_hour()
        cap = int(np.ceil(self.context_hours * frames_per_hour))
        self._buffer = RingBuffer(capacity=cap, dtype=np.float32)
        return self._buffer

    def _apply_mask_for_storage(self, frame: Array) -> Array:
        """
        If store_masked=True and an effective mask exists, apply it before storing.

        Convention:
        - mask is (F,) with True=keep
        - frame is (T, F) or (T, F, C)
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

    def n_steps(self, analysis_start_utc: str, analysis_stop_utc: str) -> int:
        """
        Return the number of scoring windows that will be produced by `run()`.

        Parameters
        ----------
        analysis_start_utc, analysis_stop_utc
            UTC strings in ``YYYYmmdd_HHMMSS`` format.

        Returns
        -------
        int
            Number of result windows (typically hours) that will be yielded.
        """
        start_dt = parse_utc(analysis_start_utc)
        stop_dt = parse_utc(analysis_stop_utc)

        if not (start_dt < stop_dt):
            raise ValueError("analysis_start_utc must be < analysis_stop_utc.")

        return len(
            list(
                _iter_hour_starts(
                    start_dt,
                    stop_dt,
                    stride_hours=self.stride_hours,
                )
            )
        )

    def run(
        self,
        *,
        analysis_start_utc: str,
        analysis_stop_utc: str,
    ) -> Iterator[HourlyResult]:
        start_dt = parse_utc(analysis_start_utc)
        stop_dt = parse_utc(analysis_stop_utc)
        if not (start_dt < stop_dt):
            raise ValueError("analysis_start_utc must be < analysis_stop_utc.")

        if self._qs is None or self._bg is None:
            raise RuntimeError("Runner not initialized correctly (missing internal qs/bg).")

        qs = self._qs
        bg = self._bg

        half_ctx = timedelta(seconds=float(self.context_hours) * 3600.0 / 2.0)
        score_td = timedelta(seconds=float(self.score_hours) * 3600.0)
        gap_td = timedelta(seconds=float(self.gap_hours) * 3600.0)

        if self.context_hours <= 0 or self.score_hours <= 0 or self.stride_hours <= 0:
            raise ValueError("context_hours, score_hours, stride_hours must be positive.")
        if self.score_hours > self.stride_hours:
            raise ValueError("score_hours must be <= stride_hours (non-overlap assumption).")

        buf = self._ensure_buffer()
        score_kwargs = dict(self.score_kwargs or {})
        k_pca = int(self.k_pca)

        hour_starts = list(_iter_hour_starts(start_dt, stop_dt, stride_hours=self.stride_hours))
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

                ctx_frames, ctx_metas_any = buf.select_time_range(start=ctx_start, stop=ctx_stop)
                ctx_metas: List[FrameMeta] = [m for m in ctx_metas_any]  # type: ignore

                if ctx_frames.shape[0] == 0:
                    next_h_idx += 1
                    continue

                # Optional donut exclusion for fit
                if self.gap_hours > 0:
                    half_gap = gap_td / 2.0
                    excl_start = hour_start - half_gap
                    excl_stop = hour_stop + half_gap

                    keep_idx: List[int] = []
                    for i_m, m in enumerate(ctx_metas):
                        t = m.start_time_utc
                        if not (excl_start <= t < excl_stop):
                            keep_idx.append(i_m)

                    if len(keep_idx) == 0:
                        next_h_idx += 1
                        continue

                    keep_idx_arr = np.asarray(keep_idx, dtype=int)
                    ctx_frames_fit = ctx_frames[keep_idx_arr]
                    # ctx_metas_fit = [ctx_metas[i] for i in keep_idx]
                else:
                    ctx_frames_fit = ctx_frames
                    # ctx_metas_fit = ctx_metas

                # Quiet selection
                quiet_scores = qs.scores(ctx_frames_fit)

                # Only keep frames with finite quietness scores as PCA candidates
                valid_quiet = np.isfinite(quiet_scores)
                if not np.any(valid_quiet):
                    next_h_idx += 1
                    continue

                ctx_frames_quiet = ctx_frames_fit[valid_quiet]
                quiet_scores_valid = quiet_scores[valid_quiet]

                order = np.argsort(quiet_scores_valid)  # lower = quieter
                if self.n_quiet is not None:
                    nq = int(max(1, min(int(self.n_quiet), order.size)))
                else:
                    qf = float(qs.quiet_fraction)
                    nq = int(max(1, np.floor(qf * order.size)))

                quiet_idx = order[:nq]
                quiet_frames = ctx_frames_quiet[quiet_idx]

                # Final guard: PCA fit requires fully finite frames after masking
                Xq = bg._prep(quiet_frames)
                ok_fit = np.isfinite(Xq).all(axis=1)
                if not np.any(ok_fit):
                    next_h_idx += 1
                    continue

                quiet_frames = quiet_frames[ok_fit]
                nq = int(quiet_frames.shape[0])

                # Fit PCA
                bg.fit(quiet_frames)

                # # Quiet selection
                # quiet_scores = qs.scores(ctx_frames_fit)
                # order = np.argsort(quiet_scores)  # lower = quieter
                # if self.n_quiet is not None:
                #     nq = int(max(1, min(int(self.n_quiet), order.size)))
                # else:
                #     qf = float(qs.quiet_fraction)
                #     nq = int(max(1, np.floor(qf * order.size)))
                # quiet_idx = order[:nq]
                # quiet_frames = ctx_frames_fit[quiet_idx]
                #
                # # Fit PCA
                # bg.fit(quiet_frames)

                # Score hour
                hour_frames, hour_metas_any = buf.select_time_range(start=hour_start, stop=hour_stop)
                hour_metas: List[FrameMeta] = [m for m in hour_metas_any]  # type: ignore
                if hour_frames.shape[0] == 0:
                    next_h_idx += 1
                    continue

                scores = bg.score(hour_frames, k_pca=k_pca, **score_kwargs)
                scores = np.asarray(scores)
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
# Optional helper: make padded UTC strings for building ds
# -----------------------------------------------------------------------------


def padded_utc_range(
    *,
    analysis_start_utc: str,
    analysis_stop_utc: str,
    context_hours: float,
) -> Tuple[str, str]:
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
