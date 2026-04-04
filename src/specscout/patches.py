"""
Patch extraction utilities for specscout Zarr cubes.

This module is responsible for *indexing and slicing*:

- Opening a specscout Zarr store and extracting time axis metadata
- Defining patch/window specifications in *sample index space*
- Reading individual patches efficiently (only the required Zarr chunks)
- Iterating over patch start indices for sliding-window workflows

This module deliberately does **not** implement preprocessing (bandpass subtraction,
whitening, clipping) or visualization. Those belong in `specscout.preprocess`
and `specscout.viz` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import zarr

from .core import parse_utc

if TYPE_CHECKING:
    from .preprocess import PreprocessPipeline
Array = np.ndarray


@dataclass(frozen=True)
class ZarrTimeAxis:
    """
    Time axis metadata for a specscout Zarr store.

    Parameters
    ----------
    t0_unix_s
        Unix timestamp in seconds corresponding to sample index 0.
    dt_s
        Cadence in seconds between consecutive time samples.
    """

    t0_unix_s: float
    dt_s: float

    def start_time_utc(self, t_start_idx: int) -> datetime:
        """
        Convert a start index into an aware UTC datetime.

        Parameters
        ----------
        t_start_idx
            Time sample index.

        Returns
        -------
        datetime
            Timezone-aware UTC datetime corresponding to `t_start_idx`.
        """
        return datetime.fromtimestamp(self.t0_unix_s + t_start_idx * self.dt_s, tz=UTC)


@dataclass(frozen=True)
class PatchSpec:
    """
    Specification for a rectangular patch extracted from the data cube.

    Parameters
    ----------
    window_n
        Number of time samples in each patch (T dimension).
    step_n
        Step between consecutive patch start indices (stride) in samples.
    f_start
        First frequency channel index to include (inclusive).
    f_stop
        Stop frequency channel index (exclusive). If None, uses all channels.
    chans
        Which cube channels to return (e.g. (0,) for pol00, (0,1,2) for 3-channel ML).
        If None, returns all channels present in the cube.
    """

    window_n: int = 128
    step_n: int = 16
    f_start: int = 0
    f_stop: Optional[int] = None
    chans: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class Patch:
    """
    A patch plus minimal metadata.

    Attributes
    ----------
    data
        Patch data array. Shape depends on `PatchSpec.chans`:
        - (T, F) if a single channel is selected
        - (T, F, C) if multiple channels are selected
    t_start_idx
        Start index (time) in the cube.
    start_time_utc
        UTC datetime corresponding to `t_start_idx`.
    """

    data: np.ndarray
    t_start_idx: int
    start_time_utc: datetime


def open_cube(zarr_path: str) -> tuple[zarr.Array, dict, ZarrTimeAxis]:
    """
    Open a specscout Zarr store.

    Parameters
    ----------
    zarr_path
        Path to the specscout ``.zarr`` directory.

    Returns
    -------
    cube
        Zarr array of shape (nt, nfreq, nchan).
    attrs
        Zarr group attributes as a plain dict.
    time_axis
        `ZarrTimeAxis` with (t0_unix_s, dt_s) extracted from attributes.
    """
    g = zarr.open_group(zarr_path, mode="r")
    cube = g["cube"]
    attrs = dict(g.attrs)
    t0 = float(attrs["t0_unix_seconds"])
    dt = float(attrs["dt_seconds"])
    return cube, attrs, ZarrTimeAxis(t0_unix_s=t0, dt_s=dt)


def read_patch(
    cube: zarr.Array,
    time_axis: ZarrTimeAxis,
    spec: PatchSpec,
    *,
    t_start_idx: int,
    dtype: np.dtype | None = np.float32,
) -> Patch:
    """
    Read a single patch from the cube.

    Parameters
    ----------
    cube
        Zarr array of shape (nt, nfreq, nchan).
    time_axis
        Time axis metadata for converting indices to UTC datetimes.
    spec
        Patch specification (window length, frequency slice, channel selection).
    t_start_idx
        Start index along time (samples).
    dtype
        If not None, cast the returned patch to this dtype.

    Returns
    -------
    Patch
        Patch data and metadata.
    """
    nt, nfreq, nchan = cube.shape

    f_stop = spec.f_stop if spec.f_stop is not None else nfreq
    if not (0 <= spec.f_start < f_stop <= nfreq):
        raise ValueError("Invalid frequency slice.")

    if not (0 <= t_start_idx < nt):
        raise ValueError("t_start_idx out of bounds.")

    t_end = t_start_idx + spec.window_n
    if t_end > nt:
        raise ValueError("Patch extends beyond end of cube.")

    slab = cube[t_start_idx:t_end, spec.f_start : f_stop, :]

    if spec.chans is None:
        data = slab
    else:
        if len(spec.chans) == 1:
            data = slab[:, :, spec.chans[0]]
        else:
            data = slab[:, :, list(spec.chans)]

    if dtype is not None:
        data = np.asarray(data, dtype=dtype)

    return Patch(
        data=data,
        t_start_idx=t_start_idx,
        start_time_utc=time_axis.start_time_utc(t_start_idx),
    )


def iter_patch_starts(
    nt_total: int,
    *,
    window_n: int,
    step_n: int,
    t_start_idx: int = 0,
    t_stop_idx: Optional[int] = None,
) -> Iterator[int]:
    """
    Yield start indices for a sliding window over [t_start_idx, t_stop_idx).

    Parameters
    ----------
    nt_total
        Total number of time samples in the cube.
    window_n
        Window length in samples.
    step_n
        Step size between consecutive windows in samples.
    t_start_idx
        First start index to consider.
    t_stop_idx
        Stop index for start positions (exclusive). If None, uses `nt_total`.

    Yields
    ------
    int
        Start indices such that [t, t+window_n) lies within the cube.
    """
    if window_n <= 0:
        raise ValueError("window_n must be positive.")
    if step_n <= 0:
        raise ValueError("step_n must be positive.")
    if t_start_idx < 0:
        raise ValueError("t_start_idx must be >= 0.")

    if t_stop_idx is None:
        t_stop_idx = nt_total

    last_start = min(t_stop_idx, nt_total) - window_n
    if last_start < t_start_idx:
        return

    t = t_start_idx
    while t <= last_start:
        yield t
        t += step_n


def iter_patches(
    zarr_path: str,
    spec: PatchSpec,
    *,
    hours: Optional[float] = None,
    t_start_idx: int = 0,
    transform: Optional[Callable[[Patch], Patch]] = None,
) -> Iterator[Patch]:
    """
    Iterate over patches from a Zarr store.

    Parameters
    ----------
    zarr_path
        Path to the specscout ``.zarr`` store.
    spec
        Patch specification.
    hours
        If provided, limits iteration to the first `hours` from `t_start_idx`.
    t_start_idx
        First patch start index.
    transform
        Optional function applied to each Patch (e.g., preprocessing). The callable
        must accept and return a `Patch`.

    Yields
    ------
    Patch
        Patch objects containing data and metadata.
    """
    cube, _attrs, time_axis = open_cube(zarr_path)

    nt_total = cube.shape[0]
    t_stop_idx = None
    if hours is not None:
        n = int(round(hours * 3600.0 / time_axis.dt_s))
        t_stop_idx = min(nt_total, t_start_idx + n)

    for ts in iter_patch_starts(
        nt_total,
        window_n=spec.window_n,
        step_n=spec.step_n,
        t_start_idx=t_start_idx,
        t_stop_idx=t_stop_idx,
    ):
        p = read_patch(cube, time_axis, spec, t_start_idx=ts)
        if transform is not None:
            p = transform(p)
        yield p


def _coerce_utc_timestamp(t: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """
    Normalize a time input to a timezone-aware UTC pandas Timestamp.

    Parameters
    ----------
    t
        Input time. Supported types:
        - str in ``YYYYmmdd_HHMMSS`` format
        - datetime
        - pandas.Timestamp

    Returns
    -------
    pandas.Timestamp
        UTC-normalized timestamp.
    """
    if isinstance(t, str):
        return pd.Timestamp(parse_utc(t), tz="UTC")
    if isinstance(t, pd.Timestamp):
        return pd.to_datetime(t, utc=True)
    if isinstance(t, datetime):
        return pd.Timestamp(t).tz_convert("UTC") if t.tzinfo else pd.Timestamp(t, tz="UTC")
    raise TypeError(f"Unsupported time type: {type(t)!r}")


def _normalize_chans(chans: int | Sequence[int]) -> tuple[int, ...]:
    """
    Normalize a channel selection to a tuple of integers.

    Parameters
    ----------
    chans
        Either a single integer channel index or a sequence of indices.

    Returns
    -------
    tuple[int, ...]
        Normalized channel tuple.
    """
    if isinstance(chans, int):
        return (int(chans),)
    return tuple(int(c) for c in chans)


def read_time_range(
    zarr_path: str | Path,
    *,
    start_utc: str | datetime | pd.Timestamp,
    stop_utc: str | datetime | pd.Timestamp,
    chans: int | Sequence[int],
    pipe: PreprocessPipeline | None = None,
    dtype: np.dtype = np.float32,
    squeeze_single_channel: bool = True,
) -> tuple[Array, pd.DatetimeIndex, Any]:
    """
    Read a contiguous time range from a specscout Zarr cube.

    This is a low-level, sample-based reader intended for quicklooks, ROI
    inspection, and general data extraction. It reads the requested interval
    directly from the underlying Zarr cube rather than reconstructing it from
    overlapping analysis frames.

    Parameters
    ----------
    zarr_path
        Path to a specscout Zarr store.
    start_utc
        Inclusive UTC start time. Supported forms:
        - ``YYYYmmdd_HHMMSS`` string
        - ``datetime``
        - ``pandas.Timestamp``
    stop_utc
        Exclusive UTC stop time. Supported forms are the same as `start_utc`.
    chans
        Channel selection from the last axis of the Zarr cube.
        Examples:
        - ``0`` for `pol00`
        - ``1`` for `pol11`
        - ``(0, 1)`` for dual-pol loading prior to a Stokes-I pipeline
        - ``(0, 1, 2, 3)`` for all stored channels
    pipe
        Optional preprocessing pipeline applied to the extracted block after
        reading. The pipeline receives the extracted array and a lightweight
        metadata object with fields such as:
        - ``start_time_utc``
        - ``t_start_idx``
        - ``dt_s``
        - ``frame_idx``

        This is intended to support transformations such as:
        - `step_stokes_i()`
        - `step_safe_db()`

    dtype
        Output dtype for the loaded array prior to any pipeline application.
    squeeze_single_channel
        If True and exactly one channel is requested, return shape ``(T, F)``
        instead of ``(T, F, 1)``.

    Returns
    -------
    data
        Extracted array. Shape is:
        - ``(T, F)`` if one channel is requested and `squeeze_single_channel=True`
        - ``(T, F, C)`` otherwise
    times
        UTC DatetimeIndex of length ``T`` giving the timestamp of each sample.
    meta
        Lightweight metadata object suitable for passing into preprocessing
        pipelines.

    Notes
    -----
    - The returned interval is clipped to the extant cube bounds.
    - If the requested time range lies fully outside the cube, an empty array
      and empty DatetimeIndex are returned.
    - Missing data internal to the cube remain as NaNs.
    - This helper is intentionally sample-based and does not use
      `SpecscoutDataset` frame construction.
    """
    zarr_path = Path(zarr_path)
    start_ts = _coerce_utc_timestamp(start_utc)
    stop_ts = _coerce_utc_timestamp(stop_utc)

    if not (start_ts < stop_ts):
        raise ValueError("start_utc must be earlier than stop_utc.")

    chans_t = _normalize_chans(chans)
    if len(chans_t) == 0:
        raise ValueError("At least one channel must be requested.")

    cube, attrs, _time_axis = open_cube(zarr_path)

    dt_s = float(attrs["dt_seconds"])
    t0_unix = float(attrs["t0_unix_seconds"])
    nt, nfreq, nchan_total = cube.shape

    for c in chans_t:
        if not (0 <= c < nchan_total):
            raise ValueError(f"Requested channel {c} out of bounds for cube with nchan={nchan_total}.")

    # Map requested times to sample indices.
    start_unix = start_ts.timestamp()
    stop_unix = stop_ts.timestamp()

    i0 = int(np.floor((start_unix - t0_unix) / dt_s))
    i1 = int(np.ceil((stop_unix - t0_unix) / dt_s))

    # Clip to cube bounds.
    i0_clip = max(0, min(nt, i0))
    i1_clip = max(0, min(nt, i1))

    if i1_clip <= i0_clip:
        empty_shape = (0, nfreq) if (len(chans_t) == 1 and squeeze_single_channel) else (0, nfreq, len(chans_t))
        data = np.empty(empty_shape, dtype=dtype)
        times = pd.DatetimeIndex([], tz="UTC")
        meta = SimpleNamespace(
            start_time_utc=start_ts.to_pydatetime(),
            t_start_idx=i0_clip,
            dt_s=dt_s,
            frame_idx=-1,
        )
        return data, times, meta

    # Read channel selection in a Zarr-friendly way.
    if len(chans_t) == 1:
        arr = np.asarray(cube[i0_clip:i1_clip, :, chans_t[0]], dtype=dtype)
        if not squeeze_single_channel:
            arr = arr[:, :, None]
    else:
        # If channels are contiguous, use a slice for efficient basic indexing.
        is_contiguous = all(chans_t[j] == chans_t[0] + j for j in range(len(chans_t)))

        if is_contiguous:
            c0 = chans_t[0]
            c1 = chans_t[-1] + 1
            arr = np.asarray(cube[i0_clip:i1_clip, :, c0:c1], dtype=dtype)
        else:
            # General fallback: read one channel at a time and stack.
            arr = np.stack(
                [np.asarray(cube[i0_clip:i1_clip, :, c], dtype=dtype) for c in chans_t],
                axis=2,
            )

    # Build per-sample UTC timestamps from cube metadata.
    sample_unix = t0_unix + np.arange(i0_clip, i1_clip, dtype=np.float64) * dt_s
    times = pd.to_datetime(sample_unix, unit="s", utc=True)

    meta = SimpleNamespace(
        start_time_utc=datetime.fromtimestamp(sample_unix[0], tz=UTC),
        t_start_idx=i0_clip,
        dt_s=dt_s,
        frame_idx=-1,
    )

    if pipe is not None:
        arr = pipe(arr, meta)

    return arr, times, meta
