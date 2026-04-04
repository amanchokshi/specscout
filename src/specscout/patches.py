"""
Low-level cube access and patch extraction utilities for specscout Zarr stores.

This module is responsible for index-based and time-based reads from specscout
Zarr cubes:

- opening a specscout Zarr store and extracting time-axis metadata
- defining rectangular patch specifications in sample-index space
- reading individual patches efficiently
- reading contiguous time ranges directly from the underlying cube

This module deliberately does not implement preprocessing, PCA modeling, or
visualization. Those belong in `specscout.preprocess`, `specscout.outlier`,
`specscout.rolling`, and `specscout.roi_search`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

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
    Time-axis metadata for a specscout Zarr store.

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
        return datetime.fromtimestamp(
            self.t0_unix_s + t_start_idx * self.dt_s,
            tz=UTC,
        )


@dataclass(frozen=True)
class PatchSpec:
    """
    Specification for a rectangular patch extracted from the data cube.

    Parameters
    ----------
    window_n
        Number of time samples in each patch (T dimension).
    step_n
        Legacy stride field retained for compatibility with older callers.
        It is not used by `read_patch`.
    f_start
        First frequency channel index to include (inclusive).
    f_stop
        Stop frequency channel index (exclusive). If None, uses all channels.
    chans
        Which cube channels to return (e.g. ``(0,)`` for pol00,
        ``(0, 1, 2)`` for a 3-channel product). If None, returns all channels
        present in the cube.
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
        - ``(T, F)`` if a single channel is selected
        - ``(T, F, C)`` if multiple channels are selected
    t_start_idx
        Start index (time) in the cube.
    start_time_utc
        UTC datetime corresponding to `t_start_idx`.
    """

    data: np.ndarray
    t_start_idx: int
    start_time_utc: datetime


@dataclass(frozen=True)
class RangeMeta:
    """
    Minimal metadata for a contiguous time-range read.

    Parameters
    ----------
    start_time_utc
        UTC datetime corresponding to the first returned sample.
    t_start_idx
        Start sample index in the cube for the returned interval.
    dt_s
        Cadence in seconds per sample.
    frame_idx
        Placeholder frame index for compatibility with transform call
        signatures. Defaults to -1 because time-range reads are not tied to a
        dataset frame.
    """

    start_time_utc: datetime
    t_start_idx: int
    dt_s: float
    frame_idx: int = -1


def open_cube(
    zarr_path: str | Path,
) -> tuple[zarr.Array, dict[str, object], ZarrTimeAxis]:
    """
    Open a specscout Zarr store.

    Parameters
    ----------
    zarr_path
        Path to the specscout ``.zarr`` directory.

    Returns
    -------
    cube
        Zarr array of shape ``(nt, nfreq, nchan)``.
    attrs
        Zarr group attributes as a plain dict.
    time_axis
        `ZarrTimeAxis` with ``(t0_unix_s, dt_s)`` extracted from attributes.
    """
    g = zarr.open_group(Path(zarr_path), mode="r")
    cube = g["cube"]
    attrs = dict(g.attrs)
    t0 = float(attrs["t0_unix_seconds"])
    dt = float(attrs["dt_seconds"])
    return cube, attrs, ZarrTimeAxis(t0_unix_s=t0, dt_s=dt)


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


def _validate_chans(chans: Optional[tuple[int, ...]], nchan: int) -> None:
    """
    Validate an optional channel selection against cube bounds.
    """
    if chans is None:
        return
    if len(chans) == 0:
        raise ValueError("spec.chans must be non-empty if provided.")
    if any((c < 0 or c >= nchan) for c in chans):
        raise ValueError(f"Channel selection {chans} out of bounds for cube with nchan={nchan}.")


def _read_channel_selection(
    cube: zarr.Array,
    *,
    t_slice: slice,
    f_slice: slice,
    chans: tuple[int, ...],
    dtype: np.dtype | None,
    squeeze_single_channel: bool,
) -> np.ndarray:
    """
    Read a channel selection from a cube slice in a Zarr-friendly way.

    Parameters
    ----------
    cube
        Zarr cube array of shape ``(nt, nfreq, nchan)``.
    t_slice
        Time slice.
    f_slice
        Frequency slice.
    chans
        Channel indices to read. Must be non-empty and already validated.
    dtype
        Optional output dtype cast.
    squeeze_single_channel
        If True, a single selected channel is returned as ``(T, F)`` rather
        than ``(T, F, 1)``.

    Returns
    -------
    ndarray
        Loaded array.
    """
    if len(chans) == 1:
        arr = np.asarray(cube[t_slice, f_slice, chans[0]], dtype=dtype)
        if not squeeze_single_channel:
            arr = arr[:, :, None]
        return arr

    is_contiguous = all(chans[j] == chans[0] + j for j in range(len(chans)))
    if is_contiguous:
        c0 = chans[0]
        c1 = chans[-1] + 1
        return np.asarray(cube[t_slice, f_slice, c0:c1], dtype=dtype)

    return np.stack(
        [np.asarray(cube[t_slice, f_slice, c], dtype=dtype) for c in chans],
        axis=2,
    )


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
        Zarr array of shape ``(nt, nfreq, nchan)``.
    time_axis
        Time-axis metadata for converting indices to UTC datetimes.
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

    _validate_chans(spec.chans, nchan)

    if not (0 <= t_start_idx < nt):
        raise ValueError("t_start_idx out of bounds.")

    t_end = t_start_idx + spec.window_n
    if t_end > nt:
        raise ValueError("Patch extends beyond end of cube.")

    t_slice = slice(t_start_idx, t_end)
    f_slice = slice(spec.f_start, f_stop)

    if spec.chans is None:
        data = np.asarray(cube[t_slice, f_slice, :], dtype=dtype)
    else:
        data = _read_channel_selection(
            cube,
            t_slice=t_slice,
            f_slice=f_slice,
            chans=spec.chans,
            dtype=dtype,
            squeeze_single_channel=True,
        )

    return Patch(
        data=data,
        t_start_idx=t_start_idx,
        start_time_utc=time_axis.start_time_utc(t_start_idx),
    )


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
        return pd.Timestamp(parse_utc(t))
    if isinstance(t, pd.Timestamp):
        return pd.to_datetime(t, utc=True)
    if isinstance(t, datetime):
        ts = pd.Timestamp(t)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    raise TypeError(f"Unsupported time type: {type(t)!r}")


def read_time_range(
    zarr_path: str | Path,
    *,
    start_utc: str | datetime | pd.Timestamp,
    stop_utc: str | datetime | pd.Timestamp,
    chans: int | Sequence[int],
    pipe: PreprocessPipeline | None = None,
    dtype: np.dtype = np.float32,
    squeeze_single_channel: bool = True,
) -> tuple[Array, pd.DatetimeIndex, RangeMeta]:
    """
    Read a contiguous time range from a specscout Zarr cube.

    This is a low-level, sample-based reader intended for quicklooks, ROI
    inspection, and general data extraction. It reads the requested interval
    directly from the underlying cube rather than reconstructing it from
    overlapping analysis frames.

    Parameters
    ----------
    zarr_path
        Path to a specscout Zarr store.
    start_utc
        Inclusive UTC start time. Supported forms:
        - ``YYYYmmdd_HHMMSS`` string
        - `datetime`
        - `pandas.Timestamp`
    stop_utc
        Exclusive UTC stop time. Supported forms are the same as `start_utc`.
    chans
        Channel selection from the last axis of the Zarr cube.
        Examples:
        - ``0`` for pol00
        - ``1`` for pol11
        - ``(0, 1)`` for dual-pol loading prior to a Stokes-I pipeline
        - ``(0, 1, 2, 3)`` for all stored channels
    pipe
        Optional preprocessing pipeline applied to the extracted block after
        reading.
    dtype
        Output dtype for the loaded array prior to any pipeline application.
    squeeze_single_channel
        If True and exactly one channel is requested, return shape ``(T, F)``
        instead of ``(T, F, 1)``.

    Returns
    -------
    data
        Extracted array. Shape is:
        - ``(T, F)`` if one channel is requested and
          `squeeze_single_channel=True`
        - ``(T, F, C)`` otherwise
    times
        UTC `DatetimeIndex` of length ``T`` giving the timestamp of each sample.
    meta
        `RangeMeta` suitable for passing into preprocessing pipelines.

    Notes
    -----
    - The returned interval is clipped to the extant cube bounds.
    - If the requested time range lies fully outside the cube, an empty array
      and empty `DatetimeIndex` are returned.
    - Missing data internal to the cube remain as NaNs.
    - This helper is intentionally sample-based and does not use
      `SpecscoutDataset` frame construction.
    """
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

    _validate_chans(chans_t, nchan_total)

    start_unix = start_ts.timestamp()
    stop_unix = stop_ts.timestamp()

    i0 = int(np.floor((start_unix - t0_unix) / dt_s))
    i1 = int(np.ceil((stop_unix - t0_unix) / dt_s))

    i0_clip = max(0, min(nt, i0))
    i1_clip = max(0, min(nt, i1))

    if i1_clip <= i0_clip:
        empty_shape = (0, nfreq) if (len(chans_t) == 1 and squeeze_single_channel) else (0, nfreq, len(chans_t))
        data = np.empty(empty_shape, dtype=dtype)
        times = pd.DatetimeIndex([], tz="UTC")
        meta = RangeMeta(
            start_time_utc=start_ts.to_pydatetime(),
            t_start_idx=i0_clip,
            dt_s=dt_s,
            frame_idx=-1,
        )
        return data, times, meta

    arr = _read_channel_selection(
        cube,
        t_slice=slice(i0_clip, i1_clip),
        f_slice=slice(None),
        chans=chans_t,
        dtype=dtype,
        squeeze_single_channel=squeeze_single_channel,
    )

    sample_unix = t0_unix + np.arange(i0_clip, i1_clip, dtype=np.float64) * dt_s
    times = pd.to_datetime(sample_unix, unit="s", utc=True)

    meta = RangeMeta(
        start_time_utc=datetime.fromtimestamp(sample_unix[0], tz=UTC),
        t_start_idx=i0_clip,
        dt_s=dt_s,
        frame_idx=-1,
    )

    if pipe is not None:
        arr = pipe(arr, meta)

    return arr, times, meta
