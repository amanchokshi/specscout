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
from typing import Callable, Iterator, Optional

import numpy as np
import zarr


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
