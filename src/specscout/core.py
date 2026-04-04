"""
Shared utilities and core types for the *specscout* package.

This module exists to prevent duplication across the codebase. It provides:

- Canonical timestamp parsing (UTC strings <-> `datetime`)
- Time-index helpers (seconds <-> sample indices)
- Small numeric utilities used throughout (clamp, safe dB conversion)
- Shared lightweight metadata containers (e.g., `FrameMeta`)
- Canonical channel labels for ALBATROS direct spectra cubes

The intent is that higher-level modules (`patches`, `preprocess`, `viz`, `dataset`)
import from here rather than re-implementing the same helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict

import numpy as np

UTC_FMT = "%Y%m%d_%H%M%S"

# Convention used across the project for the last axis of the Zarr cube.
CHAN_LABELS: Dict[int, str] = {
    0: "pol00",
    1: "pol11",
    2: "pol01_mag",
    3: "pol01_phase",
}


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
        Frame number within the requested sequence (0..n_frames-1).
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


def parse_utc(s: str) -> datetime:
    """
    Parse a UTC timestamp string into a timezone-aware `datetime`.

    Parameters
    ----------
    s
        UTC timestamp in the form ``YYYYmmdd_HHMMSS``.

    Returns
    -------
    datetime
        Timezone-aware datetime in UTC.
    """
    return datetime.strptime(s, UTC_FMT).replace(tzinfo=UTC)


def clamp_int(x: int, lo: int, hi: int) -> int:
    """
    Clamp an integer to the inclusive range [lo, hi].

    Parameters
    ----------
    x
        Input integer.
    lo
        Lower bound (inclusive).
    hi
        Upper bound (inclusive).

    Returns
    -------
    int
        Clamped integer.
    """
    return lo if x < lo else hi if x > hi else x


def seconds_to_samples(dt_s: float, seconds: float, *, min_n: int = 1) -> int:
    """
    Convert duration in seconds to nearest integer number of samples.

    Parameters
    ----------
    dt_s
        Cadence (seconds per sample).
    seconds
        Duration in seconds.
    min_n
        Minimum returned value.

    Returns
    -------
    int
        Nearest integer sample count (>= min_n).
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")
    n = int(np.floor(seconds / dt_s + 0.5))
    return max(int(min_n), n)


def time_index(unix_s: float, t0_unix_s: float, dt_s: float) -> int:
    """
    Map Unix time in seconds to the nearest sample index on a regular grid.

    Parameters
    ----------
    unix_s
        Unix timestamp in seconds.
    t0_unix_s
        Unix timestamp corresponding to sample index 0.
    dt_s
        Cadence in seconds per sample.

    Returns
    -------
    int
        Nearest integer sample index.
    """
    return int(np.floor(((unix_s - t0_unix_s) / dt_s) + 0.5))


def freq_axis_from_attrs(attrs: dict, nfreq: int) -> tuple[np.ndarray, str]:
    """
    Construct a frequency x-axis from Zarr attributes.

    If the store has `df_mhz` (and optionally `f0_mhz`), returns MHz.
    Otherwise returns channel indices.

    Parameters
    ----------
    attrs
        Zarr group attributes dictionary (typically `dict(g.attrs)`).
    nfreq
        Number of frequency channels (cube second dimension).

    Returns
    -------
    x
        1D array of length `nfreq`, either in MHz or channel indices.
    x_label
        Label suitable for plotting (e.g. "Frequency (MHz)" or "Frequency channel").
    """
    f0 = float(attrs.get("f0_mhz", 0.0))
    df = attrs.get("df_mhz", None)

    if df is None:
        return np.arange(nfreq, dtype=float), "Frequency channel"

    df = float(df)
    return f0 + df * np.arange(nfreq, dtype=float), "Frequency (MHz)"



def plan_frames(
    *,
    nt: int,
    t0_unix_s: float,
    dt_s: float,
    start_utc: str,
    stop_utc: str,
    window_seconds: float,
    step_seconds: float,
    parse_utc,  # use your existing parse_utc in core
) -> FramePlan:
    """
    Build a `FramePlan` from human-friendly UTC strings and durations.

    Parameters
    ----------
    nt
        Total number of time samples in the cube.
    t0_unix_s
        Unix seconds corresponding to sample index 0.
    dt_s
        Cadence in seconds per sample.
    start_utc, stop_utc
        UTC strings in format YYYYmmdd_HHMMSS.
    window_seconds, step_seconds
        Window length and step size in seconds.
    parse_utc
        Callable that parses UTC strings to timezone-aware datetimes.

    Returns
    -------
    FramePlan
        Plan describing valid frame start indices and counts.
    """
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive.")

    window_n = seconds_to_samples(dt_s, window_seconds, min_n=1)
    step_n = seconds_to_samples(dt_s, step_seconds, min_n=1)

    start_dt = parse_utc(start_utc)
    stop_dt = parse_utc(stop_utc)
    if stop_dt < start_dt:
        raise ValueError("stop_utc must be >= start_utc")

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
