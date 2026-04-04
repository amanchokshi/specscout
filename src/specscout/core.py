"""
Shared utilities and core types for the *specscout* package.

This module exists to prevent duplication across the codebase. It provides:

- Canonical timestamp parsing (UTC strings <-> `datetime`)
- Time-index helpers (seconds <-> sample indices)
- Small numeric utilities used throughout (clamp, safe dB conversion)
- Canonical channel labels for ALBATROS direct spectra cubes

The intent is that higher-level modules (`patches`, `preprocess`, `viz`, `dataset`)
import from here rather than re-implementing the same helpers.
"""

from __future__ import annotations

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
