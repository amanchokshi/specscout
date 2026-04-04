"""
Core constants and small shared utilities for specscout.

This module contains lightweight, package-wide helpers that are used across
multiple layers of the library without depending on higher-level concepts such
as datasets, preprocessing pipelines, PCA models, or ROI search.

Current responsibilities
------------------------
- Canonical UTC timestamp parsing via ``parse_utc``
- Small time/index conversion helpers for regular time grids
- Small numeric helpers such as integer clamping
- Frequency-axis construction from Zarr metadata
- Canonical channel labels for ALBATROS direct-spectra cubes

Design notes
------------
The intent is that higher-level modules such as `ingest`, `patches`, and
`dataset` import these shared primitives rather than re-implementing them.
This module should remain minimal and dependency-light.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

UTC_FMT = "%Y%m%d_%H%M%S"

# Convention used across the project for the last axis of the Zarr cube.
CHAN_LABELS: dict[int, str] = {
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
    Clamp an integer to the inclusive range ``[lo, hi]``.

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
    Convert a duration in seconds to the nearest integer number of samples.

    Parameters
    ----------
    dt_s
        Cadence in seconds per sample.
    seconds
        Duration in seconds.
    min_n
        Minimum returned value.

    Returns
    -------
    int
        Nearest integer sample count, guaranteed to be at least ``min_n``.
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    n = int(np.floor(seconds / dt_s + 0.5))
    return max(int(min_n), n)


def time_index(unix_s: float, t0_unix_s: float, dt_s: float) -> int:
    """
    Map a Unix timestamp to the nearest sample index on a regular grid.

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
    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    return int(np.floor(((unix_s - t0_unix_s) / dt_s) + 0.5))


def freq_axis_from_attrs(attrs: dict[str, object], nfreq: int) -> tuple[np.ndarray, str]:
    """
    Construct a frequency axis from Zarr-style metadata attributes.

    If ``df_mhz`` is present (and optionally ``f0_mhz``), the returned axis is
    in MHz. Otherwise, channel indices are returned.

    Parameters
    ----------
    attrs
        Metadata dictionary, typically ``dict(g.attrs)`` from a Zarr group.
    nfreq
        Number of frequency channels.

    Returns
    -------
    x
        1D array of length ``nfreq``, either in MHz or channel indices.
    x_label
        Axis label suitable for plotting.
    """
    f0 = float(attrs.get("f0_mhz", 0.0))
    df = attrs.get("df_mhz", None)

    if df is None:
        return np.arange(nfreq, dtype=float), "Frequency channel"

    df = float(df)
    return f0 + df * np.arange(nfreq, dtype=float), "Frequency (MHz)"
