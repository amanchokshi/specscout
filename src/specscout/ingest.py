"""
Ingest ALBATROS direct-spectra files into a regularized, chunked Zarr v3 cube.

This module is responsible for *I/O and storage*:

- Traverses a timestamped directory tree (directories named by UNIX seconds).
- Reads per-hour direct-spectra files: ``pol00``, ``pol11``, ``pol01r``, ``pol01i``.
- Aligns data onto a regular time grid of cadence ``dt_seconds``.
- Writes a single Zarr v3 store containing a dense cube with NaNs for gaps.

The output Zarr store is the canonical on-disk representation consumed by the
rest of the package (patch extraction, preprocessing, visualization, ML).

Notes
-----
- This ingest step does **not** perform bandpass subtraction or whitening.
- This implementation targets ALBATROS direct spectra which are expected to
  have exactly 2048 frequency channels. Files with unexpected shapes/channels
  are skipped and the corresponding region remains NaN in the preallocated cube.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from scio import scio
from zarr.codecs import BloscCodec, BloscShuffle

from .core import UTC_FMT, parse_utc, time_index

# Canonical file names expected within each timestamp directory.
_P00 = "pol00.scio.bz2"
_P11 = "pol11.scio.bz2"
_P01R = "pol01r.scio.bz2"
_P01I = "pol01i.scio.bz2"


@dataclass(frozen=True)
class IngestConfig:
    """
    Configuration for ingesting direct spectra into a Zarr cube.

    Parameters
    ----------
    startutc
        Inclusive start time in UTC_FMT (``YYYYmmdd_HHMMSS``).
    stoputc
        Inclusive stop time in UTC_FMT (``YYYYmmdd_HHMMSS``).
    dt_seconds
        Cadence of spectra in seconds.
    f0_mhz
        Frequency of channel 0 in MHz.
    df_mhz
        Frequency spacing between channels in MHz.
    nchan
        Number of frequency channels. For ALBATROS direct spectra this is
        expected to be 2048.
    chunk_t
        Chunk length along time (number of time samples per chunk).
    compressors
        Zarr v3 compressor stack (codecs). If None, a sensible default is used.
    batch_size
        Number of timestamp directories to read per ``scio.read_files`` batch.
    """

    startutc: str
    stoputc: str
    dt_seconds: float = 393216 * (4096 / 250e6)
    f0_mhz: float = 0.0
    df_mhz: float = 125.0 / 2048
    nchan: int = 2048
    chunk_t: int = 1024
    compressors: Optional[object] = None
    batch_size: int = 128


def iter_timestamp_dirs(root: Path) -> Iterator[tuple[int, Path]]:
    """
    Yield (unix_seconds, path) for directories whose name is exactly 10 digits.

    Parameters
    ----------
    root
        Root directory to traverse.

    Yields
    ------
    (unix_seconds, path)
        Integer unix seconds parsed from the directory name, and the directory Path.
    """
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        name = d.name
        if len(name) == 10 and name.isdigit():
            yield int(name), d


def _file_paths_for_dir(d: Path) -> tuple[str, str, str, str]:
    """
    Return absolute paths to expected files in a timestamp directory.

    Parameters
    ----------
    d
        Timestamp directory.

    Returns
    -------
    (pol00, pol11, pol01r, pol01i)
        Absolute paths as strings.
    """
    return (str(d / _P00), str(d / _P11), str(d / _P01R), str(d / _P01I))


def _default_compressors(dtype: np.dtype) -> tuple[object, ...]:
    """
    Choose a reasonable default compressor stack for Zarr v3 numeric arrays.

    Parameters
    ----------
    dtype
        Data type of the array being stored.

    Returns
    -------
    tuple
        Tuple of Zarr v3 codecs.
    """
    typesize = int(dtype.itemsize)
    return (
        BloscCodec(
            cname="zstd",
            clevel=3,
            shuffle=BloscShuffle.shuffle,
            typesize=typesize,
        ),
    )


def _compute_nt(start: datetime, stop: datetime, dt_seconds: float) -> int:
    """
    Compute the number of samples on an inclusive grid [start, stop] with spacing dt.

    Parameters
    ----------
    start
        Inclusive start time.
    stop
        Inclusive stop time.
    dt_seconds
        Cadence in seconds.

    Returns
    -------
    int
        Number of time samples (>= 1).
    """
    span = (stop - start).total_seconds()
    if span < 0:
        raise ValueError("stoputc must be >= startutc")
    return max(int(np.floor(span / dt_seconds + 0.5)) + 1, 1)


def _batched(seq: Sequence[tuple[int, Path]], n: int) -> Iterator[Sequence[tuple[int, Path]]]:
    """
    Yield successive slices of length `n` from `seq`.

    Parameters
    ----------
    seq
        Input sequence.
    n
        Batch size.

    Yields
    ------
    Sequence[tuple[int, Path]]
        Successive batches.
    """
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def infer_available_utc_bounds(root: str | Path) -> tuple[str, str]:
    """
    Infer UTC start/stop bounds from available timestamp directories under `root`.

    Parameters
    ----------
    root
        Root directory containing timestamp-named subdirectories.

    Returns
    -------
    (startutc, stoputc)
        Earliest and latest available timestamps formatted in UTC_FMT.

    Raises
    ------
    RuntimeError
        If no valid timestamp directories are found.
    """
    root_path = Path(root)
    ts_dirs = list(iter_timestamp_dirs(root_path))
    if not ts_dirs:
        raise RuntimeError("No valid timestamp directories found under root.")

    ts_vals = sorted(ts for ts, _ in ts_dirs)
    startutc = datetime.fromtimestamp(ts_vals[0], tz=UTC).strftime(UTC_FMT)
    stoputc = datetime.fromtimestamp(ts_vals[-1], tz=UTC).strftime(UTC_FMT)
    return startutc, stoputc


def ingest_direct_spectra_to_zarr(
    root: str | Path,
    out_zarr: str | Path,
    cfg: IngestConfig,
    *,
    station: str = "",
) -> Path:
    """
    Ingest direct spectra under `root` into a regularized Zarr v3 cube at `out_zarr`.

    Parameters
    ----------
    root
        Root directory containing timestamp-named subdirectories with direct spectra files.
    out_zarr
        Output path for the Zarr store (a directory).
    cfg
        Ingest configuration.
    station
        Optional station label stored in Zarr attributes (e.g., "MARS2").

    Returns
    -------
    pathlib.Path
        Path to the written Zarr store.

    Output layout
    -------------
    The output Zarr group contains one main array:

    - ``cube`` : float32, shape ``(nt, nfreq, 4)`` where the last axis is:
        0=pol00, 1=pol11, 2=pol01_mag, 3=pol01_phase

    Zarr attributes store time and frequency axis metadata:
    ``dt_seconds``, ``t0_unix_seconds``, ``f0_mhz``, ``df_mhz``, etc.
    """
    root_path = Path(root)
    out_path = Path(out_zarr)

    start_dt = parse_utc(cfg.startutc)
    stop_dt = parse_utc(cfg.stoputc)

    t0_unix_s = float(start_dt.timestamp())
    nt = _compute_nt(start_dt, stop_dt, cfg.dt_seconds)

    # Candidate directories in range.
    start_s = int(np.floor(start_dt.timestamp()))
    stop_s = int(np.ceil(stop_dt.timestamp()))
    dirs: list[tuple[int, Path]] = [(ts_s, d) for ts_s, d in iter_timestamp_dirs(root_path) if start_s <= ts_s <= stop_s]
    dirs.sort(key=lambda x: x[0])
    if not dirs:
        raise RuntimeError("No valid timestamp directories found in the specified range.")

    # Create store and arrays
    store = zarr.open_group(out_path, mode="w", zarr_format=3)

    cube_dtype = np.dtype(np.float32)
    compressors = cfg.compressors if cfg.compressors is not None else _default_compressors(cube_dtype)

    cube = store.create_array(
        name="cube",
        shape=(nt, cfg.nchan, 4),
        chunks=(cfg.chunk_t, cfg.nchan, 4),
        dtype=cube_dtype,
        fill_value=np.nan,
        compressors=compressors,
        overwrite=True,
    )

    store.attrs.update(
        {
            "dt_seconds": float(cfg.dt_seconds),
            "t0_unix_seconds": float(t0_unix_s),
            "startutc": cfg.startutc,
            "stoputc": cfg.stoputc,
            "nt": int(nt),
            "nchan": int(cfg.nchan),
            "station": station,
            "channel_labels": ["pol00", "pol11", "pol01_mag", "pol01_phase"],
            "created_utc": datetime.now(UTC).strftime(UTC_FMT),
            "source_root": str(root_path),
            "f0_mhz": float(cfg.f0_mhz),
            "df_mhz": float(cfg.df_mhz),
        }
    )

    # Read + write in batches to keep memory bounded.
    for batch in _batched(dirs, cfg.batch_size):
        pol00_files: list[str] = []
        pol11_files: list[str] = []
        pol01r_files: list[str] = []
        pol01i_files: list[str] = []
        batch_ts: list[int] = []

        for ts_s, d in batch:
            p00, p11, p01r, p01i = _file_paths_for_dir(d)
            pol00_files.append(p00)
            pol11_files.append(p11)
            pol01r_files.append(p01r)
            pol01i_files.append(p01i)
            batch_ts.append(ts_s)

        pol00_arrays = scio.read_files(pol00_files)
        pol11_arrays = scio.read_files(pol11_files)
        pol01r_arrays = scio.read_files(pol01r_files)
        pol01i_arrays = scio.read_files(pol01i_files)

        for ts_s, a00, a11, ar, ai in zip(batch_ts, pol00_arrays, pol11_arrays, pol01r_arrays, pol01i_arrays):
            # Missing any component: leave as NaNs
            if any(x is None for x in (a00, a11, ar, ai)):
                continue

            # Expect 2D arrays
            if not (a00.ndim == a11.ndim == ar.ndim == ai.ndim == 2):
                continue

            # Expect same shape
            if not (a00.shape == a11.shape == ar.shape == ai.shape):
                continue

            n_time, n_chan_this = a00.shape
            # Any wrong channel count is considered serious: skip.
            if n_chan_this != cfg.nchan:
                continue

            i0 = time_index(float(ts_s), t0_unix_s, cfg.dt_seconds)
            if i0 >= nt:
                continue

            i1 = min(i0 + n_time, nt)
            n_write = i1 - i0
            if n_write <= 0:
                continue

            # Derived channels from complex cross-pol (magnitude, phase)
            pol01 = ar[:n_write, :] + 1j * ai[:n_write, :]
            mag = np.abs(pol01)
            phs = np.angle(pol01)

            block = np.empty((n_write, cfg.nchan, 4), dtype=np.float32)
            block[:, :, 0] = a00[:n_write, :].astype(np.float32, copy=False)
            block[:, :, 1] = a11[:n_write, :].astype(np.float32, copy=False)
            block[:, :, 2] = mag.astype(np.float32, copy=False)
            block[:, :, 3] = phs.astype(np.float32, copy=False)

            cube[i0:i1, :, :] = block

    return out_path


def _build_default_out_path(
    root: Path,
    station: str,
    startutc: str,
    stoputc: str,
) -> Path:
    """
    Build a sensible default output zarr path.

    Parameters
    ----------
    root
        Root input directory.
    station
        Station name.
    startutc, stoputc
        UTC bounds in UTC_FMT.

    Returns
    -------
    pathlib.Path
        Default output path in the current working directory.
    """
    stem = f"{station}_{startutc}_{stoputc}".replace(":", "-")
    return Path.cwd() / f"{stem}.zarr"
