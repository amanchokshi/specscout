from __future__ import annotations

import argparse
from pathlib import Path

import zarr

from .ingest import (
    IngestConfig,
    infer_available_utc_bounds,
    ingest_direct_spectra_to_zarr,
)


def _build_default_out_path(
    root: Path,
    station: str,
    startutc: str,
    stoputc: str,
) -> Path:
    stem = f"{station}_{startutc}_{stoputc}".replace(":", "-")
    return Path.cwd() / f"{stem}.zarr"


def ingest_cmd(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.startutc is None or args.stoputc is None:
        inferred_startutc, inferred_stoputc = infer_available_utc_bounds(args.root)
        startutc = args.startutc if args.startutc is not None else inferred_startutc
        stoputc = args.stoputc if args.stoputc is not None else inferred_stoputc
    else:
        startutc = args.startutc
        stoputc = args.stoputc

    cfg = IngestConfig(
        startutc=startutc,
        stoputc=stoputc,
        batch_size=args.batch_size,
    )

    out_zarr = (
        args.out_zarr
        if args.out_zarr is not None
        else _build_default_out_path(
            root=args.root,
            station=args.station,
            startutc=startutc,
            stoputc=stoputc,
        )
    )

    print(f"root       : {args.root}")
    print(f"station    : {args.station}")
    print(f"startutc   : {startutc}")
    print(f"stoputc    : {stoputc}")
    print(f"batch_size : {cfg.batch_size}")
    print(f"out_zarr   : {out_zarr}")

    out = ingest_direct_spectra_to_zarr(
        root=args.root,
        out_zarr=out_zarr,
        cfg=cfg,
        station=args.station,
    )
    print(f"Wrote {out}")


def zarrmeta_cmd(args: argparse.Namespace) -> None:
    path = args.path

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    z = zarr.open_group(path, mode="r")
    attrs = dict(z.attrs)

    print(f"\nZarr: {path}\n")

    keys = [
        "station",
        "startutc",
        "stoputc",
        "created_utc",
        "nt",
        "nchan",
        "dt_seconds",
        "f0_mhz",
        "df_mhz",
        "source_root",
    ]

    for k in keys:
        v = attrs.get(k, "<missing>")
        print(f"{k:<14}: {v}")

    if args.all:
        print("\nAll attributes:")
        for k, v in attrs.items():
            print(f"{k:<14}: {v}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specscout",
        description="Specscout command line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest ALBATROS direct spectra into a Zarr cube.",
    )
    ingest.add_argument(
        "root",
        type=Path,
        help="Root directory containing timestamp directories.",
    )
    ingest.add_argument(
        "--station",
        required=True,
        help='Station label, e.g. "MARS1".',
    )
    ingest.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of timestamp directories per scio batch.",
    )
    ingest.add_argument(
        "--startutc",
        type=str,
        default=None,
        help="Optional inclusive start UTC in YYYYmmdd_HHMMSS format.",
    )
    ingest.add_argument(
        "--stoputc",
        type=str,
        default=None,
        help="Optional inclusive stop UTC in YYYYmmdd_HHMMSS format.",
    )
    ingest.add_argument(
        "--out-zarr",
        type=Path,
        default=None,
        help="Optional output .zarr path.",
    )
    ingest.set_defaults(func=ingest_cmd)

    # zarrmeta
    zmeta = subparsers.add_parser(
        "zarrmeta",
        help="Print metadata from a specscout Zarr store.",
    )
    zmeta.add_argument(
        "path",
        type=Path,
        help="Path to .zarr directory",
    )
    zmeta.add_argument(
        "--all",
        action="store_true",
        help="Print all attributes (not just key fields).",
    )
    zmeta.set_defaults(func=zarrmeta_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
