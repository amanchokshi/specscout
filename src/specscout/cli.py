from __future__ import annotations

import argparse
from pathlib import Path

import zarr

from .ingest import (
    IngestConfig,
    infer_available_utc_bounds,
    ingest_direct_spectra_to_zarr,
)
from .roi_search import run_roi_search


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


def roi_search_cmd(args: argparse.Namespace) -> None:
    result = run_roi_search(
        args.zarr_path,
        station=args.station,
        analysis_start_utc=args.startutc,
        analysis_stop_utc=args.stoputc,
        out_dir=args.out_dir,
        window_seconds=args.window_minutes * 60.0,
        step_seconds=args.step_minutes * 60.0,
        context_hours=args.context_hours,
        stride_hours=args.stride_hours,
        score_hours=args.score_hours,
        gap_hours=args.gap_hours,
        quiet_fraction=args.quiet_fraction,
        n_quiet=args.n_quiet,
        k_fit=args.k_fit,
        k_pca=args.k_pca,
        min_finite_frac=args.min_finite_frac,
        nsig=args.nsig,
        pad_minutes=args.pad_minutes,
        merge_gap_minutes=args.merge_gap_minutes,
        rfi_mask_start=args.rfi_mask_start,
        rfi_mask_stop=args.rfi_mask_stop,
        random_state=args.random_state,
    )

    print("ROI search complete.")
    print(f"  out_dir     : {result.out_dir}")
    print(f"  scores.pkl  : {result.scores_path}")
    print(f"  rois.pkl    : {result.rois_path}")
    print(f"  config.json : {result.config_path}")
    print(f"  summary.png : {result.summary_plot_path}")
    print(f"  roi plots   : {result.roi_plot_dir}")
    print(f"  n_scores    : {result.n_scores}")
    print(f"  n_rois      : {result.n_rois}")


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

    # roi-search
    roi = subparsers.add_parser(
        "roi-search",
        help="Run rolling PCA ROI search on a station-season Zarr product.",
    )
    roi.add_argument(
        "zarr_path",
        type=Path,
        help="Path to station-season .zarr directory.",
    )
    roi.add_argument(
        "--station",
        required=True,
        help='Station label, e.g. "MARS1".',
    )
    roi.add_argument(
        "--startutc",
        required=True,
        help="Analysis start UTC in YYYYmmdd_HHMMSS format.",
    )
    roi.add_argument(
        "--stoputc",
        required=True,
        help="Analysis stop UTC in YYYYmmdd_HHMMSS format.",
    )
    roi.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for scores, ROIs, config, and plots.",
    )

    roi.add_argument("--window-minutes", type=float, default=20.0)
    roi.add_argument("--step-minutes", type=float, default=5.0)

    roi.add_argument("--context-hours", type=float, default=24.0)
    roi.add_argument("--stride-hours", type=float, default=1.0)
    roi.add_argument("--score-hours", type=float, default=1.0)
    roi.add_argument("--gap-hours", type=float, default=0.0)

    roi.add_argument("--quiet-fraction", type=float, default=0.3)
    roi.add_argument("--n-quiet", type=int, default=None)

    roi.add_argument("--k-fit", type=int, default=128)
    roi.add_argument("--k-pca", type=int, default=16)

    roi.add_argument("--min-finite-frac", type=float, default=0.7)

    roi.add_argument("--nsig", type=float, default=3.0)
    roi.add_argument("--pad-minutes", type=float, default=5.0)
    roi.add_argument("--merge-gap-minutes", type=float, default=20.0)

    roi.add_argument("--rfi-mask-start", type=int, default=116)
    roi.add_argument("--rfi-mask-stop", type=int, default=384)

    roi.add_argument("--random-state", type=int, default=42)

    roi.set_defaults(func=roi_search_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
