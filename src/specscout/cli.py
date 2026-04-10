from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import zarr

from .ingest import (
    IngestConfig,
    infer_available_utc_bounds,
    ingest_direct_spectra_to_zarr,
)
from .preprocess import PreprocessPipeline, step_safe_db, step_stokes_i
from .roi import run_roi_search


def _build_default_out_path(
    root: Path,
    station: str,
    startutc: str,
    stoputc: str,
) -> Path:
    stem = f"{station}_{startutc}_{stoputc}".replace(":", "-")
    return Path.cwd() / f"{stem}.zarr"


def _build_default_stokes_i_pipe(
    *,
    zarr_path: str | Path | None = None,
    station: str | None = None,
    notes: str | None = None,
) -> PreprocessPipeline:
    """
    Build the default Stokes-I + safe_db preprocessing pipeline used by the CLI.

    This is the standard operational detection/plotting product for roi-search.
    """
    pipe = PreprocessPipeline(input_space="linear")
    md: dict[str, Any] = {}

    if zarr_path is not None:
        md["zarr_path"] = str(zarr_path)
    if station is not None:
        md["station"] = station
    if notes is not None:
        md["notes"] = notes

    if md:
        pipe = pipe.with_metadata(**md)

    return pipe.add(step_stokes_i()).add(step_safe_db(name="safe_db"))


def _add_metric_args(
    parser: argparse.ArgumentParser,
    *,
    prefix: str,
    default_method: str,
    include_positive_only: bool = False,
    include_min_finite_frac: bool = False,
) -> None:
    """
    Add metric-related argparse options to a parser.

    Parameters
    ----------
    parser
        Parser or argument group to modify.
    prefix
        Prefix for argument names, e.g. ``"quiet"`` or ``"score"``.
    default_method
        Default metric method string.
    include_positive_only
        If True, add a toggle for positive-only scoring.
    include_min_finite_frac
        If True, add ``min_finite_frac``.
    """
    parser.add_argument(
        f"--{prefix}-method",
        type=str,
        default=default_method,
        choices=[
            "p99",
            "p995",
            "p999",
            "percentile",
            "topk_sum",
            "excess_mass",
            "l1",
            "l2",
            "lp",
        ],
        help=f"{prefix.capitalize()} metric method.",
    )
    parser.add_argument(
        f"--{prefix}-q",
        type=float,
        default=99.0,
        help=f"Percentile q used when --{prefix}-method=percentile.",
    )
    parser.add_argument(
        f"--{prefix}-topk",
        type=int,
        default=2048,
        help=f"Top-k used when --{prefix}-method=topk_sum.",
    )
    parser.add_argument(
        f"--{prefix}-thr",
        type=float,
        default=3.0,
        help=f"Threshold used when --{prefix}-method=excess_mass.",
    )
    parser.add_argument(
        f"--{prefix}-p",
        type=float,
        default=4.0,
        help=f"p used when --{prefix}-method=lp.",
    )

    if include_positive_only:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            f"--{prefix}-positive-only",
            dest=f"{prefix}_positive_only",
            action="store_true",
            help=f"Score only positive residuals for {prefix}.",
        )
        group.add_argument(
            f"--{prefix}-allow-negative",
            dest=f"{prefix}_positive_only",
            action="store_false",
            help=f"Allow negative residuals to contribute to {prefix}.",
        )
        parser.set_defaults(**{f"{prefix}_positive_only": True})

    if include_min_finite_frac:
        parser.add_argument(
            f"--{prefix}-min-finite-frac",
            type=float,
            default=0.7,
            help=f"Minimum finite fraction required for {prefix} scoring.",
        )


def _build_metric_kwargs(
    *,
    args: argparse.Namespace,
    prefix: str,
    include_positive_only: bool = False,
    include_min_finite_frac: bool = False,
) -> dict[str, Any]:
    """
    Build a kwargs dict for metric-based configuration from argparse args.
    """
    method = getattr(args, f"{prefix}_method")
    kwargs: dict[str, Any] = {"method": method}

    if method == "percentile":
        kwargs["q"] = getattr(args, f"{prefix}_q")
    elif method == "topk_sum":
        kwargs["topk"] = getattr(args, f"{prefix}_topk")
    elif method == "excess_mass":
        kwargs["thr"] = getattr(args, f"{prefix}_thr")
    elif method == "lp":
        kwargs["p"] = getattr(args, f"{prefix}_p")

    if include_positive_only:
        kwargs["positive_only"] = getattr(args, f"{prefix}_positive_only")

    if include_min_finite_frac:
        kwargs["min_finite_frac"] = getattr(args, f"{prefix}_min_finite_frac")

    return kwargs


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
    quiet_selector_kwargs = _build_metric_kwargs(
        args=args,
        prefix="quiet",
        include_positive_only=False,
        include_min_finite_frac=False,
    )
    quiet_selector_kwargs["quiet_fraction"] = args.quiet_fraction

    score_kwargs = _build_metric_kwargs(
        args=args,
        prefix="score",
        include_positive_only=True,
        include_min_finite_frac=True,
    )

    detect_pipe = _build_default_stokes_i_pipe(
        zarr_path=args.zarr_path,
        station=args.station,
        notes="CLI default detection pipe: Stokes I + safe_db",
    )
    plot_pipe = _build_default_stokes_i_pipe(
        zarr_path=args.zarr_path,
        station=args.station,
        notes="CLI default plotting pipe: Stokes I + safe_db",
    )

    result = run_roi_search(
        args.zarr_path,
        station=args.station,
        analysis_start_utc=args.startutc,
        analysis_stop_utc=args.stoputc,
        out_dir=args.out_dir,
        detect_chans=(0, 1),
        detect_pipe=detect_pipe,
        plot_chans=(0, 1),
        plot_pipe=plot_pipe,
        window_seconds=args.window_minutes * 60.0,
        step_seconds=args.step_minutes * 60.0,
        context_hours=args.context_hours,
        stride_hours=args.stride_hours,
        score_hours=args.score_hours,
        gap_hours=args.gap_hours,
        n_quiet=args.n_quiet,
        k_fit=args.k_fit,
        k_pca=args.k_pca,
        nsig=args.nsig,
        pad_minutes=args.pad_minutes,
        merge_gap_minutes=args.merge_gap_minutes,
        rfi_mask_start=args.rfi_mask_start,
        rfi_mask_stop=args.rfi_mask_stop,
        random_state=args.random_state,
        quiet_selector_kwargs=quiet_selector_kwargs,
        score_kwargs=score_kwargs,
        plot_pad_minutes=args.plot_pad_minutes,
        save_plots=not args.no_plots,
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


def goes_xrs_cmd(args: argparse.Namespace) -> None:
    """
    Download GOES soft X-ray 1 s NetCDF files for a given year.

    Files are downloaded recursively from the NOAA archive using `wget` into:

        out_dir / YEAR / ...

    Parameters expected on `args`
    -----------------------------
    year
        Four-digit year to download.
    satellite
        GOES satellite name, e.g. "goes18".
    out_dir
        Root output directory.
    resume
        If True, pass `--continue` to `wget`.
    """
    year = int(args.year)
    if year < 2000 or year > 2100:
        raise ValueError("--year must be a reasonable four-digit year.")

    satellite = str(args.satellite).strip().lower()
    out_dir = Path(args.out_dir) / f"{year:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    product = "xrsf-l2-flx1s_science"
    base_url = (
        f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{satellite}/l2/data/{product}/{year:04d}/"
    )

    cmd = [
        "wget",
        "--recursive",
        "--no-parent",
        "--no-host-directories",
        "--cut-dirs=9",
        "--reject",
        "index.html*",
        "--accept",
        "*.nc",
        "--directory-prefix",
        str(out_dir),
    ]

    if args.resume:
        cmd.append("--continue")

    cmd.append(base_url)

    print(f"satellite : {satellite}")
    print(f"year      : {year:04d}")
    print(f"product   : {product}")
    print(f"url       : {base_url}")
    print(f"out_dir   : {out_dir}")
    print("running   : " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("wget was not found on this system. Please install wget and try again.") from exc

    print("GOES XRS download complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specscout",
        description="Specscout command line tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest ALBATROS direct spectra into a Zarr cube.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    zmeta = subparsers.add_parser(
        "zarrmeta",
        help="Print metadata from a specscout Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    zmeta.add_argument(
        "path",
        type=Path,
        help="Path to .zarr directory.",
    )
    zmeta.add_argument(
        "--all",
        action="store_true",
        help="Print all attributes (not just key fields).",
    )
    zmeta.set_defaults(func=zarrmeta_cmd)

    roi = subparsers.add_parser(
        "roi-search",
        help="Run rolling PCA ROI search on a station-season Zarr product.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    roi.add_argument(
        "--window-minutes",
        type=float,
        default=20.0,
        help="Frame duration in minutes.",
    )
    roi.add_argument(
        "--step-minutes",
        type=float,
        default=5.0,
        help="Frame step size in minutes.",
    )
    roi.add_argument(
        "--context-hours",
        type=float,
        default=24.0,
        help="Rolling PCA context window width in hours.",
    )
    roi.add_argument(
        "--stride-hours",
        type=float,
        default=1.0,
        help="How often to refit and rescore, in hours.",
    )
    roi.add_argument(
        "--score-hours",
        type=float,
        default=1.0,
        help="Width of scored chunk per rolling step, in hours.",
    )
    roi.add_argument(
        "--gap-hours",
        type=float,
        default=0.0,
        help="Optional donut gap around the scored interval when fitting PCA, in hours.",
    )

    roi.add_argument(
        "--quiet-fraction",
        type=float,
        default=0.3,
        help="Fraction of finite-scored context frames retained for quiet PCA training.",
    )
    roi.add_argument(
        "--n-quiet",
        type=int,
        default=None,
        help="Optional fixed number of quiet frames. Overrides quiet_fraction in the rolling runner.",
    )

    _add_metric_args(
        roi,
        prefix="quiet",
        default_method="p99",
        include_positive_only=False,
        include_min_finite_frac=False,
    )

    _add_metric_args(
        roi,
        prefix="score",
        default_method="p99",
        include_positive_only=True,
        include_min_finite_frac=True,
    )

    roi.add_argument(
        "--k-fit",
        type=int,
        default=128,
        help="Number of PCA modes fit in the quiet background model.",
    )
    roi.add_argument(
        "--k-pca",
        type=int,
        default=16,
        help="Number of PCA modes used during reconstruction for scoring.",
    )
    roi.add_argument(
        "--nsig",
        type=float,
        default=3.0,
        help="Robust sigma threshold multiplier for ROI detection.",
    )
    roi.add_argument(
        "--pad-minutes",
        type=float,
        default=5.0,
        help="Padding applied to each ROI boundary, in minutes.",
    )
    roi.add_argument(
        "--merge-gap-minutes",
        type=float,
        default=20.0,
        help="Merge ROIs separated by less than or equal to this gap, in minutes.",
    )
    roi.add_argument(
        "--rfi-mask-start",
        type=int,
        default=116,
        help="Start channel of the masked RFI range.",
    )
    roi.add_argument(
        "--rfi-mask-stop",
        type=int,
        default=384,
        help="Stop channel of the masked RFI range.",
    )
    roi.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for randomized SVD.",
    )
    roi.add_argument(
        "--plot-pad-minutes",
        type=float,
        default=5.0,
        help="Extra context shown on either side of each ROI in ROI quicklook plots.",
    )
    roi.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate summary or ROI quicklook plots.",
    )

    roi.set_defaults(func=roi_search_cmd)

    # goes-xrs
    goes = subparsers.add_parser(
        "goes-xrs",
        help="Download GOES 1 s soft X-ray lightcurves (NetCDF files) for a given year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    goes.add_argument(
        "--year",
        type=int,
        required=True,
        help="Four-digit year to download.",
    )
    goes.add_argument(
        "--satellite",
        type=str,
        default="goes18",
        help='GOES satellite name, e.g. "goes18".',
    )
    goes.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Root output directory. Files will be placed under out-dir/YEAR/...",
    )
    goes.add_argument(
        "--resume",
        action="store_true",
        help="Resume partially downloaded files using wget --continue.",
    )
    goes.set_defaults(func=goes_xrs_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
