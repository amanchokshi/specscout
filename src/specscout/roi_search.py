"""
ROI search pipeline for specscout seasonal Zarr products.

This module runs a production-style transient search over a user-specified
analysis window using:

- Stokes I as the detection product
- rolling quiet-PCA background modeling
- p99 quiet-frame selection
- p99 residual scoring
- robust sigma thresholding to define ROIs

Outputs are written to disk in a simple, inspectable format:

- scores.pkl
- rois.pkl
- config.json
- scores_with_rois.png
- rois/roi_XXXX.png

Design notes
------------
- The input Zarr is assumed to contain the full available data for a station/season.
- The requested analysis window may exceed the extant data bounds; the underlying
  dataset / rolling framework is expected to handle missing data as NaNs.
- Detection is performed on Stokes I derived from channels (0, 1).
- ROI plots are generated for all detected ROIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import cmasher as cmr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .dataset import SpecscoutDataset
from .outlier import QuietSelector, RollingPCABackground
from .patches import read_time_range
from .preprocess import PreprocessPipeline, step_safe_db, step_stokes_i
from .rolling import RollingPCARunner, padded_utc_range


@dataclass(frozen=True)
class ROI:
    """
    Region of interest in time.

    Parameters
    ----------
    start
        Inclusive ROI start time.
    stop
        Inclusive ROI stop time.
    peak_score
        Maximum frame score within the unpadded, above-threshold segment.
    sum_score
        Sum of frame scores within the unpadded, above-threshold segment.
    n_frames
        Number of above-threshold frames contributing to the ROI before padding/merge.
    """

    start: pd.Timestamp
    stop: pd.Timestamp
    peak_score: float
    sum_score: float
    n_frames: int


@dataclass(frozen=True)
class ROISearchResult:
    """
    Summary of a completed ROI search run.

    Parameters
    ----------
    out_dir
        Root output directory for this run.
    scores_path
        Pickle file containing frame-level score table.
    rois_path
        Pickle file containing ROI table.
    config_path
        JSON file containing run configuration.
    summary_plot_path
        PNG plot of frame scores with ROI overlays.
    roi_plot_dir
        Directory containing one PNG per ROI.
    n_scores
        Number of frame-level scores written.
    n_rois
        Number of ROIs written.
    """

    out_dir: Path
    scores_path: Path
    rois_path: Path
    config_path: Path
    summary_plot_path: Path
    roi_plot_dir: Path
    n_scores: int
    n_rois: int


def robust_sigma_threshold(scores: np.ndarray, *, nsig: float = 5.0) -> float:
    """
    Compute a robust threshold using the median and MAD.

    The threshold is:

        threshold = median(scores) + nsig * (1.4826 * MAD(scores))

    Parameters
    ----------
    scores
        1D array of scores. NaNs/Infs are ignored.
    nsig
        Number of robust sigma above the median.

    Returns
    -------
    float
        Threshold value. Returns NaN if no finite scores are available.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan

    med = np.median(s)
    mad = np.median(np.abs(s - med))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0:
        return float(np.percentile(s, 99.9))

    return float(med + nsig * sigma)


def find_rois_from_scores(
    df_scores: pd.DataFrame,
    *,
    nsig: float = 3.0,
    pad: timedelta = timedelta(minutes=5),
    merge_gap: timedelta = timedelta(minutes=20),
    time_col: str = "time",
    score_col: str = "score",
) -> tuple[float, list[ROI]]:
    """
    Detect ROIs from a frame-level score time series.

    A frame is active if its score is finite and exceeds a robust sigma threshold.
    Contiguous active runs are converted into ROIs, then padded and merged.

    Parameters
    ----------
    df_scores
        DataFrame containing at least columns ``time_col`` and ``score_col``.
    nsig
        Robust sigma multiplier used in thresholding.
    pad
        Time padding added to each ROI boundary.
    merge_gap
        ROIs separated by <= this gap are merged.
    time_col
        Name of timestamp column.
    score_col
        Name of score column.

    Returns
    -------
    threshold, rois
        Robust threshold and list of merged ROIs.
    """
    if time_col not in df_scores.columns or score_col not in df_scores.columns:
        raise ValueError(f"df_scores must contain columns {time_col!r} and {score_col!r}.")

    df = df_scores[[time_col, score_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).reset_index(drop=True)

    scores = df[score_col].to_numpy(dtype=float)
    threshold = robust_sigma_threshold(scores, nsig=nsig)

    active = np.isfinite(scores) & (scores > threshold)
    if (not np.any(active)) or (not np.isfinite(threshold)):
        return threshold, []

    edges = np.diff(active.astype(np.int8))
    starts = (np.where(edges == 1)[0] + 1).tolist()
    ends = (np.where(edges == -1)[0] + 1).tolist()

    if active[0]:
        starts = [0] + starts
    if active[-1]:
        ends = ends + [len(active)]

    raw_rois: list[ROI] = []
    for s_idx, e_idx in zip(starts, ends):
        t0 = df[time_col].iloc[s_idx] - pad
        t1 = df[time_col].iloc[e_idx - 1] + pad

        seg = scores[s_idx:e_idx]
        seg_f = seg[np.isfinite(seg)]
        peak = float(np.max(seg_f)) if seg_f.size else float("nan")
        summ = float(np.sum(seg_f)) if seg_f.size else float("nan")

        raw_rois.append(
            ROI(
                start=pd.Timestamp(t0),
                stop=pd.Timestamp(t1),
                peak_score=peak,
                sum_score=summ,
                n_frames=int(e_idx - s_idx),
            )
        )

    raw_rois.sort(key=lambda r: r.start)
    merged: list[ROI] = [raw_rois[0]]
    for r in raw_rois[1:]:
        prev = merged[-1]
        gap = r.start - prev.stop
        if gap <= pd.Timedelta(merge_gap):
            merged[-1] = ROI(
                start=min(prev.start, r.start),
                stop=max(prev.stop, r.stop),
                peak_score=float(np.nanmax([prev.peak_score, r.peak_score])),
                sum_score=float(np.nansum([prev.sum_score, r.sum_score])),
                n_frames=int(prev.n_frames + r.n_frames),
            )
        else:
            merged.append(r)

    return threshold, merged


def rois_to_dataframe(rois: list[ROI]) -> pd.DataFrame:
    """
    Convert ROI list to a DataFrame.
    """
    if not rois:
        return pd.DataFrame(columns=["start", "stop", "peak_score", "sum_score", "n_frames"])

    return pd.DataFrame(
        {
            "start": [r.start for r in rois],
            "stop": [r.stop for r in rois],
            "peak_score": [r.peak_score for r in rois],
            "sum_score": [r.sum_score for r in rois],
            "n_frames": [r.n_frames for r in rois],
        }
    )


def _utc_tag(ts: pd.Timestamp) -> str:
    """
    Format a UTC timestamp for filenames.
    """
    return pd.to_datetime(ts, utc=True).strftime("%Y%m%d_%H%M%S")


def plot_scores_with_rois(
    df_scores: pd.DataFrame,
    rois: list[ROI],
    *,
    threshold: Optional[float] = None,
    time_col: str = "time",
    score_col: str = "score",
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot frame-level scores with ROI overlays.

    Parameters
    ----------
    df_scores
        DataFrame containing time and score columns.
    rois
        List of detected ROIs.
    threshold
        Optional threshold line to draw.
    time_col
        Name of timestamp column.
    score_col
        Name of score column.
    title
        Optional plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    df = df_scores[[time_col, score_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[time_col], df[score_col], lw=0.7)
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Outlier score")
    ax.grid(ls=":", alpha=0.4)

    if title is not None:
        ax.set_title(title)

    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, ls="--", lw=1.0)

    for roi in rois:
        ax.axvspan(roi.start, roi.stop, alpha=0.2)

    fig.tight_layout()
    return fig, ax


def plot_roi_event(
    station: str,
    roi: ROI,
    df_scores: pd.DataFrame,
    zarr_path: str | Path,
    *,
    chans: int | tuple[int, ...] | list[int],
    pipe: PreprocessPipeline | None = None,
    plot_pad_minutes: float = 5.0,
    threshold: float | None = None,
    time_col: str = "time",
    score_col: str = "score",
    cmap=cmr.pride,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one ROI as a score panel plus a contiguous time-frequency waterfall.

    This function reads the lower panel directly from the underlying Zarr cube
    over the plotting interval, optionally applying a preprocessing pipeline.

    Parameters
    ----------
    station
        Station name used in panel annotation.
    roi
        ROI to plot. `roi.start` and `roi.stop` are assumed to already reflect
        the final padded/merged ROI bounds from ROI detection.
    df_scores
        DataFrame containing at least `time_col` and `score_col`.
    zarr_path
        Path to the source Zarr store.
    chans
        Channel selection passed to `read_time_range(...)`.
        Examples:
        - `1` for pol11
        - `(0, 1)` for Stokes-I pipelines
    pipe
        Optional preprocessing pipeline applied after reading the contiguous
        time range from Zarr.
    plot_pad_minutes
        Extra plotting context on either side of the ROI. This affects only the
        displayed time span, not the ROI definition itself.
    threshold
        Optional score threshold drawn as a dotted horizontal line.
    time_col
        Name of score timestamp column in `df_scores`.
    score_col
        Name of score column in `df_scores`.
    cmap
        Colormap used for the waterfall.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes array.
    """
    df = df_scores.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    roi_start = pd.to_datetime(roi.start, utc=True)
    roi_stop = pd.to_datetime(roi.stop, utc=True)

    plot_pad = pd.Timedelta(minutes=float(plot_pad_minutes))
    t_plot_min = roi_start - plot_pad
    t_plot_max = roi_stop + plot_pad

    # ------------------------------------------------------------------
    # Top panel: score time series
    # ------------------------------------------------------------------
    mask_plot = (df[time_col] >= t_plot_min) & (df[time_col] <= t_plot_max)
    df_plot = df.loc[mask_plot].sort_values(time_col)

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(12, 6.5),
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 3]),
    )
    ax0, ax1 = axs

    ax0.plot(df_plot[time_col], df_plot[score_col], lw=0.9)
    ax0.axvspan(roi_start, roi_stop, alpha=0.20)

    if threshold is not None and np.isfinite(threshold):
        ax0.axhline(threshold, ls=":", lw=1.2)

    ax0.set_ylabel("Score")
    ax0.grid(ls=":", alpha=0.4)
    ax0.set_xlim(t_plot_min, t_plot_max)
    ax0.tick_params(axis="x", labelbottom=False)

    roi_start_str = roi_start.round("1s").strftime("%Y-%m-%d %H:%M:%S UTC")
    roi_stop_str = roi_stop.round("1s").strftime("%Y-%m-%d %H:%M:%S UTC")

    ax0.text(
        0.01,
        0.95,
        (
            f"{roi_start_str} \u2192 {roi_stop_str}\n"
            f"peak={roi.peak_score:.3g}, sum={roi.sum_score:.3g}, "
            f"n_frames={roi.n_frames}\n"
            f"{station}"
        ),
        transform=ax0.transAxes,
        ha="left",
        va="top",
    )

    # ------------------------------------------------------------------
    # Bottom panel: direct contiguous read from Zarr
    # ------------------------------------------------------------------
    data_tf, times, _meta = read_time_range(
        zarr_path,
        start_utc=t_plot_min,
        stop_utc=t_plot_max,
        chans=chans,
        pipe=pipe,
    )

    if data_tf.size == 0 or len(times) == 0:
        ax1.text(0.5, 0.5, "No data in plotting window", ha="center", va="center")
        ax1.set_axis_off()
        fig.tight_layout()
        return fig, axs

    data_tf = np.asarray(data_tf)
    if data_tf.ndim != 2:
        raise ValueError(f"Expected plotted data with shape (T, F) after pipe application, got {data_tf.shape}.")

    nt, nfreq = data_tf.shape

    if nt < 2:
        ax1.text(0.5, 0.5, "Insufficient samples in plotting window", ha="center", va="center")
        ax1.set_axis_off()
        fig.tight_layout()
        return fig, axs

    # Use actual timestamps from the contiguous read
    times = pd.to_datetime(times, utc=True)

    # Build time edges for pcolormesh
    t_num = mdates.date2num(times.to_pydatetime())
    dt_days = np.median(np.diff(t_num))

    x_edges = np.concatenate(
        [
            [t_num[0] - 0.5 * dt_days],
            0.5 * (t_num[:-1] + t_num[1:]),
            [t_num[-1] + 0.5 * dt_days],
        ]
    )
    y_edges = np.arange(nfreq + 1) * (125.0 / 2048)

    mesh = ax1.pcolormesh(
        x_edges,
        y_edges,
        data_tf.T,
        shading="auto",
        cmap=cmap,
    )

    ax1.axvspan(roi_start, roi_stop, alpha=0.10)
    ax1.set_yticks(np.arange(0, 150, 25))
    ax1.set_ylabel("Frequency [MHz]")
    ax1.set_xlabel("UTC time")
    ax1.set_xlim(t_plot_min, t_plot_max)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="7%", pad=0.05)
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.tick_params(axis="x", direction="in", pad=-11, color="whitesmoke", labelcolor="whitesmoke")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    return fig, axs


def run_roi_search(
    zarr_path: str | Path,
    *,
    station: str,
    analysis_start_utc: str,
    analysis_stop_utc: str,
    out_dir: str | Path,
    window_seconds: float = 20 * 60,
    step_seconds: float = 5 * 60,
    context_hours: float = 24.0,
    stride_hours: float = 1.0,
    score_hours: float = 1.0,
    gap_hours: float = 0.0,
    quiet_fraction: float = 0.3,
    n_quiet: Optional[int] = None,
    k_fit: int = 128,
    k_pca: int = 16,
    min_finite_frac: float = 0.7,
    nsig: float = 3.0,
    pad_minutes: float = 5.0,
    merge_gap_minutes: float = 20.0,
    rfi_mask_start: int = 116,
    rfi_mask_stop: int = 384,
    random_state: int = 42,
) -> ROISearchResult:
    """
    Run a complete ROI search over a station-season Zarr product.

    Parameters
    ----------
    zarr_path
        Input Zarr path for one station/season.
    station
        Station label used in metadata and plot titles.
    analysis_start_utc
        Requested analysis start time in ``YYYYmmdd_HHMMSS``.
    analysis_stop_utc
        Requested analysis stop time in ``YYYYmmdd_HHMMSS``.
    out_dir
        Output directory for run products.
    window_seconds
        Frame duration for dataset extraction.
    step_seconds
        Frame step size for dataset extraction.
    context_hours
        Width of centered rolling context window for PCA.
    stride_hours
        How often to refit / rescore.
    score_hours
        Width of scored chunk per step.
    gap_hours
        Optional donut gap around the scored interval when fitting PCA.
    quiet_fraction
        Fraction of context frames used as quiet PCA training set.
    n_quiet
        Optional fixed number of quiet frames. If provided, overrides `quiet_fraction`.
    k_fit
        Number of PCA modes fit in the quiet background model.
    k_pca
        Number of PCA modes used in reconstruction during scoring.
    min_finite_frac
        Minimum finite fraction required for a frame to receive a score.
    nsig
        Robust sigma threshold multiplier for ROI detection.
    pad_minutes
        Time padding applied to each ROI boundary.
    merge_gap_minutes
        Merge ROIs separated by less than or equal to this gap.
    rfi_mask_start, rfi_mask_stop
        Frequency channel range to mask out during PCA and scoring.
    random_state
        Random seed used in randomized SVD.

    Returns
    -------
    ROISearchResult
        Paths and counts describing the completed run.
    """
    zarr_path = Path(zarr_path)
    out_dir = Path(out_dir)
    roi_plot_dir = out_dir / "rois"
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_plot_dir.mkdir(parents=True, exist_ok=True)

    ds_start_utc, ds_stop_utc = padded_utc_range(
        analysis_start_utc=analysis_start_utc,
        analysis_stop_utc=analysis_stop_utc,
        context_hours=context_hours,
    )

    zgroup = zarr.open_group(zarr_path, mode="r")
    zarr_attrs = dict(zgroup.attrs)
    zarr_startutc = zarr_attrs.get("startutc")
    zarr_stoputc = zarr_attrs.get("stoputc")

    pipe_i = (
        PreprocessPipeline(input_space="linear")
        .with_metadata(
            zarr_path=str(zarr_path),
            station=station,
            notes="Stokes I + safe_db",
        )
        .add(step_stokes_i())
        .add(step_safe_db(name="safe_db"))
    )

    ds = SpecscoutDataset(
        zarr_path,
        start_utc=ds_start_utc,
        stop_utc=ds_stop_utc,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        chans=(0, 1),
        pipe=pipe_i,
        return_meta=True,
    )

    example_x, _ = ds[0]
    example_x = np.asarray(example_x)
    if example_x.ndim != 2:
        raise ValueError(f"Expected Stokes I dataset frames with shape (T, F); got {example_x.shape}")

    _, nfreq = example_x.shape
    rfi_mask = np.ones((nfreq,), dtype=bool)
    ms = max(0, min(int(rfi_mask_start), nfreq))
    me = max(0, min(int(rfi_mask_stop), nfreq))
    if me > ms:
        rfi_mask[ms:me] = False

    qs = QuietSelector(
        method="p99",
        quiet_fraction=quiet_fraction,
        freq_mask=rfi_mask,
    )

    bg = RollingPCABackground(
        k=k_fit,
        center=True,
        freq_mask=rfi_mask,
        use_randomized=True,
        n_iter=2,
        random_state=random_state,
    )

    score_kwargs = dict(
        method="p99",
        positive_only=True,
        min_finite_frac=min_finite_frac,
    )

    runner = RollingPCARunner(
        ds=ds,
        quiet_selector=qs,
        background=bg,
        context_hours=context_hours,
        stride_hours=stride_hours,
        score_hours=score_hours,
        gap_hours=gap_hours,
        n_quiet=n_quiet,
        k_pca=k_pca,
        score_kwargs=score_kwargs,
        store_masked=True,
    )

    times: list[pd.Timestamp] = []
    frame_idx: list[int] = []
    scores: list[float] = []

    for res in runner.run(
        analysis_start_utc=analysis_start_utc,
        analysis_stop_utc=analysis_stop_utc,
    ):
        for score, meta in zip(res.scores, res.metas):
            times.append(pd.Timestamp(meta.start_time_utc))
            frame_idx.append(int(meta.frame_idx))
            scores.append(float(score))

    df_scores = (
        pd.DataFrame({"time": times, "frame_idx": frame_idx, "score": scores}).sort_values("time").reset_index(drop=True)
    )

    threshold, rois = find_rois_from_scores(
        df_scores,
        nsig=nsig,
        pad=timedelta(minutes=pad_minutes),
        merge_gap=timedelta(minutes=merge_gap_minutes),
    )
    df_rois = rois_to_dataframe(rois)

    config = {
        "zarr_path": str(zarr_path),
        "station": station,
        "analysis_start_utc": analysis_start_utc,
        "analysis_stop_utc": analysis_stop_utc,
        "requested_processing_start_utc": ds_start_utc,
        "requested_processing_stop_utc": ds_stop_utc,
        "zarr_startutc": zarr_startutc,
        "zarr_stoputc": zarr_stoputc,
        "product": "stokes_i",
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "context_hours": context_hours,
        "stride_hours": stride_hours,
        "score_hours": score_hours,
        "gap_hours": gap_hours,
        "quiet_method": "p99",
        "quiet_fraction": quiet_fraction,
        "n_quiet": n_quiet,
        "score_method": "p99",
        "k_fit": k_fit,
        "k_pca": k_pca,
        "min_finite_frac": min_finite_frac,
        "nsig": nsig,
        "pad_minutes": pad_minutes,
        "merge_gap_minutes": merge_gap_minutes,
        "rfi_mask_start": ms,
        "rfi_mask_stop": me,
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "n_scores": int(len(df_scores)),
        "n_rois": int(len(df_rois)),
    }

    scores_path = out_dir / "scores.pkl"
    rois_path = out_dir / "rois.pkl"
    config_path = out_dir / "config.json"
    summary_plot_path = out_dir / "scores_with_rois.png"

    df_scores.to_pickle(scores_path)
    df_rois.to_pickle(rois_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    fig, _ax = plot_scores_with_rois(
        df_scores,
        rois,
        threshold=threshold,
        title=f"{station}: Stokes I p99 ROI search",
    )
    fig.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for i, roi in enumerate(rois):
        pipe_i = PreprocessPipeline(input_space="linear").add(step_stokes_i()).add(step_safe_db(name="safe_db"))

        fig, axs = plot_roi_event(
            station=station,
            roi=roi,
            df_scores=df_scores,
            zarr_path=zarr_path,
            chans=(0, 1),
            pipe=pipe_i,
            plot_pad_minutes=5.0,
            threshold=threshold,
        )

        roi_tag = _utc_tag(roi.start)
        fig.savefig(
            roi_plot_dir / f"roi_{i:04d}_{roi_tag}.png",
            dpi=144,
            bbox_inches="tight",
        )
        plt.close(fig)

    return ROISearchResult(
        out_dir=out_dir,
        scores_path=scores_path,
        rois_path=rois_path,
        config_path=config_path,
        summary_plot_path=summary_plot_path,
        roi_plot_dir=roi_plot_dir,
        n_scores=len(df_scores),
        n_rois=len(df_rois),
    )
