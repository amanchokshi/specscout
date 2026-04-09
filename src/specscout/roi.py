"""
ROI detection and search workflow utilities for specscout.

This module defines:

- the ROI data model
- robust thresholding and ROI grouping from frame-level scores
- the main workflow for running rolling quiet-PCA search over a
  station-season Zarr product

The search workflow is configurable in terms of:
- detection product / preprocessing pipeline
- quiet-frame selection metric
- residual scoring metric
- ROI thresholding and merging parameters

Plotting helpers live in `specscout.viz`. CLI-level defaults such as the
standard Stokes-I detection configuration should live in `specscout.cli`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from .dataset import SpecscoutDataset
from .outlier import QuietSelector, RollingPCABackground
from .preprocess import PreprocessPipeline
from .rolling import RollingPCARunner, padded_utc_range
from .viz.static import plot_roi_event, plot_scores_with_rois


@dataclass(frozen=True)
class ROI:
    """
    Region of interest in time.

    Parameters
    ----------
    start
        Inclusive ROI start time after padding / merging.
    stop
        Inclusive ROI stop time after padding / merging.
    peak_score
        Maximum frame score within the original above-threshold run(s).
    sum_score
        Sum of frame scores within the original above-threshold run(s).
    n_frames
        Number of above-threshold frames contributing before padding / merge.
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
        1D array of scores. NaNs and infs are ignored.
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

    A frame is active if its score is finite and exceeds a robust threshold.
    Contiguous active runs are converted into ROIs, then padded and merged.

    Parameters
    ----------
    df_scores
        DataFrame containing at least columns `time_col` and `score_col`.
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
    Convert a list of ROI objects to a DataFrame.
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


def _sanitize_tag(text: str) -> str:
    """
    Convert a short label to a filename-safe tag.
    """
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "unknown"


def _metric_label(kwargs: dict[str, Any]) -> str:
    """
    Generate a short human-readable label for a metric/config dict.
    """
    method = str(kwargs.get("method", "unknown"))

    if method == "lp":
        p = kwargs.get("p", None)
        if p is None:
            return "lp"
        p_float = float(p)
        if p_float.is_integer():
            return f"l{int(p_float)}"
        return f"lp{p_float:g}"

    if method == "topk_sum":
        topk = kwargs.get("topk", None)
        return f"topk{topk}" if topk is not None else "topk"

    if method == "percentile":
        q = kwargs.get("q", None)
        if q is None:
            return "percentile"
        q_float = float(q)
        if q_float.is_integer():
            return f"p{int(q_float)}"
        return f"p{q_float:g}"

    return method


def run_roi_search(
    zarr_path: str | Path,
    *,
    station: str,
    analysis_start_utc: str,
    analysis_stop_utc: str,
    out_dir: str | Path,
    detect_chans: int | tuple[int, ...] = (0, 1),
    detect_pipe: PreprocessPipeline | None,
    plot_chans: int | tuple[int, ...] | None = None,
    plot_pipe: PreprocessPipeline | None = None,
    window_seconds: float = 20 * 60,
    step_seconds: float = 5 * 60,
    context_hours: float = 24.0,
    stride_hours: float = 1.0,
    score_hours: float = 1.0,
    gap_hours: float = 0.0,
    n_quiet: int | None = None,
    k_fit: int = 128,
    k_pca: int = 16,
    nsig: float = 3.0,
    pad_minutes: float = 5.0,
    merge_gap_minutes: float = 20.0,
    rfi_mask_start: int = 116,
    rfi_mask_stop: int = 384,
    random_state: int = 42,
    quiet_selector_kwargs: dict[str, Any],
    score_kwargs: dict[str, Any],
    plot_pad_minutes: float = 5.0,
    save_plots: bool = True,
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
    detect_chans
        Channel selection used to build the detection dataset.
    detect_pipe
        Preprocessing pipeline used for detection. Must not be None.
    plot_chans
        Channel selection used for ROI plotting. Defaults to `detect_chans`.
    plot_pipe
        Preprocessing pipeline used for ROI plotting. Defaults to `detect_pipe`.
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
    n_quiet
        Optional fixed number of quiet frames.
    k_fit
        Number of PCA modes fit in the quiet background model.
    k_pca
        Number of PCA modes used in reconstruction during scoring.
    nsig
        Robust sigma threshold multiplier for ROI detection.
    pad_minutes
        Time padding applied to each ROI boundary.
    merge_gap_minutes
        Merge ROIs separated by <= this gap.
    rfi_mask_start, rfi_mask_stop
        Frequency channel range to mask out during PCA and scoring.
    random_state
        Random seed used in randomized SVD.
    quiet_selector_kwargs
        Fully formed kwargs passed to `QuietSelector`. The effective `freq_mask`
        is overwritten internally to ensure consistency with the detection
        dataset.
    score_kwargs
        Fully formed kwargs passed to rolling scoring.
    plot_pad_minutes
        Extra context on either side of each ROI in ROI quicklook plots.
    save_plots
        If True, write summary and ROI quicklook plots.

    Returns
    -------
    ROISearchResult
        Paths and counts describing the completed run.
    """
    if detect_pipe is None:
        raise ValueError("detect_pipe must be provided.")

    zarr_path = Path(zarr_path)
    out_dir = Path(out_dir)
    roi_plot_dir = out_dir / "rois"
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_plot_dir.mkdir(parents=True, exist_ok=True)

    if plot_chans is None:
        plot_chans = detect_chans
    if plot_pipe is None:
        plot_pipe = detect_pipe

    ds_start_utc, ds_stop_utc = padded_utc_range(
        analysis_start_utc=analysis_start_utc,
        analysis_stop_utc=analysis_stop_utc,
        context_hours=context_hours,
    )

    zgroup = zarr.open_group(zarr_path, mode="r")
    zarr_attrs = dict(zgroup.attrs)
    zarr_startutc = zarr_attrs.get("startutc")
    zarr_stoputc = zarr_attrs.get("stoputc")

    ds = SpecscoutDataset(
        zarr_path,
        start_utc=ds_start_utc,
        stop_utc=ds_stop_utc,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        chans=detect_chans,
        pipe=detect_pipe,
        return_meta=True,
    )

    example_x, _ = ds[0]
    example_x = np.asarray(example_x)
    if example_x.ndim != 2:
        raise ValueError(
            "Expected detection dataset frames with shape (T, F); "
            f"got {example_x.shape}. Detection pipe should usually reduce "
            "the selected channels to a 2D product."
        )

    _, nfreq = example_x.shape
    rfi_mask = np.ones((nfreq,), dtype=bool)
    ms = max(0, min(int(rfi_mask_start), nfreq))
    me = max(0, min(int(rfi_mask_stop), nfreq))
    if me > ms:
        rfi_mask[ms:me] = False

    quiet_selector_kwargs_final = dict(quiet_selector_kwargs)
    quiet_selector_kwargs_final["freq_mask"] = rfi_mask

    score_kwargs_final = dict(score_kwargs)

    quiet_label = _sanitize_tag(_metric_label(quiet_selector_kwargs_final))
    score_label = _sanitize_tag(_metric_label(score_kwargs_final))

    qs = QuietSelector(**quiet_selector_kwargs_final)

    bg = RollingPCABackground(
        k=k_fit,
        center=True,
        # freq_mask=rfi_mask,
        use_randomized=True,
        n_iter=2,
        random_state=random_state,
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
        score_kwargs=score_kwargs_final,
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
        pd.DataFrame(
            {
                "time": times,
                "frame_idx": frame_idx,
                "score": scores,
            }
        )
        .sort_values("time")
        .reset_index(drop=True)
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
        "detect_chans": list(detect_chans) if not isinstance(detect_chans, int) else [detect_chans],
        "plot_chans": list(plot_chans) if not isinstance(plot_chans, int) else [plot_chans],
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "context_hours": context_hours,
        "stride_hours": stride_hours,
        "score_hours": score_hours,
        "gap_hours": gap_hours,
        "quiet_selector_kwargs": {k: v for k, v in quiet_selector_kwargs_final.items() if k != "freq_mask"},
        "score_kwargs": score_kwargs_final,
        "n_quiet": n_quiet,
        "k_fit": k_fit,
        "k_pca": k_pca,
        "nsig": nsig,
        "pad_minutes": pad_minutes,
        "merge_gap_minutes": merge_gap_minutes,
        "rfi_mask_start": ms,
        "rfi_mask_stop": me,
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "quiet_label": quiet_label,
        "score_label": score_label,
        "save_plots": bool(save_plots),
        "plot_pad_minutes": float(plot_pad_minutes),
        "n_scores": int(len(df_scores)),
        "n_rois": int(len(df_rois)),
        "detect_pipe": detect_pipe.to_dict(),
        "plot_pipe": plot_pipe.to_dict() if plot_pipe is not None else None,
    }

    scores_path = out_dir / "scores.pkl"
    rois_path = out_dir / "rois.pkl"
    config_path = out_dir / "config.json"
    summary_plot_path = out_dir / f"scores_with_rois_q-{quiet_label}_s-{score_label}.png"

    df_scores.to_pickle(scores_path)
    df_rois.to_pickle(rois_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    if save_plots:
        fig, _ax = plot_scores_with_rois(
            df_scores,
            rois,
            threshold=threshold,
            title=(f"{station}: ROI search (quiet={quiet_label}, score={score_label})"),
        )
        fig.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        for i, roi in enumerate(rois):
            fig, _axs = plot_roi_event(
                station=station,
                roi=roi,
                df_scores=df_scores,
                zarr_path=zarr_path,
                chans=plot_chans,
                pipe=plot_pipe,
                plot_pad_minutes=plot_pad_minutes,
                threshold=threshold,
                quiet_label=quiet_label,
                score_label=score_label,
            )

            roi_tag = _utc_tag(roi.start)
            fig.savefig(
                roi_plot_dir / f"roi_{i:04d}_{roi_tag}_q-{quiet_label}_s-{score_label}.png",
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
