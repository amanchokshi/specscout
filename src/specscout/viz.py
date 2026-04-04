"""
Interactive visualization helpers for specscout Zarr cubes (Jupyter-first).

This module provides:

1) `specscout_scrubber`
   A rolling-window viewer with an ipywidgets slider. Frames are read lazily
   from the Zarr store via `specscout.patches.read_patch`, so startup is fast
   and memory use is bounded. A preprocessing pipeline (PreprocessPipeline)
   can be applied per frame.

2) `save_scrubber_frames`
   Batch-render a sequence of sequential frames to PNGs using the same frame
   definition as the scrubber.

3) `animate_specscout_streaming`
   A streaming Matplotlib animation over an arbitrary UTC range, with optional
   PNG frame dumping and MP4 saving.

4) Outlier helpers (for PCA/SVD workflows)
   - `plot_outliers`: plot a grid of non-sequential outlier frames.
   - `outlier_scrubber`: scrub through a list of non-sequential outlier frames.
   - `save_outlier_frames`: save one or many outlier frames to disk.

Design
------
- Index planning (UTC/seconds -> frame indices) lives in `specscout.core`.
- Patch extraction lives in `specscout.patches` and `SpecscoutDataset`.
- This module focuses on plotting/widgets/animation only.

Units / colorbar labeling
-------------------------
Visualization functions do not take an explicit `units` argument.
Instead, if a preprocessing pipeline is provided, the colorbar is labeled using
`pipe.output_space`. If `pipe.output_space` is missing/None (or no pipeline is
provided), units are assumed to be "linear".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

import cmasher as cmr
import ipywidgets as widgets
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .core import (
    CHAN_LABELS,
    freq_axis_from_attrs,
    parse_utc,
)
from .dataset import FrameMeta, SpecscoutDataset, plan_frames
from .patches import PatchSpec, open_cube, read_patch, read_time_range
from .preprocess import PreprocessPipeline

...
if TYPE_CHECKING:
    from .roi import ROI


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
    quiet_label: str | None = None,
    score_label: str | None = None,
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
    quiet_label
        Short label for the quiet-selector method/config.
    score_label
        Short label for the scoring method/config.
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

    if quiet_label is not None or score_label is not None:
        qs_label = f"quiet={quiet_label or 'unknown'}, score={score_label or 'unknown'}\n"
    else:
        qs_label = ""

    label_lines = [
        f"{roi_start_str} \u2192 {roi_stop_str}",
        (f"peak={roi.peak_score:.3g}, sum={roi.sum_score:.3g}, n_frames={roi.n_frames}"),
        f"{qs_label}{station}",
    ]

    ax0.text(
        0.01,
        0.95,
        "\n".join(label_lines),
        transform=ax0.transAxes,
        ha="left",
        va="top",
    )

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
        ax1.text(
            0.5,
            0.5,
            "Insufficient samples in plotting window",
            ha="center",
            va="center",
        )
        ax1.set_axis_off()
        fig.tight_layout()
        return fig, axs

    times = pd.to_datetime(times, utc=True)
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
    cbar.ax.tick_params(
        axis="x",
        direction="in",
        pad=-11,
        color="whitesmoke",
        labelcolor="whitesmoke",
    )

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    return fig, axs


@dataclass(frozen=True)
class ScrubberResult:
    """
    Container for objects returned by `specscout_scrubber` and `outlier_scrubber`.

    Attributes
    ----------
    fig
        Matplotlib figure holding the image.
    slider
        Slider widget controlling the current frame index.
    update
        Callback that updates the image for a given frame index.
    n_frames
        Total number of frames available (sequential frames or number of outliers).
    """

    fig: plt.Figure
    slider: widgets.IntSlider
    update: Callable[[int], None]
    n_frames: int


def _infer_units(pipe: Optional[PreprocessPipeline]) -> str:
    """
    Return a short unit label for the colorbar.

    Convention:
    - If `pipe` is None, assume "linear".
    - If `pipe.output_space` is missing/None, assume "linear".
    - Otherwise, use `str(pipe.output_space)`.

    Notes
    -----
    This is intentionally simple: units here refer to the *semantic space* of the
    numbers (linear/db/z/compressed), not physical units.
    """
    if pipe is None:
        return "linear"
    out_units = getattr(pipe, "output_space", None)
    return "linear" if out_units is None else str(out_units)


def _vlims_for_image(
    img: np.ndarray,
    *,
    clim_percentiles: Tuple[float, float],
    vlims: Optional[Tuple[float, float]],
) -> tuple[float, float]:
    """
    Determine (vmin, vmax) for an image.

    Parameters
    ----------
    img
        Image array (2D), possibly containing NaNs.
    clim_percentiles
        (low, high) percentiles used when `vlims` is None.
    vlims
        Optional fixed (vmin, vmax) override.

    Returns
    -------
    vmin, vmax
        Color limits as floats.

    Notes
    -----
    If `vlims` is None, limits are computed *per image* using nanpercentiles.
    """
    if vlims is not None:
        return float(vlims[0]), float(vlims[1])
    lo, hi = clim_percentiles
    vmin = float(np.nanpercentile(img, lo))
    vmax = float(np.nanpercentile(img, hi))
    return vmin, vmax


def _make_frame_loader(
    *,
    cube,
    time_axis,
    plan,
    chan: int,
    pipe: Optional[PreprocessPipeline],
) -> Callable[[int], tuple[np.ndarray, FrameMeta]]:
    """
    Return a function that loads (frame, meta) for a given sequential frame index.

    All data access is routed through `read_patch()` to keep slicing consistent
    with dataset generation.

    Parameters
    ----------
    cube, time_axis
        Opened Zarr cube and time axis.
    plan
        FramePlan from `core.plan_frames`.
    chan
        Cube channel index to load.
    pipe
        Optional preprocessing pipeline applied per frame.

    Returns
    -------
    load
        Callable `load(frame_idx) -> (frame, meta)` where `frame` is a 2D array (T, F).

    Notes
    -----
    No unit conversion is performed here. If you want dB, whitening, clipping,
    etc., build a `PreprocessPipeline` that includes those steps.
    """
    spec = PatchSpec(
        window_n=plan.window_n,
        step_n=plan.step_n,
        f_start=0,
        f_stop=None,  # full band
        chans=(chan,),  # one cube channel -> returns (T, F)
    )

    def load(frame_idx: int) -> tuple[np.ndarray, FrameMeta]:
        frame_idx = int(frame_idx)
        t_start_idx = plan.i_start + frame_idx * plan.step_n

        p = read_patch(
            cube,
            time_axis,
            spec,
            t_start_idx=t_start_idx,
            dtype=np.float32,
        )
        frame = p.data  # (T, F) because single chan selected

        meta = FrameMeta(
            t_start_idx=p.t_start_idx,
            t_end_idx=p.t_start_idx + plan.window_n,
            frame_idx=frame_idx,
            start_time_utc=p.start_time_utc,
            dt_s=plan.dt_s,
        )

        if pipe is not None:
            frame = pipe(frame, meta)

        return frame, meta

    return load


def specscout_scrubber(
    zarr_path: str,
    *,
    start_utc: str,
    stop_utc: str,
    window_seconds: float,
    step_seconds: float,
    chan: int = 0,
    cmap=cmr.pride,
    pipe: Optional[PreprocessPipeline] = None,
    clim_percentiles: Tuple[float, float] = (1.0, 99.0),
    vlims: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (7.5, 5.5),
) -> ScrubberResult:
    """
    Create an interactive rolling-window viewer for a specscout Zarr cube.

    Parameters
    ----------
    zarr_path
        Path to the Zarr store directory.
    start_utc, stop_utc
        Time range to scrub through, format ``YYYYmmdd_HHMMSS`` (UTC).
    window_seconds, step_seconds
        Window length and step size in seconds. Rounded to nearest whole number
        of samples based on the cube cadence.
    chan
        Cube channel index (0..nchan-1).
    cmap
        Matplotlib colormap.
    pipe
        Optional `PreprocessPipeline` applied to each frame.
        The colorbar label uses `pipe.output_space` when available.
    clim_percentiles
        (low, high) percentiles used to determine color limits when `vlims` is None.
        Limits are computed per frame (dynamic) unless `vlims` is provided.
    vlims
        Optional fixed (vmin, vmax) color limits. If provided, these limits are
        used for all frames. If None, limits are computed per frame.
    figsize
        Figure size in inches.

    Returns
    -------
    ScrubberResult
        Keep this referenced in the notebook to avoid garbage collection.
    """
    if chan not in CHAN_LABELS:
        raise ValueError("chan must be one of {0, 1, 2, 3}.")

    cube, attrs, time_axis = open_cube(zarr_path)
    nt, nfreq, nchan = cube.shape
    if chan >= nchan:
        raise ValueError(f"chan={chan} out of bounds for cube with nchan={nchan}.")

    plan = plan_frames(
        nt=nt,
        t0_unix_s=time_axis.t0_unix_s,
        dt_s=time_axis.dt_s,
        start_utc=start_utc,
        stop_utc=stop_utc,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        parse_utc=parse_utc,
    )

    freqs, x_label = freq_axis_from_attrs(attrs, nfreq)
    label = CHAN_LABELS[chan]
    units = _infer_units(pipe)

    load_frame = _make_frame_loader(
        cube=cube,
        time_axis=time_axis,
        plan=plan,
        chan=chan,
        pipe=pipe,
    )

    fig, ax = plt.subplots(figsize=figsize)

    img0, meta0 = load_frame(0)
    extent = [float(freqs[0]), float(freqs[-1]), float(window_seconds), 0.0]

    im = ax.imshow(
        img0,
        aspect="auto",
        interpolation="none",
        extent=extent,
        cmap=cmap,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time since start (s)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{label} ({units})")

    vmin0, vmax0 = _vlims_for_image(img0, clim_percentiles=clim_percentiles, vlims=vlims)
    im.set_clim(vmin0, vmax0)

    ax.set_title(f"{label} — frame 0/{plan.n_frames - 1} — start {meta0.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    def update(frame_idx: int) -> None:
        img, meta = load_frame(int(frame_idx))
        im.set_data(img)

        vmin, vmax = _vlims_for_image(img, clim_percentiles=clim_percentiles, vlims=vlims)
        im.set_clim(vmin, vmax)

        ax.set_title(
            f"{label} — frame {int(frame_idx)}/{plan.n_frames - 1} — start "
            f"{meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        fig.canvas.draw_idle()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=plan.n_frames - 1,
        step=1,
        description="Frame",
        continuous_update=True,
        readout=True,
        layout=widgets.Layout(width="80%"),
    )
    slider.observe(lambda change: update(int(change["new"])), names="value")

    return ScrubberResult(fig=fig, slider=slider, update=update, n_frames=plan.n_frames)


def save_scrubber_frames(
    zarr_path: str,
    *,
    out_dir: str | Path,
    start_utc: str,
    stop_utc: str,
    window_seconds: float,
    step_seconds: float,
    chan: int = 0,
    cmap=cmr.pride,
    pipe: Optional[PreprocessPipeline] = None,
    dpi: int = 240,
    clim_percentiles: Tuple[float, float] = (1.0, 99.0),
    vlims: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (7.5, 5.5),
) -> Path:
    """
    Save a sequence of sequential rolling-window frames to PNG files.

    Each file is named ``"{frame_idx:05d}_{isotime}.png"`` where `isotime` is the
    UTC timestamp of the frame start time.

    Parameters
    ----------
    zarr_path
        Path to Zarr store.
    out_dir
        Output directory (created if needed).
    start_utc, stop_utc
        UTC timestamps defining the span.
    window_seconds, step_seconds
        Frame window and stride in seconds.
    chan
        Cube channel index.
    cmap
        Matplotlib colormap.
    pipe
        Optional preprocessing pipeline applied per frame.
    dpi
        Output DPI.
    clim_percentiles
        Percentiles used for per-frame scaling when `vlims` is None.
    vlims
        Optional fixed (vmin, vmax) for all frames.
    figsize
        Figure size in inches.

    Returns
    -------
    pathlib.Path
        The output directory path.
    """
    if chan not in CHAN_LABELS:
        raise ValueError("chan must be one of {0, 1, 2, 3}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube, attrs, time_axis = open_cube(zarr_path)
    nt, nfreq, nchan = cube.shape
    if chan >= nchan:
        raise ValueError(f"chan={chan} out of bounds for cube with nchan={nchan}.")

    plan = plan_frames(
        nt=nt,
        t0_unix_s=time_axis.t0_unix_s,
        dt_s=time_axis.dt_s,
        start_utc=start_utc,
        stop_utc=stop_utc,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        parse_utc=parse_utc,
    )

    freqs, x_label = freq_axis_from_attrs(attrs, nfreq)
    label = CHAN_LABELS[chan]
    units = _infer_units(pipe)

    spec = PatchSpec(
        window_n=plan.window_n,
        step_n=plan.step_n,
        f_start=0,
        f_stop=None,
        chans=(chan,),
    )

    def load_frame(frame_idx: int) -> tuple[np.ndarray, FrameMeta]:
        frame_idx = int(frame_idx)
        t_start_idx = plan.i_start + frame_idx * plan.step_n

        p = read_patch(
            cube,
            time_axis,
            spec,
            t_start_idx=t_start_idx,
            dtype=np.float32,
        )
        frame = p.data

        meta = FrameMeta(
            t_start_idx=p.t_start_idx,
            t_end_idx=p.t_start_idx + plan.window_n,
            frame_idx=frame_idx,
            start_time_utc=p.start_time_utc,
            dt_s=plan.dt_s,
        )

        if pipe is not None:
            frame = pipe(frame, meta)

        return frame, meta

    fig, ax = plt.subplots(figsize=figsize)

    img0, meta0 = load_frame(0)
    extent = [float(freqs[0]), float(freqs[-1]), float(window_seconds), 0.0]

    im = ax.imshow(
        img0,
        aspect="auto",
        interpolation="none",
        extent=extent,
        cmap=cmap,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time since start (s)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{label} ({units})")

    vmin0, vmax0 = _vlims_for_image(img0, clim_percentiles=clim_percentiles, vlims=vlims)
    im.set_clim(vmin0, vmax0)

    ax.set_title(f"{label}: [0/{plan.n_frames - 1}]: {meta0.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    for frame_idx in range(plan.n_frames):
        img, meta = load_frame(frame_idx)
        im.set_data(img)

        vmin, vmax = _vlims_for_image(img, clim_percentiles=clim_percentiles, vlims=vlims)
        im.set_clim(vmin, vmax)

        ax.set_title(f"{label}: [{frame_idx}/{plan.n_frames - 1}]: {meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        isotime = meta.start_time_utc.strftime("%Y%m%d_%H%M")
        fname = out_dir / f"{frame_idx:05d}_{isotime}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return out_dir


def animate_specscout_streaming(
    zarr_path: str,
    *,
    start_utc: str,
    stop_utc: str,
    window_seconds: float,
    step_seconds: float,
    chan: int = 0,
    cmap=cmr.pride,
    pipe: Optional[PreprocessPipeline] = None,
    clim_percentiles: Tuple[float, float] = (1.0, 99.0),
    vlims: Optional[Tuple[float, float]] = None,
    out_dir: str | None = None,
    save_mp4: str | None = None,
    fps: int = 24,
    dpi: int = 150,
    interval_ms: int = 50,
    show: bool = True,
) -> FuncAnimation:
    """
    Streaming sliding-window waterfall animation over an arbitrary UTC time range.

    Parameters
    ----------
    zarr_path
        Path to Zarr store.
    start_utc, stop_utc
        UTC timestamps defining the span.
    window_seconds, step_seconds
        Frame window and stride in seconds.
    chan
        Cube channel index.
    cmap
        Matplotlib colormap.
    pipe
        Optional preprocessing pipeline applied per frame.
    clim_percentiles
        Percentiles used for per-frame scaling when `vlims` is None.
    vlims
        Optional fixed (vmin, vmax) for all frames.
    out_dir
        If provided, dumps PNG frames to this directory during animation.
    save_mp4
        If provided, saves the animation to this filepath.
    fps
        FPS used when saving MP4.
    dpi
        DPI used for PNG/MP4 output.
    interval_ms
        Animation update interval in milliseconds.
    show
        If True, displays the animation inline / in a window. If False, closes the figure.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Keep a reference to this object alive in notebooks.
    """
    if chan not in CHAN_LABELS:
        raise ValueError("chan must be one of {0, 1, 2, 3}.")

    cube, attrs, time_axis = open_cube(zarr_path)
    nt, nfreq, nchan = cube.shape
    if chan >= nchan:
        raise ValueError(f"chan={chan} out of bounds for cube with nchan={nchan}.")

    plan = plan_frames(
        nt=nt,
        t0_unix_s=time_axis.t0_unix_s,
        dt_s=time_axis.dt_s,
        start_utc=start_utc,
        stop_utc=stop_utc,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        parse_utc=parse_utc,
    )

    freqs, x_label = freq_axis_from_attrs(attrs, nfreq)
    label = CHAN_LABELS[chan]
    units = _infer_units(pipe)

    out_path: Optional[Path] = None
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    load_frame = _make_frame_loader(
        cube=cube,
        time_axis=time_axis,
        plan=plan,
        chan=chan,
        pipe=pipe,
    )

    fig, ax = plt.subplots()

    img0, meta0 = load_frame(0)
    extent = [float(freqs[0]), float(freqs[-1]), float(window_seconds), 0.0]

    im = ax.imshow(
        img0,
        aspect="auto",
        interpolation="none",
        extent=extent,
        cmap=cmap,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time since start (s)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{label} ({units})")

    vmin0, vmax0 = _vlims_for_image(img0, clim_percentiles=clim_percentiles, vlims=vlims)
    im.set_clim(vmin0, vmax0)

    ax.set_title(f"{label} — start {meta0.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    def update(frame_idx: int):
        img, meta = load_frame(int(frame_idx))
        im.set_data(img)

        vmin, vmax = _vlims_for_image(img, clim_percentiles=clim_percentiles, vlims=vlims)
        im.set_clim(vmin, vmax)

        ax.set_title(
            f"{label}: [ {int(frame_idx)}/{plan.n_frames - 1} ]: {meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

        if out_path is not None:
            iso = meta.start_time_utc.strftime("%Y%m%d_%H%M%S")
            fname = out_path / f"{int(frame_idx):05d}_{iso}.png"
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")

        return (im,)

    ani = FuncAnimation(
        fig,
        update,
        frames=plan.n_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_mp4 is not None:
        ani.save(save_mp4, fps=fps, dpi=dpi)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


# -----------------------------------------------------------------------------
# Outlier visualization helpers (non-sequential FrameMeta lists)
# -----------------------------------------------------------------------------


MaybeMeta = Union[FrameMeta, Sequence[FrameMeta]]


def _normalize_metas(metas: MaybeMeta) -> list[FrameMeta]:
    """
    Normalize `metas` into a list of FrameMeta.

    Parameters
    ----------
    metas
        Either a single FrameMeta or a sequence of FrameMeta objects.

    Returns
    -------
    list[FrameMeta]
        List form of input.
    """
    if isinstance(metas, FrameMeta):
        return [metas]
    return [m for m in metas]


def _load_outlier_frame(
    ds: SpecscoutDataset,
    meta: FrameMeta,
    *,
    pipe: Optional[PreprocessPipeline],
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, FrameMeta]:
    """
    Load one outlier frame using `SpecscoutDataset.load_by_t_start_idx`.

    Parameters
    ----------
    ds
        Dataset used for slicing (window_n, freq slice, chans).
    meta
        FrameMeta identifying the outlier via `meta.t_start_idx`.
    pipe
        Optional preprocessing pipeline applied for visualization.
    dtype
        dtype used for reading the patch from disk before applying `pipe`.

    Returns
    -------
    img, loaded_meta
        - img: 2D array (T, F) for plotting (if multi-channel, uses channel 0)
        - loaded_meta: FrameMeta returned by the dataset loader

    Notes
    -----
    This function intentionally does *not* depend on `ds.pipe`. It always loads
    raw patches (apply_pipe=False) and then applies the provided `pipe` argument.
    That makes it easy to compare “inspection space” vs “detection space” plots
    using the same dataset.
    """
    x, loaded_meta = ds.load_by_t_start_idx(
        meta.t_start_idx,
        frame_idx=getattr(meta, "frame_idx", -1),
        apply_pipe=False,
        return_meta=True,
        dtype=dtype,
    )

    if pipe is not None:
        x = pipe(np.asarray(x), loaded_meta)
    else:
        x = np.asarray(x)

    # If multi-channel data, show channel 0 by default for 2D waterfall plotting.
    if x.ndim == 3:
        x = x[:, :, 0]

    return x, loaded_meta


def outlier_scrubber(
    ds: SpecscoutDataset,
    metas: Sequence[FrameMeta],
    *,
    pipe: Optional[PreprocessPipeline] = None,
    cmap=cmr.pride,
    clim_percentiles: Tuple[float, float] = (1.0, 99.0),
    vlims: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (7.5, 5.5),
) -> ScrubberResult:
    """
    Create an interactive scrubber over a list of outliers (non-sequential).

    Parameters
    ----------
    ds
        Dataset used for slicing (window length, frequency slice, channels).
    metas
        Sequence of FrameMeta objects defining outliers to view. The slider index
        selects from this list in the given order.
    pipe
        Optional preprocessing pipeline applied per outlier before plotting.
    cmap
        Matplotlib colormap.
    clim_percentiles
        (low, high) percentiles used for per-frame color scaling when `vlims` is None.
        Outlier scrubbers are intentionally *per-frame* by default to avoid hiding
        structure due to a single global extreme outlier.
    vlims
        Optional fixed (vmin, vmax) applied to all outliers.
    figsize
        Figure size in inches.

    Returns
    -------
    ScrubberResult
        Same return container as `specscout_scrubber`.
    """
    metas_list = list(metas)
    if len(metas_list) == 0:
        raise ValueError("metas must be non-empty.")

    freqs, x_label = ds.freq_axis()
    fp = ds.plan.frame_plan
    window_seconds = float(fp.window_n * fp.dt_s)
    extent = [float(freqs[0]), float(freqs[-1]), window_seconds, 0.0]
    units = _infer_units(pipe)

    fig, ax = plt.subplots(figsize=figsize)

    img0, meta0 = _load_outlier_frame(ds, metas_list[0], pipe=pipe)
    im = ax.imshow(
        img0,
        aspect="auto",
        interpolation="none",
        extent=extent,
        cmap=cmap,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time since start (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"({units})")

    def _set_title(i: int, m: FrameMeta) -> None:
        ax.set_title(
            f"Outlier {i}/{len(metas_list) - 1} — "
            f"{m.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"(t_start_idx={m.t_start_idx})"
        )

    def update(outlier_idx: int) -> None:
        i = int(outlier_idx)
        img, m = _load_outlier_frame(ds, metas_list[i], pipe=pipe)
        im.set_data(img)

        vmin, vmax = _vlims_for_image(img, clim_percentiles=clim_percentiles, vlims=vlims)
        im.set_clim(vmin, vmax)

        _set_title(i, m)
        fig.canvas.draw_idle()

    vmin0, vmax0 = _vlims_for_image(img0, clim_percentiles=clim_percentiles, vlims=vlims)
    im.set_clim(vmin0, vmax0)
    _set_title(0, meta0)

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(metas_list) - 1,
        step=1,
        description="Outlier",
        continuous_update=True,
        readout=True,
        layout=widgets.Layout(width="80%"),
    )
    slider.observe(lambda change: update(int(change["new"])), names="value")

    return ScrubberResult(fig=fig, slider=slider, update=update, n_frames=len(metas_list))


def save_outlier_frames(
    ds: SpecscoutDataset,
    metas: MaybeMeta,
    *,
    out_dir: str | Path,
    pipe: Optional[PreprocessPipeline] = None,
    cmap=cmr.pride,
    clim_percentiles: Tuple[float, float] = (1.0, 99.0),
    vlims: Optional[Tuple[float, float]] = None,
    dpi: int = 240,
    figsize: Tuple[float, float] = (7.5, 5.5),
    name_template: str = "{i:05d}_{isotime}_idx{t_start_idx}.png",
) -> list[Path]:
    """
    Save one or many outlier frames to PNG files.

    Parameters
    ----------
    ds
        Dataset used for slicing (window length, frequency slice, channels).
    metas
        Either a single FrameMeta or a sequence of FrameMeta objects.
        Each meta identifies a frame via `meta.t_start_idx`.
    out_dir
        Output directory (created if needed).
    pipe
        Optional preprocessing pipeline applied per outlier before plotting.
    cmap
        Matplotlib colormap.
    clim_percentiles
        (low, high) percentiles used for per-frame scaling when `vlims` is None.
    vlims
        Optional fixed (vmin, vmax) for all saved frames.
    dpi
        Output DPI.
    figsize
        Figure size in inches.
    name_template
        Filename template evaluated with:
            - i: outlier index in the provided list (0..)
            - isotime: UTC string "YYYYmmdd_HHMMSS" of the frame start time
            - t_start_idx: integer sample index
        Example default:
            "{i:05d}_{isotime}_idx{t_start_idx}.png"

    Returns
    -------
    list[pathlib.Path]
        List of resolved filepaths written, in the same order as `metas`.

    Notes
    -----
    - Uses dataset slicing so outputs match exactly what your ML pipeline sees.
    - Uses a single figure reused across frames (efficient; no artist accumulation).
    - Per-frame scaling is used unless `vlims` is provided.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas_list = _normalize_metas(metas)
    if len(metas_list) == 0:
        raise ValueError("metas must be non-empty.")

    freqs, x_label = ds.freq_axis()
    fp = ds.plan.frame_plan
    window_seconds = float(fp.window_n * fp.dt_s)
    extent = [float(freqs[0]), float(freqs[-1]), window_seconds, 0.0]
    units = _infer_units(pipe)

    # Initialize with first image
    img0, meta0 = _load_outlier_frame(ds, metas_list[0], pipe=pipe)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        img0,
        aspect="auto",
        interpolation="none",
        extent=extent,
        cmap=cmap,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time since start (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"({units})")

    written: list[Path] = []

    for i, meta in enumerate(metas_list):
        img, m = _load_outlier_frame(ds, meta, pipe=pipe)
        im.set_data(img)

        vmin, vmax = _vlims_for_image(img, clim_percentiles=clim_percentiles, vlims=vlims)
        im.set_clim(vmin, vmax)

        ax.set_title(
            f"Outlier {i}/{len(metas_list) - 1} — "
            f"{m.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"(t_start_idx={m.t_start_idx})"
        )

        isotime = m.start_time_utc.strftime("%Y%m%d_%H%M%S")
        fname = name_template.format(i=i, isotime=isotime, t_start_idx=int(m.t_start_idx))
        path = (out_dir / fname).resolve()

        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)

    plt.close(fig)
    return written


if __name__ == "__main__":
    from preprocess import (
        ContextSpec,
        PreprocessPipeline,
        step_safe_db,
        # step_mad_whitener_from_zarr,
        # step_softclip,
    )

    zarr_path = "../data/MARS2_20240801_20240901.zarr"

    start_utc = "20240801_014000"
    stop_utc = "20240810_000000"

    step_seconds = 5 * 60
    window_seconds = 20 * 60
    ctx_window_seconds = 6 * 60 * 60
    donut_gap_seconds = 60 * 60
    whiten_alpha = 0.5
    asinh_alpha = 0.9
    softclip_level = 9.0
    chan = 0

    ctx = ContextSpec(
        baseline_seconds=ctx_window_seconds,
        mode="donut",
        gap_seconds=donut_gap_seconds,
    )

    pipe = (
        PreprocessPipeline(input_space="linear")
        .with_metadata(
            zarr_path=zarr_path,
            notes=(f"baseline={int(ctx_window_seconds / 3600)}h donut, chan={chan}: {CHAN_LABELS[chan]}"),
        )
        .add(
            step_safe_db(
                name="safe_db",
            )
        )
        # .add(
        #     step_mad_whitener_from_zarr(
        #         zarr_path,
        #         chan=chan,
        #         ctx=ctx,
        #         alpha=whiten_alpha,
        #         name=f"mad_whiten_{CHAN_LABELS[chan]}",
        #     )
        # )
        # .add(
        #     step_softclip(
        #         kind="asinh",
        #         clip=softclip_level,
        #         alpha=asinh_alpha,
        #         name="softclip",
        #         # defaults are in_space="z", out_space="compressed"
        #     )
        # )
    )

    print(pipe.summary())

    ani = animate_specscout_streaming(
        zarr_path,
        start_utc=start_utc,
        stop_utc=stop_utc,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        chan=chan,
        pipe=pipe,
        # vlims=(-2, 9),
        # save_mp4="pol00.mp4",
        fps=24,
        dpi=300,
        interval_ms=100,
        show=True,
    )

    # out_path = save_scrubber_frames(
    #     zarr_path,
    #     start_utc=start_utc,
    #     stop_utc=stop_utc,
    #     window_seconds=window_seconds,
    #     step_seconds=step_seconds,
    #     chan=chan,
    #     pipe=pipe,
    #     dpi=300,
    #     out_dir="./tmp"
    # )
