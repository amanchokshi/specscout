"""
Static plotting utilities for specscout.

This module contains non-interactive Matplotlib helpers for:

- plotting a single dataset frame
- plotting an arbitrary contiguous time range read directly from a Zarr store
- plotting ROI summaries and ROI event quicklooks
- saving sequential or arbitrary selected frames to PNG files
- recipe-based diagnostic plots for common instrumental and Stokes products

All plotting functions are compatible with preprocessing pipelines. In
particular, they support:

- raw cube channels in linear or dB space
- Stokes I products
- multi-channel products such as full Stokes (I, Q, U, V)

The plotting layer treats arrays as either:

- ``(T, F)``: one waterfall panel
- ``(T, F, C)``: a grid of ``C`` waterfall panels

Units / colorbar labeling
-------------------------
Visualization functions do not take an explicit units argument. If a
preprocessing pipeline is provided, colorbar labels are inferred from
``pipe.output_space``. Otherwise units are assumed to be ``"linear"``.

Recipe plots may use mixed semantics across panels, for example:

- ``stokes_I_dB``
- ``Q/I``
- ``U/I``
- ``V/I``

These recipe plots therefore support per-panel colorbar labels and colormaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import cmasher as cmr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..core import channel_names_from_indices, freq_axis_from_attrs
from ..dataset import FrameMeta, SpecscoutDataset
from ..patches import open_cube, read_time_range
from ..preprocess import PreprocessPipeline

if TYPE_CHECKING:
    from ..roi import ROI


def _infer_units(pipe: Optional[PreprocessPipeline]) -> str:
    """
    Infer a short unit / semantic-space label from a preprocessing pipeline.
    """
    if pipe is None:
        return "linear"
    out_units = getattr(pipe, "output_space", None)
    return "linear" if out_units is None else str(out_units)


def _centered_vlims_for_image(
    img: np.ndarray,
    *,
    clim_percentiles: tuple[float, float],
    vlims: tuple[float, float] | None,
) -> tuple[float, float]:
    """
    Determine symmetric color limits around zero for one image.

    When `vlims` is not provided, symmetric limits are estimated from finite
    pixels using the larger absolute value of the requested lower/upper
    percentiles. Fully non-finite images fall back to ``(-1, 1)``.
    """
    if vlims is not None:
        return float(vlims[0]), float(vlims[1])

    finite = np.isfinite(img)
    if not np.any(finite):
        return -1.0, 1.0

    lo, hi = clim_percentiles
    vlo = float(np.nanpercentile(img, lo))
    vhi = float(np.nanpercentile(img, hi))
    vmax = max(abs(vlo), abs(vhi))

    if not np.isfinite(vmax) or vmax == 0.0:
        return -1.0, 1.0

    return -vmax, vmax


def _vlims_for_image(
    img: np.ndarray,
    *,
    clim_percentiles: tuple[float, float],
    vlims: tuple[float, float] | None,
) -> tuple[float, float]:
    """
    Determine color limits for one image.

    When `vlims` is not provided, limits are estimated from finite pixels using
    the requested percentiles. Fully non-finite images fall back to ``(0, 1)``.
    Degenerate limits are also guarded against.
    """
    if vlims is not None:
        return float(vlims[0]), float(vlims[1])

    finite = np.isfinite(img)
    if not np.any(finite):
        return 0.0, 1.0

    lo, hi = clim_percentiles
    vmin = float(np.nanpercentile(img, lo))
    vmax = float(np.nanpercentile(img, hi))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0, 1.0

    return vmin, vmax


def _panel_layout(n_panels: int) -> tuple[int, int]:
    """
    Choose a simple rows/cols layout for a panel grid.
    """
    if n_panels <= 1:
        return 1, 1
    if n_panels <= 2:
        return 1, 2
    if n_panels <= 4:
        return 2, 2

    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def _resolve_panel_labels(
    *,
    channel_labels: Sequence[str] | None,
    pipe: PreprocessPipeline | None,
    chans: int | Sequence[int] | None,
    n_panels: int,
) -> list[str]:
    """
    Resolve panel labels for plotting.

    Priority
    --------
    1. Explicit `channel_labels`
    2. Pipeline output channel names
    3. Raw channel indices mapped through `channel_names_from_indices`
    4. Generic fallback labels
    """
    if channel_labels is not None:
        labels = [str(x) for x in channel_labels]
        if len(labels) != n_panels:
            raise ValueError("channel_labels length does not match number of panels.")
        return labels

    if pipe is not None:
        names = getattr(pipe, "output_channel_names", None)
        if names is not None:
            names_l = [str(x) for x in names]
            if len(names_l) == n_panels:
                return names_l
            if n_panels == 1 and len(names_l) >= 1:
                return [names_l[0]]

    if chans is not None:
        names_l = [str(x) for x in channel_names_from_indices(chans)]
        if len(names_l) == n_panels:
            return names_l
        if n_panels == 1 and len(names_l) >= 1:
            return [names_l[0]]

    return [f"ch{j}" for j in range(n_panels)]


def _time_edges_from_datetimes(times: pd.DatetimeIndex) -> np.ndarray:
    """
    Build pcolormesh x-edges from sample timestamps.

    Parameters
    ----------
    times
        UTC DatetimeIndex with at least two entries.
    """
    if len(times) < 2:
        raise ValueError("times must contain at least 2 samples.")
    t_num = mdates.date2num(times.to_pydatetime())
    dt_days = np.median(np.diff(t_num))
    return np.concatenate(
        [
            [t_num[0] - 0.5 * dt_days],
            0.5 * (t_num[:-1] + t_num[1:]),
            [t_num[-1] + 0.5 * dt_days],
        ]
    )


def _time_range_extent_from_frame(
    frame: np.ndarray,
    *,
    meta: FrameMeta,
    freqs: np.ndarray,
) -> list[float]:
    """
    Build imshow extent for a single frame waterfall.
    """
    t_seconds = float(frame.shape[0]) * float(meta.dt_s)
    return [float(freqs[0]), float(freqs[-1]), t_seconds, 0.0]


def _normalize_panel_arg(
    value,
    *,
    n_panels: int,
    name: str,
):
    """
    Normalize a scalar-or-sequence plotting argument to a per-panel list.
    """
    if value is None:
        return [None] * n_panels

    if isinstance(value, (list, tuple)):
        if len(value) != n_panels:
            raise ValueError(f"{name} sequence must match number of panels.")
        return list(value)

    return [value] * n_panels


def _load_frame_for_plot(
    ds: SpecscoutDataset,
    *,
    idx: int | None = None,
    meta: FrameMeta | None = None,
    pipe: PreprocessPipeline | None = None,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, FrameMeta]:
    """
    Load a single frame for plotting, optionally applying a plotting pipeline.
    """
    if (idx is None) == (meta is None):
        raise ValueError("Provide exactly one of idx or meta.")

    if idx is not None:
        x, loaded_meta = ds.load_by_t_start_idx(
            ds.plan.frame_plan.i_start + int(idx) * ds.plan.frame_plan.step_n,
            frame_idx=int(idx),
            apply_pipe=False,
            return_meta=True,
            dtype=dtype,
        )
    else:
        x, loaded_meta = ds.load_by_t_start_idx(
            int(meta.t_start_idx),
            frame_idx=int(meta.frame_idx),
            apply_pipe=False,
            return_meta=True,
            dtype=dtype,
        )

    x = np.asarray(x)
    if pipe is not None:
        x = pipe(x, loaded_meta)

    return x, loaded_meta


def _plot_loaded_frame(
    data: np.ndarray,
    *,
    ds: SpecscoutDataset,
    loaded_meta: FrameMeta,
    pipe: PreprocessPipeline | None = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    cbar_units: str | Sequence[str] | None = None,
    center_zero: bool | Sequence[bool] = False,
    figsize: tuple[float, float] = (8.5, 5.5),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot an already loaded dataset frame.
    """
    freqs, _x_label = ds.freq_axis()
    units = _infer_units(pipe)
    n_panels = data.shape[2] if data.ndim == 3 else 1

    resolved_labels = _resolve_panel_labels(
        channel_labels=channel_labels,
        pipe=pipe,
        chans=ds.plan.chans,
        n_panels=n_panels,
    )

    if title is None:
        title = f"Frame start {loaded_meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    return _plot_waterfall_grid(
        data,
        x_mode="frame",
        freqs=freqs,
        frame_meta=loaded_meta,
        channel_labels=resolved_labels,
        cmap=cmap,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        cbar_units=cbar_units if cbar_units is not None else units,
        center_zero=center_zero,
        figsize=figsize,
        title=title,
    )


def _plot_waterfall_grid(
    data: np.ndarray,
    *,
    x_mode: str,
    freqs: np.ndarray,
    frame_meta: FrameMeta | None = None,
    times: pd.DatetimeIndex | None = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    cbar_units: str | Sequence[str] = "linear",
    center_zero: bool | Sequence[bool] = False,
    figsize: tuple[float, float] = (8.5, 5.5),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Generic waterfall plotting for either one panel or a panel grid.

    Parameters
    ----------
    data
        Array of shape ``(T, F)`` or ``(T, F, C)``.
    x_mode
        Either ``"frame"`` or ``"timerange"``.
    freqs
        Frequency axis values.
    frame_meta
        Required when ``x_mode="frame"``.
    times
        Required when ``x_mode="timerange"``.
    channel_labels
        Optional labels for each panel.
    cmap
        Either a single colormap or a sequence of length `n_panels`.
    clim_percentiles
        Percentiles used when `vlims` is None.
    vlims
        Either None, a single `(vmin, vmax)` pair, or a sequence of per-panel
        `(vmin, vmax)` pairs.
    cbar_units
        Either a single colorbar label or a sequence of per-panel labels.
    center_zero
        Either a single bool or a sequence of per-panel bools. When True for a
        panel, color limits are chosen symmetrically around zero unless
        overridden by an explicit `vlims`.
    """
    arr = np.asarray(data)

    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError(f"Expected data with shape (T, F) or (T, F, C), got {arr.shape}.")

    _nt, nfreq, n_panels = arr.shape

    if nfreq != len(freqs):
        raise ValueError("Frequency axis length does not match data shape.")

    if channel_labels is None:
        channel_labels = [f"ch{j}" for j in range(n_panels)]
    else:
        channel_labels = list(channel_labels)
        if len(channel_labels) != n_panels:
            raise ValueError("channel_labels length does not match number of panels.")

    cmaps = _normalize_panel_arg(cmap, n_panels=n_panels, name="cmap")
    vlims_list = _normalize_panel_arg(vlims, n_panels=n_panels, name="vlims")
    units_list = _normalize_panel_arg(
        cbar_units,
        n_panels=n_panels,
        name="cbar_units",
    )
    center_zero_list = _normalize_panel_arg(
        center_zero,
        n_panels=n_panels,
        name="center_zero",
    )

    nrows, ncols = _panel_layout(n_panels)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize
            if n_panels == 1
            else (
                max(figsize[0], 4.8 * ncols),
                max(figsize[1], 3.8 * nrows),
            )
        ),
        squeeze=False,
        sharex=(x_mode == "timerange"),
        sharey=True,
    )
    axs_flat = axs.ravel()

    if x_mode == "frame":
        if frame_meta is None:
            raise ValueError("frame_meta must be provided when x_mode='frame'.")
        extent = _time_range_extent_from_frame(
            arr[:, :, 0],
            meta=frame_meta,
            freqs=freqs,
        )
    else:
        if times is None or len(times) < 2:
            raise ValueError("times must be provided with at least 2 samples when x_mode='timerange'.")
        x_edges = _time_edges_from_datetimes(times)
        y_edges = np.concatenate(
            [
                [freqs[0] - 0.5 * (freqs[1] - freqs[0])],
                0.5 * (freqs[:-1] + freqs[1:]),
                [freqs[-1] + 0.5 * (freqs[-1] - freqs[-2])],
            ]
        )

    for j in range(n_panels):
        ax = axs_flat[j]
        img = arr[:, :, j]
        panel_label = str(channel_labels[j])
        panel_cmap = cmaps[j]
        panel_vlims = vlims_list[j]
        panel_units = units_list[j]
        panel_center_zero = bool(center_zero_list[j])

        if x_mode == "frame":
            im = ax.imshow(
                img,
                aspect="auto",
                interpolation="none",
                extent=extent,
                cmap=panel_cmap,
            )
            ax.set_xlabel("Frequency [MHz]")
            ax.set_ylabel("Time since start (s)")
        else:
            im = ax.pcolormesh(
                x_edges,
                y_edges,
                img.T,
                shading="auto",
                cmap=panel_cmap,
            )
            ax.set_xlabel("UTC time")
            ax.set_ylabel("Frequency [MHz]")

            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        if panel_center_zero:
            vmin, vmax = _centered_vlims_for_image(
                img,
                clim_percentiles=clim_percentiles,
                vlims=panel_vlims,
            )
        else:
            vmin, vmax = _vlims_for_image(
                img,
                clim_percentiles=clim_percentiles,
                vlims=panel_vlims,
            )

        im.set_clim(vmin, vmax)
        ax.set_title(panel_label)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("" if panel_units is None else str(panel_units))

    for j in range(n_panels, len(axs_flat)):
        axs_flat[j].set_axis_off()

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axs


def _resolve_plot_pipe(
    ds: SpecscoutDataset,
    pipe: PreprocessPipeline | None,
) -> PreprocessPipeline | None:
    """
    Resolve the plotting pipeline.

    Priority
    --------
    1. Explicit `pipe` argument passed to the plotting function
    2. Dataset pipeline `ds.pipe`
    """
    return pipe if pipe is not None else ds.pipe


def plot_frame(
    ds: SpecscoutDataset,
    *,
    idx: int | None = None,
    meta: FrameMeta | None = None,
    pipe: PreprocessPipeline | None = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    cbar_units: str | Sequence[str] | None = None,
    center_zero: bool | Sequence[bool] = False,
    figsize: tuple[float, float] = (8.5, 5.5),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a single dataset frame.

    Exactly one of `idx` or `meta` must be provided.
    """

    pipe = _resolve_plot_pipe(ds, pipe)
    data, loaded_meta = _load_frame_for_plot(ds, idx=idx, meta=meta, pipe=pipe)

    return _plot_loaded_frame(
        data,
        ds=ds,
        loaded_meta=loaded_meta,
        pipe=pipe,
        channel_labels=channel_labels,
        cmap=cmap,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        cbar_units=cbar_units,
        center_zero=center_zero,
        figsize=figsize,
        title=title,
    )


def plot_time_range(
    zarr_path: str | Path,
    *,
    start_utc: str | pd.Timestamp,
    stop_utc: str | pd.Timestamp,
    chans: int | Sequence[int],
    pipe: PreprocessPipeline | None = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    cbar_units: str | Sequence[str] | None = None,
    center_zero: bool | Sequence[bool] = False,
    figsize: tuple[float, float] = (10.0, 5.5),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Read an arbitrary contiguous time range from a Zarr store and plot it.
    """
    data, times, _meta = read_time_range(
        zarr_path,
        start_utc=start_utc,
        stop_utc=stop_utc,
        chans=chans,
        pipe=pipe,
    )

    if data.size == 0 or len(times) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No data in requested time range",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return fig, np.asarray([ax])

    cube, attrs, _time_axis = open_cube(zarr_path)
    nfreq = data.shape[1]
    freqs_all, _x_label = freq_axis_from_attrs(attrs, cube.shape[1])
    freqs = np.asarray(freqs_all[:nfreq], dtype=float)
    units = _infer_units(pipe)

    n_panels = data.shape[2] if data.ndim == 3 else 1
    resolved_labels = _resolve_panel_labels(
        channel_labels=channel_labels,
        pipe=pipe,
        chans=chans,
        n_panels=n_panels,
    )

    if title is None:
        start_ts = pd.to_datetime(times[0], utc=True).round("1s")
        stop_ts = pd.to_datetime(times[-1], utc=True).round("1s")
        title = f"{start_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} → {stop_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    return _plot_waterfall_grid(
        data,
        x_mode="timerange",
        freqs=freqs,
        times=pd.to_datetime(times, utc=True),
        channel_labels=resolved_labels,
        cmap=cmap,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        cbar_units=cbar_units if cbar_units is not None else units,
        center_zero=center_zero,
        figsize=figsize,
        title=title,
    )


def plot_scores_with_rois(
    df_scores: pd.DataFrame,
    rois: list[ROI],
    *,
    threshold: float | None = None,
    time_col: str = "time",
    score_col: str = "score",
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot frame-level scores with ROI overlays.
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
    Plot one ROI as a score panel plus a contiguous waterfall quicklook.

    Notes
    -----
    The plotting pipeline must produce a 2D ``(T, F)`` array.
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
        f"{roi_start_str} → {roi_stop_str}",
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

    cube, attrs, _time_axis = open_cube(zarr_path)
    freqs, _x_label = freq_axis_from_attrs(attrs, cube.shape[1])
    freqs = np.asarray(freqs[:nfreq], dtype=float)

    times = pd.to_datetime(times, utc=True)
    x_edges = _time_edges_from_datetimes(times)
    y_edges = np.concatenate(
        [
            [freqs[0] - 0.5 * (freqs[1] - freqs[0])],
            0.5 * (freqs[:-1] + freqs[1:]),
            [freqs[-1] + 0.5 * (freqs[-1] - freqs[-2])],
        ]
    )

    mesh = ax1.pcolormesh(
        x_edges,
        y_edges,
        data_tf.T,
        shading="auto",
        cmap=cmap,
    )

    ax1.axvspan(roi_start, roi_stop, alpha=0.10)
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
    cbar.set_label(_infer_units(pipe))

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    return fig, axs


def _reject_db_recipe_pipe(
    pipe: PreprocessPipeline | None,
    *,
    func_name: str,
    expected: str,
) -> None:
    """
    Reject recipe plotting pipelines that already output dB-valued data.

    Parameters
    ----------
    pipe
        Optional preprocessing pipeline passed by the caller.
    func_name
        Public plotting function name for the error message.
    expected
        Short description of the expected input semantics.
    """
    if pipe is None:
        return

    out_space = getattr(pipe, "output_space", None)
    if out_space == "db":
        raise ValueError(
            f"{func_name} expects {expected}, but received a pipeline with "
            "output_space='db'. Pass raw linear inputs or a linear-valued "
            "transform pipeline instead."
        )


def _build_instrumental_quicklook(
    data: np.ndarray,
) -> tuple[np.ndarray, list[str], list, list[str], list[bool]]:
    """
    Build instrumental mixed quicklook:

    - pol00_dB
    - pol11_dB
    - pol01_mag_dB
    - pol01_phase
    """
    x = np.asarray(data)
    if x.ndim != 3 or x.shape[2] != 4:
        raise ValueError("Instrumental quicklook requires data with shape (T, F, 4).")

    pol00 = x[:, :, 0]
    pol11 = x[:, :, 1]
    pol01_mag = x[:, :, 2]
    pol01_phase = x[:, :, 3]

    eps = 1e-12

    def _safe_db_local(arr: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.array(arr, copy=True, dtype=np.float32)
            finite = np.isfinite(out)
            out[finite] = np.maximum(out[finite], eps)
            return 10.0 * np.log10(out)

    out = np.stack(
        [
            _safe_db_local(pol00),
            _safe_db_local(pol11),
            _safe_db_local(pol01_mag),
            np.asarray(pol01_phase, dtype=np.float32),
        ],
        axis=2,
    )

    labels = ["pol00_dB", "pol11_dB", "pol01_mag_dB", "pol01_phase"]
    cmaps = [cmr.pride, cmr.pride, cmr.pride, "PuOr"]
    units = ["db", "db", "db", "rad"]
    center_zero = [False, False, False, False]

    return out, labels, cmaps, units, center_zero


def _build_stokes_fractional_quicklook(
    data: np.ndarray,
) -> tuple[np.ndarray, list[str], list, list[str], list[tuple[float, float] | None], list[bool]]:
    """
    Build Stokes diagnostic quicklook:

    - stokes_I_dB
    - Q/I
    - U/I
    - V/I
    """
    x = np.asarray(data)
    if x.ndim != 3 or x.shape[2] != 4:
        raise ValueError("Stokes fractional quicklook requires data with shape (T, F, 4).")

    I = x[:, :, 0]
    Q = x[:, :, 1]
    U = x[:, :, 2]
    V = x[:, :, 3]

    eps = 1e-12
    I_safe = np.where(np.abs(I) > eps, I, np.nan)

    def _safe_db_local(arr: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.array(arr, copy=True, dtype=np.float32)
            finite = np.isfinite(out)
            out[finite] = np.maximum(out[finite], eps)
            return 10.0 * np.log10(out)

    out = np.stack(
        [
            _safe_db_local(I),
            np.asarray(Q / I_safe, dtype=np.float32),
            np.asarray(U / I_safe, dtype=np.float32),
            np.asarray(V / I_safe, dtype=np.float32),
        ],
        axis=2,
    )

    labels = ["stokes_I_dB", "Q/I", "U/I", "V/I"]
    cmaps = [cmr.pride, "PuOr", "PuOr", "PuOr"]
    units = ["db", "ratio", "ratio", "ratio"]
    vlims = [None, None, None, None]
    center_zero = [False, True, True, True]

    return out, labels, cmaps, units, vlims, center_zero


def plot_frame_instrumental_quicklook(
    ds: SpecscoutDataset,
    *,
    idx: int | None = None,
    meta: FrameMeta | None = None,
    pipe: PreprocessPipeline | None = None,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    figsize: tuple[float, float] = (10.0, 7.0),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a 4-panel instrumental quicklook for one frame.

    Expected pipeline output shape is ``(T, F, 4)`` with channels ordered as:
    ``(pol00, pol11, pol01_mag, pol01_phase)``.

    Expects raw linear instrumental channels
    """

    _reject_db_recipe_pipe(
        pipe,
        func_name="plot_frame_instrumental_quicklook",
        expected="linear instrumental inputs",
    )

    pipe = _resolve_plot_pipe(ds, pipe)
    data, loaded_meta = _load_frame_for_plot(ds, idx=idx, meta=meta, pipe=pipe)
    quick, labels, cmaps, units, center_zero = _build_instrumental_quicklook(data)

    if title is None:
        title = f"Instrumental quicklook — {loaded_meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    return _plot_loaded_frame(
        quick,
        ds=ds,
        loaded_meta=loaded_meta,
        pipe=None,
        channel_labels=labels,
        cmap=cmaps,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        cbar_units=units,
        center_zero=center_zero,
        figsize=figsize,
        title=title,
    )


def plot_time_range_instrumental_quicklook(
    zarr_path: str | Path,
    *,
    start_utc: str | pd.Timestamp,
    stop_utc: str | pd.Timestamp,
    chans: int | Sequence[int],
    pipe: PreprocessPipeline | None = None,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    figsize: tuple[float, float] = (11.0, 7.5),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a 4-panel instrumental quicklook over a contiguous time range.

    Expected pipeline output shape is ``(T, F, 4)`` with channels ordered as:
    ``(pol00, pol11, pol01_mag, pol01_phase)``.

    Expects raw linear instrumental channels
    """

    _reject_db_recipe_pipe(
        pipe,
        func_name="plot_frame_instrumental_quicklook",
        expected="linear instrumental inputs",
    )

    data, times, _meta = read_time_range(
        zarr_path,
        start_utc=start_utc,
        stop_utc=stop_utc,
        chans=chans,
        pipe=pipe,
    )

    if data.size == 0 or len(times) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data in requested time range", ha="center", va="center")
        ax.set_axis_off()
        return fig, np.asarray([ax])

    quick, labels, cmaps, units, center_zero = _build_instrumental_quicklook(data)

    cube, attrs, _time_axis = open_cube(zarr_path)
    freqs_all, _ = freq_axis_from_attrs(attrs, cube.shape[1])
    freqs = np.asarray(freqs_all[: quick.shape[1]], dtype=float)

    if title is None:
        start_ts = pd.to_datetime(times[0], utc=True).round("1s")
        stop_ts = pd.to_datetime(times[-1], utc=True).round("1s")
        title = (
            "Instrumental quicklook — "
            f"{start_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} → "
            f"{stop_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    return _plot_waterfall_grid(
        quick,
        x_mode="timerange",
        freqs=freqs,
        times=pd.to_datetime(times, utc=True),
        channel_labels=labels,
        cmap=cmaps,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        cbar_units=units,
        center_zero=center_zero,
        figsize=figsize,
        title=title,
    )


def plot_frame_stokes_fractional(
    ds: SpecscoutDataset,
    *,
    idx: int | None = None,
    meta: FrameMeta | None = None,
    pipe: PreprocessPipeline | None = None,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    figsize: tuple[float, float] = (10.0, 7.0),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a 4-panel Stokes diagnostic quicklook for one frame.

    Expected pipeline output shape is ``(T, F, 4)`` with channels ordered as:
    ``(stokes_I, stokes_Q, stokes_U, stokes_V)``.

    Expects linear Stokes IQUV inputs
    """

    _reject_db_recipe_pipe(
        pipe,
        func_name="plot_frame_instrumental_quicklook",
        expected="linear instrumental inputs",
    )

    pipe = _resolve_plot_pipe(ds, pipe)
    data, loaded_meta = _load_frame_for_plot(ds, idx=idx, meta=meta, pipe=pipe)
    quick, labels, cmaps, units, recipe_vlims, center_zero = _build_stokes_fractional_quicklook(data)

    use_vlims = recipe_vlims if vlims is None else vlims

    if title is None:
        title = f"Stokes diagnostic — {loaded_meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    return _plot_loaded_frame(
        quick,
        ds=ds,
        loaded_meta=loaded_meta,
        pipe=None,
        channel_labels=labels,
        cmap=cmaps,
        clim_percentiles=clim_percentiles,
        vlims=use_vlims,
        cbar_units=units,
        center_zero=center_zero,
        figsize=figsize,
        title=title,
    )


def plot_time_range_stokes_fractional(
    zarr_path: str | Path,
    *,
    start_utc: str | pd.Timestamp,
    stop_utc: str | pd.Timestamp,
    chans: int | Sequence[int],
    pipe: PreprocessPipeline | None = None,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    figsize: tuple[float, float] = (11.0, 7.5),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a 4-panel Stokes diagnostic quicklook over a contiguous time range.

    Expected pipeline output shape is ``(T, F, 4)`` with channels ordered as:
    ``(stokes_I, stokes_Q, stokes_U, stokes_V)``.

    Expects linear Stokes IQUV inputs
    """

    _reject_db_recipe_pipe(
        pipe,
        func_name="plot_frame_instrumental_quicklook",
        expected="linear instrumental inputs",
    )

    data, times, _meta = read_time_range(
        zarr_path,
        start_utc=start_utc,
        stop_utc=stop_utc,
        chans=chans,
        pipe=pipe,
    )

    if data.size == 0 or len(times) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data in requested time range", ha="center", va="center")
        ax.set_axis_off()
        return fig, np.asarray([ax])

    quick, labels, cmaps, units, recipe_vlims, center_zero = _build_stokes_fractional_quicklook(data)

    use_vlims = recipe_vlims if vlims is None else vlims

    cube, attrs, _time_axis = open_cube(zarr_path)
    freqs_all, _ = freq_axis_from_attrs(attrs, cube.shape[1])
    freqs = np.asarray(freqs_all[: quick.shape[1]], dtype=float)

    if title is None:
        start_ts = pd.to_datetime(times[0], utc=True).round("1s")
        stop_ts = pd.to_datetime(times[-1], utc=True).round("1s")
        title = (
            f"Stokes diagnostic — {start_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} → {stop_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    return _plot_waterfall_grid(
        quick,
        x_mode="timerange",
        freqs=freqs,
        times=pd.to_datetime(times, utc=True),
        channel_labels=labels,
        cmap=cmaps,
        clim_percentiles=clim_percentiles,
        vlims=use_vlims,
        cbar_units=units,
        center_zero=center_zero,
        figsize=figsize,
        title=title,
    )


def save_frame_sequence(
    ds: SpecscoutDataset,
    *,
    out_dir: str | Path,
    start_idx: int = 0,
    stop_idx: int | None = None,
    pipe: PreprocessPipeline | None = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    cbar_units: str | Sequence[str] | None = None,
    center_zero: bool | Sequence[bool] = False,
    dpi: int = 240,
    figsize: tuple[float, float] = (8.5, 5.5),
) -> list[Path]:
    """
    Save a sequential range of dataset frames to PNG files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if stop_idx is None:
        stop_idx = len(ds)

    written: list[Path] = []
    for idx in range(int(start_idx), int(stop_idx)):
        pipe = _resolve_plot_pipe(ds, pipe)
        data, loaded_meta = _load_frame_for_plot(ds, idx=idx, pipe=pipe)
        fig, _axs = _plot_loaded_frame(
            data,
            ds=ds,
            loaded_meta=loaded_meta,
            pipe=pipe,
            channel_labels=channel_labels,
            cmap=cmap,
            clim_percentiles=clim_percentiles,
            vlims=vlims,
            cbar_units=cbar_units,
            center_zero=center_zero,
            figsize=figsize,
        )

        isotime = loaded_meta.start_time_utc.strftime("%Y%m%d_%H%M%S")
        path = (out_dir / f"{idx:05d}_{isotime}.png").resolve()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def save_frames_by_meta(
    ds: SpecscoutDataset,
    metas: FrameMeta | Sequence[FrameMeta],
    *,
    out_dir: str | Path,
    pipe: PreprocessPipeline | None = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
    cbar_units: str | Sequence[str] | None = None,
    center_zero: bool | Sequence[bool] = False,
    dpi: int = 240,
    figsize: tuple[float, float] = (8.5, 5.5),
    name_template: str = "{i:05d}_{isotime}_idx{t_start_idx}.png",
) -> list[Path]:
    """
    Save one or many arbitrary frames identified by FrameMeta to PNG files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas_list = [metas] if isinstance(metas, FrameMeta) else list(metas)
    if len(metas_list) == 0:
        raise ValueError("metas must be non-empty.")

    written: list[Path] = []
    for i, meta in enumerate(metas_list):
        pipe = _resolve_plot_pipe(ds, pipe)
        data, loaded_meta = _load_frame_for_plot(ds, meta=meta, pipe=pipe)
        fig, _axs = _plot_loaded_frame(
            data,
            ds=ds,
            loaded_meta=loaded_meta,
            pipe=pipe,
            channel_labels=channel_labels,
            cmap=cmap,
            clim_percentiles=clim_percentiles,
            vlims=vlims,
            cbar_units=cbar_units,
            center_zero=center_zero,
            figsize=figsize,
        )

        isotime = loaded_meta.start_time_utc.strftime("%Y%m%d_%H%M%S")
        fname = name_template.format(
            i=i,
            isotime=isotime,
            t_start_idx=int(loaded_meta.t_start_idx),
        )
        path = (out_dir / fname).resolve()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written
