"""
Interactive notebook-first visualization tools for specscout.

This module provides widget-based scrubbing helpers for inspecting:

- sequential dataset frames
- arbitrary non-sequential frames identified by `FrameMeta`

These tools are intended primarily for Jupyter workflows and are compatible
with preprocessing pipelines, including raw channels, Stokes I, or multi-panel
products such as full Stokes.

Design notes
------------
- Data loading is delegated to `SpecscoutDataset.load_by_t_start_idx(...)`.
- Rendering is delegated to the same conventions used by `viz.static`.
- Interactive functions return a `ScrubberResult`; keep a reference to this
  object alive in notebooks to avoid garbage collection issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cmasher as cmr
import ipywidgets as widgets
import matplotlib.pyplot as plt

from ..dataset import FrameMeta, SpecscoutDataset
from ..preprocess import PreprocessPipeline
from .static import (
    _infer_panel_labels,
    _infer_units,
    _load_frame_for_plot,
    _plot_waterfall_grid,
)


@dataclass(frozen=True)
class ScrubberResult:
    """
    Container returned by interactive scrubber functions.

    Attributes
    ----------
    fig
        Matplotlib figure holding the scrubbed image(s).
    slider
        Widget slider controlling the current frame selection.
    update
        Callback that updates the figure for a given slider index.
    n_frames
        Total number of selectable frames.
    """

    fig: plt.Figure
    slider: widgets.IntSlider
    update: Callable[[int], None]
    n_frames: int


def _normalize_metas(metas: FrameMeta | Sequence[FrameMeta]) -> list[FrameMeta]:
    """
    Normalize one or many `FrameMeta` objects into a list.
    """
    if isinstance(metas, FrameMeta):
        return [metas]
    return list(metas)


def scrub_frames_sequence(
    ds: SpecscoutDataset,
    *,
    start_idx: int = 0,
    stop_idx: int | None = None,
    pipe: Optional[PreprocessPipeline] = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8.5, 5.5),
) -> ScrubberResult:
    """
    Create an interactive scrubber over a sequential range of dataset frames.

    Parameters
    ----------
    ds
        Dataset used to read frames.
    start_idx
        First dataset frame index included in the scrubber.
    stop_idx
        Exclusive stop dataset index. Defaults to `len(ds)`.
    pipe
        Optional plotting pipeline applied after reading each frame.
    channel_labels
        Optional labels for multi-panel data.
    cmap
        Colormap. Defaults to ``cmr.pride``.
    clim_percentiles
        Percentiles used for dynamic per-frame color scaling when `vlims` is None.
    vlims
        Optional fixed color limits.
    figsize
        Figure size in inches.

    Returns
    -------
    ScrubberResult
        Interactive scrubber container.
    """
    n_total = len(ds)
    if stop_idx is None:
        stop_idx = n_total

    start_i = int(start_idx)
    stop_i = int(stop_idx)

    if not (0 <= start_i < stop_i <= n_total):
        raise ValueError(f"Invalid frame range [{start_i}, {stop_i}) for dataset of length {n_total}.")

    frame_indices = list(range(start_i, stop_i))

    data0, meta0 = _load_frame_for_plot(ds, idx=frame_indices[0], pipe=pipe)
    freqs, _x_label = ds.freq_axis()
    units = _infer_units(pipe)

    if data0.ndim == 3 and channel_labels is None:
        channel_labels = _infer_panel_labels(
            pipe=pipe,
            chans=ds.plan.chans,
            n_panels=data0.shape[2],
        )

    title0 = f"Frame {frame_indices[0]}/{n_total - 1} — {meta0.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    fig, _axs = _plot_waterfall_grid(
        data0,
        x_mode="frame",
        freqs=freqs,
        frame_meta=meta0,
        channel_labels=channel_labels,
        cmap=cmap,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        figsize=figsize,
        units=units,
        title=title0,
    )

    def update(slider_idx: int) -> None:
        real_idx = frame_indices[int(slider_idx)]
        data, meta = _load_frame_for_plot(ds, idx=real_idx, pipe=pipe)

        fig.clf()
        title = f"Frame {real_idx}/{n_total - 1} — {meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        _plot_waterfall_grid(
            data,
            x_mode="frame",
            freqs=freqs,
            frame_meta=meta,
            channel_labels=channel_labels,
            cmap=cmap,
            clim_percentiles=clim_percentiles,
            vlims=vlims,
            figsize=figsize,
            units=units,
            title=title,
        )
        fig.canvas.draw_idle()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(frame_indices) - 1,
        step=1,
        description="Frame",
        continuous_update=True,
        readout=True,
        layout=widgets.Layout(width="80%"),
    )
    slider.observe(lambda change: update(int(change["new"])), names="value")

    return ScrubberResult(
        fig=fig,
        slider=slider,
        update=update,
        n_frames=len(frame_indices),
    )


def scrub_frames_by_meta(
    ds: SpecscoutDataset,
    metas: FrameMeta | Sequence[FrameMeta],
    *,
    pipe: Optional[PreprocessPipeline] = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8.5, 5.5),
) -> ScrubberResult:
    """
    Create an interactive scrubber over arbitrary frames identified by FrameMeta.

    Parameters
    ----------
    ds
        Dataset used to read frames.
    metas
        One or many metadata objects identifying the frames to inspect.
    pipe
        Optional plotting pipeline applied after reading each frame.
    channel_labels
        Optional labels for multi-panel data.
    cmap
        Colormap. Defaults to ``cmr.pride``.
    clim_percentiles
        Percentiles used for dynamic per-frame color scaling when `vlims` is None.
    vlims
        Optional fixed color limits.
    figsize
        Figure size in inches.

    Returns
    -------
    ScrubberResult
        Interactive scrubber container.
    """
    metas_list = _normalize_metas(metas)
    if len(metas_list) == 0:
        raise ValueError("metas must be non-empty.")

    data0, meta0 = _load_frame_for_plot(ds, meta=metas_list[0], pipe=pipe)
    freqs, _x_label = ds.freq_axis()
    units = _infer_units(pipe)

    if data0.ndim == 3 and channel_labels is None:
        channel_labels = _infer_panel_labels(
            pipe=pipe,
            chans=ds.plan.chans,
            n_panels=data0.shape[2],
        )

    title0 = (
        f"Frame 0/{len(metas_list) - 1} — "
        f"{meta0.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"(t_start_idx={meta0.t_start_idx})"
    )

    fig, _axs = _plot_waterfall_grid(
        data0,
        x_mode="frame",
        freqs=freqs,
        frame_meta=meta0,
        channel_labels=channel_labels,
        cmap=cmap,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        figsize=figsize,
        units=units,
        title=title0,
    )

    def update(slider_idx: int) -> None:
        i = int(slider_idx)
        data, meta = _load_frame_for_plot(ds, meta=metas_list[i], pipe=pipe)

        fig.clf()
        title = (
            f"Frame {i}/{len(metas_list) - 1} — "
            f"{meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"(t_start_idx={meta.t_start_idx})"
        )
        _plot_waterfall_grid(
            data,
            x_mode="frame",
            freqs=freqs,
            frame_meta=meta,
            channel_labels=channel_labels,
            cmap=cmap,
            clim_percentiles=clim_percentiles,
            vlims=vlims,
            figsize=figsize,
            units=units,
            title=title,
        )
        fig.canvas.draw_idle()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(metas_list) - 1,
        step=1,
        description="Frame",
        continuous_update=True,
        readout=True,
        layout=widgets.Layout(width="80%"),
    )
    slider.observe(lambda change: update(int(change["new"])), names="value")

    return ScrubberResult(
        fig=fig,
        slider=slider,
        update=update,
        n_frames=len(metas_list),
    )
