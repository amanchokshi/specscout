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
from IPython.display import clear_output, display

from ..dataset import FrameMeta, SpecscoutDataset
from ..preprocess import PreprocessPipeline
from .static import _load_frame_for_plot, _plot_loaded_frame


@dataclass
class ScrubberResult:
    """
    Container returned by interactive scrubber functions.

    Attributes
    ----------
    fig
        Most recently rendered matplotlib figure.
    slider
        Widget slider controlling the current frame selection.
    output
        Output widget containing the rendered figure.
    container
        VBox containing slider + output, convenient for notebook display.
    update
        Callback that updates the figure for a given slider index.
    n_frames
        Total number of selectable frames.
    """

    fig: plt.Figure | None
    slider: widgets.IntSlider
    output: widgets.Output
    container: widgets.VBox
    update: Callable[[int], None]
    n_frames: int


def _normalize_metas(metas: FrameMeta | Sequence[FrameMeta]) -> list[FrameMeta]:
    """
    Normalize one or many `FrameMeta` objects into a list.
    """
    if isinstance(metas, FrameMeta):
        return [metas]
    return list(metas)


def _resolve_plot_pipe(
    ds: SpecscoutDataset,
    pipe: PreprocessPipeline | None,
) -> PreprocessPipeline | None:
    """
    Resolve the plotting pipeline.

    Priority
    --------
    1. Explicit `pipe` argument
    2. Dataset pipeline `ds.pipe`
    """
    return pipe if pipe is not None else ds.pipe


def _render_sequence_frame(
    *,
    ds: SpecscoutDataset,
    frame_idx: int,
    n_total: int,
    plot_pipe: PreprocessPipeline | None,
    channel_labels: Sequence[str] | None,
    cmap,
    clim_percentiles: tuple[float, float],
    vlims,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, FrameMeta]:
    """
    Render one frame from a sequential dataset index.
    """
    data, meta = _load_frame_for_plot(ds, idx=frame_idx, pipe=plot_pipe)
    title = f"Frame {frame_idx}/{n_total - 1} — {meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    fig, _axs = _plot_loaded_frame(
        data,
        ds=ds,
        loaded_meta=meta,
        pipe=plot_pipe,
        channel_labels=channel_labels,
        cmap=cmap,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        figsize=figsize,
        title=title,
    )
    return fig, meta


def _render_meta_frame(
    *,
    ds: SpecscoutDataset,
    meta: FrameMeta,
    idx_in_list: int,
    n_total: int,
    plot_pipe: PreprocessPipeline | None,
    channel_labels: Sequence[str] | None,
    cmap,
    clim_percentiles: tuple[float, float],
    vlims,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, FrameMeta]:
    """
    Render one frame identified by explicit FrameMeta.
    """
    data, loaded_meta = _load_frame_for_plot(ds, meta=meta, pipe=plot_pipe)
    title = (
        f"Frame {idx_in_list}/{n_total - 1} — "
        f"{loaded_meta.start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"(t_start_idx={loaded_meta.t_start_idx})"
    )

    fig, _axs = _plot_loaded_frame(
        data,
        ds=ds,
        loaded_meta=loaded_meta,
        pipe=plot_pipe,
        channel_labels=channel_labels,
        cmap=cmap,
        clim_percentiles=clim_percentiles,
        vlims=vlims,
        figsize=figsize,
        title=title,
    )
    return fig, loaded_meta


def scrub_frames_sequence(
    ds: SpecscoutDataset,
    *,
    start_idx: int = 0,
    stop_idx: int | None = None,
    pipe: Optional[PreprocessPipeline] = None,
    channel_labels: Sequence[str] | None = None,
    cmap=cmr.pride,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    vlims=None,
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
        Optional plotting pipeline applied after reading each frame. If not
        provided, defaults to `ds.pipe`.
    channel_labels
        Optional labels for panel titles.
    cmap
        Colormap. Defaults to ``cmr.pride``.
    clim_percentiles
        Percentiles used for dynamic per-frame color scaling when `vlims` is
        None.
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
    plot_pipe = _resolve_plot_pipe(ds, pipe)

    output = widgets.Output()
    fig_holder: dict[str, plt.Figure | None] = {"fig": None}

    def update(slider_idx: int) -> None:
        real_idx = frame_indices[int(slider_idx)]

        with output:
            clear_output(wait=True)
            fig, _meta = _render_sequence_frame(
                ds=ds,
                frame_idx=real_idx,
                n_total=n_total,
                plot_pipe=plot_pipe,
                channel_labels=channel_labels,
                cmap=cmap,
                clim_percentiles=clim_percentiles,
                vlims=vlims,
                figsize=figsize,
            )
            fig_holder["fig"] = fig
            display(fig)
            plt.close(fig)

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

    # Initial render
    update(0)

    container = widgets.VBox([slider, output])

    return ScrubberResult(
        fig=fig_holder["fig"],
        slider=slider,
        output=output,
        container=container,
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
    vlims=None,
    figsize: tuple[float, float] = (8.5, 5.5),
) -> ScrubberResult:
    """
    Create an interactive scrubber over arbitrary frames identified by
    `FrameMeta`.

    Parameters
    ----------
    ds
        Dataset used to read frames.
    metas
        One or many metadata objects identifying the frames to inspect.
    pipe
        Optional plotting pipeline applied after reading each frame. If not
        provided, defaults to `ds.pipe`.
    channel_labels
        Optional labels for panel titles.
    cmap
        Colormap. Defaults to ``cmr.pride``.
    clim_percentiles
        Percentiles used for dynamic per-frame color scaling when `vlims` is
        None.
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

    plot_pipe = _resolve_plot_pipe(ds, pipe)

    output = widgets.Output()
    fig_holder: dict[str, plt.Figure | None] = {"fig": None}

    def update(slider_idx: int) -> None:
        i = int(slider_idx)

        with output:
            clear_output(wait=True)
            fig, _meta = _render_meta_frame(
                ds=ds,
                meta=metas_list[i],
                idx_in_list=i,
                n_total=len(metas_list),
                plot_pipe=plot_pipe,
                channel_labels=channel_labels,
                cmap=cmap,
                clim_percentiles=clim_percentiles,
                vlims=vlims,
                figsize=figsize,
            )
            fig_holder["fig"] = fig
            display(fig)
            plt.close(fig)

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

    # Initial render
    update(0)

    container = widgets.VBox([slider, output])

    return ScrubberResult(
        fig=fig_holder["fig"],
        slider=slider,
        output=output,
        container=container,
        update=update,
        n_frames=len(metas_list),
    )
