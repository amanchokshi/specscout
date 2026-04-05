"""
Visualization utilities for specscout.

This package provides:

- static Matplotlib plotting helpers in `specscout.viz.static`
- interactive notebook scrubbers in `specscout.viz.interactive`
"""

from .interactive import ScrubberResult, scrub_frames_by_meta, scrub_frames_sequence
from .static import (
    plot_frame,
    plot_roi_event,
    plot_scores_with_rois,
    plot_time_range,
    save_frame_sequence,
    save_frames_by_meta,
)

__all__ = [
    "ScrubberResult",
    "plot_frame",
    "plot_roi_event",
    "plot_scores_with_rois",
    "plot_time_range",
    "save_frame_sequence",
    "save_frames_by_meta",
    "scrub_frames_by_meta",
    "scrub_frames_sequence",
]
