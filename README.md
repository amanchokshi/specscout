# specscout

A pipeline for ingesting, preprocessing, and detecting transient events in ALBATROS low-frequency radio data

---

## Overview

`specscout` provides tools to:

- Ingest ALBATROS direct-spectra data into a regularized Zarr format
- Apply preprocessing pipelines (e.g. safe dB transforms, Stokes conversion)
- Model background structure using rolling PCA
- Detect transient events using robust outlier scoring
- Identify regions of interest (ROIs) in time–frequency data
- Visualize spectra, scores, and detections

The package is designed for large-scale, long-duration datasets with gaps, systematics, and non-stationary backgrounds.

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management.

Clone the repository and install in editable mode:

```bash
git clone https://github.com/amanchokshi/specscout.git
cd specscout

uv sync
```

This installs both the package and development dependencies.
