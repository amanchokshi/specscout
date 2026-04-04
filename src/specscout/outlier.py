"""
Outlier detection utilities for specscout time-frequency frames.

This module provides two closely related building blocks for large-scale frame
analysis:

1. `QuietSelector`
   Computes a per-frame quietness score and selects a quiet subset of frames for
   background modeling.

2. `RollingPCABackground`
   Fits a PCA background model on quiet frames and scores candidate outliers
   using residuals in the same feature space.

Design principles
-----------------
- Work in a stable upstream transform space (typically `safe_db`).
- Use a consistent feature representation: masked frames flattened to `(N, D)`.
- Use the same metric vocabulary for both quiet-frame ranking and outlier
  scoring.
- Allow missing values to be handled explicitly and intentionally.

Masking convention
------------------
Most entry points accept an optional boolean frequency mask `freq_mask` of
shape `(F,)`:

- `True`  => keep the channel
- `False` => mask / ignore the channel

The mask is applied along the frequency axis before flattening.

Metric vocabulary
-----------------
Both `QuietSelector` and `RollingPCABackground` use the same metric core via
`compute_metric()`.

Supported methods:
- percentile aliases: `"p99"`, `"p995"`, `"p999"`
- generic percentile: `"percentile"` (uses `q`)
- tail / threshold metrics: `"topk_sum"`, `"excess_mass"`
- norm metrics: `"l1"`, `"l2"`, `"lp"`

Typical usage
-------------
Build a dataset that returns frames already transformed into `safe_db`:

    pipe_db = PreprocessPipeline(input_space="linear").add(step_safe_db(...))
    ds = SpecscoutDataset(..., pipe=pipe_db, return_meta=True)

Select quiet frames and fit a background PCA:

    qs = QuietSelector(method="p99", quiet_fraction=0.3, freq_mask=rfi_mask)
    bg = RollingPCABackground(k=128, center=True, freq_mask=rfi_mask)

    frames, metas = ds.to_numpy_batch(...)
    quiet_idx = qs.select_quiet(frames)
    bg.fit(frames[quiet_idx])

Score outliers using residuals:

    scores = bg.score(
        frames,
        k_pca=16,
        metric_kwargs={
            "method": "topk_sum",
            "topk": 2048,
            "positive_only": True,
            "min_finite_frac": 0.7,
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from sklearn.utils.extmath import randomized_svd

Array = np.ndarray

MetricMethod = Literal[
    "p99",
    "p995",
    "p999",
    "percentile",
    "topk_sum",
    "excess_mass",
    "lp",
    "l1",
    "l2",
]


# -----------------------------------------------------------------------------
# Shape / masking helpers
# -----------------------------------------------------------------------------


def _validate_frame_batch(frames: Array) -> tuple[int, int, int]:
    """
    Validate that `frames` has shape `(N, T, F)` or `(N, T, F, C)`.

    Parameters
    ----------
    frames
        Input frame batch.

    Returns
    -------
    N, T, F
        Batch size and time/frequency dimensions.

    Notes
    -----
    If a channel dimension is present, it is treated as extra features during
    flattening.
    """
    if frames.ndim == 3:
        n, t, f = frames.shape
        return int(n), int(t), int(f)
    if frames.ndim == 4:
        n, t, f, _c = frames.shape
        return int(n), int(t), int(f)
    raise ValueError("Expected frames with shape (N, T, F) or (N, T, F, C).")


def _as_freq_mask(freq_mask: Optional[Array], f: int) -> Array:
    """
    Normalize `freq_mask` to a boolean array of shape `(F,)`.

    Parameters
    ----------
    freq_mask
        Optional frequency mask.
    f
        Number of frequency channels.

    Returns
    -------
    mask
        Boolean mask of shape `(F,)` where `True` means keep.

    Notes
    -----
    Convention:
    - `True`  => channel is kept
    - `False` => channel is masked / ignored
    """
    if freq_mask is None:
        return np.ones((f,), dtype=bool)

    m = np.asarray(freq_mask, dtype=bool)
    if m.ndim != 1 or m.shape[0] != f:
        raise ValueError(f"freq_mask must have shape (F,), got {m.shape}, expected ({f},).")
    return m


def _frames_to_X(frames: Array, *, freq_mask: Optional[Array]) -> Array:
    """
    Convert raw frames to a canonical feature matrix `X` of shape `(N, D)`.

    Parameters
    ----------
    frames
        Frame batch of shape `(N, T, F)` or `(N, T, F, C)`.
    freq_mask
        Optional boolean mask of shape `(F,)`.

    Returns
    -------
    X
        Flattened feature matrix of shape `(N, D)`.

    Notes
    -----
    Steps:
    - validate the frame batch shape
    - apply `freq_mask` along the frequency axis
    - flatten `(T, F[, C])` into one feature dimension
    """
    n, _t, f = _validate_frame_batch(frames)
    m = _as_freq_mask(freq_mask, f)

    if frames.ndim == 3:
        x = frames[:, :, m]
    else:
        x = frames[:, :, m, :]

    return np.reshape(x, (n, -1))


# -----------------------------------------------------------------------------
# Unified metric core
# -----------------------------------------------------------------------------


def _compute_metric_on_valid_rows(
    Xv: Array,
    *,
    method: MetricMethod,
    q: float,
    topk: int,
    thr: float,
    p: float,
) -> Array:
    """
    Compute a per-row metric on a matrix whose rows are already known to be valid.

    Parameters
    ----------
    Xv
        2D feature matrix `(Nv, D)` containing only rows that passed the
        `min_finite_frac` validity check.
    method
        Metric method.
    q, topk, thr, p
        Method-specific parameters.

    Returns
    -------
    scores
        1D array of length `Nv`.
    """
    m = str(method)

    if m == "p99":
        return np.nanpercentile(Xv, 99.0, axis=1)

    if m == "p995":
        return np.nanpercentile(Xv, 99.5, axis=1)

    if m == "p999":
        return np.nanpercentile(Xv, 99.9, axis=1)

    if m == "percentile":
        return np.nanpercentile(Xv, float(q), axis=1)

    if m == "topk_sum":
        X2 = np.where(np.isfinite(Xv), Xv, -np.inf)
        kk = int(max(1, min(int(topk), X2.shape[1])))
        part = np.partition(X2, X2.shape[1] - kk, axis=1)[:, -kk:]
        sv = np.sum(part, axis=1)
        sv[~np.isfinite(sv)] = np.nan
        return sv

    if m == "excess_mass":
        return np.nansum(np.maximum(Xv - float(thr), 0.0), axis=1)

    if m == "l1":
        return np.nansum(np.abs(Xv), axis=1)

    if m == "l2":
        X2 = np.where(np.isfinite(Xv), Xv, 0.0)
        return np.sqrt(np.sum(X2 * X2, axis=1))

    if m == "lp":
        pp = float(p)
        if pp < 1.0:
            raise ValueError("p must be >= 1.")
        X2 = np.where(np.isfinite(Xv), Xv, 0.0)
        return np.sum(np.abs(X2) ** pp, axis=1) ** (1.0 / pp)

    raise ValueError(f"Unknown method: {method!r}")


def compute_metric(
    X: Array,
    *,
    method: MetricMethod,
    q: float = 99.9,
    topk: int = 2048,
    thr: float = 3.0,
    p: float = 4.0,
    min_finite_frac: float = 1.0,
    normalize_missing: bool = False,
) -> Array:
    """
    Compute a per-row metric on a feature matrix with NaN-aware handling.

    Parameters
    ----------
    X
        Feature matrix of shape `(N, D)`.
    method
        Metric method.
    q
        Percentile used when `method="percentile"`.
    topk
        Number of largest elements summed when `method="topk_sum"`.
    thr
        Threshold used when `method="excess_mass"`.
    p
        Exponent used when `method="lp"`.
    min_finite_frac
        Minimum fraction of finite values required for a row to receive a
        score. Rows failing this requirement receive NaN.
    normalize_missing
        If True, metrics based on summation / norms are rescaled to compensate
        for missing values so scores remain more comparable across rows with
        different finite fractions.

    Returns
    -------
    scores
        Array of shape `(N,)`. Invalid rows receive NaN.

    Notes
    -----
    Percentile-based metrics are never missingness-normalized.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be (N, D).")

    if not (0.0 < float(min_finite_frac) <= 1.0):
        raise ValueError("min_finite_frac must be in (0, 1].")

    n, d = X.shape
    finite = np.isfinite(X)
    n_finite = finite.sum(axis=1).astype(np.float64)
    frac = n_finite / float(d)

    valid = (n_finite > 0) & (frac >= float(min_finite_frac))

    scores = np.full((n,), np.nan, dtype=np.float64)
    if not np.any(valid):
        return scores

    Xv = X[valid]
    sv = _compute_metric_on_valid_rows(
        Xv,
        method=method,
        q=float(q),
        topk=int(topk),
        thr=float(thr),
        p=float(p),
    )

    if normalize_missing and str(method) not in {"p99", "p995", "p999", "percentile"}:
        nv = np.maximum(n_finite[valid], 1.0)
        if method in {"topk_sum", "excess_mass", "l1"}:
            sv = sv * (float(d) / nv)
        elif method == "l2":
            sv = sv * np.sqrt(float(d) / nv)
        elif method == "lp":
            sv = sv * (float(d) / nv) ** (1.0 / float(p))

    scores[valid] = sv
    return scores


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QuietSelector:
    """
    Select a quiet subset of frames for background modeling.

    Quietness is computed from the same masked feature representation used
    elsewhere in this module. Lower metric values are interpreted as quieter.

    Parameters
    ----------
    method
        Quietness metric. Lower is quieter.
    quiet_fraction
        Fraction of finite-scored frames to keep as quiet. Must lie in `(0, 1]`.
    freq_mask
        Optional boolean mask of shape `(F,)`. `True` means keep the channel.
    q, topk, thr, p
        Metric-specific parameters passed to `compute_metric()`.
    min_finite_frac
        Minimum finite fraction required for a frame to receive a quietness
        score. Frames failing this are excluded from quiet selection.
    normalize_missing
        If True, apply missingness normalization for norm/sum-based metrics.
    """

    method: MetricMethod = "p99"
    quiet_fraction: float = 0.7
    freq_mask: Optional[Array] = None

    q: float = 99.0
    topk: int = 2048
    thr: float = 3.0
    p: float = 4.0
    min_finite_frac: float = 1.0
    normalize_missing: bool = False

    def scores(self, frames: Array) -> Array:
        """
        Compute per-frame quietness scores.

        Parameters
        ----------
        frames
            Frame batch of shape `(N, T, F)` or `(N, T, F, C)`.

        Returns
        -------
        scores
            1D array of shape `(N,)`. Lower is quieter.
        """
        X = _frames_to_X(frames, freq_mask=self.freq_mask)
        return compute_metric(
            X,
            method=self.method,
            q=float(self.q),
            topk=int(self.topk),
            thr=float(self.thr),
            p=float(self.p),
            min_finite_frac=float(self.min_finite_frac),
            normalize_missing=bool(self.normalize_missing),
        )

    def select_quiet(self, frames: Array) -> Array:
        """
        Return indices of quiet frames, sorted by ascending quietness.

        Parameters
        ----------
        frames
            Frame batch of shape `(N, T, F)` or `(N, T, F, C)`.

        Returns
        -------
        idx
            1D integer array of selected frame indices.

        Notes
        -----
        Only frames with finite quietness scores are eligible for selection.
        Frames whose scores are NaN are treated as unusable rather than merely
        "non-quiet".
        """
        qf = float(self.quiet_fraction)
        if not (0.0 < qf <= 1.0):
            raise ValueError("quiet_fraction must be in (0, 1].")

        s = self.scores(frames)
        finite = np.isfinite(s)
        if not np.any(finite):
            raise ValueError("No finite quietness scores available.")

        idx_valid = np.where(finite)[0]
        s_valid = s[finite]

        k = int(max(1, np.floor(qf * s_valid.size)))
        order = np.argsort(s_valid)
        return idx_valid[order[:k]]


@dataclass
class RollingPCABackground:
    """
    PCA background model fit on quiet frames.

    This object is intentionally stateless with respect to time: it does not
    manage dataset iteration itself. Instead, callers are expected to:

    - assemble batches or rolling contexts of frames externally
    - select quiet frames with `QuietSelector`
    - call `fit()` on the quiet subset
    - call `reconstruct()`, `residuals()`, or `score()` on frames of interest

    Parameters
    ----------
    k
        Number of PCA components to fit.
    center
        If True, mean-center features before SVD.
    freq_mask
        Optional boolean mask of shape `(F,)` applied before flattening.
    use_randomized
        If True, use scikit-learn's randomized SVD.
    n_iter
        Power iterations for randomized SVD.
    random_state
        Seed for randomized SVD.

    Learned attributes
    ------------------
    mu_
        Mean vector of shape `(D,)` if `center=True`, else None.
    Vt_
        PCA basis vectors of shape `(k, D)`.
    S_
        Singular values of shape `(k,)`.

    Notes
    -----
    - The PCA model is trained in the same masked feature space used for
      scoring.
    - During `fit()`, rows containing any non-finite feature values are
      dropped conservatively.
    """

    k: int = 256
    center: bool = True
    freq_mask: Optional[Array] = None
    use_randomized: bool = True
    n_iter: int = 2
    random_state: int = 0

    mu_: Optional[Array] = None
    Vt_: Optional[Array] = None
    S_: Optional[Array] = None

    def _prep(self, frames: Array) -> Array:
        """
        Apply frequency masking (if any) and flatten to feature matrix `(N, D)`.
        """
        return _frames_to_X(frames, freq_mask=self.freq_mask)

    def fit(self, quiet_frames: Array) -> RollingPCABackground:
        """
        Fit the PCA model on quiet frames.

        Parameters
        ----------
        quiet_frames
            Array of shape `(Nq, T, F)` or `(Nq, T, F, C)`.

        Returns
        -------
        self
        """
        X = self._prep(quiet_frames)

        ok = np.isfinite(X).all(axis=1)
        X = X[ok]
        if X.shape[0] == 0:
            raise ValueError("No finite quiet frames to fit PCA.")

        if self.center:
            mu = np.mean(X, axis=0)
            Xc = X - mu
            self.mu_ = mu
        else:
            Xc = X
            self.mu_ = None

        k = int(self.k)
        if k <= 0:
            raise ValueError("k must be positive.")
        k = min(k, min(int(Xc.shape[0]), int(Xc.shape[1])))

        if self.use_randomized:
            _U, S, Vt = randomized_svd(
                Xc,
                n_components=k,
                n_iter=int(self.n_iter),
                random_state=int(self.random_state),
            )
        else:
            _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            S = S[:k]
            Vt = Vt[:k, :]

        self.S_ = np.asarray(S)
        self.Vt_ = np.asarray(Vt)
        return self

    def reconstruct(self, frames: Array, *, k: Optional[int] = None) -> Array:
        """
        Compute a low-rank reconstruction of frames in feature space.

        Parameters
        ----------
        frames
            Array of shape `(N, T, F)` or `(N, T, F, C)`.
        k
            Number of PCA components to use. Defaults to `self.k`.

        Returns
        -------
        X_hat
            Reconstructed feature matrix of shape `(N, D)`.

        Notes
        -----
        Missing values are treated as "no deviation" only for the projection
        step. That is, NaNs are zero-filled when projecting into the PCA
        subspace, but residuals are still formed against the original feature
        matrix so missing entries propagate naturally back into the residual.
        """
        if self.Vt_ is None:
            raise RuntimeError("Model is not fit yet. Call fit() first.")

        X = self._prep(frames)

        if self.center and self.mu_ is not None:
            Xc = X - self.mu_
        else:
            Xc = X

        finite = np.isfinite(Xc)
        Xc_proj = np.where(finite, Xc, 0.0)

        kk = int(self.k if k is None else k)
        kk = min(kk, int(self.Vt_.shape[0]))

        V = self.Vt_[:kk, :]
        C = Xc_proj @ V.T
        X_hat = C @ V

        if self.center and self.mu_ is not None:
            X_hat = X_hat + self.mu_

        return X_hat

    def residuals(self, frames: Array, *, k: Optional[int] = None) -> Array:
        """
        Compute residuals in feature space: `R = X - X_hat`.

        Parameters
        ----------
        frames
            Array of shape `(N, T, F)` or `(N, T, F, C)`.
        k
            Number of PCA components used in reconstruction.

        Returns
        -------
        R
            Residual feature matrix of shape `(N, D)`.
        """
        X = self._prep(frames)
        X_hat = self.reconstruct(frames, k=k)
        return X - X_hat

    def score(
        self,
        frames: Array,
        *,
        k_pca: Optional[int] = None,
        metric_kwargs: Optional[dict[str, Any]] = None,
        **metric_overrides: Any,
    ) -> Array:
        """
        Score outliers using PCA residuals.

        Parameters
        ----------
        frames
            Array of shape `(N, T, F)` or `(N, T, F, C)`.
        k_pca
            Number of PCA modes to use for reconstruction. If None, uses `self.k`.
        metric_kwargs
            Optional metric configuration dictionary. Supported keys include:
            - `method`
            - `q`
            - `topk`
            - `thr`
            - `p`
            - `positive_only`
            - `min_finite_frac`
            - `normalize_missing`
        **metric_overrides
            Additional keyword overrides merged on top of `metric_kwargs`.

        Returns
        -------
        scores
            1D array of shape `(N,)`. Higher means more outlier-like.

        Notes
        -----
        This method accepts a config dict so it can integrate cleanly with
        higher-level workflow code that already stores metric parameters as
        dictionaries.
        """
        cfg: dict[str, Any] = {
            "method": "topk_sum",
            "q": 99.9,
            "topk": 2048,
            "thr": 3.0,
            "p": 4.0,
            "positive_only": True,
            "min_finite_frac": 0.7,
            "normalize_missing": True,
        }
        if metric_kwargs is not None:
            cfg.update(metric_kwargs)
        if metric_overrides:
            cfg.update(metric_overrides)

        positive_only = bool(cfg.pop("positive_only", True))

        R = self.residuals(frames, k=k_pca)
        if positive_only:
            R = np.clip(R, 0.0, None)

        return compute_metric(R, **cfg)
