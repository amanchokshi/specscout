"""
Outlier detection utilities for specscout time–frequency frames.

This module provides two building blocks for large-scale, streaming analysis:

1) QuietSelector
   Computes a per-frame "quietness" score (using only non-flagged channels)
   and selects a quiet subset of frames for background modeling.

2) RollingPCABackground
   Fits a PCA background model on quiet frames (optionally within a rolling
   context managed outside this module) and scores outliers using residuals.
   Outlier scoring can be restricted to positive residuals (brighter-than-model).

Key design principles
---------------------
- Work in a stable space (typically `safe_db`) via a PreprocessPipeline upstream.
- Avoid bias from RFI-contaminated channels during quiet selection by masking.
- Fit the background model on "quiet" frames only.
- Outlier scoring can be based on positive residuals (bright events).

Masking / RFI handling
----------------------
Most entry points accept an optional boolean frequency mask `freq_mask` of shape (F,).
Convention:
- True  => channel is "good" (kept)
- False => channel is masked/ignored

- Quiet selection ALWAYS uses only unmasked channels.
- PCA fitting and residual scoring can also ignore masked channels (recommended for
  robust detection).

Metrics: unified MetricMethod
-----------------------------
Both QuietSelector and RollingPCABackground use the same metric core.

The core operates on a canonical representation:
- a feature matrix X of shape (N, D), where each row is a flattened frame (optionally
  masked in frequency beforehand), OR a residual matrix R of the same shape.

QuietSelector:
- computes X from raw frames, then scores using a chosen MetricMethod.
- "quiet" = smaller scores (ascending).

RollingPCABackground:
- computes residuals R in feature space (N, D),
- optionally clips to positive residuals,
- scores using the same MetricMethod.
- "outlier" = larger scores (descending).

Dependencies
------------
- NumPy is required.
- If scikit-learn is available, randomized SVD is used by default for efficiency.
  Otherwise falls back to np.linalg.svd.

Typical usage
-------------
Build a dataset that returns frames already transformed into `safe_db`:

    pipe_db = PreprocessPipeline(input_space="linear").add(step_safe_db(...))
    ds = SpecscoutDataset(..., pipe=pipe_db, return_meta=True)

Select quiet frames and fit a background PCA:

    qs = QuietSelector(method="p99", quiet_fraction=0.7, freq_mask=rfi_mask)
    bg = RollingPCABackground(k=256, center=True, freq_mask=rfi_mask)

    frames, metas = ds.to_numpy_batch(...)
    quiet_idx = qs.select_quiet(frames)
    bg.fit(frames[quiet_idx])

Score outliers on residuals (bright-only):

    scores = bg.score(frames, method="topk_sum", topk=2048, positive_only=True)
    outlier_idx = np.argsort(scores)[::-1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
from sklearn.utils.extmath import randomized_svd

Array = np.ndarray

# A single unified metric vocabulary for both "quietness" and "outlier score".
MetricMethod = Literal[
    # percentile aliases (no params required)
    "p99",
    "p995",
    "p999",
    # generic percentile (uses q)
    "percentile",
    # strong-tail mass
    "topk_sum",
    "excess_mass",
    # norms
    "lp",
    "l1",
    "l2",
]


# -----------------------------------------------------------------------------
# Shape/masking helpers (frames -> canonical feature matrix X with shape (N, D))
# -----------------------------------------------------------------------------


def _require_2d_frames(frames: Array) -> tuple[int, int, int]:
    """
    Validate frame batch has shape (N, T, F) or (N, T, F, C).

    Returns
    -------
    N, T, F
        Batch size and time/frequency dimensions. If frames include a channel
        dimension, it is treated as additional features during flattening.
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
    Normalize `freq_mask` to a boolean array of shape (F,).

    Convention
    ----------
    True means "keep", False means "masked".
    """
    if freq_mask is None:
        return np.ones((f,), dtype=bool)
    m = np.asarray(freq_mask, dtype=bool)
    if m.ndim != 1 or m.shape[0] != f:
        raise ValueError(f"freq_mask must have shape (F,), got {m.shape}, expected ({f},).")
    return m


def _masked_view(frames: Array, freq_mask: Array) -> Array:
    """
    Apply `freq_mask` to frames along frequency axis.

    Parameters
    ----------
    frames
        (N, T, F) or (N, T, F, C)
    freq_mask
        (F,) boolean. True kept.

    Returns
    -------
    masked
        (N, T, F_kept) or (N, T, F_kept, C)
    """
    if frames.ndim == 3:
        return frames[:, :, freq_mask]
    return frames[:, :, freq_mask, :]


def _flatten_frames(frames: Array) -> Array:
    """
    Flatten frames to feature rows.

    (N, T, F)    -> (N, T*F)
    (N, T, F, C) -> (N, T*F*C)

    Notes
    -----
    Channel dimension (if present) is treated as additional features.
    """
    n = int(frames.shape[0])
    return np.reshape(frames, (n, -1))


def _frames_to_X(frames: Array, *, freq_mask: Optional[Array]) -> Array:
    """
    Convert raw frames to canonical feature matrix X of shape (N, D).

    Steps
    -----
    - Validate shape is (N, T, F) or (N, T, F, C)
    - Apply `freq_mask` along frequency axis if provided
    - Flatten (T, F[, C]) into feature dimension D
    """
    _n, _t, f = _require_2d_frames(frames)
    m = _as_freq_mask(freq_mask, f)
    x = _masked_view(frames, m)
    return _flatten_frames(x)


# -----------------------------------------------------------------------------
# Unified metric core (operates on canonical X with shape (N, D))
# -----------------------------------------------------------------------------


def _row_percentile(X: Array, q: float) -> Array:
    """Per-row percentile over features, ignoring NaNs."""
    return np.nanpercentile(X, float(q), axis=1)


def _row_topk_sum(X: Array, k: int) -> Array:
    """
    Per-row sum of the top-k largest values.

    Notes
    -----
    - Uses np.partition for efficiency.
    - NaNs are treated as -inf (ignored).
    """
    X2 = np.where(np.isfinite(X), X, -np.inf)
    k = int(max(1, min(int(k), X2.shape[1])))
    part = np.partition(X2, X2.shape[1] - k, axis=1)[:, -k:]
    s = np.sum(part, axis=1)
    s[~np.isfinite(s)] = np.nan
    return s


def _row_excess_mass(X: Array, thr: float) -> Array:
    """Per-row sum(max(x - thr, 0)), ignoring NaNs."""
    return np.nansum(np.maximum(X - float(thr), 0.0), axis=1)


def _row_lp(X: Array, p: float) -> Array:
    """
    Per-row Lp norm (p>=1), treating NaNs as 0.

    This is: (sum |x|^p)^(1/p)
    """
    pp = float(p)
    if pp < 1.0:
        raise ValueError("p must be >= 1.")
    X2 = np.where(np.isfinite(X), X, 0.0)
    return np.sum(np.abs(X2) ** pp, axis=1) ** (1.0 / pp)


def _row_l1(X: Array) -> Array:
    """Per-row L1 sum of absolute values, ignoring NaNs."""
    return np.nansum(np.abs(X), axis=1)


def _row_l2(X: Array) -> Array:
    """Per-row L2 norm, treating NaNs as 0."""
    X2 = np.where(np.isfinite(X), X, 0.0)
    return np.sqrt(np.sum(X2 * X2, axis=1))


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

    This function evaluates an outlier or quietness metric independently
    for each row of a 2D feature matrix. NaN values are handled explicitly
    so that frames with partial missing data can still be scored.

    Parameters
    ----------
    X : Array
        Feature matrix of shape (N, D) where each row corresponds to one
        frame and columns correspond to flattened features (e.g., time–
        frequency pixels after masking and preprocessing).

    method : MetricMethod
        Name of the metric to compute. Supported methods are:

        Percentile-based
            "p99", "p995", "p999"
                Fixed percentile aliases.
            "percentile"
                Percentile specified by ``q``.

        Tail / threshold metrics
            "topk_sum"
                Sum of the largest ``topk`` values in each row.
            "excess_mass"
                Sum of ``max(x - thr, 0)`` across the row.

        Norm-based metrics
            "l1"
                L1 norm (sum of absolute values).
            "l2"
                L2 norm.
            "lp"
                General Lp norm using exponent ``p``.

    q : float, optional
        Percentile used when ``method="percentile"`` (default 99.9).

    topk : int, optional
        Number of largest elements summed when ``method="topk_sum"``.

    thr : float, optional
        Threshold used for ``method="excess_mass"``.

    p : float, optional
        Exponent used when ``method="lp"`` (must be >= 1).

    min_finite_frac : float, optional
        Minimum fraction of finite values required in a row for it to be
        considered valid. Rows with fewer finite values than this threshold
        receive a score of NaN.

        For example:
        - ``1.0`` requires all elements to be finite.
        - ``0.7`` allows up to 30% missing values.

    normalize_missing : bool, optional
        If True, metrics that depend on summation or norms are rescaled to
        compensate for missing features so that scores remain comparable
        across rows with different numbers of finite values.

        Normalization rules:

        - L1-like metrics ("topk_sum", "excess_mass", "l1")
              scale by ``D / n_finite``
        - L2 norm
              scale by ``sqrt(D / n_finite)``
        - Lp norm
              scale by ``(D / n_finite) ** (1/p)``

        Percentile-based metrics are not normalized.

    Returns
    -------
    scores : Array
        Array of shape (N,) containing the computed metric for each row.
        Rows failing the ``min_finite_frac`` requirement receive NaN.

    Notes
    -----
    - NaN values are ignored during metric evaluation using NumPy
      ``nan*`` functions where appropriate.
    - Rows with no finite values are always assigned NaN.
    - The validity check (``min_finite_frac``) is applied before computing
      metrics to avoid warnings from functions such as
      ``np.nanpercentile`` on all-NaN slices.
    - Missingness normalization can help prevent partially-missing rows
      from systematically producing smaller scores simply due to having
      fewer contributing features.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be (N, D).")

    N, D = X.shape
    finite = np.isfinite(X)
    n_finite = finite.sum(axis=1).astype(np.float64)
    frac = n_finite / float(D)

    valid = (n_finite > 0) & (frac >= float(min_finite_frac))

    scores = np.full((N,), np.nan, dtype=np.float64)
    if not np.any(valid):
        return scores

    Xv = X[valid]

    # --- compute metric on valid rows (NaN-aware inside each) ---
    m = str(method)
    if m == "p99":
        sv = np.nanpercentile(Xv, 99.0, axis=1)
    elif m == "p995":
        sv = np.nanpercentile(Xv, 99.5, axis=1)
    elif m == "p999":
        sv = np.nanpercentile(Xv, 99.9, axis=1)
    elif m == "percentile":
        sv = np.nanpercentile(Xv, float(q), axis=1)
    elif m == "topk_sum":
        X2 = np.where(np.isfinite(Xv), Xv, -np.inf)
        kk = int(max(1, min(int(topk), X2.shape[1])))
        part = np.partition(X2, X2.shape[1] - kk, axis=1)[:, -kk:]
        sv = np.sum(part, axis=1)
        sv[~np.isfinite(sv)] = np.nan
    elif m == "excess_mass":
        sv = np.nansum(np.maximum(Xv - float(thr), 0.0), axis=1)
    elif m == "l1":
        sv = np.nansum(np.abs(Xv), axis=1)
    elif m == "l2":
        X2 = np.where(np.isfinite(Xv), Xv, 0.0)
        sv = np.sqrt(np.sum(X2 * X2, axis=1))
    elif m == "lp":
        pp = float(p)
        if pp < 1.0:
            raise ValueError("p must be >= 1.")
        X2 = np.where(np.isfinite(Xv), Xv, 0.0)
        sv = np.sum(np.abs(X2) ** pp, axis=1) ** (1.0 / pp)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # --- optional missingness normalization (not for percentiles) ---
    if normalize_missing and m not in {"p99", "p995", "p999", "percentile"}:
        nv = n_finite[valid]
        # guard
        nv = np.maximum(nv, 1.0)
        if m in {"topk_sum", "excess_mass", "l1"}:
            sv = sv * (float(D) / nv)
        elif m == "l2":
            sv = sv * np.sqrt(float(D) / nv)
        elif m == "lp":
            sv = sv * (float(D) / nv) ** (1.0 / float(p))

    scores[valid] = sv
    return scores


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QuietSelector:
    """
    Select a quiet subset of frames for background modeling.

    Quietness is computed from *unmasked* frequency channels only.

    Parameters
    ----------
    method
        Quietness metric. Lower is "quieter".
        Uses the unified MetricMethod vocabulary.

        Recommended starting points:
        - "p99": conservative, tends to reject frames with any bright-ish pixels
        - "topk_sum": more sensitive to sparse bright structure if topk is smallish
        - "excess_mass": good for "above-threshold" type quietness
    quiet_fraction
        Fraction of frames to keep as quiet (0 < quiet_fraction <= 1).
        Selection is by ascending quietness score.
    freq_mask
        Boolean mask of shape (F,). True means keep the channel.
        If None, all channels are used.

    Metric parameters
    -----------------
    q
        Used when method="percentile".
    topk
        Used for method="topk_sum".
    thr
        Used for method="excess_mass".
    p
        Used for method="lp".
    """

    method: MetricMethod = "p99"
    quiet_fraction: float = 0.7
    freq_mask: Optional[Array] = None

    q: float = 99.0
    topk: int = 2048
    thr: float = 3.0
    p: float = 4.0

    def scores(self, frames: Array) -> Array:
        """
        Compute per-frame quietness scores.

        Parameters
        ----------
        frames
            Frame batch of shape (N, T, F) or (N, T, F, C).

        Returns
        -------
        scores
            (N,) array. Lower is quieter.
        """
        X = _frames_to_X(frames, freq_mask=self.freq_mask)
        return compute_metric(
            X,
            method=self.method,
            q=float(self.q),
            topk=int(self.topk),
            thr=float(self.thr),
            p=float(self.p),
        )

    def select_quiet(self, frames: Array) -> Array:
        """
        Return indices of quiet frames (ascending by quietness).

        Parameters
        ----------
        frames
            Frame batch of shape (N, T, F) or (N, T, F, C).

        Returns
        -------
        idx
            1D integer array of selected indices (sorted ascending by quietness).

        Notes
        -----
        NaN scores are treated as non-quiet and tend to appear at the end of the
        ordering produced by np.argsort.
        """
        s = self.scores(frames)
        n = int(s.size)

        qf = float(self.quiet_fraction)
        if not (0.0 < qf <= 1.0):
            raise ValueError("quiet_fraction must be in (0, 1].")

        k = int(max(1, np.floor(qf * n)))
        order = np.argsort(s)
        return order[:k]


@dataclass
class RollingPCABackground:
    """
    PCA background model fit on quiet frames.

    This object is intentionally "stateless with respect to time": it does not
    manage dataset iteration by itself. Instead, you:
      - build batches/chunks of frames externally,
      - select quiet frames with QuietSelector,
      - call fit() on the quiet subset,
      - call reconstruct()/residuals()/score() on frames of interest.

    Parameters
    ----------
    k
        Number of PCA components to fit.
    center
        If True, mean-center features before SVD (classic PCA). Strongly recommended.
    freq_mask
        Optional boolean mask of shape (F,) applied before flattening.
        If provided, PCA ignores masked channels entirely.
    use_randomized
        If True and scikit-learn is available, use randomized SVD.
    n_iter
        Power iterations for randomized SVD (typical 1–3).
    random_state
        Seed for randomized SVD.

    Learned attributes (after fit)
    ------------------------------
    mu_
        Mean vector (D,) if center=True, else None.
    Vt_
        Components (k, D). Rows are basis vectors in feature space.
    S_
        Singular values (k,).

    Notes
    -----
    - The PCA model is trained in the same feature space produced by masking +
      flattening. If you pass freq_mask, you should use the same mask for quiet
      selection and scoring for consistency.
    - NaN handling: during fit(), any frame rows containing non-finite features
      are dropped (conservative).
    """

    k: int = 256
    center: bool = True
    freq_mask: Optional[Array] = None
    use_randomized: bool = True
    n_iter: int = 2
    random_state: int = 0

    # learned
    mu_: Optional[Array] = None
    Vt_: Optional[Array] = None
    S_: Optional[Array] = None

    def _prep(self, frames: Array) -> Array:
        """Apply mask (if any) and flatten to feature matrix X (N, D)."""
        return _frames_to_X(frames, freq_mask=self.freq_mask)

    def fit(self, quiet_frames: Array) -> "RollingPCABackground":
        """
        Fit PCA model on quiet frames.

        Parameters
        ----------
        quiet_frames
            Array of shape (Nq, T, F) or (Nq, T, F, C).

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
        Low-rank reconstruction of frames in feature space.

        Parameters
        ----------
        frames
            (N, T, F) or (N, T, F, C)
        k
            Number of PCA components to use. Defaults to self.k.

        Returns
        -------
        X_hat
            (N, D) reconstruction in feature space.
        """
        if self.Vt_ is None:
            raise RuntimeError("Model is not fit yet. Call fit() first.")

        X = self._prep(frames)  # (N, D)

        if self.center and self.mu_ is not None:
            Xc = X - self.mu_
        else:
            Xc = X

        # NaN-robust projection input: treat missing entries as "no deviation"
        finite = np.isfinite(Xc)
        Xc_proj = np.where(finite, Xc, 0.0)

        kk = int(self.k if k is None else k)
        kk = min(kk, int(self.Vt_.shape[0]))

        V = self.Vt_[:kk, :]  # (kk, D)
        C = Xc_proj @ V.T  # (N, kk)
        X_hat = C @ V  # (N, D)

        if self.center and self.mu_ is not None:
            X_hat = X_hat + self.mu_

        return X_hat

    def residuals(self, frames: Array, *, k: Optional[int] = None) -> Array:
        """
        Residuals in feature space: R = X - X_hat.

        Returns
        -------
        R
            (N, D) residual feature matrix.
        """
        X = self._prep(frames)
        X_hat = self.reconstruct(frames, k=k)
        return X - X_hat

    def score(
        self,
        frames: Array,
        *,
        method: MetricMethod = "topk_sum",
        k_pca: Optional[int] = None,
        positive_only: bool = True,
        # metric params
        q: float = 99.9,
        topk: int = 2048,
        thr: float = 3.0,
        p: float = 4.0,
        min_finite_frac: float = 0.7,
        normalize_missing: bool = True,
    ) -> Array:
        """
        Score outliers using residuals against the PCA background model.

        Parameters
        ----------
        frames
            (N, T, F) or (N, T, F, C).
        method
            MetricMethod applied to residuals (after optional positive clipping).
        k_pca
            Number of PCA modes to use for reconstruction. If None, uses self.k.
        positive_only
            If True, residuals are clipped to positive values (bright-only detection).
        q, topk, thr, p
            Metric parameters (see compute_metric).

        Returns
        -------
        scores
            (N,) array. Higher means "more outlier-like".
        """
        R = self.residuals(frames, k=k_pca)
        if positive_only:
            R = np.clip(R, 0.0, None)

        return compute_metric(
            R,
            method=method,
            q=float(q),
            topk=int(topk),
            thr=float(thr),
            p=float(p),
            min_finite_frac=float(min_finite_frac),
            normalize_missing=bool(normalize_missing),
        )


# -----------------------------------------------------------------------------
# Small rolling helper (pure numpy, no I/O)
# -----------------------------------------------------------------------------


def rolling_windows(
    indices: Iterable[int],
    *,
    window: int,
    stride: int,
) -> Iterable[list[int]]:
    """
    Yield overlapping windows of indices.

    Parameters
    ----------
    indices
        Iterable of integer indices (e.g. range(len(ds))).
    window
        Window length (number of indices).
    stride
        Step between window starts.

    Yields
    ------
    win
        List of indices for each window.
    """
    idx = [int(i) for i in indices]
    n = len(idx)
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive.")
    for s in range(0, max(0, n - window + 1), stride):
        yield idx[s : s + window]
