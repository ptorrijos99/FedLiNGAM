"""
Cumulant Estimators for Causal Discovery.

This module provides unbiased estimators for 3rd and 4th-order cumulants
and cross-cumulants, designed for both local (single-machine) and federated
learning scenarios.

Architecture
------------
The module is organized in three layers:

1. **Raw Moment Computation** — ``compute_local_raw_stats``
   Computes additive sufficient statistics (sums of powers / cross-products)
   from a local dataset.  These are the *only* quantities that leave each
   client in a federated setting.

2. **Aggregation & Cumulant Estimation** — ``aggregate_and_compute_cumulants``
   Aggregates raw moment sums from one or more clients, centers them using
   the global mean, and applies bias-corrected (k-statistic) formulas to
   obtain unbiased cumulant estimates up to 4th order.

3. **Convenience Wrappers** — ``unbiased_cumulant3``, ``unbiased_cumulant4``,
   ``cross_cumulants3``, ``cross_cumulants4``, ``compute_local_cumulants``,
   ``compute_sufficient_statistics``
   Thin wrappers that present the same API as the original single-machine
   functions, internally delegating to the raw-stats pipeline.

References
----------
[1] Chen et al., "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery", IEEE TNNLS, 2025.
[2] Schefczik & Hägele, "Ready-to-use unbiased estimators for
    multivariate cumulants including one that outperforms x³",
    arXiv:1904.12154, 2019.
[3] Chen et al., "FedISHC: Federated Causal Discovery via Higher-Order
    Cumulants", AAAI / arXiv, 2025.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Union

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# §1  Raw Moment Computation (client-side, privacy-safe)
# ═══════════════════════════════════════════════════════════════════════════

def compute_local_raw_stats(X: np.ndarray, order: int = 4) -> Dict[str, Union[int, np.ndarray]]:
    """
    Compute local raw moment sums suitable for federated aggregation.

    All returned quantities are *additive* across disjoint subsets of
    samples, which is the key property that enables secure aggregation
    without sharing raw data.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Local dataset with *n* samples and *p* variables.
    order : {3, 4}, default 4
        Maximum order of moments to compute.

    Returns
    -------
    dict
        Always present:
            n           : int            — sample count
            sum_x       : (p,)           — Σ xᵢ
            sum_xx      : (p, p)         — Σ xᵢ xⱼ  (includes diagonal)
            sum_x3      : (p,)           — Σ xᵢ³
            sum_x1y2    : (p, p)         — Σ xᵢ xⱼ²   (i ≠ j; diagonal = 0)
            sum_x2y1    : (p, p)         — Σ xᵢ² xⱼ   (i ≠ j; diagonal = 0)
        When ``order >= 4``:
            sum_x4      : (p,)           — Σ xᵢ⁴
            sum_x1y3    : (p, p)         — Σ xᵢ xⱼ³
            sum_x3y1    : (p, p)         — Σ xᵢ³ xⱼ
            sum_x2y2    : (p, p)         — Σ xᵢ² xⱼ²
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # --- 1st / 2nd order ---
    sum_x = np.sum(X, axis=0)                  # (p,)
    sum_xx = X.T @ X                            # (p, p)

    # --- Pre-compute element-wise powers ---
    X2 = X ** 2

    # --- 3rd order ---
    sum_x3 = np.sum(X2 * X, axis=0)            # (p,)  — xᵢ³ = xᵢ² · xᵢ
    sum_x1y2 = np.zeros((p, p))
    sum_x2y1 = np.zeros((p, p))

    for i in range(p):
        xi, xi2 = X[:, i], X2[:, i]
        for j in range(p):
            if i == j:
                continue
            xj2 = X2[:, j]
            sum_x1y2[i, j] = xi @ xj2           # Σ xᵢ xⱼ²
            sum_x2y1[i, j] = xi2 @ X[:, j]      # Σ xᵢ² xⱼ

    stats: Dict[str, Union[int, np.ndarray]] = {
        "n": n,
        "sum_x": sum_x,
        "sum_xx": sum_xx,
        "sum_x3": sum_x3,
        "sum_x1y2": sum_x1y2,
        "sum_x2y1": sum_x2y1,
    }

    if order < 4:
        return stats

    # --- 4th order ---
    X3 = X2 * X
    sum_x4 = np.sum(X2 ** 2, axis=0)           # (p,)
    sum_x1y3 = np.zeros((p, p))
    sum_x3y1 = np.zeros((p, p))
    sum_x2y2 = np.zeros((p, p))

    for i in range(p):
        xi, xi2, xi3 = X[:, i], X2[:, i], X3[:, i]
        for j in range(p):
            if i == j:
                continue
            xj, xj2, xj3 = X[:, j], X2[:, j], X3[:, j]
            sum_x1y3[i, j] = xi @ xj3           # Σ xᵢ xⱼ³
            sum_x3y1[i, j] = xi3 @ xj           # Σ xᵢ³ xⱼ
            sum_x2y2[i, j] = xi2 @ xj2          # Σ xᵢ² xⱼ²

    stats.update({
        "sum_x4": sum_x4,
        "sum_x1y3": sum_x1y3,
        "sum_x3y1": sum_x3y1,
        "sum_x2y2": sum_x2y2,
    })

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# §2  Aggregation & Global Cumulant Estimation (server-side)
# ═══════════════════════════════════════════════════════════════════════════

_MIN_SAMPLES_3RD = 3
_MIN_SAMPLES_4TH = 4


def aggregate_and_compute_cumulants(
    client_stats_list: List[Dict[str, Union[int, np.ndarray]]],
) -> Dict[str, Union[int, np.ndarray]]:
    """
    Aggregate raw moment sums from multiple clients and compute global
    unbiased cumulant estimates.

    Parameters
    ----------
    client_stats_list : list of dict
        Each element is the output of :func:`compute_local_raw_stats` from
        one client.  All clients must share the same variable set (i.e.
        horizontal FL).

    Returns
    -------
    dict
        n    : int          — total sample count
        cov  : (p, p)       — sample covariance matrix
        c3   : (p,)         — C₃(xᵢ)
        c12  : (p, p)       — C₁,₂(xᵢ, xⱼ) = Cum(xᵢ, xⱼ, xⱼ)
        c21  : (p, p)       — C₂,₁(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xⱼ)
        When 4th-order stats are available:
            c4   : (p,)     — C₄(xᵢ)
            c13  : (p, p)   — C₁,₃(xᵢ, xⱼ) = Cum(xᵢ, xⱼ, xⱼ, xⱼ)
            c31  : (p, p)   — C₃,₁(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xᵢ, xⱼ)
            c22  : (p, p)   — C₂,₂(xᵢ, xⱼ) = Cum(xᵢ, xᵢ, xⱼ, xⱼ)

    Notes
    -----
    Bias correction follows the k-statistic formulas [2]:

    * 3rd order:  k₃ = N / ((N−1)(N−2)) · M₃
    * 4th order:  k₄ = [N(N+1) M₄ − 3(N−1) M₂²] / [(N−1)(N−2)(N−3)]

    where Mₖ denotes the sum of k-th centered powers.
    """
    if not client_stats_list:
        return {}

    p = len(client_stats_list[0]["sum_x"])
    has_4th = "sum_x4" in client_stats_list[0]

    # ── Step 1: Sum all raw statistics across clients ──────────────────
    totals = _aggregate_raw_stats(client_stats_list, p, has_4th)

    N = totals["n"]
    mu = totals["sum_x"] / N

    # ── Step 2: Sample covariance ──────────────────────────────────────
    cov = (totals["sum_xx"] - N * np.outer(mu, mu)) / max(N - 1, 1)

    # ── Step 3: 3rd-order cumulants ───────────────────────────────────
    c3, c12, c21 = _compute_3rd_order(totals, mu, N, p)

    result: Dict[str, Union[int, np.ndarray]] = {
        "n": N,
        "cov": cov,
        "c3": c3,
        "c12": c12,
        "c21": c21,
    }

    if not has_4th:
        return result

    # ── Step 4: 4th-order cumulants ───────────────────────────────────
    c4, c13, c31, c22 = _compute_4th_order(totals, mu, N, p)

    result.update({"c4": c4, "c13": c13, "c31": c31, "c22": c22})
    return result


# ── Private helpers for aggregation ───────────────────────────────────────

def _aggregate_raw_stats(
    stats_list: List[Dict], p: int, has_4th: bool
) -> Dict[str, Union[int, np.ndarray]]:
    """Sum raw moment arrays across all clients."""
    totals: Dict[str, Union[int, np.ndarray]] = {
        "n": 0,
        "sum_x": np.zeros(p),
        "sum_xx": np.zeros((p, p)),
        "sum_x3": np.zeros(p),
        "sum_x1y2": np.zeros((p, p)),
        "sum_x2y1": np.zeros((p, p)),
    }

    if has_4th:
        totals.update({
            "sum_x4": np.zeros(p),
            "sum_x1y3": np.zeros((p, p)),
            "sum_x3y1": np.zeros((p, p)),
            "sum_x2y2": np.zeros((p, p)),
        })

    for s in stats_list:
        totals["n"] += s["n"]
        for key in totals:
            if key == "n":
                continue
            if key in s:
                totals[key] = totals[key] + s[key]

    return totals


def _compute_3rd_order(
    totals: Dict, mu: np.ndarray, N: int, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unbiased 3rd-order marginal and cross-cumulants.

    Centering identity used:
        Σ(x − μ)³ = S₃ − 3μ S₂ + 2Nμ³
    """
    if N < _MIN_SAMPLES_3RD:
        warnings.warn(
            f"Total sample size N={N} < {_MIN_SAMPLES_3RD}; "
            "3rd-order cumulants are unreliable.",
            stacklevel=3,
        )
        return np.zeros(p), np.zeros((p, p)), np.zeros((p, p))

    factor3 = N / ((N - 1) * (N - 2))

    S_xx = totals["sum_xx"]
    S_x3 = totals["sum_x3"]
    S_x1y2 = totals["sum_x1y2"]
    S_x2y1 = totals["sum_x2y1"]

    # ── Marginal C₃(xᵢ) ──
    #   Σ(xᵢ − μᵢ)³ = S₃[i] − 3μᵢ·S₂[i] + 2N·μᵢ³
    sum_cen_x3 = S_x3 - 3 * mu * np.diag(S_xx) + 2 * N * (mu ** 3)
    c3 = factor3 * sum_cen_x3

    # ── Cross-cumulants C₁,₂ and C₂,₁ ──
    c12 = np.zeros((p, p))
    c21 = np.zeros((p, p))

    for i in range(p):
        c12[i, i] = c3[i]
        c21[i, i] = c3[i]
        for j in range(p):
            if i == j:
                continue

            mi, mj = mu[i], mu[j]

            # Σ(xᵢ−μᵢ)(xⱼ−μⱼ)²
            #   = S_x1y2[i,j] − 2μⱼ·S_xx[i,j] − μᵢ·S_xx[j,j] + 2N·μᵢ·μⱼ²
            M_12 = (
                S_x1y2[i, j]
                - 2 * mj * S_xx[i, j]
                - mi * S_xx[j, j]
                + 2 * N * mi * mj ** 2
            )
            c12[i, j] = factor3 * M_12

            # Σ(xᵢ−μᵢ)²(xⱼ−μⱼ)  — symmetric swap of roles
            #   = S_x2y1[i,j] − 2μᵢ·S_xx[i,j] − μⱼ·S_xx[i,i] + 2N·μⱼ·μᵢ²
            M_21 = (
                S_x2y1[i, j]
                - 2 * mi * S_xx[i, j]
                - mj * S_xx[i, i]
                + 2 * N * mj * mi ** 2
            )
            c21[i, j] = factor3 * M_21

    return c3, c12, c21


def _compute_4th_order(
    totals: Dict, mu: np.ndarray, N: int, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unbiased 4th-order marginal and cross-cumulants.

    k-statistic formula:
        k₄ = [N(N+1)·M₄ − 3(N−1)·M₂²] / [(N−1)(N−2)(N−3)]

    Centering identities used:
        Σ(x−μ)⁴ = S₄ − 4μ S₃ + 6μ² S₂ − 3Nμ⁴
        Σ(x−μₓ)(y−μᵧ)³ = S_{x¹y³} − μₓ S_{y³} − 3μᵧ S_{x¹y²}
                          + 3μₓμᵧ S_{y²} + 3μᵧ² S_{xy} − 3Nμₓμᵧ³
    """
    if N < _MIN_SAMPLES_4TH:
        warnings.warn(
            f"Total sample size N={N} < {_MIN_SAMPLES_4TH}; "
            "4th-order cumulants are unreliable.",
            stacklevel=3,
        )
        z = np.zeros(p)
        zz = np.zeros((p, p))
        return z, zz.copy(), zz.copy(), zz.copy()

    D4 = (N - 1) * (N - 2) * (N - 3)
    alpha4 = N * (N + 1)
    beta4 = N - 1

    S_xx = totals["sum_xx"]
    S_x3 = totals["sum_x3"]
    S_x4 = totals["sum_x4"]
    S_x1y2 = totals["sum_x1y2"]
    S_x2y1 = totals["sum_x2y1"]
    S_x1y3 = totals["sum_x1y3"]
    S_x3y1 = totals["sum_x3y1"]
    S_x2y2 = totals["sum_x2y2"]

    # ── Centered 2nd-moment sums (reused in bias correction) ──
    #   M₂[i] = Σ(xᵢ − μᵢ)² = S_xx[i,i] − N μᵢ²
    M2_diag = np.diag(S_xx) - N * mu ** 2

    # ── Marginal C₄(xᵢ) ──
    #   M₄ = S₄ − 4μ S₃ + 6μ² S₂ − 3Nμ⁴
    M4_vec = S_x4 - 4 * mu * S_x3 + 6 * mu ** 2 * np.diag(S_xx) - 3 * N * mu ** 4
    c4 = (alpha4 * M4_vec - 3 * beta4 * M2_diag ** 2) / D4

    # ── Cross-cumulants C₁,₃ / C₃,₁ / C₂,₂ ──
    c13 = np.zeros((p, p))
    c31 = np.zeros((p, p))
    c22 = np.zeros((p, p))

    for i in range(p):
        # Diagonal: all cross-cumulants collapse to the marginal
        c13[i, i] = c4[i]
        c31[i, i] = c4[i]
        c22[i, i] = c4[i]

        for j in range(p):
            if i == j:
                continue

            mi, mj = mu[i], mu[j]

            # --- M₁₃ = Σ(xᵢ − μᵢ)(xⱼ − μⱼ)³ ---
            #   = S_{x¹y³} − μᵢ·S_{y³} − 3μⱼ·S_{x¹y²}
            #     + 3μᵢμⱼ·S_{y²} + 3μⱼ²·S_{xy} − 3N·μᵢ·μⱼ³
            M13 = (
                S_x1y3[i, j]
                - mi * S_x3[j]
                - 3 * mj * S_x1y2[i, j]
                + 3 * mi * mj * S_xx[j, j]
                + 3 * mj ** 2 * S_xx[i, j]
                - 3 * N * mi * mj ** 3           # ← μⱼ³ (not μⱼ²)
            )

            # --- M₃₁ = Σ(xᵢ − μᵢ)³(xⱼ − μⱼ)  [swap i ↔ j roles in M₁₃] ---
            #   = S_{x³y¹} − μⱼ·S_{x³} − 3μᵢ·S_{x²y¹}
            #     + 3μⱼμᵢ·S_{x²} + 3μᵢ²·S_{xy} − 3N·μⱼ·μᵢ³
            M31 = (
                S_x3y1[i, j]
                - mj * S_x3[i]
                - 3 * mi * S_x2y1[i, j]
                + 3 * mj * mi * S_xx[i, i]
                + 3 * mi ** 2 * S_xx[i, j]
                - 3 * N * mj * mi ** 3           # ← μᵢ³ (not μᵢ²)
            )

            # --- M₂₂ = Σ(xᵢ − μᵢ)²(xⱼ − μⱼ)² ---
            #   = S_{x²y²} − 2μⱼ·S_{x²y¹} − 2μᵢ·S_{x¹y²}
            #     + μⱼ²·S_{x²} + μᵢ²·S_{y²} + 4μᵢμⱼ·S_{xy} − 3N·μᵢ²μⱼ²
            M22 = (
                S_x2y2[i, j]
                - 2 * mj * S_x2y1[i, j]
                - 2 * mi * S_x1y2[i, j]
                + mj ** 2 * S_xx[i, i]
                + mi ** 2 * S_xx[j, j]
                + 4 * mi * mj * S_xx[i, j]
                - 3 * N * mi ** 2 * mj ** 2
            )

            # --- Centered 2nd-moment cross-sum ---
            #   M₁₁ = Σ(xᵢ − μᵢ)(xⱼ − μⱼ) = S_xx[i,j] − N·μᵢμⱼ
            M11 = S_xx[i, j] - N * mi * mj
            M20 = M2_diag[i]
            M02 = M2_diag[j]

            # --- Unbiased k-statistics ---
            c13[i, j] = (alpha4 * M13 - 3 * beta4 * M11 * M02) / D4
            c31[i, j] = (alpha4 * M31 - 3 * beta4 * M11 * M20) / D4
            c22[i, j] = (alpha4 * M22 - beta4 * (M20 * M02 + 2 * M11 ** 2)) / D4

    return c4, c13, c31, c22


# ═══════════════════════════════════════════════════════════════════════════
# §3  Convenience Wrappers (backward-compatible API)
# ═══════════════════════════════════════════════════════════════════════════

# ── Single-vector estimators ──────────────────────────────────────────────

def unbiased_cumulant3(
    x: np.ndarray, y: np.ndarray | None = None, z: np.ndarray | None = None
) -> float:
    """
    Unbiased 3rd-order cumulant estimator.

    * ``unbiased_cumulant3(x)``       → C₃(x) = Cum(x, x, x)
    * ``unbiased_cumulant3(x, y, z)`` → Cum(x, y, z)

    Uses k-statistic:  k₃ = N / ((N−1)(N−2)) · Σ(xᵢ−μₓ)(yᵢ−μᵧ)(zᵢ−μ_z)
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        y = z = x
    else:
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float) if z is not None else y

    n = len(x)
    if n < _MIN_SAMPLES_3RD:
        raise ValueError(
            f"Need at least {_MIN_SAMPLES_3RD} samples; got {n}."
        )

    xc = x - x.mean()
    yc = y - y.mean()
    zc = z - z.mean()

    factor = n / ((n - 1) * (n - 2))
    return float(factor * np.sum(xc * yc * zc))


def unbiased_cumulant4(x: np.ndarray) -> float:
    """
    Unbiased 4th-order marginal cumulant (excess kurtosis numerator).

    k₄ = [N(N+1)·M₄ − 3(N−1)·M₂²] / [(N−1)(N−2)(N−3)]

    where M₄ = Σ(xᵢ − x̄)⁴  and  M₂ = Σ(xᵢ − x̄)².
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < _MIN_SAMPLES_4TH:
        raise ValueError(
            f"Need at least {_MIN_SAMPLES_4TH} samples; got {n}."
        )

    xc = x - x.mean()
    M2 = np.sum(xc ** 2)
    M4 = np.sum(xc ** 4)

    D = (n - 1) * (n - 2) * (n - 3)
    return float((n * (n + 1) * M4 - 3 * (n - 1) * M2 ** 2) / D)


# ── Pairwise cross-cumulant estimators ────────────────────────────────────

def cross_cumulants3(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Unbiased 3rd-order cross-cumulants between *x* and *y*.

    Returns
    -------
    (C₁₂, C₂₁) : tuple of float
        C₁₂ = Cum(x, y, y)
        C₂₁ = Cum(x, x, y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < _MIN_SAMPLES_3RD:
        raise ValueError(
            f"Need at least {_MIN_SAMPLES_3RD} samples; got {n}."
        )

    xc = x - x.mean()
    yc = y - y.mean()
    factor = n / ((n - 1) * (n - 2))

    c12 = float(factor * np.sum(xc * yc ** 2))
    c21 = float(factor * np.sum(xc ** 2 * yc))
    return c12, c21


def cross_cumulants4(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Unbiased 4th-order cross-cumulants between *x* and *y*.

    Returns
    -------
    (C₁₃, C₃₁, C₂₂) : tuple of float
        C₁₃ = Cum(x, y, y, y)
        C₃₁ = Cum(x, x, x, y)
        C₂₂ = Cum(x, x, y, y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < _MIN_SAMPLES_4TH:
        raise ValueError(
            f"Need at least {_MIN_SAMPLES_4TH} samples; got {n}."
        )

    xc = x - x.mean()
    yc = y - y.mean()

    s20 = np.sum(xc ** 2)
    s02 = np.sum(yc ** 2)
    s11 = np.sum(xc * yc)

    s13 = np.sum(xc * yc ** 3)
    s31 = np.sum(xc ** 3 * yc)
    s22 = np.sum(xc ** 2 * yc ** 2)

    D = (n - 1) * (n - 2) * (n - 3)
    alpha = n * (n + 1)
    beta = n - 1

    k13 = (alpha * s13 - 3 * beta * s11 * s02) / D
    k31 = (alpha * s31 - 3 * beta * s11 * s20) / D
    k22 = (alpha * s22 - beta * (s20 * s02 + 2 * s11 ** 2)) / D

    return float(k13), float(k31), float(k22)


# ── Dataset-level wrappers (legacy API) ───────────────────────────────────

def compute_local_cumulants(X: np.ndarray) -> Dict[str, Union[list, float]]:
    """
    Compute all pairwise 3rd-order cumulants for a local dataset.

    This is the legacy API used by FedISHC clients.  Internally delegates
    to the raw-stats → aggregation pipeline (single-client case).

    Returns
    -------
    dict with keys ``'c3'``, ``'c12'``, ``'c21'`` (as lists), ``'n'``.
    """
    raw = compute_local_raw_stats(X, order=3)
    stats = aggregate_and_compute_cumulants([raw])
    return {
        "c3": stats["c3"].tolist(),
        "c12": stats["c12"].tolist(),
        "c21": stats["c21"].tolist(),
        "n": stats["n"],
    }


def compute_sufficient_statistics(X: np.ndarray) -> Dict[str, Union[list, float]]:
    """
    Compute 3rd-order cumulants + covariance for a local dataset.

    Returns
    -------
    dict with keys ``'c3'``, ``'c12'``, ``'c21'``, ``'cov'`` (as lists),
    ``'n'``.
    """
    raw = compute_local_raw_stats(X, order=3)
    stats = aggregate_and_compute_cumulants([raw])
    return {
        "c3": stats["c3"].tolist(),
        "c12": stats["c12"].tolist(),
        "c21": stats["c21"].tolist(),
        "cov": stats["cov"].tolist(),
        "n": stats["n"],
    }


def compute_4th_order_local_stats(X: np.ndarray) -> Dict[str, list]:
    """
    Compute all pairwise 4th-order cumulants for a local dataset.

    Returns
    -------
    dict with keys ``'c4'``, ``'c13'``, ``'c31'``, ``'c22'`` (as lists).
    """
    raw = compute_local_raw_stats(X, order=4)
    stats = aggregate_and_compute_cumulants([raw])
    return {
        "c4": stats["c4"].tolist(),
        "c13": stats["c13"].tolist(),
        "c31": stats["c31"].tolist(),
        "c22": stats["c22"].tolist(),
    }


def compute_sufficient_statistics_4th(X: np.ndarray) -> Dict[str, Union[list, float]]:
    """
    Compute 3rd + 4th-order cumulants and covariance for a local dataset.

    Returns
    -------
    dict with keys ``'c3'``, ``'c12'``, ``'c21'``, ``'cov'``, ``'c4'``,
    ``'c13'``, ``'c31'``, ``'c22'`` (as lists), ``'n'``.
    """
    raw = compute_local_raw_stats(X, order=4)
    stats = aggregate_and_compute_cumulants([raw])
    return {
        "c3": stats["c3"].tolist(),
        "c12": stats["c12"].tolist(),
        "c21": stats["c21"].tolist(),
        "cov": stats["cov"].tolist(),
        "c4": stats["c4"].tolist(),
        "c13": stats["c13"].tolist(),
        "c31": stats["c31"].tolist(),
        "c22": stats["c22"].tolist(),
        "n": stats["n"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# §4  Second-Order Aggregation (for kernel-based federated methods)
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_global_second_order(
    client_stats_list: List[Dict[str, Union[int, np.ndarray]]],
) -> Dict[str, Union[int, np.ndarray]]:
    """
    Aggregate {N_k, sum_x, sum_xx} from clients to produce global mean and covariance.

    This is the minimal Round 0 aggregation needed for kernel-based federated
    algorithms (e.g. FedHSIC) that don't require higher-order cumulants.

    Parameters
    ----------
    client_stats_list : list of dict
        Each dict must have keys: 'N_k' (int), 'sum_x' (p,), 'sum_xx' (p, p).

    Returns
    -------
    dict with:
        N          : int       — total sample count
        global_mean: (p,)      — weighted mean
        global_cov : (p, p)    — population covariance (ddof=0)
        global_std : (p,)      — per-variable std (clamped at 1e-10)
    """
    N = sum(c['N_k'] for c in client_stats_list)
    sum_x = sum(c['sum_x'] for c in client_stats_list)
    sum_xx = sum(c['sum_xx'] for c in client_stats_list)

    global_mean = sum_x / N
    global_cov = sum_xx / N - np.outer(global_mean, global_mean)
    global_std = np.sqrt(np.maximum(np.diag(global_cov), 0.0))
    global_std = np.where(global_std < 1e-10, 1e-10, global_std)

    return {
        'N': N,
        'global_mean': global_mean,
        'global_cov': global_cov,
        'global_std': global_std,
    }