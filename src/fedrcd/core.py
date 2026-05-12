"""
FedRCD: Federated Regression-Directed Cumulants.

Three variants of the same algorithm family:

  fedrdc_noniterative()  [FedRCD-NI]
      One-shot: computes τ once on the original aggregated stats and sorts
      globally. Federated counterpart of HCLiNGAM (Algorithm 2). Requires
      only 1 communication round. Fast but ignores residual structure.

  fedrdc()               [FedRCD]
      Iterative algebraic deflation: updates cumulant statistics in-place
      using β coefficients (OLS) after each source removal. One-shot —
      no further communication after the initial stats exchange. Exact
      under the true LiNGAM model; accumulates numerical error under
      finite samples or model misspecification.

  fedrdc_exact()         [FedRCD-Exact]
      Multi-round exact deflation: p-1 rounds of communication. Each
      round the server broadcasts (source, β) and clients deflate their
      LOCAL raw data, then recompute fresh sufficient statistics. No
      algebraic approximation — exact federated counterpart of HC
      (Algorithm 1). Higher communication cost but no error accumulation.

  ishc()
      Improved ISHC baseline using β-deflation (kept for comparison).
"""

import numpy as np
from typing import List, Tuple, Dict
from ..common.metrics import compute_tau_matrix_3rd, compute_tau_matrix_4th_stats
from ..common.solvers import _estimate_B_from_cov, _estimate_B_lasso_from_cov


def fedrdc(
    stats: Dict, 
    adaptive: bool = False, 
    use_lasso: bool = False,
    lmbda_type: str = 'fixed'
) -> Tuple[List[int], np.ndarray]:
    """
    FedRDC Server-Side Solver.

    Identifies causal order iteratively, then re-estimates B from the
    original covariance using OLS.

    Parameters
    ----------
    stats : dict
        Aggregated statistics containing 'cov', 'c3', 'c12', 'c21',
        and optionally 'c4', 'c13', 'c31', 'c22'.
    adaptive : bool
        If True and 4th-order stats available, uses τ₄ by default and
        falls back to τ₃ only if kurtosis is too low. If False, uses
        τ₃ only.
    """
    cov = np.array(stats['cov'], dtype=float)
    cov_original = cov.copy()
    c3  = np.array(stats['c3'],  dtype=float).copy()
    c12 = np.array(stats['c12'], dtype=float).copy()
    c21 = np.array(stats['c21'], dtype=float).copy()

    has_4th = 'c4' in stats
    if has_4th:
        c4  = np.array(stats['c4'],  dtype=float).copy()
        c13 = np.array(stats['c13'], dtype=float).copy()
        c31 = np.array(stats['c31'], dtype=float).copy()
        c22 = np.array(stats['c22'], dtype=float).copy()

    p = len(c3)
    active = list(range(p))
    causal_order: List[int] = []

    for iteration in range(p - 1):
        # ── 1. Source Identification ──────────────────────────────────
        # Default: use 4th-order tau when available (stronger discriminator)
        use_4th = has_4th
        if adaptive and has_4th:
            # Fall back to 3rd-order only if kurtosis signal is too weak
            active_vars = cov.diagonal()[active]
            active_std = np.sqrt(np.maximum(active_vars, 1e-15))
            active_c4 = c4[active]
            active_kurt = active_c4 / (active_std ** 4)
            if np.max(np.abs(active_kurt)) < 0.1:
                # Very low kurtosis → 4th-order τ unreliable, try 3rd
                active_c3 = c3[active]
                active_skew = active_c3 / (active_std ** 3)
                if np.max(np.abs(active_skew)) > 0.15:
                    use_4th = False

        if use_4th:
            tau = compute_tau_matrix_4th_stats(
                c4[active],
                c13[np.ix_(active, active)],
                c31[np.ix_(active, active)],
                c22[np.ix_(active, active)],
                list(range(len(active)))
            )
        else:
            tau = compute_tau_matrix_3rd(
                c3[active],
                c12[np.ix_(active, active)],
                c21[np.ix_(active, active)],
                list(range(len(active)))
            )

        t_tau = np.sum(tau, axis=1)
        src_local = int(np.argmin(t_tau))
        s = active[src_local]
        causal_order.append(s)

        # ── 2. Compute β (OLS) — used for ALL updates ────────────────
        var_s = cov[s, s]
        if np.abs(var_s) < 1e-15:
            var_s = 1e-15

        betas = {}
        for j in active:
            if j == s:
                continue
            betas[j] = cov[j, s] / var_s

        # ── 3. Update all statistics using β ─────────────────────────
        remaining = [j for j in active if j != s]

        for i in remaining:
            bi = betas[i]

            # Covariance update
            for j in remaining:
                bj = betas[j]
                cov[i, j] -= bi * bj * var_s

            # 3rd-order cumulant updates (using β, not α)
            c3_s = c3[s]
            c3[i] = (c3[i]
                     - (bi ** 3) * c3_s
                     - 3 * bi * c21[i, s]
                     + 3 * (bi ** 2) * c12[i, s])

            for j in remaining:
                if i == j:
                    continue
                bj = betas[j]
                c12[i, j] -= bi * (bj ** 2) * c3_s
                c21[i, j] -= (bi ** 2) * bj * c3_s

            # 4th-order cumulant updates (using β)
            if has_4th:
                c4_s = c4[s]
                c4[i] = (c4[i]
                         - 4 * bi * c31[i, s]
                         + 6 * (bi ** 2) * c22[i, s]
                         - 4 * (bi ** 3) * c13[i, s]
                         + (bi ** 4) * c4_s)

                for j in remaining:
                    if i == j:
                        continue
                    bj = betas[j]
                    c13[i, j] -= bi * (bj ** 3) * c4_s
                    c31[i, j] -= (bi ** 3) * bj * c4_s
                    c22[i, j] -= (bi ** 2) * (bj ** 2) * c4_s

        # Update diagonals
        for j in remaining:
            c12[j, j] = c3[j]
            c21[j, j] = c3[j]
            if has_4th:
                c13[j, j] = c4[j]
                c31[j, j] = c4[j]
                c22[j, j] = c4[j]

        active = remaining

    if active:
        causal_order.append(active[0])

    # ── 4. Re-estimate B from ORIGINAL covariance ─────────────────────
    n_samples = stats.get('n', 1000)
    if use_lasso:
        B = _estimate_B_lasso_from_cov(
            causal_order, cov_original, 
            n_samples=n_samples, lmbda_type=lmbda_type
        )
    else:
        B = _estimate_B_from_cov(causal_order, cov_original)

    return causal_order, B


def ishc(
    stats: Dict, 
    use_lasso: bool = False,
    lmbda_type: str = 'fixed'
) -> Tuple[List[int], np.ndarray]:
    """
    ISHC variant using β (OLS) for deflation instead of α (cumulant ratio).

    This is our improved version. The original FedISHC (in fedishc/)
    is kept as-is since it's a published baseline.
    """
    c3  = np.array(stats['c3'],  dtype=float).copy()
    c12 = np.array(stats['c12'], dtype=float).copy()
    c21 = np.array(stats['c21'], dtype=float).copy()
    cov = np.array(stats['cov'], dtype=float)
    cov_original = cov.copy()

    p = len(c3)
    active = list(range(p))
    causal_order: List[int] = []

    from ..common.metrics import compute_tau_matrix_3rd

    for _ in range(p - 1):
        tau = compute_tau_matrix_3rd(
            c3[active],
            c12[np.ix_(active, active)],
            c21[np.ix_(active, active)],
            list(range(len(active)))
        )
        t_tau = np.sum(tau, axis=1)
        src_local = int(np.argmin(t_tau))
        s = active[src_local]
        causal_order.append(s)
        active.pop(src_local)

        # Use β (OLS) instead of α (c21/c3)
        var_s = cov[s, s]
        if np.abs(var_s) < 1e-15:
            var_s = 1e-15
        c3_s = c3[s]

        for j in active:
            bj = cov[j, s] / var_s

            # C3 update (full formula with β)
            c3[j] = (c3[j]
                     - (bj ** 3) * c3_s
                     - 3 * bj * c21[j, s]
                     + 3 * (bj ** 2) * c12[j, s])

            # Cross-cumulant updates
            for k in active:
                if k == j:
                    continue
                bk = cov[k, s] / var_s
                c12[j, k] -= bj * (bk ** 2) * c3_s
                c21[j, k] -= (bj ** 2) * bk * c3_s

            # Also update covariance for next iteration's β
            for k in active:
                cov[j, k] -= bj * (cov[k, s] / var_s) * var_s

        # Update diagonals
        for j in active:
            c12[j, j] = c3[j]
            c21[j, j] = c3[j]

    if active:
        causal_order.append(active[0])

    # Re-estimate B from original covariance
    n_samples = stats.get('n', 1000)
    if use_lasso:
        B = _estimate_B_lasso_from_cov(
            causal_order, cov_original, 
            n_samples=n_samples, lmbda_type=lmbda_type
        )
    else:
        B = _estimate_B_from_cov(causal_order, cov_original)

    return causal_order, B


# ═══════════════════════════════════════════════════════════════════════
# Non-iterative solvers (global τ sort — no deflation)
# ═══════════════════════════════════════════════════════════════════════

def fedrdc_noniterative(
    stats: Dict,
    adaptive: bool = False,
    use_lasso: bool = False,
    lmbda_type: str = 'fixed'
) -> Tuple[List[int], np.ndarray]:
    """
    Non-iterative FedRDC: global τ₄ sort + OLS/Lasso B.

    Avoids iterative cumulant deflation entirely. Computes τ₄ once
    from the aggregated stats and sorts by row sum (like HCLiNGAM).
    Then estimates B from the original covariance.
    """
    cov = np.array(stats['cov'], dtype=float)
    has_4th = 'c4' in stats

    if has_4th:
        c4  = np.array(stats['c4'],  dtype=float)
        c13 = np.array(stats['c13'], dtype=float)
        c31 = np.array(stats['c31'], dtype=float)
        c22 = np.array(stats['c22'], dtype=float)
        p = len(c4)
        tau = compute_tau_matrix_4th_stats(
            c4, c13, c31, c22, list(range(p))
        )
    else:
        c3  = np.array(stats['c3'],  dtype=float)
        c12 = np.array(stats['c12'], dtype=float)
        c21 = np.array(stats['c21'], dtype=float)
        p = len(c3)
        tau = compute_tau_matrix_3rd(
            c3, c12, c21, list(range(p))
        )

    # Sort by total τ score (ascending = most exogenous first)
    t_tau = np.sum(tau, axis=1)
    causal_order = np.argsort(t_tau).tolist()

    # B re-estimation
    n_samples = stats.get('n', 1000)
    if use_lasso:
        B = _estimate_B_lasso_from_cov(
            causal_order, cov,
            n_samples=n_samples, lmbda_type=lmbda_type
        )
    else:
        B = _estimate_B_from_cov(causal_order, cov)

    return causal_order, B


# ═══════════════════════════════════════════════════════════════════════
# Multi-round exact solver (clients deflate local data each round)
# ═══════════════════════════════════════════════════════════════════════

def fedrdc_exact(
    clients_X: List[np.ndarray],
    use_lasso: bool = False,
    lmbda_type: str = 'fixed'
) -> Tuple[List[int], np.ndarray]:
    """
    FedRCD-Exact: multi-round exact federated deflation.

    Runs p-1 communication rounds. Each round:
      1. Clients compute fresh sufficient statistics from their current
         (deflated) local data and send them to the server.
      2. Server aggregates stats, identifies the source variable via τ₄,
         and computes OLS coefficients β from the current covariance.
      3. Server broadcasts (src_local_idx, β) back to clients.
      4. Clients deflate: x_j := x_j − β_j · x_s  for all j ≠ s,
         then drop column s from their local data.

    Because statistics are recomputed from actual residuals at every round,
    no algebraic approximation is involved and numerical errors do not
    accumulate across steps. This is the exact federated counterpart of
    HC (Algorithm 1 in Chen et al., 2025).

    Parameters
    ----------
    clients_X : list of ndarray, each (n_k, p)
        Local datasets. Raw data is used only locally; only sufficient
        statistics (moment sums) leave each client.
    use_lasso : bool
        If True, estimate B with Adaptive Lasso; otherwise OLS.
    lmbda_type : str
        Lambda selection for Lasso: 'fixed' or 'bic'.

    Returns
    -------
    causal_order : list of int
        Estimated causal order, source first.
    B : ndarray, shape (p, p)
        Estimated adjacency matrix.

    Communication cost
    ------------------
    p-1 rounds × O(p²) statistics per round = O(p³) total.
    """
    from ..common.cumulants import compute_local_raw_stats, aggregate_and_compute_cumulants

    p = clients_X[0].shape[1]
    causal_order: List[int] = []

    # Each client maintains a working copy of its (progressively deflated) data.
    # Only sufficient statistics cross the network — raw data stays local.
    X_local = [X.copy() for X in clients_X]

    # Round 0 also gives us the original global covariance for final B estimation.
    init_raw = [compute_local_raw_stats(X, order=4) for X in X_local]
    init_agg = aggregate_and_compute_cumulants(init_raw)
    cov_original = np.array(init_agg['cov'])
    n_total = int(init_agg['n'])

    # active_global[local_idx] → original variable index
    active_global: List[int] = list(range(p))

    # Reuse the stats already computed for round 0
    agg = init_agg

    for step in range(p - 1):
        n_active = len(active_global)

        # ── 1. Source identification via τ₄ ──────────────────────────
        c4  = np.array(agg['c4'])
        c13 = np.array(agg['c13'])
        c31 = np.array(agg['c31'])
        c22 = np.array(agg['c22'])

        tau = compute_tau_matrix_4th_stats(
            c4, c13, c31, c22, list(range(n_active))
        )
        src_local = int(np.argmin(np.sum(tau, axis=1)))
        causal_order.append(active_global[src_local])

        # ── 2. Compute β from current covariance ─────────────────────
        cov_curr = np.array(agg['cov'])
        var_s = max(float(cov_curr[src_local, src_local]), 1e-15)
        # betas[j] = OLS coefficient for regressing x_j on x_s
        betas = cov_curr[:, src_local] / var_s  # shape (n_active,)

        # ── 3. Clients deflate locally and drop source column ─────────
        remaining_local = [j for j in range(n_active) if j != src_local]
        new_X_local = []
        for X in X_local:
            x_s = X[:, src_local]
            X_new = np.column_stack([
                X[:, j] - betas[j] * x_s for j in remaining_local
            ])
            new_X_local.append(X_new)
        X_local = new_X_local
        active_global = [active_global[j] for j in remaining_local]

        # ── 4. Clients send fresh stats for the next round ────────────
        if step < p - 2:
            raw = [compute_local_raw_stats(X, order=4) for X in X_local]
            agg = aggregate_and_compute_cumulants(raw)

    if active_global:
        causal_order.append(active_global[0])

    # ── 5. Re-estimate B from ORIGINAL covariance ─────────────────────
    if use_lasso:
        B = _estimate_B_lasso_from_cov(
            causal_order, cov_original,
            n_samples=n_total, lmbda_type=lmbda_type
        )
    else:
        B = _estimate_B_from_cov(causal_order, cov_original)

    return causal_order, B
