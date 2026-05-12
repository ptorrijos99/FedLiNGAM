"""
Common solver utilities for Federated Causal Discovery.

Includes regression-based estimation of the adjacency matrix B from
covariance matrices (OLS and Adaptive Lasso) and coordinate descent
optimizers.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional


def _estimate_B_from_cov(
    causal_order: List[int],
    cov: np.ndarray,
    prune_thres: float = 0.0,
) -> np.ndarray:
    """
    Re-estimate adjacency matrix B by OLS from the original covariance.

    Parameters
    ----------
    causal_order : list of int
        Variable indices from source to sink.
    cov : ndarray, shape (p, p)
        Global covariance matrix.
    prune_thres : float
        Smallest absolute value to keep in B.

    Returns
    -------
    B : ndarray, shape (p, p)
        Estimated DAG adjacency matrix.
    """
    p = cov.shape[0]
    B = np.zeros((p, p))
    for idx in range(1, len(causal_order)):
        child = causal_order[idx]
        parents = causal_order[:idx]
        Sigma_pp = cov[np.ix_(parents, parents)]
        Sigma_pc = cov[parents, child]
        try:
            beta = np.linalg.solve(Sigma_pp, Sigma_pc)
        except np.linalg.LinAlgError:
            # Handle near-singular covariance
            beta = np.linalg.lstsq(Sigma_pp, Sigma_pc, rcond=None)[0]
        B[child, parents] = beta
    
    if prune_thres > 0:
        B[np.abs(B) < prune_thres] = 0.0
    return B


def _estimate_B_lasso_from_cov(
    causal_order: List[int],
    cov: np.ndarray,
    n_samples: int = 1000,
    lmbda_type: str = 'fixed',
    prune_thres: float = 0.05,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Re-estimate B by Adaptive Lasso from the global covariance.

    Parameters
    ----------
    causal_order : list of int
        Topological order.
    cov : ndarray, shape (p, p)
        Global covariance matrix.
    n_samples : int
        Total number of samples (used for BIC).
    lmbda_type : str
        'fixed' for the 0.01 heuristic, 'bic' for analytical selection.
    prune_thres : float
        Final hard-threshold for zeros.
    gamma : float
        Power parameter for adaptive weights.
    """
    p = cov.shape[0]
    B = np.zeros((p, p))
    for idx in range(1, len(causal_order)):
        child = causal_order[idx]
        parents = causal_order[:idx]
        Sigma_pp = cov[np.ix_(parents, parents)]
        Sigma_pc = cov[parents, child]
        Sigma_cc = cov[child, child]

        # 1. Compute OLS for adaptive weights
        try:
            beta_ols = np.linalg.solve(Sigma_pp, Sigma_pc)
        except np.linalg.LinAlgError:
            beta_ols = np.linalg.lstsq(Sigma_pp, Sigma_pc, rcond=None)[0]
        
        weights = 1.0 / (np.abs(beta_ols) ** gamma + 1e-10)
        
        if lmbda_type == 'fixed':
            # 2a. Fixed heuristic: proportional to average parent variance
            lmbda = 0.01 * np.mean(np.diag(Sigma_pp))
            beta = _solve_lasso_cd(Sigma_pp, Sigma_pc, lmbda, weights)
        elif lmbda_type == 'bic':
            # 2b. BIC selection: search over a grid
            lmbda_max = np.max(np.abs(Sigma_pc)) if len(Sigma_pc) > 0 else 0
            if lmbda_max < 1e-12:
                beta = np.zeros(len(parents))
            else:
                # 20 points from 0.1% of max to max
                lambdas = np.logspace(np.log10(lmbda_max * 1e-3), np.log10(lmbda_max), 20)
                best_bic = np.inf
                beta = np.zeros(len(parents))
                log_n = np.log(n_samples)
                
                for l in lambdas:
                    curr_beta = _solve_lasso_cd(Sigma_pp, Sigma_pc, l, weights)
                    # RSS/n = Var(y) - 2*beta*Sxy + beta*Sxx*beta
                    rss_n = Sigma_cc - 2 * curr_beta.dot(Sigma_pc) + curr_beta.dot(Sigma_pp).dot(curr_beta)
                    rss_n = max(rss_n, 1e-12)
                    k = np.count_nonzero(curr_beta)
                    bic = n_samples * np.log(rss_n) + k * log_n
                    if bic < best_bic:
                        best_bic = bic
                        beta = curr_beta
        else:
            raise ValueError(f"Unknown lmbda_type: {lmbda_type}")
            
        B[child, parents] = beta

    B[np.abs(B) < prune_thres] = 0.0
    return B


def _solve_lasso_cd(
    Cxx: np.ndarray,
    Cxy: np.ndarray,
    lmbda: float,
    weights: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Coordinate descent Lasso optimizer using only covariance statistics.
    
    Solves: min_beta  1/2 beta' Cxx beta - Cxy' beta + lambda |w * beta|
    """
    q = len(Cxy)
    beta = np.zeros(q)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(q):
            # Partial residual: r_j = y'x_j - sum_{k!=j} x_k'x_j * beta_k
            r_j = Cxy[j] - (Cxx[j, :].dot(beta) - Cxx[j, j] * beta[j])
            
            # Soft thresholding
            threshold = lmbda * weights[j]
            if r_j > threshold:
                beta[j] = (r_j - threshold) / Cxx[j, j]
            elif r_j < -threshold:
                beta[j] = (r_j + threshold) / Cxx[j, j]
            else:
                beta[j] = 0.0
        
        if np.linalg.norm(beta - beta_old) < tol:
            break
    return beta


# ═══════════════════════════════════════════════════════════════════════════
# Public convenience API
# ═══════════════════════════════════════════════════════════════════════════

def estimate_B(
    causal_order: List[int],
    cov: np.ndarray,
    *,
    use_lasso: bool = False,
    n_samples: int = 1000,
    lmbda_type: str = 'fixed',
    prune_thres: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Estimate the adjacency matrix B given a causal order and covariance.

    Convenience wrapper around :func:`_estimate_B_from_cov` (OLS) and
    :func:`_estimate_B_lasso_from_cov` (Adaptive Lasso) that eliminates
    the repeated if/else pattern in strategy code.

    Parameters
    ----------
    causal_order : list[int]
        Source-to-sink ordering.
    cov : ndarray, shape (p, p)
        Global covariance matrix.
    use_lasso : bool
        If True, use Adaptive Lasso; otherwise OLS.
    n_samples : int
        Total sample count (needed for BIC in Lasso).
    lmbda_type : str
        'fixed' or 'bic' for lambda selection.
    prune_thres : float
        Threshold for pruning near-zero entries.
    gamma : float
        Power for Adaptive Lasso weights.

    Returns
    -------
    B : ndarray, shape (p, p)
    """
    if use_lasso:
        return _estimate_B_lasso_from_cov(
            causal_order, cov,
            n_samples=n_samples,
            lmbda_type=lmbda_type,
            prune_thres=prune_thres,
            gamma=gamma,
        )
    return _estimate_B_from_cov(causal_order, cov, prune_thres=prune_thres)

