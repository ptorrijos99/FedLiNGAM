"""
ICA-to-LiNGAM Post-Processing (shared by FedICA and FedTD).

Converts an estimated unmixing matrix W from ICA/JADE into a causal
order and DAG adjacency matrix B via the procedure of Shimizu et al. (2006).

Steps:
  1. Row permutation via Hungarian algorithm to maximize |diag(W)|.
  2. Row normalisation so diagonal = 1.
  3. B = I - W_norm.
  4. Iterative pruning (Algorithm C) to find a DAG ordering.
  5. Enforce strict lower-triangularity given the ordering.

References
----------
- Shimizu, S. et al. (2006). A linear non-Gaussian acyclic model for
  causal discovery. JMLR, 7, 2003-2030.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import linear_sum_assignment


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def ica_to_lingam(
    W_est: np.ndarray,
    prune_thres: float = 0.05,
) -> Tuple[List[int], np.ndarray]:
    """
    Convert an estimated unmixing matrix W to a causal order and B matrix.

    Uses the robust strategy from the 'lingam' package:
      1. Hungarian permutation with scale-invariant cost (1/|W|).
      2. Row normalisation.
      3. B = I - W_norm.
      4. Iterative pruning → DAG test → causal order.
      5. Enforce DAG.

    Parameters
    ----------
    W_est : ndarray, shape (p, p)
        Estimated unmixing matrix from FastICA or JADE.

    Returns
    -------
    causal_order : list[int]
        Variable indices from most exogenous (source) to least (sink).
    B : ndarray, shape (p, p)
        Estimated adjacency matrix.
    """
    p = W_est.shape[0]

    # Step 1 — Hungarian permutation (scale-invariant cost 1/|W|)
    with np.errstate(divide='ignore'):
        cost = 1.0 / (np.abs(W_est) + 1e-12)
    row_ind, col_ind = linear_sum_assignment(cost)

    PW = np.zeros_like(W_est)
    PW[col_ind] = W_est[row_ind]

    # Step 2 — Normalise diagonal to 1
    diags = np.diag(PW).copy()
    diags[np.abs(diags) < 1e-12] = 1.0
    W_norm = PW / diags[:, np.newaxis]

    # Step 3 — Candidate B
    B_cand = np.eye(p) - W_norm

    # Step 4 — Iterative pruning to find DAG ordering
    causal_order = _iterative_pruning_order(B_cand)

    # Step 5 — Enforce strict lower-triangularity
    B_final = enforce_dag(B_cand, causal_order)

    return causal_order, B_final


def enforce_dag(B: np.ndarray, causal_order: List[int]) -> np.ndarray:
    """
    Zero out entries violating the ordering (upper triangle when permuted).

    Keep ``B[i, j]`` only if ``pos[j] < pos[i]`` (j strictly precedes i).
    Vectorised implementation via NumPy broadcasting.

    Parameters
    ----------
    B : ndarray, shape (p, p)
        Candidate adjacency matrix.
    causal_order : list of int
        Source-to-sink ordering.

    Returns
    -------
    B_out : ndarray, shape (p, p)
        DAG-consistent adjacency matrix.
    """
    p = B.shape[0]
    pos = np.empty(p, dtype=np.intp)
    pos[causal_order] = np.arange(p)
    mask = pos[np.newaxis, :] >= pos[:, np.newaxis]
    B_out = B.copy()
    B_out[mask] = 0.0
    return B_out


def try_topological_sort(B: np.ndarray) -> Optional[List[int]]:
    """
    Kahn's algorithm: return topological order if B is a DAG, else None.

    Convention: ``B[i, j] != 0`` ⟹ j → i (j is parent of i).
    """
    p = B.shape[0]
    adj = np.abs(B) > 0
    in_deg = adj.sum(axis=1).astype(np.intp)
    processed = np.zeros(p, dtype=bool)
    order: List[int] = []

    for _ in range(p):
        candidates = np.where((in_deg == 0) & ~processed)[0]
        if candidates.size == 0:
            return None
        node = int(candidates[0])
        order.append(node)
        processed[node] = True
        children = np.where(adj[:, node] & ~processed)[0]
        in_deg[children] -= 1

    return order


def heuristic_ordering(B: np.ndarray) -> List[int]:
    """
    Fallback: greedily peel node with smallest incoming edge mass.
    """
    p = B.shape[0]
    B_abs = np.abs(B.copy())
    np.fill_diagonal(B_abs, 0.0)
    processed = np.zeros(p, dtype=bool)
    order: List[int] = []

    for _ in range(p):
        mass = B_abs.sum(axis=1)
        mass[processed] = np.inf
        node = int(np.argmin(mass))
        order.append(node)
        processed[node] = True
        B_abs[:, node] = 0.0

    return order


def whitening_matrix(cov: np.ndarray) -> np.ndarray:
    """
    Compute Σ^{-1/2} via eigendecomposition.

    Handles near-singular covariance by clamping small eigenvalues.

    Parameters
    ----------
    cov : ndarray, shape (p, p)
        Symmetric positive semi-definite covariance matrix.

    Returns
    -------
    W_white : ndarray, shape (p, p)
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eps = max(1e-15, eigvals.max() * 1e-12)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]) @ eigvecs.T


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _iterative_pruning_order(B: np.ndarray) -> List[int]:
    """
    Find a causal order by iterative pruning (Shimizu 2006, Algorithm C).

    1. Sort all off-diagonal entries by |value|.
    2. Zero out the p(p+1)/2 smallest entries.
    3. Check if a clean DAG exists.
    4. If not, zero out the next smallest entry and repeat.
    """
    p = B.shape[0]
    B_work = B.copy()
    np.fill_diagonal(B_work, 0.0)

    rows, cols = np.where(~np.eye(p, dtype=bool))
    vals = np.abs(B_work[rows, cols])
    sort_idx = np.argsort(vals)

    n_start = min(p * (p + 1) // 2, len(sort_idx))

    for n_zero in range(n_start, len(sort_idx) + 1):
        B_test = B_work.copy()
        to_zero = sort_idx[:n_zero]
        B_test[rows[to_zero], cols[to_zero]] = 0.0

        order = _search_causal_order(B_test)
        if order is not None:
            return order

    return heuristic_ordering(B)


def _search_causal_order(matrix: np.ndarray) -> Optional[List[int]]:
    """
    Try to find a causal order by peeling rows that are all-zero.

    This is the 'lingam' package approach: repeatedly find a row with
    all entries below threshold, record it, and remove it.
    """
    causal_order = []
    row_num = matrix.shape[0]
    original_index = np.arange(row_num)
    curr_mat = matrix.copy()

    while len(curr_mat) > 0:
        row_index_list = np.where(np.sum(np.abs(curr_mat), axis=1) < 1e-12)[0]
        if len(row_index_list) == 0:
            break
        target_index = row_index_list[0]
        causal_order.append(int(original_index[target_index]))
        original_index = np.delete(original_index, target_index, axis=0)
        mask = np.delete(np.arange(len(curr_mat)), target_index, axis=0)
        curr_mat = curr_mat[mask][:, mask]

    return causal_order if len(causal_order) == row_num else None
