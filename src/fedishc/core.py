"""
Core ISHC Algorithm Implementation.

This module implements the Iterative Source Identification via Higher-order
Cumulants (ISHC) algorithm using 3rd-order cumulants, as adapted for the
Federated Setting (FedISHC), and a no-deflation variant (FedHC).

Algorithms
----------
- ishc       : Full FedISHC with iterative source identification and 3rd-order
               algebraic deflation (Chen et al., 2026, AAAI).
- ishc_4th   : Ablation of FedISHC where source identification uses tau^(4) but
               deflation remains 3rd-order. Demonstrates Proposition 2 of the
               accompanying paper (order-mismatch degeneracy).
- hc         : FedHC, the no-deflation variant of FedISHC. Equivalent to
               sorting by tau^(3) row sums in a single pass.
- compute_tau_matrix: Helper to compute the independence measure matrix.

References
----------
.. [1] Chen et al., "Horizontal and Vertical Federated Causal Structure Learning
       via Higher-Order Cumulants", AAAI 2026.
"""

from typing import List, Tuple, Dict, Union, Optional

import numpy as np

from ..common.metrics import compute_tau_matrix_3rd as compute_tau_matrix
from ..common.metrics import compute_tau_matrix_4th_stats


# Numerical safeguards used by every variant
_EPS = 1e-6
_ALPHA_CLIP = 10.0


def ishc(
    cumulants: Dict[str, Union[List, np.ndarray]],
) -> Tuple[List[int], np.ndarray]:
    """
    FedISHC: causal-order recovery via iterative source identification and
    3rd-order algebraic deflation.

    This method reconstructs the causal order and the adjacency matrix B
    from a dictionary of aggregated 3rd-order cumulants. It does not require
    access to the original data samples, making it suitable for federated
    learning under the protocol of Chen et al. (2026).

    Parameters
    ----------
    cumulants : dict
        Aggregated cumulant statistics:
        - 'c3'  : (p,)  3rd marginal cumulants
        - 'c12' : (p,p) Cum(x_i, x_j, x_j)
        - 'c21' : (p,p) Cum(x_i, x_i, x_j)

    Returns
    -------
    causal_order : list of int
        Estimated causal order. The first element is the root of the DAG.
    B : np.ndarray of shape (p, p)
        Estimated weighted adjacency matrix such that X = B X + E,
        recovered as B = I - A^{-1}.
    """
    c3 = np.asarray(cumulants['c3'], dtype=float)
    c12 = np.asarray(cumulants['c12'], dtype=float)
    c21 = np.asarray(cumulants['c21'], dtype=float)

    p = len(c3)
    active_indices = list(range(p))
    causal_order: List[int] = []

    # Mixing matrix A (X = A * E). Initialize as identity.
    A = np.eye(p)

    # Work on copies to avoid modifying inputs
    curr_c3 = c3.copy()
    curr_c12 = c12.copy()
    curr_c21 = c21.copy()

    for _ in range(p - 1):
        # 1. Source identification on currently active variables
        tau = compute_tau_matrix(curr_c3, curr_c12, curr_c21, active_indices)
        t_tau_scores = np.sum(tau, axis=1)

        min_idx_local = int(np.argmin(t_tau_scores))
        source_idx = active_indices[min_idx_local]
        causal_order.append(source_idx)

        s = source_idx
        active_indices.pop(min_idx_local)

        # 2. Algebraic deflation of remaining variables
        # Freeze c3_s before mutating curr_c3: deflation uses the value at the
        # moment s was identified, not the running estimate.
        c3_s = curr_c3[s]

        for xj in active_indices:
            # alpha = C_{2,1}(x_s, x_j) / C_3(x_s), the effect of source s on x_j
            if np.abs(c3_s) < _EPS:
                alpha = 0.0
            else:
                alpha = np.clip(curr_c21[s, xj] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP)

            A[xj, s] = alpha

            # Update C3(x_j): C3'(x_j) = C3(x_j) - alpha^3 * C3(s)
            curr_c3[xj] = curr_c3[xj] - (alpha ** 3) * c3_s

            for xk in active_indices:
                # lambda = effect of source s on x_k
                if np.abs(c3_s) > _EPS:
                    lam = np.clip(curr_c21[s, xk] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP)
                else:
                    lam = 0.0

                # C1,2'(x_j, x_k) = C1,2(x_j, x_k) - alpha * lambda^2 * C3(s)
                curr_c12[xj, xk] -= alpha * (lam ** 2) * c3_s
                # C2,1'(x_j, x_k) = C2,1(x_j, x_k) - alpha^2 * lambda * C3(s)
                curr_c21[xj, xk] -= (alpha ** 2) * lam * c3_s

    # Last variable
    if active_indices:
        causal_order.append(active_indices[0])

    # Recover B from A
    try:
        A_inv = np.linalg.inv(A)
        B = np.eye(p) - A_inv
    except np.linalg.LinAlgError:
        print("Warning: mixing matrix A is singular. Returning zero B.")
        B = np.zeros((p, p))

    return causal_order, B


def ishc_4th(
    cumulants: Dict[str, Union[List, np.ndarray]],
) -> Tuple[List[int], np.ndarray]:
    """
    FedISHC(tau^(4)): ablation that swaps only the source identification
    criterion (tau^(3) -> tau^(4)) while retaining 3rd-order algebraic
    deflation.

    By Proposition 2 of the accompanying paper (order-mismatch degeneracy)
    this procedure is non-iterative in effect: the 4th-order cumulants that
    drive tau^(4) are left invariant by the 3rd-order updates, so every
    iteration evaluates the same matrix on the active set. The expected
    output equals an argsort of the initial tau^(4) row sums.

    Parameters
    ----------
    cumulants : dict
        Required keys:
        - 'c3', 'c12', 'c21'        : 3rd-order arrays (used for deflation)
        - 'c4', 'c13', 'c31', 'c22' : 4th-order arrays (used for tau^(4))
    """
    c3 = np.asarray(cumulants['c3'], dtype=float)
    c12 = np.asarray(cumulants['c12'], dtype=float)
    c21 = np.asarray(cumulants['c21'], dtype=float)
    c4 = np.asarray(cumulants['c4'], dtype=float).copy()
    c13 = np.asarray(cumulants['c13'], dtype=float).copy()
    c31 = np.asarray(cumulants['c31'], dtype=float).copy()
    c22 = np.asarray(cumulants['c22'], dtype=float).copy()

    p = len(c3)
    active_indices = list(range(p))
    causal_order: List[int] = []
    A = np.eye(p)

    curr_c3 = c3.copy()
    curr_c12 = c12.copy()
    curr_c21 = c21.copy()

    for _ in range(p - 1):
        n_act = len(active_indices)
        local = list(range(n_act))

        # Source identification: 4th-order tau
        tau = compute_tau_matrix_4th_stats(
            c4[active_indices],
            c13[np.ix_(active_indices, active_indices)],
            c31[np.ix_(active_indices, active_indices)],
            c22[np.ix_(active_indices, active_indices)],
            local,
        )
        t_tau_scores = np.sum(tau, axis=1)
        min_idx_local = int(np.argmin(t_tau_scores))
        source_idx = active_indices[min_idx_local]
        causal_order.append(source_idx)

        s = source_idx
        active_indices.pop(min_idx_local)

        # Deflation: 3rd-order, with the original alpha coefficient
        c3_s = curr_c3[s]

        for xj in active_indices:
            if np.abs(c3_s) < _EPS:
                alpha = 0.0
            else:
                alpha = np.clip(curr_c21[s, xj] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP)

            A[xj, s] = alpha
            curr_c3[xj] = curr_c3[xj] - (alpha ** 3) * c3_s

            for xk in active_indices:
                if np.abs(c3_s) > _EPS:
                    lam = np.clip(curr_c21[s, xk] / c3_s, -_ALPHA_CLIP, _ALPHA_CLIP)
                else:
                    lam = 0.0
                curr_c12[xj, xk] -= alpha * (lam ** 2) * c3_s
                curr_c21[xj, xk] -= (alpha ** 2) * lam * c3_s

    if active_indices:
        causal_order.append(active_indices[0])

    try:
        A_inv = np.linalg.inv(A)
        B = np.eye(p) - A_inv
    except np.linalg.LinAlgError:
        print("Warning: mixing matrix A is singular. Returning zero B.")
        B = np.zeros((p, p))

    return causal_order, B


def hc(
    cumulants: Dict[str, Union[List, np.ndarray]],
    iterative: bool = False,
) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    FedHC: no-deflation variant of FedISHC.

    Identifies the causal order from the 3rd-order tau matrix without
    removing the influence of identified sources. By the same argument as
    Proposition 2 of the accompanying paper, the iterative loop and a single
    argsort of the initial tau row sums are expected to coincide on
    population statistics; under finite samples they may differ slightly,
    which is itself useful diagnostic information. The flag ``iterative``
    selects between the two implementations for testing.

    Parameters
    ----------
    cumulants : dict
        Aggregated cumulant statistics:
        - 'c3'  : (p,)  3rd marginal cumulants
        - 'c12' : (p,p) Cum(x_i, x_j, x_j)
        - 'c21' : (p,p) Cum(x_i, x_i, x_j)
    iterative : bool, optional (default False)
        If False, compute the order in one pass (faster, recommended).
        If True, run the iterative argmin loop. Useful as a sanity check
        that the two implementations agree on a given dataset.

    Returns
    -------
    causal_order : list of int
        Estimated causal order.
    B : None
        FedHC does not estimate edge weights. The caller is expected to
        recover B externally (e.g., via adaptive Lasso on the aggregated
        covariance matrix), as in FedRCD-NI in the accompanying paper.

    Notes
    -----
    Returning ``None`` (rather than an all-zero matrix) is intentional: a
    zero matrix would silently break SHD or precision/recall computations
    if accidentally piped into downstream code, while ``None`` produces an
    immediate, visible error.
    """
    c3 = np.asarray(cumulants['c3'], dtype=float)
    c12 = np.asarray(cumulants['c12'], dtype=float)
    c21 = np.asarray(cumulants['c21'], dtype=float)
    p = len(c3)

    if not iterative:
        # Single-pass implementation: equivalent to the iterative form on
        # population statistics by Proposition 2 of the paper.
        tau = compute_tau_matrix(c3, c12, c21, list(range(p)))
        t_tau_scores = np.sum(tau, axis=1)
        causal_order: List[int] = list(np.argsort(t_tau_scores).tolist())
        return causal_order, None

    # Iterative form, kept for testing equivalence with the single-pass form.
    active_indices = list(range(p))
    causal_order = []

    for _ in range(p - 1):
        tau = compute_tau_matrix(c3, c12, c21, active_indices)
        t_tau_scores = np.sum(tau, axis=1)
        min_idx_local = int(np.argmin(t_tau_scores))
        source_idx = active_indices[min_idx_local]
        causal_order.append(source_idx)
        active_indices.pop(min_idx_local)

    if active_indices:
        causal_order.append(active_indices[0])

    return causal_order, None