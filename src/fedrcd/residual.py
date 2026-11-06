"""The fourth-order residual score, with and without scale, from the aggregated cumulants.

For a candidate cause i of j, with OLS residual r = x_j - beta x_i, the score is

    m31[i, j]   = |E[x_i^3 r]| = |k31 - beta k4|
    m31n[i, j]  = m31[i, j] / (sigma_i^3 sigma_j)

m31 is the statistic of Wang & Drton (2020, Theorem 1) divided by Var(x_i). It is zero
when x_i is a cause of x_j and generically non-zero in the other direction. Both use
only the statistics FedRCD already aggregates. The extended article uses the pair as an
instrument: dividing by the marginal scales removes whatever accuracy the scale
carried, and nothing else.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..common.solvers import _estimate_B_from_cov, _estimate_B_lasso_from_cov

SCORES = ("tau4", "m31", "m31n")


def score_matrix(agg: Dict, score: str) -> np.ndarray:
    """Pairwise score matrix; entry [i, j] is small when x_i is a cause of x_j."""
    cov = np.asarray(agg["cov"], dtype=float)
    var = np.clip(np.diag(cov), 1e-15, None)
    beta = cov / var[:, None]                     # beta[i, j]: regress x_j on x_i
    c4 = np.asarray(agg["c4"], dtype=float)
    c13 = np.asarray(agg["c13"], dtype=float)
    c31 = np.asarray(agg["c31"], dtype=float)
    c22 = np.asarray(agg["c22"], dtype=float)
    if score == "tau4":
        S = c4[:, None] * c13 - c22 * c31
    elif score == "m31":
        S = c31 - beta * c4[:, None]
    elif score == "m31n":
        sd = np.sqrt(var)
        S = (c31 - beta * c4[:, None]) / (sd[:, None] ** 3 * sd[None, :])
    else:
        raise ValueError(f"unknown score {score!r}")
    S = np.abs(S)
    np.fill_diagonal(S, 0.0)
    return S


def fedrcd_ni_score(agg: Dict, score: str = "m31", use_lasso: bool = True,
                    lmbda_type: str = "fixed") -> Tuple[List[int], np.ndarray]:
    """One round: sort by the row sums of `score` (FedRCD-NI with a different score)."""
    order = np.argsort(score_matrix(agg, score).sum(axis=1), kind="stable").tolist()
    cov = np.asarray(agg["cov"], dtype=float)
    n = int(agg["n"])
    if use_lasso:
        B = _estimate_B_lasso_from_cov(order, cov, n_samples=n, lmbda_type=lmbda_type)
    else:
        B = _estimate_B_from_cov(order, cov)
    return order, B
