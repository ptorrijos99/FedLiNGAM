"""
Screened unlearning for the multi-round FedRCD-X protocol.

The problem
-----------
The single-round FedRCD variants unlearn exactly: raw moments are additive, the
server subtracts the forgotten contribution and re-solves. FedRCD-X has no such
shortcut. Round t+1 ran on residuals the clients built with a deflation
coefficient derived from the round-t aggregate, and an erasure changes that
coefficient. Two things go stale:

  (a) the leaving party's own contribution to every round's aggregate.
      Recoverable exactly -- the server already received per-client statistics
      each round, so it subtracts them.

  (b) the residuals the REMAINING clients computed, which used beta_old rather
      than beta_new.

Why (b) cannot be repaired algebraically
----------------------------------------
Write v for the round-(t-1) variables and s for that round's source. Clients
formed w_j = v_j - beta_j v_s; after erasure the target is
w'_j = w_j - delta_j v_s. By multilinearity the SELF cumulants kappa_4(w'_j) and
every bivariate quantity involving v_s follow from the round-(t-1) aggregate.
But tau^(4) also needs the PAIR terms kappa_{a,b}(w'_i, w'_j) for i, j both
different from s, and expanding those produces cumulants in three distinct
indices (i, j, s). Those trivariates are never transmitted.

They could be: for a FIXED source s the required blocks are O(p^2), not O(p^3),
so one round of staleness is repairable at the protocol's existing order of
cost. The catch is that the correction has to be applied at every round, and the
trivariates transmitted at round t were themselves computed on data already
stale with respect to round t-1. Repairing round t+1 exactly would need
4-variate blocks involving two sources, round t+2 five-variate, and so on. The
requirement compounds, so there is no fixed-order-cost exact scheme.

What this module does instead
-----------------------------
Subtract what is exactly subtractable (a), then use the resulting aggregate as a
SCREEN: at each round, recompute the argmin and compare it with the source the
protocol originally chose. While they agree, accept the prefix; at the first
disagreement, replay the remaining rounds with the clients still present.

This is a screening test, not a proof: the screen runs on residuals that are
still stale by delta_beta, so it can in principle certify a prefix that a true
re-run would not have produced. Its false-certification rate is an empirical
quantity, measured in `experiments/deployment/screened_unlearning.py`.

One thing IS exact regardless: FedRCD-X's B depends on the data only through the
causal order and the original global covariance, and the covariance is exactly
subtractable. So whenever the recovered order matches the re-run order, B is
bit-exact -- there is no separate error in the weights.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

from ..common.cumulants import compute_local_raw_stats, aggregate_and_compute_cumulants
from ..common.metrics import compute_tau_matrix_4th_stats
from ..common.solvers import _estimate_B_from_cov, _estimate_B_lasso_from_cov


# ═══════════════════════════════════════════════════════════════════════
# Recording the protocol as it runs
# ═══════════════════════════════════════════════════════════════════════

def fedrcd_exact_recorded(clients_X: List[np.ndarray]) -> Dict:
    """Run FedRCD-X and keep a transcript of what the server legitimately saw.

    The per-client raw-moment arrays of each round are exactly the messages the
    server received in order to aggregate them, so retaining them adds no new
    disclosure. (It does add storage: O(K p^2) per round.)
    """
    p = clients_X[0].shape[1]
    K = len(clients_X)
    X_local = [X.copy().astype(float) for X in clients_X]

    per_client_round: List[List[Dict]] = []
    sources: List[int] = []
    active_round: List[List[int]] = []

    raw = [compute_local_raw_stats(X, order=4) for X in X_local]
    per_client_round.append(raw)
    agg = aggregate_and_compute_cumulants(raw)

    active_global = list(range(p))
    causal_order: List[int] = []

    for step in range(p - 1):
        n_active = len(active_global)
        active_round.append(list(active_global))

        src_local = _argmin_source(agg, n_active)
        sources.append(src_local)
        causal_order.append(active_global[src_local])

        cov_curr = np.array(agg["cov"], dtype=float)
        var_s = max(float(cov_curr[src_local, src_local]), 1e-15)
        betas = cov_curr[:, src_local] / var_s

        remaining = [j for j in range(n_active) if j != src_local]
        X_local = [np.column_stack([X[:, j] - betas[j] * X[:, src_local]
                                    for j in remaining]) for X in X_local]
        active_global = [active_global[j] for j in remaining]

        if step < p - 2:
            raw = [compute_local_raw_stats(X, order=4) for X in X_local]
            per_client_round.append(raw)
            agg = aggregate_and_compute_cumulants(raw)

    causal_order.append(active_global[0])

    return {"order": causal_order, "sources": sources, "active": active_round,
            "per_client_round": per_client_round, "p": p, "K": K,
            "clients_X": clients_X}


def _argmin_source(agg: Dict, n_active: int) -> int:
    tau = compute_tau_matrix_4th_stats(
        np.array(agg["c4"]), np.array(agg["c13"]),
        np.array(agg["c31"]), np.array(agg["c22"]),
        list(range(n_active)))
    return int(np.argmin(np.sum(tau, axis=1)))


# ═══════════════════════════════════════════════════════════════════════
# Unlearning
# ═══════════════════════════════════════════════════════════════════════

def _sub(a: Dict, b: Dict) -> Dict:
    return {k: a[k] - b[k] for k in a}


def _round_aggregate(record: Dict, keep: List[int],
                     forget_stats: Optional[Dict[int, List[Dict]]], t: int) -> Dict:
    """Round-t aggregate over the retained data, by exact subtraction."""
    parts = []
    for k in keep:
        s = record["per_client_round"][t][k]
        if forget_stats and k in forget_stats and t < len(forget_stats[k]):
            s = _sub(s, forget_stats[k][t])
        parts.append(s)
    return aggregate_and_compute_cumulants(parts)


def screened_unlearn(record: Dict,
                     keep_client: Optional[Callable[[int], bool]] = None,
                     forget_stats: Optional[Dict[int, List[Dict]]] = None,
                     replay_data: Optional[List[np.ndarray]] = None,
                     use_lasso: bool = True,
                     lmbda_type: str = "fixed") -> Dict:
    """Honour an erasure request against a recorded FedRCD-X run.

    Parameters
    ----------
    record : output of `fedrcd_exact_recorded`.
    keep_client : predicate on client index; False means forget that client.
    forget_stats : {client_idx: [per-round stats of the rows to forget]} for
        instance-level erasure.
    replay_data : the retained clients' raw data, used only if the screen fails
        and a suffix replay is required. Passing None reports the divergence
        without replaying.

    Returns
    -------
    dict with the recovered order and B, the round at which the screen failed
    (`p` if it never did), and how many rounds were served without contacting
    any client.
    """
    p = record["p"]
    K = record["K"]
    keep = [k for k in range(K) if (keep_client is None or keep_client(k))]

    order: List[int] = []
    divergence = p

    for t, stored_src in enumerate(record["sources"]):
        if t >= len(record["per_client_round"]):
            break
        agg = _round_aggregate(record, keep, forget_stats, t)
        n_active = len(record["active"][t])
        if _argmin_source(agg, n_active) != stored_src:
            divergence = t
            break
        order.append(record["active"][t][stored_src])

    screened_rounds = len(order)

    if divergence < p:
        remaining = [v for v in range(p) if v not in order]
        if replay_data is not None and len(remaining) > 1:
            order = order + _replay_suffix(replay_data, order, remaining)
        else:
            order = order + remaining
    else:
        order = order + [v for v in range(p) if v not in order]

    agg0 = _round_aggregate(record, keep, forget_stats, 0)
    cov0 = np.array(agg0["cov"], dtype=float)
    n_tot = int(agg0["n"])
    if use_lasso:
        B = _estimate_B_lasso_from_cov(order, cov0, n_samples=n_tot,
                                       lmbda_type=lmbda_type)
    else:
        B = _estimate_B_from_cov(order, cov0)

    return {"order": order, "B": B,
            "first_replay_round": divergence,
            "screened_rounds": screened_rounds,
            "replayed_rounds": max(0, (p - 1) - screened_rounds),
            "fully_screened": divergence >= p}


def _replay_suffix(replay_data: List[np.ndarray], prefix: List[int],
                   remaining: List[int]) -> List[int]:
    """Re-run FedRCD-X from the divergent round, on the retained data only.

    The prefix is applied as real deflations so the replay starts from exactly
    the state the protocol would have been in.
    """
    X_local = [X.copy().astype(float) for X in replay_data]
    active = list(range(X_local[0].shape[1]))

    for v in prefix:
        idx = active.index(v)
        agg = aggregate_and_compute_cumulants(
            [compute_local_raw_stats(X, order=4) for X in X_local])
        cov = np.array(agg["cov"], dtype=float)
        betas = cov[:, idx] / max(float(cov[idx, idx]), 1e-15)
        rem = [j for j in range(len(active)) if j != idx]
        X_local = [np.column_stack([X[:, j] - betas[j] * X[:, idx] for j in rem])
                   for X in X_local]
        active = [active[j] for j in rem]

    suffix: List[int] = []
    while len(active) > 1:
        agg = aggregate_and_compute_cumulants(
            [compute_local_raw_stats(X, order=4) for X in X_local])
        idx = _argmin_source(agg, len(active))
        suffix.append(active[idx])
        cov = np.array(agg["cov"], dtype=float)
        betas = cov[:, idx] / max(float(cov[idx, idx]), 1e-15)
        rem = [j for j in range(len(active)) if j != idx]
        X_local = [np.column_stack([X[:, j] - betas[j] * X[:, idx] for j in rem])
                   for X in X_local]
        active = [active[j] for j in rem]
    suffix.append(active[0])
    return suffix
