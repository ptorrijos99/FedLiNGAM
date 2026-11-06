"""The methods the extended article compares, behind one signature.

Each method takes the list of client arrays and a seed and returns a `Result`.
Every method is then scored on the same three reconstructions of B from the
pooled covariance (fixed-lambda adaptive Lasso, BIC adaptive Lasso, OLS), so
differences in the graph come from the order and not from the regression.
Methods that estimate B themselves also keep their native matrix.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ADMM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "notears-admm")

from src.common.cumulants import compute_local_raw_stats, aggregate_and_compute_cumulants  # noqa: E402
from src.common.solvers import _estimate_B_from_cov, _estimate_B_lasso_from_cov           # noqa: E402
from src.fedishc.core import ishc, ishc_4th, hc as fedhc_solver                           # noqa: E402
from src.fedrcd.core import fedrcd, fedrcd_noniterative, fedrcd_exact                     # noqa: E402
from src.fedrcd.residual import fedrcd_ni_score                                         # noqa: E402

from experiments.benchmark.metrics import r2_scores, topological_order                        # noqa: E402

OLS_THRESHOLD = 0.05


@dataclass
class Result:
    order: list[int]
    B_native: np.ndarray | None = None
    t_client: float = np.nan          # seconds per client, summed over rounds
    t_server: float = np.nan
    floats: float = np.nan            # total floats on the wire, all clients, all rounds
    rounds: float = np.nan
    extra: dict = field(default_factory=dict)


def pooled(clients):
    return np.vstack(clients)


def pooled_cov(clients):
    X = pooled(clients)
    Xc = X - X.mean(0)
    return Xc.T @ Xc / max(len(X) - 1, 1)


def reconstructions(order, cov, n):
    """The three uniform B estimates every method is scored on."""
    return {
        # Pruning thresholds as in the PGM benchmark: 0.05 for fixed lambda, 1e-5 for BIC.
        "lfix": _estimate_B_lasso_from_cov(order, cov, n_samples=n, lmbda_type="fixed", prune_thres=0.05),
        "lbic": _estimate_B_lasso_from_cov(order, cov, n_samples=n, lmbda_type="bic", prune_thres=1e-5),
        "ols": _estimate_B_from_cov(order, cov, prune_thres=OLS_THRESHOLD),
    }


# ── cumulant family ─────────────────────────────────────────────────────────

def _payload(p, K, order):
    """Floats one round of a cumulant protocol puts on the wire."""
    per_client = p + p * p + 3 * p * p
    if order >= 4:
        per_client += 4 * p * p
    return per_client * K


def _stats(clients, order):
    t0 = time.perf_counter()
    local = [compute_local_raw_stats(c, order=order) for c in clients]
    t_client = (time.perf_counter() - t0) / len(clients)
    return aggregate_and_compute_cumulants(local), t_client


def _oneshot(order_needed, solve):
    def run(clients, seed):
        agg, tc = _stats(clients, order_needed)
        p, K = clients[0].shape[1], len(clients)
        t0 = time.perf_counter()
        order = solve(agg)
        return Result(order=list(map(int, order)), t_client=tc,
                      t_server=time.perf_counter() - t0,
                      floats=_payload(p, K, order_needed), rounds=1)
    return run


def _legacy3(agg):
    return {k: agg[k] for k in ("c3", "c12", "c21", "n")}


def _stats4(agg):
    return {k: agg[k] for k in ("c3", "c12", "c21", "c4", "c13", "c31", "c22", "n")}


def fedrcd_x(clients, seed):
    p, K = clients[0].shape[1], len(clients)
    _, tc = _stats(clients, 4)                 # one round's client cost, for scale
    t0 = time.perf_counter()
    order, _ = fedrcd_exact(clients, use_lasso=True, lmbda_type="fixed")
    total = time.perf_counter() - t0
    floats = sum(_payload(q, K, 4) for q in range(p, 1, -1))
    return Result(order=list(map(int, order)), t_client=tc * (p - 1),
                  t_server=total, floats=floats, rounds=p - 1,
                  extra={"t_server_includes_client_deflation": True})


# ── centralised baselines ───────────────────────────────────────────────────

def _central(make):
    def run(clients, seed):
        X = pooled(clients)
        t0 = time.perf_counter()
        model = make(seed).fit(X)
        return Result(order=list(map(int, model.causal_order_)),
                      B_native=np.asarray(model.adjacency_matrix_, dtype=float),
                      t_server=time.perf_counter() - t0, rounds=0)
    return run


def _hc(method):
    from src.hclingam.hclingam import HCLiNGAM
    return lambda seed: HCLiNGAM(method=method)


def _lingam(name):
    def make(seed):
        import lingam
        return lingam.DirectLiNGAM() if name == "direct" else lingam.ICALiNGAM(random_state=seed)
    return make


# ── null models ─────────────────────────────────────────────────────────────

def varsort(clients, seed):
    """Order by marginal variance, ascending. Reads only the ladder."""
    t0 = time.perf_counter()
    cov = pooled_cov(clients)
    p, K = clients[0].shape[1], len(clients)
    # After standardisation every variance is one up to rounding; break those ties at
    # random, or the column order (causal in most generators) decides the sort.
    var = np.round(np.diag(cov), 10)
    order = np.lexsort((np.random.default_rng(seed).random(p), var))
    return Result(order=order.tolist(), t_server=time.perf_counter() - t0,
                  floats=(p + p * p) * K, rounds=1)


def r2sort(clients, seed):
    """Order by R^2 on all other variables, ascending (Reisach et al., 2023)."""
    t0 = time.perf_counter()
    order = np.argsort(r2_scores(pooled_cov(clients)), kind="stable")
    p, K = clients[0].shape[1], len(clients)
    return Result(order=order.tolist(), t_server=time.perf_counter() - t0,
                  floats=(p + p * p) * K, rounds=1)


def random_order(clients, seed):
    p = clients[0].shape[1]
    return Result(order=np.random.default_rng(seed).permutation(p).tolist(), rounds=0)


# ── NOTEARS-ADMM (Ng & Zhang, AISTATS 2022) ─────────────────────────────────

def _is_dag(W):
    A = (np.abs(W) > 0).astype(float)
    p = A.shape[0]
    M = np.eye(p)
    for _ in range(p):
        M = M @ A
        if np.trace(M) > 0:
            return False
    return True


def _threshold_till_dag(W):
    W = W.copy()
    if _is_dag(W):
        return W
    idx = np.argwhere(W != 0)
    for i, j in sorted(idx.tolist(), key=lambda ij: abs(W[ij[0], ij[1]])):
        W[i, j] = 0.0
        if _is_dag(W):
            break
    return W


def notears_admm(clients, seed, lambda1=0.01, max_iter=200, rho_max=1e16, w_threshold=0.3,
                 budget_s=1800.0):
    """The authors' linear ADMM loop, unchanged except that it counts rounds.

    Clients must hold equal sample sizes, so each is trimmed to the smallest.
    A wall-clock budget stops a run that would otherwise block a worker; hitting
    it is recorded, not hidden.
    """
    if ADMM_DIR not in sys.path:
        sys.path.insert(0, ADMM_DIR)
    import torch
    from notears_admm import linear_admm as la

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.double)
    torch.manual_seed(seed)
    np.random.seed(seed)

    m = min(len(c) for c in clients)
    Xs = np.stack([c[:m] for c in clients])
    K, n, d = Xs.shape

    t0 = time.perf_counter()
    local_models = [la.LinearModel(d) for _ in range(K)]
    global_model = la.LinearModel(d)
    rho1, rho2, alpha, h = 0.001, 0.001, 0.0, np.inf
    betas = np.zeros((K, d, d))
    Xs = Xs - np.mean(Xs, axis=1, keepdims=True)
    covs_emp = la.compute_empirical_covs(Xs)
    rounds = 0
    for _ in range(max_iter):
        for k in range(K):
            la.solve_local_subproblem(covs_emp[k], local_models[k], global_model, betas[k], rho2)
        la.solve_global_subproblem(local_models, global_model, lambda1, alpha, betas, rho1, rho2)
        with torch.no_grad():
            h = global_model.h_func().item()
        alpha, betas, rho1, rho2 = la.update_admm_params(alpha, betas, rho1, rho2, local_models,
                                                         global_model, h, rho_max)
        rounds += 1
        if rho1 >= rho_max and rho2 >= rho_max:
            break
        if time.perf_counter() - t0 > budget_s:
            break
    W = global_model.adj()
    elapsed = time.perf_counter() - t0

    W = W.copy()
    W[np.abs(W) <= w_threshold] = 0.0
    W = _threshold_till_dag(W)
    B = W.T                                     # their W[i, j] is i -> j
    # One uplink of d*d local weights per client and one downlink of the global
    # matrix per client, every round.
    # Variables unordered by the estimated DAG get a random relative order; by index
    # they would inherit the generator's causal order.
    order = topological_order(B, rng=np.random.default_rng(seed))
    return Result(order=order, B_native=B, t_server=elapsed,
                  floats=rounds * 2 * K * d * d, rounds=rounds,
                  extra={"admm_final_h": h,
                         "admm_hit_budget": bool(time.perf_counter() - t0 > budget_s)})


# ── registry ────────────────────────────────────────────────────────────────

REGISTRY = {
    # FedRCD
    "FedRCD-NI": (_oneshot(4, lambda a: fedrcd_noniterative(a, adaptive=False, use_lasso=True, lmbda_type="fixed")[0]), "federated"),
    "FedRCD-Def": (_oneshot(4, lambda a: fedrcd(a, adaptive=False, use_lasso=True, lmbda_type="fixed")[0]), "federated"),
    "FedRCD-X": (fedrcd_x, "federated"),
    # federated cumulant baselines and the ablation of Section 3.2
    "FedISHC": (_oneshot(3, lambda a: ishc(_legacy3(a))[0]), "federated"),
    "FedHC": (_oneshot(3, lambda a: fedhc_solver(_legacy3(a))[0]), "federated"),
    # The PGM figures were produced with this form: argmin over a shrinking active
    # set with no deflation. It is not identical to the single-pass sort above.
    "FedHC-iter": (_oneshot(3, lambda a: fedhc_solver(_legacy3(a), iterative=True)[0]), "federated"),
    "FedISHC-4": (_oneshot(4, lambda a: ishc_4th(_stats4(a))[0]), "federated"),
    # the residual score with and without scale (src/fedrcd/residual.py), Section 5.2
    "FedRCD-NI-M": (_oneshot(4, lambda a: fedrcd_ni_score(a, "m31")[0]), "federated"),
    "FedRCD-NI-Mn": (_oneshot(4, lambda a: fedrcd_ni_score(a, "m31n")[0]), "federated"),
    # federated, different paradigm
    "NOTEARS-ADMM": (notears_admm, "federated"),
    # centralised
    "HC": (_central(_hc("hc")), "central"),
    "HC-LiNGAM": (_central(_hc("hc-lingam")), "central"),
    "DirectLiNGAM": (_central(_lingam("direct")), "central"),
    "ICA-LiNGAM": (_central(_lingam("ica")), "central"),
    # reference orders
    "VarSort": (varsort, "null"),
    "R2Sort": (r2sort, "null"),
    "Random": (random_order, "null"),
}

CUMULANT = ["FedRCD-NI", "FedRCD-Def", "FedRCD-X", "FedISHC", "FedHC", "FedHC-iter", "FedISHC-4"]
CENTRAL = ["HC", "HC-LiNGAM", "DirectLiNGAM", "ICA-LiNGAM"]
NULLS = ["VarSort", "R2Sort", "Random"]
INSTRUMENT = ["FedRCD-NI-M", "FedRCD-NI-Mn"]
ALL = CUMULANT + ["NOTEARS-ADMM"] + CENTRAL + NULLS
