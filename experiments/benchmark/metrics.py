"""Every metric the extended article reports, computed from one run's raw output.

Convention throughout: ``B[i, j] != 0`` means ``j -> i`` (LiNGAM).

A run stores its causal order and the estimated matrices, so anything added here
later is a recomputation over saved output, not a re-run.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from experiments.benchmark.sid import sid

EDGE_TOL = 1e-8


def _support(B: np.ndarray, tol: float = EDGE_TOL) -> np.ndarray:
    return np.abs(B) > tol


def reachability(B: np.ndarray) -> np.ndarray:
    """``R[i, j]`` is True when ``j`` is a proper ancestor of ``i``."""
    p = B.shape[0]
    R = _support(B)
    for _ in range(max(p.bit_length(), 1) + 1):
        R = R | ((R.astype(np.int64) @ R.astype(np.int64)) > 0)
    np.fill_diagonal(R, False)
    return R


def topological_order(B: np.ndarray, rng: np.random.Generator | None = None) -> list[int]:
    """Kahn's algorithm. Without `rng`, ties are broken by index (deterministic, fine for
    ground truth); with it, by a random pick. An estimator's order must use `rng`: the
    generators list variables in causal order, so index ties would be free wins."""
    A = _support(B)
    p = A.shape[0]
    indeg = A.sum(axis=1).astype(int)
    order, ready = [], sorted(np.flatnonzero(indeg == 0).tolist())
    while ready:
        v = ready.pop(0 if rng is None else int(rng.integers(len(ready))))
        order.append(v)
        for c in np.flatnonzero(A[:, v]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(int(c))
                ready.sort()
    if len(order) != p:
        raise ValueError("matrix is not acyclic")
    return order


# ── order metrics ───────────────────────────────────────────────────────────

def ancestral_accuracy(B_true: np.ndarray, order: list[int]) -> float:
    """Definition 3 of the PGM paper: share of ancestral pairs ordered correctly."""
    R = reachability(B_true)
    total = R.sum()
    if total == 0:
        return 1.0
    pos = np.empty(len(order), dtype=int)
    pos[np.asarray(order)] = np.arange(len(order))
    anc_before = pos[None, :] < pos[:, None]     # [i, j]: j placed before i
    return float((R & anc_before).sum() / total)


def order_f1(true_order: list[int], order: list[int]) -> float:
    """Strict pairwise F1 against one topological sort of the truth (PGM `f1_order`)."""
    p = len(order)
    pt = np.empty(p, dtype=int)
    pt[np.asarray(true_order)] = np.arange(p)
    pe = np.empty(p, dtype=int)
    pe[np.asarray(order)] = np.arange(p)
    iu = np.triu_indices(p, 1)
    agree = np.sign(pt[:, None] - pt[None, :])[iu] == np.sign(pe[:, None] - pe[None, :])[iu]
    return float(agree.mean()) if agree.size else 1.0


def order_metrics(B_true: np.ndarray, order: list[int], true_order: list[int],
                  cov: np.ndarray) -> dict:
    var = np.diag(cov)
    r2 = r2_scores(cov)
    pos = np.empty(len(order))
    pos[np.asarray(order)] = np.arange(len(order))
    return {
        "anc_acc": ancestral_accuracy(B_true, order),
        "f1_order": order_f1(true_order, order),
        "order_str": " ".join(map(str, order)),
        # Kendall tau between the returned position and the variance / R^2 rank:
        # how much of the answer is the variance ladder or the R^2 ladder.
        "ladder_agreement": float(kendalltau(pos, var)[0]),
        "r2_ladder_agreement": float(kendalltau(pos, r2)[0]),
        "sid_order": sid_of_order(B_true, order),
    }


def sid_of_order(B_true: np.ndarray, order: list[int]) -> int:
    """SID of the complete DAG consistent with `order`: the error the order alone causes."""
    p = len(order)
    full = np.zeros((p, p), dtype=bool)
    for k, child in enumerate(order):
        full[child, order[:k]] = True
    return sid_lingam(B_true, full)


# ── graph metrics ───────────────────────────────────────────────────────────

def sid_lingam(B_true: np.ndarray, B_est: np.ndarray) -> int:
    """SID with LiNGAM-convention matrices (`sid` wants ``A[u, v]`` for ``u -> v``)."""
    return sid(_support(B_true).T, _support(B_est).T)


def graph_metrics(B_true: np.ndarray, B_est: np.ndarray, prefix: str) -> dict:
    T, E = _support(B_true), _support(B_est)
    p = T.shape[0]
    n_true, n_est = int(T.sum()), int(E.sum())
    tp = int((T & E).sum())
    fp, fn = n_est - tp, n_true - tp

    T_sk, E_sk = T | T.T, E | E.T
    iu = np.triu_indices(p, 1)
    sk_t, sk_e = T_sk[iu], E_sk[iu]
    sk_tp = int((sk_t & sk_e).sum())

    extra = int((E & ~T_sk).sum())
    missing = int((T & ~E_sk).sum())
    reversed_ = int((T & E.T).sum())

    def f1(tp_, fp_, fn_):
        prec = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        rec = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    ep, er, ef = f1(tp, fp, fn)
    sp, sr, sf = f1(sk_tp, int(sk_e.sum()) - sk_tp, int(sk_t.sum()) - sk_tp)
    # Arc F1 restricted to correctly recovered adjacencies: orientation alone.
    oriented_right = tp
    adj_right = sk_tp
    orient_acc = oriented_right / adj_right if adj_right else np.nan

    shd_pgm = int((T ^ E).sum())                 # PGM definition: a reversal counts 2
    shd = extra + missing + reversed_            # usual definition: a reversal counts 1
    sid_val = sid_lingam(B_true, B_est)

    true_w = B_true[T]
    est_w = B_est[T]
    diff = B_est - B_true
    out = {
        "n_edges_true": n_true,
        "n_edges_est": n_est,
        "shd": shd,
        "shd_pgm": shd_pgm,
        "shd_extra": extra,
        "shd_missing": missing,
        "shd_reversed": reversed_,
        "shd_ratio": shd / n_true if n_true else np.nan,   # empty graph scores 1
        "edge_prec": ep, "edge_rec": er, "edge_f1": ef,
        "skel_prec": sp, "skel_rec": sr, "skel_f1": sf,
        "orient_acc": orient_acc,
        "sid": sid_val,
        "sid_norm": sid_val / (p * (p - 1)) if p > 1 else 0.0,
        "b_mse": float(np.mean(diff ** 2)),
        "b_frob_rel": float(np.linalg.norm(diff) / max(np.linalg.norm(B_true), 1e-12)),
        "b_mae_true_edges": float(np.mean(np.abs(est_w - true_w))) if n_true else np.nan,
        "sign_acc": float(np.mean(np.sign(est_w) == np.sign(true_w))) if n_true else np.nan,
    }
    return {f"{prefix}_{k}": v for k, v in out.items()}


# ── data diagnostics ────────────────────────────────────────────────────────

def r2_scores(cov: np.ndarray) -> np.ndarray:
    """R^2 of each variable regressed on all others, from the covariance alone."""
    prec = np.linalg.pinv(cov)
    return 1.0 - 1.0 / np.clip(np.diag(prec) * np.diag(cov), 1.0, None)


def sortability(B_true: np.ndarray, scores: np.ndarray) -> float:
    """Share of ancestral pairs along which `scores` increases (Reisach et al.)."""
    R = reachability(B_true)
    total = R.sum()
    if total == 0:
        return 0.5
    inc = scores[None, :] < scores[:, None]      # [i, j]: ancestor j scores lower
    ties = scores[None, :] == scores[:, None]
    return float(((R & inc).sum() + 0.5 * (R & ties).sum()) / total)


def data_diagnostics(B_true: np.ndarray, X: np.ndarray, cov: np.ndarray) -> dict:
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    skew = np.mean(Z ** 3, axis=0)
    kurt = np.mean(Z ** 4, axis=0) - 3.0
    return {
        "varsortability": sortability(B_true, np.diag(cov)),
        "r2_sortability": sortability(B_true, r2_scores(cov)),
        "mean_abs_skew": float(np.mean(np.abs(skew))),
        "mean_abs_exkurt": float(np.mean(np.abs(kurt))),
        "min_abs_exkurt": float(np.min(np.abs(kurt))),
    }
