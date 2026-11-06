"""
Cumulant aggregation under partial variable coverage (vertical / hybrid FL).

`src.common.cumulants` carries a single scalar sample count and states in its
docstring that all clients must share the same variable set. That is horizontal
federation only, so the vertical and hybrid claims of the PGM paper cannot be
run from it. This module supplies what those regimes need.

What changes
------------
When client k observes only the subset S_k, the sample count backing an entry
depends on the entry:

    N_i   = sum of n_k over clients holding i
    N_ij  = sum of n_k over clients holding both i and j

and every centring must use a mean taken over *the same rows* as the moment it
centres: a marginal mean over N_i rows cannot centre a cross moment over N_ij
rows. So each client also reports, per pair it covers, the partial sums of
x_i, x_i^2, x_i^3, x_i^4 restricted to that pair's rows (`P1`..`P4`). Within one
client this is pure bookkeeping -- all its observed columns share all its rows --
but across clients the restriction is what makes the estimates correct.

Cost: four extra p x p arrays, so the payload stays O(p^2).

Every formula mirrors `_compute_3rd_order` / `_compute_4th_order` of the
horizontal module entry by entry, with N replaced by N_ij and the marginal
sums replaced by their pair-restricted counterparts. Consequently, when every
client observes every variable this module reproduces the horizontal result
exactly; `tests/test_partial_coverage.py` asserts that.

Identifiability needs every pair covered by at least one client
(Chen et al., 2026); `coverage_report` checks it and names the gaps.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Client side
# ═══════════════════════════════════════════════════════════════════════

def compute_local_raw_stats_partial(X: np.ndarray,
                                    observed: Sequence[int],
                                    p: int,
                                    order: int = 4,
                                    group: int = 0) -> Dict[str, np.ndarray]:
    """Raw moment sums from a client holding only `observed` columns.

    Parameters
    ----------
    X : (n_k, len(observed)) local data, columns ordered as `observed`.
    observed : global indices of the columns this client holds.
    p : number of variables in the federation.
    group : identifier of the ROW block this client holds. Clients sharing a
        group hold the same samples, so their contributions must not be added
        (that would count each observation once per client). Clients in
        different groups hold disjoint samples and are summed. Pure horizontal
        federation gives every client its own group; pure vertical puts every
        client in one group; hybrid has several clients per group.

    All arrays come back at full p x p (or p) size with zeros outside the
    observed block, so the server adds them without knowing who holds what.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    obs = list(observed)
    if X.shape[1] != len(obs):
        raise ValueError(f"X has {X.shape[1]} columns, observed lists {len(obs)}")
    idx = np.array(obs, dtype=int)
    m = len(obs)

    X2 = X ** 2
    X3 = X2 * X
    X4 = X2 * X2

    def emb1(v):
        out = np.zeros(p)
        out[idx] = v
        return out

    def emb2(M):
        out = np.zeros((p, p))
        out[np.ix_(idx, idx)] = M
        return out

    ones = np.ones((m, m))
    s1, s2, s3, s4 = X.sum(0), X2.sum(0), X3.sum(0), X4.sum(0)

    cnt1 = np.zeros(p); cnt1[idx] = n
    cnt2 = np.zeros((p, p)); cnt2[np.ix_(idx, idx)] = n

    stats = {
        "group": group,
        "cnt1": cnt1,
        "cnt2": cnt2,
        "sum_x":    emb1(s1),
        "sum_xx":   emb2(X.T @ X),
        "sum_x3":   emb1(s3),
        # P<k>[i, j] = sum of x_i^k over the rows where i and j are both held
        "P1": emb2(s1[:, None] * ones),
        "P2": emb2(s2[:, None] * ones),
        "P3": emb2(s3[:, None] * ones),
        "sum_x1y2": emb2(X.T @ X2),
        "sum_x2y1": emb2(X2.T @ X),
    }
    if order >= 4:
        stats.update({
            "sum_x4":   emb1(s4),
            "P4": emb2(s4[:, None] * ones),
            "sum_x1y3": emb2(X.T @ X3),
            "sum_x3y1": emb2(X3.T @ X),
            "sum_x2y2": emb2(X2.T @ X2),
        })
    return stats


# ═══════════════════════════════════════════════════════════════════════
# Server side
# ═══════════════════════════════════════════════════════════════════════

def coverage_report(client_stats_list: List[Dict], p: int) -> Dict:
    """Which pairs are covered, by how many samples, and by how many clients.

    The client count is a privacy diagnostic, not just a bookkeeping one. A pair
    covered by exactly ONE client is exposed outright: the aggregate for that
    pair *is* that client's contribution, so the server reads its bivariate
    moments with no inference at all. That is a stronger leak than the
    before-and-after subtraction attack, which at least needs two observations.
    Finer vertical splits cover more pairs singly, so coverage efficiency and
    privacy pull in opposite directions.
    """
    cnt2 = np.zeros((p, p))
    holders = np.zeros((p, p), dtype=int)
    for s in client_stats_list:
        cnt2 = cnt2 + s["cnt2"]
        holders = holders + (s["cnt2"] > 0).astype(int)
    off = ~np.eye(p, dtype=bool)
    missing = [(int(a), int(b)) for a, b in np.argwhere((cnt2 == 0) & off) if a < b]
    single = [(int(a), int(b)) for a, b in np.argwhere((holders == 1) & off) if a < b]
    n_off = int(off.sum() // 2)
    return {
        "N_ij": cnt2,
        "holders": holders,
        "n_missing_pairs": len(missing),
        "missing": missing,
        "fully_covered": not missing,
        "min_pair_count": float(cnt2[off].min()) if p > 1 else 0.0,
        # privacy diagnostics
        "n_singly_covered": len(single),
        "singly_covered": single,
        "frac_singly_covered": len(single) / n_off if n_off else 0.0,
        "min_holders": int(holders[off].min()) if p > 1 else 0,
    }


def aggregate_partial(client_stats_list: List[Dict],
                      p: Optional[int] = None) -> Dict:
    """Global cumulants from partially-overlapping clients."""
    if not client_stats_list:
        return {}
    if p is None:
        p = len(client_stats_list[0]["cnt1"])

    # Within a sample group every client holds the same rows, so an entry must
    # be taken from ONE covering client per group -- adding them would count
    # each observation once per client. Across groups the rows are disjoint and
    # the sums are the intended federated aggregation.
    keys = [k for k in client_stats_list[0] if k != "group"]
    tot = {k: np.zeros_like(np.asarray(client_stats_list[0][k], dtype=float))
           for k in keys}

    groups: Dict[int, List[Dict]] = {}
    for s in client_stats_list:
        groups.setdefault(int(s.get("group", 0)), []).append(s)

    for members in groups.values():
        # per-group accumulator, plus a mask of entries already claimed
        claimed1 = np.zeros(p, dtype=bool)
        claimed2 = np.zeros((p, p), dtype=bool)
        for s in members:
            has1 = s["cnt1"] > 0
            has2 = s["cnt2"] > 0
            take1 = has1 & ~claimed1
            take2 = has2 & ~claimed2
            for k in keys:
                arr = np.asarray(s[k], dtype=float)
                if arr.ndim == 1:
                    tot[k] = tot[k] + np.where(take1, arr, 0.0)
                else:
                    tot[k] = tot[k] + np.where(take2, arr, 0.0)
            claimed1 |= take1
            claimed2 |= take2

    N1 = tot["cnt1"]
    N2 = tot["cnt2"].copy()
    np.fill_diagonal(N2, N1)
    s1 = np.where(N1 > 0, N1, 1.0)
    s2 = np.where(N2 > 0, N2, 1.0)

    Sxx = tot["sum_xx"]
    mu = tot["sum_x"] / s1                       # marginal mean, over N_i
    A = tot["P1"] / s2                           # E[x_i] on pair (i, j) rows
    B = A.T                                      # E[x_j] on the same rows
    P2, P3 = tot["P2"], tot["P3"]

    # ── covariance: off-diagonal over N_ij, diagonal over N_i ────────
    M11 = Sxx - N2 * A * B
    cov = M11 / np.maximum(N2 - 1.0, 1.0)
    diag_var = (np.diag(Sxx) - N1 * mu ** 2) / np.maximum(N1 - 1.0, 1.0)
    np.fill_diagonal(cov, diag_var)
    cov = 0.5 * (cov + cov.T)

    # ── third order ──────────────────────────────────────────────────
    f3_1 = np.where(N1 >= 3, N1 / np.maximum((N1 - 1) * (N1 - 2), 1.0), 0.0)
    c3 = f3_1 * (tot["sum_x3"] - 3 * mu * np.diag(Sxx) + 2 * N1 * mu ** 3)

    f3_2 = np.where(N2 >= 3, N2 / np.maximum((N2 - 1) * (N2 - 2), 1.0), 0.0)
    M12 = tot["sum_x1y2"] - 2 * B * Sxx - A * P2.T + 2 * N2 * A * B ** 2
    M21 = tot["sum_x2y1"] - 2 * A * Sxx - B * P2 + 2 * N2 * B * A ** 2
    c12, c21 = f3_2 * M12, f3_2 * M21
    # On the diagonal every cross cumulant collapses to the marginal, matching
    # the horizontal aggregator.
    np.fill_diagonal(c12, 0.0)
    np.fill_diagonal(c21, 0.0)
    c12 += np.diag(c3)
    c21 += np.diag(c3)

    out = {"n": float(N1.max()) if p else 0.0, "N1": N1, "N2": N2,
           "cov": cov, "c3": c3, "c12": c12, "c21": c21,
           "coverage": coverage_report(client_stats_list, p)}

    if "sum_x4" not in tot:
        return out

    # ── fourth order ─────────────────────────────────────────────────
    M2d = np.diag(Sxx) - N1 * mu ** 2
    D4_1 = np.maximum((N1 - 1) * (N1 - 2) * (N1 - 3), 1.0)
    M4 = (tot["sum_x4"] - 4 * mu * tot["sum_x3"]
          + 6 * mu ** 2 * np.diag(Sxx) - 3 * N1 * mu ** 4)
    c4 = np.where(N1 >= 4,
                  (N1 * (N1 + 1) * M4 - 3 * (N1 - 1) * M2d ** 2) / D4_1, 0.0)

    P4 = tot["P4"]
    M13 = (tot["sum_x1y3"] - A * P3.T - 3 * B * tot["sum_x1y2"]
           + 3 * A * B * P2.T + 3 * B ** 2 * Sxx - 3 * N2 * A * B ** 3)
    M31 = (tot["sum_x3y1"] - B * P3 - 3 * A * tot["sum_x2y1"]
           + 3 * A * B * P2 + 3 * A ** 2 * Sxx - 3 * N2 * B * A ** 3)
    M22 = (tot["sum_x2y2"] - 2 * B * tot["sum_x2y1"] - 2 * A * tot["sum_x1y2"]
           + B ** 2 * P2 + A ** 2 * P2.T + 4 * A * B * Sxx
           - 3 * N2 * A ** 2 * B ** 2)
    M20 = P2 - N2 * A ** 2                       # centred, on pair rows
    M02 = P2.T - N2 * B ** 2

    a4 = N2 * (N2 + 1)
    b4 = N2 - 1
    D4 = np.maximum((N2 - 1) * (N2 - 2) * (N2 - 3), 1.0)
    ok = N2 >= 4
    c13 = np.where(ok, (a4 * M13 - 3 * b4 * M11 * M02) / D4, 0.0)
    c31 = np.where(ok, (a4 * M31 - 3 * b4 * M11 * M20) / D4, 0.0)
    c22 = np.where(ok, (a4 * M22 - b4 * (M20 * M02 + 2 * M11 ** 2)) / D4, 0.0)
    for M in (c13, c31, c22):
        np.fill_diagonal(M, 0.0)
    c13 += np.diag(c4); c31 += np.diag(c4); c22 += np.diag(c4)

    # P4 participates only through the marginal path; kept for completeness
    _ = P4
    out.update({"c4": c4, "c13": c13, "c31": c31, "c22": c22})
    return out


# ═══════════════════════════════════════════════════════════════════════
# Partition builders
# ═══════════════════════════════════════════════════════════════════════

def vertical_partition(p: int, K: int, block: int, rng) -> List[List[int]]:
    """K clients of `block` variables each, resampled until every pair is
    covered by at least one client."""
    if block < 2:
        raise ValueError("block must be >= 2 for any pair to be covered")
    for _ in range(50_000):
        sets = [sorted(rng.choice(p, size=block, replace=False).tolist())
                for _ in range(K)]
        seen = np.eye(p, dtype=bool)
        for s in sets:
            seen[np.ix_(s, s)] = True
        if seen.all():
            return sets
    raise RuntimeError(
        f"could not cover every pair with K={K} clients of size {block} "
        f"(p={p}): need K*C(block,2) comfortably above C(p,2)")
