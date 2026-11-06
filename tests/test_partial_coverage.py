"""
Correctness tests for vertical / hybrid cumulant aggregation.

The decisive test is the reduction: when every client observes every variable,
`aggregate_partial` must reproduce `aggregate_and_compute_cumulants` to
floating-point precision. If that fails, nothing downstream can be trusted.

Run: python tests/test_partial_coverage.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.cumulants import (                       # noqa: E402
    compute_local_raw_stats, aggregate_and_compute_cumulants,
)
from src.common.cumulants_partial import (               # noqa: E402
    compute_local_raw_stats_partial, aggregate_partial, coverage_report,
    vertical_partition,
)

KEYS = ["cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22"]


def _data(n=4000, p=6, seed=0):
    rng = np.random.default_rng(seed)
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            if rng.random() < 0.4:
                B[i, j] = rng.uniform(0.5, 1.5) * (1 if rng.random() < .5 else -1)
    E = np.stack([rng.exponential(1.0, n) - 1.0 for _ in range(p)])
    return np.linalg.solve(np.eye(p) - B, E).T, B


def test_reduces_to_horizontal(K=4, p=6, n=4000):
    """Full coverage on every client => identical to the horizontal path."""
    X, _ = _data(n=n, p=p)
    shards = np.array_split(X, K)

    ref = aggregate_and_compute_cumulants(
        [compute_local_raw_stats(s, order=4) for s in shards])
    got = aggregate_partial(
        [compute_local_raw_stats_partial(s, list(range(p)), p, order=4, group=g)
         for g, s in enumerate(shards)], p)

    ok = True
    for k in KEYS:
        a, b = np.asarray(ref[k], float), np.asarray(got[k], float)
        err = np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12)
        status = "ok " if err < 1e-9 else "FAIL"
        if err >= 1e-9:
            ok = False
        print(f"  {status} {k:5s} max rel err = {err:.2e}")
    return ok


def test_vertical_recovers_horizontal(p=6, n=20000):
    """Vertical split of the SAME rows: every client sees all rows but only
    some columns. Pair statistics are then backed by identical row sets, so the
    estimates must again match the centralised ones."""
    X, _ = _data(n=n, p=p)
    rng = np.random.default_rng(1)
    sets = vertical_partition(p, K=6, block=4, rng=rng)

    ref = aggregate_and_compute_cumulants(
        [compute_local_raw_stats(X, order=4)])
    parts = [compute_local_raw_stats_partial(X[:, s], s, p, order=4, group=0)
             for s in sets]
    got = aggregate_partial(parts, p)

    cov = coverage_report(parts, p)
    print(f"  coverage: all pairs = {cov['fully_covered']}, "
          f"min N_ij = {cov['min_pair_count']:.0f}")

    ok = cov["fully_covered"]
    for k in KEYS:
        a, b = np.asarray(ref[k], float), np.asarray(got[k], float)
        err = np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12)
        status = "ok " if err < 1e-9 else "FAIL"
        if err >= 1e-9:
            ok = False
        print(f"  {status} {k:5s} max rel err = {err:.2e}")
    return ok


def test_gap_is_reported(p=6):
    """A pair nobody covers must be reported, not silently zeroed."""
    X, _ = _data(n=1000, p=p)
    sets = [[0, 1, 2], [2, 3], [3, 4, 5]]        # (0,5) never co-observed
    parts = [compute_local_raw_stats_partial(X[:, s], s, p, order=4, group=0)
             for s in sets]
    cov = coverage_report(parts, p)
    print(f"  missing pairs detected: {cov['n_missing_pairs']} "
          f"e.g. {cov['missing'][:3]}")
    return (not cov["fully_covered"]) and (0, 5) in cov["missing"]


if __name__ == "__main__":
    print("1) reduction to the horizontal aggregator")
    r1 = test_reduces_to_horizontal()
    print("\n2) vertical split, same rows")
    r2 = test_vertical_recovers_horizontal()
    print("\n3) uncovered pair is reported")
    r3 = test_gap_is_reported()
    print(f"\nRESULT: {'ALL PASSED' if (r1 and r2 and r3) else 'FAILURES PRESENT'}")
    sys.exit(0 if (r1 and r2 and r3) else 1)
