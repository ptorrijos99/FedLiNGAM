"""
Exact unlearning and privacy are one property, read twice.

The PGM paper sells exact unlearning as a feature: the server subtracts a
client's contribution from the aggregate and recovers exactly the statistics it
would have had if that client had never participated. The same sentence,
rearranged, is an attack: a server that can subtract a client's contribution can
ISOLATE it, and the difference between the aggregate before and after that
client left is precisely the client's own sufficient statistics.

This makes the point concrete. An honest-but-curious server observes the
aggregate at two moments -- before and after a client departs -- and reconstructs
that client's private summaries from the difference. We measure how well, with
and without the DP noise of `dp_one_round.py`, and we quantify what the reconstruction
actually reveals: per-variable means, variances, skewness and kurtosis of a
single institution's patients.

Usage: python experiments/deployment/reconstruction.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import (   # noqa: E402
    _split_data, PGMSyntheticDataGenerator,
)
from src.common.cumulants import compute_local_raw_stats   # noqa: E402
from src.common import dp                                  # noqa: E402

OUT = "experiments/deployment/results/reconstruction.csv"
EPS_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float("inf")]
DELTA = 1e-5
CLIP = 4.0


def _sum_stats(stats_list):
    out = {k: (np.zeros_like(np.asarray(v, dtype=float))
               if isinstance(v, np.ndarray) else 0)
           for k, v in stats_list[0].items()}
    for s in stats_list:
        for k in out:
            out[k] = out[k] + s[k]
    return out


def _summaries(stats):
    """Per-variable mean, variance, skewness, kurtosis from raw-moment sums."""
    n = stats["n"]
    m1 = stats["sum_x"] / n
    m2 = np.diag(stats["sum_xx"]) / n
    m3 = stats["sum_x3"] / n
    m4 = stats["sum_x4"] / n
    var = np.maximum(m2 - m1 ** 2, 1e-12)
    sd = np.sqrt(var)
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / sd ** 3
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / var ** 2 - 3.0
    return np.concatenate([m1, var, skew, kurt])


def run(seeds=15, p=20, n_per_client=2000, K=10, dist="mixed", density=0.2):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []

    sens = dp.moment_sensitivity(p, CLIP, order=4)

    for seed in range(seeds):
        gen = PGMSyntheticDataGenerator(seed=seed)
        X, _, _ = gen.generate(n_samples=n_per_client * K, p=p,
                               graph_density=density, dist_type=dist)
        data = _split_data(X, K, np.random.default_rng(seed))
        Xall = np.vstack(data)
        mu, sd = Xall.mean(0), Xall.std(0)
        sd[sd == 0] = 1.0
        clipped = [dp.clip((Xk - mu) / sd, CLIP) for Xk in data]

        stats = [compute_local_raw_stats(Xk, order=4) for Xk in clipped]
        victim = 3
        truth = _summaries(stats[victim])

        agg_all = _sum_stats(stats)
        agg_wo = _sum_stats([s for i, s in enumerate(stats) if i != victim])

        for eps in EPS_GRID:
            rng = np.random.default_rng(50_000 + seed)
            rho = dp.rho_from_eps_delta(eps, DELTA)
            # Two releases are observed, so the budget splits over both.
            sigma = dp.gaussian_sigma(sens, rho / 2 if np.isfinite(rho) else rho)

            a1 = dp.privatise_moment_stats(agg_all, sigma, rng)
            a2 = dp.privatise_moment_stats(agg_wo, sigma, rng)
            diff = {k: (a1[k] - a2[k]) for k in a1}
            diff["n"] = stats[victim]["n"]        # participation count is public

            try:
                est = _summaries(diff)
            except Exception:
                continue

            denom = np.where(np.abs(truth) > 1e-9, np.abs(truth), 1.0)
            rel = np.abs(est - truth) / denom
            blocks = ["mean", "variance", "skewness", "kurtosis"]
            for b, sl in zip(blocks, [slice(0, p), slice(p, 2 * p),
                                      slice(2 * p, 3 * p), slice(3 * p, 4 * p)]):
                rows.append(dict(seed=seed, eps=eps, quantity=b,
                                 median_rel_err=float(np.median(rel[sl])),
                                 sigma=sigma))
        print(f"  seed {seed} done")
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\n=== Reconstructing one client's private summaries from two aggregates ===")
    print("median relative error, victim = 1 of %d clients, n_k = %d\n" % (K, n_per_client))
    print(df.pivot_table(index="quantity", columns="eps",
                         values="median_rel_err", aggfunc="median")
            .reindex(["mean", "variance", "skewness", "kurtosis"])
            .round(4).to_string())
    print("\nAn error near 0 means the server recovered that statistic exactly.")
    print("The eps=inf column is the protocol exactly as published at PGM.")


if __name__ == "__main__":
    run()
