"""
Differential privacy on the one-round cumulant protocols.

The cumulant protocols release raw moment sums once. Per-record sensitivity of a
fourth-order entry is C^4, so the Gaussian noise scales as the fourth power of the
clipping bound C. Accounting is zCDP, converted to (eps, delta = 1e-5).

Usage: python experiments/deployment/dp_one_round.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import (   # noqa: E402
    ancestral_accuracy, _split_data, PGMSyntheticDataGenerator,
)
from src.common.cumulants import (          # noqa: E402
    compute_local_raw_stats, aggregate_and_compute_cumulants,
)
from src.common import dp                   # noqa: E402
from src.fedrcd.core import fedrcd_noniterative, fedrcd                     # noqa: E402
from src.fedishc.core import hc as fedhc_solver                             # noqa: E402

OUT = "experiments/deployment/results/dp_one_round.csv"
EPS_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float("inf")]
DELTA = 1e-5
CLIP = 4.0


def clip_bound(eps):
    """Clipping only bounds the sensitivity, so the noise-free reference skips it."""
    return CLIP if np.isfinite(eps) else np.inf


# ─────────────────────────────────────────────────────────────────────────
# DP variant of the one-round pipelines
# ─────────────────────────────────────────────────────────────────────────

def dp_cumulant_order(data, eps, rng, solver, order=4):
    """One-shot cumulant protocol under a single Gaussian release."""
    p = data[0].shape[1]
    rho = dp.rho_from_eps_delta(eps, DELTA)
    sens = dp.moment_sensitivity(p, CLIP, order=order)
    sigma = dp.gaussian_sigma(sens, rho)

    agg_raw = _aggregate_clipped(data, order, clip_bound(eps))
    noisy = dp.symmetrise(dp.privatise_moment_stats(agg_raw, sigma, rng))
    agg = aggregate_and_compute_cumulants([noisy])
    if order < 4:
        legacy = {k: agg[k] for k in ("c3", "c12", "c21", "n")}
        return fedhc_solver(legacy)[0], sigma
    return solver(agg, adaptive=False, use_lasso=True, lmbda_type="fixed")[0], sigma


def _aggregate_clipped(data, order, C):
    """Scale by the global std, clip, then form the per-client moment sums."""
    Xall = np.vstack(data)
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd == 0] = 1.0
    stats = [compute_local_raw_stats(dp.clip((Xk - mu) / sd, C), order=order)
             for Xk in data]
    out = {k: (np.zeros_like(np.asarray(v, dtype=float))
               if isinstance(v, np.ndarray) else 0)
           for k, v in stats[0].items()}
    for s in stats:
        for k in out:
            out[k] = out[k] + s[k]
    return out


# ─────────────────────────────────────────────────────────────────────────

def run(seeds=10, p=20, n_per_client=2000, K=10, dist="mixed", density=0.2):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    if os.path.exists(OUT):
        rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["eps"], r["seed"], r["method"]) for r in rows}

    for eps in EPS_GRID:
        for seed in range(seeds):
            gen = PGMSyntheticDataGenerator(seed=seed)
            X, B_true, _ = gen.generate(n_samples=n_per_client * K, p=p,
                                        graph_density=density, dist_type=dist)
            data = _split_data(X, K, np.random.default_rng(seed))
            rng = np.random.default_rng(90_000 + seed)

            specs = [
                ("FedRCD-NI", lambda: dp_cumulant_order(data, eps, rng,
                                                        fedrcd_noniterative, 4)),
                ("FedRCD-Def", lambda: dp_cumulant_order(data, eps, rng,
                                                         fedrcd, 4)),
                ("FedHC", lambda: dp_cumulant_order(data, eps, rng, None, 3)),
            ]
            for name, fn in specs:
                if (eps, seed, name) in done:
                    continue
                try:
                    o, sigma = fn()
                    acc = ancestral_accuracy(B_true, o)
                except Exception as e:
                    print(f"    {name} eps={eps} seed={seed}: {e}")
                    acc, sigma = np.nan, np.nan
                rows.append(dict(eps=eps, seed=seed, method=name,
                                 AncAcc=acc, sigma=sigma))
                done.add((eps, seed, name))
            pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  eps={eps} done")

    df = pd.DataFrame(rows)
    print("\n=== AncAcc vs privacy budget (delta=%g, clip C=%g, p=%d, K=%d, N=%d) ==="
          % (DELTA, CLIP, p, K, n_per_client * K))
    print(df.pivot_table(index="method", columns="eps", values="AncAcc",
                         aggfunc="mean").round(3).to_string())
    print("\n=== Noise scale actually applied (sigma on the transmitted sums) ===")
    print(df.pivot_table(index="method", columns="eps", values="sigma",
                         aggfunc="mean").round(1).to_string())
    print("\nAnalytic sensitivities at C=%g, p=%d:" % (CLIP, p))
    print("  cumulant 4th-order, one release : %.1f"
          % dp.moment_sensitivity(p, CLIP, 4))
    print("  cumulant 3rd-order, one release : %.1f"
          % dp.moment_sensitivity(p, CLIP, 3))


if __name__ == "__main__":
    run()
