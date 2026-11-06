"""
the FedRCD-X arm of the DP sweep.

`dp_one_round.py` covers only the single-round protocols. FedRCD-X is multi-round,
so it cannot share their accounting: it publishes fourth-order statistics p-1
times, and under zCDP the budget splits across all of them. Composition is
exactly the axis on which multi-round protocols are expected to lose, so leaving
it out would understate the cost of the paper's most accurate variant.

Per round the sensitivity is the same as one single-round release (the client
sends the same moment arrays, over the current active set), but the budget per
round is rho_total / (p-1), so the noise scale grows by sqrt(p-1) relative to
the one-shot variants.

Usage: python experiments/deployment/dp_fedrcdx.py
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
from src.common.metrics import compute_tau_matrix_4th_stats   # noqa: E402

OUT = "experiments/deployment/results/dp_fedrcdx.csv"
EPS_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float("inf")]
DELTA = 1e-5
CLIP = 4.0


def dp_fedrcdx_order(data, eps, rng):
    """FedRCD-X with a Gaussian release per round, zCDP-composed over p-1 rounds."""
    p = data[0].shape[1]
    rho_total = dp.rho_from_eps_delta(eps, DELTA)
    n_rounds = p - 1
    rho_round = rho_total / n_rounds if np.isfinite(rho_total) else float("inf")

    Xall = np.vstack(data)
    mu0, sd0 = Xall.mean(0), Xall.std(0)
    sd0[sd0 == 0] = 1.0
    # Clipping only bounds the sensitivity, so the noise-free reference skips it
    C = CLIP if np.isfinite(eps) else np.inf
    X_local = [dp.clip((Xk - mu0) / sd0, C).astype(float) for Xk in data]

    active = list(range(p))
    order = []
    sigmas = []

    for _ in range(p - 1):
        m = len(active)
        sens = dp.moment_sensitivity(m, CLIP, order=4)
        sigma = dp.gaussian_sigma(sens, rho_round)
        sigmas.append(sigma)

        raw = [compute_local_raw_stats(X, order=4) for X in X_local]
        summed = {k: (sum(r[k] for r in raw) if k != "n"
                      else sum(r["n"] for r in raw)) for k in raw[0]}
        noisy = dp.symmetrise(dp.privatise_moment_stats(summed, sigma, rng))
        agg = aggregate_and_compute_cumulants([noisy])

        tau = compute_tau_matrix_4th_stats(
            np.array(agg["c4"]), np.array(agg["c13"]),
            np.array(agg["c31"]), np.array(agg["c22"]), list(range(m)))
        src = int(np.argmin(np.sum(tau, axis=1)))
        order.append(active[src])

        cov = np.array(agg["cov"], dtype=float)
        betas = cov[:, src] / max(float(cov[src, src]), 1e-12)
        rem = [j for j in range(m) if j != src]
        X_local = [np.column_stack([X[:, j] - betas[j] * X[:, src] for j in rem])
                   for X in X_local]
        active = [active[j] for j in rem]

    order.append(active[0])
    # report the WORST round (the first, where the active set is largest),
    # not the last, which is the most favourable
    return order, float(max(sigmas))


def run(seeds=10, p=20, n_per_client=2000, K=10, dist="mixed", density=0.2):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    if os.path.exists(OUT):
        rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["eps"], r["seed"]) for r in rows}

    for eps in EPS_GRID:
        for seed in range(seeds):
            if (eps, seed) in done:
                continue
            gen = PGMSyntheticDataGenerator(seed=seed)
            X, B_true, _ = gen.generate(n_samples=n_per_client * K, p=p,
                                        graph_density=density, dist_type=dist)
            data = _split_data(X, K, np.random.default_rng(seed))
            rng = np.random.default_rng(90_000 + seed)
            try:
                o, sigma = dp_fedrcdx_order(data, eps, rng)
                acc = ancestral_accuracy(B_true, o)
            except Exception as e:
                print(f"    eps={eps} seed={seed}: {e}", flush=True)
                acc, sigma = np.nan, np.nan
            rows.append(dict(eps=eps, seed=seed, method="FedRCD-X",
                             AncAcc=acc, sigma=sigma))
            done.add((eps, seed))
            pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  eps={eps} done", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== FedRCD-X under DP (per-round release, zCDP over p-1 rounds) ===")
    print(df.pivot_table(index="method", columns="eps", values="AncAcc",
                         aggfunc="mean").round(3).to_string())
    print("\nnoise scale per round:")
    print(df.pivot_table(index="method", columns="eps", values="sigma",
                         aggfunc="mean").round(1).to_string())
    print(f"\nsingle-round sensitivity at p={p}, C={CLIP}: "
          f"{dp.moment_sensitivity(p, CLIP, 4):.0f}; "
          f"budget per round is rho/{p-1}, so sigma grows by "
          f"sqrt({p-1}) = {np.sqrt(p-1):.2f} relative to a one-shot release.")


if __name__ == "__main__":
    run()
