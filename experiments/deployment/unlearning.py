"""
Federated unlearning: exactness and cost.

The PGM paper claims exact federated unlearning for the single-round variants
(FedRCD-NI, FedRCD-Def) and states that the multi-round variant (FedRCD-X)
loses it, but reports no measurement.

We measure, for a GDPR right-to-erasure request of a given granularity:
  * exactness -- max |B_unlearn - B_retrain| and whether the order matches;
  * cost      -- server time to honour the request, and client-side work
                 (raw-moment recomputation vs. a full protocol replay).

Usage: python experiments/deployment/unlearning.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import PGMSyntheticDataGenerator, _split_data   # noqa: E402
from src.common.cumulants import (                                              # noqa: E402
    compute_local_raw_stats, aggregate_and_compute_cumulants,
)
from src.fedrcd.core import fedrcd, fedrcd_noniterative, fedrcd_exact  # noqa: E402

OUT = "experiments/deployment/results/unlearning.csv"


def _subtract(a: dict, b: dict) -> dict:
    """Raw moments are additive, hence subtractive: this is the unlearning step."""
    return {k: (a[k] - b[k]) for k in a}


def run(p=20, n_per_client=2000, K=10, seeds=10, dist="mixed"):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []

    for seed in range(seeds):
        gen = PGMSyntheticDataGenerator(seed=seed)
        X, B_true, _ = gen.generate(n_samples=n_per_client * K, p=p,
                                    graph_density=0.2, dist_type=dist)
        rng = np.random.default_rng(seed)
        shards = _split_data(X, K, rng)

        # ---- baseline: everyone participates ---------------------------------
        stats4 = [compute_local_raw_stats(s, order=4) for s in shards]

        for gran, keep in [
            ("one client", lambda i: i != 3),
            ("half the federation", lambda i: i >= K // 2),
        ]:
            kept_idx = [i for i in range(K) if keep(i)]
            kept_shards = [shards[i] for i in kept_idx]

            # === single-round variants: unlearn = subtract raw moments =======
            for name, solver in [("FedRCD-NI", fedrcd_noniterative),
                                 ("FedRCD-Def", fedrcd)]:
                t0 = time.perf_counter()
                agg_unl = aggregate_and_compute_cumulants([stats4[i] for i in kept_idx])
                o_unl, B_unl = solver(agg_unl, adaptive=False, use_lasso=True,
                                      lmbda_type="fixed")
                t_unl = time.perf_counter() - t0

                t0 = time.perf_counter()
                fresh = [compute_local_raw_stats(s, order=4) for s in kept_shards]
                agg_ret = aggregate_and_compute_cumulants(fresh)
                o_ret, B_ret = solver(agg_ret, adaptive=False, use_lasso=True,
                                      lmbda_type="fixed")
                t_ret = time.perf_counter() - t0

                rows.append(dict(seed=seed, method=name, granularity=gran,
                                 order_match=int(list(o_unl) == list(o_ret)),
                                 max_abs_diff=float(np.max(np.abs(B_unl - B_ret))),
                                 t_unlearn=t_unl, t_retrain=t_ret,
                                 client_recompute=0))

            # === instance level: drop 1% of one client's rows ================
            n0 = shards[0].shape[0]
            n_forget = max(1, n0 // 100)
            forget_stats = compute_local_raw_stats(shards[0][:n_forget], order=4)
            patched = _subtract(stats4[0], forget_stats)
            t0 = time.perf_counter()
            agg_unl = aggregate_and_compute_cumulants([patched] + stats4[1:])
            o_unl, B_unl = fedrcd_noniterative(agg_unl, adaptive=False,
                                               use_lasso=True, lmbda_type="fixed")
            t_unl = time.perf_counter() - t0

            t0 = time.perf_counter()
            fresh0 = compute_local_raw_stats(shards[0][n_forget:], order=4)
            agg_ret = aggregate_and_compute_cumulants([fresh0] + stats4[1:])
            o_ret, B_ret = fedrcd_noniterative(agg_ret, adaptive=False,
                                               use_lasso=True, lmbda_type="fixed")
            t_ret = time.perf_counter() - t0
            rows.append(dict(seed=seed, method="FedRCD-NI", granularity="1% of one client",
                             order_match=int(list(o_unl) == list(o_ret)),
                             max_abs_diff=float(np.max(np.abs(B_unl - B_ret))),
                             t_unlearn=t_unl, t_retrain=t_ret, client_recompute=0))

            # === multi-round variant: no shortcut, must replay the protocol ===
            for name, fn in [("FedRCD-X", lambda d: fedrcd_exact(
                                  d, use_lasso=True, lmbda_type="fixed"))]:
                t0 = time.perf_counter()
                o_ret, B_ret = fn(kept_shards)
                t_ret = time.perf_counter() - t0
                rows.append(dict(seed=seed, method=name, granularity=gran,
                                 order_match=-1, max_abs_diff=np.nan,
                                 t_unlearn=np.nan, t_retrain=t_ret,
                                 client_recompute=1))

        print(f"  seed {seed} done")
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\n=== Unlearning: exactness ===")
    print(df.groupby(["method", "granularity"])[["order_match", "max_abs_diff"]]
            .mean().to_string())
    print("\n=== Unlearning: server seconds to honour the request ===")
    print(df.groupby(["method", "granularity"])[["t_unlearn", "t_retrain"]]
            .mean().round(4).to_string())
    spd = df.dropna(subset=["t_unlearn"])
    print("\nmean speed-up (retrain / unlearn): %.1fx"
          % (spd["t_retrain"] / spd["t_unlearn"]).mean())


if __name__ == "__main__":
    run()
