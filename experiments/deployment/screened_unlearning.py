"""
Does screened unlearning of FedRCD-X actually work?

`src/fedrcd/unlearn.py` subtracts the forgotten contribution exactly and then
uses the result as a SCREEN on the source selected at each round. The screen is
not a proof, because it runs on residuals still stale by delta_beta. Two numbers
decide whether it is worth anything:

  false certification rate -- how often the screen accepts all p-1 rounds but
      the resulting order differs from an honest re-run. This is the error the
      scheme can actually make, and it must be small or the scheme is unsound.

  rounds saved -- how many rounds are served without contacting a client.

Ground truth is `fedrcd_exact` re-run on the retained data.

Usage: python experiments/deployment/screened_unlearning.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import PGMSyntheticDataGenerator, _split_data   # noqa: E402
from src.common.cumulants import compute_local_raw_stats                        # noqa: E402
from src.fedrcd.core import fedrcd_exact                                        # noqa: E402
from src.fedrcd.unlearn import fedrcd_exact_recorded, screened_unlearn          # noqa: E402

OUT = "experiments/deployment/results/screened_unlearning.csv"


def run(seeds=15, p=20, n_per_client=2000, K=10, dist="mixed", density=0.2):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []

    for seed in range(seeds):
        gen = PGMSyntheticDataGenerator(seed=seed)
        X, _, _ = gen.generate(n_samples=n_per_client * K, p=p,
                               graph_density=density, dist_type=dist)
        shards = _split_data(X, K, np.random.default_rng(seed))
        record = fedrcd_exact_recorded(shards)

        scenarios = [
            ("1% of one client", None, {0: None}),
            ("one client", lambda i: i != 3, None),
            ("half the federation", lambda i: i >= K // 2, None),
        ]

        for label, keep_fn, inst in scenarios:
            forget_stats = None
            if inst is not None:
                # Rebuild the per-round statistics of the rows being erased by
                # replaying the recorded deflations on those rows only.
                n_forget = max(1, shards[0].shape[0] // 100)
                forget_stats = {0: _forget_trace(record, shards, 0, n_forget)}
                kept = [s if i != 0 else s[n_forget:] for i, s in enumerate(shards)]
                keep_pred = None
            else:
                kept = [s for i, s in enumerate(shards) if keep_fn(i)]
                keep_pred = keep_fn

            t0 = time.perf_counter()
            res = screened_unlearn(record, keep_client=keep_pred,
                                   forget_stats=forget_stats,
                                   replay_data=kept)
            t_screen = time.perf_counter() - t0

            t0 = time.perf_counter()
            truth, _ = fedrcd_exact(kept, use_lasso=True, lmbda_type="fixed")
            t_retrain = time.perf_counter() - t0

            correct = list(res["order"]) == list(truth)
            rows.append(dict(
                seed=seed, scenario=label,
                fully_screened=int(res["fully_screened"]),
                correct=int(correct),
                false_certification=int(res["fully_screened"] and not correct),
                screened_rounds=res["screened_rounds"],
                replayed_rounds=res["replayed_rounds"],
                frac_saved=res["screened_rounds"] / (p - 1),
                t_screen=t_screen, t_retrain=t_retrain))
        print(f"  seed {seed} done")
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\n=== Screened unlearning of FedRCD-X (p=%d, K=%d, %d seeds) ===" % (p, K, seeds))
    print(df.groupby("scenario")[
        ["fully_screened", "correct", "false_certification",
         "frac_saved", "t_screen", "t_retrain"]].mean().round(3).to_string())
    print("\nfalse_certification = screen accepted every round but the order "
          "differs from an honest re-run. This is the scheme's actual error rate.")


def _forget_trace(record, shards, client, n_forget):
    """Per-round statistics of the erased rows, following the recorded sources."""
    rows_X = shards[client][:n_forget].copy().astype(float)
    active = list(range(record["p"]))
    trace = [compute_local_raw_stats(rows_X, order=4)]

    for t, src_local in enumerate(record["sources"][:-1]):
        agg_cols = record["active"][t]
        idx = src_local
        # Recompute the broadcast betas from the recorded full aggregate.
        from src.common.cumulants import aggregate_and_compute_cumulants
        agg = aggregate_and_compute_cumulants(record["per_client_round"][t])
        cov = np.array(agg["cov"], dtype=float)
        betas = cov[:, idx] / max(float(cov[idx, idx]), 1e-15)
        rem = [j for j in range(len(agg_cols)) if j != idx]
        rows_X = np.column_stack([rows_X[:, j] - betas[j] * rows_X[:, idx]
                                  for j in rem])
        active = [active[j] for j in rem]
        trace.append(compute_local_raw_stats(rows_X, order=4))
    return trace


if __name__ == "__main__":
    run()
