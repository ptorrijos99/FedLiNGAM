"""
What a client-level erasure costs under cohort aggregation.

With cohorts of m, forgetting one client deletes its whole cohort: re-reporting
the cohort without the departing member would let the server difference the two
sums and recover that member, which is the leak cohorts exist to close. The
other m - 1 members' data are lost with it. This measures the accuracy of the
model after one such erasure, on raw and on standardised data, with the clients
of cohorts.py.

Usage:  python experiments/deployment/cohort_erasure.py [--seeds 15]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import (   # noqa: E402
    ancestral_accuracy, PGMSyntheticDataGenerator, standardise,
)
from experiments.deployment.cohorts import noniid_clients, COHORT_SIZES   # noqa: E402
from src.common.cumulants import compute_local_raw_stats, aggregate_and_compute_cumulants   # noqa: E402
from src.fedrcd.core import fedrcd_noniterative   # noqa: E402

OUT = "experiments/deployment/results/cohort_erasure.csv"


def ni_order(clients):
    agg = aggregate_and_compute_cumulants([compute_local_raw_stats(c, order=4) for c in clients])
    return fedrcd_noniterative(agg, adaptive=False, use_lasso=True, lmbda_type="fixed")[0]


def run(seeds=15, p=20, n=20_000, K=10, density=0.2, dist="mixed"):
    rows = []
    for seed in range(seeds):
        X, B_true, _ = PGMSyntheticDataGenerator(seed=seed).generate(
            n_samples=n, p=p, graph_density=density, dist_type=dist)
        clients, _ = noniid_clients(X, B_true, K, np.random.default_rng(20_000 + seed))
        for m in [0] + [c for c in COHORT_SIZES if c < K]:
            kept = clients[m:] if m else clients       # m = 0: nothing erased
            for setting, data in (("raw", kept), ("std", standardise(kept))):
                rows.append(dict(seed=seed, cohort_size=m, setting=setting,
                                 clients_left=len(kept),
                                 AncAcc=ancestral_accuracy(B_true, ni_order(data))))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.pivot_table(index="cohort_size", columns="setting", values="AncAcc",
                         aggfunc=["mean", "sem"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    run(seeds=ap.parse_args().seeds)
