"""
Cohort aggregation: the unlearning/leak tension as a dial.

reconstruction.py shows that exact unlearning and exact reconstruction are the same
subtraction, and suggests that secure aggregation closes
the leak and destroys the unlearning property with it.  Stated that way it is a
binary choice, and it is not: between "the server holds every client's terms"
and "the server holds only the total" there is a cohort size.

The protocol: clients are partitioned into fixed cohorts of m, each cohort's
contributions are summed before the server sees them, and erasure happens at
cohort granularity --- a departing client's cohort re-reports its sum.
Aggregation is associative, so this changes nothing about the recovered model
(asserted here, not assumed).  What it changes is what the subtraction attack
returns: the cohort's aggregate instead of the victim's own statistics.

m = 1 is the protocol as published: exact unlearning, and the exact leak of
reconstruction.py.  m = K is secure aggregation: no leak, no unlearning.  In between both
degrade together, and this measures how fast.

The clients here are deliberately NON-IID --- each draws its own per-variable
noise scales --- because with identical clients a cohort average is a fine
estimate of any member and the question does not arise.  Heterogeneity is what
makes "whose statistics are these" a real question, and it is the realistic
case: institutions differ.

Usage:  python experiments/deployment/cohorts.py [--seeds 15]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import (   # noqa: E402
    ancestral_accuracy, PGMSyntheticDataGenerator, _build_fed_stats,
    _run_fed_algos,
)
from src.common.cumulants import (          # noqa: E402
    compute_local_raw_stats, aggregate_and_compute_cumulants,
)

OUT = "experiments/deployment/results/cohorts.csv"
COHORT_SIZES = [1, 2, 3, 5, 10]


def noniid_clients(X, B_true, K, rng):
    """Split the sample and give each client its own per-variable scales.

    A diagonal rescaling per client leaves the causal order untouched, so the
    federated model is unaffected; what it creates is a private profile for
    each client, which is what the attack is trying to recover.
    """
    p = X.shape[1]
    idx = rng.permutation(X.shape[0])
    shards = np.array_split(idx, K)
    clients, scales = [], []
    for sh in shards:
        s = rng.lognormal(mean=0.0, sigma=0.5, size=p)
        clients.append(X[sh] * s)
        scales.append(s)
    return clients, scales


def moments_of(Xk):
    """The per-variable summaries the reconstruction attack targets."""
    mu = Xk.mean(axis=0)
    var = Xk.var(axis=0)
    sd = np.sqrt(np.maximum(var, 1e-15))
    skew = (((Xk - mu) / sd) ** 3).mean(axis=0)
    kurt = (((Xk - mu) / sd) ** 4).mean(axis=0) - 3
    return dict(mean=mu, var=var, skew=skew, kurt=kurt)


def cohort_estimate(clients, cohort):
    """What the server recovers by subtracting one cohort's aggregate.

    Summing raw statistics over the cohort and dividing by the cohort's sample
    count is exactly what the attacker can compute, and for a cohort of one it
    is the victim's own summary.
    """
    stacked = np.vstack([clients[k] for k in cohort])
    return moments_of(stacked)


def relerr(est, truth):
    return float(np.median(np.abs(est - truth) / np.maximum(np.abs(truth), 1e-12)))


def run(seeds=15, p=20, n=20_000, K=10, density=0.2, dist="mixed"):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {(r["seed"], r["cohort_size"]) for r in rows}

    for seed in range(seeds):
        gen = PGMSyntheticDataGenerator(seed=seed)
        X, B_true, _ = gen.generate(n_samples=n, p=p, graph_density=density,
                                    dist_type=dist)
        rng = np.random.default_rng(20_000 + seed)
        clients, _ = noniid_clients(X, B_true, K, rng)

        victim = 0
        truth = moments_of(clients[victim])

        # the model itself, computed once: cohort aggregation must not move it
        stats = _build_fed_stats(clients, ["FedRCD-NI"])
        order_ref = _run_fed_algos(stats, clients, ["FedRCD-NI"])["FedRCD-NI"][0]
        anc_ref = ancestral_accuracy(B_true, order_ref)

        for m in COHORT_SIZES:
            if (seed, m) in done:
                continue
            cohort = list(range(0, min(m, K)))       # the victim's cohort
            est = cohort_estimate(clients, cohort)

            # associativity check: summing within cohorts first must give the
            # same aggregate, so the same order and the same accuracy
            groups = [list(range(i, min(i + m, K))) for i in range(0, K, m)]
            pooled = [np.vstack([clients[k] for k in g]) for g in groups]
            agg_batched = aggregate_and_compute_cumulants(
                [compute_local_raw_stats(c, order=4) for c in pooled])
            agg_flat = aggregate_and_compute_cumulants(
                [compute_local_raw_stats(c, order=4) for c in clients])
            cov_gap = float(np.max(np.abs(np.asarray(agg_batched["cov"])
                                          - np.asarray(agg_flat["cov"]))))

            rows.append(dict(
                seed=seed, cohort_size=m,
                clients_needed_to_unlearn=len(cohort) - 1,
                mean_relerr=relerr(est["mean"], truth["mean"]),
                var_relerr=relerr(est["var"], truth["var"]),
                skew_relerr=relerr(est["skew"], truth["skew"]),
                kurt_relerr=relerr(est["kurt"], truth["kurt"]),
                AncAcc=anc_ref, cov_gap=cov_gap))
            done.add((seed, m))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  seed {seed} done", flush=True)

    report(pd.DataFrame(rows))


def report(df):
    print("\n=== What the subtraction attack recovers about one client, "
          "by cohort size ===")
    print("(median relative error of the victim's per-variable summaries; "
          "0 = exact recovery)")
    cols = ["mean_relerr", "var_relerr", "skew_relerr", "kurt_relerr",
            "clients_needed_to_unlearn"]
    print(df.groupby("cohort_size")[cols].mean().round(4).to_string())
    print("\nlargest covariance difference between cohort-wise and flat "
          "aggregation (must be 0: the model is unchanged):")
    print(df.groupby("cohort_size")["cov_gap"].max().to_string())
    print("\nAncAcc (identical by construction, reported as a check):")
    print(df.groupby("cohort_size")["AncAcc"].mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        report(pd.read_csv(OUT))
    else:
        run(seeds=a.seeds)
