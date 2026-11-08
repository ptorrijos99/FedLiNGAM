"""
Differential privacy against the server itself.

dp_one_round.py adds the Gaussian noise once to the aggregated sums, which protects the
released model but not the clients from an honest-but-curious server: that
server sums the noiseless contributions itself. Here every client clips its
standardised data and adds its own noise, calibrated to one record, before it
sends anything. The server therefore sees only noisy contributions, and the
aggregate carries K times the variance of a single release.

Two quantities are measured on the same releases:
  * the ancestral accuracy of each method on the noisy aggregate;
  * what the subtraction attack recovers about one departing client: its
    per-variable mean, variance, skewness and kurtosis, compared with the
    client's true summaries in the same units as cohorts.py.

Record-level privacy bounds what any single observation changes. It does not
bound the client's aggregate profile, which is an average over its records.

Usage:  python experiments/deployment/dp_clients.py [--seeds 10]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import (   # noqa: E402
    ancestral_accuracy, _split_data, PGMSyntheticDataGenerator,
)
from experiments.deployment.cohorts import (   # noqa: E402
    noniid_clients, moments_of, relerr,
)
from src.common.cumulants import (          # noqa: E402
    compute_local_raw_stats, aggregate_and_compute_cumulants,
)
from src.common import dp                   # noqa: E402
from src.common.metrics import compute_tau_matrix_4th_stats   # noqa: E402
from src.fedrcd.core import fedrcd_noniterative, fedrcd        # noqa: E402
from src.fedishc.core import hc as fedhc_solver                # noqa: E402

OUT = "experiments/deployment/results/dp_clients.csv"
EPS_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")]
DELTA = 1e-5
CLIP = 4.0


def _scaled(data, eps):
    """Standardise with the global moments (federable) and clip under a finite budget."""
    Xall = np.vstack(data)
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd == 0] = 1.0
    C = CLIP if np.isfinite(eps) else np.inf
    return [dp.clip((Xk - mu) / sd, C).astype(float) for Xk in data], mu, sd


def _noisy_release(Xk, order, sigma, rng):
    """One client's raw-moment sums with its own Gaussian noise."""
    return dp.privatise_moment_stats(compute_local_raw_stats(Xk, order=order), sigma, rng)


def _sum(releases):
    out = {}
    for k in releases[0]:
        out[k] = sum(r[k] for r in releases)
    return dp.symmetrise(out)


def one_round(data, eps, rng, order, solver):
    """A one-round method on client-noised statistics."""
    p = data[0].shape[1]
    sigma = dp.gaussian_sigma(dp.moment_sensitivity(p, CLIP, order=order),
                              dp.rho_from_eps_delta(eps, DELTA))
    Xs, _, _ = _scaled(data, eps)
    agg = aggregate_and_compute_cumulants([_sum([_noisy_release(X, order, sigma, rng) for X in Xs])])
    if order < 4:
        return fedhc_solver({k: agg[k] for k in ("c3", "c12", "c21", "n")})[0]
    return solver(agg, adaptive=False, use_lasso=True, lmbda_type="fixed")[0]


def fedrcdx(data, eps, rng):
    """FedRCD-X with every client noising every round, zCDP-composed over p - 1 rounds."""
    p = data[0].shape[1]
    rho_total = dp.rho_from_eps_delta(eps, DELTA)
    rho_round = rho_total / (p - 1) if np.isfinite(rho_total) else float("inf")
    Xs, _, _ = _scaled(data, eps)
    active, order = list(range(p)), []
    for _ in range(p - 1):
        m = len(active)
        sigma = dp.gaussian_sigma(dp.moment_sensitivity(m, CLIP, order=4), rho_round)
        agg = aggregate_and_compute_cumulants([_sum([_noisy_release(X, 4, sigma, rng) for X in Xs])])
        tau = compute_tau_matrix_4th_stats(np.array(agg["c4"]), np.array(agg["c13"]),
                                           np.array(agg["c31"]), np.array(agg["c22"]),
                                           list(range(m)))
        src = int(np.argmin(np.sum(tau, axis=1)))
        order.append(active[src])
        cov = np.array(agg["cov"], dtype=float)
        betas = cov[:, src] / max(float(cov[src, src]), 1e-12)
        rem = [j for j in range(m) if j != src]
        Xs = [np.column_stack([X[:, j] - betas[j] * X[:, src] for j in rem]) for X in Xs]
        active = [active[j] for j in rem]
    order.append(active[0])
    return order


def moments_from_sums(s, mu, sd):
    """The departing client's summaries from its noisy sums, back in the original units."""
    n = s["n"]
    m1 = s["sum_x"] / n
    m2 = np.diag(s["sum_xx"]) / n
    m3 = s["sum_x3"] / n
    m4 = s["sum_x4"] / n
    var_z = m2 - m1 ** 2                       # may be negative once noise is added
    v = np.maximum(var_z, 1e-12)
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return dict(mean=mu + sd * m1, var=sd ** 2 * var_z,
                skew=c3 / v ** 1.5, kurt=c4 / v ** 2 - 3)


def attack(seed, eps, p=20, n=20_000, K=10, density=0.2, dist="mixed"):
    """The heterogeneous clients of cohorts.py; the victim releases once under its own noise."""
    X, B_true, _ = PGMSyntheticDataGenerator(seed=seed).generate(
        n_samples=n, p=p, graph_density=density, dist_type=dist)
    rng = np.random.default_rng(20_000 + seed)
    clients, _ = noniid_clients(X, B_true, K, rng)
    truth = moments_of(clients[0])
    Xs, mu, sd = _scaled(clients, eps)
    sigma = dp.gaussian_sigma(dp.moment_sensitivity(p, CLIP, order=4),
                              dp.rho_from_eps_delta(eps, DELTA))
    est = moments_from_sums(_noisy_release(Xs[0], 4, sigma, np.random.default_rng(70_000 + seed)), mu, sd)
    return {f"{k}_relerr": relerr(est[k], truth[k]) for k in truth}


def run(seeds=10, p=20, n_per_client=2000, K=10, dist="mixed", density=0.2):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {(r["eps"], r["seed"], r["method"]) for r in rows}
    for eps in EPS_GRID:
        for seed in range(seeds):
            X, B_true, _ = PGMSyntheticDataGenerator(seed=seed).generate(
                n_samples=n_per_client * K, p=p, graph_density=density, dist_type=dist)
            data = _split_data(X, K, np.random.default_rng(seed))
            rng = np.random.default_rng(90_000 + seed)
            specs = [("FedRCD-NI", lambda: one_round(data, eps, rng, 4, fedrcd_noniterative)),
                     ("FedRCD-Def", lambda: one_round(data, eps, rng, 4, fedrcd)),
                     ("FedHC", lambda: one_round(data, eps, rng, 3, None)),
                     ("FedRCD-X", lambda: fedrcdx(data, eps, rng)),
                     ("Attack", None)]
            for name, fn in specs:
                if (eps, seed, name) in done:
                    continue
                row = dict(eps=eps, seed=seed, method=name)
                if name == "Attack":
                    row.update(attack(seed, eps, p=p, n=n_per_client * K, K=K, density=density))
                else:
                    row["AncAcc"] = ancestral_accuracy(B_true, fn())
                rows.append(row)
                done.add((eps, seed, name))
            pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  eps={eps} done", flush=True)
    report(pd.DataFrame(rows))


def report(df):
    print("\n=== AncAcc with noise added by each client (delta=%g, clip C=%g) ===" % (DELTA, CLIP))
    print(df[df.method != "Attack"].pivot_table(index="method", columns="eps", values="AncAcc").round(3).to_string())
    print("\n=== Median relative error of the departing client's summaries ===")
    cols = ["mean_relerr", "var_relerr", "skew_relerr", "kurt_relerr"]
    print(df[df.method == "Attack"].groupby("eps")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        report(pd.read_csv(OUT))
    else:
        run(seeds=a.seeds)
