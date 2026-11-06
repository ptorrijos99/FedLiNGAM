"""
Deleting single records: what the difference of two aggregates reveals.

reconstruction.py and cohorts.py attack a departing client. A data-protection request usually
concerns one person, and the protocol honours it by subtracting that record's
raw moments. The server then sees the difference of two aggregates, which is
exactly the deleted records' moment sums. For one record the first block IS the
record. Processing deletions in batches of b, the per-instance analogue of a
cohort, hides each record inside the sum of b, but the release has about
4p + 3p^2 numbers for b*p unknowns, so by counting alone the batch is
determined (up to row order) whenever b < ~3p.

The attack is plain moment matching: find b rows whose moment sums equal the
released ones, from several random starts, and match the result to the true
rows with the Hungarian algorithm. The baseline gives every deleted record the
batch mean, which is what the release reveals if it cannot be inverted.

Usage: python experiments/deployment/record_erasure.py [--seeds 5]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.deployment.common import PGMSyntheticDataGenerator   # noqa: E402
from src.common.cumulants import compute_local_raw_stats            # noqa: E402

OUT = "experiments/deployment/results/record_erasure.csv"
BATCH_SIZES = [1, 2, 3, 5, 10, 20, 30, 40, 60, 80, 120]
P_GRID = [10, 20]
RECOVERED = 0.05          # a record counts as recovered below this relative error
torch.set_default_dtype(torch.float64)

# Release blocks in the order of `compute_local_raw_stats`
BLOCKS = ["sum_x", "sum_xx", "sum_x3", "sum_x1y2", "sum_x2y1",
          "sum_x4", "sum_x1y3", "sum_x3y1", "sum_x2y2"]


def moment_sums(X):
    """The release of `compute_local_raw_stats`, off-diagonals kept, in torch."""
    X2, X3 = X ** 2, X ** 3
    off = 1.0 - torch.eye(X.shape[1])
    return dict(sum_x=X.sum(0), sum_xx=X.T @ X, sum_x3=X3.sum(0),
                sum_x1y2=(X.T @ X2) * off, sum_x2y1=(X2.T @ X) * off,
                sum_x4=(X2 ** 2).sum(0), sum_x1y3=(X.T @ X3) * off,
                sum_x3y1=(X3.T @ X) * off, sum_x2y2=(X2.T @ X2) * off)


def released_difference(before, after):
    """What the server computes: the deleted records' raw moment sums."""
    return {k: torch.as_tensor(np.asarray(before[k], float) - np.asarray(after[k], float))
            for k in BLOCKS}


def invert(target, b, p, rng, restarts=6, iters=500, steps=20):
    """Moment matching from random starts; returns the best batch and its loss."""
    scale = {k: v.abs() + b for k, v in target.items()}
    mu = (target["sum_x"] / b).numpy()
    sd = np.sqrt(np.maximum(np.diag(target["sum_xx"].numpy()) / b - mu ** 2, 1e-6))
    best, best_loss = None, np.inf
    for _ in range(restarts):
        Y = torch.tensor(mu + sd * rng.standard_normal((b, p)), requires_grad=True)
        opt = torch.optim.LBFGS([Y], lr=1, max_iter=iters, tolerance_grad=1e-12,
                                tolerance_change=1e-14, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            m = moment_sums(Y)
            loss = sum((((m[k] - target[k]) / scale[k]) ** 2).sum() for k in BLOCKS)
            loss.backward()
            return loss

        loss = np.inf
        for _ in range(steps):
            prev, loss = loss, float(opt.step(closure))
            if loss < 1e-14 or loss > 0.999 * prev:
                break
        if loss < best_loss:
            best, best_loss = Y.detach().numpy().copy(), loss
        if best_loss < 1e-12:
            break
    return best, best_loss


def record_errors(est, truth):
    """Relative error of each true record against its matched estimate."""
    cost = ((truth[:, None, :] - est[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    return np.linalg.norm(truth[r] - est[c], axis=1) / np.linalg.norm(truth[r], axis=1)


def task(args):
    """One (p, seed, b) cell; the batch drawn depends only on these three."""
    p, seed, b, n, density, dist = args
    torch.set_num_threads(1)
    X, _, _ = PGMSyntheticDataGenerator(seed=seed).generate(
        n_samples=n, p=p, graph_density=density, dist_type=dist)
    X = (X - X.mean(0)) / X.std(0)
    rng = np.random.default_rng([70_000, p, seed, b])
    client = X[: n // 10]                         # the deleting client's shard
    before = compute_local_raw_stats(client, order=4)
    gone = rng.choice(len(client), size=b, replace=False)
    after = compute_local_raw_stats(np.delete(client, gone, axis=0), order=4)
    target = released_difference(before, after)
    truth = client[gone]

    est, loss = invert(target, b, p, rng)
    err = record_errors(est, truth)
    base = record_errors(np.repeat(truth.mean(0, keepdims=True), b, 0), truth)
    n_stats = sum(int(np.count_nonzero(np.asarray(before[k]))) for k in BLOCKS)
    return dict(p=p, seed=seed, b=b, unknowns=b * p, released=n_stats,
                loss=loss, median_relerr=float(np.median(err)),
                recovered=float(np.mean(err < RECOVERED)),
                baseline_relerr=float(np.median(base)))


def run(seeds=5, n=20_000, density=0.2, dist="mixed", workers=10):
    from multiprocessing import Pool
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {(r["p"], r["seed"], r["b"]) for r in rows}
    todo = [(p, seed, b, n, density, dist) for p in P_GRID for b in BATCH_SIZES
            for seed in range(seeds) if (p, seed, b) not in done]

    with Pool(workers) as pool:
        for r in pool.imap_unordered(task, todo):
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  p={r['p']} seed={r['seed']} b={r['b']:4d}  loss={r['loss']:.1e}  "
                  f"relerr={r['median_relerr']:.3f}  recovered={r['recovered']:.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    print("\n=== Records recovered from the release of one batched deletion ===")
    print(df.pivot_table(index="b", columns="p",
                         values=["recovered", "median_relerr", "baseline_relerr"],
                         aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    run(seeds=ap.parse_args().seeds)
