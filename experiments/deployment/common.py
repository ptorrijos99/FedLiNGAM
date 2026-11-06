"""Shared helpers for the deployment experiments."""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.pgm26.benchmark import (          # noqa: E402
    PGMSyntheticDataGenerator, _build_fed_stats, _run_fed_algos, _split_data,
)


def ancestral_accuracy(B_true, order):
    """AncAcc (Def. 3 of the PGM paper): fraction of ancestral pairs ordered right."""
    p = B_true.shape[0]
    A = (np.abs(B_true) > 1e-8).astype(int)
    reach = A.copy()
    for _ in range(p):
        reach = ((reach + reach @ A) > 0).astype(int)
    pos = {v: k for k, v in enumerate(order)}
    pairs = correct = 0
    for i in range(p):
        for j in range(p):
            if reach[i, j]:
                pairs += 1
                if pos[j] < pos[i]:
                    correct += 1
    return correct / max(pairs, 1)


def varsortability(B_true, var):
    """Reisach et al. (2021) varsortability v of a dataset."""
    p = B_true.shape[0]
    A = (np.abs(B_true) > 1e-8).astype(int)
    reach = A.copy()
    for _ in range(p):
        reach = ((reach + reach @ A) > 0).astype(int)
    tot = inc = 0
    for i in range(p):
        for j in range(p):
            if reach[i, j]:            # j is an ancestor of i
                tot += 1
                if var[j] < var[i]:
                    inc += 1
    return inc / max(tot, 1) if tot else 0.5


def standardise(client_data):
    """Marginal standardisation using the pooled mean/std (a diagonal rescaling)."""
    Xall = np.vstack(client_data)
    sd = Xall.std(axis=0)
    sd[sd == 0] = 1.0
    mu = Xall.mean(axis=0)
    return [(c - mu) / sd for c in client_data]


SYMMETRIC = ['uniform', 'laplace', 'logistic', 't']
SKEWED = ['exponential', 'gamma', 'pareto']
