"""
Differential privacy for the transmitted statistics.

The cumulant protocols release sums over records, so the Gaussian mechanism applies
directly once each record's contribution is bounded. Records are clipped
per-coordinate to [-C, C] after standardisation by the global scale, which makes
every released quantity bounded and gives closed-form sensitivities.

Accounting is in zero-concentrated DP (Bun & Steinke, 2016), which composes
additively and converts cleanly to (eps, delta):

    Gaussian mechanism, L2 sensitivity D, noise sigma  ->  rho = D^2 / (2 sigma^2)
    k rounds                                           ->  rho_total = sum rho_i
    conversion                                         ->  eps = rho + 2 sqrt(rho ln(1/delta))

With |x_i| <= C, a fourth-order raw moment sum has per-record sensitivity C^4, so
the L2 sensitivity of the release is ~ sqrt(3) p C^4: it grows as the fourth power
of the clipping bound. `experiments/deployment/dp_one_round.py` measures what that costs.
"""

import math
from typing import Dict

import numpy as np

# Blocks of `compute_local_raw_stats`, with the exponent of C that bounds a
# single record's contribution to one entry, and the entry count as a function
# of p.
_MOMENT_BLOCKS = {
    "sum_x":    (1, "p"),
    "sum_xx":   (2, "p2"),
    "sum_x3":   (3, "p"),
    "sum_x1y2": (3, "p2"),
    "sum_x2y1": (3, "p2"),
    "sum_x4":   (4, "p"),
    "sum_x1y3": (4, "p2"),
    "sum_x3y1": (4, "p2"),
    "sum_x2y2": (4, "p2"),
}


def rho_from_eps_delta(eps: float, delta: float = 1e-5) -> float:
    """Largest zCDP rho whose (eps, delta) guarantee is at most `eps`.

    Inverts eps = rho + 2 sqrt(rho ln(1/delta)) for rho.
    """
    if not np.isfinite(eps):
        return float("inf")
    L = math.log(1.0 / delta)
    # rho + 2 sqrt(L) sqrt(rho) - eps = 0, quadratic in sqrt(rho)
    a, b, c = 1.0, 2.0 * math.sqrt(L), -eps
    r = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return r * r


def gaussian_sigma(sensitivity: float, rho: float) -> float:
    """Noise scale for one Gaussian release at the given zCDP budget."""
    if not np.isfinite(rho) or rho <= 0:
        return 0.0 if not np.isfinite(rho) else float("inf")
    return sensitivity / math.sqrt(2.0 * rho)


def clip(X: np.ndarray, C: float) -> np.ndarray:
    """Per-coordinate clipping, applied after scaling by the global std."""
    return np.clip(X, -C, C)


def moment_sensitivity(p: int, C: float, order: int = 4) -> float:
    """L2 sensitivity of one record's contribution to the raw-moment release."""
    total = 0.0
    for name, (power, shape) in _MOMENT_BLOCKS.items():
        if order < 4 and name in ("sum_x4", "sum_x1y3", "sum_x3y1", "sum_x2y2"):
            continue
        count = p if shape == "p" else p * p
        total += count * (C ** (2 * power))
    return math.sqrt(total)


def privatise_moment_stats(agg: Dict, sigma: float, rng) -> Dict:
    """Add Gaussian noise to an aggregated raw-moment dict, in place of the sum.

    `n` is left exact: the number of participating records is public in this
    setting (the server knows who joined), so it carries no per-record secret.
    """
    if sigma <= 0:
        return dict(agg)
    out = {}
    for k, v in agg.items():
        if k == "n":
            out[k] = v
        elif isinstance(v, np.ndarray):
            out[k] = v + rng.normal(0.0, sigma, size=v.shape)
        else:
            out[k] = v
    return out


def symmetrise(agg: Dict) -> Dict:
    """Restore the symmetry that independent noise destroys.

    sum_xx and sum_x2y2 are symmetric by construction; averaging the noisy
    matrix with its transpose halves the variance on those blocks for free.
    """
    out = dict(agg)
    for k in ("sum_xx", "sum_x2y2"):
        if k in out and isinstance(out[k], np.ndarray) and out[k].ndim == 2:
            out[k] = 0.5 * (out[k] + out[k].T)
    return out
