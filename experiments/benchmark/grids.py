"""The experiment grids of the extended article, one entry per experiment id.

A task is one data draw (cell + seed); every listed method runs on it under every
listed setting. Centralised methods and reference sorts see the pooled sample, which
does not depend on K, so in K sweeps they run only at the reference K.

    P_main        main grid: p, K and eleven noise settings
    P_bnlearn     eight bnlearn topologies
    E_gauss       disturbances interpolated from Gaussian to non-Gaussian
    E_sample      sample size up to 500 000, five noise families
    E_sample2     the same on the six remaining families
    E_violations  one modelling assumption broken at a time
    E_real        Sachs split by reagent condition, and the Danube discharge series
    E_scale       communication and running time against p
"""

from __future__ import annotations

from itertools import product

from experiments.benchmark import data as D
from experiments.benchmark.methods import ALL, CUMULANT, INSTRUMENT, NULLS

REF_K = 10
BOTH = ["raw", "std"]
PGM_DISTS = ["gaussian_cubed", "pareto", "logistic", "uniform", "laplace", "poisson",
             "exponential", "t", "gamma", "exponential_cubed", "mixed"]
TOPOLOGIES = ["asia", "sachs", "child", "insurance", "water", "mildew", "alarm", "barley"]


def _task(exp, cell, seed, methods, settings=BOTH):
    return {"exp": exp, "cell": cell, "seed": seed, "methods": list(methods), "settings": list(settings)}


def p_main():
    out = []
    for p, K, dist, seed in product([10, 20, 30, 50], [1, 5, 10, 20, 50], PGM_DISTS, range(10)):
        methods = ALL + INSTRUMENT if K == REF_K else CUMULANT
        out.append(_task("P_main", dict(kind="pgm", p=p, n=2000, K=K, dist=dist, density=0.2), seed, methods))
    return out


def p_bnlearn():
    return [_task("P_bnlearn", dict(kind="bnlearn", graph=g, n=2000, K=REF_K, dist=d), s, ALL + INSTRUMENT)
            for g, d, s in product(TOPOLOGIES, PGM_DISTS, range(10))]


def e_gauss():
    return [_task("E_gauss", dict(kind="gauss", p=20, n=2000, K=REF_K, lam=lam, density=0.2), s, ALL + INSTRUMENT)
            for lam, s in product([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], range(15))]


def e_sample():
    return [_task("E_sample", dict(kind="pgm", p=20, n=N, K=REF_K, dist=d, density=0.2), s, ALL + INSTRUMENT)
            for N, d, s in product([2_000, 5_000, 20_000, 50_000, 200_000, 500_000],
                                   ["laplace", "uniform", "exponential", "gamma", "mixed"], range(10))]


def e_sample2():
    return [_task("E_sample2", dict(kind="pgm", p=20, n=N, K=REF_K, dist=d, density=0.2), s, ALL + INSTRUMENT)
            for N, d, s in product([2_000, 5_000, 20_000, 50_000, 200_000, 500_000],
                                   ["gaussian_cubed", "pareto", "logistic", "poisson", "t", "exponential_cubed"],
                                   range(10))]


def e_violations():
    kinds = ["clean", "confounded", "nonlinear_tanh", "nonlinear_square", "nonlinear_sin",
             "measurement", "gaussian"]
    return [_task("E_violations", dict(kind="violation", violation=k, p=20, n=2000, K=REF_K, density=0.2), s,
                  ALL + INSTRUMENT)
            for k, s in product(kinds, range(15))]


def e_real():
    out = []
    for K, s in product([1, 5, 10], range(5)):
        methods = ALL + INSTRUMENT if K == REF_K else CUMULANT
        out.append(_task("E_real", dict(kind="real", dataset="danube_raw", K=K), s, methods))
    for s in range(5):
        out.append(_task("E_real", dict(kind="real", dataset="sachs_conditions", K=0), s, ALL + INSTRUMENT))
    return out


def e_scale():
    out = []
    for p, K, s in product([10, 20, 30, 50, 80], [5, 10, 20, 50], range(5)):
        methods = list(CUMULANT) + NULLS
        if K == REF_K:
            methods += ["NOTEARS-ADMM"]
        out.append(_task("E_scale", dict(kind="pgm", p=p, n=2000 * K, K=K, dist="mixed", density=0.2), s, methods,
                         ["raw"]))
    return out


GRIDS = {
    "P_main": p_main, "P_bnlearn": p_bnlearn, "E_gauss": e_gauss, "E_sample": e_sample,
    "E_sample2": e_sample2, "E_violations": e_violations, "E_real": e_real, "E_scale": e_scale,
}


def make_data(cell, seed):
    """Return (clients, B_true, true_order, extra) for a cell."""
    kind = cell["kind"]
    if kind == "pgm":
        c, B, o = D.pgm_cell(cell["p"], cell["n"], cell["K"], cell["dist"], cell["density"], seed)
        return c, B, o, {"regime": D.regime(cell["dist"])}
    if kind == "bnlearn":
        c, B, o = D.bnlearn_cell(cell["graph"], cell["n"], cell["K"], cell["dist"], seed)
        return c, B, o, {"regime": D.regime(cell["dist"])}
    if kind == "gauss":
        c, B, o = D.gaussianity_cell(cell["p"], cell["n"], cell["K"], cell["lam"], cell["density"], seed)
        return c, B, o, {}
    if kind == "violation":
        return D.violation_cell(cell["violation"], cell["p"], cell["n"], cell["K"], cell["density"], seed)
    if kind == "real":
        c, B, o, info = D.real_dataset(cell["dataset"], cell["K"], seed)
        extra = {"split": info["split"], "n_clients_actual": len(c)}
        extra.update({f"suff_{k}": v for k, v in info["sufficiency"].items()})
        return c, B, o, extra
    raise ValueError(kind)
