"""Data for every experiment: returns clients, the true B, and a true order.

Synthetic cells reuse the PGM generator and the PGM client split with the same
seeds, so the raw-data numbers of the published paper are reproducible from here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.benchmark.metrics import topological_order                         # noqa: E402

# ── copied verbatim from experiments/pgm26/benchmark.py ─────────────────────
# Importing that module loads bnlearn and its dependencies in every worker
# (~700 MB each). test_reproduce_pgm.py guards the copy against drift.

W_MIN, W_MAX = 0.5, 1.5


class PGMSyntheticDataGenerator:
    """
    Data generator strictly following DirectLiNGAM (2011) settings for the PGM benchmark:
    - Scales are static, NO massive heteroscedastic factors [1.5 => 1500].
    - Coefficients are strictly Uniform in [-1.5, -0.5] U [0.5, 1.5]
    - ER structure (Lower triangular random)
    """
    DISTRIBUTIONS = [
        'gaussian', 'gaussian_cubed', 'pareto', 'logistic', 'uniform', 
        'laplace', 'poisson', 'exponential', 't', 
        'gamma', 'exponential_cubed'
    ]
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def _get_noise(self, n_samples: int, dist_type: str, scale: float = 1.0) -> np.ndarray:
        if dist_type == 'gaussian': noise = self.rng.normal(0, scale, n_samples)
        elif dist_type == 'gaussian_cubed': noise = self.rng.normal(0, scale, n_samples) ** 3
        elif dist_type == 'pareto': noise = self.rng.pareto(3.0, n_samples) * scale
        elif dist_type == 'logistic': noise = self.rng.logistic(loc=0, scale=scale, size=n_samples)
        elif dist_type == 'uniform': noise = self.rng.uniform(-scale*np.sqrt(3), scale*np.sqrt(3), n_samples)
        elif dist_type == 'laplace': noise = self.rng.laplace(loc=0, scale=scale, size=n_samples)
        elif dist_type == 'poisson': noise = self.rng.poisson(scale, n_samples) - scale
        elif dist_type == 'exponential': noise = self.rng.exponential(scale, n_samples) - scale
        elif dist_type == 't': noise = self.rng.standard_t(5, n_samples) * scale
        elif dist_type == 'gamma': noise = self.rng.gamma(shape=2.0, scale=scale, size=n_samples) - 2.0*scale
        elif dist_type == 'exponential_cubed': noise = np.power(self.rng.exponential(scale, n_samples), 3)
        else: raise ValueError(f"Unknown distribution: {dist_type}")
        return noise - np.mean(noise)

    def generate(self, n_samples: int, p: int, graph_density: float = 1.0, dist_type: str = 'exponential') -> tuple:
        topo_order = list(range(p))
        B = np.zeros((p, p))
        for i in range(p):
            for j in range(i):
                if self.rng.random() < graph_density:
                    coeff = self.rng.uniform(0.5, 1.5)
                    if self.rng.random() < 0.5: coeff *= -1
                    B[i, j] = coeff
                    
        E = np.zeros((p, n_samples))
        
        valid_dists = [d for d in self.DISTRIBUTIONS if d not in ["gaussian"]]
        
        for i in range(p):
            # 1. Distribution shape
            if dist_type.startswith('mixed'):
                node_dist = str(self.rng.choice(valid_dists))
            else:
                node_dist = dist_type.replace('_hetero', '')
                
            # 2. Distribution scale
            if dist_type.endswith('_hetero'):
                # HC-LiNGAM / FedISHC heteroscedastic setup: 4 specific ranges for scale parameters
                scale_ranges = [(0.5, 1.5), (10, 15), (100, 150), (1000, 1500)]
                range_idx = self.rng.integers(0, len(scale_ranges))
                low, high = scale_ranges[range_idx]
                node_scale = self.rng.uniform(low, high)
            else:
                node_scale = 1.0
                
            E[i, :] = self._get_noise(n_samples, node_dist, scale=node_scale)
            
        try:
            X = np.dot(np.linalg.inv(np.eye(p) - B), E).T
        except np.linalg.LinAlgError:
            return self.generate(n_samples, p, graph_density, dist_type)
        return X, B, topo_order


def _split_data(X: np.ndarray, n_clients: int, rng) -> list:
    """Randomly split X into n_clients horizontal shards."""
    idx = np.array_split(rng.permutation(X.shape[0]), n_clients)
    return [X[i] for i in idx]


def _assign_weights(B_bin: np.ndarray, rng) -> np.ndarray:
    """Assign weights in [-W_MAX, -W_MIN] ∪ [W_MIN, W_MAX] to binary adjacency."""
    B = B_bin.copy()
    rows, cols = np.where(B_bin > 0)
    for i, j in zip(rows, cols):
        w = rng.uniform(W_MIN, W_MAX)
        B[i, j] = w if rng.random() > 0.5 else -w
    return B


def _generate_from_dag(B: np.ndarray, n: int, dist_type: str, rng):
    """Generate X = (I-B)^{-1} E using SyntheticDataGenerator noise logic."""
    p = B.shape[0]
    # Reuse SyntheticDataGenerator for noise generation logic but not state
    gen = PGMSyntheticDataGenerator(seed=int(rng.integers(1e9)))

    # Generate noise per variable (reuse internal _get_noise method)
    E = np.zeros((p, n))
    valid_dists = [d for d in gen.DISTRIBUTIONS if d not in ["gaussian"]]
    
    for i in range(p):
        # 1. Distribution shape
        if dist_type.startswith('mixed'):
            node_dist = str(rng.choice(valid_dists))
        else:
            node_dist = dist_type.replace('_hetero', '')
            
        # 2. Distribution scale
        if dist_type.endswith('_hetero'):
            scale_ranges = [(0.5, 1.5), (10, 15), (100, 150), (1000, 1500)]
            range_idx = rng.integers(0, len(scale_ranges))
            low, high = scale_ranges[range_idx]
            node_scale = rng.uniform(low, high)
        else:
            node_scale = 1.0
            
        E[i, :] = gen._get_noise(n, node_dist, scale=node_scale)

    try:
        X_pn = np.linalg.solve(np.eye(p) - B, E)
    except np.linalg.LinAlgError:
        X_pn = np.linalg.inv(np.eye(p) - B) @ E

    return X_pn.T   # (n, p)


FAMILIES = [d for d in PGMSyntheticDataGenerator.DISTRIBUTIONS if d != "gaussian"]
SYMMETRIC = ["logistic", "uniform", "laplace", "t"]
SKEWED = ["gaussian_cubed", "pareto", "poisson", "exponential", "gamma", "exponential_cubed"]


def regime(dist: str) -> str:
    base = dist.replace("_hetero", "")
    if base.startswith("mixed"):
        return "mixed"
    return "symmetric" if base in SYMMETRIC else "skewed"


def standardise(clients):
    """Marginal standardisation with pooled mean and sd (federable from second moments)."""
    X = np.vstack(clients)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    mu = X.mean(axis=0)
    return [(c - mu) / sd for c in clients]



def _random_B(rng, p, density, w=(0.5, 1.5)):
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            if rng.random() < density:
                c = rng.uniform(*w)
                B[i, j] = c if rng.random() < 0.5 else -c
    return B


# ── the PGM generator ───────────────────────────────────────────────────────

def pgm_cell(p, n, K, dist, density, seed):
    """Exactly the PGM benchmark's data and split for (p, n, K, dist, seed)."""
    rng = np.random.default_rng(seed)
    X, B, order = PGMSyntheticDataGenerator(seed=seed).generate(
        n_samples=n, p=p, graph_density=density, dist_type=dist)
    return _split_data(X, K, rng), B, list(order)


# ── bnlearn topologies ──────────────────────────────────────────────────────

def read_xbif(path):
    """Variable names in declaration order, and the binary B (``B[child, parent] = 1``)."""
    import xml.etree.ElementTree as ET
    net = ET.parse(path).getroot().find("NETWORK")
    names = [v.findtext("NAME").strip() for v in net.findall("VARIABLE")]
    index = {name: i for i, name in enumerate(names)}
    B = np.zeros((len(names), len(names)))
    for d in net.findall("DEFINITION"):
        child = index[d.findtext("FOR").strip()]
        for g in d.findall("GIVEN"):
            B[child, index[g.text.strip()]] = 1.0
    return names, B


def load_topology(name):
    return read_xbif(os.path.join(ROOT, "data", "BNs", f"{name}.xbif"))[1]


def bnlearn_cell(graph, n, K, dist, seed):
    """PGM bnlearn pipeline: fixed topology, random weights, PGM noise."""
    rng = np.random.default_rng(seed)
    B = _assign_weights(load_topology(graph), rng)
    X = _generate_from_dag(B, n, dist, rng)
    return _split_data(X, K, rng), B, topological_order(B)


# ── perturbations ───────────────────────────────────────────────────

def _mix(rng, B, E, n):
    return np.linalg.solve(np.eye(B.shape[0]) - B, E).T


def gaussianity_cell(p, n, K, lam, density, seed):
    """e = sqrt(1 - lam^2) N(0,1) + lam * standardised non-Gaussian."""
    rng = np.random.default_rng(1000 + seed)
    gen = PGMSyntheticDataGenerator(seed=seed)
    B = _random_B(rng, p, density)
    E = []
    for _ in range(p):
        s = gen._get_noise(n, str(rng.choice(FAMILIES)))
        s = (s - s.mean()) / max(s.std(), 1e-12)
        E.append(np.sqrt(max(1 - lam ** 2, 0.0)) * rng.normal(0, 1, n) + lam * s)
    X = _mix(rng, B, np.stack(E), n)
    return _split_data(X, K, np.random.default_rng(seed)), B, list(range(p))



NONLINEAR = {
    # tanh keeps a linear term near the origin, so it is reported with its linear
    # R^2 rather than trusted; the other two have no linear component at all.
    "tanh": np.tanh,
    "square": lambda x: x ** 2 - 1.0,
    "sin": lambda x: np.sin(2.0 * x),
}


def violation_cell(kind, p, n, K, density, seed, strength=0.5, n_latent=3):
    """One LiNGAM assumption broken at a time, plus genuinely nonlinear edges."""
    rng = np.random.default_rng(4000 + seed)
    gen = PGMSyntheticDataGenerator(seed=seed)
    B = _random_B(rng, p, density)
    noise = lambda: np.stack([gen._get_noise(n, str(rng.choice(FAMILIES))) for _ in range(p)])  # noqa: E731
    extra = {}

    if kind == "gaussian":
        X = _mix(rng, B, rng.normal(0, 1, (p, n)), n)
    elif kind == "clean":
        X = _mix(rng, B, noise(), n)
    elif kind == "measurement":
        X = _mix(rng, B, noise(), n)
        X = X + rng.normal(0, strength * X.std(axis=0), X.shape)
    elif kind == "confounded":
        E = noise()
        L = np.stack([gen._get_noise(n, str(rng.choice(FAMILIES))) for _ in range(n_latent)])
        load = np.zeros((p, n_latent))
        for l in range(n_latent):
            for t in rng.choice(p, size=min(3, p), replace=False):
                w = rng.uniform(0.5, 1.5)
                load[t, l] = w if rng.random() < 0.5 else -w
        X = _mix(rng, B, E + load @ L, n)
    elif kind.startswith("nonlinear_"):
        f = NONLINEAR[kind.split("_", 1)[1]]
        E = noise()
        X = np.zeros((n, p))
        nl = (rng.random((p, p)) < strength) & (B != 0)
        r2 = []
        for i in range(p):
            acc = E[i].copy()
            for j in np.flatnonzero(B[i]):
                xj = X[:, j]
                if nl[i, j]:
                    z = (xj - xj.mean()) / max(xj.std(), 1e-12)
                    fz = f(z)
                    r2.append(np.corrcoef(fz, xj)[0, 1] ** 2)
                    acc = acc + B[i, j] * xj.std() * fz
                else:
                    acc = acc + B[i, j] * xj
            X[:, i] = acc
        # How much of each nonlinear mechanism a straight line still explains.
        extra["linear_r2_nonlinear_edges"] = float(np.mean(r2)) if r2 else np.nan
    else:
        raise ValueError(kind)
    return _split_data(X, K, np.random.default_rng(seed)), B, list(range(p)), extra



# ── real data ───────────────────────────────────────────────────────────────

# Rows of the first nine Sachs reagent conditions, in file order: the pooled 7466-row
# file of `cdt` is their concatenation, so each condition is one client.
SACHS_CONDITIONS = (853, 902, 911, 723, 810, 799, 848, 913, 707)
# Column names of the Sachs CSV, mapped to the names of the reference network.
SACHS_NAMES = {"praf": "Raf", "pmek": "Mek", "plcg": "Plcg", "PIP2": "PIP2", "PIP3": "PIP3",
               "p44/42": "Erk", "pakts473": "Akt", "PKA": "PKA", "PKC": "PKC", "P38": "P38",
               "pjnk": "Jnk"}
# Days between retained rows of the raw Danube series (lag-1 autocorrelation 0.90).
DANUBE_THINNING = 7


def _sachs(data_dir):
    path =os.path.join(data_dir, "sachs_data_cdt_7466_11.csv")
    frame = pd.read_csv(path)
    if len(frame) != sum(SACHS_CONDITIONS):
        raise ValueError(f"expected {sum(SACHS_CONDITIONS)} rows, found {len(frame)}")
    values = np.log(frame.values.astype(float) + 1e-8)
    bounds = np.cumsum((0, *SACHS_CONDITIONS))
    clients = [values[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    reference, B_ref = read_xbif(os.path.join(data_dir, "BNs", "sachs.xbif"))
    # Reorder the reference network to the column order of the data file.
    perm = [reference.index(SACHS_NAMES[c]) for c in frame.columns]
    return clients, B_ref[np.ix_(perm, perm)]


def _danube(data_dir):
    frame = pd.read_csv(os.path.join(data_dir, "danube", "danube_data_raw.csv")).dropna()
    rows = frame.iloc[::DANUBE_THINNING].values.astype(float)
    edges = pd.read_csv(os.path.join(data_dir, "danube", "danube_edges.csv"))
    B = np.zeros((rows.shape[1],) * 2)
    for u, v in edges.values[:, :2].astype(int):         # upstream -> downstream, 1-based
        B[v - 1, u - 1] = 1.0
    return rows, B


def sufficiency(X, B):
    """Mean absolute correlation over the pairs the reference graph calls marginally independent.

    A causally sufficient DAG leaves those pairs uncorrelated; strong correlation there
    means a cause outside the graph drives both.
    """
    from experiments.benchmark.sid import descendant_matrix
    A = np.abs(B).T > 0                                  # A[u, v]: u -> v
    anc = descendant_matrix(A) | np.eye(len(A), dtype=bool)
    corr = np.abs(np.corrcoef(X.T))
    p = len(A)
    # d-separated by the empty set iff no node is an ancestor (or self) of both.
    v = np.array([corr[i, j] for i in range(p) for j in range(i + 1, p)
                  if not (anc[:, i] & anc[:, j]).any()])
    if not v.size:
        return {"pairs": 0.0, "mean_abs_r": np.nan, "median_abs_r": np.nan, "share_above_half": np.nan}
    return {"pairs": float(len(v)), "mean_abs_r": float(v.mean()),
            "median_abs_r": float(np.median(v)), "share_above_half": float((v > 0.5).mean())}


def real_dataset(name, K, seed):
    """Sachs split by reagent condition (natural), or the Danube series split at random into K.

    Returns (clients, B_true, true_order, info).
    """
    data_dir = os.path.join(ROOT, "data")
    if name == "sachs_conditions":
        clients, B = _sachs(data_dir)
        split = "natural"
    elif name == "danube_raw":
        rows, B = _danube(data_dir)
        clients, split = _split_data(rows, K, np.random.default_rng(seed)), "random"
    else:
        raise ValueError(name)
    info = {"split": split, "sufficiency": sufficiency(np.vstack(clients), B)}
    return clients, B, topological_order(B), info
