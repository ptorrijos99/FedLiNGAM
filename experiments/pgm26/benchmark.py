"""
PGM 2026 — Main Benchmark Runner
==================================
Runs the cumulant-based federated causal discovery benchmark for
"Federated Causal Discovery via Shared Cumulant Statistics".

Modes
-----
main        -- p x K grid, n=2000 fixed (Table 1)
n_ablation  -- n ablation, p=20 K=5 fixed
bnlearn     -- real graph structure (asia/sachs/child/alarm) x K grid, n=2000
scalability -- extended K sweep [1..100], p=20 n=5000
smoke       -- 1 cell from each mode (quick sanity check)
all         -- main + n_ablation + bnlearn

Resume logic
------------
- Results are stored incrementally to CSV.
- On startup, existing CSV is loaded and processed keys are tracked.
- A cell is skipped entirely if ALL algorithms for that (cell_tag, seed)
  are already in the CSV -- data generation is not even triggered.
- Centralised baselines (HCLiNGAM, ICALiNGAM, DirectLiNGAM) are stored
  with n_clients=-1, so they are run ONCE per (cell_tag, seed) regardless
  of how many K values are swept.

Usage (from project root)
--------------------------
  python experiments/pgm26/benchmark.py --mode smoke
  python experiments/pgm26/benchmark.py --mode main
  python experiments/pgm26/benchmark.py --mode bnlearn
  python experiments/pgm26/benchmark.py --mode all
"""

import sys
import os
import io
import time
import warnings
import argparse
import traceback
import threading
from typing import List, Dict, Optional
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LassoLarsIC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Regressors in active set degenerate.")
warnings.filterwarnings("ignore", message="Early stopping the lars path")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# ── Config ────────────────────────────────────────────────────────────────────
from experiments.pgm26.config import (
    P_VALUES, MAIN_K_VALUES, DEFAULT_N, DEFAULT_P, DEFAULT_K,
    GRAPH_DENSITY, N_SEEDS, NOISE_DISTRIBUTIONS, EXCLUDE_FROM_AGGREGATE,
    N_ABLATION_VALUES, N_ABLATION_SEEDS,
    BNLEARN_GRAPHS, BNLEARN_N, BNLEARN_K_VALUES, BNLEARN_SEEDS,
    RESULTS_DIR, SKIP_ABOVE_P, CENTRAL_N_CLIENTS,
    W_MIN, W_MAX,
)

# ── Data / Evaluation utils ───────────────────────────────────────────────────
from experiments.utils.evaluation import count_accuracy_combined, count_accuracy_adjacency

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

# ── Core algorithm imports (no Flower needed for benchmarking) ─────────────────
from src.common.cumulants import compute_local_raw_stats, aggregate_and_compute_cumulants
from src.common.solvers import _estimate_B_from_cov, _estimate_B_lasso_from_cov
from src.fedishc.core import ishc as fedishc_solver, ishc_4th as fedishc4_solver, hc as fedhc_solver
from src.fedrcd.core import fedrdc, fedrdc_noniterative, fedrdc_exact

# ── Centralised baselines ─────────────────────────────────────────────────────
HAS_LINGAM = False
try:
    import lingam
    HAS_LINGAM = True
except ImportError:
    print("Warning: 'lingam' not found -> skipping ICALiNGAM / DirectLiNGAM.")

HAS_HCLINGAM = False
try:
    from src.hclingam.hclingam import HCLiNGAM
    HAS_HCLINGAM = True
except Exception:
    print("Warning: local HCLiNGAM not importable.")

# ── bnlearn ───────────────────────────────────────────────────────────────────
HAS_BNLEARN = False
try:
    import bnlearn as bn
    HAS_BNLEARN = True
except ImportError:
    print("Warning: 'bnlearn' not found -> bnlearn mode will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_data(X: np.ndarray, n_clients: int, rng) -> list:
    """Randomly split X into n_clients horizontal shards."""
    idx = np.array_split(rng.permutation(X.shape[0]), n_clients)
    return [X[i] for i in idx]


def _get_order(model, p: int) -> list:
    o = getattr(model, "causal_order_", None)
    return list(range(p)) if o is None else list(o)


def _get_adj(model, p: int) -> np.ndarray:
    return getattr(model, "adjacency_matrix_", np.zeros((p, p)))


def _get_detailed_edge_counts(B_true, B_est, threshold=0.05):
    """
    Decompose SHD into Extra, Missing, and Reversed.
    - Extra: Edge in B_est but no edge (any direction) in B_true.
    - Missing: Edge in B_true but no edge (any direction) in B_est.
    - Reversed: Edge exists in both but direction is swapped.
    """
    true_edges = (np.abs(B_true) > 1e-6)
    est_edges = (np.abs(B_est) > threshold)
    
    true_skeleton = true_edges | true_edges.T
    est_skeleton = est_edges | est_edges.T
    
    extra = np.sum(est_edges & ~true_skeleton)
    missing = np.sum(true_edges & ~est_skeleton)
    reversed_edges = np.sum(true_edges & est_edges.T)
    
    return int(extra), int(missing), int(reversed_edges)


def _compute_metrics(B_true, B_lasso_fixed, B_lasso_bic, true_order, est_order) -> dict:
    """Compute metrics using two uniform Lasso reconstructions.

    B_lasso_fixed : Lasso with fixed-λ heuristic.
                    Central methods: from X via _reconstruct_and_prune.
                    Federated methods: algorithm's own native output (already
                    computed with lmbda_type='fixed' from Σ — no duplication).
    B_lasso_bic   : Lasso with BIC-selected λ.
                    Central methods: LassoLarsIC from X (exact LARS path).
                    Federated methods: coordinate-descent BIC from Σ via
                    _reconstruct_and_prune_fed.
    """
    m_order = count_accuracy_combined(B_true, B_lasso_bic, true_order, est_order)

    m_fixed = count_accuracy_adjacency(B_true, B_lasso_fixed, threshold=1e-5)
    ex_f, ms_f, rv_f = _get_detailed_edge_counts(B_true, B_lasso_fixed, 1e-5)

    m_bic = count_accuracy_adjacency(B_true, B_lasso_bic, threshold=1e-5)
    ex_b, ms_b, rv_b = _get_detailed_edge_counts(B_true, B_lasso_bic, 1e-5)

    return {
        # Order-based
        "f1_reach":         m_order["f1_reach"],
        "f1_order":         m_order["f1_order"],

        # Lasso fixed-λ
        "lasso_fixed_shd":  m_fixed["shd"],
        "lasso_fixed_extra":ex_f,
        "lasso_fixed_miss": ms_f,
        "lasso_fixed_rev":  rv_f,
        "lasso_fixed_f1":   m_fixed["f1_edges"],
        "lasso_fixed_prec": m_fixed["precision_edges"],
        "lasso_fixed_rec":  m_fixed["recall_edges"],

        # Lasso BIC
        "lasso_bic_shd":    m_bic["shd"],
        "lasso_bic_extra":  ex_b,
        "lasso_bic_miss":   ms_b,
        "lasso_bic_rev":    rv_b,
        "lasso_bic_f1":     m_bic["f1_edges"],
        "lasso_bic_prec":   m_bic["precision_edges"],
        "lasso_bic_rec":    m_bic["recall_edges"],
    }


def _reconstruct_and_prune(X: np.ndarray, causal_order: list):
    """
    Given a causal order, compute two Lasso reconstructions from raw data X.

    Returns
    -------
    B_lasso_fixed : Lasso with fixed-λ heuristic (via covariance, consistent
                    with the federated approach in _reconstruct_and_prune_fed).
    B_lasso_bic   : Lasso with BIC-selected λ via exact LassoLarsIC LARS path.
    """
    n, p = X.shape
    mu = np.mean(X, axis=0)
    cov = (X - mu).T @ (X - mu) / max(n - 1, 1)

    # Lasso fixed-λ: computed from covariance (same formula as federated)
    B_lasso_fixed = _estimate_B_lasso_from_cov(causal_order, cov,
                                                n_samples=n, lmbda_type='fixed',
                                                prune_thres=1e-5)

    # Lasso-BIC: exact LARS path from X (better for central, has raw data)
    std_devs = np.std(X, axis=0)
    std_devs[std_devs == 0] = 1e-8
    X_std = (X - mu) / std_devs

    B_lasso_bic = np.zeros((p, p))
    for i in range(1, p):
        target  = causal_order[i]
        parents = causal_order[:i]
        y = X_std[:, target]
        Z = X_std[:, parents]
        try:
            reg = LassoLarsIC(criterion="bic", fit_intercept=True)
            reg.fit(Z, y)
            for k, p_idx in enumerate(parents):
                if abs(reg.coef_[k]) > 1e-5:
                    B_lasso_bic[target, p_idx] = reg.coef_[k] * (std_devs[target] / std_devs[p_idx])
        except Exception as e:
            if os.environ.get("PGM_DEBUG"):
                print(f"  [prune error] LassoLarsIC failed for node {target}: {e}")

    return B_lasso_fixed, B_lasso_bic


def _reconstruct_and_prune_fed(cov: np.ndarray, n: int, causal_order: list) -> np.ndarray:
    """
    Federated Lasso-BIC reconstruction from the global covariance matrix only.

    The fixed-λ Lasso (B_lasso_fixed) is NOT computed here — it is taken
    directly from the algorithm's native output (already computed from Σ
    inside _run_fed_algos to avoid duplication).

    Returns
    -------
    B_lasso_bic : Lasso-BIC from Σ (coordinate descent, BIC over a log-grid).
    """
    return _estimate_B_lasso_from_cov(causal_order, cov, n_samples=n,
                                       lmbda_type='bic', prune_thres=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Resume helpers (Thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

_csv_lock = threading.Lock()

def _init_results(filename: str):
    """Load existing CSV; return (records_list, processed_keys_set)."""
    with _csv_lock:
        if not os.path.exists(filename):
            return [], set()
        try:
            df = pd.read_csv(filename)
            records = df.to_dict("records")
            keys = {(r["cell"], r["seed"], r["method"]) for r in records}
            print(f"  Resuming: {len(df)} records loaded from {os.path.basename(filename)}")
            return records, keys
        except Exception as e:
            print(f"  Warning: could not read {filename} ({e}). Starting fresh.")
            return [], set()


def _save(records: list, filename: str):
    """Save records to CSV in a thread-safe way."""
    if not records:
        return
    with _csv_lock:
        _save_unlocked(records, filename)

def _save_unlocked(records: list, filename: str):
    """Save records to CSV without locking (caller must lock)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(records).to_csv(filename, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Cumulant statistics helpers (shared across modes)
# ─────────────────────────────────────────────────────────────────────────────

def _build_fed_stats(client_data: list, fed_needed: list = None) -> dict:
    """Compute 3rd and 4th-order cumulants from client shards."""
    local_3 = [compute_local_raw_stats(c, order=3) for c in client_data]
    agg_3   = aggregate_and_compute_cumulants(local_3)

    n_total = agg_4 = None
    if fed_needed is None or any(a in fed_needed for a in ["FedRCD", "FedRCD-NI", "FedISHC-4"]):
        local_4 = [compute_local_raw_stats(c, order=4) for c in client_data]
        agg_4   = aggregate_and_compute_cumulants(local_4)
        n_total = agg_4["n"]

    return {"agg_3": agg_3, "agg_4": agg_4, "n_total": n_total}


def _run_fed_algos(stats: dict, client_data: list, fed_needed: list = None) -> dict:
    """Run federated algorithms. FedRCD-Exact requires raw client_data."""
    results = {}
    if fed_needed is None: fed_needed = ALL_FED_ALGO_NAMES

    # FedHC
    if "FedHC" in fed_needed:
        legacy = {k: stats["agg_3"][k] for k in ("c3", "c12", "c21", "n")}
        t0 = time.time()
        causal_order_hc, _ = fedhc_solver(legacy)
        cov_hc = np.array(stats["agg_3"]["cov"], dtype=float)
        n_hc   = int(stats["agg_3"]["n"])
        B_hc   = _estimate_B_lasso_from_cov(causal_order_hc, cov_hc,
                                               n_samples=n_hc, lmbda_type="fixed")
        results["FedHC"] = (causal_order_hc, B_hc, time.time() - t0)

    # FedISHC
    if "FedISHC" in fed_needed:
        legacy = {k: stats["agg_3"][k] for k in ("c3", "c12", "c21", "n")}
        t0 = time.time()
        causal_order_ishc, _ = fedishc_solver(legacy)
        cov_ishc = np.array(stats["agg_3"]["cov"], dtype=float)
        n_ishc   = int(stats["agg_3"]["n"])
        B_ishc   = _estimate_B_lasso_from_cov(causal_order_ishc, cov_ishc,
                                               n_samples=n_ishc, lmbda_type="fixed")
        results["FedISHC"] = (causal_order_ishc, B_ishc, time.time() - t0)

    # FedISHC-4  (τ₄ identification + original α deflation — ablation variant)
    if "FedISHC-4" in fed_needed:
        stats4 = {k: stats["agg_4"][k] for k in ("c3", "c12", "c21", "c4", "c13", "c31", "c22", "n")}
        t0 = time.time()
        causal_order_ishc4, _ = fedishc4_solver(stats4)
        cov_ishc4 = np.array(stats["agg_4"]["cov"], dtype=float)
        n_ishc4   = int(stats["agg_4"]["n"])
        B_ishc4   = _estimate_B_lasso_from_cov(causal_order_ishc4, cov_ishc4,
                                               n_samples=n_ishc4, lmbda_type="fixed")
        results["FedISHC-4"] = (causal_order_ishc4, B_ishc4, time.time() - t0)

    # FedRCD
    if "FedRCD" in fed_needed:
        t0 = time.time()
        results["FedRCD"] = (*fedrdc(stats["agg_4"], adaptive=False, use_lasso=True, lmbda_type="fixed"),
                             time.time() - t0)

    # FedRCD-NI
    if "FedRCD-NI" in fed_needed:
        t0 = time.time()
        results["FedRCD-NI"] = (*fedrdc_noniterative(stats["agg_4"], adaptive=False, use_lasso=True,
                                                      lmbda_type="fixed"),
                                 time.time() - t0)


    # FedRCD-Exact (Iterative)
    if "FedRCD-Exact" in fed_needed:
        t0 = time.time()
        results["FedRCD-Exact"] = (*fedrdc_exact(client_data, use_lasso=True, lmbda_type="fixed"),
                                   time.time() - t0)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Core cell runner — synthetic data
# ─────────────────────────────────────────────────────────────────────────────

# We put the slowest algorithms at the end so partial results for faster methods are saved sooner.
ALL_FED_ALGO_NAMES    = ["FedHC", "FedISHC", "FedISHC-4", "FedRCD", "FedRCD-NI", "FedRCD-Exact"]
ALL_CENTRAL_NAMES     = ["HC", "HCLiNGAM", "ICALiNGAM", "DirectLiNGAM"]
ALL_ALGO_NAMES        = ALL_FED_ALGO_NAMES + ALL_CENTRAL_NAMES


def _cell_complete(cell_id: str, seed: int, processed: set) -> bool:
    """Return True if (cell_id, seed, name) for ALL algorithms is in processed."""
    # Check federated
    for a in ALL_FED_ALGO_NAMES:
        if (cell_id, seed, a) not in processed:
            return False
    return True

def _central_complete(cell_tag_k1: str, seed: int, processed: set, p: int) -> bool:
    """Return True if all appropriate centralized baselines are done for k=1."""
    skip_dir = SKIP_ABOVE_P.get("DirectLiNGAM", float("inf"))
    skip_ica = SKIP_ABOVE_P.get("ICALiNGAM",    float("inf"))
    skip_hc  = SKIP_ABOVE_P.get("HC",           float("inf"))

    expected = []
    if HAS_HCLINGAM:
        expected.append("HCLiNGAM")
        if p <= skip_hc:
            expected.append("HC")
    if HAS_LINGAM:
        if p <= skip_ica: expected.append("ICALiNGAM")
        if p <= skip_dir: expected.append("DirectLiNGAM")

    return all((cell_tag_k1, seed, a) in processed for a in expected)


def run_cell_synthetic(
    f_tag: str, c_tag: str, p: int, n: int, n_clients: int,
    dist_type: str, density: float, seed: int,
    records: list, processed: set, filename: str,
    save_every: int = 20
):
    """Run all algorithms on one synthetic (p, n, K, dist, seed) cell."""
    # Tags already provided

    # 1. Skip if everything done? (This is a bit complex now, we check parts)
    fed_done = _cell_complete(f_tag, seed, processed)
    central_done = _central_complete(c_tag, seed, processed, p)
    if fed_done and central_done:
        return

    # 2. Generate data (Reuse same data for Fed and Central)
    rng = np.random.default_rng(seed)
    gen = PGMSyntheticDataGenerator(seed=seed)
    try:
        X, B_true, true_order = gen.generate(
            n_samples=n, p=p, graph_density=density, dist_type=dist_type
        )
    except Exception as e:
        if os.environ.get("PGM_DEBUG"): traceback.print_exc()
        print(f"  [gen error] p={p} dist={dist_type} seed={seed}: {e}")
        return

    # 3. Centralised baselines (Save with n_clients=1)
    if not central_done:
        skip_dir = SKIP_ABOVE_P.get("DirectLiNGAM", float("inf"))
        skip_ica = SKIP_ABOVE_P.get("ICALiNGAM",    float("inf"))
        skip_hc  = SKIP_ABOVE_P.get("HC",           float("inf"))

        central_algos = {}
        if HAS_HCLINGAM and (c_tag, seed, "HCLiNGAM") not in processed:
            central_algos["HCLiNGAM"] = lambda: HCLiNGAM(method="hc-lingam").fit(X)
        if HAS_HCLINGAM and p <= skip_hc and (c_tag, seed, "HC") not in processed:
            central_algos["HC"] = lambda: HCLiNGAM(method="hc").fit(X)
        if HAS_LINGAM and p <= skip_ica and (c_tag, seed, "ICALiNGAM") not in processed:
            central_algos["ICALiNGAM"] = lambda: lingam.ICALiNGAM().fit(X)
        if HAS_LINGAM and p <= skip_dir and (c_tag, seed, "DirectLiNGAM") not in processed:
            central_algos["DirectLiNGAM"] = lambda: lingam.DirectLiNGAM().fit(X)

        for name, fn in central_algos.items():
            t0 = time.time()
            try:
                model = fn()
                runtime = time.time() - t0
                est_order = _get_order(model, p)
                try:
                    try:
                        B_lasso_fixed, B_lasso_bic = _reconstruct_and_prune(X, est_order)
                    except Exception:
                        B_lasso_fixed = _get_adj(model, p)
                        B_lasso_bic   = B_lasso_fixed
                    m = _compute_metrics(B_true, B_lasso_fixed, B_lasso_bic, true_order, est_order)
                    with _csv_lock:
                        records.append({
                            "cell": c_tag, "method": name, "seed": seed,
                            "graph": "synthetic", "p": p, "n": n, "n_clients": CENTRAL_N_CLIENTS,
                            "dist": dist_type, "density": density, "runtime": runtime, **m
                        })
                        processed.add((c_tag, seed, name))
                        if len(records) % save_every == 0:
                            _save_unlocked(records, filename)
                except Exception as e:
                    print(f"  [prune error] {name} {c_tag}: {e}")
            except Exception as e:
                print(f"  [{name}] p={p} dist={dist_type} seed={seed}: {e}")

    # 4. Federated algorithms
    if not fed_done:
        client_data = _split_data(X, n_clients, rng)
        fed_needed = [a for a in ALL_FED_ALGO_NAMES
                      if (f_tag, seed, a) not in processed]
        if fed_needed:
            try:
                stats = _build_fed_stats(client_data, fed_needed)
                fed_cov = stats["agg_3"]["cov"]
                fed_n   = stats["agg_3"]["n"]
                fed_results = _run_fed_algos(stats, client_data, fed_needed)
                for name in fed_needed:
                    if name not in fed_results: continue
                    causal_order, B_lasso_fixed, runtime = fed_results[name]
                    try:
                        try:
                            B_lasso_bic = _reconstruct_and_prune_fed(fed_cov, fed_n, causal_order)
                        except Exception:
                            B_lasso_bic = B_lasso_fixed
                        m = _compute_metrics(B_true, B_lasso_fixed, B_lasso_bic, true_order, causal_order)
                        with _csv_lock:
                            records.append({
                                "cell": f_tag, "method": name, "seed": seed,
                                "graph": "synthetic", "p": p, "n": n, "n_clients": n_clients,
                                "dist": dist_type, "density": density, "runtime": runtime, **m
                            })
                            processed.add((f_tag, seed, name))
                            if len(records) % save_every == 0:
                                _save_unlocked(records, filename)
                    except Exception as e:
                        print(f"  [prune error] {name} {f_tag}: {e}")
            except Exception as e:
                print(f"  [stats error] f_tag={f_tag} seed={seed}: {e}")

    if len(records) % save_every == 0:
        _save(records, filename)


# ─────────────────────────────────────────────────────────────────────────────
# bnlearn helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topological_sort(B: np.ndarray) -> list:
    """Kahn's algorithm: B[i,j] != 0 means j -> i (LiNGAM convention)."""
    p = B.shape[0]
    # In-degree: how many j have B[i,j] != 0?
    remaining = set(range(p))
    order = []
    while remaining:
        sources = [
            i for i in remaining
            if all(abs(B[i, j]) < 1e-10 for j in remaining if j != i)
        ]
        if not sources:
            # Numerical issue / cycle: pick min in-degree
            sources = [min(remaining,
                           key=lambda i: sum(abs(B[i, j]) > 1e-10
                                             for j in remaining if j != i))]
        node = sources[0]
        order.append(node)
        remaining.remove(node)
    return order


def _load_bnlearn_dag(name: str):
    """Load bnlearn DAG; return (B_binary, var_names, true_order).
    B_binary uses LiNGAM convention: B[i,j] != 0 means j -> i."""
    
    # Try local BIF first if it exists
    bif_path = os.path.join(ROOT, "data", "BNs", "BIF", f"{name}.bif")
    if os.path.exists(bif_path):
        import_name = bif_path
    else:
        import_name = name

    with redirect_stdout(io.StringIO()):
        model = bn.import_DAG(import_name, verbose=0)
        
    if model is None or len(model.keys()) == 0:
        raise ValueError(f"bnlearn could not load DAG '{name}' (import_name='{import_name}')")

    adj   = model["adjmat"]          # adj[src, dst] = True means src -> dst
    names = list(adj.columns)
    # Transpose to get LiNGAM convention: B[dst, src] != 0 means src -> dst
    B_bin = adj.values.astype(float).T
    true_order = _topological_sort(B_bin)
    return B_bin, names, true_order


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


def run_cell_bnlearn(
    f_tag: str, c_tag: str, graph_name: str, B_bin: np.ndarray,
    true_order_base: list, n: int, n_clients: int,
    dist_type: str, seed: int,
    records: list, processed: set, filename: str,
    save_every: int = 20
):
    """Run all algorithms on one bnlearn (graph, K, dist, seed) cell."""
    p = B_bin.shape[0]
    
    fed_done = _cell_complete(f_tag, seed, processed)
    central_done = _central_complete(c_tag, seed, processed, p)
    if fed_done and central_done:
        return

    rng    = np.random.default_rng(seed)
    B_true = _assign_weights(B_bin, rng)
    true_order = true_order_base

    try:
        X = _generate_from_dag(B_true, n, dist_type, rng)
    except Exception as e:
        print(f"  [gen error] {f_tag} seed={seed}: {e}")
        return

    # 1. Centralised
    if not central_done:
        skip_dir = SKIP_ABOVE_P.get("DirectLiNGAM", float("inf"))
        skip_ica = SKIP_ABOVE_P.get("ICALiNGAM",    float("inf"))
        skip_hc  = SKIP_ABOVE_P.get("HC",           float("inf"))

        central_algos = {}
        if HAS_HCLINGAM and (c_tag, seed, "HCLiNGAM") not in processed:
            central_algos["HCLiNGAM"] = lambda: HCLiNGAM(method="hc-lingam").fit(X)
        if HAS_HCLINGAM and p <= skip_hc and (c_tag, seed, "HC") not in processed:
            central_algos["HC"] = lambda: HCLiNGAM(method="hc").fit(X)
        if HAS_LINGAM and p <= skip_ica and (c_tag, seed, "ICALiNGAM") not in processed:
            central_algos["ICALiNGAM"] = lambda: lingam.ICALiNGAM().fit(X)
        if HAS_LINGAM and p <= skip_dir and (c_tag, seed, "DirectLiNGAM") not in processed:
            central_algos["DirectLiNGAM"] = lambda: lingam.DirectLiNGAM().fit(X)

        for name, fn in central_algos.items():
            t0 = time.time()
            try:
                model = fn()
                runtime = time.time() - t0
                est_order = _get_order(model, p)
                try:
                    try:
                        B_lasso_fixed, B_lasso_bic = _reconstruct_and_prune(X, est_order)
                    except Exception:
                        B_lasso_fixed = _get_adj(model, p)
                        B_lasso_bic   = B_lasso_fixed
                    m = _compute_metrics(B_true, B_lasso_fixed, B_lasso_bic, true_order, est_order)
                    with _csv_lock:
                        records.append({
                            "cell": c_tag, "method": name, "seed": seed,
                            "graph": graph_name, "p": p, "n": n, "n_clients": CENTRAL_N_CLIENTS,
                            "dist": dist_type,
                            "density": float(B_bin.sum())/(p*(p-1)) if p>1 else 0.0,
                            "runtime": runtime, **m
                        })
                        processed.add((c_tag, seed, name))
                        if len(records) % save_every == 0:
                            _save_unlocked(records, filename)
                except Exception as e:
                    print(f"  [prune error] {name} {c_tag}: {e}")
            except Exception as e:
                print(f"  [{name}] {c_tag} seed={seed}: {e}")

    # 2. Federated
    if not fed_done:
        client_data = _split_data(X, n_clients, rng)
        fed_needed = [a for a in ALL_FED_ALGO_NAMES
                      if (f_tag, seed, a) not in processed]
        if fed_needed:
            try:
                stats = _build_fed_stats(client_data, fed_needed)
                fed_cov = stats["agg_3"]["cov"]
                fed_n   = stats["agg_3"]["n"]
                fed_results = _run_fed_algos(stats, client_data, fed_needed)
                for name in fed_needed:
                    if name not in fed_results: continue
                    causal_order, B_lasso_fixed, runtime = fed_results[name]
                    try:
                        try:
                            B_lasso_bic = _reconstruct_and_prune_fed(fed_cov, fed_n, causal_order)
                        except Exception:
                            B_lasso_bic = B_lasso_fixed
                        m = _compute_metrics(B_true, B_lasso_fixed, B_lasso_bic, true_order, causal_order)
                        with _csv_lock:
                            records.append({
                                "cell": f_tag, "method": name, "seed": seed,
                                "graph": graph_name, "p": p, "n": n, "n_clients": n_clients,
                                "dist": dist_type,
                                "density": float(B_bin.sum())/(p*(p-1)) if p>1 else 0.0,
                                "runtime": runtime, **m
                            })
                            processed.add((f_tag, seed, name))
                            if len(records) % save_every == 0:
                                _save_unlocked(records, filename)
                    except Exception as e:
                        print(f"  [prune error] {name} {f_tag}: {e}")
            except Exception as e:
                print(f"  [stats error] {f_tag} seed={seed}: {e}")

    if len(records) % save_every == 0:
        _save(records, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark modes
# ─────────────────────────────────────────────────────────────────────────────

def run_main_benchmark(
    smoke: bool = False,
    dist_filter: Optional[str] = None,
    p_filter: Optional[int] = None,
    k_filter: Optional[int] = None,
    n_jobs: int = 1
):
    """Main grid: p x K x 10 dists x 10 seeds, n=2000 fixed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname   = os.path.join(RESULTS_DIR, "main_grid.csv")
    records, processed = _init_results(fname)

    p_vals  = [P_VALUES[0]]               if smoke else P_VALUES
    k_vals  = [MAIN_K_VALUES[0]]          if smoke else MAIN_K_VALUES
    dists   = [d for d in NOISE_DISTRIBUTIONS
               if d not in EXCLUDE_FROM_AGGREGATE][:2 if smoke else None]
    seeds   = [0]                         if smoke else range(N_SEEDS)

    if dist_filter:
        dists = [d for d in dists if d == dist_filter]
    if p_filter:
        p_vals = [p for p in p_vals if p == p_filter]
    if k_filter:
        k_vals = [k for k in k_vals if k == k_filter]

    total = len(p_vals) * len(k_vals) * len(dists) * len(list(seeds))
    print(f"\n=== Main Grid ({'SMOKE' if smoke else 'FULL'}) | "
          f"p={p_vals} K={k_vals} | {len(dists)} dists x {int(len(seeds))} seeds = {total} cells ===")

    
    tasks = []
    for p in p_vals:
        for dist in dists:
            for seed in seeds:
                tasks.append((p, dist, seed))

    def _process_task(task):
        p, dist, seed = task
        for k in k_vals:
            run_cell_synthetic(
                f_tag=f"p{p}_k{k}_{dist}",
                c_tag=f"p{p}_k1_{dist}",
                p=p, n=DEFAULT_N, n_clients=k,
                dist_type=dist, density=GRAPH_DENSITY,
                seed=seed,
                records=records, processed=processed, filename=fname
            )
            pbar.update(1)

    if n_jobs > 1:
        from joblib import Parallel, delayed
        print(f"Running in parallel with {n_jobs} jobs (by p, dist, seed)...")
        pbar = tqdm(total=total, desc="Main grid")
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_task)(t) for t in tasks
        )
    else:
        pbar = tqdm(total=total, desc="Main grid")
        for t in tasks: _process_task(t)
    pbar.close()

    _save(records, fname)
    print(f"\nDone. {len(records)} records -> {fname}")


def run_n_ablation(smoke: bool = False, dist_filter: Optional[str] = None, n_jobs: int = 1):
    """n ablation: various n, p=20, K=5 fixed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname   = os.path.join(RESULTS_DIR, "n_ablation.csv")
    records, processed = _init_results(fname)

    n_vals  = [N_ABLATION_VALUES[0]] if smoke else N_ABLATION_VALUES
    dists   = [d for d in NOISE_DISTRIBUTIONS
               if d not in EXCLUDE_FROM_AGGREGATE][:2 if smoke else None]
    
    if dist_filter:
        dists = [d for d in dists if d == dist_filter]
    
    seeds   = [0]                    if smoke else range(N_ABLATION_SEEDS)

    tasks = []
    for n in n_vals:
        for dist in dists:
            effective_seeds = range(10) if dist == 'mixed' else range(N_ABLATION_SEEDS)
            for seed in effective_seeds:
                tasks.append((n, dist, seed))

    total = len(tasks)
    print(f"\n=== n-Ablation | n={n_vals} | {len(dists)} dists | Total cells = {total} ===")

    def _process_task(task):
        n, dist, seed = task
        run_cell_synthetic(
            f_tag=f"nabl_n{n}_k{DEFAULT_K}_{dist}",
            c_tag=f"nabl_n{n}_k1_{dist}",
            p=DEFAULT_P, n=n, n_clients=DEFAULT_K,
            dist_type=dist, density=GRAPH_DENSITY,
            seed=seed,
            records=records, processed=processed, filename=fname
        )
        pbar.update(1)

    if n_jobs > 1:
        from joblib import Parallel, delayed
        print(f"Running in parallel with {n_jobs} jobs...")
        pbar = tqdm(total=total, desc="n-ablation")
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_task)(t) for t in tasks
        )
    else:
        pbar = tqdm(total=total, desc="n-ablation")
        for t in tasks: _process_task(t)
    pbar.close()

    _save(records, fname)
    print(f"\nDone. {len(records)} records -> {fname}")


def run_density_ablation(smoke: bool = False, dist_filter: Optional[str] = None, n_jobs: int = 1):
    """Density ablation: various graph densities, p=20, K=10 fixed."""
    from experiments.pgm26.config import DENSITY_VALUES, DENSITY_SEEDS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname   = os.path.join(RESULTS_DIR, "density_ablation.csv")
    records, processed = _init_results(fname)

    d_vals  = [DENSITY_VALUES[0]] if smoke else DENSITY_VALUES
    dists   = [d for d in NOISE_DISTRIBUTIONS
               if d not in EXCLUDE_FROM_AGGREGATE][:2 if smoke else None]
               
    if dist_filter:
        dists = [d for d in dists if d == dist_filter]
               
    seeds   = [0]                    if smoke else range(DENSITY_SEEDS)

    tasks = []
    for d in d_vals:
        for dist in dists:
            effective_seeds = range(10) if dist == 'mixed' else range(DENSITY_SEEDS)
            for seed in effective_seeds:
                tasks.append((d, dist, seed))

    total = len(tasks)
    print(f"\n=== Density Ablation | density={d_vals} | {len(dists)} dists | Total cells = {total} ===")

    def _process_task(task):
        d, dist, seed = task
        run_cell_synthetic(
            f_tag=f"dabl_d{d}_k{DEFAULT_K}_{dist}",
            c_tag=f"dabl_d{d}_k1_{dist}",
            p=DEFAULT_P, n=DEFAULT_N, n_clients=DEFAULT_K,
            dist_type=dist, density=d,
            seed=seed,
            records=records, processed=processed, filename=fname
        )
        pbar.update(1)

    if n_jobs > 1:
        from joblib import Parallel, delayed
        print(f"Running in parallel with {n_jobs} jobs...")
        pbar = tqdm(total=total, desc="density-ablation")
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_task)(t) for t in tasks
        )
    else:
        pbar = tqdm(total=total, desc="density-ablation")
        for t in tasks: _process_task(t)
    pbar.close()

    _save(records, fname)
    print(f"\nDone. {len(records)} records -> {fname}")


def run_hetero_benchmark(smoke: bool = False):
    """FedISHC Heteroscedastic Replication: large variance differences, mixed distributions."""
    from experiments.pgm26.config import HETERO_P_VALUES, HETERO_K_VALUES, HETERO_N, HETERO_SEEDS
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname   = os.path.join(RESULTS_DIR, "heteroscedastic.csv")
    records, processed = _init_results(fname)

    p_vals  = [HETERO_P_VALUES[0]]    if smoke else HETERO_P_VALUES
    k_vals  = [HETERO_K_VALUES[0]]    if smoke else HETERO_K_VALUES
    dists   = ["mixed_hetero"]
    seeds   = [0]                     if smoke else range(HETERO_SEEDS)

    total = len(p_vals) * len(k_vals) * len(list(seeds))
    print(f"\n=== Heteroscedastic Grid ({'SMOKE' if smoke else 'FULL'}) | "
          f"p={p_vals} K={k_vals} | 1 dist (mixed_hetero) x {int(len(seeds))} seeds = {total} cells ===")

    pbar = tqdm(total=total, desc="Hetero grid")
    for p in p_vals:
        for k in k_vals:
            for dist in dists:
                for seed in seeds:
                    run_cell_synthetic(
                        f_tag=f"hetero_p{p}_k{k}_{dist}",
                        c_tag=f"hetero_p{p}_k1_{dist}",
                        p=p, n=HETERO_N, n_clients=k,
                        dist_type=dist, density=GRAPH_DENSITY,
                        seed=seed,
                        records=records, processed=processed, filename=fname
                    )
                    pbar.update(1)
                
    pbar.close()
    _save(records, fname)
    print(f"\nDone. {len(records)} records -> {fname}")


def _load_real_sachs_data(filename="sachs.data.txt_853_11.gz"):
    """Load Sachs real data and its bnlearn ground truth from the local data folder."""
    data_path = os.path.join(ROOT, "data", filename)
    if not os.path.exists(data_path):
        print(f"Sachs data not found at {data_path}")
        return None, None, None

    # Load Data
    is_csv = filename.endswith('.csv')
    if is_csv:
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(data_path, sep=r'\s+', compression='gzip')
        
    X_df = (df - df.mean()) / df.std()
    var_names = list(df.columns)

    # Sachs Protein Mapping (p-protein -> Canonical)
    mapping = {
        'praf': 'Raf', 'pmek': 'Mek', 'plcg': 'Plcg', 
        'PIP2': 'PIP2', 'PIP3': 'PIP3', 'p44/42': 'Erk', 
        'pakts473': 'Akt', 'PKA': 'PKA', 'PKC': 'PKC', 
        'P38': 'P38', 'pjnk': 'Jnk'
    }
    
    # Map var_names to canonical if they exist in mapping
    mapped_names = [mapping.get(v, v) for v in var_names]

    # Load Ground Truth from bnlearn
    try:
        if not HAS_BNLEARN:
            raise ImportError("bnlearn required for ground truth.")
        model = bn.import_DAG('sachs', verbose=0)
        adj = model['adjmat']
        B_true = adj.values.astype(float).T
        bn_vars = list(adj.columns)
        
        # Align using mapped_names
        idx = [bn_vars.index(v) for v in mapped_names]
        B_true = B_true[np.ix_(idx, idx)]
        
        # Topological Sort for true_order
        true_order = _topological_sort(B_true)
            
        return X_df.values, B_true, true_order
    except Exception as e:
        print(f"Warning: Sachs Ground Truth alignment failed: {e}")
        return X_df.values, None, list(range(len(var_names)))


def run_real_sachs_benchmark(smoke: bool = False):
    """Real Data (Sachs): Observational biological dataset."""
    from experiments.pgm26.config import MAIN_K_VALUES, N_SEEDS
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname   = os.path.join(RESULTS_DIR, "real_sachs.csv")
    records, processed = _init_results(fname)

    k_vals  = [MAIN_K_VALUES[0]]    if smoke else MAIN_K_VALUES
    seeds   = [0]                   if smoke else range(N_SEEDS)
    
    # Run for both observational (853) and interventional (7466) if available, 
    # or just whichever we find locally. Let's do 853 for the smoke test.
    datasets = ["sachs.data.txt_853_11.gz"]
    if not smoke:
        datasets.append("sachs_data_cdt_7466_11.csv")

    for dataset_fname in datasets:
        X, B_bin, true_order = _load_real_sachs_data(dataset_fname)
        if X is None or B_bin is None:
            print(f"Skipping real Sachs benchmark for {dataset_fname}: Could not load data or DAG.")
            continue
            
        n_samples, p = X.shape

        total = len(k_vals) * len(list(seeds))
        print(f"\n=== Real Sachs ({dataset_fname} | {'SMOKE' if smoke else 'FULL'}) | "
              f"p={p} n={n_samples} K={k_vals} x {int(len(seeds))} seeds = {total} cells ===")

        pbar = tqdm(total=total, desc=f"Sachs {n_samples}")
        for k in k_vals:
            for seed in seeds:
                f_tag = f"real_sachs_{n_samples}_k{k}"
                c_tag = f"real_sachs_{n_samples}_k1"
            
                fed_done = _cell_complete(f_tag, seed, processed)
                central_done = _central_complete(c_tag, seed, processed, p)
                if fed_done and central_done:
                    pbar.update(1)
                    continue

                rng = np.random.default_rng(seed)
                # Shuffle data for the federated split to simulate random clients
                idx = rng.permutation(n_samples)
                X_shuffled = X[idx]

                # 1. Centralised
                if not central_done:
                    skip_dir = SKIP_ABOVE_P.get("DirectLiNGAM", float("inf"))
                    skip_ica = SKIP_ABOVE_P.get("ICALiNGAM",    float("inf"))
                    skip_hc  = SKIP_ABOVE_P.get("HC",           float("inf"))

                    central_algos = {}
                    if HAS_HCLINGAM and (c_tag, seed, "HCLiNGAM") not in processed:
                        central_algos["HCLiNGAM"] = lambda: HCLiNGAM(method="hc-lingam").fit(X_shuffled)
                    if HAS_HCLINGAM and p <= skip_hc and (c_tag, seed, "HC") not in processed:
                        central_algos["HC"] = lambda: HCLiNGAM(method="hc").fit(X_shuffled)
                    if HAS_LINGAM and p <= skip_ica and (c_tag, seed, "ICALiNGAM") not in processed:
                        central_algos["ICALiNGAM"] = lambda: lingam.ICALiNGAM().fit(X_shuffled)
                    if HAS_LINGAM and p <= skip_dir and (c_tag, seed, "DirectLiNGAM") not in processed:
                        central_algos["DirectLiNGAM"] = lambda: lingam.DirectLiNGAM().fit(X_shuffled)

                    for name, fn in central_algos.items():
                        t0 = time.time()
                        try:
                            model = fn()
                            runtime = time.time() - t0
                            est_order = _get_order(model, p)
                            try:
                                # Use the true binary matrix as B_true for SHD computation
                                try:
                                    B_lasso_fixed, B_lasso_bic = _reconstruct_and_prune(X_shuffled, est_order)
                                except Exception:
                                    B_lasso_fixed = _get_adj(model, p)
                                    B_lasso_bic   = B_lasso_fixed
                                m = _compute_metrics(B_bin, B_lasso_fixed, B_lasso_bic, true_order, est_order)
                                with _csv_lock:
                                    records.append({
                                        "cell": c_tag, "method": name, "seed": seed,
                                        "graph": "sachs_real", "p": p, "n": n_samples, "n_clients": CENTRAL_N_CLIENTS,
                                        "dist": "real", "density": float(B_bin.sum())/(p*(p-1)) if p>1 else 0.0,
                                        "runtime": runtime, **m
                                    })
                                    processed.add((c_tag, seed, name))
                                    if len(records) % 20 == 0:
                                        _save_unlocked(records, fname)
                            except Exception as e:
                                print(f"  [prune error] {name} {c_tag}: {e}")
                        except Exception as e:
                            print(f"  [{name}] {c_tag} seed={seed}: {e}")

                # 2. Federated
                if not fed_done:
                    client_data = _split_data(X_shuffled, k, rng)
                    fed_needed = [a for a in ALL_FED_ALGO_NAMES if (f_tag, seed, a) not in processed]
                    if fed_needed:
                        try:
                            stats = _build_fed_stats(client_data, fed_needed)
                            fed_cov = stats["agg_3"]["cov"]
                            fed_n   = stats["agg_3"]["n"]
                            fed_results = _run_fed_algos(stats, client_data, fed_needed)
                            for name in fed_needed:
                                if name not in fed_results: continue
                                causal_order, B_lasso_fixed, runtime = fed_results[name]
                                try:
                                    try:
                                        B_lasso_bic = _reconstruct_and_prune_fed(fed_cov, fed_n, causal_order)
                                    except Exception:
                                        B_lasso_bic = B_lasso_fixed
                                    m = _compute_metrics(B_bin, B_lasso_fixed, B_lasso_bic, true_order, causal_order)
                                    with _csv_lock:
                                        records.append({
                                            "cell": f_tag, "method": name, "seed": seed,
                                            "graph": "sachs_real", "p": p, "n": n_samples, "n_clients": k,
                                            "dist": "real", "density": float(B_bin.sum())/(p*(p-1)) if p>1 else 0.0,
                                            "runtime": runtime, **m
                                        })
                                        processed.add((f_tag, seed, name))
                                        if len(records) % 20 == 0:
                                            _save_unlocked(records, fname)
                                except Exception as e:
                                    print(f"  [prune error] {name} {f_tag}: {e}")
                        except Exception as e:
                            print(f"  [stats error] {f_tag} seed={seed}: {e}")

                pbar.update(1)
                
        if 'pbar' in locals() and hasattr(pbar, 'close'):
            pbar.close()
    _save(records, fname)
    print(f"\nDone. {len(records)} records -> {fname}")


def run_bnlearn_benchmark(smoke: bool = False, n_jobs: int = 1, graph_filter: str = None):
    """bnlearn mode: real graph structure x K x 10 dists x 10 seeds."""
    if not HAS_BNLEARN:
        print("Skipping bnlearn mode: 'bnlearn' package not installed.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname   = os.path.join(RESULTS_DIR, "bnlearn.csv")
    records, processed = _init_results(fname)

    graphs  = [BNLEARN_GRAPHS[0]]    if smoke else BNLEARN_GRAPHS
    if graph_filter:
        graphs = [g for g in graphs if g == graph_filter]
        
    k_vals  = [BNLEARN_K_VALUES[0]]   if smoke else BNLEARN_K_VALUES
    dists   = [d for d in NOISE_DISTRIBUTIONS
               if d not in EXCLUDE_FROM_AGGREGATE][:2 if smoke else None]
    seeds   = [0]                    if smoke else range(BNLEARN_SEEDS)

    # Pre-load all graph structures (do this once, not inside the loop)
    dag_cache = {}
    for g in graphs:
        try:
            B_bin, var_names, true_order = _load_bnlearn_dag(g)
            dag_cache[g] = (B_bin, var_names, true_order)
            print(f"  Loaded '{g}': p={B_bin.shape[0]}, arcs={int(B_bin.sum())}")
        except Exception as e:
            print(f"  Warning: could not load bnlearn DAG '{g}': {e}")

    total = len(dag_cache) * len(k_vals) * len(dists) * len(list(seeds))
    print(f"\n=== BNLearn ({'SMOKE' if smoke else 'FULL'}) | "
          f"graphs={list(dag_cache.keys())} K={k_vals} | "
          f"{len(dists)} dists x {len(list(seeds))} seeds = {total} cells ===")

    
    tasks = []
    for g_name, (B_bin, _, true_order) in dag_cache.items():
        for dist in dists:
            for seed in seeds:
                tasks.append((g_name, B_bin, true_order, dist, seed))
                
    def _process_task(task):
        g_name, B_bin, true_order, dist, seed = task
        for k in k_vals:
            run_cell_bnlearn(
                f_tag=f"bn_{g_name}_k{k}_{dist}",
                c_tag=f"bn_{g_name}_k1_{dist}",
                graph_name=g_name,
                B_bin=B_bin,
                true_order_base=true_order,
                n=BNLEARN_N,
                n_clients=k,
                dist_type=dist,
                seed=seed,
                records=records, processed=processed, filename=fname
            )
            pbar.update(1)

    if n_jobs > 1:
        from joblib import Parallel, delayed
        print(f"Running in parallel with {n_jobs} jobs (by graph, dist, seed)...")
        pbar = tqdm(total=total, desc="bnlearn")
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_task)(t) for t in tasks
        )
    else:
        pbar = tqdm(total=total, desc="bnlearn")
        for t in tasks: _process_task(t)
    pbar.close()

    _save(records, fname)
    print(f"\nDone. {len(records)} records -> {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Variance Stratification Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def run_variance_diagnostic(n_seeds: int = 30):
    """
    Variance Stratification Diagnostic.
    Tests if performance is driven by variance heterogeneity or causal asymmetry.
    """
    from experiments.utils.evaluation import compute_varsortability
    
    fname = os.path.join(RESULTS_DIR, "variance_stratification.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    p, n, k = 20, 2000, 10
    density = 0.2
    
    noise_families = {
        'symmetric':  ['uniform', 'laplace', 'logistic', 't'],
        'asymmetric': ['exponential', 'gamma', 'pareto'],
    }
    
    records, processed = _init_results(fname)
    
    def ancestral_accuracy(B_true, order):
        p = B_true.shape[0]
        A = (np.abs(B_true) > 1e-8).astype(int)
        reach = A.copy()
        for _ in range(p):
            reach = ((reach + reach @ A) > 0).astype(int)
        pos = {v: k for k, v in enumerate(order)}
        pairs, correct = 0, 0
        for i in range(p):
            for j in range(p):
                if reach[i, j]:
                    pairs += 1
                    if pos[j] < pos[i]:
                        correct += 1
        return correct / max(pairs, 1)

    def standardise_data(client_data):
        Xall = np.vstack(client_data)
        sd = Xall.std(axis=0)
        sd[sd == 0] = 1.0
        mu = Xall.mean(axis=0)
        return [(c - mu) / sd for c in client_data]

    all_methods = ["FedHC", "FedISHC", "FedISHC-4", "FedRCD", "FedRCD-NI", "FedRCD-Exact", "DirectLiNGAM"]

    for family, dists in noise_families.items():
        for dist in dists:
            print(f"Running killer experiment for {dist}...")
            pbar = tqdm(total=n_seeds, desc=f"{dist}")
            for seed in range(n_seeds):
                rng = np.random.default_rng(10_000 + seed)
                gen = PGMSyntheticDataGenerator(seed=seed)
                X, B_true, _ = gen.generate(n_samples=n, p=p, graph_density=density, dist_type=dist)
                
                clients_raw = _split_data(X, k, rng)
                clients_norm = standardise_data(clients_raw)
                
                for setting, data in [("raw", clients_raw), ("normalised", clients_norm)]:
                    X_stack = np.vstack(data)
                    v_score = compute_varsortability(B_true, np.var(X_stack, axis=0))
                    
                    # Get stats and run algos
                    fed_needed = [m for m in all_methods if m.startswith("Fed")]
                    stats = _build_fed_stats(data, fed_needed)
                    results = _run_fed_algos(stats, data, fed_needed)
                    
                    if "DirectLiNGAM" in all_methods and HAS_LINGAM:
                        t0 = time.time()
                        model = lingam.DirectLiNGAM().fit(X_stack)
                        results["DirectLiNGAM"] = (list(model.causal_order_), None, time.time() - t0)
                    
                    for name, res in results.items():
                        if (dist, seed, name, setting) in processed: continue
                        
                        causal_order = res[0]
                        acc = ancestral_accuracy(B_true, causal_order)
                        
                        with _csv_lock:
                            records.append({
                                "family": family, "dist": dist, "seed": seed,
                                "method": name, "setting": setting, 
                                "AncAcc": acc, "v": v_score
                            })
                            processed.add((dist, seed, name, setting))
                
                pbar.update(1)
                if (seed + 1) % 5 == 0:
                    _save(records, fname)
            pbar.close()

    _save(records, fname)
    print(f"\nVariance Stratification Diagnostic Done. Results in {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PGM 2026 benchmark — Federated Causal Discovery",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["main", "n_ablation", "density", "bnlearn", "hetero", "real_sachs", "killer", "smoke", "all"],
        default="main",
        help=(
            "main        -- p x K grid, n=2000 (Table 1)\n"
            "n_ablation  -- n ablation, p=20 K=10 fixed\n"
            "density     -- density ablation, p=20 K=10 fixed\n"
            "bnlearn     -- real graph structure (asia/sachs/child/alarm) x K grid\n"
            "hetero      -- heteroscedastic noise (FedISHC replication)\n"
            "real_sachs  -- real biological observational Sachs dataset\n"
            "killer      -- variance stratification diagnostic (killer experiment)\n"
            "smoke       -- 1 cell from each mode\n"
            "all         -- main + n_ablation + density + bnlearn + hetero + real_sachs + killer"
        ),
    )
    parser.add_argument("--dist", type=str, help="Filter by noise distribution (e.g. exponential)")
    parser.add_argument("--p", type=int, help="Filter by dimension p")
    parser.add_argument("--k", type=int, help="Filter by number of clients K")
    parser.add_argument("--graph", type=str, help="Filter by BNLearn graph name (e.g. child)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    
    args = parser.parse_args()

    if args.mode == "smoke":
        run_main_benchmark(smoke=True, dist_filter=args.dist, p_filter=args.p, k_filter=args.k, n_jobs=args.n_jobs)
        run_n_ablation(smoke=True, n_jobs=args.n_jobs)
        run_bnlearn_benchmark(smoke=True, n_jobs=args.n_jobs, graph_filter=args.graph)
        run_density_ablation(smoke=True, n_jobs=args.n_jobs)
        run_hetero_benchmark(smoke=True)
        run_real_sachs_benchmark(smoke=True)
        run_variance_diagnostic(n_seeds=2)
    elif args.mode == "all":
        run_main_benchmark(n_jobs=args.n_jobs)
        run_n_ablation(n_jobs=args.n_jobs)
        run_density_ablation(n_jobs=args.n_jobs)
        run_bnlearn_benchmark(n_jobs=args.n_jobs)
        run_hetero_benchmark()
        run_real_sachs_benchmark()
        run_variance_diagnostic()
    elif args.mode == "main":
        run_main_benchmark(dist_filter=args.dist, p_filter=args.p, k_filter=args.k, n_jobs=args.n_jobs)
    elif args.mode == "n_ablation":
        run_n_ablation(dist_filter=args.dist, n_jobs=args.n_jobs)
    elif args.mode == "density":
        run_density_ablation(dist_filter=args.dist, n_jobs=args.n_jobs)
    elif args.mode == "bnlearn":
        run_bnlearn_benchmark(n_jobs=args.n_jobs, graph_filter=args.graph)
    elif args.mode == "hetero":
        run_hetero_benchmark()
    elif args.mode == "real_sachs":
        run_real_sachs_benchmark()
    elif args.mode == "killer":
        run_variance_diagnostic()
        
if __name__ == "__main__":
    main()
