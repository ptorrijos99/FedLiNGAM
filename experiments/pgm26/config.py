"""
PGM 2026 — Experiment Configuration
=====================================
All parameters for the cumulant-based federated causal discovery benchmark.
Edit here to change the experimental grid; benchmark.py reads from these constants.

Experiment structure
--------------------
main        : p x K x 11 dists x 5 seeds   (n fixed at DEFAULT_N=2000)
n_ablation  : n x 11 dists x 5 seeds        (p=20, K=10 fixed)
bnlearn     : graph x K x 11 dists x 5 seeds (n=2000 fixed)
"""

# ---------------------------------------------------------------------------
# Main grid  (Table 1 in paper: p x K, n fixed)
# ---------------------------------------------------------------------------

P_VALUES         = [10, 20, 30, 50]    # dimensionalities (max 30)
MAIN_K_VALUES    = [1, 5, 10, 20, 50]   # federated clients (main table)
DEFAULT_N        = 2_000                 # fixed n for main grid
DEFAULT_P        = 20                    # fixed p for ablations
DEFAULT_K        = 10                    # fixed K for n-ablation / bnlearn

# Graph density (ER model — fraction of possible edges)
GRAPH_DENSITY    = 0.2    # ≈ 2 parents per node on average (ER-2)

# Edge weight range: uniform in [-W_MAX, -W_MIN] ∪ [W_MIN, W_MAX]
W_MIN, W_MAX     = 0.5, 1.5   # follows DirectLiNGAM (2011) benchmark

# Number of random seeds per cell
N_SEEDS          = 5
N_SEEDS_ABLATION = 20

# ---------------------------------------------------------------------------
# Noise distributions (11 families)
# ---------------------------------------------------------------------------
# "gaussian" is theoretically non-identifiable → excluded from aggregate Stats
# but kept in the config so we can run it as a robustness curiosity.

NOISE_DISTRIBUTIONS = [
    "gaussian",           # Non-identifiable; included for robustness check only
    "gaussian_cubed",     # Mild kurtosis
    "pareto",             # Heavy-tailed, skewed
    "logistic",           # Near-Gaussian symmetric
    "uniform",            # Bounded symmetric
    "laplace",            # Double-exponential (high kurtosis)
    "poisson",            # Discrete, positive skew
    "exponential",        # Skewed — most common in literature
    "t",                  # Medium-tailed (df=5)
    "gamma",              # Positive, skewed
    "exponential_cubed",  # Extreme skew / heavy tail
    "mixed",              # Each variable randomly assigned a distribution
]

# ---------------------------------------------------------------------------
# FedISHC Heteroscedastic Replication
# ---------------------------------------------------------------------------
HETERO_P_VALUES = [10, 20, 30, 50]
HETERO_K_VALUES = [1, 5, 10, 20, 50]
HETERO_N        = 2_000
HETERO_SEEDS    = N_SEEDS

EXCLUDE_FROM_AGGREGATE = ["gaussian"]   # excluded when averaging per-dist results

# ---------------------------------------------------------------------------
# n-ablation  (vary n, p=20, K=10)
# ---------------------------------------------------------------------------

N_ABLATION_VALUES = [250, 500, 1_000, 2_000, 5_000]
N_ABLATION_SEEDS  = N_SEEDS_ABLATION

# ---------------------------------------------------------------------------
# Density ablation  (vary density, p=20, K=10)
# ---------------------------------------------------------------------------
DENSITY_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4]
DENSITY_SEEDS  = N_SEEDS_ABLATION

# ---------------------------------------------------------------------------
# bnlearn experiments  (real graph structure, random weights + continuous noise)
# Mirrors main grid: sweeps K clients, n=2000 fixed, all 10 non-Gaussian dists.
# ---------------------------------------------------------------------------

BNLEARN_GRAPHS   = ["asia", "sachs", "child", "insurance", "water", "mildew", "alarm", "barley"]


BNLEARN_N        = 2_000   # fixed n (matches DEFAULT_N for comparability)
BNLEARN_K_VALUES = MAIN_K_VALUES   # same K sweep as main grid
BNLEARN_SEEDS    = N_SEEDS

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESULTS_DIR = "experiments/pgm26/results"
FIGURES_DIR = "experiments/pgm26/figures"

# ---------------------------------------------------------------------------
# n_clients convention
# ---------------------------------------------------------------------------
# Centralised baselines (HC, HCLiNGAM, ICALiNGAM, DirectLiNGAM) are stored
# with n_clients = -1 so that they are run once per (cell_tag, seed)
# regardless of which K value is being swept.
CENTRAL_N_CLIENTS = 1

# ---------------------------------------------------------------------------
# Methods to skip above a given p (too slow)
# ---------------------------------------------------------------------------
SKIP_ABOVE_P = {
    "DirectLiNGAM": 100,   # kernel-based independence test: O(n p^3)
    "ICALiNGAM":    100,  # FastICA convergence fragile at very high p
    "HC":           100,  # Hill Climbing can be slow at high P
}

