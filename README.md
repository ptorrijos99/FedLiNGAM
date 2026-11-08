# Federated Causal Discovery via Regression-Directed Cumulants

Implementation and experiments of "Federated Causal Discovery via Regression-Directed Cumulants" (PGM 2026) and of its extended journal version.

## Overview

The code learns a linear non-Gaussian acyclic model from data partitioned across clients, which share only aggregated raw-moment statistics. The FedRCD family:

* **FedRCD-NI**: one round, a single sort of the fourth-order scores.
* **FedRCD-Def**: one round, algebraic deflation of the aggregated cumulants on the server.
* **FedRCD-X**: p-1 rounds, exact deflation of the data on the clients.

Baselines: FedISHC and FedHC (federated, third order), NOTEARS-ADMM (federated, continuous optimisation), and HC, HC-LiNGAM, DirectLiNGAM and ICA-LiNGAM (centralised).

## Layout

* `src/fedrcd/`: the FedRCD family (`core.py`), exact unlearning (`unlearn.py`) and the residual score used as an instrument in the extended article (`residual.py`).
* `src/fedishc/`: FedISHC and FedHC. The deflation coefficient is `kappa_{2,1}(x_s, x_j) / kappa_3(x_s)`; the PGM results for FedISHC were produced with its arguments swapped.
* `src/common/`: cumulant aggregation (`cumulants.py`), aggregation under vertical and hybrid partitions (`cumulants_partial.py`), the Gaussian mechanism (`dp.py`), and the solvers for B.
* `experiments/pgm26/`: the PGM 2026 benchmark.
* `experiments/benchmark/`: the grids, methods, metrics, figures and tables of the extended article. NOTEARS-ADMM is the authors' code (`third_party/notears-admm`, Apache 2.0).
* `experiments/deployment/`: the unlearning, privacy and record-erasure experiments of the extended article.
* `data/`: the network topologies and the two real datasets (see `data/README.md`).

## Installation

    pip install -r requirements.txt

## PGM 2026

    python experiments/pgm26/benchmark.py --mode main
    python experiments/pgm26/benchmark.py --mode stratification

## Extended version

Every synthetic draw is fixed by its cell and seed, so each grid regenerates the same data. Run a grid (shards go to `~/fedrcd_results`, or `FEDRCD_RESULTS`), merge it, then build the tables and figures:

    python -m experiments.benchmark.run --list
    python -m experiments.benchmark.run --exp P_main --jobs 8
    python -m experiments.benchmark.run --collect P_main
    python -m experiments.benchmark.tables
    python -m experiments.benchmark.figures

The grids are `P_main`, `P_bnlearn`, `E_gauss`, `E_sample`, `E_sample2`, `E_violations`, `E_real` and `E_scale`; `--chunk i --chunks n` runs one slice of a grid, for SLURM arrays. The deployment experiments are standalone scripts:

    python experiments/deployment/unlearning.py            # exact unlearning
    python experiments/deployment/screened_unlearning.py   # screening test of FedRCD-X
    python experiments/deployment/dp_one_round.py          # Gaussian mechanism, one-round methods
    python experiments/deployment/dp_fedrcdx.py            # Gaussian mechanism, FedRCD-X
    python experiments/deployment/dp_clients.py            # Gaussian mechanism added by each client
    python experiments/deployment/reconstruction.py        # reconstruction by subtraction
    python experiments/deployment/cohorts.py               # cohort aggregation
    python experiments/deployment/cohort_erasure.py        # erasure at cohort granularity
    python experiments/deployment/record_erasure.py        # batched record erasure

    python -m pytest tests                                 # vertical and hybrid aggregation
