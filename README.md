# Federated Causal Discovery via Shared Cumulant Statistics (FedLiNGAM)

This repository contains the implementation and experiments for the paper "Federated Causal Discovery via Shared Cumulant Statistics" submitted to PGM 2026.

## Overview

The repository includes the implementation of:
- **FedISHC**: Federated Iterative Second-order High-order Cumulants.
- **FedRCD**: Federated Recursive Causal Discovery.
- **HC-LiNGAM**: Hill-Climbing LiNGAM (Baseline).

The code is designed to perform federated causal structure learning from horizontally partitioned data across multiple clients without sharing raw data, relying instead on shared aggregated cumulant statistics.

## Project Structure

- `src/`: Core implementation of the algorithms.
  - `fedishc/`: FedISHC implementation.
  - `fedrcd/`: FedRCD implementation.
  - `hclingam/`: HC-LiNGAM implementation.
  - `common/`: Shared utilities for cumulant computation and solving.
- `experiments/`: Scripts to reproduce the experimental results.
  - `pgm26/`: Main benchmark and diagnostic experiments.
    - `benchmark.py`: Main runner for experiments.
    - `config.py`: Experimental parameters and grid definitions.
  - `utils/`: Data generation and evaluation metrics.
- `data/`: Directory for datasets (structure only).

## Installation

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

To run the main benchmark:

```bash
python experiments/pgm26/benchmark.py --mode main
```

To run the variance stratification diagnostic:

```bash
python experiments/pgm26/benchmark.py --mode killer
```

Results are stored in the `experiments/pgm26/results/` directory as CSV files.
