# Federated Causal Discovery via Regression-Directed Cumulants

This repository contains the implementation and experiments for the submission "Federated Causal Discovery via Regression-Directed Cumulants" to PGM 2026.

## Overview

The code performs federated causal structure learning from horizontally partitioned data across multiple clients without sharing raw data. It relies on shared aggregated cumulant statistics.

The repository includes the implementation of the FedRCD family of algorithms and several baselines:
* **FedRCD-NI**: Non-iterative global sort.
* **FedRCD-Def**: Iterative algebraic deflation on the server.
* **FedRCD-X**: Multi-round exact deflation on the clients.
* **FedISHC**: Baseline with third-order deflation.
* **FedHC**: Baseline without deflation.
* **HC-LiNGAM**: Centralised baseline.
* **HC**: Centralised baseline.

## Installation

Install the required dependencies:

    pip install -r requirements.txt

## Running Experiments

Run the main benchmark:

    python experiments/pgm26/benchmark.py --mode main

Run the variance stratification diagnostic:

    python experiments/pgm26/benchmark.py --mode stratification
