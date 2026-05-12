"""
FedRCD Flower Clients.

Two clients matching the two protocol families:

  FedRDCClient       — one-shot (FedRCD, FedRCD-NI)
      Sends raw moment sums once; server does all computation.

  FedRDCExactClient  — multi-round (FedRCD-Exact)
      Stateful client: receives (src_local, betas) from the server each
      round, deflates its local data in-place, then sends fresh stats.
      Raw data never leaves the client.
"""

import flwr as fl
import numpy as np
from typing import Dict, List, Tuple
from ..common.cumulants import compute_local_raw_stats


def _pack_stats(stats: dict) -> List[np.ndarray]:
    """Serialize raw moment stats into a flat list of arrays for Flower."""
    return [
        stats['sum_x'], stats['sum_xx'],
        stats['sum_x3'], stats['sum_x1y2'], stats['sum_x2y1'],
        stats['sum_x4'], stats['sum_x1y3'], stats['sum_x3y1'], stats['sum_x2y2'],
    ]


class FedRDCClient(fl.client.NumPyClient):
    """
    One-shot Flower client for FedRCD and FedRCD-NI.

    Sends raw moment sums once; server aggregates and runs the solver.
    Parameters from the server are ignored (no iterative model update).
    """

    def __init__(self, X: np.ndarray, client_id: str = ""):
        self.X = X
        self.client_id = client_id
        self.B_global = None

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return _pack_stats(compute_local_raw_stats(self.X, order=4))

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        return _pack_stats(compute_local_raw_stats(self.X, order=4)), self.X.shape[0], {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        if parameters:
            self.B_global = parameters[0]
        return 0.0, self.X.shape[0], {}


class FedRDCExactClient(fl.client.NumPyClient):
    """
    Multi-round Flower client for FedRCD-Exact.

    Maintains a working copy of the local data (X_curr) that is
    progressively deflated each round:

      Round 1: server sends nothing → client returns initial stats.
      Round r≥2: server sends (src_local, betas) via parameters/config
                 → client deflates X_curr, drops source column,
                 → returns fresh stats from the deflated data.

    Raw data never leaves the client; only sufficient statistics
    (moment sums) are transmitted.
    """

    def __init__(self, X: np.ndarray, client_id: str = ""):
        self.X_curr = X.copy()
        self.client_id = client_id
        self.B_global = None

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return _pack_stats(compute_local_raw_stats(self.X_curr, order=4))

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        If the server sent deflation instructions, apply them first.
        Then return fresh stats from the (updated) local data.
        """
        if parameters and 'src_local' in config:
            src_local = int(config['src_local'])
            betas = parameters[0]                     # shape (n_active,)

            n_active = self.X_curr.shape[1]
            x_s = self.X_curr[:, src_local].copy()
            remaining = [j for j in range(n_active) if j != src_local]
            self.X_curr = np.column_stack([
                self.X_curr[:, j] - betas[j] * x_s for j in remaining
            ])

        stats = compute_local_raw_stats(self.X_curr, order=4)
        return _pack_stats(stats), self.X_curr.shape[0], {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        if parameters:
            self.B_global = parameters[0]
        return 0.0, self.X_curr.shape[0], {}
