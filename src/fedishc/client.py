"""
FedISHC Flower Client.

Computes local raw moment sums for 3rd-order cumulant estimation and
sends them to the server. Uses the shared pipeline from common.cumulants,
which ensures correct global centering upon aggregation.
"""

import flwr as fl
import numpy as np
from typing import Dict, List, Tuple
from ..common.cumulants import compute_local_raw_stats


class CumulantClient(fl.client.NumPyClient):
    """
    Flower client for FedISHC.

    Sends raw moment sums (order=3) as parameters:
    [sum_x, sum_xx, sum_x3, sum_x1y2, sum_x2y1]
    """

    def __init__(self, X: np.ndarray):
        self.X = X
        self.B_global = None

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        stats = compute_local_raw_stats(self.X, order=3)
        return [
            stats['sum_x'], stats['sum_xx'],
            stats['sum_x3'], stats['sum_x1y2'], stats['sum_x2y1'],
        ]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Compute and return local raw moment sums.
        """
        stats = compute_local_raw_stats(self.X, order=3)
        params = [
            stats['sum_x'], stats['sum_xx'],
            stats['sum_x3'], stats['sum_x1y2'], stats['sum_x2y1'],
        ]
        return params, self.X.shape[0], {}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict]:
        if parameters:
            self.B_global = parameters[0]
        return 0.0, self.X.shape[0], {"status": "received_B"}
