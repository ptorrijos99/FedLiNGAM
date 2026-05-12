"""
FedISHC Flower Strategy.

Aggregates raw moment sums from clients via ``aggregate_and_compute_cumulants``
and runs the ISHC solver. Stores per-client raw stats for exact unlearning.
"""

import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy

from .core import ishc
from ..common.cumulants import aggregate_and_compute_cumulants


class FedISHCStrategy(fl.server.strategy.Strategy):
    """
    One-shot federated strategy for FedISHC.

    Round 1: Clients send raw moment sums → Server aggregates → Solver.
    """

    def __init__(
        self,
        solver_fn: Callable = ishc,
        min_available_clients: int = 2,
    ):
        self.solver_fn = solver_fn
        self.min_available_clients = min_available_clients
        self.result_B: Optional[np.ndarray] = None
        self.result_order: Optional[List[int]] = None

        # Per-client raw stats for exact unlearning
        self.client_contributions: Dict[str, Dict] = {}
        self.global_stats: Optional[Dict] = None
        self.total_samples: int = 0

    # ── Flower interface ──────────────────────────────────────────────

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round > 1:
            return []
        clients = client_manager.sample(
            num_clients=self.min_available_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(c, fl.common.FitIns(parameters, {})) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # ── 1. Unpack raw stats from clients ──────────────────────────
        client_stats_list = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            n_k = fit_res.num_examples
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)

            # [sum_x, sum_xx, sum_x3, sum_x1y2, sum_x2y1]
            raw_stats = {
                'n': n_k,
                'sum_x': params[0],
                'sum_xx': params[1],
                'sum_x3': params[2],
                'sum_x1y2': params[3],
                'sum_x2y1': params[4],
            }

            # Store for unlearning
            self.client_contributions[cid] = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in raw_stats.items()
            }
            client_stats_list.append(raw_stats)

        # ── 2. Aggregate raw sums → compute global cumulants ──────────
        self.global_stats = aggregate_and_compute_cumulants(client_stats_list)
        self.total_samples = self.global_stats['n']

        # ── 3. Run ISHC solver ────────────────────────────────────────
        causal_order, B_est = self.solver_fn(self.global_stats)

        self.result_B = B_est
        self.result_order = causal_order

        # ── 4. Pack result ────────────────────────────────────────────
        parameters_out = fl.common.ndarrays_to_parameters([B_est])
        metrics = {
            "n_clients": len(results),
            "total_samples": self.total_samples,
        }
        return parameters_out, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        if server_round > 1:
            return []
        clients = client_manager.sample(
            num_clients=self.min_available_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(c, fl.common.EvaluateIns(parameters, {})) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        return 0.0, {}

    def evaluate(self, server_round, parameters):
        return None

    # ── Causal Unlearning ─────────────────────────────────────────────

    def unlearn(self, client_cid: str) -> Tuple[List[int], np.ndarray]:
        """
        Exact unlearning: remove client's raw stats and re-aggregate.
        """
        if client_cid not in self.client_contributions:
            raise ValueError(f"Client '{client_cid}' not found.")

        remaining_stats = [
            s for cid, s in self.client_contributions.items()
            if cid != client_cid
        ]
        if not remaining_stats:
            raise ValueError("Cannot unlearn: no samples would remain.")

        self.global_stats = aggregate_and_compute_cumulants(remaining_stats)
        self.total_samples = self.global_stats['n']

        causal_order, B_est = self.solver_fn(self.global_stats)
        self.result_B = B_est
        self.result_order = causal_order
        del self.client_contributions[client_cid]

        return causal_order, B_est

    def unlearn_batch(self, client_cids: List[str]) -> Tuple[List[int], np.ndarray]:
        """Remove multiple clients at once."""
        forget_set = set(client_cids)
        remaining_stats = [
            s for cid, s in self.client_contributions.items()
            if cid not in forget_set
        ]
        if not remaining_stats:
            raise ValueError("Cannot unlearn: no samples would remain.")

        self.global_stats = aggregate_and_compute_cumulants(remaining_stats)
        self.total_samples = self.global_stats['n']

        causal_order, B_est = self.solver_fn(self.global_stats)
        self.result_B = B_est
        self.result_order = causal_order
        for cid in client_cids:
            del self.client_contributions[cid]

        return causal_order, B_est

    def unlearn_instances(self, client_cid: str, delta_stats: dict,
                          mode: str = 'exact') -> Tuple[List[int], np.ndarray]:
        """
        Remove specific instances from a client's contribution.

        Parameters
        ----------
        client_cid : str
            The client whose instances are being forgotten.
        delta_stats : dict
            Raw moment sums for the forgotten instances.
        mode : str
            Ignored (always exact for one-shot algorithms).
        """
        if client_cid not in self.client_contributions:
            raise ValueError(f"Client '{client_cid}' not found.")

        stored = self.client_contributions[client_cid]
        new_n = stored['n'] - delta_stats['n']
        if new_n <= 0:
            raise ValueError("Cannot unlearn: client would have no samples remaining.")

        updated = {'n': new_n}
        for key in stored:
            if key == 'n':
                continue
            updated[key] = stored[key] - delta_stats[key]
        self.client_contributions[client_cid] = updated

        all_stats = list(self.client_contributions.values())
        self.global_stats = aggregate_and_compute_cumulants(all_stats)
        self.total_samples = self.global_stats['n']

        causal_order, B_est = self.solver_fn(self.global_stats)
        self.result_B = B_est
        self.result_order = causal_order
        return causal_order, B_est
