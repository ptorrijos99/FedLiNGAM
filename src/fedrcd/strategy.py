"""
FedRCD Flower Strategies.

Two strategies matching the two protocol families:

  FedRDCStrategy      — one-shot (FedRCD, FedRCD-NI, ISHC)
      1 Flower round: clients send raw moment sums, server aggregates
      and runs the solver. Supports exact causal unlearning.

  FedRDCExactStrategy — multi-round (FedRCD-Exact)
      p-1 Flower rounds: each round the server identifies a source,
      computes β, and sends (src_local, β) to clients. Clients deflate
      their local data and return fresh stats. No algebraic approximation.
      Requires num_rounds = p-1 when calling fl.server.start_server().
"""

import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Parameters, Scalar, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from .core import fedrdc, fedrdc_noniterative, ishc
from ..common.cumulants import aggregate_and_compute_cumulants
from ..common.metrics import compute_tau_matrix_4th_stats
from ..common.solvers import _estimate_B_from_cov, _estimate_B_lasso_from_cov


class FedRDCStrategy(fl.server.strategy.Strategy):
    """
    One-shot federated strategy for FedRCD, FedRCD-NI and ISHC.

    Round 1: Clients send raw moment sums → Server aggregates → Server runs solver.
    """

    def __init__(
        self,
        solver: str = 'fedrdc',
        min_available_clients: int = 2,
    ):
        """
        Parameters
        ----------
        solver : str
            'fedrdc'    — iterative algebraic deflation (FedRCD)
            'fedrdc_ni' — global tau sort, no deflation  (FedRCD-NI)
            'ishc'      — improved ISHC baseline
        min_available_clients : int
            Minimum number of clients to wait for.
        """
        if solver == 'fedrdc':
            self.solver_fn = fedrdc
        elif solver == 'fedrdc_ni':
            self.solver_fn = fedrdc_noniterative
        elif solver == 'ishc':
            self.solver_fn = ishc
        else:
            raise ValueError(f"Unknown solver: {solver!r}. Use 'fedrdc', 'fedrdc_ni' or 'ishc'.")

        self.solver_name = solver
        self.min_available_clients = min_available_clients

        # Results
        self.result_B: Optional[np.ndarray] = None
        self.result_order: Optional[List[int]] = None

        # Per-client raw stats (for exact unlearning via subtraction)
        # Maps client_cid -> dict with raw moment sums + n
        self.client_contributions: Dict[str, Dict] = {}

        # Aggregated global statistics (cumulants computed from raw sums)
        self.global_stats: Optional[Dict] = None
        self.total_samples: int = 0

    # ── Flower interface ──────────────────────────────────────────────

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        if server_round > 1:
            return []

        clients = client_manager.sample(
            num_clients=self.min_available_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fl.common.FitIns(parameters, {})) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # ── 1. Unpack raw stats and store per-client contributions ────
        client_stats_list = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            n_k = fit_res.num_examples
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)

            # Reconstruct raw stats dict from the serialized array list
            # [sum_x, sum_xx, sum_x3, sum_x1y2, sum_x2y1,
            #  sum_x4, sum_x1y3, sum_x3y1, sum_x2y2]
            raw_stats = {
                'n': n_k,
                'sum_x': params[0],
                'sum_xx': params[1],
                'sum_x3': params[2],
                'sum_x1y2': params[3],
                'sum_x2y1': params[4],
            }
            # 4th-order stats (if present)
            if len(params) >= 9:
                raw_stats.update({
                    'sum_x4': params[5],
                    'sum_x1y3': params[6],
                    'sum_x3y1': params[7],
                    'sum_x2y2': params[8],
                })

            # Store for unlearning
            self.client_contributions[cid] = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in raw_stats.items()
            }

            client_stats_list.append(raw_stats)

        # ── 2. Aggregate raw sums → compute global cumulants ──────────
        self.global_stats = aggregate_and_compute_cumulants(client_stats_list)
        self.total_samples = self.global_stats['n']

        # ── 3. Run solver ─────────────────────────────────────────────
        causal_order, B_est = self.solver_fn(self.global_stats)

        self.result_B = B_est
        self.result_order = causal_order

        # ── 4. Pack result ────────────────────────────────────────────
        parameters_out = fl.common.ndarrays_to_parameters([B_est])
        metrics = {
            "solver": self.solver_name,
            "n_clients": len(results),
            "total_samples": self.total_samples,
        }
        return parameters_out, metrics

    def configure_evaluate(
        self, server_round, parameters, client_manager
    ) -> list:
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
        Remove a client's contribution and re-run the solver.

        This is EXACT unlearning: raw moment sums are subtracted and
        cumulants are recomputed from scratch, giving results identical
        to re-training without the client's data.

        Parameters
        ----------
        client_cid : str
            The client ID to "forget".

        Returns
        -------
        causal_order : List[int]
            Updated causal order.
        B : np.ndarray
            Updated adjacency matrix.
        """
        if client_cid not in self.client_contributions:
            raise ValueError(f"Client '{client_cid}' not found in contributions.")

        # Collect remaining client stats
        remaining_stats = [
            stats for cid, stats in self.client_contributions.items()
            if cid != client_cid
        ]

        if not remaining_stats:
            raise ValueError("Cannot unlearn: no samples would remain.")

        # Re-aggregate from scratch (exact)
        self.global_stats = aggregate_and_compute_cumulants(remaining_stats)
        self.total_samples = self.global_stats['n']

        # Re-run solver
        causal_order, B_est = self.solver_fn(self.global_stats)

        # Update state
        self.result_B = B_est
        self.result_order = causal_order
        del self.client_contributions[client_cid]

        return causal_order, B_est

    def unlearn_batch(self, client_cids: List[str]) -> Tuple[List[int], np.ndarray]:
        """Remove multiple clients at once."""
        for cid in client_cids:
            if cid not in self.client_contributions:
                raise ValueError(f"Client '{cid}' not found.")

        # Collect remaining client stats
        forget_set = set(client_cids)
        remaining_stats = [
            stats for cid, stats in self.client_contributions.items()
            if cid not in forget_set
        ]

        if not remaining_stats:
            raise ValueError("Cannot unlearn: no samples would remain.")

        # Re-aggregate from scratch (exact)
        self.global_stats = aggregate_and_compute_cumulants(remaining_stats)
        self.total_samples = self.global_stats['n']

        # Re-run solver
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


# ═══════════════════════════════════════════════════════════════════════
# Multi-round strategy for FedRCD-Exact
# ═══════════════════════════════════════════════════════════════════════

def _unpack_raw_stats(fit_res: FitRes) -> dict:
    """Deserialize raw moment arrays from a Flower FitRes."""
    n_k = fit_res.num_examples
    params = parameters_to_ndarrays(fit_res.parameters)
    return {
        'n': n_k,
        'sum_x':   params[0], 'sum_xx':  params[1],
        'sum_x3':  params[2], 'sum_x1y2': params[3], 'sum_x2y1': params[4],
        'sum_x4':  params[5], 'sum_x1y3': params[6],
        'sum_x3y1': params[7], 'sum_x2y2': params[8],
    }


class FedRDCExactStrategy(fl.server.strategy.Strategy):
    """
    Multi-round federated strategy for FedRCD-Exact.

    Runs p-1 Flower rounds:
      Round 1 : clients send initial stats (no deflation yet).
      Round r≥2: server sends (src_local, betas) to clients; clients
                 deflate and return fresh stats.

    Usage
    -----
    Pass ``num_rounds = p - 1`` to ``fl.server.start_server()``.

        strategy = FedRDCExactStrategy(p=8)
        fl.server.start_server(config=fl.server.ServerConfig(num_rounds=7),
                               strategy=strategy)
        order, B = strategy.result_order, strategy.result_B
    """

    def __init__(
        self,
        p: int,
        use_lasso: bool = False,
        lmbda_type: str = 'fixed',
        min_available_clients: int = 2,
    ):
        self.p = p
        self.use_lasso = use_lasso
        self.lmbda_type = lmbda_type
        self.min_available_clients = min_available_clients

        # State updated across rounds
        self.causal_order: List[int] = []
        self.active_global: List[int] = list(range(p))
        self.cov_original: Optional[np.ndarray] = None
        self.n_total: int = 0

        # Deflation instructions broadcast to clients in the NEXT round
        # (set in aggregate_fit, read in configure_fit of round+1)
        self._next_src_local: Optional[int] = None
        self._next_betas: Optional[np.ndarray] = None

        # Final results (available after the last round)
        self.result_order: Optional[List[int]] = None
        self.result_B: Optional[np.ndarray] = None

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        if server_round > self.p - 1:
            return []

        clients = client_manager.sample(
            num_clients=self.min_available_clients,
            min_num_clients=self.min_available_clients,
        )

        if server_round == 1:
            # First round: no deflation, just collect initial stats
            fit_ins = fl.common.FitIns(ndarrays_to_parameters([]), {})
        else:
            # Subsequent rounds: send deflation instructions
            fit_ins = fl.common.FitIns(
                ndarrays_to_parameters([self._next_betas]),
                {'src_local': self._next_src_local},
            )

        return [(c, fit_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # ── 1. Aggregate fresh stats from clients ─────────────────────
        client_stats = [_unpack_raw_stats(res) for _, res in results]
        agg = aggregate_and_compute_cumulants(client_stats)

        if server_round == 1:
            self.cov_original = np.array(agg['cov'])
            self.n_total = int(agg['n'])

        # ── 2. Identify source via τ₄ ─────────────────────────────────
        n_active = len(self.active_global)
        c4  = np.array(agg['c4'])
        c13 = np.array(agg['c13'])
        c31 = np.array(agg['c31'])
        c22 = np.array(agg['c22'])

        tau = compute_tau_matrix_4th_stats(c4, c13, c31, c22, list(range(n_active)))
        src_local = int(np.argmin(np.sum(tau, axis=1)))
        self.causal_order.append(self.active_global[src_local])

        # ── 3. Compute β for next round's deflation ───────────────────
        cov_curr = np.array(agg['cov'])
        var_s = max(float(cov_curr[src_local, src_local]), 1e-15)
        self._next_betas = cov_curr[:, src_local] / var_s
        self._next_src_local = src_local

        self.active_global = [g for i, g in enumerate(self.active_global) if i != src_local]

        # ── 4. If last round: add final variable and estimate B ───────
        if server_round == self.p - 1:
            if self.active_global:
                self.causal_order.append(self.active_global[0])

            if self.use_lasso:
                B = _estimate_B_lasso_from_cov(
                    self.causal_order, self.cov_original,
                    n_samples=self.n_total, lmbda_type=self.lmbda_type,
                )
            else:
                B = _estimate_B_from_cov(self.causal_order, self.cov_original)

            self.result_order = self.causal_order
            self.result_B = B
            return ndarrays_to_parameters([B]), {'causal_order': str(self.causal_order)}

        return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager) -> list:
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return 0.0, {}

    def evaluate(self, server_round, parameters):
        return None
