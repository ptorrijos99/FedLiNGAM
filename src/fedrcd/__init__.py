"""
FedRDC: Federated Regression-Directed Cumulants for Causal Discovery.

A federated causal discovery method that combines:
  - 2nd-order statistics (covariance) for exact regression
  - 3rd-order cumulants (tau) for source identification

Supports exact causal unlearning via statistics subtraction.
"""

from .core import fedrdc, fedrdc_noniterative, fedrdc_exact, ishc
from ..common.cumulants import compute_local_raw_stats, aggregate_and_compute_cumulants

__all__ = [
    'fedrdc',              # FedRCD:       iterative algebraic deflation (one-shot)
    'fedrdc_noniterative', # FedRCD-NI:    global τ sort, no deflation   (one-shot)
    'fedrdc_exact',        # FedRCD-Exact: multi-round, clients deflate local data
    'ishc',
    'compute_local_raw_stats', 'aggregate_and_compute_cumulants',
]

# Flower-dependent components (optional)
try:
    from .client import FedRDCClient, FedRDCExactClient
    from .strategy import FedRDCStrategy, FedRDCExactStrategy
    __all__ += ['FedRDCClient', 'FedRDCExactClient', 'FedRDCStrategy', 'FedRDCExactStrategy']
except ImportError:
    pass
