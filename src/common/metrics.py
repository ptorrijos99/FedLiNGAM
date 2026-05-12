"""
Metric Utilities for Causal Discovery (Common Library).

This module provides Tau scores and other metrics.
"""

import numpy as np
from typing import List, Tuple

from .cumulants import unbiased_cumulant4, cross_cumulants4

def compute_tau_matrix_3rd(
    c3: np.ndarray, 
    c12: np.ndarray, 
    c21: np.ndarray, 
    active_indices: List[int]
) -> np.ndarray:
    """
    Compute 3rd-order Tau matrix.
    """
    n_active = len(active_indices)
    tau = np.zeros((n_active, n_active))
    
    for idx_i, i in enumerate(active_indices):
        for idx_j, j in enumerate(active_indices):
            if i == j: continue
            
            # tau_ij = | C3(xi) * C12(xi, xj) - C21(xi, xj) * C12(xj, xi) |
            val = c3[i] * c12[i, j] - c21[i, j] * c12[j, i]
            tau[idx_i, idx_j] = np.abs(val)
            
    return tau

def compute_tau_matrix_4th(X: np.ndarray, active_indices: List[int] = None) -> np.ndarray:
    """
    Compute 4th-order Tau matrix.
    
    If active_indices provided, returns square matrix of that size.
    Else returns full p x p matrix for all variables in X.
    """
    n, p = X.shape
    if active_indices is None:
        active_indices = list(range(p))
    n_active = len(active_indices)
    
    tau = np.zeros((n_active, n_active))
    
    # Pre-compute C4 for active set
    c4_local = []
    for idx in active_indices:
        c4_local.append(unbiased_cumulant4(X[:, idx]))
    
    for i in range(n_active):
        real_i = active_indices[i]
        for j in range(n_active):
            if i == j: continue
            real_j = active_indices[j]
            
            c13, c31, c22 = cross_cumulants4(X[:, real_i], X[:, real_j])
            
            # tau_ij = | C4(xi) * C13(xi, xj) - C22(xi, xj) * C31(xi, xj) |
            val = c4_local[i] * c13 - c22 * c31
            tau[i, j] = np.abs(val)
            
    return tau

def compute_tau_matrix_4th_stats(
    c4: np.ndarray,
    c13: np.ndarray,
    c31: np.ndarray,
    c22: np.ndarray,
    active_indices: List[int]
) -> np.ndarray:
    """
    Compute 4th-order Tau matrix from pre-computed cumulants.
    
    Parameters
    ----------
    c4 : (p,) array - Marginal 4th cumulants C4(xi)
    c13 : (p, p) array - Cum(xi, xj, xj, xj)
    c31 : (p, p) array - Cum(xi, xi, xi, xj)
    c22 : (p, p) array - Cum(xi, xi, xj, xj)
    active_indices : List of active variable indices
    """
    n_active = len(active_indices)
    tau = np.zeros((n_active, n_active))
    
    for idx_i, i in enumerate(active_indices):
        for idx_j, j in enumerate(active_indices):
            if i == j: continue
            
            # tau_ij = | C4(xi) * C1,3(xi, xj) - C2,2(xi, xj) * C3,1(xi, xj) |
            # Check orientation of input matrices:
            # c13[i, j] should be Cum(xi, xj, xj, xj) -> C1,3(xi, xj)
            # c31[i, j] should be Cum(xi, xi, xi, xj) -> C3,1(xi, xj)
            # c22[i, j] should be Cum(xi, xi, xj, xj) -> C2,2(xi, xj)
            
            val = c4[i] * c13[i, j] - c22[i, j] * c31[i, j]
            tau[idx_i, idx_j] = np.abs(val)
            
    return tau

def permutation_test_4th(xi: np.ndarray, xj: np.ndarray, n_permutations: int = 1000) -> float:
    """
    Permutation test for 4th-order independence.
    """
    c4 = unbiased_cumulant4(xi)
    c13, c31, c22 = cross_cumulants4(xi, xj)
    tau_obs = np.abs(c4 * c13 - c22 * c31)
    
    count = 0
    xj_perm = xj.copy()
    
    for _ in range(n_permutations):
        np.random.shuffle(xj_perm)
        c13_p, c31_p, c22_p = cross_cumulants4(xi, xj_perm)
        tau_perm = np.abs(c4 * c13_p - c22_p * c31_p)
        
        if tau_perm >= tau_obs:
            count += 1
            
    return count / n_permutations
