import numpy as np
from typing import List, Tuple
from ..common.cumulants import unbiased_cumulant4, cross_cumulants4
from ..common.metrics import compute_tau_matrix_4th

class HCLiNGAM:
    """
    Implementation of HC and HC-LiNGAM algorithms for Causal Discovery.

    This class implements the methods described in Chen et al. (2025) for
    identifying causal structures in the presence of heteroscedastic noise.
    
    The algorithm typically proceeds in two steps:
    1.  **Causal Ordering**: Identifying the causal order of variables.
    2.  **Parameter Estimation**: Estimating the adjacency matrix (DAG) consistent with the order.

    Parameters
    ----------
    method : {'hc', 'hc-lingam'}, default='hc'
        The algorithm to use for causal ordering:
        - 'hc': Algorithm 1 (Iterative Regression). Similar to DirectLiNGAM but using 
          4th-order cumulant criteria (Tau) to identify exogenous variables.
        - 'hc-lingam': Algorithm 2 (Regression-free). Uses a global Tau score 
          to sort variables, which is more efficient but assumes the global score 
          ordering holds.

    Attributes
    ----------
    causal_order_ : list of int
        The estimated causal ordering of the variables. The i-th element 
        is the index of the variable at the i-th position in the causal order.
        
    adjacency_matrix_ : array-like of shape (n_features, n_features)
        The estimated adjacency matrix B, where B[i, j] represents the 
        coefficient of the edge j -> i.

    References
    ----------
    Chen, Z., et al. (2025). "Causal Discovery in the Presence of Heteroscedastic Noise."
    """
    def __init__(self, method: str = 'hc'):
        if method not in ['hc', 'hc-lingam']:
            raise ValueError("method must be 'hc' or 'hc-lingam'")
        self.method = method
        self.adjacency_matrix_ = None
        self.causal_order_ = None

    def fit(self, X: np.ndarray):
        """
        Fit the model to the dataset X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.array(X)
        n, p = X.shape

        # Step 1: Center the data
        X = X - np.mean(X, axis=0)

        # Step 2: Estimate Causal Order
        if self.method == 'hc-lingam':
            self.causal_order_ = self._fit_hc_lingam(X)
        elif self.method == 'hc':
            self.causal_order_ = self._fit_hc(X)
        else:
            raise ValueError(f"Unknown method {self.method}")

        # Step 3: Estimate Adjacency Matrix B based on Causal Order
        self.adjacency_matrix_ = self._estimate_adjacency_matrix(X, self.causal_order_)
        
        return self

    
    # _unbiased_cumulant4 and _cross_cumulants replaced by common library imports

    def _compute_tau_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise tau matrix.
        """
        return compute_tau_matrix_4th(X)

    def measure_cause_difference(self, xi: np.ndarray, xj: np.ndarray) -> float:
        """
        Calculate cause difference tau_ij from xi to xj (Definition 2).
        
        tau_ij = | C4(xi) * C1,3(xi, xj) - C2,2(xi, xj) * C3,1(xi, xj) |
        
        A value close to 0 indicates that xi -> xj is plausible under the 
        heteroscedastic noise model.
        """
        c4 = unbiased_cumulant4(xi)
        c13, c31, c22 = cross_cumulants4(xi, xj)
        return np.abs(c4 * c13 - c22 * c31)

    def permutation_test(self, xi: np.ndarray, xj: np.ndarray, n_permutations: int = 1000) -> float:
        """
        Perform permutation test to test H0: tau_ij = 0.
        """
        # Could use common.metrics.permutation_test_4th directly, but keeping wrapper for class consistency
        from ..common.metrics import permutation_test_4th
        return permutation_test_4th(xi, xj, n_permutations)

    def identify_pairwise_relationship(self, xi: np.ndarray, xj: np.ndarray, alpha: float = 0.05, n_permutations: int = 200) -> Tuple[str, float, float]:
        """
        Identify causal relationship between xi and xj based on hypothesis tests.

        Returns
        -------
        result : {'independent', 'i->j', 'j->i', 'undetermined'}
        p_ij : float
            p-value for H0: xi -> xj (tau_ij = 0)
        p_ji : float
            p-value for H0: xj -> xi (tau_ji = 0)
        """
        p_ij = self.permutation_test(xi, xj, n_permutations)
        p_ji = self.permutation_test(xj, xi, n_permutations)
        
        accept_H0_ij = p_ij > alpha # tau_ij = 0
        accept_H0_ji = p_ji > alpha # tau_ji = 0
        
        if accept_H0_ij and accept_H0_ji:
            return 'independent', p_ij, p_ji
        elif accept_H0_ij and not accept_H0_ji:
            return 'i->j', p_ij, p_ji
        elif not accept_H0_ij and accept_H0_ji:
            return 'j->i', p_ij, p_ji
        else:
            return 'undetermined', p_ij, p_ji

    def _fit_hc_lingam(self, X: np.ndarray) -> List[int]:
        """
        Algorithm 2: HC-LiNGAM (Regression-free global ordering).
        
        Sorts variables based on total Tau score. The variable with the 
        minimal total score is considered the most exogenous.
        """
        n, p = X.shape
        U = list(range(p))
        
        if p < 2:
            return U

        # 1. Calculate Tau Matrix (global)
        tau = self._compute_tau_matrix(X)
        
        # 2. Total cause difference score
        # T_tau(xi) = sum_{j != i} tau_ij
        # Sort indices by score ascending: minimal score -> root
        t_tau_scores = np.sum(tau, axis=1)
        
        causal_order = np.argsort(t_tau_scores)
        
        return causal_order.tolist()

    def _fit_hc(self, X: np.ndarray) -> List[int]:
        """
        Algorithm 1: HC (Iterative Regression).
        
        Identify one exogenous variable at a time, regress its effect out 
        from the remaining variables, and repeat.
        """
        n, p = X.shape
        K = []
        X_curr = X.copy()
        active_indices = list(range(p))
        
        for _ in range(p - 1):
            n_curr = len(active_indices)
            # Calculate Tau for current active set
            # We can use common function but it assumes full matrix or list of indices
            # Let's use the efficient vectorized approach from common if possible or manual
            
            # Since active_indices maps local index to global index:
            # X_curr is already the reduced dataset (n, n_curr).
            # We just need to compute tau on X_curr.
            tau_local = compute_tau_matrix_4th(X_curr)
            
            # Find root in current set (min row sum)
            t_tau = np.sum(tau_local, axis=1)
            m_local_idx = np.argmin(t_tau)
            m_global_idx = active_indices[m_local_idx]
            
            K.append(m_global_idx)
            
            # Remove effect of x_m from others (Regression)
            # x_j := x_j - (cov(xj, xm)/var(xm)) * xm
            xm_data = X_curr[:, m_local_idx]
            var_xm = np.var(xm_data)
            
            new_X = []
            new_active_indices = []
            
            for i in range(n_curr):
                if i == m_local_idx:
                    continue
                
                xi_data = X_curr[:, i]
                cov_im = np.cov(xi_data, xm_data, bias=True)[0, 1]
                coef = cov_im / var_xm
                resid = xi_data - coef * xm_data
                
                new_X.append(resid)
                new_active_indices.append(active_indices[i])
            
            X_curr = np.array(new_X).T
            active_indices = new_active_indices
            
        # Add the last remaining variable
        K.append(active_indices[0])
        
        return K

    def _estimate_adjacency_matrix(self, X: np.ndarray, order: List[int]) -> np.ndarray:
        """
        Estimate B given the causal order K using OLS regression.
        
        For each variable, regress it on its predecessors in the order.
        """
        p = X.shape[1]
        B = np.zeros((p, p))
        
        for i_idx, target_var in enumerate(order):
            if i_idx == 0:
                continue
                
            predictors_indices = order[:i_idx]
            
            # Regress X[:, target_var] on X[:, predictors_indices]
            y = X[:, target_var]
            A = X[:, predictors_indices]
            
            # OLS: coef = (A^T A)^-1 A^T y
            coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            
            # Fill B
            for k, split_idx in enumerate(predictors_indices):
                B[target_var, split_idx] = coefs[k]
                
        return B
