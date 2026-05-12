"""
Data Generation for HC-LiNGAM Experiments (Replication of Chen et al. 2025)
"""

import numpy as np
from scipy import stats, sparse
from typing import Tuple, List, Optional

class SyntheticDataGenerator:
    """
    Generates synthetic data according to the LiNGAM model with various noise distributions.
    
    Distributions from paper (Section VII.A):
    a) cubed Gaussian
    b) Pareto
    c) Logistic
    d) Uniform
    e) Laplace
    f) Poisson
    g) Exponential
    h) t-distribution
    i) Gamma
    j) Cubed Exponential
    k) Multiple distributions (heteroscedastic)
    """
    
    DISTRIBUTIONS = [
        'gaussian', 'gaussian_cubed', 'pareto', 'logistic', 'uniform', 
        'laplace', 'poisson', 'exponential', 't', 
        'gamma', 'exponential_cubed'
    ]
    
    SCALE_RANGES = [
        (0.5, 1.5), (10, 15), (100, 150), (1000, 1500)
    ]
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def _get_noise(self, n_samples: int, dist_type: str, scale: float) -> np.ndarray:
        """
        Generates noise using the scale parameter directly in the distribution.
        Does NOT normalize variance to 1, preserving the natural variance of the distribution.
        """
        if dist_type == 'gaussian':
            noise = self.rng.normal(0, scale, n_samples)

        elif dist_type == 'gaussian_cubed':
            # (N(0, 1))^3 * scale? Or N(0, scale)^3? 
            # Paper: "cubed Gaussian distribution"
            # User suggestion: noise = self.rng.normal(0, scale, n) ** 3
            # This makes sense if we consider the variable itself is Gaussian with scale, then cubed.
            noise = self.rng.normal(0, scale, n_samples) ** 3
        
        elif dist_type == 'pareto':
            # Pareto(b). Paper doesn't specify b, user suggests standard.
            # Scale usually acts as linear multiplier for Pareto?
            # Or x_m parameter? Scipy pareto is b / x^{b+1}. 
            # Let's use user's pattern: scale is a multiplier or parameter.
            # For pareto, variance is infinite if b <= 2. 
            b = 3.0
            noise = self.rng.pareto(b, n_samples) * scale
            
        elif dist_type == 'logistic':
            noise = self.rng.logistic(loc=0, scale=scale, size=n_samples)
            
        elif dist_type == 'uniform':
            # Var(U(-a, a)) = (2a)^2 / 12 = 4a^2/12 = a^2/3.
            # If we want something related to scale...
            # User suggestion: a = scale * sqrt(3) -> Var = scale^2
            a = scale * np.sqrt(3)
            noise = self.rng.uniform(-a, a, n_samples)
            
        elif dist_type == 'laplace':
            noise = self.rng.laplace(loc=0, scale=scale, size=n_samples)
            
        elif dist_type == 'poisson':
            # Poisson(lam). Var=lam.
            # If scale is the parameter, lam = scale?
            # Scale ranges are like [10, 15] or [1000, 1500].
            # Poisson with lam=1000 is very Gaussian.
            # We center it: P(lam) - lam.
            lam = scale
            noise = self.rng.poisson(lam, n_samples) - lam
            
        elif dist_type == 'exponential':
            # Exp(scale). Scale in numpy is 1/lambda (beta).
            # Var = scale^2.
            noise = self.rng.exponential(scale, n_samples) - scale
            
        elif dist_type == 't':
            # t(df). Var = df/(df-2) for df>2.
            # Scale is multiplier.
            df = 5
            noise = self.rng.standard_t(df, n_samples) * scale
            
        elif dist_type == 'gamma':
            # Gamma(k, theta). Var = k*theta^2.
            # User didn't specify k. Paper says "Gamma distribution".
            k = 2.0
            theta = scale 
            noise = self.rng.gamma(shape=k, scale=theta, size=n_samples) - k*theta
            
        elif dist_type == 'exponential_cubed':
            # (Exp(scale))^3
            x = self.rng.exponential(scale, n_samples)
            noise = np.power(x, 3) 
            # This will have massive mean, must center.
            
        else:
            raise ValueError(f"Unknown distribution: {dist_type}")
            
        # Center the noise (Critical for LiNGAM)
        return noise - np.mean(noise)

    def generate(
        self, 
        n_samples: int, 
        p: int, 
        graph_density: float = 1.0, 
        dist_type: str = 'exponential',
        heteroscedastic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Generate dataset X, adjacency matrix B, and topological order.
        Uses canonical topological order [0, 1, ..., p-1].
        """
        # 1. Canonical Topological Order
        topo_order = list(range(p))
        B = np.zeros((p, p))
        
        # 2. Generate DAG (Lower Triangular)
        # B[i, j] means j -> i. 
        # Since order is 0..p-1, we enforce i > j for edges.
        for i in range(p):
            for j in range(i):
                if self.rng.random() < graph_density:
                    # Coeffs: [-0.8, -0.2] U [0.2, 0.8]
                    coeff = self.rng.uniform(0.2, 0.8)
                    if self.rng.random() < 0.5:
                        coeff *= -1
                    
                    B[i, j] = coeff
                    
        # 3. Generate Noise (E)
        E = np.zeros((p, n_samples))
        
        for i in range(p):
            if heteroscedastic:
                current_dist = self.rng.choice(self.DISTRIBUTIONS)
            else:
                current_dist = dist_type
            
            # Sample scale parameter randomly
            scale_range = self.SCALE_RANGES[self.rng.integers(len(self.SCALE_RANGES))]
            scale = self.rng.uniform(*scale_range)
                
            E[i, :] = self._get_noise(n_samples, current_dist, scale)
            
        # 4. Generate X
        # X = (I - B)^-1 E
        # Since B is strictly lower triangular, I-B is triangular with diagonal 1.
        # We can solve triangular.
        # But brute force inversion is fine for p<=50.
        
        try:
            I = np.eye(p)
            A_inv = np.linalg.inv(I - B)
            X = np.dot(A_inv, E) # (p, n)
            X = X.T # (n, p)
        except np.linalg.LinAlgError:
            return self.generate(n_samples, p, graph_density, dist_type, heteroscedastic)
        
        return X, B, topo_order

