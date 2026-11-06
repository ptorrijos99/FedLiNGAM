# FedISHC: Federated Iterative Source-identification via Higher-order Cumulants

FedISHC is a federated causal discovery algorithm that identifies the topological order and mixing matrix of a LiNGAM model using **3rd-order cumulants**. It is a statistics-only, one-shot communication algorithm designed for distributed privacy-preserving environments.

## 1. Problem Definition
The algorithm assumes the model $X = AX + E$, where $A$ is a lower-triangular mixing matrix. It aims to estimate $A$ by iteratively extracting source variables from aggregated 3rd-order moments.

## 2. Statistical Framework: 3rd-Order Cumulants
FedISHC focuses on the skewness of the variables. It uses three types of 3rd-order cumulants:
- $c_3(x_j) = \text{cum}(x_j, x_j, x_j)$ (Marginal skewness)
- $c_{12}(x_j, x_s) = \text{cum}(x_j, x_s, x_s)$
- $c_{21}(x_j, x_s) = \text{cum}(x_j, x_j, x_s)$

## 3. Algorithm Stages

### 3.1. Aggregation
1. **Local Phase**: Clients compute local raw 3rd-order moment sums.
2. **Aggregation**: Server computes global unbiased 3rd-order cumulants.

### 3.2. Iterative Identification
For $p-1$ iterations:
1. **Source Discovery**: Identify variable $s$ that minimizes the total dependence score $\sum_j \tau(s, j)$, where $\tau$ is the 3rd-order tau score.
2. **Influence Estimation**: Calculate the coefficient $\alpha_{js} = \frac{c_{21}(x_j, x_s)}{c_3(x_s)}$. This representing the direct effect of $s$ on $j$.
3. **Cumulant Deflation**: Update the remaining cumulants by subtracting the contribution of $s$:
   - $c_3(x_j)' = c_3(x_j) - \alpha_{js}^3 c_3(x_s)$
   - $c_{12}(x_j, x_k)' = c_{12}(x_j, x_k) - \alpha_{js} \alpha_{ks}^2 c_3(x_s)$
   - $c_{21}(x_j, x_k)' = c_{21}(x_j, x_k) - \alpha_{js}^2 \alpha_{ks} c_3(x_s)$
4. **Mixing Matrix Construction**: Store $\alpha_{js}$ in the mixing matrix $A$.

### 3.3. Structural Recovery
The final adjacency matrix $B$ is recovered using the identity:
$$B = I - A^{-1}$$
Where $A$ is the estimated lower-triangular mixing matrix (permuted according to the found order).

## 4. Federated Protocol
- **Communication Round**: 1 round (One-shot).
- **Communication Complexity**: $O(p^2)$ indices (sending $c_3, c_{12}, c_{21}$).

## 5. Analysis

### 5.1. Advantages
- **Simplicity**: Easy to implement and computationally efficient.
- **Low Communication**: Requires only the 3rd-order moment matrices.
- **Wait-free**: Clients only need to be online for one aggregation.

### 5.2. Disadvantages
- **Gaussian Limitation**: Cannot identify structure if noise distributions have zero skewness (Gaussian or symmetric distributions).
- **Instability**: The estimator $\alpha = c_{21}/c_3$ can be unstable if $c_3$ is small, leading to large estimation errors.
- **Error Propagation**: No covariance-based correction (unlike FedRCD or FedHybrid), which can lead to rapid error accumulation in the cumulant updates.

## 6. Complexity
- **Communication**: $1 \text{ round} \times O(p^2)$ floats.
- **Computational (Local)**: $O(n p^2)$.
- **Computational (Server)**: $O(p^3)$.
