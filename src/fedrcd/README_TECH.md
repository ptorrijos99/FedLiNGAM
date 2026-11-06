# FedRCD: Federated Regression-Directed Cumulants

Three variants of the same algorithm family, differing only in how
deflation is performed and how many communication rounds are required.

| Variant | Function | Rounds | Deflation | Centralised counterpart |
|---------|----------|--------|-----------|------------------------|
| **FedRCD-NI** | `fedrcd_noniterative` | 1 | none — global τ sort | HCLiNGAM (Algorithm 2) |
| **FedRCD** | `fedrcd` | 1 | algebraic update of cumulants | *(federated approximation, no exact CL counterpart)* |
| **FedRCD-Exact** | `fedrcd_exact` | p−1 | clients deflate local data | HC (Algorithm 1) |

---

## 1. Problem Definition

Recover causal structure $B$ and topological order $K$ from $C$ distributed
datasets $\{X_k\}_{k=1}^C$ without sharing raw data.  The model is
$x = Bx + e$ where $B$ is strictly lower-triangular and $e$ has
mutually independent, non-Gaussian components.

Aggregated quantities available to the server:

- $N = \sum_k n_k$ — total samples
- $\Sigma$ — global covariance
- $\kappa_3, \kappa_4$ — global 3rd- and 4th-order cumulants

---

## 2. Source Identification: Tau Scores

All three variants identify the source variable (most exogenous) as
the one minimising $T_\tau(i) = \sum_{j \neq i} \tau(i,j)$.

Two score variants (chosen adaptively by FedRCD and FedRCD-NI):

$$\tau_3(i,j) = \bigl|\kappa_3(x_i)\,\kappa_{12}(x_i,x_j) - \kappa_{21}(x_i,x_j)\,\kappa_{12}(x_j,x_i)\bigr|$$

$$\tau_4(i,j) = \bigl|\kappa_4(x_i)\,\kappa_{13}(x_i,x_j) - \kappa_{22}(x_i,x_j)\,\kappa_{31}(x_i,x_j)\bigr|$$

**Adaptive selection** (FedRCD, FedRCD-NI): use $\tau_4$ by default;
fall back to $\tau_3$ only if kurtosis $< 0.1$ and skewness $> 0.15$.

---

## 3. Deflation Strategies

### 3.1 FedRCD-NI — no deflation

Computes the full $\tau$ matrix once on the original aggregated
statistics and returns `argsort` of row sums.  Single communication
round, no iterative updates.

### 3.2 FedRCD — algebraic deflation on statistics

After identifying source $s$, updates the server-side cumulant arrays
in-place using the OLS coefficient $\beta_j = \Sigma_{js}/\Sigma_{ss}$:

**Covariance:**
$$\Sigma_{ij}' = \Sigma_{ij} - \beta_i\,\beta_j\,\Sigma_{ss}$$

**3rd-order marginal:**
$$\kappa_3(x_i)' = \kappa_3(x_i) - \beta_i^3\,\kappa_3(x_s)
  - 3\beta_i\,\kappa_{21}(x_i,x_s)
  + 3\beta_i^2\,\kappa_{12}(x_i,x_s)$$

**3rd-order cross (off-diagonal):**
$$\kappa_{12}(x_i,x_j)' = \kappa_{12}(x_i,x_j) - \beta_i\,\beta_j^2\,\kappa_3(x_s)$$
$$\kappa_{21}(x_i,x_j)' = \kappa_{21}(x_i,x_j) - \beta_i^2\,\beta_j\,\kappa_3(x_s)$$

**4th-order** analogous expansions (see `core.py`).

These formulas are exact under the true LiNGAM model and infinite
samples.  Under finite samples or misspecification, numerical errors
accumulate across the $p-1$ iterations — which is why FedRCD can
perform worse than the non-iterative FedRCD-NI in practice.

### 3.3 FedRCD-Exact — local data deflation (multi-round)

Exact federated counterpart of HC (Algorithm 1).  Each of the $p-1$
rounds proceeds as:

1. Clients compute fresh sufficient statistics from their **current**
   (already deflated) local data and send them to the server.
2. Server aggregates → identifies source $s$ via $\tau_4$ → computes
   $\beta_j = \Sigma_{js}/\Sigma_{ss}$ from the current covariance.
3. Server broadcasts $(s_{\text{local}},\, \boldsymbol{\beta})$.
4. Each client deflates locally:
   $x_j^{(r+1)} = x_j^{(r)} - \beta_j\, x_s^{(r)}$, drops column $s$.

Because fresh statistics are recomputed from actual residuals every
round, there is **no algebraic approximation and no error accumulation**.

---

## 4. B Matrix Estimation (shared by all three variants)

After the causal order $K$ is found, $B$ is re-estimated from the
**original** global covariance $\Sigma$ (saved before any deflation):

- **OLS**: $B_{\text{child}, \text{parents}} = \Sigma_{\text{pp}}^{-1}\,\Sigma_{\text{pc}}$
- **Adaptive Lasso** (`use_lasso=True`): coordinate-descent with
  OLS-initialised adaptive weights; optional BIC $\lambda$ selection.

This step is **fully federated**: $\Sigma$ was computed server-side
from aggregated raw moment sums $(\sum x_i,\, \sum x_i x_j)$ sent by
clients.  No raw data leaves any client at any point.

---

## 5. Communication Protocols

### FedRCD-NI and FedRCD (one-shot)

```
Client k                          Server
  compute_local_raw_stats(X_k) ──→ aggregate_and_compute_cumulants()
                                    fedrcd / fedrcd_noniterative()
                               ←── B (optional broadcast)
```

Flower classes: `FedRCDClient`, `FedRCDStrategy(solver='fedrcd'|'fedrcd_ni')`.

### FedRCD-Exact (multi-round)

```
Round 1:  Client sends initial stats ──→ Server finds s₀, computes β₀
Round 2:  Server sends (s₀,β₀)      ──→ Client deflates, sends fresh stats
          ...
Round p-1: Server finds s_{p-2}, adds last variable, estimates B
```

Flower classes: `FedRCDExactClient`, `FedRCDExactStrategy(p=p)`.
Requires `num_rounds = p - 1` in `fl.server.start_server()`.

---

## 6. Trade-offs

| Property | FedRCD-NI | FedRCD | FedRCD-Exact |
|----------|-----------|--------|--------------|
| Communication rounds | 1 | 1 | p−1 |
| Data transmitted | O(p²) stats | O(p²) stats | O(p²) stats × (p−1) rounds |
| Error accumulation | none | yes (finite sample) | none |
| Exact unlearning | ✓ (subtract raw sums) | ✓ | ✗ (stateful deflation) |
| Privacy level | same for all three — only sufficient statistics leave clients |

---

## 7. Complexity

| | FedRCD-NI | FedRCD | FedRCD-Exact |
|--|-----------|--------|--------------|
| Local compute | $O(np^2)$ | $O(np^2)$ | $O(np^2)$ per round |
| Server compute | $O(p^2)$ | $O(p^3)$ | $O(p^2)$ per round |
| Total comm. | $O(p^2)$ | $O(p^2)$ | $O(p^3)$ |
