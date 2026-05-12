"""
Evaluation Metrics for HC-LiNGAM Experiments
Implements F1 score based on CAUSAL ORDER as specified in paper.

Paper evaluates "causal orders between two variables",
meaning PAIRWISE ORDERINGS, not individual edges.
"""

import numpy as np
from typing import Dict, List, Union


def compute_varsortability(
    B_true: np.ndarray, 
    variances: np.ndarray
) -> float:
    """
    Compute varsortability (v) as defined in Reisach et al. (2021).
    
    v is the fraction of directed paths i -> ... -> j such that Var(Xi) < Var(Xj).
    Standardization leads to v = 0.5 (by tie-breaking).
    
    Args:
        B_true: True adjacency matrix (p x p), where B[i,j] != 0 means j -> i.
        variances: Vector of marginal variances (p,).
        
    Returns:
        v: float in [0, 1]
    """
    p = B_true.shape[0]
    
    # 1. Compute Reachability (Transitive Closure)
    # R[i, j] = 1 if j is an ancestor of i
    adj_bool = (np.abs(B_true) > 1e-8).astype(int)
    R = np.eye(p, dtype=bool) | (adj_bool != 0)
    for _ in range(p.bit_length()): 
        R = R | (R @ R)
    
    # Remove diagonal (self-paths of length 0 are not part of the definition)
    np.fill_diagonal(R, False)
    
    # 2. Sum 'increasing' indicator over all paths (ancestor-descendant pairs)
    # v = sum_{i->j} [ Var(Xi) < Var(Xj) ] / sum_{i->j} [ 1 ]
    # In our matrix R: R[i, j] means j -> ... -> i
    # So we check Var(j) < Var(i)
    
    pairs = 0
    correct = 0.0
    
    for i in range(p):
        for j in range(p):
            if R[i, j]: # j is ancestor of i
                pairs += 1
                # prioritize tie-breaking for standardized data
                if np.isclose(variances[j], variances[i], atol=1e-5):
                    correct += 0.5
                elif variances[j] < variances[i]:
                    correct += 1.0
                    
    if pairs == 0:
        return 1.0 # Vacuously varsortable
        
    return correct / pairs


def count_accuracy_order(
    true_order: List[int], 
    estimated_order: List[int]
) -> Dict[str, float]:
    """
    Compute F1 score based on CAUSAL ORDER (pairwise orderings).
    
    This is the CORRECT interpretation of the paper's metric.
    
    For a causal order [0, 1, 2, 3, 4]:
    - Implies 0→1, 0→2, 0→3, 0→4, 1→2, 1→3, 1→4, 2→3, 2→4, 3→4
    - Total: C(5,2) = 10 pairwise orderings
    
    Args:
        true_order: Ground truth causal order
        estimated_order: Estimated causal order
        
    Returns:
        Dictionary with f1, precision, recall
        
    Example:
        true_order = [0, 1, 2, 3, 4]
        estimated_order = [0, 2, 1, 3, 4]
        
        True pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        Est pairs:  (0,2), (0,1), (0,3), (0,4), (2,1), (2,3), (2,4), (1,3), (1,4), (3,4)
        
        Correct: (0,1), (0,2), (0,3), (0,4), (1,3), (1,4), (2,3), (2,4), (3,4) = 9/10
        
        F1 = 0.9
    """
    p = len(true_order)
    
    if p != len(estimated_order):
        raise ValueError("Orders must have same length")
    
    # Generate all pairwise orderings
    true_pairs = set()
    for i in range(p):
        for j in range(i + 1, p):
            # In true_order, position i comes before position j
            # So true_order[i] → true_order[j]
            true_pairs.add((true_order[i], true_order[j]))
    
    estimated_pairs = set()
    for i in range(p):
        for j in range(i + 1, p):
            estimated_pairs.add((estimated_order[i], estimated_order[j]))
    
    # Calculate metrics
    tp = len(true_pairs & estimated_pairs)  # Correctly ordered pairs
    
    # Precision and Recall
    precision = tp / len(estimated_pairs) if len(estimated_pairs) > 0 else 0.0
    recall = tp / len(true_pairs) if len(true_pairs) > 0 else 0.0
    
    # F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'total_true_pairs': len(true_pairs),
        'total_estimated_pairs': len(estimated_pairs)
    }

def count_accuracy_reachability(
    B_true: np.ndarray, 
    estimated_order: List[int]
) -> Dict[str, float]:
    """
    Compute reachability accuracy metrics using vectorized operations.
    
    Optimized to O(p^3) for matrix multiplication but implemented
    in optimized C/Fortran backend via numpy, avoiding Python loops.
    """
    p = B_true.shape[0]
    
    # 1. Compute Reachability (Transitive Closure)
    # Using matrix power series or simple multiplication for small graphs.
    # For DAGs, (I + |B|)^p > 0 works to find all paths.
    adj_bool = (np.abs(B_true) > 1e-6).astype(int)
    # Fast reachability for small-medium p:
    # R = (I + A)^p. For strictly binary/bool, we can just loop matmul.
    R = np.eye(p, dtype=bool) | (adj_bool != 0)
    for _ in range(p.bit_length()): # Logarithmic squaring
        R = R | (R @ R)
    
    # Remove diagonal (self-loops not needed for ordering)
    np.fill_diagonal(R, False)

    # 2. Reorder R according to estimated_order
    # If estimated_order is correct, all true connections (R[i, j]=1)
    # should appear in the upper triangle (i < j).
    idx = np.array(estimated_order)
    # Create the permuted matrix: rows/cols sorted by estimated rank
    R_permuted = R[np.ix_(idx, idx)]

    # 3. Calculate metrics
    # TP: True connections that appear in the lower triangle of the permuted matrix
    # (i.e., if j is ancestor of i, idx.index(j) < idx.index(i)).
    # Because R[i, j] = 1 means j is ancestor of i, in the permuted matrix R_permuted[a, b] = 1 
    # means idx[b] is ancestor of idx[a]. For a correct order, b < a.
    tp = np.sum(np.tril(R_permuted, k=-1))
    
    # Total connections in the true graph (mandatory pairs)
    total_mandatory = np.sum(R)
    
    # For total orders on DAGs, Precision = Recall = F1
    score = tp / total_mandatory if total_mandatory > 0 else 1.0
    
    return {
        'f1': score,
        'precision': score,
        'recall': score,
        'tp': int(tp),
        'total_mandatory': int(total_mandatory)
    }


def count_accuracy_adjacency(
    B_true: np.ndarray, 
    B_est: np.ndarray,
    threshold: float = 1e-6
) -> Dict[str, float]:
    """
    Compute metrics based on ADJACENCY MATRIX (edges).
    
    Args:
        B_true: True adjacency matrix (p x p)
        B_est: Estimated adjacency matrix (p x p)
        threshold: Threshold for considering estimated edge as present
        
    Returns:
        Dictionary with f1, precision, recall, shd
    """
    # Binarize matrices
    true_edges = (np.abs(B_true) > 1e-6)
    est_edges = (np.abs(B_est) > threshold)
    
    # Calculate TP, FP, FN
    tp = np.sum(true_edges & est_edges)
    fp = np.sum(est_edges & ~true_edges)
    fn = np.sum(~est_edges & true_edges)
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance
    shd = np.sum(true_edges ^ est_edges)
    
    return {
        'f1_edges': f1,
        'precision_edges': precision,
        'recall_edges': recall,
        'shd': shd,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def count_accuracy_combined(
    B_true: np.ndarray,
    B_est: np.ndarray,
    true_order: List[int],
    estimated_order: List[int]
) -> Dict[str, float]:
    """
    Compute BOTH order-based and edge-based metrics.
    
    Returns combined dictionary with all metrics.
    Primary metric (as in paper): F1 based on order.
    Auxiliary metric: F1 based on edges.
    """
    # Order-based metrics (STRICT - ALL PAIRS)
    order_metrics = count_accuracy_order(true_order, estimated_order)
    
    # Reachability-based metrics (FLEXIBLE - ONLY MANDATORY PAIRS)
    reach_metrics = count_accuracy_reachability(B_true, estimated_order)
    
    # Edge-based metrics (AUXILIARY)
    edge_metrics = count_accuracy_adjacency(B_true, B_est)
    
    # Combine with clear naming
    combined = {
        # Primary (strict)
        'f1': order_metrics['f1'],
        'precision': order_metrics['precision'],
        'recall': order_metrics['recall'],
        
        'f1_order': order_metrics['f1'],
        'precision_order': order_metrics['precision'],
        'recall_order': order_metrics['recall'],
        
        # Reachability (Equivalent orders allowed)
        'f1_reach': reach_metrics['f1'],
        'precision_reach': reach_metrics['precision'],
        'recall_reach': reach_metrics['recall'],
        
        # Auxiliary (Edges)
        'f1_edges': edge_metrics['f1_edges'],
        'precision_edges': edge_metrics['precision_edges'],
        'recall_edges': edge_metrics['recall_edges'],
        'shd': edge_metrics['shd'],
        
        # Detailed counts
        'tp_order': order_metrics['tp'],
        'total_pairs': order_metrics['total_true_pairs'],
        'tp_reach': reach_metrics['tp'],
        'total_mandatory': reach_metrics['total_mandatory'],
        'tp_edges': edge_metrics['tp'],
        'fp_edges': edge_metrics['fp'],
        'fn_edges': edge_metrics['fn']
    }
    
    return combined


# Backwards compatible function (uses ORDER metric)
def count_accuracy(
    B_true: np.ndarray,
    B_est: np.ndarray,
    true_order: List[int] = None,
    estimated_order: List[int] = None
) -> Dict[str, float]:
    """
    Backwards compatible wrapper.
    
    If orders provided: Use ORDER-based F1 (CORRECT for paper)
    If orders not provided: Use EDGE-based F1 (fallback)
    """
    if true_order is not None and estimated_order is not None:
        # Use ORDER-based metric (correct for paper)
        return count_accuracy_order(true_order, estimated_order)
    else:
        # Fallback to edge-based
        print("Warning: Falling back to edge-based accuracy (orders not provided)")
        return count_accuracy_adjacency(B_true, B_est)