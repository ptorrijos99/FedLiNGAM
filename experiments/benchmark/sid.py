"""Structural intervention distance (Peters and Buhlmann, 2015) on adjacency matrices.

Convention here: ``A[u, v]`` is True for an edge ``u -> v``. A pair (x, y) counts as
wrong when y is a parent of x in the estimate but a descendant of x in the truth, or,
otherwise, when the estimated parent set of x fails the adjustment criterion for
(x, y) in the true graph (Shpitser, VanderWeele and Robins, 2010).
"""

from __future__ import annotations

import numpy as np


def _topological_order(A: np.ndarray) -> list[int]:
    indeg = A.sum(axis=0).astype(int)
    ready = [v for v in range(A.shape[0]) if indeg[v] == 0]
    order = []
    while ready:
        v = ready.pop()
        order.append(v)
        for c in np.flatnonzero(A[v]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(int(c))
    if len(order) != A.shape[0]:
        raise ValueError("graph is not acyclic")
    return order


def descendant_matrix(A: np.ndarray) -> np.ndarray:
    """``reach[i, j]`` is True when there is a directed path of length >= 1 from i to j."""
    p = A.shape[0]
    reach = np.zeros((p, p), dtype=bool)
    for v in reversed(_topological_order(A)):
        row = reach[v]
        for c in np.flatnonzero(A[v]):
            row[c] = True
            row |= reach[c]
    return reach


def _reachable(parents, children, source: int, z_mask: np.ndarray) -> np.ndarray:
    """Nodes d-connected to `source` given the set in `z_mask` (Koller and Friedman, Alg. 3.1)."""
    p = len(parents)
    in_ancestry = z_mask.copy()
    stack = [v for v in range(p) if z_mask[v]]
    while stack:
        v = stack.pop()
        for u in parents[v]:
            if not in_ancestry[u]:
                in_ancestry[u] = True
                stack.append(u)

    # Direction 0: entered from a child; 1: entered from a parent.
    visited = np.zeros(2 * p, dtype=bool)
    reached = np.zeros(p, dtype=bool)
    frontier = [(source, 0)]
    while frontier:
        v, direction = frontier.pop()
        state = 2 * v + direction
        if visited[state]:
            continue
        visited[state] = True
        blocked = bool(z_mask[v])
        if not blocked:
            reached[v] = True
        if direction == 0:
            if not blocked:
                for u in parents[v]:
                    if not visited[2 * u]:
                        frontier.append((u, 0))
                for c in children[v]:
                    if not visited[2 * c + 1]:
                        frontier.append((c, 1))
        else:
            if not blocked:
                for c in children[v]:
                    if not visited[2 * c + 1]:
                        frontier.append((c, 1))
            if in_ancestry[v]:
                for u in parents[v]:
                    if not visited[2 * u]:
                        frontier.append((u, 0))
    return reached


def _adjustment_ok(parents, children, reach, x: int, y: int, z_mask: np.ndarray) -> bool:
    """Adjustment criterion for (x, y); the adjacency lists are restored before returning."""
    p = reach.shape[0]
    if reach[x, y]:
        on_path = np.asarray(reach[x] & reach[:, y], dtype=bool)
        on_path[y] = True
        on_path[x] = False
    else:
        on_path = np.zeros(p, dtype=bool)

    # No member of the set may descend from a node on a causal path.
    forbidden = on_path.copy()
    for w in np.nonzero(on_path)[0]:
        forbidden |= reach[w]
    if bool((forbidden & z_mask).any()):
        return False

    # d-separation in the proper back-door graph, which drops the first edge of each
    # causal path.
    cut = [c for c in children[x] if on_path[c]]
    if cut:
        children[x] = [c for c in children[x] if not on_path[c]]
        for c in cut:
            parents[c] = [u for u in parents[c] if u != x]
    try:
        return not bool(_reachable(parents, children, x, z_mask)[y])
    finally:
        if cut:
            children[x] = sorted([*children[x], *cut])
            for c in cut:
                parents[c] = sorted([*parents[c], x])


def sid(A_true: np.ndarray, A_est: np.ndarray) -> int:
    """Number of ordered pairs whose interventional distribution `A_est` gets wrong."""
    A_true, A_est = np.asarray(A_true, dtype=bool), np.asarray(A_est, dtype=bool)
    p = A_true.shape[0]
    if A_est.shape != (p, p):
        raise ValueError("the two graphs differ in node count")
    reach = descendant_matrix(A_true)
    parents = [sorted(np.flatnonzero(A_true[:, v]).tolist()) for v in range(p)]
    children = [sorted(np.flatnonzero(A_true[v]).tolist()) for v in range(p)]

    wrong = 0
    for x in range(p):
        parents_est = set(np.flatnonzero(A_est[:, x]).tolist())
        z_mask = np.zeros(p, dtype=bool)
        if parents_est:
            z_mask[list(parents_est)] = True
        for y in range(p):
            if x == y:
                continue
            if y in parents_est:
                if reach[x, y]:
                    wrong += 1
            elif not _adjustment_ok(parents, children, reach, x, y, z_mask):
                wrong += 1
    return wrong
