"""Bounded search for an exact, consecutive-predecessor SBBU order."""

import numpy as np

from ...constraints import ConstraintSet
from ..run import SolveBudget

_MAX_SECONDS = 0.05
_BUDGET_FRACTION = 0.05
_MIN_VISIT_LIMIT = 4096


def find_order(constraints: ConstraintSet, budget: SolveBudget) -> list[int] | None:
    """Find a full exact-band order without certifying geometry or completeness.

    Parameters
    ----------
    constraints : ConstraintSet
        Original observations; only unambiguous exact hard edges are used.
    budget : SolveBudget
        Whole-call budget, also charged for graph construction and search.

    Returns
    -------
    list[int] | None
        Original node indices in working order, or None if bounded search finds
        no order. None does not prove that no such order or realization exists.

    Raises
    ------
    SolveTimeoutError
        The whole-call deadline expires, independently of the local search cap.
    """
    reserved_seconds = budget.remaining - min(
        _MAX_SECONDS, _BUDGET_FRACTION * (budget.deadline - budget.started)
    )

    def available() -> bool:
        return budget.remaining > reserved_seconds

    n = constraints.num_nodes
    if n < 3:
        return None
    exact = np.flatnonzero(constraints.lower_bounds == constraints.upper_bounds)
    if not available() or len(exact) < 3 * n - 6:
        return None

    adjacency: list[set[int]] = [set() for _ in range(n)]
    rows, columns = constraints.rows, constraints.columns
    for count, index in enumerate(exact):
        if count % 128 == 0 and not available():
            return None
        first, second = int(rows[index]), int(columns[index])
        adjacency[first].add(second)
        adjacency[second].add(first)

    degrees = [len(neighbors) for neighbors in adjacency]
    if not available() or min(degrees) < min(3, n - 1):
        return None

    def priority(node: int) -> tuple[int, int]:
        return degrees[node], node

    path: list[int] = []
    used: set[int] = set()
    choices = [sorted(range(n), key=priority, reverse=True)]
    visits = 0
    visit_limit = max(_MIN_VISIT_LIMIT, 2 * n)
    while choices:
        if not available() or visits >= visit_limit:
            return None
        if not choices[-1]:
            choices.pop()
            if path:
                used.remove(path.pop())
            continue

        node = choices[-1].pop()
        visits += 1
        path.append(node)
        used.add(node)
        if len(path) == n:
            return path if available() else None

        candidates = adjacency[node].copy()
        for predecessor in path[-3:-1]:
            candidates.intersection_update(adjacency[predecessor])
        candidates.difference_update(used)
        choices.append(sorted(candidates, key=priority, reverse=True))

    budget.check()
    return None
