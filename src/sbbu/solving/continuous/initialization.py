"""Partial spectral initialization for complete interval bounds."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh

from ...constraints import ConstraintSet
from ..run import SolveBudget


class _CenteredGram(LinearOperator):
    """Apply the centered midpoint-distance Gram matrix without storing it."""

    def __init__(
        self, squared_distances: NDArray[np.float64], budget: SolveBudget
    ) -> None:
        self.squared_distances = squared_distances
        self.budget = budget
        super().__init__(dtype=np.dtype(np.float64), shape=squared_distances.shape)

    def _matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.budget.check()
        centered = x.ravel() - x.mean()
        product = self.squared_distances @ centered
        result = -0.5 * (product - product.mean())
        self.budget.check()
        return result


def _canonicalize(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(coordinates, dtype=np.float64).copy()
    result -= result.mean(axis=0, keepdims=True)
    for axis in range(result.shape[1]):
        column = result[:, axis]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            result[:, axis] *= -1.0
    return result


def partial_spectral(
    constraints: ConstraintSet, budget: SolveBudget
) -> NDArray[np.float64]:
    """Return a three-dimensional midpoint-Gram proposal with a fixed local seed.

    Parameters
    ----------
    constraints : ConstraintSet
        Complete finite hard interval observations for at least four nodes.
    budget : SolveBudget
        Absolute budget shared with subsequent refinement stages.

    Returns
    -------
    NDArray[np.float64]
        Centered coordinates in original node order.

    Raises
    ------
    SolveTimeoutError
        If spectral preparation or eigensolver work reaches the budget.
    ValueError
        If the bounds are not complete.
    """
    if not constraints.is_complete:
        raise ValueError("Spectral initialization requires complete hard bounds")
    budget.check()
    lower, upper = constraints.bounds_matrices()
    midpoint_squared = (lower + (upper - lower) * 0.5) ** 2
    operator = _CenteredGram(midpoint_squared, budget)
    rng = np.random.default_rng(1729)
    v0 = rng.normal(size=constraints.num_nodes)
    values, vectors = eigsh(
        operator,
        k=3,
        which="LA",
        tol=1e-6,
        maxiter=300,
        v0=v0,
    )
    budget.check()
    coordinates = vectors * np.sqrt(np.maximum(values, 0.0))[None, :]
    result = _canonicalize(coordinates)
    budget.check()
    return result
