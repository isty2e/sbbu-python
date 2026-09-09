"""
Distance-constraint metric computations for solver runtime state.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..constraints import DistanceConstraint
from .types import ProblemState


class MetricsCalculator:
    """Compute distance-geometry quality metrics from runtime state."""

    @staticmethod
    def constraint_errors(
        constraints: Sequence[DistanceConstraint],
        coordinates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute finite-distance bound violations in constraint order.

        Parameters
        ----------
        constraints : Sequence[DistanceConstraint]
            Canonical constraints whose node indices address ``coordinates``.
        coordinates : NDArray[np.float64]
            Node coordinates with shape ``(N, 3)``.

        Returns
        -------
        NDArray[np.float64]
            Nonnegative violations with shape ``(len(constraints),)``.
            Nonfinite computed distances have infinite violation.
        """
        count = len(constraints)
        rows = np.fromiter((c.i for c in constraints), dtype=np.intp, count=count)
        columns = np.fromiter((c.j for c in constraints), dtype=np.intp, count=count)
        lower = np.fromiter(
            (c.lower_bound for c in constraints), dtype=np.float64, count=count
        )
        upper = np.fromiter(
            (c.upper_bound for c in constraints),
            dtype=np.float64,
            count=count,
        )
        distances = np.linalg.norm(coordinates[rows] - coordinates[columns], axis=1)
        errors = np.maximum(np.maximum(lower - distances, distances - upper), 0.0)
        return np.where(np.isfinite(distances), errors, np.inf)

    def populate_distance_errors(
        self, state: ProblemState, constraints: list[DistanceConstraint]
    ) -> None:
        """Populate hard-constraint error statistics."""
        errors = self.constraint_errors(constraints, state.coordinates)
        if errors.size:
            state.stats.mean_distance_error = float(np.mean(errors))
            state.stats.largest_distance_error = float(np.max(errors))
        else:
            state.stats.mean_distance_error = 0.0
            state.stats.largest_distance_error = 0.0

    def populate_soft_ambiguous_violations(self, state: ProblemState) -> None:
        """Populate soft-ambiguous violation statistics."""
        violations = []
        for pair, candidates in state.active_soft_ambiguous_constraints:
            i, j = pair
            pos_i = state.coordinates[i]
            pos_j = state.coordinates[j]
            actual_distance = float(np.linalg.norm(pos_i - pos_j))
            candidate_violations = [
                constraint.violation(actual_distance) for constraint in candidates
            ]
            violations.append(min(candidate_violations))

        state.stats.soft_ambiguous_constraints = len(
            state.active_soft_ambiguous_constraints
        )
        if violations:
            state.stats.mean_soft_violation = float(np.mean(violations))
            state.stats.max_soft_violation = float(np.max(violations))
        else:
            state.stats.mean_soft_violation = 0.0
            state.stats.max_soft_violation = 0.0
