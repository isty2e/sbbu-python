"""
Distance-constraint metric computations for solver runtime state.
"""

import numpy as np

from ..constraints import DistanceConstraint
from .types import ProblemState


class MetricsCalculator:
    """Compute distance-geometry quality metrics from runtime state."""

    @staticmethod
    def constraint_bounds(constraint: DistanceConstraint) -> tuple[float, float]:
        """Return [lower, upper] bounds with exact constraints normalized."""
        upper = (
            constraint.upper_bound
            if constraint.upper_bound is not None
            else constraint.lower_bound
        )
        return constraint.lower_bound, upper

    @classmethod
    def constraint_violation(
        cls, constraint: DistanceConstraint, distance: float
    ) -> float:
        """Return absolute violation of a constraint interval."""
        if not np.isfinite(distance):
            return float("inf")
        lower, upper = cls.constraint_bounds(constraint)
        if distance < lower:
            return lower - distance
        if distance > upper:
            return distance - upper
        return 0.0

    def populate_distance_errors(
        self, state: ProblemState, constraints: list[DistanceConstraint]
    ) -> None:
        """Populate hard-constraint error statistics."""
        errors = []
        for constraint in constraints:
            pos_i = state.coordinates[constraint.i]
            pos_j = state.coordinates[constraint.j]
            actual_distance = float(np.linalg.norm(pos_i - pos_j))
            error = self.constraint_violation(constraint, actual_distance)
            errors.append(error)

        if errors:
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
                self.constraint_violation(constraint, actual_distance)
                for constraint in candidates
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
