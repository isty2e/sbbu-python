"""
Core runtime and configuration types for the SBBU solver.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..constraints import DistanceConstraint
from ..validation import validate_positive_float
from .cluster import ReflectionCluster, UnionFind


class SBBUTimeoutError(RuntimeError):
    """Signal that the configured solver time budget has been exceeded."""


class SBBUSolveInfeasibleError(RuntimeError):
    """Signal that the current constraint set is infeasible for SBBU solving."""


@dataclass
class SBBUConfig:
    """
    Configuration options for :class:`SBBUSolver`.

    Attributes
    ----------
    distance_tolerance : float
        Maximum acceptable numeric error for constraint checks.
    max_iterations : int
        Upper bound on branch-and-bound state evaluations per constraint.
    max_time : float
        Default solve time budget in seconds.
    verbose : bool
        Whether to emit progress logs while solving.
    enable_soft_pruning : bool
        Whether to run soft-ambiguous pruning and refinement.
    soft_pruning_max_rounds : int
        Maximum refinement rounds for soft-pruning.
    soft_pruning_acceptance_tolerance : float
        Maximum violation allowed for promoting a soft candidate.
    soft_pruning_candidate_window : float
        Violation window above the best candidate used to retain alternatives.
    soft_pruning_min_margin : float
        Minimum violation gap required between best and second-best candidates
        before promotion.
    """

    distance_tolerance: float = 1e-7  # Match C++ tolerance
    max_iterations: int = int(1e9)
    max_time: float = 60.0
    verbose: bool = True
    enable_soft_pruning: bool = True
    soft_pruning_max_rounds: int = 3
    soft_pruning_acceptance_tolerance: float = 5e-2
    soft_pruning_candidate_window: float = 2e-1
    soft_pruning_min_margin: float = 2e-2

    def __post_init__(self) -> None:
        """Validate configuration values."""
        validate_positive_float(self.distance_tolerance, "distance_tolerance")
        validate_positive_float(self.max_time, "max_time")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
        ):
            raise ValueError(
                f"max_iterations must be a positive integer, got {self.max_iterations}"
            )
        if (
            isinstance(self.soft_pruning_max_rounds, bool)
            or not isinstance(self.soft_pruning_max_rounds, int)
            or self.soft_pruning_max_rounds <= 0
        ):
            raise ValueError(
                "soft_pruning_max_rounds must be a positive integer, "
                f"got {self.soft_pruning_max_rounds}"
            )

        threshold_fields = (
            (
                "soft_pruning_acceptance_tolerance",
                self.soft_pruning_acceptance_tolerance,
            ),
            ("soft_pruning_candidate_window", self.soft_pruning_candidate_window),
            ("soft_pruning_min_margin", self.soft_pruning_min_margin),
        )
        for field_name, value in threshold_fields:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number, got {type(value)}")
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")


@dataclass
class SBBUStats:
    """
    Statistics collected during a solver run.

    Attributes
    ----------
    solve_time : float
        Total wall-clock solve time in seconds.
    iterations_used : int
        Number of branch-and-bound iterations consumed.
    constraints_processed : int
        Count of long-range constraints processed.
    mean_distance_error : float
        Mean hard-constraint violation.
    largest_distance_error : float
        Maximum hard-constraint violation.
    soft_ambiguous_constraints : int
        Number of unresolved soft-ambiguous groups after refinement.
    mean_soft_violation : float
        Mean minimal violation across unresolved soft-ambiguous groups.
    max_soft_violation : float
        Maximum minimal violation across unresolved soft-ambiguous groups.
    soft_pruning_rounds : int
        Number of soft-pruning refinement rounds executed.
    soft_pruned_constraints : int
        Number of soft constraints promoted to hard constraints.
    soft_pruned_candidates : int
        Number of soft candidates removed during refinement.
    """

    solve_time: float = 0.0
    iterations_used: int = 0
    constraints_processed: int = 0
    mean_distance_error: float = 0.0
    largest_distance_error: float = 0.0
    soft_ambiguous_constraints: int = 0
    mean_soft_violation: float = 0.0
    max_soft_violation: float = 0.0
    soft_pruning_rounds: int = 0
    soft_pruned_constraints: int = 0
    soft_pruned_candidates: int = 0


@dataclass
class ProblemState:
    """Mutable runtime state for one solver pass."""

    coordinates: NDArray[np.float64]
    current_node: int
    active_soft_ambiguous_constraints: list[
        tuple[tuple[int, int], list[DistanceConstraint]]
    ]
    union_find: UnionFind
    clusters: list[ReflectionCluster]
    cluster_nodes: list[int]
    reflection_flags: list[bool]
    best_reflection_flags: list[bool]
    num_cluster_nodes: int
    stats: SBBUStats


@dataclass
class RefinementContext:
    """Dependencies and state required by one refinement run."""

    state: ProblemState
    config: SBBUConfig
    num_nodes: int
    hard_constraints: list[DistanceConstraint]
    constraint_violation: Callable[[DistanceConstraint, float], float]
    check_time_limit: Callable[[], None]
    run_refinement_solve: Callable[
        [list[DistanceConstraint]], tuple[NDArray[np.float64], SBBUStats]
    ]
