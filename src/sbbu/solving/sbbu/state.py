"""
Core runtime and configuration types for the SBBU solver.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...constraints import ConstraintSet, DistanceConstraint
from ...validation import validate_positive_float
from ..run import SBBUStage, SolveError, SolveTimeoutError
from .cluster import ReflectionCluster, UnionFind


class SBBUTimeoutError(SolveTimeoutError):
    """Signal that the configured solver time budget has been exceeded."""


class SBBUSolveInfeasibleError(SolveError):
    """Signal that SBBU could not satisfy a constraint within its search limits."""


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
class SBBUWork:
    """Accumulate native SBBU counters within one active solve."""

    iterations: int = 0
    constraints_processed: int = 0
    soft_pruning_rounds: int = 0
    soft_pruned_constraints: int = 0
    soft_pruned_candidates: int = 0


@dataclass
class SBBUState:
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
    work: SBBUWork

    def stage(self, seconds: float) -> SBBUStage:
        """Snapshot native work and remaining soft-alternative quality."""
        violations = []
        for (i, j), candidates in self.active_soft_ambiguous_constraints:
            distance = float(np.linalg.norm(self.coordinates[i] - self.coordinates[j]))
            violations.append(min(c.violation(distance) for c in candidates))
        return SBBUStage(
            seconds=seconds,
            iterations=self.work.iterations,
            constraints_processed=self.work.constraints_processed,
            soft_pruning_rounds=self.work.soft_pruning_rounds,
            soft_pruned_constraints=self.work.soft_pruned_constraints,
            soft_pruned_candidates=self.work.soft_pruned_candidates,
            unresolved_soft_constraints=len(violations),
            mean_soft_violation=float(np.mean(violations)) if violations else 0.0,
            max_soft_violation=max(violations, default=0.0),
        )


@dataclass
class RefinementContext:
    """Dependencies and state required by one refinement run.

    Parameters
    ----------
    state : SBBUState
        Runtime coordinates and statistics updated during refinement.
    config : SBBUConfig
        Refinement policy and numerical acceptance limits.
    constraints : ConstraintSet
        Canonical original observations; working scalar projections are deferred
        until soft refinement is actually needed.
    check_time_limit : Callable[[], None]
        Check the enclosing solve deadline, raising when its budget is exhausted.
    run_refinement_solve : Callable
        Solve a supplied list of hard constraints, returning coordinates and
        statistics or raising on failure.
    """

    state: SBBUState
    config: SBBUConfig
    constraints: ConstraintSet
    check_time_limit: Callable[[], None]
    run_refinement_solve: Callable[
        [list[DistanceConstraint]], tuple[NDArray[np.float64], SBBUStage]
    ]
