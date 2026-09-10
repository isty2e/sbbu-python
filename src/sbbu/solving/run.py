"""Solve budgets, phase diagnostics, and original-constraint result publication."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from ..constraints import ConstraintSet
from ..timer import get_wall_time
from ..validation import validate_positive_float

Termination = Literal[
    "feasible", "stationary", "deadline", "iteration_limit", "numerical_error"
]


@dataclass(frozen=True)
class SBBUStage:
    """Report one SBBU run, including nested soft-pruning work.

    Parameters
    ----------
    seconds : float
        Wall time spent in the SBBU stage.
    iterations : int
        Native branch-and-bound iteration counter, not a continuous iteration count.
    constraints_processed : int
        Number of long-range hard constraints processed.
    soft_pruning_rounds : int
        Soft-pruning rounds attempted.
    soft_pruned_constraints : int
        Soft groups successfully promoted to hard constraints.
    soft_pruned_candidates : int
        Alternatives removed during refinement.
    unresolved_soft_constraints : int
        Soft groups still unresolved after refinement.
    mean_soft_violation : float
        Mean minimum violation across the remaining working soft groups.
    max_soft_violation : float
        Largest minimum violation across those groups.
    termination : Termination
        Stage stop cause; bounded failure is not an infeasibility certificate.
    """

    seconds: float = 0.0
    iterations: int = 0
    constraints_processed: int = 0
    soft_pruning_rounds: int = 0
    soft_pruned_constraints: int = 0
    soft_pruned_candidates: int = 0
    unresolved_soft_constraints: int = 0
    mean_soft_violation: float = 0.0
    max_soft_violation: float = 0.0
    termination: Termination = "feasible"
    algorithm: ClassVar[Literal["sbbu"]] = "sbbu"


@dataclass(frozen=True)
class MMStage:
    """Report MM work without conflating it with TRF evaluations.

    Parameters
    ----------
    seconds : float
        Stage wall time, including preparation and cooperative overshoot.
    steps : int | None
        Completed updates, or None if a failed backend did not report its work.
    rank_lifts : int | None
        Missing coordinate directions perturbed, or None for unavailable work.
    termination : Termination
        Last attempt's stop cause, independent of the retained candidate's quality.
    """

    seconds: float
    steps: int | None
    rank_lifts: int | None
    termination: Termination
    algorithm: ClassVar[Literal["mm"]] = "mm"


@dataclass(frozen=True)
class TRFStage:
    """Report TRF work and its internal linear solver.

    Parameters
    ----------
    seconds : float
        Stage wall time, including preparation.
    evaluations : int | None
        Residual callback count, or None for unavailable work.
    jacobians : int | None
        Jacobian callback count, or None for unavailable work.
    matvecs : int | None
        Forward product count; zero for dense, None for unavailable work.
    rmatvecs : int | None
        Transpose product count; zero for dense, None for unavailable work.
    backend : {"dense", "lsmr"}
        Internal linear solver, not another public method choice.
    termination : Termination
        Last attempt's stop cause, independent of retained candidate quality.
    """

    seconds: float
    evaluations: int | None
    jacobians: int | None
    matvecs: int | None
    rmatvecs: int | None
    backend: Literal["dense", "lsmr"]
    termination: Termination
    algorithm: ClassVar[Literal["trf"]] = "trf"


@dataclass(frozen=True)
class SolveStats:
    """Summarize a solve accepted against its original hard constraints.

    Parameters
    ----------
    solve_time : float
        Wall time through final validation, excluding input parsing/normalization.
        It includes initialization, preparation, stage transitions and validation;
        it need not equal the sum of stage times.
    mean_distance_error : float
        Mean absolute violation of the original normalized hard bounds.
    largest_distance_error : float
        Maximum violation of those bounds.
    stages : tuple[SBBUStage | MMStage | TRFStage, ...]
        Executed algorithm stages in order. Initialization may already yield a
        valid candidate, requiring no continuous refinement stages.
    """

    solve_time: float
    mean_distance_error: float
    largest_distance_error: float
    stages: tuple[SBBUStage | MMStage | TRFStage, ...]


class SolveError(RuntimeError):
    """Report unsuccessful reconstruction without certifying infeasibility.

    Parameters
    ----------
    message : str
        Failure description.
    stages : tuple[SBBUStage | MMStage | TRFStage, ...], optional
        Completed or failed stage diagnostics available at failure.
    """

    def __init__(
        self,
        message: str,
        *,
        stages: tuple[SBBUStage | MMStage | TRFStage, ...] = (),
    ) -> None:
        super().__init__(message)
        self.stages = stages


class SolveTimeoutError(SolveError):
    """Report exhaustion of the shared cooperative solve deadline."""


class UnsupportedProblemError(ValueError):
    """Report valid constraints outside the currently supported solver envelope."""


@dataclass(frozen=True)
class SolveBudget:
    """Carry one absolute monotonic deadline across a solve's nested operations.

    Parameters
    ----------
    started : float
        Monotonic start timestamp.
    deadline : float
        Absolute monotonic deadline. Nested operations retain this exact value.
    """

    started: float
    deadline: float

    @classmethod
    def start(cls, max_time: float) -> "SolveBudget":
        """Start a validated budget.

        Parameters
        ----------
        max_time : float
            Finite positive duration in seconds.

        Returns
        -------
        SolveBudget
            A newly started solve budget.
        """
        validate_positive_float(max_time, "max_time")
        started = get_wall_time()
        return cls(started, started + max_time)

    def check(self, *, stages: tuple[SBBUStage | MMStage | TRFStage, ...] = ()) -> None:
        """Reject elapsed deadlines while retaining available diagnostics.

        Parameters
        ----------
        stages : tuple[SBBUStage | MMStage | TRFStage, ...], optional
            Stage evidence already available at this boundary.

        Raises
        ------
        SolveTimeoutError
            When the absolute monotonic deadline is reached.
        """
        if get_wall_time() >= self.deadline:
            raise SolveTimeoutError("Solve time limit exceeded", stages=stages)

    @property
    def remaining(self) -> float:
        """Return positive seconds left, raising SolveTimeoutError if exhausted."""
        remaining = self.deadline - get_wall_time()
        if remaining <= 0:
            raise SolveTimeoutError("Solve time limit exceeded")
        return remaining

    @property
    def elapsed(self) -> float:
        """Return seconds elapsed since this solve began."""
        return get_wall_time() - self.started


def finalize(
    constraints: ConstraintSet,
    coordinates: NDArray[np.float64],
    budget: SolveBudget,
    tolerance: float,
    stages: tuple[SBBUStage | MMStage | TRFStage, ...],
) -> SolveStats:
    """Publish diagnostics after original-bound and deadline validation.

    Parameters
    ----------
    constraints : ConstraintSet
        Original normalized hard bounds, never promoted working constraints.
    coordinates : NDArray[np.float64]
        Owned final coordinates of shape ``(num_nodes, 3)``.
    budget : SolveBudget
        The unchanged whole-call deadline.
    tolerance : float
        Validated absolute maximum-error tolerance.
    stages : tuple[SBBUStage | MMStage | TRFStage, ...]
        Available native stage reports, not feasibility certificates.

    Returns
    -------
    SolveStats
        Frozen common errors and native diagnostics.

    Raises
    ------
    SolveError
        If coordinates are invalid or violate original hard bounds.
    SolveTimeoutError
        If the deadline is reached, including during final verification.
    """
    budget.check(stages=stages)
    errors = constraints.constraint_errors(coordinates)
    largest = float(np.max(errors, initial=0.0))
    if not np.isfinite(largest):
        raise SolveError(
            "Solver returned invalid or numerically unassessable coordinates",
            stages=stages,
        )
    if largest > tolerance:
        raise SolveError(
            f"Original hard constraints not satisfied: error={largest:.6e}, "
            f"tolerance={tolerance:.6e}",
            stages=stages,
        )
    mean = float(np.mean(errors)) if errors.size else 0.0
    stats = SolveStats(budget.elapsed, mean, largest, stages)
    budget.check(stages=stages)
    return stats
