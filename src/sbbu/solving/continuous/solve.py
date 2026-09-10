"""Continuous capability admission and shared-budget stage scheduling."""

from typing import Literal, NoReturn

import numpy as np
from numpy.typing import NDArray

from ...constraints import ConstraintSet
from ..run import MMStage, SolveBudget, SolveError, SolveTimeoutError, TRFStage
from .initialization import partial_spectral
from .mm import solve_mm
from .state import BestCandidate
from .trf import solve_trf


def applicability_error(constraints: ConstraintSet) -> str | None:
    """Explain the current continuous backend's temporary support boundary.

    Parameters
    ----------
    constraints : ConstraintSet
        Valid normalized observations in original node order.

    Returns
    -------
    str | None
        Unsupported-capability reason, or None. This is not an input-validity
        test or an infeasibility claim. Missing-edge support awaits verification.
    """
    if constraints.num_nodes < 4:
        return "Continuous solving currently requires at least four nodes"
    if not constraints.is_complete:
        return (
            "Continuous solving temporarily requires a complete unambiguous hard graph; "
            "missing-edge support has not yet been verified"
        )
    if np.any(constraints.lower_bounds == constraints.upper_bounds):
        return "Continuous solving currently requires strictly positive-width intervals"
    return None


def _raise_stage_failure(
    error: Exception,
    previous: list[MMStage | TRFStage],
    unavailable: MMStage | TRFStage,
) -> NoReturn:
    available = error.stages if isinstance(error, SolveError) else ()
    stages = (*previous, *(available or (unavailable,)))
    failure = SolveTimeoutError if isinstance(error, SolveTimeoutError) else SolveError
    raise failure(str(error), stages=stages) from error


def solve(
    constraints: ConstraintSet,
    budget: SolveBudget,
    tolerance: float,
    method: Literal["mm", "trf"] | None,
) -> tuple[NDArray[np.float64], tuple[MMStage | TRFStage, ...]]:
    """Initialize and refine an admitted problem under the unchanged deadline.

    Parameters
    ----------
    constraints : ConstraintSet
        Complete positive-width hard intervals admitted by the solve operation.
    budget : SolveBudget
        Whole-call monotonic deadline.
    tolerance : float
        Validated maximum original-bound error tolerance.
    method : {"mm", "trf", None}
        Validated explicit method or provisional size-based policy. None uses
        a nominal 50 ms MM prefix through 50 nodes, otherwise direct TRF/LSMR50.

    Returns
    -------
    tuple[NDArray[np.float64], tuple[MMStage | TRFStage, ...]]
        Owned best candidate and ordered native reports. The outer operation
        still performs final original-bound and deadline verification.

    Raises
    ------
    SolveError
        Numerical failure or termination without a feasible candidate.
    SolveTimeoutError
        Whole-call deadline exhaustion, never just the MM prefix boundary.
    """
    stages: list[MMStage | TRFStage] = []
    try:
        budget.check()
        initial = partial_spectral(constraints, budget)
        best = BestCandidate(constraints, initial)
        budget.check()
    except SolveTimeoutError:
        raise
    except (
        FloatingPointError,
        MemoryError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        raise SolveError("Spectral initialization failed") from error
    if not np.isfinite(best.error):
        raise SolveError("Spectral initialization returned an invalid candidate")
    if best.error <= tolerance:
        return best.coordinates, ()

    if method == "mm" or (method is None and constraints.num_nodes <= 50):
        started = budget.elapsed
        prefix_deadline = budget.deadline
        if method is None:
            prefix_deadline = min(prefix_deadline, budget.started + started + 0.05)
        try:
            candidate, report = solve_mm(
                constraints,
                best.coordinates,
                budget,
                tolerance,
                deadline=prefix_deadline,
            )
        except (
            FloatingPointError,
            MemoryError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            reason = (
                "deadline"
                if isinstance(error, SolveTimeoutError)
                else "numerical_error"
            )
            _raise_stage_failure(
                error, stages, MMStage(budget.elapsed - started, None, None, reason)
            )
        stages.append(report)
        budget.check(stages=tuple(stages))
        if report.termination == "numerical_error":
            raise SolveError("MM reported numerical_error", stages=tuple(stages))
        if not np.isfinite(best.consider(candidate)):
            raise SolveError("MM returned an invalid candidate", stages=tuple(stages))
        budget.check(stages=tuple(stages))
        if best.error <= tolerance:
            return best.coordinates, tuple(stages)
        if method == "mm":
            raise SolveError(
                "MM stopped without a feasible candidate", stages=tuple(stages)
            )

    budget.check(stages=tuple(stages))
    started = budget.elapsed
    candidate = best.coordinates
    try:
        candidate, trf_report = solve_trf(
            constraints, best.coordinates, budget, tolerance
        )
    except (
        FloatingPointError,
        MemoryError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        reason = (
            "deadline" if isinstance(error, SolveTimeoutError) else "numerical_error"
        )
        backend = "dense" if constraints.num_nodes <= 50 else "lsmr"
        _raise_stage_failure(
            error,
            stages,
            TRFStage(budget.elapsed - started, None, None, None, None, backend, reason),
        )
    stages.append(trf_report)
    budget.check(stages=tuple(stages))
    if trf_report.termination == "numerical_error":
        raise SolveError("TRF reported numerical_error", stages=tuple(stages))
    if trf_report.termination == "deadline":
        raise SolveTimeoutError(
            "TRF reported deadline exhaustion", stages=tuple(stages)
        )
    if not np.isfinite(best.consider(candidate)):
        raise SolveError("TRF returned an invalid candidate", stages=tuple(stages))
    budget.check(stages=tuple(stages))
    if best.error <= tolerance:
        return best.coordinates, tuple(stages)
    raise SolveError("TRF stopped without a feasible candidate", stages=tuple(stages))
