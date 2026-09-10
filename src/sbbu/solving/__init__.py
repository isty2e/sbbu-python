"""Reconstruction admission, dispatch and original-bound publication."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..constraints import ConstraintSet
from ..validation import validate_positive_float
from .continuous.solve import applicability_error as continuous_applicability_error
from .continuous.solve import solve as solve_continuous
from .run import SolveBudget, SolveStats, UnsupportedProblemError, finalize
from .sbbu.solver import SBBUSolver
from .sbbu.state import SBBUConfig


def solve(
    constraints: ConstraintSet,
    distance_tolerance: float,
    max_time: float,
    verbose: bool,
    method: Literal["mm", "trf"] | None,
) -> tuple[NDArray[np.float64], SolveStats]:
    """Reconstruct original bounds with current-order SBBU taking precedence.

    Parameters
    ----------
    constraints : ConstraintSet
        Normalized immutable observations in original node order.
    distance_tolerance : float
        Finite positive absolute hard-bound acceptance tolerance.
    max_time : float
        Finite positive whole-solve duration, excluding input normalization.
    verbose : bool
        Whether SBBU emits progress logs.
    method : {"mm", "trf", None}
        Non-SBBU method or automatic continuous policy. This option is validated
        even when current-order SBBU is applicable.

    Returns
    -------
    tuple[NDArray[np.float64], SolveStats]
        Owned coordinates and frozen reports after final verification.

    Raises
    ------
    TypeError
        Wrong option type.
    ValueError
        Invalid option value.
    UnsupportedProblemError
        Valid observations outside both current backend capabilities.
    SolveError
        Numerical failure or bounded search ending without a valid realization.
    SolveTimeoutError
        Shared deadline exhaustion, including publication work.
    """
    validate_positive_float(distance_tolerance, "distance_tolerance")
    validate_positive_float(max_time, "max_time")
    if method is not None and not isinstance(method, str):
        raise TypeError("method must be a string or None")
    if method not in (None, "mm", "trf"):
        raise ValueError("method must be 'mm', 'trf', or None")

    sbbu_reason = SBBUSolver.applicability_error(constraints)
    if sbbu_reason is None:
        budget = SolveBudget.start(max_time)
        solver = SBBUSolver(
            constraints,
            SBBUConfig(
                distance_tolerance=distance_tolerance,
                max_time=max_time,
                verbose=verbose,
            ),
        )
        stats = solver._solve(budget)
        coordinates = solver.solution_coordinates
        budget.check(stages=stats.stages)
        return coordinates, stats

    continuous_reason = continuous_applicability_error(constraints)
    if continuous_reason is not None:
        raise UnsupportedProblemError(f"{sbbu_reason}. {continuous_reason}")
    budget = SolveBudget.start(max_time)
    candidate, stages = solve_continuous(
        constraints, budget, distance_tolerance, method
    )
    budget.check(stages=stages)
    coordinates = candidate.copy()
    stats = finalize(constraints, coordinates, budget, distance_tolerance, stages)
    budget.check(stages=stages)
    return coordinates, stats
