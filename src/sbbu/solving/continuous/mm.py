"""Safeguarded accelerated majorization-minimization for complete intervals."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...constraints import ConstraintSet
from ..run import MMStage, SolveBudget, SolveError, SolveTimeoutError, Termination
from .state import BestCandidate


class _StageDeadline(Exception):
    pass


@dataclass
class _MMWork:
    steps: int = 0
    rank_lifts: int = 0


def _check(budget: SolveBudget, deadline: float) -> None:
    budget.check()
    if budget.started + budget.elapsed >= deadline:
        raise _StageDeadline


def interval_stress(
    constraints: ConstraintSet, coordinates: NDArray[np.float64]
) -> float:
    """Evaluate squared interval residuals for MM's acceleration safeguard.

    Parameters
    ----------
    constraints : ConstraintSet
        Original hard bounds.
    coordinates : NDArray[np.float64]
        Proposed coordinates with shape ``(num_nodes, 3)``.

    Returns
    -------
    float
        Sum of squared signed residuals, not a feasibility certificate.
    """
    distance = np.linalg.norm(
        coordinates[constraints.rows] - coordinates[constraints.columns], axis=1
    )
    residual = distance - np.clip(
        distance, constraints.lower_bounds, constraints.upper_bounds
    )
    return float(np.sum(residual * residual))


def stress_step(
    constraints: ConstraintSet, coordinates: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply the complete-graph interval majorization update.

    Parameters
    ----------
    constraints : ConstraintSet
        Complete original hard bounds; completeness is a backend precondition.
    coordinates : NDArray[np.float64]
        Current coordinates with shape ``(num_nodes, 3)``.

    Returns
    -------
    NDArray[np.float64]
        Centered plain MM proposal. Collapsed pairs contribute zero direction.
    """
    first, second = constraints.rows, constraints.columns
    differences = coordinates[first] - coordinates[second]
    distances = np.linalg.norm(differences, axis=1)
    targets = np.clip(distances, constraints.lower_bounds, constraints.upper_bounds)
    ratios = np.divide(
        targets, distances, out=np.zeros_like(distances), where=distances > 1e-15
    )
    update = np.zeros_like(coordinates)
    np.add.at(update, first, ratios[:, None] * differences)
    np.add.at(update, second, -ratios[:, None] * differences)
    result = update / constraints.num_nodes
    result -= result.mean(axis=0, keepdims=True)
    return result


def _rank_lift(
    constraints: ConstraintSet,
    coordinates: NDArray[np.float64],
    budget: SolveBudget,
    deadline: float,
    work: _MMWork,
) -> NDArray[np.float64]:
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    threshold = (
        float(singular.max(initial=0.0)) * max(centered.shape) * np.finfo(float).eps
    )
    rank = int(np.count_nonzero(singular > threshold))
    if rank >= 3:
        return coordinates
    scale = 0.25 * float(np.median(constraints.upper_bounds - constraints.lower_bounds))
    if not np.isfinite(scale) or scale <= 0.0:
        return coordinates
    rng = np.random.default_rng(1729)
    lifted = coordinates.copy()
    for direction in vh[rank:3]:
        _check(budget, deadline)
        lifted += np.outer(rng.normal(size=coordinates.shape[0]), direction) * scale
        work.rank_lifts += 1
    lifted -= lifted.mean(axis=0, keepdims=True)
    return lifted


def solve_mm(
    constraints: ConstraintSet,
    initial: NDArray[np.float64],
    budget: SolveBudget,
    tolerance: float,
    *,
    deadline: float | None = None,
) -> tuple[NDArray[np.float64], MMStage]:
    """Run accelerated MM with original-bound retention and an optional prefix cap.

    Parameters
    ----------
    constraints : ConstraintSet
        Complete positive-width hard intervals.
    initial : NDArray[np.float64]
        Initial ``(num_nodes, 3)`` coordinates, never mutated.
    budget : SolveBudget
        Whole-call monotonic deadline.
    tolerance : float
        Maximum original hard-bound violation for a feasible candidate.
    deadline : float, optional
        Earlier absolute deadline in the same clock domain as ``budget``.

    Returns
    -------
    tuple[NDArray[np.float64], MMStage]
        Owned best candidate and native work/termination diagnostics. Prefix
        expiry is not whole-call expiry and is not a feasibility certificate.

    Raises
    ------
    SolveError
        Numerical failure, with available MM work and the original exception.
    SolveTimeoutError
        Whole-call expiry before a candidate can be retained.
    """
    started = budget.elapsed
    stage_deadline = (
        budget.deadline if deadline is None else min(deadline, budget.deadline)
    )
    work = _MMWork()
    termination: Termination = "stationary"

    def stage() -> MMStage:
        return MMStage(
            budget.elapsed - started, work.steps, work.rank_lifts, termination
        )

    # Establish owned retention before the prefix can expire during preparation.
    try:
        budget.check()
        best = BestCandidate(constraints, initial)
    except SolveTimeoutError as error:
        termination = "deadline"
        raise SolveTimeoutError(str(error), stages=(stage(),)) from error
    except (
        FloatingPointError,
        MemoryError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        termination = "numerical_error"
        raise SolveError("MM preparation failed", stages=(stage(),)) from error

    try:
        _check(budget, stage_deadline)
        if not np.isfinite(best.error):
            raise FloatingPointError("Nonfinite MM initial candidate")
        if best.error <= tolerance:
            termination = "feasible"
        else:
            current = np.asarray(
                initial - initial.mean(axis=0, keepdims=True), dtype=np.float64
            )
            best.consider(current)
            current = _rank_lift(constraints, current, budget, stage_deadline, work)
            best.consider(current)
            history_images: list[NDArray[np.float64]] = []
            history_residuals: list[NDArray[np.float64]] = []
            for _ in range(20_000):
                _check(budget, stage_deadline)
                if best.error <= tolerance:
                    termination = "feasible"
                    break
                plain = stress_step(constraints, current)
                work.steps += 1
                history_images.append(plain.ravel().copy())
                history_residuals.append(
                    np.asarray((plain - current).ravel(), dtype=np.float64)
                )
                if len(history_images) > 6:
                    history_images.pop(0)
                    history_residuals.pop(0)
                candidate = plain
                if len(history_images) >= 2:
                    delta_residuals = np.diff(
                        np.stack(history_residuals, axis=1), axis=1
                    )
                    delta_images = np.diff(np.stack(history_images, axis=1), axis=1)
                    weights = np.linalg.lstsq(
                        delta_residuals, history_residuals[-1], rcond=1e-10
                    )[0]
                    extrapolated = np.asarray(
                        plain - (delta_images @ weights).reshape(current.shape),
                        dtype=np.float64,
                    )
                    if np.isfinite(extrapolated).all() and interval_stress(
                        constraints, extrapolated
                    ) < interval_stress(constraints, plain):
                        candidate = extrapolated
                error = best.consider(candidate)
                if not np.isfinite(error):
                    raise FloatingPointError("Nonfinite MM proposal")
                if best.error <= tolerance:
                    termination = "feasible"
                    break
                change = float(np.linalg.norm(candidate - current))
                current = candidate
                if change <= 1e-13 * max(1.0, float(np.linalg.norm(current))):
                    termination = "stationary"
                    break
            else:
                termination = "iteration_limit"
    except (_StageDeadline, SolveTimeoutError):
        termination = "deadline"
    except (
        FloatingPointError,
        MemoryError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        termination = "numerical_error"
        raise SolveError(
            "MM refinement failed numerically", stages=(stage(),)
        ) from error
    return best.coordinates, stage()
