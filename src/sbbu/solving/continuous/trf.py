"""Dense and compact matrix-free trust-region least-squares refinement."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator

from ...constraints import ConstraintSet
from ..run import SolveBudget, SolveError, SolveTimeoutError, Termination, TRFStage
from .state import BestCandidate


class _FeasibleFound(Exception):
    pass


@dataclass
class TRFWork:
    """Accumulate native callback work within one TRF invocation.

    Parameters
    ----------
    evaluations : int
        Residual evaluations begun after deadline admission.
    jacobians : int
        Jacobian evaluations begun after deadline admission.
    matvecs : int
        Forward products begun.
    rmatvecs : int
        Transpose products begun.
    """

    evaluations: int = 0
    jacobians: int = 0
    matvecs: int = 0
    rmatvecs: int = 0


class PairResiduals:
    """Project canonical observed intervals into residuals and derivatives.

    Parameters
    ----------
    constraints : ConstraintSet
        Original hard bounds. Solver admission, not this derivative projection,
        determines which graph structures are currently supported.
    """

    def __init__(self, constraints: ConstraintSet) -> None:
        self.first = constraints.rows
        self.second = constraints.columns
        self.lower = constraints.lower_bounds
        self.upper = constraints.upper_bounds
        self.n = constraints.num_nodes

    def unpack(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Expand free variables with node zero fixed at the origin.

        Parameters
        ----------
        vector : NDArray[np.float64]
            Coordinates for nodes one through ``n - 1``, flattened.

        Returns
        -------
        NDArray[np.float64]
            Full ``(n, 3)`` coordinate array in original order.
        """
        coordinates = np.zeros((self.n, 3), dtype=np.float64)
        coordinates[1:] = vector.reshape(self.n - 1, 3)
        return coordinates

    def residual(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate signed residuals for every original observed pair.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Full ``(n, 3)`` coordinates in original node order.

        Returns
        -------
        NDArray[np.float64]
            Signed distance minus its projection onto each original interval.
        """
        distance = np.linalg.norm(
            coordinates[self.first] - coordinates[self.second], axis=1
        )
        return distance - np.clip(distance, self.lower, self.upper)

    def directions(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linearize active residuals, choosing zero directions at collapsed pairs.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Full ``(n, 3)`` coordinates in original node order.

        Returns
        -------
        NDArray[np.float64]
            Owned ``(number_of_pairs, 3)`` directions, zero for inactive rows.
            At a bound knot, use the active-side derivative.
        """
        difference = coordinates[self.first] - coordinates[self.second]
        distance = np.linalg.norm(difference, axis=1)
        active = (distance <= self.lower) | (distance >= self.upper)
        return np.divide(
            difference,
            distance[:, None],
            out=np.zeros_like(difference),
            where=(active & (distance > 0.0))[:, None],
        )

    def dense_jacobian(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build all pair rows of the anchored dense Jacobian.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Full ``(n, 3)`` coordinates in original node order.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(number_of_pairs, 3 * (n - 1))``.
        """
        directions = self.directions(coordinates)
        result = np.zeros((len(self.first), 3 * (self.n - 1)), dtype=np.float64)
        rows = np.arange(len(self.first))
        free_first = self.first > 0
        first_columns = 3 * (self.first[free_first] - 1)[:, None] + np.arange(3)
        second_columns = 3 * (self.second - 1)[:, None] + np.arange(3)
        result[rows[:, None], second_columns] = -directions
        result[rows[free_first, None], first_columns] = directions[free_first]
        return result

    def compact_jacobian(
        self, coordinates: NDArray[np.float64], budget: SolveBudget, work: TRFWork
    ) -> LinearOperator:
        """Freeze active support while preserving the full original row space.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Full ``(n, 3)`` coordinates in original node order.
        budget : SolveBudget
            Shared deadline checked during preparation and every product.
        work : TRFWork
            Invocation-owned counters, including this Jacobian evaluation.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            Operator with shape ``(number_of_pairs, 3 * (n - 1))`` whose saved
            directions do not change when coordinates or later Jacobians change.
        """
        budget.check()
        work.jacobians += 1
        directions = self.directions(coordinates)
        rows = np.flatnonzero(np.any(directions != 0.0, axis=1))
        operator = _CompactJacobian(self, rows, directions[rows], budget, work)
        budget.check()
        return operator


class _CompactJacobian(LinearOperator):
    def __init__(
        self,
        pairs: PairResiduals,
        rows: NDArray[np.intp],
        directions: NDArray[np.float64],
        budget: SolveBudget,
        work: TRFWork,
    ) -> None:
        self.pairs = pairs
        self.rows = rows
        self.first = pairs.first[rows]
        self.second = pairs.second[rows]
        self.directions = directions
        self.budget = budget
        self.work = work
        super().__init__(
            dtype=np.dtype(np.float64), shape=(len(pairs.first), 3 * (pairs.n - 1))
        )

    def _matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.budget.check()
        self.work.matvecs += 1
        coordinates = self.pairs.unpack(x)
        result = np.zeros(len(self.pairs.first), dtype=np.float64)
        result[self.rows] = np.sum(
            self.directions * (coordinates[self.first] - coordinates[self.second]),
            axis=1,
        )
        self.budget.check()
        return result

    def _rmatvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.budget.check()
        self.work.rmatvecs += 1
        selected = x.ravel()[self.rows]
        result = np.empty((self.pairs.n, 3), dtype=np.float64)
        for axis in range(3):
            weights = self.directions[:, axis] * selected
            result[:, axis] = np.bincount(
                self.first, weights=weights, minlength=self.pairs.n
            ) - np.bincount(self.second, weights=weights, minlength=self.pairs.n)
        self.budget.check()
        return result[1:].ravel()


def solve_trf(
    constraints: ConstraintSet,
    initial: NDArray[np.float64],
    budget: SolveBudget,
    tolerance: float,
) -> tuple[NDArray[np.float64], TRFStage]:
    """Refine with dense TRF through 50 nodes, otherwise compact TRF/LSMR50.

    Parameters
    ----------
    constraints : ConstraintSet
        Original hard intervals admitted by the continuous solver.
    initial : NDArray[np.float64]
        Initial ``(num_nodes, 3)`` coordinates, never mutated.
    budget : SolveBudget
        Whole-call deadline shared by setup, callbacks and products.
    tolerance : float
        Maximum original-bound error for early feasible termination.

    Returns
    -------
    tuple[NDArray[np.float64], TRFStage]
        Owned lower-error candidate and native work/termination. Stationarity or
        an iteration limit is not an infeasibility certificate.

    Raises
    ------
    SolveError
        Numerical failure, retaining native work and the original exception.
    SolveTimeoutError
        Whole-call deadline exhaustion, retaining native work.
    """
    started = budget.elapsed
    backend: Literal["dense", "lsmr"] = (
        "dense" if constraints.num_nodes <= 50 else "lsmr"
    )
    work = TRFWork()
    termination: Termination = "stationary"

    def stage() -> TRFStage:
        return TRFStage(
            budget.elapsed - started,
            work.evaluations,
            work.jacobians,
            work.matvecs,
            work.rmatvecs,
            backend,
            termination,
        )

    try:
        budget.check()
        pairs = PairResiduals(constraints)
        retained = BestCandidate(constraints, initial)
        if not np.isfinite(retained.error):
            raise FloatingPointError("Nonfinite TRF initial candidate")
        if retained.error <= tolerance:
            termination = "feasible"
        else:
            translated = initial - initial[0]
            retained.consider(translated)

            def residual(vector: NDArray[np.float64]) -> NDArray[np.float64]:
                budget.check()
                work.evaluations += 1
                candidate = pairs.unpack(vector)
                values = pairs.residual(candidate)
                retained.consider(candidate)
                if retained.error <= tolerance:
                    raise _FeasibleFound
                budget.check()
                return values

            def jacobian(
                vector: NDArray[np.float64],
            ) -> NDArray[np.float64] | LinearOperator:
                budget.check()
                candidate = pairs.unpack(vector)
                if backend == "dense":
                    work.jacobians += 1
                    result = pairs.dense_jacobian(candidate)
                    budget.check()
                    return result
                return pairs.compact_jacobian(candidate, budget, work)

            try:
                result = least_squares(
                    residual,
                    translated[1:].ravel(),
                    jac=jacobian,
                    method="trf",
                    tr_solver="exact" if backend == "dense" else "lsmr",
                    tr_options={} if backend == "dense" else {"maxiter": 50},
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                    max_nfev=2000,
                )
            except _FeasibleFound:
                termination = "feasible"
            else:
                retained.consider(pairs.unpack(result.x))
                termination = "iteration_limit" if result.status == 0 else "stationary"
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
        raise SolveError(
            "TRF refinement failed numerically", stages=(stage(),)
        ) from error
    budget.check(stages=(stage(),))
    return retained.coordinates, stage()
