"""Continuous derivatives, original-edge oracles and stage transitions."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

import sbbu.solving.continuous.solve as scheduler
from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.solving.continuous import initialization, mm, trf
from sbbu.solving.continuous.state import BestCandidate
from sbbu.solving.run import (
    MMStage,
    SolveBudget,
    SolveError,
    SolveTimeoutError,
    TRFStage,
)


def _problem(n: int = 6) -> tuple[ConstraintSet, NDArray[np.float64]]:
    rng = np.random.RandomState(1729 + n)
    coordinates = rng.normal(size=(n, 3))
    rows, columns = np.triu_indices(n, k=1)
    distances = np.linalg.norm(coordinates[rows] - coordinates[columns], axis=1)
    alpha = rng.uniform(0.1, 0.9, size=len(rows))
    problem = ConstraintSet.from_bounds_arrays(
        n,
        rows,
        columns,
        distances * (1 - 0.05 * alpha),
        distances * (1 + 0.05 * (1 - alpha)),
    )
    return problem, coordinates


def _oracle(problem: ConstraintSet, coordinates: NDArray[np.float64]) -> float:
    if coordinates.shape != (problem.num_nodes, 3):
        return math.inf
    if not all(math.isfinite(float(value)) for value in coordinates.flat):
        return math.inf
    errors = []
    for i, j, lower, upper in zip(
        problem.rows,
        problem.columns,
        problem.lower_bounds,
        problem.upper_bounds,
        strict=True,
    ):
        distance = math.dist(
            [float(value) for value in coordinates[i]],
            [float(value) for value in coordinates[j]],
        )
        errors.append(max(float(lower) - distance, distance - float(upper), 0.0))
    return max(errors, default=0.0)


def _active_problem():
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.2, 0.0], [0.2, 1.1, 0.1], [0.1, 0.2, 1.3]]
    )
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    observations = []
    for index, (i, j) in enumerate(pairs):
        distance = float(np.linalg.norm(coordinates[i] - coordinates[j]))
        if index in (0, 3, 5):
            lower, upper = distance + 0.3, distance + 0.6
        elif index in (1, 4):
            lower, upper = distance - 0.6, distance - 0.3
        else:
            lower, upper = distance - 0.2, distance + 0.2
        observations.append(DistanceConstraint(i, j, lower, upper))
    return trf.PairResiduals(ConstraintSet(4, observations)), coordinates


def test_active_anchored_and_both_free_derivatives():
    pairs, coordinates = _active_problem()
    vector = coordinates[1:].ravel()
    jacobian = pairs.dense_jacobian(coordinates)
    residual = pairs.residual(coordinates)
    assert residual[0] < 0 and residual[1] > 0
    assert residual[3] < 0 and residual[4] > 0
    assert np.count_nonzero(jacobian[3]) == 6
    finite_difference = np.empty_like(jacobian)
    for index in range(vector.size):
        plus, minus = vector.copy(), vector.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite_difference[:, index] = (
            pairs.residual(pairs.unpack(plus)) - pairs.residual(pairs.unpack(minus))
        ) / 2e-6
    np.testing.assert_allclose(jacobian, finite_difference, atol=1e-8)


def test_compact_adjoint_matrix_products_and_frozen_support():
    pairs, coordinates = _active_problem()
    work = trf.TRFWork()
    operator = pairs.compact_jacobian(coordinates, SolveBudget.start(1.0), work)
    dense = pairs.dense_jacobian(coordinates)
    rng = np.random.RandomState(6)
    vector, weights = rng.normal(size=9), rng.normal(size=6)
    matrix = rng.normal(size=(9, 3))
    np.testing.assert_allclose(operator @ vector, dense @ vector)
    np.testing.assert_allclose(operator.T @ weights, dense.T @ weights)
    np.testing.assert_allclose(operator.matmat(matrix), dense @ matrix)
    assert np.dot(operator @ vector, weights) == pytest.approx(
        np.dot(vector, operator.T @ weights)
    )
    coordinates[1:] *= 3
    pairs.compact_jacobian(coordinates, SolveBudget.start(1.0), work)
    assert not np.allclose(dense, pairs.dense_jacobian(coordinates))
    np.testing.assert_allclose(operator @ vector, dense @ vector)
    np.testing.assert_allclose(operator.T @ weights, dense.T @ weights)
    assert work.jacobians == 2 and work.matvecs > 0 and work.rmatvecs > 0


@pytest.mark.parametrize("collapsed", [False, True])
def test_zero_support_keeps_all_original_rows(collapsed):
    problem, coordinates = _problem()
    if collapsed:
        coordinates[:] = 0
    pairs = trf.PairResiduals(problem)
    operator = pairs.compact_jacobian(
        coordinates, SolveBudget.start(1.0), trf.TRFWork()
    )
    assert operator.shape == (15, 15)
    np.testing.assert_array_equal(operator @ np.ones(15), np.zeros(15))
    np.testing.assert_array_equal(operator.T @ np.ones(15), np.zeros(15))
    if collapsed:
        assert np.all(pairs.residual(coordinates) < 0)


def test_product_deadlines_check_after_work_too(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    pairs, coordinates = _active_problem()
    operator = pairs.compact_jacobian(coordinates, SolveBudget(0.0, 1.0), trf.TRFWork())
    original = pairs.unpack

    def expire(vector):
        result = original(vector)
        clock[0] = 2.0
        return result

    monkeypatch.setattr(pairs, "unpack", expire)
    with pytest.raises(SolveTimeoutError):
        operator @ np.ones(9)
    with pytest.raises(SolveTimeoutError):
        operator.T @ np.ones(6)


@pytest.mark.parametrize("method", ["mm", "trf"])
def test_real_refinement_against_independent_original_edge_oracle(method):
    problem, witness = _problem()
    initial = witness + np.random.RandomState(19).normal(scale=0.1, size=witness.shape)
    saved, lower, upper = (
        initial.copy(),
        problem.lower_bounds.copy(),
        problem.upper_bounds.copy(),
    )
    refine = mm.solve_mm if method == "mm" else trf.solve_trf
    candidate, report = refine(problem, initial, SolveBudget.start(1.0), 1e-7)
    assert _oracle(problem, candidate) <= 1e-7
    assert report.termination == "feasible"
    np.testing.assert_array_equal(initial, saved)
    np.testing.assert_array_equal(problem.lower_bounds, lower)
    np.testing.assert_array_equal(problem.upper_bounds, upper)


def test_collapsed_mm_can_lift_missing_directions():
    problem, witness = _problem(4)
    initial = np.zeros_like(witness)
    candidate, report = mm.solve_mm(problem, initial, SolveBudget.start(1.0), 1e-7)
    assert report.rank_lifts == 3
    assert _oracle(problem, candidate) <= 1e-7
    np.testing.assert_array_equal(initial, 0)


def test_mm_lift_preserves_existing_plane_and_local_rng():
    problem, coordinates = _problem()
    coordinates[:, 2] = 0.0
    coordinates -= coordinates.mean(axis=0)
    original_rng = np.random.get_state()
    work = mm._MMWork()
    budget = SolveBudget.start(1.0)
    lifted = mm._rank_lift(problem, coordinates, budget, budget.deadline, work)
    np.testing.assert_allclose(lifted[:, :2], coordinates[:, :2], atol=1e-12)
    assert np.linalg.matrix_rank(lifted) == 3 and work.rank_lifts == 1
    after_rng = np.random.get_state()
    np.testing.assert_equal(original_rng, after_rng)


def test_prefix_uses_budget_clock_not_perf_counter(monkeypatch):
    origin = 1e12
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: origin + 2.0)
    problem, points = _problem()
    initial = np.zeros_like(points)
    candidate, report = mm.solve_mm(
        problem, initial, SolveBudget(origin, origin + 5), 1e-7, deadline=origin + 1
    )
    assert report.termination == "deadline" and report.steps == 0
    np.testing.assert_array_equal(candidate, initial)


@pytest.mark.parametrize("method", ["mm", "trf", None])
@pytest.mark.parametrize("n", [50, 51])
def test_explicit_methods_and_automatic_size_boundary(monkeypatch, method, n):
    problem, witness = _problem(n)
    initial = np.zeros_like(witness)
    calls = []
    budget = SolveBudget.start(1.0)
    monkeypatch.setattr(scheduler, "partial_spectral", lambda *args: initial)

    def run_mm(constraints, points, received_budget, tolerance, *, deadline):
        assert received_budget is budget
        calls.append("mm")
        if method == "mm":
            assert deadline == budget.deadline
        else:
            assert deadline <= budget.started + 0.1
        return (witness if method == "mm" else initial), MMStage(
            0.0, 1, 0, "stationary"
        )

    def run_trf(constraints, points, received_budget, tolerance):
        assert received_budget is budget
        calls.append("trf")
        return witness, TRFStage(
            0.0, 1, 1, 0, 0, "dense" if n <= 50 else "lsmr", "stationary"
        )

    monkeypatch.setattr(scheduler, "solve_mm", run_mm)
    monkeypatch.setattr(scheduler, "solve_trf", run_trf)
    candidate, reports = scheduler.solve(problem, budget, 1e-7, method)
    expected = (
        ["mm"]
        if method == "mm"
        else ["mm", "trf"]
        if method is None and n <= 50
        else ["trf"]
    )
    assert calls == expected
    assert [report.algorithm for report in reports] == expected
    assert _oracle(problem, candidate) <= 1e-7


def test_scheduler_keeps_lower_error_before_trf(monkeypatch):
    problem, witness = _problem()
    initial = witness * 0.8
    worse = np.zeros_like(witness)
    assert _oracle(problem, initial) < _oracle(problem, worse)
    monkeypatch.setattr(scheduler, "partial_spectral", lambda *args: initial)
    monkeypatch.setattr(
        scheduler,
        "solve_mm",
        lambda *args, **kwargs: (worse, MMStage(0.0, 1, 0, "stationary")),
    )

    def finish(constraints, points, budget, tolerance):
        np.testing.assert_array_equal(points, initial)
        return witness, TRFStage(0.0, 1, 1, 0, 0, "dense", "stationary")

    monkeypatch.setattr(scheduler, "solve_trf", finish)
    candidate, reports = scheduler.solve(problem, SolveBudget.start(1.0), 1e-7, None)
    assert _oracle(problem, candidate) <= 1e-7
    assert reports[-1].termination == "stationary"


@pytest.mark.parametrize("failure", ["raised", "returned", "invalid", "whole-deadline"])
def test_mm_failure_does_not_silently_switch(monkeypatch, failure):
    clock = [0.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    problem, witness = _problem()
    monkeypatch.setattr(
        scheduler, "partial_spectral", lambda *args: np.zeros_like(witness)
    )

    def fail(*args, **kwargs):
        if failure == "raised":
            raise FloatingPointError("MM backend failed")
        if failure == "whole-deadline":
            clock[0] = 2.0
        candidate = np.full_like(witness, np.nan) if failure == "invalid" else witness
        termination = "numerical_error" if failure == "returned" else "deadline"
        return candidate, MMStage(0.0, 4, 0, termination)

    def forbidden(*args, **kwargs):
        raise AssertionError("TRF must not follow failed MM")

    monkeypatch.setattr(scheduler, "solve_mm", fail)
    monkeypatch.setattr(scheduler, "solve_trf", forbidden)
    with pytest.raises(SolveError) as caught:
        scheduler.solve(problem, SolveBudget(0.0, 1.0), 1e-7, None)
    assert len(caught.value.stages) == 1
    report = caught.value.stages[0]
    assert isinstance(report, MMStage)
    assert report.steps is None if failure == "raised" else report.steps == 4
    if failure == "raised":
        assert isinstance(caught.value.__cause__, FloatingPointError)
    if failure == "whole-deadline":
        assert isinstance(caught.value, SolveTimeoutError)


def test_prefix_expiry_can_switch_without_refresh(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    problem, witness = _problem()
    initial = np.zeros_like(witness)
    budget = SolveBudget(100.0, 101.0)
    monkeypatch.setattr(scheduler, "partial_spectral", lambda *args: initial)

    def prefix(*args, deadline, **kwargs):
        assert deadline == 100.05
        clock[0] = 100.06
        return initial, MMStage(0.06, 0, 0, "deadline")

    def finish(constraints, points, received_budget, tolerance):
        assert received_budget is budget and received_budget.deadline == 101.0
        return witness, TRFStage(0.0, 1, 1, 0, 0, "dense", "feasible")

    monkeypatch.setattr(scheduler, "solve_mm", prefix)
    monkeypatch.setattr(scheduler, "solve_trf", finish)
    assert _oracle(problem, scheduler.solve(problem, budget, 1e-7, None)[0]) <= 1e-7


@pytest.mark.parametrize("n", [6, 51])
def test_trf_backend_and_native_status_are_not_certification(monkeypatch, n):
    problem, witness = _problem(n)
    initial = np.zeros_like(witness)

    def optimizer(fun, x0, *, jac, tr_solver, tr_options, **kwargs):
        fun(x0)
        derivative = jac(x0)
        assert isinstance(derivative, LinearOperator) == (n > 50)
        assert tr_solver == ("lsmr" if n > 50 else "exact")
        assert tr_options == ({"maxiter": 50} if n > 50 else {})
        return SimpleNamespace(x=x0, status=1)

    monkeypatch.setattr(trf, "least_squares", optimizer)
    candidate, report = trf.solve_trf(problem, initial, SolveBudget.start(1.0), 1e-7)
    assert report.termination == "stationary"
    assert _oracle(problem, candidate) > 1e-7
    assert report.evaluations == 1 and report.jacobians == 1


@pytest.mark.parametrize(
    "failure_type", [FloatingPointError, ValueError, np.linalg.LinAlgError]
)
def test_mm_and_trf_keep_reported_work_on_failure(monkeypatch, failure_type):
    problem, witness = _problem()
    initial = witness * 0.8

    def failed_step(*args):
        raise failure_type("numeric MM")

    monkeypatch.setattr(mm, "stress_step", failed_step)
    with pytest.raises(SolveError) as caught:
        mm.solve_mm(problem, initial, SolveBudget.start(1.0), 1e-7)
    report = caught.value.stages[0]
    assert isinstance(report, MMStage) and report.steps == 0
    assert isinstance(caught.value.__cause__, failure_type)

    def failed_optimizer(fun, x0, **kwargs):
        fun(x0)
        raise failure_type("numeric TRF")

    monkeypatch.setattr(trf, "least_squares", failed_optimizer)
    with pytest.raises(SolveError) as caught:
        trf.solve_trf(problem, initial, SolveBudget.start(1.0), 1e-7)
    report = caught.value.stages[0]
    assert isinstance(report, TRFStage) and report.evaluations == 1
    assert isinstance(caught.value.__cause__, failure_type)


def test_candidate_retention_is_owned_and_not_an_average():
    problem, witness = _problem()
    best = BestCandidate(problem, witness)
    worse = np.zeros_like(witness)
    best.consider(worse)
    np.testing.assert_array_equal(best.coordinates, witness)
    witness[:] = 0
    assert _oracle(problem, best.coordinates) == 0.0


def test_spectral_uses_largest_algebraic_and_old_scipy_v0(monkeypatch):
    problem, _ = _problem(4)
    calls = []

    def spectrum(operator, *, k, which, tol, maxiter, v0):
        assert k == 3 and which == "LA" and tol == 1e-6 and maxiter == 300
        calls.append(v0.copy())
        dense = operator.matmat(np.eye(4))
        values, vectors = np.linalg.eigh(dense)
        return values[-3:], vectors[:, -3:]

    monkeypatch.setattr(initialization, "eigsh", spectrum)
    first = initialization.partial_spectral(problem, SolveBudget.start(1.0))
    second = initialization.partial_spectral(problem, SolveBudget.start(1.0))
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 3) and np.isfinite(first).all()


def test_returned_trf_numerical_error_cannot_be_hidden_by_good_coordinates(monkeypatch):
    problem, witness = _problem()
    monkeypatch.setattr(
        scheduler, "partial_spectral", lambda *args: np.zeros_like(witness)
    )
    report = TRFStage(0.0, 7, 3, 0, 0, "dense", "numerical_error")
    monkeypatch.setattr(scheduler, "solve_trf", lambda *args: (witness, report))
    with pytest.raises(SolveError, match="numerical_error") as caught:
        scheduler.solve(problem, SolveBudget.start(1.0), 1e-7, "trf")
    assert caught.value.stages == (report,)


def test_expired_call_never_enters_initializer(monkeypatch):
    problem, _ = _problem()

    def forbidden(*args):
        raise AssertionError("Expired call entered initialization")

    monkeypatch.setattr(scheduler, "partial_spectral", forbidden)
    with pytest.raises(SolveTimeoutError):
        scheduler.solve(problem, SolveBudget(0.0, 0.0), 1e-7, None)


def test_acceleration_rejects_worse_stress_and_bounds_history(monkeypatch):
    problem, witness = _problem()
    seen, plain_images, columns = [], [], []

    def plain_step(constraints, current):
        seen.append(current.copy())
        if len(seen) == 9:
            raise RuntimeError("bounded safeguard probe")
        proposal = current * 0.99
        plain_images.append(proposal.copy())
        return proposal

    def aggressive_weights(matrix, residual, *, rcond):
        columns.append(matrix.shape[1])
        return (np.full(matrix.shape[1], 1e12),)

    monkeypatch.setattr(mm, "stress_step", plain_step)
    monkeypatch.setattr(mm.np.linalg, "lstsq", aggressive_weights)
    with pytest.raises(SolveError):
        mm.solve_mm(problem, witness * 2, SolveBudget.start(1.0), 1e-7)
    assert len(seen) == 9 and max(columns) == 5
    for actual, plain in zip(seen[1:], plain_images, strict=True):
        np.testing.assert_array_equal(actual, plain)
