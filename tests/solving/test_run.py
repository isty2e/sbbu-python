"""Exercise shared deadlines and original-constraint publication."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from sbbu import SBBUConfig, SBBUSolver, SolveStats
from sbbu.adapters import MatrixAdapter
from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.solving.run import (
    MMStage,
    SBBUStage,
    SolveBudget,
    SolveError,
    SolveTimeoutError,
    finalize,
)


def test_budget_uses_an_absolute_monotonic_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    budget = SolveBudget.start(2.0)
    assert budget.deadline == 102.0
    clock[0] = 101.0
    assert budget.remaining == 1.0
    assert budget.elapsed == 1.0
    clock[0] = 102.0
    with pytest.raises(SolveTimeoutError):
        budget.check()
    with pytest.raises(SolveTimeoutError):
        _ = budget.remaining


@pytest.mark.parametrize(
    "coordinates",
    [
        np.zeros((3, 3)),
        np.full((4, 3), np.nan),
        np.zeros((4, 3)),
        np.ones((4, 3), dtype=complex),
    ],
)
def test_native_feasible_status_cannot_publish_bad_coordinates(coordinates):
    constraints = ConstraintSet(4, [DistanceConstraint(0, 1, 1.0, 1.0)])
    stage = MMStage(0.0, 1, 0, "feasible")
    with pytest.raises(SolveError):
        finalize(constraints, coordinates, SolveBudget.start(1.0), 1e-7, (stage,))


def test_feasible_coordinates_after_deadline_are_not_published(monkeypatch):
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: 2.0)
    constraints = ConstraintSet(3, [DistanceConstraint(0, 1, 1.0, 1.0)])
    points = np.zeros((3, 3))
    points[1, 0] = 1.0
    with pytest.raises(SolveTimeoutError):
        finalize(constraints, points, SolveBudget(0.0, 1.0), 1e-7, ())


def test_validation_cost_is_inside_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    constraints = ConstraintSet(3, [DistanceConstraint(0, 1, 1.0, 1.0)])
    points = np.zeros((3, 3))
    points[1, 0] = 1.0
    original = constraints.constraint_errors

    def expire_after_check(coordinates):
        errors = original(coordinates)
        clock[0] = 2.0
        return errors

    monkeypatch.setattr(constraints, "constraint_errors", expire_after_check)
    with pytest.raises(SolveTimeoutError):
        finalize(constraints, points, SolveBudget(0.0, 1.0), 1e-7, ())


@pytest.mark.parametrize(
    "field_name", ["solve_time", "mean_distance_error", "largest_distance_error"]
)
def test_stats_are_frozen_common_snapshots(simple_tetrahedron, field_name):
    _, constraints = simple_tetrahedron
    solver = SBBUSolver(constraints, SBBUConfig(verbose=False))
    stats = solver.solve()
    assert isinstance(stats, SolveStats)
    assert len(stats.stages) == 1 and isinstance(stats.stages[0], SBBUStage)
    with pytest.raises(FrozenInstanceError):
        setattr(stats, field_name, 0.0)
    before = stats.stages[0]
    solver.solve()
    assert stats.stages[0] is before
    np.testing.assert_allclose(
        constraints.constraint_errors(solver.solution_coordinates), 0, atol=1e-7
    )


def test_nested_soft_refinement_retains_the_identical_budget(monkeypatch):
    points = np.random.RandomState(1729).normal(size=(5, 3))
    constraints = [
        DistanceConstraint(
            i,
            j,
            float(np.linalg.norm(points[i] - points[j])),
            float(np.linalg.norm(points[i] - points[j])),
        )
        for j in range(5)
        for i in range(max(0, j - 3), j)
    ]
    base = SBBUSolver(ConstraintSet(5, constraints), SBBUConfig(verbose=False))
    base.solve()
    d04 = float(
        np.linalg.norm(base.solution_coordinates[0] - base.solution_coordinates[4])
    )
    constraints.extend(
        [
            DistanceConstraint(0, 4, d04 - 0.01, d04 + 0.01),
            DistanceConstraint(0, 4, d04 + 1.0, d04 + 1.1),
        ]
    )
    budgets = []
    original = SBBUSolver._solve

    def capture_budget(self, budget):
        budgets.append(budget)
        return original(self, budget)

    monkeypatch.setattr(SBBUSolver, "_solve", capture_budget)
    stats = SBBUSolver(ConstraintSet(5, constraints), SBBUConfig(verbose=False)).solve()
    assert len(budgets) >= 2
    assert all(budget is budgets[0] for budget in budgets)
    stage = stats.stages[0]
    assert isinstance(stage, SBBUStage)
    assert stage.soft_pruned_constraints == 1


def test_inactive_refinement_does_not_materialize_all_hard_pairs(monkeypatch):
    points = np.random.RandomState(42).normal(size=(8, 3))
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    problem = MatrixAdapter.from_distance_matrix(distances)

    def forbidden():
        raise AssertionError("Inactive refinement projected every pair")

    monkeypatch.setattr(problem, "get_hard_constraints", forbidden)
    solver = SBBUSolver(problem, SBBUConfig(verbose=False))
    solver.solve()
    assert problem.maximum_violation(solver.solution_coordinates) <= 1e-7


def test_finalizer_timeout_keeps_sbbu_diagnostics_and_clears_previous_result(
    monkeypatch, simple_tetrahedron
):
    _, problem = simple_tetrahedron
    clock = [0.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    solver = SBBUSolver(problem, SBBUConfig(verbose=False))
    solver.solve(max_time=1.0)
    original = problem.constraint_errors

    def delayed_errors(coordinates):
        result = original(coordinates)
        clock[0] = 2.0
        return result

    def delayed_finalize(constraints, coordinates, budget, tolerance, stages):
        monkeypatch.setattr(problem, "constraint_errors", delayed_errors)
        return finalize(constraints, coordinates, budget, tolerance, stages)

    monkeypatch.setattr("sbbu.solving.sbbu.solver.finalize", delayed_finalize)
    with pytest.raises(SolveTimeoutError) as caught:
        solver.solve(max_time=1.0)
    assert len(caught.value.stages) == 1
    assert isinstance(caught.value.stages[0], SBBUStage)
    with pytest.raises(RuntimeError):
        _ = solver.solution_coordinates
