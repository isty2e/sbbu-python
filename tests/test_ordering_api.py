"""Reordered SBBU through public adapters and original-order publication."""

import math

import numpy as np
import pytest

import sbbu
import sbbu.solving as operation
from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.solving.run import SolveError, SolveTimeoutError, UnsupportedProblemError

_CHAIN = (4, 1, 6, 0, 7, 2, 5, 3)


@pytest.fixture
def mixed_problem():
    points = np.random.RandomState(17).normal(size=(8, 3))
    rank = np.argsort(_CHAIN)
    observations = []
    for first in range(8):
        for second in range(first + 1, 8):
            distance = math.dist(points[first], points[second])
            width = 0.0 if abs(rank[first] - rank[second]) <= 3 else 0.01
            observations.append(
                DistanceConstraint(first, second, distance - width, distance + width)
            )
    return ConstraintSet(8, observations)


@pytest.mark.parametrize("method", [None, "mm", "trf"])
def test_all_entrypoints_restore_original_ids_and_verify_original_edges(
    mixed_problem, tmp_path, method
):
    problem = mixed_problem
    assert operation.SBBUSolver.applicability_error(problem) is not None
    lower, upper = problem.bounds_matrices()
    distances = np.where(lower == upper, lower, np.nan)
    snapshots = [array.copy() for array in (lower, upper, distances)]
    edges = [(c.i, c.j, c.lower_bound, c.upper_bound) for c in problem.constraints]
    nmr = tmp_path / "reordered.nmr"
    nmr.write_text(
        "".join(
            f"{i + 1} {j + 1} {lo:.17g} {hi:.17g} C C ALA ALA\n"
            for i, j, lo, hi in edges
        )
    )
    exact_edges = [edge for edge in edges if edge[2] == edge[3]]
    results = [
        (
            sbbu.solve_from_distance_matrix(distances, method=method, verbose=False),
            exact_edges,
        ),
        (
            sbbu.solve_from_bounds_matrices(lower, upper, method=method, verbose=False),
            edges,
        ),
        (sbbu.solve_from_edge_list(8, edges, method=method, verbose=False), edges),
        (sbbu.solve_from_nmr(nmr, method=method, verbose=False), edges),
    ]
    for (coordinates, stats), original_edges in results:
        assert coordinates.shape == (8, 3) and np.isfinite(coordinates).all()
        errors = [
            max(
                lo - math.dist(coordinates[i], coordinates[j]),
                math.dist(coordinates[i], coordinates[j]) - hi,
                0.0,
            )
            for i, j, lo, hi in original_edges
        ]
        assert max(errors) <= 1e-7
        assert stats.largest_distance_error == pytest.approx(max(errors), abs=1e-14)
        assert stats.mean_distance_error == pytest.approx(np.mean(errors), abs=1e-14)
        assert len(stats.stages) == 1 and stats.stages[0].algorithm == "sbbu"
    for array, snapshot in zip((lower, upper, distances), snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def test_direct_solver_still_requires_current_order(mixed_problem):
    with pytest.raises(ValueError, match="Sequential constraint"):
        sbbu.SBBUSolver(mixed_problem)


def test_reordering_preserves_soft_alternatives_through_refinement(mixed_problem):
    observations = mixed_problem.constraints
    index = next(i for i, c in enumerate(observations) if not c.is_exact)
    interval = observations.pop(index)
    observations.extend(
        [
            interval,
            DistanceConstraint(
                interval.i,
                interval.j,
                interval.upper_bound + 10.0,
                interval.upper_bound + 11.0,
            ),
        ]
    )
    problem = ConstraintSet(8, observations)
    coordinates, stats = operation.solve(problem, 1e-7, 1.0, False, None)
    assert problem.maximum_violation(coordinates) <= 1e-7
    assert problem.constraints == observations
    assert len(problem.get_soft_ambiguous_constraints()) == 1
    stage = stats.stages[0]
    assert isinstance(stage, sbbu.SBBUStage)
    assert stage.unresolved_soft_constraints + stage.soft_pruned_constraints == 1


@pytest.mark.parametrize("failure", [SolveError, SolveTimeoutError, ValueError])
def test_reordered_sbbu_failure_does_not_retry(mixed_problem, monkeypatch, failure):
    calls = []
    fault = failure("selected SBBU failed")

    def find(constraints, budget):
        calls.append("ordering")
        return list(_CHAIN)

    def fail(*args):
        calls.append("sbbu")
        raise fault

    def forbidden(*args):
        raise AssertionError("Unexpected continuous fallback")

    monkeypatch.setattr(operation, "find_order", find)
    monkeypatch.setattr(operation.SBBUSolver, "_solve", fail)
    monkeypatch.setattr(operation, "solve_continuous", forbidden)
    with pytest.raises(failure, match="selected SBBU failed") as caught:
        operation.solve(mixed_problem, 1e-7, 1.0, False, "trf")
    assert caught.value is fault
    if failure is SolveTimeoutError:
        assert str(fault) == "selected SBBU failed"
    assert calls == ["ordering", "sbbu"]


def test_constraint_failure_identifies_the_working_to_input_map(
    mixed_problem, monkeypatch
):
    observations = mixed_problem.constraints
    index = next(i for i, c in enumerate(observations) if not c.is_exact)
    pair = observations[index]
    observations[index] = DistanceConstraint(pair.i, pair.j, 100.0, 101.0)
    monkeypatch.setattr(operation, "find_order", lambda *args: list(_CHAIN))
    with pytest.raises(SolveError) as caught:
        operation.solve(ConstraintSet(8, observations), 1e-7, 1.0, False, None)
    first, second = sorted((_CHAIN.index(pair.i) + 1, _CHAIN.index(pair.j) + 1))
    assert f"Constraint ({first}, {second}," in str(caught.value)
    assert f"input IDs in solver order: {list(_CHAIN)}" in str(caught.value)
    assert "one-based positions" in str(caught.value)


def test_no_found_order_is_not_an_infeasibility_claim(mixed_problem, monkeypatch):
    monkeypatch.setattr(operation, "find_order", lambda *args: None)
    with pytest.raises(UnsupportedProblemError, match="does not prove"):
        operation.solve(mixed_problem, 1e-7, 1.0, False, None)


@pytest.mark.parametrize("order", [None, list(_CHAIN)])
def test_whole_deadline_wins_over_ordering_result(mixed_problem, monkeypatch, order):
    now = [100.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: now[0])

    def late(*args):
        now[0] = 101.0
        return order

    monkeypatch.setattr(operation, "find_order", late)
    with pytest.raises(SolveTimeoutError):
        operation.solve(mixed_problem, 1e-7, 1.0, False, None)


def test_search_and_projection_share_the_original_budget(mixed_problem, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: now[0])
    seen = []
    original_solve = operation.SBBUSolver._solve
    original_reordered = ConstraintSet.reordered

    def find(constraints, budget):
        seen.append(budget)
        now[0] += 0.02
        return list(_CHAIN)

    def reordered(self, order):
        result = original_reordered(self, order)
        now[0] += 0.03
        return result

    def solve(self, budget):
        assert budget is seen[0] and budget.deadline == 101.0
        return original_solve(self, budget)

    monkeypatch.setattr(operation, "find_order", find)
    monkeypatch.setattr(ConstraintSet, "reordered", reordered)
    monkeypatch.setattr(operation.SBBUSolver, "_solve", solve)
    coordinates, stats = operation.solve(mixed_problem, 1e-7, 1.0, False, None)
    assert mixed_problem.maximum_violation(coordinates) <= 1e-7
    assert stats.solve_time == pytest.approx(0.05)


def test_projection_expiry_prevents_solver_entry(mixed_problem, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: now[0])
    original = ConstraintSet.reordered

    def late(self, order):
        result = original(self, order)
        now[0] = 101.0
        return result

    def forbidden(*args):
        raise AssertionError("Expired projection entered SBBU")

    monkeypatch.setattr(ConstraintSet, "reordered", late)
    monkeypatch.setattr(operation.SBBUSolver, "_solve", forbidden)
    with pytest.raises(SolveTimeoutError):
        operation.solve(mixed_problem, 1e-7, 1.0, False, None)


def test_publication_validates_original_not_remapped_bounds(mixed_problem, monkeypatch):
    original = ConstraintSet.reordered

    def corrupted(self, order):
        result = original(self, order)
        return ConstraintSet(
            result.num_nodes,
            [
                DistanceConstraint(c.i, c.j, c.lower_bound * 2, c.upper_bound * 2)
                for c in result.constraints
            ],
        )

    monkeypatch.setattr(ConstraintSet, "reordered", corrupted)
    with pytest.raises(SolveError, match="Original hard constraints"):
        operation.solve(mixed_problem, 1e-7, 1.0, False, None)


def test_final_original_validation_cannot_publish_after_deadline(
    mixed_problem, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: now[0])
    original = operation.finalize

    def late(*args):
        now[0] = 101.0
        return original(*args)

    monkeypatch.setattr(operation, "finalize", late)
    with pytest.raises(SolveTimeoutError) as caught:
        operation.solve(mixed_problem, 1e-7, 1.0, False, None)
    assert len(caught.value.stages) == 1
