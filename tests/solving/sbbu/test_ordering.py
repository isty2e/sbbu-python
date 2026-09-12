"""Exact-band search, bounded work, and non-selection semantics."""

import numpy as np
import pytest

from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.solving.run import SolveBudget, SolveTimeoutError
from sbbu.solving.sbbu import ordering
from sbbu.solving.sbbu.solver import SBBUSolver


def _graph(n: int, pairs: list[tuple[int, int]]) -> ConstraintSet:
    return ConstraintSet(n, [DistanceConstraint(i, j, 1.0, 1.0) for i, j in pairs])


def _band(n: int) -> ConstraintSet:
    return _graph(n, [(i, j) for i in range(n) for j in range(i + 1, min(i + 4, n))])


def test_search_backtracks_out_of_a_greedy_dead_end(monkeypatch):
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: 100.0)
    pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 6),
        (0, 7),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (3, 5),
        (3, 6),
        (3, 7),
        (4, 5),
        (4, 7),
        (5, 6),
        (5, 7),
        (6, 7),
    ]
    problem = _graph(8, pairs)
    adjacency = [set() for _ in range(8)]
    for first, second in pairs:
        adjacency[first].add(second)
        adjacency[second].add(first)
    # The degree/ID-first prefix 4,1,5 has no unused common neighbor.
    assert not (adjacency[4] & adjacency[1] & adjacency[5]) - {4, 1, 5}
    order = ordering.find_order(problem, SolveBudget.start(1.0))
    assert order is not None
    assert sorted(order) == list(range(8))
    assert SBBUSolver.applicability_error(problem.reordered(order)) is None
    assert ordering.find_order(problem, SolveBudget.start(1.0)) == order


def test_long_chain_search_does_not_depend_on_recursion_depth(monkeypatch):
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: 100.0)
    problem = _band(1200)
    permutation = np.random.RandomState(19).permutation(1200).tolist()
    shuffled = problem.reordered(permutation)
    order = ordering.find_order(shuffled, SolveBudget.start(1.0))
    assert order is not None
    assert SBBUSolver.applicability_error(shuffled.reordered(order)) is None


def test_positive_width_and_ambiguous_edges_cannot_complete_the_band():
    problem = _band(5)
    observations = problem.constraints
    last = observations.pop()
    observations.append(DistanceConstraint(last.i, last.j, 1.0, np.nextafter(1.0, 2.0)))
    assert (
        ordering.find_order(ConstraintSet(5, observations), SolveBudget.start(1.0))
        is None
    )
    observations[-1] = last
    observations.append(DistanceConstraint(last.i, last.j, 2.0, 2.0))
    assert (
        ordering.find_order(ConstraintSet(5, observations), SolveBudget.start(1.0))
        is None
    )


def test_duplicate_intersection_can_supply_an_exact_edge():
    observations = _band(5).constraints
    last = observations.pop()
    observations.extend(
        [
            DistanceConstraint(last.i, last.j, 0.9, 1.0),
            DistanceConstraint(last.i, last.j, 1.0, 1.1),
        ]
    )
    problem = ConstraintSet(5, observations)
    order = ordering.find_order(problem, SolveBudget.start(1.0))
    assert order is not None
    assert SBBUSolver.applicability_error(problem.reordered(order)) is None


def test_work_limit_stops_search_even_with_a_frozen_clock(monkeypatch):
    calls = [0]

    def clock():
        calls[0] += 1
        return 100.0

    monkeypatch.setattr("sbbu.solving.run.get_wall_time", clock)
    monkeypatch.setattr(ordering, "_MIN_VISIT_LIMIT", 0)
    pairs = [
        (i, j)
        for start in (0, 6)
        for i in range(start, start + 6)
        for j in range(i + 1, start + 6)
    ]
    assert ordering.find_order(_graph(12, pairs), SolveBudget.start(1.0)) is None
    assert calls[0] < 150


def test_search_slice_includes_graph_construction_but_does_not_expire_whole_budget(
    monkeypatch,
):
    now = [100.0]

    def clock():
        now[0] += 0.001
        return now[0]

    monkeypatch.setattr("sbbu.solving.run.get_wall_time", clock)
    budget = SolveBudget.start(1.0)
    assert ordering.find_order(_band(4000), budget) is None
    assert budget.remaining > 0.9
    assert budget.elapsed >= 0.05


def test_whole_deadline_is_not_an_ordering_miss(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: now[0])
    budget = SolveBudget.start(1.0)
    now[0] = 101.0
    with pytest.raises(SolveTimeoutError):
        ordering.find_order(_band(5), budget)
