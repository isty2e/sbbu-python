from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import sbbu.solving.sbbu.solver as solver_module
from sbbu import SBBUConfig, SBBUSolver, solve_from_distance_matrix
from sbbu.adapters import MatrixAdapter
from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.solving.run import SBBUStage, SolveStats
from sbbu.solving.sbbu.state import SBBUTimeoutError


def _sbbu_stage(stats: SolveStats) -> SBBUStage:
    assert len(stats.stages) == 1
    stage = stats.stages[0]
    assert isinstance(stage, SBBUStage)
    return stage


@pytest.fixture
def issue_distances() -> NDArray[np.float64]:
    return np.loadtxt(Path(__file__).parent / "test_data" / "issue1_distances.txt")


def distances_of(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)


@pytest.mark.parametrize("permutation_seed", range(12))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_issue_distance_matrix_is_recovered_after_relabeling(
    issue_distances, permutation_seed, dtype
):
    original = issue_distances.astype(dtype)
    permutation = np.random.RandomState(permutation_seed).permutation(len(original))
    supplied = original[np.ix_(permutation, permutation)]
    before = supplied.copy()

    coordinates, stats = solve_from_distance_matrix(
        supplied, distance_tolerance=1e-4, max_time=1.0, verbose=False
    )

    restored = coordinates[np.argsort(permutation)]
    errors = np.abs(distances_of(restored) - original.astype(np.float64))
    assert np.isfinite(restored).all()
    assert np.max(errors) <= 1e-4
    assert stats.largest_distance_error == pytest.approx(np.max(errors), abs=1e-12)
    np.testing.assert_array_equal(supplied, before)


@pytest.mark.parametrize("swap", [False, True])
def test_issue_diameter_endpoint_swap(issue_distances, swap):
    order = np.arange(len(issue_distances))
    if swap:
        order[0], order[31] = order[31], order[0]
    supplied = issue_distances[np.ix_(order, order)]
    solver = SBBUSolver(
        MatrixAdapter.from_distance_matrix(supplied),
        SBBUConfig(distance_tolerance=1e-4, max_time=1.0, verbose=False),
    )
    solver.solve()
    restored = solver.solution_coordinates[np.argsort(order)]
    assert np.max(np.abs(distances_of(restored) - issue_distances)) <= 1e-4


def test_exact_sparse_issue_does_not_commit_first_approximate_branch(issue_distances):
    count = len(issue_distances)
    centering = np.eye(count) - np.ones((count, count)) / count
    values, vectors = np.linalg.eigh(-0.5 * centering @ issue_distances**2 @ centering)
    exact_coordinates = vectors[:, -3:] * np.sqrt(values[-3:])
    supplied = distances_of(exact_coordinates)
    supplied[0, -1] = supplied[-1, 0] = np.nan
    coordinates, _ = solve_from_distance_matrix(
        supplied, distance_tolerance=1e-4, max_time=1.0, verbose=False
    )
    known = np.isfinite(supplied)
    assert np.max(np.abs(distances_of(coordinates)[known] - supplied[known])) <= 1e-4


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("thickness", [1.0, 1e-4])
def test_exact_anisotropic_cliques_preserve_units(scale, thickness):
    points = np.random.RandomState(1409).normal(size=(30, 3))
    points[:, 2] *= thickness
    supplied = distances_of(points) * scale
    coordinates, _ = solve_from_distance_matrix(
        supplied, distance_tolerance=1e-6 * scale, max_time=1.0, verbose=False
    )
    assert np.max(np.abs(distances_of(coordinates) - supplied)) <= 1e-6 * scale


@pytest.mark.parametrize("permutation_seed", range(4))
def test_symmetric_cube_relabeling(permutation_seed):
    points = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    order = np.random.RandomState(permutation_seed).permutation(len(points))
    supplied = distances_of(points[order])
    coordinates, _ = solve_from_distance_matrix(
        supplied, distance_tolerance=1e-6, max_time=1.0, verbose=False
    )
    assert np.max(np.abs(distances_of(coordinates) - supplied)) <= 1e-6


@pytest.mark.parametrize("scalar_type", [float, np.float32, np.float64])
def test_distance_scalars_are_normalized_before_arithmetic(scalar_type):
    value = scalar_type(1.3734981)
    constraint = DistanceConstraint(0, 1, value, value)
    assert constraint.lower_bound**2 == float(value) ** 2


def test_inconsistent_complete_matrix_is_not_published():
    points = np.random.RandomState(8).normal(size=(10, 3))
    supplied = distances_of(points)
    supplied[0, -1] = supplied[-1, 0] = 100.0
    solver = SBBUSolver(
        MatrixAdapter.from_distance_matrix(supplied),
        SBBUConfig(distance_tolerance=1e-6, max_time=1.0, verbose=False),
    )
    with pytest.raises(RuntimeError):
        solver.solve()
    with pytest.raises(RuntimeError, match="No solved coordinates"):
        _ = solver.solution_coordinates


def test_interval_and_ambiguous_edges_keep_their_meanings():
    points = np.random.RandomState(8).normal(size=(10, 3))
    supplied = distances_of(points)
    constraints = [
        DistanceConstraint(i, j, float(supplied[i, j]), float(supplied[i, j]))
        for i in range(10)
        for j in range(i + 1, min(i + 4, 10))
    ]
    distance = float(supplied[0, 4])
    constraints.append(DistanceConstraint(0, 4, distance - 0.1, distance + 0.1))
    constraints.extend(
        [DistanceConstraint(0, 9, 1.0, 1.0), DistanceConstraint(0, 9, 100.0, 100.0)]
    )
    constraint_set = ConstraintSet(10, constraints)
    solver = SBBUSolver(
        constraint_set,
        SBBUConfig(distance_tolerance=1e-6, enable_soft_pruning=False, verbose=False),
    )
    solver.solve()
    actual = distances_of(solver.solution_coordinates)
    for constraint in constraint_set.get_hard_constraints():
        assert constraint.violation(float(actual[constraint.i, constraint.j])) <= 1e-6
    assert len(constraint_set.get_soft_ambiguous_constraints()) == 1


@pytest.mark.parametrize("distance", [0.0, 1.0, 1.5, 2.0, 3.0, np.nan, np.inf])
def test_bulk_constraint_errors_match_scalar_semantics(distance):
    constraints = [
        DistanceConstraint(0, 1, 1.5, 1.5),
        DistanceConstraint(0, 2, 1.0, 2.0),
    ]
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0], [distance, 0.0, 0.0]]
    )
    expected = [c.violation(distance) for c in constraints]
    np.testing.assert_array_equal(
        ConstraintSet(3, constraints).constraint_errors(coordinates), expected
    )


def test_complete_projection_preserves_canonical_constraints():
    points = np.random.RandomState(8).normal(size=(5, 3))
    supplied = distances_of(points)
    canonical = MatrixAdapter.from_distance_matrix(supplied)
    projected = canonical.get_complete_distance_matrix()
    assert projected is not None
    np.testing.assert_array_equal(projected, supplied)
    projected[0, 1] = 100.0
    np.testing.assert_array_equal(canonical.get_complete_distance_matrix(), supplied)

    constraints = canonical.get_hard_constraints()
    bounded = [
        DistanceConstraint(c.i, c.j, c.lower_bound - 0.01, c.lower_bound + 0.01)
        if (c.i, c.j) == (0, 4)
        else c
        for c in constraints
    ]
    assert ConstraintSet(5, bounded).get_complete_distance_matrix() is None
    ambiguous = constraints + [DistanceConstraint(0, 4, 100.0, 100.0)]
    assert ConstraintSet(5, ambiguous).get_complete_distance_matrix() is None


def test_rejected_clique_candidate_leaves_original_search_available(monkeypatch):
    points = np.random.RandomState(8).normal(size=(10, 3))
    supplied = distances_of(points)
    monkeypatch.setattr(
        solver_module, "clique_realizations", lambda _: iter([np.full((10, 3), np.nan)])
    )
    coordinates, _ = solve_from_distance_matrix(supplied, verbose=False)
    assert np.max(np.abs(distances_of(coordinates) - supplied)) <= 1e-7


def test_clique_timeout_clears_previous_solution(monkeypatch):
    points = np.random.RandomState(8).normal(size=(10, 3))
    solver = SBBUSolver(
        MatrixAdapter.from_distance_matrix(distances_of(points)),
        SBBUConfig(max_time=1.0, verbose=False),
    )
    solver.solve()
    clock = [0.0]

    def expired_candidate(_: NDArray[np.float64]) -> Iterator[NDArray[np.float64]]:
        clock[0] = 2.0
        yield points

    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    monkeypatch.setattr(solver_module, "clique_realizations", expired_candidate)
    with pytest.raises(SBBUTimeoutError):
        solver.solve()
    with pytest.raises(RuntimeError, match="No solved coordinates"):
        _ = solver.solution_coordinates


@pytest.mark.parametrize("closing_node", [25, 29])
@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("perturbation", [0.0, 1e-8])
def test_sparse_long_span_stops_only_after_global_feasibility(
    closing_node, reflect, perturbation
):
    points = np.random.RandomState(91).normal(size=(30, 3))
    supplied = distances_of(points)
    rows, columns = np.indices(supplied.shape)
    supplied[np.abs(rows - columns) > 3] = np.nan
    witness, _ = solve_from_distance_matrix(supplied, verbose=False)
    if reflect:
        normal = np.cross(witness[2] - witness[1], witness[3] - witness[1])
        normal /= np.linalg.norm(normal)
        offsets = (witness[4:] - witness[1]) @ normal
        witness[4:] -= 2.0 * offsets[:, None] * normal
    supplied[0, closing_node] = supplied[closing_node, 0] = (
        np.linalg.norm(witness[0] - witness[closing_node]) + perturbation
    )
    solver = SBBUSolver(
        MatrixAdapter.from_distance_matrix(supplied),
        SBBUConfig(max_iterations=1024, max_time=1.0, verbose=False),
    )
    stats = solver.solve()
    coordinates = solver.solution_coordinates
    known = np.isfinite(supplied)
    assert np.max(np.abs(distances_of(coordinates)[known] - supplied[known])) <= 1e-7
    assert _sbbu_stage(stats).iterations <= 3


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, SBBUTimeoutError])
def test_failed_forward_completion_restores_state_or_propagates_timeout(
    monkeypatch, failure
):
    points = np.random.RandomState(91).normal(size=(30, 3))
    supplied = distances_of(points)
    rows, columns = np.indices(supplied.shape)
    supplied[np.abs(rows - columns) > 3] = np.nan
    witness, _ = solve_from_distance_matrix(supplied, verbose=False)
    supplied[0, 25] = supplied[25, 0] = np.linalg.norm(witness[0] - witness[25])
    solver = SBBUSolver(
        MatrixAdapter.from_distance_matrix(supplied),
        SBBUConfig(max_iterations=1024, max_time=1.0, verbose=False),
    )
    solver.solve()
    extend = solver._extend_to_node
    injected = False

    def fail_after_extension(target: int) -> None:
        nonlocal injected
        extend(target)
        if target == 29 and not injected:
            injected = True
            solver._state_required().coordinates[:] = np.nan
            raise failure("injected trial-completion failure")

    monkeypatch.setattr(solver, "_extend_to_node", fail_after_extension)
    if failure is SBBUTimeoutError:
        with pytest.raises(SBBUTimeoutError, match="trial-completion"):
            solver.solve()
        with pytest.raises(RuntimeError, match="No solved coordinates"):
            _ = solver.solution_coordinates
    else:
        solver.solve()
        coordinates = solver.solution_coordinates
        known = np.isfinite(supplied)
        assert np.isfinite(coordinates).all()
        assert (
            np.max(np.abs(distances_of(coordinates)[known] - supplied[known])) <= 1e-7
        )
    assert injected


def test_rank_truncation_cannot_collapse_the_positive_x_seed():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    supplied = distances_of(points)
    coordinates, _ = solve_from_distance_matrix(supplied, verbose=False, max_time=1.0)
    assert coordinates[1, 0] > 0.0
    np.testing.assert_array_equal(coordinates[0], np.zeros(3))
    assert np.max(np.abs(distances_of(coordinates) - supplied)) <= 1e-7


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matrix_symmetry_retains_scalar_promotion(dtype):
    for value in np.geomspace(1e-6, 1e6, 100):
        for factor in (
            1.0 - 1.0001e-5,
            1.0 - 0.9999e-5,
            1.0 + 0.9999e-5,
            1.0 + 1.0001e-5,
        ):
            supplied = np.array([[0.0, value], [value * factor, 0.0]], dtype=dtype)
            expected = np.isclose(supplied[0, 1], supplied[1, 0], atol=1e-12)
            assert (
                MatrixAdapter.validate_distance_matrix(supplied, tolerance=1e-12)
                == expected
            )
    for steps in range(84, 168):
        lower = dtype((steps * 100000 - 1) * 2.0**-23)
        upper = dtype(lower + steps * 2.0**-23)
        supplied = np.array([[0.0, upper], [lower, 0.0]], dtype=dtype)
        expected = np.isclose(supplied[0, 1], supplied[1, 0], atol=1e-12)
        assert (
            MatrixAdapter.validate_distance_matrix(supplied, tolerance=1e-12)
            == expected
        )
