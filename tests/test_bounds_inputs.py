"""Closed-bound input contracts, independent of solver applicability."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import sbbu
from sbbu.adapters import MatrixAdapter, NMRAdapter
from sbbu.constraints import ConstraintSet, DistanceConstraint


def triangle_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.ones((3, 3))
    upper = lower * 2
    np.fill_diagonal(lower, 0)
    np.fill_diagonal(upper, 0)
    return lower, upper


def test_closed_intervals_and_exact_equality() -> None:
    exact = DistanceConstraint(0, 1, 1.0, 1.0)
    narrow = DistanceConstraint(0, 1, 1.0, np.nextafter(1.0, 2.0))
    assert exact.is_exact
    assert not narrow.is_exact
    assert exact.violation(1.5) == 0.5
    assert narrow.upper_bound > narrow.lower_bound


def test_issue3_interval_graph_constructs_but_solver_refuses() -> None:
    constraints = ConstraintSet(
        3,
        [DistanceConstraint(i, j, 1.0, 2.0) for i in range(3) for j in range(i + 1, 3)],
    )
    assert len(constraints.get_hard_constraints()) == 3
    with pytest.raises(ValueError, match="must be exact"):
        sbbu.SBBUSolver(constraints)


def test_narrow_sequential_interval_is_not_reinterpreted_as_exact() -> None:
    constraints = ConstraintSet(
        3,
        [
            DistanceConstraint(0, 1, 1.0, np.nextafter(1.0, 2.0)),
            DistanceConstraint(0, 2, 1.0, 1.0),
            DistanceConstraint(1, 2, 1.0, 1.0),
        ],
    )
    with pytest.raises(ValueError, match="must be exact"):
        sbbu.SBBUSolver(constraints, sbbu.SBBUConfig(distance_tolerance=0.1))


def test_bounds_matrix_preserves_inputs_and_intervals() -> None:
    lower, upper = triangle_bounds()
    before_lower, before_upper = lower.copy(), upper.copy()
    constraints = MatrixAdapter.from_bounds_matrices(lower, upper)
    np.testing.assert_array_equal(lower, before_lower)
    np.testing.assert_array_equal(upper, before_upper)
    assert constraints.num_nodes == 3
    assert constraints.get_hard_constraints() == [
        DistanceConstraint(i, j, lower[i, j], upper[i, j])
        for i in range(3)
        for j in range(i + 1, 3)
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_bounds_matrix_exact_matches_distance_matrix(dtype: str) -> None:
    lower, _ = triangle_bounds()
    distances = lower.astype(dtype)
    expected = MatrixAdapter.from_distance_matrix(distances)
    actual = MatrixAdapter.from_bounds_matrices(distances, distances)
    assert actual.get_hard_constraints() == expected.get_hard_constraints()
    coordinates, stats = sbbu.solve_from_bounds_matrices(
        distances, distances, verbose=False
    )
    errors = actual.constraint_errors(coordinates)
    assert errors.max() < 1e-7
    assert stats.largest_distance_error < 1e-7


def test_nmr_and_edge_list_preserve_same_bounds(tmp_path: Path) -> None:
    path = tmp_path / "interval.nmr"
    path.write_text(
        "1 2 1.0 2.0 H H ALA GLY\n1 3 1.1 2.1 H H ALA GLY\n2 3 1.2 2.2 H H ALA GLY\n"
    )
    nmr = NMRAdapter.from_nmr_file(path)
    edges = [(1, 0, 1.0, 2.0), (0, 2, 1.1, 2.1), (1, 2, 1.2, 2.2)]
    edge_constraints = MatrixAdapter.from_edge_list(3, edges)
    assert nmr.get_hard_constraints() == edge_constraints.get_hard_constraints()
    assert len(nmr.constraints) == 3
    with pytest.raises(ValueError, match="must be exact"):
        sbbu.SBBUSolver(nmr)


def test_nmr_preserves_narrow_interval(tmp_path: Path) -> None:
    path = tmp_path / "narrow.nmr"
    upper = float(np.nextafter(1.0, 2.0))
    path.write_text(f"1 2 1.0 {upper!r} H H ALA GLY\n")
    constraint = NMRAdapter.from_nmr_file(path).get_constraint(0, 1)
    assert constraint is not None
    assert constraint.upper_bound == upper
    assert not constraint.is_exact


def test_intersection_does_not_round_a_narrow_interval() -> None:
    upper = float(np.nextafter(1.0, 2.0))
    constraints = ConstraintSet(
        5, [DistanceConstraint(0, 4, 0.5, upper), DistanceConstraint(0, 4, 1.0, 2.0)]
    )
    assert constraints.get_constraint(0, 4) == DistanceConstraint(0, 4, 1.0, upper)
    assert constraints.get_complete_distance_matrix() is None


def test_exact_and_containing_interval_intersect_without_loss() -> None:
    constraints = ConstraintSet(
        2, [DistanceConstraint(0, 1, 1.0, 1.0), DistanceConstraint(0, 1, 0.9, 1.1)]
    )
    assert constraints.get_constraint(0, 1) == DistanceConstraint(0, 1, 1.0, 1.0)


def test_bounds_matrix_paired_missing_entries() -> None:
    lower, upper = triangle_bounds()
    lower[0, 1] = lower[1, 0] = upper[0, 1] = upper[1, 0] = np.nan
    constraints = MatrixAdapter.from_bounds_matrices(lower, upper)
    assert constraints.get_constraint(0, 1) is None
    with pytest.raises(ValueError, match="Missing required sequential"):
        sbbu.SBBUSolver(constraints)
    with pytest.raises(ValueError):
        MatrixAdapter.from_bounds_matrices(lower, upper, ignore_nan=False)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, -1.0, 0.0])
def test_bounds_matrix_invalid_endpoints(bad: float) -> None:
    lower, upper = triangle_bounds()
    lower[0, 1] = lower[1, 0] = bad
    with pytest.raises(ValueError):
        MatrixAdapter.from_bounds_matrices(lower, upper)


@pytest.mark.parametrize(
    "failure",
    [
        "shape",
        "nonsquare",
        "asymmetric",
        "diagonal",
        "half_nan",
        "reverse",
        "infinite",
        "complex",
    ],
)
def test_bounds_matrix_rejects_malformed_pairs(failure: str) -> None:
    lower, upper = triangle_bounds()
    if failure == "shape":
        upper = upper[:2, :2]
    elif failure == "nonsquare":
        lower = lower[:, :2]
    elif failure == "asymmetric":
        lower[0, 1] = np.nextafter(1.0, 2.0)
    elif failure == "diagonal":
        upper[0, 0] = 1.0
    elif failure == "half_nan":
        lower[0, 1] = lower[1, 0] = np.nan
    elif failure == "reverse":
        lower[0, 1] = lower[1, 0] = np.nextafter(2.0, 3.0)
    elif failure == "infinite":
        upper[0, 1] = upper[1, 0] = np.inf
    else:
        lower = lower.astype(complex)
    with pytest.raises(ValueError):
        MatrixAdapter.from_bounds_matrices(lower, upper)


def test_mixed_bounds_solve_and_metrics_match_original_observations() -> None:
    points = np.random.RandomState(8).normal(size=(5, 3))
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    lower, upper = distances.copy(), distances.copy()
    lower[0, 4] = lower[4, 0] = distances[0, 4] - 0.1
    upper[0, 4] = upper[4, 0] = distances[0, 4] + 0.1
    coordinates, stats = sbbu.solve_from_bounds_matrices(lower, upper, verbose=False)
    realized = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    errors = np.maximum(np.maximum(lower - realized, realized - upper), 0)
    assert errors.max() <= 1e-7
    assert stats.largest_distance_error == pytest.approx(errors.max())
    constraints = MatrixAdapter.from_bounds_matrices(lower, upper)
    edges = [(c.i, c.j, c.lower_bound, c.upper_bound) for c in constraints.constraints]
    edge_coordinates, _ = sbbu.solve_from_edge_list(5, edges, verbose=False)
    np.testing.assert_allclose(edge_coordinates, coordinates, atol=1e-12)


def test_bounds_matrix_tiny_interval_disables_complete_exact_initialization() -> None:
    lower = np.ones((4, 4))
    np.fill_diagonal(lower, 0.0)
    upper = lower.copy()
    upper[0, 1] = upper[1, 0] = np.nextafter(1.0, 2.0)
    constraints = MatrixAdapter.from_bounds_matrices(lower, upper)
    assert constraints.get_complete_distance_matrix() is None
    with pytest.raises(ValueError, match="must be exact"):
        sbbu.solve_from_bounds_matrices(lower, upper, verbose=False)


def test_disjoint_duplicates_are_not_joined_by_tolerance() -> None:
    next_value = float(np.nextafter(1.0, 2.0))
    observations = [
        DistanceConstraint(0, 4, 0.9, 1.0),
        DistanceConstraint(0, 4, next_value, 1.1),
    ]
    constraints = ConstraintSet(5, observations)
    assert constraints.get_constraint(0, 4) is None
    assert constraints.get_soft_ambiguous_constraints() == [((0, 4), observations)]


@pytest.mark.parametrize("bounds", [(None, 1.0), (1.0, None), (None, None)])
def test_one_sided_bounds_are_not_canonical(
    bounds: tuple[float | None, float | None],
) -> None:
    with pytest.raises(TypeError):
        DistanceConstraint(0, 1, *bounds)  # type: ignore[arg-type]


def test_edge_list_preserves_isolated_nodes() -> None:
    constraints = MatrixAdapter.from_edge_list(4, [(1, 0, 1.0, 2.0)])
    assert constraints.num_nodes == 4
    assert constraints.get_hard_constraints() == [DistanceConstraint(0, 1, 1.0, 2.0)]


def test_nmr_missing_predecessors_are_valid_input(tmp_path: Path) -> None:
    path = tmp_path / "missing.nmr"
    path.write_text("1 2 1.0 1.0 H H ALA GLY\n2 3 1.0 1.0 H H ALA GLY\n")
    constraints = NMRAdapter.from_nmr_file(path)
    assert constraints.num_nodes == 3
    assert constraints.get_constraint(0, 2) is None
    with pytest.raises(ValueError, match="Missing required sequential"):
        sbbu.SBBUSolver(constraints)
