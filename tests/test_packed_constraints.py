"""Exercise canonical packed bounds through construction and evaluation."""

import math
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from sbbu.adapters import MatrixAdapter
from sbbu.constraints import ConstraintSet, DistanceConstraint


def test_bulk_and_observation_construction_agree():
    rows = np.array([0, 0, 1], dtype=np.intp)
    columns = np.array([1, 2, 3], dtype=np.intp)
    lower = np.array([1.0, 2.0, 3.0])
    upper = np.array([1.0, 2.5, 4.0])
    bulk = ConstraintSet.from_bounds_arrays(5, rows, columns, lower, upper)
    scalar = ConstraintSet(
        5,
        [
            DistanceConstraint(int(i), int(j), float(lo), float(hi))
            for i, j, lo, hi in zip(rows, columns, lower, upper, strict=True)
        ],
    )
    assert bulk.get_hard_constraints() == scalar.get_hard_constraints()
    assert bulk.get_constraint(3, 1) == DistanceConstraint(1, 3, 3.0, 4.0)
    assert bulk.get_constraint(3, 4) is None
    assert not bulk.is_complete
    coordinates = np.random.RandomState(6).normal(size=(5, 3))
    np.testing.assert_array_equal(
        bulk.constraint_errors(coordinates), scalar.constraint_errors(coordinates)
    )
    lower[:] = 99.0
    rows[:] = 4
    assert bulk.get_constraint(0, 1) == DistanceConstraint(0, 1, 1.0, 1.0)
    for column in (bulk.rows, bulk.columns, bulk.lower_bounds, bulk.upper_bounds):
        with pytest.raises(ValueError):
            column[0] = 0
        with pytest.raises(ValueError):
            column.flags.writeable = True


def test_original_observations_and_disjoint_alternatives_are_snapshots():
    observations = [
        DistanceConstraint(0, 1, 1.0, 3.0),
        DistanceConstraint(0, 1, 2.0, 4.0),
        DistanceConstraint(0, 2, 1.0, 1.1),
        DistanceConstraint(0, 2, 2.0, 2.1),
    ]
    problem = ConstraintSet(4, observations)
    observations.clear()
    assert len(problem.constraints) == 4
    assert problem.get_hard_constraints() == [DistanceConstraint(0, 1, 2.0, 3.0)]
    alternatives = problem.get_soft_ambiguous_constraints()
    alternatives[0][1].clear()
    assert len(problem.get_soft_ambiguous_constraints()[0][1]) == 2
    problem.constraints.clear()
    assert len(problem.constraints) == 4


@pytest.mark.parametrize(
    "rows,columns",
    [
        ([0, 0], [1, 1]),
        ([1, 0], [2, 1]),
        ([-1, 0], [1, 2]),
        ([0, 1], [0, 2]),
        ([0, 1], [1, 4]),
    ],
)
def test_bulk_rejects_noncanonical_pairs(rows, columns):
    with pytest.raises(ValueError):
        ConstraintSet.from_bounds_arrays(
            4, np.array(rows), np.array(columns), np.ones(2), np.ones(2)
        )


@pytest.mark.parametrize(
    "lower,upper", [(0.0, 1.0), (2.0, 1.0), (np.nan, 1.0), (1.0, np.inf)]
)
def test_bulk_rejects_invalid_bounds(lower, upper):
    with pytest.raises(ValueError):
        ConstraintSet.from_bounds_arrays(
            3, np.array([0]), np.array([1]), np.array([lower]), np.array([upper])
        )


def test_bulk_rejects_wrong_columns_and_preserves_narrow_intervals():
    with pytest.raises(TypeError):
        ConstraintSet.from_bounds_arrays(
            3, np.array([0.0]), np.array([1]), np.ones(1), np.ones(1)
        )
    with pytest.raises(ValueError):
        ConstraintSet.from_bounds_arrays(
            3, np.array([0]), np.array([1]), np.ones(2), np.ones(1)
        )
    upper = np.nextafter(1.0, 2.0)
    problem = ConstraintSet.from_bounds_arrays(
        3, np.array([0]), np.array([1]), np.ones(1), np.array([upper])
    )
    constraint = problem.get_constraint(0, 1)
    assert constraint is not None and not constraint.is_exact
    assert constraint.upper_bound == upper


@pytest.mark.parametrize("kind", ["missing-node", "nan-isolated", "complex"])
def test_invalid_coordinates_do_not_pass_with_unconstrained_nodes(kind):
    problem = ConstraintSet(4, [DistanceConstraint(0, 1, 1.0, 1.0)])
    coordinates = np.zeros((4, 3))
    coordinates[1, 0] = 1.0
    if kind == "missing-node":
        coordinates = coordinates[:3]
    elif kind == "nan-isolated":
        coordinates[-1, 0] = np.nan
    else:
        coordinates = coordinates.astype(complex) * 1j
    assert math.isinf(problem.maximum_violation(cast(NDArray[np.float64], coordinates)))


def test_empty_graph_and_integer_coordinates_have_truthful_metrics():
    empty = ConstraintSet(5, [])
    assert empty.maximum_violation(np.zeros((5, 3))) == 0.0
    assert math.isinf(empty.maximum_violation(np.zeros((4, 3))))
    problem = ConstraintSet(2, [DistanceConstraint(0, 1, 1.0, 2.0)])
    coordinates = np.array([[0, 0, 0], [10**12, 0, 0]], dtype=np.int64)
    assert (
        problem.maximum_violation(cast(NDArray[np.float64], coordinates))
        == 10**12 - 2.0
    )


def test_matrix_ingestion_does_not_construct_per_pair_objects(monkeypatch):
    def unexpected_scalar(*args, **kwargs):
        raise AssertionError("Matrix ingestion materialized a scalar constraint")

    monkeypatch.setattr("sbbu.constraints.DistanceConstraint", unexpected_scalar)
    points = np.random.RandomState(1729).normal(size=(100, 3))
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    for problem in (
        MatrixAdapter.from_distance_matrix(distances),
        MatrixAdapter.from_bounds_matrices(distances * 0.99, distances * 1.01),
        MatrixAdapter.from_adjacency_matrix(np.ones((100, 100)), distances),
    ):
        assert problem.is_complete
        assert problem.maximum_violation(points) == 0.0


def test_matrix_projection_does_not_invent_missing_observations():
    partial = ConstraintSet(4, [DistanceConstraint(0, 1, 1.0, 2.0)])
    with pytest.raises(ValueError, match="all hard pairs"):
        partial.bounds_matrices()
    assert partial.get_constraint(0, 2) is None


@pytest.mark.parametrize("lower,upper", [(2**53, 2**53 + 1), (2**53 + 1, 2**53 + 2)])
def test_lossy_numeric_conversion_never_changes_original_bounds(lower, upper):
    with pytest.raises(ValueError, match="losslessly"):
        DistanceConstraint(0, 1, lower, upper)
    lower_matrix = np.full((4, 4), lower, dtype=np.int64)
    upper_matrix = np.full((4, 4), upper, dtype=np.int64)
    np.fill_diagonal(lower_matrix, 0)
    np.fill_diagonal(upper_matrix, 0)
    with pytest.raises(ValueError, match="losslessly"):
        MatrixAdapter.from_bounds_matrices(
            cast(NDArray[np.float64], lower_matrix),
            cast(NDArray[np.float64], upper_matrix),
        )


def test_representable_large_integer_width_stays_positive():
    lower, upper = 2**53, 2**53 + 2
    constraint = DistanceConstraint(0, 1, lower, upper)
    assert not constraint.is_exact
    assert constraint.lower_bound == lower and constraint.upper_bound == upper


def test_extended_precision_interval_is_not_rounded_to_a_point():
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("No wider floating-point dtype on this platform")
    lower = np.ones((4, 4), dtype=np.longdouble)
    upper = np.nextafter(lower, lower * 2)
    np.fill_diagonal(lower, 0)
    np.fill_diagonal(upper, 0)
    with pytest.raises(ValueError, match="losslessly"):
        MatrixAdapter.from_bounds_matrices(
            cast(NDArray[np.float64], lower), cast(NDArray[np.float64], upper)
        )
