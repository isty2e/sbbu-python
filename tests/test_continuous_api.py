"""Public reconstruction dispatch, ingestion and publication boundaries."""

import math

import numpy as np
import pytest

import sbbu
import sbbu.solving as operation
from sbbu.adapters import MatrixAdapter
from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.solving.run import (
    MMStage,
    SBBUStage,
    SolveError,
    SolveTimeoutError,
    UnsupportedProblemError,
)


@pytest.fixture
def exact_entrypoints(tmp_path, simple_tetrahedron):
    points, problem = simple_tetrahedron
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    observations = problem.get_hard_constraints()
    edges = [(c.i, c.j, c.lower_bound, c.upper_bound) for c in observations]
    nmr = tmp_path / "exact.nmr"
    nmr.write_text(
        "".join(
            f"{c.i + 1} {c.j + 1} {c.lower_bound:.17g} {c.upper_bound:.17g} C C ALA ALA\n"
            for c in observations
        )
    )
    return [
        lambda method: sbbu.solve_from_distance_matrix(
            distances, verbose=False, method=method
        ),
        lambda method: sbbu.solve_from_bounds_matrices(
            distances, distances, verbose=False, method=method
        ),
        lambda method: sbbu.solve_from_edge_list(
            4, edges, verbose=False, method=method
        ),
        lambda method: sbbu.solve_from_nmr(nmr, verbose=False, method=method),
    ]


@pytest.mark.parametrize("method", [None, "mm", "trf"])
def test_all_entrypoints_prefer_applicable_sbbu(exact_entrypoints, monkeypatch, method):
    def forbidden(*args):
        raise AssertionError("Continuous backend entered for applicable SBBU")

    monkeypatch.setattr(operation, "solve_continuous", forbidden)
    for entrypoint in exact_entrypoints:
        coordinates, stats = entrypoint(method)
        assert coordinates.shape == (4, 3)
        assert isinstance(stats, sbbu.SolveStats)
        assert len(stats.stages) == 1 and isinstance(stats.stages[0], SBBUStage)


@pytest.mark.parametrize(
    "method,error",
    [
        (True, TypeError),
        (0, TypeError),
        ([], TypeError),
        ("auto", ValueError),
        ("MM", ValueError),
    ],
)
def test_all_entrypoints_validate_method_even_when_sbbu_would_win(
    exact_entrypoints, method, error
):
    for entrypoint in exact_entrypoints:
        with pytest.raises(error, match="method"):
            entrypoint(method)


@pytest.mark.parametrize("failure", [SolveError, SolveTimeoutError])
def test_sbbu_runtime_failure_is_not_inapplicability(
    exact_entrypoints, monkeypatch, failure
):
    def failed(*args):
        raise failure("SBBU execution failed")

    def forbidden(*args):
        raise AssertionError("SBBU failure triggered a continuous retry")

    monkeypatch.setattr(operation.SBBUSolver, "_solve", failed)
    monkeypatch.setattr(operation, "solve_continuous", forbidden)
    for entrypoint in exact_entrypoints:
        with pytest.raises(failure, match="SBBU execution failed"):
            entrypoint("trf")


def test_missing_edges_are_valid_but_continuous_support_is_temporary():
    problem = ConstraintSet(5, [DistanceConstraint(0, 1, 1.0, 2.0)])
    assert problem.get_constraint(0, 2) is None
    with pytest.raises(UnsupportedProblemError, match="temporarily.*missing-edge"):
        operation.solve(problem, 1e-7, 1.0, False, "trf")


def test_positive_width_complete_inputs_across_bounds_edges_and_nmr(tmp_path):
    rng = np.random.RandomState(91)
    witness = rng.normal(size=(8, 3))
    distances = np.linalg.norm(witness[:, None] - witness[None, :], axis=-1)
    rows, columns = np.triu_indices(8, k=1)
    alpha = rng.uniform(0.1, 0.9, size=len(rows))
    lower, upper = distances.copy(), distances.copy()
    lower[rows, columns] *= 1 - 0.02 * alpha
    upper[rows, columns] *= 1 + 0.02 * (1 - alpha)
    lower[columns, rows] = lower[rows, columns]
    upper[columns, rows] = upper[rows, columns]
    edges = [
        (int(i), int(j), float(lower[i, j]), float(upper[i, j]))
        for i, j in zip(rows, columns, strict=True)
    ]
    nmr = tmp_path / "intervals.nmr"
    nmr.write_text(
        "".join(
            f"{i + 1} {j + 1} {lo:.17g} {hi:.17g} C C ALA ALA\n"
            for i, j, lo, hi in edges
        )
    )
    results = [
        sbbu.solve_from_bounds_matrices(lower, upper, max_time=1.0, verbose=False),
        sbbu.solve_from_edge_list(8, edges, max_time=1.0, verbose=False, method="mm"),
        sbbu.solve_from_nmr(nmr, max_time=1.0, verbose=False, method="trf"),
    ]
    for coordinates, stats in results:
        error = max(
            max(
                lo - math.dist(coordinates[i], coordinates[j]),
                math.dist(coordinates[i], coordinates[j]) - hi,
                0.0,
            )
            for i, j, lo, hi in edges
        )
        assert error <= 1e-7
        assert stats.largest_distance_error == pytest.approx(error, abs=1e-14)
        assert all(stage.algorithm != "sbbu" for stage in stats.stages)
        assert stats.solve_time < 1.0


def test_public_finalizer_does_not_trust_backend_feasible_status(monkeypatch):
    lower = np.ones((4, 4)) - np.eye(4)
    upper = lower * 2
    monkeypatch.setattr(
        operation,
        "solve_continuous",
        lambda *args: (np.zeros((4, 3)), (MMStage(0.0, 1, 0, "feasible"),)),
    )
    with pytest.raises(SolveError, match="Original hard constraints"):
        sbbu.solve_from_bounds_matrices(lower, upper, verbose=False)


def test_publication_after_backend_return_cannot_refresh_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("sbbu.solving.run.get_wall_time", lambda: clock[0])
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)

    def late(*args):
        clock[0] = 2.0
        return points, (MMStage(2.0, 1, 0, "feasible"),)

    monkeypatch.setattr(operation, "solve_continuous", late)
    with pytest.raises(SolveTimeoutError) as caught:
        sbbu.solve_from_bounds_matrices(
            distances * 0.9, distances * 1.1, max_time=1.0, verbose=False
        )
    assert len(caught.value.stages) == 1


def test_mixed_exact_continuous_admission_is_not_model_validation():
    lower = np.ones((4, 4)) - np.eye(4)
    upper = lower * 2
    upper[0, 3] = upper[3, 0] = lower[0, 3]
    problem = MatrixAdapter.from_bounds_matrices(lower, upper)
    assert problem.is_complete
    with pytest.raises(UnsupportedProblemError, match="positive-width"):
        operation.solve(problem, 1e-7, 1.0, False, None)
