"""
Tests for the high-level SBBU API.
"""

import numpy as np
import pytest

import sbbu
from sbbu.core.solver import SBBUConfig as CoreSBBUConfig
from sbbu.core.solver import SBBUSolver as CoreSBBUSolver


@pytest.mark.unit
def test_top_level_solver_exports():
    """Top-level package should export core solver entry types."""
    assert sbbu.SBBUConfig is CoreSBBUConfig
    assert sbbu.SBBUSolver is CoreSBBUSolver


@pytest.mark.unit
def test_solve_from_distance_matrix(distance_matrix_4x4):
    """Test solving from distance matrix."""
    coords, stats = sbbu.solve_from_distance_matrix(distance_matrix_4x4, verbose=False)

    assert coords.shape == (4, 3)
    assert stats.solve_time > 0
    assert stats.mean_distance_error < 1e-6

    # Check distance reconstruction
    for i in range(4):
        for j in range(i + 1, 4):
            expected_dist = distance_matrix_4x4[i, j]
            actual_dist = np.linalg.norm(coords[i] - coords[j])
            assert abs(expected_dist - actual_dist) < 1e-6


@pytest.mark.unit
def test_solve_from_edge_list(edge_list_simple):
    """Test solving from edge list."""
    num_nodes, edges = edge_list_simple

    coords, stats = sbbu.solve_from_edge_list(num_nodes, edges, verbose=False)

    assert coords.shape == (num_nodes, 3)
    assert stats.solve_time > 0
    assert stats.mean_distance_error < 1e-6

    # Check edge distance reconstruction
    for i, j, expected_dist, _upper in edges:
        actual_dist = np.linalg.norm(coords[i] - coords[j])
        assert abs(expected_dist - actual_dist) < 1e-6


@pytest.mark.unit
def test_solve_from_sparse_matrix(sparse_distance_matrix):
    """Test solving from sparse distance matrix."""
    coords, stats = sbbu.solve_from_distance_matrix(
        sparse_distance_matrix, verbose=False, ignore_nan=True
    )

    assert coords.shape == (5, 3)
    assert stats.solve_time > 0

    # Check that non-NaN distances are preserved
    n = sparse_distance_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            expected_dist = sparse_distance_matrix[i, j]
            if not np.isnan(expected_dist):
                actual_dist = np.linalg.norm(coords[i] - coords[j])
                assert abs(expected_dist - actual_dist) < 1e-6


@pytest.mark.integration
def test_solve_from_nmr(nmr_1n6t):
    """Test solving from NMR file."""
    if not nmr_1n6t.exists():
        pytest.skip("Test NMR file not found")

    coords, stats = sbbu.solve_from_nmr(nmr_1n6t, verbose=False)

    assert coords.shape == (30, 3)
    assert stats.solve_time > 0
    assert stats.mean_distance_error < 1e-10
    assert stats.largest_distance_error < 1e-8


@pytest.mark.integration
def test_nmr_accuracy_vs_reference(nmr_1n6t, sol_1n6t):
    """Test NMR solving accuracy against C++ reference."""
    if not (nmr_1n6t.exists() and sol_1n6t.exists()):
        pytest.skip("Test files not found")

    coords, _ = sbbu.solve_from_nmr(nmr_1n6t, verbose=False)
    ref_coords = np.loadtxt(sol_1n6t)

    # Check coordinate differences
    max_diff = 0
    for i in range(coords.shape[0]):
        diff = np.linalg.norm(coords[i] - ref_coords[i])
        max_diff = max(max_diff, diff)

    # Should match C++ reference within numerical precision
    assert max_diff < 1e-6


@pytest.mark.unit
def test_create_test_constraints():
    """Test synthetic constraint generation."""
    constraints = sbbu.create_test_constraints(
        num_nodes=6, connectivity=0.5, noise_level=0.01, random_seed=42
    )

    assert constraints.num_nodes == 6
    assert len(constraints.constraints) > 0

    # Should have sequential constraints
    sequential = constraints.get_sequential_constraints()
    assert len(sequential) >= 5  # At least 0-1, 1-2, 2-3, 3-4, 4-5

    # Should have some long-range constraints
    long_range = constraints.get_long_range_constraints()
    assert len(long_range) >= 0


@pytest.mark.unit
@pytest.mark.parametrize("connectivity", [-0.1, 1.1])
def test_create_test_constraints_rejects_connectivity_out_of_range(connectivity):
    """Synthetic generator should enforce connectivity in [0, 1]."""
    with pytest.raises(ValueError, match="connectivity"):
        sbbu.create_test_constraints(
            num_nodes=6,
            connectivity=connectivity,
            noise_level=0.0,
            random_seed=42,
        )


@pytest.mark.unit
def test_create_test_constraints_rejects_negative_noise_level():
    """Synthetic generator should reject negative noise scales."""
    with pytest.raises(ValueError, match="noise_level"):
        sbbu.create_test_constraints(
            num_nodes=6,
            connectivity=0.5,
            noise_level=-0.01,
            random_seed=42,
        )


@pytest.mark.unit
def test_create_test_constraints_rejects_non_positive_num_nodes():
    """Synthetic generator should reject non-positive node counts."""
    with pytest.raises(ValueError, match="num_nodes"):
        sbbu.create_test_constraints(
            num_nodes=0,
            connectivity=0.5,
            noise_level=0.0,
            random_seed=42,
        )


@pytest.mark.unit
def test_api_error_handling():
    """Test API error handling."""
    # Empty distance matrix
    with pytest.raises(ValueError):
        sbbu.solve_from_distance_matrix(np.array([[]]))

    # Negative distances
    bad_matrix = np.array([[0, -1], [-1, 0]])
    with pytest.raises(ValueError):
        sbbu.solve_from_distance_matrix(bad_matrix)

    # Edge list with negative distances
    bad_edges = [(0, 1, -1.0, -1.0)]
    with pytest.raises(ValueError):
        sbbu.solve_from_edge_list(2, bad_edges)


@pytest.mark.unit
def test_api_configuration():
    """Test API with different configurations."""
    matrix = np.array([[0.0, 1.0, 1.4], [1.0, 0.0, 1.0], [1.4, 1.0, 0.0]])

    # Test with custom tolerance
    coords, stats = sbbu.solve_from_distance_matrix(
        matrix, distance_tolerance=1e-5, max_time=30.0, verbose=False
    )

    assert coords.shape == (3, 3)
    assert stats.solve_time > 0


@pytest.mark.unit
def test_solve_from_distance_matrix_rejects_infinite_entries_early():
    """API should fail fast on infinite off-diagonal distances."""
    matrix = np.array(
        [
            [0.0, np.inf, 1.4],
            [np.inf, 0.0, 1.0],
            [1.4, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        sbbu.solve_from_distance_matrix(matrix, verbose=False)


@pytest.mark.unit
def test_solve_from_distance_matrix_rejects_zero_off_diagonal_early():
    """API should fail fast on zero off-diagonal distances."""
    matrix = np.array(
        [
            [0.0, 0.0, 1.4],
            [0.0, 0.0, 1.0],
            [1.4, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="positive"):
        sbbu.solve_from_distance_matrix(matrix, verbose=False)


@pytest.mark.unit
def test_solve_from_distance_matrix_allows_inf_on_nonrequired_edge_with_override(
    sparse_distance_matrix,
):
    """ignore_inf=True should keep solving when only non-required edges are infinite."""
    matrix = sparse_distance_matrix.copy()
    matrix[0, 4] = np.inf
    matrix[4, 0] = np.inf

    coords, stats = sbbu.solve_from_distance_matrix(
        matrix,
        verbose=False,
        ignore_nan=True,
        ignore_inf=True,
    )

    assert coords.shape == (5, 3)
    assert stats.solve_time > 0


@pytest.mark.unit
def test_solve_from_distance_matrix_allows_zero_on_nonrequired_edge_with_override(
    sparse_distance_matrix,
):
    """ignore_zero=True should keep solving when only non-required edges are zero."""
    matrix = sparse_distance_matrix.copy()
    matrix[0, 4] = 0.0
    matrix[4, 0] = 0.0

    coords, stats = sbbu.solve_from_distance_matrix(
        matrix,
        verbose=False,
        ignore_nan=True,
        ignore_zero=True,
    )

    assert coords.shape == (5, 3)
    assert stats.solve_time > 0


@pytest.mark.parametrize("num_nodes", [1.5, True, "6", None])
def test_create_test_constraints_rejects_node_count_type(
    num_nodes: float | str | None,
) -> None:
    with pytest.raises(TypeError, match="num_nodes"):
        sbbu.create_test_constraints(num_nodes=num_nodes)


@pytest.mark.parametrize("connectivity", [True, "0.5", None])
def test_create_test_constraints_rejects_connectivity_type(
    connectivity: bool | str | None,
) -> None:
    with pytest.raises(TypeError, match="connectivity"):
        sbbu.create_test_constraints(connectivity=connectivity)


@pytest.mark.parametrize("noise_level", [True, "0.1", None])
def test_create_test_constraints_rejects_noise_type(
    noise_level: bool | str | None,
) -> None:
    with pytest.raises(TypeError, match="noise_level"):
        sbbu.create_test_constraints(noise_level=noise_level)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_create_test_constraints_rejects_nonfinite_numeric_values(value: float) -> None:
    with pytest.raises(ValueError, match="connectivity"):
        sbbu.create_test_constraints(connectivity=value)
    with pytest.raises(ValueError, match="noise_level"):
        sbbu.create_test_constraints(noise_level=value)
