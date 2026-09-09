"""
Tests for SBBU adapters (NMR, Matrix, etc.).
"""

import numpy as np
import pytest

import sbbu
from sbbu.adapters import MatrixAdapter, NMRAdapter
from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.core.solver import SBBUSolver


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix():
    """Test MatrixAdapter distance matrix conversion."""
    matrix = np.array(
        [
            [0.0, 1.0, 1.4, 1.7, np.nan],
            [1.0, 0.0, 1.0, 2.0, 1.8],
            [1.4, 1.0, 0.0, 1.2, 1.6],
            [1.7, 2.0, 1.2, 0.0, 1.0],
            [np.nan, 1.8, 1.6, 1.0, 0.0],
        ]
    )

    constraint_set = MatrixAdapter.from_distance_matrix(matrix)

    assert constraint_set.num_nodes == 5
    # Should skip NaN values
    assert len(constraint_set.constraints) == 9  # 10 total - 1 NaN


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix_respects_ignore_zero():
    """ignore_zero=False should fail fast on non-positive off-diagonal entries."""
    matrix = np.array(
        [
            [0.0, 1.0, 1.4, 1.7, 0.0],
            [1.0, 0.0, 1.0, 1.4, 2.0],
            [1.4, 1.0, 0.0, 1.0, 1.4],
            [1.7, 1.4, 1.0, 0.0, 1.0],
            [0.0, 2.0, 1.4, 1.0, 0.0],
        ]
    )

    constraint_set = MatrixAdapter.from_distance_matrix(matrix, ignore_zero=True)
    assert len(constraint_set.constraints) == 9

    with pytest.raises(ValueError, match="positive"):
        MatrixAdapter.from_distance_matrix(matrix, ignore_zero=False)


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix_rejects_asymmetric_input():
    """Distance matrix ingestion should reject asymmetric inputs."""
    matrix = np.array(
        [
            [0.0, 1.0, 1.4],
            [2.0, 0.0, 1.1],
            [1.4, 1.1, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="symmetric"):
        MatrixAdapter.from_distance_matrix(matrix)


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix_rejects_asymmetric_nan_pattern():
    """One-sided NaN entries should fail symmetry validation."""
    matrix = np.array([[0.0, np.nan], [1.0, 0.0]])

    with pytest.raises(ValueError, match="symmetric"):
        MatrixAdapter.from_distance_matrix(matrix)


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix_rejects_non_zero_diagonal():
    """Distance matrix ingestion should require a zero diagonal."""
    matrix = np.array([[1.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="zero diagonal"):
        MatrixAdapter.from_distance_matrix(matrix)


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix_rejects_nan_when_ignore_disabled():
    """ignore_nan=False should fail fast on NaN entries."""
    matrix = np.array([[0.0, np.nan], [np.nan, 0.0]])

    with pytest.raises(ValueError, match="finite"):
        MatrixAdapter.from_distance_matrix(matrix, ignore_nan=False)


@pytest.mark.unit
def test_solver_rejects_missing_sequential_after_nan_ingestion():
    """Missing pairs can be represented but may prevent sequential placement."""
    matrix = np.array(
        [
            [0.0, 1.0, np.nan, 1.7],
            [1.0, 0.0, 1.0, 1.4],
            [np.nan, 1.0, 0.0, 1.0],
            [1.7, 1.4, 1.0, 0.0],
        ]
    )

    constraints = MatrixAdapter.from_distance_matrix(matrix, ignore_nan=True)
    assert constraints.get_constraint(0, 2) is None
    with pytest.raises(ValueError, match="Missing required sequential"):
        SBBUSolver(constraints)


@pytest.mark.unit
def test_solver_rejects_missing_sequential_after_zero_ingestion():
    """Explicitly skipped zeros do not supply a required sequential edge."""
    matrix = np.array(
        [
            [0.0, 1.0, 0.0, 1.7],
            [1.0, 0.0, 1.0, 1.4],
            [0.0, 1.0, 0.0, 1.0],
            [1.7, 1.4, 1.0, 0.0],
        ]
    )

    constraints = MatrixAdapter.from_distance_matrix(matrix, ignore_zero=True)
    assert constraints.get_constraint(0, 2) is None
    with pytest.raises(ValueError, match="Missing required sequential"):
        SBBUSolver(constraints)


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix_allows_mirrored_nan_with_ignore_enabled():
    """Mirrored NaN long-range pairs should be accepted when ignore_nan=True."""
    matrix = np.array(
        [
            [0.0, 1.0, 1.4, 1.7, np.nan],
            [1.0, 0.0, 1.0, 1.4, 2.0],
            [1.4, 1.0, 0.0, 1.0, 1.4],
            [1.7, 1.4, 1.0, 0.0, 1.0],
            [np.nan, 2.0, 1.4, 1.0, 0.0],
        ]
    )

    constraint_set = MatrixAdapter.from_distance_matrix(matrix, ignore_nan=True)

    assert constraint_set.num_nodes == 5
    assert len(constraint_set.constraints) == 9


@pytest.mark.unit
def test_matrix_adapter_from_distance_matrix_rejects_inf_when_ignore_disabled():
    """ignore_inf=False should fail fast on infinite entries."""
    matrix = np.array([[0.0, np.inf], [np.inf, 0.0]])

    with pytest.raises(ValueError, match="finite"):
        MatrixAdapter.from_distance_matrix(matrix, ignore_inf=False)


@pytest.mark.unit
def test_matrix_adapter_from_edge_list():
    """Test MatrixAdapter edge list conversion."""
    edges = [(0, 1, 1.0, 1.0), (1, 2, 1.5, 1.5), (0, 2, 2.0, 2.0)]

    constraint_set = MatrixAdapter.from_edge_list(3, edges)

    assert constraint_set.num_nodes == 3
    assert len(constraint_set.constraints) == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    "edges",
    [
        [(0.0, 1, 1.0, 1.0), (0, 2, 1.5, 1.5), (1, 2, 2.0, 2.0)],
        [(0, 1.0, 1.0, 1.0), (0, 2, 1.5, 1.5), (1, 2, 2.0, 2.0)],
        [(False, True, 1.0, 1.0), (0, 2, 1.5, 1.5), (1, 2, 2.0, 2.0)],
    ],
)
def test_matrix_adapter_from_edge_list_rejects_non_integer_indices(edges):
    """Edge-list ingestion should fail fast on non-integer node indices."""
    with pytest.raises(TypeError, match="integers"):
        MatrixAdapter.from_edge_list(3, edges)  # type: ignore[arg-type]


@pytest.mark.unit
def test_matrix_adapter_from_adjacency():
    """Test MatrixAdapter adjacency matrix conversion."""
    adjacency = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])

    distances = np.array(
        [
            [0.0, 1.0, 1.4, 0.0],
            [1.0, 0.0, 1.0, 2.0],
            [1.4, 1.0, 0.0, 1.2],
            [0.0, 2.0, 1.2, 0.0],
        ]
    )

    constraint_set = MatrixAdapter.from_adjacency_matrix(adjacency, distances)

    assert constraint_set.num_nodes == 4
    assert len(constraint_set.constraints) == 5  # Only connected edges


@pytest.mark.unit
def test_matrix_adapter_from_adjacency_rejects_non_square_same_shape():
    """Adjacency ingestion should reject non-square matrices."""
    adjacency = np.array([[0, 1, 0], [1, 0, 1]])
    distances = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.2]])

    with pytest.raises(ValueError, match="square"):
        MatrixAdapter.from_adjacency_matrix(adjacency, distances)


@pytest.mark.unit
def test_matrix_adapter_from_adjacency_rejects_asymmetric_adjacency():
    """Adjacency ingestion should reject asymmetric connectivity matrices."""
    adjacency = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )
    distances = np.array(
        [
            [0.0, 1.0, 1.4],
            [1.0, 0.0, 1.0],
            [1.4, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="symmetric"):
        MatrixAdapter.from_adjacency_matrix(adjacency, distances)


@pytest.mark.unit
def test_matrix_adapter_from_adjacency_rejects_asymmetric_distance_matrix():
    """Adjacency ingestion should reject asymmetric distance values."""
    adjacency = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    distances = np.array(
        [
            [0.0, 1.0, 1.4],
            [1.1, 0.0, 1.0],
            [1.4, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="symmetric"):
        MatrixAdapter.from_adjacency_matrix(adjacency, distances)


@pytest.mark.unit
def test_matrix_adapter_from_adjacency_rejects_connected_invalid_distance():
    """Connected adjacency entries must have finite positive distances."""
    adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    distances = np.array(
        [
            [0.0, np.nan, 1.4],
            [np.nan, 0.0, 1.0],
            [1.4, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="invalid distance"):
        MatrixAdapter.from_adjacency_matrix(adjacency, distances)


@pytest.mark.unit
def test_matrix_adapter_to_distance_matrix():
    """Test converting coordinates back to distance matrix."""
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Full matrix
    dist_matrix = MatrixAdapter.to_distance_matrix(coords)

    assert dist_matrix.shape == (4, 4)
    assert np.allclose(np.diag(dist_matrix), 0.0)  # Diagonal should be zero
    assert np.allclose(dist_matrix, dist_matrix.T)  # Should be symmetric

    # Check some known distances
    assert abs(dist_matrix[0, 1] - 1.0) < 1e-10
    assert abs(dist_matrix[0, 2] - 1.0) < 1e-10
    assert abs(dist_matrix[1, 2] - np.sqrt(2)) < 1e-10


@pytest.mark.unit
def test_matrix_adapter_sparse_to_distance_matrix():
    """Test converting coordinates to sparse distance matrix."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    sparse_indices = [(0, 1), (1, 2)]
    dist_matrix = MatrixAdapter.to_distance_matrix(coords, sparse_indices)

    assert dist_matrix.shape == (3, 3)
    assert np.allclose(np.diag(dist_matrix), 0.0)

    # Should have values only for specified indices
    assert not np.isnan(dist_matrix[0, 1])
    assert not np.isnan(dist_matrix[1, 2])
    assert np.isnan(dist_matrix[0, 2])


@pytest.mark.unit
def test_matrix_adapter_sparse_indices_validate_bounds():
    """Sparse distance conversion should fail fast on out-of-range indices."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    with pytest.raises(ValueError, match="out of range"):
        MatrixAdapter.to_distance_matrix(coords, [(0, 3)])


@pytest.mark.unit
def test_matrix_adapter_sparse_indices_reject_negative_indices():
    """Sparse distance conversion should reject negative node indices."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    with pytest.raises(ValueError, match="out of range"):
        MatrixAdapter.to_distance_matrix(coords, [(-1, 1)])


@pytest.mark.unit
def test_matrix_adapter_validate_distance_matrix():
    """Test distance matrix validation."""
    # Valid matrix
    valid_matrix = np.array([[0.0, 1.0, 1.4], [1.0, 0.0, 1.0], [1.4, 1.0, 0.0]])
    assert MatrixAdapter.validate_distance_matrix(valid_matrix)

    # Invalid: non-square
    invalid_matrix = np.array([[0.0, 1.0], [1.0, 0.0], [1.4, 1.0]])
    assert not MatrixAdapter.validate_distance_matrix(invalid_matrix)

    # Invalid: non-zero diagonal
    invalid_matrix = np.array([[1.0, 1.0], [1.0, 0.0]])
    assert not MatrixAdapter.validate_distance_matrix(invalid_matrix)

    # Invalid: asymmetric
    invalid_matrix = np.array([[0.0, 1.0], [2.0, 0.0]])
    assert not MatrixAdapter.validate_distance_matrix(invalid_matrix)


@pytest.mark.integration
def test_nmr_adapter_from_file(nmr_1n6t):
    """Test NMRAdapter file loading."""
    if not nmr_1n6t.exists():
        pytest.skip("Test NMR file not found")

    constraint_set = NMRAdapter.from_nmr_file(nmr_1n6t)

    assert constraint_set.num_nodes == 30
    assert len(constraint_set.constraints) > 0

    # Should have both sequential and long-range constraints
    sequential = constraint_set.get_sequential_constraints()
    long_range = constraint_set.get_long_range_constraints()

    assert len(sequential) > 0
    assert len(long_range) > 0


@pytest.mark.unit
def test_nmr_adapter_parser_initialization_does_not_write_stdout(tmp_path, capsys):
    """Initializing parser through adapter should not print to stdout."""
    nmr_path = tmp_path / "minimal.nmr"
    nmr_path.write_text("1 2 1.0 1.0 H H ALA GLY\n", encoding="utf-8")

    capsys.readouterr()
    NMRAdapter.from_nmr_file(nmr_path)
    captured = capsys.readouterr()

    assert captured.out == ""


@pytest.mark.integration
def test_nmr_adapter_rejects_short_malformed_rows(tmp_path):
    """NMR ingestion should fail fast on malformed short rows."""
    nmr_path = tmp_path / "malformed.nmr"
    nmr_path.write_text(
        "1 2 1.0 1.0 H H ALA\n1 3 1.2 1.2 H H ALA GLY\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected at least 8 fields"):
        NMRAdapter.from_nmr_file(nmr_path)


@pytest.mark.integration
def test_nmr_adapter_accepts_utf8_bom_header(tmp_path):
    """NMR ingestion should accept UTF-8 BOM-prefixed files."""
    nmr_path = tmp_path / "bom.nmr"
    nmr_path.write_text(
        "\ufeff1 2 1.0 1.0 H H ALA GLY\n"
        "1 3 1.4 1.4 H H ALA GLY\n"
        "2 3 1.0 1.0 H H ALA GLY\n",
        encoding="utf-8",
    )

    constraints = NMRAdapter.from_nmr_file(nmr_path)

    assert constraints.num_nodes == 3
    assert len(constraints.constraints) == 3


@pytest.mark.integration
def test_nmr_adapter_rejects_non_contiguous_node_ids(tmp_path):
    """NMR ingestion should fail on sparse node IDs that explode node count."""
    nmr_path = tmp_path / "sparse_ids.nmr"
    nmr_path.write_text(
        "1 100000 1.0 1.0 H H ALA GLY\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        NMRAdapter.from_nmr_file(nmr_path)


@pytest.mark.integration
def test_solver_rejects_nmr_sequential_alternatives(tmp_path):
    """NMR ingestion preserves alternatives that SBBU cannot use as predecessors."""
    nmr_path = tmp_path / "dup_seq_conflict.nmr"
    nmr_path.write_text(
        "1 2 1.0 1.0 H H ALA GLY\n"
        "1 2 1.1 1.1 H H ALA GLY\n"
        "1 3 1.4 1.4 H H ALA GLY\n"
        "2 3 1.0 1.0 H H ALA GLY\n",
        encoding="utf-8",
    )

    constraints = NMRAdapter.from_nmr_file(nmr_path)
    with pytest.raises(ValueError, match="Missing required sequential"):
        SBBUSolver(constraints)


@pytest.mark.integration
def test_nmr_adapter_solution_output(nmr_1n6t, tmp_path):
    """Test NMRAdapter solution file output."""
    if not nmr_1n6t.exists():
        pytest.skip("Test NMR file not found")

    # Solve and save
    coords, _ = sbbu.solve_from_nmr(nmr_1n6t, verbose=False)
    output_file = tmp_path / "test_output.sol"

    NMRAdapter.to_solution_file(coords, output_file, verbose=False)

    # Check file was created
    assert output_file.exists()

    # Check content format
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == coords.shape[0]

    # Check first line format (should be 3 numbers)
    first_line_parts = lines[0].split()
    assert len(first_line_parts) == 3

    # Should be able to parse as floats
    for part in first_line_parts:
        float(part)


@pytest.mark.unit
def test_nmr_adapter_solution_output_with_nmr_suffix(tmp_path):
    """Saving with .nmr path should produce *_sbbu.sol output."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    input_path = tmp_path / "example.nmr"

    NMRAdapter.to_solution_file(coords, input_path, verbose=False)

    expected_path = tmp_path / "example_sbbu.sol"
    assert expected_path.exists()


@pytest.mark.unit
def test_nmr_adapter_solution_output_rejects_non_3d_rows(tmp_path):
    """Solution writer should validate coordinate shape as (N, 3)."""
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    output_path = tmp_path / "bad_shape.sol"

    with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
        NMRAdapter.to_solution_file(coords, output_path, verbose=False)


@pytest.mark.unit
def test_nmr_adapter_solution_output_rejects_non_finite_coordinates(tmp_path):
    """Solution writer should reject non-finite coordinates."""
    coords = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]])
    output_path = tmp_path / "non_finite.sol"

    with pytest.raises(ValueError, match="finite"):
        NMRAdapter.to_solution_file(coords, output_path, verbose=False)


@pytest.mark.unit
def test_nmr_adapter_solution_output_rejects_infinite_coordinates(tmp_path):
    """Solution writer should reject infinite coordinates."""
    coords = np.array([[0.0, 0.0, 0.0], [np.inf, 1.0, 1.0]])
    output_path = tmp_path / "infinite.sol"

    with pytest.raises(ValueError, match="finite"):
        NMRAdapter.to_solution_file(coords, output_path, verbose=False)


@pytest.mark.unit
def test_nmr_adapter_verbose_output_does_not_write_stdout(tmp_path, capsys):
    """Verbose solution output should use logging instead of direct stdout writes."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    output_path = tmp_path / "verbose_output.sol"

    capsys.readouterr()
    NMRAdapter.to_solution_file(coords, output_path, verbose=True)
    captured = capsys.readouterr()

    assert captured.out == ""


@pytest.mark.unit
def test_adapter_error_handling():
    """Test adapter error handling."""
    # MatrixAdapter errors
    with pytest.raises(Exception):
        # Non-square matrix
        bad_matrix = np.array([[1, 2, 3]])
        MatrixAdapter.from_distance_matrix(bad_matrix)

    with pytest.raises(Exception):
        # Mismatched adjacency and distance matrices
        adj = np.array([[0, 1], [1, 0]])
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        MatrixAdapter.from_adjacency_matrix(adj, dist)

    # Edge list errors
    with pytest.raises(Exception):
        # Negative distance
        bad_edges = [(0, 1, -1.0, -1.0)]
        MatrixAdapter.from_edge_list(2, bad_edges)


@pytest.mark.unit
def test_constraint_set_edge_methods():
    """Test ConstraintSet edge classification methods."""
    constraints = [
        DistanceConstraint(0, 1, 1.0, 1.0),  # Sequential
        DistanceConstraint(1, 2, 1.0, 1.0),  # Sequential
        DistanceConstraint(2, 3, 1.0, 1.0),  # Sequential
        DistanceConstraint(0, 4, 2.0, 2.0),  # Long-range (4-0=4 > 3)
        DistanceConstraint(1, 5, 2.1, 2.1),  # Long-range (5-1=4 > 3)
    ]

    constraint_set = ConstraintSet(6, constraints)

    sequential = constraint_set.get_sequential_constraints()
    long_range = constraint_set.get_long_range_constraints()

    assert len(sequential) == 3
    assert len(long_range) == 2

    # Check constraint lookup
    assert constraint_set.has_constraint(0, 1)
    assert constraint_set.has_constraint(1, 0)  # Should work both ways
    assert not constraint_set.has_constraint(0, 3)
