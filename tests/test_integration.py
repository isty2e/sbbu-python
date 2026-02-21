"""
Integration tests for SBBU algorithm.
"""

import numpy as np
import pytest

import sbbu
from sbbu.core.solver import SBBUConfig, SBBUSolver


@pytest.mark.integration
@pytest.mark.slow
def test_nmr_files_accuracy(nmr_1n6t, sol_1n6t, nmr_1adx, sol_1adx):
    """Test accuracy on real NMR files against C++ reference."""
    test_cases = [
        (nmr_1n6t, sol_1n6t, "1n6t"),
        (nmr_1adx, sol_1adx, "1adx"),
    ]

    for nmr_file, sol_file, name in test_cases:
        if not (nmr_file.exists() and sol_file.exists()):
            pytest.skip(f"Test files for {name} not found")

        # Solve with SBBU
        coords, stats = sbbu.solve_from_nmr(nmr_file, verbose=False)

        # Load reference
        ref_coords = np.loadtxt(sol_file)

        # Check basic properties
        assert coords.shape == ref_coords.shape
        assert stats.solve_time > 0
        assert stats.mean_distance_error < 1e-10

        # Check accuracy against reference
        max_diff = np.max(
            [np.linalg.norm(coords[i] - ref_coords[i]) for i in range(coords.shape[0])]
        )

        # Should match C++ reference within numerical precision
        assert max_diff < 1e-6, (
            f"Failed accuracy test for {name}: max_diff = {max_diff}"
        )


@pytest.mark.integration
def test_api_consistency():
    """Test that different APIs produce consistent results."""
    # Create test case with known solution
    true_coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0], [0.5, 0.289, 0.816]]
    )

    n = true_coords.shape[0]

    # Create distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(true_coords[i] - true_coords[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Create edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_matrix[i, j]
            edges.append((i, j, dist))

    # Solve with different APIs
    coords1, _ = sbbu.solve_from_distance_matrix(distance_matrix, verbose=False)
    coords2, _ = sbbu.solve_from_edge_list(n, edges, verbose=False)

    # Results should be consistent (within numerical precision)
    max_diff = np.max([np.linalg.norm(coords1[i] - coords2[i]) for i in range(n)])

    assert max_diff < 1e-10


@pytest.mark.integration
def test_large_problem_performance():
    """Test performance on larger synthetic problems."""
    # Create truly large problem with no noise (only sparsity challenge)
    constraints = sbbu.create_test_constraints(
        num_nodes=100, connectivity=0.1, noise_level=0.0, random_seed=42
    )

    # Use strict tolerance for noise-free data
    config = SBBUConfig(distance_tolerance=1e-6, max_time=30.0, verbose=False)

    solver = SBBUSolver(constraints, config)
    stats = solver.solve()

    coords = solver.solution_coordinates

    # Check basic properties
    assert coords.shape == (100, 3)
    assert stats.solve_time < 30.0  # Should finish within time limit
    assert (
        stats.mean_distance_error < 1e-10
    )  # Should be extremely accurate with no noise

    # Check constraint satisfaction
    for constraint in constraints.constraints:
        actual_dist = np.linalg.norm(coords[constraint.i] - coords[constraint.j])
        error = abs(actual_dist - constraint.distance)
        assert error < config.distance_tolerance


@pytest.mark.integration
def test_noisy_data_robustness():
    """Test robustness to noisy distance data."""
    # Create clean test case using isolated RNG
    rng = np.random.RandomState(42)

    true_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816],
            [1.5, 0.289, 0.816],
        ]
    )

    n = true_coords.shape[0]

    # Add noise to distances using isolated RNG
    noise_level = 0.005  # 0.5% noise level

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            true_dist = np.linalg.norm(true_coords[i] - true_coords[j])
            noise = rng.normal(0, noise_level * true_dist)
            noisy_dist = max(0.1, true_dist + noise)
            edges.append((i, j, noisy_dist))

    # Solve with noisy data and appropriate tolerance
    coords, stats = sbbu.solve_from_edge_list(
        n, edges, distance_tolerance=5e-3, verbose=False
    )

    # Should still converge
    assert coords.shape == (n, 3)
    assert stats.solve_time > 0

    # Reconstruction error should be reasonable given noise
    reconstruction_errors = []
    for i, j, noisy_dist in edges:
        actual_dist = np.linalg.norm(coords[i] - coords[j])
        error = abs(actual_dist - noisy_dist)
        reconstruction_errors.append(error)

    mean_error = np.mean(reconstruction_errors)
    assert mean_error < 1e-3  # Should be small despite noise


@pytest.mark.integration
def test_sparse_connectivity():
    """Test with sparse connectivity (minimal constraints)."""
    # Create geometrically feasible sparse case using real coordinates
    rng = np.random.RandomState(42)
    n = 8

    # Generate true coordinates (chain-like structure)
    true_coords = np.zeros((n, 3))
    for i in range(n):
        true_coords[i] = [
            i * 1.2 + rng.normal(0, 0.1),
            rng.normal(0, 0.3),
            rng.normal(0, 0.2),
        ]

    # Calculate true distances
    edges = []

    # Sequential connectivity (required)
    for i in range(n - 1):
        for j in range(i + 1, min(i + 4, n)):
            dist = np.linalg.norm(true_coords[i] - true_coords[j])
            edges.append((i, j, dist))

    # Add minimal long-range constraints (geometrically feasible)
    long_range_pairs = [(0, 5), (1, 6), (2, 7)]
    for i, j in long_range_pairs:
        if j < n:
            dist = np.linalg.norm(true_coords[i] - true_coords[j])
            edges.append((i, j, dist))

    coords, stats = sbbu.solve_from_edge_list(n, edges, verbose=False)

    # Should solve successfully
    assert coords.shape == (n, 3)
    assert stats.solve_time > 0

    # Check constraint satisfaction
    for i, j, target_dist in edges:
        actual_dist = np.linalg.norm(coords[i] - coords[j])
        error = abs(actual_dist - target_dist)
        assert error < 1e-6


@pytest.mark.integration
def test_configuration_robustness():
    """Test different solver configurations."""
    # Simple test case
    matrix = np.array(
        [
            [0.0, 1.0, 1.4, 1.7],
            [1.0, 0.0, 1.0, 1.4],
            [1.4, 1.0, 0.0, 1.0],
            [1.7, 1.4, 1.0, 0.0],
        ]
    )

    configs = [
        SBBUConfig(distance_tolerance=1e-5, verbose=False),
        SBBUConfig(distance_tolerance=1e-7, verbose=False),
        SBBUConfig(distance_tolerance=1e-6, max_time=5.0, verbose=False),
    ]

    results = []
    for config in configs:
        coords, stats = sbbu.solve_from_distance_matrix(
            matrix,
            distance_tolerance=config.distance_tolerance,
            max_time=config.max_time,
            verbose=config.verbose,
        )
        results.append((coords, stats))

    # All should succeed
    for coords, stats in results:
        assert coords.shape == (4, 3)
        assert stats.solve_time > 0

    # Results should be consistent
    coords1, _ = results[0]
    coords2, _ = results[1]

    max_diff = np.max([np.linalg.norm(coords1[i] - coords2[i]) for i in range(4)])

    assert max_diff < 1e-4  # Should be reasonably close
