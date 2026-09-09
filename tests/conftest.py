"""
Pytest configuration and fixtures for SBBU tests.
"""

# Ensure we can import sbbu from src/
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.core.solver import SBBUConfig


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def nmr_1n6t(test_data_dir: Path) -> Path:
    """Path to 1n6t.nmr test file."""
    return test_data_dir / "1n6t.nmr"


@pytest.fixture(scope="session")
def sol_1n6t(test_data_dir: Path) -> Path:
    """Path to 1n6t reference solution."""
    return test_data_dir / "1n6t_sbbu.sol"


@pytest.fixture(scope="session")
def nmr_1adx(test_data_dir: Path) -> Path:
    """Path to 1adx.nmr test file."""
    return test_data_dir / "1adx.nmr"


@pytest.fixture(scope="session")
def sol_1adx(test_data_dir: Path) -> Path:
    """Path to 1adx reference solution."""
    return test_data_dir / "1adx_sbbu.sol"


@pytest.fixture
def simple_tetrahedron() -> tuple[np.ndarray, ConstraintSet]:
    """
    Simple 4-node tetrahedron test case with exact distances.

    Returns:
        Tuple of (true_coordinates, constraint_set)
    """
    # Regular tetrahedron coordinates
    true_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
        ]
    )

    # Create constraints from true distances
    constraints = []
    n = true_coords.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(true_coords[i] - true_coords[j]))
            constraints.append(DistanceConstraint(i, j, dist, dist))

    constraint_set = ConstraintSet(n, constraints)
    return true_coords, constraint_set


@pytest.fixture
def simple_chain() -> tuple[np.ndarray, ConstraintSet]:
    """
    Simple 5-node chain test case.

    Returns:
        Tuple of (true_coordinates, constraint_set)
    """
    # Chain along x-axis with unit distances
    true_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    # Add sequential constraints (required) + some long-range
    constraints = []
    n = true_coords.shape[0]

    # Sequential constraints (i, i+1, i+2, i+3)
    for i in range(n):
        for j in range(i + 1, min(i + 4, n)):
            dist = float(np.linalg.norm(true_coords[i] - true_coords[j]))
            constraints.append(DistanceConstraint(i, j, dist, dist))

    # Add a few long-range constraints
    for i, j in [(0, 4)]:  # End-to-end distance
        if j < n:
            dist = float(np.linalg.norm(true_coords[i] - true_coords[j]))
            constraints.append(DistanceConstraint(i, j, dist, dist))

    constraint_set = ConstraintSet(n, constraints)
    return true_coords, constraint_set


@pytest.fixture
def distance_matrix_4x4() -> np.ndarray:
    """4x4 distance matrix for testing."""
    return np.array(
        [
            [0.0, 1.0, 1.414, 1.732],
            [1.0, 0.0, 1.0, 1.414],
            [1.414, 1.0, 0.0, 1.0],
            [1.732, 1.414, 1.0, 0.0],
        ]
    )


@pytest.fixture
def sparse_distance_matrix() -> np.ndarray:
    """Sparse distance matrix with NaN values."""
    matrix = np.full((5, 5), np.nan)
    np.fill_diagonal(matrix, 0.0)

    # Add required sequential constraints for SBBU initialization
    known_distances = [
        # Required for initialization
        (0, 1, 1.2),
        (0, 2, 1.7),
        (1, 2, 1.1),
        # Additional sequential
        (1, 3, 1.3),
        (2, 3, 1.0),
        (2, 4, 1.4),
        (3, 4, 1.0),
        # Long-range
        (0, 3, 2.1),
        (1, 4, 2.0),
    ]

    for i, j, dist in known_distances:
        matrix[i, j] = dist
        matrix[j, i] = dist

    return matrix


@pytest.fixture
def edge_list_simple() -> tuple[int, list[tuple[int, int, float, float]]]:
    """Simple edge list for testing."""
    num_nodes = 4
    edges = [
        (0, 1, 1.0, 1.0),
        (1, 2, 1.0, 1.0),
        (2, 3, 1.0, 1.0),
        (0, 2, 1.414, 1.414),
        (1, 3, 1.414, 1.414),
        (0, 3, 1.732, 1.732),
    ]
    return num_nodes, edges


@pytest.fixture
def sbbu_config_fast() -> SBBUConfig:
    """Fast SBBU configuration for testing."""
    return SBBUConfig(
        distance_tolerance=1e-6,  # Slightly relaxed for speed
        max_time=10.0,  # Short time limit
        verbose=False,  # Quiet during tests
    )


@pytest.fixture
def sbbu_config_accurate() -> SBBUConfig:
    """Accurate SBBU configuration for testing."""
    return SBBUConfig(
        distance_tolerance=1e-7,  # Match C++ precision
        max_time=60.0,
        verbose=False,
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
