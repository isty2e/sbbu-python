"""
High-level API for SBBU algorithm.

This module provides convenient functions for common SBBU use cases.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .adapters.matrix_adapter import MatrixAdapter
from .adapters.nmr_adapter import NMRAdapter
from .constraints import ConstraintSet
from .core.solver import SBBUConfig, SBBUSolver, SBBUStats


def solve_from_nmr(
    nmr_file: str | Path,
    distance_tolerance: float = 1e-7,
    max_time: float = 60.0,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], SBBUStats]:
    """
    Solve SBBU from an NMR file.

    Parameters
    ----------
    nmr_file : str | pathlib.Path
        Path to the NMR input file.
    distance_tolerance : float, default=1e-7
        Distance tolerance used by the solver.
    max_time : float, default=60.0
        Maximum solving time in seconds.
    verbose : bool, default=True
        If ``True``, emit solver progress logs.

    Returns
    -------
    tuple[numpy.typing.NDArray[numpy.float64], SBBUStats]
        A tuple ``(coordinates, stats)`` with solved coordinates and run statistics.

    Raises
    ------
    Exception
        Raised when loading input, constructing configuration, or solving fails.
    """
    constraints = NMRAdapter.from_nmr_file(nmr_file)
    config = SBBUConfig(
        distance_tolerance=distance_tolerance, max_time=max_time, verbose=verbose
    )

    solver = SBBUSolver(constraints, config)
    stats = solver.solve()
    return solver.solution_coordinates, stats


def solve_from_distance_matrix(
    distance_matrix: NDArray[np.float64],
    distance_tolerance: float = 1e-7,
    max_time: float = 60.0,
    verbose: bool = True,
    ignore_nan: bool = True,
    ignore_inf: bool = False,
    ignore_zero: bool = False,
) -> tuple[NDArray[np.float64], SBBUStats]:
    """
    Solve SBBU from a distance matrix.

    Parameters
    ----------
    distance_matrix : numpy.typing.NDArray[numpy.float64]
        Square distance matrix. Missing distances may be encoded as ``NaN``.
    distance_tolerance : float, default=1e-7
        Distance tolerance used by the solver.
    max_time : float, default=60.0
        Maximum solving time in seconds.
    verbose : bool, default=True
        If ``True``, emit solver progress logs.
    ignore_nan : bool, default=True
        If ``True``, ignore ``NaN`` entries in the distance matrix.
    ignore_inf : bool, default=False
        If ``True``, ignore infinite entries in the distance matrix.
    ignore_zero : bool, default=False
        If ``True``, ignore non-positive off-diagonal distances.

    Returns
    -------
    tuple[numpy.typing.NDArray[numpy.float64], SBBUStats]
        A tuple ``(coordinates, stats)`` with solved coordinates and run statistics.

    Raises
    ------
    Exception
        Raised when parsing input, constructing configuration, or solving fails.
    """
    constraints = MatrixAdapter.from_distance_matrix(
        distance_matrix,
        ignore_nan=ignore_nan,
        ignore_inf=ignore_inf,
        ignore_zero=ignore_zero,
    )
    config = SBBUConfig(
        distance_tolerance=distance_tolerance, max_time=max_time, verbose=verbose
    )

    solver = SBBUSolver(constraints, config)
    stats = solver.solve()
    return solver.solution_coordinates, stats


def solve_from_bounds_matrices(
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    distance_tolerance: float = 1e-7,
    max_time: float = 60.0,
    verbose: bool = True,
    *,
    ignore_nan: bool = True,
) -> tuple[NDArray[np.float64], SBBUStats]:
    """Solve from paired distance-bound matrices.

    Parameters
    ----------
    lower_bounds : NDArray[np.float64]
        Symmetric square lower bounds with zero diagonal.
    upper_bounds : NDArray[np.float64]
        Same-shaped symmetric upper bounds with zero diagonal. Equal endpoints
        denote exact distances; SBBU requires exact sequential predecessors.
    distance_tolerance : float, default=1e-7
        Absolute tolerance for accepting constraint violations, not for deciding
        whether an input interval is exact.
    max_time : float, default=60.0
        Maximum solving time in seconds.
    verbose : bool, default=True
        Whether to emit solver progress logs.
    ignore_nan : bool, default=True
        Omit pairs with matching NaNs in both bounds and directions.

    Returns
    -------
    tuple[NDArray[np.float64], SBBUStats]
        Coordinates in input node order and solver statistics.

    Raises
    ------
    ValueError
        If matrix bounds are invalid or required exact predecessors are absent.
    SBBUSolveInfeasibleError
        If the bounded search cannot satisfy the hard constraints.
    SBBUTimeoutError
        If the solve exceeds the time limit.
    """
    constraints = MatrixAdapter.from_bounds_matrices(
        lower_bounds, upper_bounds, ignore_nan=ignore_nan
    )
    config = SBBUConfig(
        distance_tolerance=distance_tolerance, max_time=max_time, verbose=verbose
    )
    solver = SBBUSolver(constraints, config)
    stats = solver.solve()
    return solver.solution_coordinates, stats


def solve_from_edge_list(
    num_nodes: int,
    edges: Sequence[tuple[int, int, float, float]],
    distance_tolerance: float = 1e-7,
    max_time: float = 60.0,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], SBBUStats]:
    """
    Solve SBBU from an edge list.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes.
    edges : Sequence[tuple[int, int, float, float]]
        Distance constraints as ``(i, j, lower_bound, upper_bound)`` tuples.
        Equal bounds specify an exact distance.
    distance_tolerance : float, default=1e-7
        Distance tolerance used by the solver.
    max_time : float, default=60.0
        Maximum solving time in seconds.
    verbose : bool, default=True
        If ``True``, emit solver progress logs.

    Returns
    -------
    tuple[numpy.typing.NDArray[numpy.float64], SBBUStats]
        A tuple ``(coordinates, stats)`` with solved coordinates and run statistics.

    Raises
    ------
    Exception
        Raised when parsing input, constructing configuration, or solving fails.
    """
    constraints = MatrixAdapter.from_edge_list(num_nodes, edges)
    config = SBBUConfig(
        distance_tolerance=distance_tolerance, max_time=max_time, verbose=verbose
    )

    solver = SBBUSolver(constraints, config)
    stats = solver.solve()
    return solver.solution_coordinates, stats


def create_test_constraints(
    num_nodes: int = 10,
    connectivity: float = 0.3,
    noise_level: float = 0.0,
    random_seed: int | None = None,
) -> ConstraintSet:
    """
    Create synthetic test constraints for SBBU development/testing.

    Parameters
    ----------
    num_nodes : int, default=10
        Number of nodes.
    connectivity : float, default=0.3
        Fraction of possible edges to include, in ``[0.0, 1.0]``.
    noise_level : float, default=0.0
        Relative Gaussian noise scale applied to sampled distances.
    random_seed : int | None, default=None
        Seed used to initialize an isolated ``numpy.random.RandomState``.

    Returns
    -------
    ConstraintSet
        Synthetic distance constraints generated from random 3D coordinates.

    Raises
    ------
    TypeError
        If ``num_nodes`` is not an integer, or ``connectivity`` or ``noise_level``
        is not an integer or float. Booleans are not accepted.
    ValueError
        If ``num_nodes`` is nonpositive, ``connectivity`` is not finite within
        ``[0.0, 1.0]``, or ``noise_level`` is negative/nonfinite.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise TypeError(f"num_nodes must be an integer, got {num_nodes}")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if isinstance(connectivity, bool) or not isinstance(connectivity, (int, float)):
        raise TypeError(f"connectivity must be an integer or float, got {connectivity}")
    if not np.isfinite(connectivity) or not (0.0 <= float(connectivity) <= 1.0):
        raise ValueError(f"connectivity must be in [0, 1], got {connectivity}")
    if isinstance(noise_level, bool) or not isinstance(noise_level, (int, float)):
        raise TypeError(f"noise_level must be an integer or float, got {noise_level}")
    if not np.isfinite(noise_level) or float(noise_level) < 0.0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")

    # Create isolated RNG instance
    rng = np.random.RandomState(random_seed)

    # 1. Generate true coordinates (reasonable scale)
    true_coords = rng.randn(num_nodes, 3) * 2.0

    # 2. Calculate all true distances
    true_distances = {}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(true_coords[i] - true_coords[j])
            true_distances[(i, j)] = dist

    # 3. Categorize edges by separation
    all_edges = list(true_distances.keys())
    sequential_edges = [(i, j) for i, j in all_edges if j - i <= 3]
    long_range_edges = [(i, j) for i, j in all_edges if j - i > 3]

    # 4. Always include sequential edges, sample long-range edges
    total_possible_edges = len(all_edges)
    target_edges = int(total_possible_edges * connectivity)
    num_long_range = max(0, target_edges - len(sequential_edges))

    selected_edges = sequential_edges.copy()
    if num_long_range > 0 and len(long_range_edges) > 0:
        sampled_indices = rng.choice(
            len(long_range_edges),
            min(num_long_range, len(long_range_edges)),
            replace=False,
        )
        selected_long_range = [long_range_edges[idx] for idx in sampled_indices]
        selected_edges.extend(selected_long_range)

    # 5. Add noise to selected edges
    edges = []
    for i, j in selected_edges:
        true_dist = true_distances[(i, j)]
        if noise_level > 0:
            noise = rng.normal(0, noise_level * true_dist)
            noisy_dist = max(0.1, true_dist + noise)
        else:
            noisy_dist = true_dist
        distance = float(noisy_dist)
        edges.append((i, j, distance, distance))

    return MatrixAdapter.from_edge_list(num_nodes, edges)
