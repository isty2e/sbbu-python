"""
SBBU (Symmetry-based Build-up) implementation.

This package provides a pythonic, modular implementation of the SBBU algorithm
for solving Distance Geometry Problems (DGP) from various input sources.

Usage:
    from sbbu import solve_from_nmr, solve_from_distance_matrix, solve_from_edge_list

    # For NMR files
    coords, stats = solve_from_nmr('protein.nmr')

    # For distance matrices (e.g., from ML predictions)
    coords, stats = solve_from_distance_matrix(distance_matrix)

    # For graph-like data
    edges = [(0, 1, 1.2, 1.2), (1, 2, 1.1, 1.1), (0, 2, 1.3, 1.3)]
    coords, stats = solve_from_edge_list(3, edges)
"""

from importlib.metadata import PackageNotFoundError, version

# High-level API (recommended for most users)
from .api import (
    create_test_constraints,
    solve_from_bounds_matrices,
    solve_from_distance_matrix,
    solve_from_edge_list,
    solve_from_nmr,
)
from .core.solver import SBBUConfig, SBBUSolver
from .solving.run import (
    MMStage,
    SBBUStage,
    SolveError,
    SolveStats,
    SolveTimeoutError,
    TRFStage,
    UnsupportedProblemError,
)

try:
    __version__ = version("sbbu")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "MMStage",
    "SBBUConfig",
    "SBBUSolver",
    "SBBUStage",
    "SolveError",
    "SolveStats",
    "SolveTimeoutError",
    "TRFStage",
    "UnsupportedProblemError",
    "create_test_constraints",
    "solve_from_bounds_matrices",
    "solve_from_distance_matrix",
    "solve_from_edge_list",
    "solve_from_nmr",
]
