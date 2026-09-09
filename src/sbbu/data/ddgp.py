"""
Discretizable Distance Geometry Problem (DDGP) implementation.

This module provides a modern implementation of DDGP data structures and algorithms.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .nmr import Edge, NMRParser


class DDGProblem:
    """Represent a discretizable distance geometry problem.

    Parameters
    ----------
    file_path : str | Path
        Path to the NMR constraint file.
    distance_tolerance : float, default=1e-3
        Absolute tolerance used for distance constraint checks.
    """

    def __init__(self, file_path: str | Path, distance_tolerance: float = 1e-3) -> None:
        """Initialize a DDGP instance from an NMR constraint file.

        Parameters
        ----------
        file_path : str | Path
            Path to the NMR constraint file.
        distance_tolerance : float, default=1e-3
            Absolute tolerance used for distance constraint checks.
        """
        # Parse NMR data
        self.nmr_parser = NMRParser(file_path, distance_tolerance)
        self.distance_tolerance = distance_tolerance

        # Copy basic properties
        self.num_nodes = self.nmr_parser.num_nodes
        self.num_edges = self.nmr_parser.num_edges
        self.edges = self.nmr_parser.edges

        # Build DDGP-specific structures
        self._build_ddgp_constraints()

    def _build_ddgp_constraints(self) -> None:
        """Build DDGP-specific constraint arrays."""
        # Arrays for exact distances to previous 3 atoms
        self.a_squared = np.zeros(self.num_nodes, dtype=np.float64)  # Distance² to i-1
        self.b_squared = np.zeros(self.num_nodes, dtype=np.float64)  # Distance² to i-2
        self.c_squared = np.zeros(self.num_nodes, dtype=np.float64)  # Distance² to i-3

        # Extract exact distances for consecutive atoms
        for i in range(self.num_nodes):
            if i > 0:
                lower, _upper = self._get_exact_bound(i, i - 1)
                self.a_squared[i] = lower * lower

            if i > 1:
                lower, _upper = self._get_exact_bound(i, i - 2)
                self.b_squared[i] = lower * lower

            if i > 2:
                lower, _upper = self._get_exact_bound(i, i - 3)
                self.c_squared[i] = lower * lower

    def _get_exact_bound(self, i: int, j: int) -> tuple[float, float]:
        """Get bounds for edge and verify it's exact (lower == upper)."""
        lower, upper = self.nmr_parser.get_bounds(i, j)
        if abs(lower - upper) > 1e-10:  # Stricter tolerance for exact constraints
            raise ValueError(f"Edge ({i}, {j}) is not exact: bounds=[{lower}, {upper}]")
        return lower, upper

    def get_bounds(self, i: int, j: int) -> tuple[float, float]:
        """Return distance bounds for an edge.

        Parameters
        ----------
        i : int
            First node index.
        j : int
            Second node index.

        Returns
        -------
        tuple[float, float]
            Lower and upper distance bounds for edge ``(i, j)``.

        Raises
        ------
        ValueError
            If the edge is not present.
        """
        return self.nmr_parser.get_bounds(i, j)

    def find_bounds(self, i: int, j: int) -> tuple[float, float] | None:
        """Return distance bounds for an edge when available.

        Parameters
        ----------
        i : int
            First node index.
        j : int
            Second node index.

        Returns
        -------
        tuple[float, float] | None
            Lower and upper bounds for edge ``(i, j)``, or ``None`` if absent.
        """
        return self.nmr_parser.find_bounds(i, j)

    def get_lower_bound(self, i: int, j: int) -> float:
        """Return the lower distance bound for an edge.

        Parameters
        ----------
        i : int
            First node index.
        j : int
            Second node index.

        Returns
        -------
        float
            Lower distance bound for edge ``(i, j)``.

        Raises
        ------
        ValueError
            If the edge is not present.
        """
        return self.nmr_parser.get_lower_bound(i, j)

    def get_upper_bound(self, i: int, j: int) -> float:
        """Return the upper distance bound for an edge.

        Parameters
        ----------
        i : int
            First node index.
        j : int
            Second node index.

        Returns
        -------
        float
            Upper distance bound for edge ``(i, j)``.

        Raises
        ------
        ValueError
            If the edge is not present.
        """
        return self.nmr_parser.get_upper_bound(i, j)

    def is_feasible(
        self, coordinates: NDArray[np.float64], node: int | None = None
    ) -> bool:
        """Check whether coordinates satisfy distance constraints.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Coordinate array with shape ``(n, 3)``.
        node : int | None, default=None
            Node index to validate. If ``None``, all nodes are checked.

        Returns
        -------
        bool
            ``True`` if all checked constraints are satisfied, else ``False``.

        Raises
        ------
        ValueError
            If coordinates are invalid for this problem.
        """
        return self.nmr_parser.is_feasible(coordinates, node)

    def assert_feasibility(self, coordinates: NDArray[np.float64]) -> None:
        """Assert that coordinates satisfy all distance constraints.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Coordinate array with shape ``(n, 3)``.

        Raises
        ------
        RuntimeError
            If any constraint violation exceeds the configured tolerance.
        """
        self.nmr_parser.assert_feasibility(coordinates)

    def mean_distance_error(self, coordinates: NDArray[np.float64]) -> float:
        """Compute the mean relative distance error.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Coordinate array with shape ``(n, 3)``.

        Returns
        -------
        float
            Mean relative error across checked constraints.
        """
        total_error = 0.0
        num_constraints = 0

        for i in range(self.num_nodes):
            for k in range(self.nmr_parser.row_ptr[i], self.nmr_parser.row_ptr[i + 1]):
                j = self.nmr_parser.col_indices[k]
                if j > i:  # Avoid double counting
                    break

                pos_i = coordinates[i]
                pos_j = coordinates[j]
                actual_distance = float(np.linalg.norm(pos_i - pos_j))

                # Use lower bound as reference distance
                reference_distance = self.nmr_parser.lower_bounds[k]
                relative_error = (
                    abs(actual_distance - reference_distance) / reference_distance
                )

                total_error += relative_error
                num_constraints += 1

        return total_error / num_constraints if num_constraints > 0 else 0.0

    def largest_distance_error(self, coordinates: NDArray[np.float64]) -> float:
        """Compute the largest absolute distance error.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Coordinate array with shape ``(n, 3)``.

        Returns
        -------
        float
            Maximum absolute error across checked constraints.
        """
        max_error = 0.0

        for i in range(self.num_nodes):
            for k in range(self.nmr_parser.row_ptr[i], self.nmr_parser.row_ptr[i + 1]):
                j = self.nmr_parser.col_indices[k]
                if j > i:  # Avoid double counting
                    break

                pos_i = coordinates[i]
                pos_j = coordinates[j]
                actual_distance = float(np.linalg.norm(pos_i - pos_j))

                # Use midpoint of bounds as reference
                lower = self.nmr_parser.lower_bounds[k]
                upper = self.nmr_parser.upper_bounds[k]
                reference_distance = 0.5 * (lower + upper)

                error = abs(actual_distance - reference_distance)
                max_error = max(max_error, error)

        return max_error

    def get_long_range_edges(self, min_separation: int = 3) -> list[Edge]:
        """Return long-range edges used by SBBU.

        Parameters
        ----------
        min_separation : int, default=3
            Minimum index separation between edge endpoints.

        Returns
        -------
        list[Edge]
            Edges with separation greater than ``min_separation``, sorted by
            ``(j, j - i)``.
        """
        long_range_edges = []

        for edge in self.edges:
            if edge.j - edge.i > min_separation:
                long_range_edges.append(edge)

        # Sort by (j, separation) as in original implementation
        long_range_edges.sort(key=lambda e: (e.j, e.j - e.i))

        return long_range_edges

    @property
    def file_path(self) -> Path:
        """Return the source NMR constraint file path.

        Returns
        -------
        Path
            File path used to construct this problem.
        """
        return self.nmr_parser.file_path
