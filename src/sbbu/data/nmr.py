"""
NMR data parser for molecular distance constraints.

This module provides a modern, type-safe parser for NMR distance constraint files.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

try:
    from typing import Self
except ImportError:  # pragma: no cover - Python < 3.11
    from typing_extensions import Self

from ..validation import (
    validate_distance_bounds,
    validate_file_path,
    validate_positive_float,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Edge:
    """Represent a distance constraint between two atoms.

    Attributes
    ----------
    i : int
        First atom index (0-based).
    j : int
        Second atom index (0-based).
    lower_bound : float
        Minimum allowed distance.
    upper_bound : float
        Maximum allowed distance.
    """

    i: int  # First atom index (0-based)
    j: int  # Second atom index (0-based)
    lower_bound: float  # Minimum allowed distance
    upper_bound: float  # Maximum allowed distance

    def __post_init__(self) -> None:
        """Validate edge parameters."""
        if self.i < 0 or self.j < 0:
            raise ValueError(
                f"Atom indices must be non-negative: i={self.i}, j={self.j}"
            )
        validate_distance_bounds(self.lower_bound, self.upper_bound)

    def __lt__(self, other: Self) -> bool:
        """Define ordering for sorting edges."""
        return (self.i, self.j) < (other.i, other.j)


class NMRData(NamedTuple):
    """Store one parsed constraint row from an NMR file.

    Attributes
    ----------
    i : int
        First atom index (1-based in file data).
    j : int
        Second atom index (1-based in file data).
    lower : float
        Lower distance bound.
    upper : float
        Upper distance bound.
    atom_i : str
        Atom name for index ``i``.
    atom_j : str
        Atom name for index ``j``.
    amino_i : str
        Residue name for index ``i``.
    amino_j : str
        Residue name for index ``j``.
    """

    i: int
    j: int
    lower: float
    upper: float
    atom_i: str
    atom_j: str
    amino_i: str
    amino_j: str


class NMRParser:
    """Parse and validate NMR distance-constraint files.

    Parameters
    ----------
    file_path : str | Path
        Path to the constraint file.
    distance_tolerance : float, default=1e-3
        Absolute tolerance for feasibility checks.
    """

    def __init__(self, file_path: str | Path, distance_tolerance: float = 1e-3):
        """Initialize parser state and parse the constraint file.

        Parameters
        ----------
        file_path : str | Path
            Path to the NMR constraint file.
        distance_tolerance : float, default=1e-3
            Absolute tolerance used for distance constraint checks.
        """
        self.file_path = validate_file_path(file_path)
        validate_positive_float(distance_tolerance, "distance_tolerance")
        self.distance_tolerance = distance_tolerance

        # Parse the file
        self._raw_data = self._parse_file()
        self._build_structure()

    def _parse_file(self) -> list[NMRData]:
        """Parse NMR constraint file."""
        raw_data = []

        with open(self.file_path, encoding="utf-8-sig") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    raise ValueError(
                        "Invalid data on line "
                        f"{line_num} of {self.file_path}: expected at least 8 fields, "
                        f"got {len(parts)}: {line}"
                    )

                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    lower = float(parts[2])
                    upper = float(parts[3])
                    atom_i = parts[4]
                    atom_j = parts[5]
                    amino_i = parts[6]
                    amino_j = parts[7]

                    raw_data.append(
                        NMRData(i, j, lower, upper, atom_i, atom_j, amino_i, amino_j)
                    )

                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"Invalid data on line {line_num} of {self.file_path}: {line}"
                    ) from e

        if not raw_data:
            raise ValueError(f"No valid constraint data found in {self.file_path}")

        return raw_data

    def _build_structure(self) -> None:
        """Build internal data structures from parsed data."""
        # Create edges (both directions for each constraint)
        edges = []
        max_node = 0
        unique_nodes: set[int] = set()

        for data in self._raw_data:
            if data.i <= 0 or data.j <= 0:
                raise ValueError(
                    f"Atom indices must be positive 1-based values, got ({data.i}, {data.j})"
                )

            # Convert to 0-based indexing
            i, j = data.i - 1, data.j - 1
            max_node = max(max_node, i, j)
            unique_nodes.add(i)
            unique_nodes.add(j)

            # Add both directions
            edges.append(Edge(i, j, data.lower, data.upper))
            edges.append(Edge(j, i, data.lower, data.upper))

        if len(unique_nodes) != max_node + 1:
            raise ValueError(
                "Node indices must be contiguous 1-based values with no gaps "
                f"(found {len(unique_nodes)} unique nodes up to index {max_node + 1})"
            )

        # Sort edges for CSR structure
        edges.sort()

        self.num_nodes = max_node + 1
        self.num_edges = len(edges)
        self.edges = edges

        logger.debug("NMR: nnodes = %d", self.num_nodes)
        logger.debug("NMR: nedges = %d", self.num_edges)

        # Build CSR structure
        self._build_csr()

    def _build_csr(self) -> None:
        """Build Compressed Sparse Row structure for efficient access."""
        # CSR row pointers
        self.row_ptr = np.zeros(self.num_nodes + 1, dtype=np.int32)

        # Count edges per node
        for edge in self.edges:
            self.row_ptr[edge.i + 1] += 1

        # Convert counts to cumulative sum
        for i in range(self.num_nodes):
            self.row_ptr[i + 1] += self.row_ptr[i]

        # CSR column indices and values
        self.col_indices = np.zeros(self.num_edges, dtype=np.int32)
        self.lower_bounds = np.zeros(self.num_edges, dtype=np.float64)
        self.upper_bounds = np.zeros(self.num_edges, dtype=np.float64)

        for k, edge in enumerate(self.edges):
            self.col_indices[k] = edge.j
            self.lower_bounds[k] = edge.lower_bound
            self.upper_bounds[k] = edge.upper_bound

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
        bounds = self.find_bounds(i, j)
        if bounds is not None:
            return bounds

        raise ValueError(f"No edge found between nodes {i} and {j}")

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
        if i < j:
            i, j = j, i  # Ensure i >= j for consistent lookup

        # Search in CSR structure
        for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
            if self.col_indices[k] == j:
                return float(self.lower_bounds[k]), float(self.upper_bounds[k])
        return None

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
        return self.get_bounds(i, j)[0]

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
        return self.get_bounds(i, j)[1]

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
            If coordinates are malformed or ``node`` is out of range.
        """
        if coordinates.size % 3 != 0:
            raise ValueError("Coordinates array size must be multiple of 3")

        num_coords = coordinates.size // 3
        if num_coords < self.num_nodes:
            raise ValueError(
                f"Need coordinates for {self.num_nodes} nodes, got {num_coords}"
            )

        if node is not None:
            return self._check_node_feasibility(coordinates, node)
        else:
            return self._check_all_feasibility(coordinates)

    def _check_node_feasibility(
        self, coordinates: NDArray[np.float64], node: int
    ) -> bool:
        """Check feasibility for a specific node."""
        if node >= self.num_nodes:
            raise ValueError(f"Invalid node index {node}, max is {self.num_nodes - 1}")

        for k in range(self.row_ptr[node], self.row_ptr[node + 1]):
            j = self.col_indices[k]
            if j > node:  # Avoid checking same constraint twice
                break

            pos_i = coordinates[node]
            pos_j = coordinates[j]
            distance = float(np.linalg.norm(pos_i - pos_j))

            lower, upper = self.lower_bounds[k], self.upper_bounds[k]
            if (
                distance < lower - self.distance_tolerance
                or distance > upper + self.distance_tolerance
            ):
                return False

        return True

    def _check_all_feasibility(self, coordinates: NDArray[np.float64]) -> bool:
        """Check feasibility for all nodes."""
        for i in range(self.num_nodes):
            if not self._check_node_feasibility(coordinates, i):
                return False
        return True

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
        for i in range(self.num_nodes):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = self.col_indices[k]
                if j > i:  # Avoid checking same constraint twice
                    break

                pos_i = coordinates[i]
                pos_j = coordinates[j]
                distance = float(np.linalg.norm(pos_i - pos_j))

                lower, upper = self.lower_bounds[k], self.upper_bounds[k]
                err_lower = lower - distance
                err_upper = distance - upper

                if (
                    err_lower > self.distance_tolerance
                    or err_upper > self.distance_tolerance
                ):
                    raise RuntimeError(
                        f"Distance constraint violation between nodes {i} and {j}: "
                        f"distance={distance:.6f}, bounds=[{lower:.6f}, {upper:.6f}], "
                        f"tolerance={self.distance_tolerance:.6f}, "
                        f"err_lower={err_lower:.6f}, err_upper={err_upper:.6f}"
                    )
