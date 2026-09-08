"""
Matrix/array adapter for SBBU algorithm.
"""

import numpy as np
from numpy.typing import NDArray

from ..constraints import ConstraintSet, DistanceConstraint


class MatrixAdapter:
    """
    Adapt matrix-like inputs to SBBU constraint representations.

    Notes
    -----
    This class provides stateless conversion helpers for dense/sparse distance
    matrices, edge lists, and adjacency-backed distance data.
    """

    @staticmethod
    def from_distance_matrix(
        distance_matrix: NDArray[np.float64],
        ignore_nan: bool = True,
        ignore_inf: bool = True,
        ignore_zero: bool = True,
    ) -> ConstraintSet:
        """
        Create a constraint set from a square distance matrix.

        Parameters
        ----------
        distance_matrix : NDArray[np.float64]
            Square matrix containing pairwise distances.
        ignore_nan : bool, default=True
            Whether to skip ``NaN`` entries.
        ignore_inf : bool, default=True
            Whether to skip infinite entries.
        ignore_zero : bool, default=True
            Whether to skip non-positive distances. If ``False``, non-positive
            values raise ``ValueError``.

        Returns
        -------
        ConstraintSet
            Constraint set ready for SBBU solving.

        Raises
        ------
        ValueError
            If ``distance_matrix`` is not a valid symmetric square distance
            matrix.
        """
        if not MatrixAdapter.validate_distance_matrix(distance_matrix):
            raise ValueError(
                "Distance matrix must be square, symmetric, and have a zero diagonal"
            )

        return ConstraintSet.from_distance_matrix(
            distance_matrix, ignore_nan, ignore_inf, ignore_zero
        )

    @staticmethod
    def from_edge_list(
        num_nodes: int, edges: list[tuple[int, int, float]]
    ) -> ConstraintSet:
        """
        Create a constraint set from an edge list.

        Parameters
        ----------
        num_nodes : int
            Total number of nodes.
        edges : list[tuple[int, int, float]]
            List of ``(i, j, distance)`` triples.

        Returns
        -------
        ConstraintSet
            Constraint set ready for SBBU solving.

        Raises
        ------
        ValueError
            If edge endpoints or distances are invalid for
            :class:`~sbbu.constraints.DistanceConstraint`.
        """
        return ConstraintSet.from_edge_list(num_nodes, edges)

    @staticmethod
    def from_adjacency_matrix(
        adjacency_matrix: NDArray[np.float64], distance_matrix: NDArray[np.float64]
    ) -> ConstraintSet:
        """
        Create a constraint set from adjacency and distance matrices.

        Parameters
        ----------
        adjacency_matrix : NDArray[np.float64]
            Square connectivity matrix where positive entries indicate edges.
        distance_matrix : NDArray[np.float64]
            Square matrix containing pairwise distances.

        Returns
        -------
        ConstraintSet
            Constraint set containing valid connected pairs.

        Raises
        ------
        ValueError
            If input matrices are not two-dimensional square arrays with equal
            shape, or if a connected pair has an invalid distance value.
        """
        if adjacency_matrix.ndim != 2 or distance_matrix.ndim != 2:
            raise ValueError("Adjacency and distance matrices must be 2D")
        if (
            adjacency_matrix.shape[0] != adjacency_matrix.shape[1]
            or distance_matrix.shape[0] != distance_matrix.shape[1]
        ):
            raise ValueError("Adjacency and distance matrices must be square")
        if adjacency_matrix.shape != distance_matrix.shape:
            raise ValueError("Adjacency and distance matrices must have same shape")
        if not np.array_equal(adjacency_matrix, adjacency_matrix.T):
            raise ValueError("Adjacency matrix must be symmetric")
        if not np.allclose(
            distance_matrix, distance_matrix.T, atol=1e-12, equal_nan=True
        ):
            raise ValueError("Distance matrix must be symmetric")

        num_nodes = adjacency_matrix.shape[0]
        constraints = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adjacency_matrix[i, j] > 0:  # Connected
                    dist = distance_matrix[i, j]
                    if np.isnan(dist) or np.isinf(dist) or dist <= 0:
                        raise ValueError(
                            f"Connected edge ({i}, {j}) has invalid distance {dist}"
                        )
                    constraints.append(DistanceConstraint(i, j, dist))

        return ConstraintSet(num_nodes, constraints)

    @staticmethod
    def to_distance_matrix(
        coordinates: NDArray[np.float64],
        sparse_indices: list[tuple[int, int]] | None = None,
    ) -> NDArray[np.float64]:
        """
        Convert coordinates to a dense or sparse distance matrix.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Coordinate array with shape ``(N, 3)``.
        sparse_indices : list[tuple[int, int]] | None, default=None
            Optional node-index pairs to evaluate. If ``None``, all pairwise
            distances are computed.

        Returns
        -------
        NDArray[np.float64]
            Distance matrix with shape ``(N, N)``.
        """
        num_nodes = coordinates.shape[0]

        if sparse_indices is None:
            # Full matrix
            distance_matrix = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = np.linalg.norm(coordinates[i] - coordinates[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            return distance_matrix
        else:
            # Sparse matrix (only compute specified pairs)
            distance_matrix = np.full((num_nodes, num_nodes), np.nan)
            np.fill_diagonal(distance_matrix, 0.0)  # Distance to self is 0

            for i, j in sparse_indices:
                if i < 0 or i >= num_nodes or j < 0 or j >= num_nodes:
                    raise ValueError(
                        f"sparse_indices pair ({i}, {j}) out of range for {num_nodes} nodes"
                    )
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

            return distance_matrix

    @staticmethod
    def validate_distance_matrix(
        distance_matrix: NDArray[np.float64], tolerance: float = 1e-9
    ) -> bool:
        """
        Validate structural properties of a distance matrix.

        Parameters
        ----------
        distance_matrix : NDArray[np.float64]
            Candidate distance matrix.
        tolerance : float, default=1e-9
            Absolute tolerance used for zero-diagonal and symmetry checks.

        Returns
        -------
        bool
            ``True`` if the matrix is square, has a zero diagonal within
            tolerance, and is symmetric (treating matching ``NaN`` entries as
            valid).
        """
        if distance_matrix.ndim != 2:
            return False

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            return False

        # Check diagonal is zero
        if not np.allclose(np.diag(distance_matrix), 0.0, atol=tolerance):
            return False

        rows, columns = np.triu_indices(distance_matrix.shape[0], k=1)
        # Match scalar isclose promotion across NumPy versions.
        comparison_dtype = np.result_type(distance_matrix.dtype.type(0), 1.0)
        return bool(
            np.all(
                np.isclose(
                    distance_matrix[rows, columns].astype(comparison_dtype, copy=False),
                    distance_matrix[columns, rows].astype(comparison_dtype, copy=False),
                    atol=tolerance,
                    equal_nan=True,
                )
            )
        )
