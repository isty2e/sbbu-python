"""
Matrix/array adapter for SBBU algorithm.
"""

from collections.abc import Sequence

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
            Constraint set without a guarantee of solver applicability.

        Raises
        ------
        ValueError
            If the matrix is not real, square, symmetric or zero-diagonal,
            or an unskipped distance is nonfinite or nonpositive.
        """
        if not MatrixAdapter.validate_distance_matrix(distance_matrix):
            raise ValueError(
                "Distance matrix must be square, symmetric, and have a zero diagonal"
            )

        num_nodes = distance_matrix.shape[0]
        rows, columns = np.triu_indices(num_nodes, k=1)
        distances = distance_matrix[rows, columns]
        omitted = np.zeros(len(rows), dtype=bool)
        if ignore_nan:
            omitted |= np.isnan(distances)
        if ignore_inf:
            omitted |= np.isinf(distances)
        if ignore_zero:
            omitted |= np.isfinite(distances) & (distances <= 0)
        present = ~omitted
        return ConstraintSet.from_bounds_arrays(
            num_nodes,
            rows[present],
            columns[present],
            distances[present],
            distances[present],
        )

    @staticmethod
    def from_bounds_matrices(
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        *,
        ignore_nan: bool = True,
    ) -> ConstraintSet:
        """Create constraints from paired finite distance bounds.

        Parameters
        ----------
        lower_bounds : NDArray[np.float64]
            Symmetric square matrix of positive lower bounds, with zero diagonal.
        upper_bounds : NDArray[np.float64]
            Same-shaped symmetric upper bounds, with zero diagonal. Each upper
            bound must be at least its lower bound. Equality denotes exact input.
        ignore_nan : bool, default=True
            Omit a pair only when both bounds are NaN in both directions.

        Returns
        -------
        ConstraintSet
            Canonical intervals, without a guarantee of solver applicability.

        Raises
        ------
        ValueError
            If shapes, symmetry, diagonals, missingness or bounds are invalid.
            Infinity and one-sided bounds are unsupported.

        Notes
        -----
        Symmetry and bound ordering are checked without tolerance. Positive-width
        intervals are preserved, even when narrower than solver tolerance.
        """
        for bounds in (lower_bounds, upper_bounds):
            if bounds.ndim != 2 or bounds.shape[0] != bounds.shape[1]:
                raise ValueError("Bounds matrices must be square and two-dimensional")
            if bounds.dtype.kind not in "iuf":
                raise ValueError("Bounds matrices must contain real numeric values")
            if not np.all(np.diag(bounds) == 0):
                raise ValueError("Bounds matrices must have zero diagonals")
            if not np.array_equal(bounds, bounds.T, equal_nan=True):
                raise ValueError("Bounds matrices must be symmetric")
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError("Bounds matrices must have the same shape")

        missing = np.isnan(lower_bounds)
        if not np.array_equal(missing, np.isnan(upper_bounds)):
            raise ValueError("Missing lower and upper bounds must match")
        if not ignore_nan and np.any(missing):
            raise ValueError("Bounds must be finite when ignore_nan=False")
        if np.any(np.isinf(lower_bounds)) or np.any(np.isinf(upper_bounds)):
            raise ValueError("Bounds must be finite; one-sided bounds are unsupported")

        num_nodes = lower_bounds.shape[0]
        rows, columns = np.triu_indices(num_nodes, k=1)
        present = ~missing[rows, columns]
        rows, columns = rows[present], columns[present]
        return ConstraintSet.from_bounds_arrays(
            num_nodes,
            rows,
            columns,
            lower_bounds[rows, columns],
            upper_bounds[rows, columns],
        )

    @staticmethod
    def from_edge_list(
        num_nodes: int, edges: Sequence[tuple[int, int, float, float]]
    ) -> ConstraintSet:
        """Create constraints from four-field edge records.

        Parameters
        ----------
        num_nodes : int
            Total number of nodes, including isolated nodes.
        edges : Sequence[tuple[int, int, float, float]]
            Records ``(i, j, lower_bound, upper_bound)``. Equal bounds denote
            exact distances. Either endpoint order is accepted.

        Returns
        -------
        ConstraintSet
            Canonical intervals, without a guarantee of solver applicability.

        Raises
        ------
        ValueError
            If a record has the wrong length, endpoints or bounds are invalid.
        TypeError
            If node indices are not integers.
        """
        constraints = []
        for i, j, lower, upper in edges:
            if i > j:
                i, j = j, i
            constraints.append(DistanceConstraint(i, j, lower, upper))
        return ConstraintSet(num_nodes, constraints)

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
        rows, columns = np.triu_indices(num_nodes, k=1)
        connected = adjacency_matrix[rows, columns] > 0
        rows, columns = rows[connected], columns[connected]
        distances = distance_matrix[rows, columns]
        if distances.dtype.kind not in "iuf":
            raise ValueError("Connected distances must be real numbers")
        invalid = ~np.isfinite(distances) | (distances <= 0)
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"Connected edge ({rows[index]}, {columns[index]}) "
                f"has invalid distance {distances[index]}"
            )
        return ConstraintSet.from_bounds_arrays(
            num_nodes, rows, columns, distances, distances
        )

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
        if distance_matrix.ndim != 2 or distance_matrix.dtype.kind not in "iuf":
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
