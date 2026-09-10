"""Canonical distance observations and packed hard-bound evaluation."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DistanceConstraint:
    """Represent a bounded distance between two nodes.

    Parameters
    ----------
    i : int
        Index of the first node. Must be smaller than ``j``.
    j : int
        Index of the second node.
    lower_bound : float
        Strictly positive lower bound, losslessly representable as float64.
    upper_bound : float
        Finite upper bound, at least ``lower_bound``. Equal endpoints denote
        an exact distance; positive-width intervals are never rounded to exact.
    """

    i: int
    j: int
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.i, bool)
            or isinstance(self.j, bool)
            or not isinstance(self.i, int)
            or not isinstance(self.j, int)
        ):
            raise TypeError(
                f"i and j must be integers, got types ({type(self.i)}, {type(self.j)})"
            )
        if self.i >= self.j:
            raise ValueError(f"i ({self.i}) must be less than j ({self.j})")
        if not math.isfinite(self.lower_bound):
            raise ValueError(f"lower_bound must be finite, got {self.lower_bound}")
        if self.lower_bound <= 0:
            raise ValueError(f"lower_bound must be positive, got {self.lower_bound}")
        if not math.isfinite(self.upper_bound):
            raise ValueError(f"upper_bound must be finite, got {self.upper_bound}")
        if self.upper_bound < self.lower_bound:
            raise ValueError(
                f"upper_bound ({self.upper_bound}) must be at least lower_bound ({self.lower_bound})"
            )
        lower, upper = float(self.lower_bound), float(self.upper_bound)
        for source, converted in ((self.lower_bound, lower), (self.upper_bound, upper)):
            exact_source = int(source) if isinstance(source, np.integer) else source
            if exact_source != converted:
                raise ValueError("Bounds must be losslessly representable as float64")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

    @property
    def is_exact(self) -> bool:
        """Return whether the interval has equal endpoints.

        Returns
        -------
        bool
            True only for a point interval, independently of solver tolerance.
        """
        return self.lower_bound == self.upper_bound

    def violation(self, distance: float) -> float:
        """Return the absolute distance outside this interval.

        Parameters
        ----------
        distance : float
            Measured distance between the endpoints.

        Returns
        -------
        float
            Zero inside the interval, otherwise distance to its nearest bound.
            Nonfinite measurements have infinite violation.
        """
        if not math.isfinite(distance):
            return math.inf
        return max(self.lower_bound - distance, distance - self.upper_bound, 0.0)


class ConstraintSet:
    """Own distance observations independently of solver applicability.

    Parameters
    ----------
    num_nodes : int
        Number of nodes, including isolated nodes.
    constraints : Sequence[DistanceConstraint]
        Observations in input order. Intersecting duplicates become one hard
        interval; disjoint duplicates remain soft alternatives.

    Notes
    -----
    Construction snapshots inputs. Normalized hard bounds are stored as owned
    read-only columns in lexicographic pair order. Scalar queries materialize
    projections; mutating a returned list does not change this problem.
    Neither construction nor completeness certifies geometric feasibility.
    """

    def __init__(
        self, num_nodes: int, constraints: Sequence[DistanceConstraint]
    ) -> None:
        self._validate_num_nodes(num_nodes)
        self._num_nodes = num_nodes
        self._observations: tuple[DistanceConstraint, ...] | None = tuple(constraints)
        self._soft: dict[tuple[int, int], tuple[DistanceConstraint, ...]] = {}
        by_pair: dict[tuple[int, int], list[DistanceConstraint]] = {}
        for constraint in self._observations:
            if constraint.i < 0:
                raise ValueError(
                    f"Constraint has negative node index: ({constraint.i}, {constraint.j})"
                )
            if constraint.j >= num_nodes:
                raise ValueError(
                    f"Constraint references node {constraint.j} but num_nodes is {num_nodes}"
                )
            by_pair.setdefault((constraint.i, constraint.j), []).append(constraint)

        hard = []
        for pair, observations in sorted(by_pair.items()):
            lower = max(c.lower_bound for c in observations)
            upper = min(c.upper_bound for c in observations)
            if lower > upper:
                self._soft[pair] = tuple(observations)
            else:
                hard.append(DistanceConstraint(*pair, lower, upper))
        count = len(hard)
        self._set_rows(
            np.fromiter((c.i for c in hard), dtype=np.intp, count=count),
            np.fromiter((c.j for c in hard), dtype=np.intp, count=count),
            np.fromiter((c.lower_bound for c in hard), dtype=np.float64, count=count),
            np.fromiter((c.upper_bound for c in hard), dtype=np.float64, count=count),
        )

    @staticmethod
    def _validate_num_nodes(num_nodes: int) -> None:
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be an integer")
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")

    @staticmethod
    def _copy_bound_column(values: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = np.array(values, dtype=np.float64, copy=True)
                if (
                    values.dtype.kind in "iu" or values.dtype.itemsize > 8
                ) and not np.array_equal(result.astype(values.dtype), values):
                    raise ValueError(
                        "Bounds must be losslessly representable as float64"
                    )
        except FloatingPointError as error:
            raise ValueError(
                "Bounds must be losslessly representable as float64"
            ) from error
        if not np.isfinite(result).all():
            raise ValueError("Bounds must be finite in float64")
        return result

    def _set_rows(
        self,
        rows: NDArray[np.intp],
        columns: NDArray[np.intp],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> None:
        arrays = (rows, columns, lower, upper)
        if any(a.ndim != 1 or a.shape != rows.shape for a in arrays):
            raise ValueError(
                "Bound columns must be equal-length one-dimensional arrays"
            )
        if rows.dtype.kind not in "iu" or columns.dtype.kind not in "iu":
            raise TypeError("Node columns must contain integers")
        if lower.dtype.kind not in "iuf" or upper.dtype.kind not in "iuf":
            raise TypeError("Bounds must contain real numbers")
        if (
            np.any(rows < 0)
            or np.any(rows >= columns)
            or np.any(columns >= self.num_nodes)
        ):
            raise ValueError("Pairs must satisfy 0 <= i < j < num_nodes")
        if np.any(columns > np.iinfo(np.intp).max):
            raise ValueError("Node indices exceed the platform array index range")
        if np.any(rows[1:] < rows[:-1]) or np.any(
            (rows[1:] == rows[:-1]) & (columns[1:] <= columns[:-1])
        ):
            raise ValueError("Bulk pairs must be sorted and unique")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("Bounds must be finite")
        if np.any(lower <= 0):
            raise ValueError("lower_bound must be positive")
        if np.any(upper < lower):
            raise ValueError("upper_bound must be at least lower_bound")
        self._rows = np.array(rows, dtype=np.intp, copy=True)
        self._columns = np.array(columns, dtype=np.intp, copy=True)
        self._lower = self._copy_bound_column(lower)
        self._upper = self._copy_bound_column(upper)
        for array in (self._rows, self._columns, self._lower, self._upper):
            array.flags.writeable = False

    @classmethod
    def from_bounds_arrays(
        cls,
        num_nodes: int,
        rows: NDArray[np.intp],
        columns: NDArray[np.intp],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> "ConstraintSet":
        """Snapshot sorted unique bounds without constructing scalar observations.

        Parameters
        ----------
        num_nodes : int
            Number of nodes, including isolated nodes.
        rows, columns : NDArray[np.intp]
            One-dimensional endpoints in strictly increasing lexicographic order,
            with ``0 <= rows < columns < num_nodes``.
        lower_bounds, upper_bounds : NDArray[np.float64]
            Same-length finite positive ordered bounds. Numeric inputs must be
            losslessly representable as float64; conversion never narrows bounds.

        Returns
        -------
        ConstraintSet
            Owned read-only bounds with no soft alternatives.

        Raises
        ------
        TypeError
            If the node count, index columns or bound dtypes are invalid.
        ValueError
            If dimensions, indices, ordering, uniqueness or bounds are invalid.

        Notes
        -----
        Use the ordinary constructor for duplicate observations requiring
        intersection or soft-alternative normalization.
        """
        result = cls(num_nodes, [])
        result._set_rows(rows, columns, lower_bounds, upper_bounds)
        result._observations = None
        return result

    @property
    def num_nodes(self) -> int:
        """Return the original number of nodes, including isolated nodes."""
        return self._num_nodes

    @property
    def rows(self) -> NDArray[np.intp]:
        """Return read-only first endpoints of normalized hard pairs."""
        return self._rows.view()

    @property
    def columns(self) -> NDArray[np.intp]:
        """Return read-only second endpoints of normalized hard pairs."""
        return self._columns.view()

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        """Return read-only lower bounds in normalized hard-pair order."""
        return self._lower.view()

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        """Return read-only upper bounds in normalized hard-pair order."""
        return self._upper.view()

    @property
    def constraints(self) -> list[DistanceConstraint]:
        """Return a copy of original observations in their input order."""
        if self._observations is not None:
            return list(self._observations)
        return self.get_hard_constraints()

    @property
    def is_complete(self) -> bool:
        """Return whether hard constraints cover every distinct node pair."""
        return len(self._rows) == self.num_nodes * (self.num_nodes - 1) // 2

    def constraint_errors(
        self, coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate every normalized hard interval in canonical pair order.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Proposed coordinates with shape ``(num_nodes, 3)``.

        Returns
        -------
        NDArray[np.float64]
            Absolute bound violations. Malformed or nonfinite coordinates yield
            infinity, including at unconstrained nodes and for empty edge sets.
        """
        if (
            coordinates.shape != (self.num_nodes, 3)
            or coordinates.dtype.kind not in "iuf"
            or not np.isfinite(coordinates).all()
        ):
            return np.full(max(1, len(self._rows)), np.inf)
        positions = np.asarray(coordinates, dtype=np.float64)
        distances = np.linalg.norm(
            positions[self._rows] - positions[self._columns], axis=1
        )
        errors = np.maximum(
            np.maximum(self._lower - distances, distances - self._upper), 0.0
        )
        return np.where(np.isfinite(distances), errors, np.inf)

    def maximum_violation(self, coordinates: NDArray[np.float64]) -> float:
        """Return maximum original-hard-bound error, or infinity for invalid coordinates.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Candidate in original node order, with shape ``(num_nodes, 3)``.

        Returns
        -------
        float
            Maximum violation; zero for a valid candidate with no hard edges.
        """
        return float(np.max(self.constraint_errors(coordinates), initial=0.0))

    def bounds_matrices(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project complete hard bounds into two new symmetric matrices.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Lower and upper ``(num_nodes, num_nodes)`` bounds with zero diagonals.

        Raises
        ------
        ValueError
            If a hard pair is missing; no distance is invented for missing edges.
        """
        if not self.is_complete:
            raise ValueError("A bounds-matrix projection requires all hard pairs")
        lower = np.zeros((self.num_nodes, self.num_nodes))
        upper = np.zeros_like(lower)
        lower[self._rows, self._columns] = lower[self._columns, self._rows] = (
            self._lower
        )
        upper[self._rows, self._columns] = upper[self._columns, self._rows] = (
            self._upper
        )
        return lower, upper

    def _constraint_at(self, index: int) -> DistanceConstraint:
        return DistanceConstraint(
            int(self._rows[index]),
            int(self._columns[index]),
            float(self._lower[index]),
            float(self._upper[index]),
        )

    def get_hard_constraints(self) -> list[DistanceConstraint]:
        """Return normalized hard constraints in pair order.

        Returns
        -------
        list[DistanceConstraint]
            Scalar projections of the canonical hard bounds.
        """
        return [self._constraint_at(i) for i in range(len(self._rows))]

    def get_complete_distance_matrix(self) -> NDArray[np.float64] | None:
        """Return complete exact hard distances for at least four nodes.

        Returns
        -------
        NDArray[np.float64] | None
            A new symmetric original-order matrix, or None for fewer than four
            nodes, missing hard pairs or any positive-width interval.
        """
        if (
            self.num_nodes < 4
            or not self.is_complete
            or np.any(self._lower != self._upper)
        ):
            return None
        distances = np.zeros((self.num_nodes, self.num_nodes))
        distances[self._rows, self._columns] = distances[self._columns, self._rows] = (
            self._lower
        )
        return distances

    def get_sequential_constraints(self) -> list[DistanceConstraint]:
        """Return hard pairs with index separation at most three.

        Returns
        -------
        list[DistanceConstraint]
            Pair-ordered scalar constraints, without an exactness guarantee.
        """
        return [
            self._constraint_at(int(i))
            for i in np.flatnonzero(self._columns - self._rows <= 3)
        ]

    def get_long_range_constraints(self) -> list[DistanceConstraint]:
        """Return hard pairs with separation greater than three.

        Returns
        -------
        list[DistanceConstraint]
            Scalar constraints sorted by ``(j, j - i)``.
        """
        indices = np.flatnonzero(self._columns - self._rows > 3)
        order = np.lexsort(
            (self._columns[indices] - self._rows[indices], self._columns[indices])
        )
        return [self._constraint_at(int(i)) for i in indices[order]]

    def get_soft_ambiguous_constraints(
        self,
    ) -> list[tuple[tuple[int, int], list[DistanceConstraint]]]:
        """Return copies of disjoint observation groups.

        Returns
        -------
        list[tuple[tuple[int, int], list[DistanceConstraint]]]
            Sorted pairs and their original, input-ordered alternatives.
        """
        return [(pair, list(values)) for pair, values in sorted(self._soft.items())]

    def get_constraint(self, i: int, j: int) -> DistanceConstraint | None:
        """Look up a normalized hard interval in either endpoint order.

        Parameters
        ----------
        i, j : int
            Node indices.

        Returns
        -------
        DistanceConstraint | None
            The hard constraint, or None for a missing or ambiguous pair.
        """
        if i > j:
            i, j = j, i
        start = int(np.searchsorted(self._rows, i, side="left"))
        end = int(np.searchsorted(self._rows, i, side="right"))
        index = start + int(np.searchsorted(self._columns[start:end], j))
        if index == end or self._columns[index] != j:
            return None
        return self._constraint_at(index)

    def has_constraint(self, i: int, j: int) -> bool:
        """Return whether a normalized hard pair exists.

        Parameters
        ----------
        i, j : int
            Node indices in either order.

        Returns
        -------
        bool
            Whether the pair has a hard constraint.
        """
        return self.get_constraint(i, j) is not None
