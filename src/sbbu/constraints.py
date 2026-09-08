"""
Distance constraint representations for SBBU algorithm.
"""

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

try:
    from typing import Self
except ImportError:  # pragma: no cover - Python < 3.11
    from typing_extensions import Self


@dataclass(frozen=True)
class DistanceConstraint:
    """
    Represent a bounded distance between two nodes.

    Parameters
    ----------
    i : int
        Index of the first node. Must be smaller than ``j``.
    j : int
        Index of the second node.
    lower_bound : float
        Strictly positive lower bound of the distance.
    upper_bound : float | None, default=None
        Optional upper bound. ``None`` denotes an exact distance equal to
        ``lower_bound``.
    """

    i: int  # First node index
    j: int  # Second node index
    lower_bound: float  # Minimum distance
    upper_bound: float | None = None  # Maximum distance (None = no upper bound)

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
        if self.upper_bound is not None and not math.isfinite(self.upper_bound):
            raise ValueError(f"upper_bound must be finite, got {self.upper_bound}")
        if self.upper_bound is not None and self.upper_bound <= self.lower_bound:
            raise ValueError(
                f"upper_bound ({self.upper_bound}) must be greater than lower_bound ({self.lower_bound})"
            )

        object.__setattr__(self, "lower_bound", float(self.lower_bound))
        if self.upper_bound is not None:
            object.__setattr__(self, "upper_bound", float(self.upper_bound))

    @property
    def distance(self) -> float:
        """
        Return the primary distance value used by SBBU.

        Returns
        -------
        float
            The lower bound of this constraint.
        """
        return self.lower_bound


@dataclass
class ConstraintSet:
    """
    Store and normalize distance constraints for SBBU solving.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the problem.
    constraints : list[DistanceConstraint]
        Input constraints to normalize into hard and soft-ambiguous groups.
    """

    num_nodes: int
    constraints: list[DistanceConstraint]
    _constraint_lookup: dict[tuple[int, int], DistanceConstraint] = field(
        init=False, repr=False
    )
    _hard_constraints: list[DistanceConstraint] = field(init=False, repr=False)
    _soft_ambiguous_constraints: dict[tuple[int, int], list[DistanceConstraint]] = (
        field(init=False, repr=False)
    )

    _SEQUENTIAL_EXACT_TOLERANCE: ClassVar[float] = 1e-10
    _INTERSECTION_TOLERANCE: ClassVar[float] = 1e-12

    def __post_init__(self):
        # Validate constraints
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {self.num_nodes}")

        self._constraint_lookup = {}
        self._hard_constraints = []
        self._soft_ambiguous_constraints = {}
        max_node = -1
        constraints_by_pair: dict[tuple[int, int], list[DistanceConstraint]] = {}

        for constraint in self.constraints:
            if constraint.i < 0 or constraint.j < 0:
                raise ValueError(
                    f"Constraint has negative node index: ({constraint.i}, {constraint.j})"
                )

            max_node = max(max_node, constraint.i, constraint.j)
            key = (constraint.i, constraint.j)
            if key not in constraints_by_pair:
                constraints_by_pair[key] = []
            constraints_by_pair[key].append(constraint)

        if max_node >= self.num_nodes:
            raise ValueError(
                f"Constraint references node {max_node} but num_nodes is {self.num_nodes}"
            )

        for (i, j), pair_constraints in sorted(constraints_by_pair.items()):
            if len(pair_constraints) == 1 and pair_constraints[0].upper_bound is None:
                constraint = pair_constraints[0]
                self._hard_constraints.append(constraint)
                self._constraint_lookup[(i, j)] = constraint
                continue

            if j - i <= 3:
                sequential_constraint = self._build_sequential_constraint(
                    i, j, pair_constraints
                )
                self._hard_constraints.append(sequential_constraint)
                self._constraint_lookup[(i, j)] = sequential_constraint
                continue

            long_range_constraint = self._build_long_range_constraint(pair_constraints)
            if long_range_constraint is not None:
                self._hard_constraints.append(long_range_constraint)
                self._constraint_lookup[(i, j)] = long_range_constraint
            else:
                self._soft_ambiguous_constraints[(i, j)] = pair_constraints

    @classmethod
    def _normalized_upper_bound(cls, constraint: DistanceConstraint) -> float:
        """Return the upper bound, interpreting None as exact lower bound."""
        if constraint.upper_bound is None:
            return constraint.lower_bound
        return constraint.upper_bound

    @classmethod
    def _build_sequential_constraint(
        cls, i: int, j: int, pair_constraints: list[DistanceConstraint]
    ) -> DistanceConstraint:
        """
        Build the unique sequential constraint.

        Sequential constraints are required by trilateration and must be exact and
        mutually consistent across duplicate entries.
        """
        reference = pair_constraints[0]
        reference_upper = cls._normalized_upper_bound(reference)
        if (
            abs(reference_upper - reference.lower_bound)
            > cls._SEQUENTIAL_EXACT_TOLERANCE
        ):
            raise ValueError(
                f"Sequential constraint ({i}, {j}) must be exact, got "
                f"[{reference.lower_bound}, {reference_upper}]"
            )

        exact_distance = reference.lower_bound
        for constraint in pair_constraints[1:]:
            upper = cls._normalized_upper_bound(constraint)
            if abs(upper - constraint.lower_bound) > cls._SEQUENTIAL_EXACT_TOLERANCE:
                raise ValueError(
                    f"Sequential constraint ({i}, {j}) must be exact, got "
                    f"[{constraint.lower_bound}, {upper}]"
                )
            if (
                abs(constraint.lower_bound - exact_distance)
                > cls._SEQUENTIAL_EXACT_TOLERANCE
            ):
                raise ValueError(
                    f"Conflicting sequential constraints for ({i}, {j}): "
                    f"{exact_distance} vs {constraint.lower_bound}"
                )

        return DistanceConstraint(i, j, exact_distance)

    @classmethod
    def _build_long_range_constraint(
        cls, pair_constraints: list[DistanceConstraint]
    ) -> DistanceConstraint | None:
        """
        Build a merged long-range hard constraint.

        If duplicates have empty intersection, the pair is treated as soft-ambiguous
        and no hard constraint is returned.
        """
        i = pair_constraints[0].i
        j = pair_constraints[0].j

        merged_lower = max(constraint.lower_bound for constraint in pair_constraints)
        merged_upper = min(
            cls._normalized_upper_bound(constraint) for constraint in pair_constraints
        )

        if merged_lower > merged_upper + cls._INTERSECTION_TOLERANCE:
            return None

        if abs(merged_upper - merged_lower) <= cls._INTERSECTION_TOLERANCE:
            return DistanceConstraint(i, j, merged_lower)

        return DistanceConstraint(i, j, merged_lower, upper_bound=merged_upper)

    def get_hard_constraints(self) -> list[DistanceConstraint]:
        """
        Return normalized hard constraints.

        Returns
        -------
        list[DistanceConstraint]
            Hard constraints produced after duplicate/conflict normalization.
        """
        return list(self._hard_constraints)

    def get_complete_distance_matrix(self) -> NDArray[np.float64] | None:
        """Return exact hard distances only when they cover every node pair.

        Returns
        -------
        NDArray[np.float64] | None
            A new symmetric ``(num_nodes, num_nodes)`` matrix in original node
            order, or ``None`` for fewer than four nodes, missing hard pairs,
            or interval constraints. Soft alternatives are never promoted.
        """
        if self.num_nodes < 4 or len(self._hard_constraints) != (
            self.num_nodes * (self.num_nodes - 1) // 2
        ):
            return None
        if any(c.upper_bound is not None for c in self._hard_constraints):
            return None

        distances = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for constraint in self._hard_constraints:
            distances[constraint.i, constraint.j] = constraint.lower_bound
            distances[constraint.j, constraint.i] = constraint.lower_bound
        return distances

    def get_sequential_constraints(self) -> list[DistanceConstraint]:
        """
        Return sequential hard constraints.

        Returns
        -------
        list[DistanceConstraint]
            Constraints where ``j - i <= 3``.
        """
        sequential = []
        for c in self._hard_constraints:
            if c.j - c.i <= 3:  # Sequential connectivity
                sequential.append(c)
        return sequential

    def get_long_range_constraints(self) -> list[DistanceConstraint]:
        """
        Return long-range hard constraints.

        Returns
        -------
        list[DistanceConstraint]
            Constraints where ``j - i > 3``, sorted by ``(j, j - i)`` for
            stable search order.
        """
        long_range = []
        for c in self._hard_constraints:
            if c.j - c.i > 3:  # Long-range
                long_range.append(c)
        long_range.sort(key=lambda c: (c.j, c.j - c.i))
        return long_range

    def get_soft_ambiguous_constraints(
        self,
    ) -> list[tuple[tuple[int, int], list[DistanceConstraint]]]:
        """
        Return soft-ambiguous long-range constraint groups.

        Returns
        -------
        list[tuple[tuple[int, int], list[DistanceConstraint]]]
            Sorted ``((i, j), constraints)`` pairs for duplicates whose
            interval intersection is empty.
        """
        return [
            (pair, list(constraints))
            for pair, constraints in sorted(self._soft_ambiguous_constraints.items())
        ]

    def get_constraint(self, i: int, j: int) -> DistanceConstraint | None:
        """
        Get a normalized hard constraint for a node pair.

        Parameters
        ----------
        i : int
            First node index.
        j : int
            Second node index.

        Returns
        -------
        DistanceConstraint | None
            The normalized hard constraint for the pair, or ``None`` if absent.
        """
        if i > j:
            i, j = j, i  # Ensure i < j

        return self._constraint_lookup.get((i, j))

    def has_constraint(self, i: int, j: int) -> bool:
        """
        Check whether a normalized hard constraint exists for a pair.

        Parameters
        ----------
        i : int
            First node index.
        j : int
            Second node index.

        Returns
        -------
        bool
            ``True`` if a hard constraint exists for the pair, else ``False``.
        """
        return self.get_constraint(i, j) is not None

    @classmethod
    def from_edge_list(
        cls, num_nodes: int, edges: list[tuple[int, int, float]]
    ) -> Self:
        """
        Build a constraint set from edge tuples.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the problem.
        edges : list[tuple[int, int, float]]
            Edge list in ``(i, j, distance)`` format.

        Returns
        -------
        ConstraintSet
            New constraint set built from the provided edges.

        Raises
        ------
        ValueError
            If node indices or distances are invalid.
        """
        constraints = []
        for i, j, dist in edges:
            if i > j:
                i, j = j, i  # Ensure i < j
            constraints.append(DistanceConstraint(i, j, dist))
        return cls(num_nodes, constraints)

    @classmethod
    def from_distance_matrix(
        cls,
        distance_matrix: NDArray[np.float64],
        ignore_nan: bool = True,
        ignore_inf: bool = True,
        ignore_zero: bool = True,
    ) -> Self:
        """
        Build a constraint set from a square distance matrix.

        Parameters
        ----------
        distance_matrix : NDArray[np.float64]
            Pairwise distance matrix of shape ``(N, N)``.
        ignore_nan : bool, default=True
            Whether to skip ``NaN`` entries.
        ignore_inf : bool, default=True
            Whether to skip infinite entries.
        ignore_zero : bool, default=True
            Whether to skip non-positive distance values.

        Returns
        -------
        ConstraintSet
            New constraint set built from valid matrix entries.

        Raises
        ------
        ValueError
            If ``distance_matrix`` is not two-dimensional and square.
        """
        if distance_matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {distance_matrix.ndim}D")
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(
                f"Expected square matrix, got shape {distance_matrix.shape}"
            )

        num_nodes = distance_matrix.shape[0]
        constraints = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Only upper triangle
                dist = distance_matrix[i, j]
                is_required_sequential = j - i <= 3

                if np.isnan(dist):
                    if ignore_nan:
                        if is_required_sequential:
                            raise ValueError(
                                "Required sequential distance "
                                f"({i}, {j}) is missing (NaN)"
                            )
                        continue
                    raise ValueError(
                        f"Distance at ({i}, {j}) must be finite, got {dist}"
                    )

                if np.isinf(dist):
                    if ignore_inf:
                        if is_required_sequential:
                            raise ValueError(
                                "Required sequential distance "
                                f"({i}, {j}) is missing (infinite)"
                            )
                        continue
                    raise ValueError(
                        f"Distance at ({i}, {j}) must be finite, got {dist}"
                    )

                if dist <= 0:
                    if ignore_zero:
                        if is_required_sequential:
                            raise ValueError(
                                "Required sequential distance "
                                f"({i}, {j}) must be positive, got {dist}"
                            )
                        continue
                    raise ValueError(
                        f"Non-positive distance at ({i}, {j}) with ignore_zero=False: {dist}"
                    )

                constraints.append(DistanceConstraint(i, j, dist))

        return cls(num_nodes, constraints)
