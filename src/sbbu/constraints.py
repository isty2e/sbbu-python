"""
Distance constraint representations for SBBU algorithm.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


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
    upper_bound : float
        Finite upper bound, at least ``lower_bound``. Equal endpoints denote
        an exact distance; positive-width intervals are never rounded to exact.
    """

    i: int  # First node index
    j: int  # Second node index
    lower_bound: float  # Minimum distance
    upper_bound: float  # Maximum distance

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

        object.__setattr__(self, "lower_bound", float(self.lower_bound))
        object.__setattr__(self, "upper_bound", float(self.upper_bound))

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


@dataclass
class ConstraintSet:
    """
    Store distance constraints independently of solver applicability.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the problem.
    constraints : list[DistanceConstraint]
        Observations grouped by node pair. Intersecting duplicates become one
        hard interval; disjoint duplicates remain soft alternatives. Construction
        checks neither discretization requirements nor global feasibility.
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

    def __post_init__(self) -> None:
        if isinstance(self.num_nodes, bool) or not isinstance(self.num_nodes, int):
            raise TypeError("num_nodes must be an integer")
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

        for pair, pair_constraints in sorted(constraints_by_pair.items()):
            if len(pair_constraints) == 1:
                constraint = pair_constraints[0]
            else:
                lower = max(c.lower_bound for c in pair_constraints)
                upper = min(c.upper_bound for c in pair_constraints)
                if lower > upper:
                    self._soft_ambiguous_constraints[pair] = pair_constraints
                    continue
                constraint = DistanceConstraint(*pair, lower, upper)

            self._hard_constraints.append(constraint)
            self._constraint_lookup[pair] = constraint

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
        if any(not c.is_exact for c in self._hard_constraints):
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
        Return soft-ambiguous constraint groups.

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
