"""
Core SBBU algorithm implementation (NMR-agnostic).

This module provides the pure SBBU algorithm that works with distance constraints,
independent of input format (NMR, PDB, matrices, etc.).
"""

import logging
import math
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ..constraints import ConstraintSet, DistanceConstraint
from ..geometry import normalize_vector, quadratic_solver_point
from ..timer import get_wall_time
from ..validation import validate_positive_float
from .cluster import ReflectionCluster, UnionFind
from .metrics import MetricsCalculator
from .refiner import SoftPruningRefiner
from .types import (
    ProblemState,
    RefinementContext,
    SBBUConfig,
    SBBUStats,
    SBBUSolveInfeasibleError,
    SBBUTimeoutError,
)

logger = logging.getLogger(__name__)


class SBBUSolver:
    """
    Branch-and-bound solver for the SBBU distance-geometry formulation.

    Parameters
    ----------
    constraints : ConstraintSet
        Canonical distance constraints to satisfy.
    config : SBBUConfig | None, optional
        Solver configuration. When ``None``, defaults from ``SBBUConfig`` are
        used.
    """

    def __init__(self, constraints: ConstraintSet, config: SBBUConfig | None = None):
        """Initialize the SBBU branch-and-bound solver.

        Parameters
        ----------
        constraints : ConstraintSet
            Distance constraints to satisfy.
        config : SBBUConfig | None, optional
            Solver configuration. When ``None``, defaults from ``SBBUConfig`` are
            used.
        """
        self.constraints = constraints
        self.config = config or SBBUConfig()
        if self.constraints.num_nodes < 3:
            raise ValueError(
                "SBBUSolver requires at least 3 nodes for geometric initialization"
            )
        self._metrics_calculator = MetricsCalculator()
        self._soft_pruning_refiner = SoftPruningRefiner()
        self._validate_required_sequential_constraints()

        # Get long-range constraints for processing
        self.long_range_constraints = constraints.get_long_range_constraints()
        self.soft_ambiguous_constraints = constraints.get_soft_ambiguous_constraints()

        # Runtime limits are configured by solve()
        self._solve_deadline: float | None = None
        self._solve_time_limit: float | None = None
        self._has_solution = False
        self._last_coordinates: NDArray[np.float64] | None = None
        self._last_stats: SBBUStats | None = None

        # Runtime state is initialized lazily in solve().
        self._state: ProblemState | None = None

    def _validate_required_sequential_constraints(self) -> None:
        """Validate mandatory exact sequential constraints for DDGP trilateration."""
        num_nodes = self.constraints.num_nodes
        for j in range(num_nodes):
            for offset in (1, 2, 3):
                i = j - offset
                if i < 0:
                    continue

                constraint = self.constraints.get_constraint(i, j)
                if constraint is None:
                    raise ValueError(
                        f"Missing required sequential constraint ({i}, {j}) "
                        "for DDGP trilateration"
                    )

                upper = (
                    constraint.upper_bound
                    if constraint.upper_bound is not None
                    else constraint.lower_bound
                )
                if abs(upper - constraint.lower_bound) > self.config.distance_tolerance:
                    raise ValueError(
                        f"Sequential constraint ({i}, {j}) must be exact; "
                        f"got bounds [{constraint.lower_bound}, {upper}]"
                    )

    def _state_required(self) -> ProblemState:
        """Return active runtime state during solve-time internal execution."""
        if self._state is None:
            raise RuntimeError("Runtime state unavailable. Call solve() first.")
        return self._state

    def _reset_runtime_state(self) -> None:
        """Initialize mutable runtime state for one solve pass."""
        num_nodes = self.constraints.num_nodes

        # Coordinate matrix and trilateration cursor.
        coordinates = np.zeros((num_nodes, 3), dtype=np.float64)
        current_node = 0

        # Fresh soft-ambiguous candidate lists for this run.
        active_soft_ambiguous_constraints = [
            (pair, list(candidates))
            for pair, candidates in self.soft_ambiguous_constraints
        ]

        # Fresh Union-Find and reflection clusters.
        union_find = UnionFind(num_nodes)
        clusters = [ReflectionCluster(num_nodes, coordinates) for _ in range(num_nodes)]
        for i in range(num_nodes):
            clusters[i].start_node = i
            clusters[i].end_node = i

        # Working arrays for branch-and-bound bookkeeping.
        cluster_nodes = [0] * (num_nodes * 2)
        reflection_flags = [False] * (num_nodes * 2)
        best_reflection_flags = [False] * (num_nodes * 2)
        num_cluster_nodes = 0

        # Per-run statistics.
        stats = SBBUStats()

        self._state = ProblemState(
            coordinates=coordinates,
            current_node=current_node,
            active_soft_ambiguous_constraints=active_soft_ambiguous_constraints,
            union_find=union_find,
            clusters=clusters,
            cluster_nodes=cluster_nodes,
            reflection_flags=reflection_flags,
            best_reflection_flags=best_reflection_flags,
            num_cluster_nodes=num_cluster_nodes,
            stats=stats,
        )

    def _initialize_first_three_nodes(self) -> None:
        """Initialize the first three nodes using exact distance constraints."""
        state = self._state_required()

        # Node 0 at origin
        state.coordinates[0] = [0.0, 0.0, 0.0]

        # Node 1 on x-axis
        d01 = self._get_exact_distance(0, 1)
        state.coordinates[1] = [d01, 0.0, 0.0]

        # Node 2 in xy-plane
        d02 = self._get_exact_distance(0, 2)
        d12 = self._get_exact_distance(1, 2)

        # Use cosine rule to find coordinates
        x2 = (d02 * d02 - d12 * d12 + d01 * d01) / (2.0 * d01)
        y2_squared = d02 * d02 - x2 * x2
        if y2_squared < -self.config.distance_tolerance:
            raise ValueError("First three constraints are geometrically infeasible")
        y2 = math.sqrt(max(0.0, y2_squared))
        if self.constraints.num_nodes > 3 and y2 <= self.config.distance_tolerance:
            raise ValueError(
                "First three nodes must be non-collinear for 3D trilateration"
            )
        state.coordinates[2] = [x2, y2, 0.0]

        state.current_node = 2

    def _get_exact_distance(self, i: int, j: int) -> float:
        """
        Get an exact distance value for sequential trilateration constraints.

        SBBU's forward placement requires exact distances for (i, i-1), (i, i-2),
        and (i, i-3). Interval bounds are accepted only for long-range constraints.
        """
        constraint = self.constraints.get_constraint(i, j)
        if not constraint:
            raise ValueError(f"No constraint found between nodes {i} and {j}")

        upper = constraint.upper_bound
        if (
            upper is not None
            and abs(upper - constraint.lower_bound) > self.config.distance_tolerance
        ):
            raise ValueError(
                f"Sequential constraint ({i}, {j}) must be exact; "
                f"got bounds [{constraint.lower_bound}, {upper}]"
            )

        return constraint.lower_bound

    def _process_long_range_constraints(self, state: ProblemState) -> None:
        """Process all long-range constraints with periodic time checks."""
        for index, constraint in enumerate(self.long_range_constraints):
            self._check_time_limit()
            self.solve_constraint(constraint)
            state.stats.constraints_processed = index + 1

    def _extend_to_node(self, target_node: int) -> None:
        """Extend the coordinate array to include nodes up to target_node."""
        state = self._state_required()
        if target_node < 0 or target_node >= self.constraints.num_nodes:
            raise ValueError(
                f"target_node {target_node} out of range [0, {self.constraints.num_nodes - 1}]"
            )
        if target_node <= state.current_node:
            return

        for i in range(state.current_node + 1, target_node + 1):
            self._check_time_limit()
            self._calculate_node_position(i)
        state.current_node = target_node

    def _calculate_node_position(self, node: int) -> None:
        """Calculate position of a node using distance constraints."""
        state = self._state_required()

        reference_offsets = (1, 2, 3)
        references = tuple(
            state.coordinates[node - offset] for offset in reference_offsets
        )
        target_distance_squares = tuple(
            self._get_exact_distance(node, node - offset) ** 2
            for offset in reference_offsets
        )

        # Check if the current position already satisfies all local constraints.
        current_pos = state.coordinates[node]
        errors = [
            abs(
                float(np.dot(reference - current_pos, reference - current_pos))
                - target_sq
            )
            for reference, target_sq in zip(
                references, target_distance_squares, strict=True
            )
        ]
        if all(error < self.config.distance_tolerance for error in errors):
            return

        ref_a, ref_b, ref_c = references
        dist_a_sq, dist_b_sq, dist_c_sq = target_distance_squares

        # Calculate plane normal
        u = ref_b - ref_a
        v = ref_c - ref_a
        normal = np.cross(u, v)
        normal = normalize_vector(normal)

        # Solve for point on plane
        plane_point = np.zeros(3, dtype=np.float64)
        quadratic_solver_point(
            plane_point, normal, ref_a, dist_a_sq, ref_b, dist_b_sq, ref_c, dist_c_sq
        )

        # Calculate step along normal
        diff = plane_point - ref_a
        beta = float(np.dot(diff, diff))
        if dist_a_sq + self.config.distance_tolerance < beta:
            raise RuntimeError(f"Cannot calculate position for node {node}")

        alpha = math.sqrt(abs(dist_a_sq - beta))

        # Final position: plane_point + alpha * normal
        final_position = plane_point + alpha * normal
        state.coordinates[node] = final_position

    def solve_constraint(self, constraint: DistanceConstraint) -> None:
        """
        Solve one long-range distance constraint.

        Parameters
        ----------
        constraint : DistanceConstraint
            Long-range constraint to satisfy with cluster reflections.

        Raises
        ------
        RuntimeError
            If branch-and-bound cannot satisfy the constraint within tolerance
            or if solver limits are exceeded.
        ValueError
            If sequential constraints needed during extension are missing or
            inconsistent.
        """
        state = self._state_required()

        # Find cluster roots
        root_i = state.union_find.find(constraint.i + 3)  # +3 for DDGP offset
        root_j = state.union_find.find(constraint.j)

        if root_i == root_j:
            return  # Already in same cluster

        # Extend coordinates to cover the constraint
        self._extend_to_node(constraint.j)

        # Build cluster information
        cluster = self._build_cluster_info(constraint, root_i, root_j)

        # Run branch-and-bound search
        self._branch_and_bound_search(constraint, cluster)

    def _build_cluster_info(
        self, constraint: DistanceConstraint, root_i: int, root_j: int
    ) -> ReflectionCluster:
        """Build cluster information for branch-and-bound."""
        state = self._state_required()
        num_clusters = len(state.clusters)
        if root_i < 0 or root_i >= num_clusters:
            raise ValueError(
                f"root_i {root_i} out of range [0, {num_clusters - 1}]"
            )
        if root_j < 0 or root_j >= num_clusters:
            raise ValueError(
                f"root_j {root_j} out of range [0, {num_clusters - 1}]"
            )

        cr = state.clusters[root_i]  # C++ cr = m_c[r]
        cj = state.clusters[root_j]  # C++ cj = m_c[j]

        # Build decision vector
        # m_n = 0; // number of decisions to be taken
        state.num_cluster_nodes = 0  # C++ m_n

        # cluster_t *ck = &cr;
        k = root_i  # C++ k = r
        ck = state.clusters[k]  # C++ ck = &m_c[k]

        # for (int k = r; ck->m_i <= edge.m_j;)
        visited_roots: set[int] = set()

        while ck.start_node <= constraint.j:  # C++ condition: ck->m_i <= edge.m_j
            self._check_time_limit()

            if k in visited_roots:
                raise RuntimeError(
                    "SBBU Core: inconsistent cluster structure while building decisions"
                )
            visited_roots.add(k)

            # merge_cluster(k, r);
            state.union_find.union(k, root_i)

            # add to decision vector: m_d[m_n] = k;
            if state.num_cluster_nodes >= len(state.cluster_nodes):
                raise RuntimeError(
                    "SBBU Core: decision vector overflow while building clusters"
                )
            state.cluster_nodes[state.num_cluster_nodes] = k
            # reset the flip vector: m_f[m_n] = false;
            state.reflection_flags[state.num_cluster_nodes] = False
            # ++m_n;
            state.num_cluster_nodes += 1

            # last feasible cluster: if (ck->m_j + 1 == m_nnodes) break;
            if ck.end_node + 1 >= self.constraints.num_nodes:
                break

            # k = find_root(ck->m_j + 1);
            k = state.union_find.find(ck.end_node + 1)
            # ck = &m_c[k];
            ck = state.clusters[k]

        # cr.create_planes(cj.m_j, m_d, m_n);
        if state.num_cluster_nodes > 0:
            cr.create_reflection_planes(
                cj.end_node, state.cluster_nodes[: state.num_cluster_nodes]
            )

        return cr

    def _branch_and_bound_search(
        self, constraint: DistanceConstraint, cluster: ReflectionCluster
    ) -> None:
        """Run branch-and-bound search to satisfy constraint."""
        state = self._state_required()

        min_error = float("inf")
        best_found = False
        num_decisions = state.num_cluster_nodes
        evaluated_states = 0

        if num_decisions > 0:
            # Start from the all-False reflection state.
            for idx in range(num_decisions):
                state.reflection_flags[idx] = False
            cluster.reflect_single_node(state.reflection_flags[:num_decisions])

        # Enumerate states without duplicates.
        total_states = 1 << num_decisions if num_decisions > 0 else 1
        max_states = min(total_states, self.config.max_iterations)
        current_state = 0

        while evaluated_states < max_states:
            self._check_time_limit()

            pos_i = state.coordinates[constraint.i]
            pos_j = state.coordinates[constraint.j]
            current_distance = float(np.linalg.norm(pos_i - pos_j))
            error = self._metrics_calculator.constraint_violation(
                constraint, current_distance
            )

            if error < min_error:
                min_error = error
                state.best_reflection_flags[:num_decisions] = state.reflection_flags[
                    :num_decisions
                ]
                best_found = True

                # No need to keep searching once the current state satisfies the bound.
                if error <= self.config.distance_tolerance:
                    evaluated_states += 1
                    break

            evaluated_states += 1
            if evaluated_states >= max_states:
                break

            # Increment binary state and toggle only changed bits.
            next_state = current_state + 1
            changed_mask = current_state ^ next_state
            while changed_mask:
                changed_bit = changed_mask & -changed_mask
                bit_index = changed_bit.bit_length() - 1
                state.reflection_flags[bit_index] = not state.reflection_flags[
                    bit_index
                ]
                changed_mask ^= changed_bit

            current_state = next_state
            if num_decisions > 0:
                cluster.reflect_single_node(state.reflection_flags[:num_decisions])

        self._check_time_limit()

        # Apply best solution found
        if best_found:
            cluster.reflect_all_nodes(
                state.cluster_nodes[:num_decisions],
                state.best_reflection_flags[:num_decisions],
            )

        # Check final constraint satisfaction
        lower, upper = self._metrics_calculator.constraint_bounds(constraint)
        if min_error > self.config.distance_tolerance:
            raise SBBUSolveInfeasibleError(
                f"Constraint ({constraint.i + 1}, {constraint.j + 1}, bounds=[{lower}, {upper}]) could not be solved "
                f"(error={min_error:.6e}, tolerance={self.config.distance_tolerance:.6e})"
            )

        state.stats.iterations_used += max(0, evaluated_states - 1)

    def solve(self, max_time: float | None = None) -> SBBUStats:
        """
        Solve all configured hard constraints and optional refinements.

        Parameters
        ----------
        max_time : float | None, optional
            Override for solve time budget in seconds. When ``None``,
            ``config.max_time`` is used.

        Returns
        -------
        SBBUStats
            Aggregated solve statistics for the completed run.

        Raises
        ------
        RuntimeError
            If constraints cannot be satisfied or the time budget is exceeded.
        ValueError
            If ``max_time`` is not strictly positive, or if required sequential
            constraints are missing/geometrically infeasible.
        """
        if max_time is None:
            time_limit = self.config.max_time
        else:
            validate_positive_float(max_time, "max_time")
            time_limit = float(max_time)

        start_time = get_wall_time()
        self._solve_deadline = start_time + time_limit
        self._solve_time_limit = time_limit
        self._has_solution = False
        self._last_coordinates = None
        self._last_stats = None

        try:
            self._reset_runtime_state()
            state = self._state_required()
            self._initialize_first_three_nodes()

            if self.config.verbose:
                logger.info("SBBU Core: tmax = %s", time_limit)
                logger.info("SBBU Core: dtol = %s", self.config.distance_tolerance)
                logger.info("SBBU Core: imax = %s", self.config.max_iterations)
                logger.info(
                    "SBBU Core: long_range_constraints = %d",
                    len(self.long_range_constraints),
                )
                logger.info(
                    "SBBU Core: soft_ambiguous_constraints = %d",
                    len(self.soft_ambiguous_constraints),
                )
                if self.config.enable_soft_pruning:
                    logger.info(
                        "SBBU Core: soft_pruning = enabled (max_rounds=%d)",
                        self.config.soft_pruning_max_rounds,
                    )

            # Process all long-range constraints
            self._process_long_range_constraints(state)

            # Finalize solution
            self._check_time_limit()
            self._extend_to_node(self.constraints.num_nodes - 1)
            self._soft_pruning_refiner.run(self._build_refinement_context(state))
            self._check_time_limit()
            self._validate_solution()

            # Calculate final statistics
            state.stats.solve_time = get_wall_time() - start_time
            self._metrics_calculator.populate_distance_errors(
                state, self.constraints.get_hard_constraints()
            )
            self._metrics_calculator.populate_soft_ambiguous_violations(state)
            self._check_time_limit()

            if self.config.verbose:
                logger.info(
                    "SBBU Core: solution found after %.6f secs",
                    state.stats.solve_time,
                )
                logger.info(
                    "SBBU Core: MDE = %.6e, LDE = %.6e",
                    state.stats.mean_distance_error,
                    state.stats.largest_distance_error,
                )
                if self.config.enable_soft_pruning:
                    logger.info(
                        "SBBU Core: soft pruning rounds = %d, promoted_pairs = %d, "
                        "pruned_candidates = %d",
                        state.stats.soft_pruning_rounds,
                        state.stats.soft_pruned_constraints,
                        state.stats.soft_pruned_candidates,
                    )
                if state.stats.soft_ambiguous_constraints > 0:
                    logger.info(
                        "SBBU Core: soft violations = %.6e mean, %.6e max over %d "
                        "ambiguous pairs",
                        state.stats.mean_soft_violation,
                        state.stats.max_soft_violation,
                        state.stats.soft_ambiguous_constraints,
                    )

            self._last_coordinates = state.coordinates.copy()
            self._last_stats = replace(state.stats)
            self._has_solution = True
            return self._last_stats

        except Exception:
            if self._state is not None:
                self._state.stats.solve_time = get_wall_time() - start_time
            raise
        finally:
            self._solve_deadline = None
            self._solve_time_limit = None
            self._state = None

    def _check_time_limit(self) -> None:
        """Raise RuntimeError if solve() time budget has been exceeded."""
        if self._solve_deadline is None:
            return

        elapsed = get_wall_time()
        if elapsed >= self._solve_deadline:
            tmax = self._solve_time_limit if self._solve_time_limit is not None else 0.0
            raise SBBUTimeoutError(f"SBBU Core: time exceeded (tmax = {tmax})")

    def _remaining_time_budget(self) -> float:
        """Get remaining solve time budget in seconds."""
        if self._solve_deadline is None:
            return self.config.max_time

        remaining = self._solve_deadline - get_wall_time()
        if remaining <= 0.0:
            self._check_time_limit()
        return max(remaining, 1e-6)

    def _make_refinement_config(self) -> SBBUConfig:
        """Build a non-recursive config for soft-pruning refinement solves."""
        return SBBUConfig(
            distance_tolerance=self.config.distance_tolerance,
            max_iterations=self.config.max_iterations,
            max_time=self.config.max_time,
            verbose=False,
            enable_soft_pruning=False,
            soft_pruning_max_rounds=self.config.soft_pruning_max_rounds,
            soft_pruning_acceptance_tolerance=self.config.soft_pruning_acceptance_tolerance,
            soft_pruning_candidate_window=self.config.soft_pruning_candidate_window,
            soft_pruning_min_margin=self.config.soft_pruning_min_margin,
        )

    def _build_refinement_context(self, state: ProblemState) -> RefinementContext:
        """Build refinement dependencies for one solve run."""
        return RefinementContext(
            state=state,
            config=self.config,
            num_nodes=self.constraints.num_nodes,
            hard_constraints=self.constraints.get_hard_constraints(),
            constraint_violation=self._metrics_calculator.constraint_violation,
            check_time_limit=self._check_time_limit,
            run_refinement_solve=self._run_refinement_solve,
        )

    def _run_refinement_solve(
        self, candidate_hard_constraints: list[DistanceConstraint]
    ) -> tuple[NDArray[np.float64], SBBUStats]:
        """Run one refinement solve and return coordinates with statistics."""
        refinement_solver = type(self)(
            ConstraintSet(self.constraints.num_nodes, candidate_hard_constraints),
            self._make_refinement_config(),
        )
        refinement_stats = refinement_solver.solve(
            max_time=self._remaining_time_budget()
        )
        return refinement_solver.solution_coordinates, refinement_stats

    def _validate_solution(self) -> None:
        """Validate that all constraints are satisfied."""
        state = self._state_required()
        for constraint in self.constraints.get_hard_constraints():
            pos_i = state.coordinates[constraint.i]
            pos_j = state.coordinates[constraint.j]
            actual_distance = float(np.linalg.norm(pos_i - pos_j))
            error = self._metrics_calculator.constraint_violation(
                constraint, actual_distance
            )
            lower, upper = self._metrics_calculator.constraint_bounds(constraint)

            if error > self.config.distance_tolerance:
                raise SBBUSolveInfeasibleError(
                    f"Constraint ({constraint.i}, {constraint.j}) not satisfied: "
                    f"expected in [{lower:.6f}, {upper:.6f}], got {actual_distance:.6f} "
                    f"(error = {error:.6e} > tolerance = {self.config.distance_tolerance:.6e})"
                )

    @property
    def solution_coordinates(self) -> NDArray[np.float64]:
        """
        Return a copy of the current coordinate matrix.

        Returns
        -------
        NDArray[np.float64]
            Coordinate array copy with shape ``(num_nodes, 3)``.

        Raises
        ------
        RuntimeError
            If ``solve()`` has not completed successfully yet.
        """
        if not self._has_solution or self._last_coordinates is None:
            raise RuntimeError("No solved coordinates available. Call solve() first.")
        return self._last_coordinates.copy()

    def get_node_position(self, node: int) -> NDArray[np.float64]:
        """
        Return a copy of one node position.

        Parameters
        ----------
        node : int
            Node index to read.

        Returns
        -------
        NDArray[np.float64]
            Coordinate vector copy for ``node`` with shape ``(3,)``.

        Raises
        ------
        RuntimeError
            If ``solve()`` has not completed successfully yet.
        ValueError
            If ``node`` is outside the valid index range.
        """
        if not self._has_solution or self._last_coordinates is None:
            raise RuntimeError("No solved coordinates available. Call solve() first.")
        if node < 0 or node >= self.constraints.num_nodes:
            raise ValueError(
                f"Node {node} out of range (max: {self.constraints.num_nodes - 1})"
            )
        return self._last_coordinates[node].copy()
