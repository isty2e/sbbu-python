"""
Soft-pruning refinement loop for SBBU.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from ..constraints import DistanceConstraint
from .types import (
    ProblemState,
    RefinementContext,
    SBBUStats,
    SBBUSolveInfeasibleError,
    SBBUTimeoutError,
)

logger = logging.getLogger(__name__)

_AmbiguousGroup = tuple[tuple[int, int], list[DistanceConstraint]]


class SoftPruningRefiner:
    """Run optional soft-pruning refinement passes."""

    def run(self, context: RefinementContext) -> None:
        """
        Iteratively filter soft-ambiguous long-range groups and promote confident pairs.

        Promotion is conservative: a candidate is promoted only when its violation is
        small and clearly better than alternatives in the same ambiguous group.
        """
        state = context.state
        if not context.config.enable_soft_pruning:
            return
        if not state.active_soft_ambiguous_constraints:
            return

        unresolved = [
            (pair, list(candidates))
            for pair, candidates in state.active_soft_ambiguous_constraints
        ]
        hard_constraints = list(context.hard_constraints)
        promoted_constraints: list[DistanceConstraint] = []
        pruned_candidates = 0
        rounds_used = 0

        for _ in range(context.config.soft_pruning_max_rounds):
            if not unresolved:
                break

            context.check_time_limit()
            rounds_used += 1

            (
                next_unresolved,
                newly_promoted,
                promoted_groups,
                round_pruned_candidates,
                any_candidate_pruned,
            ) = self._process_round(context, unresolved)
            pruned_candidates += round_pruned_candidates

            if newly_promoted:
                candidate_hard_constraints = hard_constraints + newly_promoted
                try:
                    coordinates, refinement_stats = context.run_refinement_solve(
                        candidate_hard_constraints
                    )
                except SBBUTimeoutError:
                    raise
                except SBBUSolveInfeasibleError as error:
                    logger.debug(
                        "SBBU Core: refinement round skipped after promoted candidates "
                        "failed to solve (%s)",
                        error,
                    )
                    unresolved = next_unresolved + promoted_groups
                    if not any_candidate_pruned:
                        break
                    continue

                self._apply_refinement_solution(
                    state,
                    coordinates,
                    refinement_stats,
                    context.num_nodes,
                )
                promoted_constraints.extend(newly_promoted)
                hard_constraints = candidate_hard_constraints
                unresolved = next_unresolved
                continue

            unresolved = next_unresolved
            if not any_candidate_pruned:
                break

        self._finalize_refinement(
            state,
            unresolved,
            rounds_used,
            promoted_constraints,
            pruned_candidates,
        )

    def _process_round(
        self,
        context: RefinementContext,
        unresolved: list[_AmbiguousGroup],
    ) -> tuple[
        list[_AmbiguousGroup], list[DistanceConstraint], list[_AmbiguousGroup], int, bool
    ]:
        """Process one refinement round over all unresolved ambiguous groups."""
        next_unresolved: list[_AmbiguousGroup] = []
        newly_promoted: list[DistanceConstraint] = []
        promoted_groups: list[_AmbiguousGroup] = []
        pruned_candidates = 0
        any_candidate_pruned = False

        for pair, candidates in unresolved:
            context.check_time_limit()
            (
                retained_constraints,
                promoted_constraint,
                removed_candidates,
            ) = self._evaluate_pair(context, pair, candidates)

            if removed_candidates > 0:
                pruned_candidates += removed_candidates
                any_candidate_pruned = True

            if promoted_constraint is not None:
                newly_promoted.append(promoted_constraint)
                promoted_groups.append((pair, retained_constraints))
            else:
                next_unresolved.append((pair, retained_constraints))

        return (
            next_unresolved,
            newly_promoted,
            promoted_groups,
            pruned_candidates,
            any_candidate_pruned,
        )

    @staticmethod
    def _evaluate_pair(
        context: RefinementContext,
        pair: tuple[int, int],
        candidates: list[DistanceConstraint],
    ) -> tuple[list[DistanceConstraint], DistanceConstraint | None, int]:
        """Evaluate one ambiguous pair and decide retain/prune/promote actions."""
        state = context.state
        i, j = pair
        pos_i = state.coordinates[i]
        pos_j = state.coordinates[j]
        actual_distance = float(np.linalg.norm(pos_i - pos_j))

        scored = [
            (
                constraint,
                context.constraint_violation(constraint, actual_distance),
            )
            for constraint in candidates
        ]
        scored.sort(key=lambda item: item[1])

        best_constraint, best_violation = scored[0]
        second_best_violation = scored[1][1] if len(scored) > 1 else float("inf")
        best_margin = second_best_violation - best_violation

        retain_cutoff = best_violation + context.config.soft_pruning_candidate_window
        retained_constraints = [
            constraint
            for constraint, violation in scored
            if violation <= retain_cutoff + context.config.distance_tolerance
        ]

        removed_candidates = len(candidates) - len(retained_constraints)
        should_promote = (
            best_violation <= context.config.soft_pruning_acceptance_tolerance
            and best_margin >= context.config.soft_pruning_min_margin
        )

        promoted_constraint: DistanceConstraint | None = None
        if should_promote:
            promoted_constraint = DistanceConstraint(
                i=best_constraint.i,
                j=best_constraint.j,
                lower_bound=best_constraint.lower_bound,
                upper_bound=best_constraint.upper_bound,
            )

        return retained_constraints, promoted_constraint, removed_candidates

    @staticmethod
    def _apply_refinement_solution(
        state: ProblemState,
        coordinates: NDArray[np.float64],
        refinement_stats: SBBUStats,
        num_nodes: int,
    ) -> None:
        """Apply successful refinement results into the active solver state."""
        state.coordinates[:] = coordinates
        state.current_node = num_nodes - 1
        state.stats.iterations_used += refinement_stats.iterations_used
        state.stats.constraints_processed = max(
            state.stats.constraints_processed,
            refinement_stats.constraints_processed,
        )

    @staticmethod
    def _finalize_refinement(
        state: ProblemState,
        unresolved: list[_AmbiguousGroup],
        rounds_used: int,
        promoted_constraints: list[DistanceConstraint],
        pruned_candidates: int,
    ) -> None:
        """Persist refinement outcomes into runtime statistics/state."""
        state.active_soft_ambiguous_constraints = unresolved
        state.stats.soft_pruning_rounds = rounds_used
        state.stats.soft_pruned_constraints = len(promoted_constraints)
        state.stats.soft_pruned_candidates = pruned_candidates
