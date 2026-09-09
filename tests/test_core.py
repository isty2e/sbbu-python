"""
Tests for the core SBBU algorithm components.
"""

import numpy as np
import pytest

from sbbu.constraints import ConstraintSet, DistanceConstraint
from sbbu.core.cluster import ReflectionCluster, UnionFind
from sbbu.core.refiner import SoftPruningRefiner
from sbbu.core.solver import SBBUConfig, SBBUSolver
from sbbu.core.types import (
    ProblemState,
    RefinementContext,
    SBBUSolveInfeasibleError,
    SBBUStats,
    SBBUTimeoutError,
)


@pytest.mark.unit
def test_constraint_creation():
    """Test creating distance constraints."""
    constraint = DistanceConstraint(0, 1, 1.5, 1.5)

    assert constraint.i == 0
    assert constraint.j == 1
    assert constraint.lower_bound == 1.5
    assert constraint.upper_bound == 1.5
    assert constraint.is_exact


@pytest.mark.unit
def test_constraint_validation():
    """Test constraint validation."""
    # Invalid order (i >= j)
    with pytest.raises(ValueError):
        DistanceConstraint(1, 0, 1.5, 1.5)

    # Negative distance
    with pytest.raises(ValueError):
        DistanceConstraint(0, 1, -1.0, -1.0)

    # Invalid upper bound
    with pytest.raises(ValueError):
        DistanceConstraint(0, 1, 2.0, upper_bound=1.0)

    # Negative node index should be rejected at set level
    with pytest.raises(ValueError):
        ConstraintSet(2, [DistanceConstraint(-1, 1, 1.0, 1.0)])


@pytest.mark.unit
def test_constraint_validation_rejects_non_finite_bounds():
    """Distance constraints should reject NaN/inf bounds."""
    with pytest.raises(ValueError, match="finite"):
        DistanceConstraint(0, 1, np.nan, np.nan)

    with pytest.raises(ValueError, match="finite"):
        DistanceConstraint(0, 1, np.inf, np.inf)

    with pytest.raises(ValueError, match="finite"):
        DistanceConstraint(0, 1, 1.0, upper_bound=np.nan)

    with pytest.raises(ValueError, match="finite"):
        DistanceConstraint(0, 1, 1.0, upper_bound=np.inf)


@pytest.mark.unit
@pytest.mark.parametrize("indices", [(0.0, 1), (0, 1.0), (False, True)])
def test_constraint_validation_rejects_non_integer_indices(indices):
    """Distance constraints should reject non-integer/bool node indices."""
    i, j = indices
    with pytest.raises(TypeError, match="integers"):
        DistanceConstraint(i, j, 1.0, 1.0)  # type: ignore[arg-type]


@pytest.mark.unit
def test_constraint_set_creation(simple_tetrahedron):
    """Test creating constraint sets."""
    true_coords, constraint_set = simple_tetrahedron

    assert constraint_set.num_nodes == 4
    assert len(constraint_set.constraints) == 6  # Complete graph

    # Check constraint lookup
    c01 = constraint_set.get_constraint(0, 1)
    assert c01 is not None
    assert c01.i == 0
    assert c01.j == 1

    # Non-existent constraint
    c_none = constraint_set.get_constraint(0, 10)
    assert c_none is None


@pytest.mark.unit
def test_constraint_set_edge_classification(simple_chain):
    """Test sequential vs long-range edge classification."""
    _, constraint_set = simple_chain

    sequential = constraint_set.get_sequential_constraints()
    long_range = constraint_set.get_long_range_constraints()

    # Should have sequential edges
    assert len(sequential) > 0
    for c in sequential:
        assert c.j - c.i <= 3

    # Should have long-range edges
    assert len(long_range) > 0
    for c in long_range:
        assert c.j - c.i > 3


@pytest.mark.unit
def test_long_range_constraint_ordering():
    """Long-range constraints should be sorted by target node then separation."""
    constraints = [
        DistanceConstraint(0, 7, 2.0, 2.0),
        DistanceConstraint(1, 5, 2.0, 2.0),
        DistanceConstraint(0, 6, 2.0, 2.0),
        DistanceConstraint(2, 6, 2.0, 2.0),
    ]
    constraint_set = ConstraintSet(8, constraints)

    long_range = constraint_set.get_long_range_constraints()
    actual_order = [(c.i, c.j) for c in long_range]

    assert actual_order == [(1, 5), (2, 6), (0, 6), (0, 7)]


def _five_node_reference_constraints() -> tuple[np.ndarray, list[DistanceConstraint]]:
    """Build a non-degenerate 5-node reference system with exact sequential edges."""
    true_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816],
            [1.5, 0.289, 0.816],
        ]
    )

    constraints = []
    for i in range(5):
        for j in range(i + 1, min(i + 4, 5)):
            distance = float(np.linalg.norm(true_coords[i] - true_coords[j]))
            constraints.append(DistanceConstraint(i, j, distance, distance))
    return true_coords, constraints


def _six_node_reference_constraints() -> tuple[np.ndarray, list[DistanceConstraint]]:
    """Build a non-degenerate 6-node system with one long-range edge."""
    true_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816],
            [1.5, 0.289, 0.816],
            [2.0, 1.0, 0.2],
        ]
    )

    constraints = []
    for i in range(6):
        for j in range(i + 1, min(i + 4, 6)):
            distance = float(np.linalg.norm(true_coords[i] - true_coords[j]))
            constraints.append(DistanceConstraint(i, j, distance, distance))

    distance = float(np.linalg.norm(true_coords[0] - true_coords[5]))
    constraints.append(DistanceConstraint(0, 5, distance, distance))
    return true_coords, constraints


@pytest.mark.unit
def test_overlapping_long_range_duplicates_are_intersected():
    """Overlapping long-range duplicates should become one merged hard interval."""
    _, constraints = _five_node_reference_constraints()
    constraints.extend(
        [
            DistanceConstraint(0, 4, 2.0, upper_bound=3.0),
            DistanceConstraint(0, 4, 2.5, upper_bound=3.5),
        ]
    )

    constraint_set = ConstraintSet(5, constraints)
    merged = constraint_set.get_constraint(0, 4)

    assert merged is not None
    assert abs(merged.lower_bound - 2.5) < 1e-12
    assert abs(merged.upper_bound - 3.0) < 1e-12
    assert len(constraint_set.get_soft_ambiguous_constraints()) == 0


@pytest.mark.unit
def test_conflicting_long_range_duplicates_become_soft_ambiguous():
    """Conflicting long-range duplicates should be tracked as soft-ambiguous only."""
    _, constraints = _five_node_reference_constraints()
    constraints.extend(
        [
            DistanceConstraint(0, 4, 1.9, upper_bound=2.0),
            DistanceConstraint(0, 4, 2.5, upper_bound=2.6),
        ]
    )

    constraint_set = ConstraintSet(5, constraints)

    assert constraint_set.get_constraint(0, 4) is None
    assert len(constraint_set.get_long_range_constraints()) == 0

    soft_constraints = constraint_set.get_soft_ambiguous_constraints()
    assert len(soft_constraints) == 1
    pair, options = soft_constraints[0]
    assert pair == (0, 4)
    assert len(options) == 2


@pytest.mark.unit
def test_solver_rejects_ambiguous_sequential_duplicates():
    """The graph retains alternatives; sequential placement requires one hard value."""
    constraints = [
        DistanceConstraint(0, 1, 1.0, 1.0),
        DistanceConstraint(0, 1, 1.1, 1.1),
        DistanceConstraint(0, 2, 1.4, 1.4),
        DistanceConstraint(1, 2, 1.0, 1.0),
    ]

    constraint_set = ConstraintSet(3, constraints)
    assert len(constraint_set.get_soft_ambiguous_constraints()) == 1
    with pytest.raises(ValueError, match="Missing required sequential"):
        SBBUSolver(constraint_set)


@pytest.mark.unit
def test_solver_accepts_long_range_interval(sbbu_config_fast):
    """Long-range constraints with bounds should accept in-interval solutions."""
    true_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816],
            [1.5, 0.289, 0.816],
        ]
    )

    constraints = []
    for i in range(5):
        for j in range(i + 1, min(i + 4, 5)):
            distance = float(np.linalg.norm(true_coords[i] - true_coords[j]))
            constraints.append(DistanceConstraint(i, j, distance, distance))

    d04 = float(np.linalg.norm(true_coords[0] - true_coords[4]))
    interval_edge = DistanceConstraint(0, 4, d04 - 0.2, upper_bound=d04 + 0.2)
    constraints.append(interval_edge)

    constraint_set = ConstraintSet(5, constraints)
    solver = SBBUSolver(constraint_set, sbbu_config_fast)
    solver.solve()

    coords = solver.solution_coordinates
    distance_04 = float(np.linalg.norm(coords[0] - coords[4]))
    assert (
        interval_edge.lower_bound - 1e-9
        <= distance_04
        <= interval_edge.upper_bound + 1e-9
    )


@pytest.mark.unit
def test_solver_reports_soft_ambiguous_violations(sbbu_config_fast):
    """Soft-ambiguous pairs should be scored, not treated as hard failures."""
    true_coords, constraints = _five_node_reference_constraints()
    d04 = float(np.linalg.norm(true_coords[0] - true_coords[4]))
    constraints.extend(
        [
            DistanceConstraint(0, 4, d04 - 0.05, upper_bound=d04 + 0.05),
            DistanceConstraint(0, 4, d04 + 0.8, upper_bound=d04 + 0.9),
        ]
    )

    config = SBBUConfig(
        distance_tolerance=sbbu_config_fast.distance_tolerance,
        max_iterations=sbbu_config_fast.max_iterations,
        max_time=sbbu_config_fast.max_time,
        verbose=False,
        enable_soft_pruning=False,
    )
    solver = SBBUSolver(ConstraintSet(5, constraints), config)
    stats = solver.solve()

    assert stats.soft_ambiguous_constraints == 1
    assert stats.mean_soft_violation < 1e-6
    assert stats.max_soft_violation < 1e-6


@pytest.mark.unit
def test_soft_pruning_promotes_confident_candidate():
    """Soft pruning should promote a clear winner and remove the ambiguous group."""
    true_coords, constraints = _five_node_reference_constraints()
    d04 = float(np.linalg.norm(true_coords[0] - true_coords[4]))
    constraints.extend(
        [
            DistanceConstraint(0, 4, d04 - 0.03, upper_bound=d04 + 0.03),
            DistanceConstraint(0, 4, d04 + 0.8, upper_bound=d04 + 0.9),
        ]
    )

    config = SBBUConfig(
        distance_tolerance=1e-6,
        max_iterations=1_000_000,
        max_time=5.0,
        verbose=False,
        enable_soft_pruning=True,
        soft_pruning_max_rounds=2,
        soft_pruning_acceptance_tolerance=5e-2,
        soft_pruning_candidate_window=2e-1,
        soft_pruning_min_margin=2e-2,
    )
    solver = SBBUSolver(ConstraintSet(5, constraints), config)
    stats = solver.solve()

    assert stats.soft_pruning_rounds >= 1
    assert stats.soft_pruned_constraints == 1
    assert stats.soft_ambiguous_constraints == 0
    assert stats.mean_soft_violation == 0.0
    assert stats.max_soft_violation == 0.0


@pytest.mark.unit
def test_soft_pruning_refinement_failure_keeps_ambiguous_group(monkeypatch):
    """Failed refinement solve should not discard unresolved ambiguous groups."""
    true_coords, constraints = _five_node_reference_constraints()
    d04 = float(np.linalg.norm(true_coords[0] - true_coords[4]))
    constraints.extend(
        [
            DistanceConstraint(0, 4, d04 - 0.03, upper_bound=d04 + 0.03),
            DistanceConstraint(0, 4, d04 + 0.8, upper_bound=d04 + 0.9),
        ]
    )

    config = SBBUConfig(
        distance_tolerance=1e-6,
        max_iterations=1_000_000,
        max_time=5.0,
        verbose=False,
        enable_soft_pruning=True,
        soft_pruning_max_rounds=2,
        soft_pruning_acceptance_tolerance=5e-2,
        soft_pruning_candidate_window=2e-1,
        soft_pruning_min_margin=2e-2,
    )
    solver = SBBUSolver(ConstraintSet(5, constraints), config)
    monkeypatch.setattr(
        solver,
        "_run_refinement_solve",
        lambda _candidate_hard_constraints: (_ for _ in ()).throw(
            SBBUSolveInfeasibleError("forced-refinement-infeasible")
        ),
    )

    stats = solver.solve()

    assert stats.soft_pruning_rounds >= 1
    assert stats.soft_pruned_constraints == 0
    assert stats.soft_ambiguous_constraints == 1


@pytest.mark.unit
def test_soft_pruning_timeout_error_is_propagated(monkeypatch):
    """Timeout-like refinement errors should propagate to solve()."""
    true_coords, constraints = _five_node_reference_constraints()
    d04 = float(np.linalg.norm(true_coords[0] - true_coords[4]))
    constraints.extend(
        [
            DistanceConstraint(0, 4, d04 - 0.03, upper_bound=d04 + 0.03),
            DistanceConstraint(0, 4, d04 + 0.8, upper_bound=d04 + 0.9),
        ]
    )

    config = SBBUConfig(
        distance_tolerance=1e-6,
        max_iterations=1_000_000,
        max_time=5.0,
        verbose=False,
        enable_soft_pruning=True,
        soft_pruning_max_rounds=2,
        soft_pruning_acceptance_tolerance=5e-2,
        soft_pruning_candidate_window=2e-1,
        soft_pruning_min_margin=2e-2,
    )
    solver = SBBUSolver(ConstraintSet(5, constraints), config)
    monkeypatch.setattr(
        solver,
        "_run_refinement_solve",
        lambda _candidate_hard_constraints: (_ for _ in ()).throw(
            SBBUTimeoutError("SBBU Core: time exceeded (tmax = 0.0)")
        ),
    )

    with pytest.raises(RuntimeError, match="time exceeded"):
        solver.solve()


@pytest.mark.unit
def test_soft_pruning_generic_runtime_error_is_propagated(monkeypatch):
    """Unexpected runtime errors in refinement should not be swallowed."""
    true_coords, constraints = _five_node_reference_constraints()
    d04 = float(np.linalg.norm(true_coords[0] - true_coords[4]))
    constraints.extend(
        [
            DistanceConstraint(0, 4, d04 - 0.03, upper_bound=d04 + 0.03),
            DistanceConstraint(0, 4, d04 + 0.8, upper_bound=d04 + 0.9),
        ]
    )

    config = SBBUConfig(
        distance_tolerance=1e-6,
        max_iterations=1_000_000,
        max_time=5.0,
        verbose=False,
        enable_soft_pruning=True,
        soft_pruning_max_rounds=2,
        soft_pruning_acceptance_tolerance=5e-2,
        soft_pruning_candidate_window=2e-1,
        soft_pruning_min_margin=2e-2,
    )
    solver = SBBUSolver(ConstraintSet(5, constraints), config)
    monkeypatch.setattr(
        solver,
        "_run_refinement_solve",
        lambda _candidate_hard_constraints: (_ for _ in ()).throw(
            RuntimeError("forced-generic-runtime-error")
        ),
    )

    with pytest.raises(RuntimeError, match="forced-generic-runtime-error"):
        solver.solve()


@pytest.mark.unit
def test_soft_pruning_does_not_duplicate_promoted_constraints_across_rounds():
    """Refinement rounds should not duplicate already-promoted hard constraints."""
    num_nodes = 6
    coordinates = np.zeros((num_nodes, 3), dtype=np.float64)
    union_find = UnionFind(num_nodes)
    clusters = [ReflectionCluster(num_nodes, coordinates) for _ in range(num_nodes)]
    for idx, cluster in enumerate(clusters):
        cluster.start_node = idx
        cluster.end_node = idx

    state = ProblemState(
        coordinates=coordinates,
        current_node=num_nodes - 1,
        active_soft_ambiguous_constraints=[
            (
                (0, 4),
                [
                    DistanceConstraint(0, 4, 1.0, upper_bound=1.1),
                    DistanceConstraint(0, 4, 2.0, upper_bound=2.1),
                ],
            )
        ],
        union_find=union_find,
        clusters=clusters,
        cluster_nodes=[0] * (num_nodes * 2),
        reflection_flags=[False] * (num_nodes * 2),
        best_reflection_flags=[False] * (num_nodes * 2),
        num_cluster_nodes=0,
        stats=SBBUStats(),
    )

    hard_constraints = [
        DistanceConstraint(0, 1, 1.0, 1.0),
        DistanceConstraint(0, 2, 1.5, 1.5),
        DistanceConstraint(1, 2, 1.0, 1.0),
    ]
    promoted_round_1 = DistanceConstraint(0, 4, 1.0, upper_bound=1.1)
    promoted_round_2 = DistanceConstraint(1, 5, 1.0, upper_bound=1.1)

    config = SBBUConfig(
        distance_tolerance=1e-6,
        max_iterations=64,
        max_time=1.0,
        verbose=False,
        enable_soft_pruning=True,
        soft_pruning_max_rounds=2,
        soft_pruning_acceptance_tolerance=1.0,
        soft_pruning_candidate_window=1.0,
        soft_pruning_min_margin=0.0,
    )

    refiner = SoftPruningRefiner()
    calls = {"round": 0, "candidate_sizes": []}

    def fake_process_round(_context, _unresolved):
        calls["round"] += 1
        if calls["round"] == 1:
            return (
                [((1, 5), [promoted_round_2])],
                [promoted_round_1],
                [((0, 4), [promoted_round_1])],
                0,
                True,
            )
        return (
            [],
            [promoted_round_2],
            [((1, 5), [promoted_round_2])],
            0,
            True,
        )

    refiner._process_round = fake_process_round  # type: ignore[method-assign]

    context = RefinementContext(
        state=state,
        config=config,
        num_nodes=num_nodes,
        hard_constraints=hard_constraints,
        check_time_limit=lambda: None,
        run_refinement_solve=lambda candidate_hard_constraints: (
            calls["candidate_sizes"].append(len(candidate_hard_constraints))
            or coordinates.copy(),
            SBBUStats(),
        ),
    )

    refiner.run(context)

    assert calls["candidate_sizes"] == [4, 5]


@pytest.mark.unit
def test_solver_rejects_interval_sequential_constraint(sbbu_config_fast):
    """Sequential trilateration constraints must stay exact for SBBU placement."""
    constraints = [
        DistanceConstraint(0, 1, 1.0, upper_bound=1.2),  # Non-exact sequential edge
        DistanceConstraint(0, 2, 1.4, 1.4),
        DistanceConstraint(1, 2, 1.0, 1.0),
        DistanceConstraint(0, 3, 1.7, 1.7),
        DistanceConstraint(1, 3, 1.4, 1.4),
        DistanceConstraint(2, 3, 1.0, 1.0),
    ]
    with pytest.raises(ValueError, match="must be exact"):
        constraint_set = ConstraintSet(4, constraints)
        SBBUSolver(constraint_set, sbbu_config_fast).solve()


@pytest.mark.unit
def test_sbbu_config():
    """Test SBBU configuration."""
    config = SBBUConfig(
        distance_tolerance=1e-6, max_iterations=100000, max_time=30.0, verbose=False
    )

    assert config.distance_tolerance == 1e-6
    assert config.max_iterations == 100000
    assert config.max_time == 30.0
    assert config.verbose is False


@pytest.mark.unit
def test_sbbu_config_rejects_non_positive_max_time():
    """SBBUConfig must enforce a strictly positive max_time."""
    with pytest.raises(ValueError, match="max_time must be positive"):
        SBBUConfig(max_time=0.0)

    with pytest.raises(ValueError, match="max_time must be positive"):
        SBBUConfig(max_time=-1.0)


@pytest.mark.unit
def test_core_solver_initialization(simple_tetrahedron):
    """Constructor should be side-effect light and defer runtime setup to solve()."""
    _, constraint_set = simple_tetrahedron
    config = SBBUConfig(verbose=False)

    solver = SBBUSolver(constraint_set, config)

    assert solver.constraints == constraint_set
    assert solver.config == config
    with pytest.raises(RuntimeError, match="Call solve\\(\\) first"):
        _ = solver.solution_coordinates


@pytest.mark.unit
def test_first_three_nodes_initialization(simple_tetrahedron):
    """First three-node geometric seed should be valid during solve()."""
    _, constraint_set = simple_tetrahedron
    config = SBBUConfig(verbose=False)

    solver = SBBUSolver(constraint_set, config)
    solver.solve()
    coords = solver.solution_coordinates

    # Node 0 should be at origin
    assert np.allclose(coords[0], [0, 0, 0])

    # Node 1 should be on x-axis
    assert coords[1][0] > 0
    assert abs(coords[1][1]) < 1e-10
    assert abs(coords[1][2]) < 1e-10

    # Node 2 should be in xy-plane
    assert abs(coords[2][2]) < 1e-10


@pytest.mark.unit
def test_core_solver_simple_case(simple_tetrahedron, sbbu_config_fast):
    """Test core solver on simple tetrahedron."""
    true_coords, constraint_set = simple_tetrahedron

    solver = SBBUSolver(constraint_set, sbbu_config_fast)
    stats = solver.solve()

    assert stats.solve_time > 0
    assert stats.constraints_processed >= 0
    assert stats.mean_distance_error < 1e-6

    # Check that all constraints are satisfied
    coords = solver.solution_coordinates
    for constraint in constraint_set.constraints:
        actual_dist = np.linalg.norm(coords[constraint.i] - coords[constraint.j])
        error = abs(actual_dist - constraint.lower_bound)
        assert error < sbbu_config_fast.distance_tolerance


@pytest.mark.unit
def test_solve_rejects_non_positive_override_max_time(
    simple_tetrahedron, sbbu_config_fast
):
    """solve(max_time=...) must reject non-positive overrides."""
    _, constraint_set = simple_tetrahedron
    solver = SBBUSolver(constraint_set, sbbu_config_fast)

    with pytest.raises(ValueError, match="max_time must be positive"):
        solver.solve(max_time=0.0)

    with pytest.raises(ValueError, match="max_time must be positive"):
        solver.solve(max_time=-0.5)


@pytest.mark.unit
def test_solve_rejects_non_finite_override_max_time(
    simple_tetrahedron, sbbu_config_fast
):
    """solve(max_time=...) must reject NaN/inf overrides."""
    _, constraint_set = simple_tetrahedron
    solver = SBBUSolver(constraint_set, sbbu_config_fast)

    with pytest.raises(ValueError, match="finite"):
        solver.solve(max_time=np.nan)

    with pytest.raises(ValueError, match="finite"):
        solver.solve(max_time=np.inf)


@pytest.mark.unit
def test_verbose_solver_logging_does_not_write_stdout(simple_tetrahedron, capsys):
    """Verbose mode should use logging instead of direct stdout writes."""
    _, constraint_set = simple_tetrahedron
    solver = SBBUSolver(constraint_set, SBBUConfig(verbose=True, max_time=5.0))

    capsys.readouterr()
    solver.solve()
    captured = capsys.readouterr()

    assert captured.out == ""


@pytest.mark.unit
def test_solver_error_handling():
    """Test solver error handling."""
    # Missing constraint for initialization
    constraints = [DistanceConstraint(0, 2, 1.0, 1.0)]  # Missing 0-1, 1-2
    constraint_set = ConstraintSet(3, constraints)

    config = SBBUConfig(verbose=False)

    # Should fail during solve-time initialization
    with pytest.raises(ValueError):
        SBBUSolver(constraint_set, config).solve()


@pytest.mark.unit
def test_solver_rejects_problems_with_fewer_than_three_nodes():
    """SBBUSolver should fail fast on unsupported problem sizes."""
    constraints = [DistanceConstraint(0, 1, 1.0, 1.0)]

    with pytest.raises(ValueError, match="at least 3 nodes"):
        SBBUSolver(ConstraintSet(2, constraints), SBBUConfig(verbose=False))


@pytest.mark.unit
def test_node_position_getter(simple_tetrahedron, sbbu_config_fast):
    """Test getting individual node positions."""
    _, constraint_set = simple_tetrahedron

    solver = SBBUSolver(constraint_set, sbbu_config_fast)
    solver.solve()

    # Test valid node
    pos = solver.get_node_position(0)
    assert pos.shape == (3,)

    # Test invalid node
    with pytest.raises(ValueError):
        solver.get_node_position(10)


@pytest.mark.unit
def test_solution_coordinates_copy(simple_tetrahedron, sbbu_config_fast):
    """Test that solution coordinates return a copy."""
    _, constraint_set = simple_tetrahedron

    solver = SBBUSolver(constraint_set, sbbu_config_fast)
    solver.solve()

    coords1 = solver.solution_coordinates
    coords2 = solver.solution_coordinates

    # Should be equal but not the same object
    assert np.allclose(coords1, coords2)
    assert coords1 is not coords2

    # Modifying copy shouldn't affect original
    coords1[0, 0] = 999.0
    coords3 = solver.solution_coordinates
    assert not np.allclose(coords1, coords3)


@pytest.mark.unit
def test_solution_coordinates_are_isolated_from_runtime_mutation(
    simple_tetrahedron, sbbu_config_fast
):
    """Public snapshot should remain available after runtime state is discarded."""
    _, constraint_set = simple_tetrahedron
    solver = SBBUSolver(constraint_set, sbbu_config_fast)
    solver.solve()

    snapshot = solver.solution_coordinates

    assert np.allclose(snapshot, solver.solution_coordinates)
    with pytest.raises(RuntimeError, match="Runtime state unavailable"):
        solver._state_required()


@pytest.mark.unit
def test_solver_reentrant_resets_runtime_state_after_mutation(sbbu_config_fast):
    """Repeated solve() should reset runtime state and keep results stable."""
    true_coords, constraints = _five_node_reference_constraints()
    distance = float(np.linalg.norm(true_coords[0] - true_coords[4]))
    constraints.append(DistanceConstraint(0, 4, distance, distance))
    solver = SBBUSolver(ConstraintSet(5, constraints), sbbu_config_fast)

    solver.solve()
    first_snapshot = solver.solution_coordinates.copy()

    stats_second = solver.solve()
    coords_second = solver.solution_coordinates

    assert stats_second.constraints_processed == 1
    assert stats_second.iterations_used >= 0
    assert np.allclose(first_snapshot, coords_second)

    tolerance = sbbu_config_fast.distance_tolerance
    for constraint in constraints:
        actual_distance = float(
            np.linalg.norm(coords_second[constraint.i] - coords_second[constraint.j])
        )
        upper = constraint.upper_bound
        assert (
            constraint.lower_bound - tolerance <= actual_distance <= upper + tolerance
        )


@pytest.mark.unit
def test_build_cluster_info_rejects_out_of_range_root_i():
    """Cluster builder should reject out-of-range root_i values."""
    _, constraints = _six_node_reference_constraints()
    solver = SBBUSolver(
        ConstraintSet(6, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=10.0, verbose=False),
    )
    solver._reset_runtime_state()
    constraint = solver.constraints.get_constraint(0, 5)
    assert constraint is not None

    with pytest.raises(ValueError, match="root_i"):
        solver._build_cluster_info(constraint, root_i=6, root_j=5)


@pytest.mark.unit
def test_build_cluster_info_rejects_out_of_range_root_j():
    """Cluster builder should reject out-of-range root_j values."""
    _, constraints = _six_node_reference_constraints()
    solver = SBBUSolver(
        ConstraintSet(6, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=10.0, verbose=False),
    )
    solver._reset_runtime_state()
    constraint = solver.constraints.get_constraint(0, 5)
    assert constraint is not None

    with pytest.raises(ValueError, match="root_j"):
        solver._build_cluster_info(constraint, root_i=3, root_j=6)


@pytest.mark.unit
def test_build_cluster_info_detects_visited_root_cycle():
    """Cluster builder should fail fast on repeated roots in one build pass."""
    _, constraints = _six_node_reference_constraints()
    solver = SBBUSolver(
        ConstraintSet(6, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=10.0, verbose=False),
    )
    solver._reset_runtime_state()
    state = solver._state_required()
    constraint = solver.constraints.get_constraint(0, 5)
    assert constraint is not None

    # Force find(4) == 3 so the traversal repeats root 3.
    state.union_find.union(4, 3)

    with pytest.raises(RuntimeError, match="inconsistent cluster structure"):
        solver._build_cluster_info(constraint, root_i=3, root_j=5)


@pytest.mark.unit
def test_build_cluster_info_raises_on_decision_vector_overflow():
    """Cluster builder should detect decision-vector capacity overflow."""
    _, constraints = _six_node_reference_constraints()
    solver = SBBUSolver(
        ConstraintSet(6, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=10.0, verbose=False),
    )
    solver._reset_runtime_state()
    state = solver._state_required()
    constraint = solver.constraints.get_constraint(0, 5)
    assert constraint is not None

    state.cluster_nodes = []

    with pytest.raises(RuntimeError, match="decision vector overflow"):
        solver._build_cluster_info(constraint, root_i=3, root_j=5)


@pytest.mark.unit
def test_build_cluster_info_resets_flags_and_records_decision_order():
    """Cluster builder should reset reflection flags and preserve visit order."""
    _, constraints = _six_node_reference_constraints()
    solver = SBBUSolver(
        ConstraintSet(6, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=10.0, verbose=False),
    )
    solver._reset_runtime_state()
    state = solver._state_required()
    constraint = solver.constraints.get_constraint(0, 5)
    assert constraint is not None

    solver._initialize_first_three_nodes()
    solver._extend_to_node(5)
    state.reflection_flags[:3] = [True, True, True]
    cluster = solver._build_cluster_info(constraint, root_i=3, root_j=5)

    assert cluster is state.clusters[3]
    assert state.num_cluster_nodes == 3
    assert state.cluster_nodes[:3] == [3, 4, 5]
    assert state.reflection_flags[:3] == [False, False, False]


@pytest.mark.unit
def test_build_cluster_info_creates_reflection_planes_for_eligible_nodes():
    """Cluster builder should create one plane per eligible decision node."""
    _, constraints = _six_node_reference_constraints()
    solver = SBBUSolver(
        ConstraintSet(6, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=10.0, verbose=False),
    )
    solver._reset_runtime_state()
    constraint = solver.constraints.get_constraint(0, 5)
    assert constraint is not None

    solver._initialize_first_three_nodes()
    solver._extend_to_node(5)
    cluster = solver._build_cluster_info(constraint, root_i=3, root_j=5)

    assert cluster.num_planes == 3


@pytest.mark.unit
def test_branch_search_checks_time_limit(simple_tetrahedron):
    """Inner branch search must invoke time checks to enforce max_time reliably."""
    _, constraint_set = simple_tetrahedron
    solver = SBBUSolver(
        constraint_set,
        SBBUConfig(
            distance_tolerance=1e-6, max_iterations=16, max_time=10.0, verbose=False
        ),
    )
    solver._reset_runtime_state()
    state = solver._state_required()

    state.num_cluster_nodes = 2
    state.reflection_flags[:2] = [False, False]
    state.best_reflection_flags[:2] = [False, False]

    cluster = state.clusters[0]
    cluster.end_node = 0
    cluster.num_planes = 0
    cluster.target_position[:] = state.coordinates[0]

    def forced_timeout() -> None:
        raise RuntimeError("forced-time-check")

    setattr(solver, "_check_time_limit", forced_timeout)

    with pytest.raises(RuntimeError, match="forced-time-check"):
        solver._branch_and_bound_search(DistanceConstraint(0, 3, 1.7, 1.7), cluster)


@pytest.mark.unit
def test_branch_search_checks_time_limit_each_state(simple_tetrahedron):
    """Branch search should check time on each evaluated state."""
    _, constraint_set = simple_tetrahedron
    solver = SBBUSolver(
        constraint_set,
        SBBUConfig(
            distance_tolerance=1e-6, max_iterations=8, max_time=10.0, verbose=False
        ),
    )
    solver._reset_runtime_state()
    state = solver._state_required()

    state.num_cluster_nodes = 3
    state.reflection_flags[:3] = [False, False, False]
    state.best_reflection_flags[:3] = [False, False, False]

    cluster = state.clusters[0]
    cluster.end_node = 0
    cluster.num_planes = 0
    cluster.target_position[:] = state.coordinates[0]

    check_calls = {"count": 0}

    def count_time_check() -> None:
        check_calls["count"] += 1

    setattr(solver, "_check_time_limit", count_time_check)

    with pytest.raises(RuntimeError, match="could not be solved"):
        solver._branch_and_bound_search(DistanceConstraint(0, 3, 1.7, 1.7), cluster)

    assert check_calls["count"] >= 8


@pytest.mark.unit
def test_soft_pruning_checks_time_limit_while_scanning_pairs():
    """Soft pruning should enforce time checks while iterating unresolved groups."""
    num_nodes = 6
    coordinates = np.zeros((num_nodes, 3), dtype=np.float64)
    union_find = UnionFind(num_nodes)
    clusters = [ReflectionCluster(num_nodes, coordinates) for _ in range(num_nodes)]
    for idx, cluster in enumerate(clusters):
        cluster.start_node = idx
        cluster.end_node = idx

    state = ProblemState(
        coordinates=coordinates,
        current_node=num_nodes - 1,
        active_soft_ambiguous_constraints=[
            (
                (0, 4),
                [
                    DistanceConstraint(0, 4, 1.0, upper_bound=1.1),
                    DistanceConstraint(0, 4, 2.0, upper_bound=2.1),
                ],
            ),
            (
                (1, 5),
                [
                    DistanceConstraint(1, 5, 1.0, upper_bound=1.1),
                    DistanceConstraint(1, 5, 2.0, upper_bound=2.1),
                ],
            ),
        ],
        union_find=union_find,
        clusters=clusters,
        cluster_nodes=[0] * (num_nodes * 2),
        reflection_flags=[False] * (num_nodes * 2),
        best_reflection_flags=[False] * (num_nodes * 2),
        num_cluster_nodes=0,
        stats=SBBUStats(),
    )

    config = SBBUConfig(
        distance_tolerance=1e-6,
        max_iterations=32,
        max_time=1.0,
        verbose=False,
        enable_soft_pruning=True,
        soft_pruning_max_rounds=1,
        soft_pruning_acceptance_tolerance=0.0,
        soft_pruning_candidate_window=0.0,
        soft_pruning_min_margin=1.0,
    )

    check_calls = {"count": 0}

    def forced_refiner_timeout() -> None:
        check_calls["count"] += 1
        if check_calls["count"] >= 2:
            raise RuntimeError("forced-refiner-time-check")

    context = RefinementContext(
        state=state,
        config=config,
        num_nodes=num_nodes,
        hard_constraints=[],
        check_time_limit=forced_refiner_timeout,
        run_refinement_solve=lambda _constraints: (_ for _ in ()).throw(
            AssertionError("refinement solve should not run in this test")
        ),
    )

    with pytest.raises(RuntimeError, match="forced-refiner-time-check"):
        SoftPruningRefiner().run(context)


@pytest.mark.unit
def test_solver_rejects_collinear_seed_nodes():
    """Solver should fail fast when first three nodes are collinear."""
    constraints = [
        DistanceConstraint(0, 1, 1.0, 1.0),
        DistanceConstraint(0, 2, 2.0, 2.0),
        DistanceConstraint(1, 2, 1.0, 1.0),
        DistanceConstraint(0, 3, 3.0, 3.0),
        DistanceConstraint(1, 3, 2.0, 2.0),
        DistanceConstraint(2, 3, 1.0, 1.0),
    ]
    solver = SBBUSolver(
        ConstraintSet(4, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=1.0, verbose=False),
    )

    with pytest.raises(ValueError, match="non-collinear"):
        solver.solve()


@pytest.mark.unit
def test_solver_allows_collinear_seed_when_problem_has_three_nodes():
    """Three-node problems should allow collinear seeds without trilateration."""
    constraints = [
        DistanceConstraint(0, 1, 1.0, 1.0),
        DistanceConstraint(0, 2, 2.0, 2.0),
        DistanceConstraint(1, 2, 1.0, 1.0),
    ]
    solver = SBBUSolver(
        ConstraintSet(3, constraints),
        SBBUConfig(distance_tolerance=1e-6, max_time=1.0, verbose=False),
    )

    stats = solver.solve()
    coordinates = solver.solution_coordinates

    assert coordinates.shape == (3, 3)
    assert stats.mean_distance_error <= 1e-6


@pytest.mark.unit
def test_extend_to_node_checks_time_limit():
    """Node-extension loop should enforce time checks between placements."""
    _, constraints = _five_node_reference_constraints()
    constraint_set = ConstraintSet(5, constraints)
    solver = SBBUSolver(
        constraint_set,
        SBBUConfig(
            distance_tolerance=1e-6,
            max_iterations=100,
            max_time=10.0,
            verbose=False,
        ),
    )
    solver._reset_runtime_state()
    solver._initialize_first_three_nodes()

    def forced_timeout() -> None:
        raise RuntimeError("forced-extend-time-check")

    setattr(solver, "_check_time_limit", forced_timeout)

    with pytest.raises(RuntimeError, match="forced-extend-time-check"):
        solver._extend_to_node(constraint_set.num_nodes - 1)


@pytest.mark.unit
def test_extend_to_node_does_not_regress_current_node():
    """Extending to a past node should leave current_node unchanged."""
    _, constraints = _five_node_reference_constraints()
    constraint_set = ConstraintSet(5, constraints)
    solver = SBBUSolver(
        constraint_set,
        SBBUConfig(
            distance_tolerance=1e-6,
            max_iterations=100,
            max_time=10.0,
            verbose=False,
        ),
    )
    solver._reset_runtime_state()
    solver._initialize_first_three_nodes()
    solver._extend_to_node(4)

    state = solver._state_required()
    current_before = state.current_node
    coordinates_before = state.coordinates.copy()

    solver._extend_to_node(2)

    assert state.current_node == current_before
    assert np.allclose(state.coordinates, coordinates_before)


@pytest.mark.unit
def test_extend_to_node_rejects_out_of_range_target():
    """Extending past the last node should fail fast with clear error."""
    _, constraints = _five_node_reference_constraints()
    constraint_set = ConstraintSet(5, constraints)
    solver = SBBUSolver(
        constraint_set,
        SBBUConfig(
            distance_tolerance=1e-6,
            max_iterations=100,
            max_time=10.0,
            verbose=False,
        ),
    )
    solver._reset_runtime_state()
    solver._initialize_first_three_nodes()

    with pytest.raises(ValueError, match="out of range"):
        solver._extend_to_node(5)

    with pytest.raises(ValueError, match="out of range"):
        solver._extend_to_node(-1)


@pytest.mark.unit
def test_remaining_time_budget_raises_when_deadline_is_reached(
    simple_tetrahedron, monkeypatch
):
    """Remaining budget helper should fail when wall-time reaches the deadline."""
    _, constraint_set = simple_tetrahedron
    solver = SBBUSolver(constraint_set, SBBUConfig(verbose=False))
    solver._solve_deadline = 42.0
    solver._solve_time_limit = 1.0

    monkeypatch.setattr("sbbu.core.solver.get_wall_time", lambda: 42.0)

    with pytest.raises(RuntimeError, match="time exceeded"):
        solver._remaining_time_budget()


@pytest.mark.unit
def test_solve_checks_timeout_after_refinement_hook(monkeypatch):
    """Solver should re-check time limit after refinement stage."""
    constraints = [
        DistanceConstraint(0, 1, 1.0, 1.0),
        DistanceConstraint(0, 2, 2.0, 2.0),
        DistanceConstraint(1, 2, 1.0, 1.0),
    ]
    solver = SBBUSolver(
        ConstraintSet(3, constraints),
        SBBUConfig(
            distance_tolerance=1e-6,
            max_iterations=100,
            max_time=10.0,
            verbose=False,
            enable_soft_pruning=True,
        ),
    )

    def expire_deadline(_context) -> None:
        solver._solve_deadline = 0.0
        solver._solve_time_limit = 10.0

    monkeypatch.setattr(solver._soft_pruning_refiner, "run", expire_deadline)

    with pytest.raises(RuntimeError, match="time exceeded"):
        solver.solve()


@pytest.mark.unit
def test_reflect_all_nodes_aligns_with_active_cluster_indices():
    """Reflection should not apply skipped (<3) decision indices as plane starts."""
    coordinates = np.array(
        [
            [0.0, 0.0, 1.0],  # node 0 (off plane)
            [0.0, 0.0, 0.0],  # node 1
            [1.0, 0.0, 0.0],  # node 2
            [0.0, 1.0, 0.0],  # node 3
            [0.0, 0.0, 2.0],  # node 4 (target)
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(5, coordinates)
    cluster.create_reflection_planes(node=4, cluster_indices=[0, 4])
    original_node0 = coordinates[0].copy()

    cluster.reflect_all_nodes([0, 4], [True])

    assert np.allclose(coordinates[0], original_node0)
    assert np.allclose(coordinates[4], [0.0, 0.0, -2.0])


@pytest.mark.unit
def test_create_reflection_planes_rejects_out_of_range_cluster_index():
    """Plane creation should fail fast on invalid cluster indices."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(5, coordinates)

    with pytest.raises(ValueError, match="out of range"):
        cluster.create_reflection_planes(node=4, cluster_indices=[6])


@pytest.mark.unit
def test_reflect_single_node_rejects_short_flag_vector():
    """Single-node reflection should validate flag-vector length."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[3])

    with pytest.raises(ValueError, match="reflection_flags"):
        cluster.reflect_single_node([])


@pytest.mark.unit
def test_reflect_all_nodes_rejects_short_flag_vector():
    """All-node reflection should validate flag-vector length."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[3])

    with pytest.raises(ValueError, match="reflection_flags"):
        cluster.reflect_all_nodes([3], [])


@pytest.mark.unit
def test_create_reflection_planes_skips_all_sub3_indices():
    """Indices below 3 should produce no reflection planes."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[0, 1, 2])

    assert cluster.num_planes == 0


@pytest.mark.unit
def test_union_find_path_compression_returns_stable_root():
    """Path compression should keep roots stable across repeated lookups."""
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)

    root_first = uf.find(0)
    root_second = uf.find(0)

    assert root_first == root_second == uf.find(3)


@pytest.mark.unit
def test_reflect_all_nodes_rejects_cluster_start_past_end_node():
    """Reflection should reject cluster starts that exceed end_node."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(5, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[4])

    with pytest.raises(ValueError, match="must not exceed end_node"):
        cluster.reflect_all_nodes([4], [True])


@pytest.mark.unit
def test_reflect_all_nodes_requires_exact_active_cluster_count():
    """Reflection should reject mismatched active-cluster/plane mapping."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(5, coordinates)
    cluster.create_reflection_planes(node=4, cluster_indices=[4])

    with pytest.raises(ValueError, match="must match plane count"):
        cluster.reflect_all_nodes([4, 3], [True])


@pytest.mark.unit
def test_union_find_rejects_out_of_range_queries():
    """UnionFind should fail fast on out-of-range indices."""
    uf = UnionFind(4)

    with pytest.raises(ValueError, match="out of range"):
        uf.find(4)

    with pytest.raises(ValueError, match="out of range"):
        uf.find(-1)

    with pytest.raises(ValueError, match="out of range"):
        uf.union(0, 4)

    with pytest.raises(ValueError, match="out of range"):
        uf.union(-1, 0)


@pytest.mark.unit
def test_create_reflection_planes_rejects_out_of_range_target_node():
    """Plane creation should reject invalid target node indices."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)

    with pytest.raises(ValueError, match="out of range"):
        cluster.create_reflection_planes(node=4, cluster_indices=[3])

    with pytest.raises(ValueError, match="out of range"):
        cluster.create_reflection_planes(node=-1, cluster_indices=[3])


@pytest.mark.unit
def test_create_reflection_planes_rejects_excessive_plane_count():
    """Plane creation should reject more planes than allocated storage."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)

    with pytest.raises(ValueError, match="Too many reflection planes"):
        cluster.create_reflection_planes(node=3, cluster_indices=[3, 3, 3, 3, 3])


@pytest.mark.unit
def test_reflect_all_nodes_rejects_invalid_cluster_index():
    """All-node reflection should reject invalid cluster start indices."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[3])

    with pytest.raises(ValueError, match="out of range"):
        cluster.reflect_all_nodes([3, 10], [True])


@pytest.mark.unit
def test_reflect_all_nodes_rejects_incompatible_active_cluster_indices():
    """All-node reflection should reject cluster indices incompatible with planes."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[3])

    with pytest.raises(ValueError, match="active cluster index count"):
        cluster.reflect_all_nodes([0, 1, 2], [True])


@pytest.mark.unit
def test_reflect_single_node_all_false_keeps_coordinates():
    """Single-node reflection with all-false flags should be a no-op."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[3])
    before = coordinates[3].copy()

    cluster.reflect_single_node([False])

    assert np.allclose(coordinates[3], before)


@pytest.mark.unit
def test_reflect_all_nodes_with_no_planes_is_noop():
    """All-node reflection with zero planes should not mutate coordinates."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[0, 1, 2])
    before = coordinates.copy()

    cluster.reflect_all_nodes([0, 1, 2], [])

    assert np.allclose(coordinates, before)


@pytest.mark.unit
def test_reflect_single_node_with_no_planes_is_noop():
    """Single-node reflection with zero planes should not mutate coordinates."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(4, coordinates)
    cluster.create_reflection_planes(node=3, cluster_indices=[0, 1, 2])
    before = coordinates[3].copy()

    cluster.reflect_single_node([])

    assert np.allclose(coordinates[3], before)


@pytest.mark.unit
def test_solver_accepts_near_collinear_seed_above_tolerance():
    """Near-collinear seeds slightly above tolerance should remain solvable."""
    epsilon = 2.0e-6
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, epsilon, 0.0],
            [0.5, 0.5, 0.8],
        ],
        dtype=np.float64,
    )
    constraints = []
    for i in range(4):
        for j in range(i + 1, min(i + 4, 4)):
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            constraints.append(DistanceConstraint(i, j, distance, distance))

    solver = SBBUSolver(
        ConstraintSet(4, constraints),
        SBBUConfig(distance_tolerance=1.0e-6, max_time=5.0, verbose=False),
    )
    stats = solver.solve()

    assert solver.solution_coordinates.shape == (4, 3)
    assert stats.mean_distance_error <= 1e-4


@pytest.mark.unit
def test_union_find_rejects_non_positive_size():
    """UnionFind should reject zero/negative sizes."""
    with pytest.raises(ValueError, match="positive"):
        UnionFind(0)

    with pytest.raises(ValueError, match="positive"):
        UnionFind(-1)


@pytest.mark.unit
def test_reflect_all_nodes_rejects_short_cluster_indices_for_plane_count():
    """All-node reflection should reject too-few active cluster starts."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(6, coordinates)
    cluster.create_reflection_planes(node=5, cluster_indices=[4, 5])

    with pytest.raises(ValueError, match="must match plane count"):
        cluster.reflect_all_nodes([5], [True, True])


@pytest.mark.unit
def test_reflect_all_nodes_rejects_short_reflection_flags_for_plane_count():
    """All-node reflection should reject too-short flag vectors."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(6, coordinates)
    cluster.create_reflection_planes(node=5, cluster_indices=[4, 5])

    with pytest.raises(ValueError, match="reflection_flags"):
        cluster.reflect_all_nodes([4, 5], [True])


@pytest.mark.unit
def test_reflect_single_node_accepts_exact_flag_length():
    """Single-node reflection should accept flags exactly equal to plane count."""
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cluster = ReflectionCluster(5, coordinates)
    cluster.create_reflection_planes(node=4, cluster_indices=[4])
    before = coordinates[4].copy()

    cluster.reflect_single_node([False])

    assert np.allclose(coordinates[4], before)


@pytest.mark.unit
def test_union_find_union_idempotent_on_same_set():
    """Repeated union calls on same set should stay stable."""
    uf = UnionFind(4)
    uf.union(0, 1)
    root_before = uf.find(0)

    uf.union(0, 1)
    uf.union(1, 0)

    assert uf.find(0) == root_before
    assert uf.find(1) == root_before


@pytest.mark.unit
def test_union_find_find_handles_deep_parent_chain_without_recursion_error():
    """Deep parent chains should not depend on Python recursion depth."""
    size = 3_000
    uf = UnionFind(size)

    for i in range(size - 1):
        uf.parent[i] = i + 1
    uf.parent[size - 1] = size - 1

    root = uf.find(0)

    assert root == size - 1
    assert uf.parent[0] == size - 1


@pytest.mark.unit
def test_constraint_violation_treats_nan_distance_as_infinite_error():
    """NaN distances should never be treated as satisfied constraints."""
    constraint = DistanceConstraint(0, 1, 1.0, upper_bound=2.0)

    violation = constraint.violation(float("nan"))

    assert np.isinf(violation)
