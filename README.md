# SBBU: Distance Geometry in Python

Reconstruct 3D coordinates from pairwise distances or distance bounds using SBBU,
MM, and TRF. Inputs can be NMR restraint files, distance matrices, bounds matrices,
or edge lists.

The SBBU implementation is based on
[michaelsouza/sbbu](https://github.com/michaelsouza/sbbu).

## Installation

From a source checkout:

```bash
pip install .
```

Requires Python ≥ 3.10, NumPy ≥ 1.21.0, and SciPy ≥ 1.10.0.

## Quick Start

```python
import numpy as np
import sbbu

# Exact distances
matrix = np.array(
    [
        [0.0, 1.2, 1.8],
        [1.2, 0.0, 1.1],
        [1.8, 1.1, 0.0],
    ]
)
coordinates, stats = sbbu.solve_from_distance_matrix(matrix)

# Distance intervals
lower = np.ones((4, 4)) - np.eye(4)
upper = 2 * lower
coordinates, stats = sbbu.solve_from_bounds_matrices(lower, upper, method="trf")

# Edge list: (first node, second node, lower bound, upper bound)
edges = [(0, 1, 1.2, 1.2), (1, 2, 1.1, 1.1), (0, 2, 1.8, 1.8)]
coordinates, stats = sbbu.solve_from_edge_list(3, edges)

# NMR restraint file
coordinates, stats = sbbu.solve_from_nmr("protein.nmr")
```

Returned coordinates have shape `(num_nodes, 3)` and retain the input node order.

## Inputs

Specify each observed distance as a finite, positive closed interval `[lower, upper]`.
A distance is exact only when its bounds are equal, regardless of solver tolerance.

- **Distance matrices:** square and symmetric, with a zero diagonal. NaNs are
  skipped by default. Set `ignore_inf` or `ignore_zero` to also skip infinite or
  non-positive off-diagonal entries, respectively.
- **Bounds matrices:** matching square, symmetric matrices with zero diagonals
  and `0 < lower <= upper` for observed pairs. To omit a pair, use NaNs in both
  matrices and both directions. Set `ignore_nan=False` to reject missing pairs.
- **Edge lists:** `(i, j, lower, upper)` tuples with zero-based node indices.
  Leave out missing pairs.
- **NMR files:** restraint files containing lower and upper distance bounds.

Duplicate observations are intersected. If the intersection is empty, they become
soft alternatives rather than constraints that must all hold.

One-sided or zero-distance bounds are unsupported; infinity is not an open bound.
Bounds use float64. Lossy conversions are rejected, but precision already lost in
an input array cannot be recovered.

## Solver Selection

Choose `method` by keyword in any of the four `solve_from_*` functions:

| Value | Selection when SBBU is not applicable |
| --- | --- |
| `None` (default) | Automatic MM/TRF selection |
| `"mm"` | MM refinement |
| `"trf"` | TRF refinement |

**SBBU always takes precedence** when the input order provides exact, unambiguous
distances for every pair with `1 <= j - i <= 3`. Nodes are not reordered
automatically, and a failed SBBU run is not retried with another solver.

SBBU allows long-range distances to be missing or interval-valued. MM and TRF
currently require at least four nodes, a complete unambiguous graph, and
strictly positive-width intervals on every pair. They cannot yet solve
missing-edge inputs, though the input formats allow them. Mixed exact/interval
observations and ambiguous alternatives require SBBU's exact predecessors.

Valid inputs may still fail to solve. Failure, stationarity, or a time limit does
not prove that no realization exists.

## Options and Results

```python
coordinates, stats = sbbu.solve_from_bounds_matrices(
    lower,
    upper,
    method=None,
    distance_tolerance=1e-7,
    max_time=60.0,
    verbose=False,
)

print(stats.solve_time)
print(stats.largest_distance_error)
for stage in stats.stages:
    print(stage.algorithm, stage.termination, stage.seconds)
```

- `distance_tolerance`: maximum absolute violation allowed for an original hard
  bound. Input intervals are unchanged.
- `max_time`: time limit for numerical solving and final validation, excluding
  input parsing and normalization. Calls may overrun this cooperative limit, but
  no result is returned if the deadline check fails.
- `verbose`: SBBU progress logging.

The `solve_from_*` functions and `SBBUSolver.solve()` use immutable `SolveStats`
records:

- `solve_time`: elapsed solve time through final validation.
- `mean_distance_error`, `largest_distance_error`: mean and maximum violations of
  the original normalized hard bounds.
- `stages`: algorithm reports in execution order. Empty if initialization already
  satisfies the bounds. SBBU's `iterations`, MM's `steps`, and TRF's `evaluations`
  count different kinds of work.

Successful results contain finite coordinates satisfying every original hard
bound at the requested tolerance. Different valid realizations or coordinate
frames may be returned.

`SolveError` indicates failed reconstruction, with available diagnostics in
`error.stages`. Its subclass `SolveTimeoutError` indicates deadline exhaustion.
Inputs outside the supported solver capabilities raise `UnsupportedProblemError`,
a subclass of `ValueError`.

### Direct SBBU Use

```python
from sbbu.adapters import MatrixAdapter
from sbbu.core.solver import SBBUConfig, SBBUSolver

constraints = MatrixAdapter.from_distance_matrix(matrix)
solver = SBBUSolver(constraints, SBBUConfig(distance_tolerance=1e-7, verbose=False))
stats = solver.solve()
coordinates = solver.solution_coordinates
```

For custom observations, create
`DistanceConstraint(i, j, lower_bound, upper_bound)` objects and collect them in a
`ConstraintSet`.

### Migration Notes

- `SolveStats` replaces `SBBUStats`. Algorithm-specific counters now belong to
  `stats.stages`, not top-level statistics fields.
- Bounds are explicit: use `DistanceConstraint(i, j, d, d)` and `(i, j, d, d)`
  edges for exact distances. Use `MatrixAdapter` instead of the former
  `ConstraintSet.from_*` constructors.
- Use `lower_bound`, `upper_bound`, and `is_exact` instead of `constraint.distance`.
  Configure tolerance on the solver, not on `NMRAdapter.from_nmr_file`.

## Citation

If this project is useful in your work, please cite the original SBBU paper:

Gonçalves D. S.; Lavor C.; Liberti L.; Souza M. A New Algorithm for the KDMDGP
Subclass of Distance Geometry Problems with Exact Distances. Algorithmica 2021,
83 (8), 2400-2426. https://doi.org/10.1007/s00453-021-00835-6

```bibtex
@article{Goncalves2021KDMDGP,
  author  = {Douglas Soares Gon{\c{c}}alves and Carlile Lavor and Leo Liberti and Michael Souza},
  title   = {A New Algorithm for the \({}^{\mbox{K}}\)DMDGP Subclass of Distance Geometry Problems with Exact Distances},
  journal = {Algorithmica},
  year    = {2021},
  volume  = {83},
  number  = {8},
  pages   = {2400--2426},
  doi     = {10.1007/S00453-021-00835-6},
  url     = {https://doi.org/10.1007/s00453-021-00835-6}
}
```
