# SBBU: Symmetry-based Build-up

A Python implementation of the SBBU algorithm for solving Distance Geometry Problems (DGP).
This project is a Python port of the original C++ implementation:
[michaelsouza/sbbu](https://github.com/michaelsouza/sbbu)

## Features

- **Multiple Input Formats**: NMR files, distance matrices, bounds matrices, edge lists
- **High Performance**: Optimized core algorithm aligned with the original C++ reference
- **Extensive Testing**: Comprehensive test suite with pytest

## Quick Start

### Basic Usage

```python
import sbbu
import numpy as np

# From NMR files
coords, stats = sbbu.solve_from_nmr('protein.nmr')

# From distance matrices (e.g., ML predictions)
distance_matrix = np.array([...])  # Your NxN distance matrix
coords, stats = sbbu.solve_from_distance_matrix(distance_matrix)

# From edge lists (graph-like data)
edges = [(0, 1, 1.2, 1.2), (1, 2, 1.1, 1.1), (0, 2, 1.8, 1.8)]
coords, stats = sbbu.solve_from_edge_list(3, edges)
```

### Advanced Usage

```python
from sbbu.adapters import NMRAdapter, MatrixAdapter
from sbbu.core.solver import SBBUConfig, SBBUSolver

# Custom configuration
config = SBBUConfig(
    distance_tolerance=1e-7,
    max_time=60.0,
    verbose=True
)

# Load constraints from custom source
constraints = MatrixAdapter.from_distance_matrix(your_matrix)

# Solve with custom settings
solver = SBBUSolver(constraints, config)
stats = solver.solve()
coordinates = solver.solution_coordinates
```

## Algorithm Overview

SBBU solves the Distance Geometry Problem (DGP) by:

1. **Initialization**: Places first 3 nodes using exact distance constraints
2. **Sequential Extension**: Calculates positions for nodes 4,5,6,... using trilateration
3. **Constraint Satisfaction**: Uses branch-and-bound with geometric reflections to satisfy long-range distance constraints

For a complete graph of exact hard distances, an additional initializer selects four
well-separated anchors and places all nodes in that shared frame. It factors only a
`3 × 3` Gram matrix, avoiding sequential amplification of small rounding errors.
Full- and lower-rank candidates are accepted only after checking **every original
hard constraint** at the requested tolerance. Node IDs and the first-three-node
coordinate convention are preserved. This is an extension of the reference SBBU
implementation, not a global MDS fit or a continuous-optimization fallback.

Supported sparse inputs, long-range intervals and soft alternatives retain the
reflection search. Exact sequential predecessor distances are still required. The same
search is used if the initializer cannot produce an acceptable candidate. Local
reflection enumeration selects the lowest-error candidate within `max_iterations`,
rather than stopping at the first approximate tolerance pass for one edge. For
complete coordinates or large search spaces, SBBU may instead finish sequential
placement and stop after validating **all** original hard constraints. Rejected
trial completions restore the search state. The `max_time` budget is checked across
initialization, search, refinement, and final validation.

### Numerical scope

- `distance_tolerance` is a maximum absolute hard-constraint violation, not an
  inferred measurement uncertainty or a tolerance that is increased on failure.
- Input scalar values are retained in double-precision computation; converting
  float32 input cannot recover precision already lost in the supplied distances.
- Solving every noisy or degenerate instance is **not guaranteed**. In particular,
  uncertain sequential interval distances can be represented but are unsupported
  by the current solver (see [Constraint contract](#constraint-contract)).
- Failure to find a realization is not a global mathematical infeasibility
  certificate. Search budgets and numerical conditioning can cause failure.
- Exact coordinate permutation equivariance is not promised: coordinate frames,
  reflections, and nonunique sparse realizations can differ. Returned coordinates
  always use the original node indices and must pass the original constraints.

## Input Formats

### Constraint contract

Every `DistanceConstraint(i, j, lower_bound, upper_bound)` represents a finite,
positive closed interval. Both bounds are required. Equal endpoints denote an
exact distance; a positive-width interval is never rounded to exact based on a
tolerance. Exact distance inputs are normalized to `[d, d]`.

Adapters validate input format and construct a `ConstraintSet`; they do not assert
that SBBU can solve it. `SBBUSolver` requires an unambiguous exact distance for
every pair with `1 <= j - i <= 3`. It rejects missing or interval predecessors,
even if an interval is narrower than `distance_tolerance`. Other hard distances
may be intervals. These checks establish applicability, not global feasibility.

Duplicate observations for a pair are intersected. An empty intersection retains
the observations as soft alternatives, rather than inventing a distance. Such
alternatives cannot supply a required exact predecessor.

**Unsupported:** one-sided bounds and interval-predecessor reconstruction are
outside this API redesign. They are deferred until algorithm support is pursued;
`None` and infinity do not represent open bounds. Missing observations can instead
be omitted from an edge list or represented by paired NaNs in bounds matrices.

**Breaking API change:** replace `DistanceConstraint(i, j, d)` with
`DistanceConstraint(i, j, d, d)`, and three-field edges with `(i, j, d, d)`.
Use `MatrixAdapter` instead of the removed `ConstraintSet.from_*` constructors.
Use explicit bounds or `is_exact` instead of the former `distance` projection;
scalar violations are available as `constraint.violation(measured_distance)`.
`NMRAdapter.from_nmr_file` accepts only the path; configure acceptance tolerance
on the solver or `solve_from_nmr`, not on input conversion.

### 1. NMR Files
Standard NMR restraint files for protein structure determination:
```python
import sbbu

coords, stats = sbbu.solve_from_nmr('1abc.nmr')
```

### 2. Distance Matrices
Full or sparse distance matrices (NaN for missing distances):
```python
import numpy as np
import sbbu

# Full matrix
matrix = np.array([[0, 1.2, 1.8], [1.2, 0, 1.1], [1.8, 1.1, 0]])
coords, stats = sbbu.solve_from_distance_matrix(matrix)

# Sparse matrix
sparse_matrix = np.array([[0, 1.2, np.nan], [1.2, 0, 1.1], [np.nan, 1.1, 0]])
from sbbu.adapters import MatrixAdapter

constraints = MatrixAdapter.from_distance_matrix(sparse_matrix, ignore_nan=True)
# This graph can be stored, but SBBU cannot solve it: distance (0, 2) is missing.
```

### 3. Bounds Matrices

```python
import numpy as np
from sbbu.adapters import MatrixAdapter

lower_bounds = np.array([[0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0]])
upper_bounds = np.array([[0, 2.0, 2.0], [2.0, 0, 2.0], [2.0, 2.0, 0]])
constraints = MatrixAdapter.from_bounds_matrices(lower_bounds, upper_bounds)
# Construction succeeds. SBBU rejects these interval predecessors.
```

Bounds matrices must have matching square shapes, exact symmetry, zero diagonals,
and positive finite off-diagonal bounds with `lower_bounds <= upper_bounds`.
Matching NaNs in both bounds and both directions omit a pair by default; set
`ignore_nan=False` to reject them. Partially missing pairs and infinities are invalid.
For solver-compatible inputs, use
`sbbu.solve_from_bounds_matrices(lower_bounds, upper_bounds)`.

### 4. Edge Lists
Graph-like representations:
```python
import sbbu

edges = [
    (0, 1, 1.2, 1.2),  # exact distance
    (1, 2, 1.1, 1.1),
    (0, 2, 1.8, 1.8),
]
coords, stats = sbbu.solve_from_edge_list(num_nodes=3, edges=edges)
```

## Applications

- **Protein Structure Determination**: From NMR restraints
- **Machine Learning**: Convert distance predictions to 3D structures  

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

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.21.0
- typing_extensions ≥ 4.0.0 (Python < 3.11)

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev,test]"

# Run all tests
pytest
```
