"""
NMR file adapter for SBBU algorithm.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..constraints import ConstraintSet, DistanceConstraint
from ..data.ddgp import DDGProblem

logger = logging.getLogger(__name__)


class NMRAdapter:
    """
    Adapt NMR constraint files to and from SBBU-compatible artifacts.

    Notes
    -----
    The adapter reuses :class:`~sbbu.data.ddgp.DDGProblem` for parsing and
    converts parsed constraints into canonical :class:`ConstraintSet` objects.
    """

    @staticmethod
    def from_nmr_file(
        nmr_file: str | Path, distance_tolerance: float = 1e-7
    ) -> ConstraintSet:
        """
        Load constraints from an NMR file.

        Parameters
        ----------
        nmr_file : str | Path
            Path to the input ``.nmr`` file.
        distance_tolerance : float, default=1e-7
            Tolerance passed to :class:`~sbbu.data.ddgp.DDGProblem`.

        Returns
        -------
        ConstraintSet
            Constraint set built from sequential and long-range DDGP edges.

        Raises
        ------
        FileNotFoundError
            If ``nmr_file`` does not exist.
        ValueError
            If the NMR/DDGP content cannot be parsed into valid constraints.
        """
        # Use existing DDGProblem for NMR parsing
        ddgp = DDGProblem(nmr_file, distance_tolerance)

        # Validate that duplicated sequential pairs remain mutually consistent.
        sequential_bounds_by_pair: dict[tuple[int, int], tuple[float, float]] = {}
        sequential_tolerance = 1e-10
        for edge in ddgp.edges:
            i, j = edge.i, edge.j
            if i > j:
                i, j = j, i
            if j - i > 3:
                continue

            pair = (i, j)
            bounds = (edge.lower_bound, edge.upper_bound)
            previous_bounds = sequential_bounds_by_pair.get(pair)
            if previous_bounds is None:
                sequential_bounds_by_pair[pair] = bounds
                continue

            if (
                abs(previous_bounds[0] - bounds[0]) > sequential_tolerance
                or abs(previous_bounds[1] - bounds[1]) > sequential_tolerance
            ):
                raise ValueError(
                    "Conflicting sequential constraints for "
                    f"({i}, {j}): {previous_bounds} vs {bounds}"
                )

        # Convert to constraint set
        constraints = []

        # Add distance constraints from DDGP edges
        edges = ddgp.get_long_range_edges()

        # Add sequential edges (0-1, 1-2, 2-3, etc.)
        for i in range(ddgp.num_nodes - 1):
            for j in range(i + 1, min(i + 4, ddgp.num_nodes)):
                bounds = ddgp.find_bounds(i, j)
                if bounds is None:
                    raise ValueError(
                        f"Missing required sequential constraint ({i}, {j}) in {nmr_file}"
                    )

                lower_bound, upper_bound = bounds
                if abs(upper_bound - lower_bound) > 1e-10:
                    raise ValueError(
                        f"Sequential constraint ({i}, {j}) must be exact, got "
                        f"[{lower_bound}, {upper_bound}]"
                    )
                constraints.append(
                    DistanceConstraint(
                        i=i,
                        j=j,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound
                        if abs(upper_bound - lower_bound) > 1e-10
                        else None,
                    )
                )

        # Add long-range edges
        for edge in edges:
            constraints.append(
                DistanceConstraint(
                    i=edge.i,
                    j=edge.j,
                    lower_bound=edge.lower_bound,
                    upper_bound=edge.upper_bound
                    if abs(edge.upper_bound - edge.lower_bound) > 1e-10
                    else None,
                )
            )

        return ConstraintSet(ddgp.num_nodes, constraints)

    @staticmethod
    def to_solution_file(
        coordinates: NDArray[np.float64],
        output_file: str | Path,
        verbose: bool = True,
    ) -> None:
        """
        Write coordinates to a ``.sol`` solution file.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape ``(N, 3)``.
        output_file : str | Path
            Base output path. If ``.nmr`` is provided, ``_sbbu.sol`` is used.
        verbose : bool, default=True
            Whether to emit an info log with the destination path.

        Returns
        -------
        None
            This method writes the solution file to disk.

        Raises
        ------
        ValueError
            If ``coordinates`` is not a finite array with shape ``(N, 3)``.
        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                f"coordinates must have shape (N, 3), got {coordinates.shape}"
            )
        if not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")

        path = Path(output_file)

        # Generate output filename
        if path.suffix == ".nmr":
            output_path = path.with_name(f"{path.stem}_sbbu.sol")
        else:
            output_path = path.with_suffix(".sol")

        if verbose:
            logger.info("SBBU: saving solution to %s", output_path)

        # Write coordinates
        with open(output_path, "w") as f:
            for i in range(coordinates.shape[0]):
                x, y, z = coordinates[i]
                f.write(f"{x:.18g} {y:.18g} {z:.18g}\n")
