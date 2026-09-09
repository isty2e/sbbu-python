"""
NMR file adapter for SBBU algorithm.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..constraints import ConstraintSet, DistanceConstraint
from ..data.nmr import NMRParser

logger = logging.getLogger(__name__)


class NMRAdapter:
    """
    Read NMR constraints and write coordinate solutions.

    Notes
    -----
    The adapter reuses :class:`~sbbu.data.nmr.NMRParser` for parsing and
    converts parsed constraints into canonical :class:`ConstraintSet` objects.
    """

    @staticmethod
    def from_nmr_file(nmr_file: str | Path) -> ConstraintSet:
        """
        Load constraints from an NMR file.

        Parameters
        ----------
        nmr_file : str | Path
            Path to the input ``.nmr`` file.

        Returns
        -------
        ConstraintSet
            Constraint set containing the original finite bounds, independently of
            solver applicability.

        Raises
        ------
        FileNotFoundError
            If ``nmr_file`` does not exist.
        ValueError
            If the NMR content cannot be parsed into valid finite bounds.
        """
        parser = NMRParser(nmr_file)
        # The parser emits both orientations; retain each input observation once.
        constraints = [
            DistanceConstraint(edge.i, edge.j, edge.lower_bound, edge.upper_bound)
            for edge in parser.edges
            if edge.i <= edge.j
        ]
        return ConstraintSet(parser.num_nodes, constraints)

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
