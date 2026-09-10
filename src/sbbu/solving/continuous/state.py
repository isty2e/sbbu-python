"""Original-bound candidate retention within continuous solving."""

import numpy as np
from numpy.typing import NDArray

from ...constraints import ConstraintSet


class BestCandidate:
    """Retain an owned candidate by its original maximum bound violation.

    Parameters
    ----------
    constraints : ConstraintSet
        One immutable problem whose node order and bounds govern every comparison.
    initial : NDArray[np.float64]
        Initial ``(n, 3)`` coordinates, copied without changing their gauge.

    Notes
    -----
    Retention does not certify public success. The solve operation still checks
    the original constraints and deadline before publication.
    """

    def __init__(
        self, constraints: ConstraintSet, initial: NDArray[np.float64]
    ) -> None:
        self._constraints = constraints
        self.coordinates = initial.copy()
        self.error = constraints.maximum_violation(self.coordinates)

    def consider(self, coordinates: NDArray[np.float64]) -> float:
        """Keep a copy only when the original maximum error strictly improves.

        Parameters
        ----------
        coordinates : NDArray[np.float64]
            Proposal in the same problem's original node order.

        Returns
        -------
        float
            Proposal error, whether or not the proposal replaces the incumbent.
        """
        error = self._constraints.maximum_violation(coordinates)
        if np.isfinite(error) and error < self.error:
            self.coordinates = coordinates.copy()
            self.error = error
        return error
