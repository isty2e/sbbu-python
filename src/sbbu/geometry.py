"""
Geometry and linear-algebra helpers for SBBU computations.
"""

import numpy as np
from numpy.typing import NDArray


def normalize_vector(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize a 3D vector to unit length.

    Parameters
    ----------
    v : numpy.typing.NDArray[numpy.float64]
        Input vector.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Unit vector in the same direction as ``v``.

    Raises
    ------
    ValueError
        If ``v`` is the zero vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


def quadratic_solver_point(
    point: NDArray[np.float64],
    normal: NDArray[np.float64],
    ref_a: NDArray[np.float64],
    dist_a_sq: float,
    ref_b: NDArray[np.float64],
    dist_b_sq: float,
    ref_c: NDArray[np.float64],
    dist_c_sq: float,
) -> None:
    """
    Solve for a point given distance constraints to three reference points.

    This function modifies ``point`` in-place.

    Parameters
    ----------
    point : numpy.typing.NDArray[numpy.float64]
        Output point updated in-place with the solved coordinates.
    normal : numpy.typing.NDArray[numpy.float64]
        Normal vector defining the plane equation.
    ref_a : numpy.typing.NDArray[numpy.float64]
        First reference point.
    dist_a_sq : float
        Squared distance from the solution to ``ref_a``.
    ref_b : numpy.typing.NDArray[numpy.float64]
        Second reference point.
    dist_b_sq : float
        Squared distance from the solution to ``ref_b``.
    ref_c : numpy.typing.NDArray[numpy.float64]
        Third reference point.
    dist_c_sq : float
        Squared distance from the solution to ``ref_c``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the derived linear system is singular and cannot be solved.
    """
    # Calculate squared norms of reference points using numpy
    a_norm_sq = float(np.linalg.norm(ref_a) ** 2)
    b_norm_sq = float(np.linalg.norm(ref_b) ** 2)
    c_norm_sq = float(np.linalg.norm(ref_c) ** 2)

    # Set up linear system matrix using vectorized operations
    matrix = np.vstack([normal, ref_a - ref_b, ref_a - ref_c])

    # Set up right-hand side vector
    rhs = np.array(
        [
            np.dot(ref_a, normal),
            (dist_b_sq - dist_a_sq + a_norm_sq - b_norm_sq) / 2.0,
            (dist_c_sq - dist_a_sq + a_norm_sq - c_norm_sq) / 2.0,
        ]
    )

    # Solve using numpy's linear solver
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular - cannot solve linear system")

    # Update point in-place
    point[:] = solution
