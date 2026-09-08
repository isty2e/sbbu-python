"""
Geometry and linear-algebra helpers for SBBU computations.
"""

from collections.abc import Iterator

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
    norm = float(np.linalg.norm(v))
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


def clique_realizations(
    distances: NDArray[np.float64],
) -> Iterator[NDArray[np.float64]]:
    """Generate fixed-anchor trilateration candidates for a complete graph.

    Parameters
    ----------
    distances : NDArray[np.float64]
        Symmetric ``(N, N)`` distances with a zero diagonal, finite positive
        off-diagonal entries, and ``N >= 4``. Supplied by the canonical exact
        hard-constraint projection.

    Yields
    ------
    NDArray[np.float64]
        ``(N, 3)`` candidates in original node order, descending in numerical
        rank. Callers must check all original constraints before acceptance.

    Notes
    -----
    Four geometrically spread anchors share one coordinate frame. Only their
    three-by-three Gram matrix is factored; there is no global spectral fit
    or chain of newly estimated coordinates. Work and storage are quadratic
    in ``N``, including distance scaling, and the factorization size is fixed.
    """
    scale = float(np.max(distances))
    squared = (distances / scale) ** 2
    first, second = (
        int(index) for index in np.unravel_index(np.argmax(squared), squared.shape)
    )
    baseline = np.sqrt(squared[first, second])
    along = (squared[first] + squared[first, second] - squared[second]) / (
        2.0 * baseline
    )
    heights_squared = squared[first] - along**2
    heights_squared[[first, second]] = -np.inf
    third = int(np.argmax(heights_squared))
    height = np.sqrt(max(0.0, heights_squared[third]))

    if height > 0.0:
        across = (
            squared[first]
            + squared[first, third]
            - squared[third]
            - 2.0 * along * along[third]
        ) / (2.0 * height)
        off_plane = squared[first] - along**2 - across**2
    else:
        off_plane = np.zeros(len(distances))
    off_plane[[first, second, third]] = -np.inf
    fourth = int(np.argmax(off_plane))

    anchors = np.array([second, third, fourth])
    radii = squared[first, anchors]
    gram = (radii[:, None] + radii[None, :] - squared[np.ix_(anchors, anchors)]) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    cross_products = (
        squared[:, first, None] + radii[None, :] - squared[:, anchors]
    ) / 2.0
    threshold = 3.0 * np.finfo(np.float64).eps * eigenvalues[-1]
    rank = int(np.count_nonzero(eigenvalues > threshold))

    for dimension in range(rank, 0, -1):
        coordinates = np.zeros((len(distances), 3), dtype=np.float64)
        coordinates[:, :dimension] = cross_products @ (
            eigenvectors[:, -dimension:] / np.sqrt(eigenvalues[-dimension:])
        )
        coordinates *= scale
        coordinates -= coordinates[0].copy()

        # Retain the public first-three-node gauge without forcing noisy lengths.
        basis, _ = np.linalg.qr(coordinates[1:3].T, mode="complete")
        if np.dot(coordinates[1], basis[:, 0]) < 0.0:
            basis[:, 0] *= -1.0
        if np.dot(coordinates[2], basis[:, 1]) < 0.0:
            basis[:, 1] *= -1.0
        transformed = coordinates @ basis
        if transformed[1, 0] <= 0.0:
            continue
        if transformed[3, 2] < 0.0:
            transformed[:, 2] *= -1.0
        yield transformed
