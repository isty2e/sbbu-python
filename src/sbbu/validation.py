"""
Input validation and error handling utilities.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def validate_distance_bounds(lower: float, upper: float) -> None:
    """
    Validate lower and upper distance bounds.

    Parameters
    ----------
    lower : float
        Proposed lower distance bound.
    upper : float
        Proposed upper distance bound.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If either bound is invalid or if ``lower > upper``.
    """
    if not np.isfinite(lower):
        raise ValueError(f"Lower bound must be finite, got {lower}")
    if not np.isfinite(upper):
        raise ValueError(f"Upper bound must be finite, got {upper}")
    if lower <= 0:
        raise ValueError(f"Lower bound must be positive, got {lower}")
    if upper < 0:
        raise ValueError(f"Upper bound must be non-negative, got {upper}")
    if lower > upper:
        raise ValueError(f"Lower bound {lower} must not exceed upper bound {upper}")


def validate_coordinates(coords: NDArray[np.float64]) -> None:
    """
    Validate a single 3D coordinate vector.

    Parameters
    ----------
    coords : numpy.typing.NDArray[numpy.float64]
        Coordinate vector expected to have shape ``(3,)`` and finite values.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If shape is not ``(3,)`` or values are not finite.
    """
    if coords.shape != (3,):
        raise ValueError(f"Coordinates must be shape (3,), got {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite")


def validate_file_path(path: str | Path) -> Path:
    """
    Validate that a path exists and points to a file.

    Parameters
    ----------
    path : str | pathlib.Path
        Input filesystem path.

    Returns
    -------
    pathlib.Path
        Normalized ``Path`` object for the validated file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path exists but is not a file.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    if not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path_obj}")
    return path_obj


def validate_positive_float(value: float, name: str) -> None:
    """
    Validate that a named numeric value is strictly positive.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``value`` is not numeric.
    ValueError
        If ``value`` is non-finite or less than or equal to zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value)}")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative_int(value: int, name: str) -> None:
    """
    Validate that a named integer value is non-negative.

    Parameters
    ----------
    value : int
        Value to validate.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``value`` is not an integer.
    ValueError
        If ``value`` is negative.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value)}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
