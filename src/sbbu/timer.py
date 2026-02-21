"""
Timer utilities for SBBU algorithm.

Simple timing functions to replace OpenMP timing.
"""

import time

try:
    from typing import Self
except ImportError:  # pragma: no cover - Python < 3.11
    from typing_extensions import Self


def get_wall_time() -> float:
    """
    Return the current monotonic time source.

    Returns
    -------
    float
        Current monotonic time in seconds.
    """
    return time.monotonic()


class Timer:
    """
    Context manager that measures elapsed wall-clock time.

    Parameters
    ----------
    name : str, optional
        Label used when converting the timer to a string.
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self) -> Self:
        self.start_time = get_wall_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = get_wall_time()

    @property
    def elapsed(self) -> float:
        """
        Elapsed wall-clock seconds between context entry and exit.

        Returns
        -------
        float
            Elapsed duration in seconds.
        """
        return self.end_time - self.start_time

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.6f} seconds"
