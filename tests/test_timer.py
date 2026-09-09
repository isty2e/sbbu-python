"""Tests for timer utilities."""

import pytest

from sbbu import timer


@pytest.mark.unit
def test_get_wall_time_uses_monotonic_clock(monkeypatch):
    """Wall-time helper should use monotonic clock for timeout safety."""
    monkeypatch.setattr(timer.time, "monotonic", lambda: 123.456)
    monkeypatch.setattr(timer.time, "time", lambda: 999.999)

    assert timer.get_wall_time() == 123.456
