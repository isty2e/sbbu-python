"""
Tests for validation helpers.
"""

import numpy as np
import pytest

from sbbu.core.solver import SBBUConfig
from sbbu.validation import validate_distance_bounds, validate_positive_float


@pytest.mark.unit
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_validate_positive_float_rejects_non_finite_values(value):
    """Positive-float validation should reject NaN and infinities."""
    with pytest.raises(ValueError, match="finite"):
        validate_positive_float(value, "max_time")


@pytest.mark.unit
def test_validate_positive_float_rejects_bool_values():
    """Validation should not accept bool as a numeric float input."""
    with pytest.raises(TypeError, match="number"):
        validate_positive_float(True, "max_time")


@pytest.mark.unit
def test_validate_positive_float_accepts_numpy_float():
    """Validation should accept finite positive numpy float values."""
    validate_positive_float(np.float64(1.5), "max_time")


@pytest.mark.unit
def test_validate_distance_bounds_rejects_non_positive_lower_bound():
    """Distance bounds validation should reject zero lower bounds."""
    with pytest.raises(ValueError, match="positive"):
        validate_distance_bounds(0.0, 1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (np.nan, 1.0),
        (1.0, np.nan),
        (np.inf, 2.0),
        (1.0, np.inf),
        (-np.inf, 2.0),
        (1.0, -np.inf),
    ],
)
def test_validate_distance_bounds_rejects_non_finite_bounds(lower, upper):
    """Distance bounds validation should reject NaN/inf values."""
    with pytest.raises(ValueError, match="finite"):
        validate_distance_bounds(lower, upper)


@pytest.mark.unit
def test_sbbu_config_rejects_non_finite_max_time():
    """SBBUConfig should reject non-finite max_time values."""
    with pytest.raises(ValueError, match="finite"):
        SBBUConfig(max_time=np.nan)

    with pytest.raises(ValueError, match="finite"):
        SBBUConfig(max_time=np.inf)


@pytest.mark.unit
@pytest.mark.parametrize("value", [0.0, -1e-6, np.nan, np.inf])
def test_sbbu_config_rejects_invalid_distance_tolerance(value):
    """SBBUConfig should reject non-positive or non-finite tolerance values."""
    with pytest.raises(ValueError):
        SBBUConfig(distance_tolerance=value)


@pytest.mark.unit
def test_sbbu_config_rejects_invalid_max_iterations_values():
    """max_iterations should reject non-positive/non-integer values."""
    with pytest.raises(ValueError, match="positive integer"):
        SBBUConfig(max_iterations=0)

    with pytest.raises(ValueError, match="positive integer"):
        SBBUConfig(max_iterations=-1)

    with pytest.raises(ValueError, match="positive integer"):
        SBBUConfig(max_iterations=1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive integer"):
        SBBUConfig(max_iterations="10")  # type: ignore[arg-type]


@pytest.mark.unit
def test_sbbu_config_rejects_non_positive_soft_pruning_rounds():
    """Soft-pruning rounds must be a strictly positive integer."""
    with pytest.raises(ValueError, match="positive"):
        SBBUConfig(soft_pruning_max_rounds=0)

    with pytest.raises(ValueError, match="positive"):
        SBBUConfig(soft_pruning_max_rounds=-1)


@pytest.mark.unit
def test_sbbu_config_rejects_negative_soft_pruning_thresholds():
    """Soft-pruning thresholds should be non-negative."""
    with pytest.raises(ValueError, match="non-negative"):
        SBBUConfig(soft_pruning_acceptance_tolerance=-1e-4)

    with pytest.raises(ValueError, match="non-negative"):
        SBBUConfig(soft_pruning_candidate_window=-1e-4)

    with pytest.raises(ValueError, match="non-negative"):
        SBBUConfig(soft_pruning_min_margin=-1e-4)


@pytest.mark.unit
def test_sbbu_config_accepts_zero_soft_pruning_thresholds():
    """Soft-pruning threshold fields should allow exact zero."""
    config = SBBUConfig(
        soft_pruning_acceptance_tolerance=0.0,
        soft_pruning_candidate_window=0.0,
        soft_pruning_min_margin=0.0,
    )

    assert config.soft_pruning_acceptance_tolerance == 0.0
    assert config.soft_pruning_candidate_window == 0.0
    assert config.soft_pruning_min_margin == 0.0


@pytest.mark.unit
def test_sbbu_config_rejects_invalid_integer_fields():
    """Integer config fields should reject bool and non-positive values."""
    with pytest.raises(ValueError, match="positive integer"):
        SBBUConfig(max_iterations=True)

    with pytest.raises(ValueError, match="positive integer"):
        SBBUConfig(soft_pruning_max_rounds=True)


@pytest.mark.unit
def test_sbbu_config_rejects_non_integer_soft_pruning_rounds():
    """soft_pruning_max_rounds should reject non-integer numeric values."""
    with pytest.raises(ValueError, match="positive integer"):
        SBBUConfig(soft_pruning_max_rounds=1.5)  # type: ignore[arg-type]


@pytest.mark.unit
def test_sbbu_config_rejects_non_numeric_distance_tolerance():
    """distance_tolerance should reject non-numeric values."""
    with pytest.raises(TypeError, match="number"):
        SBBUConfig(distance_tolerance="1e-6")  # type: ignore[arg-type]


@pytest.mark.unit
def test_sbbu_config_rejects_non_numeric_soft_pruning_thresholds():
    """Soft-pruning thresholds should reject non-numeric values."""
    with pytest.raises(TypeError, match="number"):
        SBBUConfig(soft_pruning_acceptance_tolerance="0.1")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="number"):
        SBBUConfig(soft_pruning_candidate_window="0.1")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="number"):
        SBBUConfig(soft_pruning_min_margin="0.1")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("field_name", [
    "soft_pruning_acceptance_tolerance",
    "soft_pruning_candidate_window",
    "soft_pruning_min_margin",
])
def test_sbbu_config_rejects_non_finite_soft_pruning_thresholds(field_name):
    """Soft-pruning thresholds should reject non-finite numeric values."""
    with pytest.raises(ValueError, match="finite"):
        SBBUConfig(**{field_name: np.nan})

    with pytest.raises(ValueError, match="finite"):
        SBBUConfig(**{field_name: np.inf})
