"""Explicit, deterministic conversions at the Python/Orekit boundary."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ScalarOrArray = TypeVar("ScalarOrArray", float, FloatArray)

KM_TO_M = 1_000.0
KM3_TO_M3 = KM_TO_M**3


def _scaled(values: ArrayLike | float, factor: float) -> FloatArray | float:
    """Scale a scalar or array without any implicit unit inference."""
    result = np.asarray(values, dtype=np.float64) * np.float64(factor)
    return float(result) if result.ndim == 0 else result


def position_km_to_m(values: ArrayLike | float) -> FloatArray | float:
    """Convert kilometres to metres."""
    return _scaled(values, KM_TO_M)


def position_m_to_km(values: ArrayLike | float) -> FloatArray | float:
    """Convert metres to kilometres."""
    return _scaled(values, 1.0 / KM_TO_M)


def velocity_km_s_to_m_s(values: ArrayLike | float) -> FloatArray | float:
    """Convert kilometres per second to metres per second."""
    return _scaled(values, KM_TO_M)


def velocity_m_s_to_km_s(values: ArrayLike | float) -> FloatArray | float:
    """Convert metres per second to kilometres per second."""
    return _scaled(values, 1.0 / KM_TO_M)


def acceleration_km_s2_to_m_s2(values: ArrayLike | float) -> FloatArray | float:
    """Convert kilometres per second squared to metres per second squared."""
    return _scaled(values, KM_TO_M)


def acceleration_m_s2_to_km_s2(values: ArrayLike | float) -> FloatArray | float:
    """Convert metres per second squared to kilometres per second squared."""
    return _scaled(values, 1.0 / KM_TO_M)


def mu_km3_s2_to_m3_s2(value: float) -> float:
    """Convert a gravitational parameter from km^3/s^2 to m^3/s^2."""
    return float(value) * KM3_TO_M3


def mu_m3_s2_to_km3_s2(value: float) -> float:
    """Convert a gravitational parameter from m^3/s^2 to km^3/s^2."""
    return float(value) / KM3_TO_M3
