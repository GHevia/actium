from __future__ import annotations

import numpy as np

from actium import (
    acceleration_km_s2_to_m_s2,
    acceleration_m_s2_to_km_s2,
    mu_km3_s2_to_m3_s2,
    mu_m3_s2_to_km3_s2,
    position_km_to_m,
    position_m_to_km,
    velocity_km_s_to_m_s,
    velocity_m_s_to_km_s,
)


def test_scalar_conversions_use_explicit_decimal_scales() -> None:
    assert position_km_to_m(1.25) == 1_250.0
    assert position_m_to_km(1_250.0) == 1.25
    assert velocity_km_s_to_m_s(7.5) == 7_500.0
    assert velocity_m_s_to_km_s(7_500.0) == 7.5
    assert acceleration_km_s2_to_m_s2(0.008) == 8.0
    assert acceleration_m_s2_to_km_s2(8.0) == 0.008
    assert mu_km3_s2_to_m3_s2(398_600.4418) == 398_600_441_800_000.0
    assert mu_m3_s2_to_km3_s2(398_600_441_800_000.0) == 398_600.4418


def test_array_conversions_preserve_shape_and_round_trip() -> None:
    source = np.asarray([[1.0, -2.0, 3.25], [4.5, 5.0, -6.0]])
    converted = position_km_to_m(source)

    assert isinstance(converted, np.ndarray)
    assert converted.shape == source.shape
    np.testing.assert_array_equal(position_m_to_km(converted), source)
