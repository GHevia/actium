from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from actium import (
    State,
    propagate_j2,
    propagate_numerical,
    propagate_sun_moon,
    sun_moon_positions,
    two_body_acceleration,
)

MU_EARTH_KM3_S2 = 398_600.4418
EARTH_RADIUS_KM = 6_378.1363
EARTH_J2 = 1.08262668e-3
SUN_MU_KM3_S2 = 132_712_440_018.0
MOON_MU_KM3_S2 = 4_904.8695
EPOCH = "2026-01-01T00:00:00Z"


@pytest.fixture(scope="module")
def inclined_state() -> State:
    return State([7_000.0, -1_200.0, 1_300.0], [2.1, 7.0, 1.0])


def _closed_form_j2(positions_km: np.ndarray) -> np.ndarray:
    radius_sq = np.sum(positions_km**2, axis=1)
    radius = np.sqrt(radius_sq)
    z_ratio = positions_km[:, 2] ** 2 / radius_sq
    scale = 1.5 * EARTH_J2 * MU_EARTH_KM3_S2 * EARTH_RADIUS_KM**2 / radius**5
    return scale[:, None] * np.column_stack(
        (
            positions_km[:, 0] * (5.0 * z_ratio - 1.0),
            positions_km[:, 1] * (5.0 * z_ratio - 1.0),
            positions_km[:, 2] * (5.0 * z_ratio - 3.0),
        )
    )


def _third_body(position_km, body_position_km, mu_km3_s2):
    relative = body_position_km - position_km
    return mu_km3_s2 * (
        relative / np.linalg.norm(relative, axis=1)[:, None] ** 3
        - body_position_km / np.linalg.norm(body_position_km, axis=1)[:, None] ** 3
    )


def test_j2_acceleration_matches_closed_form_at_propagated_states(inclined_state) -> None:
    trajectory = propagate_j2(
        inclined_state,
        [0.0, 60.0, 600.0],
        mu_km3_s2=MU_EARTH_KM3_S2,
        equatorial_radius_km=EARTH_RADIUS_KM,
        j2=EARTH_J2,
    )
    central = two_body_acceleration(
        trajectory.positions_km,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )

    np.testing.assert_allclose(
        trajectory.accelerations_km_s2 - central,
        _closed_form_j2(trajectory.positions_km),
        rtol=3.0e-13,
        atol=2.0e-18,
    )


def test_zero_j2_reduces_to_numerical_two_body(inclined_state) -> None:
    times = [0.0, 60.0, 600.0]
    reference = propagate_numerical(
        inclined_state,
        times,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    candidate = propagate_j2(
        inclined_state,
        times,
        mu_km3_s2=MU_EARTH_KM3_S2,
        equatorial_radius_km=EARTH_RADIUS_KM,
        j2=0.0,
    )
    np.testing.assert_allclose(candidate.states, reference.states, rtol=0.0, atol=1.0e-12)


@pytest.mark.skipif(importlib.util.find_spec("orekitdata") is None, reason="orekitdata unavailable")
def test_sun_moon_acceleration_uses_orekit_positions_and_explicit_mu(inclined_state) -> None:
    times = np.asarray([0.0, 600.0, 3_600.0])
    trajectory = propagate_sun_moon(
        inclined_state,
        times,
        epoch=EPOCH,
        mu_km3_s2=MU_EARTH_KM3_S2,
        sun_mu_km3_s2=SUN_MU_KM3_S2,
        moon_mu_km3_s2=MOON_MU_KM3_S2,
    )
    body_positions = sun_moon_positions(times, epoch=EPOCH)
    central = two_body_acceleration(
        trajectory.positions_km,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    expected = (
        central
        + _third_body(
            trajectory.positions_km,
            body_positions["sun"],
            SUN_MU_KM3_S2,
        )
        + _third_body(
            trajectory.positions_km,
            body_positions["moon"],
            MOON_MU_KM3_S2,
        )
    )

    np.testing.assert_allclose(
        trajectory.accelerations_km_s2,
        expected,
        rtol=3.0e-14,
        atol=1.0e-18,
    )


@pytest.mark.skipif(importlib.util.find_spec("orekitdata") is None, reason="orekitdata unavailable")
def test_individual_sun_and_moon_switches_change_the_trajectory(inclined_state) -> None:
    kwargs = {
        "epoch": EPOCH,
        "mu_km3_s2": MU_EARTH_KM3_S2,
        "sun_mu_km3_s2": SUN_MU_KM3_S2,
        "moon_mu_km3_s2": MOON_MU_KM3_S2,
    }
    sun = propagate_sun_moon(
        inclined_state,
        [0.0, 3_600.0],
        include_sun=True,
        include_moon=False,
        **kwargs,
    )
    moon = propagate_sun_moon(
        inclined_state,
        [0.0, 3_600.0],
        include_sun=False,
        include_moon=True,
        **kwargs,
    )

    assert not np.array_equal(sun.states[-1], moon.states[-1])


def test_perturbation_inputs_are_explicitly_validated(inclined_state) -> None:
    with pytest.raises(ValueError, match="equatorial_radius_km"):
        propagate_j2(
            inclined_state,
            [0.0],
            mu_km3_s2=MU_EARTH_KM3_S2,
            equatorial_radius_km=0.0,
            j2=EARTH_J2,
        )
    with pytest.raises(ValueError, match="at least one"):
        propagate_sun_moon(
            inclined_state,
            [0.0],
            epoch=EPOCH,
            mu_km3_s2=MU_EARTH_KM3_S2,
            include_sun=False,
            include_moon=False,
        )
