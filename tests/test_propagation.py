from __future__ import annotations

import inspect

import numpy as np
import pytest

from actium import (
    State,
    compare_trajectories,
    propagate_keplerian,
    propagate_numerical,
    two_body_acceleration,
    two_body_acceleration_si,
)

MU_EARTH_KM3_S2 = 398_600.4418


@pytest.fixture(scope="module")
def circular_state() -> State:
    radius_km = 7_000.0
    return State(
        [radius_km, 0.0, 0.0],
        [0.0, np.sqrt(MU_EARTH_KM3_S2 / radius_km), 0.0],
    )


def test_mu_is_required_and_eme2000_is_default() -> None:
    for function in (propagate_keplerian, propagate_numerical):
        signature = inspect.signature(function)
        assert signature.parameters["mu_km3_s2"].default is inspect.Parameter.empty
        assert signature.parameters["frame"].default == "EME2000"


def test_analytical_kepler_preserves_initial_state_and_caller_time_order(circular_state) -> None:
    times_s = np.asarray([120.0, 0.0, -60.0, 120.0])
    trajectory = propagate_keplerian(
        circular_state,
        times_s,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )

    assert trajectory.frame == "EME2000"
    np.testing.assert_array_equal(trajectory.times_s, times_s)
    np.testing.assert_allclose(trajectory.state_at(1).vector, circular_state.vector, atol=1.0e-12)
    np.testing.assert_array_equal(trajectory.states[0], trajectory.states[3])
    assert trajectory.accelerations_km_s2 is not None


def test_analytical_kepler_closes_a_circular_period(circular_state) -> None:
    period_s = 2.0 * np.pi * np.sqrt(7_000.0**3 / MU_EARTH_KM3_S2)
    trajectory = propagate_keplerian(
        circular_state,
        [0.0, period_s],
        mu_km3_s2=MU_EARTH_KM3_S2,
    )

    np.testing.assert_allclose(trajectory.states[1], trajectory.states[0], atol=1.0e-9)


def test_numerical_two_body_agrees_with_orekit_analytical_path(circular_state) -> None:
    times_s = np.asarray([0.0, 30.0, 300.0, 900.0, 1_800.0, 3_600.0])
    analytical = propagate_keplerian(
        circular_state,
        times_s,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    numerical = propagate_numerical(
        circular_state,
        times_s,
        mu_km3_s2=MU_EARTH_KM3_S2,
        position_tolerance_m=1.0e-7,
    )
    comparison = compare_trajectories(analytical, numerical)

    assert comparison.max_position_error_m < 1.0e-3
    assert comparison.max_velocity_error_m_s < 1.0e-6
    assert comparison.max_acceleration_error_m_s2 < 1.0e-9

    expected_acceleration = two_body_acceleration(
        numerical.positions_km,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    np.testing.assert_allclose(
        numerical.accelerations_km_s2,
        expected_acceleration,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_numerical_samples_are_independent_of_output_order(circular_state) -> None:
    ordered = propagate_numerical(
        circular_state,
        [0.0, 60.0, 120.0],
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    permuted = propagate_numerical(
        circular_state,
        [120.0, 0.0, 60.0],
        mu_km3_s2=MU_EARTH_KM3_S2,
    )

    np.testing.assert_array_equal(permuted.states[[1, 2, 0]], ordered.states)


def test_two_body_acceleration_supports_single_and_batched_si_and_km_inputs() -> None:
    km = two_body_acceleration([7_000.0, 0.0, 0.0], mu_km3_s2=MU_EARTH_KM3_S2)
    si = two_body_acceleration_si(
        [[7_000_000.0, 0.0, 0.0], [0.0, 7_000_000.0, 0.0]],
        mu_m3_s2=MU_EARTH_KM3_S2 * 1.0e9,
    )

    np.testing.assert_allclose(si[0], km * 1_000.0)
    np.testing.assert_allclose(si[1], [0.0, si[0, 0], 0.0])


@pytest.mark.parametrize("mu", [0.0, -1.0, np.nan])
def test_propagators_reject_invalid_mu_without_starting_propagation(circular_state, mu) -> None:
    with pytest.raises(ValueError, match="mu_km3_s2"):
        propagate_keplerian(circular_state, [0.0], mu_km3_s2=mu)


def test_numerical_rejects_inconsistent_step_configuration(circular_state) -> None:
    with pytest.raises(ValueError, match="initial_step_s"):
        propagate_numerical(
            circular_state,
            [0.0],
            mu_km3_s2=MU_EARTH_KM3_S2,
            min_step_s=1.0,
            initial_step_s=0.5,
        )
