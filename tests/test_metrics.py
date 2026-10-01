from __future__ import annotations

import numpy as np
import pytest

from actium import Trajectory, compare_trajectories


def _trajectory(*, offset_m: float = 0.0, frame: str = "EME2000", with_acceleration=True):
    acceleration = [[-8.0, 0.0, 0.0], [-7.9, -0.1, 0.0]] if with_acceleration else None
    return Trajectory.from_si(
        [0.0, 60.0],
        [[7_000_000.0 + offset_m, 0.0, 0.0], [6_980_000.0 + offset_m, 450_000.0, 0.0]],
        [[0.0, 7_500.0, 0.0], [-500.0, 7_480.0, 0.0]],
        acceleration,
        frame=frame,
    )


def test_comparison_reports_candidate_minus_reference_full_history() -> None:
    reference = _trajectory()
    candidate = _trajectory(offset_m=2.0)
    comparison = compare_trajectories(reference, candidate)

    np.testing.assert_allclose(comparison.position_difference_m, [[2.0, 0.0, 0.0]] * 2)
    np.testing.assert_array_equal(comparison.velocity_difference_m_s, np.zeros((2, 3)))
    np.testing.assert_array_equal(comparison.acceleration_difference_m_s2, np.zeros((2, 3)))
    assert comparison.state_difference_si.shape == (2, 6)
    assert comparison.max_position_error_m == pytest.approx(2.0)
    assert comparison.rms_position_error_m == pytest.approx(2.0)
    assert comparison.max_velocity_error_m_s == 0.0
    assert comparison.max_acceleration_error_m_s2 == 0.0
    assert comparison.within(position_m=2.0, velocity_m_s=0.0, acceleration_m_s2=0.0)
    assert not comparison.within(position_m=1.99, velocity_m_s=0.0)


def test_comparison_rejects_non_identical_grids_frames_and_acceleration_contracts() -> None:
    reference = _trajectory()
    shifted_times = Trajectory.from_si(
        [0.0, 60.01],
        reference.states_si[:, :3],
        reference.states_si[:, 3:],
        reference.accelerations_m_s2,
    )
    with pytest.raises(ValueError, match="output times"):
        compare_trajectories(reference, shifted_times)
    compare_trajectories(reference, shifted_times, time_tolerance_s=0.02)

    with pytest.raises(ValueError, match="frames"):
        compare_trajectories(reference, _trajectory(frame="GCRF"))
    with pytest.raises(ValueError, match="accelerations"):
        compare_trajectories(reference, _trajectory(with_acceleration=False))


def test_comparison_without_accelerations_reports_none() -> None:
    comparison = compare_trajectories(
        _trajectory(with_acceleration=False),
        _trajectory(with_acceleration=False),
    )
    assert comparison.acceleration_error_norm_m_s2 is None
    assert comparison.max_acceleration_error_m_s2 is None
    assert comparison.summary()["rms_acceleration_error_m_s2"] is None
