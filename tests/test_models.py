from __future__ import annotations

import numpy as np
import pytest

from actium import State, Trajectory


def test_state_copies_validates_and_freezes_arrays() -> None:
    position = np.asarray([7_000.0, 1.0, -2.0])
    state = State(position, [0.0, 7.5, 1.0])
    position[0] = 0.0

    np.testing.assert_array_equal(state.position_km, [7_000.0, 1.0, -2.0])
    np.testing.assert_array_equal(state.vector, [7_000.0, 1.0, -2.0, 0.0, 7.5, 1.0])
    with pytest.raises(ValueError):
        state.position_km[0] = 1.0


@pytest.mark.parametrize(
    ("position", "velocity", "message"),
    [
        ([1.0, 2.0], [1.0, 2.0, 3.0], "position_km"),
        ([1.0, 2.0, 3.0], [1.0, 2.0], "velocity_km_s"),
        ([np.nan, 2.0, 3.0], [1.0, 2.0, 3.0], "finite"),
    ],
)
def test_state_rejects_malformed_vectors(position, velocity, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        State(position, velocity)


def test_state_si_conversion_is_explicit_and_reversible() -> None:
    state = State.from_si([7_000_000.0, 2_000.0, -3_000.0], [0.0, 7_500.0, 1_000.0])
    position_m, velocity_m_s = state.to_si()

    np.testing.assert_array_equal(state.position_km, [7_000.0, 2.0, -3.0])
    np.testing.assert_array_equal(state.velocity_km_s, [0.0, 7.5, 1.0])
    np.testing.assert_array_equal(position_m, [7_000_000.0, 2_000.0, -3_000.0])
    np.testing.assert_array_equal(velocity_m_s, [0.0, 7_500.0, 1_000.0])


def test_trajectory_exposes_km_and_si_history_views() -> None:
    trajectory = Trajectory.from_si(
        [0.0, 10.0],
        [[7_000_000.0, 0.0, 0.0], [6_999_000.0, 75_000.0, 0.0]],
        [[0.0, 7_500.0, 0.0], [-80.0, 7_499.0, 0.0]],
        [[-8.0, 0.0, 0.0], [-7.99, -0.08, 0.0]],
    )

    assert len(trajectory) == 2
    assert trajectory.states.shape == (2, 6)
    assert trajectory.history.shape == (2, 7)
    np.testing.assert_array_equal(trajectory.history[:, -1], [0.0, 10.0])
    np.testing.assert_array_equal(trajectory.states_si[:, :3], trajectory.positions_km * 1_000.0)
    np.testing.assert_array_equal(trajectory.accelerations_m_s2[0], [-8.0, 0.0, 0.0])
    np.testing.assert_array_equal(trajectory.state_at(1).vector, trajectory.states[1])


def test_trajectory_rejects_shape_mismatch_and_empty_times() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Trajectory([], [], [])
    with pytest.raises(ValueError, match="positions_km"):
        Trajectory([0.0, 1.0], [[1.0, 2.0, 3.0]], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="accelerations_km_s2"):
        Trajectory([0.0], [[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], np.zeros((2, 3)))
