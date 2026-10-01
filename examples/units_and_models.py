"""Show the explicit SI boundary and immutable history model."""

from __future__ import annotations

from actium import State, Trajectory


def main() -> None:
    state = State.from_si(
        position_m=[7_000_000.0, 0.0, 0.0],
        velocity_m_s=[0.0, 7_546.053290107542, 0.0],
    )
    position_m, velocity_m_s = state.to_si()
    print("Actium state:", state.vector)
    print("SI position:", position_m)
    print("SI velocity:", velocity_m_s)

    trajectory = Trajectory(
        times_s=[0.0],
        positions_km=[state.position_km],
        velocities_km_s=[state.velocity_km_s],
        accelerations_km_s2=[[-0.00813470289387755, 0.0, 0.0]],
    )
    print("[state, time] row:", trajectory.history[0])


if __name__ == "__main__":
    main()
