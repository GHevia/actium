"""Propagate one orbit with the analytical and numerical Orekit paths."""

from __future__ import annotations

import numpy as np

from actium import State, compare_trajectories, propagate_keplerian, propagate_numerical

MU_EARTH_KM3_S2 = 398_600.4418


def main() -> None:
    radius_km = 7_000.0
    initial = State(
        [radius_km, 0.0, 0.0],
        [0.0, np.sqrt(MU_EARTH_KM3_S2 / radius_km), 0.0],
    )
    times_s = np.arange(0.0, 3_601.0, 300.0)

    analytical = propagate_keplerian(
        initial,
        times_s,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    numerical = propagate_numerical(
        initial,
        times_s,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    comparison = compare_trajectories(analytical, numerical)

    print("time_s     |dr|_m       |dv|_m/s     |da|_m/s^2")
    acceleration_norms = comparison.acceleration_error_norm_m_s2
    assert acceleration_norms is not None
    for time_s, dr_m, dv_m_s, da_m_s2 in zip(
        times_s,
        comparison.position_error_norm_m,
        comparison.velocity_error_norm_m_s,
        acceleration_norms,
        strict=True,
    ):
        print(f"{time_s:6.0f}  {dr_m:12.5e}  {dv_m_s:12.5e}  {da_m_s2:12.5e}")


if __name__ == "__main__":
    main()
