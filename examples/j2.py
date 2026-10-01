"""Propagate central gravity plus an explicit Earth J2 term."""

from __future__ import annotations

import numpy as np

from actium import State, compare_trajectories, propagate_j2, propagate_numerical

MU_EARTH_KM3_S2 = 398_600.4418
EARTH_RADIUS_KM = 6_378.1363
EARTH_J2 = 1.08262668e-3


def main() -> None:
    initial = State(
        position_km=[-2_806.3182300452064, 6_750.454822659758, -1_886.8908112853997],
        velocity_km_s=[-4.030598697633472, 0.9770643994799905, 5.994992595088863],
    )
    times_s = np.arange(0.0, 6.0 * 3_600.0 + 1.0, 180.0)
    two_body = propagate_numerical(initial, times_s, mu_km3_s2=MU_EARTH_KM3_S2)
    j2 = propagate_j2(
        initial,
        times_s,
        mu_km3_s2=MU_EARTH_KM3_S2,
        equatorial_radius_km=EARTH_RADIUS_KM,
        j2=EARTH_J2,
    )

    print(compare_trajectories(two_body, j2).summary())
    print("final J2 acceleration [km/s^2]:", j2.accelerations_km_s2[-1])


if __name__ == "__main__":
    main()
