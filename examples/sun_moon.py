"""Propagate explicit Sun and Moon third-body gravity from Orekit ephemerides."""

from __future__ import annotations

import numpy as np

from actium import State, propagate_sun_moon, sun_moon_positions

MU_EARTH_KM3_S2 = 398_600.4418
EPOCH = "2026-01-01T00:00:00Z"


def main() -> None:
    initial = State(
        position_km=[-2_806.3182300452064, 6_750.454822659758, -1_886.8908112853997],
        velocity_km_s=[-4.030598697633472, 0.9770643994799905, 5.994992595088863],
    )
    times_s = np.arange(0.0, 6.0 * 3_600.0 + 1.0, 180.0)
    trajectory = propagate_sun_moon(
        initial,
        times_s,
        epoch=EPOCH,
        mu_km3_s2=MU_EARTH_KM3_S2,
        include_sun=True,
        include_moon=True,
    )
    bodies = sun_moon_positions(times_s[[0, -1]], epoch=EPOCH)

    print("final spacecraft state [km, km/s]:", trajectory.states[-1])
    print("Sun positions at endpoints [km]:\n", bodies["sun"])
    print("Moon positions at endpoints [km]:\n", bodies["moon"])


if __name__ == "__main__":
    main()
