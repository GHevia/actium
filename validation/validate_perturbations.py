"""Compare J2 and Sun/Moon gravity using the same orbit, epoch, and frame.

Each case adds one selected perturbation to Earth point-mass gravity. Orekit
and Octavian use independent integrators and native ephemerides, so the small
remaining disagreement includes ephemeris and integration differences.
Requires the Orekit data package described in README.md.
"""

from pathlib import Path

import numpy as np
from _report import report_comparison
from octavian import EARTH, MOON, SUN, Perturbations, propagate, state
from octavian.data.ephemeris import sample_sun_moon_positions_eci_tod
from octavian.dynamics import ThirdBodyTable, gravity_acceleration_components

from actium import State, Trajectory, configure_orekit_data, propagate_j2, propagate_sun_moon
from actium._orekit import classes

# 1. All cases start from this Earth orbit and absolute epoch.
position_m = [-2_806_318.2300452064, 6_750_454.822659758, -1_886_890.8112853996]
velocity_m_s = [-4_030.598697633472, 977.0643994799905, 5_994.992595088863]
epoch = "2026-01-01T00:00:00Z"
times_s = np.linspace(0.0, 6 * 3600.0, 121)
cases = {
    "J2": Perturbations(j2=True),
    "Sun": Perturbations(sun=True),
    "Moon": Perturbations(moon=True),
    "Sun and Moon": Perturbations(sun=True, moon=True),
}

# Sun/Moon positions in Octavian use true-of-date axes. Match those axes in
# Orekit; comparing identical numbers in different frames would compare different orbits.
configure_orekit_data()
classes()
from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.utils import IERSConventions  # noqa: E402

true_of_date = FramesFactory.getTOD(IERSConventions.IERS_1996, True)
all_passed = True
for name, forces in cases.items():
    frame = "EME2000" if forces.j2 else true_of_date
    frame_name = frame if isinstance(frame, str) else str(frame.getName())

    # 2. Propagate directly in Octavian; no deputy or relative orbit is needed.
    history = propagate.inertial(
        state(position_m, velocity_m_s),
        times_s,
        perturbations=forces,
        initial_epoch=epoch,
        max_step_s=5.0,
        ephemeris_step_s=300.0,
    )

    # 3. Select the equivalent Orekit force model and matching constants.
    initial = State.from_si(position_m, velocity_m_s)
    if forces.j2:
        orekit = propagate_j2(
            initial,
            times_s,
            mu_km3_s2=EARTH.mu_m3ps2 / 1e9,
            equatorial_radius_km=EARTH.mean_radius_m / 1e3,
            j2=EARTH.j2_coefficient,
            frame=frame,
            position_tolerance_m=1e-7,
        )
    else:
        orekit = propagate_sun_moon(
            initial,
            times_s,
            epoch=epoch,
            mu_km3_s2=EARTH.mu_m3ps2 / 1e9,
            include_sun=forces.sun,
            include_moon=forces.moon,
            sun_mu_km3_s2=SUN.mu_m3ps2 / 1e9,
            moon_mu_km3_s2=MOON.mu_m3ps2 / 1e9,
            frame=frame,
            position_tolerance_m=1e-7,
        )

    # 4. Evaluate Octavian acceleration along its trajectory using the same
    # ephemeris sampling as its propagator, then compare full state histories.
    tables = []
    if forces.sun or forces.moon:
        table_times, body_positions = sample_sun_moon_positions_eci_tod(
            initial_epoch=epoch,
            duration_s=float(times_s[-1]),
            step_s=300.0,
        )
        for body in (SUN, MOON):
            if body.name.lower() in forces.active_third_bodies():
                tables.append(
                    ThirdBodyTable(
                        name=body.name.lower(),
                        mu_m3ps2=body.mu_m3ps2,
                        position_table=None,
                        times_s=table_times,
                        positions_eci_m=body_positions[body.name.lower()],
                    )
                )
    accelerations = np.array(
        [
            gravity_acceleration_components(
                row[:3],
                time_s=row[6],
                mu_m3ps2=EARTH.mu_m3ps2,
                include_j2=forces.j2,
                central_body_radius_m=EARTH.mean_radius_m,
                j2_coefficient=EARTH.j2_coefficient,
                third_body_tables=tables,
            )
            for row in history
        ]
    )
    octavian = Trajectory.from_si(
        times_s, history[:, :3], history[:, 3:6], accelerations, frame=frame_name
    )
    difference = report_comparison(
        octavian,
        orekit,
        Path(__file__).parent / "results/perturbations_intro" / name.lower().replace(" ", "_"),
        title=f"Earth gravity plus {name}: {frame_name}",
    )
    all_passed = (
        difference.within(position_m=0.1, velocity_m_s=1e-4, acceleration_m_s2=1e-6) and all_passed
    )

if not all_passed:
    raise SystemExit("A comparison exceeds the documented tolerances; inspect the saved reports.")
