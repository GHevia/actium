"""Compare EGM2008 gravity using Octavian/ASSET and Orekit/Actium.

Both engines read the same measured coefficients and use uniform Earth rotation.
The independent algorithms should give the same orbit up to integration error.
This comparison excludes tides, drag, third bodies and a full Earth-orientation model.
"""

from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
from _report import report_comparison
from octavian import SphericalHarmonics, state
from octavian.dynamics import PerturbedECI, gravity_acceleration_components

from actium import SphericalHarmonicGravity, State, Trajectory, propagate_spherical_harmonics

# 1. Choose the gravity resolution and one shared orbit in EME2000 coordinates.
degree = 200
order = 200
position_m = [7_000_000.0, 100_000.0, 300_000.0]
velocity_m_s = [0.0, 7_400.0, 1_000.0]
times_s = np.linspace(0.0, 6 * 3600.0, 121)
gravity = SphericalHarmonics.earth(degree=degree, order=order)

# 2. Octavian loads coefficients and GM automatically; ASSET integrates the EOM.
ode = PerturbedECI(spherical_harmonics=gravity)
integrator = ode.integrator(10.0)
integrator.setAbsTol(2e-8)
integrator.setRelTol(1e-13)
initial = state(position_m, velocity_m_s)
history = np.asarray(
    integrator.integrate_dense(
        np.r_[initial.r_m, initial.v_mps, 0.0],
        float(times_s[-1]),
        len(times_s),
    )
)
accelerations = np.array(
    [
        gravity_acceleration_components(
            row[:3], time_s=row[6], mu_m3ps2=ode.mu, spherical_harmonics=gravity
        )
        for row in history
    ]
)
octavian = Trajectory.from_si(history[:, 6], history[:, :3], history[:, 3:6], accelerations)

# 3. Orekit reads the coefficient file with its own ICGEM parser and evaluates
# Holmes–Featherstone gravity. Actium converts the SI initial state to km/km/s.
coefficient_file = files("octavian").joinpath("data/gravity/egm2008_200.gfc.gz")
with as_file(coefficient_file) as path:
    orekit_gravity = SphericalHarmonicGravity(path, degree=degree, order=order)
orekit = propagate_spherical_harmonics(
    State.from_si(position_m, velocity_m_s),
    times_s,
    gravity=orekit_gravity,
    position_tolerance_m=1e-9,
    max_step_s=60.0,
)

# 4. Compare forces at the SAME positions to separate gravity-model agreement
# from small trajectory differences accumulated by the two integrators.
force_differences = []
for row in history:
    octavian_force = gravity.acceleration(row[:3], time_s=row[6])
    orekit_force = orekit_gravity.perturbing_acceleration(row[:3] / 1e3, time_s=row[6]) * 1e3
    force_differences.append(np.linalg.norm(orekit_force - octavian_force))
print(f"Maximum same-state gravity difference: {max(force_differences):.6g} m/s²")

# 5. Plot overlapping orbits and small differences at the shared output times.
difference = report_comparison(
    octavian,
    orekit,
    Path(__file__).parent / "results/spherical_harmonics_intro/comparison",
    title=f"EGM2008 {degree}×{order}: Octavian/ASSET and Orekit",
)
if (
    not difference.within(position_m=1e-3, velocity_m_s=1e-6, acceleration_m_s2=1e-9)
    or max(force_differences) > 1e-11
):
    raise SystemExit("Comparison exceeds the documented tolerances; inspect the saved report.")
