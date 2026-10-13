"""Compare two-body motion: one initial orbit, two independent propagators.

Run from the Actium checkout after installing Actium and Octavian in the same
Python environment. Edit the state and output times below; no arguments are needed.
"""

from pathlib import Path

import numpy as np
from _report import report_comparison
from octavian import EARTH, propagate, state
from octavian.dynamics import gravity_acceleration_components

from actium import State, Trajectory, propagate_keplerian

# 1. Share the physical inputs. Octavian uses metres; Actium uses kilometres.
position_m = [7_000_000.0, 0.0, 300_000.0]
velocity_m_s = [0.0, 7_400.0, 1_000.0]
times_s = np.linspace(0.0, 6 * 3600.0, 121)
mu = EARTH.mu_m3ps2

# 2. Propagate with Octavian's analytical two-body model.
history = propagate.two_body(state(position_m, velocity_m_s), times_s, mu_m3ps2=mu)
accelerations = np.array([gravity_acceleration_components(row[:3], mu_m3ps2=mu) for row in history])
octavian = Trajectory.from_si(times_s, history[:, :3], history[:, 3:6], accelerations)

# 3. Propagate the same orbit independently with Orekit via Actium.
orekit = propagate_keplerian(
    State.from_si(position_m, velocity_m_s),
    times_s,
    mu_km3_s2=mu / 1e9,
)

# 4. Compare at identical times. Both histories use the EME2000 frame.
difference = report_comparison(
    octavian,
    orekit,
    Path(__file__).parent / "results/two_body/comparison",
    title="Two-body gravity: Octavian and Orekit",
)
print("The paths overlap; the difference plot resolves the numerical disagreement.")
if not difference.within(position_m=1e-3, velocity_m_s=1e-6, acceleration_m_s2=1e-9):
    raise SystemExit("Comparison exceeds the documented tolerances; inspect the saved report.")
