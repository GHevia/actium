"""Validate Octavian/ASSET EGM2008 against Actium/Orekit, with matched physics.

Plain Python: edit the settings below and run this file. Java and Actium must
be installed in the environment used for Octavian. No Orekit data archive is
required: the frame is deliberately uniform Z rotation, not ITRF/EOP.
"""

from __future__ import annotations

import csv
import json
import sys
from importlib.metadata import version
from importlib.resources import as_file, files
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from actium import (  # noqa: E402
    SphericalHarmonicGravity,
    State,
    Trajectory,
    compare_trajectories,
    propagate_spherical_harmonics,
    two_body_acceleration_si,
)

# Editable settings. Only the bridge imports Octavian; Actium stays independent.
octavian_path = Path(__file__).resolve().parents[3] / "octavian"
output_dir = Path(__file__).resolve().parents[1] / "results" / "spherical_harmonics"
degrees = [20, 100, 200]
times_s = np.linspace(0.0, 6 * 3600.0, 121)
rotation = dict(rotation_rate_radps=7.292115e-5, reference_angle_rad=0.4, reference_time_s=1234.0)
scenarios = {
    "inclined_leo": State([7000.0, 100.0, 300.0], [0.0, 7.4, 1.0]),
    "near_polar_leo": State([7000.0, 0.0, 50.0], [0.0, 0.12, 7.5]),
}
settings = {
    "baseline": dict(asset_abs=5e-8, asset_rel=2e-13, orekit_position=1e-8, max_step=90.0),
    "tight": dict(asset_abs=2e-8, asset_rel=1e-13, orekit_position=1e-9, max_step=60.0),
}
gates = dict(position_m=1e-3, velocity_m_s=1e-6, acceleration_m_s2=1e-9)
force_gate_m_s2 = 1e-11

sys.path.insert(0, str(octavian_path))
from octavian import SphericalHarmonics  # noqa: E402
from octavian.dynamics import PerturbedECI  # noqa: E402

output_dir.mkdir(parents=True, exist_ok=True)
reports = []
force_rows = []
resources = files("octavian").joinpath("data/gravity")
manifest = json.loads(resources.joinpath("egm2008_200.json").read_text())

with as_file(resources.joinpath("egm2008_200.gfc.gz")) as coefficient_path:
    for degree in degrees:
        gravity = SphericalHarmonics.earth(degree=degree, **rotation)
        reference = SphericalHarmonicGravity(coefficient_path, degree=degree, **rotation)
        mu = gravity.reference_mu_m3ps2
        assert reference.provider.getMu() == mu
        assert reference.provider.getAe() == gravity.reference_radius_m
        # Independent parsers must agree on every selected coefficient.
        normalized = reference.provider.onDate(reference.initial_date)
        for n in range(degree + 1):
            for m in range(n + 1):
                assert normalized.getNormalizedCnm(n, m) == gravity.cosine[n][m]
                assert normalized.getNormalizedSnm(n, m) == gravity.sine[n][m]

        # Same-state force tests isolate the equations from integrator errors.
        # Avoid the exact polar coordinate singularity in Orekit's gradient.
        points = [
            [7e6, 0, 0],
            [1.0, 0.0, 7e6],
            [1.0, 0.0, -7e6],
            [6.4e6, -2.2e6, 3.1e6],
            [4.2e7, 1e6, -2e6],
        ]
        for t in (0.0, 1234.0, 21600.0):
            for point in points:
                actual = gravity.acceleration(point, mu_m3ps2=mu, time_s=t)
                expected = (
                    reference.perturbing_acceleration(np.asarray(point) / 1e3, time_s=t) * 1e3
                )
                difference = actual - expected
                force_rows.append(
                    dict(
                        degree=degree,
                        time_s=t,
                        x_m=point[0],
                        y_m=point[1],
                        z_m=point[2],
                        dx_m_s2=difference[0],
                        dy_m_s2=difference[1],
                        dz_m_s2=difference[2],
                        norm_m_s2=float(np.linalg.norm(difference)),
                    )
                )
        max_force = max(row["norm_m_s2"] for row in force_rows if row["degree"] == degree)

        for name, initial in scenarios.items():
            states_by_setting = {}
            for setting_name, options in settings.items():
                orekit = propagate_spherical_harmonics(
                    initial,
                    times_s,
                    gravity=reference,
                    position_tolerance_m=options["orekit_position"],
                    max_step_s=options["max_step"],
                )
                ode = PerturbedECI(mu_m3ps2=mu, spherical_harmonics=gravity)
                integrator = ode.integrator(10.0)
                integrator.setAbsTol(options["asset_abs"])
                integrator.setRelTol(options["asset_rel"])
                position_m, velocity_m_s = initial.to_si()
                start = np.r_[position_m, velocity_m_s, 0.0]
                history = np.asarray(
                    integrator.integrate_dense(start, float(times_s[-1]), len(times_s))
                )
                np.testing.assert_allclose(history[:, 6], times_s, rtol=0, atol=1e-9)
                acceleration = two_body_acceleration_si(history[:, :3], mu_m3_s2=mu)
                force_at_same_states = []
                for index, t in enumerate(times_s):
                    harmonic = gravity.acceleration(history[index, :3], mu_m3ps2=mu, time_s=t)
                    acceleration[index] += harmonic
                    same = (
                        reference.perturbing_acceleration(history[index, :3] / 1e3, time_s=t) * 1e3
                    )
                    force_at_same_states.append(np.linalg.norm(harmonic - same))
                octavian = Trajectory(
                    times_s,
                    history[:, :3] / 1e3,
                    history[:, 3:6] / 1e3,
                    acceleration / 1e3,
                    frame="EME2000",
                )
                # Explicit convention: Actium/Orekit minus Octavian/ASSET.
                difference = compare_trajectories(octavian, orekit)
                same_force_max = max(max_force, max(force_at_same_states))
                passed = difference.within(**gates) and same_force_max <= force_gate_m_s2
                report = dict(
                    case=name,
                    degree=degree,
                    settings=setting_name,
                    **difference.summary(),
                    same_state_force_max_m_s2=float(same_force_max),
                    passed=bool(passed),
                )
                reports.append(report)
                states_by_setting[setting_name] = (octavian, orekit)
                stem = f"{name}_{degree}_{setting_name}"
                columns = np.column_stack(
                    (
                        times_s,
                        octavian.states_si,
                        orekit.states_si,
                        octavian.accelerations_m_s2,
                        orekit.accelerations_m_s2,
                        difference.position_difference_m,
                        difference.velocity_difference_m_s,
                        difference.acceleration_difference_m_s2,
                        difference.position_error_norm_m,
                        difference.velocity_error_norm_m_s,
                        difference.acceleration_error_norm_m_s2,
                        force_at_same_states,
                    )
                )
                headers = ["time_s"]
                for prefix in ("octavian", "orekit"):
                    headers += [
                        f"{prefix}_{x}" for x in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
                    ]
                for prefix in (
                    "octavian",
                    "orekit",
                    "difference_position",
                    "difference_velocity",
                    "difference_acceleration",
                ):
                    unit = (
                        "m_s2"
                        if prefix in ("octavian", "orekit", "difference_acceleration")
                        else ("m" if prefix == "difference_position" else "m_s")
                    )
                    headers += [f"{prefix}_{axis}_{unit}" for axis in "xyz"]
                headers += [
                    "position_error_m",
                    "velocity_error_m_s",
                    "acceleration_error_m_s2",
                    "same_state_force_error_m_s2",
                ]
                np.savetxt(
                    output_dir / f"{stem}.csv",
                    columns,
                    delimiter=",",
                    header=",".join(headers),
                    comments="",
                )
                fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
                for axis, values, label in zip(
                    axes,
                    [
                        difference.position_error_norm_m,
                        difference.velocity_error_norm_m_s,
                        difference.acceleration_error_norm_m_s2,
                    ],
                    ["Position [m]", "Velocity [m/s]", "Acceleration [m/s²]"],
                    strict=True,
                ):
                    axis.plot(times_s / 3600, values)
                    axis.set_ylabel(label)
                    axis.grid(alpha=0.3)
                axes[-1].set_xlabel("Elapsed time [h]")
                fig.suptitle(
                    f"EGM2008 {degree}×{degree}: {name}, {setting_name}\nOrekit minus ASSET; matched uniform rotation"
                )
                fig.savefig(output_dir / f"{stem}.png", dpi=150)
                plt.close(fig)
                print(json.dumps(report), flush=True)
            # Record tolerance sensitivity within each independent integrator.
            convergence = {}
            for index, label in enumerate(("octavian", "orekit")):
                delta = compare_trajectories(
                    states_by_setting["baseline"][index],
                    states_by_setting["tight"][index],
                )
                convergence[label] = delta.summary()
            reports[-1]["baseline_to_tight"] = convergence

with (output_dir / "same_state_forces.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(force_rows[0]))
    writer.writeheader()
    writer.writerows(force_rows)
summary = dict(
    model=manifest,
    orekit_jpype=version("orekit-jpype"),
    actium=version("actium"),
    octavian=version("octavian"),
    duration_s=float(times_s[-1]),
    samples=len(times_s),
    rotation=rotation,
    gates=gates,
    force_gate_m_s2=force_gate_m_s2,
    integrators=settings,
    difference_convention="Orekit minus ASSET",
    reports=reports,
    scope="Point mass plus static EGM2008 in a uniformly rotating frame; no ITRF/EOP, tides, third bodies, drag or SRP",
)
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if not all(report["passed"] for report in reports):
    raise SystemExit(
        "Spherical-harmonic validation FAILED; inspect the saved CSV/PNG/JSON artifacts"
    )
print(f"All {len(reports)} comparisons passed. Artifacts: {output_dir}")
