# Compare Octavian and Orekit

The comparison scripts are outside `src/actium`. Only validation bridge scripts import Octavian, keeping the installable
reference implementation independent.

Start with the sequential Python scripts below. Each defines shared inputs,
propagates with Octavian and Orekit, and plots the overlapping orbits alongside
position and velocity differences. CSV files preserve both trajectories and
signed differences. Edit the settings near the top; no arguments are required.
Install both projects in the same environment using the
[setup guide](../README.md#octavian-validation).

```bash
python validation/validate_octavian.py
python validation/validate_perturbations.py
python validation/validate_spherical_harmonics.py
```

Two-body propagation needs no external data. Sun/Moon comparisons require the
Orekit data package. The harmonic comparison independently reads Octavian's
bundled EGM2008 coefficients; select degree and order in the script. It requires
Octavian 0.4.18 or the `agent/readable-gravity-workflows` branch.

The plots demonstrate agreement under matched assumptions. The initial state,
units, frame, constants, and forces must match. Small trajectory differences
also depend on integrator tolerances; same-state force comparisons isolate the
dynamics from those integration errors.

## Detailed regression campaigns

The multi-case checks, force diagnostics, convergence sweeps, and machine-readable
reports remain in `validation/regression/`, with the acceptance gates below.

The harness uses:

- the same initial Cartesian state;
- the same caller-supplied gravitational parameter;
- the same EME2000 interpretation;
- the exact same elapsed output-time array; and
- full position, velocity, and point-mass acceleration histories.

Run all representative cases:

```bash
python validation/regression/two_body.py \
  --octavian-path ../octavian \
  --details-csv validation/results/two_body_differences.csv
```

Use `--method analytical` or `--method numerical` to isolate an Orekit path,
and use `--scenario circular_leo` to isolate a case. The CSV contains Octavian
and Actium state/acceleration components plus signed Actium-minus-Octavian
differences and Euclidean error norms at every output time.

The default pass/fail gates are 1 mm position, 1 micrometre/s velocity, and
1e-9 m/s^2 acceleration. Override them with `--position-tolerance-m`,
`--velocity-tolerance-m-s`, and `--acceleration-tolerance-m-s2` when studying
integrator settings rather than enforcing the default regression gate.

Inspect the worst state and acceleration vectors in a generated report:

```bash
python validation/summarize_differences.py \
  validation/results/two_body_differences.csv
```

## J2, Sun, and Moon

Actium provides explicit Orekit J2 and solar/lunar third-body propagators. Run
the independent Octavian checks and create state/acceleration error plots with:

```bash
python validation/regression/perturbations.py \
  --octavian-path ../octavian \
  --output-dir validation/results/perturbations
```

The harness checks J2, Sun-only, Moon-only, and Sun+Moon histories. The
``native-force`` column evaluates both force laws at the same spacecraft states
and in the same coordinates while retaining each implementation's native body
ephemeris. J2 is compared in EME2000. Sun/Moon cases explicitly use Orekit
`TOD/1996 simple EOP`, the Orekit realization matching Octavian's SPICE
`ECI_TOD` convention. Actium's public propagation default remains EME2000. The
``shared-force`` column then evaluates Octavian's force equations using the
exact Orekit body positions. Together they distinguish force-law differences
from native ephemeris differences. The report also prints maximum Sun/Moon
position differences in the shared frame. All CSV and PNG artifacts are
written before the command returns a nonzero status for any failed tolerance.

The recorded Actium 0.2 validation used Orekit data commit
`baf158744d38ec76cf94e2d396280d545b9f0ba2`. The GitHub workflows pin that
revision so ephemeris updates cannot silently move the regression baseline.

### Recorded 2026-08-29 result

The six-hour, 121-sample run used 0.1 m position, 1e-4 m/s velocity, and
1e-6 m/s² acceleration gates:

| case | max position | max velocity | max acceleration | native-force acceleration | shared-force acceleration | result |
|---|---:|---:|---:|---:|---:|:---:|
| J2 | 3.400e-3 m | 2.327e-6 m/s | 2.257e-9 m/s² | 5.864e-15 m/s² | 5.864e-15 m/s² | PASS |
| Sun | 3.389e-3 m | 2.315e-6 m/s | 2.232e-9 m/s² | 8.095e-13 m/s² | 6.453e-15 m/s² | PASS |
| Moon | 3.380e-3 m | 2.306e-6 m/s | 2.236e-9 m/s² | 1.609e-12 m/s² | 5.318e-15 m/s² | PASS |
| Sun+Moon | 3.376e-3 m | 2.302e-6 m/s | 2.227e-9 m/s² | 9.164e-13 m/s² | 5.371e-15 m/s² | PASS |

Every case clears every gate. In the matched TOD frame, the maximum native
body-position differences are 64.3933 km for the Sun and 0.204263 km for the
Moon. Their dynamical effect remains at about 1e-12 m/s², while the
shared-ephemeris force equations agree near 6e-15 m/s². The roughly 3.4 mm
history difference is the Octavian fixed-step RK4 integration floor: the same
error appears in unperturbed two-body propagation and drops to micrometres when
the Octavian maximum step is reduced from 5 s to 1 s.

## EGM2008 spherical harmonics

`regression/spherical_harmonics.py` is a plain Python bridge with editable settings
and no CLI arguments. Run it in an environment containing Actium, Octavian/ASSET,
matplotlib and Java. The default checkout path is the sibling `../octavian`. See the
[setup guide](../README.md#octavian-validation) for dependencies.

```bash
python validation/regression/spherical_harmonics.py
```

Orekit's ICGEM reader loads the packaged NGA EGM2008 subset independently of
Octavian's loader. Every selected coefficient, GM, and reference radius is
checked. Orekit uses Holmes–Featherstone gravity; Octavian uses its native
Cartesian recurrence. Both use the same EME2000 coordinate interpretation and
uniform Z rotation with nonzero initial angle/reference time. No Orekit data
archive, tides, drag, SRP, third-body forces or full Earth-orientation model is
included; this isolates the implemented gravity model.

The campaign covers degree/order 20, 100 and 200, inclined and near-polar LEO,
six hours and 121 samples, and baseline/tight tolerances. Fixed same-state
force cases also cover near-pole and high-altitude positions. Reports include
within-integrator baseline-to-tight differences. Gates stay at 1 mm position,
1 µm/s velocity, 1e-9 m/s² history acceleration, and 1e-11 m/s² same-state force.

All full-history CSVs, error PNGs, same-state force CSV, and a summary JSON with
model provenance and versions are written to `validation/results/spherical_harmonics`
before a failed gate returns nonzero. Differences are **Orekit minus ASSET**.

Force comparisons isolate the gravity equations; baseline and tighter integration
settings assess trajectory sensitivity to numerical tolerances. Both checks are
required because force agreement alone does not bound accumulated state error.

### Recorded six-hour EGM2008 result

All 12 comparisons passed (two orbit types × three degrees × two tolerance
settings). Each row reports the maximum across both orbit cases and both
settings, over all 121 samples:

| degree/order | position [m] | velocity [m/s] | history acceleration [m/s²] | same-state force [m/s²] |
| --- | ---: | ---: | ---: | ---: |
| 20×20 | 2.168e-04 | 1.362e-07 | 4.956e-10 | 7.692e-17 |
| 100×100 | 2.085e-04 | 3.112e-07 | 2.805e-10 | 2.187e-16 |
| 200×200 | 3.267e-04 | 4.007e-07 | 4.284e-10 | 4.165e-16 |

These bounds apply to the documented matched-physics campaign, not arbitrary
missions or full Earth-orientation models. The recorded reference uses
Orekit-JPype 13.1.8.0. See `summary.json` for every case and convergence metrics.
