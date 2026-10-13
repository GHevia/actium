# Actium

Actium is a Python interface to
[Orekit](https://www.orekit.org/) through `orekit-jpype`. It provides an
independent dynamics reference for validating Octavian trajectories. The
installable `actium` package never imports Octavian; the bridge lives in the
separate `validation/` scripts.

Supported capabilities include:

- immutable NumPy `State` and `Trajectory` dataclasses;
- analytical and numerical two-body propagation;
- numerical J2 and static spherical-harmonic propagation;
- an explicit numerical Sun/Moon third-body propagator using Orekit
  ephemerides;
- explicit km/km/s public units and deterministic SI conversions at the
  Orekit boundary;
- full-history position, velocity, and acceleration comparison metrics; and
- Octavian-versus-Actium validation at identical output times, with CSV and
  plot artifacts.

The reference model excludes orbit determination, measurements, maneuvers,
event detection, solar-radiation pressure, drag, or general force-model
framework. Its named propagators are kept explicit for auditability.

## Installation

Orekit requires Java 11 or newer. With a system JDK:

```bash
python -m pip install -e .
```

Without a system JDK, install the optional pip-provided runtime:

```bash
python -m pip install -e ".[jdk]"
```

J2 and two-body propagation do not require an Orekit data archive. Sun/Moon
propagation does. Install Orekit's official data package directly from its
repository:

```bash
python -m pip install \
  "git+https://gitlab.orekit.org/orekit/orekit-data.git"
```

Actium will discover that package automatically. Alternatively, pass
`orekit_data_path` as a local Orekit data directory or zip. The JVM and data
configuration start lazily and cannot be replaced in the same Python process.

## Propagation

The public state unit convention is km and km/s. Every central and third-body
gravitational parameter is supplied in km^3/s^2. Actium never selects a
central-body `mu` implicitly. Orekit internally receives m, m/s, and m^3/s^2.

```python
import numpy as np

from actium import State, propagate_keplerian, propagate_numerical

mu_earth_km3_s2 = 398_600.4418
initial = State(
    position_km=[7_000.0, 0.0, 0.0],
    velocity_km_s=[0.0, np.sqrt(mu_earth_km3_s2 / 7_000.0), 0.0],
)
times_s = np.arange(0.0, 3_601.0, 60.0)

analytical = propagate_keplerian(initial, times_s, mu_km3_s2=mu_earth_km3_s2)
numerical = propagate_numerical(initial, times_s, mu_km3_s2=mu_earth_km3_s2)
```

J2 is also entirely caller-defined:

```python
from actium import propagate_j2

j2_history = propagate_j2(
    initial,
    times_s,
    mu_km3_s2=mu_earth_km3_s2,
    equatorial_radius_km=6_378.1363,
    j2=1.08262668e-3,
)
```

Sun and Moon use real Orekit ephemerides and therefore require an absolute UTC
epoch. Either body can be selected independently, and both gravitational
parameters may be overridden:

```python
from actium import propagate_sun_moon

third_body_history = propagate_sun_moon(
    initial,
    times_s,
    epoch="2026-01-01T00:00:00Z",
    mu_km3_s2=mu_earth_km3_s2,
    include_sun=True,
    include_moon=True,
)
```

`EME2000` is the default frame. `frame="GCRF"` or an already-created Orekit
`Frame` may be supplied explicitly. Elapsed output times are returned in caller
order. Analytical times may be negative, unsorted, or repeated. Numerical propagation starts from the same initial state. Spherical-harmonic
propagation samples a continuous ephemeris, preserving unsorted and repeated
output times.

Runnable examples are in [`examples/`](examples/).

## Comparing full histories

```python
from actium import compare_trajectories

difference = compare_trajectories(analytical, numerical)
print(difference.summary())

assert difference.within(
    position_m=1.0e-3,
    velocity_m_s=1.0e-6,
    acceleration_m_s2=1.0e-9,
)
```

Differences have the explicit sign convention `candidate - reference`. The
comparison refuses mismatched sample counts, output times, frames, or asymmetric
acceleration availability. It never interpolates one history onto another.

## Octavian validation

Actium runs in its own repository. Install both packages in the same environment;
Octavian is only a dependency of the comparison scripts. For a source-based setup,
clone the repositories as siblings and follow Octavian's installation instructions
for ASSET and the native gravity backend:

```bash
git clone https://github.com/GHevia/actium.git
git clone https://github.com/GHevia/octavian.git
cd actium
python -m pip install -e ".[dev,jdk]"
python -m pip install -e ../octavian
```

Spherical-harmonic validation requires an Octavian version providing
`SphericalHarmonics.earth()` and its compiled backend. Until that feature reaches
Octavian's default branch, select `agent/readable-gravity-workflows` in the Octavian
checkout and follow its native build instructions. Compatible published wheels
include the native backend. The reference package itself requires neither ASSET
nor Octavian.

Run the introductory comparisons:

```bash
python validation/validate_octavian.py
python validation/validate_perturbations.py
python validation/validate_spherical_harmonics.py
```

Each script shows shared inputs, Octavian propagation, Orekit propagation, and
plotted differences in sequence. Settings are ordinary Python variables.
The independent reference implementation remains in `src/actium`; reusable CSV
and plot code lives in `validation/_report.py`.

Detailed sweeps and ephemeris diagnostics remain in `validation/regression/`.
See [the validation guide](validation/README.md) for frames, assumptions,
acceptance gates, and recorded results.

## Development and publishing

```bash
python -m pip install -e ".[dev,jdk]"
python -m pytest
ruff check .
python -m build
```

PyPI Trusted Publishing and the included release
workflow are documented in [`PUBLISHING.md`](PUBLISHING.md).

## Static spherical harmonics

`SphericalHarmonicGravity(path, degree=200)` reads fully normalized ICGEM data
with Orekit's own parser and configures Holmes–Featherstone gravity. The file
supplies GM/radius; public position/velocity units remain km and km/s.
`propagate_spherical_harmonics(initial, times, gravity=gravity)` uses Orekit
DP853 and samples a continuous numerical ephemeris at caller times. Uniform
Z rotation is explicit; this is not an ITRF/EOP model. No Orekit data archive
is needed for this matched-physics comparison.

Run the plain Python bridge with settings defined in the script:

```bash
python validation/validate_spherical_harmonics.py
```

It uses the installed Octavian package's NGA EGM2008 file, independently
reads it in each implementation, and compares forces and six-hour trajectories
at the selected degree and order. The detailed campaign additionally covers
20×20, 100×100, and 200×200 across two orbits and two integration settings. See [the validation guide](validation/README.md) for setup and recorded results.
