"""Explicit Orekit propagators for Actium's supported gravity models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._orekit import classes, resolve_frame
from .ephemeris import (
    MOON_MU_KM3_S2,
    SUN_MU_KM3_S2,
    absolute_date,
    configure_orekit_data,
)
from .models import State, Trajectory
from .units import (
    acceleration_m_s2_to_km_s2,
    mu_km3_s2_to_m3_s2,
    position_m_to_km,
    velocity_m_s_to_km_s,
)

FloatArray = NDArray[np.float64]


def _validate_inputs(
    initial_state: State,
    times_s: ArrayLike,
    mu_km3_s2: float,
) -> tuple[FloatArray, float]:
    if not isinstance(initial_state, State):
        raise TypeError("initial_state must be an actium.State")
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least one finite value")
    mu_si = mu_km3_s2_to_m3_s2(float(mu_km3_s2))
    if not np.isfinite(mu_si) or mu_si <= 0.0:
        raise ValueError("mu_km3_s2 must be finite and positive")
    return times, mu_si


def _initial_orbit(
    initial_state: State,
    frame: Any,
    mu_m3_s2: float,
    initial_date: Any,
) -> Any:
    api = classes()
    position_m, velocity_m_s = initial_state.to_si()
    position = api["Vector3D"](*(float(value) for value in position_m))
    velocity = api["Vector3D"](*(float(value) for value in velocity_m_s))
    pv = api["PVCoordinates"](position, velocity)
    return api["CartesianOrbit"](
        pv,
        frame,
        initial_date,
        mu_m3_s2,
    )


def _vector3_to_numpy(vector: Any) -> FloatArray:
    return np.asarray([vector.getX(), vector.getY(), vector.getZ()], dtype=np.float64)


def _sample_propagator(
    propagator: Any,
    times_s: FloatArray,
    frame: Any,
    initial_date: Any,
    mu_m3_s2: float,
    force_models: tuple[Any, ...] = (),
) -> Trajectory:
    positions_m = np.empty((times_s.size, 3), dtype=np.float64)
    velocities_m_s = np.empty((times_s.size, 3), dtype=np.float64)
    accelerations_m_s2 = np.empty((times_s.size, 3), dtype=np.float64)

    for index, elapsed_s in enumerate(times_s):
        date = initial_date.shiftedBy(float(elapsed_s))
        state = propagator.propagate(date)
        pv = state.getPVCoordinates(frame)
        positions_m[index] = _vector3_to_numpy(pv.getPosition())
        velocities_m_s[index] = _vector3_to_numpy(pv.getVelocity())
        acceleration = two_body_acceleration_si(
            positions_m[index],
            mu_m3_s2=mu_m3_s2,
        )
        for model in force_models:
            acceleration = acceleration + _vector3_to_numpy(
                model.acceleration(state, model.getParameters(date))
            )
        accelerations_m_s2[index] = acceleration

    return Trajectory(
        times_s=times_s,
        positions_km=position_m_to_km(positions_m),
        velocities_km_s=velocity_m_s_to_km_s(velocities_m_s),
        accelerations_km_s2=acceleration_m_s2_to_km_s2(accelerations_m_s2),
        frame=str(frame.getName()),
    )


def two_body_acceleration_si(
    positions_m: ArrayLike,
    *,
    mu_m3_s2: float,
) -> FloatArray:
    """Evaluate ``-mu*r/|r|^3`` in SI units for one or many positions."""
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.shape == (3,):
        positions = positions.reshape(1, 3)
        squeeze = True
    elif positions.ndim == 2 and positions.shape[1] == 3:
        squeeze = False
    else:
        raise ValueError("positions_m must have shape (3,) or (N, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must contain only finite values")
    mu = float(mu_m3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu_m3_s2 must be finite and positive")
    radii = np.linalg.norm(positions, axis=1)
    if np.any(radii == 0.0):
        raise ValueError("positions_m must have non-zero norm")
    accelerations = -mu * positions / radii[:, None] ** 3
    return accelerations[0] if squeeze else accelerations


def two_body_acceleration(
    positions_km: ArrayLike,
    *,
    mu_km3_s2: float,
) -> FloatArray:
    """Evaluate ``-mu*r/|r|^3`` in km/s^2 for one or many positions."""
    positions = np.asarray(positions_km, dtype=np.float64)
    mu = float(mu_km3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu_km3_s2 must be finite and positive")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_km must contain only finite values")
    if positions.shape == (3,):
        radius = float(np.linalg.norm(positions))
        if radius == 0.0:
            raise ValueError("positions_km must have non-zero norm")
        return -mu * positions / radius**3
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_km must have shape (3,) or (N, 3)")
    radii = np.linalg.norm(positions, axis=1)
    if np.any(radii == 0.0):
        raise ValueError("positions_km must have non-zero norm")
    return -mu * positions / radii[:, None] ** 3


def propagate_keplerian(
    initial_state: State,
    times_s: ArrayLike,
    *,
    mu_km3_s2: float,
    frame: str | Any = "EME2000",
) -> Trajectory:
    """Propagate a Cartesian state with Orekit's analytical Kepler solver.

    ``mu_km3_s2`` is required: Actium never silently selects a central body.
    Times are elapsed seconds from an arbitrary J2000 epoch and may be
    negative, unsorted, or repeated.
    """
    times, mu_si = _validate_inputs(initial_state, times_s, mu_km3_s2)
    initial_date = classes()["AbsoluteDate"].J2000_EPOCH
    orekit_frame = resolve_frame(frame)
    orbit = _initial_orbit(initial_state, orekit_frame, mu_si, initial_date)
    propagator = classes()["KeplerianPropagator"](orbit, mu_si)
    return _sample_propagator(propagator, times, orekit_frame, initial_date, mu_si)


def _numerical_propagator(
    orbit: Any,
    mu_m3_s2: float,
    *,
    min_step_s: float,
    max_step_s: float,
    initial_step_s: float,
    position_tolerance_m: float,
    force_models: tuple[Any, ...] = (),
) -> Any:
    api = classes()
    provider = api["ToleranceProvider"].getDefaultToleranceProvider(position_tolerance_m)
    tolerances = provider.getTolerances(orbit, api["OrbitType"].CARTESIAN)
    integrator = api["DormandPrince853Integrator"](
        min_step_s,
        max_step_s,
        tolerances[0],
        tolerances[1],
    )
    integrator.setInitialStepSize(initial_step_s)
    propagator = api["NumericalPropagator"](integrator)
    propagator.setOrbitType(api["OrbitType"].CARTESIAN)
    propagator.setMu(mu_m3_s2)
    propagator.setInitialState(api["SpacecraftState"](orbit))
    for model in force_models:
        propagator.addForceModel(model)
    # Each target is integrated from the exact same initial state, making the
    # result independent of output ordering and sampling density.
    propagator.setResetAtEnd(False)
    return propagator


def _numeric_options(
    *,
    min_step_s: float,
    max_step_s: float,
    initial_step_s: float,
    position_tolerance_m: float,
) -> dict[str, float]:
    options = {
        "min_step_s": float(min_step_s),
        "max_step_s": float(max_step_s),
        "initial_step_s": float(initial_step_s),
        "position_tolerance_m": float(position_tolerance_m),
    }
    if not all(np.isfinite(value) and value > 0.0 for value in options.values()):
        raise ValueError(
            "numerical step sizes and position_tolerance_m must be finite and positive"
        )
    if options["min_step_s"] > options["max_step_s"]:
        raise ValueError("min_step_s must be no larger than max_step_s")
    if not options["min_step_s"] <= options["initial_step_s"] <= options["max_step_s"]:
        raise ValueError("initial_step_s must be between min_step_s and max_step_s")
    return options


def propagate_numerical(
    initial_state: State,
    times_s: ArrayLike,
    *,
    mu_km3_s2: float,
    frame: str | Any = "EME2000",
    min_step_s: float = 1.0e-3,
    max_step_s: float = 300.0,
    initial_step_s: float = 10.0,
    position_tolerance_m: float = 1.0e-6,
) -> Trajectory:
    """Propagate only Newtonian central gravity with Orekit DP853.

    The integrated coordinates are Cartesian. No force model beyond the
    central attraction represented by ``mu_km3_s2`` is configured.
    """
    times, mu_si = _validate_inputs(initial_state, times_s, mu_km3_s2)
    numeric_options = _numeric_options(
        min_step_s=min_step_s,
        max_step_s=max_step_s,
        initial_step_s=initial_step_s,
        position_tolerance_m=position_tolerance_m,
    )

    initial_date = classes()["AbsoluteDate"].J2000_EPOCH
    orekit_frame = resolve_frame(frame)
    orbit = _initial_orbit(initial_state, orekit_frame, mu_si, initial_date)
    propagator = _numerical_propagator(orbit, mu_si, **numeric_options)
    return _sample_propagator(propagator, times, orekit_frame, initial_date, mu_si)


def propagate_j2(
    initial_state: State,
    times_s: ArrayLike,
    *,
    mu_km3_s2: float,
    equatorial_radius_km: float,
    j2: float,
    frame: str | Any = "EME2000",
    min_step_s: float = 1.0e-3,
    max_step_s: float = 300.0,
    initial_step_s: float = 10.0,
    position_tolerance_m: float = 1.0e-6,
) -> Trajectory:
    """Propagate central gravity plus the caller's explicit J2 coefficient."""
    times, mu_si = _validate_inputs(initial_state, times_s, mu_km3_s2)
    radius_m = float(equatorial_radius_km) * 1_000.0
    j2_value = float(j2)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("equatorial_radius_km must be finite and positive")
    if not np.isfinite(j2_value):
        raise ValueError("j2 must be finite")
    numeric_options = _numeric_options(
        min_step_s=min_step_s,
        max_step_s=max_step_s,
        initial_step_s=initial_step_s,
        position_tolerance_m=position_tolerance_m,
    )

    api = classes()
    initial_date = api["AbsoluteDate"].J2000_EPOCH
    orekit_frame = resolve_frame(frame)
    orbit = _initial_orbit(initial_state, orekit_frame, mu_si, initial_date)
    j2_model = api["J2OnlyPerturbation"](mu_si, radius_m, j2_value, orekit_frame)
    force_models = (j2_model,)
    propagator = _numerical_propagator(
        orbit,
        mu_si,
        force_models=force_models,
        **numeric_options,
    )
    return _sample_propagator(
        propagator,
        times,
        orekit_frame,
        initial_date,
        mu_si,
        force_models,
    )


def propagate_sun_moon(
    initial_state: State,
    times_s: ArrayLike,
    *,
    epoch: str | datetime | Any,
    mu_km3_s2: float,
    include_sun: bool = True,
    include_moon: bool = True,
    sun_mu_km3_s2: float = SUN_MU_KM3_S2,
    moon_mu_km3_s2: float = MOON_MU_KM3_S2,
    frame: str | Any = "EME2000",
    orekit_data_path: str | Path | None = None,
    min_step_s: float = 1.0e-3,
    max_step_s: float = 300.0,
    initial_step_s: float = 10.0,
    position_tolerance_m: float = 1.0e-6,
) -> Trajectory:
    """Propagate central gravity plus explicit solar/lunar third-body gravity."""
    if not include_sun and not include_moon:
        raise ValueError("at least one of include_sun or include_moon must be True")
    times, mu_si = _validate_inputs(initial_state, times_s, mu_km3_s2)
    sun_mu_si = mu_km3_s2_to_m3_s2(float(sun_mu_km3_s2))
    moon_mu_si = mu_km3_s2_to_m3_s2(float(moon_mu_km3_s2))
    if include_sun and (not np.isfinite(sun_mu_si) or sun_mu_si <= 0.0):
        raise ValueError("sun_mu_km3_s2 must be finite and positive")
    if include_moon and (not np.isfinite(moon_mu_si) or moon_mu_si <= 0.0):
        raise ValueError("moon_mu_km3_s2 must be finite and positive")
    numeric_options = _numeric_options(
        min_step_s=min_step_s,
        max_step_s=max_step_s,
        initial_step_s=initial_step_s,
        position_tolerance_m=position_tolerance_m,
    )

    configure_orekit_data(orekit_data_path)
    api = classes()
    initial_date = absolute_date(epoch)
    orekit_frame = resolve_frame(frame)
    orbit = _initial_orbit(initial_state, orekit_frame, mu_si, initial_date)
    models: list[Any] = []
    if include_sun:
        models.append(
            api["ThirdBodyAttraction"](
                api["CelestialBodyFactory"].getSun(),
                "Sun",
                sun_mu_si,
            )
        )
    if include_moon:
        models.append(
            api["ThirdBodyAttraction"](
                api["CelestialBodyFactory"].getMoon(),
                "Moon",
                moon_mu_si,
            )
        )
    force_models = tuple(models)
    propagator = _numerical_propagator(
        orbit,
        mu_si,
        force_models=force_models,
        **numeric_options,
    )
    return _sample_propagator(
        propagator,
        times,
        orekit_frame,
        initial_date,
        mu_si,
        force_models,
    )
