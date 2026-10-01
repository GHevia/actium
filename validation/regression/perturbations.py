#!/usr/bin/env python3
"""Validate Octavian J2/Sun/Moon histories against Actium/Orekit and plot errors."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from actium import (  # noqa: E402
    State,
    Trajectory,
    compare_trajectories,
    configure_orekit_data,
    propagate_j2,
    propagate_sun_moon,
    sun_moon_positions,
    two_body_acceleration,
)
from actium._orekit import classes  # noqa: E402

try:  # Source-checkout package import under pytest.
    from validation.regression.two_body import _difference_rows, _write_csv  # noqa: E402
except ModuleNotFoundError:  # Direct ``python validation/...`` execution.
    from two_body import (  # type: ignore[no-redef]  # noqa: E402
        _difference_rows,
        _write_csv,
    )

MU_EARTH_M3_S2 = 3.986004418e14
MU_EARTH_KM3_S2 = MU_EARTH_M3_S2 / 1.0e9
EARTH_RADIUS_KM = 6_378.1363
EARTH_J2 = 1.08262668e-3
SUN_MU_KM3_S2 = 132_712_440_018.0
MOON_MU_KM3_S2 = 4_904.8695
EPOCH = "2026-01-01T00:00:00Z"


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    j2: bool = False
    sun: bool = False
    moon: bool = False


CASES = {
    case.name: case
    for case in (
        Case("j2", j2=True),
        Case("sun", sun=True),
        Case("moon", moon=True),
        Case("sun_moon", sun=True, moon=True),
    )
}
INITIAL_STATE = State(
    position_km=[-2_806.3182300452064, 6_750.454822659758, -1_886.8908112853997],
    velocity_km_s=[-4.030598697633472, 0.9770643994799905, 5.994992595088863],
)


@lru_cache(maxsize=1)
def _octavian_tod_frame() -> Any:
    """Return Orekit's match for Octavian's IAU-1976/1980 ECI_TOD frame."""
    configure_orekit_data()
    classes()
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions

    return FramesFactory.getTOD(IERSConventions.IERS_1996, True)


def _validation_frame(case: Case) -> str | Any:
    """Use the frame in which Octavian evaluates each force model."""
    if case.sun or case.moon:
        return _octavian_tod_frame()
    return "EME2000"


def _validation_frame_name(case: Case) -> str:
    frame = _validation_frame(case)
    return frame if isinstance(frame, str) else str(frame.getName())


def _load_octavian(path: Path | None) -> dict[str, Any]:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not (resolved / "octavian").is_dir():
            raise RuntimeError(f"--octavian-path does not contain an octavian package: {resolved}")
        sys.path.insert(0, str(resolved))
    try:
        from octavian import EARTH, Perturbations, propagate, state
        from octavian.data.ephemeris import sample_sun_moon_positions_eci_tod
        from octavian.dynamics import (
            ThirdBodyTable,
            gravity_acceleration_components,
            j2_acceleration_components,
            third_body_acceleration_components,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Octavian is not importable. Install it or pass --octavian-path to its checkout."
        ) from exc
    return {
        "EARTH": EARTH,
        "Perturbations": Perturbations,
        "ThirdBodyTable": ThirdBodyTable,
        "gravity": gravity_acceleration_components,
        "j2": j2_acceleration_components,
        "propagate": propagate,
        "sample_bodies": sample_sun_moon_positions_eci_tod,
        "state": state,
        "third_body": third_body_acceleration_components,
    }


def _actium_trajectory(case: Case, times_s: np.ndarray) -> Trajectory:
    common = {
        "mu_km3_s2": MU_EARTH_KM3_S2,
        "position_tolerance_m": 1.0e-7,
        "max_step_s": 120.0,
        "frame": _validation_frame(case),
    }
    if case.j2:
        return propagate_j2(
            INITIAL_STATE,
            times_s,
            equatorial_radius_km=EARTH_RADIUS_KM,
            j2=EARTH_J2,
            **common,
        )
    return propagate_sun_moon(
        INITIAL_STATE,
        times_s,
        epoch=EPOCH,
        include_sun=case.sun,
        include_moon=case.moon,
        sun_mu_km3_s2=SUN_MU_KM3_S2,
        moon_mu_km3_s2=MOON_MU_KM3_S2,
        **common,
    )


def _octavian_tables(
    api: dict[str, Any],
    case: Case,
    duration_s: float,
    ephemeris_step_s: float,
) -> tuple[Any, ...]:
    if not case.sun and not case.moon:
        return ()
    table_times, positions = api["sample_bodies"](
        initial_epoch=EPOCH,
        duration_s=duration_s,
        step_s=ephemeris_step_s,
    )
    tables = []
    for name, mu in (("sun", SUN_MU_KM3_S2), ("moon", MOON_MU_KM3_S2)):
        if getattr(case, name):
            tables.append(
                api["ThirdBodyTable"](
                    name=name,
                    mu_m3ps2=mu * 1.0e9,
                    position_table=None,
                    times_s=table_times,
                    positions_eci_m=positions[name],
                )
            )
    return tuple(tables)


def _octavian_trajectory(
    api: dict[str, Any],
    case: Case,
    times_s: np.ndarray,
    *,
    max_step_s: float,
    ephemeris_step_s: float,
) -> Trajectory:
    position_m, velocity_m_s = INITIAL_STATE.to_si()
    chief = api["state"](position_m, velocity_m_s)
    deputy = api["state"](position_m + np.asarray([100.0, 20.0, -30.0]), velocity_m_s)
    perturbations = api["Perturbations"](j2=case.j2, sun=case.sun, moon=case.moon)
    result = api["propagate"].relative(
        chief,
        None,
        times_s,
        deputy_initial_eci=deputy,
        central_body=api["EARTH"],
        perturbations=perturbations,
        initial_epoch=EPOCH if case.sun or case.moon else None,
        max_step_s=max_step_s,
        ephemeris_step_s=ephemeris_step_s,
    )
    states = np.asarray(result.chief_states_eci, dtype=np.float64)
    tables = _octavian_tables(api, case, float(times_s[-1]), ephemeris_step_s)
    accelerations = np.asarray(
        [
            api["gravity"](
                state_row[:3],
                time_s=float(time_s),
                mu_m3ps2=MU_EARTH_M3_S2,
                include_j2=case.j2,
                central_body_radius_m=EARTH_RADIUS_KM * 1.0e3,
                j2_coefficient=EARTH_J2,
                third_body_tables=tables,
            )
            for time_s, state_row in zip(times_s, states, strict=True)
        ],
        dtype=np.float64,
    )
    return Trajectory.from_si(
        times_s,
        states[:, :3],
        states[:, 3:6],
        accelerations,
        frame=_validation_frame_name(case),
    )


def _shared_ephemeris_force_error(
    api: dict[str, Any],
    case: Case,
    candidate: Trajectory,
) -> float:
    central = two_body_acceleration(
        candidate.positions_km,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    orekit_perturbation = candidate.accelerations_km_s2 - central
    octavian_perturbation = np.zeros_like(orekit_perturbation)
    if case.j2:
        octavian_perturbation = (
            np.asarray(
                [
                    api["j2"](
                        position * 1.0e3,
                        mu_m3ps2=MU_EARTH_M3_S2,
                        radius_m=EARTH_RADIUS_KM * 1.0e3,
                        j2=EARTH_J2,
                    )
                    for position in candidate.positions_km
                ]
            )
            / 1.0e3
        )
    else:
        body_positions = sun_moon_positions(
            candidate.times_s,
            epoch=EPOCH,
            frame=_validation_frame(case),
        )
        for name, mu in (("sun", SUN_MU_KM3_S2), ("moon", MOON_MU_KM3_S2)):
            if not getattr(case, name):
                continue
            octavian_perturbation += (
                np.asarray(
                    [
                        api["third_body"](
                            position * 1.0e3,
                            body_position * 1.0e3,
                            mu_m3ps2=mu * 1.0e9,
                        )
                        for position, body_position in zip(
                            candidate.positions_km,
                            body_positions[name],
                            strict=True,
                        )
                    ]
                )
                / 1.0e3
            )
    return float(np.max(np.linalg.norm(orekit_perturbation - octavian_perturbation, axis=1)))


def _native_ephemeris_force_error(
    api: dict[str, Any],
    case: Case,
    candidate: Trajectory,
    *,
    ephemeris_step_s: float,
) -> float:
    """Compare force laws at one spacecraft history using each native ephemeris."""
    central = two_body_acceleration(
        candidate.positions_km,
        mu_km3_s2=MU_EARTH_KM3_S2,
    )
    orekit_perturbation = candidate.accelerations_km_s2 - central
    octavian_perturbation = np.zeros_like(orekit_perturbation)
    if case.j2:
        octavian_perturbation = (
            np.asarray(
                [
                    api["j2"](
                        position * 1.0e3,
                        mu_m3ps2=MU_EARTH_M3_S2,
                        radius_m=EARTH_RADIUS_KM * 1.0e3,
                        j2=EARTH_J2,
                    )
                    for position in candidate.positions_km
                ]
            )
            / 1.0e3
        )
    else:
        tables = _octavian_tables(
            api,
            case,
            float(candidate.times_s[-1]),
            ephemeris_step_s,
        )
        for body in tables:
            octavian_perturbation += (
                np.asarray(
                    [
                        api["third_body"](
                            position * 1.0e3,
                            body.position_at(float(time_s)),
                            mu_m3ps2=float(body.mu_m3ps2),
                        )
                        for time_s, position in zip(
                            candidate.times_s,
                            candidate.positions_km,
                            strict=True,
                        )
                    ]
                )
                / 1.0e3
            )
    return float(np.max(np.linalg.norm(orekit_perturbation - octavian_perturbation, axis=1)))


def _native_body_position_errors_km(
    api: dict[str, Any],
    case: Case,
    candidate: Trajectory,
    *,
    ephemeris_step_s: float,
) -> dict[str, float]:
    """Return maximum Orekit-minus-SPICE body-position norms in the shared frame."""
    if not case.sun and not case.moon:
        return {}
    orekit_positions = sun_moon_positions(
        candidate.times_s,
        epoch=EPOCH,
        frame=_validation_frame(case),
    )
    tables = _octavian_tables(
        api,
        case,
        float(candidate.times_s[-1]),
        ephemeris_step_s,
    )
    errors: dict[str, float] = {}
    for body in tables:
        spice_positions_km = (
            np.asarray([body.position_at(float(time_s)) for time_s in candidate.times_s]) / 1.0e3
        )
        errors[body.name] = float(
            np.max(np.linalg.norm(orekit_positions[body.name] - spice_positions_km, axis=1))
        )
    return errors


def _plot_errors(case: Case, comparison: Any, output_path: Path) -> None:
    hours = comparison.times_s / 3_600.0
    acceleration = comparison.acceleration_difference_m_s2
    assert acceleration is not None
    series = (
        (comparison.position_difference_m, "Position difference", "m"),
        (comparison.velocity_difference_m_s, "Velocity difference", "m/s"),
        (acceleration, "Acceleration difference", "m/s²"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis, (values, title, units) in zip(axes, series, strict=True):
        for column, label in enumerate(("x", "y", "z")):
            axis.plot(hours, values[:, column], label=f"d{label}", linewidth=1.1)
        axis.plot(hours, np.linalg.norm(values, axis=1), "k--", label="norm", linewidth=1.4)
        axis.set_ylabel(units)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.legend(ncol=4, fontsize="small")
    axes[-1].set_xlabel("Elapsed time (hours)")
    figure.suptitle(f"Actium/Orekit minus Octavian: {case.name}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octavian-path", type=Path)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("validation/results/perturbations"))
    parser.add_argument("--duration-s", type=float, default=21_600.0)
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument("--octavian-max-step-s", type=float, default=5.0)
    parser.add_argument("--octavian-ephemeris-step-s", type=float, default=300.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.1)
    parser.add_argument("--velocity-tolerance-m-s", type=float, default=1.0e-4)
    parser.add_argument("--acceleration-tolerance-m-s2", type=float, default=1.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.duration_s <= 0.0 or args.samples < 2:
        raise RuntimeError("--duration-s must be positive and --samples must be at least 2")
    api = _load_octavian(args.octavian_path)
    times_s = np.linspace(0.0, args.duration_s, args.samples)
    cases = CASES.values() if args.case == "all" else [CASES[args.case]]

    print(
        "case       frame                    max|dr| m     max|dv| m/s   max|da| m/s²  "
        "native-force |da| m/s²  shared-force |da| m/s²  result"
    )
    all_rows: list[dict[str, float | str]] = []
    all_passed = True
    for case in cases:
        reference = _octavian_trajectory(
            api,
            case,
            times_s,
            max_step_s=args.octavian_max_step_s,
            ephemeris_step_s=args.octavian_ephemeris_step_s,
        )
        candidate = _actium_trajectory(case, times_s)
        comparison = compare_trajectories(reference, candidate)
        native_force_error_km_s2 = _native_ephemeris_force_error(
            api,
            case,
            candidate,
            ephemeris_step_s=args.octavian_ephemeris_step_s,
        )
        force_error_km_s2 = _shared_ephemeris_force_error(api, case, candidate)
        body_position_errors_km = _native_body_position_errors_km(
            api,
            case,
            candidate,
            ephemeris_step_s=args.octavian_ephemeris_step_s,
        )
        passed = comparison.within(
            position_m=args.position_tolerance_m,
            velocity_m_s=args.velocity_tolerance_m_s,
            acceleration_m_s2=args.acceleration_tolerance_m_s2,
        )
        all_passed = all_passed and passed
        print(
            f"{case.name:10} {_validation_frame_name(case):24} "
            f"{comparison.max_position_error_m:12.5e} "
            f"{comparison.max_velocity_error_m_s:14.5e} "
            f"{comparison.max_acceleration_error_m_s2:14.5e} "
            f"{native_force_error_km_s2 * 1.0e3:22.5e} "
            f"{force_error_km_s2 * 1.0e3:22.5e}  {'PASS' if passed else 'FAIL'}"
        )
        if body_position_errors_km:
            body_summary = ", ".join(
                f"{name}={error_km:.6g} km" for name, error_km in body_position_errors_km.items()
            )
            print(f"  native body-position max: {body_summary}")
        all_rows.extend(_difference_rows(case.name, "orekit", reference, candidate))
        plot_path = args.output_dir / f"{case.name}_errors.png"
        _plot_errors(case, comparison, plot_path)
        print(f"  plot: {plot_path}")

    csv_path = args.output_dir / "perturbation_differences.csv"
    _write_csv(csv_path, all_rows)
    print(f"Wrote {len(all_rows)} sample differences to {csv_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
