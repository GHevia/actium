#!/usr/bin/env python3
"""Compare representative Octavian and Actium two-body histories."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from actium import (
    State,
    Trajectory,
    compare_trajectories,
    propagate_keplerian,
    propagate_numerical,
)

MU_EARTH_M3_S2 = 3.986004418e14
MU_EARTH_KM3_S2 = MU_EARTH_M3_S2 / 1.0e9


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    times_s: tuple[float, ...]


SCENARIOS = {
    item.name: item
    for item in (
        Scenario(
            "circular_leo",
            (7_000_000.0, 0.0, 0.0),
            (0.0, 7_546.053290107542, 0.0),
            (0.0, 10.0, 60.0, 300.0, 900.0, 1_800.0, 3_600.0, 5_400.0),
        ),
        Scenario(
            "inclined_eccentric_leo",
            (-2_806_318.2300452064, 6_750_454.822659758, -1_886_890.8112853996),
            (-4_030.598697633472, 977.0643994799905, 5_994.992595088863),
            (0.0, 15.0, 120.0, 600.0, 1_500.0, 3_000.0, 6_000.0),
        ),
        Scenario(
            "circular_geo",
            (42_164_000.0, 0.0, 0.0),
            (0.0, 3_074.666284127684, 0.0),
            (0.0, 60.0, 600.0, 3_600.0, 10_800.0, 21_600.0, 43_200.0),
        ),
    )
}


def _load_octavian(
    octavian_path: Path | None,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    if octavian_path is not None:
        resolved = octavian_path.expanduser().resolve()
        if not (resolved / "octavian").is_dir():
            raise RuntimeError(f"--octavian-path does not contain an octavian package: {resolved}")
        sys.path.insert(0, str(resolved))
    try:
        from octavian.astro import propagate_cartesian_rv
        from octavian.dynamics import gravity_acceleration_components
    except ImportError as exc:
        raise RuntimeError(
            "Octavian is not importable. Install it or pass --octavian-path to its checkout."
        ) from exc
    return propagate_cartesian_rv, gravity_acceleration_components


def _octavian_trajectory(
    scenario: Scenario,
    propagate_state: Callable[..., Any],
    acceleration_function: Callable[..., Any],
) -> Trajectory:
    times_s = np.asarray(scenario.times_s, dtype=np.float64)
    initial = np.hstack((scenario.position_m, scenario.velocity_m_s))
    history = np.asarray(
        [propagate_state(initial, float(time_s), MU_EARTH_M3_S2) for time_s in times_s],
        dtype=np.float64,
    )
    if history.shape != (times_s.size, 6):
        raise RuntimeError(f"Octavian returned unexpected history shape {history.shape}")
    accelerations = np.asarray(
        [acceleration_function(position, mu_m3ps2=MU_EARTH_M3_S2) for position in history[:, :3]],
        dtype=np.float64,
    )
    return Trajectory.from_si(
        times_s,
        history[:, :3],
        history[:, 3:6],
        accelerations,
        frame="EME2000",
    )


def _actium_trajectory(scenario: Scenario, method: str) -> Trajectory:
    initial = State.from_si(scenario.position_m, scenario.velocity_m_s)
    propagator = propagate_keplerian if method == "analytical" else propagate_numerical
    return propagator(
        initial,
        scenario.times_s,
        mu_km3_s2=MU_EARTH_KM3_S2,
        frame="EME2000",
    )


def _difference_rows(
    scenario: str,
    method: str,
    reference: Trajectory,
    candidate: Trajectory,
) -> list[dict[str, float | str]]:
    comparison = compare_trajectories(reference, candidate)
    acceleration_difference = comparison.acceleration_difference_m_s2
    acceleration_norms = comparison.acceleration_error_norm_m_s2
    reference_accelerations = reference.accelerations_m_s2
    candidate_accelerations = candidate.accelerations_m_s2
    assert acceleration_difference is not None
    assert acceleration_norms is not None
    assert reference_accelerations is not None
    assert candidate_accelerations is not None

    rows: list[dict[str, float | str]] = []
    for index, time_s in enumerate(reference.times_s):
        row: dict[str, float | str] = {
            "scenario": scenario,
            "method": method,
            "time_s": float(time_s),
        }
        for column, axis in enumerate("xyz"):
            row[f"octavian_r{axis}_m"] = float(reference.states_si[index, column])
            row[f"actium_r{axis}_m"] = float(candidate.states_si[index, column])
            row[f"d_r{axis}_m"] = float(comparison.position_difference_m[index, column])
            row[f"octavian_v{axis}_m_s"] = float(reference.states_si[index, column + 3])
            row[f"actium_v{axis}_m_s"] = float(candidate.states_si[index, column + 3])
            row[f"d_v{axis}_m_s"] = float(comparison.velocity_difference_m_s[index, column])
            row[f"octavian_a{axis}_m_s2"] = float(reference_accelerations[index, column])
            row[f"actium_a{axis}_m_s2"] = float(candidate_accelerations[index, column])
            row[f"d_a{axis}_m_s2"] = float(acceleration_difference[index, column])
        row["position_error_norm_m"] = float(comparison.position_error_norm_m[index])
        row["velocity_error_norm_m_s"] = float(comparison.velocity_error_norm_m_s[index])
        row["acceleration_error_norm_m_s2"] = float(acceleration_norms[index])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octavian-path", type=Path)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument(
        "--method",
        choices=("all", "analytical", "numerical"),
        default="all",
    )
    parser.add_argument("--details-csv", type=Path)
    parser.add_argument("--position-tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--velocity-tolerance-m-s", type=float, default=1.0e-6)
    parser.add_argument("--acceleration-tolerance-m-s2", type=float, default=1.0e-9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    propagate_state, acceleration_function = _load_octavian(args.octavian_path)
    scenarios = SCENARIOS.values() if args.scenario == "all" else [SCENARIOS[args.scenario]]
    methods = ("analytical", "numerical") if args.method == "all" else (args.method,)

    print(
        "scenario                  method       max|dr| m     rms|dr| m  "
        "max|dv| m/s   max|da| m/s^2  result"
    )
    all_rows: list[dict[str, float | str]] = []
    all_passed = True
    for scenario in scenarios:
        reference = _octavian_trajectory(
            scenario,
            propagate_state,
            acceleration_function,
        )
        for method in methods:
            candidate = _actium_trajectory(scenario, method)
            comparison = compare_trajectories(reference, candidate)
            passed = comparison.within(
                position_m=args.position_tolerance_m,
                velocity_m_s=args.velocity_tolerance_m_s,
                acceleration_m_s2=args.acceleration_tolerance_m_s2,
            )
            all_passed = all_passed and passed
            print(
                f"{scenario.name:25} {method:10} "
                f"{comparison.max_position_error_m:12.5e} "
                f"{comparison.rms_position_error_m:12.5e} "
                f"{comparison.max_velocity_error_m_s:12.5e} "
                f"{comparison.max_acceleration_error_m_s2:14.5e}  "
                f"{'PASS' if passed else 'FAIL'}"
            )
            all_rows.extend(_difference_rows(scenario.name, method, reference, candidate))

    if args.details_csv is not None:
        _write_csv(args.details_csv, all_rows)
        print(f"Wrote {len(all_rows)} sample differences to {args.details_csv}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
