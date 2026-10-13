from __future__ import annotations

import csv
import importlib.util

import numpy as np
import pytest

from actium import Trajectory
from validation.regression.perturbations import (
    CASES,
    _validation_frame,
    _validation_frame_name,
)
from validation.regression.two_body import _difference_rows, _write_csv


def test_validation_detail_rows_include_full_state_and_acceleration_differences(tmp_path) -> None:
    reference = Trajectory.from_si(
        [0.0],
        [[7_000_000.0, 0.0, 0.0]],
        [[0.0, 7_500.0, 0.0]],
        [[-8.0, 0.0, 0.0]],
    )
    candidate = Trajectory.from_si(
        [0.0],
        [[7_000_001.0, 2.0, 3.0]],
        [[0.1, 7_500.2, 0.3]],
        [[-8.001, 0.002, 0.003]],
    )
    rows = _difference_rows("case", "analytical", reference, candidate)
    output = tmp_path / "details.csv"
    _write_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["d_rx_m"]) == 1.0
    assert float(row["d_vy_m_s"]) == pytest.approx(0.2)
    assert float(row["d_az_m_s2"]) == pytest.approx(0.003)
    assert float(row["position_error_norm_m"]) == pytest.approx(np.sqrt(14.0))


def test_j2_validation_remains_in_eme2000() -> None:
    assert _validation_frame(CASES["j2"]) == "EME2000"
    assert _validation_frame_name(CASES["j2"]) == "EME2000"


@pytest.mark.skipif(importlib.util.find_spec("orekitdata") is None, reason="orekitdata unavailable")
def test_third_body_validation_matches_octavian_tod_frame() -> None:
    assert _validation_frame_name(CASES["sun"]) == "TOD/1996 simple EOP"
    assert _validation_frame_name(CASES["moon"]) == "TOD/1996 simple EOP"


def test_intro_report_keeps_signed_si_histories(tmp_path):
    from validation._report import report_comparison

    reference = Trajectory.from_si(
        [0, 1],
        [[7e6, 0, 0], [7e6, 7500, 0]],
        [[0, 7500, 0], [0, 7500, 0]],
        [[-8, 0, 0], [-8, 0, 0]],
    )
    candidate = Trajectory.from_si(
        [0, 1],
        reference.states_si[:, :3] + [1, 0, 0],
        reference.states_si[:, 3:6],
        reference.accelerations_m_s2,
    )
    difference = report_comparison(
        reference, candidate, tmp_path / "orbit", title="Matched gravity"
    )
    assert difference.max_position_error_m == pytest.approx(1)
    with (tmp_path / "orbit.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["difference_x_m"]) == pytest.approx(1)
    assert float(row["octavian_x_m"]) == 7e6
    assert float(row["orekit_x_m"]) == 7e6 + 1
    assert (tmp_path / "orbit.png").stat().st_size > 0
