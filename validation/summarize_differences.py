#!/usr/bin/env python3
"""Print worst state and acceleration component differences from a harness CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _maximum(rows: list[dict[str, str]], column: str) -> dict[str, str]:
    return max(rows, key=lambda row: float(row[column]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No validation rows found in {args.csv_path}")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["method"])].append(row)

    for (scenario, method), group in sorted(groups.items()):
        worst_state = _maximum(group, "position_error_norm_m")
        worst_acceleration = _maximum(group, "acceleration_error_norm_m_s2")
        print(f"{scenario} / {method}")
        print(
            "  worst position: "
            f"t={float(worst_state['time_s']):g} s, "
            f"|dr|={float(worst_state['position_error_norm_m']):.6e} m, "
            f"dr=[{float(worst_state['d_rx_m']):.6e}, "
            f"{float(worst_state['d_ry_m']):.6e}, "
            f"{float(worst_state['d_rz_m']):.6e}] m"
        )
        print(
            "  velocity there: "
            f"|dv|={float(worst_state['velocity_error_norm_m_s']):.6e} m/s, "
            f"dv=[{float(worst_state['d_vx_m_s']):.6e}, "
            f"{float(worst_state['d_vy_m_s']):.6e}, "
            f"{float(worst_state['d_vz_m_s']):.6e}] m/s"
        )
        print(
            "  worst acceleration: "
            f"t={float(worst_acceleration['time_s']):g} s, "
            f"|da|={float(worst_acceleration['acceleration_error_norm_m_s2']):.6e} m/s^2, "
            f"da=[{float(worst_acceleration['d_ax_m_s2']):.6e}, "
            f"{float(worst_acceleration['d_ay_m_s2']):.6e}, "
            f"{float(worst_acceleration['d_az_m_s2']):.6e}] m/s^2"
        )


if __name__ == "__main__":
    main()
