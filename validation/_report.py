"""CSV and plotting support shared by the introductory comparisons."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from actium import compare_trajectories  # noqa: E402


def report_comparison(octavian, orekit, output, *, title):
    """Save both trajectories and signed differences; return comparison metrics."""
    difference = compare_trajectories(octavian, orekit)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        difference.times_s,
        octavian.states_si,
        orekit.states_si,
        difference.position_difference_m,
        difference.velocity_difference_m_s,
    ]
    headers = ["time_s"]
    for prefix in ("octavian", "orekit"):
        headers.extend(
            f"{prefix}_{label}" for label in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
        )
    headers.extend(f"difference_{axis}_m" for axis in "xyz")
    headers.extend(f"difference_v{axis}_m_s" for axis in "xyz")
    if difference.acceleration_difference_m_s2 is not None:
        columns.extend(
            [
                octavian.accelerations_m_s2,
                orekit.accelerations_m_s2,
                difference.acceleration_difference_m_s2,
            ]
        )
        for prefix in ("octavian", "orekit", "difference"):
            headers.extend(f"{prefix}_a{axis}_m_s2" for axis in "xyz")
    np.savetxt(
        output.with_suffix(".csv"),
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
    figure = plt.figure(figsize=(11, 7), layout="constrained")
    grid = figure.add_gridspec(2, 2, wspace=0.25)
    orbit = figure.add_subplot(grid[:, 0], projection="3d")
    for history, label, style in ((octavian, "Octavian", "-"), (orekit, "Orekit", "--")):
        orbit.plot(*history.positions_km.T, style, label=label)
    orbit.set(xlabel="X [km]", ylabel="Y [km]", zlabel="Z [km]", title="Trajectories")
    limit_km = max(
        np.linalg.norm(octavian.positions_km, axis=1).max(),
        np.linalg.norm(orekit.positions_km, axis=1).max(),
    )
    orbit.set(xlim=(-limit_km, limit_km), ylim=(-limit_km, limit_km), zlim=(-limit_km, limit_km))
    orbit.set_box_aspect((1, 1, 1))
    orbit.legend()
    for row, error, label in (
        (0, difference.position_error_norm_m, "Position difference [m]"),
        (1, difference.velocity_error_norm_m_s, "Velocity difference [m/s]"),
    ):
        axis = figure.add_subplot(grid[row, 1])
        axis.plot(difference.times_s / 3600, error)
        axis.set(xlabel="Elapsed time [h]", ylabel=label)
        axis.grid(alpha=0.3)
    figure.suptitle(title + "\nDifferences: Orekit minus Octavian")
    figure.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(figure)
    print(title)
    print(f"  Maximum position difference: {difference.max_position_error_m:.6g} m")
    print(f"  Maximum velocity difference: {difference.max_velocity_error_m_s:.6g} m/s")
    if difference.max_acceleration_error_m_s2 is not None:
        print(
            f"  Maximum acceleration difference: {difference.max_acceleration_error_m_s2:.6g} m/s²"
        )
    print(f"  Trajectories and differences: {output.with_suffix('.csv')}")
    print(f"  Plot: {output.with_suffix('.png')}")
    return difference
