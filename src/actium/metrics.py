"""Full-history trajectory difference metrics in SI units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import Trajectory

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class TrajectoryComparison:
    """Candidate-minus-reference component differences and norm summaries."""

    times_s: FloatArray
    position_difference_m: FloatArray
    velocity_difference_m_s: FloatArray
    acceleration_difference_m_s2: FloatArray | None

    @property
    def state_difference_si(self) -> FloatArray:
        """Return full Cartesian state differences as ``[dr_m, dv_m_s]``."""
        return np.column_stack((self.position_difference_m, self.velocity_difference_m_s))

    @property
    def position_error_norm_m(self) -> FloatArray:
        return np.linalg.norm(self.position_difference_m, axis=1)

    @property
    def velocity_error_norm_m_s(self) -> FloatArray:
        return np.linalg.norm(self.velocity_difference_m_s, axis=1)

    @property
    def acceleration_error_norm_m_s2(self) -> FloatArray | None:
        if self.acceleration_difference_m_s2 is None:
            return None
        return np.linalg.norm(self.acceleration_difference_m_s2, axis=1)

    @property
    def max_position_error_m(self) -> float:
        return float(np.max(self.position_error_norm_m))

    @property
    def rms_position_error_m(self) -> float:
        return float(np.sqrt(np.mean(self.position_error_norm_m**2)))

    @property
    def max_velocity_error_m_s(self) -> float:
        return float(np.max(self.velocity_error_norm_m_s))

    @property
    def rms_velocity_error_m_s(self) -> float:
        return float(np.sqrt(np.mean(self.velocity_error_norm_m_s**2)))

    @property
    def max_acceleration_error_m_s2(self) -> float | None:
        norms = self.acceleration_error_norm_m_s2
        return None if norms is None else float(np.max(norms))

    @property
    def rms_acceleration_error_m_s2(self) -> float | None:
        norms = self.acceleration_error_norm_m_s2
        return None if norms is None else float(np.sqrt(np.mean(norms**2)))

    @property
    def max_abs_state_component_si(self) -> float:
        """Return the largest absolute component across the full SI state history."""
        return float(np.max(np.abs(self.state_difference_si)))

    def within(
        self,
        *,
        position_m: float,
        velocity_m_s: float,
        acceleration_m_s2: float | None = None,
    ) -> bool:
        """Check independent Euclidean-norm tolerances over every sample."""
        passed = self.max_position_error_m <= float(
            position_m
        ) and self.max_velocity_error_m_s <= float(velocity_m_s)
        if acceleration_m_s2 is not None:
            maximum = self.max_acceleration_error_m_s2
            passed = passed and maximum is not None and maximum <= float(acceleration_m_s2)
        return bool(passed)

    def summary(self) -> dict[str, float | None]:
        """Return stable names suitable for reports or machine-readable output."""
        return {
            "max_position_error_m": self.max_position_error_m,
            "rms_position_error_m": self.rms_position_error_m,
            "max_velocity_error_m_s": self.max_velocity_error_m_s,
            "rms_velocity_error_m_s": self.rms_velocity_error_m_s,
            "max_acceleration_error_m_s2": self.max_acceleration_error_m_s2,
            "rms_acceleration_error_m_s2": self.rms_acceleration_error_m_s2,
            "max_abs_state_component_si": self.max_abs_state_component_si,
        }


def compare_trajectories(
    reference: Trajectory,
    candidate: Trajectory,
    *,
    time_tolerance_s: float = 0.0,
    require_same_frame: bool = True,
) -> TrajectoryComparison:
    """Compare full histories at identical times without interpolation.

    Differences use the sign convention ``candidate - reference``. Both
    trajectories must contain the same number of samples and matching times.
    """
    if not isinstance(reference, Trajectory) or not isinstance(candidate, Trajectory):
        raise TypeError("reference and candidate must both be actium.Trajectory objects")
    if len(reference) != len(candidate):
        raise ValueError("trajectories must have the same number of samples")
    tolerance = float(time_tolerance_s)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("time_tolerance_s must be finite and non-negative")
    if not np.allclose(reference.times_s, candidate.times_s, rtol=0.0, atol=tolerance):
        raise ValueError("trajectory output times do not match")
    if require_same_frame and reference.frame != candidate.frame:
        raise ValueError(
            f"trajectory frames do not match: {reference.frame!r} != {candidate.frame!r}"
        )
    if (reference.accelerations_km_s2 is None) != (candidate.accelerations_km_s2 is None):
        raise ValueError("both trajectories must either provide accelerations or omit them")

    acceleration_difference = None
    if reference.accelerations_m_s2 is not None:
        acceleration_difference = candidate.accelerations_m_s2 - reference.accelerations_m_s2
    return TrajectoryComparison(
        times_s=np.array(reference.times_s, copy=True),
        position_difference_m=candidate.states_si[:, :3] - reference.states_si[:, :3],
        velocity_difference_m_s=candidate.states_si[:, 3:] - reference.states_si[:, 3:],
        acceleration_difference_m_s2=acceleration_difference,
    )
