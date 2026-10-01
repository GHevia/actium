"""Actium: a tiny, independent Orekit trajectory-validation reference."""

from .ephemeris import (
    MOON_MU_KM3_S2,
    SUN_MU_KM3_S2,
    configure_orekit_data,
    sun_moon_positions,
)
from .harmonics import SphericalHarmonicGravity, propagate_spherical_harmonics
from .metrics import TrajectoryComparison, compare_trajectories
from .models import State, Trajectory
from .propagation import (
    propagate_j2,
    propagate_keplerian,
    propagate_numerical,
    propagate_sun_moon,
    two_body_acceleration,
    two_body_acceleration_si,
)
from .units import (
    acceleration_km_s2_to_m_s2,
    acceleration_m_s2_to_km_s2,
    mu_km3_s2_to_m3_s2,
    mu_m3_s2_to_km3_s2,
    position_km_to_m,
    position_m_to_km,
    velocity_km_s_to_m_s,
    velocity_m_s_to_km_s,
)

__version__ = "0.3.0"

__all__ = [
    "MOON_MU_KM3_S2",
    "SUN_MU_KM3_S2",
    "State",
    "SphericalHarmonicGravity",
    "propagate_spherical_harmonics",
    "Trajectory",
    "TrajectoryComparison",
    "acceleration_km_s2_to_m_s2",
    "acceleration_m_s2_to_km_s2",
    "compare_trajectories",
    "configure_orekit_data",
    "mu_km3_s2_to_m3_s2",
    "mu_m3_s2_to_km3_s2",
    "position_km_to_m",
    "position_m_to_km",
    "propagate_keplerian",
    "propagate_j2",
    "propagate_numerical",
    "propagate_sun_moon",
    "sun_moon_positions",
    "two_body_acceleration",
    "two_body_acceleration_si",
    "velocity_km_s_to_m_s",
    "velocity_m_s_to_km_s",
]
