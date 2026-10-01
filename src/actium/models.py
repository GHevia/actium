"""Small NumPy data models used by every Actium entry point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .units import (
    acceleration_km_s2_to_m_s2,
    acceleration_m_s2_to_km_s2,
    position_km_to_m,
    position_m_to_km,
    velocity_km_s_to_m_s,
    velocity_m_s_to_km_s,
)

FloatArray = NDArray[np.float64]


def _immutable_array(values: ArrayLike, *, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, eq=False)
class State:
    """One Cartesian state in km and km/s.

    Actium's public data model uses astrodynamics-friendly kilometre units.
    Orekit conversion is explicit and happens only at the backend boundary.
    """

    position_km: FloatArray
    velocity_km_s: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_km",
            _immutable_array(self.position_km, shape=(3,), name="position_km"),
        )
        object.__setattr__(
            self,
            "velocity_km_s",
            _immutable_array(self.velocity_km_s, shape=(3,), name="velocity_km_s"),
        )

    @property
    def vector(self) -> FloatArray:
        """Return ``[x, y, z, vx, vy, vz]`` in km and km/s."""
        return np.concatenate((self.position_km, self.velocity_km_s))

    @classmethod
    def from_si(cls, position_m: ArrayLike, velocity_m_s: ArrayLike) -> State:
        """Build a state from Orekit/Octavian-compatible SI vectors."""
        return cls(position_m_to_km(position_m), velocity_m_s_to_km_s(velocity_m_s))

    def to_si(self) -> tuple[FloatArray, FloatArray]:
        """Return copied position and velocity vectors in m and m/s."""
        return (
            np.asarray(position_km_to_m(self.position_km), dtype=np.float64),
            np.asarray(velocity_km_s_to_m_s(self.velocity_km_s), dtype=np.float64),
        )


@dataclass(frozen=True, slots=True, eq=False)
class Trajectory:
    """Cartesian samples at caller-selected elapsed times.

    Times are seconds from the initial state epoch. Sample order is preserved,
    so analytical propagation can be evaluated at unsorted or negative times.
    Accelerations are optional for imported histories and are always populated
    by Actium's propagators.
    """

    times_s: FloatArray
    positions_km: FloatArray
    velocities_km_s: FloatArray
    accelerations_km_s2: FloatArray | None = None
    frame: str = "EME2000"

    def __post_init__(self) -> None:
        times = np.array(self.times_s, dtype=np.float64, copy=True).reshape(-1)
        if times.size == 0:
            raise ValueError("times_s must contain at least one sample")
        if not np.all(np.isfinite(times)):
            raise ValueError("times_s must contain only finite values")
        times.setflags(write=False)
        count = times.size

        positions = _immutable_array(
            self.positions_km,
            shape=(count, 3),
            name="positions_km",
        )
        velocities = _immutable_array(
            self.velocities_km_s,
            shape=(count, 3),
            name="velocities_km_s",
        )
        accelerations = None
        if self.accelerations_km_s2 is not None:
            accelerations = _immutable_array(
                self.accelerations_km_s2,
                shape=(count, 3),
                name="accelerations_km_s2",
            )
        frame = str(self.frame).strip()
        if not frame:
            raise ValueError("frame must not be empty")

        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "positions_km", positions)
        object.__setattr__(self, "velocities_km_s", velocities)
        object.__setattr__(self, "accelerations_km_s2", accelerations)
        object.__setattr__(self, "frame", frame)

    def __len__(self) -> int:
        return int(self.times_s.size)

    @property
    def states(self) -> FloatArray:
        """Return an ``(N, 6)`` state history in km and km/s."""
        return np.column_stack((self.positions_km, self.velocities_km_s))

    @property
    def history(self) -> FloatArray:
        """Return ``[r, v, elapsed_time]`` rows in km, km/s, and s."""
        return np.column_stack((self.states, self.times_s))

    @property
    def states_si(self) -> FloatArray:
        """Return an ``(N, 6)`` state history in m and m/s."""
        return np.column_stack(
            (
                position_km_to_m(self.positions_km),
                velocity_km_s_to_m_s(self.velocities_km_s),
            )
        )

    @property
    def accelerations_m_s2(self) -> FloatArray | None:
        """Return accelerations in m/s^2 when present."""
        if self.accelerations_km_s2 is None:
            return None
        return np.asarray(
            acceleration_km_s2_to_m_s2(self.accelerations_km_s2),
            dtype=np.float64,
        )

    def state_at(self, index: int) -> State:
        """Return one trajectory sample as a :class:`State`."""
        return State(self.positions_km[index], self.velocities_km_s[index])

    @classmethod
    def from_si(
        cls,
        times_s: ArrayLike,
        positions_m: ArrayLike,
        velocities_m_s: ArrayLike,
        accelerations_m_s2: ArrayLike | None = None,
        *,
        frame: str = "EME2000",
    ) -> Trajectory:
        """Build a trajectory from SI arrays without unit inference."""
        accelerations = (
            None if accelerations_m_s2 is None else acceleration_m_s2_to_km_s2(accelerations_m_s2)
        )
        return cls(
            times_s,
            position_m_to_km(positions_m),
            velocity_m_s_to_km_s(velocities_m_s),
            accelerations,
            frame,
        )
