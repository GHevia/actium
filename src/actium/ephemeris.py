"""Explicit Orekit-data setup and Sun/Moon ephemeris access."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._orekit import classes, resolve_frame
from .units import position_m_to_km

FloatArray = NDArray[np.float64]

SUN_MU_KM3_S2 = 132_712_440_018.0
MOON_MU_KM3_S2 = 4_902.800118

_DATA_LOCK = Lock()
_DATA_SOURCE: str | None = None


def configure_orekit_data(path: str | Path | None = None) -> str:
    """Configure Orekit data once and return the selected source.

    With ``path=None``, Actium loads the separately installed ``orekitdata``
    package. A local Orekit data directory or zip can be supplied explicitly.
    """
    global _DATA_SOURCE

    requested = "orekitdata package" if path is None else str(Path(path).expanduser().resolve())
    with _DATA_LOCK:
        if _DATA_SOURCE is not None:
            if path is not None and requested != _DATA_SOURCE:
                raise RuntimeError(
                    f"Orekit data is already configured from {_DATA_SOURCE!r}; "
                    f"cannot switch to {requested!r} in the running JVM"
                )
            return _DATA_SOURCE

        classes()  # Java imports must follow JVM startup.
        from orekit_jpype.pyhelpers import setup_orekit_data

        try:
            if path is None:
                setup_orekit_data(from_pip_library=True)
            else:
                resolved = Path(path).expanduser().resolve()
                if not resolved.exists():
                    raise FileNotFoundError(f"Orekit data path does not exist: {resolved}")
                setup_orekit_data(filenames=str(resolved), from_pip_library=False)
        except (ImportError, TypeError) as exc:
            raise RuntimeError(
                "Sun/Moon propagation requires Orekit data. Install the official data "
                "package with: pip install "
                "'git+https://gitlab.orekit.org/orekit/orekit-data.git', or pass "
                "orekit_data_path to a local data directory/zip."
            ) from exc
        _DATA_SOURCE = requested
        return _DATA_SOURCE


def _datetime_utc(epoch: str | datetime) -> datetime:
    if isinstance(epoch, datetime):
        value = epoch
    elif isinstance(epoch, str):
        normalized = epoch.strip()
        if normalized.upper().endswith(" UTC"):
            normalized = normalized[:-4]
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("epoch must be an ISO-8601 UTC string or datetime") from exc
    else:
        raise TypeError("epoch must be an ISO-8601 UTC string or datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def absolute_date(epoch: str | datetime | Any) -> Any:
    """Convert a UTC value to Orekit ``AbsoluteDate`` after data setup."""
    if hasattr(epoch, "shiftedBy") and hasattr(epoch, "durationFrom"):
        return epoch
    value = _datetime_utc(epoch)
    api = classes()
    seconds = float(value.second) + float(value.microsecond) * 1.0e-6
    return api["AbsoluteDate"](
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        seconds,
        api["TimeScalesFactory"].getUTC(),
    )


def sun_moon_positions(
    times_s: ArrayLike,
    *,
    epoch: str | datetime | Any,
    frame: str | Any = "EME2000",
    orekit_data_path: str | Path | None = None,
) -> dict[str, FloatArray]:
    """Return Earth-centered Sun and Moon positions in km at elapsed times."""
    configure_orekit_data(orekit_data_path)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least one finite value")
    start = absolute_date(epoch)
    orekit_frame = resolve_frame(frame)
    api = classes()
    bodies = {
        "sun": api["CelestialBodyFactory"].getSun(),
        "moon": api["CelestialBodyFactory"].getMoon(),
    }
    output: dict[str, FloatArray] = {}
    for name, body in bodies.items():
        positions_m = np.empty((times.size, 3), dtype=np.float64)
        for index, elapsed_s in enumerate(times):
            vector = body.getPosition(start.shiftedBy(float(elapsed_s)), orekit_frame)
            positions_m[index] = [vector.getX(), vector.getY(), vector.getZ()]
        output[name] = np.asarray(position_m_to_km(positions_m), dtype=np.float64)
    return output
