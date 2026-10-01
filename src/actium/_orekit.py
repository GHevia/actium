"""Lazy Orekit-JPype startup and Java-object construction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

_JVM_LOCK = Lock()
_ACTIUM_JVM_READY = False


def _jdk4py_jvm_path() -> str | None:
    """Locate the optional jdk4py JVM without changing process environment."""
    try:
        import jdk4py
    except ImportError:
        return None

    java_home = Path(jdk4py.JAVA_HOME)
    candidates = (
        java_home / "lib" / "server" / "libjvm.so",
        java_home / "lib" / "server" / "libjvm.dylib",
        java_home / "bin" / "server" / "jvm.dll",
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def ensure_jvm() -> None:
    """Start Orekit's JVM once, falling back to the optional jdk4py runtime."""
    global _ACTIUM_JVM_READY

    try:
        import jpype
        import orekit_jpype
    except ImportError as exc:  # pragma: no cover - packaging normally prevents this
        raise RuntimeError(
            "Actium requires orekit-jpype; install the project dependencies"
        ) from exc

    if _ACTIUM_JVM_READY:
        return
    with _JVM_LOCK:
        if _ACTIUM_JVM_READY:
            return
        if jpype.isJVMStarted():
            orekit_jpype.initVM()
            _ACTIUM_JVM_READY = True
            return
        try:
            jpype.getDefaultJVMPath()
            jvm_path = None
        except Exception:  # JPype uses platform-specific JVM lookup exceptions.
            jvm_path = _jdk4py_jvm_path()
        if jvm_path is None:
            try:
                orekit_jpype.initVM()
            except Exception as original_error:
                raise RuntimeError(
                    "No usable Java runtime was found. Install Java 11+ or install Actium "
                    "with the 'jdk' extra: pip install -e '.[jdk]'"
                ) from original_error
        else:
            try:
                orekit_jpype.initVM(
                    vmargs="--enable-native-access=ALL-UNNAMED",
                    jvmpath=jvm_path,
                )
            except Exception as startup_error:
                # orekit-jpype currently logs getDefaultJVMPath after a
                # successful explicit-path start. That lookup can fail even
                # though the supplied JVM is running; a second call safely
                # completes Orekit's converter registration.
                if not jpype.isJVMStarted():
                    raise RuntimeError("The jdk4py JVM could not be started") from startup_error
                orekit_jpype.initVM()
        _ACTIUM_JVM_READY = True


@lru_cache(maxsize=1)
def classes() -> dict[str, Any]:
    """Return the small Java class set used by Actium after starting the JVM."""
    ensure_jvm()

    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.bodies import CelestialBodyFactory
    from org.orekit.forces.gravity import J2OnlyPerturbation, ThirdBodyAttraction
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation import SpacecraftState, ToleranceProvider
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import PVCoordinates

    return {
        "AbsoluteDate": AbsoluteDate,
        "CartesianOrbit": CartesianOrbit,
        "CelestialBodyFactory": CelestialBodyFactory,
        "DormandPrince853Integrator": DormandPrince853Integrator,
        "FramesFactory": FramesFactory,
        "KeplerianPropagator": KeplerianPropagator,
        "J2OnlyPerturbation": J2OnlyPerturbation,
        "NumericalPropagator": NumericalPropagator,
        "OrbitType": OrbitType,
        "PVCoordinates": PVCoordinates,
        "SpacecraftState": SpacecraftState,
        "ThirdBodyAttraction": ThirdBodyAttraction,
        "TimeScalesFactory": TimeScalesFactory,
        "ToleranceProvider": ToleranceProvider,
        "Vector3D": Vector3D,
    }


def resolve_frame(frame: str | Any = "EME2000") -> Any:
    """Resolve a supported inertial-frame name or pass through an Orekit Frame."""
    if not isinstance(frame, str):
        if not hasattr(frame, "getName"):
            raise TypeError("frame must be 'EME2000', 'GCRF', or an Orekit Frame")
        return frame

    api = classes()
    normalized = frame.strip().upper().replace("-", "")
    if normalized == "EME2000":
        return api["FramesFactory"].getEME2000()
    if normalized == "GCRF":
        return api["FramesFactory"].getGCRF()
    raise ValueError("frame must be 'EME2000', 'GCRF', or an Orekit Frame")
