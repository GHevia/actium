"""Independent Orekit spherical-harmonic reference with explicit body rotation."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from ._orekit import classes, resolve_frame
from .propagation import (
    _initial_orbit,
    _numeric_options,
    _numerical_propagator,
    _sample_propagator,
    _validate_inputs,
    _vector3_to_numpy,
)


class SphericalHarmonicGravity:
    """Read a static fully-normalized ICGEM file with Orekit, independently.

    No Octavian imports or recurrence are used. Public lengths use km, elapsed
    times use seconds from J2000, and the frame defaults to EME2000. The body
    rotates uniformly about inertial +Z by angle + rate*(t-reference_time_s).
    This intentionally matches a simple rotating field, not full ITRF/EOP.
    """

    def __init__(
        self,
        coefficient_path,
        *,
        degree=200,
        order=None,
        frame="EME2000",
        rotation_rate_radps=7.292115e-5,
        reference_angle_rad=0.0,
        reference_time_s=0.0,
    ):
        order = degree if order is None else order
        if (
            isinstance(degree, bool)
            or not isinstance(degree, (int, np.integer))
            or degree < 2
            or isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or not 0 <= order <= degree
        ):
            raise ValueError(
                "degree/order must be integers with 2 <= degree and 0 <= order <= degree"
            )
        if not np.isfinite(
            [rotation_rate_radps, reference_angle_rad, reference_time_s]
        ).all():
            raise ValueError("rotation parameters must be finite")
        api = classes()
        from java.io import ByteArrayInputStream
        from jpype import JProxy
        from org.hipparchus.geometry.euclidean.threed import (
            Rotation,
            RotationConvention,
            Vector3D,
        )
        from org.orekit.data import DataContext, DataProvidersManager
        from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
        from org.orekit.forces.gravity.potential import (
            ICGEMFormatReader,
            LazyLoadedGravityFields,
        )
        from org.orekit.frames import Frame, Transform, TransformProvider

        path = Path(coefficient_path)
        payload = (
            gzip.decompress(path.read_bytes())
            if path.suffix == ".gz"
            else path.read_bytes()
        )
        reader = ICGEMFormatReader(".*", False)  # Missing coefficients are errors.
        reader.setMaxParseDegree(int(degree))
        reader.setMaxParseOrder(int(order))
        reader.loadData(ByteArrayInputStream(payload), path.name.removesuffix(".gz"))
        fields = LazyLoadedGravityFields(
            DataProvidersManager(), DataContext.getDefault().getTimeScales().getTT()
        )
        fields.addPotentialCoefficientsReader(reader)
        self.provider = fields.getNormalizedProvider(int(degree), int(order))
        self.mu_km3_s2 = self.provider.getMu() / 1e9
        self.reference_radius_km = self.provider.getAe() / 1e3
        self.frame = resolve_frame(frame)
        self.initial_date = api["AbsoluteDate"].J2000_EPOCH
        epoch = self.initial_date

        def transform(date):
            elapsed = date.durationFrom(epoch)
            angle = reference_angle_rad + rotation_rate_radps * (
                elapsed - reference_time_s
            )
            # Inertial vector -> coordinates in the rotating body frame.
            rotation = Rotation(
                Vector3D.PLUS_K, -angle, RotationConvention.VECTOR_OPERATOR
            )
            return Transform(
                date, rotation, Vector3D(0.0, 0.0, float(rotation_rate_radps))
            )

        self._rotation_provider = JProxy(
            TransformProvider,
            dict={
                "getTransform": transform,
                "getStaticTransform": lambda date: transform(date).toStaticTransform(),
                "getKinematicTransform": transform,
            },
        )
        self.body_frame = Frame(
            self.frame, self._rotation_provider, "uniformly rotating gravity"
        )
        self.force_model = HolmesFeatherstoneAttractionModel(
            self.body_frame, self.provider
        )

    def perturbing_acceleration(self, position_km, *, time_s=0.0):
        """Return harmonic acceleration in inertial km/s², excluding point mass."""
        position = np.asarray(position_km, dtype=float)
        if (
            position.shape != (3,)
            or not np.isfinite(position).all()
            or np.linalg.norm(position) == 0
        ):
            raise ValueError("position_km must be a finite nonzero three-vector")
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        api = classes()
        date = self.initial_date.shiftedBy(float(time_s))
        transform = self.frame.getStaticTransformTo(self.body_frame, date)
        point = api["Vector3D"](*(float(value) * 1e3 for value in position))
        body_point = transform.transformPosition(point)
        gradient = self.force_model.gradient(date, body_point, self.provider.getMu())
        inertial = transform.getInverse().transformVector(api["Vector3D"](*gradient))
        return _vector3_to_numpy(inertial) / 1e3


def propagate_spherical_harmonics(
    initial_state,
    times_s,
    *,
    gravity: SphericalHarmonicGravity,
    min_step_s=1e-3,
    max_step_s=120.0,
    initial_step_s=10.0,
    position_tolerance_m=1e-7,
):
    """Propagate point mass plus Orekit Holmes–Featherstone gravity.

    Generates one continuous numerical ephemeris, sampled at the exact caller
    times (including unsorted/repeated samples). Times must be nonnegative.
    The coefficient file supplies GM and radius; no body defaults are chosen.
    """
    times, mu_si = _validate_inputs(initial_state, times_s, gravity.mu_km3_s2)
    if np.any(times < 0):
        raise ValueError("harmonic propagation times must be nonnegative")
    options = _numeric_options(
        min_step_s=min_step_s,
        max_step_s=max_step_s,
        initial_step_s=initial_step_s,
        position_tolerance_m=position_tolerance_m,
    )
    orbit = _initial_orbit(initial_state, gravity.frame, mu_si, gravity.initial_date)
    forces = (gravity.force_model,)
    propagator = _numerical_propagator(orbit, mu_si, force_models=forces, **options)
    if times.max() > 0:
        generator = propagator.getEphemerisGenerator()
        propagator.propagate(gravity.initial_date.shiftedBy(float(times.max())))
        sampler = generator.getGeneratedEphemeris()
    else:
        sampler = propagator
    return _sample_propagator(
        sampler, times, gravity.frame, gravity.initial_date, mu_si, forces
    )
