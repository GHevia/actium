"""Independent Orekit harmonics tests: no Octavian dependency or dataset needed."""

import numpy as np
import pytest

from actium import (
    SphericalHarmonicGravity,
    State,
    propagate_j2,
    propagate_spherical_harmonics,
)


@pytest.fixture
def coefficients(tmp_path):
    path = tmp_path / "j2.gfc"
    path.write_text("""product_type gravity_field
modelname j2_test
earth_gravity_constant 3.986004418e14
radius 6378136.3
max_degree 2
errors no
norm fully_normalized
tide_system tide_free
end_of_head
gfc 0 0 1 0
gfc 1 0 0 0
gfc 1 1 0 0
""" + f"gfc 2 0 {-1.08262668e-3/np.sqrt(5):.17g} 0\ngfc 2 1 0 0\ngfc 2 2 0 0\n")
    return path


def test_j2_limit_and_rotation(coefficients):
    gravity = SphericalHarmonicGravity(coefficients, degree=2, reference_angle_rad=0.7)
    point = np.array([7000.0, 1000.0, 2000.0])
    r2 = np.dot(point, point)
    z2 = point[2] ** 2 / r2
    scale = 1.5 * 1.08262668e-3 * 398600.4418 * 6378.1363**2 / r2**2.5
    expected = scale * point * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])
    for t in [0, 1000, 10000]:
        np.testing.assert_allclose(
            gravity.perturbing_acceleration(point, time_s=t),
            expected,
            rtol=5e-14,
            atol=1e-18,
        )


def test_harmonic_propagation_matches_orekit_j2_and_sample_order(coefficients):
    initial = State([7000, 100, 300], [0, 7.4, 1])
    times = [600, 0, 60, 600]
    gravity = SphericalHarmonicGravity(coefficients, degree=2)
    actual = propagate_spherical_harmonics(initial, times, gravity=gravity)
    expected = propagate_j2(
        initial,
        times,
        mu_km3_s2=398600.4418,
        equatorial_radius_km=6378.1363,
        j2=1.08262668e-3,
        position_tolerance_m=1e-7,
    )
    np.testing.assert_allclose(actual.states_si, expected.states_si, rtol=0, atol=2e-5)
    np.testing.assert_array_equal(actual.states_si[0], actual.states_si[3])
    zero = propagate_spherical_harmonics(initial, [0, 0], gravity=gravity)
    np.testing.assert_allclose(
        zero.states_si[0, :3], initial.to_si()[0], atol=1e-9, rtol=0
    )
    with pytest.raises(ValueError, match="nonnegative"):
        propagate_spherical_harmonics(initial, [-1, 0], gravity=gravity)


def test_missing_coefficients_are_not_silently_zero_filled(coefficients):
    coefficients.write_text(coefficients.read_text().replace("gfc 2 2 0 0\n", ""))
    with pytest.raises(Exception, match="missing gravity field coefficient"):
        SphericalHarmonicGravity(coefficients, degree=2)
