"""
Golden-file regression test for the antenna gain pipeline.

Captures the output of RxAntenna.get_gain_dbi_for_direction over a fixed grid of inputs in
a synthesized antenna setup, and compares to a stored reference. Any change to the gain
pipeline (steered-pattern sampling, steering bracket/bilinear interpolation, directivity
scaling) that drifts numerically will trip this test.

To accept a new reference (after an intentional change), run
    python tests/golden/regenerate_gains.py
and commit the updated tests/golden/reference_gains.npz.
"""

from pathlib import Path

from antenna_fixture import FREQUENCY_MHZ, compute_gain_grid, make_test_antenna
import numpy as np
import pytest

from object_definitions.Rx_antenna_def import RxAntenna

GOLDEN_PATH = Path(__file__).parent / "golden" / "reference_gains.npz"


@pytest.mark.golden
def test_antenna_gain_matches_golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden reference {GOLDEN_PATH} does not exist. "
            "Run `python tests/golden/regenerate_gains.py` to create it."
        )

    reference = np.load(GOLDEN_PATH)
    ref_gains = reference["gains"]

    antenna = make_test_antenna()
    current = compute_gain_grid(antenna)

    # NaN pattern (e.g. behind-antenna or beyond fixed-antenna FOV) must match
    np.testing.assert_array_equal(
        np.isnan(current), np.isnan(ref_gains),
        err_msg="NaN pattern differs from golden reference.",
    )

    # Finite gain values must match to ~math precision.
    finite = ~np.isnan(current)
    np.testing.assert_allclose(
        current[finite], ref_gains[finite],
        atol=1e-10, rtol=1e-10,
        err_msg="Gain values differ from golden reference. If the change was intentional, "
                "run tests/golden/regenerate_gains.py to update the baseline.",
    )


def test_steered_gain_depends_on_steering_command():
    # With steering-dependent patterns, different steering commands at a fixed look direction
    # must produce different gains (otherwise the bracket/interpolation is not being applied).
    antenna = make_test_antenna()
    # Look direction 20 deg off boresight toward +x (steering azimuth ~20 deg, elevation 0).
    direction = np.array([np.sin(np.radians(20.0)), 0.0, np.cos(np.radians(20.0))])
    gain_aligned = antenna.get_gain_dbi_for_direction(direction, FREQUENCY_MHZ, azimuth_deg=20.0, elevation_deg=0.0)
    gain_off = antenna.get_gain_dbi_for_direction(direction, FREQUENCY_MHZ, azimuth_deg=-40.0, elevation_deg=0.0)
    assert np.isfinite(gain_aligned) and np.isfinite(gain_off)
    # Steering toward the look direction yields higher gain than steering away from it.
    assert gain_aligned > gain_off


def test_auto_steer_rejects_directions_beyond_max_steering():
    antenna = make_test_antenna()  # per-axis max steering = 40 deg
    off_boresight_deg = 60.0
    direction = np.array([np.sin(np.radians(off_boresight_deg)), 0.0, np.cos(np.radians(off_boresight_deg))])
    # Auto-steer (no explicit steering) must reject a 60 deg azimuth-plane reflection (> 40 deg limit).
    assert np.isnan(antenna.get_gain_dbi_for_direction(direction, FREQUENCY_MHZ))


def test_antenna_footprint_interpolates_steered_contours():
    antenna = make_test_antenna()
    corner_bundles = {
        (0.0, 0.0, FREQUENCY_MHZ): np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        (20.0, 0.0, FREQUENCY_MHZ): np.array([[1.0, 0.0, 0.0], [0.2, 0.0, 1.0]]),
        (0.0, 20.0, FREQUENCY_MHZ): np.array([[0.0, 1.0, 0.0], [0.1, 0.1, 1.0]]),
        (20.0, 20.0, FREQUENCY_MHZ): np.array([[1.0, 1.0, 1.0], [0.2, 0.1, 1.0]]),
    }
    for key, vector_bundle in corner_bundles.items():
        antenna.antenna_lookuptable[key]["vector_bundle"] = vector_bundle

    u = np.sin(np.radians(10.0))
    direction = np.array([u, u, np.sqrt(1.0 - 2.0 * u**2)])

    actual = antenna._get_antenna_footprint(direction, FREQUENCY_MHZ)
    expected = 0.25 * sum(corner_bundles.values())
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_bilinear_pattern_sampler_uses_endpoint_exclusive_phi_grid_in_linear_units():
    # Single theta row, 4 phi bins at 0, 90, 180, 270 deg (endpoint-exclusive).
    pattern_db = np.array([[0.0, 10.0, 20.0, 30.0]])

    # phi=45 -> halfway between bin 0 (0 dB) and bin 1 (10 dB), interpolated in LINEAR power.
    expected_45 = 0.5 * 10.0 ** (0.0 / 10.0) + 0.5 * 10.0 ** (10.0 / 10.0)
    assert RxAntenna._bilinear_pattern_sample_linear(pattern_db, 0.0, 45.0) == pytest.approx(expected_45)
    # phi=90 -> exactly bin 1 (10 dB -> 10 linear).
    assert RxAntenna._bilinear_pattern_sample_linear(pattern_db, 0.0, 90.0) == pytest.approx(10.0)
    # phi=315 -> halfway between bin 3 (30 dB) and bin 0 (0 dB, wraps across the seam).
    expected_315 = 0.5 * 10.0 ** (30.0 / 10.0) + 0.5 * 10.0 ** (0.0 / 10.0)
    assert RxAntenna._bilinear_pattern_sample_linear(pattern_db, 0.0, 315.0) == pytest.approx(expected_315)


def test_antenna_footprint_directivity_uses_endpoint_exclusive_phi_grid():
    antenna = make_test_antenna()
    antenna.footprint_resolution = 4
    pattern_db = np.zeros((181, 4), dtype=float)

    directivity, _directivity_db, _contour_vectors = antenna._calc_antenna_footprint({"radiation_pattern_dB": pattern_db})

    theta_rad = np.linspace(0, 180, pattern_db.shape[0]) * np.pi / 180
    phi_rad = np.linspace(0, 360, pattern_db.shape[1], endpoint=False) * np.pi / 180
    theta_mesh, _phi_mesh = np.meshgrid(theta_rad, phi_rad, indexing="ij")
    expected_directivity, _expected_directivity_db = RxAntenna._calculate_directivity(
        np.ones_like(pattern_db), theta_rad, phi_rad, theta_mesh
    )

    assert directivity == pytest.approx(expected_directivity)
