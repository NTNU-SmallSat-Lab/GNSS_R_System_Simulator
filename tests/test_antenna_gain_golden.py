"""
Golden-file regression test for the antenna gain pipeline.

Captures the output of RxAntenna.get_gain_dbi_for_direction over a fixed grid of inputs in
a synthesized antenna setup, and compares to a stored reference. Any change to the gain
pipeline (fixed-nadir JSON sampling, FOV bracket, directivity scaling) that drifts
numerically will trip this test.

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


def test_fixed_nadir_gain_ignores_steering_arguments():
    antenna = make_test_antenna()
    current = compute_gain_grid(antenna)
    reference = current[:, :, :1, :1]
    expected = np.broadcast_to(reference, current.shape)

    np.testing.assert_array_equal(np.isnan(current), np.isnan(expected))
    finite = np.isfinite(current)
    np.testing.assert_allclose(current[finite], expected[finite], atol=1e-10, rtol=1e-10)


def test_antenna_footprint_ignores_reflection_direction():
    antenna = make_test_antenna()
    expected = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.99]])
    antenna.antenna_lookuptable[FREQUENCY_MHZ]["vector_bundle"] = expected

    actual = antenna._get_antenna_footprint(np.array([1.0, 0.0, 0.0]), FREQUENCY_MHZ)

    np.testing.assert_allclose(actual, expected)


def test_radiation_pattern_sampler_uses_endpoint_exclusive_phi_grid():
    antenna = make_test_antenna()
    antenna.antenna_lookuptable[FREQUENCY_MHZ]["radiation_pattern_dB"] = np.array([[0.0, 10.0, 20.0, 30.0]])

    assert antenna._sample_radiation_pattern_db(FREQUENCY_MHZ, 0.0, 45.0) == pytest.approx(5.0)
    assert antenna._sample_radiation_pattern_db(FREQUENCY_MHZ, 0.0, 90.0) == pytest.approx(10.0)
    assert antenna._sample_radiation_pattern_db(FREQUENCY_MHZ, 0.0, 315.0) == pytest.approx(15.0)


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
