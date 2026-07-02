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

from object_definitions.AntennaPatternGenerator_def import AntennaPatternGenerator, ArrayFactor
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
    # Look direction 20 deg off boresight toward +x (steering across_track ~20 deg, along_track 0).
    direction = np.array([np.sin(np.radians(20.0)), 0.0, np.cos(np.radians(20.0))])
    gain_aligned = antenna.get_gain_dbi_for_direction(direction, FREQUENCY_MHZ, across_track_deg=20.0, along_track_deg=0.0)
    gain_off = antenna.get_gain_dbi_for_direction(direction, FREQUENCY_MHZ, across_track_deg=-40.0, along_track_deg=0.0)
    assert np.isfinite(gain_aligned) and np.isfinite(gain_off)
    # Steering toward the look direction yields higher gain than steering away from it.
    assert gain_aligned > gain_off


def test_auto_steer_rejects_directions_beyond_max_steering():
    antenna = make_test_antenna()  # per-axis max steering = 40 deg
    off_boresight_deg = 60.0
    direction = np.array([np.sin(np.radians(off_boresight_deg)), 0.0, np.cos(np.radians(off_boresight_deg))])
    # Auto-steer (no explicit steering) must reject a 60 deg across_track-plane reflection (> 40 deg limit).
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


def test_antenna_footprint_uses_commanded_front_lobe_when_back_lobe_is_stronger():
    antenna = make_test_antenna()
    antenna.footprint_dB_contour = -3.0
    antenna.footprint_resolution = 32

    theta = np.linspace(0.0, 180.0, 181) * np.pi / 180.0
    phi = np.linspace(0.0, 360.0, 360, endpoint=False) * np.pi / 180.0
    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack(
        [
            np.sin(theta_mesh) * np.cos(phi_mesh),
            np.sin(theta_mesh) * np.sin(phi_mesh),
            np.cos(theta_mesh),
        ],
        axis=-1,
    )

    front_axis = np.array([0.0, np.sin(np.radians(20.0)), np.cos(np.radians(20.0))])
    back_axis = np.array([0.0, np.sin(np.radians(20.0)), -np.cos(np.radians(20.0))])
    front_sep = np.degrees(np.arccos(np.clip(directions @ front_axis, -1.0, 1.0)))
    back_sep = np.degrees(np.arccos(np.clip(directions @ back_axis, -1.0, 1.0)))
    pattern_db = np.maximum(-0.1 - 0.3 * front_sep, -0.3 * back_sep)

    bundle = antenna._find_contour_vectors(pattern_db, 0.0, 20.0)
    mean_axis = np.mean(bundle, axis=0)
    mean_axis = mean_axis / np.linalg.norm(mean_axis)

    assert np.dot(mean_axis, front_axis) > 0.98
    assert np.dot(mean_axis, back_axis) < 0.0


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


def test_directivity_integrates_front_hemisphere_only():
    # The array sits over a ground plane (shielded Earth-facing spacecraft panel), so the
    # isotropic-element mirror back lobe carries no power: a uniform pattern is a hemispheric
    # radiator with D = 2 (3.01 dB), and cos^2(theta) (mirror-symmetric in theta exactly like
    # the array factor) has D = 4*pi / (2*pi/3) = 6. Full-sphere normalization would halve both.
    n_theta, n_phi = 181, 72
    theta_rad = np.linspace(0, np.pi, n_theta)
    phi_rad = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)
    theta_mesh, _phi_mesh = np.meshgrid(theta_rad, phi_rad, indexing="ij")

    directivity, _ = RxAntenna._calculate_directivity(np.ones_like(theta_mesh), theta_rad, phi_rad, theta_mesh)
    assert directivity == pytest.approx(2.0, rel=1e-2)

    directivity, _ = RxAntenna._calculate_directivity(np.cos(theta_mesh) ** 2, theta_rad, phi_rad, theta_mesh)
    assert directivity == pytest.approx(6.0, rel=1e-2)


def test_array_factor_along_track_steers_x_and_across_track_steers_y():
    frequency_hz = FREQUENCY_MHZ * 1e6
    wavelength_m = 299_792_458.0 / frequency_hz
    k = 2.0 * np.pi / wavelength_m
    spacing = wavelength_m / 2.0
    command = np.radians(20.0)
    theta_samples = np.array([[command], [command]])
    phi_x_y = np.array([[0.0], [np.pi / 2.0]])

    along_track_pattern = ArrayFactor(4, 4, frequency_hz, spacing, spacing, -k * spacing * np.sin(command), 0.0)
    along_track_cuts = along_track_pattern.calculate_array_factor(theta_samples, phi_x_y)
    gain_el_x, gain_el_y = along_track_cuts[:, 0]
    assert gain_el_x == pytest.approx(0.0, abs=1e-12)
    assert gain_el_x > gain_el_y

    across_track_pattern = ArrayFactor(4, 4, frequency_hz, spacing, spacing, 0.0, -k * spacing * np.sin(command))
    across_track_cuts = across_track_pattern.calculate_array_factor(theta_samples, phi_x_y)
    gain_az_x, gain_az_y = across_track_cuts[:, 0]
    assert gain_az_y == pytest.approx(0.0, abs=1e-12)
    assert gain_az_y > gain_az_x


@pytest.mark.parametrize(
    "overrides",
    [
        {"elements_along_x": 0},
        {"elements_along_x": 3.5},
        {"d_x": 0.0},
        {"freqs": []},
        {"theta_resolution": 1},
        {"phi_resolution": 10.5},
        {"across_track_steering": [-40.0, 40.0, 4.5]},
        {"across_track_steering": [-90.0, 0.0, 2]},
    ],
)
def test_pattern_generator_rejects_nonphysical_grid_inputs(tmp_path, overrides):
    kwargs = {
        "elements_along_x": 3,
        "elements_along_y": 4,
        "d_x": 0.095,
        "d_y": 0.095,
        "freqs": [FREQUENCY_MHZ],
        "across_track_steering": [-40.0, 40.0, 5],
        "along_track_steering": [-40.0, 40.0, 5],
        "output_folder": tmp_path,
        "theta_resolution": 180,
        "phi_resolution": 360,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        AntennaPatternGenerator(**kwargs)
