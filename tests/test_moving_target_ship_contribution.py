"""Tests for the ship-contribution path in MovingTargetDetection.

These exercise the two new private methods directly on an instance that has bypassed __init__,
so we don't need a propagated constellation or a real ShipTrajectory shapefile.
"""

import datetime as dt
from types import SimpleNamespace

from _link_budget_settings import link_budget_settings
import numpy as np
import pytest

import functions.link_budget as lb
from functions.link_budget import compute_reflection_link_budget
from object_definitions.Constants_def import Constants
from object_definitions.MovingTargetDetection_def import (
    MovingTargetDetection,
    MovingTargetDetectionStruct,
)


@pytest.fixture
def mtd():
    """A MovingTargetDetection instance that has bypassed __init__; populate per-test."""
    return object.__new__(MovingTargetDetection)


BASE_TIME = dt.datetime(2026, 5, 27, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def detection_time():
    return BASE_TIME + dt.timedelta(minutes=25)             # mid-trajectory


def _trajectory(points, interval_minutes=10):
    timestamps = [BASE_TIME + dt.timedelta(minutes=interval_minutes * i) for i in range(len(points))]
    return SimpleNamespace(timestamps=timestamps, trajectory_points=points)


class TestComputeShipHeadingEci:
    def test_short_trajectory_returns_none(self, mtd, detection_time):
        # < 2 waypoints -> no heading definable.
        mtd.moving_target_object = _trajectory([(60.0, 5.0)])
        assert mtd._compute_ship_heading_eci(detection_time) is None

    def test_degenerate_segment_returns_none(self, mtd, detection_time):
        # Bracketing segment has identical endpoints -> stationary, no heading.
        mtd.moving_target_object = _trajectory([(60.0, 5.0)] * 5)
        assert mtd._compute_ship_heading_eci(detection_time) is None

    def test_mid_trajectory_uses_bracketing_segment(self, mtd, detection_time):
        # Detection at 12:25 -> falls inside the segment [12:20, 12:30] = points[2..3].
        points = [(60.0 + 0.1 * i, 5.0) for i in range(5)]   # 5 waypoints, drifting north
        mtd.moving_target_object = _trajectory(points)
        heading = mtd._compute_ship_heading_eci(detection_time)
        assert heading is not None
        assert heading.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)

    def test_detection_before_trajectory_uses_first_segment(self, mtd):
        # detection_time before the first waypoint -> use first segment (initial bearing).
        points = [(60.0 + 0.1 * i, 5.0) for i in range(5)]
        mtd.moving_target_object = _trajectory(points)
        early_detection = BASE_TIME - dt.timedelta(minutes=5)
        heading = mtd._compute_ship_heading_eci(early_detection)
        assert heading is not None                          # initial bearing is defined
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)

    def test_detection_after_trajectory_uses_last_segment(self, mtd):
        # detection_time past the last waypoint -> use final segment.
        points = [(60.0 + 0.1 * i, 5.0) for i in range(5)]
        mtd.moving_target_object = _trajectory(points)
        late_detection = BASE_TIME + dt.timedelta(hours=10)
        heading = mtd._compute_ship_heading_eci(late_detection)
        assert heading is not None
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)

    def test_interval_change_still_works(self, mtd):
        # Same logic should hold for a 5-min trajectory without code changes — the
        # helper reads waypoints directly, not the interval setting.
        points = [(60.0 + 0.1 * i, 5.0) for i in range(10)]
        mtd.moving_target_object = _trajectory(points, interval_minutes=5)
        detection = BASE_TIME + dt.timedelta(minutes=12)    # inside segment [10, 15]
        heading = mtd._compute_ship_heading_eci(detection)
        assert heading is not None
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)


def _build_refl_obj_mock():
    """One-sample refl_obj with the fields the helper reads from."""
    # Tx and Rx both directly above the surface point, near-zenith geometry.
    surf = np.array([[6.371e6, 0.0, 0.0]])
    tx_pos = np.array([[6.371e6 + 2.0e7], [0.0], [0.0]])   # ~ GPS altitude
    rx_pos = np.array([[6.371e6 + 7.0e5], [0.0], [0.0]])   # ~ LEO altitude
    rx_vel = np.array([[0.0], [7_500.0], [0.0]])
    return SimpleNamespace(
        gnss_pos_vec=tx_pos,
        sat_pos_vec=rx_pos,
        sat_vel_vec=rx_vel,
        refl_ECI_vec=surf,
        link_budget={
            "tx_to_surface_range_m": np.array([np.nan]),
            "surface_to_rx_range_m": np.array([np.nan]),
            "rx_gain_dbi": np.array([np.nan]),
            "received_power_w": np.array([1.0e-15]),
            "rcs_m2": np.array([1.0e3]),
        },
    )


def _constant_gain_antenna():
    class ConstantGainAntenna:
        R_antenna_to_body = np.eye(3)

        def resolve_steering_angles_for_direction(self, direction):
            return (0.0, 0.0) if np.isfinite(direction).all() else None

        def get_gain_dbi_for_direction(self, direction, frequency_mhz, azimuth_deg=None, elevation_deg=None):
            return 10.0 if np.isfinite(direction).all() else np.nan

    return ConstantGainAntenna()


class _RecordingFixedSteerAntenna:
    R_antenna_to_body = np.eye(3)

    def __init__(self):
        self.resolved_direction = None
        self.gain_calls = []

    def resolve_steering_angles_for_direction(self, direction):
        self.resolved_direction = np.asarray(direction, dtype=float)
        return (11.0, -7.0)

    def get_gain_dbi_for_direction(self, direction, frequency_mhz, azimuth_deg=None, elevation_deg=None):
        self.gain_calls.append(
            {
                "direction": np.asarray(direction, dtype=float),
                "frequency_mhz": frequency_mhz,
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
            }
        )
        return 6.0


class TestComputeShipPrAtEvent:
    def test_returns_nan_when_ship_dimensions_is_none(self, mtd, detection_time):
        mtd.ship_dimensions = None
        mtd.moving_target_object = _trajectory([(60.0 + 0.1 * i, 5.0) for i in range(5)])
        refl_obj = _build_refl_obj_mock()
        pr_ship_w, rcs_ship_m2, snr_ship_rf_db, noise_power_w = mtd._compute_ship_pr_at_event(
            refl_obj, 0, 1.57542e9, 27.0, detection_time, (0.0, 0.0)
        )
        assert np.isnan(pr_ship_w)
        assert np.isnan(rcs_ship_m2)
        assert np.isnan(snr_ship_rf_db)
        assert np.isnan(noise_power_w)

    def test_returns_nan_when_ship_is_stationary(self, mtd, detection_time):
        mtd.ship_dimensions = {"length_m": 100.0, "beam_m": 15.0, "freeboard_m": 5.0}
        mtd.moving_target_object = _trajectory([(60.0, 5.0)] * 5)
        mtd.link_budget_settings = link_budget_settings()
        mtd.sat_antenna = _constant_gain_antenna()
        refl_obj = _build_refl_obj_mock()
        pr_ship_w, rcs_ship_m2, snr_ship_rf_db, noise_power_w = mtd._compute_ship_pr_at_event(
            refl_obj, 0, 1.57542e9, 27.0, detection_time, (0.0, 0.0)
        )
        assert np.isnan(pr_ship_w)
        assert np.isnan(rcs_ship_m2)
        assert np.isnan(snr_ship_rf_db)
        assert np.isnan(noise_power_w)

    def test_returns_finite_pr_for_well_formed_event(self, mtd, detection_time):
        mtd.ship_dimensions = {"length_m": 100.0, "beam_m": 15.0, "freeboard_m": 5.0}
        mtd.moving_target_object = _trajectory([(0.0, 0.1 * i) for i in range(5)])
        mtd.link_budget_settings = link_budget_settings()
        mtd.sat_antenna = _constant_gain_antenna()
        refl_obj = _build_refl_obj_mock()
        pr_ship_w, rcs_ship_m2, snr_ship_rf_db, noise_power_w = mtd._compute_ship_pr_at_event(
            refl_obj, 0, 1.57542e9, 27.0, detection_time, (0.0, 0.0)
        )
        # Geometry is near-zenith with a moving ship — deck plate dominates, RCS finite and positive.
        assert np.isfinite(rcs_ship_m2)
        assert rcs_ship_m2 > 0.0
        assert np.isfinite(pr_ship_w)
        assert pr_ship_w > 0.0
        assert np.isfinite(snr_ship_rf_db)
        assert noise_power_w > 0.0

    def test_ship_gain_uses_reflection_point_steering(self, mtd, detection_time, monkeypatch):
        center_direction = np.array([0.0, 0.0, 1.0])
        ship_direction = np.array([0.2, 0.0, np.sqrt(1.0 - 0.2**2)])

        def fake_look_direction(_sat_pos_eci, _sat_vel_eci, target_pos_eci, _antenna, q_sat=None):
            if np.allclose(target_pos_eci, refl_obj.refl_ECI_vec[0]):
                return center_direction
            return ship_direction

        mtd.ship_dimensions = {"length_m": 100.0, "beam_m": 15.0, "freeboard_m": 5.0}
        mtd.moving_target_object = _trajectory([(0.0, 0.1 * i) for i in range(5)])
        mtd.link_budget_settings = link_budget_settings()
        mtd.sat_antenna = _RecordingFixedSteerAntenna()
        refl_obj = _build_refl_obj_mock()
        monkeypatch.setattr(lb, "compute_rx_look_direction_antenna_frame", fake_look_direction)

        pr_ship_w, rcs_ship_m2, snr_ship_rf_db, noise_power_w = mtd._compute_ship_pr_at_event(
            refl_obj, 0, 1.57542e9, 27.0, detection_time, (0.0, 0.0)
        )

        assert np.isfinite(pr_ship_w)
        assert np.isfinite(rcs_ship_m2)
        assert np.isfinite(snr_ship_rf_db)
        assert noise_power_w > 0.0
        np.testing.assert_allclose(mtd.sat_antenna.resolved_direction, center_direction)
        assert len(mtd.sat_antenna.gain_calls) == 1
        gain_call = mtd.sat_antenna.gain_calls[0]
        np.testing.assert_allclose(gain_call["direction"], ship_direction)
        assert gain_call["azimuth_deg"] == 11.0
        assert gain_call["elevation_deg"] == -7.0


class TestForwardShadowDeltaPr:
    def test_returns_nan_when_ship_dimensions_is_none(self, mtd, detection_time):
        mtd.ship_dimensions = None
        refl_obj = _build_refl_obj_mock()
        result = mtd._compute_forward_shadow_delta_pr_at_event(refl_obj, 0, 1.57542e9, 27.0, detection_time, (0.0, 0.0))
        assert all(np.isnan(v) for v in result)

    def test_returns_finite_shadow_power_for_well_formed_event(self, mtd, detection_time):
        # The ocean glint power the ship occults over its waterline area; finite and positive.
        mtd.ship_dimensions = {"length_m": 100.0, "beam_m": 15.0, "freeboard_m": 5.0}
        mtd.link_budget_settings = link_budget_settings()
        mtd.sat_antenna = _constant_gain_antenna()
        refl_obj = _build_refl_obj_mock()
        pr_shadow_w, rcs_shadow_m2, noise_power_w = mtd._compute_forward_shadow_delta_pr_at_event(
            refl_obj, 0, 1.57542e9, 27.0, detection_time, (0.0, 0.0)
        )
        assert np.isfinite(pr_shadow_w) and pr_shadow_w > 0.0
        assert np.isfinite(rcs_shadow_m2) and rcs_shadow_m2 > 0.0
        assert noise_power_w > 0.0


class TestForwardDetectionMetrics:
    def test_populates_signed_delta_and_ratios(self):
        event = MovingTargetDetectionStruct(pr_ocean_w=1.0e-12, noise_power_w=1.0e-15, forward_shadow_pr_w=1.0e-15)
        MovingTargetDetection._populate_forward_detection_metrics(event)
        np.testing.assert_allclose(event.forward_delta_pr_w, -1.0e-15)              # signed: ship removes glint
        np.testing.assert_allclose(event.forward_delta_pr_fraction_db, 10.0 * np.log10(1.0e-15 / 1.0e-12))  # -30 dB
        np.testing.assert_allclose(event.forward_delta_pr_over_noise_db, 0.0, atol=1e-12)                   # shadow == noise

    def test_no_op_without_shadow_power(self):
        event = MovingTargetDetectionStruct(pr_ocean_w=1.0e-12, noise_power_w=1.0e-15)
        MovingTargetDetection._populate_forward_detection_metrics(event)
        assert event.forward_delta_pr_w is None
        assert event.forward_delta_pr_fraction_db is None


class TestForwardShadowSummary:
    def test_returns_false_when_no_forward_data(self, mtd):
        mtd.gnss_constellation = {"GPS": SimpleNamespace()}
        mtd.target_detect_dict = {"GPS": [MovingTargetDetectionStruct(band="L1")]}
        assert mtd.show_forward_shadow_summary(band="L1") is False

    def test_returns_true_and_prints_with_forward_data(self, mtd, capsys):
        mtd.gnss_constellation = {"GPS": SimpleNamespace()}
        event = MovingTargetDetectionStruct(band="L1", pr_ocean_w=1.0e-12, noise_power_w=1.0e-15, forward_shadow_pr_w=1.0e-15)
        MovingTargetDetection._populate_forward_detection_metrics(event)
        mtd.target_detect_dict = {"GPS": [event]}
        assert mtd.show_forward_shadow_summary(band="L1") is True
        assert "FORWARD-SCATTER SHADOW SUMMARY" in capsys.readouterr().out


class TestDetectionStructDefaults:
    def test_legacy_path_leaves_ship_fields_none(self):
        # An untouched struct keeps all the per-event Pr fields as None — what the loop will see
        # when ship_dimensions is None (ocean-only / legacy mode).
        s = MovingTargetDetectionStruct()
        assert s.pr_ocean_w is None
        assert s.pr_ship_w is None
        assert s.pr_total_w is None
        assert s.scr_db is None
        assert s.noise_power_w is None
        assert s.snr_ocean_rf_db is None
        assert s.snr_ship_rf_db is None
        assert s.snr_total_rf_db is None
        assert s.sinr_ship_db is None
        assert s.detection_sinr_threshold_db is None
        assert s.detection_margin_db is None
        assert s.is_sinr_detection is None
        assert s.rcs_ocean_m2 is None
        assert s.rcs_ship_m2 is None


class TestDetectionMetrics:
    def test_populates_ship_sinr_and_pass_flag(self):
        event = MovingTargetDetectionStruct(
            pr_ocean_w=2.0,
            pr_ship_w=8.0,
            noise_power_w=2.0,
            detection_sinr_threshold_db=3.0,
        )

        MovingTargetDetection._populate_detection_metrics(event)

        np.testing.assert_allclose(event.pr_total_w, 10.0)
        np.testing.assert_allclose(event.scr_db, 10.0 * np.log10(8.0 / 2.0))
        np.testing.assert_allclose(event.snr_total_rf_db, 10.0 * np.log10(10.0 / 2.0))
        np.testing.assert_allclose(event.sinr_ship_db, 10.0 * np.log10(8.0 / (2.0 + 2.0)))
        np.testing.assert_allclose(event.detection_margin_db, event.sinr_ship_db - 3.0)
        assert event.is_sinr_detection is True

    def test_populates_ship_sinr_and_fail_flag(self):
        event = MovingTargetDetectionStruct(
            pr_ocean_w=2.0,
            pr_ship_w=4.0,
            noise_power_w=2.0,
            detection_sinr_threshold_db=3.1,
        )

        MovingTargetDetection._populate_detection_metrics(event)

        np.testing.assert_allclose(event.sinr_ship_db, 0.0)
        np.testing.assert_allclose(event.detection_margin_db, -3.1)
        assert event.is_sinr_detection is False


class TestLinkBudgetContributionPlot:
    def test_plot_includes_lb4c_noise_and_sinr_metrics(self, mtd, tmp_path, monkeypatch, capsys):
        mtd.gnss_constellation = {"GPS": SimpleNamespace(), "GALILEO": SimpleNamespace()}
        pass_event = MovingTargetDetectionStruct(
            band="L1",
            pr_ocean_w=1.0e-16,
            pr_ship_w=1.0e-14,
            noise_power_w=1.0e-15,
            detection_sinr_threshold_db=0.0,
        )
        fail_event = MovingTargetDetectionStruct(
            band="L1",
            pr_ocean_w=1.0e-14,
            pr_ship_w=1.0e-15,
            noise_power_w=1.0e-15,
            detection_sinr_threshold_db=0.0,
        )
        MovingTargetDetection._populate_detection_metrics(pass_event)
        MovingTargetDetection._populate_detection_metrics(fail_event)
        mtd.target_detect_dict = {"GPS": [pass_event, fail_event], "GALILEO": []}
        monkeypatch.setattr(Constants, "output_folder_path", str(tmp_path))

        mtd.show_link_budget_contributions(file_name="lb4c_plot.png", band="L1")

        # Each metric is written to its own suffixed figure.
        for suffix in ("power", "scr_hist", "sinr_hist", "margin"):
            panel_file = tmp_path / f"lb4c_plot_{suffix}.png"
            assert panel_file.is_file()
            assert panel_file.stat().st_size > 0
        assert "SINR-qualified (threshold 0.00 dB):   1 / 2" in capsys.readouterr().out

    def test_returns_true_when_plots_produced(self, mtd, tmp_path, monkeypatch):
        mtd.gnss_constellation = {"GPS": SimpleNamespace()}
        event = MovingTargetDetectionStruct(
            band="L1", pr_ocean_w=1.0e-16, pr_ship_w=1.0e-14, noise_power_w=1.0e-15, detection_sinr_threshold_db=0.0,
        )
        MovingTargetDetection._populate_detection_metrics(event)
        mtd.target_detect_dict = {"GPS": [event]}
        monkeypatch.setattr(Constants, "output_folder_path", str(tmp_path))

        assert mtd.show_link_budget_contributions(file_name="lb_plot.png", band="L1") is True

    def test_returns_false_and_writes_nothing_without_ship_pr(self, mtd, tmp_path, monkeypatch):
        # Specular-mode case: ocean Pr only, no ship backscatter Pr -> no SCR/SINR plots.
        mtd.gnss_constellation = {"GPS": SimpleNamespace()}
        event = MovingTargetDetectionStruct(band="L1", pr_ocean_w=1.0e-16)   # pr_ship_w stays None
        mtd.target_detect_dict = {"GPS": [event]}
        monkeypatch.setattr(Constants, "output_folder_path", str(tmp_path))

        assert mtd.show_link_budget_contributions(file_name="lb_plot.png", band="L1") is False
        assert list(tmp_path.iterdir()) == []


def _build_multi_sample_refl_obj(n=4):
    """Multi-sample refl_obj (spherical) for the ocean per-event equivalence checks."""
    R = 6.371e6
    incs = np.array([5.0, 15.0, 30.0, 45.0])[:n]
    surf = np.tile(np.array([R, 0.0, 0.0]), (n, 1))
    tx = np.zeros((3, n))
    rx = np.zeros((3, n))
    rxv = np.zeros((3, n))
    for i, th_deg in enumerate(incs):
        th = np.radians(th_deg)
        tx[:, i] = surf[i] + 2.0e7 * np.array([np.cos(th), 0.0, np.sin(th)])
        rx[:, i] = surf[i] + 7.0e5 / np.cos(th) * np.array([np.cos(th), 0.0, -np.sin(th)])
        rxv[:, i] = [0.0, 7_500.0, 0.0]
    return SimpleNamespace(
        gnss_pos_vec=tx,
        sat_pos_vec=rx,
        sat_vel_vec=rxv,
        refl_ECI_vec=surf,
        refl_ang_vec=incs,
        refl_ECEF_vec=None,          # spherical => geocentric normals, no ECEF needed
        time_vec=[BASE_TIME] * n,
    )


def _ocean_mtd(mtd, antenna):
    mtd.earth_model = "spherical"
    mtd.sat_antenna = antenna
    mtd.link_budget_settings = link_budget_settings()
    return mtd


def _full_series_budget(refl_obj, antenna):
    return compute_reflection_link_budget(
        tx_positions_eci=refl_obj.gnss_pos_vec,
        rx_positions_eci=refl_obj.sat_pos_vec,
        rx_velocities_eci=refl_obj.sat_vel_vec,
        reflection_positions_eci=refl_obj.refl_ECI_vec,
        incidence_deg=refl_obj.refl_ang_vec,
        antenna=antenna,
        frequency_hz=1.57542e9,
        tx_eirp_dbw=27.0,
        settings=link_budget_settings(),
    )


class TestComputeOceanPrAtEvent:
    """Deferred per-event ocean budget must equal the eager full-series value bit-for-bit."""

    def test_per_event_matches_full_series(self, mtd):
        # The deferred slice-and-recompute must reproduce the idx-th element of the full-array budget
        # the propagation used to cache, across several distinct geometries.
        refl_obj = _build_multi_sample_refl_obj()
        antenna = _constant_gain_antenna()
        full = _full_series_budget(refl_obj, antenna)
        _ocean_mtd(mtd, antenna)
        for i in range(refl_obj.refl_ECI_vec.shape[0]):
            pr_evt, rcs_evt, snr_evt, noise_evt = mtd._compute_ocean_pr_at_event(refl_obj, i, 1.57542e9, 27.0)
            np.testing.assert_allclose(pr_evt, full["received_power_w"][i], rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(rcs_evt, full["rcs_m2"][i], rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(snr_evt, full["snr_rf_db"][i], rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(noise_evt, full["noise_power_w"], rtol=1e-12, atol=0.0)

    def test_invalid_sample_is_nan_in_both(self, mtd):
        # An invalid reflection point is NaN in the full series, and the per-event recompute agrees.
        refl_obj = _build_multi_sample_refl_obj()
        refl_obj.refl_ECI_vec[2] = np.nan
        antenna = _constant_gain_antenna()
        full = _full_series_budget(refl_obj, antenna)
        _ocean_mtd(mtd, antenna)
        pr_evt, _, snr_evt, noise_evt = mtd._compute_ocean_pr_at_event(refl_obj, 2, 1.57542e9, 27.0)
        assert np.isnan(pr_evt)
        assert np.isnan(snr_evt)
        assert noise_evt > 0.0
        assert np.isnan(full["received_power_w"][2])
