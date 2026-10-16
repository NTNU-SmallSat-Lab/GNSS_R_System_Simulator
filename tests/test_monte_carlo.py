"""Deterministic MC event persistence, plotting, aggregation, and resume checks."""

import configparser
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from unittest.mock import Mock

from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from functions.mc_event_plots import _collect_event_powers, _ship_distance_km, make_event_plots, serialize_power_events
from object_definitions.MovingTargetDetection_def import MovingTargetDetectionStruct
import run_monte_carlo as mc

MODES = ["backscatter", "specular"]


def _draw():
    return {
        "lat": 66.0,
        "lon": 6.0,
        "heading_deg": 90.0,
        "epoch_off_h": 0.0,
        "wind_speed_m_s": 5.0,
        "wave_height_m": 1.0,
        "wind_direction_deg": 0.0,
        "inverse_wave_age": 0.8,
        "coherent_integration_time_s": 0.1,
        "num_incoherent_looks": 10,
        "system_noise_temperature_k": 300.0,
    }


def _event(ocean, signature, constellation="GPS", band="L1"):
    return {
        "constellation": constellation,
        "band": band,
        "time": "2025-10-01T12:00:00+00:00",
        "tx_satID": 3,
        "rx_satID": 0,
        "pr_ocean_w": ocean,
        "pr_ship_w": signature,
        "forward_shadow_pr_w": signature,
    }


def _trial(idx, events):
    return {
        **mc._blank_result(idx, _draw()),
        "n_opportunities": len(events),
        "best_metric_db": -30.0,
        "best_secondary_db": 999.0,
        "power_events": events,
    }


def _detector_event(ocean, signature, metric=-30.0, secondary=-20.0):
    event = MovingTargetDetectionStruct(
        time=datetime(2025, 10, 1, 12, tzinfo=UTC),
        band="L1",
        pr_ocean_w=np.float64(ocean),
        pr_ship_w=signature,
        forward_shadow_pr_w=signature,
        # Deliberately keep signed delta negative; the saved signature must be the magnitude.
        forward_delta_pr_w=-signature if signature is not None else None,
        sinr_ship_db=metric,
        scr_db=secondary,
        forward_delta_pr_over_noise_db=metric,
        forward_delta_pr_fraction_db=secondary,
        is_sinr_detection=False,
    )
    event.tx_satID = np.int64(3)
    event.rx_satID = np.int64(0)
    return event


@pytest.mark.parametrize("mode", MODES)
def test_serialization_retains_identity_and_below_threshold_events(mode):
    events = [_detector_event(1e-12, 1e-14), _detector_event(1e-15, 1e-17), _detector_event(1e-12, np.nan)]
    records = serialize_power_events({"GPS": events, "Galileo": []}, mode)
    decoded = json.loads(json.dumps(records, allow_nan=False))
    assert len(decoded) == 3
    assert decoded[0]["constellation"] == "GPS"
    assert decoded[0]["band"] == "L1"
    assert decoded[0]["time"] == "2025-10-01T12:00:00+00:00"
    assert decoded[0]["tx_satID"] == 3
    assert decoded[0]["rx_satID"] == 0
    signature_key = "pr_ship_w" if mode == "backscatter" else "forward_shadow_pr_w"
    assert decoded[0][signature_key] == 1e-14
    assert decoded[2][signature_key] is None
    groups, counts = _collect_event_powers([_trial(0, decoded)], mode)
    np.testing.assert_allclose(groups["GPS"], [[-120, -140, -20], [-150, -170, -20]])
    assert counts["n_events"] == 2


@pytest.mark.parametrize("mode", MODES)
def test_known_pairs_and_event_weighting(mode):
    results = [
        _trial(0, [_event(1e-12, 1e-14), _event(1e-15, 1e-14, band="L5")]),
        _trial(1, [_event(1e-10, 1e-10, "Galileo", "E1")]),
    ]
    groups, counts = _collect_event_powers(results, mode)
    np.testing.assert_allclose(groups["GPS"], [[-120, -140, -20], [-150, -140, 10]])
    np.testing.assert_allclose(groups["Galileo"], [[-100, -100, 0]])
    assert counts["n_events"] == 3
    assert counts["n_contributing_trials"] == 2
    # Trial best_secondary_db=999 must never enter the event ratios.
    assert max(rows[:, 2].max() for rows in groups.values()) == 10


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("invalid", [None, 0, -1, float("nan"), float("inf"), -float("inf"), "bad", True])
def test_invalid_pairs_are_excluded_together(mode, invalid):
    result = _trial(0, [_event(1e-12, invalid), _event(invalid, 1e-14)])
    groups, counts = _collect_event_powers([result], mode)
    assert groups == {}
    assert counts["n_invalid_events"] == 2
    assert counts["n_contributing_trials"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_extreme_finite_powers_do_not_overflow_ratio(mode):
    groups, _ = _collect_event_powers([_trial(0, [_event(1e-300, 1e300)])], mode)
    np.testing.assert_allclose(groups["GPS"], [[-3000, 3000, 6000]])


@pytest.mark.parametrize("mode", MODES)
def test_legacy_mixed_empty_and_invalid_trials(mode):
    legacy = _trial(0, [_event(1, 1)])
    del legacy["power_events"]
    results = [
        legacy,
        _trial(1, []),
        _trial(2, [_event(1e-12, 1e-14), _event(1e-11, 1e-13), _event(0, 1)]),
        _trial(3, [_event(None, 1)]),
    ]
    groups, counts = _collect_event_powers(results, mode)
    assert len(groups["GPS"]) == 2
    assert counts == {
        "n_trials": 4,
        "n_legacy_trials": 1,
        "n_contributing_trials": 1,
        "n_events": 2,
        "n_invalid_events": 2,
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("records", [[], [{"best_secondary_db": 12.0}], [{"power_events": []}], [{"power_events": [_event(0, 1)]}]])
def test_no_suitable_events_skip_new_plots(mode, records, tmp_path, capsys):
    existing = tmp_path / f"mc_{mode}_received_power.png"
    existing.write_bytes(b"previous result")
    assert make_event_plots(records, mode, tmp_path) == []
    assert existing.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [existing]
    assert "skipping new event plots" in capsys.readouterr().out


@pytest.fixture
def trial_pipeline(monkeypatch):
    def prepare(mode, events, visible=True):
        base = {
            "reflection_type": mode,
            "start_time": datetime(2025, 10, 1, tzinfo=UTC),
            "moving_target_speed": 0.1,
            "window_hours_mc": 1.0,
            "link_budget": {"detection_sinr_threshold_db": 0.0},
            "GNSS_constellations": {},
            "orbits": [{}],
            "No_satellites": 1,
            "step_size": 10,
            "earth_model": "spherical",
            "detection_bbox_buffer_deg": 1.0,
            "max_central_angle_deg": 80.0,
            "max_incidence_angle_deg": 70.0,
            "ship": {},
            "selected_signals": [],
        }
        mtd = Mock(target_detect_dict={"GPS": events}, gnss_constellation={})
        ship = Mock()
        ship.get_travel_duration_hours.return_value = 1.0
        monkeypatch.setitem(mc._SHARED, "base", base)
        monkeypatch.setitem(mc._SHARED, "antenna", object())
        monkeypatch.setattr(mc, "ShipTrajectory", Mock(return_value=ship))
        monkeypatch.setattr(mc, "Satellite", Mock())
        monkeypatch.setattr(mc, "SatelliteConstellation", Mock())
        monkeypatch.setattr(mc, "MovingTargetDetection", Mock(return_value=mtd))
        monkeypatch.setattr(mc.fn, "load_gnss_constellation_dict", Mock(return_value={"GPS": Mock()}))
        monkeypatch.setattr(mc.fn, "compute_visible_times_and_resample_ship", Mock(return_value=(None, {0: [1] if visible else []})))
        monkeypatch.setattr(mc.fn, "propagate_only_visible_satellites", Mock())
        return mtd

    return prepare


@pytest.mark.parametrize("mode", MODES)
def test_run_trial_adds_all_events_without_changing_best_metrics(mode, trial_pipeline):
    trial_pipeline(mode, [_detector_event(1e-12, 1e-14, -30, -20), _detector_event(1e-15, 1e-17, -40, -10)])
    result = mc._run_trial((7, _draw()))
    assert result["error"] is None
    assert result["idx"] == 7
    assert result["n_opportunities"] == 2
    assert result["best_metric_db"] == -30.0
    assert result["best_secondary_db"] == -10.0
    assert result["detected"] is False
    groups, counts = _collect_event_powers([result], mode)
    np.testing.assert_allclose(groups["GPS"], [[-120, -140, -20], [-150, -170, -20]])
    assert counts["n_events"] == 2


@pytest.mark.parametrize("visible", [False, True])
def test_run_trial_with_no_events(visible, trial_pipeline):
    trial_pipeline("backscatter", [], visible=visible)
    result = mc._run_trial((0, _draw()))
    assert result["power_events"] == []
    assert result["n_opportunities"] == 0
    assert result["detected"] is False
    assert result["error"] is None


def test_failed_trial_has_empty_event_records(trial_pipeline):
    mtd = trial_pipeline("backscatter", [])
    mtd.calculate_moving_target_reflection_intersection.side_effect = ValueError("fixture failure")
    result = mc._run_trial((0, _draw()))
    assert result["power_events"] == []
    assert result["error"] == "ValueError: fixture failure"


@pytest.mark.parametrize("mode,ratio_suffix", [("backscatter", "scr"), ("specular", "shadow_fraction")])
def test_standalone_pngs_show_event_population(mode, ratio_suffix, tmp_path, monkeypatch):
    legacy = _trial(9, [])
    del legacy["power_events"]
    results = [
        _trial(0, [_event(1e-12, 1e-14), _event(1e-15, 1e-14, band="L5")]),
        _trial(1, [_event(1e-10, 1e-11, "Galileo", "E1"), _event(0, 1)]),
        legacy,
    ]
    figures = []
    savefig = Figure.savefig

    def capture(fig, *args, **kwargs):
        figures.append(fig)
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = make_event_plots(results, mode, tmp_path)
    assert {Path(path).name for path in paths} == {f"mc_{mode}_received_power.png", f"mc_{mode}_{ratio_suffix}.png"}
    for path, fig in zip(paths, figures, strict=True):
        with Image.open(path) as png:
            assert png.format == "PNG"
            assert png.size == (1275, 975)
            png.verify()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        title = fig._suptitle.get_text()
        assert "Selected bands combined" in title
        assert "3 events" in title
        assert "2 contributing trials / 3 total trials" in title
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "GPS (2 events)" in labels
        assert "Galileo (1 events)" in labels
        text = " ".join([title, ax.get_title(), ax.get_xlabel(), ax.get_ylabel(), *labels]).lower()
        assert "threshold" not in text
        assert "detection" not in text
    scatter_ax, hist_ax = [fig.axes[0] for fig in figures]
    assert "[dBW]" in scatter_ax.get_xlabel()
    assert "[dBW]" in scatter_ax.get_ylabel()
    assert "[dB]" in hist_ax.get_xlabel()
    assert hist_ax.get_ylabel() == "Events"
    assert sum(len(collection.get_offsets()) for collection in scatter_ax.collections) == 3
    assert len(hist_ax.lines) == 0  # no threshold lines
    assert len(scatter_ax.lines) == (1 if mode == "backscatter" else 0)
    if mode == "backscatter":
        np.testing.assert_allclose(scatter_ax.lines[0].get_xdata(), scatter_ax.lines[0].get_ydata())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("population", ["legacy", "mixed", "new"])
def test_aggregate_cli_preserves_summaries_and_never_initializes_trials(mode, population, tmp_path, monkeypatch, capsys):
    results = [
        _trial(1, [_event(1e-12, 1e-14)]),
        _trial(0, [_event(1e-11, 1e-12, "Galileo", "E1")]),
    ]
    results[0]["detected"] = True
    results[0]["best_metric_db"] = 2.0
    if population in {"mixed", "legacy"}:
        del results[0]["power_events"]
    if population == "legacy":
        del results[1]["power_events"]
    jsonl = tmp_path / f"mc_{mode}_trials.jsonl"
    jsonl.write_text("".join(json.dumps(result) + "\n" for result in results))
    original = jsonl.read_bytes()
    config_path = tmp_path / "report.ini"
    # Minimal reporting config proves aggregate does not depend on antenna/TLE/MC inputs.
    config_path.write_text(f'[REFLECTION]\nreflection_type = "{mode}"\n[LINK BUDGET]\ndetection_sinr_threshold_db = 0.0\n')
    for name in ("initialize", "draw_trials", "run_trials_timeout_safe", "_run_trial", "_init_worker"):
        monkeypatch.setattr(mc, name, Mock(side_effect=AssertionError(f"{name} must not run")))
    monkeypatch.setattr(sys, "argv", ["run_monte_carlo.py", str(config_path), "aggregate", "--output-dir", str(tmp_path)])
    mc.main()
    assert jsonl.read_bytes() == original
    report = json.loads((tmp_path / f"mc_{mode}_results.json").read_text())
    assert report["summary"] == {
        "mode": mode,
        "n_trials": 2,
        "detection_threshold_db": 0.0,
        "pd_overall": 0.5,
        "p_detect_given_opportunity": 0.5,
        "opportunity_rate": 1.0,
        "mean_opportunities_per_trial": 1.0,
        "n_errors": 0,
    }
    assert report["trials"] == sorted(results, key=lambda trial: trial["idx"])
    for suffix in ("distribution", "pd_vs_tcoh", "pd_vs_looks", "metric_vs_tsys"):
        assert (tmp_path / f"mc_{mode}_{suffix}.png").is_file()
    assert (tmp_path / f"mc_{mode}_received_power.png").exists() == (population != "legacy")
    assert len(list(tmp_path.glob("*.png"))) == (4 if population == "legacy" else 6)
    output = capsys.readouterr().out
    assert f"excluded {2 if population == 'legacy' else 1 if population == 'mixed' else 0} trials" in output


def test_aggregate_empty_file_does_not_create_outputs(tmp_path, capsys):
    mc.aggregate_and_report("backscatter", 0, tmp_path / "absent.jsonl", tmp_path / "summary.json", tmp_path)
    assert not list(tmp_path.iterdir())
    assert "nothing to aggregate" in capsys.readouterr().out


def test_resume_skips_legacy_and_new_completed_indices(tmp_path, monkeypatch):
    completed = [_trial(0, []), _trial(1, [])]
    del completed[0]["power_events"]
    jsonl = tmp_path / "mc_backscatter_trials.jsonl"
    jsonl.write_text("".join(json.dumps(record) + "\n" for record in completed))
    original = jsonl.read_bytes()
    config = configparser.ConfigParser()
    config.read("Configuration_mc_backscatter.ini")
    settings = {"reflection_type": "backscatter", "link_budget": {"detection_sinr_threshold_db": 0}}
    monkeypatch.setattr(mc, "initialize", Mock(return_value=(config, settings)))
    draws = [_draw(), _draw(), _draw()]
    monkeypatch.setattr(mc, "draw_trials", Mock(return_value=draws))
    # No coastline files are needed for the resume fixture.
    import geopandas as gpd

    monkeypatch.setattr(gpd, "read_file", Mock(side_effect=OSError("fixture has no ocean polygons")))
    runner = Mock(return_value=iter([_trial(2, [])]))
    monkeypatch.setattr(mc, "run_trials_timeout_safe", runner)
    monkeypatch.setattr(mc, "aggregate_and_report", Mock())
    monkeypatch.setattr(sys, "argv", ["run_monte_carlo.py", "fixture.ini", "3", "--output-dir", str(tmp_path)])
    mc.main()
    assert runner.call_args.args[1] == [(2, draws[2])]
    assert jsonl.read_bytes().startswith(original)
    assert [record["idx"] for record in mc._read_jsonl(jsonl)] == [0, 1, 2]
    # With every index now saved, running again must not invoke the runner.
    runner.reset_mock()
    mc.main()
    runner.assert_not_called()


def _large_result_worker(settings, task, out_queue):
    """Spawn-safe fixture, large enough to fill an OS pipe without parent reads."""
    idx, draw = task
    result = mc._blank_result(idx, draw)
    result["power_events"] = [_event(1e-12, 1e-14, band=str(i)) for i in range(10000)]
    out_queue.put(result)


def test_large_event_payload_is_drained_before_worker_exit(monkeypatch):
    monkeypatch.setattr(mc, "_trial_worker_entry", _large_result_worker)
    results = list(mc.run_trials_timeout_safe({}, [(0, _draw())], n_workers=1, timeout_s=30))
    assert len(results) == 1
    assert results[0]["error"] is None
    assert len(results[0]["power_events"]) == 10000


def test_ship_distance_from_reflection_point():
    from shapely.geometry import Point

    event = _detector_event(1e-12, 1e-14)
    assert _ship_distance_km(event) is None  # positions unavailable
    event.pos_tgt_i = Point(10.1, 58.3)  # ship: x = lon, y = lat
    event.reflection_point_ll = (58.3, 10.1)
    assert _ship_distance_km(event) == pytest.approx(0.0, abs=1e-9)
    event.reflection_point_ll = (59.3, 10.1)  # one degree of latitude away
    assert _ship_distance_km(event) == pytest.approx(111.19, abs=0.1)
    assert serialize_power_events({"GPS": [event]}, "backscatter")[0]["ship_dist_km"] == pytest.approx(111.19, abs=0.1)
