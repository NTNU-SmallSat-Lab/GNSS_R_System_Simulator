"""
Tests for build_selected_signals — the warn-and-drop band validation behind [ANTENNA].bands.

Bands are constellation-specific signals (L1=GPS, E1=Galileo, B1I=BeiDou, ...). Invalid bands must
be warned about and dropped (never raise), so a typo or an unmatched band never aborts a run.
"""

import logging

from __init__ import build_selected_signals
from object_definitions.Constants_def import Constants

LOADED = ["GPS", "GALILEO", "BEIDOU"]


def _by_band(signals):
    return {s["band"]: s for s in signals}


class TestValidBands:
    def test_single_valid_band_resolves_constellation_freq_eirp(self):
        signals = build_selected_signals(["L1"], LOADED)
        assert len(signals) == 1
        s = signals[0]
        assert s["band"] == "L1"
        assert s["constellation"] == "GPS"
        assert s["freq_hz"] == Constants.BANDS_CENTER_FREQUENCIES_Hz["L1"]
        assert s["eirp_dbw"] == Constants.GNSS_EIRP_DBW["GPS"]["L1"]

    def test_multiple_bands_across_constellations(self):
        signals = build_selected_signals(["L1", "E1", "B1I"], LOADED)
        by = _by_band(signals)
        assert set(by) == {"L1", "E1", "B1I"}
        assert by["L1"]["constellation"] == "GPS"
        assert by["E1"]["constellation"] == "GALILEO"
        assert by["B1I"]["constellation"] == "BEIDOU"

    def test_band_order_is_preserved(self):
        signals = build_selected_signals(["E1", "L1"], LOADED)
        assert [s["band"] for s in signals] == ["E1", "L1"]

    def test_empty_and_none_yield_empty_plan(self):
        assert build_selected_signals([], LOADED) == []
        assert build_selected_signals(None, LOADED) == []

    def test_constellation_match_is_case_insensitive(self):
        # The loaded key may carry the case from the TLE filename; owner lookup is canonical (upper),
        # but the returned constellation must be the actual loaded key for downstream dict access.
        signals = build_selected_signals(["L1"], ["gps"])
        assert len(signals) == 1
        assert signals[0]["constellation"] == "gps"


class TestWarnAndDrop:
    def test_unknown_band_is_dropped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            signals = build_selected_signals(["L33"], LOADED)
        assert signals == []
        assert "not a known GNSS band" in caplog.text
        assert "L33" in caplog.text

    def test_band_without_eirp_entry_is_dropped_with_warning(self, caplog):
        # NavIC_L1 is in the frequency table but has no EIRP/constellation entry.
        with caplog.at_level(logging.WARNING):
            signals = build_selected_signals(["NavIC_L1"], LOADED)
        assert signals == []
        assert "no transmit-EIRP entry" in caplog.text

    def test_band_for_unloaded_constellation_is_dropped_with_warning(self, caplog):
        # G1 belongs to GLONASS, which is not loaded here.
        with caplog.at_level(logging.WARNING):
            signals = build_selected_signals(["G1"], ["GPS", "GALILEO"])
        assert signals == []
        assert "GLONASS" in caplog.text
        assert "not among the loaded" in caplog.text

    def test_valid_bands_survive_alongside_invalid_ones(self, caplog):
        with caplog.at_level(logging.WARNING):
            signals = build_selected_signals(["L1", "L33", "G1", "E1"], LOADED)
        # L33 (unknown) and G1 (GLONASS not loaded) are dropped; L1 and E1 survive in order.
        assert [s["band"] for s in signals] == ["L1", "E1"]
        assert "L33" in caplog.text
        assert "G1" in caplog.text
