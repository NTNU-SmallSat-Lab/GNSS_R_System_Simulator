"""Unit tests for Constants.GNSS_EIRP_DBW and the get_tx_eirp_dbw lookup helper."""

import pytest

from object_definitions.Constants_def import Constants


L1_FREQ_HZ = 1.57542e9
L5_FREQ_HZ = 1.17645e9
GLONASS_G1_FREQ_HZ = 1.602e9
QZSS_L1_FREQ_HZ = 1.57542e9


class TestGnssEirpTable:
    def test_gps_entries(self):
        assert Constants.GNSS_EIRP_DBW["GPS"]["L1"] == 27.0
        assert Constants.GNSS_EIRP_DBW["GPS"]["L2"] == 23.0
        assert Constants.GNSS_EIRP_DBW["GPS"]["L5"] == 27.0

    def test_galileo_entries(self):
        assert Constants.GNSS_EIRP_DBW["GALILEO"]["E1"] == 32.0
        assert Constants.GNSS_EIRP_DBW["GALILEO"]["E6"] == 33.49

    def test_beidou_entries(self):
        assert Constants.GNSS_EIRP_DBW["BEIDOU"]["B1C"] == 28.0
        assert Constants.GNSS_EIRP_DBW["BEIDOU"]["B3I"] == 35.0

    def test_glonass_entries(self):
        assert Constants.GNSS_EIRP_DBW["GLONASS"]["G1"] == 28.0
        assert Constants.GNSS_EIRP_DBW["GLONASS"]["G3"] == 26.0

    def test_qzss_entries(self):
        assert Constants.GNSS_EIRP_DBW["QZSS"]["QZSS_L1"] == 27.0
        assert Constants.GNSS_EIRP_DBW["QZSS"]["QZSS_L5"] == 29.1
        assert Constants.GNSS_EIRP_DBW["QZSS"]["QZSS_L6"] == 35.0

    def test_navic_intentionally_absent(self):
        assert "NAVIC" not in Constants.GNSS_EIRP_DBW

    def test_all_band_keys_present_in_band_freq_table(self):
        # Every (constellation, band) entry must map to a known carrier frequency,
        # otherwise the lookup helper can never find it.
        for constellation, table in Constants.GNSS_EIRP_DBW.items():
            for band in table:
                assert band in Constants.BANDS_CENTER_FREQUENCIES_Hz, (
                    f"{constellation}.{band} has no entry in BANDS_CENTER_FREQUENCIES_Hz"
                )


class TestGetTxEirpDbw:
    def test_gps_l1(self):
        assert Constants.get_tx_eirp_dbw("GPS", L1_FREQ_HZ) == 27.0

    def test_galileo_e1(self):
        # E1 also sits at 1575.42 MHz; the lookup returns Galileo's value, not GPS's.
        assert Constants.get_tx_eirp_dbw("GALILEO", L1_FREQ_HZ) == 32.0

    def test_beidou_b1c(self):
        assert Constants.get_tx_eirp_dbw("BEIDOU", L1_FREQ_HZ) == 28.0

    def test_glonass_g1(self):
        assert Constants.get_tx_eirp_dbw("GLONASS", GLONASS_G1_FREQ_HZ) == 28.0

    def test_qzss_l1(self):
        assert Constants.get_tx_eirp_dbw("QZSS", QZSS_L1_FREQ_HZ) == 27.0

    def test_constellation_is_case_insensitive(self):
        assert Constants.get_tx_eirp_dbw("gps", L1_FREQ_HZ) == 27.0
        assert Constants.get_tx_eirp_dbw("Galileo", L1_FREQ_HZ) == 32.0

    def test_unknown_constellation_raises(self):
        with pytest.raises(KeyError, match="No EIRP table for constellation"):
            Constants.get_tx_eirp_dbw("IRIDIUM", L1_FREQ_HZ)

    def test_glonass_at_l1_frequency_raises(self):
        # GLONASS G1 is at 1602 MHz, not L1's 1575.42 MHz; no entry should match.
        with pytest.raises(KeyError, match="No EIRP entry for constellation 'GLONASS'"):
            Constants.get_tx_eirp_dbw("GLONASS", L1_FREQ_HZ)

    def test_gps_at_arbitrary_frequency_raises(self):
        with pytest.raises(KeyError, match="No EIRP entry for constellation 'GPS'"):
            Constants.get_tx_eirp_dbw("GPS", 1.4e9)

    def test_frequency_within_tolerance_matches(self):
        # 1 MHz default tolerance; a value 500 kHz off should still map to L1.
        assert Constants.get_tx_eirp_dbw("GPS", L1_FREQ_HZ + 5e5) == 27.0

    def test_frequency_outside_tolerance_misses(self):
        # 5 MHz off is outside the 1 MHz default tolerance.
        with pytest.raises(KeyError, match="No EIRP entry"):
            Constants.get_tx_eirp_dbw("GPS", L1_FREQ_HZ + 5e6)

    def test_explicit_freq_tolerance_widens_match(self):
        # With a 10 MHz tolerance, 5 MHz off still matches.
        assert Constants.get_tx_eirp_dbw("GPS", L1_FREQ_HZ + 5e6, freq_tolerance_hz=1.0e7) == 27.0

    def test_returns_float(self):
        result = Constants.get_tx_eirp_dbw("GALILEO", 1.27875e9)  # E6
        assert isinstance(result, float)
        assert result == 33.49
