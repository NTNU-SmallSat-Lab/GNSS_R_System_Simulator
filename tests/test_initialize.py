"""Unit tests for the configuration parser in SRC/__init__.py::initialize.

These tests exercise the parsing surface only — they never propagate orbits,
generate antenna patterns, or read TLE files. To stay on the parsing side of
that line, the configs deliberately omit [ANTENNA] and [ANTENNA PATTERN]
sections (which would otherwise call AntennaPatternGenerator).

TLE filenames in [GNSS CONSTELLATIONS] are never opened by the parser; it only
resolves them against Constants.tle_folderpath, so placeholders like
'GPS_dummy.tle' are valid here.
"""

import datetime as dt
import os
from textwrap import dedent

import pytest

# SRC is on sys.path via pyproject.toml's [tool.pytest.ini_options].pythonpath,
# matching what the entry-point scripts do.
from __init__ import initialize
from object_definitions.Constants_def import Constants


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

MINIMAL_SIMULATION_SETTINGS = dedent(
    """\
    [SIMULATION SETTINGS]
    startTime = 02.04.2025 12:00:00
    stepSize = 30
    """
)

MINIMAL_ORBITS = dedent(
    """\
    [ORBITS]
    orbit1 = [99.1, 0.00001, 0.0, 0.0, 0.0, 7300e3]
    """
)

MINIMAL_GNSS = dedent(
    """\
    [GNSS CONSTELLATIONS]
    gnss_tle_files = [GPS_dummy.tle]
    """
)


def write_config(tmp_path, body: str) -> str:
    """Write the given INI body to a temporary file and return its path."""
    config_path = tmp_path / "config.ini"
    config_path.write_text(body, encoding="utf-8")
    return str(config_path)


def minimal_config_body(
    *,
    extra_sim_settings: str = "",
    orbits_section: str = MINIMAL_ORBITS,
    gnss_section: str = MINIMAL_GNSS,
    extra_sections: str = "",
) -> str:
    """Compose a minimal-but-valid INI body that initialize() can parse."""
    return "\n".join(
        [
            MINIMAL_SIMULATION_SETTINGS + extra_sim_settings,
            orbits_section,
            gnss_section,
            extra_sections,
        ]
    )


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #

class TestInitializeHappyPath:
    def test_returns_configparser_and_dict(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        config, settings = initialize(path)
        assert config is not None
        assert isinstance(settings, dict)

    def test_start_time_parsed_to_utc_datetime(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        assert settings["start_time"] == dt.datetime(2025, 4, 2, 12, 0, 0, tzinfo=dt.timezone.utc)

    def test_step_size_parsed_as_int(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        assert settings["step_size"] == 30
        assert isinstance(settings["step_size"], int)

    def test_sim_duration_parsed_when_present(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body(extra_sim_settings="simulationDuration = 4\n"))
        _, settings = initialize(path)
        assert settings["sim_duration"] == 4

    def test_sim_duration_none_when_absent(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        assert settings["sim_duration"] is None

    def test_no_satellites_matches_orbit_count(self, tmp_path):
        orbits = dedent(
            """\
            [ORBITS]
            orbit1 = [99.1, 0.00001, 0.0, 0.0, 0.0, 7300e3]
            orbit2 = [99.1, 0.00001, 90.0, 0.0, 0.0, 7300e3]
            orbit3 = [99.1, 0.00001, 180.0, 0.0, 0.0, 7300e3]
            """
        )
        path = write_config(tmp_path, minimal_config_body(orbits_section=orbits))
        _, settings = initialize(path)
        assert settings["No_satellites"] == 3
        assert len(settings["orbits"]) == 3

    def test_orbit_start_time_appended_to_each_orbit(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        # Each orbit row is [inc, ecc, raan, argP, M, sma, start_time]
        assert len(settings["orbits"][0]) == 7
        assert settings["orbits"][0][-1] == settings["start_time"]

    def test_multiple_orbits_parsed_in_order(self, tmp_path):
        orbits = dedent(
            """\
            [ORBITS]
            orbit1 = [10.0, 0.0, 0.0, 0.0, 0.0, 7000e3]
            orbit2 = [20.0, 0.0, 0.0, 0.0, 0.0, 7000e3]
            orbit3 = [30.0, 0.0, 0.0, 0.0, 0.0, 7000e3]
            """
        )
        path = write_config(tmp_path, minimal_config_body(orbits_section=orbits))
        _, settings = initialize(path)
        # Inclination is the first element — verify ordering preserved.
        assert [orbit[0] for orbit in settings["orbits"]] == [10.0, 20.0, 30.0]

    def test_orbit_loop_stops_at_first_gap(self, tmp_path):
        # orbit3 is present but orbit2 is missing → the while-loop stops after orbit1.
        orbits = dedent(
            """\
            [ORBITS]
            orbit1 = [10.0, 0.0, 0.0, 0.0, 0.0, 7000e3]
            orbit3 = [30.0, 0.0, 0.0, 0.0, 0.0, 7000e3]
            """
        )
        path = write_config(tmp_path, minimal_config_body(orbits_section=orbits))
        _, settings = initialize(path)
        assert settings["No_satellites"] == 1
        assert settings["orbits"][0][0] == 10.0


# --------------------------------------------------------------------------- #
# [GNSS CONSTELLATIONS] parsing — locks the A4 tightening                     #
# --------------------------------------------------------------------------- #

class TestGnssConstellationParsing:
    def test_single_constellation(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        assert list(settings["GNSS_constellations"].keys()) == ["GPS"]

    def test_multiple_constellations(self, tmp_path):
        gnss = dedent(
            """\
            [GNSS CONSTELLATIONS]
            gnss_tle_files = [GPS.tle, GALILEO.tle, GLONASS.tle, BEIDOU.tle]
            """
        )
        path = write_config(tmp_path, minimal_config_body(gnss_section=gnss))
        _, settings = initialize(path)
        assert set(settings["GNSS_constellations"].keys()) == {"GPS", "GALILEO", "GLONASS", "BEIDOU"}

    def test_path_resolved_against_tle_folder(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        gps_entry = settings["GNSS_constellations"]["GPS"]
        expected_path = os.path.join(Constants.tle_folderpath, "GPS_dummy.tle")
        assert gps_entry["tle_file"] == expected_path

    def test_filename_match_is_case_insensitive(self, tmp_path):
        gnss = dedent(
            """\
            [GNSS CONSTELLATIONS]
            gnss_tle_files = [galileo_operational.tle]
            """
        )
        path = write_config(tmp_path, minimal_config_body(gnss_section=gnss))
        _, settings = initialize(path)
        # re.search with re.IGNORECASE matches lowercase 'galileo', and returns the
        # matched text — i.e. the case as it appears in the filename.
        assert "galileo" in settings["GNSS_constellations"]

    def test_brackets_and_whitespace_tolerated(self, tmp_path):
        gnss = dedent(
            """\
            [GNSS CONSTELLATIONS]
            gnss_tle_files = [  GPS.tle  ,  GALILEO.tle  ]
            """
        )
        path = write_config(tmp_path, minimal_config_body(gnss_section=gnss))
        _, settings = initialize(path)
        assert set(settings["GNSS_constellations"].keys()) == {"GPS", "GALILEO"}

    def test_unknown_filename_raises_value_error(self, tmp_path):
        gnss = dedent(
            """\
            [GNSS CONSTELLATIONS]
            gnss_tle_files = [UNKNOWN_SAT.tle]
            """
        )
        path = write_config(tmp_path, minimal_config_body(gnss_section=gnss))
        with pytest.raises(ValueError, match="Cannot identify GNSS constellation"):
            initialize(path)

    def test_empty_list_raises_value_error(self, tmp_path):
        gnss = dedent(
            """\
            [GNSS CONSTELLATIONS]
            gnss_tle_files = []
            """
        )
        path = write_config(tmp_path, minimal_config_body(gnss_section=gnss))
        with pytest.raises(ValueError, match="empty"):
            initialize(path)

    def test_mix_of_known_and_unknown_raises_value_error(self, tmp_path):
        gnss = dedent(
            """\
            [GNSS CONSTELLATIONS]
            gnss_tle_files = [GPS.tle, UNKNOWN_SAT.tle]
            """
        )
        path = write_config(tmp_path, minimal_config_body(gnss_section=gnss))
        with pytest.raises(ValueError, match=r"UNKNOWN_SAT\.tle"):
            initialize(path)


# --------------------------------------------------------------------------- #
# Missing sections / keys                                                     #
# --------------------------------------------------------------------------- #

class TestMissingSectionsAndKeys:
    def test_missing_start_time_key(self, tmp_path):
        body = dedent(
            """\
            [SIMULATION SETTINGS]
            stepSize = 30
            """
        ) + MINIMAL_ORBITS + MINIMAL_GNSS
        path = write_config(tmp_path, body)
        with pytest.raises(KeyError, match="startTime"):
            initialize(path)

    def test_missing_step_size_key(self, tmp_path):
        body = dedent(
            """\
            [SIMULATION SETTINGS]
            startTime = 02.04.2025 12:00:00
            """
        ) + MINIMAL_ORBITS + MINIMAL_GNSS
        path = write_config(tmp_path, body)
        with pytest.raises(KeyError, match="stepSize"):
            initialize(path)

    def test_missing_orbits_section(self, tmp_path):
        body = MINIMAL_SIMULATION_SETTINGS + MINIMAL_GNSS
        path = write_config(tmp_path, body)
        with pytest.raises(KeyError, match="ORBITS"):
            initialize(path)

    def test_missing_gnss_section(self, tmp_path):
        body = MINIMAL_SIMULATION_SETTINGS + MINIMAL_ORBITS
        path = write_config(tmp_path, body)
        with pytest.raises(KeyError, match="GNSS CONSTELLATIONS"):
            initialize(path)

    def test_missing_gnss_tle_files_key(self, tmp_path):
        body = MINIMAL_SIMULATION_SETTINGS + MINIMAL_ORBITS + dedent(
            """\
            [GNSS CONSTELLATIONS]
            placeholder = 1
            """
        )
        path = write_config(tmp_path, body)
        with pytest.raises(KeyError, match="gnss_tle_files"):
            initialize(path)


# --------------------------------------------------------------------------- #
# Optional sections (COVERAGE, MOVING TARGET, REFLECTION, LINK BUDGET)        #
# --------------------------------------------------------------------------- #

class TestOptionalSections:
    def test_no_coverage_section_yields_none(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        assert settings["grid_size_m"] is None
        assert settings["shapefile_path"] is None

    def test_coverage_section_parses(self, tmp_path):
        coverage = dedent(
            """\
            [COVERAGE]
            grid_size_meters = 30000
            shapefile_path = data/foo.shp
            """
        )
        path = write_config(tmp_path, minimal_config_body(extra_sections=coverage))
        _, settings = initialize(path)
        assert settings["grid_size_m"] == 30000
        assert settings["shapefile_path"] == "data/foo.shp"

    def test_no_moving_target_section_yields_none(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        assert settings["moving_target_speed"] is None
        assert settings["start_coord_ll"] is None
        assert settings["end_coord_ll"] is None

    def test_moving_target_section_parses(self, tmp_path):
        moving = dedent(
            """\
            [MOVING TARGET]
            speed = 12.5
            start_pos = (60.0, 5.0)
            end_pos = (62.0, 8.0)
            """
        )
        path = write_config(tmp_path, minimal_config_body(extra_sections=moving))
        _, settings = initialize(path)
        assert settings["moving_target_speed"] == 12.5
        assert settings["start_coord_ll"] == (60.0, 5.0)
        assert settings["end_coord_ll"] == (62.0, 8.0)

    def test_no_reflection_section_defaults_to_specular(self, tmp_path):
        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        assert settings["reflection_type"] == "specular"

    def test_reflection_type_backscatter_parsed(self, tmp_path):
        reflection = dedent(
            """\
            [REFLECTION]
            reflection_type = backscatter
            """
        )
        path = write_config(tmp_path, minimal_config_body(extra_sections=reflection))
        _, settings = initialize(path)
        assert settings["reflection_type"] == "backscatter"

    def test_reflection_type_strips_surrounding_quotes(self, tmp_path):
        # The parser strips both " and ' from around the value.
        reflection = dedent(
            """\
            [REFLECTION]
            reflection_type = "backscatter"
            """
        )
        path = write_config(tmp_path, minimal_config_body(extra_sections=reflection))
        _, settings = initialize(path)
        assert settings["reflection_type"] == "backscatter"

    def test_no_link_budget_section_uses_defaults(self, tmp_path):
        # The defaults come from BistaticLinkBudgetSettings (see functions/link_budget.py).
        from functions.link_budget import BistaticLinkBudgetSettings

        path = write_config(tmp_path, minimal_config_body())
        _, settings = initialize(path)
        defaults = BistaticLinkBudgetSettings()
        for field_name in settings["link_budget"]:
            assert settings["link_budget"][field_name] == getattr(defaults, field_name)

    def test_link_budget_override_applied(self, tmp_path):
        link_budget = dedent(
            """\
            [LINK BUDGET]
            tx_eirp_dbw = 35.0
            """
        )
        path = write_config(tmp_path, minimal_config_body(extra_sections=link_budget))
        _, settings = initialize(path)
        assert settings["link_budget"]["tx_eirp_dbw"] == 35.0

    def test_link_budget_invalid_float_raises_value_error(self, tmp_path):
        link_budget = dedent(
            """\
            [LINK BUDGET]
            tx_eirp_dbw = not_a_number
            """
        )
        path = write_config(tmp_path, minimal_config_body(extra_sections=link_budget))
        with pytest.raises(ValueError, match="tx_eirp_dbw"):
            initialize(path)
