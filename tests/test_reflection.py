"""Tests for Reflection class construction. Migrated from the legacy testObjReflection in
SRC/test_main.py and rewritten as pytest tests using pytest.raises for the invalid-model case."""

import pytest

from object_definitions.Reflection_def import Reflection


class TestReflectionConstruction:
    def test_no_earth_model_defaults_to_spherical(self):
        r = Reflection(tx_satID=1, rx_satID=2)
        assert isinstance(r, Reflection)
        assert r.earth_model == "spherical"

    def test_explicit_elliptical(self):
        r = Reflection(tx_satID=1, rx_satID=2, earth_model="elliptical")
        assert isinstance(r, Reflection)
        assert r.earth_model == "elliptical"

    def test_explicit_spherical(self):
        r = Reflection(tx_satID=1, rx_satID=2, earth_model="spherical")
        assert isinstance(r, Reflection)
        assert r.earth_model == "spherical"

    def test_invalid_earth_model_raises_value_error(self):
        # Use an obviously-invalid value (no autocorrect/spell-checker will "fix" this).
        with pytest.raises(ValueError, match="Invalid earth_model"):
            Reflection(tx_satID=1, rx_satID=2, earth_model="NOT_A_VALID_EARTH_MODEL")

    def test_missing_satid_raises_value_error(self):
        with pytest.raises(ValueError):
            Reflection(tx_satID=None, rx_satID=2)
        with pytest.raises(ValueError):
            Reflection(tx_satID=1, rx_satID=None)
