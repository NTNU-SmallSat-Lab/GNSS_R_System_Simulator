"""Tests for Reflection class construction. Migrated from the legacy testObjReflection in
SRC/test_main.py and rewritten as pytest tests using pytest.raises for the invalid-model case."""

import numpy as np
import pytest

from object_definitions.Reflection_def import Reflection


class TestReflectionConstruction:
    def test_no_earth_model_defaults_to_spherical(self):
        r = Reflection(tx_satID=1, rx_satID=2, max_central_angle_deg=80.0)
        assert isinstance(r, Reflection)
        assert r.earth_model == "spherical"

    def test_explicit_elliptical(self):
        r = Reflection(tx_satID=1, rx_satID=2, earth_model="elliptical", max_central_angle_deg=80.0)
        assert isinstance(r, Reflection)
        assert r.earth_model == "elliptical"

    def test_explicit_spherical(self):
        r = Reflection(tx_satID=1, rx_satID=2, earth_model="spherical", max_central_angle_deg=80.0)
        assert isinstance(r, Reflection)
        assert r.earth_model == "spherical"

    def test_invalid_earth_model_raises_value_error(self):
        # Use an obviously-invalid value (no autocorrect/spell-checker will "fix" this).
        with pytest.raises(ValueError, match="Invalid earth_model"):
            Reflection(tx_satID=1, rx_satID=2, earth_model="NOT_A_VALID_EARTH_MODEL", max_central_angle_deg=80.0)

    def test_missing_satid_raises_value_error(self):
        with pytest.raises(ValueError):
            Reflection(tx_satID=None, rx_satID=2, max_central_angle_deg=80.0)
        with pytest.raises(ValueError):
            Reflection(tx_satID=1, rx_satID=None, max_central_angle_deg=80.0)

    def test_missing_max_central_angle_raises_type_error(self):
        # max_central_angle_deg is a keyword-only required argument; omitting it must fail loudly
        # rather than silently using a default.
        with pytest.raises(TypeError, match="max_central_angle_deg"):
            Reflection(tx_satID=1, rx_satID=2)

    @pytest.mark.parametrize("angle", [0.0, -1.0, 181.0, np.nan])
    def test_nonphysical_max_central_angle_rejected(self, angle):
        with pytest.raises(ValueError, match="max_central_angle_deg"):
            Reflection(tx_satID=1, rx_satID=2, max_central_angle_deg=angle)
