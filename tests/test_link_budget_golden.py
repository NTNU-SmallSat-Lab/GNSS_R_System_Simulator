"""Golden-file regression for the bistatic link-budget pipeline.

Loads a frozen reference scenario from `tests/golden/reference_link_budget.npz` and
re-runs the same scenario through `compute_reflection_link_budget`. Any change in the
geometry, OceanTarget physics or link equation flips this red until the baseline is
intentionally regenerated via `tests/golden/regenerate_link_budget.py`.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

_GOLDEN_DIR = Path(__file__).parent / "golden"
_REGEN_SCRIPT = _GOLDEN_DIR / "regenerate_link_budget.py"
sys.path.insert(0, str(_GOLDEN_DIR))

from regenerate_link_budget import _fixture_scenario  # noqa: E402

from functions.link_budget import compute_reflection_link_budget  # noqa: E402

REFERENCE_NPZ = _GOLDEN_DIR / "reference_link_budget.npz"


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE_NPZ.exists():
        pytest.fail(
            f"Missing {REFERENCE_NPZ}. Regenerate with: python {_REGEN_SCRIPT}"
        )
    return np.load(REFERENCE_NPZ)


@pytest.fixture(scope="module")
def fresh_link_budget():
    return compute_reflection_link_budget(**_fixture_scenario())


class TestLinkBudgetGolden:
    def test_received_power_w_matches(self, reference, fresh_link_budget):
        np.testing.assert_allclose(
            fresh_link_budget["received_power_w"],
            reference["received_power_w"],
            rtol=1e-12,
        )

    def test_received_power_dbm_matches(self, reference, fresh_link_budget):
        np.testing.assert_allclose(
            fresh_link_budget["received_power_dbm"],
            reference["received_power_dbm"],
            rtol=1e-12,
        )

    def test_rcs_matches(self, reference, fresh_link_budget):
        np.testing.assert_allclose(
            fresh_link_budget["rcs_m2"],
            reference["rcs_m2"],
            rtol=1e-12,
        )

    def test_fspl_sum_matches(self, reference, fresh_link_budget):
        np.testing.assert_allclose(
            fresh_link_budget["fspl_sum_db"],
            reference["fspl_sum_db"],
            rtol=1e-12,
        )

    def test_geometry_angles_match(self, reference, fresh_link_budget):
        np.testing.assert_allclose(fresh_link_budget["incidence_deg"], reference["incidence_deg"], atol=1e-12)
        np.testing.assert_allclose(fresh_link_budget["scattering_deg"], reference["scattering_deg"], atol=1e-12)
        np.testing.assert_allclose(fresh_link_budget["bistatic_deg"], reference["bistatic_deg"], atol=1e-12)

    def test_ranges_match(self, reference, fresh_link_budget):
        np.testing.assert_allclose(
            fresh_link_budget["tx_to_surface_range_m"],
            reference["tx_to_surface_range_m"],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            fresh_link_budget["surface_to_rx_range_m"],
            reference["surface_to_rx_range_m"],
            rtol=1e-12,
        )
