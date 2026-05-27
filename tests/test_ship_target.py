"""Smoke / structural tests for ShipTarget (flat plates + hull-water dihedrals)."""

import numpy as np
import pytest

from functions.link_budget import build_bistatic_geometry
from object_definitions.ShipTarget_def import ShipTarget
from object_definitions.Target_def import TargetRcs

WAVELENGTH_M = 0.19


def _zenith_geometry():
    """Both Tx and Rx straight above a surface point with normal +z."""
    surf = np.array([[0.0, 0.0, 0.0]])
    normal = np.array([[0.0, 0.0, 1.0]])
    tx = np.array([[0.0], [0.0], [2.0e7]])     # GPS altitude-ish
    rx = np.array([[0.0], [0.0], [7.0e5]])     # LEO altitude-ish
    return build_bistatic_geometry(tx, rx, surf, normal)


def _oblique_geometry(tx_off_x: float, rx_off_x: float):
    """Tx and Rx offset along +x to expose hull / dihedral aspects."""
    surf = np.array([[0.0, 0.0, 0.0]])
    normal = np.array([[0.0, 0.0, 1.0]])
    tx = np.array([[tx_off_x], [0.0], [2.0e7]])
    rx = np.array([[rx_off_x], [0.0], [7.0e5]])
    return build_bistatic_geometry(tx, rx, surf, normal)


def _default_heading_along_y() -> np.ndarray:
    """Bow points along +y. With mast=+z, starboard = bow x mast = +y x +z = +x."""
    return np.array([[0.0, 1.0, 0.0]])


def _ship() -> ShipTarget:
    return ShipTarget(length_m=100.0, beam_m=15.0, freeboard_m=5.0, heading_eci=_default_heading_along_y())


class TestShipTargetConstruction:
    def test_negative_length_raises(self):
        with pytest.raises(ValueError, match="length_m must be > 0"):
            ShipTarget(length_m=-1.0, beam_m=15.0, freeboard_m=5.0, heading_eci=_default_heading_along_y())

    def test_zero_beam_raises(self):
        with pytest.raises(ValueError, match="beam_m must be > 0"):
            ShipTarget(length_m=100.0, beam_m=0.0, freeboard_m=5.0, heading_eci=_default_heading_along_y())

    def test_bad_heading_shape_raises(self):
        with pytest.raises(ValueError, match="heading_eci must have shape"):
            ShipTarget(length_m=100.0, beam_m=15.0, freeboard_m=5.0, heading_eci=np.array([1.0, 0.0, 0.0]))

    def test_from_dict_classmethod(self):
        ship_dict = {"length_m": 80.0, "beam_m": 12.0, "freeboard_m": 4.0}
        target = ShipTarget.from_dict(ship_dict, heading_eci=_default_heading_along_y())
        assert target.length_m == 80.0
        assert target.beam_m == 12.0
        assert target.freeboard_m == 4.0


class TestShipTargetComputeRcs:
    def test_returns_target_rcs_instance(self):
        result = _ship().compute_rcs(_zenith_geometry(), WAVELENGTH_M)
        assert isinstance(result, TargetRcs)

    def test_info_breakdown_keys_present(self):
        result = _ship().compute_rcs(_zenith_geometry(), WAVELENGTH_M)
        for key in (
            "rcs_plates_total_m2",
            "rcs_dihedrals_total_m2",
            "rcs_dihedral_starboard_m2",
            "rcs_dihedral_port_m2",
            "rcs_per_plate_m2",
        ):
            assert key in result.info
        per_plate = result.info["rcs_per_plate_m2"]
        for name in ("deck", "starboard", "port", "bow", "stern"):
            assert name in per_plate

    def test_zenith_geometry_deck_dominates(self):
        # Tx and Rx directly above; the deck (normal +z) reflects strongly, the hull sides
        # and bow / stern are edge-on (zero), and the dihedrals are at their open-quadrant
        # boundary so they sit at the symmetry edge of the wedge check.
        result = _ship().compute_rcs(_zenith_geometry(), WAVELENGTH_M)
        per_plate = result.info["rcs_per_plate_m2"]
        deck = float(per_plate["deck"][0])
        starboard = float(per_plate["starboard"][0])
        port = float(per_plate["port"][0])
        assert deck > 0.0
        assert starboard == 0.0
        assert port == 0.0

    def test_starboard_offset_lights_up_starboard_dihedral(self):
        # Tx offset in +x (starboard side of the ship in our chosen heading). Starboard
        # dihedral should activate (wedge visibility holds); port one should not.
        geom = _oblique_geometry(tx_off_x=1.0e7, rx_off_x=1.0e7)
        result = _ship().compute_rcs(geom, WAVELENGTH_M)
        sigma_stbd = float(result.info["rcs_dihedral_starboard_m2"][0])
        sigma_port = float(result.info["rcs_dihedral_port_m2"][0])
        assert sigma_stbd > 0.0
        assert sigma_port == 0.0

    def test_port_offset_lights_up_port_dihedral(self):
        # Mirror: offset to -x (port). Port dihedral active, starboard quiet.
        geom = _oblique_geometry(tx_off_x=-1.0e7, rx_off_x=-1.0e7)
        result = _ship().compute_rcs(geom, WAVELENGTH_M)
        sigma_stbd = float(result.info["rcs_dihedral_starboard_m2"][0])
        sigma_port = float(result.info["rcs_dihedral_port_m2"][0])
        assert sigma_port > 0.0
        assert sigma_stbd == 0.0

    def test_total_equals_sum_of_components(self):
        result = _ship().compute_rcs(_oblique_geometry(tx_off_x=1.0e7, rx_off_x=1.0e7), WAVELENGTH_M)
        per_plate = result.info["rcs_per_plate_m2"]
        plate_sum = sum(np.nan_to_num(per_plate[name][0]) for name in ("deck", "starboard", "port", "bow", "stern"))
        dihedral_sum = (
            np.nan_to_num(result.info["rcs_dihedral_starboard_m2"][0])
            + np.nan_to_num(result.info["rcs_dihedral_port_m2"][0])
        )
        np.testing.assert_allclose(result.rcs_m2[0], plate_sum + dihedral_sum, rtol=1e-12)

    def test_dihedral_dominates_for_backscatter_at_l_band(self):
        # At a moderate starboard offset (backscatter-like geometry), the dihedral peak
        # 8 pi (L H)^2 / lambda^2 vastly exceeds the deck plate; the total should be dominated
        # by the starboard dihedral.
        result = _ship().compute_rcs(_oblique_geometry(tx_off_x=1.0e6, rx_off_x=1.0e6), WAVELENGTH_M)
        sigma_total = float(result.rcs_m2[0])
        sigma_stbd = float(result.info["rcs_dihedral_starboard_m2"][0])
        assert sigma_total > 0.0
        assert sigma_stbd > 0.5 * sigma_total

    def test_heading_length_mismatch_raises(self):
        ship = _ship()                                                   # heading has 1 sample
        # Build a 2-sample geometry that doesn't match.
        surf = np.tile(np.array([[0.0, 0.0, 0.0]]), (2, 1))
        normal = np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1))
        tx = np.tile(np.array([[0.0], [0.0], [2.0e7]]), (1, 2))
        rx = np.tile(np.array([[0.0], [0.0], [7.0e5]]), (1, 2))
        geom = build_bistatic_geometry(tx, rx, surf, normal)
        with pytest.raises(ValueError, match="heading_eci has 1 samples"):
            ship.compute_rcs(geom, WAVELENGTH_M)

    def test_invalid_heading_propagates_nan(self):
        # Heading parallel to surface normal -> bow re-orthogonalisation fails -> NaN RCS.
        bad_heading = np.array([[0.0, 0.0, 1.0]])
        ship = ShipTarget(length_m=100.0, beam_m=15.0, freeboard_m=5.0, heading_eci=bad_heading)
        result = ship.compute_rcs(_zenith_geometry(), WAVELENGTH_M)
        assert np.isnan(result.rcs_m2[0])
