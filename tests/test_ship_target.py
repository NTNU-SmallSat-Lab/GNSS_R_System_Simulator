"""Smoke / structural tests for ShipTarget (flat plates + hull-water dihedrals)."""

import numpy as np
import pytest

from functions.link_budget import build_bistatic_geometry
from functions.ocean_physics import circular_pol_reflectivities, seawater_permittivity
from functions.ship_physics import dihedral_rcs
from object_definitions.Constants_def import Constants
from object_definitions.OceanTarget_def import OceanTarget
from object_definitions.ShipTarget_def import ShipTarget
from object_definitions.Target_def import TargetRcs

WAVELENGTH_M = 0.19
CALM_SEA = OceanTarget(wave_height_m=0.0, wind_speed_m_s=5.0, wind_direction_deg=0.0, sea_temperature_c=15.0, salinity_psu=35.0)


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
    return ShipTarget(length_m=100.0, beam_m=15.0, freeboard_m=5.0, heading_eci=_default_heading_along_y(), ocean=CALM_SEA)


class TestShipTargetConstruction:
    def test_negative_length_raises(self):
        with pytest.raises(ValueError, match="length_m must be > 0"):
            ShipTarget(length_m=-1.0, beam_m=15.0, freeboard_m=5.0, heading_eci=_default_heading_along_y(), ocean=CALM_SEA)

    def test_zero_beam_raises(self):
        with pytest.raises(ValueError, match="beam_m must be > 0"):
            ShipTarget(length_m=100.0, beam_m=0.0, freeboard_m=5.0, heading_eci=_default_heading_along_y(), ocean=CALM_SEA)

    def test_bad_heading_shape_raises(self):
        with pytest.raises(ValueError, match="heading_eci must have shape"):
            ShipTarget(length_m=100.0, beam_m=15.0, freeboard_m=5.0, heading_eci=np.array([1.0, 0.0, 0.0]), ocean=CALM_SEA)

    def test_from_dict_classmethod(self):
        ship_dict = {"length_m": 80.0, "beam_m": 12.0, "freeboard_m": 4.0}
        target = ShipTarget.from_dict(ship_dict, heading_eci=_default_heading_along_y(), ocean=CALM_SEA)
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
            "rcs_dihedrals_diffuse_m2",
            "rcs_dihedral_starboard_m2",
            "rcs_dihedral_port_m2",
            "rcs_dihedral_bow_m2",
            "rcs_dihedral_stern_m2",
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
        dihedral_sum = sum(np.nan_to_num(result.info[f"rcs_dihedral_{name}_m2"][0]) for name in ("starboard", "port", "bow", "stern"))
        np.testing.assert_allclose(result.rcs_m2[0], plate_sum + dihedral_sum, rtol=1e-12)

    def test_bow_corner_is_a_side_corner_with_the_beam_as_edge(self):
        # Tx and Rx ahead and to starboard. Swapping length and beam and turning the heading to -x makes the
        # starboard face point along +y, where the first ship's bow points: both corners must be identical.
        surf, up = np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]])
        geom = build_bistatic_geometry(np.array([[0.3e7], [1.2e7], [2.0e7]]), np.array([[1.0e5], [4.0e5], [7.0e5]]), surf, up)
        info = _ship().compute_rcs(geom, WAVELENGTH_M).info
        turned = ShipTarget(length_m=15.0, beam_m=100.0, freeboard_m=5.0, heading_eci=np.array([[-1.0, 0.0, 0.0]]), ocean=CALM_SEA)
        np.testing.assert_allclose(info["rcs_dihedral_bow_m2"][0], turned.compute_rcs(geom, WAVELENGTH_M).info["rcs_dihedral_starboard_m2"][0], rtol=1e-12)
        assert float(info["rcs_dihedral_bow_m2"][0]) > 0.0
        assert float(info["rcs_dihedral_stern_m2"][0]) == 0.0

    def test_dihedral_dominates_for_backscatter_at_l_band(self):
        # Tx and Rx on one line through the ship on the starboard side (backscatter geometry): on a calm sea the
        # dihedral peak 16 pi (L H sin(theta))^2 / lambda^2 times |R_RR|^2 still exceeds the plates; the total should
        # be dominated by the starboard dihedral.
        rx_off_x = 5.0e5
        geom = _oblique_geometry(tx_off_x=rx_off_x * 2.0e7 / 7.0e5, rx_off_x=rx_off_x)
        result = _ship().compute_rcs(geom, WAVELENGTH_M)
        sigma_total = float(result.rcs_m2[0])
        sigma_stbd = float(result.info["rcs_dihedral_starboard_m2"][0])
        assert sigma_total > 0.0
        assert sigma_stbd > 0.5 * sigma_total

    def test_dihedral_sea_leg_is_seawater_same_sense_with_roughness_loss(self):
        # The sea leg scales the metal corner by the seawater same-sense reflectivity |R_RR|^2 at the incidence on the
        # sea; a rough sea removes the coherent corner by exp(-kappa), kappa = (pi H_s cos(theta) / lambda)^2.
        rx_off_x = 5.0e5
        geom = _oblique_geometry(tx_off_x=rx_off_x * 2.0e7 / 7.0e5, rx_off_x=rx_off_x)
        cos_sea = float(-geom.n_tx2s[0, 2])                                           # surface normal +z
        bow, stbd, up = np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
        metal = dihedral_rcs(bow, stbd, up, 100.0, 5.0, geom.n_tx2s, geom.n_s2rx, WAVELENGTH_M)[0]
        eps_r = seawater_permittivity(Constants.SPEED_OF_LIGHT / WAVELENGTH_M, 15.0, 35.0)
        _, R_RR_sq = circular_pol_reflectivities(np.arccos(cos_sea), eps_r)
        coherent = lambda ship: float(ship.compute_rcs(geom, WAVELENGTH_M).info["rcs_dihedral_starboard_m2"][0]   # noqa: E731
                                      - ship._diffuse_corner_rcs(geom, stbd, 100.0, WAVELENGTH_M)[0])   # starboard corner minus its own diffuse part
        calm = coherent(_ship())
        np.testing.assert_allclose(calm, metal * R_RR_sq, rtol=1e-9)

        wave_height_m = 0.05
        rough_sea = OceanTarget(wave_height_m=wave_height_m, wind_speed_m_s=5.0, wind_direction_deg=0.0, sea_temperature_c=15.0, salinity_psu=35.0)
        ship = ShipTarget(length_m=100.0, beam_m=15.0, freeboard_m=5.0, heading_eci=_default_heading_along_y(), ocean=rough_sea)
        rough = coherent(ship)
        np.testing.assert_allclose(rough / calm, np.exp(-((np.pi * wave_height_m * cos_sea / WAVELENGTH_M) ** 2)), rtol=1e-9)

    def test_diffuse_corner_is_same_sense_sea_scatter_mirrored_by_the_hull(self):
        # Each path order sees the sea strip the hull mirrors (area L H tan(theta) cos(phi)) with the ocean's same-sense
        # NRCS for the hull image of Rx (sea first) or of Tx (hull first); the images are built here from mirrored positions.
        surf, up, hull = np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([1.0, 0.0, 0.0])   # starboard = +x
        tx, rx = np.array([1.2e7, 0.4e7, 2.0e7]), np.array([3.0e5, -2.0e5, 7.0e5])                  # both on the starboard side
        mirror = lambda p: p - 2.0 * (p @ hull) * hull  # noqa: E731
        expected = 0.0
        for tx_path, rx_path, to_v in ((tx, mirror(rx), rx), (mirror(tx), rx, tx)):
            info = CALM_SEA.compute_rcs(build_bistatic_geometry(tx_path[:, None], rx_path[:, None], surf, up), WAVELENGTH_M).info
            nrcs_rr = info["nrcs"] * info["fresnel_reflectivity_RR_sq"] / info["fresnel_reflectivity_RL_sq"] + info["nrcs_bragg_RR"]
            expected += float(nrcs_rr[0]) * 100.0 * 5.0 * (to_v @ hull) / to_v[2]
        result = _ship().compute_rcs(build_bistatic_geometry(tx[:, None], rx[:, None], surf, up), WAVELENGTH_M)
        np.testing.assert_allclose(result.info["rcs_dihedrals_diffuse_m2"][0], expected, rtol=1e-9)
        assert expected > 0.0

    def test_diffuse_corner_needs_tx_and_rx_on_the_hull_side(self):
        # Tx on the starboard side, Rx on the port side: neither hull side faces both (open corner quadrant).
        result = _ship().compute_rcs(_oblique_geometry(tx_off_x=1.0e7, rx_off_x=-1.0e5), WAVELENGTH_M)
        assert float(result.info["rcs_dihedrals_diffuse_m2"][0]) == 0.0

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
        ship = ShipTarget(length_m=100.0, beam_m=15.0, freeboard_m=5.0, heading_eci=bad_heading, ocean=CALM_SEA)
        result = ship.compute_rcs(_zenith_geometry(), WAVELENGTH_M)
        assert np.isnan(result.rcs_m2[0])
