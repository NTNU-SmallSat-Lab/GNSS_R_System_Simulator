"""Unit tests for Target ABC and OceanTarget."""

import dataclasses

import numpy as np
import pytest

from functions.link_budget import (
    BistaticLinkBudgetSettings,
    build_bistatic_geometry,
    compute_reflection_link_budget,
)
from object_definitions.OceanTarget_def import OceanTarget
from object_definitions.Target_def import Target, TargetRcs


def _trivial_geometry():
    """Backscatter geometry: surface at origin (+z normal); Tx 10 m up, Rx 5 m up."""
    surf = np.array([[0.0, 0.0, 0.0]])
    norm = np.array([[0.0, 0.0, 1.0]])
    tx = np.array([[0.0], [0.0], [10.0]])
    rx = np.array([[0.0], [0.0], [5.0]])
    return build_bistatic_geometry(tx, rx, surf, norm)


class TestTargetABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Target()

    def test_subclass_without_compute_rcs_cannot_be_instantiated(self):
        class IncompleteTarget(Target):
            pass

        with pytest.raises(TypeError):
            IncompleteTarget()

    def test_concrete_subclass_works(self):
        class ConstantRcs(Target):
            def compute_rcs(self, geometry, wavelength_m):
                n = geometry.range_tx_m.shape[0]
                return TargetRcs(rcs_m2=np.full(n, 1.0), info={"name": "constant"})

        geom = _trivial_geometry()
        target = ConstantRcs()
        result = target.compute_rcs(geom, wavelength_m=0.19)
        assert isinstance(result, TargetRcs)
        np.testing.assert_allclose(result.rcs_m2, [1.0])
        assert result.info == {"name": "constant"}


class TestOceanTargetConstruction:
    def test_init_stores_parameters_as_floats(self):
        target = OceanTarget(sigma0_db=-20, slope_db_per_deg=0.25, sigma_floor_db=-45, fresnel_scale=1)
        assert target.sigma0_db == -20.0
        assert target.slope_db_per_deg == 0.25
        assert target.sigma_floor_db == -45.0
        assert target.fresnel_scale == 1.0
        assert isinstance(target.sigma0_db, float)

    def test_from_settings_roundtrips_ocean_params(self):
        settings = BistaticLinkBudgetSettings()
        target = OceanTarget.from_settings(settings)
        assert target.sigma0_db == settings.ocean_sigma0_db
        assert target.slope_db_per_deg == settings.ocean_sigma_slope_db_per_deg
        assert target.sigma_floor_db == settings.ocean_sigma_floor_db
        assert target.fresnel_scale == settings.fresnel_rcs_scale


class TestOceanTargetComputeRcs:
    def test_returns_target_rcs_with_expected_shape(self):
        geom = _trivial_geometry()
        target = OceanTarget.from_settings(BistaticLinkBudgetSettings())

        result = target.compute_rcs(geom, wavelength_m=0.19)

        assert isinstance(result, TargetRcs)
        assert result.rcs_m2.shape == (1,)
        assert np.isfinite(result.rcs_m2).all()

    def test_info_contains_sigma0_db_and_fresnel_area(self):
        geom = _trivial_geometry()
        target = OceanTarget.from_settings(BistaticLinkBudgetSettings())

        result = target.compute_rcs(geom, wavelength_m=0.19)

        assert set(result.info.keys()) == {"sigma0_db", "fresnel_area_m2"}
        assert result.info["sigma0_db"].shape == (1,)
        assert result.info["fresnel_area_m2"].shape == (1,)

    def test_matches_hand_computed_value_at_zero_incidence(self):
        # incidence=0, range_tx=10, range_rx=5, wavelength=0.19, default sigma0_db=-20
        # => sigma0_lin = 0.01, fresnel_area = pi * 0.19 * 10 * 5 / 15 = pi * 0.6333..., rcs = 0.01 * area
        geom = _trivial_geometry()
        target = OceanTarget(sigma0_db=-20.0, slope_db_per_deg=0.25, sigma_floor_db=-45.0, fresnel_scale=1.0)

        result = target.compute_rcs(geom, wavelength_m=0.19)

        expected_area = np.pi * 0.19 * 10.0 * 5.0 / (10.0 + 5.0)
        expected_rcs = 0.01 * expected_area * 1.0
        np.testing.assert_allclose(result.rcs_m2, [expected_rcs], rtol=1e-12)
        np.testing.assert_allclose(result.info["sigma0_db"], [-20.0], atol=1e-12)
        np.testing.assert_allclose(result.info["fresnel_area_m2"], [expected_area], rtol=1e-12)

    def test_sigma0_floor_applied_at_high_incidence(self):
        # Override incidence to 89 deg; with slope 10 dB/deg, raw sigma0 = -20 - 890 = -910 dB,
        # which gets clamped to the floor (-30 dB).
        geom = _trivial_geometry()
        geom = dataclasses.replace(geom, incidence_angle_deg=np.array([89.0]))
        target = OceanTarget(sigma0_db=-20.0, slope_db_per_deg=10.0, sigma_floor_db=-30.0, fresnel_scale=1.0)

        result = target.compute_rcs(geom, wavelength_m=0.19)

        np.testing.assert_allclose(result.info["sigma0_db"], [-30.0])

    def test_nan_geometry_propagates_to_nan_rcs(self):
        surf = np.array([[0.0, 0.0, 0.0]])
        norm = np.array([[0.0, 0.0, 1.0]])
        tx = np.array([[np.nan], [0.0], [10.0]])
        rx = np.array([[0.0], [0.0], [5.0]])
        geom = build_bistatic_geometry(tx, rx, surf, norm)
        target = OceanTarget.from_settings(BistaticLinkBudgetSettings())

        result = target.compute_rcs(geom, wavelength_m=0.19)

        assert np.isnan(result.rcs_m2[0])
        assert np.isnan(result.info["sigma0_db"][0])
        assert np.isnan(result.info["fresnel_area_m2"][0])

    def test_fresnel_scale_multiplies_rcs(self):
        geom = _trivial_geometry()
        target_unscaled = OceanTarget(sigma0_db=-20.0, slope_db_per_deg=0.0, sigma_floor_db=-45.0, fresnel_scale=1.0)
        target_scaled = OceanTarget(sigma0_db=-20.0, slope_db_per_deg=0.0, sigma_floor_db=-45.0, fresnel_scale=3.5)

        unscaled = target_unscaled.compute_rcs(geom, wavelength_m=0.19).rcs_m2
        scaled = target_scaled.compute_rcs(geom, wavelength_m=0.19).rcs_m2

        np.testing.assert_allclose(scaled, 3.5 * unscaled, rtol=1e-12)


class TestComputeReflectionLinkBudgetIntegration:
    def _canned_inputs(self):
        tx = np.array([[8.0e6], [0.0], [0.0]])
        rx = np.array([[7.1e6], [1.0e5], [0.0]])
        rx_vel = np.array([[0.0], [7.5e3], [0.0]])
        refl = np.array([[6.371e6, 0.0, 0.0]])
        incidence_deg = np.array([5.0])
        return tx, rx, rx_vel, refl, incidence_deg

    def test_default_target_matches_explicit_ocean_target(self):
        tx, rx, rx_vel, refl, incidence_deg = self._canned_inputs()

        lb_default = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=incidence_deg,
            antenna=None,
            frequency_hz=1.57542e9,
            settings=None,
        )
        ocean = OceanTarget.from_settings(BistaticLinkBudgetSettings())
        lb_explicit = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=incidence_deg,
            antenna=None,
            frequency_hz=1.57542e9,
            settings=None,
            target=ocean,
        )
        np.testing.assert_allclose(lb_default["received_power_w"], lb_explicit["received_power_w"], rtol=1e-12)
        np.testing.assert_allclose(lb_default["rcs_m2"], lb_explicit["rcs_m2"], rtol=1e-12)

    def test_output_dict_carries_geometry_and_target_info(self):
        tx, rx, rx_vel, refl, incidence_deg = self._canned_inputs()

        lb = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=incidence_deg,
            antenna=None,
            frequency_hz=1.57542e9,
            settings=None,
        )
        for key in (
            "incidence_deg",
            "scattering_deg",
            "bistatic_deg",
            "rcs_m2",
            "target_info",
            "received_power_w",
            "received_power_dbw",
            "received_power_dbm",
        ):
            assert key in lb, f"missing key {key!r}"
        assert "sigma0_db" in lb["target_info"]
        assert "fresnel_area_m2" in lb["target_info"]

    def test_caller_supplied_incidence_overrides_geometry_default(self):
        # Verify our spherical-normal default + replace() preserves caller's incidence.
        tx, rx, rx_vel, refl, _ = self._canned_inputs()
        chosen_incidence = np.array([42.0])

        lb = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=chosen_incidence,
            antenna=None,
            frequency_hz=1.57542e9,
            settings=None,
        )
        np.testing.assert_allclose(lb["incidence_deg"], chosen_incidence)

    def test_custom_target_replaces_ocean_rcs(self):
        # Plug a non-ocean target through to confirm the abstraction works.
        class FixedRcs(Target):
            def compute_rcs(self, geometry, wavelength_m):
                n = geometry.range_tx_m.shape[0]
                return TargetRcs(rcs_m2=np.full(n, 100.0), info={"kind": "fixed"})

        tx, rx, rx_vel, refl, incidence_deg = self._canned_inputs()
        lb_fixed = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=incidence_deg,
            antenna=None,
            frequency_hz=1.57542e9,
            settings=None,
            target=FixedRcs(),
        )
        np.testing.assert_allclose(lb_fixed["rcs_m2"], [100.0])
        assert lb_fixed["target_info"] == {"kind": "fixed"}

    def test_received_power_scales_linearly_with_rcs(self):
        # Doubling RCS doubles Pr (the link equation is linear in sigma).
        tx, rx, rx_vel, refl, incidence_deg = self._canned_inputs()

        class FixedRcs(Target):
            def __init__(self, rcs):
                self.rcs = rcs

            def compute_rcs(self, geometry, wavelength_m):
                n = geometry.range_tx_m.shape[0]
                return TargetRcs(rcs_m2=np.full(n, self.rcs), info={})

        lb_a = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=incidence_deg,
            antenna=None,
            frequency_hz=1.57542e9,
            settings=None,
            target=FixedRcs(50.0),
        )
        lb_b = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=incidence_deg,
            antenna=None,
            frequency_hz=1.57542e9,
            settings=None,
            target=FixedRcs(100.0),
        )
        np.testing.assert_allclose(lb_b["received_power_w"] / lb_a["received_power_w"], [2.0], rtol=1e-12)
