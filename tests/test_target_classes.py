"""Unit tests for Target ABC and OceanTarget (geometric-optics Cox-Munk model)."""

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


def _default_ocean() -> OceanTarget:
    return OceanTarget(
        wave_height_m=1.5,
        wind_speed_m_s=7.0,
        wind_direction_deg=270.0,
        sea_temperature_c=15.0,
        salinity_psu=35.0,
    )


def _zenith_geometry():
    """Tx and Rx both straight above the surface point at (R, 0, 0). Pure backscatter, theta=0."""
    R = 6.371e6
    surf = np.array([[R, 0.0, 0.0]])
    normal = np.array([[1.0, 0.0, 0.0]])
    tx = np.array([[3 * R], [0.0], [0.0]])
    rx = np.array([[2 * R], [0.0], [0.0]])
    return build_bistatic_geometry(tx, rx, surf, normal)


def _mirror_geometry(theta_deg: float):
    """Symmetric specular geometry at incidence theta_deg, with surface normal +z."""
    surf = np.array([[0.0, 0.0, 0.0]])
    normal = np.array([[0.0, 0.0, 1.0]])
    th = np.radians(theta_deg)
    L = 1.0e6
    tx = np.array([[-L * np.sin(th)], [0.0], [L * np.cos(th)]])
    rx = np.array([[L * np.sin(th)], [0.0], [L * np.cos(th)]])
    geom = build_bistatic_geometry(tx, rx, surf, normal)
    return dataclasses.replace(geom, incidence_angle_deg=np.array([float(theta_deg)]))


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
            def compute_rcs(self, geometry, wavelength_m, illuminated_area_m2=None):
                n = geometry.range_tx_m.shape[0]
                return TargetRcs(rcs_m2=np.full(n, 1.0), info={"name": "constant"})

        geom = _zenith_geometry()
        result = ConstantRcs().compute_rcs(geom, wavelength_m=0.19)
        np.testing.assert_allclose(result.rcs_m2, [1.0])
        assert result.info == {"name": "constant"}


class TestOceanTargetConstruction:
    def test_init_stores_parameters_as_floats(self):
        target = OceanTarget(
            wave_height_m=2,
            wind_speed_m_s=8,
            wind_direction_deg=180,
            sea_temperature_c=12,
            salinity_psu=34,
        )
        assert target.wave_height_m == 2.0
        assert target.wind_speed_m_s == 8.0
        assert target.wind_direction_deg == 180.0
        assert target.sea_temperature_c == 12.0
        assert target.salinity_psu == 34.0
        assert isinstance(target.wave_height_m, float)

    def test_from_settings_roundtrips_ocean_params(self):
        settings = BistaticLinkBudgetSettings()
        target = OceanTarget.from_settings(settings)
        assert target.wave_height_m == settings.ocean_wave_height_m
        assert target.wind_speed_m_s == settings.ocean_wind_speed_m_s
        assert target.wind_direction_deg == settings.ocean_wind_direction_deg
        assert target.sea_temperature_c == settings.ocean_sea_temperature_c
        assert target.salinity_psu == settings.ocean_salinity_psu
        assert target.inverse_wave_age == settings.ocean_inverse_wave_age

    def test_negative_wave_height_raises(self):
        with pytest.raises(ValueError, match="wave_height_m"):
            OceanTarget(-0.1, 7.0, 0.0, 15.0, 35.0)

    def test_zero_or_negative_wind_speed_raises(self):
        with pytest.raises(ValueError, match="wind_speed_m_s"):
            OceanTarget(1.0, 0.0, 0.0, 15.0, 35.0)
        with pytest.raises(ValueError, match="wind_speed_m_s"):
            OceanTarget(1.0, -1.0, 0.0, 15.0, 35.0)

    def test_implausible_temperature_raises(self):
        with pytest.raises(ValueError, match="sea_temperature_c"):
            OceanTarget(1.0, 7.0, 0.0, -200.0, 35.0)

    def test_implausible_salinity_raises(self):
        with pytest.raises(ValueError, match="salinity_psu"):
            OceanTarget(1.0, 7.0, 0.0, 15.0, -1.0)
        with pytest.raises(ValueError, match="salinity_psu"):
            OceanTarget(1.0, 7.0, 0.0, 15.0, 100.0)

    def test_implausible_inverse_wave_age_raises(self):
        with pytest.raises(ValueError, match="inverse_wave_age"):
            OceanTarget(1.0, 7.0, 0.0, 15.0, 35.0, inverse_wave_age=0.5)
        with pytest.raises(ValueError, match="inverse_wave_age"):
            OceanTarget(1.0, 7.0, 0.0, 15.0, 35.0, inverse_wave_age=6.0)


class TestOceanTargetComputeRcs:
    def test_returns_target_rcs_with_expected_shape(self):
        geom = _zenith_geometry()

        result = _default_ocean().compute_rcs(geom, wavelength_m=0.19)

        assert isinstance(result, TargetRcs)
        assert result.rcs_m2.shape == (1,)
        assert np.isfinite(result.rcs_m2).all()
        assert result.rcs_m2[0] > 0.0

    def test_info_dict_carries_expected_diagnostics(self):
        geom = _zenith_geometry()

        result = _default_ocean().compute_rcs(geom, wavelength_m=0.19)

        for key in (
            "fresnel_reflectivity_RL_sq",
            "fresnel_reflectivity_RR_sq",
            "slope_pdf",
            "slope_upwind",
            "slope_crosswind",
            "fresnel_area_m2",
            "rayleigh_roughness_kappa",
            "nrcs",
        ):
            assert key in result.info
            assert result.info[key].shape == (1,)

    def test_mirror_geometry_gives_zero_slope(self):
        # At specular mirror geometry the required facet slope is zero -> Cox-Munk peaks.
        geom = _mirror_geometry(theta_deg=20.0)
        result = _default_ocean().compute_rcs(geom, wavelength_m=0.19)
        np.testing.assert_allclose(result.info["slope_upwind"], [0.0], atol=1e-12)
        np.testing.assert_allclose(result.info["slope_crosswind"], [0.0], atol=1e-12)

    def test_backscatter_geometry_gives_zero_slope(self):
        geom = _zenith_geometry()
        result = _default_ocean().compute_rcs(geom, wavelength_m=0.19)
        np.testing.assert_allclose(result.info["slope_upwind"], [0.0], atol=1e-12)
        np.testing.assert_allclose(result.info["slope_crosswind"], [0.0], atol=1e-12)

    def test_coherent_dominates_for_calm_low_freq_smooth_surface(self):
        # Tiny wave height (sigma_h = 1 mm) at L1 keeps the Rayleigh attenuation kappa small,
        # so the coherent term survives. The incoherent term still exists but is comparable
        # to or smaller than coherent at very low wind.
        geom = _mirror_geometry(theta_deg=10.0)
        result = OceanTarget(1e-3, 1.0, 0.0, 15.0, 35.0).compute_rcs(geom, 0.19)
        assert result.info["rcs_coherent_m2"][0] > 0.0
        assert result.info["rayleigh_roughness_kappa"][0] < 1.0    # Rayleigh-smooth regime

    def test_coherent_collapses_for_rough_sea(self):
        # At a typical sea state (sigma_h = 1.5 m) the Rayleigh attenuation kills the coherent term.
        geom = _mirror_geometry(theta_deg=10.0)
        result = _default_ocean().compute_rcs(geom, 0.19)
        coh = result.info["rcs_coherent_m2"][0]
        inc = result.info["rcs_incoherent_m2"][0]
        assert coh < 1e-6 * inc, f"coherent {coh:.3e} not negligible vs incoherent {inc:.3e}"

    def test_glistening_area_grows_with_wind(self):
        # Wider Cox-Munk slope variance -> bigger glistening zone.
        geom = _mirror_geometry(theta_deg=20.0)
        a_calm = OceanTarget(1.0, 3.0, 0.0, 15.0, 35.0).compute_rcs(geom, 0.19).info["glistening_area_m2"][0]
        a_rough = OceanTarget(1.0, 15.0, 0.0, 15.0, 35.0).compute_rcs(geom, 0.19).info["glistening_area_m2"][0]
        assert a_rough > a_calm

    def test_nan_geometry_propagates_to_nan_rcs(self):
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0]])
        tx = np.array([[np.nan], [0.0], [10.0]])
        rx = np.array([[0.0], [0.0], [5.0]])
        geom = build_bistatic_geometry(tx, rx, surf, normal)

        result = _default_ocean().compute_rcs(geom, wavelength_m=0.19)

        assert np.isnan(result.rcs_m2[0])
        assert np.isnan(result.info["nrcs"][0])

    def test_rayleigh_kappa_grows_with_wave_height(self):
        geom = _mirror_geometry(theta_deg=10.0)
        kappa_calm = OceanTarget(0.1, 5.0, 0.0, 15.0, 35.0).compute_rcs(geom, 0.19).info["rayleigh_roughness_kappa"]
        kappa_storm = OceanTarget(5.0, 5.0, 0.0, 15.0, 35.0).compute_rcs(geom, 0.19).info["rayleigh_roughness_kappa"]
        assert kappa_storm[0] > kappa_calm[0]


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
            tx_eirp_dbw=27.0,
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
            tx_eirp_dbw=27.0,
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
            tx_eirp_dbw=27.0,
            settings=None,
        )
        for key in (
            "incidence_deg",
            "scattering_deg",
            "bistatic_deg",
            "rcs_m2",
            "target_info",
            "received_power_w",
        ):
            assert key in lb
        assert "fresnel_reflectivity_RL_sq" in lb["target_info"]
        assert "slope_pdf" in lb["target_info"]

    def test_custom_target_replaces_ocean_rcs(self):
        class FixedRcs(Target):
            def compute_rcs(self, geometry, wavelength_m, illuminated_area_m2=None):
                n = geometry.range_tx_m.shape[0]
                return TargetRcs(rcs_m2=np.full(n, 100.0), info={"kind": "fixed"})

        tx, rx, rx_vel, refl, incidence_deg = self._canned_inputs()
        lb = compute_reflection_link_budget(
            tx_positions_eci=tx,
            rx_positions_eci=rx,
            rx_velocities_eci=rx_vel,
            reflection_positions_eci=refl,
            incidence_deg=incidence_deg,
            antenna=None,
            frequency_hz=1.57542e9,
            tx_eirp_dbw=27.0,
            settings=None,
            target=FixedRcs(),
        )
        np.testing.assert_allclose(lb["rcs_m2"], [100.0])
        assert lb["target_info"] == {"kind": "fixed"}

    def test_received_power_scales_linearly_with_rcs(self):
        tx, rx, rx_vel, refl, incidence_deg = self._canned_inputs()

        class FixedRcs(Target):
            def __init__(self, rcs):
                self.rcs = rcs

            def compute_rcs(self, geometry, wavelength_m, illuminated_area_m2=None):
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
            tx_eirp_dbw=27.0,
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
            tx_eirp_dbw=27.0,
            settings=None,
            target=FixedRcs(100.0),
        )
        np.testing.assert_allclose(lb_b["received_power_w"] / lb_a["received_power_w"], [2.0], rtol=1e-12)
