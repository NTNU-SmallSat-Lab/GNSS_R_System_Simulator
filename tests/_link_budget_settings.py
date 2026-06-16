"""Canonical fully-specified BistaticLinkBudgetSettings for tests.

Production has no default fallbacks for physical ocean/loss inputs, so tests supply a
complete settings object for those values. Optional noise/patch controls can still be
overridden here where a test needs to vary one.
"""

from functions.link_budget import BistaticLinkBudgetSettings


def link_budget_settings(**overrides) -> BistaticLinkBudgetSettings:
    """Return a complete BistaticLinkBudgetSettings; pass field=value to override any default test value."""
    values = {
        "system_losses_db": 2.0,
        "polarization_losses_db": 0.5,
        "noise_bandwidth_hz": 2.0e6,
        "system_noise_temperature_k": 290.0,
        "detection_sinr_threshold_db": 0.0,
        "ocean_wave_height_m": 1.5,
        "ocean_wind_speed_m_s": 7.0,
        "ocean_wind_direction_deg": 270.0,
        "ocean_sea_temperature_c": 15.0,
        "ocean_salinity_psu": 35.0,
        "ocean_inverse_wave_age": 0.84,
    }
    values.update(overrides)
    return BistaticLinkBudgetSettings(**values)
