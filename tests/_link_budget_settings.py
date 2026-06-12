"""Canonical fully-specified BistaticLinkBudgetSettings for tests.

Production has no default fallbacks (every field is mandatory), so tests must supply a
complete settings object. This helper is the single place that holds the standard test
values; override individual fields via keyword arguments where a test needs to vary one.
"""

from functions.link_budget import BistaticLinkBudgetSettings


def link_budget_settings(**overrides) -> BistaticLinkBudgetSettings:
    """Return a complete BistaticLinkBudgetSettings; pass field=value to override any default test value."""
    values = dict(
        system_losses_db=2.0,
        polarization_losses_db=0.5,
        ocean_wave_height_m=1.5,
        ocean_wind_speed_m_s=7.0,
        ocean_wind_direction_deg=270.0,
        ocean_sea_temperature_c=15.0,
        ocean_salinity_psu=35.0,
        ocean_inverse_wave_age=0.84,
    )
    values.update(overrides)
    return BistaticLinkBudgetSettings(**values)
