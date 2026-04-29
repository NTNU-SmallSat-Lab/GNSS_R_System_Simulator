"""Sanity check: at low latitudes, the elliptical specular point should
agree with the spherical FF result to within a few km. Catches regressions
in the SLSQP setup, frame conversions, or residual definition."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pymap3d

from object_definitions.Constants_def import Constants
from object_definitions.Reflection_def import Reflection


def main() -> None:
    R_LEO = Constants.RADIUS_EARTH_EQUAT + 600e3   # ~600 km altitude
    R_GNSS = 26.6e6                                # MEO

    # Equatorial-plane geometries, varying central angle. Low latitude => the
    # flattening effect on the specular point is small, so ellipse ~ sphere.
    t = np.datetime64("2026-01-01T00:00:00")
    refl = Reflection(tx_satID="dummy", rx_satID="dummy",
                      earth_model="elliptical", angle=45)

    cases = []
    for sep_deg in (10.0, 20.0, 30.0, 40.0):
        Rx = np.array([R_LEO, 0.0, 0.0])
        a = np.deg2rad(sep_deg)
        Tx = R_GNSS * np.array([np.cos(a), np.sin(a), 0.0])

        S_ell, _, _ = refl.find_specular_point_ellipse(Tx, Rx, t)
        S_sph, _, _ = refl.get_reflection_point_ff(Tx, Rx)
        if np.any(np.isnan(S_ell)) or np.any(np.isnan(S_sph)):
            raise AssertionError(f"NaN at sep={sep_deg} deg: ellipse={S_ell}, sphere={S_sph}")

        lat_e, lon_e, _ = pymap3d.eci2geodetic(*S_ell, t)
        lat_s, lon_s, _ = pymap3d.eci2geodetic(*S_sph, t)

        # Great-circle separation, mean Earth radius
        dlat = np.deg2rad(lat_e - lat_s)
        dlon = np.deg2rad(lon_e - lon_s)
        h = (np.sin(dlat / 2) ** 2
             + np.cos(np.deg2rad(lat_s)) * np.cos(np.deg2rad(lat_e)) * np.sin(dlon / 2) ** 2)
        d_m = 2 * Constants.R_EARTH_MEAN * np.arcsin(np.sqrt(h))
        cases.append((sep_deg, d_m, lat_e))

        assert d_m < 5e3, (
            f"sep={sep_deg} deg: ellipse vs sphere specular disagreement {d_m/1e3:.2f} km "
            f"(>5 km tolerance). lat_ellipse={lat_e:.3f} deg"
        )

    print("Specular ellipse <-> sphere consistency OK:")
    for sep, d, lat in cases:
        print(f"  sep={sep:>5.1f} deg  delta={d:7.1f} m  lat={lat:+.3f} deg")


if __name__ == "__main__":
    main()
