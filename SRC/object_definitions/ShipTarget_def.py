"""Ship RCS model: rectangular hull broken into five flat plates plus four hull-water dihedrals.

Plates:    deck, port hull, starboard hull, bow, stern
Dihedrals: starboard, port, bow and stern hull-water (along the waterline of each hull face)

Each plate uses the PO bistatic RCS in `functions.ship_physics.flat_plate_rcs` with back-face culling.
Each dihedral uses the 90-deg corner RCS in `functions.ship_physics.dihedral_rcs` with wedge visibility,
times the seawater sea-leg factor |R_RR|^2 exp(-kappa), plus the diffuse corner: the same-sense sea scatter
of the OceanTarget that reaches Rx through the hull mirror.
Contributions sum non-coherently; the `info` dict breaks the total down per component.

The ship body frame is built per sample:
    bow_hat       = provided heading (re-orthogonalised against the local surface normal)
    mast_hat      = surface normal (geometry.n_surface)
    starboard_hat = bow_hat x mast_hat   (right-hand rule)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from functions.link_budget import build_bistatic_geometry
from functions.ocean_physics import circular_pol_reflectivities, rayleigh_roughness, seawater_permittivity
from functions.ship_physics import dihedral_rcs, flat_plate_rcs
from object_definitions.Constants_def import Constants
from object_definitions.Target_def import Target, TargetRcs

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry
    from object_definitions.OceanTarget_def import OceanTarget


class ShipTarget(Target):
    """Ship RCS parameterised by length, beam, freeboard, per-sample heading vectors, and the sea it floats on."""

    def __init__(
        self,
        length_m: float,
        beam_m: float,
        freeboard_m: float,
        heading_eci: np.ndarray,
        ocean: OceanTarget,
    ) -> None:
        if length_m <= 0.0:
            raise ValueError(f"length_m must be > 0; got {length_m}")
        if beam_m <= 0.0:
            raise ValueError(f"beam_m must be > 0; got {beam_m}")
        if freeboard_m <= 0.0:
            raise ValueError(f"freeboard_m must be > 0; got {freeboard_m}")

        heading_eci = np.asarray(heading_eci, dtype=float)
        if heading_eci.ndim != 2 or heading_eci.shape[1] != 3:
            raise ValueError(f"heading_eci must have shape (n, 3); got {heading_eci.shape}")

        self.length_m = float(length_m)
        self.beam_m = float(beam_m)
        self.freeboard_m = float(freeboard_m)
        self.heading_eci = heading_eci
        self.ocean = ocean

    @classmethod
    def from_dict(cls, ship_dict: dict, heading_eci: np.ndarray, ocean: OceanTarget) -> ShipTarget:
        """Build a ShipTarget from the INI-parsed dimensions dict, a heading array and the sea state."""
        return cls(
            length_m=ship_dict["length_m"],
            beam_m=ship_dict["beam_m"],
            freeboard_m=ship_dict["freeboard_m"],
            heading_eci=heading_eci,
            ocean=ocean,
        )

    def compute_rcs(self, geometry: BistaticGeometry, wavelength_m: float, illuminated_area_m2: np.ndarray | None = None) -> TargetRcs:
        """Per-sample ship RCS as the non-coherent sum of five plates and four hull-water dihedrals.

        ``illuminated_area_m2`` is ignored: a ship is a point target with an absolute RCS. The ``info``
        dict breaks the total down per plate and per dihedral.
        """
        # illuminated_area_m2 is unused: a ship is a point target with an absolute RCS (no area factor).
        n_tx2s = geometry.n_tx2s
        n_s2rx = geometry.n_s2rx
        mast_hat = geometry.n_surface                                  # outward surface normal = up
        n_samples = n_tx2s.shape[0]

        if self.heading_eci.shape[0] != n_samples:
            raise ValueError(
                f"heading_eci has {self.heading_eci.shape[0]} samples but geometry has {n_samples}"
            )

        # Build ship body frame; bow re-orthogonalised against mast so it lies in the tangent plane.
        bow_input = self.heading_eci
        bow_proj_mast = np.einsum("ij,ij->i", bow_input, mast_hat)     # bow component along mast
        bow_tangent = bow_input - bow_proj_mast[:, None] * mast_hat    # remove vertical component
        bow_mag = np.linalg.norm(bow_tangent, axis=1)
        valid_axes = (
            np.isfinite(mast_hat).all(axis=1)
            & np.isfinite(bow_input).all(axis=1)
            & (bow_mag > 1e-9)
        )

        bow_hat = np.full_like(mast_hat, np.nan)
        bow_hat[valid_axes] = bow_tangent[valid_axes] / bow_mag[valid_axes, None]

        starboard_hat = np.full_like(mast_hat, np.nan)
        starboard_hat[valid_axes] = np.cross(bow_hat[valid_axes], mast_hat[valid_axes])

        port_hat = -starboard_hat                                      # by symmetry
        stern_hat = -bow_hat

        # Plate definitions: (name, outward_normal, edge_1, length_1_m, edge_2, length_2_m)
        plate_specs = [
            ("deck",      mast_hat,      bow_hat,       self.length_m,  starboard_hat, self.beam_m),
            ("starboard", starboard_hat, bow_hat,       self.length_m,  mast_hat,      self.freeboard_m),
            ("port",      port_hat,      bow_hat,       self.length_m,  mast_hat,      self.freeboard_m),
            ("bow",       bow_hat,       starboard_hat, self.beam_m,    mast_hat,      self.freeboard_m),
            ("stern",     stern_hat,     starboard_hat, self.beam_m,    mast_hat,      self.freeboard_m),
        ]

        per_plate_rcs: dict[str, np.ndarray] = {}
        rcs_plates_total = np.zeros(n_samples, dtype=float)
        for name, n_plate, edge_1, len_1, edge_2, len_2 in plate_specs:
            sigma = flat_plate_rcs(
                n_plate=n_plate,
                edge_1=edge_1,
                edge_2=edge_2,
                length_edge_1_m=len_1,
                length_edge_2_m=len_2,
                n_tx2s=n_tx2s,
                n_s2rx=n_s2rx,
                wavelength_m=wavelength_m,
            )
            sigma = np.where(valid_axes, sigma, np.nan)
            per_plate_rcs[name] = sigma
            rcs_plates_total = rcs_plates_total + np.where(np.isfinite(sigma), sigma, 0.0)

        # Sea leg of the corner: seawater, not metal. The double bounce keeps the circular sense, so only the
        # sea's same-sense amplitude R_RR reaches the LHCP receiver; roughness removes the coherent part by exp(-kappa).
        cos_sea = np.clip(np.einsum("ij,ij->i", -n_tx2s, mast_hat), 0.0, 1.0)     # incidence cosine on the sea leg, both path orders
        eps_r = seawater_permittivity(Constants.SPEED_OF_LIGHT / wavelength_m, self.ocean.sea_temperature_c, self.ocean.salinity_psu)
        _, R_RR_sq = circular_pol_reflectivities(np.arccos(cos_sea), eps_r)
        kappa = rayleigh_roughness(self.ocean.wave_height_m, cos_sea, wavelength_m)           # Rayleigh roughness, sigma_h = H_s / 4
        sea_leg = R_RR_sq * np.exp(-kappa)

        # Hull-water dihedrals: (name, hull-leg outward normal, edge, edge length); the other leg is the local water surface.
        corner_specs = [
            ("starboard", starboard_hat, bow_hat,       self.length_m),
            ("port",      port_hat,      bow_hat,       self.length_m),
            ("bow",       bow_hat,       starboard_hat, self.beam_m),
            ("stern",     stern_hat,     starboard_hat, self.beam_m),
        ]

        per_corner_rcs: dict[str, np.ndarray] = {}
        rcs_dihedrals_total = np.zeros(n_samples, dtype=float)
        rcs_diffuse_total = np.zeros(n_samples, dtype=float)
        for name, n_hull, edge_hat, len_edge in corner_specs:
            diffuse = self._diffuse_corner_rcs(geometry, n_hull, len_edge, wavelength_m)
            sigma = diffuse + sea_leg * dihedral_rcs(
                edge_hat=edge_hat,
                leg_1_normal=n_hull,
                leg_2_normal=mast_hat,
                length_edge_m=len_edge,
                leg_dimension_m=self.freeboard_m,
                n_tx2s=n_tx2s,
                n_s2rx=n_s2rx,
                wavelength_m=wavelength_m,
            )
            sigma = np.where(valid_axes, sigma, np.nan)
            per_corner_rcs[f"rcs_dihedral_{name}_m2"] = sigma
            rcs_dihedrals_total = rcs_dihedrals_total + np.where(np.isfinite(sigma), sigma, 0.0)
            rcs_diffuse_total = rcs_diffuse_total + diffuse

        rcs_total = rcs_plates_total + rcs_dihedrals_total
        rcs_total = np.where(valid_axes, rcs_total, np.nan)

        return TargetRcs(
            rcs_m2=rcs_total,
            info={
                "rcs_plates_total_m2": np.where(valid_axes, rcs_plates_total, np.nan),
                "rcs_dihedrals_total_m2": np.where(valid_axes, rcs_dihedrals_total, np.nan),
                "rcs_dihedrals_diffuse_m2": np.where(valid_axes, rcs_diffuse_total, np.nan),
                **per_corner_rcs,
                "rcs_per_plate_m2": per_plate_rcs,
            },
        )

    def _diffuse_corner_rcs(self, geometry: BistaticGeometry, hull_hat: np.ndarray, edge_length_m: float, wavelength_m: float) -> np.ndarray:
        """Diffuse hull-sea corner: same-sense sea scatter that reaches Rx through the hull mirror, both path orders."""
        up, to_tx, to_rx = geometry.n_surface, -geometry.n_tx2s, geometry.n_s2rx
        a_hull, b_hull = np.einsum("ij,ij->i", to_tx, hull_hat), np.einsum("ij,ij->i", to_rx, hull_hat)
        a_sea, b_sea = np.einsum("ij,ij->i", to_tx, up), np.einsum("ij,ij->i", to_rx, up)
        rcs = np.zeros(up.shape[0])
        vis = np.flatnonzero((a_hull > 0.0) & (b_hull > 0.0) & (a_sea > 0.0) & (b_sea > 0.0))   # open corner quadrant
        if vis.size == 0:
            return rcs

        hull, range_tx, range_rx = hull_hat[vis], geometry.range_tx_m[vis, None], geometry.range_rx_m[vis, None]
        paths = (
            (to_tx[vis], to_rx[vis] - 2.0 * b_hull[vis, None] * hull, b_hull[vis] / b_sea[vis]),   # sea first: scatter toward the hull image of Rx
            (to_tx[vis] - 2.0 * a_hull[vis, None] * hull, to_rx[vis], a_hull[vis] / a_sea[vis]),   # hull first: sea lit from the hull image of Tx
        )
        for to_tx_path, to_rx_path, tan_cos in paths:
            path = build_bistatic_geometry((range_tx * to_tx_path).T, (range_rx * to_rx_path).T, np.zeros_like(hull), up[vis])
            info = self.ocean.compute_rcs(path, wavelength_m).info
            nrcs_rr = info["nrcs"] * info["fresnel_reflectivity_RR_sq"] / info["fresnel_reflectivity_RL_sq"] + info["nrcs_bragg_RR"]   # same-sense GO + Bragg
            rcs[vis] += nrcs_rr * edge_length_m * self.freeboard_m * tan_cos      # sea strip the hull mirrors: L_e H tan(theta) cos(phi)
        return rcs
