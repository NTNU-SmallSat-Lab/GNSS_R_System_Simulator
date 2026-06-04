"""Ship RCS model: rectangular hull broken into five flat plates plus two hull-water dihedrals.

Plates:    deck, port hull, starboard hull, bow, stern
Dihedrals: port hull-water, starboard hull-water (along the waterline on each side)

Each plate uses the PO bistatic RCS in `functions.ship_physics.flat_plate_rcs` with back-face culling.
Each dihedral uses the 90-deg corner RCS in `functions.ship_physics.dihedral_rcs` with wedge visibility.
Contributions sum non-coherently; the `info` dict breaks the total down per component.

The ship body frame is built per sample:
    bow_hat       = provided heading (re-orthogonalised against the local surface normal)
    mast_hat      = surface normal (geometry.n_surface)
    starboard_hat = bow_hat x mast_hat   (right-hand rule)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from functions.ship_physics import dihedral_rcs, flat_plate_rcs
from object_definitions.Target_def import Target, TargetRcs

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry


class ShipTarget(Target):
    """Ship RCS parameterised by length, beam, freeboard, and per-sample heading vectors."""

    def __init__(
        self,
        length_m: float,
        beam_m: float,
        freeboard_m: float,
        heading_eci: np.ndarray,
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

    @classmethod
    def from_dict(cls, ship_dict: dict, heading_eci: np.ndarray) -> ShipTarget:
        """Build a ShipTarget from the INI-parsed dimensions dict and a heading array."""
        return cls(
            length_m=ship_dict["length_m"],
            beam_m=ship_dict["beam_m"],
            freeboard_m=ship_dict["freeboard_m"],
            heading_eci=heading_eci,
        )

    def compute_rcs(self, geometry: BistaticGeometry, wavelength_m: float, illuminated_area_m2: np.ndarray | None = None) -> TargetRcs:
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

        # Hull-water dihedrals: edge along bow, legs are hull-side outward and the local water surface up.
        rcs_dihedral_stbd = dihedral_rcs(
            edge_hat=bow_hat,
            leg_1_normal=starboard_hat,
            leg_2_normal=mast_hat,
            length_edge_m=self.length_m,
            leg_dimension_m=self.freeboard_m,
            n_tx2s=n_tx2s,
            n_s2rx=n_s2rx,
            wavelength_m=wavelength_m,
        )
        rcs_dihedral_stbd = np.where(valid_axes, rcs_dihedral_stbd, np.nan)

        rcs_dihedral_port = dihedral_rcs(
            edge_hat=bow_hat,
            leg_1_normal=port_hat,
            leg_2_normal=mast_hat,
            length_edge_m=self.length_m,
            leg_dimension_m=self.freeboard_m,
            n_tx2s=n_tx2s,
            n_s2rx=n_s2rx,
            wavelength_m=wavelength_m,
        )
        rcs_dihedral_port = np.where(valid_axes, rcs_dihedral_port, np.nan)

        rcs_dihedrals_total = (
            np.where(np.isfinite(rcs_dihedral_stbd), rcs_dihedral_stbd, 0.0)
            + np.where(np.isfinite(rcs_dihedral_port), rcs_dihedral_port, 0.0)
        )

        rcs_total = rcs_plates_total + rcs_dihedrals_total
        rcs_total = np.where(valid_axes, rcs_total, np.nan)

        return TargetRcs(
            rcs_m2=rcs_total,
            info={
                "rcs_plates_total_m2": np.where(valid_axes, rcs_plates_total, np.nan),
                "rcs_dihedrals_total_m2": np.where(valid_axes, rcs_dihedrals_total, np.nan),
                "rcs_dihedral_starboard_m2": rcs_dihedral_stbd,
                "rcs_dihedral_port_m2": rcs_dihedral_port,
                "rcs_per_plate_m2": per_plate_rcs,
            },
        )
