"""
CELL-CENTRIC COVERAGE CALCULATION for LEO-PNT Backscatter Footprint Analysis

Coverage criteria:
  1. Cell is within the Tx beam cone  (nadir angle from Tx ≤ TX_BEAM_HALF_ANGLE_DEG)
  2. Cell is within the steered Rx beam cone  (angle from Rx boresight ≤ RX_BEAM_HALF_ANGLE_DEG)
     The Rx boresight is steered to a fixed fraction between the Tx nadir and the geometric
     horizon in the Tx direction (steer_fraction: 0=Tx nadir, 1=horizon).
     The beam half-angle is derived from the antenna pattern at rx_beam_db_threshold.

Note: compute_footprint() was tried but its polygon construction breaks down when the beam
is steered >40° off nadir (the slant projection creates an unclosed arc that Shapely closes
into a massive wedge). A direct cone-angle check is equivalent and numerically stable at
all steering angles.
"""

import numpy as np
import pymap3d
from tqdm import tqdm

import functions.link_budget as lb
from functions.bistatic_geometry import tx_eirp_dbw as _tx_eirp_dbw, cell_snr_db as _cell_snr_db


class CoverageCellCentric:
    """Cell-centric coverage calculation for backscatter Tx-Rx pairs."""

    EARTH_RADIUS_M = 6_371_000.0
    TX_BEAM_HALF_ANGLE_DEG = 60.0

    def __init__(self, coverage_obj, antenna=None, steer_fraction=0.2,
                 rx_beam_db_threshold=-3.0, link_budget_settings=None,
                 nadir_pointing=False):
        """
        INPUTS:
            coverage_obj:          [Coverage]       Coverage object with reflection_dict populated
            antenna:               [RxAntenna]      Rx antenna object
            steer_fraction:        [float]          0 = steer toward Tx nadir, 1 = toward horizon
            rx_beam_db_threshold:  [float]          dB level that defines the Rx beam edge (e.g. -3.0)
            link_budget_settings:  [dict | None]    Link budget parameters; enables SNR tracking
        """
        self.coverage = coverage_obj
        self.antenna = antenna
        self.steer_fraction = steer_fraction
        self.rx_beam_db_threshold = rx_beam_db_threshold
        self.rx_beam_half_angle_deg = self._get_rx_beam_half_angle(antenna, rx_beam_db_threshold)
        self.tx_only_counts = {}   # gnss_name -> ndarray[n_cells] — filled after calculate_coverage_cell_centric
        self.snr_max = {}          # gnss_name -> ndarray[n_cells] of peak SNR [dB], only when lb enabled
        self.link_budget_settings = link_budget_settings
        self.nadir_pointing = nadir_pointing
        print(f"Rx beam half-angle at {rx_beam_db_threshold} dB: {self.rx_beam_half_angle_deg:.1f}°  "
              f"({'nadir' if nadir_pointing else f'steer={steer_fraction}'})")

        if link_budget_settings is not None:
            # Rx peak gain from antenna directivity (first lookup entry, steered at nadir)
            if antenna is not None:
                first_key = list(antenna.antenna_lookuptable.keys())[0]
                self.rx_peak_gain_dbi = float(antenna.antenna_lookuptable[first_key]["directivity_dB"])
            else:
                self.rx_peak_gain_dbi = 15.0  # ~3×4 array estimate
            self.cell_area_m2 = (getattr(coverage_obj, "grid_size_m", 30_000)) ** 2
            self.freq_hz = float(link_budget_settings.get("tx_frequency_mhz", 1176.45)) * 1e6
            tx_type = link_budget_settings.get("tx_type", "gnss")
            print(f"Link budget: tx_type={tx_type}, Rx peak={self.rx_peak_gain_dbi:.1f} dBi, "
                  f"cell={self.cell_area_m2:.2e} m², f={self.freq_hz/1e6:.2f} MHz")

    # ------------------------------------------------------------------
    # Antenna beam half-angle
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rx_beam_half_angle(antenna, db_threshold):
        """
        Scan the antenna pattern to find the nadir angle at which gain first
        drops to db_threshold below the peak (same logic as visualize_horizon_circles).
        Returns the maximum such angle across all phi slices.
        """
        if antenna is None:
            return 17.2  # fallback: -3 dB value from geometric config

        lookup_key = list(antenna.antenna_lookuptable.keys())[0]
        pattern_db = antenna.antenna_lookuptable[lookup_key]["radiation_pattern_dB"]
        n_theta = pattern_db.shape[0]
        theta_rad = np.linspace(0, np.pi, n_theta)
        angles = []

        for phi_i in range(pattern_db.shape[1]):
            slc = pattern_db[:, phi_i]
            peak = int(np.argmax(slc))
            for d in [1, -1]:
                for i in range(peak + d, (n_theta if d == 1 else -1), d):
                    if slc[i] <= db_threshold:
                        prev = i - d
                        if slc[prev] != slc[i]:
                            t = (db_threshold - slc[prev]) / (slc[i] - slc[prev])
                            theta = theta_rad[prev] + t * (theta_rad[i] - theta_rad[prev])
                        else:
                            theta = theta_rad[i]
                        angles.append(np.degrees(theta))
                        break
                else:
                    continue
                break

        return float(np.max(angles)) if angles else 17.2

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _steer_target_eci(self, rx_pos_eci, tx_pos_eci):
        """
        Beam boresight target on Earth surface in ECI (metres).
        Interpolates on the sphere between Tx nadir (f=0) and the geometric
        horizon in the Tx azimuth (f=1).
        """
        R = self.EARTH_RADIUS_M
        rx_up = rx_pos_eci / np.linalg.norm(rx_pos_eci)
        tx_nadir_unit = tx_pos_eci / np.linalg.norm(tx_pos_eci)

        tx_horiz = tx_nadir_unit - np.dot(tx_nadir_unit, rx_up) * rx_up
        norm = np.linalg.norm(tx_horiz)
        if norm < 1e-10:
            return tx_nadir_unit * R
        tx_horiz /= norm

        rx_r = np.linalg.norm(rx_pos_eci)
        eta = np.arccos(R / rx_r)
        horizon_unit = np.cos(eta) * rx_up + np.sin(eta) * tx_horiz

        f = self.steer_fraction
        target_unit = (1.0 - f) * tx_nadir_unit + f * horizon_unit
        target_unit /= np.linalg.norm(target_unit)
        return target_unit * R

    @staticmethod
    def _angle_deg(v1, v2):
        """Angle in degrees between two vectors."""
        return np.degrees(np.arccos(
            np.clip(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)), -1.0, 1.0)
        ))

    # ------------------------------------------------------------------
    # Main coverage calculation
    # ------------------------------------------------------------------

    def calculate_coverage_cell_centric(self, times, verbose=False):
        """
        For each grid cell determine which (Tx, Rx, time) combinations illuminate it.

        Coverage gate:
          (a) Cell is inside the Tx 60° beam cone  (nadir angle from Tx ≤ 60°)
          (b) Cell is inside the steered Rx beam cone  (angle from boresight ≤ beam half-angle)

        INPUTS:
            times:   [list[datetime]]  Simulation time vector
            verbose: [bool]            Show progress bar
        """
        if self.coverage.reflection_dict is None:
            print("Reflection dictionary is empty. Coverage is zero.")
            return

        if self.antenna is None:
            raise ValueError("RxAntenna object required — pass antenna= to CoverageCellCentric.")

        num_cells = len(self.coverage.grid_gdf)
        num_gnss = len(self.coverage.reflection_dict)

        with tqdm(total=num_cells * num_gnss, desc="Cell-centric coverage",
                  unit="cell", disable=not verbose) as pbar:

            for gnss_name, _gnss_value in self.coverage.gnss_constellation.items():
                refl_vector = self.coverage.reflection_dict[gnss_name]

                # --- Precompute per-(pair, time) quantities in ECEF ---
                # Angles are frame-invariant (ECI↔ECEF is a pure rotation), so
                # doing everything in ECEF avoids per-cell coordinate conversions.
                # Entry: (rx_ecef, boresight_ecef_unit, tx_ecef, valid, power, angle)
                pre = {}
                if verbose:
                    print(f"\nPrecomputing boresights for {gnss_name} ...")
                for refl_idx, refl_obj in enumerate(refl_vector):
                    for time_idx, time in enumerate(times):
                        if np.isnan(refl_obj.refl_LL_vec[time_idx, 0]):
                            pre[(refl_idx, time_idx)] = None
                            continue

                        rx_eci = refl_obj.sat_pos_vec[:, time_idx]
                        tx_eci = refl_obj.gnss_pos_vec[:, time_idx]

                        # Rx boresight target: nadir (straight down) or steered toward Tx
                        if self.nadir_pointing:
                            target_eci = rx_eci / np.linalg.norm(rx_eci) * self.EARTH_RADIUS_M
                        else:
                            target_eci = self._steer_target_eci(rx_eci, tx_eci)
                        boresight_eci = target_eci - rx_eci
                        boresight_eci /= np.linalg.norm(boresight_eci)

                        # Convert positions to ECEF once (angles are the same in both frames)
                        rx_ecef  = np.array(pymap3d.eci2ecef(rx_eci[0],  rx_eci[1],  rx_eci[2],  time))
                        tx_ecef  = np.array(pymap3d.eci2ecef(tx_eci[0],  tx_eci[1],  tx_eci[2],  time))

                        # Boresight direction in ECEF (rotate the ECI vector; same rotation as positions)
                        # Use finite-difference: target_ecef - rx_ecef  (target is on Earth surface)
                        target_ecef = np.array(pymap3d.eci2ecef(
                            target_eci[0], target_eci[1], target_eci[2], time
                        ))
                        bs_ecef = target_ecef - rx_ecef
                        bs_ecef /= np.linalg.norm(bs_ecef)

                        pre[(refl_idx, time_idx)] = (
                            rx_ecef, bs_ecef, tx_ecef,
                            refl_obj.power[time_idx],
                            refl_obj.refl_ang_vec[time_idx],
                            refl_obj.tx_satID,
                            refl_obj.rx_satID,
                        )

                cos_rx_thresh = np.cos(np.radians(self.rx_beam_half_angle_deg))
                cos_tx_thresh = np.cos(np.radians(self.TX_BEAM_HALF_ANGLE_DEG))
                R_m2 = self.EARTH_RADIUS_M ** 2   # visibility threshold: dot(sat_ecef, cell_ecef) ≥ R²
                num_cells = len(self.coverage.grid_gdf)
                tx_only = np.zeros(num_cells, dtype=np.int32)
                if self.link_budget_settings is not None:
                    self.snr_max[gnss_name] = np.full(num_cells, -np.inf)

                for cell_idx, cell_row in self.coverage.grid_gdf.iterrows():
                    cell_centroid = cell_row["geometry"].centroid
                    cell_ecef = np.array(pymap3d.geodetic2ecef(
                        cell_centroid.y, cell_centroid.x, 0.0
                    ))

                    for refl_idx in range(len(refl_vector)):
                        for time_idx, time in enumerate(times):

                            entry = pre.get((refl_idx, time_idx))
                            if entry is None:
                                continue

                            rx_ecef, bs_ecef, tx_ecef, power, incidence_angle, tx_satID, rx_satID = entry

                            # Earth-occlusion guard: cell must be on the near side of Earth
                            # for both Tx and Rx.  dot(sat_ecef, cell_ecef) ≥ R² ↔ cell within
                            # the satellite's visible hemisphere (standard spherical-Earth test).
                            if np.dot(tx_ecef, cell_ecef) < R_m2:
                                continue
                            if np.dot(rx_ecef, cell_ecef) < R_m2:
                                continue

                            # (a) Tx beam cone: nadir angle from Tx to cell ≤ 60°
                            tx_nadir = -tx_ecef / np.linalg.norm(tx_ecef)
                            tx_to_cell = cell_ecef - tx_ecef
                            norm_tc = np.linalg.norm(tx_to_cell)
                            if norm_tc == 0:
                                continue
                            if np.dot(tx_nadir, tx_to_cell / norm_tc) < cos_tx_thresh:
                                continue

                            # Tx illuminates this cell — count it regardless of Rx
                            tx_only[cell_idx] += 1

                            # (b) Rx beam cone: angle from Rx boresight to cell ≤ half-angle
                            rx_to_cell = cell_ecef - rx_ecef
                            norm = np.linalg.norm(rx_to_cell)
                            if norm == 0:
                                continue
                            if np.dot(bs_ecef, rx_to_cell / norm) < cos_rx_thresh:
                                continue

                            # (c) Optional SNR tracking (link budget)
                            if self.link_budget_settings is not None:
                                r_tx = np.linalg.norm(cell_ecef - tx_ecef)
                                r_rx = float(norm)
                                eirp = _tx_eirp_dbw(tx_ecef, cell_ecef, self.link_budget_settings)
                                sigma0_db = float(self.link_budget_settings.get("ocean_sigma0_db", -20.0))
                                snr = _cell_snr_db(eirp, self.rx_peak_gain_dbi,
                                                   r_tx, r_rx,
                                                   self.cell_area_m2, sigma0_db,
                                                   self.freq_hz, self.link_budget_settings)
                                if snr > self.snr_max[gnss_name][cell_idx]:
                                    self.snr_max[gnss_name][cell_idx] = snr

                            self.coverage.coverage_dict[gnss_name][cell_idx].times.append(time)
                            self.coverage.coverage_dict[gnss_name][cell_idx].tx_satID.append(tx_satID)
                            self.coverage.coverage_dict[gnss_name][cell_idx].rx_satID.append(rx_satID)
                            self.coverage.coverage_dict[gnss_name][cell_idx].angles.append(incidence_angle)
                            self.coverage.coverage_dict[gnss_name][cell_idx].power.append(power)

                    pbar.update(1)

                self.tx_only_counts[gnss_name] = tx_only

    # ------------------------------------------------------------------
    # Legacy / utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bistatic_angle(tx_pos_eci, rx_pos_eci, refl_pos_eci):
        to_tx = (tx_pos_eci - refl_pos_eci) / np.linalg.norm(tx_pos_eci - refl_pos_eci)
        to_rx = (rx_pos_eci - refl_pos_eci) / np.linalg.norm(rx_pos_eci - refl_pos_eci)
        return np.degrees(np.arccos(np.clip(np.dot(to_tx, to_rx), -1.0, 1.0)))

    def compare_approaches(self, times):
        """Compare trajectory-based vs cell-centric coverage cell counts."""
        print("\n" + "=" * 70)
        print("COVERAGE APPROACH COMPARISON")
        print("=" * 70)

        trajectory_coverage = {
            gnss_name: sum(
                1 for idx in range(len(self.coverage.grid_gdf))
                if len(self.coverage.coverage_dict[gnss_name][idx].times) > 0
            )
            for gnss_name in self.coverage.gnss_constellation.keys()
        }

        self.calculate_coverage_cell_centric(times, verbose=False)

        print(f"\n{'Constellation':<20} {'Trajectory':<15} {'Cell-Centric':<15} {'Diff':<10}")
        print("-" * 60)
        for gnss_name in self.coverage.gnss_constellation.keys():
            traj = trajectory_coverage.get(gnss_name, 0)
            cc = sum(
                1 for idx in range(len(self.coverage.grid_gdf))
                if len(self.coverage.coverage_dict[gnss_name][idx].times) > 0
            )
            print(f"{gnss_name:<20} {traj:<15} {cc:<15} {cc - traj:+}")
        print("=" * 70)
