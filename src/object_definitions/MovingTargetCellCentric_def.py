"""
Cell-centric detection criterion for a moving target.

At each timestep t, a ship at position P is considered detectable if:
  1. P is within the Tx 60° nadir cone  (Earth-illuminated footprint)
  2. P is within the steered Rx receive beam  (angle from boresight ≤ rx_beam_half_angle)

This is independent of whether a valid specular/backscatter reflection point
exists — it asks purely: can the ship backscatter reach the Rx receiver?
"""

import datetime as dt
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pymap3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import ConvexHull
from tqdm import tqdm

from functions.bistatic_geometry import (
    EARTH_RADIUS_M,
    TX_BEAM_HALF_ANGLE_DEG,
    get_rx_beam_half_angle,
    steer_target_eci,
)
from object_definitions.Constants_def import Constants


@dataclass
class MTCCDetection:
    """One cell-centric detection event along the ship track."""
    time:       dt.datetime
    ship_lat:   float
    ship_lon:   float
    tx_sat_idx: int
    rx_sat_idx: int
    t_idx:      int          # index into timestamps_resampled / positions_eci
    gnss_name:  str


class MovingTargetCellCentric:
    """
    Cell-centric beam-overlap check for a moving ship.

    Usage (after propagation):
        cc = MovingTargetCellCentric(mtd, antenna, steer_fraction=0.2)
        cc.calculate(verbose=True)
        cc.show_detections(file_name="Detection_cellcentric.png")
    """

    def __init__(self, mtd_obj, antenna, steer_fraction=0.2,
                 rx_beam_db_threshold=-3.0, max_bistatic_angle_deg=None):
        self.mtd              = mtd_obj
        self.antenna          = antenna
        self.steer_fraction   = steer_fraction
        self.db_threshold     = rx_beam_db_threshold
        self.max_bistatic_angle_deg = max_bistatic_angle_deg  # None = no limit
        self.rx_beam_half_angle_deg = get_rx_beam_half_angle(antenna, rx_beam_db_threshold)
        self.detections       = {}   # gnss_name -> list[MTCCDetection]

        # Pre-compute -3 dB antenna contour vectors (antenna frame, unit vectors)
        self._contour_vecs = self._build_contour(antenna, rx_beam_db_threshold)
        self._R_ant2body   = getattr(antenna, "R_antenna_to_body", np.eye(3))

        bistatic_str = (f"{max_bistatic_angle_deg}°"
                        if max_bistatic_angle_deg is not None else "unlimited")
        print(f"[MovingTargetCellCentric] Rx beam ½-angle at "
              f"{rx_beam_db_threshold} dB: {self.rx_beam_half_angle_deg:.1f}°  |  "
              f"steer_fraction={steer_fraction}  |  "
              f"bistatic limit={bistatic_str}  |  "
              f"contour: {len(self._contour_vecs)} vectors")

    # ------------------------------------------------------------------
    @staticmethod
    def _build_contour(antenna, db_threshold):
        """Extract the dB-threshold contour as unit vectors in antenna frame."""
        orig = antenna.footprint_dB_contour
        antenna.footprint_dB_contour = db_threshold

        freq = antenna.frequencies[0]
        lkey = antenna._resolve_unsteered_footprint_lookup_key(freq)
        pat  = antenna.antenna_lookuptable[lkey]["radiation_pattern_dB"]
        nt, nphi = pat.shape
        th = np.linspace(0, np.pi,    nt)
        ph = np.linspace(0, 2*np.pi, nphi)
        th_m, ph_m = np.meshgrid(th, ph, indexing='ij')

        raw    = antenna._find_contour_vectors(pat, th, ph, th_m, ph_m)
        smooth = antenna._smooth_antenna_footprint(raw)

        antenna.footprint_dB_contour = orig
        return smooth   # shape (N, 3), unit vectors

    # ------------------------------------------------------------------
    def _steered_footprint_latlons(self, rx_eci, rx_vel, tx_eci, epoch):
        """
        Compute the steered -3 dB footprint on Earth's surface for one
        (Rx, Tx) pair at one epoch.  Returns (lats, lons) lists.
        """
        R = EARTH_RADIUS_M

        rx_up  = rx_eci / np.linalg.norm(rx_eci)
        tx_nd  = tx_eci / np.linalg.norm(tx_eci)

        # Steer target (same logic as steer_target_eci but we need the full beam)
        tx_h = tx_nd - np.dot(tx_nd, rx_up) * rx_up
        n = np.linalg.norm(tx_h)
        if n < 1e-10:
            return [], []
        tx_h /= n
        eta      = np.arccos(np.clip(R / np.linalg.norm(rx_eci), -1, 1))
        h_unit   = np.cos(eta) * rx_up + np.sin(eta) * tx_h
        tgt_unit = (1 - self.steer_fraction) * tx_nd + self.steer_fraction * h_unit
        tgt_unit /= np.linalg.norm(tgt_unit)
        tgt_eci  = tgt_unit * R

        # Body-frame → ECI rotation (x=fwd, y=cross, z=nadir)
        rx_fwd  = rx_vel - np.dot(rx_vel, rx_up) * rx_up
        nf = np.linalg.norm(rx_fwd)
        if nf < 1e-10:
            return [], []
        rx_fwd /= nf
        rx_crs   = np.cross(rx_up, rx_fwd)
        R_b2eci  = np.column_stack([rx_fwd, rx_crs, -rx_up]) @ self._R_ant2body

        # Rodrigues rotation: rotate nadir (0,0,1) → look direction in antenna frame
        look_eci = tgt_eci - rx_eci
        look_eci /= np.linalg.norm(look_eci)
        look_ant  = R_b2eci.T @ look_eci
        look_ant /= np.linalg.norm(look_ant)

        bs = np.array([0., 0., 1.])
        ax_rot = np.cross(bs, look_ant)
        nax = np.linalg.norm(ax_rot)
        if nax > 1e-10:
            ax_rot /= nax
            ang  = np.arccos(np.clip(np.dot(bs, look_ant), -1.0, 1.0))
            K    = np.array([[0, -ax_rot[2], ax_rot[1]],
                             [ax_rot[2], 0, -ax_rot[0]],
                             [-ax_rot[1], ax_rot[0], 0]])
            R_st = np.eye(3) + np.sin(ang)*K + (1 - np.cos(ang))*(K @ K)
            steered = self._contour_vecs @ R_st.T
        else:
            steered = self._contour_vecs

        # Rotate to ECI and ray-cast to Earth
        vecs_eci = (R_b2eci @ steered.T).T
        lats, lons = [], []
        rx_r2 = np.dot(rx_eci, rx_eci)
        for v in vecs_eci:
            b    = np.dot(rx_eci, v)
            disc = b**2 - (rx_r2 - R**2)
            if disc < 0:
                continue
            t_hit = -b - np.sqrt(disc)
            if t_hit < 0:
                continue
            pt_eci  = rx_eci + t_hit * v
            pt_ecef = np.array(pymap3d.eci2ecef(*pt_eci, epoch))
            la, lo, _ = pymap3d.ecef2geodetic(*pt_ecef)
            lats.append(float(la))
            lons.append(float(lo))
        if len(lats) < 3:
            return [], []

        # Convex hull guarantees a non-self-intersecting boundary, eliminating
        # ring artifacts that angular sort can produce for concave limb footprints.
        pts = np.column_stack([lons, lats])
        try:
            hull = ConvexHull(pts)
            verts = hull.vertices
        except Exception:
            return [], []
        return [lats[i] for i in verts], [lons[i] for i in verts]

    # ------------------------------------------------------------------
    def calculate(self, verbose=True):
        """
        Iterate over every visible timestep and every (Rx, Tx) pair.
        Records a detection whenever the ship position satisfies both
        the Tx 60° cone and the steered Rx beam criteria.
        """
        cos_rx       = np.cos(np.radians(self.rx_beam_half_angle_deg))
        cos_tx       = np.cos(np.radians(TX_BEAM_HALF_ANGLE_DEG))
        R_m2         = EARTH_RADIUS_M ** 2
        cos_bistatic = (np.cos(np.radians(self.max_bistatic_angle_deg))
                        if self.max_bistatic_angle_deg is not None else -2.0)

        timestamps     = self.mtd.timestamps_resampled
        ship_positions = self.mtd.ship_positions_resampled

        ship_ecef_arr = np.array([
            pymap3d.geodetic2ecef(pt.y, pt.x, 0.0)
            for pt in ship_positions
        ])

        sat_const = self.mtd.sat_constellation
        gnss_dict = self.mtd.gnss_constellation

        for gnss_name, gnss_const in gnss_dict.items():
            detections = []
            n_rx = len(sat_const.satellites)
            n_tx = len(gnss_const.satellites)
            n_t  = len(timestamps)

            with tqdm(total=n_t, desc=f"Cell-centric MTD [{gnss_name}]",
                      unit="step", disable=not verbose) as pbar:

                for t_idx, (time_i, ship_ecef) in enumerate(
                        zip(timestamps, ship_ecef_arr)):

                    ship_lat  = ship_positions[t_idx].y
                    ship_lon  = ship_positions[t_idx].x
                    ship_ecef = np.asarray(ship_ecef)

                    for rx_i, rx_sat in enumerate(sat_const.satellites):
                        rx_eci = rx_sat.positions_eci[:, t_idx]
                        if np.any(np.isnan(rx_eci)):
                            continue

                        rx_ecef = np.array(pymap3d.eci2ecef(*rx_eci, time_i))

                        for tx_i, tx_sat in enumerate(gnss_const.satellites):
                            tx_eci = tx_sat.positions_eci[:, t_idx]
                            if np.any(np.isnan(tx_eci)):
                                continue

                            tx_ecef = np.array(pymap3d.eci2ecef(*tx_eci, time_i))

                            # ── 1. Earth-occlusion guard ───────────────
                            if np.dot(tx_ecef, ship_ecef) < R_m2:
                                continue
                            if np.dot(rx_ecef, ship_ecef) < R_m2:
                                continue

                            # ── 2. Tx 60° nadir cone ──────────────────
                            tx_nadir   = -tx_ecef / np.linalg.norm(tx_ecef)
                            tx_to_ship = ship_ecef - tx_ecef
                            norm_ts    = np.linalg.norm(tx_to_ship)
                            if norm_ts == 0:
                                continue
                            if np.dot(tx_nadir, tx_to_ship / norm_ts) < cos_tx:
                                continue

                            # ── 3. Steered Rx beam ─────────────────────
                            tgt_eci  = steer_target_eci(rx_eci, tx_eci,
                                                        self.steer_fraction)
                            tgt_ecef = np.array(pymap3d.eci2ecef(*tgt_eci, time_i))
                            bs_ecef  = tgt_ecef - rx_ecef
                            bs_norm  = np.linalg.norm(bs_ecef)
                            if bs_norm == 0:
                                continue
                            bs_ecef /= bs_norm

                            rx_to_ship = ship_ecef - rx_ecef
                            norm_rs    = np.linalg.norm(rx_to_ship)
                            if norm_rs == 0:
                                continue
                            if np.dot(bs_ecef, rx_to_ship / norm_rs) < cos_rx:
                                continue

                            # ── 4. Bistatic angle at ship ──────────────
                            if self.max_bistatic_angle_deg is not None:
                                dir_to_tx = tx_ecef - ship_ecef
                                dir_to_tx /= np.linalg.norm(dir_to_tx)
                                dir_to_rx = rx_ecef - ship_ecef
                                dir_to_rx /= np.linalg.norm(dir_to_rx)
                                if np.dot(dir_to_tx, dir_to_rx) < cos_bistatic:
                                    continue

                            detections.append(MTCCDetection(
                                time       = time_i,
                                ship_lat   = ship_lat,
                                ship_lon   = ship_lon,
                                tx_sat_idx = tx_i,
                                rx_sat_idx = rx_i,
                                t_idx      = t_idx,
                                gnss_name  = gnss_name,
                            ))

                    pbar.update(1)

            self.detections[gnss_name] = detections
            print(f"  [{gnss_name}] {len(detections)} detection events "
                  f"over {n_t} timesteps  ({n_rx} Rx × {n_tx} Tx)")

    # ------------------------------------------------------------------
    def show_detections(self, file_name=None, show=False, footprints=True):
        """
        Plot ship track with detection events.  If footprints=True, draw the
        steered -3 dB beam contour for every unique (t_idx, rx_i, tx_i) event.
        """
        fig, ax = plt.subplots(figsize=(14, 9),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        pc = ccrs.PlateCarree()

        waypts = self.mtd.moving_target_object.trajectory_points
        lats   = [w[0] for w in waypts]
        lons   = [w[1] for w in waypts]
        margin = 4.0
        ax.set_extent([min(lons)-margin, max(lons)+margin,
                       min(lats)-margin, max(lats)+margin], crs=pc)

        ax.add_feature(cfeature.LAND,      facecolor="#d8d8d8", zorder=1)
        ax.add_feature(cfeature.OCEAN,     facecolor="#cce5f0", alpha=0.6, zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False; gl.right_labels = False

        ax.plot([w[1] for w in waypts], [w[0] for w in waypts],
                "b-", lw=2, alpha=0.5, label="Ship track", transform=pc, zorder=3)

        sat_const  = self.mtd.sat_constellation
        gnss_dict  = self.mtd.gnss_constellation
        timestamps = self.mtd.timestamps_resampled

        # ── Per-Rx colour map ─────────────────────────────────────────
        all_dets  = [d for dets in self.detections.values() for d in dets]
        rx_ids    = sorted(set(d.rx_sat_idx for d in all_dets))
        n_rx      = max(len(rx_ids), 1)
        rx_palette = plt.cm.tab10(np.linspace(0, 0.9, n_rx))
        rx_color  = {rx_i: rx_palette[k] for k, rx_i in enumerate(rx_ids)}
        rx_label  = {rx_i: f"Rx-{rx_i + 1}" for rx_i in rx_ids}

        total        = 0
        fp_seen      = set()   # (t_idx, rx_i, tx_i) — deduplicate across constellations
        rx_fp_labeled = set()  # which Rx sats have a legend entry for the footprint
        rx_sc_labeled = set()  # which Rx sats have a legend entry for the scatter

        for gnss_name, dets in self.detections.items():
            if not dets:
                continue

            gnss_const = gnss_dict[gnss_name]

            # ── Footprints coloured by Rx satellite ───────────────────
            if footprints:
                for d in dets:
                    key = (d.t_idx, d.rx_sat_idx, d.tx_sat_idx)
                    if key in fp_seen:
                        continue
                    fp_seen.add(key)

                    col    = rx_color[d.rx_sat_idx]
                    rx_sat = sat_const.satellites[d.rx_sat_idx]
                    tx_sat = gnss_const.satellites[d.tx_sat_idx]
                    rx_eci = rx_sat.positions_eci[:, d.t_idx]
                    tx_eci = tx_sat.positions_eci[:, d.t_idx]
                    rx_vel = rx_sat.velocities_eci[:, d.t_idx]

                    if np.any(np.isnan(rx_eci)) or np.any(np.isnan(rx_vel)):
                        continue

                    fp_lats, fp_lons = self._steered_footprint_latlons(
                        rx_eci, rx_vel, tx_eci, timestamps[d.t_idx])

                    if not fp_lats:
                        continue
                    if (max(fp_lats) - min(fp_lats) < 2.0 and
                            max(fp_lons) - min(fp_lons) < 3.0):
                        continue

                    fp_lbl = (f"{rx_label[d.rx_sat_idx]} −3 dB beam"
                              if d.rx_sat_idx not in rx_fp_labeled else None)
                    ax.plot(fp_lons + [fp_lons[0]], fp_lats + [fp_lats[0]],
                            "-", color=col, lw=1.0, alpha=0.7,
                            transform=pc, zorder=4, label=fp_lbl)
                    rx_fp_labeled.add(d.rx_sat_idx)

            # ── Detection markers grouped by Rx satellite ─────────────
            from itertools import groupby
            dets_sorted = sorted(dets, key=lambda d: d.rx_sat_idx)
            for rx_i, group in groupby(dets_sorted, key=lambda d: d.rx_sat_idx):
                group = list(group)
                col   = rx_color[rx_i]
                counts = Counter((d.ship_lat, d.ship_lon) for d in group)
                lons_d = [ll[1] for ll in counts]
                lats_d = [ll[0] for ll in counts]
                sizes  = [40 + 8 * v for v in counts.values()]
                sc_lbl = (f"{rx_label[rx_i]}: {len(group)} events"
                          if rx_i not in rx_sc_labeled else None)
                ax.scatter(lons_d, lats_d, marker="*", c=[col], s=sizes,
                           zorder=8, transform=pc, alpha=0.95,
                           edgecolors="k", linewidths=0.4, label=sc_lbl)
                rx_sc_labeled.add(rx_i)
                total += len(group)

        bistatic_str = (f"  +  bistatic ≤ {self.max_bistatic_angle_deg}°"
                        if self.max_bistatic_angle_deg is not None else "")
        ax.set_title(
            f"Cell-centric moving target detection\n"
            f"Tx {TX_BEAM_HALF_ANGLE_DEG:.0f}° cone  +  "
            f"steered Rx {self.rx_beam_half_angle_deg:.1f}° beam  "
            f"(steer_fraction={self.steer_fraction})"
            f"{bistatic_str}  |  {total} total events",
            fontsize=11, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        plt.tight_layout()

        if file_name:
            path = os.path.join(Constants.output_folder_path, file_name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        if show:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    def show_timeline(self, file_name=None, show=False, time_bin_h=5.0):
        """
        Three-panel summary figure:
          Left  : detection events on the ship track (lat vs elapsed time),
                  coloured by Rx satellite
          Top-R : total events per Rx satellite (horizontal bar)
          Bot-R : detection density over time (stacked histogram, bin=time_bin_h)
        """
        all_dets = [d for dets in self.detections.values() for d in dets]
        if not all_dets:
            print("[show_timeline] No detections to plot.")
            return

        start_time = self.mtd.timestamps_resampled[0]

        # ── colour map (same rule as show_detections) ─────────────────
        rx_ids    = sorted(set(d.rx_sat_idx for d in all_dets))
        n_rx      = max(len(rx_ids), 1)
        rx_palette = plt.cm.tab10(np.linspace(0, 0.9, n_rx))
        rx_color  = {rx_i: rx_palette[k] for k, rx_i in enumerate(rx_ids)}
        rx_label  = {rx_i: f"Rx-{rx_i + 1}" for rx_i in rx_ids}

        # ── build per-detection arrays ─────────────────────────────────
        elapsed = np.array([
            (self.mtd.timestamps_resampled[d.t_idx] - start_time).total_seconds() / 3600
            for d in all_dets
        ])
        lats   = np.array([d.ship_lat for d in all_dets])
        rx_arr = np.array([d.rx_sat_idx for d in all_dets])

        # Full ship track for reference
        ship_times = np.array([
            (ts - start_time).total_seconds() / 3600
            for ts in self.mtd.timestamps_resampled
        ])
        ship_lats  = np.array([pt.y for pt in self.mtd.ship_positions_resampled])

        total_h   = ship_times[-1]
        n_bins    = max(1, int(np.ceil(total_h / time_bin_h)))
        bin_edges = np.linspace(0, total_h, n_bins + 1)

        bistatic_str = (f"  ·  bistatic ≤ {self.max_bistatic_angle_deg}°"
                        if self.max_bistatic_angle_deg is not None else "")
        suptitle = (f"Cell-centric MTD  ·  {len(all_dets)} events"
                    f"  ·  {n_rx} Rx sat{'s' if n_rx > 1 else ''}{bistatic_str}")

        fig = plt.figure(figsize=(16, 7))
        gs  = fig.add_gridspec(2, 2, width_ratios=[2.2, 1],
                               hspace=0.38, wspace=0.32)
        ax_tl = fig.add_subplot(gs[:, 0])   # timeline (full left column)
        ax_br = fig.add_subplot(gs[0, 1])   # events per Rx (top-right)
        ax_bl = fig.add_subplot(gs[1, 1])   # time histogram (bottom-right)

        # ── Panel 1: timeline ─────────────────────────────────────────
        ax_tl.plot(ship_times, ship_lats, "-", color="#aaaaaa",
                   lw=1.5, alpha=0.6, zorder=1, label="Ship track")
        for rx_i in rx_ids:
            mask = rx_arr == rx_i
            ax_tl.scatter(elapsed[mask], lats[mask],
                          color=rx_color[rx_i], s=60, zorder=3,
                          label=f"{rx_label[rx_i]} ({mask.sum()} events)",
                          edgecolors="k", linewidths=0.4, marker="*")
        ax_tl.set_xlabel("Elapsed time  [h]", fontsize=10)
        ax_tl.set_ylabel("Ship latitude  [°N]", fontsize=10)
        ax_tl.set_title("Detection events along track", fontsize=10)
        ax_tl.legend(fontsize=8, loc="upper left", framealpha=0.9,
                     ncol=max(1, n_rx // 6 + 1))
        ax_tl.grid(True, alpha=0.3)
        ax_tl.set_xlim(0, total_h)

        # ── Panel 2: events per Rx ────────────────────────────────────
        counts_rx = [np.sum(rx_arr == rx_i) for rx_i in rx_ids]
        bars = ax_br.barh(
            [rx_label[rx_i] for rx_i in rx_ids],
            counts_rx,
            color=[rx_color[rx_i] for rx_i in rx_ids],
            edgecolor="k", linewidth=0.5,
        )
        ax_br.bar_label(bars, padding=3, fontsize=8)
        ax_br.set_xlabel("Detection events", fontsize=9)
        ax_br.set_title("Events per Rx satellite", fontsize=10)
        ax_br.grid(axis="x", alpha=0.3)
        ax_br.invert_yaxis()

        # ── Panel 3: stacked time histogram ───────────────────────────
        hist_data = [elapsed[rx_arr == rx_i] for rx_i in rx_ids]
        ax_bl.hist(
            hist_data,
            bins=bin_edges,
            stacked=True,
            color=[rx_color[rx_i] for rx_i in rx_ids],
            label=[rx_label[rx_i] for rx_i in rx_ids],
            edgecolor="k", linewidth=0.4,
        )
        ax_bl.set_xlabel("Elapsed time  [h]", fontsize=9)
        ax_bl.set_ylabel("Events per bin", fontsize=9)
        ax_bl.set_title(f"Detection density  (bin = {time_bin_h:.0f} h)", fontsize=10)
        ax_bl.grid(axis="y", alpha=0.3)
        ax_bl.set_xlim(0, total_h)
        if n_rx > 1:
            ax_bl.legend(fontsize=7, ncol=max(1, n_rx // 4 + 1))

        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()

        if file_name:
            path = os.path.join(Constants.output_folder_path, file_name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        if show:
            plt.show()
        plt.close()
