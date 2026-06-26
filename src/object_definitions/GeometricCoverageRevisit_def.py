"""
GEOMETRIC COVERAGE WITH REVISIT TIME ANALYSIS

Tracks:
1. Number of times each cell is geometrically illuminated (passes)
2. Revisit time (time between successive passes)
3. Pass timing statistics (first pass, last pass, pass intervals)
"""

import numpy as np
import pymap3d
from tqdm import tqdm


class GeometricCoverageRevisit:
    """Geometric coverage with revisit time analysis."""

    def __init__(self, sat_const, grid_gdf, min_tx_elevation_deg=30.0, max_bistatic_angle_deg=10.0):
        """
        Initialize geometric coverage with revisit tracking.

        INPUTS:
            sat_const: [SatelliteConstellation] Rx satellite constellation
            grid_gdf: [GeoDataFrame] Grid cells with geometry
            min_tx_elevation_deg: [float] Minimum Tx elevation (default 30°)
            max_bistatic_angle_deg: [float] Maximum bistatic angle (default 10°)
        """
        self.sat_const = sat_const
        self.grid_gdf = grid_gdf
        self.min_tx_elevation_deg = min_tx_elevation_deg
        self.max_bistatic_angle_deg = max_bistatic_angle_deg

        # Store results per cell
        self.pass_count = {}  # {cell_idx: number of passes}
        self.pass_times = {}  # {cell_idx: [times of illumination]}
        self.revisit_times = {}  # {cell_idx: [intervals between passes]}
        self.coverage_details = {}  # {cell_idx: [(time, tx_id, rx_id, tx_elev, rx_elev, bistatic_angle)]}

    def compute_elevation_angles_vectorized(self, sat_pos_ecef, cell_ecef_array):
        """Compute elevation angles from satellite to ALL cells simultaneously (VECTORIZED)."""
        los_vectors = sat_pos_ecef[np.newaxis, :] - cell_ecef_array
        nadir_directions = -cell_ecef_array / np.linalg.norm(cell_ecef_array, axis=1, keepdims=True)
        los_norms = np.linalg.norm(los_vectors, axis=1)
        cos_elevs = np.sum(los_vectors * nadir_directions, axis=1) / los_norms
        cos_elevs = np.clip(cos_elevs, -1.0, 1.0)
        elevation_rads = np.arccos(cos_elevs)
        elevation_degs = 90.0 - np.degrees(elevation_rads)
        return elevation_degs

    def compute_bistatic_angles_vectorized(self, tx_pos_ecef, rx_pos_ecef, cell_ecef_array):
        """Compute bistatic angles at ALL cells simultaneously (VECTORIZED)."""
        to_tx = tx_pos_ecef[np.newaxis, :] - cell_ecef_array
        to_rx = rx_pos_ecef[np.newaxis, :] - cell_ecef_array
        to_tx_norm = to_tx / np.linalg.norm(to_tx, axis=1, keepdims=True)
        to_rx_norm = to_rx / np.linalg.norm(to_rx, axis=1, keepdims=True)
        cos_angles = np.sum(to_tx_norm * to_rx_norm, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        bistatic_rads = np.arccos(cos_angles)
        bistatic_degs = np.degrees(bistatic_rads)
        return bistatic_degs

    def is_in_tx_beam_vectorized(self, tx_elevation_degs):
        """Check if targets are within Tx beam (VECTORIZED)."""
        return tx_elevation_degs >= self.min_tx_elevation_deg

    def is_in_rx_beam_vectorized(self, rx_elevation_degs):
        """Check if targets are within Rx beam (VECTORIZED)."""
        min_rx_elev = 10.0
        max_rx_elev = 60.0
        return (rx_elevation_degs >= min_rx_elev) & (rx_elevation_degs <= max_rx_elev)

    def calculate_geometric_coverage_with_revisit(self, times, gnss_pos_vec, verbose=False):
        """
        Calculate geometric coverage with revisit time tracking.

        For each timestep and Tx-Rx pair, compute elevations for ALL cells
        simultaneously using NumPy broadcasting, and track pass times.

        INPUTS:
            times: [list[datetime]] Time vector
            gnss_pos_vec: [dict] {gnss_name: np.ndarray} Tx positions in ECI
            verbose: [bool] Print progress

        OUTPUT:
            None - results stored in pass_count, pass_times, revisit_times
        """
        num_cells = len(self.grid_gdf)
        num_rx = self.sat_const.no_satellites
        num_times = len(times)

        # Pre-compute cell ECEF positions
        cell_lats = np.array([self.grid_gdf.iloc[i].geometry.centroid.y for i in range(num_cells)])
        cell_lons = np.array([self.grid_gdf.iloc[i].geometry.centroid.x for i in range(num_cells)])
        cell_ecef_array = np.array(
            [pymap3d.geodetic2ecef(cell_lats[i], cell_lons[i], 0) for i in range(num_cells)]
        )

        # Initialize tracking
        for cell_idx in range(num_cells):
            self.pass_times[cell_idx] = []
            self.coverage_details[cell_idx] = []

        with tqdm(
            total=num_times,
            desc="Geometric coverage with revisit",
            unit="timestep",
            disable=not verbose,
        ) as pbar:

            for time_idx, time in enumerate(times):

                # Get Rx positions at this time
                for rx_idx in range(num_rx):
                    rx_sat = self.sat_const.satellites[rx_idx]
                    rx_pos_eci = rx_sat.positions_eci[:, time_idx]
                    rx_pos_ecef = np.array(pymap3d.eci2ecef(rx_pos_eci[0], rx_pos_eci[1], rx_pos_eci[2], time))

                    # Compute Rx elevations for ALL cells
                    try:
                        rx_elevs = self.compute_elevation_angles_vectorized(rx_pos_ecef, cell_ecef_array)
                        rx_in_beam = self.is_in_rx_beam_vectorized(rx_elevs)
                    except Exception as e:
                        pbar.update(1)
                        continue

                    # Get Tx positions at this time
                    for gnss_name, tx_positions in gnss_pos_vec.items():
                        num_tx = tx_positions.shape[1]

                        for tx_idx in range(num_tx):
                            tx_pos_eci = tx_positions[:, tx_idx, time_idx]
                            tx_pos_ecef = np.array(
                                pymap3d.eci2ecef(tx_pos_eci[0], tx_pos_eci[1], tx_pos_eci[2], time)
                            )

                            # Compute Tx elevations for ALL cells
                            try:
                                tx_elevs = self.compute_elevation_angles_vectorized(tx_pos_ecef, cell_ecef_array)
                                tx_in_beam = self.is_in_tx_beam_vectorized(tx_elevs)
                            except Exception as e:
                                continue

                            # Compute bistatic angles for ALL cells
                            try:
                                bistatic_angles = self.compute_bistatic_angles_vectorized(
                                    tx_pos_ecef, rx_pos_ecef, cell_ecef_array
                                )
                                bistatic_ok = bistatic_angles <= self.max_bistatic_angle_deg
                            except Exception as e:
                                continue

                            # Find cells that satisfy ALL constraints
                            all_constraints = rx_in_beam & tx_in_beam & bistatic_ok
                            covered_cell_indices = np.where(all_constraints)[0]

                            # Track pass times and details
                            for cell_idx in covered_cell_indices:
                                self.pass_times[cell_idx].append(time)
                                self.coverage_details[cell_idx].append(
                                    (
                                        time,
                                        gnss_name,
                                        tx_idx,
                                        rx_idx,
                                        tx_elevs[cell_idx],
                                        rx_elevs[cell_idx],
                                        bistatic_angles[cell_idx],
                                    )
                                )

                pbar.update(1)

        # Calculate pass counts and revisit times
        self._calculate_revisit_statistics()

    def _calculate_revisit_statistics(self):
        """
        Calculate pass counts and revisit times from pass_times.

        A "pass" is a continuous period of illumination.
        Revisit time is the gap between the end of one pass and the start of the next.
        """
        for cell_idx in range(len(self.grid_gdf)):
            times = self.pass_times[cell_idx]

            if not times:
                self.pass_count[cell_idx] = 0
                self.revisit_times[cell_idx] = []
                continue

            # Sort times to identify continuous passes
            times_sorted = sorted(times)

            # Group consecutive timesteps into passes
            # A pass ends when there's a gap > 75 minutes (orbital period ~96 min, illumination ~10 min)
            # Gap between actual passes: ~86 minutes, so 75 min threshold splits them while grouping timesteps
            timestep_seconds = 30  # From config: step_size = 30 seconds
            max_gap_within_pass = 75 * 60  # 75 minutes - gaps larger than this separate passes
            passes = []  # List of (pass_start_time, pass_end_time)
            current_pass_start = times_sorted[0]
            current_pass_end = times_sorted[0]

            for i in range(1, len(times_sorted)):
                time_diff = (times_sorted[i] - times_sorted[i-1]).total_seconds()

                if time_diff <= max_gap_within_pass:
                    # Continue current pass (allow gaps up to 5 minutes due to elevation constraints)
                    current_pass_end = times_sorted[i]
                else:
                    # Large gap detected - end current pass and start new one
                    passes.append((current_pass_start, current_pass_end))
                    current_pass_start = times_sorted[i]
                    current_pass_end = times_sorted[i]

            # Don't forget the last pass
            passes.append((current_pass_start, current_pass_end))

            self.pass_count[cell_idx] = len(passes)

            # Calculate revisit times (gaps between passes)
            if len(passes) > 1:
                revisit_intervals = []
                for i in range(1, len(passes)):
                    gap = (passes[i][0] - passes[i-1][1]).total_seconds()
                    revisit_intervals.append(gap)
                self.revisit_times[cell_idx] = np.array(revisit_intervals)
            else:
                self.revisit_times[cell_idx] = np.array([])

    def print_statistics(self):
        """Print pass count statistics."""
        print("\n" + "=" * 70)
        print("GEOMETRIC COVERAGE - PASS COUNT ANALYSIS")
        print("=" * 70)

        pass_counts = list(self.pass_count.values())
        covered_cells = sum(1 for c in pass_counts if c > 0)

        print(f"\nCoverage Statistics:")
        print(f"  Total cells: {len(self.grid_gdf)}")
        print(f"  Covered cells (≥1 pass): {covered_cells}")
        print(f"  Coverage %: {100 * covered_cells / len(self.grid_gdf):.2f}%")

        print(f"\nPass Count per Cell (24-hour simulation):")
        valid_counts = [c for c in pass_counts if c > 0]
        if valid_counts:
            print(f"  Min passes: {min(valid_counts)}")
            print(f"  Max passes: {max(valid_counts)}")
            print(f"  Mean passes: {np.mean(valid_counts):.2f}")
            print(f"  Median passes: {np.median(valid_counts):.1f}")
            print(f"  Std dev: {np.std(valid_counts):.2f}")

            # Histogram of pass counts
            unique_counts = sorted(set(valid_counts))
            print(f"\n  Pass count distribution:")
            for count in unique_counts:
                num_cells = sum(1 for c in valid_counts if c == count)
                pct = 100 * num_cells / len(valid_counts)
                print(f"    {count:3d} pass(es): {num_cells:4d} cells ({pct:5.1f}%)")

        print("\n" + "=" * 70)
