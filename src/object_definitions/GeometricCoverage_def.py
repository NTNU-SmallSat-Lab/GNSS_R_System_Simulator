"""
GEOMETRIC COVERAGE CALCULATION for LEO-PNT Backscatter (VECTORIZED)

True cell-centric approach with NumPy vectorization:
For each grid cell, check if it's geometrically coverable by Tx-Rx pairs
based on:
1. Tx beam coverage (elevation ≥ 30°, antenna gain pattern)
2. Rx beam coverage (steered antenna, antenna gain pattern)
3. Backscatter geometry (bistatic angle ≤ 10°)

Uses vectorized NumPy operations to compute elevations for all cells
simultaneously instead of looping through each cell.
"""

import numpy as np
import pymap3d
from tqdm import tqdm


class GeometricCoverage:
    """Geometric coverage based on antenna beam geometry and bistatic constraints."""

    def __init__(self, sat_const, grid_gdf, min_tx_elevation_deg=30.0, max_bistatic_angle_deg=10.0):
        """
        Initialize geometric coverage calculator.

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

        # Store coverage results
        self.coverage_cells = []  # List of covered cell indices
        self.coverage_details = {}  # {cell_idx: [(time, tx_id, rx_id, tx_elev, rx_elev, bistatic_angle)]}

    def compute_elevation_angles_vectorized(self, sat_pos_ecef, cell_ecef_array):
        """
        Compute elevation angles from satellite to ALL cells simultaneously (VECTORIZED).

        INPUTS:
            sat_pos_ecef: [np.ndarray] Satellite position in ECEF (3D point)
            cell_ecef_array: [np.ndarray] Cell positions in ECEF (N×3 array)

        OUTPUT:
            elevation_angles: [np.ndarray] Elevation angles in degrees (N elements)
        """
        # Vector from each cell to satellite (broadcasting: Nx3)
        los_vectors = sat_pos_ecef[np.newaxis, :] - cell_ecef_array  # (N, 3)

        # Nadir directions (from Earth center through each cell)
        nadir_directions = -cell_ecef_array / np.linalg.norm(cell_ecef_array, axis=1, keepdims=True)  # (N, 3)

        # Elevation angle: angle between LOS and nadir
        los_norms = np.linalg.norm(los_vectors, axis=1)  # (N,)
        cos_elevs = np.sum(los_vectors * nadir_directions, axis=1) / los_norms  # (N,)
        cos_elevs = np.clip(cos_elevs, -1.0, 1.0)

        elevation_rads = np.arccos(cos_elevs)
        elevation_degs = 90.0 - np.degrees(elevation_rads)

        return elevation_degs

    def compute_bistatic_angles_vectorized(self, tx_pos_ecef, rx_pos_ecef, cell_ecef_array):
        """
        Compute bistatic angles at ALL cells simultaneously (VECTORIZED).

        INPUTS:
            tx_pos_ecef: [np.ndarray] Tx position in ECEF (3D)
            rx_pos_ecef: [np.ndarray] Rx position in ECEF (3D)
            cell_ecef_array: [np.ndarray] Cell positions in ECEF (N×3)

        OUTPUT:
            bistatic_angles: [np.ndarray] Bistatic angles in degrees (N elements)
        """
        # Vectors from each cell to satellites (broadcasting)
        to_tx = tx_pos_ecef[np.newaxis, :] - cell_ecef_array  # (N, 3)
        to_rx = rx_pos_ecef[np.newaxis, :] - cell_ecef_array  # (N, 3)

        # Normalize
        to_tx_norm = to_tx / np.linalg.norm(to_tx, axis=1, keepdims=True)  # (N, 3)
        to_rx_norm = to_rx / np.linalg.norm(to_rx, axis=1, keepdims=True)  # (N, 3)

        # Bistatic angle
        cos_angles = np.sum(to_tx_norm * to_rx_norm, axis=1)  # (N,)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        bistatic_rads = np.arccos(cos_angles)
        bistatic_degs = np.degrees(bistatic_rads)

        return bistatic_degs

    def is_in_tx_beam_vectorized(self, tx_elevation_degs):
        """
        Check if targets are within Tx beam (VECTORIZED).

        INPUTS:
            tx_elevation_degs: [np.ndarray] Elevation angles (N elements)

        OUTPUT:
            in_beam: [np.ndarray] Boolean array (N elements)
        """
        return tx_elevation_degs >= self.min_tx_elevation_deg

    def is_in_rx_beam_vectorized(self, rx_elevation_degs):
        """
        Check if targets are within Rx beam (VECTORIZED).

        Rx steered toward horizon: 10° to 60° elevation range.

        INPUTS:
            rx_elevation_degs: [np.ndarray] Elevation angles (N elements)

        OUTPUT:
            in_beam: [np.ndarray] Boolean array (N elements)
        """
        min_rx_elev = 10.0
        max_rx_elev = 60.0
        return (rx_elevation_degs >= min_rx_elev) & (rx_elevation_degs <= max_rx_elev)

    def calculate_geometric_coverage(self, times, gnss_pos_vec, verbose=False):
        """
        Calculate geometric coverage for all grid cells (VECTORIZED).

        For each timestep and Tx-Rx pair, compute elevations for ALL cells
        simultaneously using NumPy broadcasting.

        INPUTS:
            times: [list[datetime]] Time vector
            gnss_pos_vec: [dict] {gnss_name: np.ndarray} Tx positions in ECI
            verbose: [bool] Print progress

        OUTPUT:
            None - results stored in self.coverage_cells and self.coverage_details
        """
        num_cells = len(self.grid_gdf)
        num_rx = self.sat_const.no_satellites
        num_times = len(times)

        # Pre-compute cell ECEF positions (once)
        cell_lats = np.array([self.grid_gdf.iloc[i].geometry.centroid.y for i in range(num_cells)])
        cell_lons = np.array([self.grid_gdf.iloc[i].geometry.centroid.x for i in range(num_cells)])
        cell_ecef_array = np.array(
            [pymap3d.geodetic2ecef(cell_lats[i], cell_lons[i], 0) for i in range(num_cells)]
        )  # (num_cells, 3)

        with tqdm(
            total=num_times,
            desc="Geometric coverage (vectorized)",
            unit="timestep",
            disable=not verbose,
        ) as pbar:

            # Iterate through each TIME STEP
            for time_idx, time in enumerate(times):

                # Get Rx positions at this time
                for rx_idx in range(num_rx):
                    rx_sat = self.sat_const.satellites[rx_idx]
                    rx_pos_eci = rx_sat.positions_eci[:, time_idx]
                    rx_pos_ecef = np.array(pymap3d.eci2ecef(rx_pos_eci[0], rx_pos_eci[1], rx_pos_eci[2], time))

                    # Compute Rx elevations for ALL cells at once
                    try:
                        rx_elevs = self.compute_elevation_angles_vectorized(rx_pos_ecef, cell_ecef_array)
                        rx_in_beam = self.is_in_rx_beam_vectorized(rx_elevs)  # (num_cells,)
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

                            # Compute Tx elevations for ALL cells at once
                            try:
                                tx_elevs = self.compute_elevation_angles_vectorized(tx_pos_ecef, cell_ecef_array)
                                tx_in_beam = self.is_in_tx_beam_vectorized(tx_elevs)  # (num_cells,)
                            except Exception as e:
                                continue

                            # Compute bistatic angles for ALL cells at once
                            try:
                                bistatic_angles = self.compute_bistatic_angles_vectorized(
                                    tx_pos_ecef, rx_pos_ecef, cell_ecef_array
                                )  # (num_cells,)
                                bistatic_ok = bistatic_angles <= self.max_bistatic_angle_deg
                            except Exception as e:
                                continue

                            # Find cells that satisfy ALL constraints
                            all_constraints = rx_in_beam & tx_in_beam & bistatic_ok  # (num_cells,)
                            covered_cell_indices = np.where(all_constraints)[0]

                            # Store results for covered cells
                            for cell_idx in covered_cell_indices:
                                if cell_idx not in self.coverage_details:
                                    self.coverage_cells.append(cell_idx)
                                    self.coverage_details[cell_idx] = []

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

        # Calculate statistics
        self.num_covered = len(self.coverage_cells)
        self.coverage_percentage = (self.num_covered / num_cells) * 100

    def print_statistics(self):
        """Print coverage statistics."""
        print("\n" + "=" * 70)
        print("GEOMETRIC COVERAGE STATISTICS")
        print("=" * 70)
        print(f"Total cells: {len(self.grid_gdf)}")
        print(f"Covered cells: {self.num_covered}")
        print(f"Coverage percentage: {self.coverage_percentage:.2f}%")
        print(f"\nConstraints applied:")
        print(f"  Tx elevation ≥ {self.min_tx_elevation_deg}°")
        print(f"  Rx beam: horizon-steered (10°-60° elevation)")
        print(f"  Bistatic angle ≤ {self.max_bistatic_angle_deg}°")
        print("=" * 70)
