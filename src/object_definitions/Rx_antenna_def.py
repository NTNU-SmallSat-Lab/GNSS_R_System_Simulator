import numpy as np
from scipy.interpolate import splev, splprep


class RxAntenna:
    def __init__(self, settings: dict = None):
        """
        Initialize the RxAntenna object.

        INPUTS:
            - antenna_beamwidth: [list] Beam angles in degrees for different support angles
            - antenna_gain: [float] Antenna gain in dB
            - steering_angle_deg: [float] Steering angle in degrees
            - polarizations: [list] Antenna polarizations (e.g., ["LHCP", "RHCP"])
            - bands: [list] Antenna bands
        """
        if settings is None:
            raise ValueError("Settings dictionary must be provided.")
        self.beam_angle_deg = settings.get("antenna_beamwidth")  # Antenna gain in dB
        self.antenna_gain = settings.get("antenna_gain")  # Antenna gain in dB
        self.steering_angle_deg = settings.get("steering_angle_deg")  # Steering angle in degrees
        self.polarizations = settings.get("polarizations") # Antenna polarizations (e.g., ["LHCP", "RHCP"])
        self.bands = settings.get("bands")  # Antenna bands (e.g., ["L1", "L2", "L3", "L5", "L6", "E1", "E5a", "E5b", "E6", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"])
        self.footprint_resolution = 64  # Number of points to create a circle (higher resolution for smoother circles) | hardcoded to 64
        self.R_antenna_to_body = np.array([ # Rotation matrix from antenna frame to body-fixed frame
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])

        self._initialize()

    def _initialize(self):
        """
        Initialize the antenna properties.
        """
        # Calculate "footprint" of the antenna at 1m distance using splines
        self.antenna_footprint_a = self._calc_antenna_footprint()

    def _calc_antenna_footprint(self):
        """
        Calculate the antenna footprint with vectors radiating from the antenna position outwards in the antenna frame
        """
        # Calculate angular positions of the antenna beam
        no_of_beamwidth_entries = len(self.beam_angle_deg)
        if no_of_beamwidth_entries < 2:
            raise ValueError("Antenna beamwidth must have at least two entries for the footprint calculation.")
        if no_of_beamwidth_entries > self.footprint_resolution:
            raise ValueError(f"Antenna beamwidth entries exceed the footprint resolution of {self.footprint_resolution}. Please adjust the settings.")
        # First entry = 0deg last entry = 90deg (points are mirrored on x-axis and y-axis)
        spline_point_support_angles = np.linspace(0, 90, no_of_beamwidth_entries)

        antenna_footprint_support_points = np.zeros((4*(no_of_beamwidth_entries - 1) + 1, 2)) # Add 1 point to make it a closed loop
        for i, spline_sup_angle_i in enumerate(spline_point_support_angles):
            # Calculate the x and y coordinates of the antenna footprint at 1m distance
            dist_from_origin = np.tan(np.radians(self.beam_angle_deg[i]))  # Distance from the origin in meters
            y = np.cos(np.radians(spline_sup_angle_i)) * dist_from_origin
            x = np.sin(np.radians(spline_sup_angle_i)) * dist_from_origin
            # 1st quadrant points
            antenna_footprint_support_points[i] = (x, y)

            if i == 0:
                # Mirror on the x-axis only
                antenna_footprint_support_points[2*no_of_beamwidth_entries - 2] = (x, -y)
            elif i == no_of_beamwidth_entries - 1:
                # Mirror on the y-axis only
                antenna_footprint_support_points[3*no_of_beamwidth_entries - 3] = (-x, y)
            else:
                # Mirrored points for the 2nd quadrant (x positive, y negative)
                antenna_footprint_support_points[2*(no_of_beamwidth_entries - 1) - i] = (x, -y)
                # Mirrored points for the third quadrant (x negative, y negative)
                antenna_footprint_support_points[2*(no_of_beamwidth_entries - 1) + i] = (-x, -y)
                # Mirrored points for the fourth quadrant (x positive, y negative)
                antenna_footprint_support_points[4*(no_of_beamwidth_entries - 1) - i] = (-x, y)
        # Add the first point again to close the loop
        antenna_footprint_support_points[-1] = antenna_footprint_support_points[0]

        # Create a spline representation of the antenna footprint
        tck, u = splprep(antenna_footprint_support_points.T, s=0, per=True)

        # Generate smooth curve (antenna footprint) using the spline representation
        u_fine = np.linspace(0, 1, self.footprint_resolution)  # High resolution
        smooth_x, smooth_y = splev(u_fine, tck)

        # Create "unit-vectors" radiating from the antenna position outwards in the antenna frame
        antenna_footprint_vectors = np.zeros((self.footprint_resolution, 3))
        for i in range(self.footprint_resolution):
            # Calculate the direction of the vector
            direction = np.array([smooth_x[i], smooth_y[i], 1.0])
            # Normalize the vector to create a unit vector
            antenna_footprint_vectors[i] = direction / np.linalg.norm(direction)

        self.antenna_footprint_vectors = antenna_footprint_vectors

        ##DEBUG Visualize the antenna footprint in 3D
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.quiver(0, 0, 0, self.antenna_footprint_vectors[:, 0], self.antenna_footprint_vectors[:, 1], self.antenna_footprint_vectors[:, 2])
        # plt.show(block=True)
        ## DEBUG SHOW THE SPLINE 1m IN FRONT OF THE ANTENNA
        # fig, ax = plt.subplots()
        # ax.plot(smooth_x, smooth_y, 'b-', label='Antenna Footprint')
        # ax.scatter(antenna_footprint_support_points[:, 0], antenna_footprint_support_points[:, 1], c='r', label='Support Points')
        # ax.set_xlabel('X (m)')
        # ax.set_ylabel('Y (m)')
        # ax.set_title('Antenna Footprint')
        # ax.legend()
        # plt.show(block=True)

    def _get_antenna_footprint(self, refl_dir_antenna_frame):
        """
        Get the antenna footprint in the antenna coordinate system.
        """
        nadir = np.array([0, 0, 1])  # Nadir pointing vector in antenna coordinate system
        refl_dir_antenna_frame = np.asarray(refl_dir_antenna_frame)
        if refl_dir_antenna_frame.shape != (3,):
            raise ValueError("Input 'refl_dir_antenna_frame' must be a 3-element vector.")

        # Check if the direction is the same as nadir (no rotation needed)
        if np.allclose(refl_dir_antenna_frame, nadir):
            return self.antenna_footprint_vectors

        # Compute rotation axis and angle
        axis = np.cross(nadir, refl_dir_antenna_frame)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            # refl_dir_antenna_frame is opposite to nadir, rotate 180 degrees around any perpendicular axis
            axis = np.array([1, 0, 0])  # Arbitrary perpendicular axis
            angle = np.pi
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(nadir, refl_dir_antenna_frame), -1.0, 1.0))

        # Compute rotation matrix using Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
#        R_nadir_to_reflection = fn.rotation_matrix_from_vectors(n_an_a, n_ar_a)# Nadir pointing vector in antenna coordinate system
        # Rotate all footprint vectors
#        rotated_footprint_vectors = (R @ self.antenna_footprint_vectors.T).T
        rotated_footprint_vectors = self.antenna_footprint_vectors @ R.T  # Rotate the footprint vectors to the reflection beam coordinate system

        return rotated_footprint_vectors
