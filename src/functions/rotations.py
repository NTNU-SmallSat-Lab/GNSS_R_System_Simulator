import numpy as np


def quat_to_rotation_matrix(q: list) -> np.ndarray:
    """
    =========================================================================================================
    Convert quaternion ``[w, x, y, z]`` to a 3x3 rotation matrix.

    INPUTS:
        q: [list]   Quaternion as [w, x, y, z]

    OUTPUT:
        R: [np.ndarray, shape (3, 3)]   Rotation matrix
    =========================================================================================================
    """
    w, x, y, z = q
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )
