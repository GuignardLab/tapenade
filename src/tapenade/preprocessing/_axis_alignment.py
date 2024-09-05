import numpy as np
from sklearn.decomposition import PCA


def _signed_angle(v1: np.ndarray, v2: np.ndarray, vn: np.ndarray) -> float:
    """
    Calculate the signed angle between two vectors in 3D space.

    Parameters:
    - v1: numpy array, first vector
    - v2: numpy array, second vector
    - vn: numpy array, normal vector

    Returns:
    - angle: float, signed angle between the two vectors
    """

    angle = np.arctan2(np.dot(vn, np.cross(v1, v2)), np.dot(v1, v2))

    return angle


def _compute_rotation_angle_and_indices(
    mask_for_pca: np.ndarray,
    target_axis: str,
    rotation_plane: str,
) -> tuple[float, tuple[int, int]]:
    """
    Compute the rotation angle and indices for axis alignment.

    Parameters:
    - mask_for_pca: numpy array, the mask array.
    - target_axis: str, the target axis for alignment ('X', 'Y', or 'Z').
    - rotation_plane: str, the rotation plane for alignment ('XY', 'XZ', or 'YZ').
    - temporal_slice: slice, the temporal slice to use for alignment.

    Returns:
    - rotation_angle: float, the rotation angle in degrees.
    - rotation_plane_indices: tuple, the indices of the rotation plane.

    """

    # Perform PCA on the mask to determine the major axis
    pca = PCA(n_components=3)
    pca.fit(np.argwhere(mask_for_pca))
    major_axis_vector = pca.components_[0]

    plane_normal_vector = {
        "XY": np.array([1, 0, 0]),
        "XZ": np.array([0, 1, 0]),
        "YZ": np.array([0, 0, 1]),
    }[rotation_plane]

    # Remove the component of the major axis vector that is perpendicular to the rotation plane
    major_axis_vector -= (
        np.dot(major_axis_vector, plane_normal_vector) * plane_normal_vector
    )
    major_axis_vector = major_axis_vector / np.linalg.norm(major_axis_vector)
    major_axis_vector *= np.sign(np.max(major_axis_vector))

    target_axis_vector = {
        "Z": np.array([1, 0, 0]),
        "Y": np.array([0, 1, 0]),
        "X": np.array([0, 0, 1]),
    }[target_axis]

    # Calculate the rotation angle between the major axis and the target axis
    rotation_angle = (
        _signed_angle(
            major_axis_vector, target_axis_vector, plane_normal_vector
        )
        * 180
        / np.pi
    )

    # respects the right-hand rule wrt the corresponding normal vector
    rotation_plane_indices = {
        "XY": (-1, -2),
        "XZ": (-3, -1),
        "YZ": (-2, -3),
    }[rotation_plane]

    return rotation_angle, rotation_plane_indices
