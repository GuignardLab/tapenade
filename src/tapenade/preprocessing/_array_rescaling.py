import numpy as np
from scipy.ndimage import zoom


def _change_array_pixelsize(
    array: np.ndarray,
    input_pixelsize: tuple[float, float, float] = (1, 1, 1),
    output_pixelsize: tuple[float, float, float] = (1, 1, 1),
    order: int = 1,
):
    """
    Make the input array isotropic by resampling it to the target spacing.

    Parameters:
    - mask (ndarray): The mask indicating the valid region.
    - image (ndarray): The image to be resampled.
    - labels (ndarray): The segmentation labels.
    - input_pixelsize (tuple): The input pixel size (e.g in microns).
    - output_pixelsize (tuple): The output pixel size (e.g in microns).
    - order (int): The order of the spline interpolation.

    Returns:
        ndarray: The resampled mask, image, or labels.
    """

    reshape_factors = np.array(input_pixelsize) / np.array(output_pixelsize)

    array_isotropic = zoom(array, reshape_factors, order=order)

    return array_isotropic
