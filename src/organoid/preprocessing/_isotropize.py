from scipy.ndimage import zoom
import numpy as np
from typing import Tuple



def _make_array_isotropic(
        mask: np.ndarray = None, 
        image: np.ndarray = None,
        labels: np.ndarray = None,
        input_pixelsize: Tuple[float, float, float] = (1, 1, 1),
        output_pixelsize: Tuple[float, float, float] = (1, 1, 1),
        order: int = 1
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
    
    if mask is not None:
        mask_isotropic = zoom(mask, reshape_factors, order=0)
    if image is not None:
        image_isotropic = zoom(image, reshape_factors, order=order)
    if labels is not None:
        labels_isotropic = zoom(labels, reshape_factors, order=0)

    if mask is not None and image is not None and labels is not None:
        return mask_isotropic, image_isotropic, labels_isotropic
    elif mask is not None and image is not None:
        return mask_isotropic, image_isotropic
    elif mask is not None and labels is not None:
        return mask_isotropic, labels_isotropic
    elif image is not None and labels is not None:
        return image_isotropic, labels_isotropic
    elif mask is not None:
        return mask_isotropic
    elif image is not None:
        return image_isotropic
    elif labels is not None:
        return labels_isotropic