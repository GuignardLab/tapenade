import numpy as np
from scipy.optimize import minimize_scalar

from tapenade.preprocessing._smoothing import _masked_smooth_gaussian


def _median_absolute_deviation(array: np.ndarray) -> float:
    return np.nanmedian(np.abs(array - np.nanmedian(array)))


def _nans_outside_mask(array: np.ndarray, mask: np.ndarray):
    return np.where(mask, array, np.nan)


def _optimize_sigma(
    ref_array: np.ndarray, mask: np.ndarray, labels_mask: np.ndarray
):

    def opt_func(
        sigma, ref_array: np.ndarray, mask: np.ndarray, labels_mask: np.ndarray
    ):
        ref_array_smooth = _masked_smooth_gaussian(
            ref_array, sigmas=sigma, mask_for_volume=labels_mask, mask=mask
        )

        if labels_mask is not None:
            res = np.divide(ref_array, ref_array_smooth, where=labels_mask)
            res = _nans_outside_mask(res, labels_mask)
        elif mask is not None:
            res = np.divide(ref_array, ref_array_smooth, where=mask)
            res = _nans_outside_mask(res, mask)
        else:
            res = ref_array / ref_array_smooth

        return _median_absolute_deviation(np.nanmedian(res, axis=(1, 2)))

    res = minimize_scalar(
        opt_func,
        args=(ref_array, mask, labels_mask),
        bounds=(10, 30),
        method="bounded",
        options={"maxiter": 5, "disp": 3},
    )

    return res.x


def _find_reference_plane_from_medians(array: np.ndarray):
    ref_ind = np.nanargmax(np.nanmedian(array, axis=(1, 2)))
    return ref_ind


def _normalize_intensity(
    array: np.ndarray,
    ref_array: np.ndarray,
    sigma=None,
    mask: np.ndarray = None,
    labels: np.ndarray = None,
    image_wavelength: float = None,
    width=3,
):
    """
    Normalize the intensity of an array based on a reference array assumed to have
    ideally homogeneous signal (e.g DAPI).

    Parameters:
    - array (ndarray): The input array to be normalized.
    - ref_array (ndarray): The reference array used for normalization.
    - sigma (float, optional): The standard deviation for Gaussian smoothing of the reference array. Default is None,
        but a value is required right now since setting 'sigma' to None is not implemented.
    - mask (ndarray, optional): A binary mask of the sample. Default is None.
    - labels (ndarray, optional): An array of labels indicating the instances in which the reference
        signal is expressed, e.g nuclei. Default is None.
    - image_wavelength (float, optional): The wavelength of the image. Default is None.
    - width (int, optional): The number of neighboring planes to consider for reference plane calculation. Default is 5.

    Returns:
    - array_norm (ndarray): The normalized input array.
    - ref_array_norm (ndarray): The normalized reference array.
    """

    num_z_slices = array.shape[0]

    labels_mask = None if labels is None else labels.astype(bool)

    # apply wavelength correction
    if image_wavelength is not None:

        # this is a temporary solution while no continuous fit is implemented
        coeff_dict = {
            405: 1.0,
            488: 0.82,
            555: 0.51,
            647: 0.47,
        }

        if image_wavelength not in coeff_dict:
            raise ValueError(
                f"Image wavelength {image_wavelength} not supported."
            )
        
        exponentiation_coeff = coeff_dict[image_wavelength]
        
        ref_array = ref_array ** exponentiation_coeff

    # compute smoothed reference array for normalization
    if sigma is None:
        sigma = _optimize_sigma(ref_array, mask, labels_mask)
        print("sigma = ", sigma)
    ref_array_smooth = _masked_smooth_gaussian(
        ref_array, sigmas=sigma, mask_for_volume=labels_mask, mask=mask
    )

    if mask is not None:
        array = _nans_outside_mask(array, mask)
        ref_array = _nans_outside_mask(ref_array, mask)
        ref_array_smooth = _nans_outside_mask(ref_array_smooth, mask)

        mask_divide = mask
    else:
        mask_divide = True

    # normalize array and reference array
    array_norm = np.divide(array, ref_array_smooth, where=mask_divide)
    ref_array_norm = np.divide(ref_array, ref_array_smooth, where=mask_divide)

    if mask is not None:
        array_norm = _nans_outside_mask(array_norm, mask)
        ref_array_norm = _nans_outside_mask(ref_array_norm, mask)

    # rectify median intensity in both normalized arrays
    # to that of the median of the brightest consecutive planes
    z_ref = _find_reference_plane_from_medians(ref_array)
    z_ref_norm = _find_reference_plane_from_medians(ref_array_norm)

    sl = slice(max(0, z_ref - width), min(num_z_slices, z_ref + width))
    sl_norm = slice(
        max(0, z_ref_norm - width), min(num_z_slices, z_ref_norm + width)
    )

    array_normalization_factor = np.nanmedian(
        array_norm[sl_norm]
    ) / np.nanmedian(array[sl])

    array_norm = array_norm / array_normalization_factor

    return array_norm
