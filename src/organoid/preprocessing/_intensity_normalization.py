import numpy as np
from organoid.preprocessing._smoothing import _smooth_gaussian
from scipy.optimize import minimize_scalar



def _median_absolute_deviation(array):
    return np.nanmedian(np.abs(array - np.nanmedian(array)))


def _nans_outside_mask(array, mask):
    return np.where(mask, array, np.nan)


def _optimize_sigma(ref_array, mask, labels_mask):

    def opt_func(sigma, labels_mask, mask):
        ref_array_smooth = _smooth_gaussian(
            ref_array, 
            sigma=sigma,
            mask_for_volume=labels_mask, 
            mask=mask
        )
        return _median_absolute_deviation(np.nanmedian(ref_array_smooth, axis=(1,2)))
    
    res = minimize_scalar(opt_func, args=(labels_mask, mask), bounds=(10, 30), method='bounded')

    return res.x

def _find_reference_plane_from_medians(array):
    ref_ind = np.nanargmax(np.nanmedian(array, axis=(1,2)))
    return ref_ind
   

def _normalize_intensity(array, ref_array, sigma=None, mask=None, labels=None, width=3):
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
    - width (int, optional): The number of neighboring planes to consider for reference plane calculation. Default is 5.

    Returns:
    - array_norm (ndarray): The normalized input array.
    - ref_array_norm (ndarray): The normalized reference array.
    """
    
    num_z_slices = array.shape[0]
    
    if labels is None:
        labels_mask = None
    else:
        labels_mask = labels.astype(bool)
    
    # compute smoothed reference array for normalization
    if sigma is None:
        sigma = _optimize_sigma(ref_array, mask, labels_mask)

    ref_array_smooth = _smooth_gaussian(
        ref_array, 
        sigma=sigma,
        mask_for_volume=labels_mask, 
        mask=mask
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

    # rectify median intensity in both normalized arrays
    # to that of the median of the brightest neighboring planes
    z_ref = _find_reference_plane_from_medians(ref_array)
    z_ref_norm = _find_reference_plane_from_medians(ref_array_norm)

    sl = slice(max(0, z_ref-width), min(num_z_slices, z_ref+width))
    sl_norm = slice(max(0, z_ref_norm-width), min(num_z_slices, z_ref_norm+width))

    ref_array_normalization_factor = np.nanmedian(ref_array_norm[sl_norm]) / np.nanmedian(ref_array[sl])
    array_normalization_factor = np.nanmedian(array_norm[sl_norm]) / np.nanmedian(array[sl])

    ref_array_norm = ref_array_norm / ref_array_normalization_factor
    array_norm = array_norm / array_normalization_factor
    
    return(array_norm, ref_array_norm)