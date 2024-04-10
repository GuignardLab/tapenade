import scipy.ndimage as ndi
import numpy as np
from scipy.stats import linregress
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter as scipy_gaussian


def change_voxelsize(
    array,
    input_vs: tuple = (1, 1, 1),
    output_vs: tuple = (1, 1, 1),
    order: int = 1,
):
    """
    Rescale an array to a new voxel size.
    :param array: The array to rescale.
    :param input_voxelsize: The voxel size of the input array.
    :param output_voxelsize: The voxel size of the output array.
    :param order: The order of the spline interpolation, default is 1.
    :return: The rescaled array.
    """
    assert len(input_vs) == len(output_vs)
    zoom_factor = tuple(
        [input_vs[i] / output_vs[i] for i in range(len(input_vs))]
    )
    return ndi.zoom(array, zoom=zoom_factor, order=order)



def find_nearest_index(array, values):
    array = np.asarray(array)
    # the last dim must be 1 to broadcast in (array - values) below.
    values = np.expand_dims(values, axis=-1) 
    indices = np.abs(array - values).argmin(axis=-1)

    return indices

def filter_percentiles(X, percentilesX: tuple = (1, 99), Y = None, percentilesY: tuple = None):

    if Y is None:

        down, up = percentilesX

        percentile_down = np.percentile(X, down)
        percentile_up = np.percentile(X, up)

        mask = np.logical_and(X>percentile_down, X<percentile_up)

        return X[mask]

    else:

        downX, upX = percentilesX

        if percentilesY is None:
            downY, upY = percentilesX
        else:
            downY, upY = percentilesY

        percentile_downX = np.percentile(X, downX)
        percentile_downY = np.percentile(Y, downY)

        percentile_upX = np.percentile(X, upX)
        percentile_upY = np.percentile(Y, upY)

        maskX = np.logical_and(X>percentile_downX, X<percentile_upX)
        maskY = np.logical_and(Y>percentile_downY, Y<percentile_upY)

        mask = np.logical_and(maskX, maskY)

        return X[mask], Y[mask]



def linear_fit(x, y, robust: bool = False, return_r2: bool = False,
               robust_params_init: tuple = None, robust_f_scale: float = None):

    if not robust:
        res = linregress(x, y)

        if return_r2:
            return res.intercept, res.slope, res.rvalue**2
        else:
            return res.intercept, res.slope
    
    else:
        def f(params, x, y):
            return params[0] + params[1] * x - y

        if robust_params_init is None:
            robust_params_init = np.ones(2)

        res_robust = least_squares(f, robust_params_init, args=(x, y),
                                   loss='soft_l1', f_scale=robust_f_scale)

        if return_r2:
            raise NotImplementedError
        else:
            return res_robust.x[0], res_robust.x[1]
        
        
def smooth_gaussian(array, sigma, scale: tuple, mask=None, mask_for_volume=None, return_effective_volume: bool=False):
    """
    Performs convolution of 'array' with a gaussian kernel of
    width(s) 'sigma'. 
    If 'mask' is specified, the convolution will not take the
    masked value into account.
    """

    scale=np.array(scale)
    sigmas = sigma/scale
    
    if mask is None:
        #return skimage_gaussian(array, sigmas, preserve_range=True, mode='constant', cval=0.0)
        return scipy_gaussian(array, sigmas, mode='constant', cval=0.0)
    else:

        if mask_for_volume is None:
            mask_for_volume = mask.copy()

        mask = mask.astype(bool)
        mask_for_volume = mask_for_volume.astype(bool)
        array_copy = array.copy() * 1.0

        array_copy[~mask_for_volume] = 0.0

        smooth_array = scipy_gaussian(
            array_copy, sigmas,
            mode='constant', cval=0.0
        )

        smooth_array_copy = smooth_array.copy()
        
        # calculate renormalization factor for masked gaussian (the 'effective'
        # volume of the gaussian kernel taking the mask into account)
        effective_volume = scipy_gaussian(
            mask_for_volume*1.0, sigmas,
            mode='constant', cval=0.0
        )

        smooth_array[mask] = smooth_array[mask] / effective_volume[mask]
        smooth_array[~mask] = 0.0

        if return_effective_volume:
            return smooth_array, effective_volume, smooth_array_copy
        else:
            return smooth_array
