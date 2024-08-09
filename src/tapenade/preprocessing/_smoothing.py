import numpy as np
import os

from functools import partial
from scipy.ndimage import gaussian_filter as scipy_gaussian
from tqdm.contrib.concurrent import process_map

from tqdm import tqdm
from typing import Union, Optional, List, Tuple


def _smooth_gaussian(array, sigmas, mask=None, mask_for_volume=None):
    """
    If 'mask' is specified, the convolution will not take the
    masked value into account.
    """

    sigmas = sigmas if isinstance(sigmas, (int, float)) else np.array(sigmas)

    if mask is None:
        # return skimage_gaussian(array, sigmas, preserve_range=True, mode='constant', cval=0.0)
        return scipy_gaussian(array, sigmas, mode="nearest", cval=0.0)
    else:

        mask = mask.astype(bool)

        if mask_for_volume is None:

            smooth_array = scipy_gaussian(
                np.where(mask, array.astype(np.float32), 0.0),
                sigmas,
                mode="constant",
                cval=0.0,
                truncate=3.0,
            )

            # calculate renormalization factor for masked gaussian (the 'effective'
            # volume of the gaussian kernel taking the mask into account)
            effective_volume = scipy_gaussian(
                mask.astype(np.float32), sigmas, mode="constant", cval=0.0, truncate=3.0
            )

        else:
            mask_for_volume = mask_for_volume.astype(bool)

            smooth_array = scipy_gaussian(
                np.where(mask_for_volume, array.astype(np.float32), 0.0),
                sigmas,
                mode="constant",
                cval=0.0,
                truncate=3.0,
            )

            # calculate renormalization factor for masked gaussian (the 'effective'
            # volume of the gaussian kernel taking the mask into account)
            effective_volume = scipy_gaussian(
                mask_for_volume.astype(np.float32),
                sigmas,
                mode="constant",
                cval=0.0,
                truncate=3.0,
            )

        smooth_array = np.where(
            mask,
            np.divide(smooth_array, effective_volume, where=mask),
            0.0,
        )

        return smooth_array


def _parallel_gaussian_smooth(
    input_tuple: Tuple[np.ndarray, np.ndarray],
    sigmas: Union[float, List[float]],
) -> np.ndarray:
    data, mask, mask_for_volume = input_tuple
    return _smooth_gaussian(data, sigmas, mask, mask_for_volume)


def _gaussian_smooth(
    image: np.ndarray,
    sigmas: Union[float, List[float]],
    mask: Optional[np.ndarray] = None,
    mask_for_volume: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to an image or a sequence of images.

    Args:
        image (ndarray): The input image or sequence of images.
        sigmas (float or list of floats): The standard deviation(s) of the Gaussian kernel.
        mask (ndarray, optional): The mask indicating the regions of interest. Default is None.
        mask_for_volume (ndarray, optional): The mask indicating the regions of interest for volume calculation. Default is None.
        n_jobs (int, optional): The number of parallel jobs to run. Default is -1, which uses all available CPU cores.

    Returns:
        ndarray: The smoothed image or sequence of images.
    """

    is_temporal = image.ndim == 4

    if is_temporal:

        if mask is None:
            mask = [None] * image.shape[0]

        if mask_for_volume is None:
            mask_for_volume = [None] * image.shape[0]

        func = partial(_parallel_gaussian_smooth, sigmas=sigmas)

        if n_jobs == 1:

            iterable = tqdm(
                zip(image, mask, mask_for_volume), total=len(image), desc="Smoothing image"
            )

            return np.array([func(elem) for elem in iterable])

        else:
            elems = [elem for elem in zip(image, mask)]

            max_workers = (
                os.cpu_count() if n_jobs == -1 else min(n_jobs, os.cpu_count())
            )
            result = process_map(
                func, elems, max_workers=max_workers, desc="Smoothing image"
            )

            return np.array(result)

    else:
        return _smooth_gaussian(image, sigmas, mask, mask_for_volume)
    






def gaussian_smooth_dense_two_arrays_gpu(
    datas: List[np.ndarray],
    sigmas: Union[float, List[float]],
    mask: np.ndarray = None,
    masks_for_volume: Union[np.ndarray, List[np.ndarray]] = None,
):
    """
    Inputs in data are assumed to be non-temporal, 3D arrays.
    """

    from pyclesperanto_prototype import gaussian_blur
    

    if isinstance(sigmas, (int, float)):
        sigmas = [sigmas] * 3

    if mask is None:
        smoothed1 = np.array(gaussian_blur(
            datas[0].astype(np.float16), 
            sigma_x=sigmas[0], 
            sigma_y=sigmas[1], 
            sigma_z=sigmas[2]
        ))
        smoothed2 = np.array(gaussian_blur(
            datas[1].astype(np.float16), 
            sigma_x=sigmas[0], 
            sigma_y=sigmas[1], 
            sigma_z=sigmas[2]
        ))

    elif masks_for_volume is None:

        smoothed1 = np.array(gaussian_blur(
            np.where(mask, datas[0].astype(np.float16), 0.0),
            sigma_x=sigmas[0],
            sigma_y=sigmas[1],
            sigma_z=sigmas[2],
        ))

        smoothed2 = np.array(gaussian_blur(
            np.where(mask, datas[1].astype(np.float16), 0.0),
            sigma_x=sigmas[0],
            sigma_y=sigmas[1],
            sigma_z=sigmas[2],
        ))

        effective_volume = np.array(gaussian_blur(
            mask.astype(np.float16),
            sigma_x=sigmas[0],
            sigma_y=sigmas[1],
            sigma_z=sigmas[2],
        ))

        smoothed1 = np.where(
            mask,
            np.divide(smoothed1, effective_volume, where=mask),
            0.0
        )

        smoothed2 = np.where(
            mask,
            np.divide(smoothed2, effective_volume, where=mask),
            0.0
        )

    else: # both masks and masks_for_volume are not None

        if isinstance(masks_for_volume, np.ndarray):
            
            smoothed1 = np.array(gaussian_blur(
                np.where(masks_for_volume, datas[0].astype(np.float16), 0.0),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            ))

            smoothed2 = np.array(gaussian_blur(
                np.where(masks_for_volume, datas[1].astype(np.float16), 0.0),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            ))

            effective_volume = np.array(gaussian_blur(
                masks_for_volume.astype(np.float16),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            ))

            smoothed1 = np.where(
                mask,
                np.divide(smoothed1, effective_volume, where=mask),
                0.0
            )

            smoothed2 = np.where(
                mask,
                np.divide(smoothed2, effective_volume, where=mask),
                0.0
            )


        elif isinstance(masks_for_volume, list):

            smoothed1 = np.array(gaussian_blur(
                np.where(masks_for_volume[0], datas[0].astype(np.float16), 0.0),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            ))
        
            smoothed2 = np.array(gaussian_blur(
                np.where(masks_for_volume[1], datas[1].astype(np.float16), 0.0),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            ))

            effective_volume1 = np.array(gaussian_blur(
                masks_for_volume[0].astype(np.float16),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            ))

            effective_volume2 = np.array(gaussian_blur(
                masks_for_volume[1].astype(np.float16),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            ))

            smoothed1 = np.where(
                mask,
                np.divide(smoothed1, effective_volume1, where=mask),
                0.0
            )

            smoothed2 = np.where(
                mask,
                np.divide(smoothed2, effective_volume2, where=mask),
                0.0
            )

    return smoothed1, smoothed2
