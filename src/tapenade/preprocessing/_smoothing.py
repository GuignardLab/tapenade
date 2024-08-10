import numpy as np
import os

from functools import partial
from scipy.ndimage import gaussian_filter as scipy_gaussian
from tqdm.contrib.concurrent import process_map

from tqdm import tqdm
from typing import Union, Optional, List, Tuple


def _masked_smooth_gaussian(array, sigmas, mask=None, mask_for_volume=None):
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
    return _masked_smooth_gaussian(data, sigmas, mask, mask_for_volume)



