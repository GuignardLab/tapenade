from itertools import product

import numpy as np
from scipy.ndimage import zoom


def _local_contrast_enhancement(
    image: np.ndarray,
    box_size: int,
    perc_low: float,
    perc_high: float,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Performs local histogram stretching by applying the following steps:
        1. compute the percentile values in a neighborhood around voxels chosen on a regular grid.
        2. interpolate these values on each voxel of the image.
        3. perform a linear mapping between the interpolated percentile values and the values 0 and 1.
        4. clip the image voxel values between 0 and 1.

    Parameters
    ----------
    image : np.ndarray
        The image to enhance.
    box_size : int
        size of the neighborhood box around each voxel to compute the percentile values.
    perc_low : float
        percentile value to use as the low value of the intensity mapping (will be mapped to 0).
    perc_high : float
        percentile value to use as the high value of the intensity mapping (will be mapped to 1).
    mask : np.ndarray
        An optional mask boolean array from which background values will be excluded from the computation.

    Returns
    -------
    image_norm : np.ndarray
        The enhanced image.
    """

    # Compute necessary variables
    array_shape = image.shape
    box_length = 2 * box_size + 1
    grid_shape = [int(np.ceil(s / box_length)) + 1 for s in array_shape]
    grid_positions = [
        np.linspace(0, s - 1, n_boxes)
        for s, n_boxes in zip(array_shape, grid_shape, strict=False)
    ]

    percs_low = np.zeros(grid_shape)
    percs_high = np.zeros(grid_shape)

    percs_positions = [np.arange(s) for s in grid_shape]

    # Compute percentile values for each neighborhood
    for indices_grid, indices_percs in zip(
        product(*grid_positions), product(*percs_positions), strict=False
    ):
        indices_grid = np.array(indices_grid).round().astype(int)

        slices = []
        for index, s in zip(indices_grid, array_shape, strict=False):
            start = min(max(index - box_size, 0), s - box_length)
            stop = min(max(index + box_size + 1, box_length), s)
            slices.append(slice(start, stop))

        box = image[tuple(slices)].copy()

        # for debug, actually should not be necessary
        if box.size < 0.2 * box_length**image.ndim:
            continue

        if mask is not None:
            box = box[mask[tuple(slices)]]
            if box.size == 0:
                continue

        val_low, val_high = np.nanpercentile(box, [perc_low, perc_high])

        if np.isnan(val_low) or np.isnan(val_high):
            continue

        percs_low[tuple(indices_percs)] = val_low
        percs_high[tuple(indices_percs)] = val_high

    # Interpolate percentile values for each voxel
    full_percs_low = zoom(
        percs_low,
        zoom=[s / n for s, n in zip(array_shape, grid_shape, strict=False)],
        order=1,
    )

    full_percs_high = zoom(
        percs_high,
        zoom=[s / n for s, n in zip(array_shape, grid_shape, strict=False)],
        order=1,
    )

    denom = full_percs_high - full_percs_low
    denom[denom == 0] = 1
    denom += 1e-8

    # Perform linear mapping and clip image voxel values
    # image_norm = np.clip((image - full_percs_low) / denom, 0, 1)
    image_norm = (image - full_percs_low) / denom

    return image_norm
