from skimage.measure import regionprops
import numpy as np
from os import cpu_count
from tqdm.contrib.concurrent import process_map


def _remove_labels_outside_of_mask(labels, mask):
    """
    Helper function to remove labels outside (or at the border) of the mask.

    Args:
        labels (ndarray): The segmentation labels.
        mask (ndarray): The mask indicating the valid region.

    Returns:
        ndarray: The post-processed segmentation labels.
    """
    props = regionprops(labels) 

    for prop in props:
        mask_roi = mask[prop.slice]

        # Check if the label is at least partially outside the mask
        if np.any(~mask_roi):

            # Calculate the volume of the label that intersects the mask
            volume_inside = np.logical_and(prop.image, mask_roi).sum()

            # If the volume inside is smaller than the total volume of the label,
            # remove the label from the labels array
            if volume_inside < prop.area:
                labels_roi = labels[prop.slice]
                labels_roi[labels_roi == prop.index] = 0

    return labels


def remove_labels_outside_of_mask(labels, mask, n_jobs=-1):
    """
    Removes labels outside (or at the border) of the mask.

    Parameters:
    - labels (ndarray): The segmentation labels.
    - mask (ndarray): The mask indicating the valid region.
    - n_jobs (int): The number of parallel jobs to run. If -1, use all available CPUs.

    Returns:
        ndarray: The post-processed segmentation labels.
    """
    is_temporal = labels.ndim == 4

    if is_temporal:

        if n_jobs == 1:
            # Process each label and mask pair sequentially
            labels_filtered = np.array([
                _remove_labels_outside_of_mask(lab, ma)
                for lab, ma in zip(labels, mask)
            ])
        else:
            max_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

            # Process each label and mask pair in parallel using multiple workers
            labels_filtered = np.array(
                process_map(
                    _remove_labels_outside_of_mask,
                    labels,
                    mask,
                    max_workers=max_workers,
                    desc='Removing labels outside of mask'
                )
            )
        
    else:
        # Process the single label and mask pair
        labels_filtered = _remove_labels_outside_of_mask(labels, mask)

    return labels_filtered
            



