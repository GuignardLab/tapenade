import numpy as np
from skimage.measure import regionprops


def _remove_labels_outside_of_mask(
    labels: np.ndarray, mask: np.ndarray
) -> np.ndarray:
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
