from os import cpu_count
from tqdm import tqdm
import numpy as np
from tqdm.contrib.concurrent import process_map

from tapenade.preprocessing._labels_masking import (
    _remove_labels_outside_of_mask,
)


def remove_labels_outside_of_mask(
    mask: np.ndarray, labels: np.ndarray, n_jobs: int = -1
) -> np.ndarray:
    """
    Removes labels outside (or at the border) of the mask.

    Parameters:
    - mask (ndarray): The mask indicating the valid region.
    - labels (ndarray): The segmentation labels.
    - n_jobs (int): The number of parallel jobs to run. If -1, use all available CPUs.

    Returns:
        ndarray: The post-processed segmentation labels.
    """
    is_temporal = labels.ndim == 4

    if is_temporal:

        if n_jobs == 1:
            # Process each label and mask pair sequentially
            labels_filtered = np.array(
                [
                    _remove_labels_outside_of_mask(lab, ma)
                    for lab, ma in zip(labels, mask, strict=False)
                ]
            )
        else:
            max_workers = (
                cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
            )

            # Process each label and mask pair in parallel using multiple workers
            labels_filtered = np.array(
                process_map(
                    _remove_labels_outside_of_mask,
                    labels,
                    mask,
                    max_workers=max_workers,
                    desc="Removing labels outside of mask",
                )
            )

    else:
        # Process the single label and mask pair
        labels_filtered = _remove_labels_outside_of_mask(labels, mask)

    return labels_filtered


def find_seg_errors(segmentation: np.ndarray, image: np.ndarray):
    """
    Compute statisitics of intensity of each label to try to find out exceptions/anomalies and detect segmentation errors.

    Parameters:
    segmentation: array containing the labels
    image : intensity image, has to be the same size as segmentatio,

    Returns:
    The intensity_distribution array, for which each line corresponds to a label in the segmentation.
    Column 1 is the label's id
    Col 2 is its mean intensity (in the ROI)
    Col 3 is the std of its intensity distrib
    Col 4 is the ratio of standard deviation and mean. We use this value to detect wrong segmentations

    """
    list_labels = list(np.unique(segmentation))
    list_labels.remove(0)
    print(len(list_labels), " labels to process")

    intensity_distribution = np.zeros((len(list_labels), 4))
    for index, label in tqdm(enumerate(list_labels)):
        intensity_distribution[index, 0] = label
        mask = segmentation == label
        list_pix = image[mask]
        n, bins = np.histogram(list_pix)
        mids = 0.5 * (bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean) ** 2, weights=n)
        intensity_distribution[index, 1] = mean
        intensity_distribution[index, 2] = var
        intensity_distribution[index, 3] = var / mean

    return intensity_distribution


def tresh_distribution(
    intensity_distribution: np.ndarray,
    threshold: float,
    column_number: int = 3,
):
    """
    Thresholds the Distribution to find labels that could be wrong segmentation.

    Parameters:
    intensity_distribution: Array computed with the function find_seg_errors, that shows average intensity, std and std/mean for each label of the distrbution
    threshold : threshold value above which cell will be considered wrong. To get an idea, visualize the distrbution (examples are in the notebooks)
    column_number : index to consider for thre threshold.
                    If you want to discrimnate on the mean intensity of the ROI, choose 1.
                    If you want to discriminate on the std, choose 2.
                    If you want to discriminate over the std divided by the mean intensity for each label, choose 3.

    Returns:
    A list of labels that might be false segmentations.

    """

    id_merged_cells = []
    for index, intensity in enumerate(
        intensity_distribution[:, column_number]
    ):
        if intensity > threshold:
            id_merged_cells.append(int(intensity_distribution[index, 0]))
    return id_merged_cells

def remove_small_objects(segmentation: np.ndarray, min_size: int):
    """
    Remove small objects from a segmentation, using the threshold volume given as a parameter.

    Parameters:
    segmentation: array containing the labels
    min_size: minimum size of the objects to keep

    Returns:
    The modified segmentation array.
    """
    seg_filt = np.copy(segmentation)
    unique_labels, label_counts = np.unique(segmentation, return_counts=True)

    smallest_labels = unique_labels[np.argsort(label_counts)]
    smallest_volumes = np.sort(label_counts)

    for label, volume in zip(smallest_labels, smallest_volumes):
        if volume<min_size :
            seg_filt[segmentation==label]=0

    return(seg_filt)