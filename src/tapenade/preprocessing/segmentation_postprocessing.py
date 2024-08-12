from os import cpu_count

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
