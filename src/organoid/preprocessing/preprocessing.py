import numpy as np

from functools import partial
from os import cpu_count
from scipy.ndimage import zoom, rotate, uniform_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.signal import argrelextrema
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from sklearn.decomposition import PCA
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from organoid.preprocessing._smoothing import gaussian_smooth
from organoid.preprocessing._local_normalization import local_normalization

"""
In typical order:
    1. making array isotropic
    2. compute mask
    3. local image normalization
    (4. spatial regi, not covered here)
    (4. segmentation, not covered here)
    (5. registration, not covered here)
    6. aligning major axis
    7. cropping array using mask
"""


def local_image_normalization(image, box_size, perc_low, perc_high):
    """
    Performs local image normalization on either a single image or a temporal stack of images.
    Stretches the image histogram in local neighborhoods by remapping intesities in the range
    [perc_low, perc_high] to the range [0, 1]. 
    This helps to enhance the contrast and improve the visibility of structures in the image.

    Parameters:
    - image: numpy array, input image or temporal stack of images
    - box_size: int, size of the local neighborhood for normalization
    - perc_low: float, lower percentile for intensity normalization
    - perc_high: float, upper percentile for intensity normalization

    Returns:
    - image_norm: numpy array, normalized image or stack of normalized images
    """

    is_temporal = image.ndim == 4

    if is_temporal:
        # Apply local normalization to each time frame in the temporal stack
        image_norm = np.array(
            [local_normalization(
                image[ind_t], 
                box_size=box_size, 
                perc_low=perc_low, 
                perc_high=perc_high
            ) for ind_t in tqdm(range(image.shape[0]), desc="Local normalization")]
        )
    else:
        # Apply local normalization to the image
        image_norm = local_normalization(image, box_size=box_size, 
                                         perc_low=perc_low, perc_high=perc_high)
        
    return image_norm

def make_array_isotropic(image, voxel_scale=None, zoom_factors=None, order=1, n_jobs=-1):
    """
    Resizes an input image to have isotropic voxel dimensions.

    Parameters:
    - image: numpy array, input image
    - voxel_scale: tuple or list, voxel size (e.g in um for each dimension)
    - order: int, order of interpolation for resizing (defaults to 1 for 
      linear interpolation). Choose 0 for nearest-neighbor interpolation 
      (e.g. for label images)  
    - n_jobs: int, optional number of parallel jobs for resizing (default: -1)

    Returns:
    - resized_image: numpy array, resized image with isotropic voxel dimensions
    """

    is_temporal = image.ndim == 4
    
    if voxel_scale is not None:
        zoom_factors = np.array(voxel_scale)
    elif zoom_factors is not None:
        zoom_factors = np.array(zoom_factors)
    else:
        raise ValueError("Either voxel_scale or zoom_factors must be provided.")

    if is_temporal:
        
        if n_jobs == 1:
            # Sequential resizing of each time frame
            resized_image = np.array(
                [zoom(image[ind_t], zoom_factors, order=order) \
                 for ind_t in tqdm(range(image.shape[0]), desc="Making array isotropic")]
            )
        
        else:
            # Parallel resizing of each time frame using multiple processes
            func_parallel = partial(zoom, zoom=zoom_factors, order=order)

            max_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

            resized_image = np.array(
                process_map(
                    func_parallel, image, max_workers=max_workers, desc="Making array isotropic"
                )
            )

    else:
        # Resizing the whole image
        resized_image = zoom(image, zoom_factors, order=order)

    return resized_image


def compute_mask_OLD(image, sigma_blur, threshold=None, n_jobs=-1):
    """
    Computes a binary mask from an input image using Gaussian smoothing and thresholding.

    Parameters:
    - image: numpy array, input image
    - sigma_blur: float, standard deviation for Gaussian smoothing
    - threshold: float, optional threshold value for binarization (default: None)
    - n_jobs: int, optional number of parallel jobs for smoothing (default: -1)

    Returns:
    - mask: numpy array, binary mask computed from the input image
    """

    is_temporal = image.ndim == 4

    # Apply Gaussian smoothing to the image
    blurred = gaussian_smooth(
        image, 
        sigmas=sigma_blur,
        n_jobs=n_jobs,
    )

    if is_temporal:
        # If the image is temporal, apply thresholding to each time frame
        mask = np.zeros_like(blurred, dtype=bool)
        for ind_t in tqdm(range(image.shape[0]), desc="Thresholding image"):

            blurred_stack = blurred[ind_t]

            if threshold is None:
                # Compute the threshold using Otsu's method
                threshold = 0.6 * threshold_otsu(blurred_stack)

            mask[ind_t] = blurred_stack > threshold

    else:
        # If the image is not temporal, apply thresholding to the whole image
        if threshold is None:
            # Compute the threshold using Otsu's method
            threshold = 0.6 * threshold_otsu(blurred)
        
        mask = blurred > threshold

    return mask

def variance_threshold_binarization(image, box_size):
    """
    Perform variance threshold binarization on an input image.

    Parameters:
    - image: numpy array, input image
    - box_size: int, size of the local neighborhood for computing variance

    Returns:
    - binary_mask: numpy array, binary mask computed using variance thresholding
    """

    # Compute the variance of the image using a local neighborhood
    variance1 = uniform_filter(image, box_size)**2
    variance2 = uniform_filter(image**2, box_size)
    sigma = np.sqrt(variance2 - variance1)

    # Compute the histogram of the variance values
    freqs, bins = np.histogram(sigma.ravel(), bins=256)

    # Find the local minima in the histogram
    # Specifically, it finds weak minima (can be less or equal to the neighboring
    # points) and choses the lowest excluding the boundaries.
    threshold_candidates_args = argrelextrema(freqs, np.less_equal, order=1)[0][1:-1]

    # Select the threshold value as the one with the minimum frequency
    threshold = bins[threshold_candidates_args[np.argmin(freqs[threshold_candidates_args])]]

    # Create a binary mask using the variance threshold
    binary_mask = sigma > threshold

    return binary_mask

def get_largest_connected_component(array):
    """
    Get the largest connected component in a binary array.

    Parameters:
    - array: Binary array.

    Returns:
    - mask_largest_cc: Binary mask of the largest connected component.
    """
    labels = label(array)
    mask_largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return mask_largest_cc

def refine_raw_mask(mask):
    """
    Refine the raw mask by keeping only the largest connected component and filling holes.

    Parameters:
    - mask: numpy array, binary mask

    Returns:
    - refined_mask: numpy array, refined binary mask
    """
    # Keep only the largest connected component
    mask = get_largest_connected_component(mask)
    # Fill holes in the mask
    refined_mask = binary_fill_holes(mask)
    return refined_mask

def process_mask(im, box_size):
    """
    Process the mask for the given image.

    Parameters:
    - im: numpy array, input image
    - box_size: int, size of the box for variance calculation. Should typically be 
        between 1 and 3 times the typical object diameter.

    Returns:
    - mask: numpy array, binary mask of the same shape as the input image
    """

    # Compute the mask using variance threshold binarization
    mask = variance_threshold_binarization(im, box_size)

    # Refine the mask by keeping only the largest connected component and filling holes
    mask = refine_raw_mask(mask)

    return mask


def compute_mask(image, box_size, n_jobs=-1):
    """
    Compute the mask for the given image using variance threshold binarization.

    Parameters:
    - image: numpy array, input image
    - box_size: int, size of the box for variance calculation. Should typicall be 
      between and 1 and 3 times the typical object diameter.
    - n_jobs: int, number of parallel jobs to run (-1 for using all available CPUs)

    Returns:
    - mask: numpy array, binary mask of the same shape as the input image
    """

    is_temporal = image.ndim == 4

    if is_temporal:
        # func_parallel = partial(variance_threshold_binarization, box_size=box_size)
        func_parallel = partial(process_mask, box_size=box_size)

        if n_jobs == 1:
            # Sequential processing
            mask = np.array(
                [func_parallel(im) for im in tqdm(image, desc="Thresholding image")]
            )
        else:
            # Parallel processing
            max_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

            mask = np.array(
                process_map(
                    func_parallel, image, max_workers=max_workers, desc="Thresholding image"
                )
            )
    else:
        # Single image processing
        mask = process_mask(image, box_size)

    return mask

def signed_angle(v1, v2, vn):
    """
    Calculate the signed angle between two vectors in 3D space.

    Parameters:
    - v1: numpy array, first vector
    - v2: numpy array, second vector
    - vn: numpy array, normal vector

    Returns:
    - angle: float, signed angle between the two vectors
    """

    angle = np.arctan2(
        np.dot(vn, np.cross(v1, v2)),
        np.dot(v1, v2)
    )

    return angle

        

def align_array_major_axis(target_axis: str, rotation_plane: str, 
                           mask, image=None, labels=None, order=1,
                           temporal_slice=None, n_jobs=-1):
    """
    Aligns the major axis of an array to a target axis in a specified rotation plane.
    This function uses Principal Component Analysis (PCA) to determine the major axis of the array,
    and then rotates the array to align the major axis with the target axis.

    Parameters:
    - target_axis: str, the target axis to align the major axis with ('X', 'Y', or 'Z')
    - rotation_plane: str, the rotation plane to perform the rotation in ('XY', 'XZ', or 'YZ')
    - mask: numpy array, binary mask indicating the region of interest
    - image: numpy array, input image or temporal stack of images (optional)
    - labels: numpy array, labels corresponding to the mask (optional)
    - order: int, order of interpolation for image rotation (default: 1)
    - temporal_slice: int, optional temporal slicing applied to the mask before computing its major axis (default: None)
    - n_jobs: int, number of parallel jobs to use (-1 for all available CPUs, 1 for sequential execution) (default: -1)

    Returns:
    - If both image and labels are provided:
        - mask_rotated: numpy array, rotated mask
        - image_rotated: numpy array, rotated image
        - labels_rotated: numpy array, rotated labels
    - If only image is provided:
        - mask_rotated: numpy array, rotated mask
        - image_rotated: numpy array, rotated image
    - If only labels is provided:
        - mask_rotated: numpy array, rotated mask
        - labels_rotated: numpy array, rotated labels
    - If neither image nor labels is provided:
        - mask_rotated: numpy array, rotated mask
    """

    is_temporal = mask.ndim == 4

    temporal_slice = slice(None) if temporal_slice is None else temporal_slice

    # Compute the mask that will be used to compute the major axis.
    # If the mask is a temporal array, the major axis is computed by aggregating the mask
    # instances in temporal_slice. 
    mask_for_pca = np.any(mask[temporal_slice], axis=0) if is_temporal else mask
    
    # Perform PCA on the mask to determine the major axis
    pca = PCA(n_components=3)
    pca.fit(np.argwhere(mask_for_pca))
    major_axis_vector = pca.components_[0]

    plane_normal_vector = {
        'XY': np.array([1, 0, 0]),
        'XZ': np.array([0, 1, 0]),
        'YZ': np.array([0, 0, 1])
    }[rotation_plane]

    # Remove the component of the major axis vector that is perpendicular to the rotation plane
    major_axis_vector -= np.dot(major_axis_vector, plane_normal_vector) * plane_normal_vector
    major_axis_vector = major_axis_vector / np.linalg.norm(major_axis_vector)
    major_axis_vector *= np.sign(np.max(major_axis_vector))

    target_axis_vector = {
        'Z': np.array([1, 0, 0]),
        'Y': np.array([0, 1, 0]),
        'X': np.array([0, 0, 1])
    }[target_axis]

    # Calculate the rotation angle between the major axis and the target axis
    rotation_angle = signed_angle(major_axis_vector, target_axis_vector, plane_normal_vector) * 180 / np.pi

    rotation_plane_indices = {
        'XY': (-1, -2), # respects the right-hand rule wrt the corresponding normal vector
        'XZ': (-3, -1),
        'YZ': (-2, -3)
    }[rotation_plane]

    # Define the rotation functions
    func_rotate_image = partial(rotate, angle=rotation_angle, 
                                axes=rotation_plane_indices, 
                                reshape=True, order=order)
    func_rotate = partial(rotate, angle=rotation_angle, 
                         axes=rotation_plane_indices, 
                          reshape=True, order=0)

    if is_temporal and n_jobs != 1:
        # Rotate the arrays in parallel if the array is temporal and parallel execution is enabled
        max_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

        mask_rotated = np.array(
            process_map(func_rotate, mask, max_workers=max_workers, desc="Aligning mask")
        )
        if image is not None:
            image_rotated = np.array(
                process_map(func_rotate_image, image, max_workers=max_workers, desc="Aligning image")
            )
        if labels is not None:
            labels_rotated = np.array(
                process_map(func_rotate, labels, max_workers=max_workers, desc="Aligning labels")
            )
        
    else:
        # Rotate the arrays in block
        mask_rotated = func_rotate(mask)
        if image is not None:
            image_rotated = func_rotate_image(image)
        if labels is not None:
            labels_rotated = func_rotate(labels)

    if image is not None and labels is not None:
        return mask_rotated, image_rotated, labels_rotated
    elif image is not None:
        return mask_rotated, image_rotated
    elif labels is not None:
        return mask_rotated, labels_rotated
    else:
        return mask_rotated


def crop_array_using_mask(array, mask, margin=0):
    """
    Crop an array using a binary mask. If the array is temporal, the cropping
    slice is computed by aggregating mask instances at all times. 

    Parameters:
    - array: numpy array, input array to be cropped
    - mask: numpy array, binary mask indicating the region of interest
    - margin: int, optional margin to add around the mask (default: 0)

    Returns:
    - cropped_array: numpy array, cropped array based on the mask
    """

    is_temporal = array.ndim == 4

    # Compute aggregated mask if array is temporal
    mask_for_slice = np.any(mask, axis=0) if is_temporal else mask

    # Get the mask slice
    mask_slice = regionprops(mask_for_slice.astype(int))[0].slice

    # Add margin to the slice if specified
    if margin > 0:
        mask_slice = tuple(
            slice(
                max(0, mask_slice[i].start - margin),
                min(mask_slice[i].stop + margin, array.shape[i])
            ) for i in range(3)
        )
    
    mask_slice = (slice(None),) + mask_slice if is_temporal else mask_slice
    # Apply the slice to the array
    array_cropped = array[mask_slice]
    mask_cropped = mask[mask_slice]

    return array_cropped, mask_cropped
