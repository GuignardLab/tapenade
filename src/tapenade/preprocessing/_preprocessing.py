import concurrent.futures
from functools import partial
from os import cpu_count
from typing import Optional, Union

import numpy as np
import tifffile
from scipy.ndimage import rotate
from skimage.measure import regionprops
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tapenade.preprocessing._array_rescaling import _change_array_pixelsize
from tapenade.preprocessing._local_equalization import _local_equalization
from tapenade.preprocessing._thresholding import _compute_mask
from tapenade.preprocessing._axis_alignment import (
    _compute_rotation_angle_and_indices,
)
from tapenade.preprocessing._intensity_normalization import (
    _normalize_intensity,
)
from tapenade.preprocessing._local_equalization import _local_equalization
from tapenade.preprocessing._smoothing import (
    _masked_smooth_gaussian,
    _masked_smooth_gaussian_sparse
)
from tapenade.preprocessing._segmentation import _segment_stardist
from tapenade.preprocessing._thresholding import _compute_mask

"""
#! TODO:
    - refactor every multiprocessing calls through a single decorator
    - update notebook to reflect changes
"""

"""
In typical order:
    1. making array isotropic
    2. compute mask
    3. local image equalization
    (4. segmentation, not covered here)
    (5. spatial registration, not covered here)
    (6. temporal registration, not covered here)
    7. aligning major axis
    (8. cropping array using mask)
"""


def isotropize_and_normalize(mask,image,labels,scale,sigma:float=None,pos_ref:int=0) :
    """
    Make an image isotropic and normalized with respect to a reference channel. Works for multichannel images (ZCYX convention) or single channel images (ZYX convention).
    Parameters
    ----------
    mask : np.array (bool)
        mask of the image
    image : np.array
        image to normalize
    labels : np.array
        labels of the mask
    scale : tuple
        scale factors for the isotropic transformation
    sigma : int
        sigma for the gaussian filter
    pos_ref : int
        position of the reference channel, starting from 0
    Returns
    -------
    norm_image : np.array
        normalized and isotropic image
    """

    if len(image.shape) > 3:  # if multichannel image
        nb_channels = image.shape[1]
        assert (
            pos_ref < nb_channels
        ), "The position of the reference channel is greater than the number of channels. Choose 0 if the first channel is the reference, 1 if the second channel is the reference, etc."
        iso_image = []
        liste_channels = np.linspace(
            0, nb_channels - 1, nb_channels, dtype=int
        )
        for ch in liste_channels:
            channel = image[:, ch, :, :]
            (mask_iso, channel_iso, seg_iso) = change_arrays_pixelsize(
                mask=mask,
                image=channel,
                labels=labels,
                input_pixelsize=scale,
                output_pixelsize=(1, 1, 1),
                order=1,
                n_jobs=-1,
            )
            iso_image.append(channel_iso)

        iso_image = np.array(iso_image)
        iso_image = iso_image.transpose(1, 0, 2, 3)  # stay in convention ZCYX

        ref_channel = iso_image[
            :, pos_ref, :, :
        ]  # should check before if pos_ref>=iso_image.shape[1]
        liste_float_channels = np.delete(liste_channels, pos_ref)
        norm_image = np.zeros_like(iso_image)
        for ch_float in liste_float_channels:
            channel = iso_image[:, ch_float, :, :]
            channel_norm, ref_norm = normalize_intensity(
                image=channel,
                ref_image=ref_channel,
                mask=mask_iso,
                labels=seg_iso,
                sigma=sigma,
            )
            norm_image[:, ch_float, :, :] = channel_norm
        norm_image[:, pos_ref, :, :] = ref_norm

    else:  # 3D data, one channel
        (mask_iso, iso_image, seg_iso) = change_arrays_pixelsize(
            mask=mask,
            image=image,
            labels=labels,
            reshape_factors=np.divide(scale, (1, 1, 1)),
        )
        norm_image, _ = normalize_intensity(
            image=iso_image,
            ref_image=iso_image,
            mask=mask_iso,
            labels=seg_iso,
            sigma=sigma,
        )

    return (mask_iso, norm_image, seg_iso)



def change_array_pixelsize(
    array: np.ndarray,
    input_pixelsize: tuple[float, float, float] = (1, 1, 1),
    output_pixelsize: tuple[float, float, float] = (1, 1, 1),
    order: int = 1,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Resizes an input image to have isotropic voxel dimensions.

    Parameters:
    - array: numpy array
    - input_pixelsize: tuple of floats, input pixel dimensions (e.g. in microns)
    - output_pixelsize: tuple of floats, output pixel dimensions (e.g. in microns)
    - order: int, order of interpolation for resizing (defaults to 1 for
      linear interpolation). Choose 0 for nearest-neighbor interpolation
      (e.g. for label images)
    - n_jobs: int, optional number of parallel jobs for resizing (default: -1)

    Returns:
    - resized_array: numpy array
    """

    is_temporal = array.ndim == 4
    n_frames = 0

    if is_temporal:
        if n_jobs == 1:
            # Sequential resizing of each time frame
            resized_array = [
                _change_array_pixelsize(arr, input_pixelsize, output_pixelsize, order=order)
                for arr in tqdm(
                    array,
                    desc="Making array isotropic",
                    total=n_frames,
                )
            ]

        else:
            # Parallel resizing of each time frame using multiple processes
            func_parallel = partial(
                _change_array_pixelsize,
                input_pixelsize=input_pixelsize,
                output_pixelsize=output_pixelsize,
                order=order,
            )

            max_workers = (
                cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
            )

            resized_array = process_map(
                func_parallel,
                array,
                max_workers=max_workers,
                desc="Changing array pixelsize",
                total=n_frames,
            )

    else:
        # Resizing the whole image
        resized_array = _change_array_pixelsize(
            array, 
            input_pixelsize=input_pixelsize,
            output_pixelsize=output_pixelsize,
            order=order,
        )

    return resized_array

def transpose_and_split_stack(
    image:np.ndarray,
    nb_channels:int,
    nb_depth:int,
    nb_Y:int,
    nb_X:int,
    nb_timepoints:int,
    bool_seperate_channels:bool
    ) -> list :

    """
    Take a stack that can be 3D, 4D (TZYX or CZYX) or 5D (CTZYX)  and re-organize it in CTZYX convention.
    Possibility to split the channels if it is a multichannel image, in order to apply the pipeline functions on the separated channels.
    In that case the output is a list of the different channels
    Parameters
    ----------
    image : np.array
        image to transpose and split
    nb_channels : int
        number of channels
    nb_depth : int
        number of pixels in the z dimension
    nb_Y : int
        number of pixels in Y
    nb_X : int
        number of pixels in X
    nb_timepoints : int
        number of timepoints if this is a time sequence
    bool_seperate_channels : bool
        if True, the channels will be split

    Returns
    -------
    output_list : list
        list of images in CTZYX convention

    """
    output_list=[] #output is a list of images
    shape = np.shape(image)  
    str_shape = np.array([str(i) for i in np.shape(image)])
    #to find the index of each dimension we compare the shape of the image with the number selected. This is done comparing string.
    ind_Z = np.argwhere(str_shape==str(nb_depth))[0][0] 
    ind_Y = np.argwhere(str_shape==str(nb_Y))[0][0]
    ind_X = np.argwhere(str_shape==str(nb_X))[0][0]
    
    if nb_timepoints !='1' :
        ind_T = np.argwhere(str_shape==str(nb_timepoints))[0][0]
        size_T = shape[ind_T]
    else :
        size_T = 1 #if not a timesequence

    if nb_channels !='1' :
        ind_C = np.argwhere(str_shape==str(nb_channels))[0][0]
        size_C = shape[ind_C]
    else :
        size_C = 1 #if only one channel

    #convention CTZYX
    if size_T > 1 and size_C > 1:
        image_transposed = np.transpose(image, (ind_C, ind_T, ind_Z,ind_Y,ind_X))
        if bool_seperate_channels :
            for c in range(size_C) :
                output_list.append(image_transposed[c,:,:,:,:])
        else : #if we don't want to split the channels, then the output list has only one element that is the multichannel image
            output_list.append(image_transposed)
    elif size_T > 1 and size_C == 1: #TZYX
        output_list.append(np.transpose(image, (ind_T, ind_Z,ind_Y,ind_X)))
    elif size_T==1 and size_C>1: #CZYX
        image_transposed = np.transpose(image, (ind_C, ind_Z,ind_Y,ind_X))
        if bool_seperate_channels :
            for c in range(size_C) :
                output_list.append(image_transposed[c,:,:,:])
        else : #if we don't want to split the channels, then the output list has only one element that is the multichannel image
            output_list.append(image_transposed)
    else : # 3D data
        output_list.append(np.transpose(image, (ind_Z,ind_Y,ind_X)))

    return output_list

def compute_mask(
    image: np.ndarray,
    method: str,
    sigma_blur: float,
    threshold_factor: float = 1,
    post_processing_method: str = "fill_holes",
    keep_largest_cc: bool = True,
    registered_image: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute the mask for the given image using the specified method.

    Parameters:
    - image: numpy array, input image
    - method: str, method to use for thresholding. Can be 'snp otsu' for Signal-Noise Product thresholding,
      or 'otsu' for Otsu's thresholding.
    - sigma_blur: float, standard deviation of the Gaussian blur. Should typically be
      around 1/3 of the typical object diameter.
    - threshold_factor: float, factor to multiply the threshold (default: 1)
    - post_processing_method: str, method to use for post-processing the mask. Can be 'convex_hull' to compute
      the convex hull of the mask, 'fill_holes' to fill holes in the mask, or 'none' to skip post-processing
      (default: 'fill_holes')
    - keep_largest_cc: bool, set to True to keep only the largest connected component in the mask (default: True)
    - n_jobs: int, number of parallel jobs to run (-1 for using all available CPUs)

    Returns:
    - mask: numpy array, binary mask of the same shape as the input image
    """

    is_temporal = image.ndim == 4

    if is_temporal:
        func_parallel = partial(
            _compute_mask,
            method=method,
            sigma_blur=sigma_blur,
            threshold_factor=threshold_factor,
            post_processing_method=post_processing_method,
            keep_largest_cc=keep_largest_cc,
            registered_image=registered_image,
        )

        if n_jobs == 1:
            # Sequential processing
            mask = np.array(
                [
                    func_parallel(im)
                    for im in tqdm(image, desc="Thresholding image")
                ]
            )
        else:
            # Parallel processing
            max_workers = (
                cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
            )

            mask = np.array(
                process_map(
                    func_parallel,
                    image,
                    max_workers=max_workers,
                    desc="Thresholding image",
                )
            )
    else:
        # Single image processing
        mask = _compute_mask(
            image,
            method=method,
            sigma_blur=sigma_blur,
            threshold_factor=threshold_factor,
            post_processing_method=post_processing_method,
            keep_largest_cc=keep_largest_cc,
            registered_image=registered_image,
        )

    return mask


def local_image_equalization(
    image: np.ndarray,
    box_size: int,
    perc_low: float,
    perc_high: float,
    mask: np.ndarray = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Performs local image equalization on either a single image or a temporal stack of images.
    Stretches the image histogram in local neighborhoods by remapping intesities in the range
    [perc_low, perc_high] to the range [0, 1].
    This helps to enhance the contrast and improve the visibility of structures in the image.

    Parameters:
    - image: numpy array, input image or temporal stack of images
    - box_size: int, size of the local neighborhood for equalization
    - perc_low: float, lower percentile for intensity equalization
    - perc_high: float, upper percentile for intensity equalization
    - mask: numpy array, binary mask used to set the background to zero (optional)
    - n_jobs: int, number of parallel jobs to use (not used currently as the function is parallelized internally)

    Returns:
    - image_norm: numpy array, equalized image or stack of equalized images
    """

    is_temporal = image.ndim == 4

    mask_is_None = mask is None

    if is_temporal:
        if mask_is_None:
            mask = [None] * image.shape[0]

        # Apply local equalization to each time frame in the temporal stack
        image_norm = np.array(
            [
                _local_equalization(
                    image[ind_t],
                    box_size=box_size,
                    perc_low=perc_low,
                    perc_high=perc_high,
                    mask=mask[ind_t],
                )
                for ind_t in tqdm(
                    range(image.shape[0]), desc="Local equalization"
                )
            ]
        )
    else:
        # Apply local equalization to the image
        image_norm = _local_equalization(
            image,
            box_size=box_size,
            perc_low=perc_low,
            perc_high=perc_high,
            mask=mask,
        )

    if not mask_is_None:
        # Set the background to zero using the mask
        image_norm = np.where(mask, image_norm, 0.0)

    return np.clip(image_norm, 0, 1)


def normalize_intensity(
    image: np.ndarray,
    ref_image: np.ndarray,
    sigma: float = None,
    mask: np.ndarray = None,
    labels: np.ndarray = None,
    width: int = 3,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the intensity of an image based on a reference image assumed to have
    ideally homogeneous signal (e.g DAPI).

    Parameters:
    - image: numpy array, input image to be normalized
    - ref_image: numpy array, reference image used for normalization
    - sigma: float, standard deviation for Gaussian smoothing of the reference
        image (default: None)
    - mask: numpy array, binary mask of the sample (default: None)
    - labels: numpy array, labels indicating the instances in which the reference
        signal is expressed (default: None)
    - width: int, number of neighboring planes to consider for reference plane
        calculation (default: 3)
    - n_jobs: int, number of parallel jobs to use (-1 for all available CPUs, 1 for
        sequential execution) (default: -1)

    Returns:
    - image_norm: numpy array, normalized input image
    - ref_image_norm: numpy array, normalized reference image
    """

    is_temporal = image.ndim == 4

    if is_temporal:
        if mask is None:
            mask = [None] * image.shape[0]

        if labels is None:
            labels = [None] * image.shape[0]

        if n_jobs == 1:
            # Sequential normalization of each time frame
            normalized_array = [
                _normalize_intensity(
                    im,
                    ref_im,
                    sigma=sigma,
                    mask=ma,
                    labels=lab,
                    width=width,
                )
                for im, ref_im, ma, lab in tqdm(
                    zip(image, ref_image, mask, labels),
                    desc="Normalizing intensity",
                    total=image.shape[0],
                )
            ]

        else:
            # Parallel normalization of each time frame using multiple processes
            func_parallel = partial(
                _normalize_intensity,
                sigma=sigma,
                mask=mask,
                labels=labels,
                width=width,
            )

            max_workers = (
                cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
            )

            normalized_array = process_map(
                func_parallel,
                zip(image, ref_image),
                max_workers=max_workers,
                desc="Normalizing intensity",
                total=image.shape[0],
            )

    else:
        # Single image normalization
        normalized_array = _normalize_intensity(
            image,
            ref_image,
            sigma=sigma,
            mask=mask,
            labels=labels,
            width=width,
        )
    
    return np.array(normalized_array)


def segment_stardist(
    image: np.ndarray,
    model_path: str,
    thresholds_dict: dict = None,
    n_jobs: int = -1,
) -> np.ndarray:
    
    """
    Predict the segmentation of an array using a StarDist model.

    Parameters:
    - image: a 3D numpy array, input image to segment
    - model_path: str, path to the StarDist model folder
    - thresholds_dict: dict, dictionary of thresholds for the model, structured like
      {'prob': 0.5, 'nms': 0.3} for probability and non-maximum suppression thresholds
      respectively (default: None)
    - n_jobs: int, not used here but kept for consistency with other functions
    """

    is_temporal = image.ndim == 4

    if is_temporal:
        labels = np.array(
            [
                _segment_stardist(
                    im,
                    model_path=model_path,
                    thresholds_dict=thresholds_dict,
                )
                for im in tqdm(image, desc="Predicting with StarDist")
            ],
            dtype=np.uint16
        )

    else:
        labels = _segment_stardist(
            image,
            model_path=model_path,
            thresholds_dict=thresholds_dict,
        )

    return labels


def align_array_major_axis(
    target_axis: str,
    rotation_plane: str,
    mask: np.ndarray,
    array: np.ndarray,
    order: int = 1,
    temporal_slice: Optional[int] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Aligns the major axis of an array to a target axis in a specified rotation plane.
    This function uses Principal Component Analysis (PCA) to determine the major axis of the array,
    and then rotates the array to align the major axis with the target axis.

    Parameters:
    - target_axis: str, the target axis to align the major axis with ('X', 'Y', or 'Z')
    - rotation_plane: str, the rotation plane to perform the rotation in ('XY', 'XZ', or 'YZ')
    - array: numpy array
    - order: int, order of interpolation for image rotation (default: 1)
    - temporal_slice: int, optional temporal slicing applied to the mask before computing its major axis (default: None)
    - n_jobs: int, number of parallel jobs to use (-1 for all available CPUs, 1 for sequential execution) (default: -1)

    Returns:
    - rotated_array: numpy array, array with the major axis aligned to the target axis
    """

    is_temporal = mask.ndim == 4

    temporal_slice = slice(None) if temporal_slice is None else temporal_slice

    # Compute the mask that will be used to compute the major axis.
    # If the mask is a temporal array, the major axis is computed by aggregating the mask
    # instances in temporal_slice.
    mask_for_pca = (
        np.any(mask[temporal_slice], axis=0) if is_temporal else mask
    )

    # Compute the rotation angle and the indices of the rotation plane
    rotation_angle, rotation_plane_indices = (
        _compute_rotation_angle_and_indices(
            mask_for_pca, target_axis, rotation_plane
        )
    )

    if array.dtype == bool:
        order = 0

    # Define the rotation functions
    func_rotate = partial(
        rotate,
        angle=rotation_angle,
        axes=rotation_plane_indices,
        reshape=True,
        order=order,
    )

    func_rotate_mask = partial(
        rotate,
        angle=rotation_angle,
        axes=rotation_plane_indices,
        reshape=True,
        order=0,
    )

    if is_temporal and n_jobs != 1:
        # Rotate the arrays in parallel if the array is temporal and parallel execution is enabled
        max_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

        array_rotated = np.array(
            process_map(
                func_rotate,
                array,
                max_workers=max_workers,
                desc="Aligning array",
            )
        )

        mask_rotated = np.array(
            process_map(
                func_rotate_mask,
                mask,
                max_workers=max_workers,
                desc="Aligning mask",
            )
        )

    else:
        # Rotate the arrays in block
        array_rotated = func_rotate(array)
        mask_rotated = func_rotate_mask(mask)

    array_rotated = np.where(mask_rotated, array_rotated, 0)

    return array_rotated


def _load_array_rotate_and_save_to_file(
    array_file: str,
    mask_file: str,
    index: int,
    path_to_save: str,
    rotation_angle: float,
    order: int,
    rotation_plane_indices: tuple[int, int],
    compress_params: dict,
):
    array = tifffile.imread(array_file)
    mask = tifffile.imread(mask_file)
    array_rotated = rotate(
        array, angle=rotation_angle, axes=rotation_plane_indices, reshape=True, order=order
    )
    mask_rotated = rotate(
        mask, angle=rotation_angle, axes=rotation_plane_indices, 
        reshape=True, order=0
    )
    array_rotated = np.where(mask_rotated, array_rotated, 0)
    if order > 1:
        # preserve the original intensity range
        array_rotated = np.clip(array_rotated, array.min(), array.max())
        
    tifffile.imwrite(
        f"{path_to_save}/aligned_{index:>04}.tif",
        array_rotated,
        **compress_params
    )


def align_array_major_axis_from_files(
    mask_files: list[str],
    array_files: list[str],
    path_to_save: str,
    compress_params: dict,
    func_params: dict, 
):
    """
    Aligns the major axis of an array to a target axis in a specified rotation plane.
    This function uses Principal Component Analysis (PCA) to determine the major axis of the array,
    and then rotates the array to align the major axis with the target axis.
    """

    mask_zyx_shape = tifffile.imread(mask_files[0]).shape
    mask_for_pca = np.zeros(mask_zyx_shape[:-3], dtype=bool)

    for mask_file in mask_files:
        mask = tifffile.imread(mask_file)
        mask_for_pca = np.logical_or(mask_for_pca, mask)

    target_axis = func_params.get("target_axis", "Z")
    rotation_plane = func_params.get("rotation_plane", "XY")
    order = func_params.get("order", 1)

    # Compute the rotation angle and the indices of the rotation plane
    rotation_angle, rotation_plane_indices = (
        _compute_rotation_angle_and_indices(
            mask_for_pca, target_axis, rotation_plane
        )
    )

    multithreaded_function = partial(
        _load_array_rotate_and_save_to_file,
        path_to_save=path_to_save,
        order=order,
        rotation_angle=rotation_angle,
        rotation_plane_indices=rotation_plane_indices,
        compress_params=compress_params,
    )

    n_jobs = func_params.get("n_jobs", -1)

    # open all array files using the multithreading library and crop the results
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        res = list(tqdm(
            executor.map(multithreaded_function, array_files, mask_files, range(len(array_files))),
            total=len(array_files),
            desc="Aligning array",
        ))
    


def crop_array_using_mask(
    mask: np.ndarray,
    array: np.ndarray,
    margin: int = 0,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Crop an array using a binary mask. If the array is temporal, the cropping
    slice is computed by aggregating mask instances at all times.

    Parameters:
    - mask: numpy array, binary mask indicating the region of interest
    - array: numpy array, array to crop based on the mask
    - margin: int, optional margin to add around the mask (default: 0)
    - n_jobs: int, number of parallel jobs to use (not used currently as the function is not computationally intensive)

    Returns:
    - cropped_array: numpy array, cropped array based on the mask
    """

    is_temporal = mask.ndim == 4

    # Compute aggregated mask if array is temporal
    mask_for_slice = np.any(mask, axis=0) if is_temporal else mask

    # Get the mask slice
    mask_slice = regionprops(mask_for_slice.astype(int))[0].slice

    mask_zyx_shape = mask.shape[1:] if is_temporal else mask.shape

    # Add margin to the slice if specified
    if margin > 0:
        mask_slice = tuple(
            slice(
                max(0, mask_slice[i].start - margin),
                min(mask_slice[i].stop + margin, mask_zyx_shape[i]),
            )
            for i in range(3)
        )

    mask_slice = (slice(None),) + mask_slice if is_temporal else mask_slice

    # Apply the slice to the array
    array_cropped = array[mask_slice]

    return array_cropped


def _extract_slice_from_file(file: str):
    """
    Extract the bounding box from a file containing a binary mask.
    """

    mask = tifffile.imread(file)
    mask_slice = regionprops(mask.astype(int))[0].slice

    return mask_slice


def _load_array_crop_and_save_to_file(
    array_file: str,
    index: int,
    path_to_save: str,
    mask_slice: tuple[slice],
    compress_params: dict,
):
    array = tifffile.imread(array_file)[mask_slice]
    tifffile.imwrite(
        f"{path_to_save}/cropped_{index:>04}.tif",
        array,
        **compress_params
    )


def crop_array_using_mask_from_files(
    mask_files: list[str],
    array_files: list[str],
    path_to_save: str,
    compress_params: dict,
    func_params: dict,
) -> np.ndarray:
    """
    Crop an array using a binary mask. If the array is temporal, the cropping
    slice is computed by aggregating mask instances at all times.
    """

    mask_zyx_shape = tifffile.imread(mask_files[0]).shape

    # open all mask files using the multithreading library
    mask_slices = []

    n_jobs = func_params.get("n_jobs", -1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        mask_slices = list(
            tqdm(
                executor.map(_extract_slice_from_file, mask_files),
                total=len(mask_files),
                desc="Extracting mask slices",
            )
        )

    # aggregate the mask slices
    mask_slice = tuple(
        slice(
            min([mask_slice[i].start for mask_slice in mask_slices]),
            max([mask_slice[i].stop for mask_slice in mask_slices]),
        )
        for i in range(3)
    )

    margin = func_params.get("margin", 0)

    # Add margin to the slice if specified
    if margin > 0:
        mask_slice = tuple(
            slice(
                max(0, mask_slice[i].start - margin),
                min(mask_slice[i].stop + margin, mask_zyx_shape[i]),
            )
            for i in range(3)
        )

    multithreaded_function = partial(
        _load_array_crop_and_save_to_file,
        path_to_save=path_to_save,
        mask_slice=mask_slice,
        compress_params=compress_params,
    )
    # open all array files using the multithreading library and crop the results
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        res = list(tqdm(
            executor.map(multithreaded_function, array_files, range(len(array_files))),
            total=len(array_files),
            desc="Cropping array",
        ))


def _parallel_gaussian_smooth(
    input_tuple: tuple[np.ndarray, np.ndarray],
    sigmas: Union[float, list[float]],
) -> np.ndarray:
    data, mask, mask_for_volume = input_tuple
    return _masked_smooth_gaussian(data, sigmas, mask, mask_for_volume)


def masked_gaussian_smooth(
    image: np.ndarray,
    sigmas: Union[float, list[float]],
    mask: Optional[np.ndarray] = None,
    mask_for_volume: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to an image or a sequence of images.

    Parameters:
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
                zip(image, mask, mask_for_volume),
                total=len(image),
                desc="Smoothing image",
            )

            return np.array([func(elem) for elem in iterable])

        else:
            elems = [elem for elem in zip(image, mask)]

            max_workers = (
                cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
            )
            result = process_map(
                func, elems, max_workers=max_workers, desc="Smoothing image"
            )

            return np.array(result)

    else:
        return _masked_smooth_gaussian(image, sigmas, mask, mask_for_volume)


def masked_gaussian_smooth_dense_two_arrays_gpu(
    datas: list[np.ndarray],
    sigmas: Union[float, list[float]],
    mask: np.ndarray = None,
    masks_for_volume: Union[np.ndarray, list[np.ndarray]] = None,
):
    """
    Inputs in data are assumed to be non-temporal, 3D arrays.
    """

    from pyclesperanto_prototype import gaussian_blur

    if isinstance(sigmas, int | float):
        sigmas = [sigmas] * 3

    if mask is None:
        smoothed1 = np.array(
            gaussian_blur(
                datas[0].astype(np.float16),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            )
        )
        smoothed2 = np.array(
            gaussian_blur(
                datas[1].astype(np.float16),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            )
        )

    elif masks_for_volume is None:

        smoothed1 = np.array(
            gaussian_blur(
                np.where(mask, datas[0].astype(np.float16), 0.0),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            )
        )

        smoothed2 = np.array(
            gaussian_blur(
                np.where(mask, datas[1].astype(np.float16), 0.0),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            )
        )

        effective_volume = np.array(
            gaussian_blur(
                mask.astype(np.float16),
                sigma_x=sigmas[0],
                sigma_y=sigmas[1],
                sigma_z=sigmas[2],
            )
        )

        smoothed1 = np.where(
            mask, np.divide(smoothed1, effective_volume, where=mask), 0.0
        )

        smoothed2 = np.where(
            mask, np.divide(smoothed2, effective_volume, where=mask), 0.0
        )

    else:  # both masks and masks_for_volume are not None

        if isinstance(masks_for_volume, np.ndarray):

            smoothed1 = np.array(
                gaussian_blur(
                    np.where(
                        masks_for_volume, datas[0].astype(np.float16), 0.0
                    ),
                    sigma_x=sigmas[0],
                    sigma_y=sigmas[1],
                    sigma_z=sigmas[2],
                )
            )

            smoothed2 = np.array(
                gaussian_blur(
                    np.where(
                        masks_for_volume, datas[1].astype(np.float16), 0.0
                    ),
                    sigma_x=sigmas[0],
                    sigma_y=sigmas[1],
                    sigma_z=sigmas[2],
                )
            )

            effective_volume = np.array(
                gaussian_blur(
                    masks_for_volume.astype(np.float16),
                    sigma_x=sigmas[0],
                    sigma_y=sigmas[1],
                    sigma_z=sigmas[2],
                )
            )

            smoothed1 = np.where(
                mask, np.divide(smoothed1, effective_volume, where=mask), 0.0
            )

            smoothed2 = np.where(
                mask, np.divide(smoothed2, effective_volume, where=mask), 0.0
            )

        elif isinstance(masks_for_volume, list):

            smoothed1 = np.array(
                gaussian_blur(
                    np.where(
                        masks_for_volume[0], datas[0].astype(np.float16), 0.0
                    ),
                    sigma_x=sigmas[0],
                    sigma_y=sigmas[1],
                    sigma_z=sigmas[2],
                )
            )

            smoothed2 = np.array(
                gaussian_blur(
                    np.where(
                        masks_for_volume[1], datas[1].astype(np.float16), 0.0
                    ),
                    sigma_x=sigmas[0],
                    sigma_y=sigmas[1],
                    sigma_z=sigmas[2],
                )
            )

            effective_volume1 = np.array(
                gaussian_blur(
                    masks_for_volume[0].astype(np.float16),
                    sigma_x=sigmas[0],
                    sigma_y=sigmas[1],
                    sigma_z=sigmas[2],
                )
            )

            effective_volume2 = np.array(
                gaussian_blur(
                    masks_for_volume[1].astype(np.float16),
                    sigma_x=sigmas[0],
                    sigma_y=sigmas[1],
                    sigma_z=sigmas[2],
                )
            )

            smoothed1 = np.where(
                mask, np.divide(smoothed1, effective_volume1, where=mask), 0.0
            )

            smoothed2 = np.where(
                mask, np.divide(smoothed2, effective_volume2, where=mask), 0.0
            )

    return smoothed1, smoothed2


def masked_gaussian_smooth_sparse(
    sparse_array: Union[np.ndarray, list[np.ndarray]],
    is_temporal: bool,
    dim_space: int,
    sigmas: Union[float, tuple[float]],
    positions: np.ndarray = None,
    n_job: int = -1,
    progress_bars: bool = True,
):
    """
    Smooth sparse data using a gaussian kernel.
    Sparse data is a numpy array with the first columns being the spatial coordinates,
    and the last columns being values of interest.

    Parameters
    ----------
    sparse_array : np.ndarray or list of np.ndarray
        Array of points to smooth in format (n_points, n_dim_space + n_dim_points). 
        The first columns (up to dim_space, i.e [:dim_space]) must be the spatial coordinates. 
        The remaining columns are the values/vectors to smooth. A temporal sparse array is a 
        list of sparse arrays, one for each time point.
    is_temporal : bool
        If True, the array is temporal and the smoothing is applied to each time step.
    dim_space : int
        Number of spatial dimensions.
    sigmas : float or list of float
        Standard deviations of the gaussian kernel.
    positions : np.ndarray, optional
        Positions where the smoothing is applied. If None, the smoothing is applied to the
        positions of the input array.
    mask : np.ndarray, optional
        Mask to apply to the positions. If None, no mask is applied.
    n_job : int, optional
        Number of jobs to run in parallel. If -1, all the available CPUs are used.
        The default is -1.

    Returns
    -------
    np.ndarray
        The smoothed sparse array.
    """

    positions_is_temporal = isinstance(positions, list)

    if is_temporal:

        if n_job == 1:

            if positions_is_temporal:
                return np.array(
                    [
                        _masked_smooth_gaussian_sparse(
                            elem, pos, sigmas, dim_space
                        )
                        for elem, pos in tqdm(
                            zip(
                                sparse_array,
                                positions,
                                disable=not progress_bars,
                            )
                        )
                    ]
                )

            else:
                return np.array(
                    [
                        _masked_smooth_gaussian_sparse(
                            elem, positions, sigmas, dim_space
                        )
                        for elem in tqdm(
                            sparse_array, disable=not progress_bars
                        )
                    ]
                )

        else:
            if not positions_is_temporal:
                # same positions for all time steps
                func = partial(
                    _masked_smooth_gaussian_sparse,
                    sigmas=sigmas,
                    dim_space=dim_space,
                    positions=positions,
                )

                max_workers = (
                    cpu_count() if n_job == -1 else min(n_job, cpu_count())
                )
                result = np.array(
                    process_map(
                        func,
                        sparse_array,
                        max_workers=max_workers,
                        disable=not progress_bars,
                    )
                )

            else:
                func = partial(
                    _masked_smooth_gaussian_sparse,
                    sigmas=sigmas,
                    dim_space=dim_space,
                )

                max_workers = (
                    cpu_count() if n_job == -1 else min(n_job, cpu_count())
                )
                result = process_map(
                    func,
                    sparse_array,
                    positions,
                    max_workers=max_workers,
                    disable=not progress_bars,
                )

            return result

    else:
        return _masked_smooth_gaussian_sparse(
            array=sparse_array,
            positions=positions,
            sigmas=sigmas,
            dim_space=dim_space,
        )

