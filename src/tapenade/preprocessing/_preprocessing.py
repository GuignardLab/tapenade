from functools import partial
from os import cpu_count
from typing import Optional, Union

import numpy as np
from scipy.ndimage import rotate
from skimage.measure import regionprops
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tapenade.preprocessing._array_rescaling import _change_arrays_pixelsize
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


def _parallel_change_arrays_pixelsize(arrays, reshape_factors, order):
    mask, image, labels = arrays
    return _change_arrays_pixelsize(
        mask=mask,
        image=image,
        labels=labels,
        reshape_factors=reshape_factors,
        order=order,
    )


def isotropize_and_normalize(
    mask, image, labels, scale, sigma: float = None, pos_ref: int = 0
):
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


def change_arrays_pixelsize(
    mask: np.ndarray = None,
    image: np.ndarray = None,
    labels: np.ndarray = None,
    input_pixelsize: tuple[float, float, float] = (1, 1, 1),
    output_pixelsize: tuple[float, float, float] = (1, 1, 1),
    order: int = 1,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Resizes an input image to have isotropic voxel dimensions.

    Parameters:
    - mask: numpy array, input mask
    - image: numpy array, input image
    - labels: numpy array, input labels
    - input_pixelsize: tuple of floats, input pixel dimensions (e.g. in microns)
    - output_pixelsize: tuple of floats, output pixel dimensions (e.g. in microns)
    - order: int, order of interpolation for resizing (defaults to 1 for
      linear interpolation). Choose 0 for nearest-neighbor interpolation
      (e.g. for label images)
    - n_jobs: int, optional number of parallel jobs for resizing (default: -1)

    Returns:
    - resized_image: numpy array, resized image with isotropic voxel dimensions
    """

    is_temporal = False
    n_frames = 0
    for arr in [mask, image, labels]:
        if arr is not None:
            is_temporal = arr.ndim == 4
            if is_temporal:
                n_frames = arr.shape[0]
            break

    mask_not_None = True
    image_not_None = True
    labels_not_None = True

    if mask is None:
        mask = [None] * n_frames if is_temporal else None
        mask_not_None = False
    if image is None:
        image = [None] * n_frames if is_temporal else None
        image_not_None = False
    if labels is None:
        labels = [None] * n_frames if is_temporal else None
        labels_not_None = False

    if is_temporal:
        if n_jobs == 1:
            # Sequential resizing of each time frame
            resized_arrays = [
                _change_arrays_pixelsize(
                    ma,
                    im,
                    labs,
                    input_pixelsize,
                    output_pixelsize,
                    order=order,
                )
                for ma, im, labs in tqdm(
                    zip(mask, image, labels, strict=False),
                    desc="Making array isotropic",
                    total=n_frames,
                )
            ]

        else:
            # Parallel resizing of each time frame using multiple processes
            func_parallel = partial(
                _parallel_change_arrays_pixelsize,
                input_pixelsize=input_pixelsize,
                output_pixelsize=output_pixelsize,
                order=order,
            )

            max_workers = (
                cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
            )

            resized_arrays = process_map(
                func_parallel,
                zip(mask, image, labels, strict=False),
                max_workers=max_workers,
                desc="Making array isotropic",
                total=n_frames,
            )

    else:
        # Resizing the whole image
        resized_arrays = _change_arrays_pixelsize(
            mask,
            image,
            labels,
            input_pixelsize=input_pixelsize,
            output_pixelsize=output_pixelsize,
            order=order,
        )

    if (
        sum([mask_not_None, image_not_None, labels_not_None]) > 1
        and is_temporal
    ):
        resized_arrays = tuple(
            map(np.array, zip(*resized_arrays, strict=False))
        )
        return resized_arrays
    else:
        return resized_arrays


def compute_mask(
    image: np.ndarray,
    method: str,
    sigma_blur: float,
    threshold_factor: float = 1,
    compute_convex_hull: bool = False,
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
    - compute_convex_hull: bool, set to True to compute the convex hull of the mask. If set to
      False, a hole-filling operation will be performed instead.
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
            compute_convex_hull=compute_convex_hull,
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
            method,
            sigma_blur,
            threshold_factor,
            compute_convex_hull,
            registered_image,
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

    return image_norm


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
            normalized_arrays = [
                _normalize_intensity(
                    im,
                    ref_im,
                    sigma=sigma,
                    mask=ma,
                    labels=lab,
                    width=width,
                )
                for im, ref_im, ma, lab in tqdm(
                    zip(image, ref_image, mask, labels, strict=False),
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

            normalized_arrays = process_map(
                func_parallel,
                zip(image, ref_image, strict=False),
                max_workers=max_workers,
                desc="Normalizing intensity",
                total=image.shape[0],
            )

    else:
        # Single image normalization
        normalized_arrays = _normalize_intensity(
            image,
            ref_image,
            sigma=sigma,
            mask=mask,
            labels=labels,
            width=width,
        )

    if is_temporal:
        normalized_arrays = tuple(
            map(np.array, zip(*normalized_arrays, strict=False))
        )
        return normalized_arrays
    else:
        return normalized_arrays


def align_array_major_axis(
    target_axis: str,
    rotation_plane: str,
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    order: int = 1,
    temporal_slice: Optional[int] = None,
    n_jobs: int = -1,
) -> Union[
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
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

    # Compute the rotation angle and the indices of the rotation plane
    rotation_angle, rotation_plane_indices = (
        _compute_rotation_angle_and_indices(
            mask, target_axis, rotation_plane, temporal_slice
        )
    )

    # Define the rotation functions
    func_rotate_image = partial(
        rotate,
        angle=rotation_angle,
        axes=rotation_plane_indices,
        reshape=True,
        order=order,
    )
    func_rotate = partial(
        rotate,
        angle=rotation_angle,
        axes=rotation_plane_indices,
        reshape=True,
        order=0,
    )

    if is_temporal and n_jobs != 1:
        # Rotate the arrays in parallel if the array is temporal and parallel execution is enabled
        max_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

        mask_rotated = np.array(
            process_map(
                func_rotate,
                mask,
                max_workers=max_workers,
                desc="Aligning mask",
            )
        )
        if image is not None:
            image_rotated = np.array(
                process_map(
                    func_rotate_image,
                    image,
                    max_workers=max_workers,
                    desc="Aligning image",
                )
            )
        if labels is not None:
            labels_rotated = np.array(
                process_map(
                    func_rotate,
                    labels,
                    max_workers=max_workers,
                    desc="Aligning labels",
                )
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


def crop_array_using_mask(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    margin: int = 0,
    n_jobs: int = -1,
) -> Union[
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Crop an array using a binary mask. If the array is temporal, the cropping
    slice is computed by aggregating mask instances at all times.

    Parameters:
    - mask: numpy array, binary mask indicating the region of interest
    - image: numpy array, input image or temporal stack of images (optional)
    - labels: numpy array, labels corresponding to the mask (optional)
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

    # Apply the slice to the arrays
    mask_cropped = mask[mask_slice]
    if image is not None:
        image_cropped = image[mask_slice]
    if labels is not None:
        labels_cropped = labels[mask_slice]

    if image is not None and labels is not None:
        return mask_cropped, image_cropped, labels_cropped
    elif image is not None:
        return mask_cropped, image_cropped
    elif labels is not None:
        return mask_cropped, labels_cropped
    else:
        return mask_cropped


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
                zip(image, mask, mask_for_volume, strict=False),
                total=len(image),
                desc="Smoothing image",
            )

            return np.array([func(elem) for elem in iterable])

        else:
            elems = [elem for elem in zip(image, mask, strict=False)]

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