import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import convex_hull_image
from skimage.transform import rescale, resize

from tapenade.preprocessing._smoothing import _masked_smooth_gaussian


def _snp_threshold_binarization(
    image: np.ndarray,
    sigma_blur: float,
    threshold_factor: float,
    registered_image: bool = False,
) -> np.ndarray:
    """
    Threshold image based on signal-and-noise product.

    Parameters:
    - image: numpy array, input image
    - sigma_blur: float, standard deviation of the Gaussian blur.
    - threshold_factor: float, factor to multiply the threshold
    - registered_image: bool, set to True if the image has been registered beforehand
      and thus contains large regions of zeros that lead to sharp intensity gradients
      at the boundaries.

    Returns:
    - binary_mask: numpy array, binary mask computed using SNP thresholding
    """

    if registered_image:
        nonzero_mask = image > 0

        blurred = _masked_smooth_gaussian(image, mask=nonzero_mask, sigmas=sigma_blur)

        blurred2 = _masked_smooth_gaussian(
            image**2, mask=nonzero_mask, sigmas=sigma_blur
        )

        sigma = blurred2 - blurred**2

        snp_array = sigma * blurred
        snp_mask = snp_array > 0

        snp_array = np.log(
            snp_array, where=np.logical_and(nonzero_mask, snp_mask)
        )
    else:
        blurred = _masked_smooth_gaussian(image, sigmas=sigma_blur)
        blurred2 = _masked_smooth_gaussian(image**2, sigmas=sigma_blur)

        sigma = blurred2 - blurred**2

        snp_array = sigma * blurred
        snp_mask = snp_array > 0
        # snp_array = np.log(snp_array, where=(snp_array != 0))

        snp_array = np.log(snp_array, where=snp_mask)

    threshold = threshold_otsu(snp_array[snp_mask]) * threshold_factor

    # Create a binary mask
    binary_mask = np.logical_and(snp_array > threshold, snp_mask)

    return binary_mask


def _otsu_threshold_binarization(
    image: np.ndarray, sigma_blur: float, threshold_factor: float
) -> np.ndarray:
    """
    Threshold image based on histogram values.

    Parameters:
    - image: numpy array, input image
    - sigma_blur: float, standard deviation of the Gaussian blur.
    - threshold_factor: float, factor to multiply the threshold

    Returns:
    - binary_mask: numpy array, binary mask computed using histogram thresholding
    """

    blurred = gaussian_filter(image, sigma=sigma_blur)

    threshold = threshold_otsu(blurred) * threshold_factor

    # Create a binary mask
    binary_mask = blurred > threshold

    return binary_mask


### POST-PROCESSING FUNCTIONS
def _get_largest_connected_component(array: np.ndarray) -> np.ndarray:
    """
    Get the largest connected component in a binary array.

    Parameters:
    - array: Binary array.

    Returns:
    - mask_largest_cc: Binary mask of the largest connected component.
    """
    labels = label(array, connectivity=1)
    mask_largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return mask_largest_cc


def _compute_mask_convex_hull(
    mask: np.ndarray, precision_factor: int
) -> np.ndarray:
    """
    Compute the convex hull of the binary mask.

    Parameters:
    - mask: numpy array, binary mask
    - precision_factor: int, factor to rescale the mask before computing the convex hull.
      Bigger values will result in faster computation but less precise convex hulls.

    Returns:
    - hull_mask: numpy array, binary mask representing the convex hull of the input mask
    """

    hull_mask = mask.copy()

    if precision_factor != 1:
        # Rescale the mask if precision_factor is not 1
        hull_mask = rescale(
            hull_mask,
            1 / precision_factor,
            anti_aliasing=False,
            order=0,
            preserve_range=True,
        )

    # Compute the convex hull of the mask
    hull_mask = convex_hull_image(hull_mask)

    if precision_factor != 1:
        # Resize the convex hull mask to the original shape
        hull_mask = resize(
            hull_mask,
            mask.shape,
            anti_aliasing=False,
            order=0,
            preserve_range=True,
        )

    return hull_mask


def _binary_fill_holes_on_each_slice(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask on each slice. Apply the binary_fill_holes function
    to each slice of the input mask twice sequentially.

    Parameters:
    - mask: numpy array, binary mask

    Returns:
    - mask: numpy array, binary mask with holes filled on each slice
    """

    filled_mask = mask.copy()

    for i in range(filled_mask.shape[2]):
        filled_mask[:, :, i] = binary_fill_holes(filled_mask[:, :, i])
    for i in range(filled_mask.shape[1]):
        filled_mask[:, i] = binary_fill_holes(filled_mask[:, i])
    for i in range(mask.shape[0]):
        filled_mask[i] = binary_fill_holes(filled_mask[i])
    # for i in range(filled_mask.shape[0]):
    #     filled_mask[i] = binary_fill_holes(filled_mask[i])
    # for i in range(filled_mask.shape[1]):
    #     filled_mask[:, i] = binary_fill_holes(filled_mask[:, i])
    # for i in range(filled_mask.shape[2]):
    #     filled_mask[:, :, i] = binary_fill_holes(filled_mask[:, :, i])

    return filled_mask


def _refine_raw_mask(
    mask: np.ndarray, post_processing_method: str, keep_largest_cc: bool
) -> np.ndarray:
    """
    Refine the raw mask by keeping only the largest connected component and filling holes.

    Parameters:
    - mask: numpy array, binary mask
    - post_processing_method: str, method to use for post-processing the mask. Can be 'convex_hull' to compute
      the convex hull of the mask, 'fill_holes' to fill holes in each 2D slice of the mask, or 'none' to skip
    - keep_largest_cc: bool, set to True to keep only the largest connected component in the mask.

    Returns:
    - refined_mask: numpy array, refined binary mask
    """
    # Keep only the largest connected component
    if keep_largest_cc:
        refined_mask = _get_largest_connected_component(mask)
    # Fill holes in the mask
    if post_processing_method == "convex_hull":
        refined_mask = _compute_mask_convex_hull(refined_mask, 3)
    elif post_processing_method == "fill_holes":
        refined_mask = binary_fill_holes(refined_mask)
        refined_mask = _binary_fill_holes_on_each_slice(refined_mask)
    else:
        pass

    return refined_mask


###


def _compute_mask(
    image: np.ndarray,
    method: str,
    sigma_blur: float,
    threshold_factor: float = 1,
    keep_largest_cc: bool = True,
    post_processing_method: str = "fill_holes",
    registered_image: bool = False,
) -> np.ndarray:
    """
    Process the mask for the given image.

    Parameters:
    - image: numpy array, input image
    - method: str, method to use for thresholding. Can be 'snp otsu' for Signal-Noise Product thresholding,
      or 'otsu' for Otsu's thresholding.
    - sigma_blur: float, standard deviation of the Gaussian blur. Should typically be
      around 1/3 of the typical object diameter.
    - threshold_factor: float, factor to multiply the threshold
    - keep_largest_cc: bool, set to True to keep only the largest connected component in the mask.
    - post_processing_method: str, method to use for post-processing the mask. Can be 'convex_hull' to compute
      the convex hull of the mask, 'fill_holes' to fill holes in each 2D slice of the mask, or 'none' to skip
    - registered_image: bool, set to True if the image has been registered beforehand and thus
      contains large regions of zeros that lead to sharp intensity gradients at the boundaries.

    Returns:
    - mask: numpy array, binary mask of the same shape as the input image
    """

    # Normalize the image
    percs = np.nanpercentile(image, [1, 99])
    im = (image - percs[0]) / (percs[1] - percs[0])
    im = np.clip(im, 0, 1).astype(np.float32)

    # Compute the mask
    if method == "snp otsu":
        mask = _snp_threshold_binarization(
            im / im.max(), sigma_blur, threshold_factor, registered_image
        )
    elif method == "otsu":
        mask = _otsu_threshold_binarization(
            im / im.max(), sigma_blur, threshold_factor
        )
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Refine the mask by keeping only the largest connected component and filling holes
    if keep_largest_cc or post_processing_method != "none":
        mask = _refine_raw_mask(mask, post_processing_method, keep_largest_cc)

    return mask
