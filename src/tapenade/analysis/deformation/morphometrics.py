import numpy as np
from skimage.measure import regionprops, label
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion
from tapenade.analysis.deformation.additional_regionprops_properties import (
    add_principal_lengths,
    add_ellipsoidal_coefficients,
)
from tapenade.preprocessing import masked_gaussian_smoothing
from tapenade.preprocessing._smoothing import _masked_smooth_gaussian


def clear_borders(labels, buffer_size=0, bgval=0, mask=None, *, out=None):
    ## from skimage.segmentation
    """Clear objects connected to the label image border.

    Parameters
    ----------
    labels : (M[, N[, ..., P]]) array of int or bool
        Imaging data labels.
    buffer_size : int, optional
        The width of the border examined.  By default, only objects
        that touch the outside of the image are removed.
    bgval : float or int, optional
        Cleared objects are set to this value.
    mask : ndarray of bool, same shape as `image`, optional.
        Image data mask. Objects in labels image overlapping with
        False pixels of mask will be removed. If defined, the
        argument buffer_size will be ignored.
    out : ndarray
        Array of the same shape as `labels`, into which the
        output is placed. By default, a new array is created.

    Returns
    -------
    out : (M[, N[, ..., P]]) array
        Imaging data labels with cleared borders

    """
    if any(buffer_size >= s for s in labels.shape) and mask is None:
        # ignore buffer_size if mask
        raise ValueError("buffer size may not be greater than labels size")

    if out is None:
        out = labels.copy()

    if mask is not None:
        err_msg = (
            f"labels and mask should have the same shape but "
            f"are {out.shape} and {mask.shape}"
        )
        if out.shape != mask.shape:
            raise (ValueError, err_msg)
        if mask.dtype != bool:
            raise TypeError("mask should be of type bool.")
        borders = ~mask
    else:
        # create borders with buffer_size
        borders = np.zeros_like(out, dtype=bool)
        ext = buffer_size + 1
        slstart = slice(ext)
        slend = slice(-ext, None)
        slices = [slice(None) for _ in out.shape]
        for d in range(out.ndim):
            slices[d] = slstart
            borders[tuple(slices)] = True
            slices[d] = slend
            borders[tuple(slices)] = True
            slices[d] = slice(None)

    # Re-label, in case we are dealing with a binary out
    # and to get consistent labeling
    labels, number = label(out, background=0, return_num=True)

    # determine all objects that are connected to borders
    borders_indices = np.unique(labels[borders])
    indices = np.arange(number + 1)
    # mask all label indices that are connected to borders
    label_mask = np.isin(indices, borders_indices)
    # create mask for pixels to clear
    mask = label_mask[labels.reshape(-1)].reshape(labels.shape)

    # clear border pixels
    out[mask] = bgval

    return out


def smooth_cellular_map(
    cellular,
    props,
    mask,
    sigma,
):
    # takes a cellular map (one value per segmented nuclei) and apply a gaussian smoothing

    centroids = np.array([prop.centroid for prop in props])
    centroids_data = np.zeros_like(mask, dtype=np.float32)
    centroids_data[tuple(centroids.T.astype(int))] = 1

    smoothed = masked_gaussian_smoothing(
        cellular,
        mask_for_volume=centroids_data,
        sigmas=sigma,
        mask=mask,
        n_jobs=-1,
    )
    return smoothed


def process_genetic_field_cellular(labels, genetic_field):
    # return cellular map (one value per segmented nuclei) of mean gene expression in that nucleus
    props_genetic_field = regionprops(labels, intensity_image=genetic_field)
    cellular_genetic_field = np.zeros_like(labels, dtype=np.float32)
    for prop in props_genetic_field:
        cellular_genetic_field[prop.slice][prop.image] = prop.mean_intensity
    return cellular_genetic_field


def process_radial_distance_cellular(mask, labels):
    # return cellular map (one value per segmented nuclei) of radial distance to the tissue surface in 3D
    radial_distance_cellular = np.zeros_like(mask, dtype=np.float32)
    distance_transform = np.zeros_like(mask, dtype=np.float32)
    props = regionprops(labels)
    for z in range(len(mask)):
        distance_transform[z] = distance_transform_edt(mask[z].astype(bool))

    for prop in props:
        dist = distance_transform[tuple(np.array(prop.centroid).astype(int))]
        radial_distance_cellular[prop.slice][prop.image] = dist
    return radial_distance_cellular


def process_axial_distance_cellular(mask, labels):
    # return cellular map (one value per segmented nuclei) of axial distance along z to the first plane in 3D
    axial_distance_cellular = np.zeros_like(mask, dtype=np.float32)

    props = regionprops(labels)
    first_x = np.min([prop.centroid[2] for prop in props])
    for prop in props:
        dist = prop.centroid[2] - first_x
        axial_distance_cellular[prop.slice][prop.image] = dist

    return axial_distance_cellular


def process_cell_density_sigma(
    props,
    mask,
    sigma,
):
    centroids = np.array([prop.centroid for prop in props])
    centroids_data = np.zeros_like(mask, dtype=np.float32)
    centroids_data[tuple(centroids.T.astype(int))] = 1

    density = masked_gaussian_smoothing(
        centroids_data,
        sigmas=sigma,
        mask=mask,
        n_jobs=-1,
    )
    return density


def process_density_gradient(
    mask,
    density,
    positions_on_grid,
):
    gradient_field = np.gradient(density)
    gradient_field = np.array(gradient_field).transpose(1, 2, 3, 0)
    gradient_field[~binary_erosion(mask)] = 0

    gradient_magnitude_field = np.linalg.norm(gradient_field, axis=-1)

    gradient_on_grid = gradient_field[tuple(positions_on_grid.T.astype(int))]

    napari_gradient_on_grid = np.zeros((len(positions_on_grid), 2, 3))

    napari_gradient_on_grid[:, 0] = positions_on_grid
    napari_gradient_on_grid[:, 1] = gradient_on_grid

    angles = np.arctan2(*(napari_gradient_on_grid[:, 1, -2:].reshape(-1, 2).T))
    angles = np.arctan2(np.sin(angles - 1), np.cos(angles - 1))
    # gradient_field can be used to compute dot product between cell density gradient and true strain tensor
    return (
        gradient_field,
        gradient_magnitude_field,
        napari_gradient_on_grid,
        angles,
    )


def process_volume_fraction_sigma(
    mask,
    labels,
    sigma,
):
    volume_data = labels.astype(bool).astype(np.float32)
    volume_fraction = masked_gaussian_smoothing(
        volume_data, sigmas=sigma, mask=mask, n_jobs=-1
    )
    return volume_fraction


def process_nuclear_volume_cellular(
    props,
    mask,
):
    centroids_data_volumes = np.zeros_like(mask, dtype=np.float32)

    for prop in props:
        centroids_data_volumes[prop.slice][prop.image] = prop.area

    return centroids_data_volumes


def process_nematic_order(mask, labels, sigma):
    props = regionprops(labels)
    n_points = len(props)
    n_dim_tensor = 9

    sparse_m_tensor = np.zeros((n_points, n_dim_tensor))
    dense_m_tensor = np.zeros((*mask.shape, n_dim_tensor), dtype=np.float32)
    dense_centroids = np.zeros(mask.shape)

    for index_label, prop in enumerate(props):
        add_principal_lengths(
            prop, add_principal_vectors=True, scale=(1, 1, 1)
        )
        orientation_vector = prop.principal_vectors[0]
        m = np.outer(orientation_vector, orientation_vector)
        sparse_m_tensor[index_label, :] = m.flatten()
        dense_m_tensor[tuple(np.array(prop.centroid).astype(int))] = (
            m.flatten()
        )
        dense_centroids[tuple(np.array(prop.centroid).astype(int))] = 1

    smoothed_dense_m_tensor = np.zeros_like(dense_m_tensor)

    for i in range(9):
        smoothed_dense_m_tensor[..., i] = _masked_smooth_gaussian(
            array=dense_m_tensor[..., i],
            sigmas=sigma,
            mask=mask,
            mask_for_volume=dense_centroids.astype(bool),
        )

    masked_flat = smoothed_dense_m_tensor[mask]
    q_matrices = masked_flat.reshape(-1, 3, 3) - (np.eye(3) / 3)
    if not np.isnan(q_matrices).any():
        eigenvalues = np.linalg.eigvalsh(q_matrices)
        s_values = np.max(eigenvalues, axis=1) * 3 / 2

        s_dense = np.zeros_like(mask, dtype=np.float32)
        s_dense[mask] = s_values
        return s_dense
    else:
        print("NaNs found in q_matrices")
        return 0


def process_ellipsoidal_coeff_cellular(props, mask):

    ellipsoidal_coefficients = np.zeros_like(mask, dtype=np.float32)
    for prop in props:
        if not hasattr(prop, "ellipsoidal_coefficients"):
            prop = add_ellipsoidal_coefficients(prop, scale=(1, 1, 1))
        l1, l2, l3 = prop.principal_lengths
        score = np.log(l2**2 / (l1 * l3))
        ellipsoidal_coefficients[prop.slice][prop.image] = score
    return ellipsoidal_coefficients


def process_oblate_prolate_cellular(props, mask):

    ellipsoidal_coefficients_oblate = np.zeros_like(mask, dtype=np.float32)
    ellipsoidal_coefficients_prolate = np.zeros_like(mask, dtype=np.float32)
    for prop in props:
        if not hasattr(prop, "ellipsoidal_coefficients"):
            prop = add_ellipsoidal_coefficients(prop, scale=(1, 1, 1))
        l1, l2, l3 = prop.principal_lengths
        ellipsoidal_coefficients_oblate[prop.slice][prop.image] = (
            l2 - l3
        ) / np.sqrt(l1 * l3)
        ellipsoidal_coefficients_prolate[prop.slice][prop.image] = (
            l1 - l2
        ) / np.sqrt(l1 * l3)
    return ellipsoidal_coefficients_oblate, ellipsoidal_coefficients_prolate


def process_true_strain_maxeig_cellular(props, mask):
    # Maximum eigenvalue of the true strain tensor of each nuclei, cellular map (one value per segmented nuclei)

    true_strain_maxeig_cellular = np.zeros_like(mask, dtype=np.float32)
    for prop in props:
        if not hasattr(prop, "principal_lengths"):
            prop = add_principal_lengths(prop, scale=(1, 1, 1))
        principal_lengths = prop.principal_lengths
        denominator = np.power(np.prod(principal_lengths), 1 / 3)
        max_eig = np.log(principal_lengths[0] / denominator)

        true_strain_maxeig_cellular[prop.slice][prop.image] = max_eig

    return true_strain_maxeig_cellular
