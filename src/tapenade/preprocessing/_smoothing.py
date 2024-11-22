from typing import Union

import numba
import numpy as np
from scipy.ndimage import gaussian_filter as scipy_gaussian
from scipy.spatial import KDTree


def _masked_smooth_gaussian(
    array: np.ndarray,
    sigmas: Union[float, tuple],
    mask: np.ndarray = None,
    mask_for_volume: np.ndarray = None,
):
    """
    If 'mask' is specified, the convolution will not take the
    masked value into account.
    """

    sigmas = sigmas if isinstance(sigmas, int | float) else np.array(sigmas)

    if mask is None:
        return scipy_gaussian(array, sigmas, mode="nearest", cval=0.0)
    else:

        mask = mask.astype(bool)

        if mask_for_volume is None:

            smooth_array = scipy_gaussian(
                np.where(mask, array.astype(np.float32), 0.0),
                sigmas,
                mode="constant",
                cval=0.0,
                truncate=3.0,
            )

            # calculate renormalization factor for masked gaussian (the 'effective'
            # volume of the gaussian kernel taking the mask into account)
            effective_volume = scipy_gaussian(
                mask.astype(np.float32),
                sigmas,
                mode="constant",
                cval=0.0,
                truncate=3.0,
            )

        else:
            mask_for_volume = mask_for_volume.astype(bool)

            smooth_array = scipy_gaussian(
                np.where(mask_for_volume, array.astype(np.float32), 0.0),
                sigmas,
                mode="constant",
                cval=0.0,
                truncate=3.0,
            )

            # calculate renormalization factor for masked gaussian (the 'effective'
            # volume of the gaussian kernel taking the mask into account)
            effective_volume = scipy_gaussian(
                mask_for_volume.astype(np.float32),
                sigmas,
                mode="constant",
                cval=0.0,
                truncate=3.0,
            )

        smooth_array = np.where(
            mask,
            np.divide(smooth_array, effective_volume, where=mask),
            0.0,
        )

        return smooth_array


@numba.jit(nopython=True, fastmath=True)
def _loop_numba(rows, cols, sparse_array_values, dists, n_points, dim_points):
    """
    Computes the actual smoothing kernel operation from the old to the new positions.
    NEVER SET PARALLEL=True IN NUMBA DECORATOR, IT WILL BREAK THE CODE.
    """

    smoothed_points = np.zeros((n_points, dim_points))
    normalization_values = np.zeros((n_points, 1))

    for i in range(len(rows)): # n_pairs iterations
        exp_dist = np.exp(-dists[i] ** 2 / 2)
        smoothed_points[rows[i]] += sparse_array_values[cols[i]] * exp_dist
        normalization_values[rows[i]] += exp_dist

    for j in range(n_points):
        if normalization_values[j] > 0:
            smoothed_points[j] /= normalization_values[j]

    return smoothed_points, normalization_values



def _masked_smooth_gaussian_sparse(
    sparse_array: np.ndarray,
    positions: np.ndarray,
    sigmas: Union[float, tuple],
    dim_space: int = 3,
):
    """
    Use this function to smooth a sparse array using a Gaussian kernel
    if you expect the output to be a sparse array as well.

    A sparse array is an array of dimension (n_points, n_dim_space + n_dim_points)
    where n_dim_space is the number of spatial dimensions and n_dim_points is the number
    of dimensions of the signal at each point, e.g n_dim_points = 1 for a scalar field,
    n_dim_points = 3 for a 3D vector field, n_dim_points = 9 for a 3x3 matrix field, etc.

    For instance, with a 3D vector field, sparse_array[0] = [z0, y0, x0, v0z, v0y, v0x],
    """

    sigmas = np.array(sigmas).reshape((1, -1))

    old_positions = sparse_array[:, :dim_space]
    positions = positions if positions is not None else old_positions.copy()

    old_tree = KDTree(old_positions / sigmas)
    new_tree = KDTree(positions / sigmas)

    n_dim = sparse_array.shape[1]
    dim_points = n_dim - dim_space
    n_points = positions.shape[0]

    smoothed_array = np.zeros((positions.shape[0], n_dim))
    smoothed_array[:, :dim_space] = positions

    # Find the nearest neighbors of each point
    sparse_dist_matrix = new_tree.sparse_distance_matrix(
        old_tree, max_distance=3, output_type="coo_matrix"
    )
    cols = sparse_dist_matrix.col # old indices, shape (n_pairs,)
    rows = sparse_dist_matrix.row # new indices, shape (n_pairs,)
    dists = sparse_dist_matrix.data # shape (n_pairs,)
    sparse_array_values = sparse_array[:, dim_space:] # shape (n_old, dim_points)

    smoothed_values, norm_values = _loop_numba(
        rows, cols, sparse_array_values, dists, n_points, dim_points
    )

    smoothed_array[:, dim_space:] = smoothed_values

    where_0 = np.where(norm_values == 0)[0]
    if len(where_0) > 0:
        # query 10 nearest neighbors if the point has no neighbors
        k = min(10, old_positions.shape[0])
        dists_nn, indices_nn = old_tree.query(positions[where_0], k=k)

        dists_nn2_2 = dists_nn ** 2 / 2
        # prevent underflow
        dists_nn2_2 = dists_nn2_2 - np.min(dists_nn2_2, axis=1).reshape(-1, 1)
        
        exp_dists = np.exp(-dists_nn2_2).reshape(-1, k, 1)
        norm_values = np.sum(exp_dists, axis=1).reshape(-1, 1)

        values_nn = sparse_array_values[indices_nn.flatten()].reshape(-1, k, dim_points)

        smoothed_values = np.sum(values_nn * exp_dists, axis=1)
        smoothed_values /= norm_values

        smoothed_array[where_0, dim_space:] = smoothed_values

    return smoothed_array