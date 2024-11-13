import numpy as np



def tensors_to_napari_vectors(
        sparse_tensors: np.ndarray, 
        is_inertia_tensor: bool, 
        volumes: np.ndarray = None, 
        return_angles: bool = False
    ):
    """
    Extracts the largest eigenvalue and eigenvector from a set of sparse tensors.
    Returns the product of the eigenvalues and eigenvectors as a set of vector
    array ready to be plotted in napari.

    Parameters
    ----------
    sparse_tensors : np.ndarray
        A 2D array of shape (n_tensors, dim_space + dim_space**2) where the first
        dim_space columns are the positions at which the tensors are defined and
        the next dim_space**2 columns are the tensors values.
    is_inertia_tensor : bool
        If True, the function will process the eigenvalues of the tensors in a
        special manner to extract the principal lengths. Otherwise, the tensors
        are simply diagonalized.
    volumes : np.ndarray, optional
        The volumes of the regions associated with the tensors. It is only useful
        (and necessary) to provide it wien is_inertia_tensor is True.
    return_angles : bool, optional
        If True, the function will return the angles of the principal vectors with
        respect to the X-axis.

    Returns
    -------
    napari_vectors : np.ndarray
        A 3D array V of shape (n_tensors, 2, dim_space) where V[i,0] is the
        position at which the vector is defined and V[i,1] is the vector itself.
    angles : np.ndarray
        The angles of the principal vectors with respect to the X-axis. Only
        returned if return_angles is True.
    """

    positions = sparse_tensors[:, :3]
    tensors = sparse_tensors[:, 3:]
    tensors = tensors.reshape(-1, 3, 3)

    # diagonalize the tensors
    eigen_values, principal_vectors = np.linalg.eigh(tensors)

    if is_inertia_tensor:
        # the principal lengths are a mix of the eigenvalues and the volumes
        axis_decoupling_matrix = np.ones((3,3))/2 - np.eye(3)
        eigen_values = np.sqrt(
            np.einsum('ij,lj->li', axis_decoupling_matrix, eigen_values)
        )
        eigen_values = eigen_values * np.sqrt(5 / volumes.reshape(-1, 1))
    
    # WARNING: the scipy documentation is wrong, the eigenvectors are in the columns
    # not the lines, i.e principal_vectors[:,:,0] is the 0th principal vector
    principal_vectors = principal_vectors.transpose(0,2,1)

    napari_vectors = np.zeros((len(positions), 2, 3))
    indices_maxs = np.nanargmax(eigen_values, axis=1)
    principal_vectors = principal_vectors[np.arange(len(eigen_values)), indices_maxs]
    eigen_values = eigen_values[np.arange(len(eigen_values)), indices_maxs]
    napari_vectors[:,0] = positions
    napari_vectors[:,1] = principal_vectors * eigen_values.reshape(-1,1)

    if return_angles:
        angles = np.arctan2(*(napari_vectors[:,1, -2:].reshape(-1, 2).T))
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        return napari_vectors, angles


    return napari_vectors
