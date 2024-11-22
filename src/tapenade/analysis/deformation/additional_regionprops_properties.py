import numba
import numpy as np


def add_tensor_moment(regionprops_prop, scale):

    regionprops_prop.tensor_moment = prop_tensor_moment(
        regionprops_prop, scale
    )

    return regionprops_prop


def add_tensor_inertia(regionprops_prop, scale):

    regionprops_prop.tensor_inertia = prop_tensor_inertia(
        regionprops_prop, scale
    )

    return regionprops_prop


def add_principal_lengths(
    regionprops_prop, scale, add_principal_vectors: bool = False
):

    if add_principal_vectors:

        principal_lengths, principal_vectors = prop_principal_lengths(
            regionprops_prop, scale=scale, return_principal_vectors=True
        )

        regionprops_prop.principal_lengths = principal_lengths
        regionprops_prop.principal_vectors = principal_vectors

    else:

        principal_lengths = prop_principal_lengths(
            regionprops_prop, scale=scale, return_principal_vectors=False
        )

        regionprops_prop.principal_lengths = principal_lengths

    return regionprops_prop


def add_ellipsoidal_coefficients(regionprops_prop, scale):

    regionprops_prop.ellipsoidal_coefficients = prop_ellipsoidal_coefficients(
        regionprops_prop, scale
    )

    return regionprops_prop


def add_ellipsoidal_nature(
    regionprops_prop, scale, similarity_threshold: float
):

    regionprops_prop.ellipsoidal_nature = prop_ellipsoidal_nature(
        regionprops_prop, scale, similarity_threshold
    )

    return regionprops_prop


def add_ellipsoidal_nature_bool(regionprops_prop, scale):

    regionprops_prop.ellipsoidal_nature_bool = prop_ellipsoidal_nature_bool(
        regionprops_prop, scale
    )

    return regionprops_prop


def add_anisotropy_coefficient(regionprops_prop, scale):

    regionprops_prop.anisotropy_coefficient = prop_anisotropy_coefficient(
        regionprops_prop, scale
    )

    return regionprops_prop


def add_true_strain_tensor(regionprops_prop, scale):

    regionprops_prop.true_strain_tensor = prop_true_strain_tensor(
        regionprops_prop, scale
    )

    return regionprops_prop


def prop_principal_lengths(
    regionprops_prop, scale, return_principal_vectors: bool = False
):
    """
    See https://math.stackexchange.com/questions/2792009/inertia-tensor-of-an-ellipsoid
    """

    if not hasattr(regionprops_prop, "tensor_inertia"):
        regionprops_prop = add_tensor_inertia(regionprops_prop, scale)

    tensor = regionprops_prop.tensor_inertia
    rescaled_tensor = tensor * 5 / regionprops_prop.area

    axis_decoupling_matrix = np.ones((3, 3)) / 2 - np.eye(3)

    if return_principal_vectors:
        eigen_values, principal_vectors = np.linalg.eigh(rescaled_tensor)
        principal_lengths = np.sqrt(axis_decoupling_matrix @ eigen_values)
        return principal_lengths, principal_vectors.T
    else:
        eigen_values = np.linalg.eigvalsh(rescaled_tensor)
        principal_lengths = np.sqrt(axis_decoupling_matrix @ eigen_values)

        return principal_lengths


@numba.jit(nopython=True, nogil=True)
def fast_tensor_inertia(centered_argwheres):
    """
    We apply the Parallel Axis Theorem on the individual voxels:
    each cubic voxel has an individual diagonal inertia tensor
    I = 1/6 * Identity, so its contribution to the inertia tensor
    of the object (on the diagonal) is 1/6 * d^2, where d is the
    distance of the voxel to the center of mass of the object.
    """

    tensor = np.zeros((3, 3), dtype=np.float32)

    for pos in centered_argwheres:

        tensor[0, 0] += pos[1] * pos[1] + pos[2] * pos[2] + 1 / 6  # y^2 + z^2
        tensor[1, 1] += pos[0] * pos[0] + pos[2] * pos[2] + 1 / 6  # x^2 + z^2
        tensor[2, 2] += pos[0] * pos[0] + pos[1] * pos[1] + 1 / 6  # x^2 + y^2

        tensor[0, 1] += -pos[0] * pos[1]
        tensor[0, 2] += -pos[0] * pos[2]
        tensor[1, 0] += -pos[1] * pos[0]
        tensor[1, 2] += -pos[1] * pos[2]
        tensor[2, 0] += -pos[2] * pos[0]
        tensor[2, 1] += -pos[2] * pos[1]

    return tensor


@numba.jit(nopython=True, nogil=True)
def fast_tensor_moment(centered_argwheres):

    tensor = np.zeros((3, 3), dtype=np.float32)

    for pos in centered_argwheres:

        tensor[0, 0] += pos[0] * pos[0]
        tensor[1, 1] += pos[1] * pos[1]
        tensor[2, 2] += pos[2] * pos[2]

        tensor[0, 1] += pos[0] * pos[1]
        tensor[0, 2] += pos[0] * pos[2]
        tensor[1, 0] += pos[1] * pos[0]
        tensor[1, 2] += pos[1] * pos[2]
        tensor[2, 0] += pos[2] * pos[0]
        tensor[2, 1] += pos[2] * pos[1]

    return tensor


def prop_tensor_inertia(regionprops_prop, scale):

    boolean_image = regionprops_prop.image

    argwheres = np.argwhere(boolean_image) * scale
    center = regionprops_prop.centroid_local * scale

    centered_argwheres = argwheres - center

    return fast_tensor_inertia(centered_argwheres)


def prop_tensor_moment(regionprops_prop, scale):

    boolean_image = regionprops_prop.image

    argwheres = np.argwhere(boolean_image) * scale
    center = regionprops_prop.centroid_local * scale

    centered_argwheres = argwheres - center

    return fast_tensor_moment(centered_argwheres)


def prop_true_strain_tensor(regionprops_prop, scale):

    if not hasattr(regionprops_prop, "principal_vectors"):
        regionprops_prop = add_principal_lengths(
            regionprops_prop, scale, add_principal_vectors=True
        )

    principal_lengths = np.array(regionprops_prop.principal_lengths)
    principal_vectors = regionprops_prop.principal_vectors

    denominator = np.power(np.product(principal_lengths), 1 / 3)
    true_strains = np.log(principal_lengths / denominator)

    tensor = (
        principal_vectors.T
        @ np.diag(true_strains)
        @ np.linalg.inv(principal_vectors.T)
    )

    return tensor


def prop_ellipsoidal_nature_bool(regionprops_prop, scale):

    if not hasattr(regionprops_prop, "ellipsoidal_coefficients"):
        regionprops_prop = add_ellipsoidal_coefficients(
            regionprops_prop, scale
        )

    l1, l2, l3 = regionprops_prop.principal_lengths

    R = (l1 * l2 * l3) ** (1 / 3)

    criterion = np.sum((np.array([l1, l2, l3]) - R) > 0)

    return criterion


def prop_ellipsoidal_nature(
    regionprops_prop, scale, similarity_threshold: float
):
    "oblate: earth = 3; prolate: rugby = 2; spheroid = 4; misc = 1"

    if not hasattr(regionprops_prop, "ellipsoidal_coefficients"):
        regionprops_prop = add_ellipsoidal_coefficients(
            regionprops_prop, scale
        )

    alpha, beta, gamma = regionprops_prop.ellipsoidal_coefficients

    alpha = alpha < similarity_threshold
    beta = beta < similarity_threshold
    gamma = gamma < similarity_threshold

    if alpha and beta and gamma:
        return 4
    elif (not alpha) and beta and (not gamma):
        return 3
    elif (not alpha) and (not beta) and gamma:
        return 2
    else:
        return 1


def prop_ellipsoidal_coefficients(regionprops_prop, scale):

    if not hasattr(regionprops_prop, "principal_lengths"):
        regionprops_prop = add_principal_lengths(regionprops_prop, scale)

    l1, l2, l3 = np.sort(regionprops_prop.principal_lengths)

    alpha = (l3 - l1) / (l1 + l3)
    beta = (l3 - l2) / (l2 + l3)
    gamma = (l2 - l1) / (l1 + l2)

    return alpha, beta, gamma


def prop_anisotropy_coefficient(regionprops_prop, scale):
    """
    Ratio of the longest to the shortest principal axis.
    """
    if not hasattr(regionprops_prop, "principal_lengths"):
        regionprops_prop = add_principal_lengths(regionprops_prop, scale)

    l1, _, l3 = regionprops_prop.principal_lengths

    return l1 / l3
