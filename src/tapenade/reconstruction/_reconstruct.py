import glob
import json
import math
import os
from pathlib import Path
from xml.dom import minidom
import tifffile
import matplotlib.pyplot as plt
import napari
import numpy as np
import registrationtools
import io
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from skimage.measure import regionprops
from skimage import io
import transforms3d._gohlketransforms as tg
from tapenade.preprocessing._preprocessing import compute_mask

def extract_positions(path_positions: str):
    """
    Extract the X and Y positions of the objects from the xml file (saved during the acquisition) and put them in a list
    Parameters
    ----------
    path_positions : str
        path to the xml file
    """
    print(path_positions)
    positions = minidom.parse(
        str(path_positions)
    )  # need to convert the path to a string, otherwise minidom does not work for Windows paths
    pos_x = positions.getElementsByTagName("dXPosition")
    pos_y = positions.getElementsByTagName("dYPosition")

    xpos = np.zeros(pos_x.length)
    ypos = np.zeros(pos_y.length)

    for i in range(pos_y.length):
        xpos[i] = pos_x[i].attributes["value"].value
        ypos[i] = pos_y[i].attributes["value"].value

    return (xpos, ypos)


def plot_positions(path_ref_positions: str, path_float_positions: str):
    """
    Plot the positions of the objects from the xml file, to visualize and manually check what number have to be associated
    Parameters
    ----------
    path_ref_positions : str
        path to the xml file for the reference view
    path_float_positions : str
        path to the xml file for the floating view

    """
    (xpos_ref, ypos_ref) = extract_positions(path_ref_positions)
    (xpos_float, ypos_float) = extract_positions(path_float_positions)

    list_number_ref = [
        i + 1 for i in range(len(xpos_ref))
    ]  # the objects will have an index going from 1 to the total number, instead of having the id chosen during the aquisition. Problem ?
    fig, ax = plt.subplots()
    ax.scatter(xpos_ref, ypos_ref, label="ref")

    for i, txt in enumerate(list_number_ref):
        ax.annotate(txt, (xpos_ref[i], ypos_ref[i]))

    list_number_float = [i + 1 for i in range(len(xpos_float))]
    ax.scatter(xpos_float, ypos_float, label="float")

    for i, txt in enumerate(list_number_float):
        ax.annotate(txt, (xpos_float[i], ypos_float[i]))

    plt.legend()


def associate_positions(path_ref_positions: str, path_float_positions: str):
    """
    Associate the objects from the reference view with the objects from the floating view, by solving a linear sum assignement between the two distribution
    Parameters
    ----------
    path_ref_positions : str
        path to the xml file for the ref view
    path_float_positions : str
        path to the xml file for the float view

    Returns
    -------
    2 lists of indices, ref then float, sorted according to the asssgnement

    """

    (xpos_ref, ypos_ref) = extract_positions(path_ref_positions)
    (xpos_float, ypos_float) = extract_positions(path_float_positions)
    ypos_float = -ypos_float
    cost = np.zeros((len(xpos_ref), len(xpos_float)))
    for i in range(len(xpos_ref)):
        for j in range(len(xpos_float)):
            cost[i, j] = math.sqrt(
                math.pow(xpos_ref[i] - xpos_float[j], 2)
                + math.pow(ypos_ref[i] - ypos_float[j], 2)
            )
    row_ind, col_ind = linear_sum_assignment(cost)
    list_row = [int(i) for i in list(row_ind + 1)]
    list_col = [int(i) for i in list(col_ind + 1)]
    return (list_row, list_col)


def create_folders(
    folder_experiment: str,
    list_ref: list,
    list_float: list,
    channels: list,
    folder_output: str = "",
):
    """
    Creates the folders to save the registered images, and save the channels separately

    Parameters
    ----------
    folder_experiment : str
        path to the main folder
    list_ref : list
        list of the refrence images
    list_float : list
        list of the floating images corresponding to the reference images
    channels : list
        list of the names of the channels
    folder_output : str, optional
        path to the output folder, if None, will be the same as the experiment folder
    """

    if folder_output == "":
        folder_output = folder_experiment

    for ind_g in range(
        len(list_ref)
    ):  # for each sample, creates a dedicated folder and save the channels separately
        filename_ref = list_ref[ind_g]
        filename_float = list_float[ind_g]
        folder_sample = Path(folder_experiment) / filename_ref

        # creates paths for the output files
        os.mkdir(os.path.join(folder_experiment, filename_ref))
        os.mkdir(os.path.join(folder_sample, "trsf"))
        os.mkdir(os.path.join(folder_sample, "raw"))
        os.mkdir(os.path.join(folder_sample, "registered"))
        os.mkdir(os.path.join(folder_sample, "fused"))
        os.mkdir(os.path.join(folder_sample, "weights"))
        os.mkdir(os.path.join(Path(folder_sample) / "weights", "before_trsf"))
        os.mkdir(os.path.join(Path(folder_sample) / "weights", "after_trsf"))

        image_ref = io.imread(Path(folder_experiment) / f"{filename_ref}.tif")
        image_float = io.imread(
            Path(folder_experiment) / f"{filename_float}.tif"
        )
        if len(channels) > 1:
            for ind_ch, ch in enumerate(channels):
                imref = image_ref[:, :, :, ind_ch]
                imfloat = image_float[:, :, :, ind_ch]
                io.imsave(
                    Path(folder_sample) / "raw" / f"{filename_ref}_{ch}.tif",
                    imref,  ##CAREFUL needs to be float32 or uint16 orint16 otherwise the blockmatching does not compute/save the result
                )  # ,imagej=True, metadata={'axes': 'TZYX'})
                io.imsave(
                    Path(folder_sample) / "raw" / f"{filename_float}_{ch}.tif",
                    imfloat,
                )  # ,imagej=True, metadata={'axes': 'TZYX'})>
        else:
            io.imsave(
                Path(folder_sample)
                / "raw"
                / f"{filename_ref}_{channels[0]}.tif",
                image_ref,  ##CAREFUL needs to be float32 or uint16 orint16 otherwise the blockmatching does not compute/save the result
            )
            io.imsave(
                Path(folder_sample)
                / "raw"
                / f"{filename_float}_{channels[0]}.tif",
                image_float,
            )


def transformation_from_plugin(path_json: str, scale: tuple = (1, 1, 1)):
    """
    Extract the transformation from the json file saved by the plugin
    Parameters
    ----------
    path_json : str
        path to the json file
    scale : tuple, optional
        scale of the image, by default (1,1,1)
    """
    with open(path_json) as f:
        data = json.load(f)
    rot_z = data["rot_z"]
    rot_y = data["rot_y"]
    rot_x = data["rot_x"]
    trans_z = data["trans_z"]
    trans_y = data["trans_y"]
    trans_x = data["trans_x"]
    # in the napari plugin, the translation computed is the one after the rotation, so we need to set the translation trans1 before rotation to 0
    init_trsfs = list_init_trsf(
        trans1=[0, 0, 0],
        trans2=[trans_z, trans_y, trans_x],
        rot=[rot_x, rot_y, rot_z],
    )
    return init_trsfs


def manual_registration_fct(
    reference_landmarks, floating_landmarks, scale: tuple = (1, 1, 1)
):
    # stolen from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    """
    Finds the transformation between 2 sets of labels in 3D.
    If the automatic registration can't find an accurate transformation, 2 options :
    1) create landmarks layers using the napari plugin manual-registration, and save the values of rotation and transaltion into a json file that you will give as the argument "input_init_trsf_from_plugin" to the function register
    2) using another software to create 2 arays of the same size as your original images (without channels) and labels of the same values fot the 2 views. Save these arrays as tif files, and give them as arguments to this function.

    Parameters
    ----------
    reference_landmarks : np.array
        Landmarks for the reference image
    floating_landmarks : np.array
        Landmarks for the floating image

    Returns
    -------
    translation and rotation to apply to the floating image to register it on the reference image
    In the following order : rotation, translation1, translation2 with translation1 the translation to apply before rotation and translation2 the translation to apply after rotation
    """

    rg_ref = regionprops(reference_landmarks)
    centroids_ref = np.array([prop.centroid for prop in rg_ref])
    centroids_ref = (centroids_ref * scale).T

    rg_float = regionprops(floating_landmarks)
    centroids_float = np.array([prop.centroid for prop in rg_float])
    centroids_float = (centroids_float * scale).T

    assert centroids_ref.shape == centroids_float.shape

    num_rows, num_cols = centroids_ref.shape
    if num_rows != 3:
        raise Exception(
            f"matrix centroids_ref is not 3xN, it is {num_rows}x{num_cols}"
        )

    num_rows, num_cols = centroids_float.shape
    if num_rows != 3:
        raise Exception(
            f"matrix centroids_float is not 3xN, it is {num_rows}x{num_cols}"
        )

    # find mean column wise
    centermass_ref = np.mean(centroids_ref, axis=1)
    centermass_float = np.mean(centroids_float, axis=1)

    # subtract mean
    centroids_ref_centered = centroids_ref - centermass_ref.reshape(3, 1)
    centroids_float_centered = centroids_float - centermass_float.reshape(3, 1)

    H = centroids_ref_centered @ np.transpose(centroids_float_centered)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centermass_ref + centermass_float

    trsf = np.identity(4)
    trsf[0:3, 0:3] = R
    trsf[0:3, 3] = t
    trsf = np.linalg.lstsq(trsf, np.identity(4))[0]
    np.savetxt("trsf.txt", trsf)

    rotation = Rotation.from_matrix(R)
    rotation_angles = rotation.as_euler("xyz", degrees=True)

    # translation1 is the translation that will center the floating landmarks on the center of the image.
    # Then, we will apply the rotation with respect to the center (the center of mass stays at the center))
    # Afterwards, we apply translation2 to center the floating landmarks that was located at the center, on the reference position.

    center_image = (np.array(reference_landmarks.shape) / 2) * scale
    translation1 = center_image - centermass_float
    translation2 = centermass_ref - center_image
    return (
        list(rotation_angles),
        list(translation1),
        list(translation2),
    )  # into list to be able to copy paste directly into the registration function


def list_init_trsf(trans1, trans2, rot):
    """
    Transforms the individual values of rotation and translation into a list of transformations to apply in the right order. This list can be given as the parameter other_trsf to the function register
    Parameters
    ----------
    trans1 : list
        list of the translations to apply to the floating image BEFORE ROTATION
    trans2 : list
        list of the translations to apply to the floating image AFTER ROTATION
    rot : list
        list of the rotations to apply to the floating image
    """

    trans2_in_um = trans2
    trans1_in_um = trans1
    init_trsf = [
        [
            "trans",
            "X",
            -trans1_in_um[2],
            "trans",
            "Y",
            -trans1_in_um[1],
            "trans",
            "Z",
            -trans1_in_um[0],
            "rot",
            "X",
            -rot[0],
            "rot",
            "Y",
            -rot[1],
            "rot",
            "Z",
            -rot[2],
            "trans",
            "X",
            -trans2_in_um[2],
            "trans",
            "Y",
            -trans2_in_um[1],
            "trans",
            "Z",
            -trans2_in_um[0],
        ]
    ]
    return init_trsf


def compute_transformation_from_trsf_files(
    path_trsf: str, return_matrix: bool = False
):

    f = open(Path(path_trsf) / "A1-rigid.trsf", "r")

    # Read and ignore header lines (NECESSARY)
    header1 = f.readline()
    header2 = f.readline()

    matrix = np.zeros((4, 4))

    index = 0
    for line in f:
        line = line.strip()
        columns = line.split()
        if index < 4:
            matrix[index, :] = np.array(columns, dtype=float)
        index = index + 1

    scale, shear, angles, trans, persp = tg.decompose_matrix(matrix)
    angles_deg = [
        math.degrees(angles[0]),
        math.degrees(angles[1]),
        math.degrees(angles[2]),
    ]
    # rotation angles (X,Y,Z) in deg:",angles_deg
    if return_matrix == True:
        return matrix
    else:
        return (trans, angles_deg)


def register(
    path_data: str,
    path_transformation: str,
    path_registered_data: str,
    reference_image: str,
    floating_image: str,
    input_voxel: list = [1, 1, 1],
    output_voxel: list = [1, 1, 1],
    compute_trsf: int = 1,
    trans1: list = None,
    rot: list = None,
    trans2: list = None,
    other_trsf: list = None,
    input_init_trsf_from_plugin: str = "",
    test_init: int = 0,
    trsf_type: str = "rigid",
    depth: int = 3,
    bbox: int = 1,
    image_interpolation: str = "linear",
    padding: int = 0,
    save_json: str = "",
    ordered_init_trsfs: bool = True,
):

    # Register the two sides of the sample, using the previously computed transformation (if any) or computing a new one
    # """
    # Parameters
    # ----------
    # path_data : str
    #     path to the raw images. In the folder structure : folder_experiment/sample_id/raw
    # path_transformation : str
    #     path to the folder where the transformations files are saved. In the folder structure : folder_experiment/sample_id/trsf
    # path_registered_data : str
    #     path where the registered images will be saved. In the folder structure : folder_experiment/sample_id/registered
    # reference_image : str
    #     name of the reference image, the 'fixed' one
    # floating_image : str
    #     name of the floating image, the one that will be registered onto the reference image
    # input_voxel : tuple, optional
    #     voxel size of the input image, by default [1,1,1]
    # output_voxel : tuple, optional
    #     voxel size of the output image, by default [1,1,1]. Can be different from the input voxel size.
    # compute_trsf : int, optional
    #     1 if the transformation has to be computed, 0 if it already exists. If you have multiple channels of the same image, it is recommended to pick one expressed homogeneously as the reference, register this channel using compute_trsf=1.
    #     Then you can use compute_trsf=0, the algo will find the pre-existing transformation to register the other channels onto the reference channel.
    # rot : list, optional
    #     list of the rotations to apply to the floating image, by default [0,0,0]
    # trans1 : list, optional
    #     list of the translations to apply to the floating image BEFORE ROTATION, by default [0,0,0]
    # trans2 : list, optional
    #     list of the translations to apply to the floating image AFTER ROTATION, by default [0,0,0]
    # other_trsf : list, optional.
    #     list of transformations to apply to the floating image, by default [], if you want give direclty your transformations.
    #     If this argument is not None, the value of the parameters rot, trans1, trans2 will be ignored.
    #     You can use flipping (flip), rotations (rot), translation (trans). Specify the axis X,Y or Z after the sample_id of the transformation. For rotations and translation, precise the value (angle or distance) after the axis.
    #     IMPORTANT : Needs 2 sets of brackets.
    #     If order_init_trsfs is True, the transformations will be applied in the order they are given in the list init_trsfs.
    #     # Example : [['flip', 'Z', 'trans', 'Z', -10,'trans','Y',100,'rot','X',-29,'rot','Y',41,'rot','Z',-2]]
    # input_init_trsf_from_plugin : str, optional
    #     path to the json file saved by the plugin, containing the transformation. If this parameter is not None, the transformation will be extracted from the json file no matter the parameters rot, trans1, trans2 or other_trsf
    # test_init : int, optional
    #     1 if we apply only the initial transformation. 0 if we apply the initial trsf + blockmatching algo. By default 0.
    # trsf_type : str, optional
    #     type of transformation to compute : rigid, affine. By default rigid.
    # depth : int, optional
    #     depth of the registration, by default 3
    # bbox : int, optional
    #     1 if the bounding box of the original image can extend, 0 if not.
    # image_interpolation : str, optional
    #     type of interpolation to apply to the image, by default 'linear'
    # padding : int, optional
    #     padding to apply to the image, by default 0
    # save_json : str
    #     if not None, save the parameters of that function in a json file at the path given.
    #     IMPORTANT : saves the parameters of the function, not the parameters of the registration, in particular the initial transformations only. To save the actual transformations, use the function "compute_transformation_from_trsf_files"
    # ordered_init_trsfs : bool, optional
    #     if True, the transformations will be applied in the order they are given in the list init_trsfs.

    # """
    if rot is None:
        rot = [0, 0, 0]
    if trans1 is None:
        trans1 = [0, 0, 0]
    if trans2 is None:
        trans2 = [0, 0, 0]

    if input_init_trsf_from_plugin != "":
        # if the user input the json file from the plugin, the json file is used as init trsf no matter the parameters rot, trans1, trans 2 or other_trsf
        init_trsfs = transformation_from_plugin(
            input_init_trsf_from_plugin, scale=input_voxel
        )
    elif input_init_trsf_from_plugin == "" and other_trsf is not None:
        print("other_trsf", other_trsf)
        # if the user defined the initial transformation with the parameter 'other_trsf' (list of transformations he wants), then we use this transformation as init_trsfs
        init_trsfs = other_trsf
    else:  # is the user did not precise other_trsf or any json file, then we use the parameters rot, trans1 and trans2 to "build" init_trsfs.
        init_trsfs = list_init_trsf(trans1=trans1, trans2=trans2, rot=rot)
    data = {
        "path_to_data": str(path_data),
        "ref_im": reference_image,
        "flo_ims": [floating_image],
        "compute_trsf": compute_trsf,
        "init_trsfs": init_trsfs,
        "trsf_paths": [str(path_transformation)],
        "trsf_types": [trsf_type],
        "ref_voxel": input_voxel,
        "flo_voxels": [input_voxel],
        "out_voxel": output_voxel,
        "test_init": test_init,
        "apply_trsf": 1,
        "out_pattern": str(path_registered_data),
        "begin": 1,
        "end": 1,
        "bbox_out": bbox,
        "image_interpolation": image_interpolation,
        "padding": padding,
        "registration_depth": depth,
        "ordered_init_trsfs": ordered_init_trsfs,
    }
    print(data)

    if save_json != "":
        json_string = json.dumps(data, indent=4)
        with open(Path(save_json) / "parameters.json", "w") as outfile:
            outfile.write(json_string)
    tr = registrationtools.SpatialRegistration(data)
    tr.run_trsf()


def check_napari(
    folder: str,
    reference_image,
    floating_image,
    additional_images: list = None,
    names_additional_images: list = None,
    scale: tuple = (1, 1, 1),
    labels: bool = False,
):
    """
    Opens the registered images in napari , to check how they overlap

    Parameters
    ----------
    folder : str
        path to the main folder. The floating image is always saved in the folder 'registered' but not the reference image.
    reference_image : str or ndarray (can be the name of the array in path_data or the array itself)
        name of the reference image, the 'fixed' one
    floating_image : str or ndarray (can be the name of the array in path_data or the array itself)
        name of the floating image, the one that will be registered onto the reference image
    additional_images : list of ndarray, optional
        list of additional images to add to the viewer, by default []
    scale : tuple, optional
        scale of the image (should be the same ), by default (1,1,1)
    labels : bool, optional
        if True, the images are considered as labels, by default False

    """

    viewer = napari.Viewer()
    if isinstance(reference_image, (str | Path)):
        if os.path.exists(
            Path(folder) / "registered" / reference_image
        ):  # if the bounding box or voxel size has changed, then the reference image has been saved as an output.
            ref_im = io.imread(Path(folder) / "registered" / reference_image)
        else:  # if the bbox did not change, the reference image has not changed and is found in the raw folder only.
            ref_im = io.imread(Path(folder) / "raw" / reference_image)
    else:
        ref_im = reference_image

    if isinstance(floating_image, (str | Path)):
        float_im = io.imread(Path(folder) / "registered" / floating_image)
    else:
        float_im = floating_image

    if not labels:
        viewer.add_image(
            ref_im, colormap="cyan", name="reference_image", scale=scale
        )
        viewer.add_image(
            float_im,
            colormap="red",
            blending="additive",
            name="floating_image",
            scale=scale,
        )
        if additional_images is not None:
            for num_im, im in enumerate(additional_images):
                viewer.add_image(
                    im,
                    colormap="green",
                    blending="additive",
                    name=names_additional_images[num_im],
                    scale=scale,
                )
    if labels:
        viewer.add_labels(ref_im, name="reference_image", scale=scale)
        viewer.add_labels(float_im, name="floating_image", scale=scale)
        if additional_images is not None:
            for num_im, im in enumerate(additional_images):
                name = names_additional_images[num_im]
                viewer.add_labels(im, name=str(name), scale=scale)

    napari.run()


def sigmoid(z, z0, p):
    """
    Sigmoid function, to weight the images by the distance to the edges
    z0 gives the middle of the sigmoid, at which weight=0.5
    p gives the slope. 5 corresponds to a low slope, wide fusion width and 25 to a strong slope, very thin fusion width.
    """
    return 1 / (1 + np.exp(-p * (z - z0)))


def fuse_sides_in_z_axis(
    path_registered_data: str,
    reference_image_reg: str,
    floating_image_reg: str,
    folder_output: str = "",
    name_output: str = "fusion",
    slope_coeff: int = 20,
    axis: int = 0,
    return_image=False,
):  # outdated version, fuses the views along z axis instead of along the sample axis (can be different if the sample moved between the 2 views)
    """
    Fuse the two sides of the sample, using the previously registered images

    Parameters
    ----------
    path_registered_data : str
        path to the registered images
    reference_image_reg : str
        name of the reference image, the 'fixed' one. Can be its name in path_registered_data or the array itself.
    floating_image_reg : str
        name of the floating image, the one that will be registered onto the reference image. Can be its name in path_registered_data or the array itself
    folder_output : str
        path to the folder where the fused image will be saved
    name_output : str, optional
    slope_coeff : int, optional
        coefficient to apply to the sigmoid function, by default 20.
        name of the output which is the fused image , by default 'fusion'
    axis : int, optional
        axis along which the fusion is done, by default 0 (z axis)
    return_image : bool, optional
        if True, returns the fused image, by default False
    """

    ###Takes the previously saved images (two registered sides)
    if isinstance(
        reference_image_reg, (str | Path)
    ):  # if the reference image is a string, then we load the image from the path
        ref_image = io.imread(Path(path_registered_data) / reference_image_reg)
    # else its the image itself
    if isinstance(floating_image_reg, (str | Path)):
        float_image = io.imread(
            Path(path_registered_data) / floating_image_reg
        )

    dtype_input = float_image.dtype
    mask_r = ref_image > 0
    mask_f = float_image > 0
    mask_fused = mask_r & mask_f
    cumsum = np.cumsum(mask_fused, axis=axis)
    # we take the cumulative sum  along the fusion axis, this will give me a linear weight.
    cumsum_normalized = cumsum / np.max(cumsum)

    # apply a sigmoid function to the linear weights, to get a smooth transition between the two sides
    w2 = sigmoid(cumsum_normalized, z0=0.5, p=slope_coeff)
    w1 = 1 - w2

    sum_weights = w1 + w2

    fusion = ref_image * w1 / sum_weights + float_image * w2 / sum_weights

    if return_image:
        return fusion
    io.imsave(Path(folder_output) / name_output, fusion.astype(dtype_input))


def fuse_sides(
    folder: str,
    reference_image: str,
    floating_image: str,
    folder_output: str = "",
    name_output: str = "fusion",
    slope_coeff: int = 20,
    axis: int = 0,
    input_voxel: list = [1, 1, 1],
    output_voxel: list = [1, 1, 1],
    trsf_type: str = "rigid",
    return_image=False,
    sigma_for_mask: int = 3,
    threshold_factor_for_mask: int = 1,
):
    """
    Fuse the two sides of the sample, using the previously registered images. Compute sigmoid weights from raw images, then register the weights using the transformation computed on the intensity image and then fuse the images using the registered weights.

    Parameters
    ----------
    folder : str
        path to the main folder
    reference_image : str
        name of the reference image, the 'fixed' one
    floating_image : str
        name of the floating image, the one that will be registered onto the reference image
    folder_output : str
        path to the folder where the fused image will be saved
    name_output : str, optional
        name of the output which is the fused image , by default 'fusion'
    slope_coeff : int, optional
        coefficient to apply to the sigmoid function, by default 20. 5 corresponds to a low slope, wide fusion width and 25 to a strong slope, very thin fusion width
    axis : int, optional
        axis along which the fusion is done, by default 0 (z axis)
    input_voxel : list, optional
        voxel size of the input image, by default [1,1,1]. Has to be the same as the intensity images.
    output_voxel : list, optional
        voxel size of the output image, by default [1,1,1].  Has to be the same as the intensity images.
    trsf_type : str, optional
        type of transformation to compute : rigid, affine. By default rigid. Has to be the same as the intensity images.
    return_image : bool, optional
        if True, returns the fused image, by default False
    sigma_for_mask : int, optional
        sigma for the gaussian blur applied to the mask, by default 1
    threshold_factor_for_mask : int, optional   
        threshold factor for the mask

    """

    if isinstance(reference_image, (str | Path)):

        ref_im = io.imread(Path(folder) / "raw" / reference_image)
        float_im = io.imread(Path(folder) / "raw" / floating_image)
    dtype_input = (
        float_im.dtype
    )  # we wil return an image that has the same dtype as the input image
    mask_r = compute_mask(image = ref_im,method = 'snp otsu',sigma_blur=sigma_for_mask,threshold_factor=threshold_factor_for_mask,compute_convex_hull=False,registered_image=False)
    mask_f = compute_mask(image = float_im,method = 'snp otsu',sigma_blur=sigma_for_mask,threshold_factor=threshold_factor_for_mask,compute_convex_hull=False,registered_image=False)

    # we compute the weights as a sigmoid function of the distance to the objective (=cumulative sum of the raw image)
    cumsum_r = np.cumsum(mask_r.astype(int), axis=axis)
    cumsum_r_normalized = cumsum_r / np.max(cumsum_r)
    w_ref = sigmoid(1 - cumsum_r_normalized, z0=0.5, p=slope_coeff)

    cumsum_f = np.cumsum(mask_f.astype(int), axis=axis)
    cumsum_f_normalized = cumsum_f / np.max(cumsum_f)
    w_float = sigmoid(1 - cumsum_f_normalized, z0=0.5, p=slope_coeff)

    folder_weight = Path(folder) / "weights"

    # saving weights (from the original images) is necessary to apply transformation
    io.imsave(
        Path(folder_weight) / "before_trsf" / f"w_float.tif",
        w_float.astype(np.float32),
    )
    io.imsave(
        Path(folder_weight) / "before_trsf" / f"w_ref.tif",
        w_ref.astype(np.float32),
    )
    register(
        path_data=Path(folder_weight) / "before_trsf",
        path_transformation=Path(folder) / "trsf",
        path_registered_data=Path(folder_weight) / "after_trsf",
        reference_image=f"w_ref.tif",
        floating_image=f"w_float.tif",
        input_voxel=input_voxel,
        output_voxel=output_voxel,
        compute_trsf=0,  # we apply the same trsf that was computed for the actual intensity images
        test_init=0,
        trsf_type=trsf_type,
    )

    w_ref_after_trsf = io.imread(
        Path(folder_weight) / "after_trsf" / "w_ref.tif"
    )  # need to register the reference image as well because of possible changes in voxel size
    w_float_after_trsf = io.imread(
        Path(folder_weight) / "after_trsf" / "w_float.tif"
    )

    ref_im_registered = io.imread(
        Path(folder) / "registered" / reference_image
    )
    float_im_registered = io.imread(
        Path(folder) / "registered" / floating_image
    )
    sum_weights = w_ref_after_trsf + w_float_after_trsf
    w_ref_after_trsf[sum_weights != 0] /= sum_weights[sum_weights != 0]
    w_float_after_trsf[sum_weights != 0] /= sum_weights[sum_weights != 0]
    fusion = np.round(ref_im_registered * w_ref_after_trsf).astype(np.uint16)
    fusion += np.round(float_im_registered * w_float_after_trsf).astype(np.uint16)
    if return_image:
        return fusion
    io.imsave(Path(folder_output) / name_output, fusion.astype(dtype_input))


def write_hyperstacks(
    path: str,
    sample_id: str,
    channels: list,
    return_image=False,
    dtype=np.float32,
):
    """
    Writes the hyperstacks, by stacking the channels of the registered images

    Parameters
    ----------
    path : str
        path to the registered images
    sample_id : str
        name of the sample
    channels : list
        list of the names of the channels
    return_image : bool, optional
        if True, returns the hyperstack, by default False
    """

    image = io.imread(
        Path(path) / f"{sample_id}_{channels[0]}.tif"
    )  # reading one image just to extract the shape of the image and initialize the hyperstack
    (z, y, x) = image.shape
    new_image = np.zeros((z, len(channels), y, x))
    for ch in range(len(channels)):
        one_channel = io.imread(Path(path) / f"{sample_id}_{channels[ch]}.tif")
        print(one_channel.shape)
        new_image[:, ch, :, :] = one_channel

    if return_image:
        return new_image

    tifffile.imwrite(
        Path(path) / f"{sample_id}_registered.tif",
        new_image.astype(dtype),
        imagej=True,
        compression=("zlib", 1),
    )  # float16 not compatible with Fiji


def add_centermass(landmarks, radius: int = 10, centermass_label: int = 10):
    """
    For debug purposes : if the registration does not work, you can use this function on the landmarks image to check if the center of mass is correcly aligned after the rotation.

    Parameters
    ----------
    landmarks : np.array
        array containing the landmarks (can be float or reference)
    radius : int, optional
        radius of the sphere to add around the center of mass, by default 10
    centermass_label : int, optional
        value of the label of the center of mass, by default 10 (arbitrary)
    """

    rg = regionprops(landmarks)
    centroids = np.array([prop.centroid for prop in rg]).T
    z, y, x = np.mean(centroids, axis=1)
    landmarks[
        int(z) - radius : int(z) + radius,
        int(y) - radius : int(y) + radius,
        int(x) - radius : int(x) + radius,
    ] = centermass_label
    return landmarks


def remove_previous_files(path):
    filelist = glob.glob(os.path.join(path, "*.tif"))
    for f in filelist:
        os.remove(f)
