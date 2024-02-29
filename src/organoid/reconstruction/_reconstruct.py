import numpy as np
import math
import matplotlib.pyplot as plt
from xml.dom import minidom
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os
from skimage.measure import regionprops
from scipy.spatial.transform import Rotation
import registrationtools
import tifffile
from xml.dom import minidom
import napari
import glob
from pathlib import Path


def extract_positions(path_positions: str):
    """
    Extract the positions of the objects from the xml file (saved during the acquisition)
    Parameters
    ----------
    path_positions : str
        path to the xml file
    """
    positions = minidom.parse(path_positions)

    pos_x = positions.getElementsByTagName("dXPosition")
    pos_y = positions.getElementsByTagName("dYPosition")

    xpos = np.zeros(pos_x.length)
    ypos = np.zeros(pos_y.length)

    for i in range(pos_y.length):
        xpos[i] = pos_x[i].attributes["value"].value
        ypos[i] = pos_y[i].attributes["value"].value

    return (xpos, ypos)


def plot_positions(path_bottom_positions: str, path_top_positions: str):
    """
    Plot the positions of the objects from the xml file, to visualize and manually check what number have to be associated
    Parameters
    ----------
    path_bottom_positions : str
        path to the xml file for the bottom view
    path_top_positions : str
        path to the xml file for the top view

    """
    (xpos_b, ypos_b) = extract_positions(path_bottom_positions)
    (xpos_t, ypos_t) = extract_positions(path_top_positions)

    list_number_b = [
        i + 1 for i in range(len(xpos_b))
    ]  # the objects will have an index going from 1 to the total number, instead of having the id chosen during the aquisition. Problem ?
    fig, ax = plt.subplots()
    ax.scatter(xpos_b, ypos_b, label="bottom")

    for i, txt in enumerate(list_number_b):
        ax.annotate(txt, (xpos_b[i], ypos_b[i]))

    list_number_t = [i + 1 for i in range(len(xpos_t))]
    ax.scatter(xpos_t, -ypos_t, label="top")

    for i, txt in enumerate(list_number_t):
        ax.annotate(txt, (xpos_t[i], -ypos_t[i]))

    plt.legend()


def associate_top_bottom(path_bottom_positions: str, path_top_positions: str):
    """
    Associate the objects from the bottom view with the objects from the top view, by solving a linear sum assignement between the two distribution
    Parameters
    ----------
    path_bottom_positions : str
        path to the xml file for the bottom view
    path_top_positions : str
        path to the xml file for the top view

    Returns
    -------
    2 lists of indices, bottom then top, sorted according to the asssgnement

    """

    (xpos_b, ypos_b) = extract_positions(path_bottom_positions)
    (xpos_t, ypos_t) = extract_positions(path_top_positions)

    cost = np.zeros((len(xpos_b), len(xpos_t)))
    for i in range(len(xpos_b)):
        for j in range(len(xpos_t)):
            cost[i, j] = math.sqrt(
                math.pow(xpos_b[i] - xpos_t[j], 2)
                + math.pow(ypos_b[i] - ypos_t[j], 2)
            )
    row_ind, col_ind = linear_sum_assignment(cost)

    return (list(row_ind + 1), list(col_ind + 1))

    # assert xpos_b.shape==xpos_t.shape==ypos_b.shape==ypos_t.shape


def create_folders(
    name_experiment: str,
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
        folder_output = name_experiment

    for ind_g in range(
        len(list_ref)
    ):  # for each sample, creates a dedicated folder and save the channels separately
        filename_ref = list_ref[ind_g]
        filename_float = list_float[ind_g]
        folder_sample = rf"{name_experiment}/{filename_ref}/"

        # creates paths for the output files
        os.mkdir(os.path.join(name_experiment, filename_ref))
        os.mkdir(os.path.join(folder_sample, "trsf"))
        os.mkdir(os.path.join(folder_sample, "raw"))
        os.mkdir(os.path.join(folder_sample, "registered"))
        os.mkdir(os.path.join(folder_sample, "fused"))

        image_ref = tifffile.imread(rf"{name_experiment}/{filename_ref}.tif")
        image_float = tifffile.imread(
            rf"{name_experiment}/{filename_float}.tif"
        )
        for ind_ch, ch in enumerate(channels):
            tifffile.imwrite(
                folder_sample + rf"/raw/{filename_ref}_{ch}.tif",
                image_ref[:, ind_ch, :, :],
            )  # ,imagej=True, metadata={'axes': 'TZYX'})
            tifffile.imwrite(
                folder_sample + rf"/raw/{filename_float}_{ch}.tif",
                image_float[:, ind_ch, :, :],
            )  # ,imagej=True, metadata={'axes': 'TZYX'})


def manual_registration_fct(
    reference_landmarks, floating_landmarks, scale: tuple = (1, 1, 1)
):
    # stolen from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    """
    Finds the transformation between 2 sets of points in 3D.
    If the automatic registration can't find an accurate transformation :
    -open your bottom and top images on a visualization software (eg Napari) and create 2 label images of the same shape, one will contain the landmarks for the bottom image and the other for the top image.
    -add landmarks, preferentially sphere on the objects you can identify on both sides. Each object needs to have the same label on both sides, top and botttom. The more landmarks, the better.
    -then gives these 2 label arrays to the function manual_registration_fct, it will compute the transformation between the two sets of points

    Parameters
    ----------
    reference_landmarks : np.array
        This image will be the reference, the 'fixed' one
    floating_landmarks : np.array
        This image will be the floating image, the one that will be registered onto the reference image

    Returns
    -------
    translation and rotation to apply to the top image to register it on the bottom image
    In the following order : rotation_z, rotation_y, rotation_x, translation_z, translation_y, translation_x.
    """

    rg_ref = regionprops(reference_landmarks)
    centroids_ref = np.array([prop.centroid for prop in rg_ref]).T
    centroids_ref = centroids_ref * scale

    rg_float = regionprops(floating_landmarks)
    centroids_float = np.array([prop.centroid for prop in rg_float]).T
    centroids_float = centroids_float * scale

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
    return (rotation_angles, translation1, translation2)


def register(
    path_data: str,
    path_transformation: str,
    path_registered_data: str,
    path_to_bin: str,
    reference_image: str,
    floating_image: str,
    input_voxel: tuple = [1, 1, 1],
    output_voxel: tuple = [1, 1, 1],
    compute_trsf: int = 1,
    init_trsfs=[
        [
            "flip",
            "Y",
            "flip",
            "Z",
            "trans",
            "Z",
            -10,
        ]
    ],
    test_init: int = 0,
    trsf_type: str = "rigid",
    depth: int = 3,
    bbox: int = 1,
    image_interpolation: str = "linear",
    padding: int = 0,
    save_json: str = "",
    ordered_init_trsfs: bool = True,
):

    ##this is under comments because if i put quotation marks, the importation fails (?)
    # Register the two sides of the sample, using the previously computed transformation (if any) or computing a new one

    # Parameters
    # ----------
    # path_data : str
    #     path to the raw images
    # path_transformation : str
    #     path to the folder where the transformations files are saved
    # path_registered_data : str
    #     path where the registered images will be saved
    # path_to_bin : str
    #     path to the bin folder : necessary to save the registered data : should be something like 'C:\Users\user\Anaconda3\envs\my_environment\Library\bin'
    # reference_image : str
    #     name of the reference image, the 'fixed' one
    # floating_image : str
    #     name of the floating image, the one that will be registered onto the reference image
    # input_voxel : tuple, optional
    #     voxel size of the input image, by default [1,1,1]
    # output_voxel : tuple, optional
    #     voxel size of the output image, by default [1,1,1]. Can be different than the input voxel size.
    # compute_trsf : int, optional
    #     1 if the transformation has to be computed, 0 if it already exists. If you have multiple channels of the same image, it is recommended to pick one expressed homogeneously as teh reference, register this channel using compute_trsf=1.
    #     Then you can use compute_trsf=0, the algo will find the pre-existing transformation to register the other channels onto the reference channel.
    # init_trsfs : list, optional.
    #     You can use flipping (flip), rotations (rot), translation (trans). Specify the axis X,Y or Z after the sample_id of the transformation. For rotations and translation, precise the value (angle or distance) after the axis.
    #     Needs 2 sets of brackets.
    #     Example : [["flip", "Z", "trans", "Z", -10,"trans","Y",100,"rot","X",-29,"rot","Y",41,"rot","Z",-2]]
    # trsf_type : str, optional
    #     type of transformation to compute : rigid, affine. By default rigid.
    # depth : int, optional
    #     depth of the registration, by default 3
    # bbox : int, optional
    #     1 if the bounding box of the original image can extend, 0 if not.
    # save_json : bool, optional
    #     if True, saves the parameters in a json file, in the main folder. By default False.

    data = {
        "path_to_bin": path_to_bin,  # necessary to register the data
        "path_to_data": str(path_data),
        "ref_im": reference_image,  # rf"{sample_id}_bot_{channel}.tif",
        "flo_ims": [floating_image],  # [rf"{sample_id}_top_{channel}.tif"],
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
    path_data: str,
    reference_image,
    floating_image,
    additional_images: list = [],
    names_additional_images: list = [],
    scale: tuple = (1, 1, 1),
    labels: bool = False,
):
    """
    Opens the registered images in napari , to check how they overlap

    Parameters
    ----------
    path_data : str
        path to the registered images
    reference_image : str or ndarray (can be the name of the array in path_data or the array itself)
        name of the reference image, the 'fixed' one
    floating_image : str or ndarray (can be the name of the array in path_data or the array itself)
        name of the floating image, the one that will be registered onto the reference image
    additional_images : list of ndarray, optional
        list of additional images to add to the viewer, by default []
    scale : tuple, optional
        scale of the image (should be the same ), by default (1,1,1)
    """

    viewer = napari.Viewer()
    if isinstance(reference_image, str):
        if os.path.exists(
            rf"{path_data}/registered/{reference_image}"
        ):  # if the bounding box or voxel size has changed, then the reference image has been saved as an output.
            ref_im = tifffile.imread(
                rf"{path_data}/registered/{reference_image}"
            )
        else:  # if the bbox did not change, the reference image has not changed and is found in the raw folder only.
            ref_im = tifffile.imread(rf"{path_data}/raw/{reference_image}")
    else:
        ref_im = reference_image

    if isinstance(floating_image, str):
        float_im = tifffile.imread(rf"{path_data}/registered/{floating_image}")
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
        if additional_images != []:
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
        if additional_images != []:
            for num_im, im in enumerate(additional_images):
                name = names_additional_images[num_im]
                viewer.add_labels(im, name=str(name), scale=scale)

    napari.run()


def sigmoid(x, x0, p):
    """
    Sigmoid function, to weight the images by the distance to the edges
    z0 gives the middle of the sigmoid, at which weight=0.5
    p gives the slope. 5 corresponds to a low slope, wide fusion width and 25 to a strong slope, very thin fusion width.
    """
    # for z in range(len(z)) :
    return 1 / (1 + np.exp(-p * (x - x0)))


def fuse_sides(
    path_registered_data: str,
    reference_image_reg: str,
    floating_image_reg: str,
    folder_output: str = "",
    name_output: str = "fusion",
    slope_coeff: int = 20,
    axis: int = 0,
):
    """
    Fuse the two sides of the sample, using the previously registered images

    Parameters
    ----------
    path_registered_data : str
        path to the registered images
    reference_image_reg : str
        name of the reference image, the 'fixed' one
    floating_image_reg : str
        name of the floating image, the one that will be registered onto the reference image
    folder_output : str
        path to the folder where the fused image will be saved
    name_output : str, optional
    slope_coeff : int, optional
        coefficient to apply to the sigmoid function, by default 20.
        name of the output which is the fused image , by default 'fusion'
    axis : int, optional
        axis along which the fusion is done, by default 0 (z axis)
    """

    ###Takes the previously saved images (two registered sides)
    ref_image = tifffile.imread(
        rf"{path_registered_data}/{reference_image_reg}"
    )
    float_image = tifffile.imread(
        rf"{path_registered_data}/{floating_image_reg}"
    )
    mask_r = ref_image > 0
    mask_f = float_image > 0
    mask_fused = mask_r & mask_f
    cumsum = np.cumsum(mask_fused, axis=axis).astype(
        np.float16
    )  # we take the cumulative sum  along the fusion axis, this will give me a linear weight.
    cumsum_normalized = cumsum / np.max(cumsum)

    # apply a sigmoid function to the linear weights, to get a smooth transition between the two sides
    w2 = sigmoid(cumsum_normalized, x0=0.5, p=slope_coeff)
    w1 = 1 - w2

    fusion = (ref_image * w1 + float_image * w2).astype(np.uint16)

    tifffile.imwrite(rf"{folder_output}/{name_output}", fusion)


def write_hyperstacks(path: str, sample_id: str, channels: list):
    image = tifffile.imread(
        Path(path) / f"{sample_id}_{channels[0]}_fused.tif", dtype=np.int16
    )
    (z, x, y) = image.shape
    new_image = np.zeros((z, len(channels), x, y))
    print(new_image.shape)
    for ch in range(len(channels)):
        one_channel = tifffile.imread(
            Path(path) / f"{sample_id}_{channels[ch]}_fused.tif",
            dtype=np.int16,
        )
        new_image[:, ch, :, :] = one_channel
        print(one_channel.shape, new_image.shape)
    tifffile.imwrite(Path(path) / f"{sample_id}_registered.tif", new_image)


def add_centermass(landmarks, radius: int = 10, centermass_label: int = 10):
    """
    For debug purposes : if the registration does not work, you can use this function on the landmarks image to check if the center of mass is correcly aligned after the rotation.

    Parameters
    ----------
    landmarks : np.array
        array containing the landmarks (can be float or reference)
    radius : int, optional
        radius of the sphere to add around the center of mass, by default 10
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


def reconstruct_foo():
    print("reconstruct")
    return -1


def pipeline_reconstruction(*args):  # image1, image2, sigma, ...
    print(*args)


def script_run():

    # Parse the arguments
    pipeline_reconstruction(1, 2, 3, 4, 5)
