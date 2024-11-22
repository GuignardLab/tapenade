import numpy as np
from skimage.transform import resize
from tapenade.preprocessing import change_array_pixelsize, segment_stardist


def predict_stardist(
    array: np.ndarray,
    model_path: str,
    input_voxelsize: tuple,
    voxelsize_model: tuple = (0.7, 0.7, 0.7),
    normalize_input: bool = True,
):
    """
    Predict the segmentation of an array using a StarDist model.

    Parameters:
    - array: a 3D numpy array, input image to segment
    - model_path: str, path to the StarDist model
    - input_voxelsize: tuple, input voxel size of the image
    - voxelsize_model: tuple, voxel size of the model
    - normalize_input: bool, whether to normalize the input image

    Returns:
    - aniso_labels: numpy array, predicted
    """
    assert len(np.shape(array)) <= 3

    data = change_array_pixelsize(
        array=array,
        input_pixelsize=input_voxelsize,
        output_pixelsize=voxelsize_model,
        order=1,
    )

    if normalize_input:
        perc_low, perc_high = np.percentile(data, (1, 99))
        data = (data - perc_low) / (perc_high - perc_low)

    labels = segment_stardist(data, model_path)

    aniso_labels = resize(
        labels,
        array.shape,
        anti_aliasing=False,
        order=0,
        preserve_range=True,
    )

    return aniso_labels.astype(np.int16)