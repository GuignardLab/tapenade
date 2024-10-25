import numpy as np
from skimage.transform import resize
from tqdm import tqdm

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


def find_seg_errors(segmentation: np.ndarray, image: np.ndarray):
    """
    Compute statisitics of intensity of each label to try to find out exceptions/anomalies and detect segmentation errors.

    Parameters:
    segmentation: array containing the labels
    image : intensity image, has to be the same size as segmentatio,

    Returns:
    The intensity_distribution array, for which each line corresponds to a label in the segmentation.
    Column 1 is the label's id
    Col 2 is its mean intensity (in the ROI)
    Col 3 is the std of its intensity distrib
    Col 4 is the ratio of standard deviation and mean. We use this value to detect wrong segmentations

    """
    list_labels = list(np.unique(segmentation))
    list_labels.remove(0)
    print(len(list_labels), " labels to process")

    intensity_distribution = np.zeros((len(list_labels), 4))
    for index, label in tqdm(enumerate(list_labels)):
        intensity_distribution[index, 0] = label
        mask = segmentation == label
        list_pix = image[mask]
        n, bins = np.histogram(list_pix)
        mids = 0.5 * (bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean) ** 2, weights=n)
        intensity_distribution[index, 1] = mean
        intensity_distribution[index, 2] = var
        intensity_distribution[index, 3] = var / mean

    return intensity_distribution


def tresh_distribution(
    intensity_distribution: np.ndarray,
    threshold: float,
    column_number: int = 3,
):
    """
    Thresholds the Distribution to find labels that could be wrong segmentation.

    Parameters:
    intensity_distribution: Array computed with the function find_seg_errors, that shows average intensity, std and std/mean for each label of the distrbution
    threshold : threshold value above which cell will be considered wrong. To get an idea, visualize the distrbution (examples are in the notebooks)
    column_number : index to consider for thre threshold.
                    If you want to discrimnate on the mean intensity of the ROI, choose 1.
                    If you want to discriminate on the std, choose 2.
                    If you want to discriminate over the std divided by the mean intensity for each label, choose 3.

    Returns:
    A list of labels that might be false segmentations.

    """

    id_merged_cells = []
    for index, intensity in enumerate(
        intensity_distribution[:, column_number]
    ):
        if intensity > threshold:
            id_merged_cells.append(int(intensity_distribution[index, 0]))
    return id_merged_cells
