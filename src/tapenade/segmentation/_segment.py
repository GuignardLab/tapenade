import warnings

import numpy as np
from tqdm import tqdm

try:
    from csbdeep.utils import normalize
    from stardist import random_label_cmap
    from stardist.models import StarDist3D
except ImportError:
    warnings.warn("Please install the required packages: pip install stardist csbdeep")

import os
from pathlib import Path

from tapenade.preprocessing import change_arrays_pixelsize


def predict_stardist(
    array: np.ndarray,
    model_path: str,
    input_voxelsize: tuple,
    voxelsize_model:tuple=(0.7,0.7,0.7),
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
    model_name = Path(model_path).stem
    directory = str(os.path.split(model_path)[0])
    model = StarDist3D(None, name=model_name, basedir=directory)

    data = change_arrays_pixelsize(
        image=array, 
        input_pixelsize=input_voxelsize, 
        output_pixelsize=voxelsize_model,
        order=1
    )
    if normalize_input:
        data = normalize(data, 1, 99)
    labels, _ = model.predict_instances(
        data, axes="ZYX", n_tiles=model._guess_n_tiles(data)
    )
    aniso_labels = change_arrays_pixelsize(
        labels=labels,
        input_pixelsize=voxelsize_model,
        output_pixelsize=input_voxelsize,
        order=0,
    )

    return np.asarray(aniso_labels).astype(np.int16)


def find_seg_errors(segmentation:np.ndarray,image:np.ndarray):
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
    print(len(list_labels),' labels to process')

    intensity_distribution=np.zeros((len(list_labels),4)) 
    for id,label in tqdm(enumerate(list_labels)) :
        intensity_distribution[id,0]=label
        mask=(segmentation==label)
        list_pix=image[mask]
        n, bins = np.histogram(list_pix)
        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean)**2, weights=n)
        intensity_distribution[id,1]=mean
        intensity_distribution[id,2]=var
        intensity_distribution[id,3]=var/mean

    return (intensity_distribution)

def tresh_distribution(intensity_distribution:np.ndarray,threshold:float,column_number:int=3) :
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


    id_merged_cells=[]
    for id,intensity in enumerate(intensity_distribution[:,column_number]) :
        if intensity>threshold :
            id_merged_cells.append(int(intensity_distribution[id,0]))
    return(id_merged_cells)

