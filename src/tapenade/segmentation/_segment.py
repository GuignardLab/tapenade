import numpy as np

try:
    from csbdeep.utils import normalize
    from stardist import random_label_cmap
    from stardist.models import StarDist3D
except ImportError:
    print("Please install the required packages: pip install stardist csbdeep")

import os
from pathlib import Path

from tapenade.utils import change_voxelsize

np.random.seed(6)
lbl_cmap = random_label_cmap


def predict_stardist(
    array,
    model_path: str,
    input_voxelsize: tuple = (1, 1, 1),
    voxelsize_model:tuple=(0.7,0.7,0.7),
    normalize_input: bool = True,
):
    """
    Predict the segmentation of an array using a StarDist model.
    :param array: The array to segment.
    :param model_path: The path to the StarDist model.
    :param input_voxelsize: The voxel size of the input array.
    :param normalize_input: Whether to normalize the input array.
    :return: The predicted segmentation.
    """
    assert len(np.shape(array)) <= 3
    model_name = Path(model_path).stem
    directory = str(os.path.split(model_path)[0])
    model = StarDist3D(None, name=model_name, basedir=directory)

    data = change_voxelsize(
        array, input_vs=input_voxelsize, output_vs=voxelsize_model, order=1
    )
    if normalize_input:
        data = normalize(data, 1, 99)
    labels, _ = model.predict_instances(
        data, axes="ZYX", n_tiles=model._guess_n_tiles(data)
    )
    aniso_labels = change_voxelsize(
        labels, input_vs=voxelsize_model, output_vs=input_voxelsize, order=0
    )

    return np.asarray(aniso_labels).astype(np.int16)
