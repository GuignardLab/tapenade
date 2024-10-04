import warnings

import numpy as np

import os
from pathlib import Path


def _load_model(model_path: str):
    """
    Load a StarDist model from a folder.

    Parameters:
    - model_path: str, path to the StarDist model folder
    """

    try:
        from stardist.models import StarDist3D
    except ImportError:
        warnings.warn("Please install Stardist for your system")

    model_name = Path(model_path).stem
    directory = str(os.path.split(model_path)[0])
    model = StarDist3D(None, name=model_name, basedir=directory)

    return model


def _segment_stardist(image: np.ndarray, model, thresholds_dict: dict):
    """
    Predict the segmentation of an array using a StarDist model.

    Parameters:
    - image: a 3D numpy array, input image to segment
    - model: StarDist model object
    - model_path: str, path to the StarDist model folder
    - thresholds_dict: dict, dictionary of thresholds for the model, structured like
      {'prob': 0.5, 'nms': 0.3} for probability and non-maximum suppression thresholds
      respectively
    """

    try:
        from stardist import gputools_available
    except ImportError:
        warnings.warn("Please install Stardist for your system")

    if gputools_available():
        model.use_gpu = True

    if thresholds_dict is not None:
        model.thresholds = thresholds_dict

    labels, _ = model.predict_instances(
        image, n_tiles=model._guess_n_tiles(image)
    )

    return labels.astype(np.uint16)


def _purge_gpu_memory():
    """
    Purge the GPU memory.
    """

    from numba import cuda

    cuda.select_device(0)
    cuda.close()
