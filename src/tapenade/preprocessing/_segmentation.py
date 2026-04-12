import warnings

import numpy as np

import os
from pathlib import Path


def _load_model_stardist(model_path: str):
    """
    Load a StarDist model from a folder.

    Parameters:
    - model_path: str, path to the StarDist model folder
    """

    try:
        from stardist.models import StarDist3D

        model_name = Path(model_path).stem
        directory = str(os.path.split(model_path)[0])
        model = StarDist3D(None, name=model_name, basedir=directory)
        return model
    except ImportError:
        warnings.warn("Please install Stardist for your system")
        raise ImportError("Please install Stardist for your system")


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

        if gputools_available():
            model.use_gpu = True

        if thresholds_dict is not None:
            model.thresholds = thresholds_dict

        labels, _ = model.predict_instances(
            image, n_tiles=model._guess_n_tiles(image)
        )

        return labels.astype(np.uint16)

    except ImportError:
        warnings.warn("Please install Stardist for your system")
        raise ImportError("Please install Stardist for your system")


def _purge_gpu_memory():
    """
    Purge the GPU memory.
    """

    from numba import cuda

    cuda.select_device(0)
    cuda.close()


def _load_model_cellpose_sam(gpu: bool = True):
    """
    Load a CellPose-SAM model.

    Parameters:
    - gpu: bool, whether to request GPU usage
    """

    try:
        from cellpose import models, core

        if gpu and not core.use_gpu():
            warnings.warn("No GPU access detected; falling back to CPU.")
            gpu = False

        model = models.CellposeModel(gpu=gpu)
        return model
    except ImportError:
        warnings.warn("Please install cellpose for your system")
        raise ImportError("Please install cellpose for your system")


def _segment_cellpose_sam(
    image: np.ndarray,
    model,
    diameter: float | None = None,
    batch_size: int = 32,
    flow3D_smooth: int = 1,
):
    """
    Predict the segmentation of an array using CellPose-SAM.

    Parameters:
    - image: a 3D numpy array, input image to segment
    - model: CellPose model object
    - diameter: float, estimated object diameter (pixels). None for auto.
    - batch_size: int, batch size for inference
    - flow3D_smooth: int, flow smoothing for 3D
    """

    labels, _, _ = model.eval(
        image,
        z_axis=0,
        channel_axis=None,
        batch_size=batch_size,
        do_3D=True,
        flow3D_smooth=flow3D_smooth,
        diameter=diameter,
    )

    return labels.astype(np.uint16)
