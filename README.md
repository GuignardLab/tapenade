# :herb: Tapenade : Thorough Analysis PipEliNe for Advanced DEep imaging

[![License MIT](https://img.shields.io/pypi/l/tapenade.svg?color=green)](https://github.com/GuignardLab/tapenade/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tapenade.svg?color=green)](https://pypi.org/project/tapenade)
[![Python Version](https://img.shields.io/pypi/pyversions/tapenade.svg?color=green)](https://python.org)
[![tests](https://github.com/GuignardLab/tapenade/workflows/tests/badge.svg)](https://github.com/GuignardLab/tapenade/actions)
[![codecov](https://codecov.io/gh/GuignardLab/tapenade/branch/main/graph/badge.svg)](https://codecov.io/gh/GuignardLab/tapenade)

<img src="https://github.com/GuignardLab/tapenade/blob/main/imgs/tapenade3.png" width="100">

A fully-visual pipeline for quantitative analysis of 3D organoid images acquired with deep imaging microscopy.

If you use this plugin for your research, please [cite us](https://github.com/GuignardLab/tapenade/blob/main/README.md#how-to-cite).

----------------------------------

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Complementary Napari plugins (for graphical user interfaces)](#complementary-napari-plugins-for-graphical-user-interfaces)
- [How to cite](#how-to-cite)
- [Contributing](#contributing)
- [License](#license)
- [Issues](#issues)

## Overview

<img src="https://github.com/GuignardLab/tapenade/blob/main/imgs/Fig_overview_github.png" width="1000">

The Tapenade pipeline is a tool for the analysis of dense 3D tissues acquired with deep imaging microscopy. It is designed to be user-friendly and to provide a comprehensive analysis of the data. The pipeline is composed of several steps, each of which can be run independently.

The pipeline is composed of the following methods:

1. **Spectral filtering**: Given a set of calibrated emission spectra, allows the unmixing of the different fluorophores present in the image. 
2. **Registration & fusion**: Allows for spatial fusion of two images (e.g. acquired with a dual-view microscope).
3. **Pre-processing**: Provides many pre-processing functions, like rescaling, masking, correction of optical artifcats, etc.
4. **Segmentation**: Detect and seperate each nuclei in the image. We provide trained weights for StarDist3D, a state-of-the-art deep learning model for nuclei segmentation.
4. **Masked smoothing**: Produces smooth fields of a given dense or sparse quantity, which allows for multiscale analysis.
5. **Spatial correlation analysis**: Computes a spatial correlation map between two continuous fields.
6. **Deformation tensors analysis**: Computes deformation tensors (inertia, true strain, etc.) from segmented objects.

All methods are explained in details in our Jupyter notebooks, which are available in the [notebooks](notebooks/) folder.


## Installation

### Main library

We recommand to install the library in a specific environement (like [conda]). To create a new environement, you can run the following command:

    conda create -n env-tapenade python=3.10
You can then activate the environement with:

    conda activate env-tapenade

For here onward, it is assumed that you are running the commands from the `env-tapenade` [conda] environement.

You can install `tapenade` via [pip]:

```shell
pip install tapenade
```

To install the latest development version:

```shell
pip install git+https://github.com/GuignardLab/tapenade.git
```

To install the latest development version in editable mode:

```shell
git clone git@github.com:GuignardLab/tapenade.git
cd tapenade
pip install -e .
```

This will install only the main library, without the libraries for the registration/fusion and segmentation methods. To install them, please follow the instructions below. 

### Registration and fusion (optional)

The registration and fusion methods require the `3D-registration` Python package. To install it, follow the instructions on the library's [repository](https://github.com/GuignardLab/tapenade/tree/main?tab=readme-ov-file#installation).

### Segmentation (optional)

We provide trained weights for StarDist3D, a state-of-the-art deep learning model for nuclei segmentation. To install Stardist3D, follow the instructions on the library's [repository](https://github.com/GuignardLab/tapenade/tree/main?tab=readme-ov-file#installation).

If you prefer to use StarDist3D with a graphical user interface, Stardist3D is also available as a plugin in several softwares, like [Napari](https://github.com/stardist/stardist-napari), [Fiji](https://imagej.net/plugins/stardist), and [Icy](https://github.com/stardist/stardist-icy) (more details on the [Stardist3D repository](https://github.com/stardist/stardist?tab=readme-ov-file#plugins-for-other-software)).

We highly recommend using the `change_array_pixelsize` and `local_contrast_enhancement` methods (defined [here](https://github.com/GuignardLab/tapenade/blob/main/src/tapenade/preprocessing/_preprocessing.py)) from our library to preprocess your data before running the inference with our custom StarDist3D weights.

Though not mandatory, we also recommend running the inference with StarDist3D on a GPU for faster results. If you don't have a GPU, you can use the ZeroCostDL4Mic Google Colab notebooks, which allow you to run the inference on a GPU for free. You can find the ZeroCostDL4Mic notebooks for StarDist3D [here](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/StarDist_3D_ZeroCostDL4Mic.ipynb).

## Usage

The methods described above are available at the following locations:

1. **Spectral filtering**: [Code](src/tapenade/preprocessing/spectral_filtering/XXX), [Notebook](notebooks/spectral_filtering_notebook.ipynb)
2. **Registration & fusion**: [Code](src/tapenade/reconstruction/_reconstruct.py), [Notebook](notebooks/registration_notebook.ipynb)
3. **Pre-processing**: This [script](src/tapenade/preprocessing/_preprocessing.py) gathers all preprocessing functions, [Notebook](notebooks/preprocessing_notebook.ipynb)
4. **Segmentation**: [Code](src/tapenade/segmentation/_segment.py), [Notebook](notebooks/segmentation_notebook.ipynb)
4. **Masked smoothing**: [Code](src/tapenade/preprocessing/_preprocessing.py) (it is one of the preprocessing function), [Notebook](notebooks/masked_gaussian_smoothing_notebook.ipynb)
5. **Spatial correlation analysis**: [Code](src/tapenade/analysis/spatial_correlation/_spatial_correlation_plotter.py), [Notebook](notebooks/spatial_correlation_analysis_notebook.ipynb)
6. **Deformation tensors analysis**: [Code](src/tapenade/analysis/deformation/deformation_quantification.py), [Notebook](notebooks/deformation_analysis_notebook.ipynb)

All methods are explained in details in our Jupyter notebooks, which are available in the [notebooks](notebooks/) folder.

## Complementary Napari plugins (for graphical user interfaces)

During the pre-processing stage, dynamical exploration and interaction led to faster tuning of the parameters by allowing direct visual feedback, and gave key biophysical insight during the analysis stage. 
We thus created three user-friendly Napari plugins designed around facilitating such interactions:

1. **napari-manual-registration** (available [here](https://github.com/GuignardLab/napari-manual-registration))
When using our automatic registration tool to spatially register two views of the same organoid, we were sometimes faced with the issue that the tool would not converge to the true registration transformation. This happens when the initial position and orientation of the floating view are too far from their target values. We thus designed a Napari plugin to quickly find a transformation that can be used to initialize our registration tool close to the optimal transformation. From two images loaded in Napari representing two views of the same organoid, the plugin allows the user to either (i) manually define a rigid transformation by continually varying 3D rotations and translations while observing the results until a satisfying fit is found, or to (ii) annotate matching salient landmarks (e.g bright dead cells or lumen-like structures) in both the reference and floating views, from which an optimal rigid transformation can be found automatically using principal component analysis. 

2. **napari-tapenade-processing** (available [here](https://github.com/GuignardLab/napari-tapenade-processing))
From a given set of raw images, segmented object instances, and object mask, the plugin allows the user to quickly run all pre-processing functions from our main pipeline with custom parameters while being able to see and interact with the result of each step. For large datasets that are cumbersome to manipulate or cannot be loaded in Napari, the plugin provides a macro recording feature: the users can experiment and design their own pipeline on a smaller subset of the dataset, then run it on the full dataset without having to load it in Napari.

2. **napari-spatial-correlation-plotter** (available [here](https://github.com/GuignardLab/napari-spatial-correlation-plotter))
This plugins allows the user to analyse the spatial correlations of two 3D fields loaded in Napari (e.g two fluorescent markers). The user can dynamically vary the analysis length scale, which corresponds to the standard deviation of the Gaussian kernel used for smoothing the 3D fields. 
If a layer of segmented nuclei instances is additionally specified, the histogram is constructed by binning values at the nuclei level (each point corresponds to an individual nucleus). Otherwise, individual voxel values are used.
The user can dynamically interact with the correlation heatmap by manually selecting a region in the plot. The corresponding cells (or voxels) that contributed to the region's statistics will be displayed in 3D on an independant Napari layer for the user to interact with and gain biological insight.

## How to cite

If you use this plugin for your research, please cite us using the following reference:

- Jules Vanaret, Alice Gros, Valentin Dunsing-Eichenauer, Agathe Rostan, Philippe Roudot, Pierre-François Lenne, Léo Guignard, Sham Tlili
bioRxiv 2024.08.13.607832; doi: https://doi.org/10.1101/2024.08.13.607832


This repository has been developed by (in alphabetical order):

- [Alice Gros](mailto:alice.gros@univ-amu.fr)
- [Jules Vanaret](mailto:jules.vanaret@univ-amu.fr)
- [Léo Guignard](mailto:leo.guignard@univ-amu.fr)
- [Valentin Dunsing-Eichenauer](valentin.dunsing@univ-amu.fr)

## Contributing

Contributions are very welcome. Tests can be run with [tox] or [pytest], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"tapenade" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

----------------------------------

This library was generated using [Cookiecutter] and a custom made template based on [@napari]'s [cookiecutter-napari-plugin] template.

[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[pip]: https://pypi.org/project/pip/
[tox]: https://tox.readthedocs.io/en/latest/
[pytest]: https://docs.pytest.org/
[conda]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[file an issue]: https://github.com/GuignardLab/tapenade/issues
