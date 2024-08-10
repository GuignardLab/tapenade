# :herb: Tapenade : Thorough Analysis PiEliNe for Advanced DEep imaging

[![License MIT](https://img.shields.io/pypi/l/tapenade.svg?color=green)](https://github.com/GuignardLab/tapenade/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tapenade.svg?color=green)](https://pypi.org/project/tapenade)
[![Python Version](https://img.shields.io/pypi/pyversions/tapenade.svg?color=green)](https://python.org)
[![tests](https://github.com/GuignardLab/tapenade/workflows/tests/badge.svg)](https://github.com/GuignardLab/tapenade/actions)
[![codecov](https://codecov.io/gh/GuignardLab/tapenade/branch/main/graph/badge.svg)](https://codecov.io/gh/GuignardLab/tapenade)

<img src="https://github.com/GuignardLab/tapenade/blob/Packaging/imgs/tapenade3.png" width="100">

A pipeline for quantitative analysis of 3D organoid images acquired with deep imaging microscopy.

If you use this plugin for your research, please [cite us](https://github.com/GuignardLab/tapenade).

This repository has been developed by (in alphabetical order):

- [Alice Gros](mailto:alice.gros@univ-amu.fr)
- [Jules Vanaret](mailto:jules.vanaret@univ-amu.fr)
- [Léo Guignard](mailto:leo.guignard@univ-amu.fr)
- [Valentin Dunsing-Eichenauer](valentin.dunsing@univ-amu.fr)

----------------------------------

## Overview

<img src="Fig_overview_github.png" width="1000">

The Tapenade pipeline is a tool for the analysis of dense 3D tissues acquired with deep imaging microscopy. It is designed to be user-friendly and to provide a comprehensive analysis of the data. The pipeline is composed of several steps, each of which can be run independently.

The pipeline is composed of the following methods:

1. **Spectral filtering**: Given a set of calibrated emission spectra, allows the unmixing of the different fluorophores present in the image. 
2. **Registration & fusion**: Allows for spatial fusion of two images (e.g. acquired with a dual-view microscope).
3. **Pre-processing**: Provides many pre-processing functions, like rescaling, masking, correction of optical artifcats, etc.
4. **Segmentation**: Detect and seperate each nuclei in the image. We provide trained weights for StarDist3D, a state-of-the-art deep learning model for nuclei segmentation.
4. **Masked smoothing**: Produces smooth fields of a given dense or sparse quantity, which allows for multiscale analysis.
5. **Spatial correlation analysis**: Computes a spatial correlation map between two continuous fields.
6. **Deformation tensors analysis**: Computes deformation tensors (inertia, true strain, etc.) from segmented objects.

All methods are explained in details in our Jupyter notebooks, which are available in the `notebooks` folder [here](notebooks/).


## Installation

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

## Usage

## Complementary Napari plugins (for graphical user interfaces)


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

[file an issue]: https://github.com/GuignardLab/tapenade/issues
