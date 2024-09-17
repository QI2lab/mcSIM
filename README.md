[![preprint](https://img.shields.io/badge/preprint-bioRxiv-blue.svg)](https://doi.org/10.1101/2020.07.27.223941)
[![paper](https://img.shields.io/badge/paper-biomedical%20optics%20express-blue.svg)](https://doi.org/10.1364/BOE.422703)
[![website](https://img.shields.io/badge/website-up-green.svg)](https://shepherdlaboratory.org/)
[![Github commit](https://img.shields.io/github/last-commit/QI2lab/mcSIM)](https://github.com/QI2lab/mcSIM)

# multicolor DMD-SIM
This repository contains code for designing, analyzing, and carrying out multicolor structured illumination microscopy
experiments based on a digital micromirror device (DMD-SIM), including DMD simulation code, DMD pattern generation, SIM reconstruction and
instrument control. It also includes a number of useful utilities for simulating the resulting diffraction 
pattern given certain DMD patterns, determining system point-spread functions and optical transfer functions, and 
determining the affine transformation between the DMD coordinates and the imaging space coordinates. The various 
files are described in more detail below.
 
# Published work using this code
* "Multicolor structured illumination microscopy and quantitative control of polychromatic light with a digital micromirror device"
([BOE](https://doi.org/10.1364/BOE.422703), 
[bioRxiv](https://doi.org/10.1101/2020.07.27.223941), 
[Zenodo](https://doi.org/10.5281/zenodo.4773865), 
[release](https://github.com/QI2lab/mcSIM/releases/tag/v1.0.0)).
* "Resolution doubling in light-sheet microscopy via oblique plane structured illumination" 
([Nat. Methods](https://doi.org/10.1038/s41592-022-01635-8), 
[bioRxiv](https://doi.org/10.1101/2022.05.19.492671), 
[Zenodo](https://doi.org/10.5281/zenodo.6419901), 
[release](https://github.com/QI2lab/mcSIM/releases/tag/v0.2.0)).
* "Spatial wavefront shaping with a multipolar-resonant metasurface for structured illumination microscopy" 
([Opt. Mat. Exp.](https://doi.org/10.1364/OME.520736), [arXiv](https://doi.org/10.48550/arXiv.2309.14456)).
* "A Structured Illumination Microscopy Framework with Spatial-Domain Noise Propagation" 
([bioRxiv](https://doi.org/10.1101/2023.12.07.570701), 
[Zenodo](https://doi.org/10.5281/zenodo.10037896), 
[v1.5.0](https://github.com/QI2lab/mcSIM/releases/tag/v1.5.0)).
* "Fourier synthesis optical diffraction tomography for kilohertz rate volumetric imaging" ([arXiv](https://arxiv.org/abs/2309.16912)).

# Installation
The best way to use this python package is to install it with pip
```
git clone https://github.com/QI2lab/mcSIM.git
cd mcSIM
pip install .
```
If you would like to edit the code, then install using the `-e` option
```
git clone https://github.com/QI2lab/mcSIM.git
cd mcSIM
pip install -e .
```

### Optional hardware control
The dependencies for the experimental control code are not installed by default because
the DMD control code relies on the Windows specific [pywinusb](https://pypi.org/project/pywinusb/). 
To install these dependencies run
```
git clone https://github.com/QI2lab/mcSIM.git
cd mcSIM
pip install .[expt_ctrl]
```

### Optional GPU support
Some functions can be optionally run on a GPU. This functionality has been tested with CUDA 11.2 and 11.8.  If GPU support is desired,
first install the CUDA toolkit manually, and then install mcSIM with pip. For example, if you are using a conda environment and
CUDA toolkit 11.8
```
conda install -c conda-forge cudatoolkit=11.8
git clone https://github.com/QI2lab/mcSIM.git
cd mcSIM
pip install .[gpu]
```
Alternatively, installing CuPy first should pull in the cudatoolkit dependencies
```commandline
conda install -c conda-forge cupy cuda-version=11.8
git clone https://github.com/QI2lab/mcSIM.git
cd mcSIM
pip install .[gpu]
```

This will install the appropriate version of [CuPy](https://cupy.dev/) and the scikit-image portion of [cuCIM](https://pypi.org/project/cucim/).
Note that the entire RAPIDS/cuCIM package cannot be installed on Windows, but the scikit-image portion can. 
Alternatively, to install the scikit-image portion of cuCIM manually run
```
pip install "git+https://github.com/rapidsai/cucim.git@v22.12.00#egg=cucim&subdirectory=python/cucim"
```
or the tag v22.12.00 can be replaced with the most recent version.

# SIM reconstruction code
### [sim_reconstruction.py](mcsim/analysis/sim_reconstruction.py)
Code for reconstructing SIM images from raw data using a Gustafsson/Heintzmann Wiener filter style reconstruction. Several
reconstruction options are available, largely following either the approach of
[openSIM](https://doi.org/10.1109/JSTQE.2016.2521542) or [fairSIM](https://doi.org/10.1038/ncomms10980).
To get started with reconstructing SIM data, 
download image data from [Zenodo](https://doi.org/10.5281/zenodo.7851110)
and place it in a directory called `examples/data`.
Then, see the example scripts
* [reconstruct_sim_simulated_single.py](examples/reconstruct_sim_simulated_single.py)
* [reconstruction_sim_simulated_timelapse.py](examples/reconstruct_sim_simulated_timelapse.py)
* [reconstruct_sim_experiment_multichannel.py](examples/reconstruct_sim_experiment_multichannel.py) 
* For a tutorial on SIM reconstruction, see our [I2K 2022 repository and talks](https://github.com/QI2lab/I2K2022-SIM).

# DMD simulation code
### [simulate_dmd.py](mcsim/analysis/simulate_dmd.py)
Code for simulating the diffraction patterns produced by the DMD. Various 
scripts illustrating the usage of this code can be found in the [examples](examples)
directory. This simulation code has many useful features, including an analytic solution
for the joint blaze/diffraction condition, pattern simulation tools, tools for extracting the intensity pattern
in the Fourier plane of a collecting lens, etc. For example usage, see
* [solve_multicolor_blaze_condition.py](examples/solve_multicolor_blaze_condition.py)
* [simulate_dmd_diffraction_pattern.py](examples/simulate_dmd_diffraction_pattern.py)
* [simulate_multicolor_sim_patterns_1d.py](examples/simulate_multicolor_sim_patterns_1d.py)
* [simulate_dmd_diffraction_fourier_plane.py](examples/simulate_dmd_diffraction_fourier_plane.py)
* [dmd_solution_explorer_gui.py](examples/dmd_solution_explorer_gui.py)

### [dmd_patterns.py](mcsim/analysis/dmd_patterns.py)
This file can be used to generate multicolor SIM patterns and other useful calibrations
patterns for the DMD. It also contains many tools for working with the basis vector/unit cell
representation of DMD patterns. This allows a complete enumeration of DMD diffraction orders
in a compact and computationally efficient form.

# Utility code
### [fit_dmd_affine.py](mcsim/analysis/fit_dmd_affine.py)
Code to fit the affine transformation between the DMD coordinates and camera coordinates
using imaging data from a DMD pattern consisting of many small points. These code relies
on tools for working with affine transformations found 
[here](https://github.com/QI2lab/localize-psf/blob/master/localize_psf/affine.py). For example usage first download the image data available on [Zenodo](https://doi.org/10.5281/zenodo.7851110),
and then see the script [calibrate_affine_xform.py](examples/calibrate_affine_xform.py).


### [localize-psf](https://github.com/QI2lab/localize-psf/blob/master/localize_psf)
Useful tools for automatically localizing sparse fluorescent beads and performing both 2D and 3D
PSF fitting using various PSF models. Also provides useful statistics and figures summarizing the results.
This code has now been split out into a [separate repository](https://github.com/QI2lab/localize-psf).
For more information about these tools, see the [readme](analysis/localize-psf/README.md)
pip installing the mcsim repository as described above will also pull in these dependencies.

# Examples
Scripts illustrated examples of different DMD simulations and analysis are stored in [examples](examples).
Associated data necessary to run some of these examples can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7851110).
These scripts assume this data has been placed in a directory called `examples/data`
  
# Hardware control code
Hardware control is based around [MicroManager2.0](https://micro-manager.org/). Currently, we control the instrument
using a fork of the [napari-micromanager](https://github.com/QI2lab/napari-micromanager) project which controls
the MicroManager core using [pymmcore-plus](https://github.com/tlambert03/pymmcore-plus). Our fork of this project
relies on MicroManager device drives to control cameras and stages, and on python code to control the DMD and DAQ.
The "device adapters" for the DMD and DAQ are found below

### [expt_ctrl/dlp6500.py](mcsim/expt_ctrl/dlp6500.py)
Code for controlling a DLP6500 or DLP9000 DMD over USB on Windows. This code was initially based on the approaches 
of [Lightcrafter6500DMDControl](https://github.com/mazurenko/Lightcrafter6500DMDControl) and
[Pycrafter6500](https://github.com/csi-dcsc/Pycrafter6500). Extension to other operating systems has not been
implemented but should be straightforward.

This file also includes functions used to define pattern sequences on the DMD using patterns have either been 
previously loaded onto the firmware using the [Texas Instruments DLP6500 and DLP9000
GUI](https://www.ti.com/tool/DLPC900REF-SW) or which are loaded "on-the-fly". There is a low-level interface
for running these patterns based directly on their index in the firmware. There is also a higher-level interface
which supports defining "channels" and "modes" which can be saved in a json configuration file.

When run as a script, this file provides a command line interface to programming DMD pattern sequences.

### [expt_ctrl/daq.py](mcsim/expt_ctrl/daq.py)
Code for controlling a national instruments DAQ through [PyDAQmx](https://pypi.org/project/PyDAQmx/)

### [expt_ctrl/daq_config.json](mcsim/expt_ctrl/daq_config.json)
DAQ configuration file describing "modes" and "channels" for the DAQ. This file is used by `nidaq` instances created
with [daq.py](mcsim/expt_ctrl/daq.py)

### [program_sim_odt.py](mcsim/expt_ctrl/program_sim_odt.py)
This file is used to create DAQ sequences for SIM and ODT experiments

### expt_ctrl/*.cfg
[MicroManager configuration files](https://micro-manager.org/wiki/Micro-Manager_Configuration_Guide#Configuration_file_syntax)
describing the equipment and settings used in the experiment. 

### useful script files
Example scripts which are useful for controlling various instruments during testing include:
* [load_dmd_pattern.py](mcsim/expt_ctrl/load_dmd_pattern.py)
* [setup_optotune_mre2.py](mcsim/expt_ctrl/setup_optotune_mre2.py)
* [set_dmd_odt_pattern.py](mcsim/expt_ctrl/set_dmd_odt_pattern.py)

# Instrument design
A [parts list](parts_list.md) is available here. Mechanical drawings and 3D models of custom parts such as the DMD mount 
are available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10892402). For a more complete description of the optical path, see our BOE paper linked above.

# Documentation
Documentation is generated from function docstrings and built with Sphinx. e.g. navigate to [docs](docs) and run
```
make html
```
Then open `docs/_build/html/index.html` in your browser

# Acknowlegements
Thank you to Songyun Gu for extending the DMD control code to work with the [DLP9000](https://www.ti.com/product/DLP9000).
Thank you to [Beibei Gao](https://github.com/Elise-9) for assistance in extending the DMD pattern generation code to work
with the staggered pixel geometry used by the [DLP4500](https://www.ti.com/product/DLP4500). 