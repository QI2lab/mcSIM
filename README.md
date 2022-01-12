# multicolor DMD-SIM
This repository contains code for designing, analyzing, and carrying out multicolor structured illumination microscopy
experiments based on a digital micromirror device (DMD-SIM), including DMD simulation code, DMD pattern generation, SIM reconstruction and
instrument control. It also includes a number of useful utilities for simulating the resulting diffraction 
pattern given certain DMD patterns, determining system point-spread functions and optical transfer functions, and 
determining the affine transformation between the DMD coordinates and the imaging space coordinates. The various 
files are described in more detail below.
 
This repository is associated with the Biomedical Optics Express paper 
[Multicolor structured illumination microscopy and quantitative control of polychromatic light with a digital micromirror device](https://doi.org/10.1364/BOE.422703)
and the [BioRxiv preprint](https://doi.org/10.1101/2020.07.27.223941).
The repository state at the time of publication is archived [here](https://doi.org/10.5281/zenodo.4773865), or available as
a [release](https://github.com/QI2lab/mcSIM/releases/tag/v1.0.0) on GitHub.

When cloning this repo, use the following command to include submodules:

`git clone --recurse-submodules https://github.com/QI2lab/mcSIM.git`

If you prefer to install this code as a python package,
```
git clone https://github.com/QI2lab/mcSIM.git
cd mcSIM
pip install .
```

# Analysis and simulation code

### [simulate_dmd.py](mcsim/analysis/simulate_dmd.py)
Code for simulating the diffraction patterns produced by the DMD. Various 
scripts illustrating the usage of this code can be found in the [examples](examples) directory

### [dmd_patterns.py](mcsim/analysis/dmd_patterns.py)
This file can be used to generate multicolor SIM patterns and other useful calibrations
patterns for the DMD.

### [sim_reconstruction.py](mcsim/analysis/sim_reconstruction.py)
Code for reconstructing SIM images from raw data using a Gustafsson/Wiener filter style reconstruction. Several different
reconstruction options are available, largely following either the approach of 
[openSIM](https://doi.org/10.1109/JSTQE.2016.2521542) or [fairSIM](https://doi.org/10.1038/ncomms10980). 

### [fit_dmd_affine.py](mcsim/analysis/fit_dmd_affine.py)
Code to fit the affine transformation between the DMD coordinates and camera coordinates using imaging data from a DMD
pattern consisting of many small points.

### [otf_tools.py](mcsim/analysis/otf_tools.py)
Code for extracting optical transfer function from measurement of the strength of various Fourier peaks for a given SIM DMD pattern.
 
### [psd.py](mcsim/analysis/psd.py)
Code for doing the periodic/smooth image decomposition, an alternative to apodization for the Fourier transform. This code is taken from https://github.com/jacobkimmel/ps_decomp (with permission), and included here for convenience.

### [analysis_tools.py](mcsim/analysis/analysis_tools.py)
Miscellaneous tools for IO, reading metadata, image processing, etc.

### [localize-psf](localize-psf/fit.py)
Useful tools for automatically localizing sparse fluorescent beads and performing both 2D and 3D
PSF fitting using various PSF models. Also provides useful statistics and figures summarizing the results.
This code has now been split out into a [separate repository](https://github.com/QI2lab/localize-psf) which is included in this repository as a
submodule for convenience. For more information about these tools, see the [readme](analysis/localize-psf/README.md)

# Examples
Scripts illustrated examples of different DMD simulations and analysis are stored in [examples](examples). Associated 
data is located in [examples/data](examples/data)
  
# Hardware control code

### [expt_ctrl/dlp6500.py](mcsim/expt_ctrl/dlp6500.py)
Code for controlling the DLP6500 DMD over USB on Windows. This code was initially based on the approaches 
of [Lightcrafter6500DMDControl](https://github.com/mazurenko/Lightcrafter6500DMDControl) and
[Pycrafter6500](https://github.com/csi-dcsc/Pycrafter6500).

### [expt_ctrl/set_dmd_sim.py](mcsim/expt_ctrl/set_dmd_sim.py)
This is the script used to define pattern sequences on the DMD using patterns which
have been previously loaded onto the firmware using the [Texas Instruments DLP6500 and DLP9000
GUI](https://www.ti.com/tool/DLPC900REF-SW). It is a command line interface to [dlp6500.py](expt_ctrl/dlp6500.py),
and it is used by [run_sim_triggerscop.bsh](expt_ctrl/run_sim_triggerscop.bsh).

### [expt_ctrl/run_sim_triggerscop.bsh](mcsim/expt_ctrl/run_sim_triggerscop.bsh)
A [beanshell script](https://beanshell.github.io/) which can be run from [MicroManager 2.0 Gamma](https://micro-manager.org/wiki/Micro-Manager)
to acquire SIM data using dlp6500.py to set the DMD patterns (assuming they have already been loaded into the 
DMD firmware as described above). This script programs a [Triggerscope 3B](https://arc.austinblanco.com/), which then provides the analog and 
digital voltages required to run the rest of the experiment. The camera free runs as the
master clock, triggering the Triggerscope. We use a customized version of the Triggerscope firmware V601. 

This script supports collecting multidimensional data with the axis order xy-position/time/z-position/channel/SIM pattern.
It also supports time-lapse imaging with arbitrary wait time.

This code was initially developed with Micro-Manager 2.0.0-gamma1 20200514, MMCore Version 10.1.0,
Device API version 69, Module API version 10. But for more recent development we are using
Micro-Manager 2.0.0-gamma1 20210516, MMCore Version 10.1.1, Device API version 70, Module API version 10.

### [expt_ctrl/SIM.cfg](mcsim/expt_ctrl/SIM.cfg)
[MicroManager configuration file](https://micro-manager.org/wiki/Micro-Manager_Configuration_Guide#Configuration_file_syntax)
describing the equipment and settings used in the experiment.
  
### [expt_ctrl/dmd_sim_umanager_plugin](mcsim/expt_ctrl/dmd_sim_umanager_plugin)
A simple [MicroManager GUI plugin](https://micro-manager.org/wiki/Version_2.0_Plugins) for controlling the DMD patterns while "cruising around" a sample before imaging.
This plugin takes input from the user and then runs [set_dmd_sim.py](expt_ctrl/set_dmd_sim.py) with
the appropriate arguments in a user specified Conda environment. 

(todo: these instructions are a work in progress. Not working yet.)
This plugin can be compiled using IntelliJ IDEA and the resulting .jar file should be copied
to the mmplugins subdirectory of the MicroManager installation directory. Specifically:
1. Import this folder as an IntelliJ IDEA project
2. Right click this folder in the Project window and click `Open Module Settings`. Or select `File -> Project Structure`
3. Go to the `libraries` tab, click add, and add all of the `.jar` files in the `plugins` subdirectory of the MicroManager install
4. Repeat this process for `ij.jar` located in the main MM install directory
5. Add the Java SDK under `Project -> Project SDK`. I used `corretto-1.8` with project language level 6.
6. Go to the `artifacts` tab and click add. Select `jar` and `From modules with dependencies`. For the main class input `ij.ImageJ`
7. Under `Build` select `Build artifacts...` and the jar file will be built
8. Copy the jar file to the `mmplugins` subdirectory
TODO: get this working and finish giving instructions...

# Instrument design
Mechanical drawings of some parts used in the setup are included in the [parts](parts) directory. For a more complete description of the setup and
a parts list, see the published paper linked above.


