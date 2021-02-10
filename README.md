# multicolorSIM
This repository contains code for performing DMD-SIM experiments, including DMD simulation code, DMD pattern generation, SIM reconstruction and
instrument control. It also includes a number of useful utilities for simulating the resulting diffraction 
pattern given certain DMD patterns, determining system point-spread functions and optical transfer functions, and 
determining the affine transformation between the DMD coordinates and the imaging space coordinates. The various 
relevant files are described in more detail below.
 
This repository is connected with the BioRxiv preprint: [Multicolor structured illumination microscopy and quantitative control of coherent light with a digital micromirror device](https://doi.org/10.1101/2020.07.27.223941 )

# Analysis and simulation code

### [analysis/simulate_dmd.py](analysis/simulate_dmd.py)
Code for simulating the diffraction patterns produced by the DMD. Various 
scripts illustrating the usage of this code can be found in the [examples](examples) directory

### [analysis/dmd_patterns.py](analysis/dmd_patterns.py)
This file can be used to generate multicolor SIM patterns and other useful calibrations
patterns for the DMD.

### [analysis/sim_reconstruction.py](analysis/sim_reconstruction.py)
Code for reconstructing SIM images from raw data

### [analysis/affine.py](analysis/affine.py)
Code to fit the affine transformation between the DMD coordinates and camera coordinates using imaging data from a DMD
pattern consisting of many small points.

### [analysis/fit_psf.py](analysis/fit_psf.py)
Code for automatically finding PSF spots on an image of a sparse bead slide, performing both 2D and 3D PSF fitting using
various PSF models, and providing useful statistics and figures summarizing the results.

### [analysis/otf_tools.py](analysis/otf_tools.py)
Code for extracting optical transfer function from measurement of the strength of various Fourier peaks for a given SIM DMD pattern.
 
### [analysis/psd.py](analysis/psd.py)
Code for doing the periodic/smooth image decomposition, an alternative to apodization for the Fourier transform. This code is taken from https://github.com/jacobkimmel/ps_decomp (with permission), and included here for convenience.

### [analysis/fit.py](analysis/fit.py)
Useful tools for fitting, used in fit_psf.py and elsewhere.

### [analysis/analysis_tools.py](analysis/analysis_tools.py)
Miscellaneous tools for IO, reading metadata, image processing, etc.

# Examples
Scripts illustrated examples of different DMD simulations and analysis are stored in [examples](examples). Associated 
data is located in [examples/data](examples/data)
  
# Hardware control code

### [expt_ctrl/dlp6500.py](expt_ctrl/dlp6500.py)
Code for controlling the DLP6500 DMD over USB. This code was initially based on the approaches 
of https://github.com/mazurenko/Lightcrafter6500DMDControl and https://github.com/csi-dcsc/Pycrafter6500 

### [expt_ctrl/set_dmd_sim.py](expt_ctrl/set_dmd_sim.py)
This is the script used to set certain pattern sequences on the DMD from patterns which
have been previously loaded onto the firmware using the [Texas Instruments DLP6500 and DLP9000
GUI](https://www.ti.com/tool/DLPC900REF-SW). This script is intended to be called from the command line, which allows run_sim_triggerscope.bsh
to utilize dlp6500.py.

### [expt_ctrl/run_sim_triggerscop.bsh](expt_ctrl/run_sim_triggerscop.bsh)
This is a beanshell script which can be run from [MicroManager 2.0 Gamma](https://micro-manager.org/wiki/Micro-Manager)
to acquire SIM data  using dlp6500.py to set the DMD patterns (assuming they have already been loaded into the 
DMD firmware). This script programs a [Triggerscope 3B](https://arc.austinblanco.com/), which then provides the analog and 
digital voltages required to run the rest of the experiment. The camera free runs as the
master clock, triggering the triggerscope. We use a customized version of the Triggerscope firmware V601.

### [expt_ctrl/SIM.cfg](expt_ctrl/SIM.cfg)
Configuration file describing experimental equipment to MicroManager
  
# Instrument design
Mechanical drawings of some parts used in the setup are included in the 
[parts](parts) directory. For a more complete description of the setup and
a parts list, see the preprint linked at the top.


