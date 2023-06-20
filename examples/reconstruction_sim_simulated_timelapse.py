"""
Reconstruct time-lapse SIM images of simulated microtubules.
"""
import datetime
import numpy as np
from pathlib import Path
import tifffile
from mcsim.analysis import sim_reconstruction as sim

# load images (images can be downloaded from Zenodo repository as described in README)
root_dir = Path(r"data")

tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
save_dir = root_dir / f"{tstamp:s}_sim_reconstruction_timelapse"

# load data
imgs = tifffile.imread(root_dir / "synthetic_microtubules.tiff")

# repeat image to simulate time-lapse
imgs = np.tile(imgs[None, ...], [61, 1, 1, 1, 1])

# physical parameters
na = 1.3 # numerical aperture
dxy = 0.065 # um
wavelength = 0.488 # um

# ###########################################
# initialize data
# ###########################################
imgset = sim.SimImageSet.initialize({"pixel_size": dxy,
                                     "na": na,
                                     "wavelength": wavelength},
                                     imgs,
                                     otf=None,
                                     wiener_parameter=0.3,
                                     frq_estimation_mode="band-correlation",
                                     # frq_guess=frqs_gt, # todo: can add frequency guesses for more reliable fitting
                                     phase_estimation_mode="wicker-iterative",
                                     phases_guess=np.array([[0, 2*np.pi / 3, 4 * np.pi / 3],
                                                            [0, 2*np.pi / 3, 4 * np.pi / 3],
                                                            [0, 2*np.pi / 3, 4 * np.pi / 3]]),
                                     combine_bands_mode="fairSIM",
                                     fmax_exclude_band0=0.4,
                                     normalize_histograms=False,
                                     background=100,
                                     gain=2,
                                     use_gpu=False)


# ###########################################
# run reconstruction
# this includes parameter estimation
# ###########################################
imgset.reconstruct(slices=(slice(0, 1),), # determine SIM parameters from first time-point
                   compute_widefield=True,
                   compute_os=False,
                   compute_deconvolved=False,
                   compute_mcnr=True)

# ###########################################
# print parameters
# ###########################################
imgset.print_parameters()

# ###########################################
# save reconstruction results
# ###########################################
imgset.save_imgs(save_dir,
                 format="tiff",  # format="zarr",
                 save_raw_data=False,
                 save_patterns=False)

# ###########################################
# save diagnostic plots
# ###########################################
imgset.plot_figs(save_dir,
                 diagnostics_only=True,
                 figsize=(20, 10),
                 imgs_dpi=300)