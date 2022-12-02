"""
Reconstruct synthetic SIM image of closely spaced line pairs.
"""
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from pathlib import Path
import numpy as np
from localize_psf.fit_psf import gridded_psf_model, get_psf_coords, psf2otf
import mcsim.analysis.sim_reconstruction as sim

tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
root_dir = Path("data")

use_gpu = False

# ############################################
# physical parameters
# ############################################
max_photons_per_pix = 1000

dxy = 0.065
nbin = 4
dxy_gt = dxy /nbin
na = 1.3
wavelength = 0.532
fmax = 2 * na / wavelength

# ############################################
# ground truth image
# ############################################
nxy_gt = 2048

def get_lines_test_pattern(img_size, angles=(15,), line_center_sep = np.arange(26, 0, -2)):
    """
    Generate patterns similar to argolight slide
    :param img_size:
    :param angles:
    :return test_patterns:
    :return line_sep: in pixels
    """

    my, mx = img_size

    width = 1
    # line_center_sep = width + line_edge_sep
    line_edge_sep = line_center_sep - width

    line_pair_sep = 2 * np.max(line_center_sep)

    n = np.sum(line_edge_sep) + len(line_center_sep) * 2 * width + (len(line_center_sep) - 1) * line_pair_sep

    gt = np.zeros((n, n))
    start = 0

    for ii in range(len(line_center_sep)):
        gt[:, start:start + width] = 1

        a = start + width + line_edge_sep[ii]
        gt[:, a:a + width] = 1

        start = a + width + line_pair_sep

    # pad image
    pxl = int(np.floor((mx - gt.shape[1]) / 2))
    pxr = int(mx - gt.shape[1] - pxl)

    pyu = int(np.floor((my - gt.shape[0]) / 2))
    pyd = int(my - gt.shape[0] - pyu)

    gtp = np.pad(gt, ((pyu, pyd), (pxl, pxr)), mode='constant')

    test_patterns = []
    for a in angles:
        img = Image.fromarray(gtp)
        test_patterns.append(np.asarray(img.rotate(a, expand=0)))

    test_patterns = np.asarray(test_patterns)

    return test_patterns, line_center_sep


gt, seps = get_lines_test_pattern((nxy_gt, nxy_gt))
gt *= max_photons_per_pix / nbin**2


# ############################################
# PSF
# ############################################
coords = get_psf_coords(gt.shape, (1, dxy_gt, dxy_gt))
psf = gridded_psf_model(wavelength, 1.5, "vectorial").model(coords, [1, 0, 0, 0, na, 0])
psf /= np.sum(psf)
otf, _ = psf2otf(psf[0], (dxy_gt, dxy_gt))

# ############################################
# synthetic SIM images
# ############################################
frqs_gt = np.array([[1, 0],
                    [np.cos(60 * np.pi/180), np.sin(60 * np.pi/180)],
                    [np.cos(120 * np.pi/180), np.sin(120 * np.pi/180)]]) * 0.8 * fmax

phases_gt = np.stack([np.array([0, 2*np.pi/3, 4*np.pi/3])] * 3, axis=0)
mod_depths_gt = np.ones((3))
amps_gt = np.ones((3, 3))

imgs, snrs, patterns = sim.get_simulated_sim_imgs(gt,
                                                  frqs_gt,
                                                  phases_gt,
                                                  mod_depths=mod_depths_gt,
                                                  gains=2,
                                                  offsets=100,
                                                  readout_noise_sds=5,
                                                  pix_size=dxy_gt,
                                                  amps=amps_gt,
                                                  otf=otf,
                                                  nbin=nbin)
imgs = imgs[:, :, 0]
patterns = patterns[:, :, 0]

# ############################################
# SIM reconstruction
# ############################################
tstart = time.perf_counter()

imgset = sim.SimImageSet({"pixel_size": dxy, "na": na, "wavelength": wavelength},
                         imgs,
                         frq_estimation_mode="band-correlation",
                         frq_guess=frqs_gt,
                         phases_guess=phases_gt,
                         phase_estimation_mode="wicker-iterative",
                         combine_bands_mode="fairSIM",
                         fmax_exclude_band0=0.4,
                         normalize_histograms=False,
                         otf=None,
                         wiener_parameter=0.3,
                         background=100,
                         gain=2,
                         min_p2nr=0.5,
                         use_gpu=use_gpu,
                         save_dir=root_dir / f"{tstamp:s}_sim_reconstruction_simulated",
                         interactive_plotting=False)

# run reconstruction
imgset.reconstruct()
# save reconstruction results
imgset.save_imgs(use_zarr=True)
# plot results
imgset.plot_figs(figsize=(20, 10),
                 imgs_dpi=300)

print(f"reconstructing images, plotting diagnostics, and saving results took {time.perf_counter() - tstart:.2f}s")
