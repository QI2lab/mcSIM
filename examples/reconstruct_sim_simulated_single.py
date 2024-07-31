"""
Reconstruct single channel, single time point synthetic SIM image of closely spaced line pairs.
"""
import datetime
import time
from PIL import Image
from pathlib import Path
import numpy as np
from localize_psf.fit_psf import gridded_psf_model, get_psf_coords
from mcsim.analysis.sim_reconstruction import (SimImageSet,
                                               get_sinusoidal_patterns,
                                               get_simulated_sim_imgs,
                                               show_sim_napari)

tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
root_dir = Path("data")

use_gpu = False

# ############################################
# physical parameters
# ############################################
max_photons_per_pix = 1000

dxy = 0.065
nbin = 4
dxy_gt = dxy / nbin
na = 1.3
wavelength = 0.532
fmax = 2 * na / wavelength

# ############################################
# ground truth image
# ############################################
nxy_gt = 2048


def get_lines_test_pattern(img_size,
                           angles=(15,),
                           line_center_sep=np.arange(26, 0, -2)):
    """
    Generate patterns similar to argolight slide

    :param img_size:
    :param angles:
    :param line_center_sep:
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
        test_patterns.append(np.asarray(img.rotate(a, expand=False)))

    test_patterns = np.asarray(test_patterns)

    return test_patterns, line_center_sep


gt, seps = get_lines_test_pattern((nxy_gt, nxy_gt))
gt *= max_photons_per_pix


# ############################################
# PSF
# ############################################
coords = get_psf_coords(gt.shape, (1, dxy_gt, dxy_gt))
psf = gridded_psf_model(wavelength, 1.5, "vectorial").model(coords, [1, 0, 0, 0, na, 0])
psf /= np.sum(psf)

# ############################################
# synthetic SIM images
# ############################################
frqs_gt = np.array([[1, 0],
                    [np.cos(60 * np.pi/180), np.sin(60 * np.pi/180)],
                    [np.cos(120 * np.pi/180), np.sin(120 * np.pi/180)],
                    ]) * 0.8 * fmax

phases_gt = np.stack([np.array([0, 2*np.pi/3, 4*np.pi/3])] * 3, axis=0)
mod_depths_gt = np.ones(3)
amps_gt = np.ones((3, 3))

patterns = get_sinusoidal_patterns(dxy,
                                   (nxy_gt // nbin, nxy_gt // nbin),
                                   np.kron(frqs_gt, np.ones((3, 1))),  # reshaped into 9 x 2
                                   phases_gt.reshape(9),
                                   np.kron(mod_depths_gt, np.ones(3)),
                                   amps_gt.reshape(9),
                                   n_oversampled=nbin
                                   )

imgs, snrs = get_simulated_sim_imgs(gt,
                                    patterns,
                                    gains=2,
                                    offsets=100,
                                    readout_noise_sds=5,
                                    psf=psf,
                                    nbin=nbin)

# reshape from 9 x nxy x nxy to 3 x 3 x nxy x nxy
imgs = imgs.reshape((3, 3, nxy_gt // nbin, nxy_gt // nbin))

# ############################################
# SIM reconstruction
# ############################################
tstart = time.perf_counter()

save_dir = root_dir / f"{tstamp:s}_sim_reconstruction_simulated"

imgset = SimImageSet.initialize({"pixel_size": dxy,
                                 "na": na,
                                 "wavelength": wavelength},
                                imgs,
                                otf=None,
                                wiener_parameter=0.3,
                                frq_estimation_mode="band-correlation",
                                frq_guess=frqs_gt,
                                phase_estimation_mode="wicker-iterative",
                                phases_guess=phases_gt,
                                combine_bands_mode="fairSIM",
                                fmax_exclude_band0=0.4,
                                normalize_histograms=False,
                                background=100,
                                gain=2,
                                min_p2nr=0.5,
                                use_gpu=use_gpu)

# run reconstruction
imgset.reconstruct(compute_widefield=True,
                   compute_os=True,
                   compute_deconvolved=True,
                   compute_mcnr=True)

# save reconstruction results
fname_out = imgset.save_imgs(save_dir,
                             format="zarr",
                             save_patterns=True,
                             save_raw_data=True)
# plot results
imgset.plot_figs(save_dir,
                 figsize=(20, 10),
                 imgs_dpi=300)

print(f"reconstructing images, plotting diagnostics, and saving results took "
      f"{time.perf_counter() - tstart:.2f}s")

# display results in napari
show_sim_napari(fname_out, clims=(0., 300.))
