"""
Reconstruct multichannel SIM images of argoSIM slide pattern of closely spaced line pairs.
The data needed to run this example can be downloaded from Zenodo as described in the README
"""
import datetime
import numpy as np
from numpy.fft import fftshift, fftfreq
import pickle
import tifffile
from pathlib import Path
from mcsim.analysis.sim_reconstruction import SimImageSet, show_sim_napari
from localize_psf.affine import xform_sinusoid_params, params2xform
from localize_psf.rois import get_centered_rois
from localize_psf.fit_psf import circ_aperture_otf

tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

# load data
root_dir = Path("data")
fname_raw_data = root_dir / "argosim_line_pairs.tif"

use_gpu = False

# ############################################
# load image data, channel/angle/phase
# ############################################
ncolors = 2
nangles = 3
nphases = 3
nx = 2048
ny = 2048
imgs = tifffile.imread(fname_raw_data).reshape([ncolors, nangles, nphases, ny, nx])

# ############################################
# set ROI to reconstruction, [cy, cx]
# ############################################
roi = get_centered_rois([791, 896], [850, 851])[0]
nx_roi = roi[3] - roi[2]
ny_roi = roi[1] - roi[0]

# ############################################
# set physical data
# ############################################
na = 1.3
pixel_size = 0.065
emission_wavelengths = [0.519, 0.580]
excitation_wavelengths = [0.465, 0.532]

# ############################################
# set OTF
# ############################################
# determined by fit to measured OTF
otf_attenuation_params = np.array([2.446])

def otf_fn(f, fmax): return (1 / (1 + (f / fmax * otf_attenuation_params[0]) ** 2) *
                             circ_aperture_otf(f, 0, na, 2 * na / fmax))


# ############################################
# load affine transformations from DMD to camera
# ############################################
affine_fnames = [root_dir / "2021-02-03_09;43;06_affine_xform_blue_z=0.pkl",
                 root_dir / "2021-02-03_09;43;06_affine_xform_green_z=0.pkl"]

affine_xforms = []
for p in affine_fnames:
    with open(p, 'rb') as f:
        affine_xforms.append(pickle.load(f)['affine_xform'])

# ############################################
# load DMD pattern information
# ############################################
dmd_pattern_data_fpath = [root_dir / "sim_patterns_period=6.01_nangles=3.pkl",
                          root_dir / "sim_patterns_period=6.82_nangles=3.pkl"]

frqs_dmd = np.zeros((ncolors, nangles, 2))
phases_dmd = np.zeros((ncolors, nangles, nphases))
for kk in range(ncolors):
    with open(dmd_pattern_data_fpath[kk], 'rb') as f:
        pattern_data = pickle.load(f)

    # DMD intensity frequency and phase (twice electric field frq/phase)
    frqs_dmd[kk] = 2 * pattern_data['frqs']
    phases_dmd[kk] = 2 * pattern_data['phases']
    dmd_nx = pattern_data['nx']
    dmd_ny = pattern_data['ny']

# ############################################
# do SIM reconstruction for each color
# ############################################
for kk in range(ncolors):
    # ###########################################
    # construct otf matrix
    # ###########################################
    fmax = 2 * na / emission_wavelengths[kk]
    fx = fftshift(fftfreq(nx_roi, pixel_size))
    fy = fftshift(fftfreq(ny_roi, pixel_size))
    ff = np.sqrt(np.expand_dims(fx, axis=0) ** 2 +
                 np.expand_dims(fy, axis=1) ** 2)
    otf = otf_fn(ff, fmax)
    otf[ff >= fmax] = 0

    # ###########################################
    # guess frequencies/phases in image ROI using affine transformations from DMD
    # output phase in FFT coordinate system, where first pixel is centered at -(n//2) along each axis
    # ###########################################
    xform_input2edge = params2xform([1, 0, (dmd_nx // 2),
                                     1, 0, (dmd_ny // 2)])
    xform_full2roi = params2xform([1, 0, -roi[2],
                                   1, 0, -roi[0]])
    xform_edge2output = params2xform([1, 0, -((roi[3] - roi[2]) // 2),
                                      1, 0, -((roi[1] - roi[0]) // 2)])
    xform_full = xform_edge2output.dot(xform_full2roi.dot(affine_xforms[kk].dot(xform_input2edge)))

    fx_guess, fy_guess, phases_guess = xform_sinusoid_params(frqs_dmd[kk, :, 0][:, None],
                                                             frqs_dmd[kk, :, 1][:, None],
                                                             phases_dmd[kk],
                                                             xform_full)
    # convert frequencies from 1/mirrors to 1/um
    frqs_guess = np.concatenate((fx_guess, fy_guess), axis=1) / pixel_size

    # ###########################################
    # initialize SIM reconstruction
    # ###########################################
    save_dir = root_dir / f"{tstamp:s}_sim_reconstruction_{excitation_wavelengths[kk] * 1e3:.0f}nm"

    imgset = SimImageSet.initialize({"pixel_size": pixel_size, "na": na, "wavelength": emission_wavelengths[kk]},
                                    imgs[kk, :, :, roi[0]:roi[1], roi[2]:roi[3]],
                                    otf=otf,
                                    wiener_parameter=0.1,
                                    frq_estimation_mode="band-correlation",
                                    frq_guess=frqs_guess,
                                    phase_estimation_mode="wicker-iterative",
                                    phases_guess=phases_guess,
                                    combine_bands_mode="fairSIM",
                                    fmax_exclude_band0=0.4,
                                    normalize_histograms=False,
                                    background=100,
                                    gain=2,
                                    min_p2nr=0.5,
                                    use_gpu=use_gpu)

    # ###########################################
    # run reconstruction
    # ###########################################
    imgset.reconstruct(compute_widefield=True,
                       compute_os=True,
                       compute_deconvolved=True,
                       compute_mcnr=True)
    imgset.print_parameters()
    fname_out = imgset.save_imgs(save_dir,
                                 format="zarr",  # "tiff"
                                 save_raw_data=False,
                                 save_patterns=False
                                 )
    imgset.plot_figs(save_dir,
                     diagnostics_only=True,
                     figsize=(20, 10),
                     imgs_dpi=300)
    show_sim_napari(fname_out)
