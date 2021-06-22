"""
Reconstruct SIM image of argoSIM slide pattern of closely spaced line pairs.
"""
import sim_reconstruction as sim
import analysis_tools as tools
import fit_psf as psf
import affine
import tifffile
import pickle
import numpy as np

# ############################################
# load image data, channel/angle/phase
# ############################################
ncolors = 2
nangles = 3
nphases = 3
nx = 2048
ny = 2048
imgs = tifffile.imread("data/argosim_line_pairs.tif").reshape([ncolors, nangles, nphases, ny, nx])

# ############################################
# set ROI to reconstruction, [cy, cx]
# ############################################
roi = tools.get_centered_roi([791, 896], [850, 850])
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
# load OTF data
# ############################################
otf_data_path = "data/2020_05_19_otf_fit_blue.pkl"

with open(otf_data_path, 'rb') as f:
    otf_data = pickle.load(f)
otf_p = otf_data['fit_params']

otf_fn = lambda f, fmax: 1 / (1 + (f / fmax * otf_p[0]) ** 2) * psf.circ_aperture_otf(f, 0, na, 2 * na / fmax)

# ############################################
# load affine transformations from DMD to camera
# ############################################
affine_fnames = ["data/2021-02-03_09;43;06_affine_xform_blue_z=0.pkl",
                 "data/2021-02-03_09;43;06_affine_xform_green_z=0.pkl"]

affine_xforms = []
for p in affine_fnames:
    with open(p, 'rb') as f:
        affine_xforms.append(pickle.load(f)['affine_xform'])

# ############################################
# load DMD patterns frequency and phase data
# ############################################
dmd_pattern_data_fpath = [r"data/sim_patterns_period=6.01_nangles=3.pkl",
                          r"data/sim_patterns_period=6.82_nangles=3.pkl"]

frqs_dmd = np.zeros((2, 3, 2))
phases_dmd = np.zeros((ncolors, nangles, nphases))
for kk in range(ncolors):
    ppath = dmd_pattern_data_fpath[kk]
    xform = affine_xforms[kk]

    with open(ppath, 'rb') as f:
        pattern_data = pickle.load(f)

    # DMD intensity frequency and phase (twice electric field frq/phase)
    frqs_dmd[kk] = 2 * pattern_data['frqs']
    phases_dmd[kk] = 2 * pattern_data['phases']
    dmd_nx = pattern_data['nx']
    dmd_ny = pattern_data['ny']

for kk in range(ncolors):
    # otf matrix
    fmax = 1 / (0.5 * emission_wavelengths[kk] / na)
    fx = tools.get_fft_frqs(nx_roi, pixel_size)
    fy = tools.get_fft_frqs(ny_roi, pixel_size)
    ff = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
    otf = otf_fn(ff, fmax)
    otf[ff >= fmax] = 0

    # guess frequencies/phases
    frqs_guess = np.zeros((nangles, 2))
    phases_guess = np.zeros((nangles, nphases))
    for ii in range(nangles):
        for jj in range(nphases):
            # estimate frequencies based on affine_xform
            frqs_guess[ii, 0], frqs_guess[ii, 1], phases_guess[ii, jj] = \
                affine.xform_sinusoid_params_roi(frqs_dmd[kk, ii, 0], frqs_dmd[kk, ii, 1],
                                                 phases_dmd[kk, ii, jj], [dmd_ny, dmd_nx], roi, xform)

    # convert from 1/mirrors to 1/um
    frqs_guess = frqs_guess / pixel_size


    imgset = sim.SimImageSet({"pixel_size": pixel_size, "na": na, "wavelength": emission_wavelengths[kk]},
                             imgs[kk, :, :, roi[0]:roi[1], roi[2]:roi[3]],
                             frq_guess=frqs_guess, phases_guess=phases_guess, otf=otf,
                             wiener_parameter=0.1, phase_estimation_mode="wicker-iterative",
                             frq_estimation_mode="band-correlation", combine_bands_mode="fairSIM",
                             normalize_histograms=True,
                             background=100, gain=2, min_p2nr=0.5,
                             save_dir="data/sim_reconstruction_%.0fnm" % (excitation_wavelengths[kk] * 1e3),
                             interactive_plotting=True, figsize=(20, 13))
    imgset.reconstruct()
    imgset.plot_figs()
    imgset.save_imgs()
    imgset.log_file.close()