"""
Tools for reconstructing 2D SIM images, using 3-angles and 3-phases.
"""
import tifffile

import analysis_tools as tools
import fit_psf as psf
import affine
import psd
import camera_noise
import fit

# general imports
import pickle
import os
import time
import datetime
import copy
import warnings
import shutil
import joblib

# numerical tools
import numpy as np
from scipy import fft
import scipy.optimize
import scipy.signal
import scipy.ndimage
from skimage.exposure import match_histograms
from PIL import Image

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.colors import LogNorm
import matplotlib.patches

def reconstruct_sim_dataset(data_dirs, pixel_size, na, emission_wavelengths, excitation_wavelengths,
                            affine_data_paths, otf_data_path, dmd_pattern_data_path,
                            nangles=3, nphases=3, npatterns_ignored=0, crop_rois=None, fit_all_sim_params=False,
                            plot_results=True,
                            channel_inds=None, zinds_to_use=None, tinds_to_use=None, xyinds_to_use=None,
                            saving=True, tif_stack=True, **kwargs):
    """
    Reconstruct entire folder of SIM data and save results in TIF stacks. Responsible for loading relevant data
    (images, affine transformations, SIM pattern information), selecting images to recombine from metadata, and
    saving results for SIM superresolution, deconvolution, widefield image, etc.

    :param list data_dirs: list of directories where data is stored
    :param float pixel_size: pixel size in ums
    :param float na: numerical aperture
    :param list emission_wavelengths: list of emission wavelengths in um
    :param list excitation_wavelengths: list of excitation wavelengths in um
    :param list affine_data_paths: list of paths to files storing data about affine transformations between DMD and camera
    space. [path_color_0, path_color_1, ...]. The affine data files store pickled dictionary objects. The dictionary
    must have an entry 'affine_xform' which contains the affine transformation matrix (in homogeneous coordinates)
    :param str otf_data_path: path to file storing optical transfer function data. Data is a pickled dictionary object
    and must have entry 'fit_params'.
    :param list dmd_pattern_data_path: list of paths to files storing data about DMD patterns for each color. Data is
    stored in a pickled dictionary object which must contain fields 'frqs', 'phases', 'nx', and 'ny'
    :param int nangles: number of angle images
    :param int nphases: number of phase images
    :param int npatterns_ignored: number of patterns to ignore at the start of each channel.
    :param crop_rois: [[ystart_0, yend_0, xstart_0, xend_0], [ystart_1, ...], ...]
    :param list img_centers: list of centers for images in each data directory to be used in cropping [[cy, cx], ...]
    :param list or int crop_sizes: list of crop sizes for each data directory
    :param list channel_inds: list of channel indices corresponding to each color. If set to None, will use [0, 1, ..., ncolors -1]
    :param list zinds_to_use: list of z-position indices to reconstruct
    :param list tinds_to_use: list of time indices to reconstruct
    :param list xyinds_to_use: list of xy-position indices to reconstruct
    :param bool saving: if True, save results
    :param bool tif_stack: save results as tif stack. Otherwise save as individual tiff images
    :param **kwargs: passed through to reconstruction

    :return np.ndarray imgs_sr:
    :return np.ndarray imgs_wf:
    :return np.ndarray imgs_deconvolved:
    :return np.ndarray imgs_os:
    """

    nfolders = len(data_dirs)
    if nfolders == 0:
        raise ValueError("No folder paths were provided.")

    ncolors = len(emission_wavelengths)
    if ncolors == 0:
        raise ValueError("No wavelength channels were provided.")

    if channel_inds is None:
        channel_inds = list(range(ncolors))

    # ensure crop_rois is a list the same size as number of folders
    if isinstance(crop_rois, list) and not isinstance(crop_rois[0], list):
        crop_rois = [crop_rois]

    if len(crop_rois) == 1 and nfolders > 1:
        crop_rois = crop_rois * nfolders

    if crop_rois is None:
        crop_rois = [None] * nfolders

    # ############################################
    # load affine data
    # ############################################
    affine_xforms = []
    for p in affine_data_paths:
        with open(p, 'rb') as f:
            affine_xforms.append(pickle.load(f)['affine_xform'])

    # ############################################
    # load DMD patterns frequency and phase data
    # ############################################
    frqs_dmd = np.zeros((ncolors, nangles, 2))
    phases_dmd = np.zeros((ncolors, nangles, nphases))
    for kk in range(ncolors):
        ppath = dmd_pattern_data_path[kk]
        xform = affine_xforms[kk]

        with open(ppath, 'rb') as f:
            pattern_data = pickle.load(f)

        # DMD intensity frequency and phase (twice electric field frq/phase)
        frqs_dmd[kk] = 2 * pattern_data['frqs']
        phases_dmd[kk] = 2 * pattern_data['phases']
        dmd_nx = pattern_data['nx']
        dmd_ny = pattern_data['ny']

    # ############################################
    # load OTF data
    # ############################################
    with open(otf_data_path, 'rb') as f:
        otf_data = pickle.load(f)
    otf_p = otf_data['fit_params']

    if len(otf_p) == 1:
        otf_fn = lambda f, fmax: 1 / (1 + (f / fmax * otf_p[0]) ** 2) * \
                                 psf.circ_aperture_otf(f, 0, na, 2 * na / fmax)
    else:
        otf_fn = lambda f, fmax: 1 / (
                    1 + (f / fmax * otf_p[0]) ** 2 + (f / fmax * otf_p[1]) ** 4 + (f / fmax * otf_p[2]) ** 6 +
                    (f / fmax * otf_p[3]) ** 8) * psf.circ_aperture_otf(f, 0, na, 2 * na / fmax)

    # ############################################
    # SIM images
    # ############################################
    for rpath, crop_roi in zip(data_dirs, crop_rois):
        folder_path, folder = os.path.split(rpath)
        print("# ################################################################################")
        print("analyzing folder: %s" % folder)
        print("located in: %s" % folder_path)

        tstamp = tools.get_timestamp()
        # path to store processed results
        if saving:
            sim_results_path = os.path.join(rpath, '%s_sim_reconstruction' % tstamp)
            if not os.path.exists(sim_results_path):
                os.mkdir(sim_results_path)
            print("save directory: %s" % sim_results_path)

            # copy useful data files here
            for kk in range(ncolors):
                # copy affine data here
                _, fname = os.path.split(affine_data_paths[kk])
                fpath = os.path.join(sim_results_path, fname)
                shutil.copyfile(affine_data_paths[kk], fpath)

                # copy otf data here
                _, fname = os.path.split(otf_data_path)
                fpath = os.path.join(sim_results_path, fname)
                shutil.copyfile(otf_data_path, fpath)

                # copy DMD pattern data here
                _, fname = os.path.split(dmd_pattern_data_path[kk])
                fpath = os.path.join(sim_results_path, fname)
                shutil.copyfile(dmd_pattern_data_path[kk], fpath)

        # load metadata
        metadata, dims, summary = tools.parse_mm_metadata(rpath)
        start_time = datetime.datetime.strptime(summary['StartTime'],  '%Y-%d-%m;%H:%M:%S.%f')
        nz = dims['z']
        nxy = dims['position']
        nt = dims['time']

        # use this construction as zinds can be different for different folders
        if zinds_to_use is None:
            zinds_to_use_temp = range(nz)
        else:
            zinds_to_use_temp = zinds_to_use
        nz_used = len(zinds_to_use_temp)

        if tinds_to_use is None:
            tinds_to_use_temp = range(nt)
        else:
            tinds_to_use_temp = tinds_to_use
        nt_used = len(tinds_to_use_temp)

        if xyinds_to_use is None:
            xyinds_to_use_temp = range(nxy)
        else:
            xyinds_to_use_temp = xyinds_to_use
        nxy_used = len(xyinds_to_use_temp)

        if pixel_size is None:
            pixel_size = metadata['PixelSizeUm'][0]

        # load metadata from one file to check size
        fname = os.path.join(rpath, metadata['FileName'].values[0])
        tif = tifffile.TiffFile(fname, multifile=False)
        ny_raw, nx_raw = tif.series[0].shape[-2:]

        if crop_roi is not None:
            # check points don't exceed image size
            if crop_roi[0] < 0:
                crop_roi[0] = 0
            if crop_roi[1] > ny_raw:
                crop_roi[1] = ny_raw
            if crop_roi[2] < 0:
                crop_roi[2] = 0
            if crop_roi[3] > nx_raw:
                crop_roi[3] = nx_raw
        else:
            crop_roi = [0, ny_raw, 0, nx_raw]

        ny = crop_roi[1] - crop_roi[0]
        nx = crop_roi[3] - crop_roi[2]

        # timing
        tstart_all = time.perf_counter()
        # arrays to save results
        imgs_sr = []
        imgs_os = []
        imgs_wf = []
        imgs_deconvolved = []
        counter = 1
        for kk in range(ncolors):
            sim_options = {'pixel_size': pixel_size, 'wavelength': emission_wavelengths[kk], 'na': na}

            # estimate otf
            fmax = 1 / (0.5 * emission_wavelengths[kk] / na)
            fx = tools.get_fft_frqs(nx, sim_options['pixel_size'])
            fy = tools.get_fft_frqs(ny, sim_options['pixel_size'])
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
                                                         phases_dmd[kk, ii, jj], [dmd_ny, dmd_nx], crop_roi, xform)

            # convert from 1/mirrors to 1/um
            frqs_guess = frqs_guess / pixel_size

            # analyze pictures
            mod_depths_real = []
            frqs_real = []
            phases_real = []
            for ixy in xyinds_to_use_temp:
                for iz in zinds_to_use_temp:
                    for ind_t in tinds_to_use_temp:
                        tstart_single_index = time.perf_counter()

                        identifier = "%.0fnm_nt=%d_nxy=%d_nz=%d" % (excitation_wavelengths[kk] * 1e3, ind_t, ixy, iz)
                        file_identifier = "nc=%d_nt=%d_nxy=%d_nz=%d" % (kk, ind_t, ixy, iz)

                        # where we will store results for this particular set
                        sim_diagnostics_path = os.path.join(sim_results_path, identifier)
                        if not os.path.exists(sim_diagnostics_path):
                            os.mkdir(sim_diagnostics_path)

                        # find images and load them
                        raw_imgs = tools.read_dataset(metadata, time_indices=ind_t, z_indices=iz, xy_indices=ixy,
                                                      user_indices={"UserChannelIndex": channel_inds[kk],
                                                                    "UserSimIndex": list(range(npatterns_ignored,
                                                                                               npatterns_ignored + nangles * nphases))})

                        # error if we have wrong number of images
                        if np.shape(raw_imgs)[0] != (nangles * nphases):
                            raise ValueError("Found %d images, but expected %d images at channel=%d,"
                                            " zindex=%d, tindex=%d, xyindex=%d" %
                                             (np.shape(raw_imgs)[0], nangles * nphases, channel_inds[kk], iz, ind_t, ixy))

                        # reshape to [nangles, nphases, ny, nx]
                        imgs_sim = raw_imgs
                        imgs_sim = imgs_sim.reshape((nangles, nphases, raw_imgs.shape[1], raw_imgs.shape[2]))
                        imgs_sim = imgs_sim[:, :, crop_roi[0]:crop_roi[1], crop_roi[2]:crop_roi[3]]

                        # instantiate reconstruction object
                        if fit_all_sim_params or ind_t == 0:
                            img_set = SimImageSet(sim_options, imgs_sim, frqs_guess, otf=otf, phases_guess=phases_guess,
                                                  save_dir=sim_diagnostics_path, **kwargs)
                            img_set.reconstruct()

                            # save fit params for next iteration
                            mod_depths_real = img_set.mod_depths
                            phases_real = img_set.phases
                            frqs_real = img_set.frqs

                        else:
                            kwargs_reduced = copy.deepcopy(kwargs)
                            kwargs_reduced["frq_estimation_mode"] = "fixed"
                            kwargs_reduced["phase_estimation_mode"] = "fixed"
                            kwargs_reduced["use_fixed_mod_depths"] = True
                            img_set = SimImageSet(sim_options, imgs_sim, frqs_real, otf=otf, phases_guess=phases_real,
                                                  mod_depths_guess=mod_depths_real, save_dir=sim_diagnostics_path,
                                                  **kwargs_reduced)
                            img_set.reconstruct()

                        if plot_results:
                            # plot results
                            img_set.plot_figs()

                        # save reconstruction summary data
                        img_set.save_result(os.path.join(sim_diagnostics_path, "sim_reconstruction_params.pkl"))

                        if saving and not tif_stack:
                            img_set.save_imgs(sim_results_path, start_time, file_identifier)
                        else:
                            # store images
                            imgs_os.append(img_set.sim_os)
                            imgs_wf.append(img_set.widefield)
                            imgs_sr.append(img_set.sim_sr)
                            imgs_deconvolved.append(img_set.widefield_deconvolution)

                        tend = time.perf_counter()
                        img_set.print_log("Reconstructed %d/%d from %s in %0.2fs" %
                                          (counter, ncolors * nt_used * nxy_used * nz_used, folder, tend - tstart_single_index))

                        # delete so destructor is called and log file closes
                        del img_set

                        counter += 1
        print("Finished %d reconstructions in %0.2fs" % (counter-1, time.perf_counter() - tstart_all))

        # #################################
        # save data for all reconstructed files
        # #################################
        if saving and tif_stack:
            tstart_save = time.perf_counter()

            # todo: want to include metadata in tif.
            # todo: this will fail if multiple positions
            fname = tools.get_unique_name(os.path.join(sim_results_path, 'widefield.tif'))
            imgs_wf = np.asarray(imgs_wf)
            wf_to_save = np.reshape(imgs_wf, [ncolors, nz_used, nt_used, imgs_wf[0].shape[-2], imgs_wf[0].shape[-1]])
            tools.save_tiff(wf_to_save, fname, dtype=np.float32, axes_order="CZTYX", hyperstack=True,
                            datetime=start_time)

            fname = tools.get_unique_name(os.path.join(sim_results_path, 'sim_os.tif'))
            imgs_os = np.asarray(imgs_os)
            sim_os = np.reshape(imgs_os, [ncolors, nz_used, nt_used, imgs_os[0].shape[-2], imgs_os[0].shape[-1]])
            tools.save_tiff(sim_os, fname, dtype=np.float32, axes_order="CZTYX", hyperstack=True, datetime=start_time)

            fname = tools.get_unique_name(os.path.join(sim_results_path, 'sim_sr.tif'))
            imgs_sr = np.asarray(imgs_sr)
            sim_to_save = np.reshape(imgs_sr, [ncolors, nz_used, nt_used, imgs_sr[0].shape[-2], imgs_sr[0].shape[-1]])
            tools.save_tiff(sim_to_save, fname, dtype=np.float32, axes_order="CZTYX", hyperstack=True,
                            datetime=start_time)

            fname = tools.get_unique_name(os.path.join(sim_results_path, 'deconvolved.tif'))
            imgs_deconvolved = np.asarray(imgs_deconvolved)
            deconvolved_to_save = np.reshape(imgs_deconvolved, [ncolors, nz_used, nt_used, imgs_deconvolved[0].shape[-2],
                                                                imgs_deconvolved[0].shape[-1]])
            tools.save_tiff(deconvolved_to_save, fname, dtype=np.float32, axes_order='CZTYX', hyperstack=True,
                            datetime=start_time)

            print("saving tiff stacks took %0.2fs" % (time.perf_counter() - tstart_save))

    return imgs_sr, imgs_wf, imgs_deconvolved, imgs_os

class SimImageSet:
    def __init__(self, physical_params, imgs, frq_guess=None, phases_guess=None, mod_depths_guess=None,
                 otf=None, wiener_parameter=0.1,
                 phase_estimation_mode="wicker-iterative", frq_estimation_mode="band-correlation", combine_bands_mode="Lal",
                 use_fixed_mod_depths=False, normalize_histograms=True, determine_amplitudes=False,
                 background=0, gain=2, max_phase_err=10 * np.pi / 180, min_p2nr=1, fbounds=(0.01, 1),
                 interactive_plotting=False, save_dir=None, use_gpu=CUPY_AVAILABLE, figsize=(20, 10)):
        """
        Reconstruct raw SIM data into widefield, SIM-SR, SIM-OS, and deconvolved images.

        :param physical_params: {'pixel_size', 'na', 'wavelength'}. Pixel size and wavelength in um
        :param imgs: nangles x nphases x ny x nx
        :param frq_guess: 2 x nangles array of guess SIM frequency values
        :param phases_guess: If use_fixed_phase is True, these phases are used
        :param mod_depths_guess: If use_fixed_mod_depths is True, these modulation depths are used
        :param otf: optical transfer function evaluated at the same frequencies as the fourier transforms of imgs.
         If None, estimate from NA.
        :param wiener_parameter: Attenuation parameter for Wiener filtering. This has a different meaning depending
        on the value of combine_bands_mode
        :param combine_bands_mode: "fairSIM" if using method of https://doi.org/10.1038/ncomms10980 or "Lal" if using method
        of https://doi.org/10.1109/jstqe.2016.2521542
        :param str phase_estimation_mode: "wicker-iterative", "real-space", "fixed"
        :param str frq_estimation_mode: "band-correlation", "fourier-transform", or "fixed"
        :param str combine_bands_mode: "Lal" or "fairSIM"
        :param bool use_fixed_mod_depths:
        :param bool normalize_histograms: for each phase, normalize histograms of images to account for laser power fluctuations
        :param background: Either a single number, or broadcastable to size of imgs. The background will be subtracted
         before running the SIM reconstruction
        :param bool determine_amplitudes: whether or not to determine amplitudes as part of Wicker phase optimization.
        These flag only matters if use_wicker is True
        :param max_phase_err: If the determined phase error between components exceeds this value, use the guess values instead
        :param min_p2nr: if the modulation-contrast-to-noise ratio is smaller than this value, use guess frequencies instead
         of fit frequencies
        :param fbounds: frequency bounds as a fraction of fmax to be used in power spectrum fit
        :param bool interactive_plotting: show plots in python GUI windows, or save outputs only
        :param save_dir: directory to save results
        :param use_gpu:
        :param figsize:
        """
        # #############################################
        # open log file
        # #############################################
        self.save_dir = save_dir

        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            self.log_file = open(os.path.join(self.save_dir, "sim_log.txt"), 'w')
        else:
            self.log_file = None

        # #############################################
        # print current time
        # #############################################
        now = datetime.datetime.now()

        self.print_log("####################################################################################")
        self.print_log("%d/%02d/%02d %02d:%02d:%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        self.print_log("####################################################################################")

        # #############################################
        # plot settings
        # #############################################
        self.interactive_plotting = interactive_plotting
        if not self.interactive_plotting:
            plt.ioff()
            plt.switch_backend("agg")

        self.figsize = figsize

        # #############################################
        # analysis settings
        # #############################################
        self.wiener_parameter = wiener_parameter
        self.normalize_histograms = normalize_histograms
        self.determine_amplitudes = determine_amplitudes
        self.max_phase_error = max_phase_err
        self.min_p2nr = min_p2nr
        self.use_gpu = use_gpu

        self.combine_mode = combine_bands_mode
        self.use_fixed_mod_depths = use_fixed_mod_depths
        self.phase_estimation_mode = phase_estimation_mode

        if phases_guess is None:
            self.frq_estimation_mode = "fourier-transform"
            self.print_log("No phase guesses provided, defaulting to frq_estimation_mode = '%s'" % "fourier_transform")
        else:
            self.frq_estimation_mode = frq_estimation_mode

        # #############################################
        # images
        # #############################################
        self.imgs = imgs.astype(np.float64)
        self.nangles, self.nphases, self.ny, self.nx = imgs.shape
        
        # #############################################
        # get basic parameters
        # #############################################
        self.dx = physical_params['pixel_size']
        self.dy = physical_params['pixel_size']
        self.na = physical_params['na']
        self.wavelength = physical_params['wavelength']

        self.fmax = 1 / (0.5 * self.wavelength / self.na)
        self.fbounds = fbounds

        self.frqs_guess = frq_guess
        self.phases_guess = phases_guess
        self.mod_depths_guess = mod_depths_guess

        # #############################################
        # get frequency data and OTF
        # #############################################
        self.fx = tools.get_fft_frqs(self.nx, self.dx)
        self.fy = tools.get_fft_frqs(self.ny, self.dy)

        if otf is None:
            otf = psf.circ_aperture_otf(self.fx[None, :], self.fy[:, None], self.na, self.wavelength)

        if np.any(otf < 0) or np.any(otf > 1):
            raise ValueError("OTF must be >= 0 and <= 1")

        self.otf = otf

        # #############################################
        # remove background
        # #############################################
        self.imgs = (self.imgs - background) / 2
        self.imgs[self.imgs <= 0] = 1e-12

        # #############################################
        # print intensity information
        # #############################################
        mean_int = np.mean(self.imgs, axis=(2, 3))
        rel_int_phases = mean_int / np.expand_dims(np.max(mean_int, axis=1), axis=1)

        mean_int_angles = np.mean(self.imgs, axis=(1, 2, 3))
        rel_int_angles = mean_int_angles / np.max(mean_int_angles)

        for ii in range(self.nangles):
            self.print_log("Angle %d,relative intensity = %0.3f" % (ii, rel_int_angles[ii]))
            self.print_log("phase relative intensities = ", end="")
            for jj in range(self.nphases):
                self.print_log("%0.3f, " % rel_int_phases[ii, jj], end="")
            self.print_log("", end="\n")

        # #############################################
        # normalize histograms for each angle
        # #############################################
        if self.normalize_histograms:
            tstart = time.perf_counter()

            for ii in range(self.nangles):
                for jj in range(1, self.nphases):
                    self.imgs[ii, jj] = match_histograms(self.imgs[ii, jj], self.imgs[ii, 0])

            self.print_log("Normalizing histograms took %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # Fourier transform SIM images
        # #############################################
        tstart = time.perf_counter()

        # todo: > 2/3 of the time of this step is taken up by the periodic/smooth decomposition. FFT ~7 times faster on GPU
        # use periodic/smooth decomposition instead of traditional apodization
        # periodic_portion = np.zeros((self.nangles, self.nphases, self.ny, self.nx), dtype=np.complex)
        # for jj in range(self.nangles):
        #     for kk in range(self.nphases):
        #         periodic_portion[jj, kk], _ = psd.periodic_smooth_decomp(self.imgs[jj, kk])

        apodization = scipy.signal.windows.tukey(self.imgs.shape[2], alpha=0.1)[None, :] * \
                      scipy.signal.windows.tukey(self.imgs.shape[3], alpha=0.1)[:, None]
        periodic_portion = self.imgs * np.expand_dims(apodization, axis=(0, 1))

        if self.use_gpu:
            self.imgs_ft = cp.asnumpy(cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(periodic_portion, axes=(2, 3)), axes=(2, 3)), axes=(2, 3)))
            # t = cp.asnumpy(cp.fft.rfft2(cp.fft.ifftshift(periodic_portion, axes=(2, 3)), axes=(2, 3)))
        else:
            self.imgs_ft = fft.fftshift(fft.fft2(fft.ifftshift(periodic_portion, axes=(2, 3)), axes=(2, 3)), axes=(2, 3))

        self.print_log("FT images took %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # get widefield image
        # #############################################
        tstart = time.perf_counter()

        self.widefield = np.nanmean(self.imgs, axis=(0, 1))
        wf_to_xform, _ = psd.periodic_smooth_decomp(self.widefield)
        self.widefield_ft = fft.fftshift(fft.fft2(fft.ifftshift(wf_to_xform)))

        self.print_log("Computing widefield image took %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # get optically sectioned image
        # #############################################
        tstart = time.perf_counter()
        self.sim_os = np.mean(sim_optical_section(self.imgs, axis=1), axis=0)
        self.print_log("Computing SIM-OS image took %0.2fs" % (time.perf_counter() - tstart))

    def __del__(self):
        if self.log_file is not None:
            try:
                self.log_file.close()
            except AttributeError:
                pass

    def reconstruct(self):
        """
        Handle SIM reconstruction, including parameter estimation, image combination, displaying useful information.
        :param figsize:
        :return:
        """

        # #############################################
        # estimate frequencies
        # #############################################
        tstart = time.perf_counter()

        if self.frq_estimation_mode == "fixed":
            self.frqs = self.frqs_guess
        elif self.frq_estimation_mode == "fourier-transform":
            # determine SIM frequency directly from Fourier transform
            self.frqs = self.estimate_sim_frqs(self.imgs_ft, self.imgs_ft)
        elif self.frq_estimation_mode == "band-correlation":
            # determine SIM frequency from separated frequency bands using guess phases
            mod_depths_temp = np.ones((self.nangles, self.nphases))
            self.bands_unmixed_ft = do_sim_band_separation(self.imgs_ft, self.phases_guess, mod_depths_temp)

            band0 = np.expand_dims(self.bands_unmixed_ft[:, 0], axis=1)
            band1 = np.expand_dims(self.bands_unmixed_ft[:, 1], axis=1)
            self.frqs = self.estimate_sim_frqs(band0, band1)
        else:
            raise ValueError("frq_estimation_mode must be 'fixed', 'fourier-transform',"
                             " or 'band-correlation' but was '%s'" % self.frq_estimation_mode)

        # for convenience also store periods and angles
        self.periods = 1 / np.sqrt(self.frqs[:, 0] ** 2 + self.frqs[:, 1] ** 2)
        self.angles = np.angle(self.frqs[:, 0] + 1j * self.frqs[:, 1])

        self.print_log("estimating %d frequencies took %0.2fs using mode '%s'" %
                       (self.nangles, time.perf_counter() - tstart, self.frq_estimation_mode))

        # #############################################
        # OTF value at frqs
        # #############################################
        otf_vals = np.zeros(self.nangles)
        
        for ii in range(self.nangles):
            ix = np.argmin(np.abs(self.frqs[ii, 0] - self.fx))
            iy = np.argmin(np.abs(self.frqs[ii, 1] - self.fy))
            otf_vals[ii] = self.otf[iy, ix]

        self.otf_at_frqs = otf_vals
        
        # #############################################
        # estimate peak heights
        # #############################################
        tstart = time.perf_counter()

        peak_heights = np.zeros((self.nangles, self.nphases))
        noise = np.zeros((self.nangles, self.nphases))
        p2nr = np.zeros((self.nangles, self.nphases))
        for ii in range(self.nangles):
            for jj in range(self.nphases):
                peak_heights[ii, jj] = np.abs(tools.get_peak_value(self.imgs_ft[ii, jj], self.fx, self.fy, self.frqs[ii], peak_pixel_size=1))
                noise[ii, jj] = np.sqrt(get_noise_power(self.imgs_ft[ii, jj], self.fx, self.fy, self.fmax))
                p2nr[ii, jj] = peak_heights[ii, jj] / noise[ii, jj]

            # if mcnr is too low use guess values instead
            if np.min(p2nr[ii]) < self.min_p2nr and self.frqs_guess is not None:
                self.frqs[ii] = self.frqs_guess[ii]
                self.print_log("Angle %d, minimum SIM peak-to-noise ratio = %0.2f is less than the minimum value, %0.2f,"
                               " so fit frequency will be replaced with guess"
                               % (ii, np.min(p2nr[ii]), self.min_p2nr))

                for jj in range(self.nphases):
                    peak_heights[ii, jj] = np.abs(tools.get_peak_value(self.imgs_ft[ii, jj], self.fx, self.fy, self.frqs[ii], peak_pixel_size=1))
                    p2nr[ii, jj] = peak_heights[ii, jj] / noise[ii, jj]

        self.cam_noise_rms = noise
        self.p2nr = p2nr

        self.print_log("estimated peak-to-noise ratio in %0.2fs" % (time.perf_counter() - tstart))
        
        # #############################################
        # estimate spatial-resolved MCNR
        # #############################################
        # following the proposal of https://doi.org/0.1038/s41592-021-01167-7
        # calculate as the ratio of the modulation size over the expected shot noise value
        # note: this is the same as sim_os / sqrt(wf_angle) up to a factor
        tstart = time.perf_counter()

        img_angle_ft = fft.fft(fft.ifftshift(self.imgs, axes=1), axis=1)
        self.mcnr = 2 * np.abs(img_angle_ft[:, 1]) / np.sqrt(np.abs(img_angle_ft[:, 0]))

        self.print_log("estimated modulation-contrast-to-noise ratio in %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # estimate phases
        # #############################################
        tstart = time.perf_counter()

        self.phases, self.amps = self.estimate_sim_phases()

        self.print_log("estimated %d phases in %0.2fs using mode '%s'" %
                       (self.nangles * self.nphases, time.perf_counter() - tstart, self.phase_estimation_mode))

        # #############################################
        # do band separation
        # #############################################
        tstart = time.perf_counter()

        self.bands_unmixed_ft = do_sim_band_separation(self.imgs_ft, self.phases, self.amps)

        self.print_log("separated bands in %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # estimate noise in component images
        # #############################################
        tstart = time.perf_counter()

        self.noise_power_bands = np.zeros((self.nangles, self.nphases))
        for ii in range(self.nangles):
            for jj in range(self.nphases):
                self.noise_power_bands[ii, jj] = get_noise_power(self.bands_unmixed_ft[ii, jj], self.fx, self.fy, self.fmax)

        self.print_log("estimated noise in %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # shift bands
        # #############################################
        tstart = time.perf_counter()

        self.bands_shifted_ft = self.shift_bands()

        self.print_log("shifted bands in %0.2fs" % (time.perf_counter() - tstart))

        tstart = time.perf_counter()
        self.otf_shifted = self.shift_otf()
        self.print_log("shifted otfs in %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # correct global phase and get modulation depths
        # #############################################
        tstart = time.perf_counter()

        # correct for wrong global phases
        self.otf_mask_threshold = 0.1
        self.phase_corrections, mags = get_band_overlap(self.bands_shifted_ft[:, 0], self.bands_shifted_ft[:, 1],
                                                        self.otf_shifted[:, 0], self.otf_shifted[:, 1],
                                                        otf_threshold=self.otf_mask_threshold)

        phase_corr_mat = np.exp(1j * np.concatenate((np.zeros((len(self.phase_corrections), 1)),
                                                     np.expand_dims(self.phase_corrections, axis=1),
                                                     np.expand_dims(-self.phase_corrections, axis=1)), axis=1))

        if self.combine_mode == "Lal":
            # estimate power spectrum parameters
            mod_depths_pspectrum, self.power_spectrum_params, self.power_spectrum_masks = self.fit_power_spectra()

        if self.use_fixed_mod_depths:
            self.print_log("using fixed modulation depth")
            self.mod_depths = self.mod_depths_guess
        else:
            if self.combine_mode == "Lal":
                self.mod_depths = mod_depths_pspectrum
            elif self.combine_mode == "fairSIM":
                self.mod_depths = np.kron(mags[:, None], np.ones((1, 3)))
            else:
                raise ValueError("combine_mode must be 'fairSIM' or 'Lal' but was '%s'" % self.combine_mode)

        self.print_log("estimated global phases and mod depths in %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # Get weights and combine bands
        # #############################################
        tstart = time.perf_counter()

        if self.combine_mode == "Lal":
            self.weights, self.weights_norm = self.get_weights_lal()
        elif self.combine_mode == "fairSIM":
            # following the approach of FairSIM: https://doi.org/10.1038/ncomms10980
            self.weights = self.otf_shifted.conj()
            self.weights_norm = self.wiener_parameter**2 + np.nansum(np.abs(self.weights) ** 2, axis=(0, 1))
        else:
            raise ValueError("combine_mode must be 'fairSIM' or 'Lal' but was '%s'" % self.combine_mode)

        self.print_log("computed band weights in %0.2fs" % (time.perf_counter() - tstart))

        # combine bands
        tstart = time.perf_counter()

        self.sim_sr_ft = np.nansum(self.bands_shifted_ft * self.weights *
                                   np.expand_dims(phase_corr_mat, axis=(2, 3)) /
                                   np.expand_dims(self.mod_depths, axis=(2, 3)), axis=(0, 1)) / self.weights_norm

        # inverse FFT to get real-space reconstructed image
        apodization = scipy.signal.windows.tukey(self.sim_sr_ft.shape[1], alpha=0.1)[None, :] * \
                      scipy.signal.windows.tukey(self.sim_sr_ft.shape[0], alpha=0.1)[:, None]

        # irfft2 ~2X faster than ifft2. But have to slice out only half the frequencies
        self.sim_sr = fft.fftshift(fft.irfft2(fft.ifftshift(self.sim_sr_ft * apodization)[:, : self.sim_sr_ft.shape[1] // 2 + 1]))
        # self.sim_sr = fft.fftshift(fft.ifft2(fft.ifftshift(self.sim_sr_ft * apodization))).real

        self.print_log("combining bands using mode '%s' and Wiener parameter %0.3f took %0.2fs" %
                       (self.combine_mode, self.wiener_parameter, time.perf_counter() - tstart))

        # #############################################
        # widefield deconvolution
        # #############################################
        tstart = time.perf_counter()

        if self.combine_mode == "Lal":
            # get signal to noise ratio
            wf_noise = get_noise_power(self.widefield_ft, self.fx, self.fy, self.fmax)
            fit_result, self.mask_wf = fit_power_spectrum(self.widefield_ft, self.otf, self.fx, self.fy, self.fmax, self.fbounds,
                                                          init_params=[None, self.power_spectrum_params[0, 0, 1], 1, wf_noise],
                                                          fixed_params=[False, True, True, True])

            self.pspec_params_wf = fit_result['fit_params']

            with np.errstate(invalid="ignore", divide="ignore"):
                ff = np.sqrt(self.fx[None, :]**2 + self.fy[:, None]**2)
                sig = power_spectrum_fn([self.pspec_params_wf[0], self.pspec_params_wf[1], self.pspec_params_wf[2], 0], ff, 1)

            wf_decon_ft = self.widefield_ft * get_wiener_filter(self.otf, sig / wf_noise)
            # upsample to make fully comparable to reconstructed image
            self.widefield_deconvolution_ft = tools.resample_bandlimited_ft(wf_decon_ft, (2, 2))
        elif self.combine_mode == "fairSIM":
            # todo: what is most fair way to deconvolve here:?

            # wf_decon_ft = self.widefield_ft * get_wiener_filter(self.otf, 1/np.sqrt(self.wiener_parameter / self.nangles))

            self.widefield_deconvolution_ft = np.nansum(self.weights[:, 0] * self.bands_shifted_ft[:, 0], axis=0) / \
                          (self.wiener_parameter**2 + np.nansum(np.abs(self.weights[:, 0])**2, axis=0))
        else:
            raise ValueError()

        self.widefield_deconvolution = fft.fftshift(fft.irfft2(
            fft.ifftshift(self.widefield_deconvolution_ft * apodization)[:, :self.widefield_deconvolution_ft.shape[1] // 2 + 1])).real
        # self.widefield_deconvolution = fft.fftshift(fft.ifft2(fft.ifftshift(self.widefield_deconvolution_ft))).real

        self.print_log("Deconvolved widefield in %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # print parameters
        # #############################################
        self.print_parameters()

    def shift_bands(self):
        """
        Shifted and upsample separated bands. Shifting done using FFT shift approach
        """

        # conceptually, just want loop to call resample_bandlimited_ft() and then translate_ft()
        # BUT: can speed up somewhat if can load all Fourier transforms on GPU at once, and limit to only the
        # number of FFT's we need. Writing function so we can do this is a bit more complicated

        f_upsample = 2
        x = tools.get_fft_pos(f_upsample * self.nx, self.dx / f_upsample, centered=False, mode='symmetric')
        y = tools.get_fft_pos(f_upsample * self.ny, self.dy / f_upsample, centered=False, mode='symmetric')

        # store expanded bands
        expanded = np.zeros((self.nangles, 2, self.ny * f_upsample, self.nx * f_upsample), dtype=np.complex)
        # exponential shift factor for the one band we will shift with FFT
        exp_factor = np.zeros((self.nangles, self.ny * f_upsample, self.nx * f_upsample), dtype=np.complex)
        # shift and filter components
        for ii in range(self.nangles):
            # loop over components [O(f)H(f), m*O(f - f_o)H(f), m*O(f + f_o)H(f)]
            # don't need to loop over m*O(f + f_o)H(f), since it is conjugate of m*O(f - f_o)H(f)

            # shift factor for m*O(f - f_o)H(f)
            exp_factor[ii] = np.exp(-1j * 2 * np.pi * (self.frqs[ii, 0] * x[None, :] + self.frqs[ii, 1] * y[:, None]))
            for jj, band_ind in enumerate([0, 1]):
                # think expand->shift->deconvolve is better than deconvolve->expand->shift, because it avoids the
                # fx=0 and fy=0 line artifacts
                expanded[ii, jj] = tools.resample_bandlimited_ft(self.bands_unmixed_ft[ii, jj], (f_upsample, f_upsample))
                # bands_shifted_ft[ii, jj] = tools.translate_ft(expanded[ii, jj], band_ind * self.frqs[ii], self.dx / f_upsample)

        # get shifted bands
        tstart = time.perf_counter()
        bands_shifted_ft = np.zeros((self.nangles, 3, self.ny * f_upsample, self.nx * f_upsample), dtype=np.complex)
        # get O(f)H(f) directly from expansion
        bands_shifted_ft[:, 0] = expanded[:, 0]

        # FFT shift to get m*O(f - f_o)H(f)
        if self.use_gpu:
            bands_shifted_ft[:, 1] = cp.asnumpy(cp.fft.fftshift(cp.fft.fft2(
                                                 cp.array(exp_factor) * cp.fft.ifft2(cp.fft.ifftshift(expanded[:, 1],
                                                 axes=(1, 2)), axes=(1, 2)), axes=(1, 2)), axes=(1, 2)))
        else:
            bands_shifted_ft[:, 1] = fft.fftshift(fft.fft2(exp_factor * fft.ifft2(fft.ifftshift(expanded[:, 1],
                                                   axes=(1, 2)), axes=(1, 2)), axes=(1, 2)), axes=(1, 2))

        # reflect m*O(f - f_o)H(f) to get m*O(f + f_o)H(f)
        bands_shifted_ft[:, 2] = tools.conj_transpose_fft(bands_shifted_ft[:, 1])

        return bands_shifted_ft

    def shift_otf(self):
        """
        Shift OTF's along with pixels. Use nearest whole pixel shift, which is much faster than FFT translation
        """
        f_upsample = 2
        otf_us = tools.resample_bandlimited_ft(self.otf, (f_upsample, f_upsample)) / f_upsample / f_upsample

        # upsampled frequency data
        fx_us = tools.get_fft_frqs(f_upsample * self.nx, self.dx / f_upsample)
        dfx_us = fx_us[1] - fx_us[0]
        fy_us = tools.get_fft_frqs(f_upsample * self.ny, self.dx / f_upsample)
        dfy_us = fy_us[1] - fy_us[0]

        otf_shifted = np.zeros((self.nangles, 3, self.ny * f_upsample, self.nx * f_upsample), dtype=np.complex)
        for ii in range(self.nangles):
            for jj, band_ind in enumerate([0, 1, -1]):
                # otf shifted
                if jj == 0:
                    otf_shifted[ii, jj] = otf_us
                else:
                    otf_shifted[ii, jj], _ = tools.translate_pix(otf_us, self.frqs[ii] * band_ind, dr=(dfx_us, dfy_us), axes=(1, 0), wrap=False)

        return otf_shifted

    def get_weights_lal(self):
        """
        Combine bands O(f)otf(f), O(f-fo)otf(f), and O(f+fo)otf(f) to do SIM reconstruction.

        Following the approach of https://doi.org/10.1109/jstqe.2016.2521542

        :return sim_sr_ft, bands_shifted_ft, weights, weight_norm, snr, snr_shifted:
        """

        # upsampled frequency data
        f_upsample = 2
        fx_us = tools.get_fft_frqs(f_upsample * self.nx, self.dx / f_upsample)
        fy_us = tools.get_fft_frqs(f_upsample * self.ny, self.dx / f_upsample)

        def ff_shift_upsample(f): return np.sqrt((fx_us[None, :] - f[0]) ** 2 + (fy_us[:, None] - f[1]) ** 2)

        # useful parameters
        snr_shifted = np.zeros(self.bands_shifted_ft.shape)
        wiener_filters = np.zeros(self.bands_shifted_ft.shape, dtype=np.complex)

        # shift and filter components
        for ii in range(self.nangles):
            # loop over components, O(f)H(f), m*O(f - f_o)H(f), m*O(f + f_o)H(f)
            for jj, band_ind in enumerate([0, 1, -1]):
                params = list(self.power_spectrum_params[ii, jj, :-1]) + [0]

                # get shifted OTF, SNR, and weights
                with np.errstate(invalid="ignore", divide="ignore"):
                    # get snr from model
                    snr_shifted[ii, jj] = power_spectrum_fn(params, ff_shift_upsample((0, 0)), 1) / self.noise_power_bands[ii, jj]
                    # correct for divergence at (fx, fy) = (0, 0)
                    snr_shifted[ii, jj][np.isinf(snr_shifted[ii, jj])] = np.max(snr_shifted[ii, jj][snr_shifted[ii, jj] < np.inf])
                    # calculate Wiener filter
                    wiener_filters[ii, jj] = get_wiener_filter(self.otf_shifted[ii, jj], snr_shifted[ii, jj])

        # get weights for averaging
        weights_norm = np.sum(snr_shifted * np.abs(self.otf_shifted) ** 2, axis=(0, 1)) + self.wiener_parameter**2
        # weighted averaging with |otf(f)|**2 * signal_power(f) / noise_power(f)
        weights = snr_shifted * np.abs(self.otf_shifted) ** 2 * wiener_filters

        return weights, weights_norm

    def estimate_sim_frqs(self, fts1, fts2):
        """
        estimate SIM frequency

        :param frq_guess:
        :return frqs:
        :return phases:
        :return peak_heights_relative: Height of the fit peak relative to the DC peak.
        """
        if self.frqs_guess is not None:
            frq_guess = self.frqs_guess
        else:
            frq_guess = [None] * self.nangles

        nangles = fts1.shape[0]

        # todo: maybe should take some average/combination of the widefield images to try and improve signal
        # e.g. could multiply each by expected phase values?
        results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
            joblib.delayed(fit_modulation_frq)(
                fts1[ii, 0], fts2[ii, 0], self.dx, self.fmax, frq_guess=frq_guess[ii])
            for ii in range(nangles)
        )

        frqs, _ = zip(*results)
        frqs = np.reshape(np.asarray(frqs), [nangles, 2])

        return frqs

    def estimate_sim_phases(self):
        """
        estimate phases for all SIM images
        """

        phases = np.zeros((self.nangles, self.nphases))
        amps = np.ones((self.nangles, self.nphases))
        phase_guess = self.phases_guess
        # mods = np.zeros((self.nangles, self.nphases))

        if self.phase_estimation_mode == "wicker-iterative":
            if phase_guess is None:
                phase_guess = [None] * self.nangles

            results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                joblib.delayed(get_phase_wicker_iterative)(
                    self.imgs_ft[ii], self.otf, self.frqs[ii], self.dx, self.fmax,
                    phases_guess=phase_guess[ii],
                    fit_amps=self.determine_amplitudes) for ii in range(self.nangles))

            # for ii in range(self.nangles):
            #     phases[ii], amps[ii] = get_phase_wicker_iterative(self.imgs_ft[ii], self.otf, self.frqs[ii], self.dx, self.fmax,
            #                                                       phases_guess=phase_guess[ii],
            #                                                       fit_amps=self.determine_amplitudes)
            phases, amps = zip(*results)
            phases = np.asarray(phases)
            amps = np.asarray(amps)

        elif self.phase_estimation_mode == "real-space":
            if phase_guess is None:
                phase_guess = np.zeros((self.nangles, self.nphases))

            # joblib a little messy because have to map one index to two
            results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                joblib.delayed(get_phase_realspace)(
                    self.imgs[np.unravel_index(aa, [self.nangles, self.nphases])],
                    self.frqs[np.unravel_index(aa, [self.nangles, self.nphases])[0]], self.dx,
                    phase_guess=phase_guess[np.unravel_index(aa, [self.nangles, self.nphases])], origin="center"
                ) for aa in range(self.nangles * self.nphases))

            phases = np.reshape(np.array(results), [self.nangles, self.nphases])
            amps = np.ones((self.nangles, self.nphases))

            # for ii in range(self.nangles):
            #     for jj in range(self.nphases):
            #         phases[ii, jj] = get_phase_realspace(self.imgs[ii, jj], self.frqs[ii], self.dx,
            #                                              phase_guess=phase_guess[ii, jj], origin="center")
            # amps = np.ones((self.nangles, self.nphases))

        elif self.phase_estimation_mode == "fixed":
            phases = self.phases_guess
        else:
            raise ValueError("phase_estimation_mode must be 'wicker-iterative', 'real-space', or 'fixed'"
                             " but was '%s'" % self.phase_estimation_mode)

        # check if phase fit was too bad, and default to guess values
        if self.phases_guess is not None:
            phase_guess_diffs = np.mod(phase_guess - phase_guess[:, 0][:, None], 2*np.pi)
            phase_diffs = np.mod(phases - phases[:, 0][:, None], 2*np.pi)

            for ii in range(self.nangles):
                diffs = np.mod(phase_guess_diffs[ii] - phase_diffs[ii], 2 * np.pi)
                condition = np.abs(diffs - 2 * np.pi) < diffs
                diffs[condition] = diffs[condition] - 2 * np.pi

                if np.any(np.abs(diffs) > self.max_phase_error):
                    phases[ii] = phase_guess[ii]
                    str = "Angle %d phase guesses are more than the maximum allowed phase error=%0.2fdeg." \
                          " Defaulting to guess values" % (ii, self.max_phase_error * 180/np.pi)

                    str += "\nfit phase diffs="
                    for jj in range(self.nphases):
                        str += "%0.2fdeg, " % (phase_diffs[ii, jj] * 180/np.pi)

                    self.print_log(str)

        return phases, amps

    def fit_power_spectra(self):
        # first average power spectrum of \sum_{angles} O(f)H(f) to get exponent
        component_zero = np.nanmean(self.bands_unmixed_ft[:, 0], axis=0)
        noise = get_noise_power(component_zero, self.fx, self.fy, self.fmax)

        fit_results_avg, mask_avg = fit_power_spectrum(component_zero, self.otf, self.fx, self.fy,
                                                       self.fmax, self.fbounds,
                                                       init_params=[None, None, 1, noise],
                                                       fixed_params=[False, False, True, True])
        fit_params_avg = fit_results_avg['fit_params']
        exponent = fit_params_avg[1]

        # now fit each other component using this same exponent
        power_spectrum_params = np.zeros((self.nangles, self.nangles, 4))
        masks = np.zeros(self.bands_unmixed_ft.shape, dtype=np.bool)
        for ii in range(self.nangles):
            for jj in range(self.nphases):

                if jj == 0:
                    # for unshifted components, fit the amplitude
                    init_params = [None, exponent, 1, self.noise_power_bands[ii, jj]]
                    fixed_params = [False, True, True, True]

                    fit_results, masks[ii, jj] = fit_power_spectrum(self.bands_unmixed_ft[ii, jj], self.otf,
                                                                    self.fx, self.fy, self.fmax, self.fbounds,
                                                                    init_params=init_params, fixed_params=fixed_params)
                    power_spectrum_params[ii, jj] = fit_results['fit_params']
                elif jj == 1:
                    # for shifted components, fit the modulation factor
                    init_params = [power_spectrum_params[ii, 0, 0], exponent, 0.5, self.noise_power_bands[ii, jj]]
                    fixed_params = [True, True, False, True]

                    fit_results, masks[ii, jj] = fit_power_spectrum(self.bands_unmixed_ft[ii, jj], self.otf,
                                                                    self.fx, self.fy, self.fmax, (0, 1), self.fbounds,
                                                                    self.frqs[ii],
                                                                    init_params=init_params, fixed_params=fixed_params)
                    power_spectrum_params[ii, jj] = fit_results['fit_params']
                elif jj == 2:
                    power_spectrum_params[ii, jj] = power_spectrum_params[ii, 1]
                else:
                    raise NotImplementedError("not implemented for nphases > 3")

        # extract mod depths from the structure
        mod_depths = np.zeros((self.nangles, self.nphases))
        mod_depths[:, 0] = 1
        for jj in range(1, self.nphases):
            mod_depths[:, jj] = power_spectrum_params[:, 1, 2]

        return mod_depths, power_spectrum_params, masks

    # printing utility functions
    def print_parameters(self):

        self.print_log("SIM reconstruction for %d angles and %d phases" % (self.nangles, self.nphases))
        self.print_log("images are size %dx%d with pixel size %0.3fum" % (self.ny, self.nx, self.dx))
        self.print_log("emission wavelength=%.0fnm and NA=%0.2f" % (self.wavelength * 1e3, self.na))

        for ii in range(self.nangles):
            self.print_log("################ Angle %d ################" % ii)

            # frequency and period data
            if self.frqs_guess is not None:
                angle_guess = np.angle(self.frqs_guess[ii, 0] + 1j * self.frqs_guess[ii, 1])
                period_guess = 1 / np.linalg.norm(self.frqs_guess[ii])

                self.print_log("Frequency guess=({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                    self.frqs_guess[ii, 0], self.frqs_guess[ii, 1], period_guess * 1e3, angle_guess * 180 / np.pi, 2 * np.pi))

            self.print_log("Frequency fit  =({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                self.frqs[ii, 0], self.frqs[ii, 1], self.periods[ii] * 1e3, self.angles[ii] * 180 / np.pi))

            # modulation depth
            self.print_log("modulation depth=%0.3f" % self.mod_depths[ii, 1])
            self.print_log("minimum peak-to-camera-noise ratio=%0.3f" % np.min(self.p2nr[ii]))

            # phase information
            self.print_log("phases  =", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%07.3fdeg, " % (self.phases[ii, jj] * 180 / np.pi), end="")
            self.print_log("%07.3fdeg" % (self.phases[ii, self.nphases - 1] * 180 / np.pi))

            if self.phases_guess is not None:
                self.print_log("guesses =", end="")
                for jj in range(self.nphases - 1):
                    self.print_log("%07.3fdeg, " % (self.phases_guess[ii, jj] * 180 / np.pi), end="")
                self.print_log("%07.3fdeg" % (self.phases_guess[ii, self.nphases - 1] * 180 / np.pi))

            self.print_log("dphases =", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%07.3fdeg, " % (np.mod(self.phases[ii, jj] - self.phases[ii, 0], 2 * np.pi) * 180 / np.pi),
                               end="")
            self.print_log("%07.3fdeg" %
                           (np.mod(self.phases[ii, self.nphases - 1] - self.phases[ii, 0], 2 * np.pi) * 180 / np.pi))

            if self.phases_guess is not None:
                self.print_log("dguesses=", end="")
                for jj in range(self.nphases - 1):
                    self.print_log("%07.3fdeg, " % (np.mod(self.phases_guess[ii, jj] - self.phases_guess[ii, 0], 2*np.pi) * 180/np.pi), end="")
                self.print_log("%07.3fdeg" % (np.mod(self.phases_guess[ii, self.nphases - 1] - self.phases_guess[ii, 0], 2*np.pi) * 180/np.pi))

            # phase corrections
            self.print_log("global phase correction=%0.2fdeg" % (self.phase_corrections[ii] * 180 / np.pi))

            self.print_log("amps =", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%05.3f, " % (self.amps[ii, jj]), end="")
            self.print_log("%05.3f" % (self.amps[ii, self.nphases - 1]))

    def print_log(self, string, **kwargs):
        """
        Print result to stdout and to a log file.

        :param string:
        :param fid: file handle. If fid=None, then will be ignored
        :param end:
        :return:
        """

        print(string, **kwargs)

        if self.log_file is not None:
            print(string, **kwargs, file=self.log_file)

    # plotting utility functions
    def plot_figs(self):
        """
        Automate plotting and saving of figures
        :return:
        """
        tstart = time.perf_counter()

        saving = self.save_dir is not None

        # todo: populate these
        figs = []
        fig_names = []

        # plot images
        figh = self.plot_sim_imgs(self.figsize)

        if saving:
            figh.savefig(os.path.join(self.save_dir, "raw_images.png"))
        if not self.interactive_plotting:
            plt.close(figh)

        # plot frequency fits
        fighs, fig_names = self.plot_frequency_fits(figsize=self.figsize)
        for fh, fn in zip(fighs, fig_names):
            if saving:
                fh.savefig(os.path.join(self.save_dir, "%s.png" % fn))
            if not self.interactive_plotting:
                plt.close(fh)

        if self.combine_mode == "Lal":
            # plot power spectrum fits
            fighs, fig_names = self.plot_power_spectrum_fits(figsize=self.figsize)
            for fh, fn in zip(fighs, fig_names):
                if saving:
                    fh.savefig(os.path.join(self.save_dir, "%s.png" % fn))
                if not self.interactive_plotting:
                    plt.close(fh)

            # widefield power spectrum fit
            figh = plot_power_spectrum_fit(self.widefield_ft, self.otf,
                                           {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                           self.pspec_params_wf, mask=self.mask_wf, figsize=self.figsize,
                                           ttl_str="Widefield power spectrum")
            if saving:
                figh.savefig(os.path.join(self.save_dir, "power_spectrum_widefield.png"))
            if not self.interactive_plotting:
                plt.close(figh)

        # plot filters used in reconstruction
        fighs, fig_names = self.plot_reconstruction_diagnostics(figsize=self.figsize)
        for fh, fn in zip(fighs, fig_names):
            if saving:
                fh.savefig(os.path.join(self.save_dir, "%s.png" % fn))
            if not self.interactive_plotting:
                plt.close(fh)

        # plot reconstruction results
        fig = self.plot_reconstruction(figsize=self.figsize)
        if saving:
            fig.savefig(os.path.join(self.save_dir, "sim_reconstruction.png"), dpi=400)
        if not self.interactive_plotting:
            plt.close(fig)

        # plot otf
        fig = self.plot_otf(figsize=self.figsize)
        if saving:
            fig.savefig(os.path.join(self.save_dir, "otf.png"))
        if not self.interactive_plotting:
            plt.close(fig)

        tend = time.perf_counter()
        self.print_log("plotting results took %0.2fs" % (tend - tstart))

        return figs, fig_names

    def plot_sim_imgs(self, figsize=(20, 10)):
        """
        Display SIM images for visual inspection

        Use this to examine SIM pictures and their fourier transforms before doing reconstruction.

        :return:
        """

        # real space coordinate data
        x = tools.get_fft_pos(self.nx, dt=self.dx)
        y = tools.get_fft_pos(self.ny, dt=self.dy)

        extent = tools.get_extent(y, x)

        # parameters for real space plot
        vmin = np.percentile(self.imgs.ravel(), 0.1)
        vmax = np.percentile(self.imgs.ravel(), 99.9)

        # ########################################
        # plot
        # ########################################
        figh = plt.figure(figsize=figsize)
        plt.suptitle('SIM images, real space and mcnr')
        grid = plt.GridSpec(self.nangles, self.nphases + 3)
        
        mean_int = np.mean(self.imgs, axis=(2, 3))
        rel_int_phases = mean_int / np.expand_dims(np.max(mean_int, axis=1), axis=1)
        
        mean_int_angles = np.mean(self.imgs, axis=(1, 2, 3))
        rel_int_angles = mean_int_angles / np.max(mean_int_angles)

        for ii in range(self.nangles):
            for jj in range(self.nphases):

                # ########################################
                # raw real-space SIM images
                # ########################################
                ax = plt.subplot(grid[ii, jj])
                ax.imshow(self.imgs[ii, jj], vmin=vmin, vmax=vmax, extent=extent, interpolation=None)


                if jj == 0:
                    tstr = 'angle %d, relative intensity=%0.3f\nphase int=' % (ii, rel_int_angles[ii])
                    for aa in range(self.nphases):
                        tstr += "%0.3f, " % rel_int_phases[ii, aa]
                    plt.ylabel(tstr)
                if ii == (self.nangles - 1):
                    plt.xlabel("Position (um)")

            # ########################################
            # histograms of real-space images
            # ########################################
            nbins = 50
            bin_edges = np.linspace(0, np.percentile(self.imgs, 99), nbins + 1)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            ax = plt.subplot(grid[ii, self.nphases])
            for jj in range(self.nphases):
                histogram, _ = np.histogram(self.imgs[ii, jj].ravel(), bins=bin_edges)
                ax.semilogy(bin_centers, histogram)

            ax.set_yticks([])
            plt.title("median %0.1f" % (np.median(self.imgs[ii, jj].ravel())))
            if ii != (self.nangles - 1):
                ax.set_xticks([])


            # ########################################
            # spatially resolved mcnr
            # ########################################
            ax = plt.subplot(grid[ii, self.nphases + 1])
            plt.imshow(self.mcnr[ii], vmin=0, vmax=np.percentile(self.mcnr[ii], 99))
            plt.colorbar()
            ax.set_xticks([])
            ax.set_yticks([])
            if ii == 0:
                plt.title("mcnr")

            # ########################################
            # mcnr histograms
            # ########################################
            nbins = 50
            bin_edges = np.linspace(0, np.percentile(self.mcnr, 99), nbins + 1)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            histogram, _ = np.histogram(self.mcnr[ii].ravel(), bins=bin_edges)

            median = np.median(self.mcnr[ii].ravel())

            #
            ax = plt.subplot(grid[ii, self.nphases + 2])
            plt.plot(bin_centers, histogram)

            ylim = ax.get_ylim()
            ax.set_ylim([0, ylim[1]])

            plt.title("mcnr median %0.3f" % median)
            ax.set_yticks([])
            if ii != (self.nangles - 1):
                ax.set_xticks([])

        return figh

    def plot_reconstruction(self, figsize=(20, 10)):
        """
        Plot SIM image and compare with 'widefield' image
        :return:
        """

        extent_wf = tools.get_extent(self.fy, self.fx)

        # for reconstructed image, must check if has been resampled
        fx = tools.get_fft_frqs(self.sim_sr.shape[1], 0.5 * self.dx)
        fy = tools.get_fft_frqs(self.sim_sr.shape[0], 0.5 * self.dx)

        extent_rec = tools.get_extent(fy, fx)

        gamma = 0.1
        min_percentile = 0.1
        max_percentile = 99.9

        # create plot
        figh = plt.figure(figsize=figsize)
        grid = plt.GridSpec(2, 3)
        # todo: print more reconstruction information here
        plt.suptitle("SIM reconstruction, w=%0.2f, phase estimation with %s" % (self.wiener_parameter, self.phase_estimation_mode))

        # widefield, real space
        ax = plt.subplot(grid[0, 0])

        vmin = np.percentile(self.widefield.ravel(), min_percentile)
        vmax = np.percentile(self.widefield.ravel(), max_percentile)
        plt.imshow(self.widefield, vmin=vmin, vmax=vmax)
        plt.title('widefield')

        # deconvolved, real space
        ax = plt.subplot(grid[0, 1])

        vmin = np.percentile(self.widefield_deconvolution.ravel(), min_percentile)
        vmax = np.percentile(self.widefield_deconvolution.ravel(), max_percentile)
        plt.imshow(self.widefield_deconvolution, vmin=vmin, vmax=vmax)
        plt.title('widefield deconvolved')

        # SIM, realspace
        ax = plt.subplot(grid[0, 2])

        vmin = np.percentile(self.sim_sr.ravel(), min_percentile)
        vmax = np.percentile(self.sim_sr.ravel(), max_percentile)
        plt.imshow(self.sim_sr, vmin=vmin, vmax=vmax)
        plt.title('SIM reconstruction')

        # widefield Fourier space
        ax = plt.subplot(grid[1, 0])
        plt.imshow(np.abs(self.widefield_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_wf)

        circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ)

        circ2 = matplotlib.patches.Circle((0, 0), radius=2 * self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ2)

        plt.xlim([-2 * self.fmax, 2 * self.fmax])
        plt.ylim([2 * self.fmax, -2 * self.fmax])

        # deconvolution Fourier space
        ax = plt.subplot(grid[1, 1])
        plt.imshow(np.abs(self.widefield_deconvolution_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_rec)
        circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ)
        circ2 = matplotlib.patches.Circle((0, 0), radius=2 * self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ2)

        plt.xlim([-2 * self.fmax, 2 * self.fmax])
        plt.ylim([2 * self.fmax, -2 * self.fmax])

        # SIM fourier space
        ax = plt.subplot(grid[1 ,2])
        plt.imshow(np.abs(self.sim_sr_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_rec)
        circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ)

        circ2 = matplotlib.patches.Circle((0, 0), radius=2 * self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ2)

        # actual maximum frequency based on real SIM frequencies
        for ii in range(self.nangles):
            ax.add_artist(matplotlib.patches.Circle((0, 0), radius=self.fmax + 1/self.periods[ii], color='k', fill=0, ls='--'))

        plt.xlim([-2 * self.fmax, 2 * self.fmax])
        plt.ylim([2 * self.fmax, -2 * self.fmax])

        return figh

    def plot_comparison_line_cut(self, start_coord, end_coord, figsize=(20, 10),
                                 plot_os_sim=False, plot_deconvolution=True):
        """
        Plot line cuts using the same regions in the widefield, deconvolved, and SR-SIM images
        :param start_coord: [xstart, ystart]
        :param end_coord: [xend, yend]
        :param figsize:
        :return figh: handle to figure
        """
        start_coord_re = [2*c for c in start_coord]
        end_coord_re = [2*c for c in end_coord]

        # get cut from widefield image
        coord_wf, cut_wf = tools.get_cut_profile(self.widefield, start_coord, end_coord, 1)
        coord_wf = self.dx * coord_wf

        coord_os, cut_os = tools.get_cut_profile(self.sim_os, start_coord, end_coord, 1)
        coord_os = self.dx * coord_os

        coord_dc, cut_dc = tools.get_cut_profile(self.widefield_deconvolution, start_coord_re, end_coord_re, 1)
        coord_dc = 0.5 * self.dx * coord_dc

        coord_sr, cut_sr = tools.get_cut_profile(self.sim_sr, start_coord_re, end_coord_re, 1)
        coord_sr = 0.5 * self.dx * coord_sr

        coords = {'wf': coord_wf, 'os': coord_os, 'dc': coord_dc, 'sr': coord_sr}
        cuts = {'wf': cut_wf, 'os': cut_os, 'dc': cut_dc, 'sr': cut_sr}

        figh = plt.figure(figsize=figsize)

        phs = []
        pnames = []

        ax = plt.subplot(1, 2, 1)
        ph, = plt.plot(coord_sr, cut_sr)
        phs.append(ph)
        pnames.append('SR-SIM')

        if plot_os_sim:
            ph, = plt.plot(coord_os, cut_os)
            phs.append(ph)
            pnames.append('OS-SIM')

        if plot_deconvolution:
            ph, = plt.plot(coord_dc, cut_dc)
            phs.append(ph)
            pnames.append('deconvolved')

        ph, = plt.plot(coord_wf, cut_wf)
        phs.append(ph)
        pnames.append('widefield')

        plt.xlabel("Position (um)")
        plt.ylabel("ADC")
        plt.legend(phs, pnames)

        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])

        plt.subplot(1, 2, 2)
        vmin = np.percentile(self.widefield.ravel(), 2)
        vmax = np.percentile(self.widefield.ravel(), 99.5)

        plt.imshow(self.widefield, vmin=vmin, vmax=vmax)
        plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], 'white')
        plt.title('widefield')

        return figh, coords, cuts

    def plot_reconstruction_diagnostics(self, figsize=(20, 10)):
        """
        Plot deconvolved components and related information to check reconstruction

        :return figh: figure handle
        """
        figs = []
        fig_names = []

        # upsampled frequency
        fx_us = tools.get_fft_frqs(2 * self.nx, 0.5 * self.dx)
        fy_us = tools.get_fft_frqs(2 * self.ny, 0.5 * self.dx)

        # plot different stages of inversion
        extent = tools.get_extent(self.fy, self.fx)
        extent_upsampled = tools.get_extent(fy_us, fx_us)

        # plot one image for each angle
        for ii in range(self.nangles):
            fig = plt.figure(figsize=figsize)
            plt.suptitle('Magnitude of Fourier transforms, angle %d\nperiod=%0.3fnm at %0.3fdeg=%0.3frad, f=(%0.3f,%0.3f) 1/um\n'
                         'mod=%0.3f, min p2nr=%0.3f, wiener param=%0.2f\n'
                         'phases (deg) =%0.2f, %0.2f, %0.2f, phase diffs (deg) =%0.2f, %0.2f, %0.2f' %
                         (ii, self.periods[ii] * 1e3, self.angles[ii] * 180 / np.pi, self.angles[ii],
                          self.frqs[ii, 0], self.frqs[ii, 1], self.mod_depths[ii, 1], np.min(self.p2nr[ii]),
                          self.wiener_parameter,
                          self.phases[ii, 0] * 180 / np.pi, self.phases[ii, 1] * 180 / np.pi, self.phases[ii, 2] * 180 / np.pi,
                          0, np.mod(self.phases[ii, 1] - self.phases[ii, 0], 2*np.pi) * 180 / np.pi,
                          np.mod(self.phases[ii, 2] - self.phases[ii, 0], 2*np.pi) * 180 / np.pi))
            grid = plt.GridSpec(self.nphases, 4)

            for jj in range(self.nphases):

                # ####################
                # raw images at different phases
                # ####################
                ax = plt.subplot(grid[jj, 0])

                to_plot = np.abs(self.imgs_ft[ii, jj])
                to_plot[to_plot <= 0] = np.nan

                ax.imshow(to_plot, norm=LogNorm(), extent=extent)

                plt.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='r', facecolor='none')
                plt.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='r', facecolor='none')

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                ax.set_xlim([-2*self.fmax, 2*self.fmax])
                ax.set_ylim([2*self.fmax, -2*self.fmax])

                ax.set_xticks([])
                ax.set_yticks([])

                if jj == 0:
                    plt.title("Raw data, phases")

                # ####################
                # separated components
                # ####################
                ax = plt.subplot(grid[jj, 1])

                to_plot = np.abs(self.bands_unmixed_ft[ii, jj])
                to_plot[to_plot <= 0] = np.nan

                im = plt.imshow(to_plot, norm=LogNorm(), extent=extent)
                clim = im.get_clim()

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                if jj == 0:
                    plt.title('O(f)otf(f)')
                    plt.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='r', facecolor='none')
                    plt.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='r', facecolor='none')
                elif jj == 1:
                    plt.title('m*O(f-fo)otf(f)')
                    plt.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='r', facecolor='none')
                elif jj == 2:
                    plt.title('m*O(f+fo)otf(f)')
                    plt.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='r', facecolor='none')

                # plt.setp(ax.get_xticklabels(), visible=False)
                # plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_xticks([])
                ax.set_yticks([])

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

                # ####################
                # shifted component
                # ####################
                ax = plt.subplot(grid[jj, 2])

                # avoid any zeros for LogNorm()
                cs_ft_toplot = np.abs(self.bands_shifted_ft[ii, jj])
                cs_ft_toplot[cs_ft_toplot <= 0] = np.nan

                im = plt.imshow(cs_ft_toplot, norm=LogNorm(), extent=extent_upsampled)

                # to keep same color scale, must correct for upsampled normalization change
                im.set_clim(tuple([4 * c for c in clim]))

                plt.scatter(0, 0, edgecolor='r', facecolor='none')

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                if jj == 0:
                    plt.title('shifted component')
                    plt.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='r', facecolor='none')
                    plt.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='r', facecolor='none')
                if jj == 1:
                    plt.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='r', facecolor='none')
                    circ2 = matplotlib.patches.Circle(-self.frqs[ii], radius=self.fmax, color='k', fill=0, ls='--')
                    ax.add_artist(circ2)
                elif jj == 2:
                    plt.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='r', facecolor='none')
                    circ2 = matplotlib.patches.Circle(self.frqs[ii], radius=self.fmax, color='k', fill=0, ls='--')
                    ax.add_artist(circ2)

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

                ax.set_xticks([])
                ax.set_yticks([])

                # ####################
                # normalized weights
                # ####################
                ax = plt.subplot(grid[jj, 3])

                to_plot = np.abs(self.weights[ii, jj]) / self.weights_norm
                to_plot[to_plot <= 0] = np.nan
                im2 = plt.imshow(to_plot, norm=LogNorm(), extent=extent_upsampled)
                im2.set_clim([1e-5, 1])
                fig.colorbar(im2)

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                if jj == 1:
                    circ2 = matplotlib.patches.Circle(-self.frqs[ii], radius=self.fmax, color='k', fill=0, ls='--')
                    ax.add_artist(circ2)
                elif jj == 2:
                    circ2 = matplotlib.patches.Circle(self.frqs[ii], radius=self.fmax, color='k', fill=0, ls='--')
                    ax.add_artist(circ2)

                if jj == 0:
                    plt.title('normalized weight')
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

                ax.set_xticks([])
                ax.set_yticks([])

            figs.append(fig)
            fig_names.append('sim_combining_angle=%d' % ii)

        # #######################
        # net weight
        # #######################
        figh = plt.figure(figsize=figsize)
        grid = plt.GridSpec(1, 2)
        plt.suptitle('Net weight, Wiener param = %0.2f' % self.wiener_parameter)

        ax = plt.subplot(grid[0, 0])
        net_weight = np.abs(np.sum(self.weights, axis=(0, 1))) / self.weights_norm
        im = ax.imshow(net_weight, extent=extent_upsampled, norm=PowerNorm(gamma=0.1))

        figh.colorbar(im, ticks=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])

        ax.set_title("non-linear scale")
        circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ)

        circ2 = matplotlib.patches.Circle((0, 0), radius=2*self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ2)

        circ3 = matplotlib.patches.Circle(self.frqs[0], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ3)

        circ4 = matplotlib.patches.Circle(-self.frqs[0], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ4)

        circ5 = matplotlib.patches.Circle(self.frqs[1], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ5)

        circ6 = matplotlib.patches.Circle(-self.frqs[1], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ6)

        circ7 = matplotlib.patches.Circle(self.frqs[2], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ7)

        circ8 = matplotlib.patches.Circle(-self.frqs[2], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ8)

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])

        ax = plt.subplot(grid[0, 1])
        ax.set_title("linear scale")
        im = ax.imshow(net_weight, extent=extent_upsampled)

        figh.colorbar(im)
        circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ)

        circ2 = matplotlib.patches.Circle((0, 0), radius=2 * self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ2)

        circ3 = matplotlib.patches.Circle(self.frqs[0], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ3)

        circ4 = matplotlib.patches.Circle(-self.frqs[0], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ4)

        circ5 = matplotlib.patches.Circle(self.frqs[1], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ5)

        circ6 = matplotlib.patches.Circle(-self.frqs[1], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ6)

        circ7 = matplotlib.patches.Circle(self.frqs[2], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ7)

        circ8 = matplotlib.patches.Circle(-self.frqs[2], radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ8)

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])

        figs.append(figh)
        fig_names.append('net_weight')

        return figs, fig_names

    def plot_power_spectrum_fits(self, figsize=(20, 10)):
        """
        Plot results of power spectrum fitting
        :param figsize:
        :return:
        """

        debug_figs = []
        debug_fig_names = []
        # individual power spectra
        for ii in range(self.nangles):
            fig = plot_power_spectrum_fit(self.bands_unmixed_ft[ii, 0], self.otf,
                                          {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                          self.power_spectrum_params[ii, 0], frq_sim=(0, 0), mask=self.power_spectrum_masks[ii, 0],
                                          figsize=figsize, ttl_str="Unshifted component, angle %d" % ii)
            debug_figs.append(fig)
            debug_fig_names.append("power_spectrum_unshifted_component_angle=%d" % ii)

            fig = plot_power_spectrum_fit(self.bands_unmixed_ft[ii, 1], self.otf,
                                          {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                          self.power_spectrum_params[ii, 1], frq_sim=self.frqs[ii], mask=self.power_spectrum_masks[ii, 1],
                                          figsize=figsize, ttl_str="Shifted component, angle %d" % ii)

            debug_figs.append(fig)
            debug_fig_names.append("power_spectrum_shifted_component_angle=%d" % ii)

        return debug_figs, debug_fig_names

    def plot_frequency_fits(self, figsize=(20, 10)):
        """
        Plot frequency fits
        :param figsize:
        :return figs: list of figure handles
        :return fig_names: list of figure names
        """
        figs = []
        fig_names = []

        if self.frqs_guess is None:
            frqs_guess = [None] * self.nangles
        else:
            frqs_guess = self.frqs_guess

        for ii in range(self.nangles):

            if self.frq_estimation_mode == "fourier-transform":
                figh = plot_correlation_fit(self.imgs_ft[ii, 0], self.imgs_ft[ii, 0], self.frqs[ii, :],
                                            self.dx, self.fmax, frqs_guess=frqs_guess[ii], figsize=figsize,
                                            ttl_str="Correlation fit, angle %d" % ii)
                figs.append(figh)
                fig_names.append("frq_fit_angle=%d_phase=%d" % (ii, 0))
            else:
                figh = plot_correlation_fit(self.bands_unmixed_ft[ii, 0],
                                            self.bands_unmixed_ft[ii, 1], self.frqs[ii, :],
                                            self.dx, self.fmax, frqs_guess=frqs_guess[ii], figsize=figsize,
                                            ttl_str="Correlation fit, angle %d" % ii)
                figs.append(figh)
                fig_names.append("frq_fit_angle=%d" % ii)

        return figs, fig_names

    def plot_otf(self, figsize=(20, 10)):
        """
        Plot optical transfer function (OTF) versus frequency. Compare with ideal OTF at the same NA, and show
        location of SIM frequencies
        :param figsize:
        :return:
        """
        figh = plt.figure(figsize=figsize)
        tstr = "OTF\nvalue at frqs="
        for ii in range(self.nangles):
            tstr += " %0.3f," % self.otf_at_frqs[ii]
        plt.suptitle(tstr)

        ff = np.sqrt(self.fx[None, :] ** 2 + self.fy[:, None] ** 2)

        otf_ideal = psf.circ_aperture_otf(ff, 0, self.na, self.wavelength)

        ax = plt.subplot(1, 2, 1)
        plt.plot(ff.ravel(), self.otf.ravel())
        plt.plot(ff.ravel(), otf_ideal.ravel())
        ylim = ax.get_ylim()

        # plot SIM frequencies
        fs = np.linalg.norm(self.frqs, axis=1)
        for ii in range(self.nangles):
            plt.plot([fs[ii], fs[ii]], ylim, 'k')

        ax.set_ylim(ylim)

        plt.xlabel("Frequency (1/um)")
        plt.ylabel("OTF")
        plt.legend(["OTF", 'Ideal OTF', 'SIM frqs'])

        plt.subplot(1, 2, 2)
        plt.title("2D OTF")
        plt.imshow(self.otf, extent=tools.get_extent(self.fy, self.fx))

        return figh

    # saving utility functions
    def save_imgs(self, save_dir=None, start_time=None, file_identifier=""):
        tstart_save = time.perf_counter()

        if save_dir is None:
            save_dir = self.save_dir

        if save_dir is not None:

            if start_time is None:
                kwargs = {}
            else:
                kwargs = {datetime:start_time}

            fname = os.path.join(save_dir, "sim_os_%s.tif" % file_identifier)
            tools.save_tiff(self.sim_os, fname, dtype=np.float32, **kwargs)

            fname = os.path.join(save_dir, "widefield_%s.tif" % file_identifier)
            tools.save_tiff(self.widefield, fname, dtype=np.float32, **kwargs)

            fname = os.path.join(save_dir, "sim_sr_%s.tif" % file_identifier)
            tools.save_tiff(self.sim_sr, fname, dtype=np.float32, **kwargs)

            fname = os.path.join(save_dir, "deconvolved_%s.tif" % file_identifier)
            tools.save_tiff(self.widefield_deconvolution, fname, dtype=np.float32, **kwargs)

            self.print_log("saving tiff files took %0.2fs" % (time.perf_counter() - tstart_save))

    def save_result(self, fname):
        """
        Save non-image fields in pickle format.

        :param fname: file path to save results
        :return results_dict: the dictionary that has been saved
        """
        tstart = time.perf_counter()

        fields_to_not_save = ['imgs', 'imgs_ft', 'sim_os', 'sim_sr', 'sim_sr_ft',
                              'widefield', 'widefield_ft', 'widefield_deconvolution', 'widefield_deconvolution_ft',
                              'bands_unmixed_ft', 'bands_shifted_ft',
                              'weights', 'weights_norm',
                              'snr', 'otf_shifted', 'power_spectrum_masks',
                              'log_file', 'mask_wf']
        # get dictionary object with images removed
        results_dict = {}
        for k, v in vars(self).items():
            if k in fields_to_not_save or k[0] == '_':
                continue
            results_dict[k] = v

        # add some useful info
        results_dict['freq units'] = '1/um'
        results_dict['period units'] = 'um'

        if fname is not None:
            with open(fname, 'wb') as f:
                pickle.dump(results_dict, f)

        self.print_log("saving results took %0.2fs" % (time.perf_counter() - tstart))

        return results_dict

# compute optical sectioned SIM image
def sim_optical_section(imgs, axis=0):
    """
    Law of signs optical sectioning reconstruction for three sim images with relative phase differences of 2*pi/3
    between each.

    Point: Let I[a] = A * [1 + m * cos(phi + phi_a)]
    Then sqrt( (I[0] - I[1])**2 + (I[1] - I[2])**2 + (I[2] - I[0])**2 ) = m*A * 3/ np.sqrt(2)

    :param imgs: image
    :param axis: axis to perform the optical sectioning computation along

    :return img_os: optically sectioned image
    """

    # put the axis we want to compute along first
    imgs = np.swapaxes(imgs, 0, axis)

    img_os = np.sqrt(2) / 3 * np.sqrt((imgs[0] - imgs[1]) ** 2 + (imgs[0] - imgs[2]) ** 2 + (imgs[1] - imgs[2]) ** 2)

    # swap the axis we moved back, if needed
    if img_os.ndim > 1 and axis != 0:
        if axis >= 1:
            # positive axis position is one less, since we consumed the 0th axis
            img_os = np.moveaxis(img_os, axis - 1, 0)
        else:
            # negative axis position is unchanged
            img_os = np.moveaxis(img_os, axis, 0)

    return img_os


def correct_modulation_for_bead_size(bead_radii, frqs):
    """
    Function for use when calibration SIM modulation depth using fluorescent beads. Assuming the beads are much smaller
    than the lattice spacing, then using the optical sectioning law of cosines type formula on the total fluorescent
    amplitude from a single isolated beads provides an estimate of the modulation depth.

    When the bead is a significant fraction of the lattice period, this modifies the modulation period. This function
    computes the correction for finite bead size.

    :param bead_radius: radius of the bead
    :param frq: frequency of the lattice
    :return mods: measure modulation depth for pattern with full contrast
    """
    # consider cosine in x-direction and spherical fluorescent object. Can divide in circles in the YZ plane, with radius
    # sqrt(R^2 - x^2), so we need to do the integral
    # \int_{-R}^R pi * (R^2 - x^2) * 0.5 * (1 + cos(2*pi*f*x + phi))
    # def integrated(r, u, f, phi): return -np.pi / (2 * np.pi * f) ** 3 * \
    #             (np.cos(phi) * ((u ** 2 - 2) * np.sin(u) + 2 * u * np.cos(u)) -
    #              np.sin(phi) * ((2 - u ** 2) * np.cos(u) + 2 * u * np.sin(u))) + \
    #              np.pi * r ** 2 / (2 * np.pi * f) * (np.cos(phi) * np.sin(u) + np.sin(phi) * np.cos(u))
    #
    # def full_int(r, f, phi): return 1 + \
    #                                 (integrated(r, 2 * np.pi * f * r, f, phi) - \
    #                                 integrated(r, -2 * np.pi * f * r, f, phi)) / \
    #                                 (4 / 3 * np.pi * r ** 3)

    def full_int(r, f, phi):
        u = 2 * np.pi * r * f
        return 1 + 3 / u**3 * (np.sin(u) - u * np.cos(u)) * np.cos(phi)

    phis = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    vals = np.zeros(3)
    for ii in range(3):
        vals[ii] = full_int(bead_radii, frqs, phis[ii])

    mods = sim_optical_section(vals, axis=0)
    return mods


# estimate frequency of modulation patterns
def fit_modulation_frq(ft1, ft2, dx, fmax=None, exclude_res=0.7, frq_guess=None, roi_pix_size=5, use_jacobian=False,
                       max_frq_shift=None, force_start_from_guess=False, keep_guess_if_better=True):
    """
    todo: drop roi_pix_size argument in favor of max_frq_shift

    Find SIM frequency from image by maximizing
    C(f) =  \sum_k img_ft(k) x img_ft*(k+f) = F(img_ft) * F(img_ft*) = F(|img_ft|^2).

    Note that there is ambiguity in the definition of this frequency, as -f will also be a peak. If frq_guess is
    provided, the peak closest to the guess will be returned. Otherwise, this function returns
    the frequency which has positive y-component. If the y-component is zero, it returns positive x-component.

    Function works in two steps: initially consider C(f) = |ft(img_ft)|^2 at the same frequency points as the DFT. Then,
    use this as the starting guess and minimize this function for other frequencies.

    Note, faster to give img as input instead of its fourier transform, because getting shifted ft of image requires
    only one fft, whereas getting shifted fourier transform of the transform itself requires two (ifft, then fft again)

    :param ft1: 2D Fourier space image
    :param ft2: 2D Fourier space image to be cross correlated with ft1
    :param dx: pixel size
    :param fmax:
    :param exclude_res: frequencies less than this fraction of fmax are excluded from peak finding. Not
    :param frq_guess: frequency guess [fx, fy]. Can be None.
    :param roi_pix_size: half-size of the region of interest around frq_guess used to estimate the peak location. This
    parameter is only used if frq_guess is provided.
    :param use_jacobian: use the jacobian during the fitting procedure. Because jacobian calculation is expensive,
    this does not speed up fitting.

    :return fit_frqs:
    :return mask: mask detailing region used in fitting
    """

    # extract options
    dy = dx
    ny, nx = ft1.shape

    # coords
    x = tools.get_fft_pos(nx, dx, centered=False, mode='symmetric')
    y = tools.get_fft_pos(ny, dy, centered=False, mode='symmetric')
    xx, yy = np.meshgrid(x, y)

    # get frequency data
    fxs = tools.get_fft_frqs(ft1.shape[1], dx)
    dfx = fxs[1] - fxs[0]
    fys = tools.get_fft_frqs(ft1.shape[0], dy)
    dfy = fys[1] - fys[0]
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff = np.sqrt(fxfx ** 2 + fyfy ** 2)

    if fmax is None:
        fmax = ff.max()

    if max_frq_shift is None:
        max_frq_shift = dfx * roi_pix_size

    # mask
    mask = np.ones(ft1.shape, dtype=np.bool)
    if frq_guess is None:
        mask[ff > fmax] = 0
        mask[ff < exclude_res * fmax] = 0
    else:
        # account for frq_guess
        f_dist_guess = np.sqrt( (fxfx - frq_guess[0])**2 + (fyfy - frq_guess[1])**2)
        mask[f_dist_guess > max_frq_shift] = 0

        # pix_x_guess = np.argmin(np.abs(fxs - frq_guess[0]))
        # pix_y_guess = np.argmin(np.abs(fys - frq_guess[1]))
        # mask = np.zeros(ft1.shape, dtype=np.bool)
        # mask[pix_y_guess - roi_pix_size : pix_y_guess + roi_pix_size + 1,
        #      pix_x_guess - roi_pix_size : pix_x_guess + roi_pix_size + 1] = 1

    # cross correlation of Fourier transforms
    # WARNING: correlate2d uses a different convention for the frequencies of the output, which will not agree with the fft convention
    # take conjugates so this will give \sum ft1 * ft2.conj()
    # scipy.signal.correlate(g1, g2)(fo) seems to compute \sum_f g1^*(f) * g2(f - fo), but I want g1^*(f)*g2(f+fo)
    cc = np.abs(scipy.signal.correlate(ft2, ft1, mode='same'))

    # get initial frq_guess by looking at cc at discrete frequency set and finding max
    max_ind = (cc * mask).argmax()
    subscript = np.unravel_index(max_ind, cc.shape)

    if frq_guess is None:
        # if not initial frq_guess, return frequency with positive fy. If fy=0, return with positive fx.
        # i.e. want to sort so that larger y-subscript is returned first
        # since img is real, also peak at [-fx, -fy]. Find those subscripts
        # todo: could exactly calculate where these will be, but this is good enough for now.
        a = np.argmin(np.abs(fxfx[subscript] + fxfx[0, :]))
        b = np.argmin(np.abs(fyfy[subscript] + fyfy[:, 0]))
        reflected_subscript = (b, a)

        subscript_list = [subscript, reflected_subscript]

        m = np.max(ft1.shape)
        subscript_list.sort(key=lambda x: x[0] * m + x[1], reverse=True)
        subscript, reflected_subscript = subscript_list

        init_params = [fxfx[subscript], fyfy[subscript]]
    else:
        if force_start_from_guess:
            init_params = frq_guess
        else:
            init_params = [fxfx[subscript], fyfy[subscript]]


    # do fitting
    img2 = fft.fftshift(fft.ifft2(fft.ifftshift(ft2)))
    # compute ft2(f + fo)
    def fft_shifted(f): return fft.fftshift(fft.fft2(np.exp(-1j*2*np.pi * (f[0] * xx + f[1] * yy)) * fft.ifftshift(img2)))
    # cross correlation
    # todo: conjugating ft2 instead of ft1, as in typical definition of cross correlation. Doesn't matter bc taking norm
    def cc_fn(f): return np.sum(ft1 * fft_shifted(f).conj())
    fft_norm = np.sum(np.abs(ft1) * np.abs(ft2))**2
    def min_fn(f): return -np.abs(cc_fn(f))**2 / fft_norm

    # derivatives and jacobian
    # these work, but the cost of evaluating this is so large that it slows down the fitting. This is not surprising,
    # as evaluating the jacobian is equivalent to ~4 function evaluations. So even though having the jacobian reduces
    # the number of function evaluations by a factor of ~3, this increase wins.
    # todo: need to check these now that added second fn to correlator
    def dfx_fft_shifted(f): return fft.fftshift(fft.fft2(-1j * 2 * np.pi * xx * np.exp(-1j * 2 * np.pi * (f[0] * xx + f[1] * yy)) * img2))
    def dfy_fft_shifted(f): return fft.fftshift(fft.fft2(-1j * 2 * np.pi * yy * np.exp(-1j * 2 * np.pi * (f[0] * xx + f[1] * yy)) * img2))
    def dfx_cc(f): return np.sum(ft2 * dfx_fft_shifted(f).conj())
    def dfy_cc(f): return np.sum(ft2 * dfy_fft_shifted(f).conj())
    def jac(f): return np.array([-2 * (cc_fn(f) * dfx_cc(f).conj()).real / fft_norm,
                                 -2 * (cc_fn(f) * dfy_cc(f).conj()).real / fft_norm])

    # enforce frequency fit in same box as guess
    lbs = (init_params[0] - max_frq_shift, init_params[1] - max_frq_shift)
    ubs = (init_params[0] + max_frq_shift, init_params[1] + max_frq_shift)
    bounds = ((lbs[0], ubs[0]), (lbs[1], ubs[1]))

    if use_jacobian:
        result = scipy.optimize.minimize(min_fn, init_params, bounds=bounds, jac=jac)
    else:
        result = scipy.optimize.minimize(min_fn, init_params, bounds=bounds)

    fit_frqs = result.x

    # ensure we never get a worse point than our initial guess
    if keep_guess_if_better and min_fn(init_params) < min_fn(fit_frqs):
        fit_frqs = init_params

    return fit_frqs, mask


def plot_correlation_fit(img1_ft, img2_ft, frqs, dx, fmax=None, frqs_guess=None, roi_size=31,
                         peak_pixels=2, figsize=(20, 10), ttl_str=""):
    """
    Display SIM parameter fitting results visually, in a way that is easy to inspect.

    Use this to plot the results of SIM frequency determination after running get_sim_frq()

    :param img_ft:
    :param frqs: fit value of frequency
    :param options:
    :param figsize:
    :param frqs_guess: guess value of frequency

    :return figh: handle to figure produced
    """

    # get physical parameters
    dy = dx

    # get frequency data
    fxs = tools.get_fft_frqs(img1_ft.shape[1], dx)
    fys = tools.get_fft_frqs(img1_ft.shape[0], dy)
    if fmax is None:
        fmax = np.sqrt(np.max(fxs)**2 + np.max(fys)**2)

    # power spectrum / cross correlation
    # cc = np.abs(img1_ft * img2_ft.conj())
    # cc = np.abs(scipy.signal.fftconvolve(img1_ft, img2_ft.conj(), mode='same'))
    cc = np.abs(scipy.signal.correlate(img2_ft, img1_ft, mode='same'))

    fx_sim, fy_sim = frqs
    # useful info to print
    period = 1 / np.sqrt(fx_sim ** 2 + fy_sim ** 2)
    angle = np.angle(fx_sim + 1j * fy_sim)

    peak_cc = tools.get_peak_value(cc, fxs, fys, [fx_sim, fy_sim], peak_pixels)

    peak1_dc = tools.get_peak_value(img1_ft, fxs, fys, [0, 0], peak_pixels)
    peak2 = tools.get_peak_value(img2_ft, fxs, fys, [fx_sim, fy_sim], peak_pixels)

    extent = tools.get_extent(fys, fxs)

    # create figure
    figh = plt.figure(figsize=figsize)
    gspec = matplotlib.gridspec.GridSpec(ncols=14, nrows=2, hspace=0.3, figure=figh)

    str = ""
    if ttl_str != "":
        str += "%s\n" % ttl_str
    # suptitle
    str += '      fit: period %0.1fnm = 1/%0.3fum at %.2fdeg=%0.3frad; f=(%0.3f,%0.3f) 1/um, peak cc=%0.3g at %0.2fdeg' % \
          (period * 1e3, 1/period, angle * 180 / np.pi, angle, fx_sim, fy_sim, np.abs(peak_cc), np.angle(peak_cc) * 180/np.pi)
    if frqs_guess is not None:
        fx_g, fy_g = frqs_guess
        period_g = 1 / np.sqrt(fx_g ** 2 + fy_g ** 2)
        angle_g = np.angle(fx_g + 1j * fy_g)
        peak_cc_g = tools.get_peak_value(cc, fxs, fys, frqs_guess, peak_pixels)

        str += '\nguess: period %0.1fnm = 1/%0.3fum at %.2fdeg=%0.3frad; f=(%0.3f,%0.3f) 1/um, peak cc=%0.3g at %0.2fdeg' % \
               (period_g * 1e3, 1/period_g, angle_g * 180 / np.pi, angle_g, fx_g, fy_g, np.abs(peak_cc_g), np.angle(peak_cc_g) * 180/np.pi)
    plt.suptitle(str)

    # #######################################
    # plot region of interest
    # #######################################
    roi_cx = np.argmin(np.abs(fx_sim - fxs))
    roi_cy = np.argmin(np.abs(fy_sim - fys))
    roi = tools.get_centered_roi([roi_cy, roi_cx], [roi_size, roi_size])

    extent_roi = tools.get_extent(fys[roi[0]:roi[1]], fxs[roi[2]:roi[3]])

    ax = plt.subplot(gspec[0, 0:6])
    ax.set_title("cross correlation, ROI")
    im1 = ax.imshow(cc[roi[0]:roi[1], roi[2]:roi[3]], interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent_roi)
    ph1 = ax.scatter(frqs[0], frqs[1], color='r', marker='x')
    if frqs_guess is not None:
        if np.linalg.norm(frqs - frqs_guess) < np.linalg.norm(frqs + frqs_guess):
            ph2 = ax.scatter(frqs_guess[0], frqs_guess[1], color='g', marker='x')
        else:
            ph2 = ax.scatter(-frqs_guess[0], -frqs_guess[1], color='g', marker='x')

    if frqs_guess is not None:
        ax.legend([ph1, ph2], ['frq fit', 'frq guess'], loc="upper right")
    else:
        ax.legend([ph1], ['frq fit'], loc="upper right")

    circ_max_frq = matplotlib.patches.Circle((0, 0), radius=fmax, color='k', fill=0, ls='--')
    ax.add_artist(circ_max_frq)

    ax.set_xlabel('fx (1/um)')
    ax.set_ylabel('fy (1/um)')


    cbar_ax = figh.add_subplot(gspec[0, 6])
    figh.colorbar(im1, cax=cbar_ax)

    ### full image ###
    ax2 = plt.subplot(gspec[0, 7:13])
    im2 = ax2.imshow(cc, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent)
    ax2.set_xlim([-fmax, fmax])
    ax2.set_ylim([fmax, -fmax])

    # plot maximum frequency
    circ_max_frq = matplotlib.patches.Circle((0, 0), radius=fmax, color='k', fill=0)
    ax2.add_artist(circ_max_frq)

    roi_rec = matplotlib.patches.Rectangle((fxs[roi[2]], fys[roi[0]]), fxs[roi[3]] - fxs[roi[2]], fys[roi[1]] - fys[roi[0]],
                                           edgecolor='k', fill=0)
    ax2.add_artist(roi_rec)

    ax2.set_title("Cross correlation, C(fo) = \sum_f img_ft1(f) x img_ft2^*(f+fo)")
    ax2.set_xlabel('fx (1/um)')
    ax2.set_ylabel('fy (1/um)')

    cbar_ax = figh.add_subplot(gspec[0, 13])
    figh.colorbar(im2, cax=cbar_ax)

    # ft 1
    ax3 = plt.subplot(gspec[1, 0:6])
    ax3.set_title("fft 1 PS near DC, peak = %0.3g at %0.2fdeg" % (np.abs(peak1_dc), np.angle(peak1_dc) * 180/np.pi))

    cx_c = np.argmin(np.abs(fxs))
    cy_c = np.argmin(np.abs(fys))
    roi_center = tools.get_centered_roi([cy_c, cx_c], [roi_size, roi_size])

    im3 = ax3.imshow(np.abs(img1_ft[roi_center[0]:roi_center[1], roi_center[2]:roi_center[3]])**2,
                     interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent)
    ax3.scatter(0, 0, color='r', marker='x')

    cbar_ax = figh.add_subplot(gspec[1, 6])
    figh.colorbar(im3, cax=cbar_ax)

    # ft 2
    ax4 = plt.subplot(gspec[1, 7:13])
    ttl_str = "fft 2 PS near fo, peak = %0.3g at %0.2fdeg" % (np.abs(peak2), np.angle(peak2) * 180 / np.pi)
    if frqs_guess is not None:
        peak2_g = tools.get_peak_value(img2_ft, fxs, fys, frqs_guess, peak_pixels)
        ttl_str += "\nguess peak = = %0.3g at %0.2fdeg" % (np.abs(peak2_g), np.angle(peak2_g) * 180 / np.pi)
    ax4.set_title(ttl_str)

    im4 = ax4.imshow(np.abs(img2_ft[roi[0]:roi[1], roi[2]:roi[3]])**2,
                     interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent_roi)
    ph1 = ax4.scatter(frqs[0], frqs[1], color='r', marker='x')
    if frqs_guess is not None:
        if np.linalg.norm(frqs - frqs_guess) < np.linalg.norm(frqs + frqs_guess):
            ph2 = ax4.scatter(frqs_guess[0], frqs_guess[1], color='g', marker='x')
        else:
            ph2 = ax4.scatter(-frqs_guess[0], -frqs_guess[1], color='g', marker='x')

    cbar_ax = figh.add_subplot(gspec[1, 13])
    figh.colorbar(im4, cax=cbar_ax)

    return figh

# estimate phase of modulation patterns
def get_phase_ft(img_ft, sim_frq, dxy, peak_pixel_size=2):
    """
    Estimate pattern phase directly from phase in Fourier transform

    :param img_ft:
    :param sim_frq:
    :param dxy:
    :return phase:
    """
    ny, nx = img_ft.shape
    fx = tools.get_fft_frqs(nx, dxy)
    fy = tools.get_fft_frqs(ny, dxy)

    phase = np.mod(np.angle(tools.get_peak_value(img_ft, fx, fy, sim_frq, peak_pixel_size=peak_pixel_size)), 2*np.pi)

    return phase


def get_phase_realspace(img, sim_frq, dxy, phase_guess=0, origin="center"):
    """
    Determine phase of pattern with a given frequency. Matches +cos(2*pi* f.*r + phi), where the origin
    is taken to be in the center of the image, or more precisely using the same coordinates as the fft
    assumes.

    To obtain the correct phase, it is necessary to have a very good frq_guess of the frequency.
    However, obtaining accurate relative phases is much less demanding.

    If you are fitting a region between [0, xmax], then a frequency error of metadata will result in a phase error of
    2*np.pi*metadata*xmax across the image. For an image with 500 pixels,
    metadata = 1e-3, dphi = pi
    metadata = 1e-4, dphi = pi/10
    metadata = 1e-5, dphi = pi/100
    and we might expect the fit will have an error of dphi/2. Part of the problem is we define the phase relative to
    the origin, which is typically at the edge of ROI, whereas the fit will tend to match the phase correctly in the
    center of the ROI.

    :param img: 2D array, must be positive
    :param sim_frq: [fx, fy]. Should be frequency (not angular frequency).
    :param dxy: pixel size (um)
    :param phase_guess: optional guess for phase
    :param origin: "center" or "edge"

    :return phase_fit: fitted value for the phase
    """
    if np.any(img < 0):
        raise ValueError('img must be strictly positive.')

    if origin == "center":
        x = tools.get_fft_pos(img.shape[1], dxy, centered=True, mode="symmetric")
        y = tools.get_fft_pos(img.shape[0], dxy, centered=True, mode="symmetric")
    elif origin == "edge":
        x = tools.get_fft_pos(img.shape[1], dxy, centered=False, mode="positive")
        y = tools.get_fft_pos(img.shape[0], dxy, centered=False, mode="positive")
    else:
        raise ValueError("'origin' must be 'center' or 'edge' but was '%s'" % origin)

    xx, yy = np.meshgrid(x, y)

    def fn(phi): return -np.cos(2*np.pi * (sim_frq[0] * xx + sim_frq[1] * yy) + phi)
    def fn_deriv(phi): return np.sin(2*np.pi * (sim_frq[0] * xx + sim_frq[1] * yy) + phi)
    def min_fn(phi): return np.sum(fn(phi) * img)
    def jac_fn(phi): return np.asarray([np.sum(fn_deriv(phi) * img),])

    # using jacobian makes faster and more robust
    result = scipy.optimize.minimize(min_fn, phase_guess, jac=jac_fn)
    # also using routine optimized for scalar univariate functions works
    #result = scipy.optimize.minimize_scalar(min_fn)
    phi_fit = np.mod(result.x, 2 * np.pi)

    return phi_fit


def get_phase_wicker_iterative(imgs_ft, otf, sim_frq, dxy, fmax, phases_guess=(0, 2 * np.pi / 3, 4 * np.pi / 3), fit_amps=True):
    """
    # TODO: this can also return components separated the opposite way of desired

    Estimate phases using the cross-correlation minimization method of Wicker, https://doi.org/10.1364/OE.21.002032.
    NOTE: the Wicker method only determines the relative phase. Here we try to estimate the absolute phase
    by additionally looking at the peak phase for one pattern

    # todo: currently hardcoded for 3 phases
    # todo: can I improve the fitting by adding jacobian?
    # todo: can get this using d(M^{-1})dphi1 = - M^{-1} * (dM/dphi1) * M^{-1}
    # todo: probably not necessary, because phases should always be close to equally spaced, so initial guess should be good

    :param imgs_ft: 3 x ny x nx. o(f), o(f-fo), o(f+fo)
    :param otf:
    :param sim_frq: [fx, fy]
    :param dxy: pixel size (um)
    :param fmax: maximum spatial frequency
    :param phase_guess: [phi1, phi2, phi3] in radians.if None will use [0, 120, 240]
    :param fit_amps: if True will also fit amplitude differences between components

    :return phases: list of phases determined using this method
    :return amps: [A1, A2, A3]. If fit_amps is False, these will all be ones.
    """

    _, ny, nx = imgs_ft.shape
    fx = tools.get_fft_frqs(nx, dxy)
    dfx = fx[1] - fx[0]
    fy = tools.get_fft_frqs(ny, dxy)
    dfy = fy[1] - fy[0]

    # compute cross correlations of data
    cross_corrs = np.zeros((3, 3, 3), dtype=np.complex)
    # C_i(k) = S(k - i*p)
    # this is the order set by matrix M, i.e.
    # M * [S(k), S(k - i*p), S(k + i*p)]
    inds = [0, 1, -1]
    for ii, mi in enumerate(inds):  # [0, 1, 2] -> [0, 1, -1]
        for jj, mj in enumerate(inds):
            for ll, ml in enumerate(inds):

                # get shifted otf -> otf(f - l * fo)
                otf_shift, _ = tools.translate_pix(otf, -ml*sim_frq, dr=(dfx, dfy), axes=(1, 0), wrap=False)

                with np.errstate(invalid="ignore", divide="ignore"):
                    weight = otf * otf_shift.conj() / (np.abs(otf_shift)**2 + np.abs(otf)**2)
                    weight[np.isnan(weight)] = 0
                    # todo: is this supposed to be assigned to?
                    weight / np.sum(weight)

                # shifted component C_j(f - l*fo)
                cshift = tools.translate_ft(imgs_ft[jj], -ml * sim_frq, dxy)
                # compute weighted cross correlation
                cross_corrs[ii, jj, ll] = np.sum(imgs_ft[ii] * cshift.conj() * weight) / np.sum(weight)

                # plt.figure()
                # nrows = 2
                # ncols = 2
                # plt.suptitle("ii = %d, jj = %d, ll = %d" % (mi, mj, ml))
                #
                # plt.subplot(nrows, ncols, 1)
                # plt.imshow(np.abs(imgs_ft[ii]), norm=PowerNorm(gamma=0.1))
                # plt.title("D_i(k)")
                #
                # plt.subplot(nrows, ncols, 2)
                # plt.imshow(np.abs(cshift), norm=PowerNorm(gamma=0.1))
                # plt.title("D_j(k-lp)")
                #
                # plt.subplot(nrows, ncols, 3)
                # plt.imshow(otf_shift)
                # plt.title('otf_j(k-lp)')
                #
                # plt.subplot(nrows, ncols, 4)
                # plt.imshow(np.abs(imgs_ft[ii] * cshift.conj()), norm=PowerNorm(gamma=0.1))
                # plt.title("D_i(k) x D_j^*(k-lp)")

                # remove extra noise correlation expected from same images
                if ml == 0 and ii == jj:
                    noise_power = get_noise_power(imgs_ft[ii], fx, fy, fmax)
                    cross_corrs[ii, jj, ll] = cross_corrs[ii, jj, ll] - noise_power

    # optimize
    if fit_amps:
        def minv(p): return np.linalg.inv(get_band_mixing_matrix([p[0], p[1], p[2]], [1, 1, 1], [1, p[3], p[4]]))
    else:
        def minv(p): return np.linalg.inv(get_band_mixing_matrix(p, [1, 1, 1]))

    # remove i = (j + l) terms
    ii_minus_jj = np.array(inds)[:, None] - np.array(inds)[None, :]

    def fn_sum(p, ll): return np.sum(np.abs(minv(p).dot(cross_corrs[:, :, ll]).dot(minv(p).conj().transpose()) *
                                        (ii_minus_jj != inds[ll]))**0.5)

    def fn(p): return np.sum([fn_sum(p, ll) for ll in range(3)])

    # can also include amplitudes and modulation depths in optimization process
    if fit_amps:
        result = scipy.optimize.minimize(fn, np.concatenate((phases_guess, np.array([1, 1]))))
        phases = result.x[0:3]
        amps = np.concatenate((np.array([1]), result.x[3:]))
    else:
        result = scipy.optimize.minimize(fn, phases_guess)
        phases = result.x
        amps = np.array([1, 1, 1])

    # estimate absolute phase by looking at phase of peak
    phase_offsets = np.angle(tools.get_peak_value(imgs_ft[0], fx, fy, sim_frq, peak_pixel_size=2))
    phases = np.mod((phases - phases[0]) + phase_offsets, 2*np.pi)

    return phases, amps

# power spectrum and modulation depths
def get_noise_power(img_ft, fxs, fys, fmax):
    """
    Estimate average noise power of an image by looking at frequencies beyond the maximum frequency
    where the OTF has support.

    :param img_ft: Fourier transform of image. Can be obtained with the idiom img_ft = fftshift(fft2(ifftshift(img)))
    :param fxs: 1D array, x-frequencies
    :param fys: 1D array, y-frequencies
    :param fmax: maximum frequency where signal may be present, i.e. (0.5*wavelength/NA)^{-1}
    :return noise_power:
    """

    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff = np.sqrt(fxfx ** 2 + fyfy ** 2)
    noise_power = np.mean(np.abs(img_ft[ff > fmax])**2)

    return noise_power


def get_mcnr(img_ft, frqs, fxs, fys, fmax, peak_pixel_size=1):
    """
    Get ratio of modulation-contrast-to-noise, which is a measure of quality of SIM contrast.
    MCNR = sqrt(peak power) / sqrt(noise power)

    see e.g. https://doi.org/10.1038/srep15915 for more discussion of the MCNR

    :param img_ft: fourier transform of given SIM image
    :param frqs: [fx, fy]
    :param fxs:
    :param fys:
    :param fmax: maximum frequency where the OTF has support

    :return mcnr:
    """

    peak_height = np.abs(tools.get_peak_value(img_ft, fxs, fys, frqs, peak_pixel_size=peak_pixel_size))
    noise = np.sqrt(get_noise_power(img_ft, fxs, fys, fmax))

    mcnr = peak_height / noise

    return mcnr


def power_spectrum_fn(p, fmag, otf):
    """
    Power spectrum of the form m**2 * A**2 * |K|^{-2*B} * |otf|^2 + noise

    :param p: [A, B, m, noise]
    :param fmag: frequency magnitude
    :param otf: optical transfer function
    :return:
    """

    val = p[2]**2 * p[0]**2 * np.abs(fmag) ** (-2 * p[1]) * np.abs(otf)**2 + p[3]

    return val


def power_spectrum_jacobian(p, fmag, otf):
    """
    jacobian of power_spectrum_fn()

    :param p: [A, B, m, noise]
    :param fmag:
    :param otf:
    :return:
    """
    return [2 * p[2] ** 2 * p[0] * np.abs(fmag) ** (-2 * p[1]) * np.abs(otf) ** 2,
            -2 * np.log(np.abs(fmag)) * p[2]**2 * p[0]**2 * np.abs(fmag)**(-2*p[1]) * np.abs(otf)**2,
            2 * p[2] * p[0]**2 * np.abs(fmag)**(-2*p[1]) * np.abs(otf)**2,
            np.ones(fmag.shape)]


def fit_power_spectrum(img_ft, otf, fxs, fys, fmax, fbounds, fbounds_shift=None,
                       frq_sim=None, init_params=None, fixed_params=None, bounds=None):
    """
    Fit power spectrum, P = |img_ft(f-fsim) * otf(f)|**2 to the form m^2 A^2*|f-fsim|^{-2B}*otf(f) + N
    A and B are the two fit parameters, and N is determined from values of img_ft in the region where the otf does not
    have support. This is determined by the provided wavelength and numerical aperture.

    # todo: instead of using fbounds and fbounds_shift, give OTF cutoff

    :param img_ft: fourier space representation of img
    :param otf: optical transfer function, same size as img_ft
    :param fxs:
    :param fys:
    :param fmax:
    :param fbounds: tuple of upper and lower bounds in units of fmax. Only frequencies in [fmax * fbounds[0], fmax * fbounds[1]]
    will be used in fit
    :param fbounds_shift: tuple of upper and lower bounds in units of fmax. These bounds are centered about frq_sim,
    meaning only points further away from frq_sim than fmax * fbounds[0], or closer than fmax * fbounds[1] will be
    consider in the fit
    :param frq_sim:
    :param init_params: either a list of [A, sigma, B, alpha, m, noise] or None. If None, will automatically guess parameters. If fit_modulation_only
    is True init_params must be provided.
    :param fixed_params:
    :param bounds:

    :return fit_results: fit results dictionary object. See analysis_tools.fit_model() for more details.
    :return mask: mask indicating region that is used for fitting
    """

    if fbounds_shift is None:
        fbounds_shift = fbounds

    if frq_sim is None:
        frq_sim = [0, 0]

    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff = np.sqrt(fxfx**2 + fyfy**2)
    ff_shift = np.sqrt((fxfx - frq_sim[0])**2 + (fyfy - frq_sim[1])**2)

    # exclude points outside of bounds, and at zero frequency (because power spectrum fn is singular there)
    in_normal_bounds = np.logical_and(ff <= fmax*fbounds[1], ff >= fmax*fbounds[0])
    in_shifted_bounds = np.logical_and(ff_shift <= fmax*fbounds_shift[1], ff_shift >= fmax * fbounds_shift[0])
    in_bounds = np.logical_and(in_normal_bounds, in_shifted_bounds)
    mask = np.logical_and(ff != 0, in_bounds)

    # estimate initial parameters if not provided. Allow subset of fit parameters to be passed as None's
    if init_params is None:
        init_params = [None] * 4

    if np.any([ip is None for ip in init_params]):
        img_guess, _, f_guess, _, _, _ = tools.azimuthal_avg(np.abs(img_ft) ** 2, ff_shift, [0, 0.1*fmax])

        noise_guess = get_noise_power(img_ft, fxs, fys, fmax)
        amp_guess = np.sqrt(img_guess[0] * f_guess[0])
        guess_params = [amp_guess, 0.5, 1, noise_guess]

        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = guess_params[ii]

    if fixed_params is None:
        fixed_params = [False] * len(init_params)

    if bounds is None:
        bounds = ((0, 0, 0, 0), (np.inf, 1.25, 1, np.inf))

    # fit function
    fit_fn = lambda p: power_spectrum_fn(p, ff_shift[mask], otf[mask])
    ps = np.abs(img_ft[mask]) ** 2

    jac_fn = lambda p: power_spectrum_jacobian(p, ff_shift[mask], otf[mask])

    # do fitting
    fit_results = fit.fit_model(ps, fit_fn, init_params, fixed_params=fixed_params,
                                  bounds=bounds, model_jacobian=jac_fn)

    return fit_results, mask


def plot_power_spectrum_fit(img_ft, otf, options, pfit, frq_sim=None, mask=None, figsize=(20, 10), ttl_str=""):
    """
    Plot results of fit_power_spectrum()

    :param img_ft:
    :param otf:
    :param options: {"pixel_size", "wavelength", "na"}
    :param pfit: [A, alpha, m, noise]
    :param frq_sim:
    :param mask:
    :param figsize:
    :return fig: handle to figure
    """

    if frq_sim is None:
        frq_sim = [0, 0]

    if mask is None:
        mask = np.ones(img_ft.shape, dtype=np.bool)

    # get physical data
    dx = options['pixel_size']
    dy = dx
    wavelength = options['wavelength']
    na = options['na']
    fmax = 1 / (0.5 * wavelength / na)

    # get frequency data
    fxs = tools.get_fft_frqs(img_ft.shape[1], dx)
    fys = tools.get_fft_frqs(img_ft.shape[0], dy)
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff_shift = np.sqrt((fxfx - frq_sim[0]) ** 2 + (fyfy - frq_sim[1]) ** 2)

    # get experimental and theoretical visualization
    ps_exp = np.abs(img_ft)**2
    ps_fit = power_spectrum_fn(pfit, ff_shift, otf)
    ps_fit_no_otf = power_spectrum_fn(list(pfit[:-1]) + [0], ff_shift, 1)

    # wienier filter, i.e. "divide" by otf
    snr = ps_fit_no_otf / pfit[-1]
    wfilter = np.abs(otf)**2 / (np.abs(otf)**4 + 1 / snr**2)
    ps_exp_deconvolved = ps_exp * wfilter
    ps_fit_deconvolved = ps_fit * wfilter

    # ######################
    # plot results
    # ######################
    extent = tools.get_extent(fys, fxs)

    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(2, 10)

    sttl_str = ""
    if ttl_str != "":
        sttl_str += "%s\n" % ttl_str
    sttl_str += 'm=%0.3g, A=%0.3g, alpha=%0.3f\nnoise=%0.3g, frq sim = (%0.2f, %0.2f) 1/um' % \
                (pfit[-2], pfit[0], pfit[1], pfit[-1], frq_sim[0], frq_sim[1])
    plt.suptitle(sttl_str)

    # ######################
    # plot power spectrum x otf vs frequency in 1D
    # ######################
    ax1 = plt.subplot(grid[0, :5])

    ax1.semilogy(ff_shift.ravel(), ps_exp.ravel(), 'k.')
    ax1.semilogy(ff_shift[mask].ravel(), ps_exp[mask].ravel(), 'b.')
    ax1.semilogy(ff_shift.ravel(), ps_fit.ravel(), 'r')

    ylims = ax1.get_ylim()
    ax1.set_ylim([ylims[0], 1.2 * np.max(ps_exp[mask].ravel())])

    ax1.set_xlabel('frequency (1/um)')
    ax1.set_ylabel('power spectrum')
    ax1.legend(['all data', 'data used to fit', 'fit'], loc="upper right")
    ax1.title.set_text('m^2 A^2 |f|^{-2*alpha} |otf(f)|^2 + N')

    # ######################
    # plot az avg divided by otf in 1D
    # ######################
    ax2 = plt.subplot(grid[0, 5:])

    ax2.semilogy(ff_shift.ravel(), ps_exp_deconvolved.ravel(), 'k.')
    ax2.semilogy(ff_shift[mask].ravel(), ps_exp_deconvolved[mask].ravel(), 'b.')
    ax2.semilogy(ff_shift.ravel(), ps_fit_no_otf.ravel(), 'g')

    ylims = ax1.get_ylim()
    ax1.set_ylim([ylims[0], 1.2 * np.max(ps_exp_deconvolved[mask].ravel())])

    ax2.title.set_text('m^2 A^2 |k|^{-2*alpha} |otf|^4/(|otf|^4 + snr^2)')
    ax2.set_xlabel('|f - f_sim| (1/um)')
    ax2.set_ylabel('power spectrum')

    # ######################
    # plot 2D power spectrum
    # ######################
    ax3 = plt.subplot(grid[1, :2])

    ax3.imshow(ps_exp, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent)

    circ = matplotlib.patches.Circle((0, 0), radius=fmax, color='k', fill=0, ls='--')
    ax3.add_artist(circ)

    circ2 = matplotlib.patches.Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--')
    ax3.add_artist(circ2)

    ax3.set_xlabel('fx (1/um)')
    ax3.set_ylabel('fy (1/um)')
    ax3.title.set_text('raw power spectrum')

    # ######################
    # 2D fit
    # ######################
    ax5 = plt.subplot(grid[1, 2:4])
    ax5.imshow(ps_fit, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent)

    circ = matplotlib.patches.Circle((0, 0), radius=fmax, color='k', fill=0, ls='--')
    ax5.add_artist(circ)

    circ2 = matplotlib.patches.Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--')
    ax5.add_artist(circ2)

    ax5.set_xlabel('fx (1/um)')
    ax5.set_ylabel('fy (1/um)')
    ax5.title.set_text('2D fit')

    # ######################
    # plot 2D power spectrum divided by otf with masked region
    # ######################
    ax4 = plt.subplot(grid[1, 6:8])
    # ps_over_otf[mask == 0] = np.nan
    ps_exp_deconvolved[mask == 0] = np.nan
    ax4.imshow(ps_exp_deconvolved, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent)

    circ = matplotlib.patches.Circle((0, 0), radius=fmax, color='k', fill=0, ls='--')
    ax4.add_artist(circ)

    circ2 = matplotlib.patches.Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--')
    ax4.add_artist(circ2)

    ax4.set_xlabel('fx (1/um)')
    ax4.set_ylabel('fy (1/um)')
    ax4.title.set_text('masked, deconvolved power spectrum')

    #
    ax4 = plt.subplot(grid[1, 8:])
    ax4.imshow(ps_fit_deconvolved, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent)

    circ = matplotlib.patches.Circle((0, 0), radius=fmax, color='k', fill=0, ls='--')
    ax4.add_artist(circ)

    circ2 = matplotlib.patches.Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--')
    ax4.add_artist(circ2)

    ax4.set_xlabel('fx (1/um)')
    ax4.set_ylabel('fy (1/um)')
    ax4.title.set_text('fit deconvolved')

    return fig

# inversion functions
def get_band_mixing_matrix(phases, mod_depths=(1, 1, 1), amps=(1, 1, 1)):
    """
    Return matrix M, which relates the measured images D to the Fluorescence profile S multiplied by the OTF H
    [[D_1(k)], [D_2(k)], [D_3(k)]] = M * [[S(k)H(k)], [S(k-p)H(k)], [S(k+p)H(k)]]

    We assume the modulation has the form [1 + m*cos(k*r + phi)], leading to
    M = [A_1 * [1, 0.5*m*exp(ip_1), 0.5*m*exp(-ip_1)],
         A_2 * [1, 0.5*m*exp(ip_2), 0.5*m*exp(-ip_2)],
         A_3 * [1, 0.5*m*exp(ip_3), 0.5*m*exp(-ip_3)]
        ]

    :param phases: [p1, p2, p3]
    :param mod_depths: [m1, m2, m3]. In most cases, m1=m2=m3
    :param amps: [A1, A2, A3]
    :return mat: matrix M
    """

    mat = []
    for p, m, a in zip(phases, mod_depths, amps):
        row_vec = a * np.array([1, 0.5 * m * np.exp(1j * p), 0.5 * m * np.exp(-1j * p)])
        mat.append(row_vec)
    mat = np.asarray(mat)

    return mat


def image_times_matrix(imgs, matrix):
    """
    Multiply a set of images by a matrix (elementwise) to create a new set of images

    Suppose we have three input images [[I_1(k), I_2(k), I_3(K)]] and these are related to a set of output images
    [[D_1(k), D_2(k), D_3(k)]] = matrix * [[I_1(k), I_2(k), I_3(K)]]

    :param imgs:
    :param matrix:
    :param imgs_out:
    :return:
    """
    nimgs, ny, nx = imgs.shape

    vec = np.reshape(imgs, [nimgs, ny * nx])
    vec_out = matrix.dot(vec)
    imgs_out = np.reshape(vec_out, [nimgs, ny, nx])

    return imgs_out


def do_sim_band_separation(imgs_ft, phases, mod_depths=None, amps=None):
    """
    Do noisy inversion of SIM data, i.e. determine
    [[S(f)H(k)], [S(f-p)H(f)], [S(f+p)H(f)]] = M^{-1} * [[D_1(f)], [D_2(f)], [D_3(f)]]

    # todo: generalize for case with more than 3 phases or angles

    :param imgs_ft: nangles x nphases x ny x nx. Fourier transform of SIM image data with zero frequency information
     in middle. i.e. as obtained from fftshift
    :param phases: array nangles x nphases listing phases
    :param mod_depths: list of length nangles. Optional. If not provided, all are set to 1.
    :param amps: list of length nangles x nphases. If not provided, all are set to 1.
    :return components_ft: nangles x nphases x ny x nx array, where the first index corresponds to the bands
    S(f)H(f), S(f-p)H(f), or S(f+p)H(f)
    """
    nangles, nphases, ny, nx = imgs_ft.shape

    # default parameters
    if mod_depths is None:
        mod_depths = np.ones((nangles, nphases))

    if amps is None:
        amps = np.ones((nangles, nphases))

    # check parameters
    if nphases != 3:
        raise NotImplementedError("only implemented for nphases=3, but nphases=%d" % nphases)

    bands_ft = np.empty((nangles, nphases, ny, nx), dtype=np.complex) * np.nan

    # try to do inversion
    for ii in range(nangles):
        mixing_mat = get_band_mixing_matrix(phases[ii], mod_depths[ii], amps[ii])

        try:
            mixing_mat_inv = np.linalg.inv(mixing_mat)
            # numerical errors are only thing preventing the last two lines from being equal up to conjugation
            if np.max(np.abs(mixing_mat_inv[1, :] - mixing_mat_inv[2, :].conj())) > 1e-13:
                raise Exception("band mixing matrix inverse had improper form")
            mixing_mat_inv[2, :] = mixing_mat_inv[1, :].conj()

            bands_ft[ii] = image_times_matrix(imgs_ft[ii], mixing_mat_inv)

        except np.linalg.LinAlgError:
            warnings.warn("warning, inversion matrix for angle index=%d is singular. bands set to nans" % ii)

    return bands_ft


def get_band_overlap(band0, band1, otf0, otf1, otf_threshold=0.1):
    """
    Get amplitude and phase of
    C = \sum [Band_0(f) * conj(Band_1(f + fo))] / \sum [ |Band_0(f)|^2]
    where Band_1(f) = O(f-fo), so Band_1(f+fo) = O(f). i.e. these are the separated SIM bands after deconvolution.

    If correct reconstruction parameters are used, expect Band_0(f) and Band_1(f) differ only by a complex constant.
    This constant contains information about the global phase offset AND the modulation depth

    Given this information, can perform the phase correction
    Band_1(f + fo) -> -> np.exp(i phase_corr) * Band_1(f + fo)

    :param band0: nangles x ny x nx. Typically S(f)otf(f) * wiener(f) ~ S(f)
    :param band1: nangles x ny x nx. Typically S(f-fo)otf(f) * wiener(f) ~ S(f-fo) after shifting to correct position
    :param mask: where mask is True, use these points to evaluate the band correlation. Typically construct by picking
    some value where otf(f) and otf(f + fo) are both > w, where w is some cutoff value.

    :return phases, mags:
    """
    mask = np.logical_and(otf0 > otf_threshold, otf1 > otf_threshold)

    nangles = band0.shape[0]
    phases = np.zeros((nangles))
    mags = np.zeros((nangles))

    # divide by OTF, but don't worry about Wiener filtering avoid problems by keeping otf_threshold large enough
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = band0 / otf0 * band1.conj() / otf1.conj()
        denominator = np.abs(band0 / otf0) ** 2

    for ii in range(nangles):
        corr = np.sum(numerator[ii][mask[ii]]) / np.sum(denominator[ii][mask[ii]])
        mags[ii] = np.abs(corr)
        phases[ii] = np.angle(corr)

    return phases, mags

# filtering and combining images
def get_wiener_filter(otf, sn_power_ratio):
    """
    Return Wiener deconvolution filter, which is used to obtain an estimate of img_ft after removing
    the effect of the OTF. Wiener filtering is the optimal (in the least-squares sense) method to recover an
    estimate of an image corrupted by noise, at least when the signal-to-noise ratio is known.

    Given a noisy image defined by
    img_ft(f) = obj(f) * otf(f) + N(f)
    the Wiener deconvolution filter is given by
    otf^*(f) / (|otf(f)|^2 + <|N(f)|>^2 / |obj(f)|^2).
    Where SNR is large, this filter approaches zero. Where SNR is small, it divides out the otf.

    :param otf:
    :param sn_power_ratio:
    :return:
    """
    wfilter = otf.conj() / (np.abs(otf) ** 2 + 1 / sn_power_ratio)
    wfilter[np.isnan(wfilter)] = 0

    return wfilter


def estimate_snr(img_ft, fxs, fys, fmax, filter_size=5):
    """
    estimate signal to noise ratio from image
    # todo: never got this to work satisfactorally...

    :param img_ft:
    :param fxs:
    :param fys:
    :param fmax:
    :param filter_size:
    :return:
    """

    kernel = np.ones((filter_size, filter_size))
    kernel = kernel / np.sum(kernel)
    sig_real = scipy.ndimage.filters.convolve(img_ft.real, kernel)
    sig_imag = scipy.ndimage.filters.convolve(img_ft.imag, kernel)

    noise_power = get_noise_power(img_ft, fxs, fys, fmax)
    # have to subtract this to remove bias. This comes from fact pixel noise always correlated with itself
    sig_power = sig_real**2 + sig_imag**2 - noise_power / kernel.size
    # this is the expect standard deviation. Set anything less than 2 of these to zero...
    sd = noise_power / kernel.size
    sig_power[sig_power < 2*sd] = 1e-12

    # snr estimate
    snr = sig_power / noise_power

    return snr


# create test data
def get_test_pattern(img_size=(2048, 2048)):
    """
     Generate a ground truth (binary) test pattern which consists stripes of variable spacing at various angles

    :param img_size: list, size of image
    :return test_pattern: numpy array
    """
    ny, nx = img_size
    # mask = np.zeros((ny, nx))

    # patterns with variable spacing
    periods = range(2, 20, 2)
    # vcounter = 0
    for ii, p in enumerate(periods):
        cell = np.zeros((p, nx))
        on_pix = int(np.ceil(p / 2))
        cell[:on_pix, :] = 1
        cell = np.tile(cell, [4, 1])

        if ii == 0:
            mask = cell
        else:
            mask = np.concatenate((mask, cell), axis=0)

    mask = mask[:, :mask.shape[0]]

    mask_block = np.concatenate((mask, np.rot90(mask)), axis=1)
    mask_block2 = np.concatenate((np.rot90(mask), mask), axis=1)

    mask_superblock = np.concatenate((mask_block, mask_block2))

    ny_reps = int(np.ceil(ny / mask_superblock.shape[0]))
    nx_reps = int(np.ceil(nx / mask_superblock.shape[1]))
    mask = np.tile(mask_superblock, [ny_reps, nx_reps])
    mask = mask[0:ny, 0:nx]

    return mask


def get_lines_test_pattern(img_size=(2048, 2048), angles=(0, 45, 90, 135)):
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
    line_center_sep = np.arange(13, 0, -1)
    line_edge_sep = line_center_sep - width

    line_pair_sep = 30

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


def get_simulated_sim_imgs(ground_truth, frqs, phases, mod_depths, max_photons, cam_gains, cam_offsets,
                           cam_readout_noise_sds, pix_size, amps=None, origin="center",
                           coherent_projection=True, otf=None, **kwargs):
    """
    Get simulated SIM images, including the effects of shot-noise and camera noise.

    :param ground_truth: ground truth image of size ny x nx
    :param frqs: SIM frequencies, of size nangles x 2. frqs[ii] = [fx, fy]
    :param phases: SIM phases in radians. Of size nangles x nphases. Phases may be different for each angle.
    :param list mod_depths: SIM pattern modulation depths. Size nangles. If pass matrices, then mod depths can vary
    spatially. Assume pattern modulation is the same for all phases of a given angle. Maybe pass list of numpy arrays
    :param max_photons: maximum photon number (i.e. photon number for pixels in ground_truth that = 1).
    :param cam_gains: gain of each pixel (or single value for all pixels)
    :param cam_offsets: offset of each pixel (or single value for all pixels)
    :param cam_readout_noise_sds: noise standard deviation for each pixel (or single value for all pixels)
    :param pix_size: pixel size in um
    :param str origin: "center" or "edge"
    :param bool coherent_projection:
    :param otf:
    :param kwargs: keyword arguments "wavlength", "otf", "na", etc. will be passed through to simulated_img()

    :return sim_imgs: nangles x nphases x ny x nx array
    """
    nangles = len(frqs)
    nphases = len(phases)
    ny, nx = ground_truth.shape

    if otf is None and not coherent_projection:
        raise ValueError("If coherent_projection is false, OTF must be provided")

    if len(mod_depths) != nangles:
        raise ValueError("mod_depths must have length nangles")

    if amps is None:
        amps = np.ones((nangles, nphases))

    if origin == "center":
        x = tools.get_fft_pos(nx, pix_size, centered=True, mode="symmetric")
        y = tools.get_fft_pos(ny, pix_size, centered=True, mode="symmetric")
    elif origin == "edge":
        x = tools.get_fft_pos(nx, pix_size, centered=False, mode="positive")
        y = tools.get_fft_pos(ny, pix_size, centered=False, mode="positive")
    else:
        raise ValueError("'origin' must be 'center' or 'edge' but was '%s'" % origin)

    xx, yy = np.meshgrid(x, y)

    if 'bin_size' in kwargs:
        nbin = kwargs['bin_size']
    else:
        nbin = 1

    sim_imgs = np.zeros((nangles, nphases, int(ny / nbin), int(nx / nbin)))
    snrs = np.zeros(sim_imgs.shape)

    for ii in range(nangles):
        for jj in range(nphases):
            # pattern = amps[ii, jj] * (1 + mod_depths[ii, jj] * np.cos(2*np.pi * (frqs[ii][0] * xx + frqs[ii][1] * yy) +
            #                                                           phases[ii, jj]))
            pattern = amps[ii, jj] * (1 + mod_depths[ii] * np.cos(2 * np.pi * (frqs[ii][0] * xx + frqs[ii][1] * yy) +
                                                        phases[ii, jj]))

            if not coherent_projection:
                pattern_ft = fft.fftshift(fft.fft2(fft.ifftshift(pattern)))
                pattern = fft.fftshift(fft.ifft2(fft.ifftshift(pattern_ft * otf))).real

            psf_mat, _ = psf.otf2psf(otf)
            sim_imgs[ii, jj], snrs[ii, jj] = camera_noise.simulated_img(ground_truth * pattern, cam_gains, cam_offsets,
                                                                        cam_readout_noise_sds, psf=psf_mat, **kwargs)

    return sim_imgs, snrs
