"""
Tools for reconstructing 2D SIM images, using 3-angles and 3-phases.
"""
# from . import analysis_tools as tools
# from . import fit_psf as psf
# from . import affine as affine
# from . import psd
import analysis_tools as tools
import fit_psf as psf
import affine
import camera_noise
import psd
import camera_noise

# general imports
import pickle
import os
import time
import datetime
import warnings
import shutil
import joblib

# numerical tools
import numpy as np
from scipy import fft
import scipy.optimize
import scipy.signal
import scipy.ndimage
import skimage
import skimage.restoration
from skimage.exposure import match_histograms
from PIL import Image

# MCMC
import theano
import theano.tensor as tt
import pymc3 as pm

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.colors import LogNorm
import matplotlib.patches

def reconstruct_folder(data_root_paths, pixel_size, na, emission_wavelengths, excitation_wavelengths,
                       affine_data_paths, otf_data_fname, dmd_pattern_data_fpath,
                       channel_inds=None, crop_image=False, img_centers=None,
                       crop_sizes=None, use_scmos_cal=False, scmos_calibration_file=None, widefield_only=False,
                       nangles=3, nphases=3, npatterns_ignored=0, saving=True,
                       zinds_to_use=None, tinds_to_use=None, xyinds_to_use=None,
                       save_tif_stack=True, **kwargs):
    """
    Reconstruct entire folder of SIM data and save results in TIF stacks. Responsible for loading relevant data
    (images, affine transformations, SIM pattern information), selecting images to recombine from metadata, and
    saving results for SIM superresolution, deconvolution, widefield image, etc.

    :param list data_root_paths: list of directories where data is stored
    :param float pixel_size: pixel size in ums. If None, read pixel size from metadata
    :param float na: numerical aperture
    :param list emission_wavelengths: list of emission wavelengths
    :param list excitation_wavelengths: list of excitation wavelengths
    :param list affine_data_paths: list of paths to files storing data about affine transformations between DMD and camera
    space. [path_color_0, path_color_1, ...]. The affine data files store pickled dictionary objects. The dictionary
    must have an entry 'affine_xform' which contains the affine transformation matrix (in homogeneous coordinates)
    :param str otf_data_fname: path to file storing optical transfer function data. Data is a pickled dictionary object
    and must have entry 'fit_params'.
    :param list dmd_pattern_data_fpath: list of paths to files storing data about DMD patterns for each color. Data is
    stored in a pickled dictionary object which must contain fields 'frqs', 'phases', 'nx', and 'ny'
    :param list channel_inds: list of channel indices corresponding to each color. If set to None, will use [0, 1, ..., ncolors -1]
    :param bool crop_image:
    :param list img_centers: list of centers for images in each data directory to be used in cropping [[cy, cx], ...]
    :param list or int crop_sizes: list of crop sizes for each data directory
    :param bool use_scmos_cal: if true correct camera counts to photons using calibration information
    :param str scmos_calibration_file: path to scmos calibration file. This should be a pickled file storing a list
    object [gainmap, means, varmap]
    :param bool widefield_only: if true only produce widefield images but don't do full reconstruction. Useful
    for diagnostic purposes
    :param int nangles:
    :param int nphases:
    :param bool npatterns_ignored: number of patterns to ignore at the start of each channel.
    :param bool saving: if True, save results
    :param bool save_tif_stack:
    :param str sim_data_export_fname:
    :param **kwargs: passed through to reconstruction

    :return np.ndarray imgs_sr:
    :return np.ndarray imgs_wf:
    :return np.ndarray imgs_deconvolved:
    :return np.ndarray imgs_os:
    """

    nfolders = len(data_root_paths)
    ncolors = len(emission_wavelengths)

    if channel_inds is None:
        channel_inds = list(range(ncolors))

    # ensure crop_sizes is a list the same size as number of folders
    if not isinstance(crop_sizes, list):
        crop_sizes = [crop_sizes]

    if len(crop_sizes) == 1 and nfolders > 1:
        crop_sizes = crop_sizes * nfolders

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
        ppath = dmd_pattern_data_fpath[kk]
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
    with open(otf_data_fname, 'rb') as f:
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
    # load camera calibration file, if we need it
    # ############################################
    if use_scmos_cal:
        with open(scmos_calibration_file, 'rb') as f:
            data = pickle.load(f)
        gain_map = data['gains']
        offsets = data['offsets']
        #varmap = data['vars']

    # ############################################
    # SIM images
    # ############################################
    if not crop_image:
        crop_sizes = [np.nan] * len(data_root_paths)
        img_centers = [[np.nan, np.nan]] * len(data_root_paths)

    for rpath, crop_size, img_center in zip(data_root_paths, crop_sizes, img_centers):
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
                _, fname = os.path.split(otf_data_fname)
                fpath = os.path.join(sim_results_path, fname)
                shutil.copyfile(otf_data_fname, fpath)

                # copy DMD pattern data here
                _, fname = os.path.split(dmd_pattern_data_fpath[kk])
                fpath = os.path.join(sim_results_path, fname)
                shutil.copyfile(dmd_pattern_data_fpath[kk], fpath)

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

        # set up image size
        # load one file to check size
        fname = os.path.join(rpath, metadata['FileName'].values[0])
        im, _, _ = tools.read_tiff(fname, [metadata['ImageIndexInFile'].values[0]])
        _, ny_raw, nx_raw = im.shape
        if crop_image:
            # or pick ROI
            roi = tools.get_centered_roi(img_center, [crop_size, crop_size])

            # check points don't exceed image size
            if roi[0] < 0:
                roi[0] = 0
            if roi[1] > ny_raw:
                roi[1] = ny_raw
            if roi[2] < 0:
                roi[2] = 0
            if roi[3] > nx_raw:
                roi[3] = nx_raw
        else:
            roi = [0, ny_raw, 0, nx_raw]

        ny = roi[1] - roi[0]
        nx = roi[3] - roi[2]

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
                                                         phases_dmd[kk, ii, jj], [dmd_ny, dmd_nx], roi, xform)

            # convert from 1/mirrors to 1/um
            frqs_guess = frqs_guess / pixel_size

            # analyze pictures
            for ii in tinds_to_use_temp:
                for bb in xyinds_to_use_temp:
                    for aa in zinds_to_use_temp:
                        tstart = time.process_time()

                        identifier = "%.0fnm_nt=%d_nxy=%d_nz=%d" % (excitation_wavelengths[kk] * 1e3, ii, bb, aa)
                        file_identifier = "nc=%d_nt=%d_nxy=%d_nz=%d" % (kk, ii, bb, aa)

                        # where we will store results for this particular set
                        if not widefield_only:
                            sim_diagnostics_path = os.path.join(sim_results_path, identifier)
                            if not os.path.exists(sim_diagnostics_path):
                                os.mkdir(sim_diagnostics_path)

                        # find images and load them
                        raw_imgs = tools.read_dataset(metadata, z_indices=aa, xy_indices=bb, time_indices=ii,
                                           user_indices={"UserChannelIndex": channel_inds[kk],
                                           "UserSimIndex": list(range(npatterns_ignored, npatterns_ignored + nangles * nphases))})

                        # error if we have wrong number of images
                        if np.shape(raw_imgs)[0] != (nangles * nphases):
                            raise Exception("Found %d images, but expected %d images at channel=%d,"
                                            " zindex=%d, tindex=%d, xyindex=%d" % (
                                            np.shape(raw_imgs)[0], nangles * nphases,
                                            channel_inds[kk], aa, ii, bb))

                        # optionally convert from ADC to photons
                        # todo: not very useful to do this way...
                        if use_scmos_cal:
                            imgs_sim = camera_noise.adc2photons(raw_imgs, gain_map, offsets)
                        else:
                            imgs_sim = raw_imgs

                        # reshape to [nangles, nphases, ny, nx]
                        imgs_sim = imgs_sim.reshape((nangles, nphases, raw_imgs.shape[1], raw_imgs.shape[2]))
                        imgs_sim = imgs_sim[:, :, roi[0]:roi[1], roi[2]:roi[3]]

                        # instantiate reconstruction object
                        r = SimImageSet(sim_options, imgs_sim, frqs_guess, phases_guess=phases_guess, otf=otf,
                                        save_dir=sim_diagnostics_path, **kwargs)

                        # if not saving stack, maybe want to handle in class?
                        if saving and not save_tif_stack:
                            fname = os.path.join(sim_results_path, "sim_os_%s.tif" % file_identifier)
                            tools.save_tiff(r.imgs_os, fname, dtype='float32', datetime=start_time)

                            fname = os.path.join(sim_results_path, "widefield_%s.tif" % file_identifier)
                            tools.save_tiff(r.widefield, fname, dtype='float32', datetime=start_time)
                        else:
                            # store widefield and os
                            imgs_os.append(r.imgs_os)
                            imgs_wf.append(r.widefield)

                        if not widefield_only:
                            # do reconstruction
                            r.reconstruct()
                            r.plot_figs()

                            if saving and not save_tif_stack:
                                fname = os.path.join(sim_results_path, "sim_sr_%s.tif" % file_identifier)
                                tools.save_tiff(r.img_sr, fname, dtype='float32', datetime=start_time)

                                fname = os.path.join(sim_results_path, "deconvolved_%s.tif" % file_identifier)
                                tools.save_tiff(r.widefield_deconvolution, fname, dtype='float32', datetime=start_time)
                            else:
                                # store sr and deconvolved
                                imgs_sr.append(r.img_sr)
                                imgs_deconvolved.append(r.widefield_deconvolution)

                            # save reconstruction summary data
                            r.save_result(os.path.join(sim_diagnostics_path, "sim_reconstruction_params.pkl"))

                        tend = time.process_time()
                        print("%d/%d from %s in %0.2fs" % (counter, ncolors * nt_used * nxy_used * nz_used, folder, tend - tstart))

                        counter += 1

        # #################################
        # save data for all reconstructed files
        # #################################
        if saving and save_tif_stack:

            # todo: want to include metadata in tif.
            fname = tools.get_unique_name(os.path.join(sim_results_path, 'widefield.tif'))
            imgs_wf = np.asarray(imgs_wf)
            wf_to_save = np.reshape(imgs_wf, [ncolors, nt_used, nz_used, imgs_wf[0].shape[-2], imgs_wf[0].shape[-1]])
            tools.save_tiff(wf_to_save, fname, dtype='float32', axes_order="CTZYX", hyperstack=True,
                            datetime=start_time)

            fname = tools.get_unique_name(os.path.join(sim_results_path, 'sim_os.tif'))
            imgs_os = np.asarray(imgs_os)
            sim_os = np.reshape(imgs_os, [ncolors, nt_used, nz_used, imgs_os[0].shape[-2], imgs_os[0].shape[-1]])
            tools.save_tiff(sim_os, fname, dtype='float32', axes_order="CTZYX", hyperstack=True,
                            datetime=start_time)

            if not widefield_only:
                fname = tools.get_unique_name(os.path.join(sim_results_path, 'sim_sr.tif'))
                imgs_sr = np.asarray(imgs_sr)
                sim_to_save = np.reshape(imgs_sr, [ncolors, nt_used, nz_used, imgs_sr[0].shape[-2], imgs_sr[0].shape[-1]])
                tools.save_tiff(sim_to_save, fname, dtype='float32', axes_order="CTZYX", hyperstack=True,
                                datetime=start_time)

                fname = tools.get_unique_name(os.path.join(sim_results_path, 'deconvolved.tif'))
                imgs_deconvolved = np.asarray(imgs_deconvolved)
                deconvolved_to_save = np.reshape(imgs_deconvolved, [ncolors, nt_used, nz_used, imgs_deconvolved[0].shape[-2],
                                                                    imgs_deconvolved[0].shape[-1]])
                tools.save_tiff(deconvolved_to_save, fname, dtype='float32', axes_order='CTZYX', hyperstack=True,
                                datetime=start_time)

    return imgs_sr, imgs_wf, imgs_deconvolved, imgs_os

class SimImageSet:
    def __init__(self, options, imgs, frq_sim_guess, otf=None,
                 wiener_parameter=1, fbounds=(0.01, 1), fbounds_shift=(0.01, 1),
                 use_bayesian=False, use_wicker=True, normalize_histograms=True, background_counts=100,
                 do_global_phase_correction=True, determine_amplitudes=False, find_frq_first=True,
                 default_to_guess_on_bad_phase_fit=True, max_phase_err=20*np.pi/180,
                 default_to_guess_on_low_mcnr=True, min_mcnr=1,
                 size_near_fo_to_remove=0,
                 phases_guess=None, mod_depths_guess=None, pspec_params_guess=None,
                 use_fixed_phase=False, use_fixed_frq=False, use_fixed_mod_depths=False,
                 plot_diagnostics=True, interactive_plotting=False, save_dir=None, figsize=(20, 10)):
        """
        Class for reconstructing a single SIM image

        :param options: {'pixel_size', 'na', 'wavelength'}. Pixel size and wavelength in um
        :param imgs: nangles x nphases x ny x nx
        :param frq_sim_guess: 2 x nangles array of guess SIM frequency values
        :param otf: optical transfer function evaluated at the same frequencies as the fourier transforms of imgs.
         If None, estimate from NA.
        :param wiener_parameter: Attenuation parameter for Wiener deconvolution. This will attenuate parts of the image
        where |otf(f)|^2 * SNR < wiener_parameter
        :param use_bayesian: Boolean. If True, use bayesian approach to estimate SIM frequency, phase, and modulation
        depth. Otherwise use classical optimization approach.
        :param plot_diagnostics: Boolean. If True, display figures to visually inspect output.
        :param use_fixed_parameters: Boolean. Whether to do parameter fitting, or to use provided parameters.
        :param phases_guess: If use_fixed_parameters is True, these phases are used. Otherwise they are ignored.
        :param mod_depths_guess: If use_fixed_parameters is True, these modulation depths are used, otherwise they are ignored.
        :param pspec_params_guess: If use_fixed_parameters is True, these power spectrum fit parameters are used, otherwise
        they are ignored.
        """
        # #############################################
        # saving information
        # #############################################
        self.save_dir = save_dir
        self.hold_figs_open = False
        self.figsize = figsize

        if self.save_dir is not None:
            self.log_file = open(os.path.join(self.save_dir, "sim_log.txt"), 'w')
        else:
            self.log_file = None

        # #############################################
        # setup plotting
        # #############################################
        if not interactive_plotting:
            plt.ioff()
            plt.switch_backend("agg")

        # #############################################
        # analysis settings
        # #############################################
        self.wiener_parameter = wiener_parameter
        self.use_bayesian = use_bayesian
        self.use_wicker = use_wicker
        self.global_phase_correction = do_global_phase_correction
        self.normalize_histograms = normalize_histograms
        self.size_near_fo_to_remove = size_near_fo_to_remove
        self.default_to_guess_on_bad_phase_fit = default_to_guess_on_bad_phase_fit
        self.max_phase_error = max_phase_err
        self.default_to_guess_on_low_mcnr = default_to_guess_on_low_mcnr
        self.min_mcnr = min_mcnr
        self.determine_amplitudes = determine_amplitudes
        self.use_fixed_phase = use_fixed_phase
        self.use_fixed_frq = use_fixed_frq
        self.use_fixed_mod_depths = use_fixed_mod_depths
        self.find_frq_first = find_frq_first
        self.plot_diagnostics = plot_diagnostics

        # #############################################
        # images
        # #############################################
        self.background_counts = background_counts
        self.imgs = imgs.astype(np.float64)
        self.nangles, self.nphases, self.ny, self.nx = imgs.shape
        
        # #############################################
        # get basic parameters
        # #############################################
        self.dx = options['pixel_size']
        self.dy = options['pixel_size']
        self.na = options['na']
        self.wavelength = options['wavelength']

        self.fmax = 1 / (0.5 * self.wavelength / self.na)
        self.fbounds = fbounds
        self.fbounds_shift = fbounds_shift

        self.frqs_guess = frq_sim_guess
        self.phases_guess = phases_guess
        self.mod_depths_guess = mod_depths_guess
        self.power_spectrum_params_guess = pspec_params_guess

        # #############################################
        # get frequency data and OTF
        # #############################################
        self.fx = tools.get_fft_frqs(self.nx, self.dx)
        self.fy = tools.get_fft_frqs(self.ny, self.dy)

        if otf is None:
            otf = psf.circ_aperture_otf(self.fx[None, :], self.fy[:, None], self.na, self.wavelength)
        self.otf = otf

        # #############################################
        # print current time
        # #############################################
        now = datetime.datetime.now()

        self.print_tee("####################################################################################", self.log_file)
        self.print_tee("%d/%02d/%02d %02d:%02d:%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second), self.log_file)
        self.print_tee("####################################################################################", self.log_file)

        # #############################################
        # normalize histograms for input images
        # #############################################
        if self.normalize_histograms:
            tstart = time.process_time()

            for ii in range(self.nangles):
                for jj in range(1, self.nphases):
                    self.imgs[ii, jj] = match_histograms(self.imgs[ii, jj], self.imgs[ii, 0])

            tend = time.process_time()
            self.print_tee("Normalizing histograms took %0.2fs" % (tend - tstart), self.log_file)

        # #############################################
        # remove background
        # #############################################
        self.imgs = self.imgs - self.background_counts
        self.imgs[self.imgs <= 0] = 1e-12

        # #############################################
        # Fourier transform SIM images
        # #############################################
        tstart = time.process_time()

        self.imgs_ft = np.zeros((self.nangles, self.nphases, self.ny, self.nx), dtype=np.complex)
        for jj in range(self.nangles):
            for kk in range(self.nphases):
                # use periodic/smooth decomposition instead of traditional apodization
                img_to_xform, _ = psd.periodic_smooth_decomp(self.imgs[jj, kk])
                self.imgs_ft[jj, kk] = fft.fftshift(fft.fft2(fft.ifftshift(img_to_xform)))

        tend = time.process_time()

        self.print_tee("FT images took %0.2fs" % (tend - tstart), self.log_file)

        # #############################################
        # get widefield image
        # #############################################
        tstart = time.process_time()

        self.widefield = get_widefield(self.imgs)
        wf_to_xform, _ = psd.periodic_smooth_decomp(self.widefield)
        self.widefield_ft = fft.fftshift(fft.fft2(fft.ifftshift(wf_to_xform)))

        tend = time.process_time()
        self.print_tee("Computing widefield image took %0.2fs" % (tend - tstart), self.log_file)

        # #############################################
        # get optically sectioned image
        # #############################################
        tstart = time.process_time()

        sim_os = np.zeros((self.nangles, self.imgs.shape[-2], self.imgs.shape[-1]))
        for ii in range(self.nangles):
            sim_os[ii] = sim_optical_section(self.imgs[ii])
        # todo: maybe want to weight by power/mod depth?
        self.imgs_os = np.mean(sim_os, axis=0)

        tend = time.process_time()
        self.print_tee("Computing OS image took %0.2fs" % (tend - tstart), self.log_file)

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()

    def reconstruct(self):
        """
        Handle SIM reconstruction, including parameter estimation, image combination, displaying useful information.
        :param figsize:
        :return:
        """
        self.print_tee("starting reconstruction", self.log_file)

        # #############################################
        # estimate SIM parameters
        # #############################################
        if self.use_bayesian:
            raise Exception("use_bayesian=True is not currently supported")
            # todo: this method no longer working. Maybe will want to revisit it at some point
            # todo: not very happy that these parameter estimation functions return figures.
            # todo: need to update to work with new way of storing power_spectrum params
            # todo: seem to have problems when I try and run many instances in a loop?
            self.print_tee("starting bayesian parameter inference", self.log_file)

            self.frqs, self.phases, self.mod_depths, bayes_figs, bayes_fig_names = \
                self.estimate_parameters_bayesian(self.frqs_guess, figsize=self.figsize)

            # in this case, still do power spectrum fitting to get model for SNR
            self.separated_components_ft = separate_components(self.imgs_ft, self.phases)

            tstart = time.process_time()

            _, self.power_spectrum_params = self.estimate_mod_depths()

            tend = time.process_time()
            self.print_tee('fit power spectrum parameters in %0.2fs' % (self.nangles, tend - tstart), self.log_file)

        else:
            # estimate frequencies
            tstart = time.time()  # since done using joblib, process_time() is not useful...

            if self.use_fixed_frq:
                self.frqs = self.frqs_guess
                self.print_tee("using fixed frequencies", self.log_file)
            else:
                if self.find_frq_first:
                    self.frqs = self.estimate_sim_frqs(self.imgs_ft, self.imgs_ft, self.frqs_guess)
                else:
                    # todo: finish implementing
                    self.print_tee("doing phase demixing prior to frequency finding", self.log_file)
                    self.separated_components_ft = separate_components(self.imgs_ft, self.phases_guess, np.ones((self.nangles, self.nphases)))
                    imgs1 = np.expand_dims(self.separated_components_ft[:, 0], axis=1)
                    imgs2 = np.expand_dims(self.separated_components_ft[:, 1], axis=1)
                    # todo: probably need some internal changes to this fn to get everything working...
                    self.frqs = self.estimate_sim_frqs(imgs1, imgs2, self.frqs_guess)

            tend = time.time()
            self.print_tee("fitting frequencies took %0.2fs" % (tend - tstart), self.log_file)

            # estimate phases
            tstart = time.process_time()
            if self.use_fixed_phase:
                self.phases = self.phases_guess
                self.amps = np.ones((self.nangles, self.nphases))
                self.print_tee("Using fixed phases", self.log_file)
            else:
                self.phases, self.amps = self.estimate_sim_phases(self.frqs, self.phases_guess)

            tend = time.process_time()
            self.print_tee("estimated %d phases in %0.2fs" % (self.nangles * self.nphases, tend - tstart), self.log_file)

            # separate components
            self.separated_components_ft = separate_components(self.imgs_ft, self.phases, self.amps)

            # estimate modulation depths and power spectrum fit parameters
            tstart = time.process_time()
            # for the moment, need to do this feet even if have fixed mod depth, because still need the
            # power spectrum parameters
            self.mod_depths, self.power_spectrum_params, self.pspec_masks = self.estimate_mod_depths()

            if self.use_fixed_mod_depths:
                self.mod_depths = self.mod_depths_guess

            tend = time.process_time()
            self.print_tee('estimated %d modulation depths in %0.2fs' % (self.nangles, tend - tstart), self.log_file)

        # #############################################
        # estimate modulation contrast to noise ratio for raw images
        # #############################################
        mcnr = np.zeros((self.nangles, self.nphases))
        for ii in range(self.nangles):
            for jj in range(self.nphases):
                mcnr[ii, jj] = get_mcnr(self.imgs_ft[ii, jj], self.frqs[ii], self.fx, self.fy, self.fmax)

            # if mcnr is too low (typically < 1), use guess values instead
            if self.default_to_guess_on_low_mcnr and np.min(mcnr[ii]) < self.min_mcnr and self.frqs_guess is not None:
                self.frqs[ii] = self.frqs_guess[ii]
                self.print_tee("Angle %d/%d, minimum mcnr = %0.2f is less than the minimum value, %0.2f,"
                          " so fit frequency will be replaced with guess"
                          % (ii + 1, self.nangles, np.min(mcnr[ii]), self.min_mcnr), self.log_file)

                for jj in range(self.nphases):
                    mcnr[ii, jj] = get_mcnr(self.imgs_ft[ii, jj], self.frqs[ii], self.fx, self.fy, self.fmax)

        self.mcnr = mcnr
        # for convenience, also save periods and angles
        self.periods = 1 / np.sqrt(self.frqs[:, 0] ** 2 + self.frqs[:, 1] ** 2)
        self.angles = np.angle(self.frqs[:, 0] + 1j * self.frqs[:, 1])

        # #############################################
        # estimate noise in component images
        # #############################################
        self.noise_power = self.power_spectrum_params[:, :, -1]

        # #############################################
        # SIM reconstruction
        # #############################################
        tstart = time.process_time()
        self.img_sr, self.img_sr_ft, self.components_deconvolved_ft, self.components_shifted_ft, \
        self.weights, self.weight_norm, self.snr, self.snr_shifted = self.combine_components()
        # self.img_sr, self.img_sr_ft, self.components_deconvolved_ft, self.components_shifted_ft, \
        #      self.weights, self.weight_norm, self.snr, self.snr_shifted = self.combine_components_fairSIM()
        tend = time.process_time()

        self.print_tee("combining components took %0.2fs" % (tend - tstart), self.log_file)

        # #############################################
        # widefield deconvolution
        # #############################################
        # get signal to noise ratio
        wf_noise = get_noise_power(self.widefield_ft, self.fx, self.fy, self.fmax)
        fit_result, self.mask_wf = fit_power_spectrum(self.widefield_ft, self.otf, self.fx, self.fy, self.fmax, self.fbounds,
                                                 init_params=[None, self.power_spectrum_params[0, 0, 1], 1, wf_noise],
                                                 fixed_params=[False, True, True, True])

        self.pspec_params_wf = fit_result['fit_params']
        ff = np.sqrt(self.fx[None, :]**2 + self.fy[:, None]**2)
        sig = power_spectrum_fn([self.pspec_params_wf[0], self.pspec_params_wf[1], self.pspec_params_wf[2], 0], ff, 1)
        wf_snr = sig / wf_noise
        # deconvolution
        wf_decon_ft, wfilter = wiener_deconvolution(self.widefield_ft, self.otf, wf_snr, snr_includes_otf=False)

        # upsample to make fully comparable to reconstructed image
        self.widefield_deconvolution_ft = tools.expand_fourier_sp(wf_decon_ft, 2, 2, centered=True)
        self.widefield_deconvolution = fft.fftshift(fft.ifft2(fft.ifftshift(self.widefield_deconvolution_ft))).real

        # #############################################
        # print parameters
        # #############################################
        self.print_parameters()

        try:
            self.log_file.close()
        except AttributeError:
            pass

    def combine_components_fairSIM(self):
        # todo: get working

        # ensure is array with correct shape
        if self.separated_components_ft.shape[1] != 3:
            raise Exception("Expected shifted_components_ft to have shape (nangles, 3, ny, nx), where components are"
                            "O(f)*otf(f), O(f-fo)*otf(f), O(f+fo)*otf(f). But size of second dimension was not 3.")

        # upsample image before shifting
        m_exp = 2
        otf = tools.expand_fourier_sp(self.otf, mx=m_exp, my=m_exp, centered=True)
        dx = self.dx / m_exp
        ny_upsample = m_exp * self.ny
        nx_upsample = m_exp * self.nx

        # frequency data
        fx = tools.get_fft_frqs(nx_upsample, dx)
        dfx = fx[1] - fx[0]
        fy = tools.get_fft_frqs(ny_upsample, dx)
        dfy = fy[1] - fy[0]
        fxfx, fyfy = np.meshgrid(fx, fy)

        w = 0.05
        denominator_noresample = w**2
        denominator = w**2

        components_deconvolved_ft = np.zeros((self.nangles, 3, self.ny, self.nx), dtype=np.complex)
        snr = np.zeros(components_deconvolved_ft.shape)
        components_shifted_ft = np.zeros((self.nangles, 3, ny_upsample, nx_upsample), dtype=np.complex)
        snr_shifted = np.zeros(components_shifted_ft.shape)
        def ff_shift_upsample(f): return np.sqrt((fxfx - f[0]) ** 2 + (fyfy - f[1]) ** 2)
        def ff_shift(f): return np.sqrt((self.fx[None, :] - f[0]) ** 2 + (self.fy[:, None] - f[1]) ** 2)

        for ii in range(self.nangles):

            # expand then shift (no shifting required for this one)
            # have to add denominator later
            components_deconvolved_ft[ii, 0] = self.otf.conj() * self.separated_components_ft[ii, 0]
            components_shifted_ft[ii, 0] = tools.expand_fourier_sp(components_deconvolved_ft[ii, 0],
                                                                   mx=m_exp, my=m_exp, centered=True)
            snr[ii, 0] = np.abs(self.otf)**2
            denominator_noresample += np.abs(self.otf)**2

            snr_shifted[ii, 0] = np.abs(otf)**2
            denominator += np.abs(otf)**2


            # 2nd component
            otf_shifted, _, _ = tools.translate_pix(otf, self.frqs[ii], dx=dfx, dy=dfy, mode='no-wrap')
            denominator += np.abs(otf_shifted)**2

            otf_shifted_noresample, _, _ = tools.translate_pix(self.otf, self.frqs[ii], dx=dfx, dy=dfy, mode="no-wrap")
            denominator_noresample += np.abs(otf_shifted_noresample)**2

            snr[ii, 1] = np.abs(self.otf) ** 2
            snr_shifted[ii, 1] = np.abs(otf_shifted) ** 2

            # shift and expand
            components_deconvolved_ft[ii, 1] = self.otf.conj() * self.separated_components_ft[ii, 1] / self.mod_depths[ii]

            mask = ff_shift_upsample(-self.frqs[ii]) <= self.fmax

            components_shifted_ft[ii, 1] = tools.translate_ft(
                tools.expand_fourier_sp(components_deconvolved_ft[ii, 1], mx=m_exp, my=m_exp, centered=True),
                self.frqs[ii], dx) * mask

            # 3rd component
            otf_shifted, _, _ = tools.translate_pix(otf, -self.frqs[ii], dx=dfx, dy=dfy, mode='no-wrap')
            denominator += np.abs(otf_shifted)**2

            otf_shifted_noresample, _, _ = tools.translate_pix(self.otf, -self.frqs[ii], dx=dfx, dy=dfy, mode="no-wrap")
            denominator_noresample += np.abs(otf_shifted_noresample)**2

            snr[ii, 2] = np.abs(self.otf) ** 2
            snr_shifted[ii, 2] = np.abs(otf_shifted) ** 2

            # mask portions we know are zero
            mask = ff_shift_upsample(self.frqs[ii]) <= self.fmax

            # deconvolve
            components_deconvolved_ft[ii, 2] = self.otf.conj() * self.separated_components_ft[ii, 2] / self.mod_depths[ii]

            # shift and expand
            components_shifted_ft[ii, 2] = tools.translate_ft(
                tools.expand_fourier_sp(components_deconvolved_ft[ii, 2], mx=m_exp, my=m_exp, centered=True),
                -self.frqs[ii], dx) * mask

        components_deconvolved_ft = components_deconvolved_ft / denominator_noresample
        components_shifted_ft = components_shifted_ft / denominator

        # correct for wrong global phases (on shifted components before weighting,
        # but then apply to weighted components)
        # todo: correct global phase correction problem...
        if self.global_phase_correction:
            self.phase_corrections = global_phase_correction(components_shifted_ft)
        else:
            self.phase_corrections = np.zeros(self.nangles)

        for ii in range(self.nangles):
            components_deconvolved_ft[ii, 1] = np.exp(1j * self.phase_corrections[ii]) * components_deconvolved_ft[ii, 1]
            components_deconvolved_ft[ii, 2] = np.exp(-1j * self.phase_corrections[ii]) * components_deconvolved_ft[ii, 2]

            components_shifted_ft[ii, 1] = np.exp(1j * self.phase_corrections[ii]) * components_shifted_ft[ii, 1]
            components_shifted_ft[ii, 2] = np.exp(-1j * self.phase_corrections[ii]) * components_shifted_ft[ii, 2]

        sim_sr_ft = np.nansum(components_shifted_ft, axis=(0, 1))

        # Fourier transform back to get real-space reconstructed image, use Tukey apodization filter
        apod = scipy.signal.windows.tukey(sim_sr_ft.shape[1], alpha=0.1)[None, :] * \
               scipy.signal.windows.tukey(sim_sr_ft.shape[0], alpha=0.1)[:, None]

        sim_sr = fft.fftshift(fft.ifft2(fft.ifftshift(sim_sr_ft * apod))).real

        #
        weights = np.zeros((self.nangles, self.nphases, ny_upsample, nx_upsample))
        weights_norm = 1
        for ii in range(self.nangles):
            for jj in range(self.nphases):
                weights[ii, jj] = denominator

        return sim_sr, sim_sr_ft, components_deconvolved_ft, components_shifted_ft, weights, weights_norm, snr, snr_shifted

    def combine_components(self):
        """
        Combine components O(f)otf(f), O(f-fo)otf(f), and O(f+fo)otf(f) to form SIM reconstruction.
        :return:
        """

        # ensure is array with correct shape
        if self.separated_components_ft.shape[1] != 3:
            raise Exception("Expected shifted_components_ft to have shape (nangles, 3, ny, nx), where components are"
                            "O(f)*otf(f), O(f-fo)*otf(f), O(f+fo)*otf(f). But size of second dimension was not 3.")

        # upsample image before shifting
        m_exp = 2
        otf = tools.expand_fourier_sp(self.otf, mx=m_exp, my=m_exp, centered=True)
        dx = self.dx / m_exp
        ny_upsample = m_exp * self.ny
        nx_upsample = m_exp * self.nx

        # frequency data
        fx = tools.get_fft_frqs(nx_upsample, dx)
        dfx = fx[1] - fx[0]
        fy = tools.get_fft_frqs(ny_upsample, dx)
        dfy = fy[1] - fy[0]
        fxfx, fyfy = np.meshgrid(fx, fy)

        # to store different params
        # wiener filtering to divide by H(k)
        components_deconvolved_ft = np.zeros((self.nangles, 3, self.ny, self.nx), dtype=np.complex)
        snr = np.zeros(components_deconvolved_ft.shape)
        # shift to correct place in frq space
        components_shifted_ft = np.zeros((self.nangles, 3, ny_upsample, nx_upsample), dtype=np.complex)
        snr_shifted = np.zeros(components_shifted_ft.shape)
        # weight and average
        components_weighted = np.zeros(components_shifted_ft.shape, dtype=np.complex)
        weights = np.zeros(components_weighted.shape)

        def ff_shift(f): return np.sqrt((self.fx[None, :] - f[0]) ** 2 + (self.fy[:, None] - f[1]) ** 2)
        def ff_shift_upsample(f): return np.sqrt((fxfx - f[0]) ** 2 + (fyfy - f[1]) ** 2)
        ff_upsample = np.sqrt(fxfx**2 + fyfy**2)

        # shift and filter components
        for ii in range(self.nangles):
            # ###########################
            # O(f)H(f)
            # ###########################
            # signal-to-noise ratio and weights
            params = list(self.power_spectrum_params[ii, 0, :-1]) + [0]
            snr[ii, 0] = power_spectrum_fn(params, ff_shift([0, 0]), 1) / self.noise_power[ii, 0]
            snr_shifted[ii, 0] = power_spectrum_fn(params, ff_upsample, 1) / self.noise_power[ii, 0]
            weights[ii, 0] = generalized_wiener_filter(otf, snr_shifted[ii, 0])

            # deconvolve
            components_deconvolved_ft[ii, 0], _ = wiener_deconvolution(self.separated_components_ft[ii, 0], self.otf, snr[ii, 0])
            # expand then shift (no shifting required for this one)
            components_shifted_ft[ii, 0] = tools.expand_fourier_sp(components_deconvolved_ft[ii, 0],
                                                                   mx=m_exp, my=m_exp, centered=True)

            if self.size_near_fo_to_remove != 0:
                to_remove = np.abs(ff_shift_upsample(-self.frqs[ii])) < self.size_near_fo_to_remove * np.linalg.norm(self.frqs[ii])
                components_shifted_ft[ii, 0][to_remove] = 0

                to_remove = np.abs(ff_shift_upsample(self.frqs[ii])) < self.size_near_fo_to_remove * np.linalg.norm(self.frqs[ii])
                components_shifted_ft[ii, 0][to_remove] = 0

            # ###########################
            # m*O(f - f_o)S(f)
            # ###########################
            # signal-to-noise ratio and weights
            params = list(self.power_spectrum_params[ii, 1, :-1]) + [0]
            snr[ii, 1] = power_spectrum_fn(params, ff_shift(self.frqs[ii]), 1) / self.noise_power[ii, 1]

            # get shifted SNR
            # pix_shift = [-int(np.round(self.frqs[ii, 0] / dfx)), -int(np.round(self.frqs[ii, 1] / dfy))]
            otf_shifted, _, _ = tools.translate_pix(otf, self.frqs[ii], dx=dfx, dy=dfy, mode='no-wrap')

            snr_shifted[ii, 1] = power_spectrum_fn(params, ff_upsample, 1) / self.noise_power[ii, 1]
            weights[ii, 1] = generalized_wiener_filter(otf_shifted, snr_shifted[ii, 1])

            # mask off regions where know FT should be zero
            mask = ff_shift_upsample(-self.frqs[ii]) <= self.fmax

            # deconvolve
            components_deconvolved_ft[ii, 1], _ = \
                wiener_deconvolution(self.separated_components_ft[ii, 1] / self.mod_depths[ii], self.otf, snr[ii, 1])

            # shift and expand
            components_shifted_ft[ii, 1] = tools.translate_ft(
                tools.expand_fourier_sp(components_deconvolved_ft[ii, 1], mx=m_exp, my=m_exp, centered=True),
                self.frqs[ii], dx) * mask

            if self.size_near_fo_to_remove != 0:
                to_remove = np.abs(ff_shift_upsample(-self.frqs[ii])) < self.size_near_fo_to_remove * np.linalg.norm(self.frqs[ii])
                components_shifted_ft[ii, 1][to_remove] = 0

            # ###########################
            # m*O(f + f_o)S(f)
            # ###########################
            # signal-to-noise ratio and weights
            params = list(self.power_spectrum_params[ii, 2, :-1]) + [0]
            snr[ii, 2] = power_spectrum_fn(params, ff_shift(-self.frqs[ii]), 1) / self.noise_power[ii, 2]

            # get shifted SNR
            # pix_shift = [-int(np.round(-self.frqs[ii, 0] / dfx)), -int(np.round(-self.frqs[ii, 1] / dfy))]
            otf_shifted, _, _ = tools.translate_pix(otf, -self.frqs[ii], dx=dfx, dy=dfy, mode='no-wrap')

            snr_shifted[ii, 2] = power_spectrum_fn(params, ff_upsample, 1) / self.noise_power[ii, 2]
            weights[ii, 2] = generalized_wiener_filter(otf_shifted, snr_shifted[ii, 2])

            # mask portions we know are zero
            mask = ff_shift_upsample(self.frqs[ii]) <= self.fmax

            # deconvolve
            components_deconvolved_ft[ii, 2], _ = \
                wiener_deconvolution(self.separated_components_ft[ii, 2] / self.mod_depths[ii], self.otf, snr[ii, 2])

            # shift and expand
            components_shifted_ft[ii, 2] = tools.translate_ft(
                                           tools.expand_fourier_sp(components_deconvolved_ft[ii, 2], mx=m_exp, my=m_exp, centered=True),
                                           -self.frqs[ii], dx) * mask

            if self.size_near_fo_to_remove != 0:
                to_remove = np.abs(ff_shift_upsample(self.frqs[ii])) < self.size_near_fo_to_remove * np.linalg.norm(self.frqs[ii])
                components_shifted_ft[ii, 2][to_remove] = 0

        # correct for wrong global phases (on shifted components before weighting,
        # but then apply to weighted components)
        if self.global_phase_correction:
            self.phase_corrections = global_phase_correction(components_shifted_ft)
        else:
            self.phase_corrections = np.zeros(self.nangles)

        # combine components
        components_weighted = components_shifted_ft * weights
        for ii in range(self.nangles):
            components_weighted[ii, 1] = np.exp(1j * self.phase_corrections[ii]) * components_weighted[ii, 1]
            components_weighted[ii, 2] = np.exp(-1j * self.phase_corrections[ii]) * components_weighted[ii, 2]

        # final averaging
        weight_norm = np.sum(weights, axis=(0, 1)) + self.wiener_parameter
        sim_sr_ft = np.nansum(components_weighted, axis=(0, 1)) / weight_norm

        # Fourier transform back to get real-space reconstructed image
        apod = scipy.signal.windows.tukey(sim_sr_ft.shape[1], alpha=0.1)[None, :] * \
               scipy.signal.windows.tukey(sim_sr_ft.shape[0], alpha=0.1)[:, None]

        sim_sr = fft.fftshift(fft.ifft2(fft.ifftshift(sim_sr_ft * apod))).real

        return sim_sr, sim_sr_ft, components_deconvolved_ft, components_shifted_ft,\
               weights, weight_norm, snr, snr_shifted

    def estimate_sim_frqs(self, fts1, fts2, frq_guess):
        """
        estimate SIM frequency

        :param frq_guess:
        :return frqs:
        :return phases:
        :return peak_heights_relative: Height of the fit peak relative to the DC peak.
        """
        nangles = fts1.shape[0]
        nphases = fts1.shape[1]

        # todo: maybe should take some average/combination of the widefield images to try and improve signal
        # e.g. could multiply each by expected phase values?
        results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
            joblib.delayed(fit_modulation_frq)(
                fts1[ii, 0], fts2[ii, 0], self.dx, self.fmax, frq_guess=frq_guess[ii])
            for ii in range(nangles)
        )

        frqs, _ = zip(*results)
        frqs = np.reshape(np.asarray(frqs), [nangles, 2])

        # estimate for each phase individually
        # results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
        #     joblib.delayed(fit_modulation_frq)(
        #         fts1[ii, jj], fts2[ii, jj], {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
        #         frq_guess=frq_guess[ii])
        #     for ii in range(nangles) for jj in range(nphases)
        # )

        # frqs, _ = zip(*results)
        # frqs = np.reshape(np.asarray(frqs), [nangles, nphases, 2])

        # frqs = np.zeros((nangles, nphases, 2))
        # for ii in range(self.nangles):
        #     for jj in range(self.nphases):
        #         frqs[ii, jj, :], mask = \
        #              fit_modulation_frq(fts1[ii, jj], fts2[ii, jj], {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
        #                                 frq_guess=frq_guess[ii])

        # get frequency and angle estimates
        # frqs = np.mean(frqs, axis=1)

        return frqs

    def estimate_sim_phases(self, frqs, phase_guess=None):
        """
        estimate phases for all SIM images
        :param frqs:
        :param phase_guess:
        :return phases:
        """

        phases = np.zeros((self.nangles, self.nphases))
        amps = np.zeros((self.nangles, self.nphases))
        # mods = np.zeros((self.nangles, self.nphases))

        if self.use_wicker:
            if phase_guess is None:
                phase_guess = [None] * self.nangles

            for ii in range(self.nangles):
                phases[ii], amps[ii] = fit_phase_wicker(self.imgs_ft[ii], self.otf, frqs[ii], self.dx, self.fmax,
                                                        phases_guess=phase_guess[ii], fit_amps=self.determine_amplitudes)
        else:
            if phase_guess is None:
                phase_guess = np.zeros((self.nangles, self.nphases))

            for ii in range(self.nangles):
                for jj in range(self.nphases):
                    phases[ii, jj] = fit_phase_realspace(self.imgs[ii, jj], frqs[ii], self.dx,
                                                         phase_guess=phase_guess[ii, jj], origin="center")
                    amps[ii, jj] = 1

        if (phase_guess is not None) and self.default_to_guess_on_bad_phase_fit:
            phase_guess_diffs = np.mod(phase_guess - phase_guess[:, 0][:, None], 2*np.pi)
            phase_diffs = np.mod(phases - phases[:, 0][:, None], 2*np.pi)

            for ii in range(self.nangles):
                diffs = np.mod(phase_guess_diffs[ii] - phase_diffs[ii], 2 * np.pi)
                condition = np.abs(diffs - 2 * np.pi) < diffs
                diffs[condition] = diffs[condition] - 2 * np.pi

                if np.any(np.abs(diffs) > self.max_phase_error):
                    phases[ii] = phase_guess[ii]
                    str = "Angle %d/%d, phase guesses are more than the maximum allowed phase error=%0.2fdeg." \
                          " Defaulting to guess values" % (ii+1, self.nangles, self.max_phase_error * 180/np.pi)

                    str += "\nfit phase diffs="
                    for jj in range(self.nphases):
                        str += "%0.2fdeg, " % (phase_diffs[ii, jj] * 180/np.pi)
                    # str += "\nguesses="
                    # for jj in range(self.nphases):
                    #     str += "%0.2fdeg, " % (phase_guess_diffs[ii, jj] * 180/np.pi)

                    # warnings.warn(str)
                    self.print_tee(str, self.log_file)

        return phases, amps

    def estimate_mod_depths(self):

        # first average power spectrum of \sum_{angles} O(f)H(f) to get exponent
        component_zero = np.nanmean(self.separated_components_ft[:, 0], axis=0)
        noise = get_noise_power(component_zero, self.fx, self.fy, self.fmax)

        fit_results_avg, mask_avg = fit_power_spectrum(component_zero, self.otf, self.fx, self.fy, self.fmax, self.fbounds,
                                               init_params=[None, None, 1, noise],
                                               fixed_params=[False, False, True, True])
        fit_params_avg = fit_results_avg['fit_params']
        exponent = fit_params_avg[1]

        # now fit each other component using this same exponent
        power_spectrum_params = np.zeros((self.nangles, self.nangles, 4))
        masks = np.zeros(self.separated_components_ft.shape, dtype=np.bool)
        for ii in range(self.nangles):
            for jj in range(self.nphases):
                noise = get_noise_power(self.separated_components_ft[ii, jj], self.fx, self.fy, self.fmax)

                if jj == 0:
                    # for unshifted components, fit the amplitude
                    init_params = [None, exponent, 1, noise]
                    fixed_params = [False, True, True, True]

                    fit_results, masks[ii, jj] = fit_power_spectrum(self.separated_components_ft[ii, jj], self.otf,
                                                           self.fx, self.fy, self.fmax, self.fbounds,
                                                           init_params=init_params, fixed_params=fixed_params)
                    power_spectrum_params[ii, jj] = fit_results['fit_params']
                elif jj == 1:
                    # for shifted components, fit the modulation factor
                    init_params = [power_spectrum_params[ii, 0, 0], exponent, 0.5, noise]
                    fixed_params = [True, True, False, True]

                    fit_results, masks[ii, jj] = fit_power_spectrum(self.separated_components_ft[ii, jj], self.otf,
                                                           self.fx, self.fy, self.fmax, (0, 1), self.fbounds_shift,
                                                           self.frqs[ii],
                                                           init_params=init_params, fixed_params=fixed_params)
                    power_spectrum_params[ii, jj] = fit_results['fit_params']
                elif jj == 2:
                    power_spectrum_params[ii, jj] = power_spectrum_params[ii, 1]
                else:
                    raise Exception("not implemented for nphases > 3")

        # extract mod depths from the structure
        mod_depths = power_spectrum_params[:, 1, 2]

        return mod_depths, power_spectrum_params, masks

    def estimate_parameters_bayesian(self, frq_sim_guess, figsize=(20, 10)):
        """
        Estimate frequencies, phases, and modulation depths from SIM images. This function utilizes the many
        estimation function (get_sim_frq, get_sim_phase, etc.) to estimate all these parameters.

        :param frq_sim_guess: 2 x nangles array of guess SIM frequency values
        :return:
        """

        debug_figs = []
        debug_fig_names = []

        frqs_sim = np.zeros((self.nangles, 2))
        phases = np.zeros((self.nangles, self.nphases))
        ms = np.zeros((self.nangles, self.nphases))
        for ii in range(self.nangles):
            for jj in range(self.nphases):

                # set up model fn
                if jj == 0:
                    frqs_sim[ii], phases[ii, jj], ms[ii, jj], I_inferred, trace = \
                        get_sim_params_mcmc(self.imgs_ft[ii, jj], self.otf, frq_sim_guess[ii], self.fx, self.fy, self.fmax, fit_shifts=True)
                else:
                    _, phases[ii, jj], ms[ii, jj], I_inferred, trace = \
                        get_sim_params_mcmc(self.imgs_ft[ii, jj], self.otf, frqs_sim[ii], self.fx, self.fy, self.fmax, fit_shifts=False)

                if self.plot_diagnostics and ii == 0:
                    figh = plot_correlation_fit(self.imgs_ft[ii, jj], self.imgs_ft[ii, jj], frqs_sim[ii],
                                                self.dx, self.fmax,
                                                frqs_guess=self.frqs_guess[ii], figsize=figsize,
                                                ttl_str="Correlation fit, angle = %d, phase=0" % ii)

                    debug_figs += [figh]
                    debug_fig_names += ['frq_fit_angle=%d' % (ii + 1)]

                if self.plot_diagnostics:
                    pm.traceplot(trace)
                    figh = plt.gcf()

                    debug_figs += [figh]
                    debug_fig_names += ['parameter_inference_angle=%d_phase=%d' % (ii + 1, jj + 1)]

        mod_depths = np.mean(ms, axis=1)

        return frqs_sim, phases, mod_depths, debug_figs, debug_fig_names

    # printing utility functions
    def print_parameters(self):

        self.print_tee("SIM reconstruction for %d angles and %d phases" % (self.nangles, self.nphases), self.log_file)
        self.print_tee("images are size %dx%d with pixel size %0.3fum" % (self.ny, self.nx, self.dx), self.log_file)
        self.print_tee("emission wavelength=%.0fnm and NA=%0.2f" % (self.wavelength * 1e3, self.na), self.log_file)

        for ii in range(self.nangles):
            self.print_tee("################ Angle %d/%d ################"
                      % (ii + 1, self.nangles), self.log_file)
            angle_guess = np.angle(self.frqs_guess[ii, 0] + 1j * self.frqs_guess[ii, 1])
            period_guess = 1 / np.linalg.norm(self.frqs_guess[ii])

            # frequency and period data
            self.print_tee("Frequency guess=({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                self.frqs_guess[ii, 0], self.frqs_guess[ii, 1], period_guess * 1e3, angle_guess * 180 / np.pi,
                                                                2 * np.pi), self.log_file)
            self.print_tee("Frequency fit  =({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                self.frqs[ii, 0], self.frqs[ii, 1], self.periods[ii] * 1e3, self.angles[ii] * 180 / np.pi),
                self.log_file)

            # modulation depth
            self.print_tee("modulation depth=%0.3f" % self.mod_depths[ii], self.log_file)
            self.print_tee("minimum mcnr=%0.3f" % np.min(self.mcnr[ii]), self.log_file)

            # phase information
            self.print_tee("phases  =", self.log_file, end="")
            for jj in range(self.nphases - 1):
                self.print_tee("%07.3fdeg, " % (self.phases[ii, jj] * 180 / np.pi), self.log_file, end="")
            self.print_tee("%07.3fdeg" % (self.phases[ii, self.nphases - 1] * 180 / np.pi), self.log_file)

            if self.phases_guess is not None:
                self.print_tee("guesses =", self.log_file, end="")
                for jj in range(self.nphases - 1):
                    self.print_tee("%07.3fdeg, " % (self.phases_guess[ii, jj] * 180 / np.pi), self.log_file, end="")
                self.print_tee("%07.3fdeg" % (self.phases_guess[ii, self.nphases - 1] * 180 / np.pi), self.log_file)

            self.print_tee("dphases =", self.log_file, end="")
            for jj in range(self.nphases - 1):
                self.print_tee("%07.3fdeg, " % (np.mod(self.phases[ii, jj] - self.phases[ii, 0], 2 * np.pi) * 180 / np.pi),
                          self.log_file, end="")
            self.print_tee("%07.3fdeg" % (np.mod(self.phases[ii, self.nphases - 1] - self.phases[ii, 0], 2 * np.pi) * 180 / np.pi),
                      self.log_file)

            if self.phases_guess is not None:
                self.print_tee("dguesses=", self.log_file, end="")
                for jj in range(self.nphases - 1):
                    self.print_tee("%07.3fdeg, " % (
                                np.mod(self.phases_guess[ii, jj] - self.phases_guess[ii, 0], 2 * np.pi) * 180 / np.pi),
                              self.log_file, end="")
                self.print_tee("%07.3fdeg" % (np.mod(self.phases_guess[ii, self.nphases - 1] - self.phases_guess[ii, 0],
                                                2 * np.pi) * 180 / np.pi), self.log_file)

            # phase corrections
            self.print_tee("global phase correction=%0.2fdeg" % (self.phase_corrections[ii] * 180 / np.pi), self.log_file)

            self.print_tee("amps =", self.log_file, end="")
            for jj in range(self.nphases - 1):
                self.print_tee("%05.3f, " % (self.amps[ii, jj]), self.log_file, end="")
            self.print_tee("%05.3f" % (self.amps[ii, self.nphases - 1]), self.log_file)

    def print_tee(self, string, fid, end="\n"):
        """
        Print result to stdout and to a log file.

        :param string:
        :param fid: file handle. If fid=None, then will be ignored
        :param end:
        :return:
        """

        print(string, end=end)

        if fid is not None:
            print(string, end=end, file=fid)

    # plotting utility functions
    def plot_figs(self):
        """
        Automate plotting and saving of figures
        :return:
        """
        tstart = time.process_time()

        saving = self.save_dir is not None

        # todo: populate these
        figs = []
        fig_names = []

        # plot images
        figh = self.plot_sim_imgs(self.frqs_guess, figsize=self.figsize)

        if saving:
            figh.savefig(os.path.join(self.save_dir, "raw_images.png"))
        if not self.hold_figs_open:
            plt.close(figh)

        # plot frequency fits
        fighs, fig_names = self.plot_frequency_fits(figsize=self.figsize)
        for fh, fn in zip(fighs, fig_names):
            if saving:
                fh.savefig(os.path.join(self.save_dir, "%s.png" % fn))
            if not self.hold_figs_open:
                plt.close(fh)

        # plot power spectrum fits
        fighs, fig_names = self.plot_power_spectrum_fits(figsize=self.figsize)
        for fh, fn in zip(fighs, fig_names):
            if saving:
                fh.savefig(os.path.join(self.save_dir, "%s.png" % fn))
            if not self.hold_figs_open:
                plt.close(fh)

        # widefield power spectrum fit
        figh = plot_power_spectrum_fit(self.widefield_ft, self.otf,
                                       {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                       self.pspec_params_wf, mask=self.mask_wf, figsize=self.figsize,
                                       ttl_str="Widefield power spectrum")
        if saving:
            figh.savefig(os.path.join(self.save_dir, "power_spectrum_widefield.png"))
        if not self.hold_figs_open:
            plt.close(figh)

        # plot filters used in reconstruction
        fighs, fig_names = self.plot_reconstruction_diagnostics(figsize=self.figsize)
        for fh, fn in zip(fighs, fig_names):
            if saving:
                fh.savefig(os.path.join(self.save_dir, "%s.png" % fn))
            if not self.hold_figs_open:
                plt.close(fh)

        # plot reconstruction results
        fig = self.plot_reconstruction(figsize=self.figsize)
        if saving:
            fig.savefig(os.path.join(self.save_dir, "sim_reconstruction.png"), dpi=400)
        if not self.hold_figs_open:
            plt.close(fig)

        # plot otf
        fig = self.plot_otf(figsize=self.figsize)
        if saving:
            fig.savefig(os.path.join(self.save_dir, "otf.png"))
        if not self.hold_figs_open:
            plt.close(fig)

        tend = time.process_time()
        # print_tee("plotting results took %0.2fs" % (tend - tstart), self.log_file)
        print("plotting results took %0.2fs" % (tend - tstart))

        return figs, fig_names

    def plot_sim_imgs(self, frqs_sim_guess=None, figsize=(20, 10)):
        """
        Display SIM images for visual inspection

        Use this to examine SIM pictures and their fourier transforms before doing reconstruction.

        :param frqs_sim_guess: if provided, plot SIM frequency or guess
        :return:
        """

        # real space coordinate data
        x = self.dx * (np.arange(self.nx) - self.nx / 2)
        y = self.dy * (np.arange(self.ny) - self.ny / 2)

        extent = tools.get_extent(y, x)

        # frequency coordinate data
        dfx = self.fx[1] - self.fx[0]
        dfy = self.fy[1] - self.fy[0]

        extent_ft = tools.get_extent(self.fx, self.fy)

        # plot FT of sim images
        figh = plt.figure(figsize=figsize)
        plt.suptitle('SIM images, real space and power spectra')
        grid = plt.GridSpec(self.nphases, self.nangles*2)

        # parameters for ft plot
        gamma = 0.1 # gamma for PowerNorm plot of power spectra

        # parameters for real space plot
        vmin = np.percentile(self.imgs.ravel(), 0.1)
        vmax = np.percentile(self.imgs.ravel(), 99.9)
        mean_int = np.mean(self.imgs, axis=(1, 2, 3))
        rel_int = mean_int / np.max(mean_int)

        for ii in range(self.nangles):
            for jj in range(self.nphases):

                # set real space image
                ax = plt.subplot(grid[jj, 2 * ii])
                ax.imshow(self.imgs[ii, jj], vmin=vmin, vmax=vmax, extent=extent, interpolation=None)

                if jj == 0:
                    plt.title('angle %d, relative intensity=%0.3f' % (ii, rel_int[ii]))
                if ii == 0:
                    plt.ylabel("Position (um)")
                if jj == (self.nphases - 1):
                    plt.xlabel("Position (um)")

                # plot power spectra
                ax = plt.subplot(grid[jj, 2*ii + 1])

                ax.imshow(np.abs(self.imgs_ft[ii, jj]) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_ft)
                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                if frqs_sim_guess is not None:
                    circ2 = matplotlib.patches.Circle((frqs_sim_guess[ii, 0], frqs_sim_guess[ii, 1]), radius=20 * dfx,
                                                      color='k', fill=0, ls='-')
                    ax.add_artist(circ2)
                    circ3 = matplotlib.patches.Circle((-frqs_sim_guess[ii, 0], -frqs_sim_guess[ii, 1]), radius=20 * dfx,
                                                      color='k', fill=0, ls='-')
                    ax.add_artist(circ3)

                    angle = np.angle(frqs_sim_guess[ii, 0] + 1j * frqs_sim_guess[ii, 1])
                    period = 1 / np.sqrt(frqs_sim_guess[ii, 0] ** 2 + frqs_sim_guess[ii, 1] ** 2)

                    if jj == 0:
                        plt.title('%0.3fdeg, %0.3fnm' % (angle * 180 / np.pi, period))

                ax.set_xlim([-self.fmax, self.fmax])
                ax.set_ylim([self.fmax, -self.fmax])

                if jj == (self.nphases - 1):
                    plt.xlabel("Frq (1/um)")

        return figh

    def plot_reconstruction(self, figsize=(20, 10)):
        """
        Plot SIM image and compare with 'widefield' image
        :return:
        """

        extent_wf = tools.get_extent(self.fy, self.fx)

        # for reconstructed image, must check if has been resampled
        fx = tools.get_fft_frqs(self.img_sr.shape[1], 0.5 * self.dx)
        fy = tools.get_fft_frqs(self.img_sr.shape[0], 0.5 * self.dx)

        extent_rec = tools.get_extent(fy, fx)

        gamma = 0.1
        min_percentile = 0.1
        max_percentile = 99.9

        # create plot
        figh = plt.figure(figsize=figsize)
        grid = plt.GridSpec(2, 3)
        # todo: print more reconstruction information here
        plt.suptitle("SIM reconstruction, w=%0.2f, wicker=%d" %
                     (self.wiener_parameter, self.use_wicker))

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

        vmin = np.percentile(self.img_sr.ravel(), min_percentile)
        vmax = np.percentile(self.img_sr.ravel(), max_percentile)
        plt.imshow(self.img_sr, vmin=vmin, vmax=vmax)
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
        plt.imshow(np.abs(self.img_sr_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_rec)
        circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ)

        circ2 = matplotlib.patches.Circle((0, 0), radius=2 * self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ2)

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

        coord_os, cut_os = tools.get_cut_profile(self.imgs_os, start_coord, end_coord, 1)
        coord_os = self.dx * coord_os

        coord_dc, cut_dc = tools.get_cut_profile(self.widefield_deconvolution, start_coord_re, end_coord_re, 1)
        coord_dc = 0.5 * self.dx * coord_dc

        coord_sr, cut_sr = tools.get_cut_profile(self.img_sr, start_coord_re, end_coord_re, 1)
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

        # check if image has been upsampled
        fx = tools.get_fft_frqs(2 * self.nx, 0.5 * self.dx)
        fy = tools.get_fft_frqs(2 * self.ny, 0.5 * self.dx)

        # plot different stages of inversion
        extent = tools.get_extent(self.fy, self.fx)
        extent_upsampled = tools.get_extent(fy, fx)

        for ii in range(self.nangles):
            fig = plt.figure(figsize=figsize)
            grid = plt.GridSpec(3, 5)

            for jj in range(self.nphases):

                # ####################
                # separated components
                # ####################
                ax = plt.subplot(grid[jj, 0])

                plt.imshow(np.abs(self.separated_components_ft[ii, jj]), norm=LogNorm(), extent=extent)

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

                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

                # ####################
                # deconvolved component
                # ####################
                ax = plt.subplot(grid[jj, 1])

                plt.imshow(np.abs(self.components_deconvolved_ft[ii, jj]), norm=LogNorm(), extent=extent)

                if jj == 0:
                    plt.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='r', facecolor='none')
                    plt.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='r', facecolor='none')
                elif jj == 1:
                    plt.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='r', facecolor='none')
                elif jj == 2:
                    plt.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='r', facecolor='none')

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                if jj == 0:
                    plt.title('deconvolved component')
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

                # ####################
                # shifted component
                # ####################
                ax = plt.subplot(grid[jj, 2])

                plt.imshow(np.abs(self.components_shifted_ft[ii, jj]), norm=LogNorm(), extent=extent_upsampled)
                plt.scatter(0, 0, edgecolor='r', facecolor='none')

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                if jj == 0:
                    plt.title('shifted component')
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

                # ####################
                # SNR
                # ####################
                ax = plt.subplot(grid[jj, 3])
                im0 = plt.imshow(self.snr_shifted[ii, jj], norm=LogNorm(), extent=extent_upsampled)

                fig.colorbar(im0)

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

                if jj == 0:
                    plt.title('SNR')
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

                # ####################
                # normalized weights
                # ####################
                ax = plt.subplot(grid[jj, 4])

                im2 = plt.imshow(self.weights[ii, jj] / self.weight_norm, norm=LogNorm(), extent=extent_upsampled)
                im2.set_clim([1e-5, 1])
                fig.colorbar(im2)

                circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
                ax.add_artist(circ)

                if jj == 0:
                    plt.title('normalized weight')
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

                plt.xlim([-2 * self.fmax, 2 * self.fmax])
                plt.ylim([2 * self.fmax, -2 * self.fmax])

            plt.suptitle('period=%0.3fnm at %0.3fdeg=%0.3frad, f=(%0.3f,%0.3f) 1/um\n'
                         'mod=%0.3f, min mcnr=%0.3f, wiener param=%0.2f\n'
                         'phases (deg) =%0.2f, %0.2f, %0.2f, phase diffs (deg) =%0.2f, %0.2f, %0.2f' %
                         (self.periods[ii] * 1e3, self.angles[ii] * 180 / np.pi, self.angles[ii],
                          self.frqs[ii, 0], self.frqs[ii, 1], self.mod_depths[ii], np.min(self.mcnr[ii]), self.wiener_parameter,
                          self.phases[ii, 0] * 180/np.pi, self.phases[ii, 1] * 180/np.pi, self.phases[ii, 2] * 180/np.pi,
                          0, np.mod(self.phases[ii, 1] - self.phases[ii, 0], 2*np.pi) * 180/np.pi,
                          np.mod(self.phases[ii, 2] - self.phases[ii, 0], 2*np.pi) * 180/np.pi))

            figs.append(fig)
            fig_names.append('sim_combining_angle=%d' % (ii + 1))

        # #######################
        # net weight
        # #######################
        figh = plt.figure(figsize=figsize)
        grid = plt.GridSpec(1, 2)
        plt.suptitle('Net weight, Wiener param = %0.2f' % self.wiener_parameter)

        ax = plt.subplot(grid[0, 0])
        net_weight = np.sum(self.weights, axis=(0, 1)) / self.weight_norm
        im = ax.imshow(net_weight, extent=extent_upsampled, norm=PowerNorm(gamma=0.1))

        figh.colorbar(im)
        ax.set_title("non-linear scale")
        circ = matplotlib.patches.Circle((0, 0), radius=self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ)

        circ2 = matplotlib.patches.Circle((0, 0), radius=2*self.fmax, color='k', fill=0, ls='--')
        ax.add_artist(circ2)

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
            fig = plot_power_spectrum_fit(self.separated_components_ft[ii, 0], self.otf,
                                          {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                          self.power_spectrum_params[ii, 0], frq_sim=(0, 0), mask=self.pspec_masks[ii, 0],
                                          figsize=figsize, ttl_str="Unshifted component, angle %d" % ii)
            debug_figs.append(fig)
            debug_fig_names.append("power_spectrum_unshifted_component_angle=%d" % ii)

            fig = plot_power_spectrum_fit(self.separated_components_ft[ii, 1], self.otf,
                                          {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                          self.power_spectrum_params[ii, 1], frq_sim=self.frqs[ii], mask=self.pspec_masks[ii, 1],
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
        for ii in range(self.nangles):

            if self.find_frq_first:
                figh = plot_correlation_fit(self.imgs_ft[ii, 0], self.imgs_ft[ii, 0], self.frqs[ii, :],
                                            self.dx, self.fmax,
                                            frqs_guess=self.frqs_guess[ii], figsize=figsize,
                                            ttl_str="Correlation fit, angle %d" % ii)
                figs.append(figh)
                fig_names.append("frq_fit_angle=%d_phase=%d" % (ii, 0))
            else:
                figh = plot_correlation_fit(self.separated_components_ft[ii, 0],
                                            self.separated_components_ft[ii, 1], self.frqs[ii, :],
                                            self.dx, self.fmax, frqs_guess=self.frqs_guess[ii], figsize=figsize,
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

        plt.suptitle("OTF")

        return figh

    # saving utility functions
    def save_result(self, fname):
        """
        Save non-image fields in pickle format.

        :param fname: file path to save results
        :return results_dict: the dictionary that has been saved
        """
        fields_to_not_save = ['imgs', 'imgs_ft',
                              'widefield', 'widefield_ft',
                              'separated_components_ft',
                              'widefield_deconvolution', 'widefield_deconvolution_ft',
                              'imgs_os',
                              'weights', 'weights_norm',
                              'deconvolved_components',
                              'components_deconvolved_ft',
                              'components_shifted_ft',
                              'snr', 'snr_shifted', 'weight_norm',
                              'img_sr', 'img_sr_ft', 'fxfx', 'fyfy', 'log_file',
                              'pspec_masks']
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

        return results_dict

# compute widefield image
def get_widefield(imgs):
    """
    Get approximate widefield image
    :param imgs: SIM angle and phase images, of shape (nangles, nphases, ny, nx)
    :return img_wf: widefield image of shapes (ny, nx)
    """
    img_wf = np.nanmean(imgs, axis=(0, 1))
    return img_wf

# compute optical sectioned SIM image
def sim_optical_section(imgs, axis=0):
    """
    Law of signs optical sectioning reconstruction for three sim images with relative phase differences of 2*pi/3
    between each.

    Point: Let I[a] = A * [1 + m * cos(phi + phi_a)]
    Then sqrt( (I[0] - I[1])**2 + (I[1] - I[2])**2 + (I[2] - I[0])**2 ) = m*A * 3/ np.sqrt(2)

    :param imgs: array of size 3 x Ny x Nx
    "param axis: axis to perform the computation along
    :return img_os: optically sectioned image
    """
    imgs = np.swapaxes(imgs, 0, axis)

    img_os = np.sqrt(2) / 3 * np.sqrt((imgs[0] - imgs[1]) ** 2 + (imgs[0] - imgs[2]) ** 2 + (imgs[1] - imgs[2]) ** 2)

    return img_os

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
    def dfx_min_fn(f): return -2 * (cc_fn(f) * dfx_cc(f).conj()).real / fft_norm
    def dfy_min_fn(f): return -2 * (cc_fn(f) * dfy_cc(f).conj()).real / fft_norm
    def jac(f): return np.array([dfx_min_fn(f), dfy_min_fn(f)])

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
def estimate_phase(img_ft, sim_frq, dxy):
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

    phase = np.mod(np.angle(tools.get_peak_value(img_ft, fx, fy, sim_frq, 2)), 2*np.pi)

    return phase

def fit_phase_realspace(img, sim_frq, dxy, phase_guess=0, origin="center"):
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
        raise Exception('img must be strictly positive.')

    #
    if origin == "center":
        x = tools.get_fft_pos(img.shape[1], dxy, centered=True, mode="symmetric")
        y = tools.get_fft_pos(img.shape[0], dxy, centered=True, mode="symmetric")
    elif origin == "edge":
        x = tools.get_fft_pos(img.shape[1], dxy, centered=False, mode="positive")
        y = tools.get_fft_pos(img.shape[0], dxy, centered=False, mode="positive")
    else:
        raise Exception("'origin' must be 'center' or 'edge' but was '%s'" % origin)

    xx, yy = np.meshgrid(x, y)

    # xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    # xx = dxy * xx
    # yy = dxy * yy

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

def fit_phase_wicker(imgs_ft, otf, sim_frq, dxy, fmax, phases_guess=None, fit_amps=True):
    """
    # TODO: this can also return components separated the opposite way of desired

    Estimate phases using the cross-correlation minimization method of Wicker et. al.
    NOTE: the Wicker method only determines the relative phase. Here we try to estimate the absolute phase
    by additionally looking at the peak phase for one pattern

    # todo: currently hardcoded for 3 phases
    # todo: can I improve the fitting by adding jacobian?
    # todo: can get this using d(M^{-1})dphi1 = - M^{-1} * (dM/dphi1) * M^{-1}
    # todo: probably not necessary, because phases should always be close to equally spaced, so initial
    # guess should be good

    See https://doi.org/10.1364/OE.21.002032 for more details about this method.
    :param imgs_ft: 3 x ny x nx. o(f), o(f-fo), o(f+fo)
    :param otf:
    :param sim_frq: [fx, fy]
    :param dxy: pixel size (um)
    :param fmax: maximum spatial frequency
    :param phase_guess: [phi1, phi2, phi3] if None will use [0, 120, 240]
    :param fit_amps: if True will also fit amplitude differences between components

    :return phases: list of phases determined using this method
    :return amps: [A1, A2, A3]. If fit_amps is False, these will all be ones.
    """
    if phases_guess is None:
        phases_guess = [0, 2*np.pi/3, 4*np.pi/3]

    _, ny, nx = imgs_ft.shape
    fx = tools.get_fft_frqs(nx, dxy)
    dfx = fx[1] - fx[0]
    fy = tools.get_fft_frqs(ny, dxy)
    dfy = fy[1] - fy[0]

    # compute cross correlations of data
    cross_corrs = np.zeros((3, 3, 3), dtype=np.complex)
    # C_i(k) = S(k - i*p)
    # this is the order set by matrix M, i.e.
    # M * [S(k), S(k-i*p), S(k + i*p)]
    inds = [0, 1, -1]
    for ii, mi in enumerate(inds):  # [0, 1, 2] -> [0, 1, -1]
        for jj, mj in enumerate(inds):
            for ll, ml in enumerate(inds):

                # get shifted otf -> otf(f - l * fo)
                # pix_shift = [int(np.round((ml) * sim_frq[0] / dfx)),
                #              int(np.round((ml) * sim_frq[1] / dfy))]
                # otf_shift = tools.translate_pix(otf, pix_shift, mode='no-wrap')

                # pix_shift = [int(np.round((ml) * sim_frq[0] / dfx)),
                #              int(np.round((ml) * sim_frq[1] / dfy))]
                otf_shift, _, _ = tools.translate_pix(otf, -ml*sim_frq, dx=dfx, dy=dfy, mode='no-wrap')

                weight = otf * otf_shift.conj() / (np.abs(otf_shift)**2 + np.abs(otf)**2)
                weight[np.isnan(weight)] = 0
                weight / np.sum(weight)

                # shifted component C_j(f - l*fo)
                cshift = tools.translate_ft(imgs_ft[jj], -ml * sim_frq, dxy, dxy)
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
        def minv(p): return np.linalg.inv(get_kmat([p[0], p[1], p[2]], [1, 1, 1], [1, p[3], p[4]]))
        # minv = lambda p: np.linalg.inv(get_kmat([p[0], p[1], p[2]], [p[3], p[4], p[5]], [1, p[6], p[7]]))
    else:
        def minv(p): return np.linalg.inv(get_kmat(p, [1, 1, 1]))

    # remove i = (j + l) terms
    ii_minus_jj = np.array(inds)[:, None] - np.array(inds)[None, :]

    def fn_sum(p, ll): return np.sum(np.abs(minv(p).dot(cross_corrs[:, :, ll]).dot(minv(p).conj().transpose()) *
                                        (ii_minus_jj != inds[ll]))**0.5)

    def fn(p): return np.sum([fn_sum(p, ll) for ll in range(3)])

    # can also include amplitudes and modulation depths in optimization process
    if fit_amps:
        result = scipy.optimize.minimize(fn, np.concatenate((phases_guess, np.array([1, 1]))))
        phases = result.x[0:3]
        # mods = result.x[3:6]
        # amps = np.concatenate((np.array([1]), result.x[6:]))
        amps = np.concatenate((np.array([1]), result.x[3:]))
    else:
        result = scipy.optimize.minimize(fn, phases_guess)
        phases = result.x
        amps = np.array([1, 1, 1])
        # mods = np.array([1, 1, 1])

    # estimate absolute phase by looking at phase of peak
    phase_offsets = np.angle(tools.get_peak_value(imgs_ft[0], fx, fy, sim_frq, peak_pixel_size=2))
    phases = np.mod((phases - phases[0]) + phase_offsets, 2*np.pi)

    # there is an ambiguity in this optimization process:

    return phases, amps

# power spectrum and modulation depths
def get_noise_power(img_ft, fxs, fys, fmax):
    """
    Get average noise power outside OTF support for an image

    :param img_ft: Fourier transform of image
    :param fxs: 1D array, x-frequencies
    :param fys: 1D array, y-frequencies
    :param fmax: maximum frequency where signal may be present, i.e. (0.5*wavelength/NA)^{-1}
    :return noise_power:
    """

    ps = np.abs(img_ft) ** 2

    # exclude regions of frequency space
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff = np.sqrt(fxfx ** 2 + fyfy ** 2)
    noise_power = np.mean(ps[ff > fmax])

    return noise_power

def get_mcnr(img_ft, frqs, fxs, fys, fmax):
    """
    Get ratio of modulation contrast to noise, which is a measure of quality of SIM contrast.

    see e.g. doi:10.1038/srep15915

     mcnr = sqrt(peak power) / sqrt(noise power)

    :param img_ft: fourier transform of given SIM image
    :param frqs: [fx, fy]
    :param fxs:
    :param fys:
    :param fmax:
    :return mcnr:
    """

    peak_height = np.abs(tools.get_peak_value(img_ft, fxs, fys, frqs, peak_pixel_size=1))
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

    # val = p[4]**2 * (p[0]**2 * np.abs(fmag) ** (-2 * p[1]) +
    #                  p[2]**2 * np.abs(fmag) ** (-2 * p[3])) * \
    #       np.abs(otf)**2 + p[5]
    val = p[2]**2 * p[0]**2 * np.abs(fmag) ** (-2 * p[1]) * np.abs(otf)**2 + p[3]

    return val

def power_spectrum_jacobian(p, fmag, otf):
    """
    jacobian of power_spectrum_fn()
    :param p:
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
    fit_results = tools.fit_model(ps, fit_fn, init_params, fixed_params=fixed_params,
                                  bounds=bounds, model_jacobian=jac_fn)

    # todo: fitting after taking log works ok for initial fit, but does not give good fit for modulation depth. Why?
    # log_ps = np.log(ps)
    # fit_fn_log = lambda p: np.log(fit_fn(p))
    # fit_results = tools.fit_model(log_ps, fit_fn_log, init_params, fixed_params=fixed_params, bounds=bounds)

    return fit_results, mask

def plot_power_spectrum_fit(img_ft, otf, options, pfit, frq_sim=None, mask=None, figsize=(20, 10), ttl_str=""):
    """
    Plot results of fit_power_spectrum()

    :param img_ft:
    :param otf:
    :param options:
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
    # plot 2D power spectrum divided by otf
    # ######################
    ax4 = plt.subplot(grid[1, 4:6])
    # ps_over_otf[mask == 0] = np.nan
    ax4.imshow(ps_exp_deconvolved, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent)

    circ = matplotlib.patches.Circle((0, 0), radius=fmax, color='k', fill=0, ls='--')
    ax4.add_artist(circ)

    circ2 = matplotlib.patches.Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--')
    ax4.add_artist(circ2)

    ax4.set_xlabel('fx (1/um)')
    ax4.set_ylabel('fy (1/um)')
    ax4.title.set_text('deconvolved power spectrum')

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

    sttl_str = ""
    if ttl_str != "":
        sttl_str += "%s\n" % ttl_str
    sttl_str += 'm=%0.3g, A=%0.3g, alpha=%0.3f\nnoise=%0.3g, frq sim = (%0.2f, %0.2f) 1/um' % \
                (pfit[-2], pfit[0], pfit[1], pfit[-1], frq_sim[0], frq_sim[1])
    plt.suptitle(sttl_str)

    return fig

# inversion functions
def get_kmat(phases, mod_depths=(1, 1, 1), amps=(1, 1, 1)):
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

def mult_img_matrix(imgs, matrix):
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

def separate_components(imgs_ft, phases, mod_depths=None, amps=None):
    """
    Do noisy inversion of SIM data, i.e. determine
    [[S(k)H(k)], [S(k-p)H(k)], [S(k+p)H(k)]] = M^{-1} * [[D_1(k)], [D_2(k)], [D_3(k)]]

    # todo: generalize for case with more than 3 phases

    :param imgs_ft: nangles x nphases x ny x nx. Fourier transform with zero frequency information in middle. i.e.
    as obtained from fftshift
    :param phases: nangles x nphases
    :param mod_depths: list of length nangles. Optional. If not provided, set to 1.
    :param amps:
    :return components_ft:
    """
    nangles, nphases, ny, nx = imgs_ft.shape

    if mod_depths is None:
        mod_depths = np.ones((nangles, nphases))

    if amps is None:
        amps = np.ones((nangles, nphases))

    if nphases != 3:
        raise Exception("noisy_inversion() is only implemented for nphases=3, but nphases=%d" % nphases)

    components_ft = np.empty((nangles, nphases, ny, nx), dtype=np.complex) * np.nan

    for ii in range(nangles):
        kmat = get_kmat(phases[ii], mod_depths[ii], amps[ii])

        # try to do inversion
        try:
            kmat_inv = np.linalg.inv(kmat)
            components_ft[ii] = mult_img_matrix(imgs_ft[ii], kmat_inv)

        except np.linalg.LinAlgError:
            warnings.warn("warning, inversion matrix for angle index=%d is singular. This data will be ignored in SIM reconstruction" % ii)

    return components_ft

def global_phase_correction(imgs_shifted_ft):
    """
    Correct phases, based on fact that img_ft(f)xI^*(f) should be real

    Should correct S(f-fo)otf(f) -> np.exp(i phase_corr) * S(f-fo)otf(f)

    imgs_ft[ii, 0] = S(f)otf(f) * wiener(f)
    imgs_ft[ii, 1] = S(f-fo)otf(f) * wiener(f) after shifting to correct position
    imgs_ft[ii, 2] = S(f+fo)otf(f) * wiener(f) after shifting to correction position

    :param imgs_shifted_ft: nangles x 3 x ny x nx
    :return phase_corrections: phase correction.
    """
    nangles = imgs_shifted_ft.shape[0]
    phase_corrections = np.zeros((nangles))

    # todo: should weight by SNR, or something like this
    for ii in range(nangles):
        phase_corrections[ii] = np.angle(np.sum(imgs_shifted_ft[ii, 0] * imgs_shifted_ft[ii, 1].conj()))

    return phase_corrections

# filtering and combining images
def wiener_deconvolution(img, otf, sn_power_ratio, snr_includes_otf=False):
    """
    Return Wiener deconvolution filter, which is used to obtain an estimate of img_ft after removing the effect of the OTF.

    Given a noisy image defined by,
    img_ft(f) = obj(f) * otf(f) + N
    the deconvolution filter is given by
    Filter = otf*(k) / (|otf(k)|^2 + noise_power/signal_power).
    i.e. where the noise the signal this will approach zero. Where signal dominates noise this divides by the otf,
     except where the otf is zero

    :param img_ft:
    :param otf:
    :param sn_power_ratio:
    :return:
    """
    if snr_includes_otf:
        wfilter = otf.conj() / (np.abs(otf)**2 * (1 + 1 / sn_power_ratio))
    else:
        wfilter = otf.conj() / (np.abs(otf) ** 2 + 1 / sn_power_ratio)

    wfilter[np.isnan(wfilter)] = 0
    img_deconvolved = img * wfilter

    return img_deconvolved, wfilter

def rl_deconvolution(img_ft, otf):
    # todo: doesn't skimage only operate on uint8 images?
    img = fft.fftshift(fft.ifft2(fft.ifftshift(img_ft)))

    psf = np.abs(fft.fftshift(fft.ifft2(otf)))
    img_deconv = skimage.restoration.richardson_lucy(img, psf, iterations=50, clip=False)
    img_deconv_ft = fft.fftshift(fft.fft2(fft.ifftshift(img_deconv)))

    return img_deconv_ft

def generalized_wiener_filter(otf, sn_power_ratio):
    """
    Return filter which is the OTF times the ratio of signal to noise power. This is a weighting factor.

    img_ft[k] * |otf(k)|**2 * signal_power/noise_power

    :param otf: optical transfer function.
    :param sn_power_ratio: ratio of signal power to noise power spectral density. Can be supplied as either a single
     number, or an array of the same size as otf
    :return filter: array the same size as otf
    """
    # filter
    wfilter = sn_power_ratio * np.abs(otf)**2

    # deal with any diverging points by averaging nearest pixels
    if np.any(np.isinf(wfilter)):
        xx, yy = np.meshgrid(range(wfilter.shape[1]), range(wfilter.shape[0]))
        xinds = xx[np.isinf(wfilter)]
        yinds = yy[np.isinf(wfilter)]

        # todo: not handling case where these points might be outside ROI
        for xi, yi in zip(xinds, yinds):
            wfilter[yi, xi] = np.mean([wfilter[yi + 1, xi], wfilter[yi - 1, xi], wfilter[yi, xi+1], wfilter[yi, xi-1]])

    return wfilter

def estimate_snr(img_ft, fxs, fys, fmax, filter_size=5):
    """
    estimate signal to noise ratio from image
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

# estimate modulation parameters using Bayesian inference. Alternative approach to the more direct techniques above.
class fftshift2dOp(theano.Op):
    # Properties attribute
    __props__ = ()

    # itypes and otypes attributes are
    # compulsory if make_node method is not defined.
    # They're the type of input and output respectively
    itypes = None
    otypes = None

    #Compulsory if itypes and otypes are not defined
    def make_node(self, sx, sy, phi):
        # make apply node
        sx = theano.tensor.as_tensor_variable(sx)
        sy = theano.tensor.as_tensor_variable(sy)
        phi = theano.tensor.as_tensor_variable(phi)
        return theano.Apply(self, [sx, sy, phi], [tt.matrix()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        sx, sy, phi = inputs_storage
        ft = self.otf * np.exp(1j * phi) * fft.fft2(np.exp(1j * 2*np.pi * (sx * self.xx + sy * self.yy)) * fft.ifft2(self.imgft))
        output_storage[0][0] = np.concatenate((ft.real, ft.imag))

    # optional:
    check_input = True

    def __init__(self, imgft, otf):
        """
        Warning: imgft should be in 'normal' ordering. i.e. do NOT fftshift before sending here.

        Will return imgft(x - sx, y - sy). i.e. will shift image towards positive direction for positive
        sx, sy

        :param imgft:
        :param xx:
        :param yy:
        """
        self.imgft = imgft
        self.otf = otf

        # choose units so that shifts are in pixels
        # assuming square image
        dt = 1 / imgft.shape[0]
        x = tools.get_fft_pos(imgft.shape[1], dt=dt, centered=False, mode='symmetric')
        y = tools.get_fft_pos(imgft.shape[0], dt=dt, centered=False, mode='symmetric')
        xx, yy = np.meshgrid(x, y)

        #self.dxy = dxy
        self.yy = yy
        self.xx = xx
        super(fftshift2dOp, self).__init__()

    def grad(self, inputs, output_grads):
        # todo: not convinced this is working. Need to test it
        sx, sy, phi = inputs

        df_dsx = self.otf * np.exp(1j * phi) * fft.fft2(1j * 2 * np.pi * self.xx * np.exp(1j * 2 * np.pi * (sx * self.xx + sy * self.yy)) * fft.ifft2(self.imgft))
        dc_dsx = output_grads[0] * df_dsx
        #dc_dsx_roi = dc_dsx[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        df_dsy = self.otf * np.exp(1j * phi) * fft.fft2(1j * 2 * np.pi * self.yy * np.exp(1j * 2 * np.pi * (sx * self.xx + sy * self.yy)) * fft.ifft2(self.imgft))
        dc_dsy = output_grads[1] * df_dsy
        #dc_dsy_roi = dc_dsy[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        df_dphi = 1j * self.otf * np.exp(1j * phi) * fft.fft2(-1j * 2 * np.pi * self.xx * np.exp(-1j * 2 * np.pi * (sx * self.xx + sy * self.yy)) * fft.ifft2(self.imgft))
        dc_dphi = output_grads[2] * df_dphi
        #dc_dphi_roi = dc_dphi[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        # return [np.concatenate((dc_dsx_roi.real, dc_dsx_roi.imag)),
        #         np.concatenate((dc_dsy_roi.real, dc_dsy_roi.imag)),
        #         np.concatenate((dc_dphi_roi.real, dc_dphi_roi.imag))]
        return [np.concatenate((dc_dsx.real, dc_dsx.imag)),
                np.concatenate((dc_dsy.real, dc_dsy.imag)),
                np.concatenate((dc_dphi.real, dc_dphi.imag))]

    def R_op(self, inputs, eval_points):
        pass

def get_sim_params_mcmc(img_ft, otf, frqs_guess, fx, fy, fmax, roi_size=81, fit_shifts=True):
    """
    Compute SIM parameters using bayesian inference

    # todo: if fit_shifts is false, automatically commute shift_vals

    :param img_ft:
    :param otf:
    :param frqs_guess:
    :param fx:
    :param fy:
    :param fmax:
    :param roi_size:
    :param fit_shifts:
    :param shift_vals:
    :return:
    """

    fx_guess, fy_guess = frqs_guess

    # get frequency information
    ny, nx = img_ft.shape
    dfx = fx[1] - fx[0]
    dfy = fy[1] - fy[0]

    noise = np.sqrt(get_noise_power(img_ft, fx, fy, fmax))

    # find pixel to center ROI at
    fx_guess_pix = np.argmin(np.abs(fx_guess - fx))
    fy_guess_pix = np.argmin(np.abs(fy_guess - fy))

    # get ROI's from frequency guesses
    model_roi = tools.get_centered_roi([fy_guess_pix, fx_guess_pix], [roi_size, roi_size])
    center_roi = tools.get_centered_roi([fy.size//2, fx.size//2], [roi_size, roi_size])

    # get image and reshape as desired
    Iroi = img_ft[model_roi[0]:model_roi[1], model_roi[2]:model_roi[3]]
    Iobs = np.concatenate((Iroi.real, Iroi.imag))

    # function to convert from all real format back to complex image
    recombine_fn = lambda mat: mat[:mat.shape[0] // 2] + 1j * mat[mat.shape[0] // 2:]

    Iroi_center = img_ft[center_roi[0]:center_roi[1], center_roi[2]:center_roi[3]]

    otf_roi = otf[model_roi[0]:model_roi[1], model_roi[2]:model_roi[3]]

    # otherwise will throw errors bc no test values associated with some theano variables.
    theano.config.compute_test_value = "warn"

    # set up model fn
    sx_var = tt.scalar('sx_var')
    sy_var = tt.scalar('sy_var')
    phi_var = tt.scalar('phi_var')
    var = fftshift2dOp(Iroi_center, otf_roi)(sx_var, sy_var, phi_var)
    model_fn = theano.function([sx_var, sy_var, phi_var], var)

    with pm.Model() as model:
        if fit_shifts:
            sx = pm.Uniform('sx', -2, 2)
            sy = pm.Uniform('sy', -2, 2)
        else:
            # in this case we take the guess value as fixed
            sx = (fx[fx_guess_pix] - fx_guess) / dfx
            sy = (fy[fy_guess_pix] - fy_guess) / dfy

        phi = pm.Normal('phi', mu=np.pi, sd=2)
        m = pm.Uniform('m', 0, 1)
        mshift = 0.5 * m * fftshift2dOp(Iroi_center, otf_roi)(sx, sy, phi)

        likelihood = pm.Normal('y', mu=mshift, sd=noise, observed=Iobs)

        trace = pm.sample(1000, cores=2, tune=6000, chains=2, discard_tuned_samples=True)

    # extract inferred values
    if fit_shifts:
        sx_inf = np.mean(trace.get_values('sx'))
        sy_inf = np.mean(trace.get_values('sy'))
    else:
        sx_inf = sx
        sy_inf = sy

    frqs_sim = [fx[fx_guess_pix] - sx_inf * dfx,
                fy[fy_guess_pix] - sy_inf * dfy]


    phi_inferred = np.mean([np.mod(np.mean(ch), 2*np.pi) for ch in trace.get_values('phi', combine=False)])
    mod_depth_inferred = np.mean(trace.get_values('m'))
    img_inferred = recombine_fn(mod_depth_inferred * model_fn(sx_inf, sy_inf, phi_inferred))

    return frqs_sim, phi_inferred, mod_depth_inferred, img_inferred, trace

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
        raise Exception("If coherent_projection is false, OTF must be provided")

    if len(mod_depths) != nangles:
        raise Exception("mod_depths must have length nangles")

    if amps is None:
        amps = np.ones((nangles, nphases))

    if origin == "center":
        x = tools.get_fft_pos(nx, pix_size, centered=True, mode="symmetric")
        y = tools.get_fft_pos(ny, pix_size, centered=True, mode="symmetric")
    elif origin == "edge":
        x = tools.get_fft_pos(nx, pix_size, centered=False, mode="positive")
        y = tools.get_fft_pos(ny, pix_size, centered=False, mode="positive")
    else:
        raise Exception("'origin' must be 'center' or 'edge' but was '%s'" % origin)

    xx, yy = np.meshgrid(x, y)

    if 'bin_size' in kwargs:
        nbin = kwargs['bin_size']
    else:
        nbin = 1

    sim_imgs = np.zeros((nangles, nphases, int(ny / nbin), int(nx / nbin)))
    snrs = np.zeros(sim_imgs.shape)
    real_max_photons = np.zeros((nangles, nphases))

    for ii in range(nangles):
        for jj in range(nphases):
            # pattern = amps[ii, jj] * (1 + mod_depths[ii, jj] * np.cos(2*np.pi * (frqs[ii][0] * xx + frqs[ii][1] * yy) +
            #                                                           phases[ii, jj]))
            pattern = amps[ii, jj] * (1 + mod_depths[ii] * np.cos(2 * np.pi * (frqs[ii][0] * xx + frqs[ii][1] * yy) +
                                                        phases[ii, jj]))

            if not coherent_projection:
                pattern_ft = fft.fftshift(fft.fft2(fft.ifftshift(pattern)))
                pattern = fft.fftshift(fft.ifft2(fft.ifftshift(pattern_ft * otf))).real

            sim_imgs[ii, jj], snrs[ii, jj], real_max_photons[ii, jj] = camera_noise.simulated_img(ground_truth * pattern, max_photons, cam_gains,
                                                           cam_offsets, cam_readout_noise_sds, pix_size, otf=otf, **kwargs)

    return sim_imgs, snrs, real_max_photons

