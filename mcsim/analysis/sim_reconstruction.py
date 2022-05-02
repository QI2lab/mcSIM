"""
Tools for reconstructing 2D SIM images from raw data.

The primary reconstruction code is contained in the class SimImageSet, which operates on a single 2D plane at a time.

reconstruct_sim_dataset() provides an example of using SimImageSet to reconstruct a larger dataset, in this case
implemented for a MicroManager dataset containing multiple z-positions, color channels, and time points.
"""

import mcsim.analysis.analysis_tools as tools
import mcsim.analysis.mm_io as mm_io
import mcsim.analysis.psd as psd
# localize psf
import localize_psf.fit as fit
import localize_psf.fit_psf
import localize_psf.affine as affine
import localize_psf.rois as rois
import localize_psf.camera as camera_noise

import pickle
import json
import os
import pathlib
from pathlib import Path
import time
import datetime
import copy
import warnings
import shutil
import joblib
import io
from io import StringIO
import numpy as np
from scipy import fft
import scipy.optimize
import scipy.signal
import scipy.ndimage
from skimage.exposure import match_histograms
import tifffile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from matplotlib.patches import Circle, Rectangle

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SimImageSet:
    def __init__(self, physical_params, imgs,
                 otf=None, wiener_parameter=0.1,
                 frq_estimation_mode="band-correlation", frq_guess=None,
                 phase_estimation_mode="wicker-iterative", phases_guess=None,
                 combine_bands_mode="fairSIM", fmax_exclude_band0=0,
                 mod_depths_guess=None, use_fixed_mod_depths=False, mod_depth_otf_mask_threshold=0.1,
                 normalize_histograms=True, determine_amplitudes=False,
                 background=0, gain=1, max_phase_err=10 * np.pi / 180, min_p2nr=1, fbounds=(0.01, 1),
                 trim_negative_values=False, upsample_widefield=False,
                 interactive_plotting=False, save_dir=None, save_suffix="", save_prefix="",
                 use_gpu=CUPY_AVAILABLE, figsize=(20, 10)):
        """
        Reconstruct raw SIM data into widefield, SIM-SR, SIM-OS, and deconvolved images using the Wiener filter
        style reconstruction of Gustafsson and Heintzmann. This code relies on various ideas developed and
        implemented elsewhere, see for example fairSIM and openSIM.

        An instance of this class may be used directly to reconstruct a single SIM image which is stored as a
        numpy array. For a typical experiment it is usually best to write a helper function to load the data and
        coordinate the SIM parameter estimation and reconstruction of e.g. various channels, z-slices, time points
        or etc. For an example of this approach, see the function reconstruct_mm_sim_dataset()

        :param physical_params: {'pixel_size', 'na', 'wavelength'}. Pixel size and emission wavelength in um
        :param imgs: nangles x nphases x ny x nx raw data to be reconstructed
        :param otf: optical transfer function evaluated at the same frequencies as the fourier transforms of imgs.
         If None, estimate from NA. This can either be an array of size ny x nx, or an array of size nangles x ny x nx
         The second case corresponds to a system that has different OTF's per SIM acquisition angle.
        :param wiener_parameter: Attenuation parameter for Wiener filtering. This has a sligtly different meaning
         depending on the value of combine_bands_mode
        :param str frq_estimation_mode: "band-correlation", "fourier-transform", or "fixed"
        "band-correlation" first unmixes the bands using the phase guess values and computes the correlation between
        the shifted and unshifted band
        "fourier-transform" correlates the Fourier transform of the image with itself.
        "fixed" uses the frq_guess values
        :param frq_guess: 2 x nangles array of guess SIM frequency values
        :param str phase_estimation_mode: "wicker-iterative", "real-space", "naive", or "fixed"
        "wicker-iterative" follows the approach of https://doi.org/10.1364/OE.21.002032.
        "real-space" follows the approach of section IV-B in https://doir.org/10.1109/JSTQE.2016.2521542.
        "naive" uses the phase of the Fourier transform of the raw data.
        "fixed" uses the values provided from phases_guess.
        :param phases_guess: nangles x nphases array of phase guesses
        :param combine_bands_mode: "fairSIM" if using method of https://doi.org/10.1038/ncomms10980 or "openSIM" if
        using method of https://doi.org/10.1109/jstqe.2016.2521542
        :param float fmax_exclude_band0: amount of the unshifted bands to exclude, as a fraction of fmax. This can
        enhance optical sectioning by replacing the low frequency information in the reconstruction with the data.
        from the shifted bands only.
        For more details on the band replacement optical sectioning approach, see https://doi.org/10.1364/BOE.5.002580
        and https://doi.org/10.1016/j.ymeth.2015.03.020
        :param mod_depths_guess: If use_fixed_mod_depths is True, these modulation depths are used
        :param bool use_fixed_mod_depths: if true, use mod_depths_guess instead of estimating the modulation depths from the data
        :param bool normalize_histograms: for each phase, normalize histograms of images to account for laser power fluctuations
        :param background: Either a single number, or broadcastable to size of imgs. The background will be subtracted
         before running the SIM reconstruction
        :param bool determine_amplitudes: whether or not to determine amplitudes as part of Wicker phase optimization.
        This flag only has an effect if phase_estimation_mode is "wicker-iterative"
        :param background: a single number, or an array which is broadcastable to the same size as images. This will
        be subtracted from the raw data before processing.
        :param gain: gain of the camera in ADU/photons. This is a single number or an array which is broadcastable to
         the same size as the images whcih is sued to convert the ADU counts to photon numbers.
        :param max_phase_err: If the determined phase error between components exceeds this value, use the phase guess
        values instead of those determined by the estimation algorithm.
        :param min_p2nr: if the peak-to-noise ratio is smaller than this value, use the frequency guesses instead
         of the frequencies determined by the estimation algorithm.
        :param fbounds: frequency bounds as a fraction of fmax to be used in power spectrum fit. todo: remove ...
        :param bool interactive_plotting: show plots in python GUI windows, or save outputs only
        :param str save_dir: directory to save results. If None, then results will not be saved
        :param bool use_gpu:
        :param figsize:
        """
        # #############################################
        # open log file
        # #############################################
        self.save_dir = Path(save_dir)
        self.save_suffix = save_suffix
        self.save_prefix = save_prefix
        self.log = StringIO() # can save this stream to a file later if desired

        if self.save_dir is not None:
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True)

        # #############################################
        # print current time
        # #############################################
        tstamp = datetime.datetime.now().strftime('%Y/%d/%m %H:%M:%S')

        self.print_log("####################################################################################")
        self.print_log(tstamp)
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
        self.trim_negative_values = trim_negative_values
        self.upsample_widefield = upsample_widefield
        if self.upsample_widefield:
            raise NotImplementedError("upsampling widefield not yet fully implemented")

        self.upsample_fact = 2
        self.combine_bands_mode = combine_bands_mode
        self.use_fixed_mod_depths = use_fixed_mod_depths
        self.phase_estimation_mode = phase_estimation_mode
        self.fmax_exclude_band0 = fmax_exclude_band0
        self.otf_mask_threshold = mod_depth_otf_mask_threshold

        if phases_guess is None:
            self.frq_estimation_mode = "fourier-transform"
            self.print_log("No phase guesses provided, defaulting to frq_estimation_mode = '%s'" % "fourier_transform")
        else:
            self.frq_estimation_mode = frq_estimation_mode

        self.frequency_units = '1/um'
        self.spatial_units = 'um'

        # #############################################
        # images
        # #############################################
        self.imgs = imgs.astype(np.float64)
        self.nangles, self.nphases, self.ny, self.nx = imgs.shape
        # hardcoded for 2D SIM
        self.nbands = 3
        self.band_inds = np.array([0, 1, -1], dtype=int)

        # #############################################
        # real space parameters
        # #############################################
        self.dx = physical_params['pixel_size']
        self.dy = physical_params['pixel_size']
        self.x = tools.get_fft_pos(self.nx, self.dx, mode='symmetric')
        self.y = tools.get_fft_pos(self.ny, self.dy, mode='symmetric')
        self.x_us = tools.get_fft_pos(self.upsample_fact * self.nx, self.dx / self.upsample_fact, mode='symmetric')
        self.y_us = tools.get_fft_pos(self.upsample_fact * self.ny, self.dy / self.upsample_fact, mode='symmetric')

        # #############################################
        # get basic parameters
        # #############################################
        self.na = physical_params['na']
        self.wavelength = physical_params['wavelength']

        self.fmax = 1 / (0.5 * self.wavelength / self.na)
        self.fbounds = fbounds

        if frq_guess is not None:
            self.frqs_guess = np.array(frq_guess)
        else:
            self.frqs_guess = None

        if phases_guess is not None:
            self.phases_guess = np.array(phases_guess)
        else:
            self.phases_guess = None

        if mod_depths_guess is not None:
            self.mod_depths_guess = np.array(mod_depths_guess)
        else:
            self.mod_depths_guess = None

        # #############################################
        # get frequency data and OTF
        # #############################################
        self.fx = fft.fftshift(fft.fftfreq(self.nx, self.dx))
        self.fy = fft.fftshift(fft.fftfreq(self.ny, self.dy))
        self.fx_us = fft.fftshift(fft.fftfreq(self.upsample_fact * self.nx, self.dx / self.upsample_fact))
        self.fy_us = fft.fftshift(fft.fftfreq(self.upsample_fact * self.ny, self.dy / self.upsample_fact))

        if otf is None:
            otf = localize_psf.fit_psf.circ_aperture_otf(np.expand_dims(self.fx, axis=0),
                                                         np.expand_dims(self.fy, axis=1),
                                                         self.na, self.wavelength)

        if np.any(otf < 0) or np.any(otf > 1):
            raise ValueError("OTF must be >= 0 and <= 1")

        # otf is stored as nangles x ny x nx array to allow for possibly different OTF's along directions
        if otf.ndim == 2:
            otf = np.tile(otf, [self.nangles, 1, 1])
        self.otf = otf

        if self.otf.shape[-2:] != self.imgs.shape[-2:]:
            raise ValueError(f"OTF shape {self.otf.shape} and image shape {self.img.shape} are not compatible")

        # #############################################
        # remove background and convert from ADU to photons
        # #############################################
        self.imgs = (self.imgs - background) / gain
        self.imgs[self.imgs <= 0] = 1e-12

        # #############################################
        # print intensity information
        # #############################################
        self.mean_intensities = np.mean(self.imgs, axis=(2, 3))

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
        # periodic_portion = np.zeros((self.nangles, self.nphases, self.ny, self.nx), dtype=complex)
        # for jj in range(self.nangles):
        #     for kk in range(self.nphases):
        #         periodic_portion[jj, kk], _ = psd.periodic_smooth_decomp(self.imgs[jj, kk])

        # real-space apodization is not so desirable because produces a roll off in the reconstruction. But seems ok.
        apodization = np.expand_dims(scipy.signal.windows.tukey(self.imgs.shape[2], alpha=0.1), axis=1) * \
                      np.expand_dims(scipy.signal.windows.tukey(self.imgs.shape[3], alpha=0.1), axis=0)
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
        # tstart = time.perf_counter()
        # os_imgs = [sim_optical_section(self.imgs[ii], phase_differences=self.phases[ii]) for ii in range(self.nangles)]
        # self.sim_os = np.mean(np.stack(os_imgs, axis=0), axis=0)
        # self.sim_os = np.mean(sim_optical_section(self.imgs, axis=1), axis=0)
        # self.print_log("Computing SIM-OS image took %0.2fs" % (time.perf_counter() - tstart))

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
            bands_unmixed_ft_temp = do_sim_band_separation(self.imgs_ft, self.phases_guess, mod_depths=np.ones((self.nangles)))

            band0 = np.expand_dims(bands_unmixed_ft_temp[:, 0], axis=1)
            band1 = np.expand_dims(bands_unmixed_ft_temp[:, 1], axis=1)
            self.frqs = self.estimate_sim_frqs(band0, band1)
        else:
            raise ValueError(f"frq_estimation_mode must be 'fixed', 'fourier-transform', or 'band-correlation'"
                             f" but was '{self.frq_estimation_mode:s}'")

        # for convenience also store periods and angles
        self.periods = 1 / np.sqrt(self.frqs[:, 0] ** 2 + self.frqs[:, 1] ** 2)
        self.angles = np.angle(self.frqs[:, 0] + 1j * self.frqs[:, 1])

        self.print_log(f"estimating {self.nangles:d} frequencies using mode {self.frq_estimation_mode:s}"
                       f" took {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # OTF value at frqs
        # #############################################
        otf_vals = np.zeros(self.nangles)
        
        for ii in range(self.nangles):
            ix = np.argmin(np.abs(self.frqs[ii, 0] - self.fx))
            iy = np.argmin(np.abs(self.frqs[ii, 1] - self.fy))
            otf_vals[ii] = self.otf[ii, iy, ix]

        self.otf_at_frqs = otf_vals
        
        # #############################################
        # estimate peak heights
        # #############################################
        tstart = time.perf_counter()

        peak_phases = np.zeros((self.nangles, self.nphases))
        peak_heights = np.zeros((self.nangles, self.nphases))
        noise = np.zeros((self.nangles, self.nphases))
        p2nr = np.zeros((self.nangles, self.nphases))
        for ii in range(self.nangles):
            for jj in range(self.nphases):
                peak_val = tools.get_peak_value(self.imgs_ft[ii, jj], self.fx, self.fy, self.frqs[ii], peak_pixel_size=1)
                peak_heights[ii, jj] = np.abs(peak_val)
                peak_phases[ii, jj] = np.angle(peak_val)
                noise[ii, jj] = np.sqrt(get_noise_power(self.imgs_ft[ii, jj], self.fx, self.fy, self.fmax))
                p2nr[ii, jj] = peak_heights[ii, jj] / noise[ii, jj]

            # if p2nr is too low use guess values instead
            if np.min(p2nr[ii]) < self.min_p2nr and self.frqs_guess is not None:
                self.frqs[ii] = self.frqs_guess[ii]
                self.print_log(f"Angle {ii:d}, minimum SIM peak-to-noise ratio = {np.min(p2nr[ii]):.2f}"
                               f" is less than the minimum value, {self.min_p2nr:.2f},"
                               " so fit frequency will be replaced with guess")

                for jj in range(self.nphases):
                    peak_val = tools.get_peak_value(self.imgs_ft[ii, jj], self.fx, self.fy, self.frqs[ii], peak_pixel_size=1)
                    peak_heights[ii, jj] = np.abs(peak_val)
                    peak_phases[ii, jj] = np.angle(peak_val)
                    p2nr[ii, jj] = peak_heights[ii, jj] / noise[ii, jj]

        self.cam_noise_rms = noise
        self.p2nr = p2nr
        self.peak_phases = peak_phases

        self.print_log(f"estimated peak-to-noise ratio in {time.perf_counter() - tstart:.2f}s")
        
        # #############################################
        # estimate spatial-resolved MCNR
        # #############################################
        # following the proposal of https://doi.org/10.1038/s41592-021-01167-7
        # calculate as the ratio of the modulation size over the expected shot noise value
        # note: this is the same as sim_os / sqrt(wf_angle) up to a factor
        tstart = time.perf_counter()

        # img_angle_ft = fft.fft(fft.ifftshift(self.imgs, axes=1), axis=1)
        # divide by nangles to remove ft normalization
        img_angle_ft = fft.fft(fft.ifftshift(self.imgs, axes=1), axis=1) / self.nangles
        # if I_j = Io * m * cos(2*pi*j), then want numerator to be 2*m. FT gives us m/2, so multiply by 4
        self.mcnr = 4 * np.abs(img_angle_ft[:, 1]) / np.sqrt(np.abs(img_angle_ft[:, 0]))

        self.print_log(f"estimated modulation-contrast-to-noise ratio in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # estimate phases
        # #############################################
        tstart = time.perf_counter()

        self.phases, self.amps = self.estimate_sim_phases()

        self.print_log(f"estimated {self.nangles * self.nphases:d} phases"
                       f" using mode {self.phase_estimation_mode:s} "
                       f"in {time.perf_counter() - tstart:.2f}")

        # #############################################
        # get optically sectioned image
        # #############################################
        tstart = time.perf_counter()
        os_imgs = [sim_optical_section(self.imgs[ii], phase_differences=self.phases[ii]) for ii in range(self.nangles)]
        self.sim_os = np.mean(np.stack(os_imgs, axis=0), axis=0)
        # self.sim_os = np.mean(sim_optical_section(self.imgs, axis=1), axis=0)
        self.print_log("Computing SIM-OS image took %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # do band separation
        # #############################################
        tstart = time.perf_counter()

        self.bands_unmixed_ft = do_sim_band_separation(self.imgs_ft, self.phases, amps=self.amps)

        self.print_log(f"separated bands in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # estimate noise in each band
        # #############################################
        tstart = time.perf_counter()

        self.noise_power_bands = np.zeros((self.nangles, self.nbands))
        for ii in range(self.nangles):
            for jj in range(self.nbands):
                self.noise_power_bands[ii, jj] = get_noise_power(self.bands_unmixed_ft[ii, jj], self.fx, self.fy, self.fmax)

        self.print_log(f"estimated noise in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # shift bands
        # #############################################
        tstart = time.perf_counter()
        self.bands_shifted_ft = self.shift_bands()
        self.print_log(f"shifted bands in {time.perf_counter() - tstart:.2f}s")

        tstart = time.perf_counter()
        self.otf_shifted = self.shift_otf()
        self.print_log(f"shifted otfs in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # correct global phase and get modulation depths
        # #############################################
        tstart = time.perf_counter()

        # correct global phases and estimate modulation depth from band correlations
        mask = np.logical_and(self.otf_shifted[:, 0] > self.otf_mask_threshold,
                              self.otf_shifted[:, 1] > self.otf_mask_threshold)

        if self.fmax_exclude_band0 > 0:
            # correct other band weights at modulation frequencies
            for ii in range(self.nangles):
                ff_us = np.sqrt(np.expand_dims(self.fx_us, axis=0) ** 2 + np.expand_dims(self.fy_us, axis=1) ** 2)
                mask[ii][ff_us < self.fmax * self.fmax_exclude_band0] = False

                ff_us = np.sqrt(np.expand_dims(self.fx_us + self.frqs[ii, 0], axis=0) ** 2 +
                                np.expand_dims(self.fy_us + self.frqs[ii, 1], axis=1) ** 2)
                mask[ii][ff_us < self.fmax * self.fmax_exclude_band0] = False

        for ii in range(self.nangles):
            if not np.any(mask[ii]):
                raise ValueError(f"band overlap mask for angle {ii:d} was all False")

        # corrected phases
        # can either think of these as (1) acting on phases such that phase -> phase - phase_correction
        # or (2) acting on bands such that band1(f) -> e^{i*phase_correction} * band1(f)
        # TODO: note, the global phases I use here have the opposite sign relative to our BOE paper eq. S47
        self.phase_corrections, mags = get_band_overlap(self.bands_shifted_ft[:, 0], self.bands_shifted_ft[:, 1],
                                                        self.otf_shifted[:, 0], self.otf_shifted[:, 1],
                                                        mask)

        if self.combine_bands_mode == "openSIM":
            # estimate power spectrum parameters
            fit_depths_pspectrum, self.power_spectrum_params, self.power_spectrum_masks = self.fit_power_spectra()

        if self.use_fixed_mod_depths:
            self.print_log("using fixed modulation depth")
            self.mod_depths = self.mod_depths_guess
        else:
            if self.combine_bands_mode == "openSIM":
                self.mod_depths = fit_depths_pspectrum[:, 1]
            elif self.combine_bands_mode == "fairSIM":
                self.mod_depths = mags
            else:
                raise ValueError(f"combine_mode must be 'fairSIM' or 'openSIM' but was '{self.combine_bands_mode:s}'")

        self.print_log("estimated global phases and mod depths in %0.2fs" % (time.perf_counter() - tstart))

        # #############################################
        # Get weights and combine bands
        # #############################################
        tstart = time.perf_counter()

        if self.combine_bands_mode == "openSIM":
            if self.fmax_exclude_band0 > 0:
                raise NotImplementedError("fmax_exlude_band0=0 is not implemented with combine_mode='openSIM'")

            self.weights, self.weights_norm = self.get_weights_open_sim()

        elif self.combine_bands_mode == "fairSIM":
            # following the approach of FairSIM: https://doi.org/10.1038/ncomms10980
            self.weights = self.otf_shifted.conj()
            weights_decon = np.array(self.weights, copy=True)

            # "fill in missing cone" by using shifted bands instead of unshifted band for values near DC
            if self.fmax_exclude_band0 > 0:

                # cubic spline which is 0 at f1 with derivative 0, and 1 at f2 with derivative 0
                # f1 = self.fmax * self.fmax_exclude_band0
                # f2 = self.fmax * self.fmax_exclude_band0 * 1.1
                # mat = np.array([[f1 ** 3, f1 ** 2, f1, 1],
                #                 [3 * f1 ** 2, 2 * f1, 1, 0],
                #                 [f2 ** 3, f2 ** 2, f2, 1],
                #                 [3 * f2 ** 2, 2 * f2, 1, 0]])
                # spline_coeffs = np.linalg.inv(mat).dot(np.array([[0], [0], [1], [0]]))
                # def spline_fn(f): return spline_coeffs[0] * f**3 + spline_coeffs[1] * f**2 + spline_coeffs[2] * f + spline_coeffs[3]

                for ii in range(self.nangles):
                    for jj, ee in enumerate(self.band_inds):
                        # correct unshifted band weights near zero frequency
                        ff_us = np.sqrt(np.expand_dims(self.fx_us + ee * self.frqs[ii, 0], axis=0)**2 +
                                        np.expand_dims(self.fy_us + ee * self.frqs[ii, 1], axis=1)**2)

                        # exclude all points inside a certain radius
                        # self.weights[ii, jj][ff_us <= (self.fmax * self.fmax_exclude_band0)] = 0

                        # connect smoothly with spline
                        # frqs_is_btw = np.logical_and(ff_us >= f1, ff_us <= f2)
                        # self.weights[ii, jj][frqs_is_btw] *= spline_fn(ff_us[frqs_is_btw])

                        # gaussian smoothing for weight
                        self.weights[ii, jj] *= (1 - np.exp(-0.5 * ff_us**2 / (self.fmax * self.fmax_exclude_band0)**2))

            self.weights_norm = self.wiener_parameter**2 + np.nansum(np.abs(self.weights) ** 2, axis=(0, 1))

        else:
            raise ValueError(f"combine_mode must be 'fairSIM' or 'openSIM' but was '{self.combine_bands_mode:s}'")

        self.print_log(f"computed band weights in {time.perf_counter() - tstart:.2f}s")

        # combine bands
        tstart = time.perf_counter()

        phase_corr_mat = np.exp(1j * np.concatenate((np.zeros((self.nangles, 1)),
                                                     np.expand_dims(self.phase_corrections, axis=1),
                                                     np.expand_dims(-self.phase_corrections, axis=1)), axis=1))

        mod_depth_corr_mat = np.concatenate((np.ones((self.nangles, 1)),
                                             np.expand_dims(1 / self.mod_depths, axis=1),
                                             np.expand_dims(1 / self.mod_depths, axis=1)), axis=1)

        # put in modulation depth and global phase corrections
        # components array useful for diagnostic plots
        self.sim_sr_ft_components = self.bands_shifted_ft * self.weights * \
                                    np.expand_dims(phase_corr_mat, axis=(2, 3)) * \
                                    np.expand_dims(mod_depth_corr_mat, axis=(2, 3)) / \
                                    np.expand_dims(self.weights_norm, axis=(0, 1))
        # final FT image
        self.sim_sr_ft = np.nansum(self.sim_sr_ft_components, axis=(0, 1))

        # inverse FFT to get real-space reconstructed image
        apodization = np.expand_dims(scipy.signal.windows.tukey(self.sim_sr_ft.shape[1], alpha=0.1), axis=0) * \
                      np.expand_dims(scipy.signal.windows.tukey(self.sim_sr_ft.shape[0], alpha=0.1), axis=1)

        # irfft2 ~2X faster than ifft2. But have to slice out only half the frequencies
        self.sim_sr = fft.fftshift(fft.irfft2(fft.ifftshift(self.sim_sr_ft * apodization)[:, : self.sim_sr_ft.shape[1] // 2 + 1]))
        # self.sim_sr = fft.fftshift(fft.ifft2(fft.ifftshift(self.sim_sr_ft * apodization))).real
        if self.trim_negative_values:
            self.sim_sr[self.sim_sr < 0] = 0

        self.print_log(f"combining bands using mode '{self.combine_bands_mode}'"
                       f" and Wiener parameter {self.wiener_parameter:.3f}"
                       f" took {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # widefield deconvolution
        # #############################################
        tstart = time.perf_counter()

        if self.combine_bands_mode == "openSIM":
            # get signal to noise ratio
            wf_noise = get_noise_power(self.widefield_ft, self.fx, self.fy, self.fmax)
            fit_result, self.mask_wf = fit_power_spectrum(self.widefield_ft, np.mean(self.otf, axis=0),
                                                          self.fx, self.fy, self.fmax, self.fbounds,
                                                          init_params=[None, self.power_spectrum_params[0, 0, 1], 1, wf_noise],
                                                          fixed_params=[False, True, True, True])

            self.pspec_params_wf = fit_result['fit_params']

            with np.errstate(invalid="ignore", divide="ignore"):
                ff = np.sqrt(np.expand_dims(self.fx, axis=0)**2 +
                             np.expand_dims(self.fy, axis=1)**2)
                sig = power_spectrum_fn([self.pspec_params_wf[0], self.pspec_params_wf[1], self.pspec_params_wf[2], 0], ff, 1)

            wf_otf = np.mean(self.otf, axis=0)
            wf_decon_ft = self.widefield_ft * get_wiener_filter(wf_otf, sig / wf_noise)
            # upsample to make fully comparable to reconstructed image
            self.widefield_deconvolution_ft = tools.resample_bandlimited_ft(wf_decon_ft, (2, 2))
        elif self.combine_bands_mode == "fairSIM":
            # wf_decon_ft = self.widefield_ft * get_wiener_filter(self.otf, 1/np.sqrt(self.wiener_parameter / self.nangles))

            self.widefield_deconvolution_ft = np.nansum(weights_decon[:, 0] * self.bands_shifted_ft[:, 0], axis=0) / \
                          (self.wiener_parameter**2 + np.nansum(np.abs(weights_decon[:, 0])**2, axis=0))
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

        # x = tools.get_fft_pos(self.upsample_fact * self.nx, self.dx / self.upsample_fact, centered=False, mode='symmetric')
        # y = tools.get_fft_pos(self.upsample_fact * self.ny, self.dy / self.upsample_fact, centered=False, mode='symmetric')

        # store expanded bands
        expanded = np.zeros((self.nangles, 2, self.ny * self.upsample_fact, self.nx * self.upsample_fact), dtype=complex)
        # exponential shift factor for the one band we will shift with FFT
        exp_factor = np.zeros((self.nangles, self.ny * self.upsample_fact, self.nx * self.upsample_fact), dtype=complex)
        # shift and filter components
        for ii in range(self.nangles):
            # loop over components [O(f)H(f), m*O(f - f_o)H(f), m*O(f + f_o)H(f)]
            # don't need to loop over m*O(f + f_o)H(f), since it is conjugate of m*O(f - f_o)H(f)

            # shift factor for m*O(f - f_o)H(f)
            exp_factor[ii] = np.exp(-1j * 2 * np.pi * (self.frqs[ii, 0] * np.expand_dims(fft.ifftshift(self.x_us), axis=0) +
                                                       self.frqs[ii, 1] * np.expand_dims(fft.ifftshift(self.y_us), axis=1)))
            for jj, band_ind in enumerate([0, 1]):
                # think expand->shift->deconvolve is better than deconvolve->expand->shift,
                # because it avoids the fx=0 and fy=0 line artifacts
                expanded[ii, jj] = tools.resample_bandlimited_ft(self.bands_unmixed_ft[ii, jj], (self.upsample_fact, self.upsample_fact))

        # get shifted bands
        bands_shifted_ft = np.zeros((self.nangles, self.nbands, self.ny * self.upsample_fact, self.nx * self.upsample_fact), dtype=complex)
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

        # upsampled frequency data
        dfx_us = self.fx_us[1] - self.fx_us[0]
        dfy_us = self.fy_us[1] - self.fy_us[0]

        otf_shifted = np.zeros((self.nangles, self.nbands, self.ny * self.upsample_fact, self.nx * self.upsample_fact), dtype=complex)
        for ii in range(self.nangles):
            otf_us = tools.resample_bandlimited_ft(self.otf[ii], (self.upsample_fact, self.upsample_fact)) / self.upsample_fact / self.upsample_fact

            for jj, band_ind in enumerate(self.band_inds):
                # compute otf(k + m * ko)
                otf_shifted[ii, jj], _ = tools.translate_pix(otf_us, self.frqs[ii] * band_ind, dr=(dfx_us, dfy_us), axes=(1, 0), wrap=False)

        return otf_shifted

    def get_weights_open_sim(self):
        """
        Combine bands O(f)otf(f), O(f-fo)otf(f), and O(f+fo)otf(f) to do SIM reconstruction.

        Following the OpenSIM approach of https://doi.org/10.1109/jstqe.2016.2521542

        :return sim_sr_ft, bands_shifted_ft, weights, weight_norm, snr, snr_shifted:
        """

        # upsampled frequency data
        def ff_shift_upsample(f): return np.sqrt((self.fx_us[None, :] - f[0]) ** 2 + (self.fy_us[:, None] - f[1]) ** 2)

        # useful parameters
        snr_shifted = np.zeros(self.bands_shifted_ft.shape)
        wiener_filters = np.zeros(self.bands_shifted_ft.shape, dtype=complex)

        # shift and filter components
        for ii in range(self.nangles):
            # loop over components, O(f)H(f), m*O(f - f_o)H(f), m*O(f + f_o)H(f)
            for jj, band_ind in enumerate(self.band_inds):
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
        results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
            joblib.delayed(fit_modulation_frq)(
                fts1[ii, 0], fts2[ii, 0], self.dx, frq_guess=frq_guess[ii])
            for ii in range(nangles)
        )

        frqs, _, _ = zip(*results)
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

            results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
                joblib.delayed(get_phase_wicker_iterative)(
                    self.imgs_ft[ii], self.otf[ii], self.frqs[ii], self.dx, self.fmax,
                    phases_guess=phase_guess[ii],
                    fit_amps=self.determine_amplitudes) for ii in range(self.nangles))
            phases, amps, _ = zip(*results)
            phases = np.asarray(phases)
            amps = np.asarray(amps)

            # for ii in range(self.nangles):
            #     phases[ii], amps[ii], _ = get_phase_wicker_iterative(self.imgs_ft[ii], self.otf[ii], self.frqs[ii], self.dx, self.fmax,
            #                                                       phases_guess=phase_guess[ii],
            #                                                       fit_amps=self.determine_amplitudes)

        elif self.phase_estimation_mode == "real-space":
            if phase_guess is None:
                phase_guess = np.zeros((self.nangles, self.nphases))

            # joblib a little messy because have to map one index to two
            results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
                joblib.delayed(get_phase_realspace)(
                    self.imgs[np.unravel_index(aa, [self.nangles, self.nphases])],
                    self.frqs[np.unravel_index(aa, [self.nangles, self.nphases])[0]], self.dx,
                    phase_guess=phase_guess[np.unravel_index(aa, [self.nangles, self.nphases])], origin="center"
                ) for aa in range(self.nangles * self.nphases))

            phases = np.reshape(np.array(results), [self.nangles, self.nphases])
            amps = np.ones((self.nangles, self.nphases))

        elif self.phase_estimation_mode == "naive":
            # phases = np.zeros((self.nangles, self.nphases))
            # for ii in range(self.nangles):
            #     for jj in range(self.nphases):
            #         phases[ii, jj] = np.angle(tools.get_peak_value(self.imgs_ft[ii, jj], self.fx, self.fy, self.frqs[ii], peak_pixel_size=2))
            phases = self.peak_phases

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

        otf_mean = np.mean(self.otf, axis=0)
        fit_results_avg, mask_avg = fit_power_spectrum(component_zero, otf_mean, self.fx, self.fy,
                                                       self.fmax, self.fbounds,
                                                       init_params=[None, None, 1, noise],
                                                       fixed_params=[False, False, True, True])
        fit_params_avg = fit_results_avg['fit_params']
        exponent = fit_params_avg[1]

        # now fit each other component using this same exponent
        power_spectrum_params = np.zeros((self.nangles, self.nphases, 4))
        masks = np.zeros(self.bands_unmixed_ft.shape, dtype=bool)
        for ii in range(self.nangles):
            for jj in range(self.nphases):

                if jj == 0:
                    # for unshifted components, fit the amplitude
                    init_params = [None, exponent, 1, self.noise_power_bands[ii, jj]]
                    fixed_params = [False, True, True, True]

                    fit_results, masks[ii, jj] = fit_power_spectrum(self.bands_unmixed_ft[ii, jj], self.otf[ii],
                                                                    self.fx, self.fy, self.fmax, self.fbounds,
                                                                    init_params=init_params, fixed_params=fixed_params)
                    power_spectrum_params[ii, jj] = fit_results['fit_params']
                elif jj == 1:
                    # for shifted components, fit the modulation factor
                    init_params = [power_spectrum_params[ii, 0, 0], exponent, 0.5, self.noise_power_bands[ii, jj]]
                    fixed_params = [True, True, False, True]

                    fit_results, masks[ii, jj] = fit_power_spectrum(self.bands_unmixed_ft[ii, jj], self.otf[ii],
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
        self.print_log("'%s' frequency estimation mode" % self.frq_estimation_mode)
        self.print_log("'%s' phase estimation mode" % self.phase_estimation_mode)
        self.print_log("'%s' band combination mode" % self.combine_bands_mode)
        self.print_log("excluded %0.2f from bands around centers" % self.fmax_exclude_band0)
        self.print_log("wiener parameter = %0.2f" % self.wiener_parameter)

        for ii in range(self.nangles):
            self.print_log("################ Angle %d ################" % ii)

            # intensity info
            self.print_log("relative intensity to max angle = %0.3f" %
                           (np.mean(self.mean_intensities[ii]) / np.max(np.mean(self.mean_intensities, axis=1))))
            self.print_log("phase relative intensities = ", end="")
            for jj in range(self.nphases):
                self.print_log("%0.3f, " % (self.mean_intensities[ii, jj] / np.max(self.mean_intensities[ii])), end="")
            self.print_log("", end="\n")

            # amplitudes
            self.print_log("amps = ", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%05.3f, " % (self.amps[ii, jj]), end="")
            self.print_log("%05.3f" % (self.amps[ii, self.nphases - 1]))

            #  peak-to-noise ratio
            self.print_log("peak-to-camera-noise ratios = %0.3f, %0.3f, %0.3f" % tuple(self.p2nr[ii]))

            # modulation depth
            self.print_log("modulation depth = %0.3f" % self.mod_depths[ii])

            # frequency and period data
            if self.frqs_guess is not None:
                angle_guess = np.angle(self.frqs_guess[ii, 0] + 1j * self.frqs_guess[ii, 1])
                period_guess = 1 / np.linalg.norm(self.frqs_guess[ii])

                self.print_log("Frequency guess= ({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                    self.frqs_guess[ii, 0], self.frqs_guess[ii, 1], period_guess * 1e3, angle_guess * 180 / np.pi, 2 * np.pi))

            self.print_log("Frequency fit  = ({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                self.frqs[ii, 0], self.frqs[ii, 1], self.periods[ii] * 1e3, self.angles[ii] * 180 / np.pi))

            # phase information
            self.print_log("peaks   = ", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%07.2fdeg, " % (np.mod(self.peak_phases[ii, jj], 2*np.pi) * 180 / np.pi), end="")
            self.print_log("%07.2fdeg" % (np.mod(self.peak_phases[ii, self.nphases - 1], 2*np.pi) * 180 / np.pi))

            # self.print_log("phases  = ", end="")
            # for jj in range(self.nphases - 1):
            #     self.print_log("%07.2fdeg, " % (self.phases[ii, jj] * 180 / np.pi), end="")
            # self.print_log("%07.2fdeg" % (self.phases[ii, self.nphases - 1] * 180 / np.pi))

            # print corrected phases
            self.print_log("phases  = ", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%07.2fdeg, " % (np.mod(self.phases[ii, jj] - self.phase_corrections[ii], 2*np.pi) * 180 / np.pi), end="")
            self.print_log("%07.2fdeg" % (np.mod(self.phases[ii, self.nphases - 1] - self.phase_corrections[ii], 2*np.pi) * 180 / np.pi))

            if self.phases_guess is not None:
                self.print_log("guesses = ", end="")
                for jj in range(self.nphases - 1):
                    self.print_log("%07.2fdeg, " % (self.phases_guess[ii, jj] * 180 / np.pi), end="")
                self.print_log("%07.2fdeg" % (self.phases_guess[ii, self.nphases - 1] * 180 / np.pi))

            self.print_log("dpeaks  = ", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%07.2fdeg, " % (np.mod(self.peak_phases[ii, jj] - self.peak_phases[ii, 0], 2 * np.pi) * 180 / np.pi),
                               end="")
            self.print_log("%07.2fdeg" % (np.mod(self.peak_phases[ii, self.nphases - 1] - self.peak_phases[ii, 0], 2 * np.pi) * 180 / np.pi))

            self.print_log("dphases = ", end="")
            for jj in range(self.nphases - 1):
                self.print_log("%07.2fdeg, " % (np.mod(self.phases[ii, jj] - self.phases[ii, 0], 2 * np.pi) * 180 / np.pi),
                               end="")
            self.print_log("%07.2fdeg" %
                           (np.mod(self.phases[ii, self.nphases - 1] - self.phases[ii, 0], 2 * np.pi) * 180 / np.pi))

            if self.phases_guess is not None:
                self.print_log("dguesses= ", end="")
                for jj in range(self.nphases - 1):
                    self.print_log("%07.2fdeg, " % (np.mod(self.phases_guess[ii, jj] - self.phases_guess[ii, 0], 2*np.pi) * 180/np.pi), end="")
                self.print_log("%07.2fdeg" % (np.mod(self.phases_guess[ii, self.nphases - 1] - self.phases_guess[ii, 0], 2*np.pi) * 180/np.pi))

            # global phase correction
            self.print_log("global phase correction=%0.2fdeg" % (self.phase_corrections[ii] * 180 / np.pi))

    def print_log(self, string, **kwargs):
        """
        Print result to stdout and to a log file.

        :param string: string to print
        :param kwargs: passed through to print()
        """

        print(string, **kwargs)
        print(string, **kwargs, file=self.log)

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
            figh.savefig(self.save_dir / f"{self.save_prefix:s}mcnr{self.save_suffix:s}.png")
        if not self.interactive_plotting:
            plt.close(figh)

        # plot frequency fits
        fighs, fig_names = self.plot_frequency_fits(figsize=self.figsize)
        for fh, fn in zip(fighs, fig_names):
            if saving:
                fh.savefig(self.save_dir / f"{self.save_prefix:s}{fn:s}{self.save_suffix:s}.png")
            if not self.interactive_plotting:
                plt.close(fh)

        if self.combine_bands_mode == "openSIM":
            # plot power spectrum fits
            fighs, fig_names = self.plot_power_spectrum_fits(figsize=self.figsize)
            for fh, fn in zip(fighs, fig_names):
                if saving:
                    fh.savefig(self.save_dir / f"{self.save_prefix:s}{fn:s}{self.save_suffix:s}.png")
                if not self.interactive_plotting:
                    plt.close(fh)

            # widefield power spectrum fit
            otf_mean = np.mean(self.otf, axis=0)
            figh = plot_power_spectrum_fit(self.widefield_ft, otf_mean,
                                           {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                           self.pspec_params_wf, mask=self.mask_wf, figsize=self.figsize,
                                           ttl_str="Widefield power spectrum")
            if saving:
                figh.savefig(self.save_dir / f"{self.save_prefix:s}power_spectrum_widefield{self.save_suffix:s}.png")
            if not self.interactive_plotting:
                plt.close(figh)

        # plot filters used in reconstruction
        fighs, fig_names = self.plot_reconstruction_diagnostics(figsize=self.figsize)
        for fh, fn in zip(fighs, fig_names):
            if saving:
                fh.savefig(self.save_dir / f"{self.save_prefix:s}{fn:s}{self.save_suffix:s}.png")
            if not self.interactive_plotting:
                plt.close(fh)

        # plot reconstruction results
        fig = self.plot_reconstruction(figsize=self.figsize)
        if saving:
            fig.savefig(self.save_dir / f"{self.save_prefix:s}sim_reconstruction{self.save_suffix:s}.png", dpi=400)
        if not self.interactive_plotting:
            plt.close(fig)

        # plot otf
        fig = self.plot_otf(figsize=self.figsize)
        if saving:
            fig.savefig(self.save_dir / f"{self.save_prefix:s}otf{self.save_suffix:s}.png")
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
        # x = tools.get_fft_pos(self.nx, dt=self.dx)
        # y = tools.get_fft_pos(self.ny, dt=self.dy)

        extent = tools.get_extent(self.y, self.x)

        # parameters for real space plot
        vmin = np.percentile(self.imgs.ravel(), 0.1)
        vmax = np.percentile(self.imgs.ravel(), 99.9)

        # to avoid errors with image that has only one value
        if vmax <= vmin:
            vmax += 1e-12

        # ########################################
        # plot
        # ########################################
        figh = plt.figure(figsize=figsize)
        figh.suptitle('SIM signal diagnostic')

        n_factor = 4 # colorbar will be 1/n_factor of images
        # 5 types of plots + 2 colorbars
        grid = figh.add_gridspec(self.nangles, n_factor * (self.nphases + 2) + 3)
        
        mean_int = np.mean(self.imgs, axis=(2, 3))
        rel_int_phases = mean_int / np.expand_dims(np.max(mean_int, axis=1), axis=1)
        
        mean_int_angles = np.mean(self.imgs, axis=(1, 2, 3))
        rel_int_angles = mean_int_angles / np.max(mean_int_angles)

        # maximum mcnr value
        vmax_mcnr = np.percentile(self.mcnr, 99)

        for ii in range(self.nangles):
            for jj in range(self.nphases):

                # ########################################
                # raw real-space SIM images
                # ########################################
                ax = figh.add_subplot(grid[ii, n_factor*jj : n_factor*(jj+1)])
                ax.imshow(self.imgs[ii, jj], vmin=vmin, vmax=vmax, extent=extent, interpolation=None, cmap="bone")

                if ii == 0:
                    ax.set_title("phase %d" % jj)
                if jj == 0:
                    tstr = 'angle %d, relative intensity=%0.3f\nphase int=' % (ii, rel_int_angles[ii])
                    for aa in range(self.nphases):
                        tstr += "%0.3f, " % rel_int_phases[ii, aa]
                    ax.set_ylabel(tstr)
                if ii == (self.nangles - 1):
                    ax.set_xlabel("Position (um)")

                if jj != 0:
                    ax.set_yticks([])

            # ########################################
            # histograms of real-space images
            # ########################################
            nbins = 50
            bin_edges = np.linspace(0, np.percentile(self.imgs, 99), nbins + 1)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            ax = figh.add_subplot(grid[ii, n_factor*self.nphases + 2 : n_factor*(self.nphases+1) + 2])
            for jj in range(self.nphases):
                histogram, _ = np.histogram(self.imgs[ii, jj].ravel(), bins=bin_edges)
                ax.semilogy(bin_centers, histogram)
                ax.set_xlim([0, bin_edges[-1]])

            ax.set_yticks([])
            if ii == 0:
                ax.set_title("image histogram\nmedian %0.1f" % (np.median(self.imgs[ii, jj].ravel())))
            else:
                ax.set_title("median %0.1f" % (np.median(self.imgs[ii, jj].ravel())))
            if ii != (self.nangles - 1):
                ax.set_xticks([])
            else:
                ax.set_xlabel("counts")


            # ########################################
            # spatially resolved mcnr
            # ########################################
            ax = figh.add_subplot(grid[ii, n_factor*(self.nphases + 1) + 2 : n_factor*(self.nphases+2) + 2])
            if vmax_mcnr <= 0:
                vmax_mcnr += 1e-12

            im = ax.imshow(self.mcnr[ii], vmin=0, vmax=vmax_mcnr, cmap="inferno")
            ax.set_xticks([])
            ax.set_yticks([])
            if ii == 0:
                ax.set_title("mcnr")

        # colorbar for images
        ax = figh.add_subplot(grid[:, n_factor*self.nphases])
        norm = PowerNorm(vmin=vmin, vmax=vmax, gamma=1)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="bone"), cax=ax)

        # colorbar for MCNR
        ax = figh.add_subplot(grid[:, n_factor*(self.nphases + 2) + 2])
        norm = PowerNorm(vmin=0, vmax=vmax_mcnr, gamma=1)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="inferno"), cax=ax, label="MCNR")

        return figh

    def plot_reconstruction(self, figsize=(20, 10)):
        """
        Plot SIM image and compare with 'widefield' image
        :return:
        """

        extent_wf = tools.get_extent(self.fy, self.fx)
        extent_rec = tools.get_extent(self.fy_us, self.fx_us)

        # x = tools.get_fft_pos(self.nx, dt=self.dx)
        # y = tools.get_fft_pos(self.ny, dt=self.dy)
        extent_wf_real = tools.get_extent(self.y, self.x)

        # x_us = tools.get_fft_pos(self.upsample_fact * self.nx, dt=self.dx / self.upsample_fact)
        # y_us = tools.get_fft_pos(self.upsample_fact * self.ny, dt=self.dy / self.upsample_fact)
        extent_us_real = tools.get_extent(self.y_us, self.x_us)


        gamma = 0.1
        min_percentile = 0.1
        max_percentile = 99.9

        # create plot
        figh = plt.figure(figsize=figsize)
        grid = figh.add_gridspec(2, 4)
        figh.suptitle("SIM reconstruction, NA=%0.2f, wavelength=%.0fnm\n"
                      "wiener parameter=%0.2f, phase estimation mode '%s', frq estimation mode '%s'\n"
                      "band combination mode '%s', band replacement using %0.2f of fmax" %
                      (self.na, self.wavelength * 1e3,
                       self.wiener_parameter, self.phase_estimation_mode, self.frq_estimation_mode,
                       self.combine_bands_mode, self.fmax_exclude_band0))

        # widefield, real space
        ax = figh.add_subplot(grid[0, 0])

        vmin = np.percentile(self.widefield.ravel(), min_percentile)
        vmax = np.percentile(self.widefield.ravel(), max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(self.widefield, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_wf_real)
        ax.set_title('widefield')
        ax.set_xlabel('x-position ($\mu m$)')
        ax.set_ylabel('y-position ($\mu m$)')

        # deconvolved, real space
        ax = figh.add_subplot(grid[0, 1])

        vmin = np.percentile(self.widefield_deconvolution.ravel(), min_percentile)
        vmax = np.percentile(self.widefield_deconvolution.ravel(), max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(self.widefield_deconvolution, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_us_real)
        ax.set_title('widefield deconvolved')
        ax.set_xlabel('x-position ($\mu m$)')

        # SIM, realspace
        ax = figh.add_subplot(grid[0, 2])
        vmin = np.percentile(self.sim_sr.ravel()[self.sim_sr.ravel() >= 0], min_percentile)
        vmax = np.percentile(self.sim_sr.ravel()[self.sim_sr.ravel() >= 0], max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(self.sim_sr, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_us_real)
        ax.set_title('SR-SIM')
        ax.set_xlabel('x-position ($\mu m$)')

        #
        ax = figh.add_subplot(grid[0, 3])
        vmin = np.percentile(self.sim_os.ravel(), min_percentile)
        vmax = np.percentile(self.sim_os.ravel(), max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(self.sim_os, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_wf_real)
        ax.set_title('OS-SIM')
        ax.set_xlabel('x-position ($\mu m$)')

        # widefield Fourier space
        ax = figh.add_subplot(grid[1, 0])
        ax.imshow(np.abs(self.widefield_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_wf, cmap="bone")

        ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))
        ax.add_artist(Circle((0, 0), radius=2 * self.fmax, color='r', fill=False, ls='--'))

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])
        ax.set_xlabel("$f_x (1/\mu m)$")
        ax.set_ylabel("$f_y (1/\mu m)$")

        # deconvolution Fourier space
        ax = figh.add_subplot(grid[1, 1])
        ax.imshow(np.abs(self.widefield_deconvolution_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_rec, cmap="bone")

        ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))
        ax.add_artist(Circle((0, 0), radius=2 * self.fmax, color='r', fill=False, ls='--'))

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])
        ax.set_xlabel("$f_x (1/\mu m)$")

        # SIM fourier space
        ax = figh.add_subplot(grid[1 ,2])
        ax.imshow(np.abs(self.sim_sr_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_rec, cmap="bone")

        ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))
        ax.add_artist(Circle((0, 0), radius=2 * self.fmax, color='r', fill=False, ls='--'))

        # actual maximum frequency based on real SIM frequencies
        for ii in range(self.nangles):
            ax.add_artist(Circle((0, 0), radius=self.fmax + 1/self.periods[ii], color='g', fill=False, ls='--'))

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])
        ax.set_xlabel("$f_x (1/\mu m)$")

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

        # todo: may need some updates. Check upsample functions in analysis_tools
        start_coord_re = [2*c for c in start_coord]
        end_coord_re = [2*c for c in end_coord]

        # get cut from widefield image
        coord_wf, cut_wf = tools.get_linecut(self.widefield, start_coord, end_coord, 1)
        coord_wf = self.dx * coord_wf

        coord_os, cut_os = tools.get_linecut(self.sim_os, start_coord, end_coord, 1)
        coord_os = self.dx * coord_os

        coord_dc, cut_dc = tools.get_linecut(self.widefield_deconvolution, start_coord_re, end_coord_re, 1)
        coord_dc = 0.5 * self.dx * coord_dc

        coord_sr, cut_sr = tools.get_linecut(self.sim_sr, start_coord_re, end_coord_re, 1)
        coord_sr = 0.5 * self.dx * coord_sr

        coords = {'wf': coord_wf, 'os': coord_os, 'dc': coord_dc, 'sr': coord_sr}
        cuts = {'wf': cut_wf, 'os': cut_os, 'dc': cut_dc, 'sr': cut_sr}

        figh = plt.figure(figsize=figsize)

        phs = []
        pnames = []

        ax = figh.add_subplot(1, 2, 1)
        ph, = ax.plot(coord_sr, cut_sr)
        phs.append(ph)
        pnames.append('SR-SIM')

        if plot_os_sim:
            ph, = ax.plot(coord_os, cut_os)
            phs.append(ph)
            pnames.append('OS-SIM')

        if plot_deconvolution:
            ph, = ax.plot(coord_dc, cut_dc)
            phs.append(ph)
            pnames.append('deconvolved')

        ph, = ax.plot(coord_wf, cut_wf)
        phs.append(ph)
        pnames.append('widefield')

        ax.set_xlabel("Position (um)")
        ax.set_ylabel("ADC")
        ax.legend(phs, pnames)

        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])

        ax = figh.add_subplot(1, 2, 2)
        vmin = np.percentile(self.widefield.ravel(), 2)
        vmax = np.percentile(self.widefield.ravel(), 99.5)

        ax.imshow(self.widefield, vmin=vmin, vmax=vmax, cmap="bone")
        ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], 'white')
        ax.set_title('widefield')

        return figh, coords, cuts

    def plot_reconstruction_diagnostics(self, figsize=(20, 10)):
        """
        Plot deconvolved components and related information to check reconstruction

        :return figh: figure handle
        """
        figs = []
        fig_names = []

        # ######################################
        # plot different stages of inversion process as diagnostic
        # ######################################
        extent = tools.get_extent(self.fy, self.fx)
        extent_upsampled = tools.get_extent(self.fy_us, self.fx_us)
        extent_upsampled_real = tools.get_extent(self.y_us, self.x_us)

        # plot one image for each angle
        for ii in range(self.nangles):
            dp1 = np.mod(self.phases[ii, 1] - self.phases[ii, 0], 2 * np.pi)
            dp2 = np.mod(self.phases[ii, 2] - self.phases[ii, 0], 2 * np.pi)
            p0 = self.phases[ii, 0]
            p1 = self.phases[ii, 1]
            p2 = self.phases[ii, 2]
            p0_corr = np.mod(self.phases[ii, 0] - self.phase_corrections[ii], 2*np.pi)
            p1_corr = np.mod(self.phases[ii, 1] - self.phase_corrections[ii], 2*np.pi)
            p2_corr = np.mod(self.phases[ii, 2] - self.phase_corrections[ii], 2*np.pi)

            fig = plt.figure(figsize=figsize)
            fig.suptitle(f'SIM bands Fourier space diagnostic, angle {ii:d}\n'
                         f'period={self.periods[ii] * 1e3:.3f}nm at '
                         f'{self.angles[ii] * 180 / np.pi:.2f}deg={self.angles[ii]:.3f}rad,'
                         f' f=({self.frqs[ii, 0]:.3f},{self.frqs[ii, 1]:.3f}) 1/um\n'
                         f'modulation contrast={self.mod_depths[ii]:.3f}, min p2nr={np.min(self.p2nr[ii]):.3f},'
                         f' $\eta$={self.wiener_parameter:.2f},'
                         f' global phase correction={self.phase_corrections[ii] * 180 / np.pi:.2f}deg\n'                         
                         f' corrected phases (deg) = {p0_corr * 180 / np.pi:.2f}, {p1_corr * 180 / np.pi:.2f}, {p2_corr * 180 / np.pi:.2f};'
                         f' phase diffs (deg) ={0:.2f}, {dp1 * 180/np.pi:.2f}, {dp2 * 180/np.pi:.2f}')

            n_factor = 5
            grid = fig.add_gridspec(self.nphases, 5 * n_factor + 2, wspace=3)

            for jj in range(self.nphases):

                # ####################
                # raw images at different phases
                # ####################
                ax = fig.add_subplot(grid[jj, 0:n_factor])
                ax.set_title("Raw data, phase %d" % jj)

                to_plot = np.abs(self.imgs_ft[ii, jj])
                to_plot[to_plot <= 0] = np.nan

                im = ax.imshow(to_plot, norm=LogNorm(), extent=extent, cmap="bone")

                ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')

                ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))

                ax.set_xlim([-2*self.fmax, 2*self.fmax])
                ax.set_ylim([2*self.fmax, -2*self.fmax])

                ax.set_xticks([])
                ax.set_yticks([])

                if jj == (self.nphases - 1):
                    ax.set_xlabel("$f_x$")
                ax.set_ylabel("$f_y$")

                # ####################
                # separated components
                # ####################
                ax = plt.subplot(grid[jj, n_factor:2*n_factor])

                to_plot = np.abs(self.bands_unmixed_ft[ii, jj])
                to_plot[to_plot <= 0] = np.nan

                im = ax.imshow(to_plot, norm=LogNorm(), extent=extent, cmap="bone")
                clim = im.get_clim()
                if clim[0] < 1e-12 and clim[1] < 1e-12:
                    clim = (0, 1)
                im.set_clim(clim)

                ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=0, ls='--'))

                if jj == 0:
                    ax.set_title('O(f)otf(f)')
                    ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                    ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
                elif jj == 1:
                    ax.set_title('m*O(f-fo)otf(f)')
                    ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                elif jj == 2:
                    ax.set_title('m*O(f+fo)otf(f)')
                    ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
                if jj == (self.nphases - 1):
                    ax.set_xlabel("$f_x$")

                ax.set_xticks([])
                ax.set_yticks([])

                ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
                ax.set_ylim([2 * self.fmax, -2 * self.fmax])

                # ####################
                # shifted component
                # ####################
                ax = fig.add_subplot(grid[jj, 2*n_factor:3*n_factor])

                # avoid any zeros for LogNorm()
                cs_ft_toplot = np.abs(self.bands_shifted_ft[ii, jj])
                cs_ft_toplot[cs_ft_toplot <= 0] = np.nan

                im = ax.imshow(cs_ft_toplot, norm=LogNorm(), extent=extent_upsampled, cmap="bone")

                # to keep same color scale, must correct for upsampled normalization change
                im.set_clim(tuple([4 * c for c in clim]))

                ax.scatter(0, 0, edgecolor='k', facecolor='none')

                ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))

                if jj == 0:
                    ax.set_title('shifted component')
                    ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                    ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
                if jj == 1:
                    ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
                    ax.add_artist(Circle(-self.frqs[ii], radius=self.fmax, color='r', fill=0, ls='--'))
                elif jj == 2:
                    ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                    ax.add_artist(Circle(self.frqs[ii], radius=self.fmax, color='r', fill=0, ls='--'))
                if jj == (self.nphases - 1):
                    ax.set_xlabel("$f_x$")

                ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
                ax.set_ylim([2 * self.fmax, -2 * self.fmax])
                ax.set_xticks([])
                ax.set_yticks([])

                # ####################
                # unnormalized weights
                # ####################
                ax = fig.add_subplot(grid[jj, 3*n_factor:4*n_factor])
                if jj == 0:
                    ax.set_title(r"$w(k)$")

                im2 = ax.imshow(np.abs(self.weights[ii, jj]), norm=PowerNorm(gamma=0.1, vmin=0), extent=extent_upsampled, cmap="bone")
                im2.set_clim([1e-5, 1])

                ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=0, ls='--'))
                if jj == 1:
                    ax.add_artist(Circle(-self.frqs[ii], radius=self.fmax, color='r', fill=0, ls='--'))
                elif jj == 2:
                    ax.add_artist(Circle(self.frqs[ii], radius=self.fmax, color='r', fill=0, ls='--'))

                if jj == (self.nphases - 1):
                    ax.set_xlabel("$f_x$")

                ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
                ax.set_ylim([2 * self.fmax, -2 * self.fmax])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                ax = fig.add_subplot(grid[jj, 4*n_factor])
                fig.colorbar(im2, cax=ax, format="%0.2g", ticks=[1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])

                # ####################
                # normalized weights
                # ####################
                ax = fig.add_subplot(grid[jj, 4*n_factor + 1: 5*n_factor + 1])
                if jj == 0:
                    ax.set_title(r"$\frac{w_i(k)}{\sum_j |w_j(k)|^2 + \eta^2}$")

                im2 = ax.imshow(np.abs(self.weights[ii, jj] / self.weights_norm), norm=PowerNorm(gamma=0.1, vmin=0), extent=extent_upsampled, cmap="bone")
                im2.set_clim([1e-5, 10])

                ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=0, ls='--'))
                if jj == 1:
                    ax.add_artist(Circle(-self.frqs[ii], radius=self.fmax, color='r', fill=0, ls='--'))
                elif jj == 2:
                    ax.add_artist(Circle(self.frqs[ii], radius=self.fmax, color='r', fill=0, ls='--'))

                if jj == (self.nphases - 1):
                    ax.set_xlabel("$f_x$")

                ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
                ax.set_ylim([2 * self.fmax, -2 * self.fmax])

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])

                # colorbar
                ax = fig.add_subplot(grid[jj, 5*n_factor + 1])
                fig.colorbar(im2, cax=ax, format="%0.2g", ticks=[10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])

            figs.append(fig)
            fig_names.append(f"band_ft_diagnostic_angle={ii:d}")

            # ######################################
            # plot real space version of 0th and +/- 1st bands
            # ######################################
            figh = plt.figure(figsize=figsize)
            figh.suptitle(f"SIM bands real space diagnostic, angle {ii:d}")

            band0 = fft.fftshift(fft.ifft2(fft.ifftshift(self.sim_sr_ft_components[ii, 0]))).real
            band1 = fft.fftshift(fft.ifft2(fft.ifftshift(self.sim_sr_ft_components[ii, 1] + self.sim_sr_ft_components[ii, 2]))).real

            color0 = np.array([0, 1, 1]) * 0.75 # cyan
            color1 = np.array([1, 0, 1]) * 0.75 # magenta

            vmax0 = np.percentile(band0, 99.9)
            vmin0 = 0

            vmax1 = np.percentile(band1, 99.9)
            vmin1 = np.percentile(band1, 5)

            img0 = (np.expand_dims(band0, axis=-1) - vmin0) / (vmax0 - vmin0) * np.expand_dims(color0, axis=(0, 1))
            img1 = (np.expand_dims(band1, axis=-1) - vmin1) / (vmax1 - vmin1) * np.expand_dims(color1, axis=(0, 1))

            ax = figh.add_subplot(1, 3, 1)
            ax.set_title("combined")

            ax.imshow(img0 + img1, extent=extent_upsampled_real)
            ax.set_xlabel("x-position ($\mu m$)")
            ax.set_ylabel("y-position ($\mu m$)")

            ax = figh.add_subplot(1, 3, 2)
            ax.set_title("band 0")
            ax.imshow(img0, extent=extent_upsampled_real)
            ax.set_xlabel("x-position ($\mu m$)")
            ax.set_ylabel("y-position ($\mu m$)")

            ax = figh.add_subplot(1, 3, 3)
            ax.set_title("band 1")
            ax.imshow(img1, extent=extent_upsampled_real)
            ax.set_xlabel("x-position ($\mu m$)")
            ax.set_ylabel("y-position ($\mu m$)")

            figs.append(figh)
            fig_names.append(f"band_real_space_diagnostic_angle={ii:d}")

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
            fig = plot_power_spectrum_fit(self.bands_unmixed_ft[ii, 0], self.otf[ii],
                                          {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                          self.power_spectrum_params[ii, 0], frq_sim=(0, 0), mask=self.power_spectrum_masks[ii, 0],
                                          figsize=figsize, ttl_str=f"Unshifted component, angle {ii:d}")
            debug_figs.append(fig)
            debug_fig_names.append(f"power_spectrum_unshifted_component_angle={ii:d}")

            fig = plot_power_spectrum_fit(self.bands_unmixed_ft[ii, 1], self.otf[ii],
                                          {'pixel_size': self.dx, 'wavelength': self.wavelength, 'na': self.na},
                                          self.power_spectrum_params[ii, 1], frq_sim=self.frqs[ii], mask=self.power_spectrum_masks[ii, 1],
                                          figsize=figsize, ttl_str=f"Shifted component, angle {ii:d}")

            debug_figs.append(fig)
            debug_fig_names.append(f"power_spectrum_shifted_component_angle={ii:d}")

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
                                            ttl_str=f"Correlation fit, angle {ii:d}")
                figs.append(figh)
                fig_names.append(f"frq_fit_angle={ii:d}_phase={0:d}")
            else:
                figh = plot_correlation_fit(self.bands_unmixed_ft[ii, 0],
                                            self.bands_unmixed_ft[ii, 1], self.frqs[ii, :],
                                            self.dx, self.fmax, frqs_guess=frqs_guess[ii], figsize=figsize,
                                            ttl_str=f"Correlation fit, angle {ii:d}, unmixing phases = {self.phases_guess[ii]}")
                figs.append(figh)
                fig_names.append(f"frq_fit_angle={ii:d}")

        return figs, fig_names

    def plot_otf(self, figsize=(20, 10)):
        """
        Plot optical transfer function (OTF) versus frequency. Compare with ideal OTF at the same NA, and show
        location of SIM frequencies
        :param figsize:
        :return:
        """
        figh = plt.figure(figsize=figsize)
        tstr = "OTF diagnostic\nvalue at frqs="
        for ii in range(self.nangles):
            tstr += f" {self.otf_at_frqs[ii]:.3f},"
        figh.suptitle(tstr)

        ff = np.sqrt(np.expand_dims(self.fx, axis=0) ** 2 + np.expand_dims(self.fy, axis=1) ** 2)

        otf_ideal = localize_psf.fit_psf.circ_aperture_otf(ff, 0, self.na, self.wavelength)

        # 1D plots
        ax = figh.add_subplot(1, 2, 1)
        ax.set_title("1D OTF")
        ax.set_xlabel("Frequency (1/um)")
        ax.set_ylabel("OTF")
        # plot real OTF's per angle
        for ii in range(self.nangles):
            ax.plot(ff.ravel(), self.otf[ii].ravel(), label=f"OTF, angle {ii:d}")
        # plot ideal OTF
        ax.plot(ff.ravel(), otf_ideal.ravel(), label="OTF ideal")
        ax.set_xlim([0, 1.2 * self.fmax])
        ylim = ax.get_ylim()

        # plot SIM frequencies
        # todo: color code to match with OTFs
        fs = np.linalg.norm(self.frqs, axis=1)
        for ii in range(self.nangles):
            if ii == 0:
                ax.plot([fs[ii], fs[ii]], ylim, 'k', label="SIM frqs")
            else:
                ax.plot([fs[ii], fs[ii]], ylim, 'k')

        ax.set_ylim(ylim)
        ax.legend()

        # 2D plot
        ax = figh.add_subplot(1, 2, 2)
        ax.set_title("Mean 2D OTF")
        ax.imshow(np.mean(self.otf, axis=0), extent=tools.get_extent(self.fy, self.fx), cmap="bone")
        ax.scatter(self.frqs[:, 0], self.frqs[:, 1], color='r', marker='o')
        ax.scatter(-self.frqs[:, 0], -self.frqs[:, 1], color='r', marker='o')
        ax.set_xlabel("$f_x (1/\mu m)$")
        ax.set_ylabel("$f_y (1/\mu m)$")
        ax.set_xlim([-self.fmax, self.fmax])
        ax.set_ylim([-self.fmax, self.fmax])

        return figh

    # saving utility functions
    def save_imgs(self, save_dir=None, start_time=None, save_suffix=None, save_prefix=None):
        tstart_save = time.perf_counter()

        if save_dir is None:
            save_dir = self.save_dir

        save_dir = Path(save_dir)

        if save_suffix is None:
            save_suffix = self.save_suffix

        if save_prefix is None:
            save_prefix = self.save_prefix

        if save_dir is not None:

            if start_time is None:
                kwargs = {}
            else:
                kwargs = {"datetime": start_time}

            dxy_wf = self.dx
            if self.upsample_widefield:
                dxy_wf = self.dx / self.upsample_fact

            fname = save_dir / f"{save_prefix:s}mcnr{save_suffix:s}.tif"
            tifffile.imwrite(fname, self.mcnr.astype(np.float32), imagej=True,
                             resolution=(1/self.dx, 1/self.dx),
                             metadata={'Info': 'modulation-contrast to noise ratio', 'unit': 'um'}, **kwargs)

            fname = save_dir / f"{save_prefix:s}sim_os{save_suffix:s}.tif"
            tifffile.imwrite(fname, self.sim_os.astype(np.float32), imagej=True,
                             resolution=(1 / dxy_wf, 1 / dxy_wf),
                             metadata={'Info': 'SIM optical-sectioning', 'unit': 'um'}, **kwargs)

            fname = save_dir / f"{save_prefix:s}widefield{save_suffix:s}.tif"
            tifffile.imwrite(fname, self.widefield.astype(np.float32), imagej=True,
                             resolution=(1 / dxy_wf, 1 / dxy_wf),
                             metadata={'Info': 'widefield', 'unit': 'um'}, **kwargs)

            fname = save_dir / f"{save_prefix:s}sim_sr{save_suffix:s}.tif"
            tifffile.imwrite(fname, self.sim_sr.astype(np.float32), imagej=True,
                             resolution=(1/(self.dx / self.upsample_fact), 1/(self.dx/ self.upsample_fact)),
                             metadata={'Info': 'SIM super-resolution', 'unit': 'um',
                                       'min': 0, 'max': np.percentile(self.sim_sr, 99.9)}, **kwargs)
            #{'Ranges': [[0.0, np.percentile(self.sim_sr, 99.9)]]}

            fname = save_dir / f"{save_prefix:s}deconvolved{save_suffix:s}.tif"
            tifffile.imwrite(fname, self.widefield_deconvolution.astype(np.float32), imagej=True,
                             resolution=(1/(self.dx / self.upsample_fact), 1/(self.dx / self.upsample_fact)),
                             metadata={'Info': 'Wiener deconvolved', 'unit': 'um'}, **kwargs)

            self.print_log(f"saving tiff files took {time.perf_counter() - tstart_save:.2f}s")

    def save_result(self, fname=None):
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
                              'fx_us', 'fy_us', 'fx', 'fy', 'x', 'y', 'x_us', 'y_us',
                              'mcnr', 'snr', 'otf', 'otf_shifted', 'power_spectrum_masks',
                              'log_file', 'mask_wf', 'log']
        # get dictionary object with images removed
        results_dict = {}
        for k, v in vars(self).items():
            if k in fields_to_not_save or k[0] == '_':
                continue

            if isinstance(v, np.ndarray):
                if v.size > 1e3:
                    continue
                results_dict[k] = v.tolist()
            elif isinstance(v, pathlib.Path):
                results_dict[k] = str(v)
            elif isinstance(v, io.IOBase):
                results_dict[k] = v.getvalue()
            else:
                results_dict[k] = v

        # get default file name
        if fname is None and self.save_dir is not None:
            fname = self.save_dir / f"{self.save_prefix:s}sim_reconstruction{self.save_suffix:s}.json"

        # save results to json file
        if fname is not None:
            with open(fname, "w") as f:
                json.dump(results_dict, f, indent="\t")

        self.print_log(f"saving results took {time.perf_counter() - tstart:.2f}s")

        return results_dict


def reconstruct_mm_sim_dataset(data_dirs, pixel_size, na, emission_wavelengths, excitation_wavelengths,
                               affine_data_paths, otf_data_path, dmd_pattern_data_path,
                               nangles=3, nphases=3, npatterns_ignored=0,
                               crop_rois=None, fit_all_sim_params=False, plot_diagnostics=True,
                               channel_inds=None, zinds_to_use=None, tinds_to_use=None, xyinds_to_use=None,
                               id_pattern="ic=%d_it=%d_ixy=%d_iz=%d", **kwargs):
    """
    Reconstruct folder of SIM data stored in TIF files. This function assumes the TIF files were generated
    using MicroManager and that the metadata has certain special user keys.

    This is an example of a helper function which loads a certain type of data and uses SimImageSet() to run a
    SIM reconstruction. For other types of data, the preferred approach is to write a function like this one.

    This function uses the MicroManager metadata to load the correct images. It is also responsible for loading
    other relevant data, such as affine transformations, OTF, SIM pattern information from .pkl files. It then
    processes the image and produces SIM superresolution images, deconvolved images, widefield images, and a variety
    of diagnostic plots which it then saves in a convenient directory structure.

    :param list[str] data_dirs: list of directories where data is stored
    :param float pixel_size: pixel size in ums
    :param float na: numerical aperture
    :param list[float] emission_wavelengths: list of emission wavelengths in um
    :param list[float] excitation_wavelengths: list of excitation wavelengths in um
    :param list[str] affine_data_paths: list of paths to files storing data about affine transformations between DMD and camera
    space. [path_color_0, path_color_1, ...]. The affine data files store pickled dictionary objects. The dictionary
    must have an entry 'affine_xform' which contains the affine transformation matrix (in homogeneous coordinates)
    :param str otf_data_path: path to file storing optical transfer function data. Data is a pickled dictionary object
    and must have entry 'fit_params'.
    :param list[str] dmd_pattern_data_path: list of paths to files storing data about DMD patterns for each color. Data is
    stored in a pickled dictionary object which must contain fields 'frqs', 'phases', 'nx', and 'ny'
    :param int nangles: number of angle images
    :param int nphases: number of phase images
    :param int npatterns_ignored: number of patterns to ignore at the start of each channel.
    :param list[list] crop_rois: [[ystart_0, yend_0, xstart_0, xend_0], [ystart_1, ...], ...]
    :param list img_centers: list of centers for images in each data directory to be used in cropping [[cy, cx], ...]
    :param list or int crop_sizes: list of crop sizes for each data directory
    :param list channel_inds: list of channel indices corresponding to each color. If set to None, will use [0, 1, ..., ncolors -1]
    :param list zinds_to_use: list of z-position indices to reconstruct
    :param list tinds_to_use: list of time indices to reconstruct
    :param list xyinds_to_use: list of xy-position indices to reconstruct
    :param **kwargs: passed through to reconstruction
    :return str sim_save_dir: directories where results were saved
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

    def otf_fn(f, fmax): return 1 / (1 + (f / fmax * otf_p[0]) ** 2) * localize_psf.fit_psf.circ_aperture_otf(f, 0, na, 2 * na / fmax)

    # ############################################
    # SIM images
    # ############################################
    save_dirs = []
    for rpath, roi in zip(data_dirs, crop_rois):
        folder_path, folder = os.path.split(rpath)
        print("# ################################################################################")
        print(f"analyzing folder: {folder:s}")
        print(f"located in: {folder_path:s}")

        tstamp =  datetime.datetime.now().strftime('%Y_%d_%m_%H;%M;%S')
        # now = datetime.datetime.now()
        # tstamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

        # path to store processed results
        sim_save_dir = os.path.join(rpath, '%s_sim_reconstruction' % tstamp)
        save_dirs.append(sim_save_dir)
        if not os.path.exists(sim_save_dir):
            os.mkdir(sim_save_dir)
        print(f"save directory: {sim_save_dir:s}")

        individual_image_save_dir = os.path.join(sim_save_dir, "individual_images")
        if not os.path.exists(individual_image_save_dir):
            os.mkdir(individual_image_save_dir)

        # copy useful data files to results dir
        for kk in range(ncolors):
            # copy affine data here
            _, fname = os.path.split(affine_data_paths[kk])
            fpath = os.path.join(sim_save_dir, fname)
            shutil.copyfile(affine_data_paths[kk], fpath)

            # copy otf data here
            _, fname = os.path.split(otf_data_path)
            fpath = os.path.join(sim_save_dir, fname)
            shutil.copyfile(otf_data_path, fpath)

            # copy DMD pattern data here
            _, fname = os.path.split(dmd_pattern_data_path[kk])
            fpath = os.path.join(sim_save_dir, fname)
            shutil.copyfile(dmd_pattern_data_path[kk], fpath)

        # load metadata
        metadata, dims, summary = mm_io.parse_mm_metadata(rpath)
        start_time = datetime.datetime.strptime(summary['StartTime'],  '%Y-%d-%m;%H:%M:%S.%f')
        nz = dims['z']
        nxy = dims['position']
        nt = dims['time']

        # z-plane spacing
        # unique_slices = np.unique(metadata["Slice"])
        unique_slices = np.unique(metadata["ZPositionUm"])
        unique_slices.sort()
        if len(unique_slices) > 1:
            dz = unique_slices[1] - unique_slices[0]
        else:
            dz = 1

        # use this construction as zinds can be different for different folders
        if zinds_to_use is None:
            zinds_to_use_temp = list(range(nz))
        else:
            zinds_to_use_temp = zinds_to_use
        nz_used = len(zinds_to_use_temp)

        if tinds_to_use is None:
            tinds_to_use_temp = list(range(nt))
        else:
            tinds_to_use_temp = tinds_to_use
        nt_used = len(tinds_to_use_temp)

        if xyinds_to_use is None:
            xyinds_to_use_temp = list(range(nxy))
        else:
            xyinds_to_use_temp = xyinds_to_use
        nxy_used = len(xyinds_to_use_temp)
        if nxy_used > 1:
            raise NotImplementedError("currently only implemented for one xy-position at a time")

        if pixel_size is None:
            pixel_size = metadata['PixelSizeUm'][0]

        # load metadata from one file to check size
        fname = os.path.join(rpath, metadata['FileName'].values[0])
        tif = tifffile.TiffFile(fname)
        ny_raw, nx_raw = tif.series[0].shape[-2:]

        if roi is None:
            roi = [0, ny_raw, 0, nx_raw]
        else:
            # check points don't exceed image size
            if roi[0] < 0:
                roi[0] = 0
            if roi[1] > ny_raw:
                roi[1] = ny_raw
            if roi[2] < 0:
                roi[2] = 0
            if roi[3] > nx_raw:
                roi[3] = nx_raw

        ny = roi[1] - roi[0]
        nx = roi[3] - roi[2]

        # timing
        tstart_all = time.perf_counter()
        counter = 1
        for kk in range(ncolors):
            # estimate otf
            fmax = 1 / (0.5 * emission_wavelengths[kk] / na)
            fx = fft.fftshift(fft.fftfreq(nx, pixel_size))
            fy = fft.fftshift(fft.fftfreq(ny, pixel_size))
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
            mod_depths_real = []
            frqs_real = []
            phases_real = []
            for ixy in xyinds_to_use_temp:
                for iz in zinds_to_use_temp:
                    for ind_t in tinds_to_use_temp:
                        tstart_single_index = time.perf_counter()

                        file_identifier = id_pattern % (kk, ind_t, ixy, iz)
                        identifier = "%.0fnm_%s" % (excitation_wavelengths[kk] * 1e3, file_identifier)

                        # where we will store results for this particular set
                        diagnostics_dir = os.path.join(sim_save_dir, identifier)
                        if not os.path.exists(diagnostics_dir):
                            os.mkdir(diagnostics_dir)

                        # find images and load them
                        img_inds = list(range(npatterns_ignored, npatterns_ignored + nangles * nphases))
                        raw_imgs = mm_io.read_mm_dataset(metadata, time_indices=ind_t, z_indices=iz, xy_indices=ixy,
                                                         user_indices={"UserChannelIndex": channel_inds[kk],
                                                                       "UserSimIndex": img_inds})

                        # error if we have wrong number of images
                        if np.shape(raw_imgs)[0] != (nangles * nphases):
                            raise ValueError("Found %d images, but expected %d images at channel=%d,"
                                            " zindex=%d, tindex=%d, xyindex=%d" %
                                             (np.shape(raw_imgs)[0], nangles * nphases, channel_inds[kk], iz, ind_t, ixy))

                        # reshape to [nangles, nphases, ny, nx]
                        imgs_sim = raw_imgs
                        imgs_sim = imgs_sim.reshape((nangles, nphases, raw_imgs.shape[1], raw_imgs.shape[2]))
                        imgs_sim = imgs_sim[:, :, roi[0]:roi[1], roi[2]:roi[3]]

                        # instantiate reconstruction object
                        if fit_all_sim_params or ind_t == 0:
                            img_set = SimImageSet({'pixel_size': pixel_size, 'wavelength': emission_wavelengths[kk], 'na': na},
                                                  imgs_sim, otf=otf, frq_guess=frqs_guess, phases_guess=phases_guess,
                                                  save_dir=diagnostics_dir, save_suffix="_%s" % file_identifier, **kwargs)
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
                            img_set = SimImageSet({'pixel_size': pixel_size, 'wavelength': emission_wavelengths[kk], 'na': na},
                                                  imgs_sim, frq_guess=frqs_real, otf=otf, phases_guess=phases_real,
                                                  mod_depths_guess=mod_depths_real,
                                                  save_dir=diagnostics_dir, save_suffix="_%s" % file_identifier,
                                                  **kwargs_reduced)
                            img_set.reconstruct()

                        if plot_diagnostics:
                            # plot results
                            img_set.plot_figs()

                        # save reconstruction summary data
                        img_set.save_result(os.path.join(diagnostics_dir, "sim_reconstruction_params.json"))
                        img_set.save_imgs(individual_image_save_dir, start_time, "_%s" % file_identifier)

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
        tstart_save = time.perf_counter()

        ch_labels = ["%.0fnm/%.0fnm" % (a * 1e3, b * 1e3) for a, b in zip(excitation_wavelengths, emission_wavelengths)]
        # colors = [np.array([0, 255, 255]), np.array([255, 0, 255]), np.array([255, 255, 0]),
        #           np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]
        # ts = np.linspace(0, 1, 255)
        #
        # luts = [(np.expand_dims(ts, axis=0) * np.expand_dims(c, axis=1)).astype(np.uint8) for c in colors]
        # luts = luts[:ncolors]

        stems = ["widefield", "deconvolved", "sim_os", "sim_sr"]
        res_factors = [1, 2, 1, 2]
        for st, rf in zip(stems, res_factors):
            fname_first = os.path.join(individual_image_save_dir, st + "_" +
                                       id_pattern % (0, tinds_to_use_temp[0], xyinds_to_use_temp[0], zinds_to_use_temp[0]) +
                                       ".tif")
            tif = tifffile.TiffFile(fname_first)
            ny_temp, nx_temp = tif.series[0].shape[-2:]

            imgs = np.zeros((ncolors, nz_used, nt_used, ny_temp, nx_temp))
            for iic, ic in enumerate(range(ncolors)):
                for iixy, ixy in enumerate(xyinds_to_use_temp):
                    for iiz, iz in enumerate(zinds_to_use_temp):
                        for iit, ind_t in enumerate(tinds_to_use_temp):
                            fname = os.path.join(individual_image_save_dir, st + "_" + id_pattern % (ic, ind_t, ixy, iz) + ".tif")
                            imgs[iic, iiz, iit] = tifffile.imread(fname)

            im_md = {"Info": "%s image reconstructed from %s" % (st, rpath),
                     "Labels": ch_labels,
                     "spacing": dz,
                     "unit": 'um'}

            if st == "sim_sr":
                # set display values for imagej
                im_md.update({"min": 0, "max": np.percentile(imgs, 99.9)})

            fname = os.path.join(sim_save_dir, '%s.tif' % st)
            imgs = tifffile.transpose_axes(imgs, "CZTYX", asaxes="TZCYXS")
            tifffile.imwrite(fname, imgs.astype(np.float32), imagej=True, datetime=start_time,
                             resolution=(1 / pixel_size * rf, 1 / pixel_size * rf),
                             metadata=im_md)

        # also for MCNR
        fname_first = os.path.join(individual_image_save_dir, "mcnr" + "_" +
                                   id_pattern % (0, tinds_to_use_temp[0], xyinds_to_use_temp[0], zinds_to_use_temp[0]) +
                                   ".tif")
        tif = tifffile.TiffFile(fname_first)
        ny_temp, nx_temp = tif.series[0].shape[-2:]

        imgs = np.zeros((ncolors, nz_used, nt_used, nphases, ny_temp, nx_temp))
        for iic, ic in enumerate(range(ncolors)):
            for iixy, ixy in enumerate(xyinds_to_use_temp):
                for iiz, iz in enumerate(zinds_to_use_temp):
                    for iit, ind_t in enumerate(tinds_to_use_temp):
                        fname = os.path.join(individual_image_save_dir,
                                             "mcnr" + "_" + id_pattern % (ic, ind_t, ixy, iz) + ".tif")
                        imgs[iic, iiz, iit] = tifffile.imread(fname)

        fname = os.path.join(sim_save_dir, 'mcnr.tif')
        imgs = tifffile.transpose_axes(imgs, "CZTQYX", asaxes="TZQCYXS")
        tifffile.imwrite(fname, imgs.astype(np.float32), imagej=True, datetime=start_time,
                         resolution=(1 / pixel_size, 1 / pixel_size),
                         metadata={"Info": "Modulation-contrast to noise-ratio (MCNR) images for each angle,"
                                           " reconstructed from %s" % (rpath),
                                   "Labels": ch_labels,
                                   "spacing": dz,
                                   "unit": 'um'})
        print("saving tiff stacks took %0.2fs" % (time.perf_counter() - tstart_save))

    return save_dirs


# compute optical sectioned SIM image
def sim_optical_section(imgs, axis=0, phase_differences=(0, 2*np.pi/3, 4*np.pi/3)):
    """
    Optical sectioning reconstruction for three SIM images with arbitrary relative phase
    differences following the approach of https://doi.org/10.1016/s0030-4018(98)00210-7

    In the most common case, where the phase differences are 0, 2*np.pi/3, and 4*np.pi/3 the result is
    Let I[a] = A * [1 + m * cos(phi + phi_a)]
    Then sqrt( (I[0] - I[1])**2 + (I[1] - I[2])**2 + (I[2] - I[0])**2 ) = m*A * 3/ np.sqrt(2)

    :param np.ndarray imgs: images stored as nD array, where one of the dimensions is of size 3.
    :param int axis: axis to perform the optical sectioning computation along. imgs.shape[axis] must = 3
    :param list[float] phase_differences: list of length 3
    :return np.ndarray img_os: optically sectioned image
    """

    if imgs.shape[axis] != 3:
        raise ValueError(f"imgs must be of size 3 along axis {axis:d}")

    if len(phase_differences) != 3:
        raise ValueError(f"phases must have length 3, but had length {len(phase_differences):d}")

    # compute inversion matrix
    p1, p2, p3 = phase_differences
    mat = np.array([[1, np.cos(p1), -np.sin(p1)],
                    [1, np.cos(p2), -np.sin(p2)],
                    [1, np.cos(p3), -np.sin(p3)]])
    inv = np.linalg.inv(mat)

    # put the axis we want to compute along first
    imgs = np.swapaxes(imgs, 0, axis)

    i_c = inv[1, 0] * imgs[0] + inv[1, 1] * imgs[1] + inv[1, 2] * imgs[2]
    i_s = inv[2, 0] * imgs[0] + inv[2, 1] * imgs[1] + inv[2, 2] * imgs[2]
    img_os = np.sqrt(i_c**2 + i_s**2)

    # img_os = np.sqrt(2) / 3 * np.sqrt((imgs[0] - imgs[1]) ** 2 + (imgs[0] - imgs[2]) ** 2 + (imgs[1] - imgs[2]) ** 2)

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
def fit_modulation_frq(ft1: np.ndarray, ft2: np.ndarray, dxy: float,
                       mask: np.ndarray = None,
                       frq_guess: tuple[float] = None, roi_pix_size: int = 5,
                       use_jacobian: bool = False, max_frq_shift=None,
                       keep_guess_if_better: bool = True):
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
    :param dxy: pixel size
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
    ny, nx = ft1.shape

    # coords
    x = tools.get_fft_pos(nx, dxy, centered=False, mode='symmetric')
    y = tools.get_fft_pos(ny, dxy, centered=False, mode='symmetric')
    xx, yy = np.meshgrid(x, y)

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(ft1.shape[1], dxy))
    dfx = fxs[1] - fxs[0]
    fys = fft.fftshift(fft.fftfreq(ft1.shape[0], dxy))
    dfy = fys[1] - fys[0]
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff = np.sqrt(fxfx ** 2 + fyfy ** 2)

    if max_frq_shift is None:
        max_frq_shift = dfx * roi_pix_size

    # mask
    if mask is None:
        mask = np.ones(ft1.shape, dtype=bool)

    if mask.shape != ft1.shape:
        raise ValueError("mask must have same shape as image")

    # update mask to account for frequency guess
    if frq_guess is not None:
        # account for frq_guess
        f_dist_guess = np.sqrt( (fxfx - frq_guess[0])**2 + (fyfy - frq_guess[1])**2)
        mask[f_dist_guess > max_frq_shift] = 0

    # cross correlation of Fourier transforms
    # WARNING: correlate2d uses a different convention for the frequencies of the output, which will not agree with the fft convention
    # take conjugates so this will give \sum ft1 * ft2.conj()
    # scipy.signal.correlate(g1, g2)(fo) seems to compute \sum_f g1^*(f) * g2(f - fo), but I want g1^*(f)*g2(f+fo)
    cc = np.abs(scipy.signal.correlate(ft2, ft1, mode='same'))

    if frq_guess is None:
        # get initial frq_guess by looking at cc at discrete frequency set and finding max
        max_ind = np.argmax(cc * mask)
        subscript = np.unravel_index(max_ind, cc.shape)

        # if cy initial frq_guess, return frequency with positive fy. If fy=0, return with positive fx.
        # i.e. want to sort so that larger y-subscript is returned first
        # since img is real, also peak at [-fx, -fy]. Find those subscripts
        # todo: could exactly calculate where these will be, but this is good enough for now.
        # a = np.argmin(np.abs(fxfx[subscript] + fxfx[0, :]))
        # b = np.argmin(np.abs(fyfy[subscript] + fyfy[:, 0]))
        # reflected_subscript = (b, a)

        # subscript_list = [subscript, reflected_subscript]
        #
        # m = np.max(ft1.shape)
        # subscript_list.sort(key=lambda x: x[0] * m + x[1], reverse=True)
        # subscript, reflected_subscript = subscript_list

        init_params = np.array([fxfx[subscript], fyfy[subscript]])
    else:
        init_params = frq_guess


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

    return fit_frqs, mask, result


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
    # normalize ...
    # img1_ft = np.array(img1_ft, copy=True) / img1_ft.size
    # img2_ft = np.array(img2_ft, copy=True) / img2_ft.size

    # get physical parameters
    dy = dx

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(img1_ft.shape[1], dx))
    fys = fft.fftshift(fft.fftfreq(img1_ft.shape[0], dy))
    if fmax is None:
        fmax = np.sqrt(np.max(fxs)**2 + np.max(fys)**2)

    # power spectrum / cross correlation
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
    gspec = figh.add_gridspec(ncols=14, nrows=2, hspace=0.3)

    str = ""
    if ttl_str != "":
        str += "%s\n" % ttl_str
    # suptitle
    str += '      fit: period %0.1fnm = 1/%0.3fum at %.2fdeg=%0.3frad; f=(%0.3f,%0.3f) 1/um, peak cc=%0.3g and %0.2fdeg' % \
          (period * 1e3, 1/period, angle * 180 / np.pi, angle, fx_sim, fy_sim,
           np.abs(peak_cc), np.angle(peak_cc) * 180/np.pi)
    if frqs_guess is not None:
        fx_g, fy_g = frqs_guess
        period_g = 1 / np.sqrt(fx_g ** 2 + fy_g ** 2)
        angle_g = np.angle(fx_g + 1j * fy_g)
        peak_cc_g = tools.get_peak_value(cc, fxs, fys, frqs_guess, peak_pixels)

        str += '\nguess: period %0.1fnm = 1/%0.3fum at %.2fdeg=%0.3frad; f=(%0.3f,%0.3f) 1/um, peak cc=%0.3g and %0.2fdeg' % \
               (period_g * 1e3, 1/period_g, angle_g * 180 / np.pi, angle_g, fx_g, fy_g,
                np.abs(peak_cc_g), np.angle(peak_cc_g) * 180/np.pi)
    figh.suptitle(str)

    # #######################################
    # plot cross-correlation region of interest
    # #######################################
    roi_cx = np.argmin(np.abs(fx_sim - fxs))
    roi_cy = np.argmin(np.abs(fy_sim - fys))
    roi = rois.get_centered_roi([roi_cy, roi_cx], [roi_size, roi_size], min_vals=[0, 0], max_vals=cc.shape)

    extent_roi = tools.get_extent(fys[roi[0]:roi[1]], fxs[roi[2]:roi[3]])

    ax = figh.add_subplot(gspec[0, 0:6])
    ax.set_title("cross correlation, ROI")
    im1 = ax.imshow(rois.cut_roi(roi, cc), interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent_roi, cmap="bone")
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

    ax.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0, ls='--'))
    ax.set_xlabel('$f_x (1/\mu m)$')
    ax.set_ylabel('$f_y (1/\mu m)$')


    cbar_ax = figh.add_subplot(gspec[0, 6])
    figh.colorbar(im1, cax=cbar_ax)

    # #######################################
    # full cross-correlation
    # #######################################
    ax2 = figh.add_subplot(gspec[0, 7:13])
    im2 = ax2.imshow(cc, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent, cmap="bone")
    ax2.set_xlim([-fmax, fmax])
    ax2.set_ylim([fmax, -fmax])

    # plot maximum frequency
    ax2.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0))

    ax2.add_artist(Rectangle((fxs[roi[2]], fys[roi[0]]), fxs[roi[3] - 1] - fxs[roi[2]], fys[roi[1] - 1] - fys[roi[0]],
                                           edgecolor='k', fill=0))

    ax2.set_title(r"$C(f_o) = \sum_f g_1(f) \times g^*_2(f+f_o)$")
    ax2.set_xlabel('$f_x (1/\mu m)$')
    ax2.set_ylabel('$f_y (1/\mu m)$')

    cbar_ax = figh.add_subplot(gspec[0, 13])
    figh.colorbar(im2, cax=cbar_ax)

    # #######################################
    # ft 1
    # #######################################
    ax3 = figh.add_subplot(gspec[1, 0:6])
    ax3.set_title(r"$|g_1(f)|^2$" + r" near DC, $g_1(0) = $"  " %0.3g and %0.2fdeg" %
                  (np.abs(peak1_dc), np.angle(peak1_dc) * 180/np.pi))
    ax3.set_xlabel('$f_x (1/\mu m)$')
    ax3.set_ylabel('$f_y (1/\mu m)$')

    cx_c = np.argmin(np.abs(fxs))
    cy_c = np.argmin(np.abs(fys))
    roi_center = rois.get_centered_roi([cy_c, cx_c], [roi[1] - roi[0], roi[3] - roi[2]], [0, 0], img1_ft.shape)
    extent_roic = tools.get_extent(fys[roi_center[0]:roi_center[1]], fxs[roi_center[2]:roi_center[3]])

    im3 = ax3.imshow(rois.cut_roi(roi_center, np.abs(img1_ft)**2),
                     interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent_roic, cmap="bone")
    ax3.scatter(0, 0, color='r', marker='x')

    cbar_ax = figh.add_subplot(gspec[1, 6])
    figh.colorbar(im3, cax=cbar_ax)

    # ft 2
    ax4 = figh.add_subplot(gspec[1, 7:13])
    ttl_str = r"$|g_2(f)|^2$" + r"near $f_o$, $g_2(f_p) =$" + " %0.3g and %0.2fdeg" % (np.abs(peak2), np.angle(peak2) * 180 / np.pi)
    if frqs_guess is not None:
        peak2_g = tools.get_peak_value(img2_ft, fxs, fys, frqs_guess, peak_pixels)
        ttl_str += "\nguess peak = %0.3g and %0.2fdeg" % (np.abs(peak2_g), np.angle(peak2_g) * 180 / np.pi)
    ax4.set_title(ttl_str)
    ax4.set_xlabel('$f_x (1/\mu m)$')

    im4 = ax4.imshow(rois.cut_roi(roi, np.abs(img2_ft)**2),
                     interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent_roi, cmap="bone")
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
    fx = fft.fftshift(fft.fftfreq(nx, dxy))
    fy = fft.fftshift(fft.fftfreq(ny, dxy))

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
        raise ValueError(f"'origin' must be 'center' or 'edge' but was '{origin:s}'")

    xx, yy = np.meshgrid(x, y)

    def fn(phi): return -np.cos(2*np.pi * (sim_frq[0] * xx + sim_frq[1] * yy) + phi)
    def fn_deriv(phi): return np.sin(2*np.pi * (sim_frq[0] * xx + sim_frq[1] * yy) + phi)
    def min_fn(phi): return np.sum(fn(phi) * img) / img.size
    def jac_fn(phi): return np.asarray([np.sum(fn_deriv(phi) * img) / img.size,])

    # using jacobian makes faster and more robust
    result = scipy.optimize.minimize(min_fn, phase_guess, jac=jac_fn)
    # also using routine optimized for scalar univariate functions works
    #result = scipy.optimize.minimize_scalar(min_fn)
    phi_fit = np.mod(result.x, 2 * np.pi)

    return phi_fit


def get_phase_wicker_iterative(imgs_ft, otf, sim_frq, dxy, fmax, phases_guess=None, fit_amps=True, debug=False):
    """
    Estimate relative phases between components using the iterative cross-correlation minimization method of Wicker,
    described in detail here https://doi.org/10.1364/OE.21.002032. This function is hard coded for 3 bands.

    NOTE: this method is not sensitive to the absolute phase, only the relative phases...

    Suppose that S(r) is the sample, h(r) is the PSF, and D_n(r) is the data. Then the separated (but unshifted) bands
    are C_m(k) = S(k - m*ko) * h_m(k), and the data vector is related to the band vector by
    D(k) = M*C(k), where M is an nphases x nbands matrix.

    This function minimizes the cross correlation between shfited bands which should not contain common information.
    Let cc^l_ij = C_i(k) \otimes C_j(k-lp). These should have low overlap for i =/= j + l. So the minimization function is
    g(M) = \sum_{i \neq l+j} |cc^l_ij|^0.5

    This can be written in terms of the correlations of data matrix, dc^l_ij in a way that minimizes numerical effort
    to compute g for different mixing matrices M.

    :param imgs_ft: array of size nphases x ny x nx, where the components are o(f), o(f-fo), o(f+fo)
    :param otf: size ny x nx
    :param sim_frq: np.array([fx, fy])
    :param float dxy: pixel size in um
    :param float fmax: maximum spatial frequency where otf has support
    :param phase_guess: [phi1, phi2, phi3] in radians.if None will use [0, 2*pi/3, 4*pi/3]
    :param fit_amps: if True will also fit amplitude differences between components
    :return phases, amps, result: where phases is a list of phases determined using this method, amps = [A1, A2, A3],
    and result is a dictionary giving information about the convergence of the optimization
    If fit_amps is False, A1=A2=A3=1
    """
    # TODO: this can also return components separated the opposite way of desired
    # todo: currently hardcoded for 3 phases
    # todo: can I improve the fitting by adding jacobian?
    # todo: can get this using d(M^{-1})dphi1 = - M^{-1} * (dM/dphi1) * M^{-1}
    # todo: probably not necessary, because phases should always be close to equally spaced, so initial guess should be good

    nphases, ny, nx = imgs_ft.shape
    fx = fft.fftshift(fft.fftfreq(nx, dxy))
    dfx = fx[1] - fx[0]
    fy = fft.fftshift(fft.fftfreq(ny, dxy))
    dfy = fy[1] - fy[0]

    # naive guess might be useful in some circumstances ...
    # phases_naive = np.zeros(nphases)
    # for ii in range(nphases):
    #     phases_naive[ii] = np.angle(tools.get_peak_value(imgs_ft[ii], fx, fy, sim_frq, peak_pixel_size=2))

    # compute cross correlations of data
    band_inds = [0, 1, -1]
    nbands = len(band_inds)
    d_cc = np.zeros((nphases, nphases, nbands), dtype=complex)
    # Band_i(k) = Obj(k - i*p) * h(k)
    # this is the order set by matrix M, i.e.
    # [D1(k), ...] = M * [Obj(k) * h(k), Obj(k - i*p) * h(k), Obj(k + i*p) * h(k)]
    for ll, ml in enumerate(band_inds):
        # get shifted otf -> otf(f - l * fo)
        otf_shift, _ = tools.translate_pix(otf, -ml * sim_frq, dr=(dfx, dfy), axes=(1, 0), wrap=False)

        with np.errstate(invalid="ignore", divide="ignore"):
            weight = otf * otf_shift.conj() / (np.abs(otf_shift) ** 2 + np.abs(otf) ** 2)
            weight[np.isnan(weight)] = 0

        for ii in range(nphases):  # [0, 1, 2] -> [0, 1, -1]
            for jj in range(nphases):
                # shifted component C_j(f - l*fo)
                band_shifted = tools.translate_ft(imgs_ft[jj], -ml * sim_frq, drs=(dxy, dxy))
                # compute weighted cross correlation
                d_cc[ii, jj, ll] = np.sum(imgs_ft[ii] * band_shifted.conj() * weight) / np.sum(weight)

                # remove extra noise correlation expected from same images
                if ml == 0 and ii == jj:
                    noise_power = get_noise_power(imgs_ft[ii], fx, fy, fmax)
                    d_cc[ii, jj, ll] = d_cc[ii, jj, ll] - noise_power

                if debug:
                    gamma = 0.1

                    figh =plt.figure(figsize=(16, 8))
                    grid = figh.add_gridspec(2, 3)
                    figh.suptitle(f"(i, j, band) = ({ii:d}, {jj:d}, {ml:d})")

                    ax = figh.add_subplot(grid[0, 0])
                    ax.imshow(np.abs(imgs_ft[ii]), norm=PowerNorm(gamma=gamma), extent=tools.get_extent(fy, fx))
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("$D_i(k)$")

                    ax = figh.add_subplot(grid[0, 1])
                    ax.imshow(np.abs(band_shifted), norm=PowerNorm(gamma=gamma), extent=tools.get_extent(fy, fx))
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("$D_j(k-lp)$")

                    ax = figh.add_subplot(grid[0, 2])
                    ax.imshow(np.abs(imgs_ft[ii] * band_shifted.conj()), norm=PowerNorm(gamma=gamma),
                              extent=tools.get_extent(fy, fx))
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("$D^l_{ij} = D_i(k) x D_j^*(k-lp)$")

                    ax = figh.add_subplot(grid[1, 0])
                    ax.imshow(otf, extent=tools.get_extent(fy, fx))
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title('$otf_i(k)$')

                    ax = figh.add_subplot(grid[1, 1])
                    ax.imshow(otf_shift, extent=tools.get_extent(fy, fx))
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title('$otf_j(k-lp)$')

                    ax = figh.add_subplot(grid[1, 2])
                    ax.imshow(weight, extent=tools.get_extent(fy, fx))
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("weight")
    # correct normalization of d_cc (inherited from FFT) so should be same for different image sizes
    d_cc = d_cc / (nx * ny)**2

    # optimize
    if fit_amps:
        # def minv(p): return np.linalg.inv(get_band_mixing_matrix([p[0], p[1], p[2]], [1, 1, 1], [1, p[3], p[4]]))
        def minv(p): return get_band_mixing_inv([0, p[0], p[1]], mod_depth=1, amps=[1, p[2], p[3]])
        def minv_jac(p):
            minv_j = get_band_mixing_inv_jac([0, p[0], p[1]], mod_depth=1, amps=[1, p[2], p[3]])
            return [minv_j[1], minv_j[2], minv_j[4], minv_j[5]]
    else:
        # def minv(p): return np.linalg.inv(get_band_mixing_matrix(p, [1, 1, 1]))
        def minv(p): return get_band_mixing_inv([0, p[0], p[1]], mod_depth=1, amps=[1, 1, 1])
        def minv_jac(p):
            minv_j = get_band_mixing_inv_jac([0, p[0], p[1]], mod_depth=1, amps=[1, 1, 1])
            return [minv_j[1], minv_j[2]]

    # condition = ii - (j + l)
    index_condition = np.expand_dims(np.array(band_inds), axis=(1, 2)) - \
                      np.expand_dims(np.array(band_inds), axis=(0, 2)) - \
                      np.expand_dims(np.array(band_inds), axis=(0, 1))

    def min_fn(p):
        m1 = minv(p)
        cc = np.zeros((nbands, nbands, nbands), dtype=complex)
        for ll in range(nbands):
            cc[..., ll] = m1.dot(d_cc[..., ll].dot(m1.conj().transpose()))

        # also normalize function by size
        g = np.sum(np.sqrt(np.abs(cc * (index_condition != 0)))) / (nbands * nbands * nbands)

        return g

    # todo: get this working, then add to fitting...
    def min_fn_jac(p):
        m1 = minv(p)
        m1_jac = minv_jac(p)
        cc = np.zeros((nbands, nbands, nbands), dtype=complex)
        cc_jac = [np.zeros(cc.shape, dtype=complex)] * len(p)
        # already have a bug when comparing cc_jac to cc ...
        for ll in range(nbands):
            cc[..., ll] = m1.dot(d_cc[..., ll].dot(m1.conj().transpose()))
            for aa in range(len(p)):
                cc_jac[aa][..., ll] = m1.dot(d_cc[..., ll].dot(m1_jac[aa].conj().transpose())) + \
                                      m1_jac[aa].dot(d_cc[..., ll].dot(m1.conj().transpose()))

        g_jac = [[]] * len(p)
        for aa in range(len(p)):
            g_jac[aa] = np.sum(0.5 / np.sqrt(np.abs(cc)) * cc_jac[aa] * (index_condition != 0)) / (nbands * nbands * nbands)

        return g_jac

    # can also include amplitudes and modulation depths in optimization process
    if fit_amps:
        if phases_guess is None:
            ip_pos = np.array([2 * np.pi / 3, 4 * np.pi / 3, 1, 1])
            ip_neg = np.array([-2 * np.pi / 3, -4 * np.pi / 3, 1, 1])
            if min_fn(ip_pos) < min_fn(ip_neg):
                init_params = ip_pos
            else:
                init_params = ip_neg
        else:
            #init_params = np.concatenate((phases_guess, np.array([1, 1])))
            init_params = np.array([phases_guess[1] - phases_guess[0], phases_guess[2] - phases_guess[0], 1, 1])

        result = scipy.optimize.minimize(min_fn, init_params)
        phases = np.array([0, result.x[0], result.x[1]])
        amps = np.array([1, result.x[2], result.x[3]])
    else:
        if phases_guess is None:
            ip = np.array([2 * np.pi / 3, 4 * np.pi / 3])
            if min_fn(ip) < min_fn(-ip):
                init_params = ip
            else:
                init_params = -ip
        else:
            init_params = np.array([phases_guess[1] - phases_guess[0], phases_guess[2] - phases_guess[0]])

        result = scipy.optimize.minimize(min_fn, init_params)
        phases = np.array([0, result.x[0], result.x[1]])
        amps = np.array([1, 1, 1])

    return phases, amps, result

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
        mask = np.ones(img_ft.shape, dtype=bool)

    # get physical data
    dx = options['pixel_size']
    dy = dx
    wavelength = options['wavelength']
    na = options['na']
    fmax = 1 / (0.5 * wavelength / na)

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(img_ft.shape[1], dx))
    fys = fft.fftshift(fft.fftfreq(img_ft.shape[0], dy))
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
    grid = fig.add_gridspec(2, 10)

    sttl_str = ""
    if ttl_str != "":
        sttl_str += "%s\n" % ttl_str
    sttl_str += 'm=%0.3g, A=%0.3g, alpha=%0.3f\nnoise=%0.3g, frq sim = (%0.2f, %0.2f) 1/um' % \
                (pfit[-2], pfit[0], pfit[1], pfit[-1], frq_sim[0], frq_sim[1])
    fig.suptitle(sttl_str)

    # ######################
    # plot power spectrum x otf vs frequency in 1D
    # ######################
    ax1 = fig.add_subplot(grid[0, :5])

    ax1.semilogy(ff_shift.ravel(), ps_exp.ravel(), 'k.')
    ax1.semilogy(ff_shift[mask].ravel(), ps_exp[mask].ravel(), 'b.')
    ax1.semilogy(ff_shift.ravel(), ps_fit.ravel(), 'r')

    ylims = ax1.get_ylim()
    ax1.set_ylim([ylims[0], 1.2 * np.max(ps_exp[mask].ravel())])

    ax1.set_xlabel('frequency (1/um)')
    ax1.set_ylabel('power spectrum')
    ax1.legend(['all data', 'data used to fit', 'fit'], loc="upper right")
    ax1.set_title('m^2 A^2 |f|^{-2*alpha} |otf(f)|^2 + N')

    # ######################
    # plot az avg divided by otf in 1D
    # ######################
    ax2 = fig.add_subplot(grid[0, 5:])

    ax2.semilogy(ff_shift.ravel(), ps_exp_deconvolved.ravel(), 'k.')
    ax2.semilogy(ff_shift[mask].ravel(), ps_exp_deconvolved[mask].ravel(), 'b.')
    ax2.semilogy(ff_shift.ravel(), ps_fit_no_otf.ravel(), 'g')

    ylims = ax1.get_ylim()
    ax1.set_ylim([ylims[0], 1.2 * np.max(ps_exp_deconvolved[mask].ravel())])

    ax2.set_title('m^2 A^2 |k|^{-2*alpha} |otf|^4/(|otf|^4 + snr^2)')
    ax2.set_xlabel('|f - f_sim| (1/um)')
    ax2.set_ylabel('power spectrum')

    # ######################
    # plot 2D power spectrum
    # ######################
    ax3 = fig.add_subplot(grid[1, :2])

    ax3.imshow(ps_exp, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent, cmap="bone")

    ax3.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0, ls='--'))

    ax3.add_artist(Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--'))

    ax3.set_xlabel('fx (1/um)')
    ax3.set_ylabel('fy (1/um)')
    ax3.set_title('raw power spectrum')

    # ######################
    # 2D fit
    # ######################
    ax5 = fig.add_subplot(grid[1, 2:4])
    ax5.imshow(ps_fit, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent, cmap="bone")

    ax5.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0, ls='--'))
    ax5.add_artist(Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--'))

    ax5.set_xlabel('fx (1/um)')
    ax5.set_ylabel('fy (1/um)')
    ax5.set_title('2D fit')

    # ######################
    # plot 2D power spectrum divided by otf with masked region
    # ######################
    ax4 = fig.add_subplot(grid[1, 6:8])
    # ps_over_otf[mask == 0] = np.nan
    ps_exp_deconvolved[mask == 0] = np.nan
    ax4.imshow(ps_exp_deconvolved, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent, cmap="bone")

    ax4.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0, ls='--'))
    ax4.add_artist(Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--'))

    ax4.set_xlabel('fx (1/um)')
    ax4.set_ylabel('fy (1/um)')
    ax4.set_title('masked, deconvolved power spectrum')

    #
    ax4 = fig.add_subplot(grid[1, 8:])
    ax4.imshow(ps_fit_deconvolved, interpolation=None, norm=PowerNorm(gamma=0.1), extent=extent, cmap="bone")

    ax4.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0, ls='--'))
    ax4.add_artist(Circle((frq_sim[0], frq_sim[1]), radius=fmax, color='k', fill=0, ls='--'))

    ax4.set_xlabel('fx (1/um)')
    ax4.set_ylabel('fy (1/um)')
    ax4.set_title('fit deconvolved')

    return fig

# inversion functions
def get_band_mixing_matrix(phases, mod_depth=1, amps=None):
    """
    Return matrix M, which relates the measured images D to the Fluorescence profile S multiplied by the OTF H
    [[D_1(k)], [D_2(k)], [D_3(k)], ...[D_n(k)]] = M * [[S(k)H(k)], [S(k-p)H(k)], [S(k+p)H(k)]]

    We assume the modulation has the form [1 + m*cos(k*r + phi)], leading to
    M = [A_1 * [1, 0.5*m*exp(ip_1), 0.5*m*exp(-ip_1)],
         A_2 * [1, 0.5*m*exp(ip_2), 0.5*m*exp(-ip_2)],
         A_3 * [1, 0.5*m*exp(ip_3), 0.5*m*exp(-ip_3)],
         ...
        ]

    :param phases: np.array([phase_1, ..., phase_n])
    :param mod_depth: np.array([m_1, m_2, ..., m_n]. In most cases, these are equal
    :param amps: np.array([a_1, a_2, ..., a_n])
    :return mat: nphases x nbands matrix
    """

    if amps is None:
        amps = np.ones(len(phases))

    mat = []
    for p, a in zip(phases, amps):
        mat.append(a * np.array([1, 0.5 * mod_depth * np.exp(1j * p), 0.5 * mod_depth * np.exp(-1j * p)]))
    mat = np.asarray(mat)

    return mat


def get_band_mixing_matrix_jac(phases, mod_depth, amps):
    """
    Get jacobian of band mixing matrix in parameters [p1, p2, p3, a1, a2, a3, m]
    @param phases:
    @param mod_depth:
    @param amps:
    @return:
    """
    p1, p2, p3 = phases
    a1, a2, a3, = amps
    m = mod_depth

    jac = [np.array([[0, 0.5 * a1 * m * 1j * np.exp(1j * p1), -0.5 * a1 * m * 1j * np.exp(-1j * p1)],
                     [0, 0, 0],
                     [0, 0, 0]]),
           np.array([[0, 0, 0],
                     [0, 0.5 * a2 * m * 1j * np.exp(1j * p2), -0.5 * a2 * m * 1j * np.exp(-1j * p2)],
                     [0, 0, 0]]),
           np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0.5 * a3 * m * 1j * np.exp(1j * p3), -0.5 * a3 * m * 1j * np.exp(-1j * p3)]]),
           np.array([[1, 0.5 * m * np.exp(1j * p1), 0.5 * m * np.exp(1j * p1)],
                     [0, 0, 0],
                     [0, 0, 0]]),
           np.array([[0, 0, 0],
                     [1, 0.5 * m * np.exp(1j * p2), 0.5 * m * np.exp(-1j * p2)],
                     [0, 0, 0]]),
           np.array([[0, 0, 0],
                     [0, 0, 0],
                     [1, 0.5 * m * np.exp(1j * p3), 0.5 * m * np.exp(-1j * p3)]]),
           np.array([[0, 0.5 * a1 * np.exp(1j * p1), 0.5 * a1 * np.exp(-1j * p1)],
                     [0, 0.5 * a2 * np.exp(1j * p2), 0.5 * a2 * np.exp(-1j * p2)],
                     [0, 0.5 * a3 * np.exp(1j * p3), 0.5 * a3 * np.exp(-1j * p3)]])
           ]

    return jac


def get_band_mixing_inv(phases, mod_depth=1, amps=None):
    """
    Get inverse of the band mixing matrix, which maps measured data to separated (but unshifted) bands

    @param phases:
    @param mod_depth:
    @param amps:
    @return mixing_mat_inv:
    """

    mixing_mat = get_band_mixing_matrix(phases, mod_depth, amps)

    if len(phases) == 3:
        # direct inversion
        try:
            mixing_mat_inv = np.linalg.inv(mixing_mat)
        except np.linalg.LinAlgError:
            warnings.warn("warning: band inversion matrix singular")
            mixing_mat_inv = np.zeros((mixing_mat.shape[1], mixing_mat.shape[0])) * np.nan

    else:
        # pseudo-inverse
        mixing_mat_inv = np.linalg.pinv(mixing_mat)

    return mixing_mat_inv


def get_band_mixing_inv2(phases, m, amps=None):
    """
    Get inverse of the band mixing matrix, which maps measured data to separated (but unshifted) bands
    Do this analytically...

    @param phases:
    @param mod_depths:
    @param amps:
    @return mixing_mat_inv:
    """
    p1, p2, p3 = phases
    a1, a2, a3 = amps
    det = 2 * 1j * (np.sin(p2 - p3) + np.sin(p1 - p2) + np.sin(p3 - p1))

    amp_part = np.array([[1/a1, 0, 0], [0, 1/a2, 0], [0, 0, 1/a3]])
    phase_part_inv = 1 / det * np.array([[np.exp(1j * (p2 - p3)) - np.exp(-1j * (p2 - p3)),
                                          np.exp(1j * (p3 - p1)) - np.exp(-1j * (p3 - p1)),
                                          np.exp(1j * (p1 - p2)) - np.exp(-1j * (p1 - p2))],
                                         [np.exp(-1j * p2) - np.exp(-1j * p3), np.exp(-1j * p3) - np.exp(-1j * p1),
                                          np.exp(1j * p1) - np.exp(-1j * p2)],
                                         [np.exp(1j * p3) - np.exp(1j * p2), np.exp(1j * p1) - np.exp(1j * p3),
                                          np.exp(1j * p2) - np.exp(1j * p1)]])
    global_part = np.array([[1, 0, 0], [0, 2 / m, 0], [0, 0, 2 / m]])
    mat_inv = global_part.dot(phase_part_inv.dot(amp_part))

    return mat_inv


def get_band_mixing_inv_jac(phases, mod_depth=1, amps=None):
    """

    @param phases:
    @param mod_depth:
    @param amps:
    @return:
    """
    mixing_jac = get_band_mixing_matrix_jac(phases, mod_depth, amps)
    mat_inv = get_band_mixing_inv2(phases, mod_depth, amps)
    inv_jac = [-mat_inv.dot(mi.dot(mat_inv)) for mi in mixing_jac]

    return inv_jac


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
        mod_depths = np.ones(nangles)

    if amps is None:
        amps = np.ones((nangles, nphases))

    # check parameters
    if nphases != 3:
        raise NotImplementedError(f"only implemented for nphases=3, but nphases={nphases:d}")

    bands_ft = np.zeros((nangles, nphases, ny, nx), dtype=complex) * np.nan

    # try to do inversion
    for ii in range(nangles):
        mixing_mat_inv = get_band_mixing_inv(phases[ii], mod_depths[ii], amps[ii])
        bands_ft[ii] = image_times_matrix(imgs_ft[ii], mixing_mat_inv)

    return bands_ft


def get_band_overlap(band0, band1, otf0, otf1, mask):
    """
    Compare the unshifted (0th) SIM band with the shifted (1st) SIM band to estimate the global phase shift and
    modulation depth.

    This is done by computing the amplitude and phase of
    C = \sum [Band_0(f) * conj(Band_1(f + fo))] / \sum [ |Band_0(f)|^2]
    where Band_1(f) = O(f-fo), so Band_1(f+fo) = O(f). i.e. these are the separated SIM.

    If correct reconstruction parameters are used, expect Band_0(f) and Band_1(f) differ only by a complex constant.
    This constant contains information about the global phase offset AND the modulation depth. i.e.
    Band_1(f) = c * Band_0(f) = m * np.exp(-i*phase_corr) * Band_0(f)
    This function extracts the complex conjugate of this value, c* = m * np.exp(i*phase_corr)

    Given this information, can perform the phase correction
    Band_1(f + fo) -> np.exp(i*phase_corr) / m * Band_1(f + fo)

    :param band0: nangles x ny x nx. Typically band0(f) = S(f) * otf(f) * wiener(f) ~ S(f)
    :param band1: nangles x ny x nx. Typically band1(f) = S((f-fo) + fo) * otf(f + fo) * wiener(f + fo),
    i.e. the separated band after shifting to correct position
    :param mask: where mask is True, use these points to evaluate the band correlation. Typically construct by picking
    some value where otf(f) and otf(f + fo) are both > w, where w is some cutoff value.

    :return phases, mags:
    """
    nangles = band0.shape[0]
    phases = np.zeros((nangles))
    mags = np.zeros((nangles))

    # divide by OTF, but don't worry about Wiener filtering. avoid problems by keeping otf_threshold large enough
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


# create test data/SIM forward model
# todo: could think about giving a 3D stack and converting this ...
def get_simulated_sim_imgs(ground_truth, frqs, phases, mod_depths,
                           gains, offsets, readout_noise_sds, pix_size, amps=None,
                           coherent_projection=True, otf=None, **kwargs):
    """
    Get simulated SIM images, including the effects of shot-noise and camera noise.

    :param ground_truth: ground truth image of size ny x nx
    :param frqs: SIM frequencies, of size nangles x 2. frqs[ii] = [fx, fy]
    :param phases: SIM phases in radians. Of size nangles x nphases. Phases may be different for each angle.
    :param list mod_depths: SIM pattern modulation depths. Size nangles. If pass matrices, then mod depths can vary
    spatially. Assume pattern modulation is the same for all phases of a given angle. Maybe pass list of numpy arrays
    :param gains: gain of each pixel (or single value for all pixels)
    :param offsets: offset of each pixel (or single value for all pixels)
    :param readout_noise_sds: noise standard deviation for each pixel (or single value for all pixels)
    :param pix_size: pixel size in um
    :param bool coherent_projection:
    :param otf: the optical transfer function evaluated at the frequencies points of the FFT of ground_truth. The
    proper frequency points can be obtained using fft.fftshift(fft.fftfreq(nx, dx)) and etc.
    :param kwargs: keyword arguments which will be passed through to simulated_img()

    :return sim_imgs: nangles x nphases x ny x nx array
    :return snrs: nangles x nphases x ny x nx array giving an estimate of the signal-to-noise ratio which will be
    accurate as long as the photon number is large enough that the Poisson distribution is close to a normal distribution
    """

    # ensure ground truth is 3D
    if ground_truth.ndim == 2:
        ground_truth = np.expand_dims(ground_truth, axis=0)
    nz, ny, nx = ground_truth.shape

    # check bin sizes
    if 'bin_size' in kwargs:
        nbin = kwargs['bin_size']
    else:
        nbin = 1

    # check phases
    if isinstance(phases, (float, int)):
        phases = np.atleast_2d(np.array(phases))

    # check mod depths
    if isinstance(mod_depths, (float, int)):
        mod_depths = np.atleast_1d(np.array(mod_depths))

    # check frequencies
    frqs = np.atleast_2d(frqs)

    nangles = len(frqs)
    nphases = len(phases)

    if otf is None and not coherent_projection:
        raise ValueError("If coherent_projection is false, OTF must be provided")

    if len(mod_depths) != nangles:
        raise ValueError("mod_depths must have length nangles")

    if amps is None:
        amps = np.ones((nangles, nphases))

    if otf is not None:
        psf, _ = localize_psf.fit_psf.otf2psf(otf)
    else:
        psf = None

    # get coordinates
    x = tools.get_fft_pos(nx, pix_size, centered=True, mode="symmetric")
    y = tools.get_fft_pos(ny, pix_size, centered=True, mode="symmetric")
    z = tools.get_fft_pos(nz, pix_size, centered=True, mode="symmetric")
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    nxb = nx / nbin
    nyb = ny / nbin
    if not nxb.is_integer() or not nyb.is_integer():
        raise Exception("The image size was not evenly divisble by the bin size")
    nxb = int(nxb)
    nyb = int(nyb)

    # phases will be the phase of the patterns in the binned coordinates
    # need to compute the phases in the unbinned coordinates
    # the effect of binning is to shift the center of the binned coordinates to the right/positive
    # think of leftmost bin.
    # For binned, this is at position -nbin x (n/nbin - 1) / 2 if n/nbin is odd, or -n/2 if n/nbin is even
    # For unbinned, add up the leftmost nbins
    # if nx and nx/nbin are even, shift = 0.5 * nbin - 0.5
    # if nx is even and nx/nbin is odd, shift = 0.5 * nbin - 1
    # if nx is odd and nx/nbin is odd, shift = 0
    # if nx is odd, nx/nbin cannot be even
    # relative to the unbinned coordinates
    if np.mod(nx, 2) == 1:
        xshift = 0
    else:
        if np.mod(nxb, 2) == 1:
            xshift = -0.5
        else:
            xshift = 0.5 * nbin - 0.5
    xshift = xshift * pix_size

    if np.mod(ny, 2) == 1:
        yshift = 0
    else:
        if np.mod(nyb, 2) == 1:
            yshift = -0.5
        else:
            yshift = 0.5 * nbin - 0.5
    yshift = yshift * pix_size

    bin2non_xform = affine.params2xform([1, 0, xshift, 1, 0, yshift])
    phases_unbinned = np.zeros(phases.shape)
    for ii in range(nangles):
        for jj in range(nphases):
            _, _, phases_unbinned[ii, jj] = affine.xform_sinusoid_params(frqs[ii, 0], frqs[ii, 1], phases[ii, jj], bin2non_xform)

    # generate images
    sim_imgs = np.zeros((nangles, nphases, nz, nyb, nxb))
    snrs = np.zeros(sim_imgs.shape)
    mcnrs = np.zeros(sim_imgs.shape)
    for ii in range(nangles):
        for jj in range(nphases):

            pattern = amps[ii, jj] * (1 + mod_depths[ii] * np.cos(2 * np.pi * (frqs[ii][0] * xx + frqs[ii][1] * yy) + phases_unbinned[ii, jj]))

            if not coherent_projection:
                pattern = localize_psf.fit_psf.blur_img_otf(pattern, otf).real

            sim_imgs[ii, jj], snrs[ii, jj] = camera_noise.simulated_img(ground_truth * pattern, gains, offsets,
                                                                        readout_noise_sds, psf=psf, **kwargs)
            # todo: compute mcnr
            mcnrs[ii, jj] = 0

    return np.squeeze(sim_imgs), np.squeeze(snrs)
