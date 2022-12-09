"""
Tools for reconstructing 2D SIM images from raw data.

The primary reconstruction code is contained in the class SimImageSet, which operates on a single 2D plane at a time.

reconstruct_sim_dataset() provides an example of using SimImageSet to reconstruct a larger dataset, in this case
implemented for a MicroManager dataset containing multiple z-positions, color channels, and time points.
"""
import time
import datetime
import copy
import warnings
from typing import Union, Optional
# parallelization
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
# numerics
import numpy as np
from scipy import fft
from scipy.optimize import minimize
from scipy.signal import correlate
from scipy.signal.windows import tukey
from skimage.exposure import match_histograms as match_histograms_cpu
# working with external files
from pathlib import Path
import shutil
from io import StringIO
# loading and exporting data
import pickle # todo: remove
import json
import tifffile
import zarr
# plotting
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm, LogNorm
from matplotlib.patches import Circle, Rectangle
# code from our projects
import mcsim.analysis.analysis_tools as tools
from mcsim.analysis import mm_io
from localize_psf import fit, affine, rois, fit_psf, camera

# GPU
_cupy_available = True
try:
    import cupy as cp
    from cucim.skimage.exposure import match_histograms as match_histograms_gpu
except ImportError:
    cp = np
    _cupy_available = False


array = Union[np.ndarray, cp.ndarray]


class SimImageSet:
    def __init__(self,
                 physical_params: dict,
                 imgs: np.ndarray,
                 otf: Optional[np.ndarray] = None,
                 wiener_parameter: float = 0.1,
                 frq_estimation_mode: str = "band-correlation",
                 frq_guess: Optional[np.ndarray] = None,
                 phase_estimation_mode: str = "wicker-iterative",
                 phases_guess: Optional[np.ndarray] = None,
                 combine_bands_mode: str = "fairSIM",
                 fmax_exclude_band0: float = 0,
                 mod_depths_guess: Optional[np.ndarray] = None,
                 use_fixed_mod_depths: bool = False,
                 mod_depth_otf_mask_threshold: float = 0.1,
                 minimum_mod_depth: float = 0.5,
                 normalize_histograms: bool = True,
                 determine_amplitudes: bool = False,
                 background: float = 0.,
                 gain: float = 1.,
                 max_phase_err: float = 10 * np.pi / 180,
                 min_p2nr: float = 1.,
                 trim_negative_values: bool = False,
                 upsample_widefield: bool = False,
                 interactive_plotting: bool = False,
                 save_dir: Optional[str] = None,
                 save_suffix: str = "",
                 save_prefix: str = "",
                 use_gpu: bool = _cupy_available):
        """
        Reconstruct raw SIM data into widefield, SIM-SR, SIM-OS, and deconvolved images using the Wiener filter
        style reconstruction of Gustafsson and Heintzmann. This code relies on various ideas developed and
        implemented elsewhere, see for example fairSIM and openSIM.

        An instance of this class may be used directly to reconstruct a single SIM image which is stored as a
        numpy array. For a typical experiment it is usually best to write a helper function to load the data and
        coordinate the SIM parameter estimation and reconstruction of e.g. various channels, z-slices, time points
        or etc. For an example of this approach, see the function reconstruct_mm_sim_dataset()

        Both the raw data and the SIM data use the same coordinates as the FFT with the origin in the center.
        i.e. the coordinates in the raw image are x = (arange(nx) - (nx // 2)) * dxy
        and for the SIM image they are            x = ((arange(2*nx) - (2*nx)//2) * 0.5 * dxy

        Note that this means they cannot be overlaid by changing the scale for the SIM image by a factor of two.
        There is an additional translation. The origin in the raw images is at pixel n//2 while those in the SIM
        images are at (2*n) // 2 = n. This translation is due to the fact that for odd n,
        n // 2 != (2*n) // 2 * 0.5 = n / 2

        :param physical_params: {'pixel_size', 'na', 'wavelength'}. Pixel size and emission wavelength in um
        :param imgs: n0 x n1 x ... nm x nangles x nphases x ny x nx raw data to be reconstructed. The first
        m-dimensions will be reconstructed in parallel. These may represent e.g. time-series and z-stack data.
        The same reconstruction parameters must be used for the full stack, so these should not represent different
        channels.
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
        :param bool determine_amplitudes: whether to determine amplitudes as part of Wicker phase optimization.
        This flag only has an effect if phase_estimation_mode is "wicker-iterative"
        :param background: a single number, or an array which is broadcastable to the same size as images. This will
        be subtracted from the raw data before processing.
        :param gain: gain of the camera in ADU/photons. This is a single number or an array which is broadcastable to
         the same size as the images whcih is sued to convert the ADU counts to photon numbers.
        :param max_phase_err: If the determined phase error between components exceeds this value, use the phase guess
        values instead of those determined by the estimation algorithm.
        :param min_p2nr: if the peak-to-noise ratio is smaller than this value, use the frequency guesses instead
         of the frequencies determined by the estimation algorithm.
        :param bool interactive_plotting: show plots in python GUI windows, or save outputs only
        :param str save_dir: directory to save results. If None, then results will not be saved
        :param bool use_gpu:
        """
        # #############################################
        # open log file
        # #############################################
        self.save_suffix = save_suffix
        self.save_prefix = save_prefix
        self.log = StringIO() # can save this stream to a file later if desired

        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.save_dir = None

        # #############################################
        # print current time
        # #############################################
        self.tstamp = datetime.datetime.now().strftime('%Y_%d_%m %H;%M;%S')

        self.print_log("####################################################################################")
        self.print_log(self.tstamp)
        self.print_log("####################################################################################")

        # #############################################
        # plot settings
        # #############################################
        self.interactive_plotting = interactive_plotting
        if not self.interactive_plotting:
            plt.ioff()
            plt.switch_backend("agg")

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
            raise NotImplementedError("upsampling widefield not yet implemented")

        self.upsample_fact = 2
        self.combine_bands_mode = combine_bands_mode
        self.use_fixed_mod_depths = use_fixed_mod_depths
        self.phase_estimation_mode = phase_estimation_mode
        self.fmax_exclude_band0 = fmax_exclude_band0
        self.otf_mask_threshold = mod_depth_otf_mask_threshold
        self.minimum_mod_depth = minimum_mod_depth

        if phases_guess is None:
            self.frq_estimation_mode = "fourier-transform"
            self.print_log(f"No phase guesses provided, defaulting to frq_estimation_mode = '{self.frq_estimation_mode:s}'")
        else:
            self.frq_estimation_mode = frq_estimation_mode

        # #############################################
        # GPU
        # #############################################
        if self.use_gpu:
            # need to disable fft plane cache, otherwise quickly run out of memory
            cp.fft._cache.PlanCache(memsize=0)

            xp = cp
            match_histograms = match_histograms_gpu
        else:
            xp = np
            match_histograms = match_histograms_cpu

        # #############################################
        # images
        # #############################################
        # todo: for full generality would want to store as npatterns x ny x nx array instead
        self.nangles, self.nphases, self.ny, self.nx = imgs.shape[-4:]
        self.n_extra_dims = imgs.ndim - 4

        # ensures imgs dask array with chunksize = 1 raw image
        chunk_size = (1,) * (self.n_extra_dims + 2) + imgs.shape[-2:]
        if not isinstance(imgs, da.core.Array):
            imgs = da.from_array(imgs, chunks=chunk_size)
        else:
            imgs = imgs.rechunk(chunk_size)

        # ensure on CPU/GPU as appropriate
        self.imgs = da.map_blocks(lambda x: xp.array(x.astype(float)), imgs, dtype=float)

        # hardcoded for 2D SIM
        self.nbands = 3
        self.band_inds = np.array([0, 1, -1], dtype=int) # bands are shifted by these multiplies of frqs

        # #############################################
        # real space parameters
        # #############################################
        self.dx = physical_params['pixel_size']
        self.dy = physical_params['pixel_size']
        self.x = (xp.arange(self.nx) - (self.nx // 2)) * self.dx
        self.y = (xp.arange(self.ny) - (self.ny // 2)) * self.dy
        self.x_us = (xp.arange(self.nx * self.upsample_fact) - (self.nx * self.upsample_fact) // 2) * (self.dx / self.upsample_fact)
        self.y_us = (xp.arange(self.ny * self.upsample_fact) - (self.ny * self.upsample_fact) // 2) * (self.dy / self.upsample_fact)

        # #############################################
        # physical parameters
        # #############################################
        self.na = physical_params['na']
        self.wavelength = physical_params['wavelength']

        self.fmax = 1 / (0.5 * self.wavelength / self.na)

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
            self.mod_depths_guess = np.ones(self.nangles)

        # #############################################
        # get frequency data and OTF
        # #############################################
        self.fx = xp.fft.fftshift(xp.fft.fftfreq(self.nx, self.dx))
        self.fy = xp.fft.fftshift(xp.fft.fftfreq(self.ny, self.dy))
        self.fx_us = xp.fft.fftshift(xp.fft.fftfreq(self.upsample_fact * self.nx, self.dx / self.upsample_fact))
        self.fy_us = xp.fft.fftshift(xp.fft.fftfreq(self.upsample_fact * self.ny, self.dy / self.upsample_fact))
        self.dfx = float(self.fx[1] - self.fx[0])
        self.dfy = float(self.fy[1] - self.fy[0])
        self.dfx_us = float(self.fx_us[1] - self.fx_us[0])
        self.dfy_us = float(self.fy_us[1] - self.fy_us[0])

        if otf is None:
            otf = fit_psf.circ_aperture_otf(np.expand_dims(self.fx, axis=0),
                                            np.expand_dims(self.fy, axis=1),
                                            self.na,
                                            self.wavelength)

        if np.any(otf < 0) or np.any(otf > 1):
            raise ValueError("OTF must be >= 0 and <= 1")

        # otf is stored as nangles x ny x nx array to allow for possibly different OTF's along directions (e.g. OPM-SIM)
        if otf.ndim == 2:
            otf = np.tile(otf, [self.nangles, 1, 1])

        self.otf = xp.array(otf)

        if self.otf.shape[-2:] != self.imgs.shape[-2:]:
            raise ValueError(f"OTF shape {self.otf.shape} and image shape {self.img.shape} are not compatible")

        # #############################################
        # remove background and convert from ADU to photons
        # #############################################
        # todo: this should probably be users responsibility before here?
        self.imgs = (self.imgs - background) / gain
        self.imgs[self.imgs <= 0] = 1e-12

        # #############################################
        # normalize histograms for each angle
        # #############################################
        if self.normalize_histograms:
            tstart_norm_histogram = time.perf_counter()


            matched_hists = da.map_blocks(match_histograms,
                                          self.imgs[..., slice(1, None), :, :],
                                          self.imgs[..., slice(0, 1), :, :],
                                          chunks=(1,) * (self.n_extra_dims + 2) + self.imgs.shape[-2:],
                                          meta=xp.array(()))
                                          #dtype=self.imgs.dtype)

            self.imgs = da.concatenate((self.imgs[..., slice(0, 1), :, :],
                                        matched_hists),
                                       axis=-3)

            self.print_log(f"Normalizing histograms took {time.perf_counter() - tstart_norm_histogram:.2f}s")

        # #############################################
        # Rechunk so working on single image at a time, which is necessary during reconstruction
        # #############################################
        new_chunks = list(self.imgs.chunksize)
        new_chunks[-4:] = self.imgs.shape[-4:]

        self.imgs = da.rechunk(self.imgs, new_chunks)

        # #############################################
        # Fourier transform SIM images
        # #############################################
        tstart = time.perf_counter()

        # real-space apodization is not so desirable because produces a roll off in the reconstruction. But seems ok.
        apodization = xp.array(np.outer(tukey(self.imgs.shape[-2], alpha=0.1),
                                        tukey(self.imgs.shape[-1], alpha=0.1)))
        # apodization = xp.expand_dims(xp.array(apodization), axis=tuple(range(self.imgs.ndim - 2)))

        # todo: when try to run fft on large arrays, much more memory than the final array is allocated
        # todo: and don't understand how to get rid of it. Possibly the individual worker processes
        # todo: each have caches that don't get deleted?
        # todo: related to? https://github.com/cupy/cupy/issues/6355
        def ft(m, use_gpu):
            # avoid issues like https://github.com/cupy/cupy/issues/6355
            if use_gpu:
                cp.fft._cache.PlanCache(memsize=0)

            result = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

            return result

        self.imgs_ft = da.map_blocks(ft, self.imgs * apodization, self.use_gpu, dtype=complex)

        self.print_log(f"FT images took {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # get widefield image
        # #############################################
        tstart = time.perf_counter()

        self.widefield = da.nanmean(self.imgs, axis=(-3, -4))
        self.widefield_ft = da.map_blocks(ft, self.widefield * apodization, self.use_gpu, dtype=complex)

        self.print_log(f"Computing widefield image took {time.perf_counter() - tstart:.2f}s")


    def estimate_parameters(self,
                            slices: Optional[tuple] = None):
        """
        Estimate SIM parameters
        @return:
        """
        self.print_log("starting parameter estimation...")

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        if slices is None:
            slices = tuple([slice(None) for _ in range(self.n_extra_dims)])

        # always average over first dims after slicing...

        imgs = da.mean(self.imgs[slices], axis=tuple(range(self.n_extra_dims))).compute()
        # mempool = cp.get_default_memory_pool()
        # mem_start = mempool.used_bytes()

        imgs_ft = da.mean(self.imgs_ft[slices], axis=tuple(range(self.n_extra_dims))).compute()

        # print((mempool.used_bytes() - mem_start) / 1e9)

        if imgs_ft.shape[0] != self.nangles or imgs_ft.shape[1] != self.nphases or imgs_ft.ndim != 4:
            raise ValueError()

        # #############################################
        # estimate frequencies
        # #############################################
        tstart = time.perf_counter()

        if self.frq_estimation_mode == "fixed":
            self.frqs = self.frqs_guess
        else:
            if self.frq_estimation_mode == "fourier-transform":
                # determine SIM frequency directly from Fourier transform
                band0 = imgs_ft[:, 0]
                band1 = imgs_ft[:, 0]

            elif self.frq_estimation_mode == "band-correlation":
                # determine SIM frequency from separated frequency bands using guess phases
                bands_unmixed_ft_temp = unmix_bands(imgs_ft,
                                                    self.phases_guess,
                                                    mod_depths=np.ones((self.nangles)))

                band0 = bands_unmixed_ft_temp[:, 0]
                band1 = bands_unmixed_ft_temp[:, 1]

            else:
                raise ValueError(f"frq_estimation_mode must be 'fixed', 'fourier-transform', or 'band-correlation'"
                                 f" but was '{self.frq_estimation_mode:s}'")

            # do frequency guess (note this is not done on GPU because scipy.optimize not supported by CuPy)
            if self.frqs_guess is not None:
                frq_guess = self.frqs_guess
            else:
                frq_guess = [None] * self.nangles

            if self.use_gpu:
                band0 = band0.get()
                band1 = band1.get()

            self.band0_frq_fit = band0
            self.band1_frq_fit = band1

            r = []
            for ii in range(self.nangles):
                r.append(dask.delayed(fit_modulation_frq)(
                    self.band0_frq_fit[ii],
                    self.band1_frq_fit[ii],
                    self.dx,
                    frq_guess=frq_guess[ii],
                    max_frq_shift=5 * self.dfx)
                )
            results = dask.compute(*r)
            frqs, _, _ = zip(*results)
            self.frqs = np.array(frqs).reshape((self.nangles, 2))

        # for convenience also store periods and angles
        self.periods = 1 / np.linalg.norm(self.frqs, axis=-1)
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
            otf_vals[ii] = self.otf[..., ii, iy, ix]

        self.otf_at_frqs = otf_vals

        # #############################################
        # estimate peak heights
        # #############################################
        tstart = time.perf_counter()

        noise = np.sqrt(get_noise_power(imgs_ft,
                                        self.fx,
                                        self.fy,
                                        self.fmax))


        peak_phases = xp.zeros((self.nangles, self.nphases))
        peak_heights = xp.zeros((self.nangles, self.nphases))
        p2nr = xp.zeros((self.nangles, self.nphases))
        for ii in range(self.nangles):
            peak_val = tools.get_peak_value(imgs_ft[ii],
                                            self.fx,
                                            self.fy,
                                            self.frqs[ii],
                                            peak_pixel_size=1)
            peak_heights[ii] = xp.abs(peak_val)
            peak_phases[ii] = xp.angle(peak_val)
            p2nr[ii] = peak_heights[ii] / noise[ii]

            # if p2nr is too low use guess values instead
            if np.min(p2nr[ii]) < self.min_p2nr and self.frqs_guess is not None:
                self.frqs[ii] = self.frqs_guess[ii]
                self.print_log(f"SIM peak-to-noise ratio for angle={ii:d} is"
                               f" {np.min(p2nr[ii]):.2f} < {self.min_p2nr:.2f}, the so frequency fit will be ignored"
                               f"and the guess value will be used instead.")

                peak_val = tools.get_peak_value(imgs_ft[ii],
                                                self.fx,
                                                self.fy,
                                                self.frqs[ii],
                                                peak_pixel_size=1)
                peak_heights[ii] = np.abs(peak_val)
                peak_phases[ii] = np.angle(peak_val)
                p2nr[ii] = peak_heights[ii] / noise[ii]

        if self.use_gpu:
            self.p2nr = p2nr.get()
            self.peak_phases = peak_phases.get()
        else:
            self.p2nr = p2nr
            self.peak_phases = peak_phases

        self.print_log(f"estimated peak-to-noise ratio in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # estimate phases
        # todo: as with frqs since cannot easily go on GPU ...
        # #############################################
        tstart = time.perf_counter()

        if self.phase_estimation_mode == "fixed":
            phases = self.phases_guess
        elif self.phase_estimation_mode == "naive":
            phases = self.peak_phases
        elif self.phase_estimation_mode == "wicker-iterative":
            phase_guess = self.phases_guess
            if phase_guess is None:
                phase_guess = [None] * self.nangles

            imft = imgs_ft
            otfs = self.otf
            if self.use_gpu:
                imft = imft.get()
                otfs = otfs.get()

            r = []
            for ii in range(self.nangles):
                r.append(dask.delayed(get_phase_wicker_iterative)(
                    imft[ii],
                    otfs[ii],
                    self.frqs[ii],
                    self.dx,
                    self.fmax,
                    phases_guess=phase_guess[ii],
                    fit_amps=self.determine_amplitudes
                ))
            results = dask.compute(*r)
            phases, amps, _ = zip(*results)
            phases = np.array(phases)
            amps = np.array(amps)

        elif self.phase_estimation_mode == "real-space":
            phase_guess = self.phases_guess
            if phase_guess is None:
                phase_guess = np.zeros((self.nangles, self.nphases))

            im = imgs
            if self.use_gpu:
                im = im.get()

            r = []
            for ii in range(self.nangles):
                for jj in range(self.nphases):
                    r.append(dask.delayed(get_phase_realspace)(
                        im[ii, jj],
                        self.frqs[ii],
                        self.dx,
                        phase_guess=phase_guess[ii, jj],
                        origin="center"
                    ))
            results = dask.compute(*r)
            phases = np.array(results).reshape((self.nangles, self.nphases))
            amps = np.ones((self.nangles, self.nphases))
        else:
            raise ValueError("phase_estimation_mode must be 'wicker-iterative', 'real-space', 'naive', or 'fixed'"
                             f" but was '{self.phase_estimation_mode}'")

        self.phases = np.array(phases)
        self.amps = np.array(amps)

        self.print_log(f"estimated {self.nangles * self.nphases:d} phases"
                       f" using mode {self.phase_estimation_mode:s} "
                       f"in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # check if phase fit was too bad, and default to guess values
        # #############################################
        if self.phases_guess is not None:
            phase_guess_diffs = np.mod(self.phases_guess - self.phases_guess[:, 0][:, None], 2 * np.pi)
            phase_diffs = np.mod(self.phases - self.phases[:, 0][:, None], 2 * np.pi)

            for ii in range(self.nangles):
                diffs = np.mod(phase_guess_diffs[ii] - phase_diffs[ii], 2 * np.pi)
                condition = np.abs(diffs - 2 * np.pi) < diffs
                diffs[condition] = diffs[condition] - 2 * np.pi

                if np.any(np.abs(diffs) > self.max_phase_error):
                    self.phases[ii] = self.phases_guess[ii]
                    strv = f"Angle {ii:d} phase guesses have more than the maximum allowed" \
                           f" phase error={self.max_phase_error * 180 / np.pi:.2f}deg." \
                           f" Defaulting to guess values"

                    strv += "\nfit phase diffs="
                    for jj in range(self.nphases):
                        strv += f"{phase_diffs[ii, jj] * 180 / np.pi:.2f}deg, "

                    self.print_log(strv)

        # #############################################
        # estimate global phase shifts/modulation depths
        # #############################################
        tstart_mod_depth = time.perf_counter()

        # do band separation
        bands_unmixed_ft = unmix_bands(imgs_ft, self.phases, amps=self.amps)
        bands_shifted_ft = shift_bands(bands_unmixed_ft, self.frqs, (self.dy, self.dx), self.upsample_fact)

        # upsample and shift OTFs
        otf_us = resample_bandlimited_ft(self.otf,
                                         (self.upsample_fact, self.upsample_fact),
                                         axes=(-1, -2)) / self.upsample_fact / self.upsample_fact

        # todo: want to be able to run this with map_blocks() like tools.translate_ft()
        otf_shifted = xp.zeros((otf_us.shape[0], self.nbands, otf_us.shape[1], otf_us.shape[2]),
                                dtype=complex)

        for ii in range(self.nangles):
            for jj, band_ind in enumerate(self.band_inds):
                # compute otf(k + m * ko)
                otf_shifted[ii, jj], _ = tools.translate_pix(otf_us[ii],
                                                             self.frqs[ii] * band_ind,
                                                             dr=(self.dfx_us, self.dfy_us),
                                                             axes=(1, 0),
                                                             wrap=False)

        # correct global phases and estimate modulation depth from band correlations

        # mask regions where OTF's are below threshold
        mask = xp.logical_and(otf_shifted[:, 0] > self.otf_mask_threshold,
                              otf_shifted[:, 1] > self.otf_mask_threshold)

        # mask regions near frequency modulation which may be corrupted by out-of-focus light
        if self.fmax_exclude_band0 > 0:
            for ii in range(self.nangles):
                # exclude positive freq
                ff_us = xp.sqrt(xp.expand_dims(self.fx_us, axis=0) ** 2 +
                                xp.expand_dims(self.fy_us, axis=1) ** 2)
                mask[ii][ff_us < self.fmax * self.fmax_exclude_band0] = False

                # exclude negative frq
                ff_us = xp.sqrt(xp.expand_dims(self.fx_us + self.frqs[ii, 0], axis=0) ** 2 +
                                xp.expand_dims(self.fy_us + self.frqs[ii, 1], axis=1) ** 2)
                mask[ii][ff_us < self.fmax * self.fmax_exclude_band0] = False

        for ii in range(self.nangles):
            if not np.any(mask[ii]):
                raise ValueError(f"band overlap mask for angle {ii:d} was all False. "
                                 f"This may indicate the SIM frequency is incorrect. Check if the frequency "
                                 f"fitting routine failed. Otherwise, reduce `otf_mask_threshold` "
                                 f"which is currently {self.otf_mask_threshold:.3f} and/or reduce "
                                 f"`fmax_exclude_band0` which is currently {self.fmax_exclude_band0:.3f}")

        # corrected phases
        # can either think of these as (1) acting on phases such that phase -> phase - phase_correction
        # or (2) acting on bands such that band1(f) -> e^{i*phase_correction} * band1(f)
        # TODO: note, the global phases I use here have the opposite sign relative to our BOE paper eq. S47
        global_phase_corrections, mags = get_band_overlap(bands_shifted_ft[..., 0, :, :],
                                                          bands_shifted_ft[..., 1, :, :],
                                                          otf_shifted[..., 0, :, :],
                                                          otf_shifted[..., 1, :, :],
                                                          mask)



        if self.phase_estimation_mode == "fixed":
            self.phase_corrections = np.zeros(global_phase_corrections.shape)
        else:
            self.phase_corrections = global_phase_corrections

        if self.use_fixed_mod_depths:
            self.print_log("using fixed modulation depth")
            self.mod_depths = self.mod_depths_guess
        else:
            self.mod_depths = mags

            for ii in range(self.nangles):
                if self.mod_depths[ii] < self.minimum_mod_depth:
                    self.print_log(f"replaced modulation depth for angle {ii:d} because estimated value"
                                   f" was less than allowed minimum,"
                                   f" {self.mod_depths[ii]:.3f} < {self.minimum_mod_depth:.3f}")
                    self.mod_depths[ii] = self.minimum_mod_depth

        self.print_log(f"estimated global phases and modulation depths in {time.perf_counter() - tstart_mod_depth:.2f}s")

    def reconstruct(self,
                    slices: Optional[tuple] = None,
                    compute_os: bool = True,
                    compute_deconvolved: bool = True,
                    compute_mcnr: bool = True):
        """
        SIM reconstruction
        """

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        # #############################################
        # parameter estimation
        # #############################################
        self.estimate_parameters(slices=slices)

        self.print_log("starting SIM reconstruction...")

        # #############################################
        # get optically sectioned image
        # #############################################
        if compute_os:
            tstart = time.perf_counter()

            os_imgs = da.stack([da.map_blocks(sim_optical_section,
                                              self.imgs[..., ii, :, :, :],
                                              phase_differences=self.phases[ii],
                                              axis=-3,
                                              drop_axis=-3,
                                              dtype=float)
                                for ii in range(self.nangles)],
                               axis=-3)
            self.sim_os = da.mean(os_imgs, axis=-3)
            self.print_log(f"Computing SIM-OS image took {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # estimate spatial-resolved MCNR
        # #############################################
        if compute_mcnr:
            # following the proposal of https://doi.org/10.1038/s41592-021-01167-7
            # calculate as the ratio of the modulation size over the expected shot noise value
            # note: this is the same as sim_os / sqrt(wf_angle) up to a factor
            tstart = time.perf_counter()

            # divide by nangles to remove ft normalization
            def ft_mcnr(m, use_gpu):
                if use_gpu:
                    cp.fft._cache.PlanCache(memsize=0)

                return xp.fft.fft(xp.fft.ifftshift(m, axes=-3), axis=-3) / self.nangles

            img_angle_ft = da.map_blocks(ft_mcnr, self.imgs, self.use_gpu, dtype=complex)
            # if I_j = Io * m * cos(2*pi*j), then want numerator to be 2*m. FT gives us m/2, so multiply by 4
            self.mcnr = (4 * da.abs(img_angle_ft[..., 1, :, :]) / da.sqrt(da.abs(img_angle_ft[..., 0, :, :])))

            self.print_log(f"estimated modulation-contrast-to-noise ratio in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # do band separation
        # bands are [O(f)H(f), m*O(f - f_o)H(f), m*O(f + f_o)H(f)]
        # #############################################
        tstart = time.perf_counter()

        self.bands_unmixed_ft = da.map_blocks(unmix_bands,
                                              self.imgs_ft,
                                              self.phases,
                                              amps=self.amps,
                                              dtype=complex)

        self.print_log(f"separated bands in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # shift bands
        # #############################################
        tstart = time.perf_counter()

        exp_chunks = list(self.bands_unmixed_ft.chunksize)
        exp_chunks[-1] *= self.upsample_fact
        exp_chunks[-2] *= self.upsample_fact

        self.bands_shifted_ft = da.map_blocks(shift_bands,
                                              self.bands_unmixed_ft,
                                              self.frqs,
                                              (self.dy, self.dx),
                                              self.upsample_fact,
                                              dtype=complex,
                                              chunks=exp_chunks
                                              )

        self.print_log(f"shifted bands in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # shift OTFs
        # #############################################
        tstart = time.perf_counter()

        otf_us = resample_bandlimited_ft(self.otf,
                                         (self.upsample_fact, self.upsample_fact),
                                         axes=(-1, -2)) / self.upsample_fact / self.upsample_fact

        # todo: want to be able to run this with map_blocks() like tools.translate_ft()
        self.otf_shifted = xp.zeros((otf_us.shape[0], self.nbands, otf_us.shape[1], otf_us.shape[2]), dtype=complex)
        for ii in range(self.nangles):
            for jj, band_ind in enumerate(self.band_inds):
                # compute otf(k + m * ko)
                self.otf_shifted[ii, jj], _ = tools.translate_pix(otf_us[ii],
                                                                  self.frqs[ii] * band_ind,
                                                                  dr=(self.dfx_us, self.dfy_us),
                                                                  axes=(1, 0),
                                                                  wrap=False)

        self.print_log(f"shifted otfs in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # Get weights and combine bands
        # #############################################
        tstart = time.perf_counter()

        if self.combine_bands_mode != "fairSIM":
            raise ValueError("combined_mode must be 'fairSIM'")
        # following the approach of FairSIM: https://doi.org/10.1038/ncomms10980
        self.weights = self.otf_shifted.conj()
        weights_decon = xp.array(self.weights, copy=True)

        # "fill in missing cone" by using shifted bands instead of unshifted band for values near DC
        if self.fmax_exclude_band0 > 0:
            for ii in range(self.nangles):
                for jj, ee in enumerate(self.band_inds):
                    # correct unshifted band weights near zero frequency
                    ff_us = xp.sqrt(xp.expand_dims(self.fx_us + ee * self.frqs[ii, 0], axis=0)**2 +
                                    xp.expand_dims(self.fy_us + ee * self.frqs[ii, 1], axis=1)**2)

                    # gaussian smoothing for weight
                    self.weights[ii, jj] *= (1 - xp.exp(-0.5 * ff_us**2 / (self.fmax * self.fmax_exclude_band0)**2))

        self.weights_norm = self.wiener_parameter**2 + xp.nansum(xp.abs(self.weights) ** 2, axis=(0, 1), keepdims=True)

        self.print_log(f"computed band weights in {time.perf_counter() - tstart:.2f}s")

        # combine bands
        tstart = time.perf_counter()

        # nangles x nphases
        corr_mat = xp.concatenate((xp.ones((self.nangles, 1)),
                                   xp.expand_dims(np.exp(1j * self.phase_corrections) / self.mod_depths, axis=1),
                                   xp.expand_dims(np.exp(-1j * self.phase_corrections) / self.mod_depths, axis=1)),
                                   axis=1)
        # expand for extra dims and xy
        # todo: think don't need last expansion bc of numpy broadcasting rules
        corr_mat = xp.expand_dims(corr_mat, axis=tuple(range(self.n_extra_dims)) + (-1, -2))

        # put in modulation depth and global phase corrections
        # components array useful for diagnostic plots
        self.sim_sr_ft_components = self.bands_shifted_ft * self.weights * corr_mat / self.weights_norm
        # final FT image
        self.sim_sr_ft = da.nansum(self.sim_sr_ft_components, axis=(-3, -4))

        # inverse FFT to get real-space reconstructed image
        apodization = np.outer(tukey(self.sim_sr_ft.shape[-2], alpha=0.1),
                               tukey(self.sim_sr_ft.shape[-1], alpha=0.1))
        apodization = xp.array(apodization)

        # irfft2 ~2X faster than ifft2. But have to slice out only half the frequencies
        # sim_sr_ft_one_side = xp.fft.ifftshift(self.sim_sr_ft * apodization, axes=(-1, -2))[:, : self.sim_sr_ft.shape[-2] // 2 + 1]
        # self.sim_sr = xp.fft.fftshift(xp.fft.irfft2(sim_sr_ft_one_side, axes=(-1, -2)), axes=(-1, -2))
        def ift(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2)).real

        self.sim_sr = da.map_blocks(ift, self.sim_sr_ft * apodization, dtype=float)
        if self.trim_negative_values:
            self.sim_sr[self.sim_sr < 0] = 0

        self.print_log(f"combining bands using mode '{self.combine_bands_mode}'"
                       f" and Wiener parameter {self.wiener_parameter:.3f}"
                       f" took {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # widefield deconvolution
        # #############################################
        if compute_deconvolved:
            tstart = time.perf_counter()

            self.widefield_deconvolution_ft = da.nansum(weights_decon[..., 0, :, :] * self.bands_shifted_ft[..., 0, :, :], axis=-3) / \
                                                  (self.wiener_parameter**2 + da.nansum(np.abs(weights_decon[..., 0, :, :])**2, axis=-3))

            self.widefield_deconvolution = da.map_blocks(ift, self.widefield_deconvolution_ft * apodization, dtype=float)

            self.print_log(f"Deconvolved widefield in {time.perf_counter() - tstart:.2f}s")

        # #############################################
        # move arrays off GPU
        # #############################################
        if self.use_gpu:
            self.fx = self.fx.get()
            self.fy = self.fy.get()
            self.fx_us = self.fx_us.get()
            self.fy_us = self.fy_us.get()
            self.x = self.x.get()
            self.y = self.y.get()
            self.x_us = self.x_us.get()
            self.y_us = self.y_us.get()

            def tocpu(c: cp.ndarray): return c.get()

            self.imgs = da.map_blocks(tocpu, self.imgs, dtype=self.imgs.dtype)
            self.imgs_ft = da.map_blocks(tocpu, self.imgs_ft, dtype=self.imgs_ft.dtype)
            self.otf = self.otf.get()

            self.mcnr = da.map_blocks(tocpu, self.mcnr, dtype=self.mcnr.dtype)
            self.widefield = da.map_blocks(tocpu, self.widefield, dtype=self.widefield.dtype)
            self.widefield_ft = da.map_blocks(tocpu, self.widefield_ft, dtype=self.widefield_ft.dtype)

            if hasattr(self, "widefield_deconvolution"):
                self.widefield_deconvolution = da.map_blocks(tocpu, self.widefield_deconvolution, dtype=self.widefield_deconvolution.dtype)
                self.widefield_deconvolution_ft = da.map_blocks(tocpu, self.widefield_deconvolution_ft, dtype=self.widefield_deconvolution.dtype)

            if hasattr(self, "sim_os"):
                self.sim_os = da.map_blocks(tocpu, self.sim_os, dtype=self.sim_os.dtype)

            self.sim_sr = da.map_blocks(tocpu, self.sim_sr, dtype=self.sim_sr.dtype)
            self.sim_sr_ft = da.map_blocks(tocpu, self.sim_sr_ft, dtype=self.sim_sr_ft.dtype)
            self.bands_unmixed_ft = da.map_blocks(tocpu, self.bands_unmixed_ft, dtype=self.bands_unmixed_ft.dtype)
            self.bands_shifted_ft = da.map_blocks(tocpu, self.bands_shifted_ft, dtype=self.bands_shifted_ft.dtype)
            self.weights = self.weights.get()
            self.weights_norm = self.weights_norm.get()
            self.sim_sr_ft_components = da.map_blocks(tocpu, self.sim_sr_ft_components, dtype=self.sim_sr_ft_components.dtype)


        # #############################################
        # print parameters
        # #############################################
        self.print_parameters()


    # printing utility functions
    def print_parameters(self):

        self.print_log(f"SIM reconstruction for {self.nangles:d} angles and {self.nphases:d} phases")
        self.print_log(f"images are size {self.ny:d}x{self.nx:d} with pixel size {self.dx:.3f}x{self.dy:.3f}um")
        self.print_log(f"emission wavelength={self.wavelength*1e3:.0f}nm and NA={self.na:.2f}")
        self.print_log(f"'{self.frq_estimation_mode:s}' frequency estimation mode")
        self.print_log(f"'{self.phase_estimation_mode:s}' phase estimation mode")
        self.print_log(f"'{self.combine_bands_mode:s}' band combination mode")
        self.print_log(f"excluded {self.fmax_exclude_band0:.2f} from bands around centers")
        self.print_log(f"wiener parameter = {self.wiener_parameter:.2f}")

        for ii in range(self.nangles):
            self.print_log(f"################ Angle {ii:d} ################")

            # amplitudes
            self.print_log("amps = ", end="")
            for jj in range(self.nphases - 1):
                self.print_log(f"{self.amps[ii, jj]:05.3f}, ", end="")
            self.print_log(f"{self.amps[ii, self.nphases - 1]:05.3f}")

            #  peak-to-noise ratio
            self.print_log("peak-to-camera-noise ratios = %0.3f, %0.3f, %0.3f" % tuple(self.p2nr[ii]))

            # modulation depth
            self.print_log(f"modulation depth = {self.mod_depths[ii]:0.3f}")

            # global phase correction
            self.print_log(f"global phase correction={self.phase_corrections[ii] * 180 / np.pi:.2f}deg")

            # frequency and period data
            if self.frqs_guess is not None:
                angle_guess = np.angle(self.frqs_guess[ii, 0] + 1j * self.frqs_guess[ii, 1])
                period_guess = 1 / np.linalg.norm(self.frqs_guess[ii])

                self.print_log("Frequency guess= ({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                    self.frqs_guess[ii, 0],
                    self.frqs_guess[ii, 1],
                    period_guess * 1e3,
                    angle_guess * 180 / np.pi))

            self.print_log("Frequency fit  = ({:+8.5f}, {:+8.5f}), period={:0.3f}nm, angle={:07.3f}deg".format(
                self.frqs[ii, 0],
                self.frqs[ii, 1],
                self.periods[ii] * 1e3,
                self.angles[ii] * 180 / np.pi))

            # phase information
            self.print_log("peaks   = ", end="")
            for jj in range(self.nphases - 1):
                self.print_log(f"{np.mod(self.peak_phases[ii, jj], 2*np.pi) * 180 / np.pi:07.2f}deg, ", end="")
            self.print_log(f"{np.mod(self.peak_phases[ii, self.nphases - 1], 2*np.pi) * 180 / np.pi:07.2f}deg")

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


    def print_log(self,
                  string: str,
                  **kwargs):
        """
        Print result to stdout and to a log file.

        :param string: string to print
        :param kwargs: passed through to print()
        """

        print(string, **kwargs)
        print(string, **kwargs, file=self.log)


    # plotting utility functions
    def plot_figs(self,
                  slices: Optional[tuple[slice]] = None,
                  figsize: tuple[float] = (20, 10),
                  diagnostics_only: bool = False,
                  imgs_dpi: int = None):

        """
        Automate plotting and saving of figures

        @param slices: tuple of slices indicating which image to plot. len(slices) = self.imgs.ndim - 4
        @param figsize:
        @param reconstruct_dpi: Set to 400 for high resolution, but long saving
        @return:
        """

        # figures/names to output
        figs = []
        fig_names = []

        # get slice to display
        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        tstart = time.perf_counter()

        saving = self.save_dir is not None

        # plot MCNR diagnostic
        fnow, fnames_now = self.plot_mcnr_diagnostic(slices, figsize=figsize)

        figs += fnow
        fig_names += fnames_now

        figh = fnow[0]
        fname = fnames_now[0]

        if saving:
            figh.savefig(self.save_dir / f"{self.save_prefix:s}{fname:s}{self.save_suffix:s}.png",
                         dpi=imgs_dpi)
        if not self.interactive_plotting:
            plt.close(figh)


        # plot frequency fits
        fighs, fig_names_now = self.plot_frequency_fits(figsize=figsize)

        figs += fighs
        fig_names += fig_names_now

        for fh, fn in zip(fighs, fig_names_now):
            if saving:
                fh.savefig(self.save_dir / f"{self.save_prefix:s}{fn:s}{self.save_suffix:s}.png")
            if not self.interactive_plotting:
                plt.close(fh)


        # plot filters used in reconstruction
        fighs, fig_names_now = self.plot_reconstruction_diagnostics(slices, figsize=figsize)

        figs += fighs
        fig_names += fig_names_now

        for fh, fn in zip(fighs, fig_names_now):
            if saving:
                fh.savefig(self.save_dir / f"{self.save_prefix:s}{fn:s}{self.save_suffix:s}.png",
                           dpi=imgs_dpi)
            if not self.interactive_plotting:
                plt.close(fh)


        # plot otf
        fig = self.plot_otf(figsize=figsize)
        fig_name_now = "otf"

        figs += [fig]
        fig_names += [fig_name_now]

        if saving:
            fig.savefig(self.save_dir / f"{self.save_prefix:s}{fig_name_now:s}{self.save_suffix:s}.png")
        if not self.interactive_plotting:
            plt.close(fig)

        if not diagnostics_only:
            # plot reconstruction results
            fig = self.plot_reconstruction(slices, figsize=figsize)
            fig_name_now = "sim_reconstruction"

            figs += [fig]
            fig_names += [fig_name_now]

            if saving:
                fig.savefig(self.save_dir / f"{self.save_prefix:s}{fig_name_now:s}{self.save_suffix:s}.png",
                            dpi=imgs_dpi)
            if not self.interactive_plotting:
                plt.close(fig)

        tend = time.perf_counter()
        self.print_log(f"plotting results took {tend - tstart:.2f}s")

        return figs, fig_names


    def plot_mcnr_diagnostic(self,
                             slices: Optional[tuple[slice]] = None,
                             figsize=(20, 10)) -> (tuple, tuple[str]):
        """
        Display SIM images for visual inspection. Use this to examine SIM pictures and their Fourier transforms
        as an aid to guessing frequencies before doing reconstruction.

        @param slices: tuple of slices indicating which image to plot. len(slices) = self.imgs.ndim - 4
        @param figsize:
        @return figs, fig_names:
        """

        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        # get slice to plot
        imgs_slice_list = slices + (slice(None),) * 4
        mcnr_slice_list = slices + (slice(None),) * 3

        imgs = self.imgs[imgs_slice_list].squeeze()
        if isinstance(imgs, da.core.Array):
            imgs = imgs.compute()

        imgs_ft = self.imgs_ft[imgs_slice_list].squeeze()
        if isinstance(imgs_ft, da.core.Array):
            imgs_ft = imgs_ft.compute()

        if hasattr(self, "mcnr"):
            mcnr = self.mcnr[mcnr_slice_list].squeeze()
            if isinstance(mcnr, da.core.Array):
                mcnr = mcnr.compute()

            vmax_mcnr = np.percentile(mcnr, 99)

        #
        extent = get_extent(self.y, self.x)
        extentk = get_extent(self.fx, self.fy)

        # parameters for real space plot
        vmin = np.percentile(imgs.ravel(), 0.1)
        vmax = np.percentile(imgs.ravel(), 99.9)

        # to avoid errors with image that has only one value
        if vmax <= vmin:
            vmax += 1e-12

        # ########################################
        # plot real-space
        # ########################################
        figh = plt.figure(figsize=figsize)
        figh.suptitle(f"SIM MCNR diagnostic, index={[s.start for s in slices[:-4]]}")

        n_factor = 4  # colorbar will be 1/n_factor of images
        # 5 types of plots + 2 colorbars
        grid = figh.add_gridspec(self.nangles, n_factor * (self.nphases + 2) + 3)
        
        mean_int = np.mean(imgs, axis=(2, 3))
        rel_int_phases = mean_int / np.expand_dims(np.max(mean_int, axis=1), axis=1)
        
        mean_int_angles = np.mean(imgs, axis=(1, 2, 3))
        rel_int_angles = mean_int_angles / np.max(mean_int_angles)

        for ii in range(self.nangles):
            for jj in range(self.nphases):

                # ########################################
                # raw real-space SIM images
                # ########################################

                ax = figh.add_subplot(grid[ii, n_factor*jj:n_factor*(jj+1)])
                ax.imshow(imgs[ii, jj], vmin=vmin, vmax=vmax, extent=extent, interpolation=None, cmap="bone")

                if ii == 0:
                    ax.set_title(f"phase {jj:d}")
                if jj == 0:
                    tstr = f'angle {ii:d}, relative intensity={rel_int_angles[ii]:.3f}\nphase int='
                    for aa in range(self.nphases):
                        tstr += f"{rel_int_phases[ii, aa]:.3f}, "
                    ax.set_ylabel(tstr)
                if ii == (self.nangles - 1):
                    ax.set_xlabel("Position (um)")

                if jj != 0:
                    ax.set_yticks([])

            # ########################################
            # histograms of real-space images
            # ########################################
            nbins = 50
            bin_edges = np.linspace(0, np.percentile(imgs, 99), nbins + 1)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            ax = figh.add_subplot(grid[ii, n_factor*self.nphases + 2:n_factor*(self.nphases+1) + 2])
            for jj in range(self.nphases):
                histogram, _ = np.histogram(imgs[ii, jj].ravel(), bins=bin_edges)
                ax.semilogy(bin_centers, histogram)
                ax.set_xlim([0, bin_edges[-1]])

            ax.set_yticks([])
            if ii == 0:
                ax.set_title("image histogram\nmedian %0.1f" % (np.median(imgs[ii, jj].ravel())))
            else:
                ax.set_title("median %0.1f" % (np.median(imgs[ii, jj].ravel())))
            if ii != (self.nangles - 1):
                ax.set_xticks([])
            else:
                ax.set_xlabel("counts")

            # ########################################
            # spatially resolved mcnr
            # ########################################
            if hasattr(self, "mcnr"):
                ax = figh.add_subplot(grid[ii, n_factor*(self.nphases + 1) + 2:n_factor*(self.nphases+2) + 2])
                if vmax_mcnr <= 0:
                    vmax_mcnr += 1e-12

                im = ax.imshow(mcnr[ii], vmin=0, vmax=vmax_mcnr, cmap="inferno")
                ax.set_xticks([])
                ax.set_yticks([])
                if ii == 0:
                    ax.set_title("mcnr")

        # colorbar for images
        ax = figh.add_subplot(grid[:, n_factor*self.nphases])
        norm = PowerNorm(vmin=vmin, vmax=vmax, gamma=1)
        plt.colorbar(ScalarMappable(norm=norm, cmap="bone"), cax=ax)

        if hasattr(self, "mcnr"):
            # colorbar for MCNR
            ax = figh.add_subplot(grid[:, n_factor*(self.nphases + 2) + 2])
            norm = PowerNorm(vmin=0, vmax=vmax_mcnr, gamma=1)
            plt.colorbar(ScalarMappable(norm=norm, cmap="inferno"), cax=ax, label="MCNR")

        # # ########################################
        # # plot k-space
        # # ########################################
        # fighk = plt.figure(figsize=figsize)
        # fighk.suptitle(f"SIM k-space diagnostic, index={[s.start for s in slices[:-4]]}")
        #
        # gridk = fighk.add_gridspec(self.nangles, self.nphases)
        #
        # norm = PowerNorm(vmin=0, vmax=np.nanmax(np.abs(imgs_ft)), gamma=0.1)
        # for ii in range(self.nangles):
        #     for jj in range(self.nphases):
        #
        #         ax = fighk.add_subplot(gridk[ii, jj])
        #         ax.imshow(np.abs(imgs_ft[ii, jj]), norm=norm, extent=extentk, cmap="bone")
        #
        #         # plot any frequency data already available
        #         if hasattr(self, "frqs_guess") and self.frqs_guess is not None:
        #             ax.scatter(-self.frqs_guess[ii, 0], -self.frqs_guess[ii, 1], edgecolor='m', facecolor='none')
        #
        #         if hasattr(self, "frqs"):
        #             ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
        #
        #         ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=0, ls='--'))
        #
        #         if jj == 0:
        #             ax.set_ylabel(f"angle = {ii:d}\n $f_y$ (1/um)")
        #         if ii == (self.nphases - 1):
        #             ax.set_xlabel("$f_x$ (1/um)")

        return [figh], ["mcnr_diagnostic"]


    def plot_reconstruction(self,
                            slices: Optional[tuple[slice]] = None,
                            figsize=(20, 10),
                            gamma=0.1):
        """
        Plot SIM image and compare with 'widefield' image
        :return:
        """

        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        wf_slice_list = slices + (slice(None),) * 2

        widefield = self.widefield[wf_slice_list].squeeze()
        widefield_ft = self.widefield_ft[wf_slice_list].squeeze()
        widefield_deconvolution = self.widefield_deconvolution[wf_slice_list].squeeze()
        widefield_deconvolution_ft = self.widefield_deconvolution_ft[wf_slice_list].squeeze()
        sim_sr = self.sim_sr[slices].squeeze()
        sim_sr_ft = self.sim_sr_ft[slices].squeeze()
        sim_os = self.sim_os[slices].squeeze()

        extent_wf = get_extent(self.fy, self.fx)
        extent_rec = get_extent(self.fy_us, self.fx_us)
        extent_wf_real = get_extent(self.y, self.x)
        extent_us_real = get_extent(self.y_us, self.x_us)

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

        vmin = np.percentile(widefield.ravel(), min_percentile)
        vmax = np.percentile(widefield.ravel(), max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(widefield, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_wf_real)
        ax.set_title('widefield')
        ax.set_xlabel('x-position ($\mu m$)')
        ax.set_ylabel('y-position ($\mu m$)')

        # deconvolved, real space
        ax = figh.add_subplot(grid[0, 1])

        vmin = np.percentile(widefield_deconvolution.ravel(), min_percentile)
        vmax = np.percentile(widefield_deconvolution.ravel(), max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(widefield_deconvolution, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_us_real)
        ax.set_title('widefield deconvolved')
        ax.set_xlabel('x-position ($\mu m$)')

        # SIM, realspace
        ax = figh.add_subplot(grid[0, 2])
        vmin = np.percentile(sim_sr.ravel()[sim_sr.ravel() >= 0], min_percentile)
        vmax = np.percentile(sim_sr.ravel()[sim_sr.ravel() >= 0], max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(sim_sr, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_us_real)
        ax.set_title('SR-SIM')
        ax.set_xlabel('x-position ($\mu m$)')

        #
        ax = figh.add_subplot(grid[0, 3])
        vmin = np.percentile(sim_os.ravel(), min_percentile)
        vmax = np.percentile(sim_os.ravel(), max_percentile)
        if vmax <= vmin:
            vmax += 1e-12
        ax.imshow(sim_os, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_wf_real)
        ax.set_title('OS-SIM')
        ax.set_xlabel('x-position ($\mu m$)')

        # widefield Fourier space
        ax = figh.add_subplot(grid[1, 0])
        ax.imshow(np.abs(widefield_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_wf, cmap="bone")

        ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))
        ax.add_artist(Circle((0, 0), radius=2 * self.fmax, color='r', fill=False, ls='--'))

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])
        ax.set_xlabel("$f_x (1/\mu m)$")
        ax.set_ylabel("$f_y (1/\mu m)$")

        # deconvolution Fourier space
        ax = figh.add_subplot(grid[1, 1])
        ax.imshow(np.abs(widefield_deconvolution_ft) ** 2,
                  norm=PowerNorm(gamma=gamma),
                  extent=extent_rec,
                  cmap="bone")

        ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))
        ax.add_artist(Circle((0, 0), radius=2 * self.fmax, color='r', fill=False, ls='--'))

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])
        ax.set_xlabel("$f_x (1/\mu m)$")

        # SIM fourier space
        ax = figh.add_subplot(grid[1, 2])
        ax.imshow(np.abs(sim_sr_ft) ** 2, norm=PowerNorm(gamma=gamma), extent=extent_rec, cmap="bone")

        ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))
        ax.add_artist(Circle((0, 0), radius=2 * self.fmax, color='r', fill=False, ls='--'))

        # actual maximum frequency based on real SIM frequencies
        for ii in range(self.nangles):
            ax.add_artist(Circle((0, 0), radius=self.fmax + 1/self.periods[ii], color='g', fill=False, ls='--'))

        ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
        ax.set_ylim([2 * self.fmax, -2 * self.fmax])
        ax.set_xlabel("$f_x (1/\mu m)$")

        return figh


    def plot_reconstruction_diagnostics(self,
                                        slices: Optional[tuple[slice]] = None,
                                        figsize=(20, 10)):
        """
        Diagnostics showing progress of SIM reconstruction

        This function can be called at any point in the reconstruction, and is useful for assessing if the guess
        frequencies are close to the actual peaks in the image, as well as the quality of the reconstruction

        :return figh: figure handle
        """
        figs = []
        fig_names = []

        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        slices_raw = slices + (slice(None),) * 4

        imgs_ft = self.imgs_ft[slices_raw].squeeze()
        if isinstance(imgs_ft, da.core.Array):
            imgs_ft = imgs_ft.compute()

        parameters_estimated = hasattr(self, "phases")
        reconstructed_already = hasattr(self, "bands_unmixed_ft")

        if reconstructed_already:
            bands_unmixed_ft = self.bands_unmixed_ft[slices_raw].squeeze()
            if isinstance(bands_unmixed_ft, da.core.Array):
                bands_unmixed_ft = bands_unmixed_ft.compute()

            bands_shifted_ft = self.bands_shifted_ft[slices_raw].squeeze()
            if isinstance(bands_shifted_ft, da.core.Array):
                bands_shifted_ft = bands_shifted_ft.compute()

            weights = self.weights
            weights_norm = self.weights_norm.squeeze()

            sim_sr_ft_components = self.sim_sr_ft_components[slices_raw].squeeze()
            if isinstance(sim_sr_ft_components, da.core.Array):
                sim_sr_ft_components = sim_sr_ft_components.compute()

        # ######################################
        # plot different stages of inversion process as diagnostic
        # ######################################
        extent = get_extent(self.fy, self.fx)
        extent_upsampled = get_extent(self.fy_us, self.fx_us)
        extent_upsampled_real = get_extent(self.y_us, self.x_us)

        # plot one image for each angle
        for ii in range(self.nangles):
            ttl_str = f"SIM bands Fourier space diagnostic, angle {ii:d}\n"

            if parameters_estimated:
                dp1 = np.mod(self.phases[ii, 1] - self.phases[ii, 0], 2 * np.pi)
                dp2 = np.mod(self.phases[ii, 2] - self.phases[ii, 0], 2 * np.pi)
                p0_corr = np.mod(self.phases[ii, 0] - self.phase_corrections[ii], 2 * np.pi)
                p1_corr = np.mod(self.phases[ii, 1] - self.phase_corrections[ii], 2 * np.pi)
                p2_corr = np.mod(self.phases[ii, 2] - self.phase_corrections[ii], 2 * np.pi)

                ttl_str += f'period={self.periods[ii] * 1e3:.3f}nm at ' \
                           f'{self.angles[ii] * 180 / np.pi:.2f}deg={self.angles[ii]:.3f}rad,' \
                           f' f=({self.frqs[ii, 0]:.3f},{self.frqs[ii, 1]:.3f}) 1/um\n' \
                           f'modulation contrast={self.mod_depths[ii]:.3f}, min p2nr={np.min(self.p2nr[ii]):.3f},' \
                           f' $\eta$={self.wiener_parameter:.2f},' \
                           f' global phase correction={self.phase_corrections[ii] * 180 / np.pi:.2f}deg\n' \
                           f' corrected phases (deg) = {p0_corr * 180 / np.pi:.2f},' \
                                                    f' {p1_corr * 180 / np.pi:.2f},' \
                                                    f' {p2_corr * 180 / np.pi:.2f};' \
                           f' phase diffs (deg) ={0:.2f}, {dp1 * 180/np.pi:.2f}, {dp2 * 180/np.pi:.2f}'

            fig = plt.figure(figsize=figsize)
            fig.suptitle(ttl_str)

            # 6 diagnostic images + 4 extra columns for colorbars
            grid = fig.add_gridspec(nrows=self.nphases,
                                    ncols=6 + 4,
                                    width_ratios=[1] * 4 + [0.2, 0.2] + [1] + [0.2, 0.2] + [1],
                                    wspace=0.1)

            for jj in range(self.nphases):

                # ####################
                # raw images at different phases
                # ####################
                ax = fig.add_subplot(grid[jj, 0])
                ax.set_title("Raw data, phase %d" % jj)

                to_plot = np.abs(imgs_ft[ii, jj])
                to_plot[to_plot <= 0] = np.nan

                im = ax.imshow(to_plot, norm=LogNorm(), extent=extent, cmap="bone")

                if parameters_estimated:
                    ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                    ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
                else:
                    if self.frqs_guess is not None:
                        ax.scatter(self.frqs_guess[ii, 0], self.frqs_guess[ii, 1], edgecolor='k', facecolor='none')
                        ax.scatter(-self.frqs_guess[ii, 0], -self.frqs_guess[ii, 1], edgecolor='k', facecolor='none')

                ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))

                ax.set_xlim([-2*self.fmax, 2*self.fmax])
                ax.set_ylim([2*self.fmax, -2*self.fmax])

                ax.set_xticks([])
                ax.set_yticks([])

                if jj == (self.nphases - 1):
                    ax.set_xlabel("$f_x$")
                ax.set_ylabel("$f_y$")

                if reconstructed_already:
                    # ####################
                    # separated components
                    # ####################
                    ax = plt.subplot(grid[jj, 1])

                    to_plot = np.abs(bands_unmixed_ft[ii, jj])
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
                    ax = fig.add_subplot(grid[jj, 2])

                    # avoid any zeros for LogNorm()
                    cs_ft_toplot = np.abs(bands_shifted_ft[ii, jj])
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
                    ax = fig.add_subplot(grid[jj, 3])
                    if jj == 0:
                        ax.set_title(r"$w(k)$")

                    im2 = ax.imshow(np.abs(weights[ii, jj]),
                                    norm=PowerNorm(gamma=0.1, vmin=0),
                                    extent=extent_upsampled,
                                    cmap="bone")
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

                    ax = fig.add_subplot(grid[jj, 4])
                    fig.colorbar(im2, cax=ax, format="%0.2g", ticks=[1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])

                    # ####################
                    # normalized weights
                    # ####################
                    ax = fig.add_subplot(grid[jj, 6])
                    if jj == 0:
                        ax.set_title(r"$\frac{w_i(k)}{\sum_j |w_j(k)|^2 + \eta^2}$")

                    im2 = ax.imshow(np.abs(weights[ii, jj] / weights_norm),
                                    norm=PowerNorm(gamma=0.1, vmin=0),
                                    extent=extent_upsampled,
                                    cmap="bone")
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
                    ax = fig.add_subplot(grid[jj, 7])
                    fig.colorbar(im2, cax=ax, format="%0.2g", ticks=[10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])

            if reconstructed_already:
                # real space bands
                band0 = fft.fftshift(fft.ifft2(fft.ifftshift(sim_sr_ft_components[ii, 0]))).real
                band1 = fft.fftshift(fft.ifft2(fft.ifftshift(sim_sr_ft_components[ii, 1] + sim_sr_ft_components[ii, 2]))).real

                color0 = np.array([0, 1, 1]) * 0.75  # cyan
                color1 = np.array([1, 0, 1]) * 0.75  # magenta

                vmax0 = np.percentile(band0, 99.9)
                vmin0 = 0

                vmax1 = np.percentile(band1, 99.9)
                vmin1 = np.percentile(band1, 5)

                img0 = (np.expand_dims(band0, axis=-1) - vmin0) / (vmax0 - vmin0) * np.expand_dims(color0, axis=(0, 1))
                img1 = (np.expand_dims(band1, axis=-1) - vmin1) / (vmax1 - vmin1) * np.expand_dims(color1, axis=(0, 1))

                # ######################################
                # plot real space version of 0th and +/- 1st bands
                # ######################################
                ax = fig.add_subplot(grid[0, 9])
                ax.set_title("combined")

                ax.imshow(img0 + img1, extent=extent_upsampled_real)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_xlabel("x-position ($\mu m$)")
                ax.set_ylabel("y-position ($\mu m$)")

                ax = fig.add_subplot(grid[1, 9])
                ax.set_title("band 0")
                ax.imshow(img0, extent=extent_upsampled_real)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_xlabel("x-position ($\mu m$)")
                ax.set_ylabel("y-position ($\mu m$)")

                ax = fig.add_subplot(grid[2, 9])
                ax.set_title("band 1")
                ax.imshow(img1, extent=extent_upsampled_real)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_xlabel("x-position ($\mu m$)")
                ax.set_ylabel("y-position ($\mu m$)")

            #
            figs.append(fig)
            fig_names.append(f"band_diagnostic_angle={ii:d}")

        return figs, fig_names


    def plot_frequency_fits(self,
                            figsize=(20, 10)):
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
            figh = plot_correlation_fit(self.band0_frq_fit[ii],
                                        self.band1_frq_fit[ii],
                                        self.frqs[ii, :],
                                        self.dx,
                                        self.fmax,
                                        frqs_guess=frqs_guess[ii],
                                        figsize=figsize,
                                        title=f"Correlation fit, angle {ii:d}, unmixing phases = {self.phases_guess[ii]}")
            figs.append(figh)
            fig_names.append(f"frq_fit_angle={ii:d}")

        return figs, fig_names


    def plot_otf(self,
                 figsize=(20, 10)):
        """
        Plot optical transfer function (OTF) versus frequency. Compare with ideal OTF at the same NA, and show
        location of SIM frequencies
        :param figsize:
        :return:
        """

        extent_fxy = get_extent(self.fy, self.fx)

        figh = plt.figure(figsize=figsize)
        tstr = "OTF diagnostic\nvalue at frqs="
        for ii in range(self.nangles):
            tstr += f" {self.otf_at_frqs[ii]:.3f},"
        figh.suptitle(tstr)

        ff = np.sqrt(np.expand_dims(self.fx, axis=0) ** 2 +
                     np.expand_dims(self.fy, axis=1) ** 2)

        otf_ideal = fit_psf.circ_aperture_otf(ff, 0, self.na, self.wavelength)

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
        ax.imshow(np.mean(self.otf, axis=0),
                  extent=extent_fxy,
                  cmap="bone")
        ax.scatter(self.frqs[:, 0], self.frqs[:, 1], color='r', marker='o')
        ax.scatter(-self.frqs[:, 0], -self.frqs[:, 1], color='r', marker='o')
        ax.set_xlabel("$f_x (1/\mu m)$")
        ax.set_ylabel("$f_y (1/\mu m)$")
        ax.set_xlim([-self.fmax, self.fmax])
        ax.set_ylim([-self.fmax, self.fmax])

        return figh


    # saving utility functions
    def save_imgs(self,
                  save_dir: Optional[str] = None,
                  start_time: Optional[str] = None,
                  save_suffix: Optional[str] = None,
                  save_prefix: Optional[str] = None,
                  use_zarr: bool = False,
                  save_patterns: bool = False,
                  save_raw_data: bool = False,
                  nmax_cores: int = -1):

        tstart_save = time.perf_counter()

        if save_dir is None:
            save_dir = self.save_dir
            if self.save_dir is None:
                raise ValueError("If no save_dir argument is provided then self.save_dir must not be None")

        save_dir = Path(save_dir)

        if save_suffix is None:
            save_suffix = self.save_suffix

        if save_prefix is None:
            save_prefix = self.save_prefix


        if start_time is None:
            kwargs = {}
        else:
            kwargs = {"datetime": start_time}

        dxy_wf = self.dx
        if self.upsample_widefield:
            dxy_wf = self.dx / self.upsample_fact

        # metadata want to save
        metadata = {}
        if start_time is not None:
            metadata["data timestemp"] = start_time
        else:
            metadata["data timestemp"] = ""

        metadata["processing timestamp"] = self.tstamp
        metadata["log"] = self.log.getvalue()


        # ###############################n
        # processing metadata
        # ###############################
        metadata["dx"] = self.dx
        metadata["dy"] = self.dy
        metadata["upsample_factor"] = self.upsample_fact
        metadata["na"] = self.na
        metadata["wavelength"] = self.wavelength
        metadata["fmax"] = self.fmax
        metadata["frequency_estimation_mode"] = self.frq_estimation_mode
        metadata["phase_estimation_mode"] = self.phase_estimation_mode
        metadata["combine_bands_mode"] = self.combine_bands_mode
        metadata["wiener_parameter"] = self.wiener_parameter
        metadata["normalize_histograms"] = self.normalize_histograms
        metadata["determine_amplitudes"] = self.determine_amplitudes
        metadata["fmax_exclude_band0"] = self.fmax_exclude_band0
        metadata["band_inds"] = self.band_inds.tolist()
        metadata["nbands"] = self.nbands
        metadata["max_phase_error"] = self.max_phase_error
        metadata["min_p2nr"] = self.min_p2nr
        metadata["otf_mask_threshold"] = self.otf_mask_threshold

        # ###############################n
        # reconstruction parameters
        # ###############################
        metadata["frqs"] = self.frqs.tolist()
        if self.frqs_guess is not None:
            metadata["frqs_guess"] = self.frqs_guess.tolist()
        else:
            metadata["frqs_guess"] = None

        metadata["phases"] = self.phases.tolist()
        metadata["phase_corrections"] = self.phase_corrections.tolist()
        if self.frqs_guess is not None:
            metadata["phases_guess"] = self.phases_guess.tolist()
        else:
            metadata["phases_guess"] = None

        metadata["modulation_depths"] = self.mod_depths.tolist()
        metadata["modulation_depths_guess"] = self.mod_depths_guess.tolist()

        # save results
        if use_zarr:
            fname = save_dir / f"{save_prefix:s}sim_results{save_suffix:s}.zarr"

            img_z = zarr.open(fname, mode="w")

            for k, v in metadata.items():
                img_z.attrs[k] = v

            # ###############################
            # images
            # ###############################

            with ProgressBar():
                future = [self.sim_sr.to_zarr(fname, component="sim_sr", compute=False)]

                attrs = ["widefield", "widefield_deconvolution", "mcnr", "sim_os"]
                if save_raw_data:
                    attrs += ["imgs"]

                for attr in attrs:
                    if hasattr(self, attr):
                        future += [getattr(self, attr).to_zarr(fname, component=attr, compute=False)]

                # dask.compute(future, num_workers=nmax_cores)
                dask.compute(future)

            if save_patterns:
                real_phases = self.phases - np.expand_dims(self.phase_corrections, axis=1)

                # on same grid
                _, _, estimated_patterns, estimated_patterns_2x = \
                    get_simulated_sim_imgs(np.ones([self.upsample_fact * self.ny, self.upsample_fact * self.nx]),
                                           frqs=self.frqs,
                                           phases=real_phases,
                                           mod_depths=self.mod_depths,
                                           gains=1,
                                           offsets=0,
                                           readout_noise_sds=0,
                                           pix_size=self.dx / self.upsample_fact,
                                           nbin=self.upsample_fact
                                           )
                img_z.array("patterns", estimated_patterns[:, :, 0].reshape([self.nangles * self.nphases, self.ny, self.nx])
                            , dtype=float, compressor="none")
                img_z.array("patterns_2x",
                            estimated_patterns_2x[:, :, 0].reshape([self.nangles * self.nphases, self.upsample_fact * self.ny, self.upsample_fact * self.nx]),
                            dtype=float, compressor="none")

        else:
            # todo: want to save without loading all data...
            with ProgressBar():
                results = dask.compute(self.widefield,
                                       self.widefield_deconvolution,
                                       self.mcnr,
                                       self.sim_os,
                                       self.sim_sr)
                widefield, widefield_deconvolution, mcnr, sim_os, sim_sr = results

            # todo: rewrite in this format
            # todo: ultimately want to save e.g. plane by plane instead of all at once
            # delayed = []
            # components = ["widefield", "widefield_deconvolution", "mcnr", "sim_os"]
            # for attr in components:
            #     fname = save_dir / f"{save_prefix:s}{attr:s}{save_suffix:s}.tif"
            #
            #     if getattr(self, attr).shape[0] == self.nx:
            #         factor = 1
            #     else:
            #         factor = self.upsample_fact
            #
            #     dask.delayed(tifffile.imwrite)(fname,
            #                                    getattr(self, attr).astype(np.float32),
            #                                    imagej=True,
            #                                    resolution=(1/self.dy * factor, 1/self.dx * factor),
            #                                    metadata={"Info": attr,
            #                                              "unit": "um"}
            #                                    )


            # save metadata to json file
            fname = save_dir / f"{save_prefix:s}sim_reconstruction{save_suffix:s}.json"
            with open(fname, "w") as f:
                json.dump(metadata, f, indent="\t")

            # save images

            # mcnr
            fname = save_dir / f"{save_prefix:s}mcnr{save_suffix:s}.tif"
            tifffile.imwrite(fname,
                             mcnr.astype(np.float32),
                             imagej=True,
                             resolution=(1/self.dx,)*2,
                             metadata={'Info': 'modulation-contrast to noise ratio',
                                       'unit': 'um'},
                             **kwargs)

            # SIM OS
            fname = save_dir / f"{save_prefix:s}sim_os{save_suffix:s}.tif"

            tifffile.imwrite(fname,
                             sim_os.astype(np.float32),
                             imagej=True,
                             resolution=(1 / dxy_wf,)*2,
                             metadata={'Info': 'SIM optical-sectioning',
                                       'unit': 'um'},
                             **kwargs)

            # widefield
            fname = save_dir / f"{save_prefix:s}widefield{save_suffix:s}.tif"

            tifffile.imwrite(fname,
                             widefield.astype(np.float32),
                             imagej=True,
                             resolution=(1 / dxy_wf,)*2,
                             metadata={'Info': 'widefield',
                                       'unit': 'um'},
                             **kwargs)

            # SIM SR
            fname = save_dir / f"{save_prefix:s}sim_sr{save_suffix:s}.tif"

            tifffile.imwrite(fname,
                             sim_sr.astype(np.float32),
                             imagej=True,
                             resolution=(1/(self.dx / self.upsample_fact),)*2,
                             metadata={'Info': 'SIM super-resolution',
                                       'unit': 'um',
                                       'min': 0,
                                       'max': np.percentile(sim_sr, 99.9)},
                             **kwargs)

            # deconvolution
            fname = save_dir / f"{save_prefix:s}deconvolved{save_suffix:s}.tif"

            tifffile.imwrite(fname,
                             widefield_deconvolution.astype(np.float32),
                             imagej=True,
                             resolution=(1/(self.dx / self.upsample_fact),
                                         1/(self.dx / self.upsample_fact)),
                             metadata={'Info': 'Wiener deconvolved',
                                       'unit': 'um'},
                             **kwargs)

        self.print_log(f"saving SIM images took {time.perf_counter() - tstart_save:.2f}s")


def show_sim_napari(fname_zarr: str,
                    block: bool = True):
    """
    Plot all images obtained from SIM reconstruction with correct scale/offset
    @param fname_zarr:
    @return viewer:
    """

    import napari

    imgz = zarr.open(fname_zarr, "r")
    wf = imgz.widefield

    dxy = imgz.attrs["dx"]
    dxy_sim = dxy / imgz.attrs["upsample_factor"]
    translate_wf = (-(wf.shape[-2] // 2) * dxy, -(wf.shape[-1] // 2) * dxy)
    translate_sim = (-((2 * wf.shape[-2]) // 2) * dxy_sim, -((2 * wf.shape[-1]) // 2) * dxy_sim)

    viewer = napari.Viewer()

    # translate to put FFT zero coordinates at origin
    if hasattr(imgz, "sim_os"):

        viewer.add_image(np.expand_dims(imgz.sim_os, axis=-3),
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         name="SIM-OS")

    if hasattr(imgz, "deconvolved"):
        viewer.add_image(np.expand_dims(imgz.deconvolved, axis=-3),
                         scale=(dxy_sim, dxy_sim),
                         translate=translate_sim,
                         name="wf deconvolved",
                         visible=False)

    if hasattr(imgz, "sim_sr"):
        viewer.add_image(np.expand_dims(imgz.sim_sr, axis=-3),
                         scale=(dxy_sim, dxy_sim),
                         translate=translate_sim,
                         name="SIM-SR",
                         contrast_limits=[0, 5000])

    viewer.add_image(np.expand_dims(wf, axis=-3),
                     scale=(dxy, dxy),
                     translate=translate_wf,
                     name="widefield")

    if hasattr(imgz, "imgs"):
        shape = imgz.imgs.shape[:-4] + (9,) + imgz.imgs.shape[-2:]

        viewer.add_image(np.reshape(imgz.imgs, shape),
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         name="raw images")

    if hasattr(imgz, "patterns"):
        viewer.add_image(imgz.patterns,
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         name="patterns")

    if hasattr(imgz, "patterns_2x"):
        viewer.add_image(imgz.patterns_2x,
                         scale=(dxy_sim, dxy_sim),
                         # translate=translate_sim,
                         translate=[a - 0.25 * dxy for a in translate_wf],
                         name="patterns upsampled")

    viewer.show(block=block)

    return viewer


def reconstruct_mm_sim_dataset(data_dirs: list[str],
                               pixel_size: float,
                               na: float,
                               emission_wavelengths: list[float],
                               excitation_wavelengths: list[float],
                               affine_data_paths: list[str],
                               otf_data_path: list[str],
                               dmd_pattern_data_path,
                               nangles: int = 3,
                               nphases: int = 3,
                               npatterns_ignored: int = 0,
                               crop_rois: list[list[int]] = None,
                               fit_all_sim_params: bool = False,
                               plot_diagnostics: bool = True,
                               channel_inds: list[int] = None,
                               zinds_to_use: list[int] = None,
                               tinds_to_use: list[int] = None,
                               xyinds_to_use: list[int] = None,
                               id_pattern: str = "ic=%d_it=%d_ixy=%d_iz=%d",
                               **kwargs):
    """
    Reconstruct folder of SIM data stored in TIF files. This function assumes the TIF files were generated
    using MicroManager and that the metadata has certain special user keys.

    This is an example of a helper function which loads a certain type of data and uses SimImageSet() to run a
    SIM reconstruction. For other types of data, the preferred approach is to write a function like this one.

    This function uses the MicroManager metadata to load the correct images. It is also responsible for loading
    other relevant data, such as affine transformations, OTF, SIM pattern information from .pkl files. It then
    processes the image and produces SIM superresolution images, deconvolved images, widefield images, and a variety
    of diagnostic plots which it then saves in a convenient directory structure.

    :param data_dirs: list of directories where data is stored
    :param pixel_size: pixel size in ums
    :param na: numerical aperture
    :param emission_wavelengths: list of emission wavelengths in um
    :param excitation_wavelengths: list of excitation wavelengths in um
    :param affine_data_paths: list of paths to files storing data about affine transformations between DMD and camera
    space. [path_color_0, path_color_1, ...]. The affine data files store pickled dictionary objects. The dictionary
    must have an entry 'affine_xform' which contains the affine transformation matrix (in homogeneous coordinates)
    :param otf_data_path: path to file storing optical transfer function data. Data is a pickled dictionary object
    and must have entry 'fit_params'.
    :param dmd_pattern_data_path: list of paths to files storing data about DMD patterns for each color. Data is
    stored in a pickled dictionary object which must contain fields 'frqs', 'phases', 'nx', and 'ny'
    :param nangles: number of angle images
    :param nphases: number of phase images
    :param npatterns_ignored: number of patterns to ignore at the start of each channel.
    :param crop_rois: [[ystart_0, yend_0, xstart_0, xend_0], [ystart_1, ...], ...]
    :param fit_all_sim_params:
    :param plot_diagnostics:
    :param channel_inds: list of channel indices corresponding to each color. If set to None, will use [0, 1, ..., ncolors -1]
    :param zinds_to_use: list of z-position indices to reconstruct
    :param tinds_to_use: list of time indices to reconstruct
    :param xyinds_to_use: list of xy-position indices to reconstruct
    :param id_pattern:
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

    def otf_fn(f, fmax): return 1 / (1 + (f / fmax * otf_p[0]) ** 2) * fit_psf.circ_aperture_otf(f, 0, na, 2 * na / fmax)

    # ############################################
    # SIM images
    # ############################################
    save_dirs = []
    for rpath, roi in zip(data_dirs, crop_rois):
        rpath = Path(rpath)
        folder_path = rpath.parent
        folder = rpath.name
        # folder_path, folder = os.path.split(rpath)
        print("# ################################################################################")
        print(f"analyzing folder: {folder:s}")
        print(f"located in: {folder_path:s}")

        tstamp = datetime.datetime.now().strftime('%Y_%d_%m_%H;%M;%S')
        # now = datetime.datetime.now()
        # tstamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

        # path to store processed results
        # sim_save_dir = os.path.join(rpath, '%s_sim_reconstruction' % tstamp)
        sim_save_dir = rpath / f'{tstamp:s}_sim_reconstruction'
        save_dirs.append(sim_save_dir)
        if not sim_save_dir.exists():
            sim_save_dir.mkdir()
        print(f"save directory: {sim_save_dir:s}")

        individual_image_save_dir = sim_save_dir / "individual_images"
        if not individual_image_save_dir.exists():
            individual_image_save_dir.mkdir()

        # copy useful data files to results dir
        for kk in range(ncolors):
            # copy affine data here
            fname = Path(affine_data_paths[kk]).name
            shutil.copyfile(affine_data_paths[kk], sim_save_dir / fname)

            # copy otf data here
            fname = Path(otf_data_path).name
            shutil.copyfile(otf_data_path, sim_save_dir / fname)

            # copy DMD pattern data here
            fname = Path(dmd_pattern_data_path[kk]).name
            shutil.copyfile(dmd_pattern_data_path[kk], sim_save_dir / fname)

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
        fname = rpath / metadata['FileName'].values[0]
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
                        diagnostics_dir = sim_save_dir / identifier
                        if not diagnostics_dir.exists():
                            diagnostics_dir.mkdir()

                        # find images and load them
                        img_inds = list(range(npatterns_ignored, npatterns_ignored + nangles * nphases))
                        raw_imgs = mm_io.read_mm_dataset(metadata,
                                                         time_indices=ind_t,
                                                         z_indices=iz,
                                                         xy_indices=ixy,
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
                            img_set = SimImageSet({'pixel_size': pixel_size,
                                                   'wavelength': emission_wavelengths[kk],
                                                   'na': na},
                                                  imgs_sim,
                                                  frq_guess=frqs_real,
                                                  otf=otf,
                                                  phases_guess=phases_real,
                                                  mod_depths_guess=mod_depths_real,
                                                  save_dir=diagnostics_dir,
                                                  save_suffix=f"_{file_identifier:s}",
                                                  **kwargs_reduced)
                            img_set.reconstruct()

                        if plot_diagnostics:
                            # plot results
                            img_set.plot_figs()

                        # save reconstruction summary data
                        img_set.save_result(diagnostics_dir / "sim_reconstruction_params.json")
                        img_set.save_imgs(individual_image_save_dir,
                                          start_time, f"_{file_identifier:s}")

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

        stems = ["widefield",
                 "deconvolved",
                 "sim_os",
                 "sim_sr"]
        res_factors = [1, 2, 1, 2]
        for st, rf in zip(stems, res_factors):
            id_str = id_pattern % (0, tinds_to_use_temp[0], xyinds_to_use_temp[0], zinds_to_use_temp[0])
            fname_first = individual_image_save_dir / f"{st:s}_{id_str:s}.tif"
            tif = tifffile.TiffFile(fname_first)
            ny_temp, nx_temp = tif.series[0].shape[-2:]

            imgs = np.zeros((ncolors, nz_used, nt_used, ny_temp, nx_temp))
            for iic, ic in enumerate(range(ncolors)):
                for iixy, ixy in enumerate(xyinds_to_use_temp):
                    for iiz, iz in enumerate(zinds_to_use_temp):
                        for iit, ind_t in enumerate(tinds_to_use_temp):
                            id_str = id_pattern % (ic, ind_t, ixy, iz)
                            imgs[iic, iiz, iit] = tifffile.imread(individual_image_save_dir / f"{st:s}_{id_str:s}.tif")

            im_md = {"Info": "%s image reconstructed from %s" % (st, rpath),
                     "Labels": ch_labels,
                     "spacing": dz,
                     "unit": 'um'}

            if st == "sim_sr":
                # set display values for imagej
                im_md.update({"min": 0, "max": np.percentile(imgs, 99.9)})

            imgs = tifffile.transpose_axes(imgs,
                                           "CZTYX",
                                           asaxes="TZCYXS")
            tifffile.imwrite(sim_save_dir / f'{st:s}.tif',
                             imgs.astype(np.float32),
                             imagej=True,
                             datetime=start_time,
                             resolution=(1 / pixel_size * rf,)*2,
                             metadata=im_md)

        # also for MCNR
        id_str = id_pattern % (0, tinds_to_use_temp[0], xyinds_to_use_temp[0], zinds_to_use_temp[0])
        fname_first = individual_image_save_dir / f"mcnr_{id_str:s}.tif"
        tif = tifffile.TiffFile(fname_first)
        ny_temp, nx_temp = tif.series[0].shape[-2:]

        imgs = np.zeros((ncolors, nz_used, nt_used, nphases, ny_temp, nx_temp))
        for iic, ic in enumerate(range(ncolors)):
            for iixy, ixy in enumerate(xyinds_to_use_temp):
                for iiz, iz in enumerate(zinds_to_use_temp):
                    for iit, ind_t in enumerate(tinds_to_use_temp):
                        id_str = id_pattern % (ic, ind_t, ixy, iz)
                        imgs[iic, iiz, iit] = tifffile.imread(individual_image_save_dir / f"mcnr_{id_str:s}.tif")

        imgs = tifffile.transpose_axes(imgs,
                                       "CZTQYX",
                                       asaxes="TZQCYXS")
        tifffile.imwrite(sim_save_dir / 'mcnr.tif',
                         imgs.astype(np.float32),
                         imagej=True,
                         datetime=start_time,
                         resolution=(1 / pixel_size,)*2,
                         metadata={"Info": f"Modulation-contrast to noise-ratio (MCNR) images for each angle,"
                                           f" reconstructed from {rpath:s}",
                                   "Labels": ch_labels,
                                   "spacing": dz,
                                   "unit": 'um'})
        print("saving tiff stacks took %0.2fs" % (time.perf_counter() - tstart_save))

    return save_dirs


# compute optical sectioned SIM image
def sim_optical_section(imgs: np.ndarray,
                        axis: int = 0,
                        phase_differences: tuple[float] = (0, 2*np.pi/3, 4*np.pi/3)):
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

    # ensure axis positive
    axis = axis % imgs.ndim

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

    # swap the axis we moved back, if needed
    if img_os.ndim > 1 and axis != 0:
        if axis >= 1:
            # positive axis position is one less, since we consumed the 0th axis
            img_os = np.moveaxis(img_os, axis - 1, 0)
        else:
            # negative axis position is unchanged
            img_os = np.moveaxis(img_os, axis, 0)

    return img_os


def correct_modulation_for_bead_size(bead_radii: float,
                                     frqs,
                                     phis=(0, 2 * np.pi / 3, 4 * np.pi / 3)):
    """
    Function for use when calibration SIM modulation depth using fluorescent beads. Assuming the beads are much smaller
    than the lattice spacing, then using the optical sectioning law of cosines type formula on the total fluorescent
    amplitude from a single isolated beads provides an estimate of the modulation depth.

    When the bead is a significant fraction of the lattice period, this modifies the modulation period. This function
    computes the correction for finite bead size.

    :param bead_radii: radius of the bead
    :param frqs: frequency of the lattice
    :param phis: phase steps of pattern
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

    # phis = np.array([])
    vals = np.zeros(3)
    for ii in range(3):
        vals[ii] = full_int(bead_radii, frqs, phis[ii])

    mods = sim_optical_section(vals, axis=0)
    return mods


# estimate frequency of modulation patterns
def fit_modulation_frq(ft1: np.ndarray,
                       ft2: np.ndarray,
                       dxy: float,
                       mask: Optional[np.ndarray] = None,
                       frq_guess: Optional[tuple[float]] = None,
                       max_frq_shift: Optional[float] = None,
                       keep_guess_if_better: bool = True) -> (np.ndarray, np.ndarray, dict):
    """
    Find SIM frequency from image by maximizing the cross correlation between ft1 and ft2
    C(df) =  \sum_f ft1(f) x ft2^*(f + df)
    modulation freqency = argmax_{df} |C(df)|

    Note that there is ambiguity in the definition of this frequency, as -f will also be a peak. If frq_guess is
    provided, the peak closest to the guess will be returned.

    :param ft1: 2D Fourier space image
    :param ft2: 2D Fourier space image to be cross correlated with ft1
    :param dxy: pixel size. Units of dxy and max_frq_shift must be consistent
    :param mask: boolean array same size as ft1 and ft2. Only consider frequency points where mask is True
    :param frq_guess: frequency guess [fx, fy]. If frequency guess is None, an initial guess will be chosen by
    finding the maximum f_guess = argmax_f CC[ft1, ft2](f), where CC[ft1, ft2] is the discrete cross-correlation
     Currently roi_pix_size is only used internally to set max_frq_shift
    :param max_frq_shift: maximum frequency shift to consider vis-a-vis the guess frequency
    :param keep_guess_if_better: keep the initial frequency guess if the cost function is more optimal
     at this point than after fitting

    :return fit_frqs, mask, fit_result:
    """

    if ft1.shape != ft2.shape:
        raise ValueError("must have ft1.shape = ft2.shape")

    if max_frq_shift is None:
        max_frq_shift = np.inf

    # mask
    if mask is None:
        mask = np.ones(ft1.shape, dtype=bool)
    else:
        mask = np.array(mask, copy=True)

    if mask.shape != ft1.shape:
        raise ValueError("mask must have same shape as ft1")

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(ft1.shape[1], dxy))
    fys = fft.fftshift(fft.fftfreq(ft1.shape[0], dxy))
    fxfx, fyfy = np.meshgrid(fxs, fys)

    # update mask to only consider region near frequency guess
    if frq_guess is not None:
        mask[np.sqrt((fxfx - frq_guess[0])**2 + (fyfy - frq_guess[1])**2) > max_frq_shift] = 0

    # ############################
    # set initial guess
    # ############################
    if frq_guess is None:
        # cross correlation of Fourier transforms
        # WARNING: correlate2d uses a different convention for the frequencies of the output, which will not agree with the fft convention
        # take conjugates so this will give \sum ft1 * ft2.conj()
        # scipy.signal.correlate(g1, g2)(fo) seems to compute \sum_f g1^*(f) x g2(f - fo), but I want g1^*(f) x g2(f+fo) # todo: is this right?
        cc = np.abs(correlate(ft2, ft1, mode='same'))

        # get initial frq_guess by looking at cc at discrete frequency set and finding max
        subscript = np.unravel_index(np.argmax(cc * mask), cc.shape)

        init_params = np.array([fxfx[subscript], fyfy[subscript]])
    else:
        init_params = frq_guess

    # ############################
    # define cross-correlation and minimization objective function
    # ############################
    # real-space coordinates
    ny, nx = ft1.shape
    x = fft.ifftshift(dxy * (np.arange(nx) - (nx // 2)))
    y = fft.ifftshift(dxy * (np.arange(ny) - (ny // 2)))
    xx, yy = np.meshgrid(x, y)

    img2 = fft.fftshift(fft.ifft2(fft.ifftshift(ft2)))

    # compute ft2(f + fo)
    def fft_shifted(f): return fft.fftshift(fft.fft2(np.exp(-1j*2*np.pi * (f[0] * xx + f[1] * yy)) * fft.ifftshift(img2)))

    # cross correlation
    # todo: conjugating ft2 instead of ft1, as in typical definition of cross correlation. Doesn't matter bc taking norm
    def cc_fn(f): return np.sum(ft1 * fft_shifted(f).conj())
    fft_norm = np.sum(np.abs(ft1) * np.abs(ft2))**2
    def min_fn(f): return -np.abs(cc_fn(f))**2 / fft_norm

    # ############################
    # do fitting
    # ############################
    # enforce frequency fit in same box as guess
    lbs = (init_params[0] - max_frq_shift,
           init_params[1] - max_frq_shift)
    ubs = (init_params[0] + max_frq_shift,
           init_params[1] + max_frq_shift)
    bounds = ((lbs[0], ubs[0]), (lbs[1], ubs[1]))

    fit_result = minimize(min_fn, init_params, bounds=bounds)

    fit_frqs = fit_result.x

    # convert to dictionary and add anythin we want to it
    fit_result = dict(fit_result)
    fit_result["init_params"] = init_params

    # ensure we never get a worse point than our initial guess
    if keep_guess_if_better and min_fn(init_params) < min_fn(fit_frqs):
        fit_frqs = init_params

    return fit_frqs, mask, fit_result


def plot_correlation_fit(img1_ft: np.ndarray,
                         img2_ft: np.ndarray,
                         frqs,
                         dxy: float,
                         fmax: Optional[float] = None,
                         frqs_guess: Optional[tuple[float]] = None,
                         roi_size: tuple[int] = (31, 31),
                         peak_pixels: int = 2,
                         figsize=(20, 10),
                         title: str = "",
                         gamma: float = 0.1,
                         cmap="bone"):
    """
    Display SIM parameter fitting results visually, in a way that is easy to inspect.

    Use this to plot the results of SIM frequency determination after running get_sim_frq()

    :param img1_ft:
    :param img2_ft
    :param frqs: fit value of frequency [fx, fy]
    :param dxy: pixel size in um
    :param fmax: maximum frequency. Will display this using a circle
    :param frqs_guess: guess frequencies [fx, fy]    
    :param roi_size:
    :param peak_pixels:    
    :param figsize:
    :param title:
    :param gamma:
    :param cmap: matplotlib colormap to use
    :return figh: handle to figure produced
    """
    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(img1_ft.shape[1], dxy))
    dfx = fxs[1] - fxs[0]
    fys = fft.fftshift(fft.fftfreq(img1_ft.shape[0], dxy))
    dfy = fys[1] - fys[0]

    extent = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx,
              fys[-1] + 0.5 * dfy, fys[0] - 0.5 * dfy]

    # power spectrum / cross correlation
    # cc = np.abs(scipy.signal.fftconvolve(img1_ft, img2_ft.conj(), mode='same'))
    cc = np.abs(correlate(img2_ft, img1_ft, mode='same'))

    # compute peak values
    fx_sim, fy_sim = frqs
    try:
        peak_cc = tools.get_peak_value(cc, fxs, fys, [fx_sim, fy_sim], peak_pixels)
        peak1_dc = tools.get_peak_value(img1_ft, fxs, fys, [0, 0], peak_pixels)
        peak2 = tools.get_peak_value(img2_ft, fxs, fys, [fx_sim, fy_sim], peak_pixels)
    except ZeroDivisionError:
        peak_cc = np.nan
        peak1_dc = np.nan
        peak2 = np.nan

    # create figure
    figh = plt.figure(figsize=figsize)
    gspec = figh.add_gridspec(ncols=4, width_ratios=[8, 1] * 2, wspace=0.5,
                              nrows=2, hspace=0.3)

    # #######################################
    # build title
    # #######################################
    str = ""
    if title != "":
        str += f"{title:s}\n"

    # print info about fit frequency
    period = 1 / np.sqrt(fx_sim ** 2 + fy_sim ** 2)
    angle = np.angle(fx_sim + 1j * fy_sim)

    str += f"      fit: period {period * 1e3:.1f}nm = 1/{1/period:.3f}um at" \
           f" {angle * 180/np.pi:.2f}deg={angle:.3f}rad;" \
           f" f=({fx_sim:.3f},{fy_sim:.3f}) 1/um," \
           f" peak cc={np.abs(peak_cc):.3g} and {np.angle(peak_cc) * 180/np.pi:.2f}deg"

    # print info about guess frequency
    if frqs_guess is not None:
        fx_g, fy_g = frqs_guess
        period_g = 1 / np.sqrt(fx_g ** 2 + fy_g ** 2)
        angle_g = np.angle(fx_g + 1j * fy_g)
        peak_cc_g = tools.get_peak_value(cc, fxs, fys, frqs_guess, peak_pixels)

        str += f"\nguess: period {period_g * 1e3:.1f}nm = 1/{1/period_g:.3f}um" \
               f" at {angle_g * 180/np.pi:.2f}deg={angle_g:.3f}rad;" \
               f" f=({fx_g:.3f},{fy_g:.3f}) 1/um," \
               f" peak cc={np.abs(peak_cc_g):.3g} and {np.angle(peak_cc_g) * 180/np.pi:.2f}deg"

    figh.suptitle(str)

    # #######################################
    # plot cross-correlation region of interest
    # #######################################
    roi_cx = np.argmin(np.abs(fx_sim - fxs))
    roi_cy = np.argmin(np.abs(fy_sim - fys))
    roi = rois.get_centered_roi([roi_cy, roi_cx],
                                roi_size,
                                min_vals=[0, 0],
                                max_vals=cc.shape)

    extent_roi = get_extent(fys[roi[0]:roi[1]], fxs[roi[2]:roi[3]])

    ax = figh.add_subplot(gspec[0, 0])
    ax.set_title("cross correlation, ROI")
    im1 = ax.imshow(rois.cut_roi(roi, cc),
                    interpolation=None,
                    norm=PowerNorm(gamma=gamma),
                    extent=extent_roi,
                    cmap=cmap)
    ax.scatter(frqs[0], frqs[1], color='r', marker='x', label="frq fit")
    if frqs_guess is not None:
        if np.linalg.norm(frqs - frqs_guess) < np.linalg.norm(frqs + frqs_guess):
            ax.scatter(frqs_guess[0], frqs_guess[1], color='g', marker='x', label="frq guess")
        else:
            ax.scatter(-frqs_guess[0], -frqs_guess[1], color='g', marker='x', label="frq guess")

    ax.legend(loc="upper right")

    if fmax is not None:
        ax.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0, ls='--'))

    ax.set_xlabel('$f_x (1/\mu m)$')
    ax.set_ylabel('$f_y (1/\mu m)$')

    # colorbar
    cbar_ax = figh.add_subplot(gspec[0, 1])
    figh.colorbar(im1, cax=cbar_ax)

    # #######################################
    # full cross-correlation
    # #######################################
    ax2 = figh.add_subplot(gspec[0, 2])
    im2 = ax2.imshow(cc, interpolation=None, norm=PowerNorm(gamma=gamma), extent=extent, cmap=cmap)

    if fmax is not None:
        ax2.set_xlim([-fmax, fmax])
        ax2.set_ylim([fmax, -fmax])

        # plot maximum frequency
        ax2.add_artist(Circle((0, 0), radius=fmax, color='k', fill=0))

    ax2.add_artist(Rectangle((fxs[roi[2]], fys[roi[0]]),
                             fxs[roi[3] - 1] - fxs[roi[2]],
                             fys[roi[1] - 1] - fys[roi[0]],
                             edgecolor='k',
                             fill=0))

    ax2.set_title(r"$C(f_o) = \sum_f g_1(f) \times g^*_2(f+f_o)$")
    ax2.set_xlabel('$f_x (1/\mu m)$')
    ax2.set_ylabel('$f_y (1/\mu m)$')

    # colorbar
    cbar_ax = figh.add_subplot(gspec[0, 3])
    figh.colorbar(im2, cax=cbar_ax)

    # #######################################
    # ft 1
    # #######################################
    ax3 = figh.add_subplot(gspec[1, 0])
    ax3.set_title(r"$|g_1(f)|^2$" + r" near DC, $g_1(0) = $"  " %0.3g and %0.2fdeg" %
                  (np.abs(peak1_dc), np.angle(peak1_dc) * 180/np.pi))
    ax3.set_xlabel('$f_x (1/\mu m)$')
    ax3.set_ylabel('$f_y (1/\mu m)$')

    cx_c = np.argmin(np.abs(fxs))
    cy_c = np.argmin(np.abs(fys))
    roi_center = rois.get_centered_roi([cy_c, cx_c], [roi[1] - roi[0], roi[3] - roi[2]], [0, 0], img1_ft.shape)
    extent_roic = get_extent(fys[roi_center[0]:roi_center[1]],
                             fxs[roi_center[2]:roi_center[3]])

    im3 = ax3.imshow(rois.cut_roi(roi_center, np.abs(img1_ft)**2),
                     interpolation=None, norm=PowerNorm(gamma=gamma), extent=extent_roic, cmap=cmap)
    ax3.scatter(0, 0, color='r', marker='x')

    # colorbar
    cbar_ax = figh.add_subplot(gspec[1, 1])
    figh.colorbar(im3, cax=cbar_ax)

    # #######################################
    # ft 2
    # #######################################
    ax4 = figh.add_subplot(gspec[1, 2])
    title = r"$|g_2(f)|^2$" + r"near $f_o$, $g_2(f_p) =$" + " %0.3g and %0.2fdeg" % \
            (np.abs(peak2), np.angle(peak2) * 180 / np.pi)
    if frqs_guess is not None:
        peak2_g = tools.get_peak_value(img2_ft, fxs, fys, frqs_guess, peak_pixels)
        title += "\nguess peak = %0.3g and %0.2fdeg" % (np.abs(peak2_g), np.angle(peak2_g) * 180 / np.pi)
    ax4.set_title(title)
    ax4.set_xlabel('$f_x (1/\mu m)$')

    im4 = ax4.imshow(rois.cut_roi(roi, np.abs(img2_ft)**2), interpolation=None, norm=PowerNorm(gamma=gamma),
                     extent=extent_roi, cmap=cmap)
    ax4.scatter(frqs[0], frqs[1], color='r', marker='x')
    if frqs_guess is not None:
        if np.linalg.norm(frqs - frqs_guess) < np.linalg.norm(frqs + frqs_guess):
            ax4.scatter(frqs_guess[0], frqs_guess[1], color='g', marker='x')
        else:
            ax4.scatter(-frqs_guess[0], -frqs_guess[1], color='g', marker='x')

    # colorbar
    cbar_ax = figh.add_subplot(gspec[1, 3])
    figh.colorbar(im4, cax=cbar_ax)

    return figh


# estimate phase of modulation patterns
def get_phase_ft(img_ft: array,
                 sim_frq,
                 dxy: float,
                 peak_pixel_size: int = 2) -> float:
    """
    Estimate pattern phase directly from phase in Fourier transform

    :param img_ft:
    :param sim_frq:
    :param dxy:
    :param peak_pixel_size:
    :return phase:
    """
    ny, nx = img_ft.shape
    fx = fft.fftshift(fft.fftfreq(nx, dxy))
    fy = fft.fftshift(fft.fftfreq(ny, dxy))

    phase = np.mod(np.angle(tools.get_peak_value(img_ft, fx, fy, sim_frq, peak_pixel_size=peak_pixel_size)), 2*np.pi)

    return phase


def get_phase_realspace(img: np.ndarray,
                        sim_frq,
                        dxy: float,
                        phase_guess: float = 0,
                        origin: str = "center"):
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

    ny, nx = img.shape

    if origin == "center":
        x = (np.arange(nx) - (nx // 2)) * dxy
        y = (np.arange(ny) - (ny // 2)) * dxy
    elif origin == "edge":
        x = np.arange(nx) * dxy
        y = np.arange(ny) * dxy
        # x = tools.get_fft_pos(nx, dxy, centered=False, mode="positive")
        # y = tools.get_fft_pos(ny, dxy, centered=False, mode="positive")
    else:
        raise ValueError(f"'origin' must be 'center' or 'edge' but was '{origin:s}'")

    xx, yy = np.meshgrid(x, y)

    def fn(phi): return -np.cos(2*np.pi * (sim_frq[0] * xx + sim_frq[1] * yy) + phi)
    def fn_deriv(phi): return np.sin(2*np.pi * (sim_frq[0] * xx + sim_frq[1] * yy) + phi)
    def min_fn(phi): return np.sum(fn(phi) * img) / img.size
    def jac_fn(phi): return np.asarray([np.sum(fn_deriv(phi) * img) / img.size, ])

    # using jacobian makes faster and more robust
    result = minimize(min_fn, phase_guess, jac=jac_fn)
    # also using routine optimized for scalar univariate functions works
    # result = scipy.optimize.minimize_scalar(min_fn)
    phi_fit = np.mod(result.x, 2 * np.pi)

    return phi_fit


def get_phase_wicker_iterative(imgs_ft: np.ndarray,
                               otf: np.ndarray,
                               sim_frq: np.ndarray,
                               dxy: float,
                               fmax: float,
                               phases_guess=None,
                               fit_amps: bool = True,
                               debug: bool = False):
    """
    Estimate relative phases between components using the iterative cross-correlation minimization method of Wicker,
    described in detail here https://doi.org/10.1364/OE.21.002032. This function is hard coded for 3 bands.

    NOTE: this method is not sensitive to the absolute phase, only the relative phases...

    Suppose that S(r) is the sample, h(r) is the PSF, and D_n(r) is the data. Then the separated (but unshifted) bands
    are C_m(k) = S(k - m*ko) * h_m(k), and the data vector is related to the band vector by
    D(k) = M*C(k), where M is an nphases x nbands matrix.

    This function minimizes the cross correlation between shfited bands which should not contain common information.
    Let cc^l_ij = C_i(k) \otimes C_j(k-lp). These should have low overlap for i =/= j + l.
    So the minimization function is
    g(M) = \sum_{i \neq l+j} |cc^l_ij|^0.5

    This can be written in terms of the correlations of data matrix, dc^l_ij in a way that minimizes numerical effort
    to compute g for different mixing matrices M.

    :param imgs_ft: array of size nphases x ny x nx, where the components are o(f), o(f-fo), o(f+fo)
    :param otf: size ny x nx
    :param sim_frq: np.array([fx, fy])
    :param float dxy: pixel size in um
    :param float fmax: maximum spatial frequency where otf has support
    :param phases_guess: [phi1, phi2, phi3] in radians.if None will use [0, 2*pi/3, 4*pi/3]
    :param fit_amps: if True will also fit amplitude differences between components
    :param debug:
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


    # compute cross correlations of data
    band_inds = [0, 1, -1]
    nbands = len(band_inds)
    d_cc = np.zeros((nphases, nphases, nbands), dtype=complex)
    # Band_i(k) = Obj(k - i*p) * h(k)
    # this is the order set by matrix M, i.e.
    # [D1(k), ...] = M * [Obj(k) * h(k), Obj(k - i*p) * h(k), Obj(k + i*p) * h(k)]
    for ll, ml in enumerate(band_inds):
        # get shifted otf -> otf(f - l * fo)
        otf_shift, _ = tools.translate_pix(otf,
                                           -ml * sim_frq,
                                           dr=(dfx, dfy),
                                           axes=(1, 0),
                                           wrap=False)

        with np.errstate(invalid="ignore", divide="ignore"):
            weight = otf * otf_shift.conj() / (np.abs(otf_shift) ** 2 + np.abs(otf) ** 2)
            weight[np.isnan(weight)] = 0

        for ii in range(nphases):  # [0, 1, 2] -> [0, 1, -1]
            for jj in range(nphases):
                # shifted component C_j(f - l*fo)
                band_shifted = tools.translate_ft(imgs_ft[jj], -ml * sim_frq[0], -ml * sim_frq[1], drs=(dxy, dxy))

                # compute weighted cross correlation
                d_cc[ii, jj, ll] = np.sum(imgs_ft[ii] * band_shifted.conj() * weight) / np.sum(weight)

                # remove extra noise correlation expected from same images
                if ml == 0 and ii == jj:
                    noise_power = get_noise_power(imgs_ft[ii], fx, fy, fmax)
                    d_cc[ii, jj, ll] = d_cc[ii, jj, ll] - noise_power

                if debug:
                    extentf = get_extent(fy, fx)
                    gamma = 0.1

                    figh = plt.figure(figsize=(16, 8))
                    grid = figh.add_gridspec(2, 3)
                    figh.suptitle(f"(i, j, band) = ({ii:d}, {jj:d}, {ml:d})")

                    ax = figh.add_subplot(grid[0, 0])
                    ax.imshow(np.abs(imgs_ft[ii]), norm=PowerNorm(gamma=gamma), extent=extentf)
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("$D_i(k)$")

                    ax = figh.add_subplot(grid[0, 1])
                    ax.imshow(np.abs(band_shifted), norm=PowerNorm(gamma=gamma), extent=extentf)
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("$D_j(k-lp)$")

                    ax = figh.add_subplot(grid[0, 2])
                    ax.imshow(np.abs(imgs_ft[ii] * band_shifted.conj()), norm=PowerNorm(gamma=gamma),
                              extent=extentf)
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("$D^l_{ij} = D_i(k) x D_j^*(k-lp)$")

                    ax = figh.add_subplot(grid[1, 0])
                    ax.imshow(otf, extent=extentf)
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title('$otf_i(k)$')

                    ax = figh.add_subplot(grid[1, 1])
                    ax.imshow(otf_shift, extent=extentf)
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title('$otf_j(k-lp)$')

                    ax = figh.add_subplot(grid[1, 2])
                    ax.imshow(weight, extent=extentf)
                    ax.plot(sim_frq[0], sim_frq[1], 'r+')
                    ax.plot(-sim_frq[0], -sim_frq[1], 'r.')
                    ax.set_title("weight")
    # correct normalization of d_cc (inherited from FFT) so should be same for different image sizes
    d_cc = d_cc / (nx * ny)**2

    # optimize
    if fit_amps:
        def minv(p): return get_band_mixing_inv([0, p[0], p[1]], mod_depth=1, amps=[1, p[2], p[3]])
    else:
        def minv(p): return get_band_mixing_inv([0, p[0], p[1]], mod_depth=1, amps=[1, 1, 1])

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
            init_params = np.array([phases_guess[1] - phases_guess[0], phases_guess[2] - phases_guess[0], 1, 1])

        result = minimize(min_fn, init_params)
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

        result = minimize(min_fn, init_params)
        phases = np.array([0, result.x[0], result.x[1]])
        amps = np.array([1, 1, 1])

    return phases, amps, result


# power spectrum and modulation depths
def get_noise_power(img_ft: np.ndarray,
                    fxs: np.ndarray,
                    fys: np.ndarray,
                    fmax: float):
    """
    Estimate average noise power of an image by looking at frequencies beyond the maximum frequency
    where the OTF has support.

    :param img_ft: Size n0 x n1 x ... x ny x nx. Fourier transform of image. Computed the noise power over the
    last two dimensions
    :param fxs: 1D array, x-frequencies
    :param fys: 1D array, y-frequencies
    :param fmax: maximum frequency where signal may be present
    :return noise_power:
    """

    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff = np.sqrt(fxfx ** 2 + fyfy ** 2)
    noise_power = np.mean(np.abs(img_ft[..., ff > fmax])**2, axis=-1)

    return noise_power


# inversion functions
def get_band_mixing_matrix(phases: list,
                           mod_depth: float = 1.,
                           amps: Optional[np.ndarray] = None) -> np.ndarray:
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


def get_band_mixing_matrix_jac(phases: list[float],
                               mod_depth: float,
                               amps: list[float]) -> list[np.ndarray]:
    """
    Get jacobian of band mixing matrix in parameters [p1, p2, p3, a1, a2, a3, m]
    @param phases:
    @param mod_depth:
    @param amps:
    @return jac:
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


def get_band_mixing_inv(phases: np.ndarray,
                        mod_depth: float = 1.,
                        amps: Optional[np.ndarray] = None) -> np.ndarray:
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


def get_band_mixing_inv2(phases: np.ndarray,
                         m: float,
                         amps: Optional[np.ndarray] = None):
    """
    Get inverse of the band mixing matrix, which maps measured data to separated (but unshifted) bands
    Do this analytically...

    @param phases:
    @param m:
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


def get_band_mixing_inv_jac(phases,
                            mod_depth: float = 1,
                            amps: Optional[np.ndarray] = None):
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


def unmix_bands(imgs_ft: array,
                phases,
                mod_depths: Optional[np.ndarray] = None,
                amps: Optional[np.ndarray] = None) -> array:
    """
    Do noisy inversion of SIM data, i.e. determine
    [[S(f)H(k)], [S(f-p)H(f)], [S(f+p)H(f)]] = M^{-1} * [[D_1(f)], [D_2(f)], [D_3(f)]]

    # todo: generalize for case with more than 3 phases or angles

    :param imgs_ft: n0 x ... x nm x nangles x nphases x ny x nx.
     Fourier transform of SIM image data with DC frequency information in middle. i.e. as obtained from fftshift
    :param phases: array nangles x nphases listing phases
    :param mod_depths: list of length nangles. Optional. If not provided, all are set to 1.
    :param amps: list of length nangles x nphases. If not provided, all are set to 1.
    :return components_ft: nangles x nphases x ny x nx array, where the first index corresponds to the bands
    S(f)H(f), S(f-p)H(f), or S(f+p)H(f)
    """
    if isinstance(imgs_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    # ensure images cupy array if doing on GPU
    imgs_ft = xp.array(imgs_ft)

    nangles, nphases, ny, nx = imgs_ft.shape[-4:]

    # keep all parameters as numpy arrays
    # default parameters
    if mod_depths is None:
        mod_depths = np.ones(nangles)
    else:
        pass

    if amps is None:
        amps = np.ones((nangles, nphases))
    else:
        pass

    # check parameters
    if nphases != 3:
        raise NotImplementedError(f"only implemented for nphases=3, but nphases={nphases:d}")

    # try to do inversion
    bands_ft = xp.zeros(imgs_ft.shape, dtype=complex) * np.nan
    for ii in range(nangles):
        mixing_mat_inv = xp.array(get_band_mixing_inv(phases[ii], mod_depths[ii], amps[ii]))
        # bands_ft[ii] = image_times_matrix(imgs_ft[ii], mixing_mat_inv)

        # todo: plenty of ways to write this more generally ... but for now this is fine
        for jj in range(nphases):
            bands_ft[..., ii, jj, :, :] = mixing_mat_inv[jj, 0] * imgs_ft[..., ii, 0, :, :] + \
                                          mixing_mat_inv[jj, 1] * imgs_ft[..., ii, 1, :, :] + \
                                          mixing_mat_inv[jj, 2] * imgs_ft[..., ii, 2, :, :]

    return bands_ft


def shift_bands(bands_unmixed_ft: array,
                frqs: array,
                drs: tuple[float],
                upsample_factor: int) -> array:
    """

    @param bands_unmixed_ft: n0 x ... x nm x 3 x ny x nx
    @param frqs: n0 x ... x nm x 2
    @param drs: (dy, dx)
    @param upsample_factor:
    @return:
    """

    if isinstance(bands_unmixed_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    dy, dx = drs

    # zero-pad bands (interpolate in realspace)
    # Only do this to one of the shifted bands. don't need to loop over m*O(f + f_o)H(f), since it is conjugate of m*O(f - f_o)H(f)
    expanded = resample_bandlimited_ft(bands_unmixed_ft[..., :2, :, :],
                                       (upsample_factor, upsample_factor),
                                       axes=(-1, -2))

    # get O(f)H(f) directly from expansion
    b0 = expanded[..., 0, :, :]
    # FFT shift to get m*O(f - f_o)H(f)
    b1 = tools.translate_ft(expanded[..., 1, :, :],
                            np.expand_dims(frqs[:, 0], axis=(-1, -2)),
                            np.expand_dims(frqs[:, 1], axis=(-1, -2)),
                            drs=(dy / upsample_factor, dx / upsample_factor))

    # reflect m*O(f - f_o)H(f) to get m*O(f + f_o)H(f)
    b2 = conj_transpose_fft(b1)

    shifted_bands_ft = xp.stack((b0, b1, b2), axis=-3)

    return shifted_bands_ft


def get_band_overlap(band0: array,
                     band1: array,
                     otf0: array,
                     otf1: array,
                     mask: array) -> (array, array):
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

    :param band0: n0 x ... x nm x nangles x ny x nx. Typically band0(f) = S(f) * otf(f) * wiener(f) ~ S(f)
    :param band1: same shape as band0. Typically band1(f) = S((f-fo) + fo) * otf(f + fo) * wiener(f + fo),
    i.e. the separated band after shifting to correct position
    :param otf0: Same shape as band0
    :param otf1:
    :param mask: same shape as band0. Where mask is True, use these points to evaluate the band correlation.
     Typically construct by picking some value where otf(f) and otf(f + fo) are both > w, where w is some cutoff value.

    :return phases, mags:
    """

    if isinstance(band0, cp.ndarray):
        xp = cp
    else:
        xp = np

    nangles, ny, nx = band0.shape[-3:]
    phases = xp.zeros(band0.shape[:-2])
    mags = xp.zeros(band0.shape[:-2])

    # divide by OTF, but don't worry about Wiener filtering. avoid problems by keeping otf_threshold large enough
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = band0 / otf0 * band1.conj() / otf1.conj()
        denominator = xp.abs(band0 / otf0) ** 2

    for ii in range(nangles):
        corr = xp.sum(numerator[..., ii, :, :][mask[..., ii, :, :]], axis=-1) / xp.sum(denominator[..., ii, :, :][mask[..., ii, :, :]], axis=-1)
        mags[..., ii] = xp.abs(corr)
        phases[..., ii] = xp.angle(corr)

    return phases, mags


def get_extent(y: np.ndarray,
               x: np.ndarray,
               origin: str = "lower") -> list[float]:
    """
    Get extent required for plotting arrays using imshow in real coordinates. The resulting list can be
    passed directly to imshow using the extent keyword.

    Here we assume the values y and x are equally spaced and describe the center coordinates of each pixel

    :param y: equally spaced y-coordinates
    :param x: equally spaced x-coordinates
    :param origin: "lower" or "upper" depending on if the y-origin is at the lower or upper edge of the image
    :return extent: [xstart, xend, ystart, yend]
    """

    dy = y[1] - y[0]
    dx = x[1] - x[0]
    if origin == "lower":
        extent = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx,
                  y[-1] + 0.5 * dy, y[0] - 0.5 * dy]
    elif origin == "upper":
        extent = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx,
                  y[0] - 0.5 * dy, y[-1] + 0.5 * dy]
    else:
        raise ValueError("origin must be 'lower' or 'upper' but was '%s'" % origin)

    return extent


# Fourier transform tools
def resample_bandlimited_ft(img_ft: array,
                            mag: tuple[int],
                            axes: tuple[int]) -> array:
    """
    Zero pad Fourier space image by adding high-frequency content. This corresponds to interpolating the real-space
    image

    Note that this is not the same as the zero padding by using ifftn s parameter, since that will pad after the least
    magnitude negative frequency, while this pads near the highest-magnitude frequencies. For more discussion of
    this point see e.g. https://github.com/numpy/numpy/issues/1346

    The expanded array is normalized so that the realspace values will match after an inverse FFT,
    thus the corresponding Fourier space components will have the relationship b_k = a_k * b.size / a.size

    @param img_ft: frequency space representation of image, arranged so that zero frequency is near the center of
    the array. The frequencies can be obtained with fftshift(fftfreq(n, dxy))
    @param mag: factor by which to oversample array. This must be an integer
    @param axes: zero-pad along these axes only
    @return img_ft_pad: expanded array
    """

    if isinstance(img_ft, cp.ndarray):
        xp = cp
    else:
        xp = np


    # make axes to operate on positive
    axes = tuple([a if a >= 0 else img_ft.ndim + a for a in axes])

    # expansion factors
    facts = np.ones(img_ft.ndim, dtype=int)
    for ii, a in enumerate(axes):
        facts[a] = mag[ii]

    # if extra padding not even (i.e. if initial array was odd) then put one more on the left
    pad_width = [(int(np.ceil((f - 1) * img_ft.shape[ii] / 2)),
                  (f - 1) * img_ft.shape[ii] // 2) for ii, f in enumerate(facts)]

    # zero pad and correct normalization
    img_ft_pad = xp.pad(img_ft,
                        pad_width=pad_width,
                        mode="constant",
                        constant_values=0) * np.prod(mag)

    # if initial array was even it had an unpaired negative frequency, but its pair is present in the larger array
    # this negative frequency was at -N/2, so this enters the IFT for a_n as a_(k=-N/2) * exp(2*np.pi*i * -n/2)
    # not that exp(2*np.pi*i * -k/2) = exp(2*np.pi*i * k/2), so this missing frequency doesn't matter for a
    # however, when we construct b, in the IFT for b_n we now have b_(k=-N/2) * exp(2*np.pi*i * -n/4)
    # Since we are supposing N is even, we must satisfy
    # b_(2n) = a_n -> b_(k=-L/2) + b_(k=L/2) = a_(k=-L/2)
    # Further, we want to ensure that b is real if a is real, which implies
    # b_(k=-N/2) = 0.5 * a(k=-N/2)
    # b_(k= N/2) = 0.5 * a(k=-N/2)
    # no complex conjugate is required for b_(k=N/2). If a_n is real, then a(k=-N/2) must also be real.
    #
    # consider the 2D case. We have an unfamiliar condition required to make a real
    # a_(ky=-N/2, kx) = conj(a_(ky=-N/2, -kx))
    # recall -N/2 <-> N/2 to make this more familiar
    # for b_(n, m) we have b_(ky=-N/2, kx) * exp(2*np.pi*i * -n/4) * exp(2*np.pi*i * kx*m/(fx*N))
    # to ensure all b_(n, m) are real we must enforce
    # b_(ky=N/2, kx) = conj(b(ky=-N/2, -kx))
    # b_(ky, kx=N/2) = conj(b(-ky, kx=-N/2))
    # on the other hand, to enforce b_(2n, 2m) = a_(n, m)
    # a(ky=-N/2,  kx) = b(ky=-N/2,  kx) + b(ky=N/2,  kx)
    # a(ky=-N/2, -kx) = b(ky=-N/2, -kx) + b(ky=N/2, -kx) = b^*(ky=-N/2, kx) + b^*(ky=N/2, kx)
    # but this second equation doesn't give us any more information than the real condition above
    # the easiest way to do this is...
    # b(ky=+/- N/2, kx) = 0.5 * a(ky=-N/2, kx)
    # for the edges, the conditions are
    # b(ky=+/- N/2, kx=+/- N/2) = 0.25 * a(ky=kx=-N/2)
    # b(ky=+/- N/2, kx=-/+ N/2) = 0.25 * a(ky=kx=-N/2)
    # loop over axes to correct nyquist frequencies
    for ii in range(len(mag)):
        m = mag[ii]
        a = axes[ii]

        if img_ft.shape[a] % 2 == 1:
            continue

        # correct nyquist frequency
        old_nyquist_ind = m * img_ft.shape[a] // 2 - img_ft.shape[a] // 2
        nyquist_slice = [slice(None, None)] * img_ft.ndim
        nyquist_slice[a] = slice(old_nyquist_ind, old_nyquist_ind + 1)

        img_ft_pad[tuple(nyquist_slice)] *= 0.5

        # paired slice
        pair_frq_ind = old_nyquist_ind + img_ft.shape[a]
        pair_slice = [slice(None, None)] * img_ft.ndim
        pair_slice[a] = slice(pair_frq_ind, pair_frq_ind + 1)

        img_ft_pad[tuple(pair_slice)] = img_ft_pad[tuple(nyquist_slice)]

    return img_ft_pad


def conj_transpose_fft(img_ft: np.ndarray,
                       axes: tuple[int] = (-1, -2)) -> np.ndarray:
    """
    Given img_ft(f), return a new array
    img_new_ft(f) := conj(img_ft(-f))

    :param img_ft:
    :param axes: axes on which to perform the transformation
    """

    if isinstance(img_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    # convert axes to positive number
    axes = np.mod(np.array(axes), img_ft.ndim)

    # flip and conjugate
    img_ft_ct = xp.flip(xp.conj(img_ft), axis=tuple(axes))

    # for odd FFT size, can simply flip the array to take f -> -f
    # for even FFT size, have on more negative frequency than positive frequency component.
    # by flipping array, have put the negative frequency components on the wrong side of the array
    # (i.e. where the positive frequency components are)
    # so must roll array to put them back on the right side
    to_roll = [a for a in axes if np.mod(img_ft.shape[a], 2) == 0]
    img_ft_ct = xp.roll(img_ft_ct, shift=[1] * len(to_roll), axis=tuple(to_roll))

    return img_ft_ct


# create test data/SIM forward model
# todo: could think about giving a 3D stack and converting this ...
def get_simulated_sim_imgs(ground_truth: array,
                           frqs: np.ndarray,
                           phases: np.ndarray,
                           mod_depths: list[float],
                           gains: Union[float, array],
                           offsets: Union[float, array],
                           readout_noise_sds: Union[float, array],
                           pix_size: float,
                           amps: Optional[np.ndarray] = None,
                           coherent_projection: bool = True,
                           otf: Optional[np.ndarray] = None,
                           nbin: int = 1,
                           **kwargs) -> (array, array, array, array):
    """
    Get simulated SIM images, including the effects of shot-noise and camera noise.

    :param ground_truth: NumPy or CuPy array of size nz x ny x nx. If
    :param frqs: SIM frequencies, of size nangles x 2. frqs[ii] = [fx, fy]
    :param phases: SIM phases in radians. Of size nangles x nphases. Phases may be different for each angle.
    :param mod_depths: SIM pattern modulation depths. Size nangles. If pass matrices, then mod depths can vary
    spatially. Assume pattern modulation is the same for all phases of a given angle. Maybe pass list of numpy arrays
    :param gains: gain of each pixel (or single value for all pixels)
    :param offsets: offset of each pixel (or single value for all pixels)
    :param readout_noise_sds: noise standard deviation for each pixel (or single value for all pixels)
    :param pix_size: pixel size of the input image (i.e. BEFORE binning). The pixel size of the output image
    will be pix_size * nbin
    :param amps:
    :param coherent_projection:
    :param otf: the optical transfer function evaluated at the frequencies points of the FFT of ground_truth. The
    proper frequency points can be obtained using fft.fftshift(fft.fftfreq(nx, dx)) and etc.
    :param kwargs: keyword arguments which will be passed through to simulated_img()

    :return sim_imgs: nangles x nphases x nz x ny x nx array
    :return snrs: nangles x nphases x nz x ny x nx array giving an estimate of the signal-to-noise ratio which will be
    accurate as long as the photon number is large enough that the Poisson distribution is close to a normal distribution
    :return patterns, snrs, patterns, raw_patterns:
    """

    # if isinstance(ground_truth, cp.ndarray):
    if isinstance(ground_truth, cp.ndarray):
        xp = cp
    else:
        xp = np

    ground_truth = xp.asarray(ground_truth)
    gains = xp.asarray(gains)
    offsets = xp.asarray(offsets)
    readout_noise_sds = xp.asarray(readout_noise_sds)

    # ensure ground truth is 3D
    if ground_truth.ndim == 2:
        ground_truth = xp.expand_dims(ground_truth, axis=0)
    nz, ny, nx = ground_truth.shape

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
        amps = xp.ones((nangles, nphases))
    else:
        amps = xp.array(amps)

    if otf is not None:
        psf, _ = fit_psf.otf2psf(otf)
    else:
        psf = None

    # get binned sizes
    nxb = nx / nbin
    nyb = ny / nbin
    if not nxb.is_integer() or not nyb.is_integer():
        raise Exception("The image size was not evenly divisible by the bin size")
    nxb = int(nxb)
    nyb = int(nyb)

    # get binned coordinates
    xb = (xp.arange(nxb) - (nxb // 2)) * (pix_size * nbin)
    yb = (xp.arange(nyb) - (nyb // 2)) * (pix_size * nbin)

    # get unbinned coordinates
    # these are not the "natural" coordinates of the unbinned pixel grid
    # define them in terms of offsets from binned grid
    subpix_offsets = np.arange(-(nbin - 1) / 2, (nbin - 1) / 2 + 1) * pix_size

    x = (np.expand_dims(xb, axis=1) + np.expand_dims(subpix_offsets, axis=0)).ravel()
    y = (np.expand_dims(yb, axis=1) + np.expand_dims(subpix_offsets, axis=0)).ravel()
    z = (xp.arange(nz) - (nz // 2)) # do not need dz, so don't have pixel size for it

    _, yy, xx = xp.meshgrid(z, y, x, indexing="ij")

    # to ensure got coordinates right, can check that the binned coordinates agree with the values obtained
    # from binning the other coordinates
    assert np.max(np.abs(xb - camera.bin(xx, bin_sizes=(1, nbin, nbin), mode="mean")[0, 0, :])) < 1e-12
    assert np.max(np.abs(yb - camera.bin(yy, bin_sizes=(1, nbin, nbin), mode="mean")[0, :, 0])) < 1e-12

    # generate images
    frqs = xp.array(frqs)

    sim_imgs = xp.zeros((nangles, nphases, nz, nyb, nxb), dtype=int)
    patterns_raw = xp.zeros((nangles, nphases, nz, ny, nx), dtype=float)
    patterns = xp.zeros((nangles, nphases, nz, nyb, nxb), dtype=float)
    snrs = xp.zeros(sim_imgs.shape)
    mcnrs = xp.zeros(sim_imgs.shape)
    for ii in range(nangles):
        for jj in range(nphases):
            patterns_raw[ii, jj] = amps[ii, jj] * 0.5 * (1 + mod_depths[ii] * xp.cos(2 * np.pi * (frqs[ii][0] * xx + frqs[ii][1] * yy) + phases[ii, jj]))

            if not coherent_projection:
                patterns_raw[ii, jj] = fit_psf.blur_img_otf(patterns_raw[ii, jj], otf).real

            # bin pattern, for reference
            patterns[ii, jj], _ = camera.simulated_img(patterns_raw[ii, jj],
                                                       gains=1,
                                                       offsets=0,
                                                       readout_noise_sds=0,
                                                       psf=None,
                                                       photon_shot_noise=False,
                                                       bin_size=nbin,
                                                       image_is_integer=False)

            # forward SIM model
            sim_imgs[ii, jj], snrs[ii, jj] = camera.simulated_img(ground_truth * patterns_raw[ii, jj],
                                                                  gains=gains,
                                                                  offsets=offsets,
                                                                  readout_noise_sds=readout_noise_sds,
                                                                  psf=psf,
                                                                  bin_size=nbin,
                                                                  **kwargs)
            # todo: compute mcnr
            mcnrs[ii, jj] = 0

    return sim_imgs, snrs, patterns, patterns_raw
