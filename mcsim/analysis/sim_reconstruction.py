"""
Tools for reconstructing 2D sinusoidal SIM images from raw data.
The primary reconstruction code is contained in the class `SimImageSet`
Suppose we illuminate an object :math:`O(r)` with a series of patterns,

.. math::

  I_{ij}(r) = A_{ij} \\left[m_i \\cos (2\\pi f_i \\cdot r + \\phi_{ij}) \\right]

Then the images we measure are

.. math::

  D_{ij}(r) = \\left[I_{ij}(r) O(r) \\right] * h(r)

where :math:`h(r)` is the point-spread function of the system.

"""
from sys import stdout
from time import perf_counter
from warnings import warn
from typing import Union, Optional
from collections.abc import Sequence
# parallelization
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask.array as da
# numerics
import numpy as np
from scipy.fft import fftshift, ifftshift, fftfreq, fft2
from scipy.optimize import minimize
from scipy.signal import correlate
from scipy.signal.windows import tukey
from skimage.exposure import match_histograms as match_histograms_cpu
# working with external files
from pathlib import Path
from io import StringIO
# loading and exporting data
import json
import tifffile
import zarr
import h5py
# plotting
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm, LogNorm
from matplotlib.patches import Circle, Rectangle
# code from our projects
from mcsim.analysis.optimize import Optimizer, soft_threshold, tv_prox, to_cpu
from mcsim.analysis.fft import ft2, ift2, irft2, conj_transpose_fft, translate_ft
from localize_psf.rois import get_centered_rois, cut_roi
from localize_psf.fit_psf import circ_aperture_otf, blur_img_psf, oversample_voxel
from localize_psf.camera import bin, bin_adjoint, simulated_img

# GPU
try:
    import cupy as cp
    from cucim.skimage.exposure import match_histograms as match_histograms_gpu
except ImportError:
    cp = None
    match_histograms_gpu = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


class SimImageSet:
    allowed_frq_estimation_modes = ["band-correlation", "fourier-transform", "fixed"]
    allowed_phase_estimation_modes = ["wicker-iterative", "real-space", "naive", "fixed"]
    allowed_combine_bands_modes = ["fairSIM"]
    allowed_reconstruction_modes = ["wiener-filter"]  # "FISTA"
    allowed_mod_depth_estimation_modes = ["band-correlation", "fixed"]

    nbands = 3  # hardcoded for 2D sinusoidal SIM
    band_inds = np.array([0, 1, -1], dtype=int)  # bands are shifted by these multiples of frqs
    band_id = np.array([[1,  0,  0,  0],
                        [0,  1,  0,  0],
                        [0, -1,  0,  0],
                        [0,  0,  1,  0],
                        [0,  0, -1,  0],
                        [0,  0,  0,  1],
                        [0,  0,  0, -1]
                        ], dtype=int)  # n_bands x (n_frqs + 1)
    upsample_fact = 2

    @classmethod
    def initialize(cls,
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
                   normalize_histograms: bool = False,
                   determine_amplitudes: bool = False,
                   background: float = 0.,
                   gain: float = 1.,
                   max_phase_err: float = 10 * np.pi / 180,
                   min_p2nr: float = 1.,
                   trim_negative_values: bool = False,
                   use_gpu: bool = cp is not None,
                   print_to_terminal: bool = True,
                   axes_names: Optional[Sequence[str]] = None
                   ):
        """
        Simplified constructor for SimImageSet. Use this constructor when all SIM reconstruction parameters
        are known at the outset (including e.g. frequency and phase guesses but not necessarily final values).
        For usage examples see

        >>> help(SimImageSet)

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
        :param frq_estimation_mode: "band-correlation", "fourier-transform", or "fixed"
           "band-correlation" first unmixes the bands using the phase guess values and computes the correlation between
           the shifted and unshifted band
           "fourier-transform" correlates the Fourier transform of the image with itself.
           "fixed" uses the frq_guess values
        :param frq_guess: 2 x nangles array of guess SIM frequency values
        :param phase_estimation_mode: "wicker-iterative", "real-space", "naive", or "fixed"
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
        :param use_fixed_mod_depths: if true, use mod_depths_guess instead of estimating the modulation depths from the data
        :param mod_depth_otf_mask_threshold:
        :param minimum_mod_depth: if modulation depth is estimated to be less than this value, it will be replaced
           with this value during reconstruction
        :param normalize_histograms: for each phase, normalize histograms of images to account for laser power fluctuations
        :param background: Either a single number, or broadcastable to size of imgs. The background will be subtracted
           before running the SIM reconstruction
        :param determine_amplitudes: whether to determine amplitudes as part of Wicker phase optimization.
           This flag only has an effect if phase_estimation_mode is "wicker-iterative"
        :param background: a single number, or an array which is broadcastable to the same size as images. This will
           be subtracted from the raw data before processing.
        :param gain: gain of the camera in ADU/photons. This is a single number or an array which is broadcastable to
           the same size as the images whcih is sued to convert the ADU counts to photon numbers.
        :param max_phase_err: If the determined phase error between components exceeds this value, use the phase guess
           values instead of those determined by the estimation algorithm.
        :param min_p2nr: if the peak-to-noise ratio is smaller than this value, use the frequency guesses instead
            of the frequencies determined by the estimation algorithm.
        :param trim_negative_values: set values in SIM-SR reconstruction which are less than zero to zero
        :param use_gpu:
        :param print_to_terminal:
        :param axes_names: axes names for all input dimensions.
        """
        
        tstart = perf_counter()

        self = cls(use_gpu, print_to_terminal)

        # #############################################
        # preprocessing
        # #############################################
        self.preprocess_data(physical_params["pixel_size"],
                             physical_params["na"],
                             physical_params["wavelength"],
                             imgs,
                             normalize_histograms,
                             gain,
                             background,
                             axes_names)

        # #############################################
        # set reconstruction settings
        # #############################################
        self.update_recon_settings(wiener_parameter=wiener_parameter,
                                   frq_estimation_mode=frq_estimation_mode,
                                   phase_estimation_mode=phase_estimation_mode,
                                   combine_bands_mode=combine_bands_mode,
                                   fmax_exclude_band0=fmax_exclude_band0,
                                   use_fixed_mod_depths=use_fixed_mod_depths,
                                   mod_depth_otf_mask_threshold=mod_depth_otf_mask_threshold,
                                   minimum_mod_depth=minimum_mod_depth,
                                   determine_amplitudes=determine_amplitudes,
                                   max_phase_err=max_phase_err,
                                   min_p2nr=min_p2nr,
                                   trim_negative_values=trim_negative_values)

        # #############################################
        # OTF
        # #############################################
        self.update_otf(otf)

        # #############################################
        # set guess parameters
        # #############################################
        self.update_parameter_guesses(frq_guess=frq_guess,
                                      phases_guess=phases_guess,
                                      mod_depths_guess=mod_depths_guess)

        self.print_log(f"initialization took {perf_counter() - tstart:.2f}s")

        return self

    def __init__(self,
                 use_gpu: bool = cp is not None,
                 print_to_terminal: bool = True):
        """
        Reconstruct raw SIM data into widefield, SIM-SR, SIM-OS, and deconvolved images using the Wiener filter
        style reconstruction of Gustafsson and Heintzmann. This code relies on various ideas developed and
        implemented elsewhere, see for example fairSIM and openSIM.

        An instance of this class may be used directly to reconstruct a single SIM image which is stored as a
        3 x 3 x ny x nx NumPy array, or a larger image series of sizes n0 x ... x nm x 3 x 3 x ny x nx as long as
        the SIM parameters are the same for all images in the stack.

        :param use_gpu: Run SIM computations on the GPU
        :param print_to_terminal: Print diagnostic information to the terminal dureing reconstruction

        Coordinates:
        Both the raw data and the SIM data use the same coordinates as the FFT with the origin in the center.
        i.e. the coordinates in the raw image are x = (arange(nx) - (nx // 2)) * dxy
        and for the SIM image they are            x = ((arange(2*nx) - (2*nx)//2) * 0.5 * dxy
        Note that this means they cannot be overlaid by changing the scale for the SIM image by a factor of two.
        There is an additional translation. The origin in the raw images is at pixel n//2 while those in the SIM
        images are at (2*n) // 2 = n. This translation is due to the fact that for odd n,
        n // 2 != (2*n) // 2 * 0.5 = n / 2

        Examples:
        For normal use cases, it is best to use the classmethod initialize() as the constructor for the class.
        initialize() requires that all SIM reconstruction settings be known when the constructor is called.

        >>> sim_obj = SimImageSet.initialize(*args, **kwargs)
        >>> sim_obj.reconstruct()  # estimate parameters and reconstruct data

        However, there will be other cases where the desired SIM reconstruction settings are not known when
        the class is initialized. In this case, more granular controller is possible

        >>> sim_obj = SimImageSet()
        >>> sim_obj.preprocess_data()
        >>> sim_obj.update_otf()
        >>> sim_obj.update_recon_settings()
        >>> sim_obj.update_parameter_guesses()
        >>> sim_obj.reconstruct()

        Results can be printed to the terminal

        >>> sim_obj.print_parameters()  # print parameters to terminal

        Results can be saved to file

        >>> sim_obj.save_imgs()  # save images to file

        Diagnostics can be printed

        >>> sim_obj.plot_figs()  # plot diagnostic figures

        """

        # #############################################
        # logging and printing results
        # #############################################
        self._streams = []
        self.log = StringIO()  # can save this stream to a file later if desired
        self.add_stream(self.log)
        if print_to_terminal:
            self.add_stream(stdout)

        # #############################################
        # GPU
        # #############################################
        self.use_gpu = use_gpu
        if self.use_gpu and cp:
            raise ValueError("'use_gpu' was true, but CuPy could not be imported")

        # #############################################
        # Preprocessing setting
        # #############################################
        self.axes_names = None
        self.nangles = None
        self.nphases = None
        self.ny = None
        self.nx = None
        self.n_extra_dims = None
        self._preprocessing_settings = {}
        self._recon_settings = {}

        # todo: could replace with fmax only
        self.na = None
        self.wavelength = None
        self.fmax = None

        self.x = None
        self.y = None
        self.x_us = None
        self.y_us = None
        self.dx = None
        self.dy = None
        self.dx_us = None
        self.dy_us = None

        self.fx = None
        self.fy = None
        self.fx_us = None
        self.fy_us = None
        self.dfx = None
        self.dfy = None
        self.dfx_us = None
        self.dfy_us = None

        # #############################################
        # guess parameters
        # #############################################
        self.frqs_guess = None
        self.phases_guess = None
        self.mod_depths_guess = None
        self.band0_frq_fit = None  # keep reference only for easy diagnostic plotting
        self.band1_frq_fit = None

        # #############################################
        # parameters
        # #############################################
        self.otf = None
        self.frqs = None
        self.phases = None
        self.phase_corrections = None
        self.amps = None
        self.mod_depths = None

        # #############################################
        # diagnostics
        # #############################################
        self.p2nr = None
        self.peak_phases = None
        self.mcnr = None

        # #############################################
        # images
        # #############################################
        self.imgs_raw = None
        self.imgs = None
        self.imgs_ft = None
        self.bands_shifted_ft = None
        self.weights = None
        self.weights_norm = None
        self.patterns = None
        self.patterns_2x = None
        self.widefield = None
        self.widefield_deconvolution = None
        self.sim_os = None
        self.sim_sr = None
        self.sim_sr_ft_components = None

    def preprocess_data(self,
                        pix_size_um: float,
                        na: float,
                        wavelength: float,
                        imgs: array,
                        normalize_histograms: bool,
                        gain: float,
                        background: float,
                        axes_names: Optional[Sequence[str]] = None
                        ):
        """
        Preprocess SIM data

        :param pix_size_um:
        :param na:
        :param wavelength:
        :param imgs:
        :param normalize_histograms:
        :param gain:
        :param background:
        :param axes_names:
        :return:
        """

        self._preprocessing_settings = {"normalize_histograms": normalize_histograms,
                                        "gain": gain,
                                        "offset": background}

        # #############################################
        # Configure CPU/GPU
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

        if axes_names is None:
            self.axes_names = ["" for _ in range(self.n_extra_dims)] + ["angles", "phases", "y", "x"]
        else:
            self.axes_names = axes_names

        if len(self.axes_names) != imgs.ndim:
            raise ValueError(f"len(axes_names)={len(self.axes_names):d} which did not match data ndim={imgs.ndim:d}")

        # ensures imgs dask array with chunksize = 1 raw image
        chunk_size = (1,) * (self.n_extra_dims + 2) + imgs.shape[-2:]

        # most expensive operation, ~ 0.1s for 9x2048x2048 images
        if not isinstance(imgs, da.core.Array):
            imgs = da.from_array(imgs, chunks=chunk_size)
        else:
            imgs = imgs.rechunk(chunk_size)

        # ensure on CPU/GPU as appropriate
        self.imgs_raw = da.map_blocks(lambda x: xp.array(x.astype(float)),
                                      imgs,
                                      dtype=float,
                                      meta=xp.array((), dtype=float)
                                      )

        # #############################################
        # real space parameters
        # #############################################
        self.dx = float(pix_size_um)
        self.dy = float(pix_size_um)
        self.x = (xp.arange(self.nx) - (self.nx // 2)) * self.dx
        self.y = (xp.arange(self.ny) - (self.ny // 2)) * self.dy
        self.dx_us = self.dx / self.upsample_fact
        self.dy_us = self.dy / self.upsample_fact
        self.x_us = (xp.arange(self.nx * self.upsample_fact) - (self.nx * self.upsample_fact) // 2) * self.dx_us
        self.y_us = (xp.arange(self.ny * self.upsample_fact) - (self.ny * self.upsample_fact) // 2) * self.dy_us

        # #############################################
        # physical parameters
        # #############################################
        self.na = na
        self.wavelength = wavelength
        self.fmax = 1 / (0.5 * self.wavelength / self.na)

        # #############################################
        # frequency data
        # #############################################
        self.fx = xp.fft.fftshift(xp.fft.fftfreq(self.nx, self.dx))
        self.fy = xp.fft.fftshift(xp.fft.fftfreq(self.ny, self.dy))
        self.fx_us = xp.fft.fftshift(xp.fft.fftfreq(self.upsample_fact * self.nx, self.dx / self.upsample_fact))
        self.fy_us = xp.fft.fftshift(xp.fft.fftfreq(self.upsample_fact * self.ny, self.dy / self.upsample_fact))
        self.dfx = float(self.fx[1] - self.fx[0])
        self.dfy = float(self.fy[1] - self.fy[0])
        self.dfx_us = float(self.fx_us[1] - self.fx_us[0])
        self.dfy_us = float(self.fy_us[1] - self.fy_us[0])

        # #############################################
        # image preprocessing
        # #############################################
        # remove background and convert from ADU to photons
        # todo: this should probably be users responsibility
        self.imgs = (self.imgs_raw - self._preprocessing_settings["offset"]) / self._preprocessing_settings["gain"]
        self.imgs[self.imgs <= 0] = 1e-12

        # normalize histograms for each angle
        if self._preprocessing_settings["normalize_histograms"]:
            # todo: should I rewrite this to handle full chunked images? Then avoid need to rechunk
            tstart_norm_histogram = perf_counter()

            matched_hists = da.map_blocks(match_histograms,
                                          self.imgs[..., slice(1, None), :, :],
                                          self.imgs[..., slice(0, 1), :, :],
                                          chunks=(1,) * (self.n_extra_dims + 2) + self.imgs.shape[-2:],
                                          dtype=self.imgs.dtype,
                                          meta=xp.array(())
                                          )

            self.imgs = da.concatenate((self.imgs[..., slice(0, 1), :, :],
                                        matched_hists),
                                       axis=-3)

            self.print_log(f"Normalizing histograms took {perf_counter() - tstart_norm_histogram:.2f}s")

        # #############################################
        # Rechunk so working on single SIM image at a time, which is necessary during reconstruction
        # #############################################
        new_chunks = list(self.imgs.chunksize)
        new_chunks[-4:] = self.imgs.shape[-4:]
        self.imgs = da.rechunk(self.imgs, new_chunks)

        # #############################################
        # Fourier transform raw SIM images
        # #############################################
        apodization = xp.outer(xp.asarray(tukey(self.ny, alpha=0.1)),
                               xp.asarray(tukey(self.nx, alpha=0.1)))

        # todo: want to do a real ft instead
        self.imgs_ft = da.map_blocks(ft2,
                                     self.imgs * apodization,
                                     dtype=complex,
                                     meta=xp.array(())
                                     )

    def update_recon_settings(self,
                              wiener_parameter: float = 0.1,
                              frq_estimation_mode: str = "band-correlation",
                              phase_estimation_mode: str = "wicker-iterative",
                              combine_bands_mode: str = "fairSIM",
                              fmax_exclude_band0: float = 0,
                              use_fixed_mod_depths: bool = False,
                              mod_depth_otf_mask_threshold: float = 0.1,
                              minimum_mod_depth: float = 0.5,
                              determine_amplitudes: bool = False,
                              max_phase_err: float = 10 * np.pi / 180,
                              min_p2nr: float = 1.,
                              trim_negative_values: bool = False,
                              ):
        """
        Set flags and settings for SIM reconstruction

        :param wiener_parameter:
        :param frq_estimation_mode:
        :param phase_estimation_mode:
        :param combine_bands_mode:
        :param fmax_exclude_band0:
        :param use_fixed_mod_depths:
        :param mod_depth_otf_mask_threshold:
        :param minimum_mod_depth:
        :param determine_amplitudes:
        :param max_phase_err:
        :param min_p2nr:
        :param trim_negative_values:
        :return:
        """
        # todo: argument checking
        reconstruction_mode = "wiener-filter"
        if reconstruction_mode not in self.allowed_reconstruction_modes:
            raise ValueError(f"reconstruction_mode must be one of {self.allowed_reconstruction_modes},"
                             f" but was {reconstruction_mode:s}")

        if phase_estimation_mode not in self.allowed_phase_estimation_modes:
            raise ValueError(f"phase_estimation_mode must be one of {self.allowed_phase_estimation_modes},"
                             f" but was {phase_estimation_mode:s}")

        # mod depth estimation mode
        if use_fixed_mod_depths:
            mod_depth_estimation_mode = "fixed"
        else:
            mod_depth_estimation_mode = "band-correlation"
        if mod_depth_estimation_mode not in self.allowed_mod_depth_estimation_modes:
            raise ValueError(f"mod_depth_estimation_mode must be one of {self.allowed_mod_depth_estimation_modes},"
                             f"but was {mod_depth_estimation_mode}")

        if wiener_parameter <= 0 or wiener_parameter > 1:
            raise ValueError(f"Wiener parameter must be between 0 and 1, but was {wiener_parameter:.3f}")

        if frq_estimation_mode not in self.allowed_frq_estimation_modes:
            raise ValueError(f"frq_estimation must be one of {self.allowed_frq_estimation_modes},"
                             f" but was {frq_estimation_mode:s}")

        self._recon_settings = {"reconstruction_mode": reconstruction_mode,
                                "phase_estimation_mode": phase_estimation_mode,
                                "frq_estimation_mode": frq_estimation_mode,
                                "combine_bands_mode": combine_bands_mode,
                                "mod_depth_estimation_mode": mod_depth_estimation_mode,
                                "wiener_parameter": wiener_parameter,
                                "determine_amplitudes": determine_amplitudes,
                                "max_phase_error": max_phase_err,
                                "min_p2nr": min_p2nr,
                                "enforce_positivity": trim_negative_values,
                                "fmax_exclude_band0": fmax_exclude_band0,
                                "otf_mask_threshold": mod_depth_otf_mask_threshold,
                                "minimum_mod_depth": minimum_mod_depth
                                }

    def update_otf(self,
                   otf: Optional[array] = None):
        """
        Set OTF

        :param otf:
        """
        # #############################################
        # OTF
        # #############################################
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        if otf is None:
            otf = circ_aperture_otf(xp.expand_dims(self.fx, axis=0),
                                    xp.expand_dims(self.fy, axis=1),
                                    self.na,
                                    self.wavelength)

        if np.any(otf < 0) or np.any(otf > 1):
            raise ValueError("OTF values must fall in [0, 1]")

        otf = xp.asarray(otf)

        # otf is stored as nangles x ny x nx array to allow for possibly different OTF's along directions (e.g. OPM-SIM)
        # todo: can I rely on broadcasting instead of tiling?
        if otf.ndim == 2:
            otf = xp.tile(otf, [self.nangles, 1, 1])

        if otf.shape[-2:] != self.imgs_raw.shape[-2:]:
            raise ValueError(f"OTF shape {otf.shape} and image shape {self.imgs_raw.shape} are not compatible")

        self.otf = otf

    def update_parameter_guesses(self,
                                 frq_guess: Optional[np.ndarray] = None,
                                 phases_guess: Optional[np.ndarray] = None,
                                 mod_depths_guess: Optional[np.ndarray] = None,
                                 ):
        """
        Set SIM parameter guess values

        :param frq_guess:
        :param phases_guess:
        :param mod_depths_guess:
        :return:
        """

        if frq_guess is not None:
            self.frqs_guess = np.array(frq_guess)
        else:
            self.frqs_guess = None

        if phases_guess is not None:
            self.phases_guess = np.array(phases_guess)
        else:
            self.phases_guess = None

        self.bands_unmixed_ft_guess = da.map_blocks(unmix_bands,
                                                    self.imgs_ft,
                                                    self.phases_guess,
                                                    mod_depths=np.ones(self.nangles),
                                                    dtype=complex,
                                                    )

        # todo: compute cross-correlations so can guess frequencies from them

        if mod_depths_guess is not None:
            self.mod_depths_guess = np.array(mod_depths_guess)
        else:
            self.mod_depths_guess = np.ones(self.nangles)

    def delete(self):
        """
        Delete data on GPU, otherwise will be retained and look like memory leak

        :return:
        """
        if self.use_gpu:
            for attr_name in dir(self):
                if attr_name in self.__dict__.keys():
                    del self.__dict__[attr_name]

    def estimate_parameters(self,
                            slices: Optional[tuple] = None,
                            frq_max_shift: Optional[float] = None,
                            frq_search_bounds: Sequence[float] = (0., np.inf)):
        """
        Estimate SIM parameters from chosen slice

        :return:
        """

        if frq_max_shift is None:
            frq_max_shift = 5 * self.dfx

        tstart_param_est = perf_counter()

        if self.phases_guess is None and self._recon_settings["frq_estimation_mode"] == "band-correlation":
            raise ValueError(f"frq_estimation_mode=`band-correlation`, but this requires phase guesses,"
                             f"and no phase guesses were provided")

        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            memory_start = mempool.used_bytes()

        self.print_log("starting parameter estimation...")

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        if self._recon_settings["frq_estimation_mode"] != "fixed" or \
           self._recon_settings["phase_estimation_mode"] != "fixed" or \
           self._recon_settings["mod_depth_estimation_mode"] != "fixed":
            # get slice of image stack to estimate SIM parameters from
            if slices is None:
                slices = tuple([slice(None) for _ in range(self.n_extra_dims)])

            # always average over first dims after slicing...
            imgs = da.mean(self.imgs[slices],
                           axis=tuple(range(self.n_extra_dims))).compute()
            imgs_ft = da.mean(self.imgs_ft[slices],
                              axis=tuple(range(self.n_extra_dims))).compute()

            if imgs_ft.shape[0] != self.nangles or imgs_ft.shape[1] != self.nphases or imgs_ft.ndim != 4:
                raise ValueError("sliced image incorrect size for parameter estimation")

        # #############################################
        # estimate frequencies
        # #############################################
        if self._recon_settings["frq_estimation_mode"] == "fixed":
            self.frqs = self.frqs_guess
        else:
            tstart = perf_counter()

            if self._recon_settings["frq_estimation_mode"] == "fourier-transform":
                # determine SIM frequency directly from Fourier transform
                band0 = imgs_ft[:, 0]
                band1 = imgs_ft[:, 0]

            elif self._recon_settings["frq_estimation_mode"] == "band-correlation":
                # determine SIM frequency from separated frequency bands using guess phases
                bands_unmixed_ft_temp = unmix_bands(imgs_ft,
                                                    self.phases_guess,
                                                    mod_depths=np.ones(self.nangles))

                band0 = bands_unmixed_ft_temp[:, 0]
                band1 = bands_unmixed_ft_temp[:, 1]

            else:
                raise ValueError(f"frq_estimation_mode must be one of {self.allowed_frq_estimation_modes}"
                                 f" but was '{self._recon_settings['frq_estimation_mode']:s}'")

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
                r.append(delayed(fit_modulation_frq)(
                    self.band0_frq_fit[ii],
                    self.band1_frq_fit[ii],
                    self.dx,
                    frq_guess=frq_guess[ii],
                    max_frq_shift=frq_max_shift,
                    fbounds=frq_search_bounds,
                    otf=self.otf[ii])
                )
            results = compute(*r)
            frqs, _, _ = zip(*results)
            self.frqs = np.array(frqs).reshape((self.nangles, 2))

            self.print_log(f"estimating {self.nangles:d} frequencies "
                           f"using mode {self._recon_settings['frq_estimation_mode']:s} "
                           f"took {perf_counter() - tstart:.2f}s")

        # #############################################
        # estimate peak heights
        # #############################################
        if self._recon_settings["frq_estimation_mode"] != "fixed":
            tstart = perf_counter()

            noise = np.sqrt(get_noise_power(imgs_ft,
                                            (self.dy, self.dx),
                                            self.fmax))

            peak_phases = xp.zeros((self.nangles, self.nphases))
            peak_heights = xp.zeros((self.nangles, self.nphases))
            p2nr = xp.zeros((self.nangles, self.nphases))
            for ii in range(self.nangles):
                peak_val = get_peak_value(imgs_ft[ii],
                                          self.fx,
                                          self.fy,
                                          self.frqs[ii],
                                          peak_pixel_size=1)
                peak_heights[ii] = xp.abs(peak_val)
                peak_phases[ii] = xp.angle(peak_val)
                p2nr[ii] = peak_heights[ii] / noise[ii]

                # if p2nr is too low use guess values instead
                if np.min(p2nr[ii]) < self._recon_settings["min_p2nr"] and self.frqs_guess is not None:
                    self.frqs[ii] = self.frqs_guess[ii]
                    self.print_log(f"SIM peak-to-noise ratio for angle={ii:d} is"
                                   f" {np.min(p2nr[ii]):.2f} < {self._recon_settings['min_p2nr']:.2f}, "
                                   f"the so frequency fit will be ignored "
                                   f"and the guess value will be used instead.")

                    peak_val = get_peak_value(imgs_ft[ii],
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

            self.print_log(f"estimated peak-to-noise ratio in {perf_counter() - tstart:.2f}s")

        # #############################################
        # estimate phases
        # todo: as with frqs since cannot easily go on GPU ...
        # #############################################
        tstart = perf_counter()

        if self._recon_settings["phase_estimation_mode"] == "fixed":
            phases = self.phases_guess
            amps = np.ones((self.nangles, self.nphases))
        elif self._recon_settings["phase_estimation_mode"] == "naive":
            phases = self.peak_phases
        elif self._recon_settings["phase_estimation_mode"] == "wicker-iterative":
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
                r.append(delayed(get_phase_wicker_iterative)(
                    imft[ii],
                    otfs[ii],
                    self.frqs[ii],
                    self.dx,
                    self.fmax,
                    phases_guess=phase_guess[ii],
                    fit_amps=self._recon_settings["determine_amplitudes"]
                ))
            results = compute(*r)
            phases, amps, _ = zip(*results)
            phases = np.array(phases)
            amps = np.array(amps)

            if self.use_gpu:
                # find this is necessary, else mempool gets too big for 8GB GPU's
                mempool.free_all_blocks()
                # self.print_log(f"after phase estimation used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB")
                # self.print_log(f"GPU memory pool = {(mempool.total_bytes()) / 1e9:.3f}GB")

        elif self._recon_settings["phase_estimation_mode"] == "real-space":
            phase_guess = self.phases_guess
            if phase_guess is None:
                phase_guess = np.zeros((self.nangles, self.nphases))

            im = imgs
            if self.use_gpu:
                im = im.get()

            r = []
            for ii in range(self.nangles):
                for jj in range(self.nphases):
                    r.append(delayed(get_phase_realspace)(
                        im[ii, jj],
                        self.frqs[ii],
                        self.dx,
                        phase_guess=phase_guess[ii, jj],
                        use_fft_origin=True
                    ))
            results = compute(*r)
            phases = np.array(results).reshape((self.nangles, self.nphases))
            amps = np.ones((self.nangles, self.nphases))

            if self.use_gpu:
                # find this is necessary, else mempool gets too big for 8GB GPU's
                mempool.free_all_blocks()
                # self.print_log(f"after phase estimation used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB")
                # self.print_log(f"GPU memory pool = {(mempool.total_bytes()) / 1e9:.3f}GB")
        else:
            raise ValueError(f"phase_estimation_mode must be one of {self.allowed_phase_estimation_modes}"
                             f" but was '{self._recon_settings['phase_estimation_mode']:s}'")

        self.phases = np.array(phases)
        self.amps = np.array(amps)

        self.print_log(f"estimated {self.nangles * self.nphases:d} phases"
                       f" using mode {self._recon_settings['phase_estimation_mode']:s} "
                       f"in {perf_counter() - tstart:.2f}s")

        # #############################################
        # check if phase fit was too bad, and default to guess values
        # #############################################
        # if self.phases_guess is not None and self.phase_estimation_mode != "fixed":
        if self.phases_guess is not None and self._recon_settings["phase_estimation_mode"] != "fixed":
            phase_guess_diffs = np.mod(self.phases_guess - self.phases_guess[:, 0][:, None], 2 * np.pi)
            phase_diffs = np.mod(self.phases - self.phases[:, 0][:, None], 2 * np.pi)

            for ii in range(self.nangles):
                diffs = np.mod(phase_guess_diffs[ii] - phase_diffs[ii], 2 * np.pi)
                condition = np.abs(diffs - 2 * np.pi) < diffs
                diffs[condition] = diffs[condition] - 2 * np.pi

                if np.any(np.abs(diffs) > self._recon_settings["max_phase_error"]):
                    self.phases[ii] = self.phases_guess[ii]
                    strv = f"Angle {ii:d} phase guesses have more than the maximum allowed" \
                           f" phase error={self._recon_settings['max_phase_error'] * 180 / np.pi:.2f}deg." \
                           f" Defaulting to guess values"

                    strv += "\nfit phase diffs="
                    for jj in range(self.nphases):
                        strv += f"{phase_diffs[ii, jj] * 180 / np.pi:.2f}deg, "

                    self.print_log(strv)

        # #############################################
        # estimate global phase shifts/modulation depths
        # #############################################
        if self._recon_settings["phase_estimation_mode"] != "fixed" or \
           self._recon_settings["mod_depth_estimation_mode"] != "fixed":
            tstart_mod_depth = perf_counter()

            # self.print_log(f"before band separation used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB")
            # self.print_log(f"GPU memory pool = {(mempool.total_bytes()) / 1e9:.3f}GB")

            # do band separation
            bands_shifted_ft = shift_bands(unmix_bands(imgs_ft, self.phases, amps=self.amps),
                                           self.frqs,
                                           (self.dy, self.dx),
                                           self.upsample_fact)

            if self.use_gpu:
                # self.print_log(f"after upsampling and band shifting used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB")
                # self.print_log(f"GPU memory pool = {(mempool.total_bytes()) / 1e9:.3f}GB")
                mempool.free_all_blocks()

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
                    otf_shifted[ii, jj], _ = translate_pix(otf_us[ii],
                                                           self.frqs[ii] * band_ind,
                                                           dr=(self.dfx_us, self.dfy_us),
                                                           axes=(1, 0),
                                                           wrap=False)

            # correct global phases and estimate modulation depth from band correlations

            # mask regions where OTF's are below threshold
            mask = xp.logical_and(otf_shifted[:, 0] > self._recon_settings["otf_mask_threshold"],
                                  otf_shifted[:, 1] > self._recon_settings["otf_mask_threshold"])

            # mask regions near frequency modulation which may be corrupted by out-of-focus light
            if self._recon_settings["fmax_exclude_band0"] > 0:
                for ii in range(self.nangles):
                    # exclude positive freq
                    ff_us = xp.sqrt(xp.expand_dims(self.fx_us, axis=0) ** 2 +
                                    xp.expand_dims(self.fy_us, axis=1) ** 2)
                    mask[ii][ff_us < self.fmax * self._recon_settings["fmax_exclude_band0"]] = False

                    # exclude negative frq
                    ff_us = xp.sqrt(xp.expand_dims(self.fx_us + self.frqs[ii, 0], axis=0) ** 2 +
                                    xp.expand_dims(self.fy_us + self.frqs[ii, 1], axis=1) ** 2)
                    mask[ii][ff_us < self.fmax * self._recon_settings["fmax_exclude_band0"]] = False

            for ii in range(self.nangles):
                if not np.any(mask[ii]):
                    raise ValueError(f"band overlap mask for angle {ii:d} was all False. "
                                     f"This may indicate the SIM frequency is incorrect. Check if the frequency "
                                     f"fitting routine failed. Otherwise, reduce `otf_mask_threshold` "
                                     f"which is currently {self._recon_settings['otf_mask_threshold']:.3f} and/or reduce "
                                     f"`fmax_exclude_band0` which is currently {self._recon_settings['fmax_exclude_band0']:.3f}")

            # corrected phases
            # can either think of these as (1) acting on phases such that phase -> phase - phase_correction
            # or (2) acting on bands such that band1(f) -> e^{i*phase_correction} * band1(f)
            # TODO: note, the global phases I use here have the opposite sign relative to our BOE paper eq. S47
            global_phase_corrections, mags = get_band_overlap(bands_shifted_ft[..., 0, :, :],
                                                              bands_shifted_ft[..., 1, :, :],
                                                              otf_shifted[..., 0, :, :],
                                                              otf_shifted[..., 1, :, :],
                                                              mask)

            self.print_log(f"estimated global phases and modulation depths in {perf_counter() - tstart_mod_depth:.2f}s")

            if self.use_gpu:
                # self.print_log(f"after phase correction used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB")
                # self.print_log(f"GPU memory pool = {(mempool.total_bytes()) / 1e9:.3f}GB")
                mempool.free_all_blocks()

            del mask
            del otf_shifted
            del bands_shifted_ft

        if self._recon_settings["phase_estimation_mode"] == "fixed":
            self.phase_corrections = xp.zeros(self.nangles)
        else:
            self.phase_corrections = global_phase_corrections

        if self._recon_settings["mod_depth_estimation_mode"] == "fixed":
            self.mod_depths = xp.asarray(self.mod_depths_guess)
        else:
            self.mod_depths = mags

            for ii in range(self.nangles):
                if self.mod_depths[ii] < self._recon_settings["minimum_mod_depth"]:
                    self.print_log(f"replaced modulation depth for angle {ii:d} because estimated value"
                                   f" was less than allowed minimum,"
                                   f" {self.mod_depths[ii]:.3f} < {self._recon_settings['minimum_mod_depth']:.3f}")
                    self.mod_depths[ii] = self._recon_settings["minimum_mod_depth"]

        # #############################################
        # estimate patterns
        # #############################################
        frqs_1d = np.kron(self.frqs, np.ones((self.nphases, 1)))
        phases_1d = (self.phases - np.expand_dims(self.phase_corrections, axis=1)).reshape(self.nangles * self.nphases)
        mods_1d = np.kron(self.mod_depths, np.ones(self.nphases))
        amps_1d = self.amps.reshape(self.nphases * self.nangles)

        p2x_delayed = delayed(get_sinusoidal_patterns)(self.dx,
                                                       (self.ny, self.nx),
                                                       frqs_1d,
                                                       phases_1d,
                                                       mods_1d,
                                                       amps_1d,
                                                       n_oversampled=self.upsample_fact)

        self.patterns_2x = da.from_delayed(p2x_delayed, shape=(self.nangles * self.nphases,
                                                               self.ny * self.upsample_fact,
                                                               self.nx * self.upsample_fact),
                                           dtype=float)
        self.patterns = da.map_blocks(bin,
                                      self.patterns_2x,
                                      (self.upsample_fact, self.upsample_fact),
                                      mode="sum",
                                      dtype=float,
                                      chunks=(self.nangles * self.nphases, self.ny, self.nx))

        # #############################################
        # print info
        # #############################################
        self.print_log(f"parameter estimation took {perf_counter() - tstart_param_est:.2f}s")

    def reconstruct(self,
                    slices: Optional[tuple] = None,
                    compute_widefield: bool = False,
                    compute_os: bool = False,
                    compute_deconvolved: bool = False,
                    compute_mcnr: bool = False,
                    compute_sr: bool = True,
                    frq_search_bounds: Sequence[float] = (0., 1.)):
        """
        SIM reconstruction

        :param slices: tuple of slice objects describing the array to use for fitting SIM parameters
        :param compute_widefield:
        :param compute_os:
        :param compute_deconvolved:
        :param compute_mcnr:
        :param compute_sr:
        :param frq_search_bounds: (fmin_search, fmax_search) as fractions of fmax. Note that the upperbound
          must be increased above 1 for TIRF-SIM data.
        :return:
        """

        if self.use_gpu and cp:
            xp = cp
        else:
            xp = np

        # #############################################
        # parameter estimation
        # #############################################
        frq_search_bounds = np.array(frq_search_bounds)
        frq_search_bounds[0] *= self.fmax
        frq_search_bounds[1] *= self.fmax

        self.estimate_parameters(slices=slices,
                                 frq_search_bounds=frq_search_bounds)

        # if self.use_gpu:
        #     mempool = cp.get_default_memory_pool()
        #     self.print_log(f"used GPU memory = {(mempool.used_bytes()) / 1e9:.3f}GB")
        #     self.print_log(f"GPU memory pool = {(mempool.total_bytes()) / 1e9:.3f}GB")

        # #############################################
        # various types of "reconstruction"
        # #############################################
        tstart_recon = perf_counter()

        # #############################################
        # get widefield image
        # #############################################
        if compute_widefield:
            apodization = xp.array(np.outer(tukey(self.ny, alpha=0.1),
                                            tukey(self.nx, alpha=0.1)))

            self.widefield = da.nanmean(self.imgs, axis=(-3, -4))

        # #############################################
        # get optically sectioned image
        # #############################################
        if compute_os:
            tstart = perf_counter()

            os_imgs = da.stack([da.map_blocks(sim_optical_section,
                                              self.imgs[..., ii, :, :, :],
                                              phase_differences=self.phases[ii],
                                              axis=-3,
                                              drop_axis=-3,
                                              dtype=float,
                                              meta=xp.array((), dtype=self.imgs.dtype))
                                for ii in range(self.nangles)], axis=-3)
            self.sim_os = da.mean(os_imgs, axis=-3)
            self.print_log(f"Computing SIM-OS image took {perf_counter() - tstart:.2f}s")

        # #############################################
        # estimate spatial-resolved MCNR
        # #############################################
        if compute_mcnr:
            # following the proposal of https://doi.org/10.1038/s41592-021-01167-7
            # calculate as the ratio of the modulation size over the expected shot noise value
            # note: this is the same as sim_os / sqrt(wf_angle) up to a factor
            tstart = perf_counter()

            # divide by nangles to remove ft normalization
            def ft_mcnr(m, nangles, use_gpu):
                return xp.fft.fft(xp.fft.ifftshift(m, axes=-3), axis=-3) / nangles

            img_angle_ft = da.map_blocks(ft_mcnr,
                                         self.imgs,
                                         self.nangles,
                                         self.use_gpu,
                                         dtype=complex,
                                         meta=xp.array((), dtype=complex))
            # if I_j = Io * m * cos(2*pi*j), then want numerator to be 2*m. FT gives us m/2, so multiply by 4
            self.mcnr = (4 * da.abs(img_angle_ft[..., 1, :, :]) / da.sqrt(da.abs(img_angle_ft[..., 0, :, :])))

            self.print_log(f"estimated modulation-contrast-to-noise ratio in {perf_counter() - tstart:.2f}s")

        # #############################################
        # compute SR-SIM image
        # #############################################
        if compute_sr:
            # #############################################
            # do band separation and then shift bands
            # results are:
            # [O(f)H(f), m*O(f - f_o)H(f), m*O(f + f_o)H(f)]
            # [O(f)H(f), m*O(f)H(f + fo),  m*O(f)H(f - fo)]
            # #############################################

            exp_chunks = list(self.imgs_ft.chunksize)
            exp_chunks[-1] *= self.upsample_fact
            exp_chunks[-2] *= self.upsample_fact

            # this approach saves ~5GB in memory pool
            def proc_bands(bands, phases, amps, frqs, dy, dx, upsample_fact, use_gpu):
                sb = shift_bands(unmix_bands(bands, phases, amps=amps), frqs, (dy, dx), upsample_fact)

                # todo: need this line or have problem with memory ... but don't really understand why
                # todo: thought this was some process issue where since dask is running a different process
                # todo: the parent process doesn't know if can use the allocated and then freed memory which becomes
                # todo: part of the pool, but other tests don't seem to support this
                if use_gpu:
                    cp.get_default_memory_pool().free_all_blocks()

                return sb

            self.bands_shifted_ft = da.map_blocks(proc_bands,
                                                  self.imgs_ft,
                                                  self.phases,
                                                  self.amps,
                                                  self.frqs,
                                                  self.dy,
                                                  self.dx,
                                                  self.upsample_fact,
                                                  self.use_gpu,
                                                  dtype=complex,
                                                  chunks=exp_chunks,
                                                  meta=xp.array((), dtype=complex)
                                                  )

            # #############################################
            # shift OTFs
            # #############################################
            otf_us = resample_bandlimited_ft(self.otf,
                                             (self.upsample_fact, self.upsample_fact),
                                             axes=(-1, -2)) / self.upsample_fact / self.upsample_fact

            self.weights = xp.zeros((otf_us.shape[0], self.nbands, otf_us.shape[1], otf_us.shape[2]), dtype=complex)
            for ii in range(self.nangles):
                for jj, band_ind in enumerate(self.band_inds):
                    # compute otf(k + m * ko)
                    # todo: want to be able to run this with map_blocks() like tools.translate_ft()
                    self.weights[ii, jj] = translate_pix(otf_us[ii],
                                                         self.frqs[ii] * band_ind,
                                                         dr=(self.dfx_us, self.dfy_us),
                                                         axes=(1, 0),
                                                         wrap=False)[0].conj()

            # #############################################
            # Get weights and do band-replacement
            # #############################################
            # "fill in missing cone" by using shifted bands instead of unshifted band for values near DC
            if self._recon_settings["fmax_exclude_band0"] > 0:
                fx_shift = xp.expand_dims(self.fx_us, axis=(0, 1, -2)) + \
                           xp.expand_dims(xp.array(self.frqs[..., 0]), axis=(1, 2, 3)) * xp.expand_dims(xp.array(self.band_inds), axis=(0, -1, -2))
                fy_shift = xp.expand_dims(self.fy_us, axis=(0, 1, -1)) + \
                           xp.expand_dims(xp.array(self.frqs[..., 1]), axis=(1, 2, 3)) * xp.expand_dims(xp.array(self.band_inds), axis=(0, -1, -2))
                self.weights *= (1 - xp.exp(-0.5 * (fx_shift**2 + fy_shift**2) / (self.fmax * self._recon_settings["fmax_exclude_band0"])**2))

            self.weights_norm = self._recon_settings["wiener_parameter"] ** 2 + xp.nansum(xp.abs(self.weights) ** 2, axis=(0, 1), keepdims=True)

            # #############################################
            # combine bands
            # following the approach of FairSIM: https://doi.org/10.1038/ncomms10980
            # #############################################
            tstart_combine_bands = perf_counter()

            # nangles x nphases
            corr_mat = xp.concatenate((xp.ones((self.nangles, 1)),
                                       xp.expand_dims(xp.exp(1j * self.phase_corrections) / self.mod_depths, axis=1),
                                       xp.expand_dims(xp.exp(-1j * self.phase_corrections) / self.mod_depths, axis=1)),
                                       axis=1)

            # expand for xy dims
            corr_mat = xp.expand_dims(corr_mat, axis=(-1, -2))

            # put in modulation depth and global phase corrections
            # components array useful for diagnostic plots
            # todo: if do not explicitely delete self.sim_sr_ft_components it will hold memory
            # self.sim_sr_ft_components = self.bands_shifted_ft * self.weights * corr_mat / self.weights_norm
            self.sim_sr_ft_components = self.bands_shifted_ft * corr_mat
            self.sim_sr_ft_components *= self.weights
            self.sim_sr_ft_components /= self.weights_norm

            # final FT image
            sim_sr_ft = da.nansum(self.sim_sr_ft_components, axis=(-3, -4))

            # inverse FFT to get real-space reconstructed image
            apodization = xp.outer(xp.asarray(tukey(sim_sr_ft.shape[-2], alpha=0.1)),
                                   xp.asarray(tukey(sim_sr_ft.shape[-1], alpha=0.1)))

            self.sim_sr = da.map_blocks(irft2,
                                        sim_sr_ft * apodization,
                                        dtype=float,
                                        meta=xp.array((), dtype=float)
                                        )

            if self._recon_settings["enforce_positivity"]:
                self.sim_sr[self.sim_sr < 0] = 0

            self.print_log(f"combining bands took {perf_counter() - tstart_combine_bands:.2f}s")

            # #############################################
            # widefield deconvolution
            # NOTE: only computed if SR-SIM is also
            # #############################################
            if compute_deconvolved:
                tstart = perf_counter()

                weights_decon = otf_us
                decon_ft = da.nansum(weights_decon * self.bands_shifted_ft[..., 0, :, :], axis=-3) / \
                                                      (self._recon_settings["wiener_parameter"]**2 + da.nansum(np.abs(weights_decon)**2, axis=-3))

                self.widefield_deconvolution = da.map_blocks(irft2,
                                                             decon_ft * apodization,
                                                             dtype=float,
                                                             meta=xp.array((), dtype=float)
                                                             )

                self.print_log(f"Deconvolved widefield in {perf_counter() - tstart:.2f}s")

        # #############################################
        # move arrays off GPU
        # #############################################
        if self.use_gpu:
            for attr_name in dir(self):
                attr = getattr(self, attr_name)

                # if cupy array, move off GPU
                if isinstance(attr, cp.ndarray):
                    setattr(self, attr_name, to_cpu(attr))

                # if dask array, move off GPU delayed
                if isinstance(attr, da.core.Array):
                    on_cpu = da.map_blocks(to_cpu, attr, dtype=attr.dtype)
                    setattr(self, attr_name, on_cpu)

        self.print_log(f"reconstruction took {perf_counter() - tstart_recon:.2f}s")

    # printing utility functions
    def print_parameters(self):
        """
        Print parameters used during SIM reconstruction

        :return:
        """
        self.print_log(f"SIM reconstruction for {self.nangles:d} angles and {self.nphases:d} phases")
        self.print_log(f"images are size {self.ny:d}x{self.nx:d} with pixel size {self.dx:.3f}x{self.dy:.3f}um")

        for k, v in self._preprocessing_settings.items():
            self.print_log(f"'{k:s}' = {v}")

        for k, v in self._recon_settings.items():
            self.print_log(f"'{k:s}' = {v}")

        # print per-angle data
        for ii in range(self.nangles):
            self.print_log(f"################ Angle {ii:d} ################")
            self.print_log(f"modulation depth = {self.mod_depths[ii]:0.3f}")

            amp_str = np.array2string(self.amps[ii], formatter={"float": lambda x: f"{x:05.3f}", "separator": ", "})
            self.print_log(f"amplitudes = {amp_str:s}")

            if self.p2nr is not None:
                self.print_log(f"peak-to-camera-noise ratios = {self.p2nr[ii]}")

            # frequency
            frq_formatter = {"float": lambda x: f"{x:7.3f}", "separator": ", "}
            if self.frqs_guess is not None:
                angle_guess = np.angle(self.frqs_guess[ii, 0] + 1j * self.frqs_guess[ii, 1])
                period_guess = 1 / np.linalg.norm(self.frqs_guess[ii])
                frq_str = np.array2string(self.frqs_guess[ii], formatter=frq_formatter)
                self.print_log(f"{'Frequency guess':15s} (fx, fy) = {frq_str:s},"
                               f" period = {period_guess*1e3:.3f}nm,"
                               f" angle ={angle_guess*180/np.pi:7.3f}deg")

            if self.frqs is not None:
                angle = np.angle(self.frqs[ii, 0] + 1j * self.frqs[ii, 1])
                period = 1 / np.linalg.norm(self.frqs[ii])
                frq_str = np.array2string(self.frqs[ii], formatter=frq_formatter)
                self.print_log(f"{'Frequency':15s} (fx, fy) = {frq_str:s},"
                               f" period = {period * 1e3:.3f}nm,"
                               f" angle ={angle * 180 / np.pi:7.3f}deg")

            # phase information
            formatter = {"float": lambda x: f"{x:7.2f}", "separator": ", "}

            if self.peak_phases is not None:
                phase_peak_diff_deg = np.mod(self.peak_phases[ii] - self.peak_phases[ii, 0], 2 * np.pi) * 180 / np.pi
                phase_peak_str = np.array2string(phase_peak_diff_deg, formatter=formatter)
                phase_peak_off_deg = np.mod(self.peak_phases[ii, 0], 2*np.pi) * 180/np.pi
                self.print_log(f"{'peak phases':12s} = {phase_peak_str:s}deg, offset={phase_peak_off_deg:7.2f}deg")

            if self.phases_guess is not None:
                phase_guess_diff_deg = np.mod(self.phases_guess[ii] - self.phases_guess[ii, 0], 2*np.pi) * 180/np.pi
                phase_guess_str = np.array2string(phase_guess_diff_deg, formatter=formatter)
                phase_guess_off_deg = np.mod(self.phases_guess[ii, 0], 2*np.pi) * 180/np.pi
                self.print_log(f"{'guess phases':12s} = {phase_guess_str:s}deg, offset={phase_guess_off_deg:7.2f}deg")

            if self.phases is not None:
                phase_diff_deg = np.mod(self.phases[ii] - self.phases[ii, 0], 2*np.pi) * 180 / np.pi
                phase_diff_str = np.array2string(phase_diff_deg, formatter=formatter)
                phase_offset = np.mod(self.phases[ii, 0] - self.phase_corrections[ii], 2*np.pi) * 180 / np.pi
                self.print_log(f"{'final phases':12s} = {phase_diff_str:s}deg, offset={phase_offset:7.2f}deg")

            self.print_log(f"global phase correction = {self.phase_corrections[ii] * 180 / np.pi:7.3f}deg")

    def add_stream(self,
                   stream):
        """
        Add stream to be used with print_log()

        :param stream:
        :return:
        """
        if stream not in self._streams:
            self._streams.append(stream)

    def print_log(self,
                  string: str,
                  **kwargs):
        """
        Print result to stdout and to a log file.

        :param string: string to print
        :param kwargs: passed through to print()
        """
        for stream in self._streams:
            print(string, **kwargs, file=stream)

    # plotting utility functions
    def plot_figs(self,
                  save_dir: str = None,
                  save_prefix: str = "",
                  save_suffix: str = "",
                  slices: Optional[tuple[slice]] = None,
                  figsize: Sequence[float, float] = (20., 10.),
                  diagnostics_only: bool = False,
                  interactive_plotting: bool = False,
                  imgs_dpi: Optional[int] = None) -> (list[Figure], list[str]):

        """
        Automate plotting and saving of figures

        :param save_dir:
        :param save_prefix:
        :param save_suffix:
        :param slices: tuple of slices indicating which image to plot. len(slices) = self.imgs.ndim - 4
        :param figsize:
        :param diagnostics_only:
        :param interactive_plotting:
        :param imgs_dpi: Set to 400 for high resolution, but slower saving
        :return:
        """

        # #############################################
        # plot settings
        # #############################################
        if not interactive_plotting:
            plt.ioff()
            plt.switch_backend("agg")

        # #############################################
        # figures/names to output
        # #############################################
        figs = []
        fig_names = []

        # get slice to display
        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        tstart = perf_counter()

        saving = save_dir is not None
        if saving:
            save_dir = Path(save_dir)

        # plot MCNR diagnostic
        fnow, fnames_now = self.plot_mcnr_diagnostic(slices, figsize=figsize)

        figs += fnow
        fig_names += fnames_now

        figh = fnow[0]
        fname = fnames_now[0]

        if saving:
            figh.savefig(save_dir / f"{save_prefix:s}{fname:s}{save_suffix:s}.png",
                         dpi=imgs_dpi)
        if not interactive_plotting:
            plt.close(figh)

        # plot frequency fits
        if self._recon_settings['frq_estimation_mode'] != "fixed":
            fighs, fig_names_now = self.plot_frequency_fits(figsize=figsize)
            figs += fighs
            fig_names += fig_names_now

            for fh, fn in zip(fighs, fig_names_now):
                if saving:
                    fh.savefig(save_dir / f"{save_prefix:s}{fn:s}{save_suffix:s}.png")
                if not interactive_plotting:
                    plt.close(fh)

        # plot filters used in reconstruction
        fighs, fig_names_now = self.plot_reconstruction_diagnostics(slices, figsize=figsize)

        figs += fighs
        fig_names += fig_names_now

        for fh, fn in zip(fighs, fig_names_now):
            if saving:
                fh.savefig(save_dir / f"{save_prefix:s}{fn:s}{save_suffix:s}.png",
                           dpi=imgs_dpi)
            if not interactive_plotting:
                plt.close(fh)

        # plot otf
        fig = self.plot_otf(figsize=figsize)
        fig_name_now = "otf"

        figs += [fig]
        fig_names += [fig_name_now]

        if saving:
            fig.savefig(save_dir / f"{save_prefix:s}{fig_name_now:s}{save_suffix:s}.png")
        if not interactive_plotting:
            plt.close(fig)

        if not diagnostics_only:
            # plot reconstruction results
            fig = self.plot_reconstruction(slices, figsize=figsize)
            fig_name_now = "sim_reconstruction"

            figs += [fig]
            fig_names += [fig_name_now]

            if saving:
                fig.savefig(save_dir / f"{save_prefix:s}{fig_name_now:s}{save_suffix:s}.png",
                            dpi=imgs_dpi)
            if not interactive_plotting:
                plt.close(fig)

        tend = perf_counter()
        self.print_log(f"plotting results took {tend - tstart:.2f}s")

        return figs, fig_names

    def plot_mcnr_diagnostic(self,
                             slices: Optional[tuple[slice]] = None,
                             figsize: Sequence[float, float] = (20., 10.),
                             **kwargs) -> (list[plt.figure], list[str]):
        """
        Display SIM images for visual inspection. Use this to examine SIM pictures and their Fourier transforms
        as an aid to guessing frequencies before doing reconstruction.

        :param slices: tuple of slices indicating which image to plot. len(slices) = self.imgs.ndim - 4
        :param figsize:
        :return figs, fig_names:
        """

        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        # get slice to plot
        imgs_slice_list = slices + (slice(None),) * 4
        mcnr_slice_list = slices + (slice(None),) * 3

        imgs = self.imgs[imgs_slice_list].squeeze()
        if isinstance(imgs, da.core.Array):
            imgs = imgs.compute()

        if self.mcnr is not None:
            mcnr = self.mcnr[mcnr_slice_list].squeeze()
            if isinstance(mcnr, da.core.Array):
                mcnr = mcnr.compute()

            vmax_mcnr = np.percentile(mcnr, 99)

        extent = [self.x[0] - 0.5 * self.dx, self.x[-1] + 0.5 * self.dx,
                  self.y[-1] + 0.5 * self.dy, self.y[0] - 0.5 * self.dy]

        # parameters for real space plot
        vmin = np.percentile(imgs.ravel(), 0.1)
        vmax = np.percentile(imgs.ravel(), 99.9)

        # to avoid errors with image that has only one value
        if vmax <= vmin:
            vmax += 1e-12

        # ########################################
        # plot real-space
        # ########################################
        figh = plt.figure(figsize=figsize, **kwargs)
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
                ax.imshow(imgs[ii, jj],
                          vmin=vmin,
                          vmax=vmax,
                          extent=extent,
                          interpolation=None,
                          cmap="bone")

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
            if self.mcnr is not None:
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

        if self.mcnr is not None:
            # colorbar for MCNR
            ax = figh.add_subplot(grid[:, n_factor*(self.nphases + 2) + 2])
            norm = PowerNorm(vmin=0, vmax=vmax_mcnr, gamma=1)
            plt.colorbar(ScalarMappable(norm=norm, cmap="inferno"), cax=ax, label="MCNR")

        return [figh], ["mcnr_diagnostic"]

    def plot_reconstruction(self,
                            slices: Optional[tuple[slice]] = None,
                            figsize: Sequence[float, float] = (20., 10.),
                            gamma: float = 0.1,
                            min_percentile: float = 0.1,
                            max_percentile: float = 99.9,
                            **kwargs) -> Figure:
        """
        Plot SIM image and compare with 'widefield' image. Pass additional keyword arguments to
        plt.figure()

        :param slices:
        :param figsize:
        :param gamma:
        :param min_percentile:
        :param max_percentile:
        :param kwargs:
        :return figh:
        """

        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        wf_slice_list = slices + (slice(None),) * 2

        # extents for plots
        extent_wf = [self.fx[0] - 0.5 * self.dfx, self.fx[-1] + 0.5 * self.dfx,
                     self.fy[-1] + 0.5 * self.dfy, self.fy[0] - 0.5 * self.dfy]
        extent_rec = [self.fx_us[0] - 0.5 * self.dfx_us, self.fx_us[-1] + 0.5 * self.dfx_us,
                      self.fy_us[-1] + 0.5 * self.dfy_us, self.fy_us[0] - 0.5 * self.dfy_us]
        extent_wf_real = [self.x[0] - 0.5 * self.dx, self.x[-1] + 0.5 * self.dx,
                          self.y[-1] + 0.5 * self.dy, self.y[0] - 0.5 * self.dy]
        extent_us_real = [self.x_us[0] - 0.5 * self.dx_us, self.x_us[-1] + 0.5 * self.dx_us,
                          self.y_us[-1] + 0.5 * self.dy_us, self.y_us[0] - 0.5 * self.dy_us]

        # create plot
        figh = plt.figure(figsize=figsize, **kwargs)
        grid = figh.add_gridspec(nrows=2, ncols=3)
        figh.suptitle(f"SIM reconstruction\n"
                      f"wiener parameter{self._recon_settings['wiener_parameter']:.2f}, "
                      f"phase estimation mode '{self._recon_settings['phase_estimation_mode']:s}', "
                      f"frq estimation mode '{self._recon_settings['frq_estimation_mode']:s}'\n"
                      f"band combination mode '{self._recon_settings['combine_bands_mode']:s}', "
                      f"band replacement using {self._recon_settings['fmax_exclude_band0']:.2f} of fmax")

        # widefield
        if self.widefield is not None:
            widefield = self.widefield[wf_slice_list].squeeze()
            widefield_ft = ft2(widefield)

            # real space
            ax = figh.add_subplot(grid[0, 0])

            vmin = np.percentile(widefield.ravel(), min_percentile)
            vmax = np.percentile(widefield.ravel(), max_percentile)
            if vmax <= vmin:
                vmax += 1e-12
            ax.imshow(widefield,
                      vmin=vmin,
                      vmax=vmax,
                      cmap="bone",
                      extent=extent_wf_real)
            ax.set_title('widefield')
            ax.set_xlabel('x-position ($\mu m$)')
            ax.set_ylabel('y-position ($\mu m$)')

            # fourier space
            ax = figh.add_subplot(grid[1, 0])
            ax.imshow(np.abs(widefield_ft) ** 2,
                      norm=PowerNorm(gamma=gamma),
                      extent=extent_wf,
                      cmap="bone")

            ax.add_artist(Circle((0, 0),
                                 radius=self.fmax,
                                 color='r',
                                 fill=False,
                                 ls='--'))
            ax.add_artist(Circle((0, 0),
                                 radius=2 * self.fmax,
                                 color='r',
                                 fill=False,
                                 ls='--'))

            ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
            ax.set_ylim([2 * self.fmax, -2 * self.fmax])
            ax.set_xlabel("$f_x (1/\mu m)$")
            ax.set_ylabel("$f_y (1/\mu m)$")

        # deconvolved
        if self.widefield_deconvolution is not None:
            widefield_deconvolution = self.widefield_deconvolution[wf_slice_list].squeeze()
            widefield_deconvolution_ft = ft2(widefield_deconvolution)

            ax = figh.add_subplot(grid[0, 2])

            vmin = np.percentile(widefield_deconvolution.ravel(), min_percentile)
            vmax = np.percentile(widefield_deconvolution.ravel(), max_percentile)
            if vmax <= vmin:
                vmax += 1e-12
            ax.imshow(widefield_deconvolution,
                      vmin=vmin,
                      vmax=vmax,
                      cmap="bone",
                      extent=extent_us_real)
            ax.set_title('widefield deconvolved')
            ax.set_xlabel('x-position ($\mu m$)')

            # deconvolution Fourier space
            ax = figh.add_subplot(grid[1, 2])
            ax.imshow(np.abs(widefield_deconvolution_ft) ** 2,
                      norm=PowerNorm(gamma=gamma),
                      extent=extent_rec,
                      cmap="bone")

            ax.add_artist(Circle((0, 0),
                                 radius=self.fmax,
                                 color='r',
                                 fill=False,
                                 ls='--'))
            ax.add_artist(Circle((0, 0),
                                 radius=2 * self.fmax,
                                 color='r',
                                 fill=False,
                                 ls='--'))

            ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
            ax.set_ylim([2 * self.fmax, -2 * self.fmax])
            ax.set_xlabel("$f_x (1/\mu m)$")

        # SIM
        if self.sim_sr is not None:
            sim_sr = self.sim_sr[slices].squeeze()
            sim_sr_ft = ft2(sim_sr)

            # real-space
            ax = figh.add_subplot(grid[0, 1])
            vmin = np.percentile(sim_sr.ravel()[sim_sr.ravel() >= 0], min_percentile)
            vmax = np.percentile(sim_sr.ravel()[sim_sr.ravel() >= 0], max_percentile)
            if vmax <= vmin:
                vmax += 1e-12
            ax.imshow(sim_sr, vmin=vmin, vmax=vmax, cmap="bone", extent=extent_us_real)
            ax.set_title('SR-SIM')
            ax.set_xlabel('x-position ($\mu m$)')

            # SIM fourier space
            ax = figh.add_subplot(grid[1, 1])
            ax.imshow(np.abs(sim_sr_ft) ** 2,
                      norm=PowerNorm(gamma=gamma),
                      extent=extent_rec,
                      cmap="bone")

            ax.add_artist(Circle((0, 0),
                                 radius=self.fmax,
                                 color='r',
                                 fill=False,
                                 ls='--'))
            ax.add_artist(Circle((0, 0),
                                 radius=2 * self.fmax,
                                 color='r',
                                 fill=False,
                                 ls='--'))

            # actual maximum frequency based on real SIM frequencies
            for ii in range(self.nangles):
                period = 1 / np.linalg.norm(self.frqs[ii], axis=-1)
                ax.add_artist(Circle((0, 0),
                                     radius=self.fmax + 1 / period,
                                     color='g',
                                     fill=False,
                                     ls='--'))

            ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
            ax.set_ylim([2 * self.fmax, -2 * self.fmax])
            ax.set_xlabel("$f_x (1/\mu m)$")

        return figh

    def plot_reconstruction_diagnostics(self,
                                        slices: Optional[tuple[slice]] = None,
                                        figsize: Sequence[float, float] = (20., 10.),
                                        **kwargs) -> (list[Figure], list[str]):
        """
        Diagnostics showing progress of SIM reconstruction
        This function can be called at any point in the reconstruction, and is useful for assessing if the guess
        frequencies are close to the actual peaks in the image, as well as the quality of the reconstruction

        :param slices:
        :param figsize:
        :param kwargs:
        :return figs, fig_names:
        """
        figs = []
        fig_names = []

        if slices is None:
            slices = tuple([slice(n // 2, n // 2 + 1) for n in self.imgs.shape[:-4]])

        slices_raw = slices + (slice(None),) * 4

        imgs_ft = self.imgs_ft[slices_raw].squeeze()
        if isinstance(imgs_ft, da.core.Array):
            imgs_ft = imgs_ft.compute()

        parameters_estimated = self.phases is not None
        reconstructed_already = self.bands_shifted_ft is not None

        if reconstructed_already:
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
        extent = [self.fx[0] - 0.5 * self.dfx,
                  self.fx[-1] + 0.5 * self.dfx,
                  self.fy[-1] + 0.5 * self.dfy,
                  self.fy[0] - 0.5 * self.dfy]
        extent_upsampled = [self.fx_us[0] - 0.5 * self.dfx_us,
                            self.fx_us[-1] + 0.5 * self.dfx_us,
                            self.fy_us[-1] + 0.5 * self.dfy_us,
                            self.fy_us[0] - 0.5 * self.dfy_us]
        extent_upsampled_real = [self.x_us[0] - 0.5 * self.dx_us,
                                 self.x_us[-1] + 0.5 * self.dx_us,
                                 self.y_us[-1] + 0.5 * self.dy_us,
                                 self.y_us[0] - 0.5 * self.dy_us]

        # plot one image for each angle
        for ii in range(self.nangles):
            ttl_str = f"SIM bands diagnostic, angle {ii:d}\n"

            if parameters_estimated:
                dp1 = np.mod(self.phases[ii, 1] - self.phases[ii, 0], 2 * np.pi)
                dp2 = np.mod(self.phases[ii, 2] - self.phases[ii, 0], 2 * np.pi)
                p0_corr = np.mod(self.phases[ii, 0] - self.phase_corrections[ii], 2 * np.pi)

                ttl_str += f'f=({self.frqs[ii, 0]:.3f},{self.frqs[ii, 1]:.3f}) 1/um\n' \
                           f'modulation contrast={self.mod_depths[ii]:.3f}, ' \
                           f'$\eta$={self._recon_settings["wiener_parameter"]:.2f},' \
                           f' phases (deg) = {0 * 180 / np.pi:.2f},' \
                                         f' {dp1 * 180 / np.pi:.2f},' \
                                         f' {dp2 * 180 / np.pi:.2f}; offset = {p0_corr * 180 / np.pi:.2f}deg'

            fig = plt.figure(figsize=figsize, **kwargs)
            fig.suptitle(ttl_str)

            # 6 diagnostic images + 4 extra columns for colorbars
            grid = fig.add_gridspec(nrows=self.nphases,
                                    ncols=8,
                                    width_ratios=[1, 1, 0.2, 0.2] + [1, 0.2, 0.2] + [2],
                                    wspace=0.1)

            vmax = np.max(np.abs(imgs_ft))
            vmin = 1e-5 * vmax

            for jj in range(self.nphases):
                # ####################
                # raw images at different phases
                # ####################
                ax = fig.add_subplot(grid[jj, 0])
                ax.set_title("Raw data, phase %d" % jj)

                to_plot = np.abs(imgs_ft[ii, jj])
                to_plot[to_plot <= 0] = np.nan

                im = ax.imshow(to_plot, norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent, cmap="bone")

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
                    # shifted component
                    # ####################
                    ax = fig.add_subplot(grid[jj, 1])

                    # avoid any zeros for LogNorm()
                    cs_ft_toplot = np.abs(bands_shifted_ft[ii, jj])
                    cs_ft_toplot[cs_ft_toplot <= 0] = np.nan

                    # to keep same color scale, must correct for upsampled normalization change
                    ax.imshow(cs_ft_toplot, norm=LogNorm(vmin=4*vmin, vmax=4*vmax), extent=extent_upsampled, cmap="bone")

                    ax.scatter(0, 0, edgecolor='k', facecolor='none')

                    ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=False, ls='--'))

                    if jj == 0:
                        ax.set_title('O(f)otf(f)')
                        ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                        ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
                    if jj == 1:
                        ax.set_title('m*O(f)otf(f+fo)')
                        ax.scatter(-self.frqs[ii, 0], -self.frqs[ii, 1], edgecolor='k', facecolor='none')
                        ax.add_artist(Circle(-self.frqs[ii], radius=self.fmax, color='m', fill=0, ls='--'))
                    elif jj == 2:
                        ax.set_title('m*O(f)otf(f-fo)')
                        ax.scatter(self.frqs[ii, 0], self.frqs[ii, 1], edgecolor='k', facecolor='none')
                        ax.add_artist(Circle(self.frqs[ii], radius=self.fmax, color='m', fill=0, ls='--'))
                    if jj == (self.nphases - 1):
                        ax.set_xlabel("$f_x$")

                    ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
                    ax.set_ylim([2 * self.fmax, -2 * self.fmax])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # colorbar
                    if jj == 0:
                        # colorbar refers to unshifted components raw values (related to shifted by factor of 4)
                        ax = fig.add_subplot(grid[:, 2])
                        plt.colorbar(im, cax=ax)

                    # ####################
                    # normalized weights
                    # ####################
                    ax = fig.add_subplot(grid[jj, 4])
                    if jj == 0:
                        ax.set_title(r"$\frac{w_i(k)}{\sum_j |w_j(k)|^2 + \eta^2}$")

                    im2 = ax.imshow(np.abs(weights[ii, jj] / weights_norm),
                                    norm=PowerNorm(gamma=0.1, vmin=1e-5, vmax=10),
                                    extent=extent_upsampled,
                                    cmap="bone")

                    ax.add_artist(Circle((0, 0), radius=self.fmax, color='r', fill=0, ls='--'))
                    if jj == 1:
                        ax.add_artist(Circle(-self.frqs[ii], radius=self.fmax, color='m', fill=0, ls='--'))
                    elif jj == 2:
                        ax.add_artist(Circle(self.frqs[ii], radius=self.fmax, color='m', fill=0, ls='--'))

                    if jj == (self.nphases - 1):
                        ax.set_xlabel("$f_x$")

                    ax.set_xlim([-2 * self.fmax, 2 * self.fmax])
                    ax.set_ylim([2 * self.fmax, -2 * self.fmax])

                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # colorbar
                    if jj == 0:
                        ax = fig.add_subplot(grid[:, 5])
                        fig.colorbar(im2, cax=ax, format="%0.2g", ticks=[10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])

            if reconstructed_already:
                # real space bands
                band0 = irft2(sim_sr_ft_components[ii, 0]).real
                band1 = irft2(sim_sr_ft_components[ii, 1] + sim_sr_ft_components[ii, 2]).real

                # combine as dual color
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
                ax = fig.add_subplot(grid[:, 7])
                ax.set_title("band 0 (cyan) and band 1 (magenta)")

                ax.imshow(img0 + img1, extent=extent_upsampled_real)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("y-position ($\mu m$)")

            figs.append(fig)
            fig_names.append(f"band_diagnostic_angle={ii:d}")

        return figs, fig_names

    def plot_frequency_fits(self,
                            figsize: Sequence[float, float] = (20., 10.)) \
            -> (list[Figure], list[str]):
        """
        Plot frequency fits

        :param figsize:
        :return figs, fig_names: list of figure handles and names
        """
        figs = []
        fig_names = []

        if self.frqs_guess is None:
            frqs_guess = [None] * self.nangles
        else:
            frqs_guess = self.frqs_guess

        for ii in range(self.nangles):
            ttl = f"Correlation fit, angle {ii:d}"
            if self.phases_guess is not None:
                ttl += f", unmixing phases = {self.phases_guess[ii]}"

            figh = plot_correlation_fit(self.band0_frq_fit[ii],
                                        self.band1_frq_fit[ii],
                                        self.frqs[ii, :],
                                        self.dx,
                                        self.fmax,
                                        frqs_guess=frqs_guess[ii],
                                        figsize=figsize,
                                        title=ttl,
                                        otf=self.otf[ii])
            figs.append(figh)
            fig_names.append(f"frq_fit_angle={ii:d}")

        return figs, fig_names

    def plot_otf(self,
                 na: Optional[float] = None,
                 wavelength: Optional[float] = None,
                 cmap: str = "brg",
                 figsize: Sequence[float, float] = (20., 10.),
                 **kwargs) -> plt.figure:
        """
        Plot optical transfer function (OTF) versus frequency and show SIM frequencies. Compare with ideal OTF if
        NA and wavelength are provided

        :param na: numerical aperture
        :param wavelength: emission wavelength in nm
        :param cmap:
        :param figsize:
        :return figh:
        """

        otf_vals = np.zeros(self.nangles)

        for ii in range(self.nangles):
            ix = np.argmin(np.abs(self.frqs[ii, 0] - self.fx))
            iy = np.argmin(np.abs(self.frqs[ii, 1] - self.fy))
            otf_vals[ii] = np.abs(self.otf[..., ii, iy, ix])

        otf_at_frqs = otf_vals

        extent_fxy = [self.fx[0] - 0.5 * self.dfx, self.fx[-1] + 0.5 * self.dfx,
                      self.fy[-1] + 0.5 * self.dfy, self.fy[0] - 0.5 * self.dfy]

        figh = plt.figure(figsize=figsize, **kwargs)
        tstr = "OTF diagnostic\nvalue at frqs="
        for ii in range(self.nangles):
            tstr += f" {otf_at_frqs[ii]:.3f},"
        figh.suptitle(tstr)

        ff = np.sqrt(np.expand_dims(self.fx, axis=0) ** 2 +
                     np.expand_dims(self.fy, axis=1) ** 2)

        # max freq to display
        fmax_disp = 1.2 * max([np.max(np.linalg.norm(self.frqs, axis=-1)), self.fmax])
        # colors fo
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, self.nangles))

        # 1D plots
        ax = figh.add_subplot(1, 2, 1)
        ax.set_title("1D OTF")
        ax.set_xlabel("Frequency (1/um)")
        ax.set_ylabel("OTF")
        # plot real OTF's per angle
        for ii in range(self.nangles):
            ax.plot(ff.ravel(),
                    self.otf[ii].ravel(),
                    c=colors[ii],
                    label=f"OTF, angle {ii:d}")

        if na is not None and wavelength is not None:
            otf_ideal = circ_aperture_otf(ff, 0, na, wavelength)
            ax.plot(ff.ravel(),
                    otf_ideal.ravel(),
                    label="OTF ideal")

        ax.axvline(self.fmax,
                   c="k",
                   label="fmax")

        # plot SIM frequencies
        fs = np.linalg.norm(self.frqs, axis=1)
        for ii in range(self.nangles):
            ax.axvline(fs[ii], c=colors[ii])

        ax.set_xlim([0, fmax_disp])
        ax.legend()

        # 2D plot
        ax = figh.add_subplot(1, 2, 2)
        ax.set_title("Mean 2D OTF")
        ax.imshow(np.mean(np.abs(self.otf), axis=0),
                  extent=extent_fxy,
                  cmap="bone")
        ax.scatter(self.frqs[:, 0],
                   self.frqs[:, 1],
                   color=colors,
                   marker='o')
        ax.add_artist(Circle((0, 0), self.fmax, facecolor='none', edgecolor='y'))
        ax.set_xlabel("$f_x (1/\mu m)$")
        ax.set_ylabel("$f_y (1/\mu m)$")
        ax.set_xlim([-fmax_disp, fmax_disp])
        ax.set_ylim([fmax_disp, -fmax_disp])

        return figh

    # saving utility functions
    def save_imgs(self,
                  save_dir: Union[str, Path],
                  save_suffix: str = "",
                  save_prefix: str = "",
                  format: str = "tiff",
                  save_patterns: bool = False,
                  save_raw_data: bool = False,
                  save_processed_data: bool = False,
                  attributes: Optional[dict] = None,
                  arrays: Optional[dict] = None,
                  compressor: Optional = None) -> Path:
        """
        Save SIM results and metadata to file

        :param save_dir: directory to save results
        :param save_suffix:
        :param save_prefix:
        :param format: "tiff", "zarr" or "hdf5". If tiff is used, metadata will be saved in a .json file
        :param save_patterns: save estimated patterns
        :param save_raw_data: save raw image data
        :param save_processed_data: save processed image data
        :param attributes: dictionary passing extra attributes which will be saved with SIM data. This data
          must be json serializable
        :param arrays: dictionary whose entries are arrays which will be stored in file
        :param compressor: if using zarr format, optionally compress results. For example, numcodecs.Zlib()
          is a lossless compressor which is often a good choice
        :return metadata_fname: when saving as tiff, this will be an auxilliary json file. For zarr and hdf5,
          it will be the zarr or hdf5 file
        """

        if attributes is None:
            attributes = {}

        # #############################################
        #  handle save_dir
        # #############################################
        tstart_save = perf_counter()

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # ###############################n
        # metadata
        # ###############################
        metadata = {"log": self.log.getvalue(),
                    "dx": self.dx,
                    "dy": self.dy,
                    "upsample_factor": self.upsample_fact,
                    "na": self.na,
                    "wavelength": self.wavelength,
                    "fmax": self.fmax,
                    "band_inds": self.band_inds.tolist(),
                    "nbands": self.nbands,
                    "reconstruction_settings": self._recon_settings,
                    "preprocessing_settings": self._preprocessing_settings,
                    "axes_names": self.axes_names
                    }

        metadata.update(attributes)

        # ###############################n
        # select attributes to save
        # ###############################n
        attrs = ["sim_sr", "widefield", "widefield_deconvolution", "mcnr", "sim_os"]

        if save_raw_data:
            attrs += ["imgs_raw"]

        if save_processed_data:
            attrs += ["imgs"]

        if save_patterns:
            attrs += ["patterns", "patterns_2x"]

        attrs = [a for a in attrs if getattr(self, a) is not None]

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
        if format == "zarr":
            fname = save_dir / f"{save_prefix:s}sim_results{save_suffix:s}.zarr"

            img_z = zarr.open(fname, mode="w")

            for k, v in metadata.items():
                img_z.attrs[k] = v

            # save additional arrays
            if arrays is not None:
                for k, v in arrays.items():
                    if k in attrs:
                        raise ValueError(f"extra array attribute {k:s} had same name as SIM attribute")

                    img_z.array(k, v, compressor=compressor, dtype=v.dtype)

            # save reconstruction later
            def save_delayed(attr):
                d = getattr(self, attr).to_zarr(fname,
                                                component=attr,
                                                compute=False,
                                                compressor=compressor)
                return d

        elif format == "hdf5":
            fname = save_dir / f"{save_prefix:s}sim_results{save_suffix:s}.hd5f"
            img_z = h5py.File(fname, "w")

            for k, v in metadata.items():
                if isinstance(v, dict):
                    # flatten dictionaries
                    for kk, vv in v.items():
                        img_z.attrs[kk] = vv
                else:
                    img_z.attrs[k] = v

            def save_delayed(attr):
                d = getattr(self, attr).to_hdf5(fname, f"/{attr:s}")
                return d

        elif format == "tiff":
            if arrays is not None:
                raise NotImplementedError()

            # save metadata to json file
            fname = save_dir / f"{save_prefix:s}sim_reconstruction{save_suffix:s}.json"
            with open(fname, "w") as f:
                json.dump(metadata, f, indent="\t")

            def save_delayed(attr):
                def _save():
                    img = np.array(getattr(self, attr).astype(np.float32))

                    if img.shape[0] == self.nx:
                        factor = 1
                    else:
                        factor = self.upsample_fact

                    # if imagej_axes_order:
                    #     use_imagej = True
                    #     img = tifffile.transpose_axes(img, axes=imagej_axes_order, asaxes="TZCYXS")

                    tifffile.imwrite(save_dir / f"{save_prefix:s}{attr:s}{save_suffix:s}.tif",
                                     img,
                                     imagej=False,
                                     resolution=(1 / self.dy / factor, 1 / self.dx / factor),
                                     metadata={"Info": f"array type = {attr:s}, axes names = {', '.join(self.axes_names):s}",
                                               "unit": "um",
                                               'min': 0,
                                               'max': float(np.max(img))
                                               }
                                     )

                return delayed(_save)()
        else:
            raise ValueError(f"format was {format:s}, but the allowed values are {['tiff', 'zarr', 'hd5f']}")

        # ###############################
        # save results
        # ###############################
        future = [save_delayed(a) for a in attrs]

        self.print_log("saving images...")
        with ProgressBar():
            compute(future)
        self.print_log(f"saving SIM images took {perf_counter() - tstart_save:.2f}s")

        return fname


def show_sim_napari(fname_zarr: Union[str, Path],
                    block: bool = True,
                    load: bool = True,
                    viewer = None,
                    clims: Sequence[float] = (0., 5000.),
                    use_um: bool = True):
    """
    Plot all images obtained from SIM reconstruction with correct scale/offset

    :param fname_zarr: file path for data saved in zarr format
    :param block: block program while displaying
    :param load: load data (vs lazy load)
    :param viewer: if viewer is supplied, plot results on this viewer
    :param clims: display limits for SR-SIM data
    :param use_um: display using real units (vs pixels)
    :return viewer:
    """

    import napari

    imgz = zarr.open(fname_zarr, "r")
    wf = imgz.widefield

    if use_um:
        dxy = imgz.attrs["dx"]
    else:
        dxy = 1

    dxy_sim = dxy / imgz.attrs["upsample_factor"]

    # translate to put FFT zero coordinates at origin
    # translate_wf = (-(wf.shape[-2] // 2) * dxy, -(wf.shape[-1] // 2) * dxy)
    translate_wf = (0, 0)
    translate_sim = (-((2 * wf.shape[-2]) // 2) * dxy_sim + (wf.shape[-2] // 2) * dxy,
                     -((2 * wf.shape[-1]) // 2) * dxy_sim + (wf.shape[-2] // 2) * dxy)
    translate_pattern_2x = [a - 0.25 * dxy for a in translate_wf]

    if viewer is None:
        viewer = napari.Viewer()

    if hasattr(imgz, "patterns"):
        if load:
            p = np.array(imgz.patterns)
        else:
            p = imgz.patterns

        viewer.add_image(p,
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         contrast_limits=[0, 20],
                         colormap="red",
                         name="patterns")

    if hasattr(imgz, "patterns_2x"):
        if load:
            p2 = np.array(imgz.patterns_2x)
        else:
            p2 = imgz.patterns_2x

        viewer.add_image(p2,
                         scale=(dxy_sim, dxy_sim),
                         translate=translate_pattern_2x,
                         contrast_limits=[0, 5],
                         colormap="red",
                         name="patterns upsampled")

    if hasattr(imgz, "sim_os"):
        sim_os = np.expand_dims(imgz.sim_os, axis=-3)

        viewer.add_image(sim_os,
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         name="SIM-OS")

    if hasattr(imgz, "deconvolved"):
        decon = np.expand_dims(imgz.deconvolved, axis=-3)
        viewer.add_image(decon,
                         scale=(dxy_sim, dxy_sim),
                         translate=translate_sim,
                         name="wf deconvolved",
                         visible=False)

    if hasattr(imgz, "sim_fista_forward_model"):
        if load:
            fwd = np.array(imgz.sim_fista_forward_model)
        else:
            fwd = imgz.sim_fista_forward_model

        viewer.add_image(fwd,
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         name="FISTA forward model")

    if hasattr(imgz, "sim_sr_fista"):
        sim_sr_fista = np.expand_dims(imgz.sim_sr_fista, axis=-3)
        viewer.add_image(sim_sr_fista,
                         scale=(dxy_sim, dxy_sim),
                         translate=translate_pattern_2x,
                         name="SIM-SR FISTA")

    if hasattr(imgz, "imgs"):
        shape = imgz.imgs.shape[:-4] + (9,) + imgz.imgs.shape[-2:]
        imgs = np.reshape(np.array(imgz.imgs), shape)

        viewer.add_image(imgs,
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         name="processed images")

    if hasattr(imgz, "imgs_raw"):
        shape = imgz.imgs_raw.shape[:-4] + (9,) + imgz.imgs_raw.shape[-2:]
        imgs_raw = np.reshape(np.array(imgz.imgs_raw), shape)

        viewer.add_image(imgs_raw,
                         scale=(dxy, dxy),
                         translate=translate_wf,
                         name="raw images")

    wf = np.expand_dims(wf, axis=-3)
    viewer.add_image(wf,
                     scale=(dxy, dxy),
                     translate=translate_wf,
                     name="widefield")

    if hasattr(imgz, "sim_sr"):
        sim_sr = np.expand_dims(imgz.sim_sr, axis=-3)
        viewer.add_image(sim_sr,
                         scale=(dxy_sim, dxy_sim),
                         translate=translate_sim,
                         name="SIM-SR",
                         contrast_limits=clims)

    # set to first position
    viewer.dims.set_current_step(axis=0, value=0)
    viewer.show(block=block)

    return viewer


# compute optical sectioned SIM image
def sim_optical_section(imgs: array,
                        axis: int = 0,
                        phase_differences: Sequence[float] = (0, 2*np.pi/3, 4*np.pi/3)) -> array:
    """
    Optical sectioning reconstruction for three SIM images with arbitrary relative phase
    differences following the approach of https://doi.org/10.1016/s0030-4018(98)00210-7

    In the most common case, where the phase differences are (0, 2*pi/3, 4*pi/3) the result is

    .. math::

      I_\\text{os}(r) &=  \\sqrt{ (I_0(r) - I_1(r))^2 + (I_1(r) - I_2(r))^2 + (I_2(r) - I_0(r))^2 }

                      &= \\frac{3}{\\sqrt{2}} mA

      I_a(r) &= A \\left[1 + m \\cos(\\phi + \\phi_a) \\right]

    :param imgs: images stored as nD array, where one of the dimensions is of size 3.
    :param axis: axis to perform the optical sectioning computation along. imgs.shape[axis] must = 3
    :param phase_differences: three phases
    :return img_os: optically sectioned image
    """

    if cp and isinstance(imgs, cp.ndarray):
        xp = cp
    else:
        xp = np

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

    # list of slice tuples, where each slice tuple selects a single phase
    slices = [tuple([slice(None) if ii != axis else slice(jj, jj + 1) for ii in range(imgs.ndim)]) for jj in range(3)]

    i_c = 0
    i_s = 0
    for ii in range(3):
        i_c += inv[1, ii] * imgs[slices[ii]]
        i_s += inv[2, ii] * imgs[slices[ii]]
    img_os = xp.squeeze(xp.sqrt(i_c**2 + i_s**2), axis=axis)

    return img_os


def correct_modulation_for_bead_size(bead_radii: float,
                                     frqs: float,
                                     phis: Sequence[float] = (0, 2 * np.pi / 3, 4 * np.pi / 3)):
    """
    Function for use when calibration SIM modulation depth using fluorescent beads. Assuming the beads are much smaller
    than the lattice spacing, then using the optical sectioning law of cosines type formula on the total fluorescent
    amplitude from a single isolated beads provides an estimate of the modulation depth.

    When the bead is a significant fraction of the lattice period, this modifies the modulation period. This function
    computes the correction for finite bead size.

    :param bead_radii: radius of the bead
    :param frqs: magnitude of the frequency of the lattice (in compatible units with bead_radii)
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
def fit_modulation_frq(mft1: np.ndarray,
                       mft2: np.ndarray,
                       dxy: float,
                       mask: Optional[np.ndarray] = None,
                       frq_guess: Optional[Sequence[float]] = None,
                       max_frq_shift: float = np.inf,
                       fbounds: Sequence[float] = (0., np.inf),
                       otf: Optional[np.ndarray] = None,
                       wiener_param: float = 0.3,
                       keep_guess_if_better: bool = True) -> (np.ndarray, np.ndarray, dict):
    """
    Find SIM frequency from image by maximizing the cross correlation between ft1 and ft2

    .. math::

       C(f') &= \\sum_f ft_1(f)  ft_2^*(f + f')

       f^\\star &= \\text{argmax}_{f'} |C(f')|

    Note that there is ambiguity in the definition of this frequency, as -f will also be a peak. If frq_guess is
    provided, the peak closest to the guess will be returned.

    :param mft1: 2D Fourier space image
    :param mft2: 2D Fourier space image to be cross correlated with ft1
    :param dxy: pixel size. Units of dxy and max_frq_shift must be consistent
    :param mask: boolean array same size as ft1 and ft2. Only consider frequency points where mask is True
    :param frq_guess: frequency guess [fx, fy]. If frequency guess is None, an initial guess will be chosen by
       finding the maximum f_guess = argmax_f CC[ft1, ft2](f), where CC[ft1, ft2] is the discrete cross-correlation
       Currently roi_pix_size is only used internally to set max_frq_shift
    :param max_frq_shift: maximum frequency shift to consider vis-a-vis the guess frequency
    :param fbounds: (min frq, max frq) bound search to
    :param otf:
    :param wiener_param:
    :param keep_guess_if_better: keep the initial frequency guess if the cost function is more optimal
       at this point than after fitting
    :return fit_frqs, mask, fit_result:

    """

    if mft1.shape != mft2.shape:
        raise ValueError("must have ft1.shape = ft2.shape")

    # must be on CPU for this function to work
    mft1 = to_cpu(mft1)
    mft2 = to_cpu(mft2)
    if otf is not None:
        otf = to_cpu(otf)

    # mask
    if mask is None:
        mask = np.ones(mft1.shape, dtype=bool)
    else:
        mask = np.array(mask, copy=True)

    if mask.shape != mft1.shape:
        raise ValueError("mask must have same shape as ft1")

    # otf
    if otf is None:
        otf = 1.
        wiener_param = 0.

    # get frequency data
    fxs = fftshift(fftfreq(mft1.shape[1], dxy))
    fys = fftshift(fftfreq(mft1.shape[0], dxy))
    fxfx, fyfy = np.meshgrid(fxs, fys)

    # ############################
    # set initial guess using cross-correlation
    # ############################
    otf_factor = np.conj(otf) / (np.abs(otf) ** 2 + wiener_param ** 2)

    if frq_guess is None:
        # cross correlation of Fourier transforms
        # WARNING: correlate2d uses a different convention for the frequencies of the output, which will not agree with the fft convention
        # cc(k) = \sum_f ft2^*(f) x ft1(f + k)

        cc = np.abs(correlate(mft2 * otf_factor,
                              mft1 * otf_factor,
                              mode='same'))

        otf_cc = np.abs(correlate(otf_factor,
                                  otf_factor,
                                  mode='same'))

        if fbounds[0] > 0:
            mask[np.sqrt(fxfx ** 2 + fyfy ** 2) < fbounds[0]] = False

        if fbounds[1] < np.inf:
            mask[np.sqrt(fxfx ** 2 + fyfy ** 2) > fbounds[1]] = False

        # get initial frq_guess by looking at cc at discrete frequency set and finding max
        with np.errstate(divide="ignore"):
            to_max = cc / otf_cc
        to_max[np.logical_not(mask)] = 0.
        subscript = np.unravel_index(np.argmax(to_max), cc.shape)

        init_params = np.array([fxfx[subscript], fyfy[subscript]])
    else:
        init_params = frq_guess

    # ############################
    # define cross-correlation and minimization objective function
    # ############################
    # real-space coordinates
    ny, nx = mft1.shape
    x = ifftshift(dxy * (np.arange(nx) - (nx // 2)))
    y = ifftshift(dxy * (np.arange(ny) - (ny // 2)))
    xx, yy = np.meshgrid(x, y)

    img2 = ift2(mft2)

    # compute ft2(f + fo)
    def fft_shifted(f): return fftshift(fft2(np.exp(-1j*2*np.pi * (f[0] * xx + f[1] * yy)) * ifftshift(img2 * otf_factor)))

    # cross correlation
    # todo: conjugating ft2 instead of ft1, as in typical definition of cross correlation. Doesn't matter bc taking norm
    def cc_fn(f): return np.sum(mft1 * otf_factor * fft_shifted(f).conj())
    fft_norm = np.sum(np.abs(mft1 * otf_factor) * np.abs(mft2 * otf_factor))**2
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
                         frqs: np.ndarray,
                         dxy: float,
                         fmax: Optional[float] = None,
                         frqs_guess: Optional[Sequence[float]] = None,
                         roi_size: Sequence[int, int] = (31, 31),
                         peak_pixels: int = 2,
                         figsize=(20, 10),
                         title: str = "",
                         gamma: float = 0.1,
                         cmap: str = "bone",
                         otf: Optional[np.ndarray] = None,
                         wiener_param: float = 0.3,
                         ) -> Figure:
    """
    Display SIM parameter fitting results visually, in a way that is easy to inspect.
    Use this to plot the results of SIM frequency determination after running get_sim_frq()

    :param img1_ft:
    :param img2_ft:
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
    :param otf:
    :param wiener_param:
    :return figh: handle to figure produced
    """
    # get frequency data
    fxs = fftshift(fftfreq(img1_ft.shape[1], dxy))
    dfx = fxs[1] - fxs[0]
    fys = fftshift(fftfreq(img1_ft.shape[0], dxy))
    dfy = fys[1] - fys[0]

    extent = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx,
              fys[-1] + 0.5 * dfy, fys[0] - 0.5 * dfy]

    if otf is None:
        otf = 1.
        wiener_param = 0.

    # power spectrum / cross correlation
    # cc = np.abs(scipy.signal.fftconvolve(img1_ft, img2_ft.conj(), mode='same'))
    # cc = np.abs(correlate(img2_ft, img1_ft, mode='same'))

    otf_factor = np.conj(otf) / (np.abs(otf) ** 2 + wiener_param ** 2)

    cc = np.abs(correlate(img2_ft * otf_factor,
                          img1_ft * otf_factor,
                          mode='same')) / \
         np.abs(correlate(otf_factor,
                          otf_factor,
                          mode='same'))

    # compute peak values
    fx_sim, fy_sim = frqs
    try:
        peak_cc = get_peak_value(cc, fxs, fys, [fx_sim, fy_sim], peak_pixels)
        peak1_dc = get_peak_value(img1_ft, fxs, fys, [0, 0], peak_pixels)
        peak2 = get_peak_value(img2_ft, fxs, fys, [fx_sim, fy_sim], peak_pixels)
    except ZeroDivisionError:
        peak_cc = np.nan
        peak1_dc = np.nan
        peak2 = np.nan

    # create figure
    figh = plt.figure(figsize=figsize)
    gspec = figh.add_gridspec(ncols=4,
                              width_ratios=[8, 1] * 2,
                              wspace=0.5,
                              nrows=2,
                              hspace=0.3)

    # #######################################
    # build title
    # #######################################
    ttl_str = ""
    if title != "":
        ttl_str += f"{title:s}\n"

    # print info about fit frequency
    period = 1 / np.sqrt(fx_sim ** 2 + fy_sim ** 2)
    angle = np.angle(fx_sim + 1j * fy_sim)

    ttl_str += f"      fit: period {period * 1e3:.1f}nm = 1/{1/period:.3f}um at" \
           f" {angle * 180/np.pi:.2f}deg={angle:.3f}rad;" \
           f" f=({fx_sim:.3f},{fy_sim:.3f}) 1/um," \
           f" peak cc={np.abs(peak_cc):.3g} and {np.angle(peak_cc) * 180/np.pi:.2f}deg"

    # print info about guess frequency
    if frqs_guess is not None:
        fx_g, fy_g = frqs_guess
        period_g = 1 / np.sqrt(fx_g ** 2 + fy_g ** 2)
        angle_g = np.angle(fx_g + 1j * fy_g)
        peak_cc_g = get_peak_value(cc, fxs, fys, frqs_guess, peak_pixels)

        ttl_str += f"\nguess: period {period_g * 1e3:.1f}nm = 1/{1/period_g:.3f}um" \
               f" at {angle_g * 180/np.pi:.2f}deg={angle_g:.3f}rad;" \
               f" f=({fx_g:.3f},{fy_g:.3f}) 1/um," \
               f" peak cc={np.abs(peak_cc_g):.3g} and {np.angle(peak_cc_g) * 180/np.pi:.2f}deg"

    figh.suptitle(ttl_str)

    # #######################################
    # plot cross-correlation region of interest
    # #######################################
    roi_cx = np.argmin(np.abs(fx_sim - fxs))
    roi_cy = np.argmin(np.abs(fy_sim - fys))
    roi = get_centered_rois([roi_cy, roi_cx],
                            roi_size,
                            min_vals=[0, 0],
                            max_vals=cc.shape)[0]

    fys_roi = fys[roi[0]:roi[1]]
    fxs_roi = fxs[roi[2]:roi[3]]
    extent_roi = [fxs_roi[0] - 0.5 * dfx, fxs_roi[-1] + 0.5 * dfx,
                  fys_roi[-1] + 0.5 * dfy, fys_roi[0] - 0.5 * dfy]

    ax = figh.add_subplot(gspec[0, 0])
    ax.set_title("cross correlation, ROI")
    im1 = ax.imshow(cut_roi(roi, cc)[0],
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
    roi_center = get_centered_rois([cy_c, cx_c], [roi[1] - roi[0], roi[3] - roi[2]], [0, 0], img1_ft.shape)[0]

    fys_roic = fys[roi_center[0]:roi_center[1]]
    fxs_roic = fxs[roi_center[2]:roi_center[3]]
    extent_roic = [fxs_roic[0] - 0.5 * dfx, fxs_roic[-1] + 0.5 * dfx,
                   fys_roic[-1] + 0.5 * dfy, fys_roic[0] - 0.5 * dfy]

    im3 = ax3.imshow(cut_roi(roi_center, np.abs(img1_ft)**2)[0],
                     interpolation=None,
                     norm=PowerNorm(gamma=gamma),
                     extent=extent_roic,
                     cmap=cmap)
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
        peak2_g = get_peak_value(img2_ft, fxs, fys, frqs_guess, peak_pixels)
        title += "\nguess peak = %0.3g and %0.2fdeg" % (np.abs(peak2_g), np.angle(peak2_g) * 180 / np.pi)
    ax4.set_title(title)
    ax4.set_xlabel('$f_x (1/\mu m)$')

    im4 = ax4.imshow(cut_roi(roi, np.abs(img2_ft)**2)[0],
                     interpolation=None,
                     norm=PowerNorm(gamma=gamma),
                     extent=extent_roi,
                     cmap=cmap)
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
                 sim_frq: array,
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
    fx = fftshift(fftfreq(nx, dxy))
    fy = fftshift(fftfreq(ny, dxy))

    phase = np.mod(np.angle(get_peak_value(img_ft, fx, fy, sim_frq, peak_pixel_size=peak_pixel_size)), 2*np.pi)

    return phase


def get_phase_realspace(img: np.ndarray,
                        sim_frq: np.ndarray,
                        dxy: float,
                        phase_guess: float = 0,
                        use_fft_origin: bool = True) -> float:
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
    :param use_fft_origin: whether to use fft origin or edge
    :return phase_fit: fitted value for the phase
    """
    if np.any(img < 0):
        raise ValueError('img must be strictly positive.')

    ny, nx = img.shape

    if use_fft_origin:
        x = (np.arange(nx) - (nx // 2)) * dxy
        y = (np.arange(ny) - (ny // 2)) * dxy
    else:
        x = np.arange(nx) * dxy
        y = np.arange(ny) * dxy

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
                               phases_guess: Optional[Sequence[float]] = None,
                               fit_amps: bool = True,
                               debug: bool = False) -> (np.ndarray, np.ndarray, dict):
    """
    Estimate relative phases between components using the iterative cross-correlation minimization method of Wicker,
    described in detail here https://doi.org/10.1364/OE.21.002032. This function is hard coded for 3 bands. This
    method is not sensitive to the absolute phase, only the relative phases.

    The optimal band unmixing matrix :math:`M`, which we parameterize by the SIM phase shifts, is given by solving
    the following optimization problem

    .. math::

      C_m(k) &= O(k - m*ko) h_m(k)

      D(k) &= M C(k)

      cc^l_ij &= C_i(k) \\otimes C_j(k-lp)

      M^* &= \\text{argmin}_M \\sum_{i \\neq l+j} |cc^l_ij|^{1/2}.

    This function minimizes the cross correlation between SIM bands which should not contain common information.
    Note that the shifted bands should have low overlap for :math:`i \\neq j + l`.
    The cost function can be rewritten in terms of the correlations of data matrix, :math: `dc^l_{ij}` in a way
    that minimizes numerical effort to recompute for different parameters.

    :param imgs_ft: array of size nphases x ny x nx, where the components are o(f), o(f-fo), o(f+fo)
    :param otf: size ny x nx
    :param sim_frq: np.array([fx, fy])
    :param float dxy: pixel size in um
    :param float fmax: maximum spatial frequency where otf has support
    :param phases_guess: [phi1, phi2, phi3] in radians. If None will use [0, 2*pi/3, 4*pi/3]
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
    fx = fftshift(fftfreq(nx, dxy))
    dfx = fx[1] - fx[0]
    fy = fftshift(fftfreq(ny, dxy))
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
        otf_shift, _ = translate_pix(otf,
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
                band_shifted = translate_ft(imgs_ft[jj], -ml * sim_frq[0], -ml * sim_frq[1], drs=(dxy, dxy))

                # compute weighted cross correlation
                d_cc[ii, jj, ll] = np.sum(imgs_ft[ii] * band_shifted.conj() * weight) / np.sum(weight)

                # remove extra noise correlation expected from same images
                if ml == 0 and ii == jj:
                    noise_power = get_noise_power(imgs_ft[ii], (dxy, dxy), fmax)
                    d_cc[ii, jj, ll] = d_cc[ii, jj, ll] - noise_power

                if debug:
                    extentf = [fx[0] - 0.5 * dfx, fx[-1] + 0.5 * dfx,
                               fy[-1] + 0.5 * dfy, fy[0] - 0.5 * dfy]
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


def get_noise_power(img_ft: array,
                    drs: Sequence[float, float],
                    fmax: float) -> array:
    """
    Estimate average noise power of an image by looking at frequencies beyond the maximum frequency
    where the OTF has support.If an nD image array is passed, compute this over the last two dimensions

    :param img_ft: Size n0 x n1 x ... x ny x nx. Fourier transform of image.
    :param drs: pixel size (dy, dx)
    :param fmax: maximum frequency where signal may be present
    :return noise_power:
    """
    if cp and isinstance(img_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    dy, dx = drs
    ny, nx = img_ft.shape[-2:]
    fxs = xp.fft.fftshift(xp.fft.fftfreq(nx, dx))
    fys = xp.fft.fftshift(xp.fft.fftfreq(ny, dy))
    fxfx, fyfy = xp.meshgrid(xp.asarray(fxs), xp.asarray(fys))
    ff = np.sqrt(fxfx ** 2 + fyfy ** 2)

    noise_power = xp.mean(xp.abs(img_ft[..., ff > fmax])**2, axis=-1)

    return noise_power


# SIM band manipulation functions
def get_band_mixing_matrix(phases: Sequence[float],
                           mod_depth: float = 1.,
                           amps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return matrix M, which relates the measured images to the object profile filtered by the OTF

    .. math::

       \\begin{pmatrix}
         D_1(f)\\\\
         D_2(f)\\\\
         D_3(f)
       \\end{pmatrix}
       = M
       \\begin{pmatrix}
       O(f)H(f)\\\\
       O(f-p)H(f)\\\\
       S(f+p)H(f)
       \\end{pmatrix}

    We assume the modulation has the form :math:`1 + m \\cos(2 \\pi f \\cdot r + \\phi)`, leading to

    .. math::

      M =
      \\begin{pmatrix}
      A_1 & \\frac{1}{2} A_1 m e^{i \\phi_1} & \\frac{1}{2} A_1 m e^{-i \\phi_1}\\\\
      A_2 & \\frac{1}{2} A_2 m e^{i \\phi_2} & \\frac{1}{2} A_2 m e^{-i \\phi_2}\\\\
      A_3 & \\frac{1}{2} A_3 m e^{i \\phi_3} & \\frac{1}{2} A_3 m e^{-i \\phi_3}
      \\end{pmatrix}

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


def get_band_mixing_matrix_jac(phases: Sequence[float, float, float],
                               mod_depth: float,
                               amps: Sequence[float, float, float]) -> list[np.ndarray]:
    """
    Get jacobian of band mixing matrix in parameters [p1, p2, p3, a1, a2, a3, m]

    :param phases:
    :param mod_depth:
    :param amps:
    :return jac:
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

    :param phases:
    :param mod_depth:
    :param amps:
    :return mixing_mat_inv:
    """

    mixing_mat = get_band_mixing_matrix(phases, mod_depth, amps)

    if len(phases) == 3:
        # direct inversion
        try:
            mixing_mat_inv = np.linalg.inv(mixing_mat)
        except np.linalg.LinAlgError:
            warn("warning: band inversion matrix singular")
            mixing_mat_inv = np.zeros((mixing_mat.shape[1], mixing_mat.shape[0])) * np.nan
    else:
        # pseudo-inverse
        mixing_mat_inv = np.linalg.pinv(mixing_mat)

    return mixing_mat_inv


def unmix_bands(imgs_ft: array,
                phases: np.ndarray,
                mod_depths: Optional[np.ndarray] = None,
                amps: Optional[np.ndarray] = None) -> array:
    """
    Do noisy inversion of SIM data, i.e. determine

    .. math::

       \\begin{pmatrix}
         O(f)H(f)\\\\
         O(f-p)H(f)\\\\
         O(f+p)H(f)
       \\end{pmatrix}
        = M^{-1}
       \\begin{pmatrix}
         D_1(f)\\\\
         D_2(f)\\\\
         D_3(f)
       \\end{pmatrix}

    :param imgs_ft: Fourier transform of SIM image data :math:`D_i(f)` as an array of
      size n0 x ... x nm x nangles x nphases x ny x nx.
      DC frequency information should be shifted to the center of the array i.e. as obtained from fftshift
    :param phases: nangles x nphases listing phases for each data image :math:`D_i`
    :param mod_depths: modulation depths for each SIM angle. Optional. If not provided, all are set to 1.
    :param amps: Amplitudes for the SIM pattern in each image, nangles x nphases. If not provided, all are set to 1.
    :return components_ft: unmixed bands of size nangles x nbands x ny x nx array, where the first index corresponds to the bands
    """
    # todo: could generalize for case with more than 3 phases or angles

    if cp and isinstance(imgs_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    # ensure images cupy array if doing on GPU
    imgs_ft = xp.asarray(imgs_ft)

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
        mixing_mat_inv = xp.asarray(get_band_mixing_inv(phases[ii], mod_depths[ii], amps[ii]))

        # todo: plenty of ways to write this more generally ... but for now this is fine
        for jj in range(nphases):
            bands_ft[..., ii, jj, :, :] = mixing_mat_inv[jj, 0] * imgs_ft[..., ii, 0, :, :] + \
                                          mixing_mat_inv[jj, 1] * imgs_ft[..., ii, 1, :, :] + \
                                          mixing_mat_inv[jj, 2] * imgs_ft[..., ii, 2, :, :]

    return bands_ft


def shift_bands(bands_unmixed_ft: array,
                frq_shifts: array,
                drs: Sequence[float, float],
                upsample_factor: int) -> array:
    """
    Shift separated SIM bands to correct locations in Fourier space

    :param bands_unmixed_ft: n0 x ... x nm x 3 x ny x nx
    :param frq_shifts: n0 x ... x nm x 2
    :param drs: (dy, dx)
    :param upsample_factor:
    :return shifted_bands_ft:
    """

    use_gpu = cp and isinstance(bands_unmixed_ft, cp.ndarray)

    if use_gpu:
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
    b1 = translate_ft(expanded[..., 1, :, :],
                      np.expand_dims(frq_shifts[:, 0], axis=(-1, -2)),
                      np.expand_dims(frq_shifts[:, 1], axis=(-1, -2)),
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

    .. math::

      C &= \\frac{\\sum_f b_0(f) b_1^*(f + f_o)}{\\sum |b_0(f)|^2}

      b_1(f + f_o) &= O(f)

    If correct reconstruction parameters are used, we expect the bands differ only by a complex constant over
    any areas where they are not noise corrupted and both the OTF and the shifted OTF have support
    This constant contains information about the global phase offset AND the modulation depth. i.e.

    .. math::

       b_1(f) = m  e^{-i \\phi_c} b_0(f)

    Given this information, can perform the phase correction

    .. math::

      b_1(f + f_o) \\to \\frac{e^{i \\phi_c}}{m} b_1(f + f_o)

    This function return :math:`m e^{i \\phi_c}`


    :param band0: n0 x ... x nm x nangles x ny x nx. Typically, band0(f) = O(f) * otf(f) * wiener(f) ~ O(f)
    :param band1: same shape as band0. Typically, band1(f) = O((f-fo) + fo) * otf(f + fo) * wiener(f + fo),
       i.e. the separated band after shifting to correct position
    :param otf0: Same shape as band0
    :param otf1:
    :param mask: same shape as band0. Where mask is True, use these points to evaluate the band correlation.
       Typically construct by picking some value where otf(f) and otf(f + fo) are both > w, where w is some cutoff value.

    :return phases, mags:
    """

    if cp and isinstance(band0, cp.ndarray):
        xp = cp
    else:
        xp = np

    nangles, ny, nx = band0.shape[-3:]
    phases = xp.zeros(band0.shape[:-2])
    mags = xp.zeros(band0.shape[:-2])

    # divide by OTF, but don't worry about Wiener filtering. avoid problems by keeping otf_threshold large enough
    # with np.errstate(invalid="ignore", divide="ignore"):
    #     numerator = band0 / otf0 * band1.conj() / otf1.conj()
    #     denominator = xp.abs(band0 / otf0) ** 2
    #
    # for ii in range(nangles):
    #     corr = xp.sum(numerator[..., ii, :, :][mask[..., ii, :, :]], axis=-1) / xp.sum(denominator[..., ii, :, :][mask[..., ii, :, :]], axis=-1)
    #     mags[..., ii] = xp.abs(corr)
    #     phases[..., ii] = xp.angle(corr)

    with np.errstate(invalid="ignore", divide="ignore"):
        for ii in range(nangles):
            # todo: use less memory?
            num = band0[..., ii, :, :][mask[..., ii, :, :]]
            num /= otf0[..., ii, :, :][mask[..., ii, :, :]]
            num *= band1.conj()[..., ii, :, :][mask[..., ii, :, :]]
            num /= otf1.conj()[..., ii, :, :][mask[..., ii, :, :]]
            sum_numerator = xp.sum(num, axis=-1)
            del num

            denom = xp.abs(band0[..., ii, :, :][mask[..., ii, :, :]])**2
            denom /= xp.abs(otf0[..., ii, :, :][mask[..., ii, :, :]])**2

            sum_denominator = xp.sum(denom, axis=-1)

            corr = sum_numerator / sum_denominator

            mags[..., ii] = xp.abs(corr)
            phases[..., ii] = xp.angle(corr)

    return phases, mags


def get_peak_value(img: array,
                   x: array,
                   y: array,
                   peak_coord: np.ndarray,
                   peak_pixel_size: int = 1) -> array:
    """
    Estimate value for a peak that is not precisely aligned to the pixel grid by performing a weighted average
    over neighboring pixels, based on how much these overlap with a rectangular area surrounding the peak.
    The size of this rectangular area is set by peak_pixel_size, given in integer multiples of a pixel.

    :param img: array of size n0 x n1 ... x ny x nx. This function operates on the last two dimensions of the array
    :param x: 1D array representing x-coordinates of images.
    :param y: 1D array representing y-coordinates of image
    :param peak_coord: peak coordinates [px, py]
    :param peak_pixel_size: number of pixels (along each direction) to sum to get peak value
    :return: estimated value of the peak
    """

    if cp and isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    px, py = peak_coord

    # frequency coordinates
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)

    # find closest pixel
    ix = int(np.argmin(np.abs(px - x)))
    iy = int(np.argmin(np.abs(py - y)))

    # get ROI around pixel for weighted averaging
    roi = get_centered_rois([iy, ix], [3 * peak_pixel_size, 3 * peak_pixel_size])[0]
    img_roi = cut_roi(roi, img)[0]

    xx_roi = xp.expand_dims(cut_roi(roi, xx)[0], axis=tuple(range(img_roi.ndim - 2)))
    yy_roi = xp.expand_dims(cut_roi(roi, yy)[0], axis=tuple(range(img_roi.ndim - 2)))

    weights = pixel_overlap(xp.array([[py, px]]),
                            xp.stack((yy_roi.ravel(), xx_roi.ravel()), axis=1),
                            [peak_pixel_size * dy, peak_pixel_size * dx],
                            [dy, dx]) / (dx * dy)

    _, weights = xp.broadcast_arrays(img_roi, weights.reshape(xx_roi.shape))

    peak_value = xp.average(img_roi, weights=weights, axis=(-1, -2))

    return peak_value


def pixel_overlap(centers1: array,
                  centers2: array,
                  lens1: Sequence[float],
                  lens2: Optional[Sequence[float]] = None) -> array:
    """
    Calculate overlap of two nd-rectangular pixels. The pixels go from coordinates
    centers[ii] - 0.5 * lens[ii] to centers[ii] + 0.5 * lens[ii].

    :param centers1: Array of size ncenters x ndims. coordinates define centers of first pixel along each dimension.
    :param centers2: Broadcastable to same size as centers1
    :param lens1: pixel 1 sizes along each dimension
    :param lens2: pixel 2 sizes along each dimension
    :return: overlap area of pixels
    """

    if cp and isinstance(centers1, cp.ndarray):
        xp = cp
    else:
        xp = np

    centers1 = xp.array(centers1)
    centers2 = xp.array(centers2)
    centers1, centers2 = xp.broadcast_arrays(centers1, centers2)

    lens1 = np.expand_dims(xp.array(lens1), axis=tuple(range(centers1.ndim - 1)))

    if lens2 is None:
        lens2 = lens1

    lens2 = xp.array(lens2)

    # compute overlaps
    lower_edge = xp.max(xp.stack((centers1 - 0.5 * lens1,
                                  centers2 - 0.5 * lens2), axis=0), axis=0)
    upper_edge = xp.min(xp.stack((centers1 + 0.5 * lens1,
                                  centers2 + 0.5 * lens2), axis=0), axis=0)
    overlaps = upper_edge - lower_edge
    overlaps[overlaps < 0] = 0
    volume_overlap = xp.prod(overlaps, axis=-1)

    return volume_overlap


# translating images
def translate_pix(img: array,
                  shifts: Sequence[float],
                  dr: Sequence[float] = (1, 1),
                  axes: Sequence[int] = (-2, -1),
                  wrap: bool = True,
                  pad_val: float = 0) -> (array, list[int]):
    """
    Translate image by given number of pixels with several different boundary conditions. If the shifts are sx, sy,
    then the image will be shifted by sx/dx and sy/dy. If these are not integers, they will be rounded to the closest
    integer.

    i.e. given img(y, x) return img(y + sy, x + sx). For example, a positive value for sx will shift the image
    to the left.

    :param img: image to translate
    :param shifts: distance to translate along each axis (s1, s2, ...). If these are not integers, then they will be
      rounded to the closest integer value.
    :param dr: size of pixels along each axis (dr1, dr2, ...)
    :param axes: identify the axes being wrapped, (a1, a2, ...)
    :param bool wrap: if true, pixels on the boundary are shifted across to the opposite side. If false, these
      parts of the array are padded with pad_val
    :param pad_val: value to pad portions of the image that would wrap around. Only used if wrap is False
    :return img_shifted, pix_shifts:
    """

    if cp and isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    # make sure axes positive
    axes = np.mod(np.array(axes), img.ndim)

    # convert pixel shifts to integers
    shifts_pix = np.array([int(np.round(-s / d)) for s, d in zip(shifts, dr)])

    # only need to operate on img if pixel shift is not zero
    if np.any(shifts_pix != 0):
        # roll arrays. If wrap is True, this is all we need to do
        for s, axis in zip(shifts_pix, axes):
            img = xp.roll(img, s, axis=axis)

        if wrap:
            pass
        else:
            # set parts of axes that have wrapped around to zero
            for s, axis in zip(shifts_pix, axes):

                if s >= 0:
                    slices = tuple([slice(0, img.shape[ii]) if ii != axis else slice(0, s) for ii in range(img.ndim)])
                else:
                    slices = tuple([slice(0, img.shape[ii]) if ii != axis else
                                    slice(s + img.shape[axis], img.shape[axis]) for ii in range(img.ndim)])

                img[slices] = pad_val

    return img, shifts_pix


# Fourier transform tools
def resample_bandlimited_ft(img_ft: array,
                            mag: Sequence[int],
                            axes: Sequence[int]) -> array:
    """
    Zero pad Fourier space image by adding high-frequency content. This corresponds to interpolating the real-space
    image

    Note that this is not the same as the zero padding by using ifftn s parameter, since that will pad after the least
    magnitude negative frequency, while this pads near the highest-magnitude frequencies. For more discussion of
    this point see e.g. https://github.com/numpy/numpy/issues/1346

    The expanded array is normalized so that the realspace values will match after an inverse FFT,
    thus the corresponding Fourier space components will have the relationship b_k = a_k * b.size / a.size

    :param img_ft: frequency space representation of image, arranged so that zero frequency is near the center of
       the array. The frequencies can be obtained with fftshift(fftfreq(n, dxy))
    :param mag: factor by which to oversample array. This must be an integer
    :param axes: zero-pad along these axes only
    :return img_ft_pad: expanded array
    """

    use_gpu = cp and isinstance(img_ft, cp.ndarray)
    if use_gpu:
        xp = cp
        mempool = cp.get_default_memory_pool()
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

    if use_gpu:
        mempool.free_all_blocks()

    return img_ft_pad


# create test data/SIM forward model
def get_simulated_sim_imgs(ground_truth: array,
                           patterns: array,
                           gains: Union[float, array],
                           offsets: Union[float, array],
                           readout_noise_sds: Union[float, array],
                           coherent_projection: bool = True,
                           psf: Optional[np.ndarray] = None,
                           nbin: int = 1,
                           **kwargs) -> (array, array, array, array):
    """
    Get simulated SIM images, including the effects of shot-noise and camera noise.

    The realistic image model relies on the localize_psf.camera function simulated_img(). See this function for
    details of the camera parameters gain, offset, etc.

    :param ground_truth: NumPy or CuPy array of size nz x ny x nx. If
    :param patterns:
    :param gains: gain of each pixel (or single value for all pixels)
    :param offsets: offset of each pixel (or single value for all pixels)
    :param readout_noise_sds: noise standard deviation for each pixel (or single value for all pixels)
    :param coherent_projection:
    :param psf: the point-spread function. Must have same dimensions as ground_truth, but may be different size
       proper frequency points can be obtained using fft.fftshift(fft.fftfreq(nx, dx)) and etc.
    :param nbin:
    :param kwargs: keyword arguments which will be passed through to simulated_img()

    :return sim_imgs, snrs:
      patterns_raw is generated on a grid which is nbin the size of the raw images. Assume the coordinates are aligned
      such that binning patterns_raw by a factor of nbin will produce patterns. Note that this grid
      is slightly different than the Wiener SIM reconstruction grid defined above because the FFT idiom
      used above implicitely assumes the origin is near the center of the array. This leads to fractional
      pixel offsets at the edge. See show_sim_napari() for an example of this offset
    """

    if cp and isinstance(ground_truth, cp.ndarray):
        xp = cp
    else:
        xp = np

    npatterns = len(patterns)

    ground_truth = xp.asarray(ground_truth)
    gains = xp.asarray(gains)
    offsets = xp.asarray(offsets)
    readout_noise_sds = xp.asarray(readout_noise_sds)

    # ensure ground truth is 3D
    if ground_truth.ndim == 2:
        ground_truth = xp.expand_dims(ground_truth, axis=0)
    nz, ny, nx = ground_truth.shape

    # get binned sizes
    nxb = nx / nbin
    nyb = ny / nbin
    if not nxb.is_integer() or not nyb.is_integer():
        raise Exception("The image size was not evenly divisible by the bin size")
    nxb = int(nxb)
    nyb = int(nyb)

    sim_imgs = xp.zeros((npatterns, nz, nyb, nxb), dtype=int)
    snrs = xp.zeros(sim_imgs.shape)
    for ii in range(npatterns):
            if not coherent_projection:
                patterns[ii] = blur_img_psf(patterns[ii], psf).real

            # forward SIM model
            # apply each pattern to every z-plane
            # divide by nbin so e.g. for uniform sample would keep number of photons fixed in pixel of final image
            sim_imgs[ii], snrs[ii] = simulated_img(ground_truth * patterns[ii] / nbin**2,
                                                   gains=gains,
                                                   offsets=offsets,
                                                   readout_noise_sds=readout_noise_sds,
                                                   psf=psf,
                                                   bin_size=nbin,
                                                   **kwargs)

    return sim_imgs, snrs


def get_sinusoidal_patterns(dxy: float,
                            size: Sequence[int, int],
                            frqs: array,
                            phases: array,
                            mod_depths: Union[array, float],
                            amps: Optional[Union[array, float]] = None,
                            n_oversampled: int = 1,
                            use_gpu: bool = False
                            ) -> array:
    """
    Generate sinusoidal SIM pattern on supersampled grid. This produces patterns suitable for use with
    get_simulated_sim_imgs()
    The supersampled grid is aligned with the final grid such that binning the supersampled grid
    by n_oversampled will match the final grid. Note that there is a sub-pixel shift between this coordinate
    system and the one used during Wiener SIM reconstruction

    :param dxy: pixel size
    :param size: (ny, nx)
    :param frqs: npatterns x 2
    :param phases: npatterns
    :param mod_depths: npatterns
    :param amps: npatterns
    :param n_oversampled: factor to oversample the grid.
    :param use_gpu:
    :return patterns:
    """

    if cp and use_gpu:
        xp = cp
    else:
        xp = np

    if amps is None:
        amps = xp.array([1])

    frqs = xp.atleast_2d(frqs)
    amps = xp.atleast_1d(amps)
    mod_depths = xp.atleast_1d(mod_depths)
    phases = xp.atleast_1d(phases)

    ny, nx = size

    # get binned coordinates
    xb = (xp.arange(nx) - (nx // 2)) * dxy
    yb = (xp.arange(ny) - (ny // 2)) * dxy
    xxb, yyb = xp.meshgrid(xb, yb)

    # get expanded coordinates
    yy, xx = oversample_voxel((yyb, xxb),
                              (dxy, dxy),
                              sf=n_oversampled,
                              expand_along_extra_dim=False)

    # SIM patterns
    patterns = amps[:, None, None] * 0.5 * (1 + mod_depths[:, None, None] * xp.cos(2 * np.pi * (frqs[:, 0][:, None, None] * xx +
                                                                                                frqs[:, 1][:, None, None] * yy) +
                                                                                   phases[:, None, None]))

    return patterns


def get_hexagonal_patterns(dxy: float,
                           size: Sequence[int, int],
                           frq_mag: float,
                           phase1s: array,
                           phase2s: array,
                           n_oversampled: int = 1,
                           use_gpu: bool = False) -> array:
    """
    Generate hexagonal SIM pattern on supersampled grid. This produces patterns suitable for use with
    get_simulated_sim_imgs()
    The supersampled grid is aligned with the final grid such that binning the supersampled grid
    by n_oversampled will match the final grid. Note that there is a sub-pixel shift between this coordinate
    system and the one used during Wiener SIM reconstruction

    :param dxy: pixel size
    :param size: (ny, nx)
    :param frq_mag:
    :param phase1s:
    :param phase2s:
    :param n_oversampled:
    :param use_gpu:
    :return patterns:
    """

    if len(phase1s) != len(phase2s):
        raise ValueError("Phases 1 and phases 2 should have equal lengths")

    if cp and use_gpu:
        xp = cp
    else:
        xp = np

    npatterns = len(phase1s)
    ny, nx = size

    # get binned coordinates
    xb = (xp.arange(nx) - (nx // 2)) * dxy
    yb = (xp.arange(ny) - (ny // 2)) * dxy
    xxb, yyb = np.meshgrid(xb, yb)

    # get expanded coordinates
    yy, xx = oversample_voxel((yyb, xxb),
                              (dxy, dxy),
                              sf=n_oversampled,
                              expand_along_extra_dim=False)
    xx = xp.asarray(xx)
    yy = xp.asarray(yy)

    phi_a = 0
    phi_b = 2 * np.pi / 3
    phi_c = 4 * np.pi / 3

    # ignoring z-component ...
    ka = np.array([np.cos(phi_a), np.sin(phi_a), 0]) * (2 * np.pi) * frq_mag
    kb = np.array([np.cos(phi_b), np.sin(phi_b), 0]) * (2 * np.pi) * frq_mag
    kc = np.array([np.cos(phi_c), np.sin(phi_c), 0]) * (2 * np.pi) * frq_mag

    def efield_plane_wave(x, y, z, kvec): return xp.exp(1j * (kvec[0] * x + kvec[1] * y + kvec[2] * z))

    def e_hex(x, y, z, alpha_b, alpha_c): return efield_plane_wave(x, y, z, ka) + \
                                                 efield_plane_wave(x, y, z, kb) * xp.exp(1j * alpha_b) + \
                                                 efield_plane_wave(x, y, z, kc) * xp.exp(1j * alpha_c)

    patterns = xp.zeros((npatterns, ny, nx))
    for ii in range(npatterns):
        patterns[ii] = 1 / 9 * xp.abs(e_hex(xx, yy, 0, phase1s[ii], phase2s[ii]))**2

    return patterns


class FistaSim(Optimizer):
    def __init__(self,
                 psf: array,
                 patterns: array,
                 imgs: array,
                 nbin: int = 1,
                 tau_tv: float = 0,
                 tau_l1: float = 0,
                 enforce_positivity: bool = True,
                 apodization: Optional[array] = None):
        """
        Infer SIM superresolution information using proximal gradient descent with
        l1 and/or total variation regularization

        :param psf:
        :param patterns:
        :param imgs:
        :param nbin:
        :param tau_tv:
        :param tau_l1:
        :param enforce_positivity:
        :param apodization:
        """

        super(FistaSim, self).__init__()

        self.prox_parameters = {"tau_tv": float(tau_tv),
                                "tau_l1": float(tau_l1),
                                "enforce_positivity": enforce_positivity
                                }

        self.nbin = nbin
        self.imgs = imgs
        self.ny, self.nx = imgs.shape[-2:]
        self.patterns = patterns
        self.n_samples = patterns.shape[0]
        self.psf = psf

        if cp and isinstance(imgs, cp.ndarray):
            xp = cp
        else:
            xp = np

        self.psf_reversed = xp.roll(xp.flip(psf, axis=(0, 1)), shift=(1, 1), axis=(0, 1))
        self.apodization = apodization

    def prox(self, x, step):
        # ###########################
        # TV proximal operators
        # ###########################
        # note cucim TV implementation requires ~10x memory as array does
        if self.prox_parameters["tau_tv"] != 0:
            x = tv_prox(x, self.prox_parameters["tau_tv"] * step)

        # ###########################
        # L1 proximal operators (softmax)
        # ###########################
        if self.prox_parameters["tau_l1"] != 0:
            x = soft_threshold(self.prox_parameters["tau_l1"] * step, x)

        # ###########################
        # projection constraints
        # ###########################
        if self.prox_parameters["enforce_positivity"]:
            x[x < 0] = 0

        return x

    def fwd_model(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        if cp and isinstance(x, cp.ndarray):
            xp = cp
        else:
            xp = np

        blurred = xp.stack([blur_img_psf(x * self.patterns[ii], self.psf, apodization=self.apodization).real
                            for ii in inds])

        return bin(blurred, (self.nbin, self.nbin), mode="sum")

    def cost(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        if cp and isinstance(x, cp.ndarray):
            xp = cp
        else:
            xp = np

        return 0.5 * xp.mean(xp.abs(self.fwd_model(x, inds=inds) - self.imgs[inds]) ** 2, axis=(1, 2))

    def gradient(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        if cp and isinstance(x, cp.ndarray):
            xp = cp
        else:
            xp = np

        img_model = self.fwd_model(x, inds=inds)
        dc_do = xp.stack([blur_img_psf(bin_adjoint(img_model[ii] - self.imgs[ind], (self.nbin, self.nbin), mode="sum"),
                                     self.psf_reversed, self.apodization).real
                        for ii, ind in enumerate(inds)])
        dc_do *= self.patterns[inds] / (self.ny * self.nx)

        return dc_do
