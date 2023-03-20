"""
Tools for reconstructing optical diffraction tomography (ODT) data
"""
import time
import datetime
import warnings
from pathlib import Path
from typing import Union, Optional
import random
import numpy as np
from numpy import fft
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.signal.windows import tukey, hann
from skimage.restoration import unwrap_phase, denoise_tv_chambolle
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask_image.ndfilters import convolve as dconvolve
# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Arc
import zarr
#
from localize_psf import fit, rois, camera, affine
import mcsim.analysis.sim_reconstruction as sim
import mcsim.analysis.analysis_tools as tools
from mcsim.analysis import field_prop

_gpu_available = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as sp_gpu
    from cucim.skimage.restoration import denoise_tv_chambolle as denoise_tv_chambolle_gpu
    # from cucim.skimage.restoration import unwrap_phase as unwrap_phase_gpu # this not implemented ...
except ImportError:
    cp = np
    sp_gpu = sp
    _gpu_available = False

_gpufit_available = True
try:
    import pygpufit.gpufit as gf
except ImportError:
    _gpufit_available = False

array = Union[np.ndarray, cp.ndarray]
csr_matrix = Union[sp.csr_matrix, sp_gpu.csr_matrix]


class tomography:
    def __init__(self,
                 imgs_raw: da.array,
                 wavelength: float,
                 no: float,
                 na_detection: float,
                 na_excitation: float,
                 dxy: float,
                 reference_frq_guess: np.ndarray,
                 hologram_frqs_guess: list[np.ndarray],
                 imgs_raw_bg: Optional[da.array] = None,
                 phase_offsets: Optional[np.ndarray] = None,
                 axes_names: Optional[list[str]] = None,
                 verbose: bool = True):
        """
        Object to reconstruct optical diffraction tomography data

        @param imgs_raw: n1 x n2 x ... x nm x npatterns x ny x nx. Data intensity images
        @param wavelength: wavelength in um
        @param no: background index of refraction
        @param na_detection:
        @param na_excitation:
        @param dxy: pixel size in um
        @param reference_frq_guess: [fx, fy] hologram reference frequency
        @param hologram_frqs_guess: list of length npatterns, where each entry is an array of frequencies contained
         in the given pattern. These arrays may be of different sizes, i.e. different patterns may contain a different
          number of frequencies. But each array should be of size n_i x 2
        @param imgs_raw_bg: background intensity images. If no background images are provided, then a time
        average of imgs_raw will be used as the background
        @param phase_offsets: phase shifts between images and corresponding background images
        @param axes_names: names of first m + 1 axes
        @param verbose:
        """
        self.verbose = verbose
        self.tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

        # image dimensions
        # todo: ensure that are dask arrays
        self.imgs_raw = imgs_raw
        self.imgs_raw_bg = imgs_raw_bg
        self.use_average_as_background = self.imgs_raw_bg is None

        self.npatterns, self.ny, self.nx = imgs_raw.shape[-3:]
        self.nextra_dims = imgs_raw.ndim - 3

        if axes_names is None:
            self.axes_names = [f"i{ii:d}" for ii in range(self.imgs_raw.ndim - 2)]
        else:
            self.axes_names = axes_names

        # reference frequency
        # shape = n0 x ... x nm x 2 where imgs have shape n0 x ... x nm x npatterns x ny x nx
        self.reference_frq = np.array(reference_frq_guess) + np.zeros(self.imgs_raw.shape[:-3] + (2,))
        self.reference_frq_bg = None

        # list of length npatterns, with each entry [1 x 1 ... x 1 x nmultiplex x 2]
        # with potential different nmultiplex for each pattern
        self.hologram_frqs = hologram_frqs_guess
        self.hologram_frqs_bg = None

        # physical parameters
        self.wavelength = wavelength
        self.no = no
        self.na_detection = na_detection
        self.na_excitation = na_excitation
        self.fmax = self.na_detection / self.wavelength

        # phase shifts
        self.phase_offsets = phase_offsets
        self.phase_offsets_bg = None

        # other hologram images we will define later
        self.holograms_ft = None
        self.holograms_ft_bg = None
        self.efield_scattered_ft = None
        self.phi_rytov_ft = None
        self.dz_final = None
        self.step = None

        self.powers_rms = None
        self.e_powers_rms = None
        self.powers_rms_bg = None
        self.e_powers_rms_bg = None

        # generate coordinates of ground truth image
        self.dxy = dxy
        self.x = (np.arange(self.nx) - (self.nx // 2)) * dxy
        self.y = (np.arange(self.ny) - (self.ny // 2)) * dxy
        self.fxs = fft.fftshift(fft.fftfreq(self.nx, self.dxy))
        self.fys = fft.fftshift(fft.fftfreq(self.ny, self.dxy))
        self.dfx = self.fxs[1] - self.fxs[0]
        self.dfy = self.fys[1] - self.fys[0]

        # where pupil allowed to be non-zero
        self.pupil_mask = np.expand_dims(np.sqrt(self.fxs[None, :]**2 + self.fys[:, None]**2) <= self.fmax,
                                         axis=tuple(range(self.nextra_dims)) + (-3,))
        # value of pupil function
        self.pupil = np.expand_dims(self.pupil_mask.astype(complex),
                                    axis=tuple(range(self.nextra_dims)) + (-3,))

        # settings
        self.reconstruction_settings = {}

    def estimate_hologram_frqs(self,
                               roi_size_pix: int = 10,
                               save_dir: Optional[str] = None,
                               use_gpufit: bool = False):
        """
        Estimate hologram frequencies from raw images.
        Guess values need to be within a few pixels for this to succeed. Can easily achieve this accuracy by
        looking at FFT
        @param roi_size_pix: ROI size (in pixels) to do frequency fitting on
        @param save_dir:
        @param use_gpufit:
        @return:
        """
        self.reconstruction_settings.update({"roi_size_pix": roi_size_pix})

        saving = save_dir is not None

        if saving:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # frequency data associated with images
        fxfx, fyfy = np.meshgrid(self.fxs, self.fys)

        def get_hologram_frqs(img,
                              fx_guesses,
                              fy_guesses,
                              use_guess_init_params,
                              saving: bool,
                              roi_size_pix: int,
                              prefix: str = "",
                              block_id=None):
            """

            @param img:
            @param fx_guesses:
            @param fy_guesses:
            @param use_guess_init_params: if True use guess not only to find an initial ROI but also as the initial
            guess in the fit
            @param saving:
            @param roi_size_pix:
            @param prefix:
            @param block_id: can use this to figure out what blcok we are in when running with dask.array map_blocks()
            @return:
            """

            nextra_dims = img.ndim - 2

            if img.shape != (1,) * nextra_dims + img.shape[-2:]:
                raise NotImplementedError()

            img = img.squeeze(axis=tuple(range(nextra_dims)))

            # this will get block id when function is run using map_blocks
            # time_index = block_id[1]
            if block_id is None:
                block_id = (0, )

            n_multiplex_this_pattern = len(fx_guesses)

            frqs_holo = np.zeros((1,) * nextra_dims + (n_multiplex_this_pattern, 2))
            for ii in range(n_multiplex_this_pattern):
                fx_guess = fx_guesses[ii]
                fy_guess = fy_guesses[ii]

                # ROI centered on frequency guess
                nx_center_2 = np.argmin(np.abs(np.squeeze(fx_guess) - self.fxs))
                ny_center_2 = np.argmin(np.abs(np.squeeze(fy_guess) - self.fys))
                roi2 = rois.get_centered_roi([ny_center_2, nx_center_2],
                                             [roi_size_pix, roi_size_pix],
                                             min_vals=[0, 0],
                                             max_vals=img.shape[-2:])

                apodization = np.outer(hann(img.shape[-2]), hann(img.shape[-1]))

                img_ref_ft = fft.fftshift(fft.fft2(fft.ifftshift(np.squeeze(img * apodization))))
                ft2 = img_ref_ft[..., roi2[0]:roi2[1], roi2[2]:roi2[3]]

                # fit to Gaussian
                fxfx_roi = rois.cut_roi(roi2, fxfx)
                fyfy_roi = rois.cut_roi(roi2, fyfy)
                if use_guess_init_params:
                    fx_ip_guess = np.squeeze(fx_guess)
                    fy_ip_guess = np.squeeze(fy_guess)
                else:
                    max_ind = np.unravel_index(np.argmax(np.abs(ft2)), ft2.shape)
                    fx_ip_guess = fxfx_roi[max_ind]
                    fy_ip_guess = fyfy_roi[max_ind]

                init_params = [np.max(np.abs(ft2)),
                               fx_ip_guess,
                               fy_ip_guess,
                               self.dfx * 3,
                               self.dfy * 3,
                               0,
                               0]

                rgauss = fit.gauss2d().fit(np.abs(ft2),
                                           (fyfy_roi, fxfx_roi),
                                           init_params=init_params,
                                           fixed_params=[False, False, False, False, False, False, True],
                                           guess_bounds=True)

                frqs_now = rgauss["fit_params"][1:3]
                frqs_holo[..., ii, :] = np.expand_dims(frqs_now, axis=list(range(fx_guess.ndim)))

                # todo: split this to another function
                if saving and np.all(np.array(block_id) == 0):
                    figh = sim.plot_correlation_fit(img_ref_ft,
                                                    img_ref_ft,
                                                    frqs_now,
                                                    self.dxy,
                                                    roi_size=(roi2[1] - roi2[0], roi2[3] - roi2[2]),
                                                    frqs_guess=np.array([np.squeeze(fx_guess), np.squeeze(fy_guess)])
                                                    )

                    figh.savefig(Path(save_dir, f"{prefix:s}=pattern_{ii:d}=multiplex_holo_frq_diagnostic.png"))
                    plt.close(figh)

            return frqs_holo

        # ############################
        # fit hologram frequencies for ALL chunks
        # ############################

        # todo: fit background images also
        if not use_gpufit or not _gpufit_available:
            delayed_frqs = []
            delayed_frqs_bg = []
            # loop over patterns
            for ii in range(self.npatterns):
                n_multiplex_this_pattern = len(self.hologram_frqs[ii])
                delayed_frqs.append(da.map_blocks(get_hologram_frqs,
                                             self.imgs_raw[..., ii, :, :],
                                             self.hologram_frqs[ii][:, 0],
                                             self.hologram_frqs[ii][:, 1],
                                             use_guess_init_params=False,
                                             saving=saving,
                                             roi_size_pix=roi_size_pix,
                                             prefix=str(ii),
                                             dtype=float,
                                             chunks=(1,) * self.nextra_dims + (n_multiplex_this_pattern, 2),
                                             drop_axis=(-1, -2),
                                             new_axis=(-1, -2)
                                             ))

                if not self.use_average_as_background:
                    delayed_frqs_bg.append(da.map_blocks(get_hologram_frqs,
                                             self.imgs_raw_bg[..., ii, :, :],
                                             self.hologram_frqs[ii][:, 0],
                                             self.hologram_frqs[ii][:, 1],
                                             use_guess_init_params=False,
                                             saving=saving,
                                             roi_size_pix=roi_size_pix,
                                             prefix=str(ii),
                                             dtype=float,
                                             chunks=(1,) * self.nextra_dims + (n_multiplex_this_pattern, 2),
                                             drop_axis=(-1, -2),
                                             new_axis=(-1, -2)
                                             ))

            # do frequency calibration
            print(f"calibrating {self.npatterns:d} patterns,"
                  f" each multiplexing {[f.shape[-2] for f in self.hologram_frqs]} plane waves")

            if self.verbose:
                with ProgressBar():
                    frqs_hologram, frqs_hologram_bg = dask.compute([delayed_frqs, delayed_frqs_bg])[0]
            else:
                frqs_hologram, frqs_hologram_bg = dask.compute([delayed_frqs, delayed_frqs_bg])[0]

            if self.use_average_as_background:
                frqs_hologram_bg = frqs_hologram

        else:
            
            raise NotImplementedError("fitting frequencies with gpufit is not yet fully implemented")
            
            tstart = time.perf_counter()

            apodization = np.outer(hann(self.ny), hann(self.nx))
            def ft_abs(img): return np.abs(fft.fftshift(fft.fft2(fft.ifftshift(img * apodization))))
            xp = np

            dfx = self.fxs[1] - self.fxs[0]
            dfy = self.fys[1] - self.fys[0]

            frqs_hologram = [np.zeros(self.imgs_raw.shape[:self.nextra_dims] + (len(f), 2)) for f in self.hologram_frqs]
            for ii in range(self.npatterns):

                imgs_ft_now = da.map_blocks(ft_abs,
                                            self.imgs_raw[..., ii, :, :],
                                            dtype=float,
                                            meta=xp.array([])
                                            )

                n_multiplex_this_pattern = len(self.hologram_frqs[ii])
                for jj in range(n_multiplex_this_pattern):
                    fx_guess = self.hologram_frqs[ii][jj, 0]
                    fy_guess = self.hologram_frqs[ii][jj, 1]

                    nx_center = np.argmin(np.abs(np.squeeze(fx_guess) - self.fxs))
                    ny_center = np.argmin(np.abs(np.squeeze(fy_guess) - self.fys))
                    roi = rois.get_centered_roi([ny_center, nx_center],
                                                 [roi_size_pix, roi_size_pix],
                                                 min_vals=[0, 0],
                                                 max_vals=(self.ny, self.nx))

                    nfits = np.prod(self.imgs_raw.shape[:self.nextra_dims])

                    data = imgs_ft_now[..., roi[0]:roi[1], roi[2]:roi[3]].compute().reshape([nfits, roi_size_pix**2]).astype(np.float32)

                    # for fitting, take coordinate of maximum pixel value as guess
                    max_ind = np.unravel_index(np.argmax(data, axis=1), (roi_size_pix, roi_size_pix))
                    fx_fit_guess = self.fxs[roi[2]:roi[3]][max_ind[1]]
                    fy_fit_guess = self.fys[roi[0]:roi[1]][max_ind[0]]

                    # symmetric 2D gaussian
                    nparams = 5
                    init_params = np.zeros((nfits, nparams), dtype=np.float32)
                    init_params[:, 0] = np.max(data, axis=1) - np.mean(data, axis=1)
                    init_params[:, 1] = (fx_fit_guess - self.fxs[roi[2]]) / dfx
                    init_params[:, 2] = (fy_fit_guess - self.fys[roi[0]]) / dfy
                    init_params[:, 3] = 3
                    init_params[:, 4] = np.mean(data, axis=1)

                    # 2D rotated
                    # nparams = 7
                    # init_params = np.zeros((nfits, nparams), dtype=np.float32)
                    # init_params[:, 0] = np.max(data, axis=1) - np.mean(data, axis=1)
                    # init_params[:, 1] = (fx_fit_guess - self.fxs[roi[2]]) / dfx
                    # init_params[:, 2] = (fy_fit_guess - self.fys[roi[0]]) / dfy
                    # init_params[:, 3] = 3
                    # init_params[:, 4] = 3
                    # init_params[:, 5] = np.mean(data, axis=1)
                    # init_params[:, 6] = 0

                    if init_params.ndim != 2 or init_params.shape != (nfits, nparams):
                        raise ValueError(f"init_params should have shape ({nfits:d}, {nparams:d}), but had shape {init_params.shape}")

                    # do fitting
                    fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data,
                                                                             None,
                                                                             gf.ModelID.GAUSS_2D,
                                                                             # gf.ModelID.GAUSS_2D_ROTATED,
                                                                             init_params,
                                                                             tolerance=1e-8,
                                                                             max_number_iterations=100,
                                                                             estimator_id=gf.EstimatorID.LSE,
                                                                             parameters_to_fit=np.ones(nparams, dtype=np.int32),
                                                                             user_info=None)

                    # defined in Gpufit/constants.h
                    # fit_states_key = {"converged": 0,
                    #                   "max_iteration": 1,
                    #                   "singular_hessian": 2,
                    #                   "neg_curvature_mle": 3,
                    #                   "gpu_not_ready": 4}

                    # convert fit centers to real units
                    frqs_hologram[ii][..., 0] = fit_params[:, 1].reshape(frqs_hologram[ii].shape[:-1]) * dfx + self.fxs[roi[2]]
                    frqs_hologram[ii][..., 1] = fit_params[:, 2].reshape(frqs_hologram[ii].shape[:-1]) * dfy + self.fys[roi[0]]

            print(f"fit frequencies in {time.perf_counter() - tstart:.2f}s")

        # for hologram interference frequencies, use frequency closer to guess value
        # ref_frq_exp = np.expand_dims(self.reference_frq, axis=-1)
        # for ii in range(self.npatterns):
        #     # todo: reference_frq size problem
        #     frq_dists_ref = np.linalg.norm(frqs_hologram[ii] - ref_frq_exp, axis=-1)
        #     frq_dists_neg_ref = np.linalg.norm(frqs_hologram[ii] + ref_frq_exp, axis=-1)
        #     frqs_hologram[ii][frq_dists_neg_ref < frq_dists_ref, :] *= -1

        self.hologram_frqs = frqs_hologram
        self.hologram_frqs_bg = frqs_hologram_bg

    def estimate_reference_frq(self,
                               mode: str = "fit",
                               save_dir: Optional[str] = None):
        """
        Estimate hologram reference frequency
        @param mode: if "fit" fit the residual speckle pattern to try and estimate the reference frequency.
         If "average" take the average of self.hologram_frqs as the reference frequency
        @param save_dir:
        @return:
        """
        
        self.reconstruction_settings.update({"reference_frequency_mode": mode})

        if mode == "fit":
            raise NotImplementedError("mode 'fit' not implemented after multiplexing code update")

            saving = save_dir is not None

            # load one slice of background data to get frequency reference. Load the first slice along all dimensions
            slices = tuple([slice(0, 1)] * self.nextra_dims + [slice(None)] * 3)
            imgs_frq_cal = np.squeeze(self.imgs_raw_bg[slices])

            imgs_frq_cal_ft = fft.fftshift(fft.fft2(fft.ifftshift(imgs_frq_cal, axes=(-1, -2)), axes=(-1, -2)),
                                           axes=(-1, -2))
            imgs_frq_cal_ft_abs_mean = np.mean(np.abs(imgs_frq_cal_ft), axis=0)

            results, circ_dbl_fn, figh_ref_frq = fit_ref_frq(imgs_frq_cal_ft_abs_mean,
                                                             self.dxy,
                                                             2*self.fmax,
                                                             show_figure=saving)
            frq_ref = results["fit_params"][:2]

            # flip sign if necessary to ensure close to guess
            if np.linalg.norm(frq_ref - self.reference_frq) > np.linalg.norm(frq_ref + self.reference_frq):
                frq_ref = -frq_ref

            frq_ref_bg = frq_ref

            if saving:
                save_dir = Path(save_dir)
                save_dir.mkdir(exist_ok=True)

                figh_ref_frq.savefig(Path(save_dir, "frequency_reference_diagnostic.png"))
        elif mode == "average":
            frq_ref = np.mean(np.stack([np.mean(f, axis=-2) for f in self.hologram_frqs], axis=-1), axis=-1)
            frq_ref_bg = np.mean(np.stack([np.mean(f, axis=-2) for f in self.hologram_frqs_bg], axis=-1), axis=-1)
        else:
            raise ValueError(f"'mode' must be '{mode:s}' but must be 'fit' or 'average'")

        self.reference_frq = frq_ref
        self.reference_frq_bg = frq_ref_bg

    def get_beam_frqs(self):
        """
        Get beam incident beam frequencies from hologram frequencies and reference frequency

        @return beam_frqs: array of size N1 x N2 ... x Nm x 3
        """

        # bxys = self.hologram_frqs - np.expand_dims(self.reference_frq, axis=0)
        # bzs = get_fz(bxys[..., 0], bxys[..., 1], self.no, self.wavelength)
        # # x, y, z
        # beam_frqs = np.stack((bxys[..., 0], bxys[..., 1], bzs), axis=-1)

        bxys = [f - np.expand_dims(self.reference_frq, axis=-2) for f in self.hologram_frqs]
        bzs = [get_fz(bxy[..., 0], bxy[..., 1], self.no, self.wavelength) for bxy in bxys]
        beam_frqs = [np.stack((bxy[..., 0], bxy[..., 1], bz), axis=-1) for bxy, bz in zip(bxys, bzs)]

        return beam_frqs

    def find_affine_xform_to_frqs(self,
                                  offsets: list[np.ndarray],
                                  save_dir: Optional[str] = None):
        """
        Fit affine transformation between device and measured frequency space.

        This could be between frequencies displayed on DMD and measured frequency space (DMD in imaging plane)
        or between mirror positions on DMD and frequency space (DMD in Fourier plane)

        @param offsets:
        @param save_dir:
        @return:
        """

        centers_dmd = np.concatenate(offsets, axis=0)
        mean_hologram_frqs = np.concatenate([np.mean(f, axis=tuple(range(self.nextra_dims))) for f in self.hologram_frqs], axis=0)
        mean_ref_frq = np.mean(self.reference_frq, axis=tuple(range(self.nextra_dims)))

        beam_frqs = np.concatenate([np.mean(f, axis=tuple(range(self.nextra_dims))) for f in self.get_beam_frqs()], axis=0)

        # fit affine transformation
        if len(mean_hologram_frqs) > 6:
            xform_dmd2frq, _, _, _ = affine.fit_xform_points_ransac(centers_dmd,
                                                                    mean_hologram_frqs,
                                                                    dist_err_max=0.1,
                                                                    niterations=100)
        else:
            # no point in RANSAC if not enough points to invert transformation
            xform_dmd2frq, _ = affine.fit_xform_points(centers_dmd, mean_hologram_frqs)

        # # map pupil positions to frequency
        frqs_from_pupil = affine.xform_points(centers_dmd, xform_dmd2frq)
        # estimate frequency of reference beam from affine transformation and previous calibration information
        frq_dmd_center = affine.xform_points(np.array([[0, 0]]), xform_dmd2frq)[0]

        # also get inverse transform and map frequencies to pupil (DMD positions)
        xform_frq2dmd = np.linalg.inv(xform_dmd2frq)
        centers_pupil_from_frq = affine.xform_points(mean_hologram_frqs, xform_frq2dmd)
        #
        center_pupil_frq_ref = affine.xform_points(mean_hologram_frqs, xform_frq2dmd)[0]
        # center_pupil_frq_ref = affine.xform_points(np.expand_dims(mean_hologram_frqs, axis=0), xform_frq2dmd)[0]

        # map maximum pupil frequency circle to DMD space
        circle_thetas = np.linspace(0, 2 * np.pi, 1001)
        frqs_pupil_boundary = self.fmax * np.stack((np.cos(circle_thetas), np.sin(circle_thetas)),
                                                     axis=1) + np.expand_dims(mean_ref_frq, axis=0)
        centers_dmd_fmax = affine.xform_points(frqs_pupil_boundary, xform_frq2dmd)
        rmax_dmd_mirrors = np.max(np.linalg.norm(centers_dmd_fmax, axis=1))

        # DMD boundary
        # south = np.zeros((nx_dmd, 2))
        # south[:, 0] = np.arange(nx_dmd) - (nx_dmd // 2)
        # south[:, 1] = 0 - (ny_dmd // 2)
        #
        # north = np.zeros((nx_dmd, 2))
        # north[:, 0] = np.arange(nx_dmd) - (nx_dmd // 2)
        # north[:, 1] = ny_dmd - (ny_dmd // 2)
        #
        # east = np.zeros((ny_dmd, 2))
        # east[:, 0] = nx_dmd - (nx_dmd // 2)
        # east[:, 1] = np.arange(ny_dmd) - (ny_dmd // 2)
        #
        # west = np.zeros((ny_dmd, 2))
        # west[:, 0] = 0 - (nx_dmd // 2)
        # west[:, 1] = np.arange(ny_dmd) - (ny_dmd // 2)
        #
        # dmd_boundary = np.concatenate((south, north, east, west), axis=0)
        # dmd_boundry_freq = affine.xform_points(dmd_boundary, xform_dmd2frq)

        # check sign of frequency reference is consistent with affine transform
        assert np.linalg.norm(frq_dmd_center + mean_ref_frq) >= np.linalg.norm(frq_dmd_center - mean_ref_frq)

        # ##############################
        # plot data
        # ##############################
        xform_params = affine.xform2params(xform_dmd2frq)

        figh = plt.figure(figsize=(20, 8))
        grid = figh.add_gridspec(1, 3)
        figh.suptitle("Mapping from pupil (DMD surface) to hologram frequencies (in object space)\n"
                      f"Reference freq = ({mean_ref_frq[0]:.3f}, {mean_ref_frq[1]:.3f}) $1/\mu m$,"
                      f" central mirror = ({center_pupil_frq_ref[0]:.1f}, {center_pupil_frq_ref[1]:.1f})\n"
                      "affine xform from DMD space to frequency space\n"
                      f"$1/M_x$ = {1 / xform_params[0]:.2f} mirror/$\mu m^{-1}$,"
                      f" $\\theta x$ = {xform_params[1] * 180 / np.pi:.2f} deg,"
                      f" $c_x$ = {xform_params[2]:.3f} $1/\mu m$\n"
                      f"$1/M_y$ = {1 / xform_params[3]:.2f} mirror/$\mu m^{-1}$,"
                      f" $\\theta y$ = {xform_params[4] * 180 / np.pi:.2f} deg,"
                      f" $c_y$ = {xform_params[5]:.3f} $1/\mu m$")

        ax = figh.add_subplot(grid[0, 0])
        ax.axis("scaled")
        ax.set_title("DMD space")
        # ax.plot(dmd_boundary[:, 0], dmd_boundary[:, 1], 'k.', label="DMD edge")
        ax.plot(centers_pupil_from_frq[..., 0],
                centers_pupil_from_frq[..., 1],
                'rx',
                label="fit hologram frequencies")
        ax.plot(centers_dmd[:, 0], centers_dmd[:, 1], 'b.', label="mirror positions")
        ax.plot(0, 0, 'g+', label="DMD center")
        ax.plot(center_pupil_frq_ref[0], center_pupil_frq_ref[1], "m3", label="reference freq")
        ax.plot(centers_dmd_fmax[:, 0], centers_dmd_fmax[:, 1], 'k', label="pupil")
        ax.set_xlim([-rmax_dmd_mirrors, rmax_dmd_mirrors])
        ax.set_ylim([-rmax_dmd_mirrors, rmax_dmd_mirrors])
        ax.legend(bbox_to_anchor=(0.2, 1.1))
        ax.set_xlabel("x-position (mirrors)")
        ax.set_ylabel("y-position (mirrors)")

        ax = figh.add_subplot(grid[0, 1])
        ax.axis("scaled")
        ax.set_title("Raw frequencies")
        # ax.plot(dmd_boundry_freq[:, 0], dmd_boundry_freq[:, 1], 'k.')
        ax.plot(mean_hologram_frqs[..., 0], mean_hologram_frqs[..., 1], 'rx')
        ax.plot(frqs_from_pupil[..., 0], frqs_from_pupil[..., 1], 'b.')
        ax.plot(frq_dmd_center[0], frq_dmd_center[1], 'g+')
        ax.plot(mean_ref_frq[0], mean_ref_frq[1], "m3")
        ax.add_artist(Circle(mean_ref_frq, radius=self.fmax, facecolor="none", edgecolor="k"))
        ax.set_xlim([-self.fmax + mean_ref_frq[0], self.fmax + mean_ref_frq[0]])
        ax.set_ylim([-self.fmax + mean_ref_frq[1], self.fmax + mean_ref_frq[1]])
        ax.set_xlabel("$f_x$ (1/$\mu m$)")
        ax.set_ylabel("$f_y$ (1/$\mu m$)")

        ax = figh.add_subplot(grid[0, 2])
        ax.axis("scaled")
        ax.set_title("Frequencies - reference frequency")
        # ax.plot(dmd_boundry_freq[:, 0] - mean_ref_frq[0], dmd_boundry_freq[:, 1] - mean_ref_frq[1], 'k.')
        ax.plot(beam_frqs[..., 0], beam_frqs[..., 1], 'rx')
        ax.plot(frqs_from_pupil[..., 0] - mean_ref_frq[0], frqs_from_pupil[..., 1] - mean_ref_frq[1], 'b.')
        ax.plot(frq_dmd_center[0] - mean_ref_frq[0], frq_dmd_center[1] - mean_ref_frq[1], 'g+')
        ax.plot(0, 0, 'm3')
        ax.add_artist(Circle((0, 0), radius=self.fmax, facecolor="none", edgecolor="k"))
        ax.set_xlim([-self.fmax, self.fmax])
        ax.set_ylim([-self.fmax, self.fmax])
        ax.set_xlabel("$f_x$ (1/$\mu m$)")
        ax.set_ylabel("$f_y$ (1/$\mu m$)")

        if save_dir is not None:
            figh.savefig(Path(save_dir, "frequency_mapping.png"))
            plt.close(figh)

        return xform_dmd2frq

    def unmix_holograms(self,
                        bg_average_axes: tuple[int],
                        kernel_size: Optional[int] = None,
                        mask: Optional[np.ndarray] = None,
                        fit_phases: bool = False,
                        correct_amplitudes: bool = True,
                        apodization: Optional[np.ndarray] = None,
                        use_gpu: bool = False):
        """
        Unmix and preprocess holograms

        Note that this only depends on reference frequencies, and not on determined hologram frequencies

        @param bg_average_axes: axes to average along when producing background images
        @param kernel_size: when averaging time points to generate a background image, this is the number of
        time points that will be used
        @param mask: area to be cut out of hologrms
        @param fit_phases: whether to fit phase differences between image and background holograms
        @param correct_amplitudes:
        @param apodization: if None use tukey apodization with alpha = 0.1. To use no apodization set equal to 1
        @param use_gpu:
        @return:
        """

        self.reconstruction_settings.update({"background_average_kernel_size": kernel_size})
        self.reconstruction_settings.update({"correct_amplitudes": correct_amplitudes})

        if use_gpu and _gpu_available:
            xp = cp
        else:
            xp = np

        if apodization is None:
            apodization = np.outer(tukey(self.ny, alpha=0.1), tukey(self.nx, alpha=0.1))

        # make ref_frq_da[..., ii] broadcastable to same size as raw images so can use with dask array
        ref_frq_da = da.from_array(np.expand_dims(self.reference_frq, axis=(-2, -3, -4)),
                                   chunks=self.imgs_raw.chunksize[:-2] + (1, 1, 2)
                                   )

        if self.use_average_as_background:
            ref_frq_bg_da = ref_frq_da
        else:
            ref_frq_bg_da = da.from_array(np.expand_dims(self.reference_frq_bg, axis=(-2, -3, -4)),
                                       chunks=self.imgs_raw_bg.chunksize[:-2] + (1, 1, 2)
                                       )

        # #########################
        # get electric field from holograms
        # #########################
        holograms_ft_raw = da.map_blocks(unmix_hologram,
                                         self.imgs_raw,
                                         self.dxy,
                                         2*self.fmax,
                                         ref_frq_da[..., 0],
                                         ref_frq_da[..., 1],
                                         apodization=apodization,
                                         dtype=complex)

        # cut off region using mask
        if mask is None:
            holograms_ft = holograms_ft_raw
        else:
            # ensure masks has same dims and chunks as holograms_ft_raw
            masks_da = da.from_array(xp.expand_dims(xp.array(mask),
                                                    axis=tuple(range(0, self.imgs_raw.ndim - mask.ndim))),
                                     chunks=holograms_ft_raw.chunksize)

            holograms_ft = da.map_blocks(cut_mask,
                                         holograms_ft_raw,
                                         masks_da,
                                         mask_val=0,
                                         dtype=complex)

        # #########################
        # get background electric field from holograms
        # #########################
        if self.use_average_as_background:
            holograms_ft_bg = holograms_ft
        else:
            holograms_ft_raw_bg = da.map_blocks(unmix_hologram,
                                                self.imgs_raw_bg,
                                                self.dxy,
                                                2*self.fmax,
                                                ref_frq_bg_da[..., 0],
                                                ref_frq_bg_da[..., 1],
                                                apodization=apodization,
                                                dtype=complex)

            if mask is None:
                holograms_ft_bg = holograms_ft_raw_bg
            else:
                # ensure masks has same dims and chunks as holograms_ft_raw
                masks_da_bg = da.from_array(xp.expand_dims(xp.array(mask),
                                                           axis=tuple(range(0, self.imgs_raw.ndim - mask.ndim))),
                                         chunks=holograms_ft_raw_bg.chunksize)
                holograms_ft_bg = da.map_blocks(cut_mask,
                                                holograms_ft_raw_bg,
                                                masks_da_bg,
                                                mask_val=0,
                                                dtype=complex)

        # #########################
        # determine phase offsets for background electric field, relative to initial slice
        # for each angle, fit compared with reference slices
        # #########################
        if fit_phases:
            # optionally determine background phase offsets
            print("computing background phase offsets")

            # take one slice from all axes which will be averaged. Otherwise, must keep the dimensions
            slices = tuple([slice(0, 1) if a in bg_average_axes else slice(None) for a in range(self.nextra_dims)] +
                           [slice(None)] * 3)

            fit_params_dask = da.map_blocks(get_global_phase_shifts,
                                            holograms_ft_bg,
                                            holograms_ft_bg[slices],  # reference slices
                                            dtype=complex,
                                            chunks=holograms_ft_bg.chunksize[:-2] + (1, 1)
                                            )

            if self.verbose:
                with ProgressBar():
                    fit_params = fit_params_dask.compute()
            else:
                fit_params = fit_params_dask.compute()

            phase_offsets_bg = xp.angle(fit_params)
            amp_corr_bg = xp.abs(fit_params)

        else:
            phase_offsets_bg = np.ones((1,) * self.nextra_dims + (self.npatterns, 1, 1))
            amp_corr_bg = np.ones(phase_offsets_bg.shape)

        self.phase_offsets_bg = phase_offsets_bg

        if correct_amplitudes:
            self.amp_corr_bg = amp_corr_bg
        else:
            self.amp_corr_bg = np.ones(amp_corr_bg.shape)

        # #########################
        # determine background electric field
        # #########################
        print("Computing background electric field")
        if kernel_size is None:
            # average all available
            # holograms_ft_bg_comp = da.mean(holograms_ft_bg * da.exp(1j * self.phase_offsets_bg),
            #                                axis=bg_average_axes,
            #                                keepdims=True)

            # todo: testing adjusting normalizations also
            holograms_ft_bg_comp = da.mean(holograms_ft_bg * da.exp(1j * self.phase_offsets_bg) * self.amp_corr_bg,
                                           axis=bg_average_axes,
                                           keepdims=True)

        else:
            # rolling average
            convolve_kernel = da.ones(tuple([kernel_size if ii in bg_average_axes else 1 for ii in range(holograms_ft_bg.ndim)]))

            # numerator = dconvolve(holograms_ft_bg * da.exp(1j * self.phase_offsets_bg),
            #                       convolve_kernel,
            #                       mode="constant",
            #                       cval=0).rechunk(holograms_ft_bg.chunksize)

            # todo: testing adjusting normalization
            numerator = dconvolve(holograms_ft_bg * da.exp(1j * self.phase_offsets_bg) * self.amp_corr_bg,
                                  convolve_kernel,
                                  mode="constant",
                                  cval=0).rechunk(holograms_ft_bg.chunksize)

            # denominator = dconvolve(da.ones(holograms_ft_bg.shape), convolve_kernel, mode="constant", cval=0)
            other_shape = da.ones(tuple([holograms_ft_bg.shape[ii] if ii in bg_average_axes else 1 for ii in range(holograms_ft_bg.ndim)]))

            denominator = dconvolve(other_shape,
                                    convolve_kernel,
                                    mode="constant",
                                    cval=0)

            holograms_ft_bg_comp = (numerator / denominator)

        if self.verbose:
            with ProgressBar():
                self.holograms_ft_bg = da.from_array(holograms_ft_bg_comp.compute(),
                                                     chunks=holograms_ft_bg.chunksize)
        else:
            self.holograms_ft_bg = da.from_array(holograms_ft_bg_comp.compute(),
                                                 chunks=holograms_ft_bg.chunksize)

        # #########################
        # determine phase offsets of electric field compared to background
        # #########################
        if self.use_average_as_background:
            phase_offsets = self.phase_offsets_bg
            amp_corr = self.amp_corr_bg
        else:
            if fit_phases:
                print("computing phase offsets")
                fit_params_dask = da.map_blocks(get_global_phase_shifts,
                                                holograms_ft,
                                                self.holograms_ft_bg,
                                                dtype=complex,
                                                chunks=holograms_ft.chunksize[:-2] + (1, 1),
                                                )

                if self.verbose:
                    with ProgressBar():
                        fit_params = fit_params_dask.compute()
                else:
                    fit_params = fit_params_dask.compute()

                phase_offsets = xp.angle(fit_params)
                amp_corr = xp.abs(fit_params)
            else:
                phase_offsets = np.ones((1,) * self.nextra_dims + (self.npatterns, 1, 1))
                amp_corr = np.ones(phase_offsets.shape)

        self.phase_offsets = phase_offsets

        if correct_amplitudes:
            self.amp_corr = amp_corr
        else:
            self.amp_corr = np.ones(amp_corr.shape)

        self.holograms_ft = holograms_ft * da.exp(1j * self.phase_offsets) * self.amp_corr

        # #########################
        # define powers
        # #########################
        # self.powers_rms = da.sqrt(da.mean(da.abs(self.imgs_raw) ** 2, axis=(-1, -2)))
        # if self.imgs_raw_bg is not None:
        #     self.powers_rms_bg = da.sqrt(da.mean(da.abs(self.imgs_raw_bg) ** 2, axis=(-1, -2)))

        self.e_powers_rms = da.sqrt(da.mean(da.abs(self.holograms_ft) ** 2, axis=(-1, -2))) / np.prod(self.holograms_ft.shape[-2:])
        self.e_powers_rms_bg = da.sqrt(da.mean(da.abs(self.holograms_ft_bg) ** 2, axis=(-1, -2))) / np.prod(self.holograms_ft_bg.shape[-2:])

    def reconstruct_n(self,
                      mode: str = "rytov",
                      solver: str = "fista",
                      scattered_field_regularization: float = 50,
                      niters: int = 100,
                      reconstruction_regularizer: float = 0.1,
                      dxy_sampling_factor: float = 1.,
                      dz_sampling_factor: float = 1.,
                      z_fov: float = 20,
                      nbin: int = 1,
                      mask: Optional[np.ndarray] = None,
                      tau_tv: float = 0,
                      tau_lasso: float = 0,
                      use_imaginary_constraint: bool = True,
                      use_real_constraint: bool = False,
                      interpolate_model: bool = True,
                      step: float = 1e-5,
                      stochastic_descent: bool = False,
                      verbose: bool = False,
                      compute_cost: bool = False,
                      use_gpu: bool = False):
        """

        @param mode: "born", "rytov", or "bpm"
        @param solver: "naive" or "fista"
        @param scattered_field_regularization:
        @param niters: number of iterations
        @param reconstruction_regularizer:
        @param dxy_sampling_factor:
        @param dz_sampling_factor:
        @param z_fov:
        @param nbin: for BPM, bin image by this factor
        @param mask:
        @param tau_tv:
        @param tau_lasso:
        @param use_imaginary_constraint:
        @param use_real_constraint:
        @param interpolate_model:
        @param step: ignored unless mode = "bpm"
        @param stochastic_descent:
        @param verbose:
        @param compute_cost:
        @param use_gpu:
        @return:
        """

        if use_gpu and _gpu_available:
            xp = cp
        else:
            xp = np

        if nbin != 1 and mode != "bpm":
            warnings.warn(f"nbin={nbin:d}, but only nbin=1 is supported for mode {mode:s}")

        # ############################
        # set grid sampling info
        # ############################
        beam_frqs = self.get_beam_frqs()
        mean_beam_frqs = [np.mean(f, axis=tuple(range(self.nextra_dims))) for f in beam_frqs]

        # get sampling so can determine new chunk sizes
        drs_v, v_size = get_reconstruction_sampling(self.no,
                                                    self.na_detection,
                                                    self.na_excitation,
                                                    self.wavelength,
                                                    self.dxy,
                                                    self.holograms_ft.shape[-2:],
                                                    z_fov,
                                                    dz_sampling_factor=dz_sampling_factor,
                                                    dxy_sampling_factor=dxy_sampling_factor)

        # if using BPM, much easier if keep pixel size the same
        if mode == "bpm":
            drs_v = (drs_v[0], self.dxy, self.dxy)
            v_size = (v_size[0], self.holograms_ft.shape[-2], self.holograms_ft.shape[-1])
            
        # must ensure chunks for reconstruction use full pattern sets
        new_chunks = list(self.holograms_ft.chunksize)
        new_chunks[-3] = self.npatterns

        # ############################
        # compute scattered field
        # ############################
        if use_gpu:
            self.holograms_ft = da.map_blocks(xp.asarray,
                                              self.holograms_ft,
                                              dtype=complex,
                                              meta=xp.array(())
                                              )
            self.holograms_ft_bg = da.map_blocks(xp.asarray,
                                                 self.holograms_ft_bg,
                                                 dtype=complex,
                                                 meta=xp.array(())
                                                 )

        # ############################
        # define different scattered field options
        # ############################
        self.efield_scattered_ft = da.map_blocks(get_scattered_field,
                                                 self.holograms_ft,
                                                 self.holograms_ft_bg,
                                                 scattered_field_regularization,
                                                 dtype=complex,
                                                 meta=xp.array(())
                                                 )

        self.phi_rytov_ft = da.map_blocks(get_rytov_phase,
                                          self.holograms_ft,
                                          self.holograms_ft_bg,
                                          scattered_field_regularization,
                                          dtype=complex,
                                          meta=xp.array(())
                                          )

        # rechunk efields since must operate on all patterns at once to get 3D volume
        if mode == "born":
            eraw_start = da.rechunk(self.efield_scattered_ft, chunks=new_chunks)
        elif mode == "rytov" or mode == "bpm":
            eraw_start = da.rechunk(self.phi_rytov_ft, chunks=new_chunks)
        else:
            raise ValueError(f"'mode' must be 'born', 'rytov', or 'bpm', but was '{mode:s}'")

        # ############################
        # scattering potential from measured data
        # ############################
        # reconstruction starting point ... put info back in right place in Fourier space

        if mode == "born" or mode == "rytov":
            start_mode = mode
        else:
            start_mode = "rytov"

        # Take first frequencies and assume these are only ones present
        # todo: in case of multiple beam frequencies, not clear what to do exactly.
        mean_beam_frqs_first = np.stack([f[0] for f in mean_beam_frqs], axis=0)

        v_fts_start = da.map_blocks(reconstruction,
                                    eraw_start,
                                    mean_beam_frqs_first[..., 0],
                                    mean_beam_frqs_first[..., 1],
                                    mean_beam_frqs_first[..., 2],
                                    self.no,
                                    self.na_detection,
                                    self.wavelength,
                                    self.dxy,
                                    drs_v,
                                    v_size,
                                    regularization=reconstruction_regularizer,
                                    mode=start_mode,
                                    no_data_value=0,
                                    dtype=complex,
                                    meta=xp.array(()),
                                    chunks=eraw_start.chunksize[:-3] + v_size)

        if mode == "born" or mode == "rytov":
            if solver == "fista":
                # define forward model
                nmax_multiplex = np.max([len(f) for f in mean_beam_frqs])
                # nth entry in list is nth set of demultiplexed frequencies
                # for images which don't have enough multiplexed frequencies, replaced by inf
                mean_beam_frqs_demultiplex = [np.stack([f[ii] if len(f) > ii else np.array([np.inf, np.inf, np.inf]) for f in mean_beam_frqs], axis=0)
                                              for ii in range(nmax_multiplex)]

                models = []
                for ii in range(len(mean_beam_frqs_demultiplex)):
                    model_now, _ = fwd_model_linear(mean_beam_frqs_demultiplex[ii][..., 0],
                                                    mean_beam_frqs_demultiplex[ii][..., 1],
                                                    mean_beam_frqs_demultiplex[ii][..., 2],
                                                    self.no,
                                                    self.na_detection,
                                                    self.wavelength,
                                                    (self.ny, self.nx),
                                                    (self.dxy, self.dxy),
                                                    v_size,
                                                    drs_v,
                                                    mode=mode,
                                                    interpolate=interpolate_model)
                    models.append(model_now)
                # full model is sum of these submodels
                model = sum(models)

                # set step size. Lipschitz constant of \nabla cost is given by the largest singular value of linear model
                # (also the square root of the largest eigenvalue of model^t * model)
                u, s, vh = sp.linalg.svds(model, k=1, which='LM')
                # todo: should also add a factor of 0.5 but maybe doesn't matter
                lipschitz_estimate = s ** 2 / (self.npatterns * self.ny * self.nx) / (self.ny * self.nx)
                self.step = float(1 / lipschitz_estimate)

                # todo: add masks
                def recon(v_fts, efield_scattered_ft, no, wavelength, step):
                    nextra_dims = v_fts.ndim - 3

                    results = grad_descent(v_fts.squeeze(axis=tuple(range(nextra_dims))),
                                           efield_scattered_ft.squeeze(axis=tuple(range(nextra_dims))),
                                           model,
                                           no,
                                           wavelength,
                                           step,
                                           niters=niters,
                                           use_fista=True,
                                           tau_tv=tau_tv,
                                           tau_lasso=tau_lasso,
                                           use_imaginary_constraint=use_imaginary_constraint,
                                           use_real_constraint=use_real_constraint,
                                           masks=None,
                                           verbose=verbose,
                                           compute_cost=compute_cost)

                    v_out_ft = results["x"].reshape((1,) * nextra_dims + v_size)

                    if isinstance(v_out_ft, cp.ndarray) and _gpu_available:
                        xp = cp
                    else:
                        xp = np

                    # inverse FFT
                    v = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(v_out_ft, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))

                    return get_n(v, no, wavelength)

            else:
                self.step = None
                # ############################
                # fill in fourier space with constraint algorithm
                # ############################
                def recon(v_fts_start, efield_scattered_ft, no, wavelength, step):
                    v_out_ft = apply_n_constraints(v_fts_start,
                                                   no,
                                                   wavelength,
                                                   n_iterations=niters,
                                                   beta=0.5,
                                                   use_raar=False,
                                                   require_real_part_greater_bg=use_real_constraint)

                    if isinstance(v_out_ft, cp.ndarray) and _gpu_available:
                        xp = cp
                    else:
                        xp = np

                    # inverse FFT
                    v = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(v_out_ft, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))

                    return get_n(v, no, wavelength)

            # get refractive index
            n = da.map_blocks(recon,
                              v_fts_start,  # initial guess
                              eraw_start,  # data
                              #masks, # todo: add
                              self.no,
                              self.wavelength,
                              self.step,
                              chunks=v_fts_start.chunksize[:-3] + v_size,
                              dtype=complex,
                              meta=xp.array(()),
                              )

            # affine transformation from reconstruction coordinates to pixel indices
            # for reconstruction, using FFT induced coordinates, i.e. zero is at array index (ny // 2, nx // 2)
            # for matrix, using image coordinates (0, 1, ..., N - 1)
            # note, due to order of operations -n//2 =/= - (n//2) when nx is odd
            xform_recon_pix2coords = affine.params2xform([drs_v[-1], 0, -(v_size[-1] // 2) * drs_v[-1],
                                                          drs_v[-2], 0, -(v_size[-2] // 2) * drs_v[-2]])

        elif mode == "bpm":
            if solver != "fista":
                raise ValueError(f"using mode='bpm', so only solver='fista' is supported, but solver='{solver:s}'")

            # position z=0 in the middle of the volume
            self.dz_final = -drs_v[0] * ((v_size[0] - 1) - v_size[0] // 2 + 0.5)

            # correct size for binning
            n_size = (v_size[0],) + (v_size[1] // nbin, v_size[2] // nbin)
            drs_n = (drs_v[0], drs_v[1] * nbin, drs_v[2] * nbin)

            # generate ATF ... ultimately want to do this based on pupil function defined in init
            fx_atf = fft.fftshift(fft.fftfreq(n_size[-1], ))[None, :]
            fy_atf = fft.fftshift(fft.fftfreq(n_size[-2]))[:, None]
            atf = (np.sqrt(fx_atf ** 2 + fy_atf ** 2) <= self.fmax).astype(complex)

            apodization_n = xp.outer(xp.asarray(tukey(n_size[-2], alpha=0.1)),
                                     xp.asarray(tukey(n_size[-1], alpha=0.1)))
            # apodization_n = None
            self.step = step

            def recon(v_ft, efields_ft, efields_bg_ft, no, wavelength, dz_final, atf, apod, step):

                if isinstance(v_ft, cp.ndarray) and _gpu_available:
                    xp = cp
                else:
                    xp = np

                # expect v is array of size 1 x 1 x ... x 1 x nz x ny x nx
                v = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(v_ft, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))
                # expect array of size 1 x 1 ... x 1 x npatterns x ny x nx
                efields = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(efields_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
                efields_bg = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(efields_bg_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

                # bin if desired
                v = camera.bin(v, [nbin, nbin], mode="mean")
                efields = camera.bin(efields, [nbin, nbin], mode="mean")
                efields_bg = camera.bin(efields_bg, [nbin, nbin], mode="mean")

                nextra_dims = v.ndim - 3
                vsize = v.shape[-3:]

                results = grad_descent_prop_model(v.squeeze(axis=tuple(range(nextra_dims))),
                                                  efields.squeeze(axis=tuple(range(nextra_dims))),
                                                  efields_bg.squeeze(axis=tuple(range(nextra_dims))),
                                                  no,
                                                  wavelength,
                                                  dz_final,
                                                  drs_n,
                                                  model="bpm",
                                                  step=step,
                                                  niters=niters,
                                                  use_fista=True,
                                                  tau_tv=tau_tv,
                                                  tau_lasso=tau_lasso,
                                                  use_imaginary_constraint=use_imaginary_constraint,
                                                  use_real_constraint=use_real_constraint,
                                                  atf=atf,
                                                  masks=None,
                                                  verbose=verbose,
                                                  stochastic_descent=stochastic_descent,
                                                  apodization=apod,
                                                  compute_cost=compute_cost)

                n = results["x"].reshape((1,) * nextra_dims + vsize)

                return n

            n = da.map_blocks(recon,
                              v_fts_start,  # initial guess
                              da.rechunk(self.holograms_ft, chunks=new_chunks),  # data
                              da.rechunk(self.holograms_ft_bg, chunks=new_chunks),  # background
                              self.no,
                              self.wavelength,
                              self.dz_final,
                              atf,
                              apodization_n,
                              self.step,
                              chunks=(1,) * self.nextra_dims + n_size,
                              dtype=complex,
                              meta=xp.array(())
                              )

            # affine transformation from reconstruction coordinates to pixel indices
            # coordinates in finer coordinates
            x = (xp.arange(v_size[-1]) - (v_size[-1] // 2)) * drs_v[-1]
            y = (xp.arange(v_size[-2]) - (v_size[-2] // 2)) * drs_v[-2]
            xb = camera.bin(x, [nbin], mode="mean")
            yb = camera.bin(y, [nbin], mode="mean")

            xform_recon_pix2coords = affine.params2xform([drs_n[-1], 0, float(xb[0]),
                                                          drs_n[-2], 0, float(yb[0])])

        else:
            raise ValueError(f"mode must be ..., but was {mode:s}")

        self.reconstruction_settings.update({"mode": mode,
                                             "solver": solver,
                                             "scattered_field_regularization": scattered_field_regularization,
                                             "niters": niters,
                                             "reconstruction_regularizer": reconstruction_regularizer,
                                             "dxy_sampling_factor": dxy_sampling_factor,
                                             "dz_sampling_factor": dz_sampling_factor,
                                             "z_fov": z_fov,
                                             "nbin": nbin,
                                             "tau_tv": tau_tv,
                                             "tau_lasso": tau_lasso,
                                             "use_imaginary_constraint": use_imaginary_constraint,
                                             "use_real_constraint": use_real_constraint,
                                             "interpolate_model": interpolate_model,
                                             "step": step,
                                             "stochastic_descent": stochastic_descent,
                                             "use_gpu": use_gpu
                                             }
                                            )

        return n, v_fts_start, drs_v, xform_recon_pix2coords


    def plot_interferograms(self,
                            nmax_to_plot: int,
                            save_dir: Optional[str] = None):
        """
        Plot nmax interferograms

        @param nmax_to_plot:
        @param save_dir:
        @return:
        """
        if not hasattr(self, "efield_scattered_ft"):
            raise ValueError()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        nmax_to_plot = np.min([nmax_to_plot, np.prod(self.holograms_ft.shape[:-2])])

        # todo: correct for new angles
        beam_frqs = self.get_beam_frqs()
        theta, phi = get_angles(beam_frqs, self.no, self.wavelength)

        def plot_interferograms(index, save_dir):
            unraveled_inds = np.unravel_index(index, self.holograms_ft.shape[:-2])

            desc = "_".join([f"{self.axes_names[ii]:s}={unraveled_inds[ii]:d}" for ii in range(len(unraveled_inds))])

            # with ProgressBar():
            efields_ft_temp, efields_scattered_ft_temp, efields_bg_ft_temp = \
                dask.compute(self.holograms_ft[unraveled_inds],
                             self.efield_scattered_ft[unraveled_inds],
                             self.holograms_ft_bg[unraveled_inds])

            beam_ind = unraveled_inds[-1]

            ttl = f"{desc.replace('_', ', '):s}\n" \
                  f"theta={theta[beam_ind] * 180 / np.pi:.2f}deg, phi={phi[beam_ind] * 180 / np.pi:.2f}deg," \
                  f" beam freq = ({beam_frqs[beam_ind, 0]:.3f}, {beam_frqs[beam_ind, 1]:.3f}, {beam_frqs[beam_ind, 2]:.3f})\n" \
                  f"holography reference freq= ({self.reference_frq[0]:.3f}, {self.reference_frq[1]:.3f})"

            figh = plot_scattered_angle(efields_ft_temp,
                                        efields_bg_ft_temp,
                                        efields_scattered_ft_temp,
                                        beam_frqs[beam_ind],
                                        self.reference_frq,
                                        self.fmax * 2,
                                        self.dxy,
                                        title=ttl)

            figh.savefig(Path(save_dir, f"{desc:s}_interferogram.png"))
            plt.close(figh)

        # collect delayed function calls
        results_interferograms = []
        for aa in range(nmax_to_plot):
            results_interferograms.append(dask.delayed(plot_interferograms(aa, save_dir)))

        # compute results = save/display figures
        print(f"displaying {len(results_interferograms):d} interferograms")
        with ProgressBar():
            dask.compute(*results_interferograms)

    def plot_frqs(self,
                  index: tuple[int],
                  time_axis: int = 1,
                  figsize: tuple[float] = (30., 8.),
                  **kwargs):
        """

        @param index: should be of length self.nextra_dims - 1. Index along these axes, but ignoring whichever
        axes is the time axis. So e.g. if the axis are position x time x z x parameter then time_axis = 1 and the index
        could be (2, 1, 0) which would selection position 2, z 1, parameter 0.
        @param time_axis:
        @param figsize:
        @param kwargs: passed through to matplotlib.pyplot.figure
        @return:
        """

        if len(index) != (self.nextra_dims - 1):
            raise ValueError(f"index={index} should have length self.nextra_dims - 1={self.nextra_dims - 1}")

        # logic for slicing out desired index
        slices = []
        index_counter = 0
        for ii in range(self.nextra_dims):
            if ii != time_axis:
                slices.append(slice(index[index_counter], index[index_counter] + 1))
            else:
                slices.append(slice(None))

        # add slices for multiplexed dimension and xy dimension
        slices = tuple(slices + [slice(None), slice(None)])
        squeeze_axes = tuple([ii for ii in range(self.nextra_dims) if ii != time_axis])

        ref_slices = slices[:-1]

        # ####################################
        # hologram frequencies
        # ####################################

        # each element of list should have shape ntimes x nmultiplex x 2
        hologram_frqs_mean = [np.mean(f, axis=1, keepdims=True) for f in self.hologram_frqs]
        hgram_frq_diffs = [(f - g)[slices].squeeze(axis=squeeze_axes)
                           for f, g in zip(self.hologram_frqs, hologram_frqs_mean)]
        # stack all hologram frqs
        hgram_frq_diffs = np.concatenate(hgram_frq_diffs, axis=1)

        # shape = ntimes x 2
        ref_frq_diffs = (self.reference_frq - np.mean(self.reference_frq, axis=1, keepdims=True))[ref_slices].squeeze(squeeze_axes)

        # plot
        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle(f"index={index}\nfrequency variation versus time")

        # plot frequency differences
        ax = figh.add_subplot(1, 3, 1)
        ax.plot(np.linalg.norm(hgram_frq_diffs, axis=-1) / self.dfx)
        ax.set_xlabel("time step")
        ax.set_ylabel("(frequency - mean) / dfx")
        ax.set_title("hologram frequency deviation amplitude")
        ax.legend([f"{ii:d}" for ii in range(self.npatterns)])

        # plot angles
        angles_unwrapped = np.unwrap(np.angle(hgram_frq_diffs[..., 0] + 1j * hgram_frq_diffs[..., 1]))

        ax = figh.add_subplot(1, 3, 2)
        ax.plot(angles_unwrapped)
        ax.set_xlabel("time step")
        ax.set_ylabel("angle (rad)")
        ax.set_title("hologram frequency deviation rotation")

        # plot mean frequency differences
        ax = figh.add_subplot(1, 3, 3)
        ax.plot(np.linalg.norm(ref_frq_diffs, axis=-1) / self.dfx)
        ax.set_xlabel("time step")
        ax.set_ylabel("(frequency norm - mean) / dfx")
        ax.set_title("reference frequency deviation amplitude")

        return figh

    def plot_phases(self,
                    index: tuple[int],
                    time_axis: int = 1,
                    figsize: tuple[float] = (30., 8.),
                    **kwargs):
        """
        Plot phase drift fits versus time for a single slice

        @param index: should be of length self.nextra_dims - 1. Index along these axes, but ignoring whichever
        axes is the time axis. So e.g. if the axis are position x time x z x parameter then time_axis = 1 and the index
        could be (2, 1, 0) which would selection position 2, z 1, parameter 0.
        @param time_axis:
        @param figsize:
        @param kwargs: passed through to matplotlib.pyplot.figure
        @return: figh
        """

        if len(index) != (self.nextra_dims - 1):
            raise ValueError(f"index={index} should have length self.nextra_dims - 1={self.nextra_dims - 1}")

        # logic for slicing out desired index
        slices = []
        index_counter = 0
        for ii in range(self.nextra_dims):
            if ii != time_axis:
                slices.append(slice(index[index_counter], index[index_counter] + 1))
            else:
                slices.append(slice(None))

        # add slices for patterns, y, x
        slices = tuple(slices + [slice(None), slice(None), slice(None)])
        squeeze_axes = tuple([ii for ii in range(self.nextra_dims) if ii != time_axis]) + (-1, -2)

        # get slice of phases
        ph = np.unwrap(self.phase_offsets[slices].squeeze(axis=squeeze_axes), axis=0)
        ph_bg = np.unwrap(self.phase_offsets_bg[slices].squeeze(axis=squeeze_axes), axis=0)

        # plot
        figh2 = plt.figure(figsize=figsize, **kwargs)
        figh2.suptitle(f"index={index}\nphase variation versus time")

        ax = figh2.add_subplot(1, 2, 1)
        ax.set_title("Phases")
        ax.plot(ph)
        ax.set_xlabel("time step")
        ax.set_ylabel("phase (rad)")

        ax = figh2.add_subplot(1, 2, 2)
        ax.set_title("Background phases")
        ax.plot(ph_bg)
        ax.set_xlabel("time step")
        ax.set_ylabel("phase (rad)")

        return figh2

    def plot_powers(self,
                    index: tuple[int],
                    time_axis: int = 1,
                    figsize: tuple[float] = (30., 8.),
                    **kwargs
                    ):
        """
        Plot hologram intensity

        @param index:
        @param time_axis:
        @param figsize:
        @param kwargs:
        @return figh, epower, epower_bg:
        """

        if len(index) != (self.nextra_dims - 1):
            raise ValueError(f"index={index} should have length self.nextra_dims - 1={self.nextra_dims - 1}")

            # logic for slicing out desired index
        slices = []
        index_counter = 0
        for ii in range(self.nextra_dims):
            if ii != time_axis:
                slices.append(slice(index[index_counter], index[index_counter] + 1))
            else:
                slices.append(slice(None))

        # add slices for patterns
        slices = tuple(slices + [slice(None)])
        squeeze_axes = tuple([ii for ii in range(self.nextra_dims) if ii != time_axis])

        # get slice of phases
        with ProgressBar():
            computed = dask.compute([self.e_powers_rms[slices].squeeze(axis=squeeze_axes),
                                     self.e_powers_rms_bg[slices].squeeze(axis=squeeze_axes)],
                                    scheduler="threads")

        epowers, epowers_bg = computed[0]
        if isinstance(epowers, cp.ndarray):
            epowers = epowers.get()
        if isinstance(epowers_bg, cp.ndarray):
            epowers_bg = epowers_bg.get()

        # slice amplitudes
        amp = self.amp_corr[slices].squeeze(axis=squeeze_axes + (-1, -2))
        amp_bg = self.amp_corr_bg[slices].squeeze(axis=squeeze_axes + (-1, -2))

        # plot
        figh2 = plt.figure(figsize=figsize, **kwargs)
        figh2.suptitle(f"index={index}\nhologram magnitude variation versus time")

        ax = figh2.add_subplot(2, 2, 1)
        ax.set_title("|E| RMS average")
        ax.plot(epowers)
        ax.set_xlabel("time step")
        ax.set_ylabel("|E|")

        ax = figh2.add_subplot(2, 2, 2)
        ax.set_title("|Ebg| RMS average")
        ax.plot(epowers_bg)
        ax.set_xlabel("time step")
        ax.set_ylabel("|E|")

        ax = figh2.add_subplot(2, 2, 3)
        ax.set_title("E fit amplitude")
        ax.plot(amp)
        ax.set_xlabel("time step")
        ax.set_ylabel("amp")

        ax = figh2.add_subplot(2, 2, 4)
        ax.set_title("Ebg bit amplitude")
        ax.plot(amp_bg)
        ax.set_xlabel("time step")
        ax.set_ylabel("amplitude")

        return figh2, epowers, epowers_bg

    def show_image(self,
                   index: tuple[int],
                   figsize: tuple[float] = (24, 10),
                   gamma: float = 0.1,
                   **kwargs):
        """
        display raw image

        @param index:
        @param figsize:
        @param gamma:
        @return:
        """
        extent = [self.x[0] - 0.5 * self.dxy, self.x[-1] + 0.5 * self.dxy,
                  self.y[-1] + 0.5 * self.dxy, self.y[0] - 0.5 * self.dxy]

        extent_f = [self.fxs[0] - 0.5 * self.dfx, self.fxs[-1] + 0.5 * self.dxy,
                    self.fys[-1] + 0.5 * self.dfy, self.fys[0] - 0.5 * self.dfy]

        # ######################
        #
        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle(f"{index}, {self.axes_names}")
        grid = figh.add_gridspec(nrows=2, ncols=4, width_ratios=[1, 0.1, 1, 0.1])

        img_now = self.imgs_raw[index].compute()
        img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img_now)))

        # raw image
        ax = figh.add_subplot(grid[0, 0])
        im = ax.imshow(img_now, extent=extent, cmap="bone")
        ax.set_title("raw image")

        ax = figh.add_subplot(grid[0, 1])
        plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[0, 2])
        im = ax.imshow(np.abs(img_ft), extent=extent_f,
                       norm=PowerNorm(gamma=gamma), cmap="bone")
        ax.set_title("raw FT")

        ax = figh.add_subplot(grid[0, 3])
        plt.colorbar(im, cax=ax)

        # hologram
        try:
            h_f = self.holograms_ft[index].compute()
            if isinstance(h_f, cp.ndarray) and _gpu_available:
                h_f = h_f.get()

            ax = figh.add_subplot(grid[1, 2])
            ax.set_title("hologram")
            im = ax.imshow(np.abs(h_f), extent=extent_f,
                           norm=PowerNorm(gamma=gamma), cmap="bone")
        except Exception as e:
            print(e)

        return figh


def soft_threshold(t: float,
                   x: array) -> array:
    """
    Softmax function, which is the proximal operator for the LASSO (L1 regularization) problem

    @param t: softmax parameter
    @param x: array to take softmax of
    @return x_out:
    """
    # x_out = np.array(x, copy=True)
    x_out = x.copy()
    x_out[x > t] -= t
    x_out[x < -t] += t
    x_out[abs(x) <= t] = 0

    return x_out


def grad_descent(v_ft_start: array,
                 e_measured_ft: array,
                 model: csr_matrix,
                 no: float,
                 wavelength: float,
                 step: float,
                 niters: int = 100,
                 use_fista: bool = True,
                 tau_tv: float = 0.,
                 tau_lasso: float = 0.,
                 use_imaginary_constraint: bool = True,
                 use_real_constraint: bool = False,
                 masks: Optional[array] = None,
                 verbose: bool = False,
                 compute_cost: bool = True) -> dict:
    """
    Perform gradient descent using a linear model

    @param v_ft_start:
    @param e_measured_ft: npatterns x ny x nx array
    @param model: CSR matrix mapping from vft to electric field
    @param step: step-size used in gradient descent
    @param niters: number of iterations
    @param use_fista:
    @param tau_tv:
    @param tau_lasso:
    @param use_imaginary_constraint:
    @param use_real_constraint:
    @param masks: # todo: add mask to remove points which we don't want to consider
    @param verbose:
    @param compute_cost:
    @return results: dict
    """
    # put on gpu optionally
    use_gpu = isinstance(v_ft_start, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
        model = sp_gpu.csr_matrix(model)
        denoise_tv = denoise_tv_chambolle_gpu
    else:
        xp = np
        denoise_tv = denoise_tv_chambolle

    v_shape = v_ft_start.shape

    # if using GPU, avoid need to transfer this back
    v_ft_start_preserved = v_ft_start.copy()

    v_ft_start = xp.asarray(v_ft_start)
    e_measured_ft = xp.asarray(e_measured_ft)
    step = xp.asarray(step)

    def ift(m): return xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(m, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))
    def ft(m): return xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(m, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))

    # prepare arguments
    npatterns, ny, nx = e_measured_ft.shape

    # model information
    model_csc = model.tocsc(copy=True)

    # cost functions and etc
    def cost(v_ft):
        # mean cost per (real-space) pixel
        # divide by nxy**2 again because using FT's and want mean cost in real space
        # recall that \sum_r |f(r)|^2 = 1/N \sum_k |f(k)|^2
        # NOTE: to get total cost, must take mean over patterns

        # operations should be written as methods of arrays (as opposed to numpy functions)
        # to be portable between numpy,cupy, dask.array
        costs = (abs(model.dot(v_ft.ravel()) - e_measured_ft.ravel()) ** 2).reshape([npatterns, ny, nx]).mean(axis=(-1, -2)) / 2 / (nx * ny)

        if use_gpu:
            costs = costs.get()

        return costs

    def grad(v_ft):
        # first division is average
        # second division converts Fourier space to real-space sum
        # factor of 0.5 in cost function killed by derivative factor of 2
        dc_dm = (model.dot(v_ft.ravel()) - e_measured_ft.ravel()) / (npatterns * ny * nx) / (ny * nx)
        dc_dv = (dc_dm[None, :].conj() * model_csc)[0].conj()

        return dc_dv

    # initialize
    tstart = time.perf_counter()
    costs = np.zeros((niters + 1, npatterns)) * np.nan
    v_ft = v_ft_start.ravel()
    q_last = 1

    if compute_cost:
        costs[0] = cost(v_ft)

    timing_names = ["iteration", "grad calculation", "ifft", "TV", "L1", "positivity", "fft", "fista", "cost"]
    timing = np.zeros((niters, len(timing_names)))

    for ii in range(niters):
        # gradient descent
        tstart_grad = time.perf_counter()
        dc_dv = grad(v_ft)
        v_ft -= step * dc_dv

        tend_grad = time.perf_counter()

        # FT so can apply proximal operators in real space
        tstart_fft = time.perf_counter()
        v = ift(v_ft.reshape(v_shape))
        tend_fft = time.perf_counter()

        # apply TV proximity operators
        # todo: better to apply this to n directly?
        tstart_tv = time.perf_counter()
        if tau_tv != 0:
            v_real = denoise_tv(v.real, weight=tau_tv, channel_axis=None)
            v_imag = denoise_tv(v.imag, weight=tau_tv, channel_axis=None)
        else:
            v_real = v.real
            v_imag = v.imag

        tend_tv = time.perf_counter()

        # apply L1 proximity operators
        tstart_l1 = time.perf_counter()

        if tau_lasso != 0:
            v_real = soft_threshold(tau_lasso, v_real)
            v_imag = soft_threshold(tau_lasso, v_imag)

        tend_l1 = time.perf_counter()

        # apply projection onto constraints (proximity operators)
        tstart_constraints = time.perf_counter()

        if use_imaginary_constraint:
            v_imag[v_imag > 0] = 0

        if use_real_constraint:
            # actually ... this is no longer right if there is any imaginary part
            v_real[v_real > 0] = 0

        tend_constraints = time.perf_counter()

        # ft back to Fourier space
        tstart_fft_back = time.perf_counter()

        v_ft_prox = ft(v_real + 1j * v_imag).ravel()

        tend_fft_back = time.perf_counter()

        # update step
        tstart_update = time.perf_counter()

        q_now = 0.5 * (1 + np.sqrt(1 + 4 * q_last ** 2))

        if ii == 0 or ii == (niters - 1) or not use_fista:
            v_ft = v_ft_prox
        else:
            v_ft = v_ft_prox + (q_last - 1) / q_now * (v_ft_prox - v_ft_prox_last)

        tend_update = time.perf_counter()

        # compute cost
        tstart_err = time.perf_counter()

        if compute_cost:
            costs[ii + 1] = cost(v_ft)

        tend_err = time.perf_counter()

        # update for next gradient-descent/FISTA iteration
        q_last = q_now
        v_ft_prox_last = v_ft_prox

        # print information
        tend_iter = time.perf_counter()
        timing[ii, 0] = tend_iter - tstart_grad
        timing[ii, 1] = tend_grad - tstart_grad
        timing[ii, 2] = tend_fft - tstart_fft
        timing[ii, 3] = tend_tv - tstart_tv
        timing[ii, 4] = tend_l1 - tstart_l1
        timing[ii, 5] = tend_constraints - tstart_constraints
        timing[ii, 6] = tend_fft_back - tstart_fft_back
        timing[ii, 7] = tend_update - tstart_update
        timing[ii, 8] = tend_err - tstart_err

        if verbose:
            print(
                f"iteration {ii + 1:d}/{niters:d},"
                f" cost={np.mean(costs[ii + 1]):.3g},"
                f" grad={tend_grad - tstart_grad:.2f}s,"
                f" fft={tend_fft - tstart_fft:.2f}s,"
                f" TV={tend_tv - tstart_tv:.2f}s,"
                f" projection={tend_constraints - tstart_constraints:.2f}s,"
                f" fft={tend_fft_back - tstart_fft_back:.2f}s,"
                f" update={tend_update - tstart_update:.2f}s,"
                f" cost={tend_err - tstart_err:.2f}s,"
                f" iter={tend_iter - tstart_grad:.2f}s,"
                f" total={time.perf_counter() - tstart:.2f}s",
                end="\r")

    # store summary results
    results = {"step_size": step,
               "niterations": niters,
               "use_fista": use_fista,
               "use_gpu": use_gpu,
               "tau_tv": tau_tv,
               "tau_l1": tau_lasso,
               "use_imaginary_constraint": use_imaginary_constraint,
               "use_real_constraint": use_real_constraint,
               "timing": timing,
               "timing_column_names": timing_names,
               "costs": costs,
               "x_init": v_ft_start_preserved,
               "x": v_ft
               }

    return results


def grad_descent_prop_model(v_start: array,
                            e_measured: array,
                            e_measured_bg: array,
                            no: float,
                            wavelength: float,
                            dz_final: float,
                            drs: float,
                            model: str,
                            step: float,
                            niters: int = 100,
                            use_fista: bool = True,
                            tau_tv: float = 0.,
                            tau_lasso: float = 0.,
                            use_imaginary_constraint: bool = True,
                            use_real_constraint: bool = False,
                            masks: Optional[array] = None,
                            verbose: bool = False,
                            compute_cost: bool = False,
                            stochastic_descent: bool = False,
                            atf: array = 1.,
                            apodization: Optional[array] = None) -> dict:
    """
    Suppose we have a 3D grid with nz voxels along the propagation direction. We define the electric field
    at the points before and after each voxel, and in an additional plane to account for the imaging. So we have
    nz + 2 electric field planes.

    @param v_start:
    @param e_measured:
    @param e_measured_bg:
    @param no:
    @param wavelength:
    @param dz_final:
    @param drs:
    @param model:
    @param step:
    @param niters:
    @param use_fista:
    @param tau_tv:
    @param tau_lasso:
    @param use_imaginary_constraint:
    @param use_real_constraint:
    @param masks:
    @param verbose: print iteration info
    @param compute_cost:
    @param stochastic_descent:
    @param atf:
    @param apodization:
    @return:
    """

    # put on gpu optionally
    use_gpu = isinstance(v_start, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
        denoise_tv = denoise_tv_chambolle_gpu
    else:
        xp = np
        denoise_tv = denoise_tv_chambolle

    v_start = xp.asarray(v_start)
    n_start = get_n(v_start, no, wavelength)
    e_measured = xp.asarray(e_measured)
    e_measured_bg = xp.asarray(e_measured_bg)
    step = xp.asarray(step)

    npatterns, ny, nx = e_measured.shape
    # nz, ny, nx = v_start.shape[-3:]
    nz = v_start.shape[-3]
    dz = drs[0]
    # backpropogate homogeneous
    efield_start = field_prop.propagate_homogeneous(e_measured_bg,
                                                    -np.array([float(dz_final) + float(dz) * nz]),
                                                    no,
                                                    drs[1:],
                                                    wavelength)[..., 0, :, :]

    def cost(n):
        forward = field_prop.propagate_inhomogeneous(efield_start,
                                                     n,
                                                     no,
                                                     drs,
                                                     wavelength,
                                                     dz_final,
                                                     atf=atf,
                                                     apodization=apodization,
                                                     model=model)

        costs = 0.5 * (abs(forward[:, -1, :, :] - e_measured) ** 2).mean(axis=(-1, -2))

        if use_gpu:
            costs = costs.get()

        return costs

    def grad(n):
        # forward propagation
        e_full = field_prop.propagate_inhomogeneous(efield_start,
                                                    n,
                                                    no,
                                                    drs,
                                                    wavelength,
                                                    dz_final,
                                                    atf=atf,
                                                    apodization=apodization,
                                                    model=model)
        # back propagation
        e_back = field_prop.backpropagate_inhomogeneous(e_full[:, -1, :, :] - e_measured,
                                                        n,
                                                        no,
                                                        drs,
                                                        wavelength,
                                                        dz_final,
                                                        atf=atf,
                                                        apodization=apodization,
                                                        model=model)

        # cost function gradient
        dc_dn = -(1j * (2 * np.pi / wavelength) * dz / (nx * ny) * e_back[:, 1:-1, :, :] * e_full[:, 1:-1, :, :].conj())

        return dc_dn

    # initialize
    tstart = time.perf_counter()
    costs = np.zeros((niters + 1, npatterns)) * np.nan
    q_last = 1
    n_start_preserved = xp.array(n_start, copy=True)
    n = n_start

    if compute_cost:
        costs[0] = cost(n)

    timing_names = ["iteration", "grad calculation", "TV", "L1", "positivity", "fista", "cost"]
    timing = np.zeros((niters, len(timing_names)))

    for ii in range(niters):
        # gradient descent
        tstart_grad = time.perf_counter()

        if stochastic_descent:
            # select random subset of angles
            num = random.sample(range(1, npatterns + 1), 1)[0]
            rand_inds = random.sample(range(npatterns), num)
            # obj -= step * xp.mean(grad(obj)[rand_inds], axis=0)

            # todo: in this case I should not compute all gradients, but only the one I need
            # n -= step * grad(n)[ii % npatterns]
            n -= step * xp.mean(grad(n)[rand_inds], axis=0)
        else:
            n -= step * xp.mean(grad(n), axis=0)

        tend_grad = time.perf_counter()

        # apply TV proximity operators
        tstart_tv = time.perf_counter()
        if tau_tv != 0:
            n_real = denoise_tv(n.real, weight=tau_tv, channel_axis=None)
            n_imag = denoise_tv(n.imag, weight=tau_tv, channel_axis=None)
        else:
            n_real = n.real
            n_imag = n.imag

        tend_tv = time.perf_counter()

        # apply L1 proximity operators
        tstart_l1 = time.perf_counter()

        if tau_lasso != 0:
            n_real = soft_threshold(tau_lasso, n_real - no) + no
            n_imag = soft_threshold(tau_lasso, n_imag)

        tend_l1 = time.perf_counter()

        # apply projection onto constraints (proximity operators)
        tstart_constraints = time.perf_counter()

        if use_imaginary_constraint:
            n_imag[n_imag < 0] = 0

        if use_real_constraint:
            # actually ... this is no longer right if there is any imaginary part
            n_real[n_real < no] = no

        tend_constraints = time.perf_counter()

        # prox value
        n_prox = n_real + 1j * n_imag

        # update step
        tstart_update = time.perf_counter()

        q_now = 0.5 * (1 + np.sqrt(1 + 4 * q_last ** 2))

        # todo: is it better to apply the proximal operator last or do FISTA as usual?
        if ii == 0 or ii == (niters - 1) or not use_fista:
            n = n_prox
        else:
            n = n_prox + (q_last - 1) / q_now * (n_prox - n_prox_last)

        tend_update = time.perf_counter()

        # compute cost
        tstart_err = time.perf_counter()

        if compute_cost:
            costs[ii + 1] = cost(n)

        tend_err = time.perf_counter()

        # update for next gradient-descent/FISTA iteration
        q_last = q_now
        n_prox_last = n_prox

        # print information
        tend_iter = time.perf_counter()
        timing[ii, 0] = tend_iter - tstart_grad
        timing[ii, 1] = tend_grad - tstart_grad
        timing[ii, 2] = tend_tv - tstart_tv
        timing[ii, 3] = tend_l1 - tstart_l1
        timing[ii, 4] = tend_constraints - tstart_constraints
        timing[ii, 5] = tend_update - tstart_update
        timing[ii, 6] = tend_err - tstart_err

        if verbose:
            print(
                f"iteration {ii + 1:d}/{niters:d},"
                f" cost={np.mean(costs[ii + 1]):.3g},"
                f" grad={tend_grad - tstart_grad:.2f}s,"                
                f" TV={tend_tv - tstart_tv:.2f}s,"
                f" projection={tend_constraints - tstart_constraints:.2f}s,"                
                f" update={tend_update - tstart_update:.2f}s,"
                f" cost={tend_err - tstart_err:.2f}s,"
                f" iter={tend_iter - tstart_grad:.2f}s,"
                f" total={time.perf_counter() - tstart:.2f}s",
                end="\r")

    # store summary results
    results = {"step_size": step,
               "niterations": niters,
               "use_fista": use_fista,
               "use_gpu": use_gpu,
               "tau_tv": tau_tv,
               "tau_l1": tau_lasso,
               "use_imaginary_constraint": use_imaginary_constraint,
               "use_real_constraint": use_real_constraint,
               "timing": timing,
               "timing_column_names": timing_names,
               "costs": costs,
               "x_init": n_start_preserved,
               "x": n
               }

    return results


def cut_mask(img: array,
             mask: array,
             mask_val: float = 0) -> array:
    """
    Mask points in image and set to a given value. Designed to be used with dask.array map_blocks()

    @param img:
    @param mask:
    @param mask_val: At any position where mask is True, replace the value of img with this value
    @return: img_masked
    """
    if isinstance(img, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    mask = xp.asarray(mask)
    img_masked = xp.array(img, copy=True)

    # if mask does not cover full image, broadcast
    mask = xp.expand_dims(mask, axis=tuple(range(img.ndim - mask.ndim)))
    _, mask = xp.broadcast_arrays(img_masked, mask)

    img_masked[mask] = mask_val

    return img_masked


# helper functions
def get_fz(fx: np.ndarray,
           fy: np.ndarray,
           no: float,
           wavelength: float) -> np.ndarray:
    """
    Get z-component of frequency given fx, fy

    @param fx: nfrqs
    @param fy:
    @param no: index of refraction
    @param wavelength: wavelength
    @return frqs_3d:
    """

    with np.errstate(invalid="ignore"):
        fzs = np.sqrt(no**2 / wavelength ** 2 - fx**2 - fy**2)

    return fzs


def get_angles(frqs: np.ndarray,
               no: float,
               wavelength: float):
    """
    Convert from frequency vectors to angle vectors. Frequency vectors should be normalized to no / wavelength
    @param frqs: (fx, fy, fz), expect |frqs| = no / wavelength
    @param no: background index of refraction
    @param wavelength:
    @return: theta, phi
    """
    frqs = np.atleast_2d(frqs)

    with np.errstate(invalid="ignore"):
        theta = np.array(np.arccos(np.dot(frqs, np.array([0, 0, 1])) / (no / wavelength)))
        theta[np.isnan(theta)] = 0
        phi = np.angle(frqs[:, 0] + 1j * frqs[:, 1])
        phi[np.isnan(phi)] = 0

    return theta, phi


def angles2frqs(no: float,
                wavelength: float,
                theta: np.ndarray,
                phi: np.ndarray):
    """
    Get frequency vector from angles
    @param no:
    @param wavelength:
    @param theta:
    @param phi:
    @return:
    """
    fz = no / wavelength * np.cos(theta)
    fy = no / wavelength * np.sin(theta) * np.sin(phi)
    fx = no / wavelength * np.sin(theta) * np.cos(phi)
    f = np.stack((fx, fy, fz), axis=1)

    return f


def get_global_phase_shifts(imgs: array,
                            ref_imgs: array,
                            thresh: Optional[float] = None) -> np.ndarray:
    """
    Given a stack of images and a reference, determine the phase shifts between images, such that
    imgs * A*np.exp(1j * phase_shift) ~ img_ref

    @param imgs: n0 x n1 x ... x n_{-2} x n_{-1} array
    @param ref_imgs: reference images. Should be broadcastable to same size as imgs.
    @param thresh: only consider points in images where both abs(imgs) and abs(ref_imgs) > thresh
    @return fit_params:
    """

    if isinstance(imgs, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    ref_imgs = xp.asarray(ref_imgs)

    # broadcast images and references images to same shapes
    imgs, ref_imgs = xp.broadcast_arrays(imgs, ref_imgs)

    # looper over all dimensions except the last two, which are the y and x dimensions respectively
    loop_shape = imgs.shape[:-2]

    fit_params = xp.zeros(loop_shape + (1, 1), dtype=complex)
    nfits = np.prod(loop_shape).astype(int)
    for ii in range(nfits):
        ind = np.unravel_index(ii, loop_shape)

        if thresh is None:
            # mask = xp.ones(imgs[ind].shape, dtype=bool)
            A = xp.expand_dims(imgs[ind].ravel(), axis=1)
            B = ref_imgs[ind].ravel()
        else:
            mask = xp.logical_and(np.abs(imgs[ind]) > thresh, xp.abs(ref_imgs) > thresh)
            A = xp.expand_dims(imgs[ind][mask], axis=1)
            B = ref_imgs[ind][mask]

        fps, _, _, _ = xp.linalg.lstsq(A, B, rcond=None)
        fit_params[ind] = fps

    return fit_params


# convert between index of refraction and scattering potential
def get_n(scattering_pot: array,
          no: float,
          wavelength: float) -> array:
    """
    convert from the scattering potential to the index of refraction

    @param scattering_pot: F(r) = - (2*np.pi / lambda)^2 * (n(r)^2 - no^2)
    @param no: background index of refraction
    @param wavelength: wavelength
    @return n: refractive index
    """
    if isinstance(scattering_pot, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    k = 2 * np.pi / wavelength
    n = xp.sqrt(-scattering_pot / k ** 2 + no ** 2)
    return n


def get_scattering_potential(n: array,
                             no: float,
                             wavelength: float) -> array:
    """
    Convert from the index of refraction to the scattering potential

    @param n:
    @param no:
    @param wavelength:
    @return:
    """
    v = - (2 * np.pi / wavelength) ** 2 * (n**2 - no**2)
    return v


def get_rytov_phase(eimgs_ft: array,
                    eimgs_bg_ft: array,
                    regularization: float) -> array:
    """
    Compute rytov phase from field and background field. The Rytov phase is \psi_s(r) where
    U_total(r) = exp[\psi_o(r) + \psi_s(r)]
    where U_o(r) = exp[\psi_o(r)] is the unscattered field

    We calculate \psi_s(r) = log | U_total(r) / U_o(r)| + 1j * unwrap[angle(U_total) - angle(U_o)]

    @param eimgs_ft: n0 x n1 ... x nm x ny x nx
    @param eimgs_bg_ft: broadcastable to same size as eimgs
    @param float regularization: regularization value. Any pixels where the background
    exceeds this value will be set to zero
    @return psi_rytov:
    """

    use_gpu = isinstance(eimgs_ft, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
    else:
        xp = np

    eimgs_ft = xp.asarray(eimgs_ft)
    eimgs_bg_ft = xp.asarray(eimgs_bg_ft)

    def ift(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    def ft(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    # broadcast arrays
    eimgs_bg = ift(eimgs_bg_ft)
    eimgs = ift(eimgs_ft)
    eimgs, eimgs_bg = xp.broadcast_arrays(eimgs, eimgs_bg)

    # output values
    phase_diff = np.mod(xp.angle(eimgs) - xp.angle(eimgs_bg), 2 * np.pi)
    # convert phase difference from interval [0, 2*np.pi) to [-np.pi, np.pi)
    phase_diff[phase_diff >= np.pi] -= 2 * np.pi

    # set real parts
    psi_rytov = xp.log(abs(eimgs) / (abs(eimgs_bg))) + 1j * 0
    # set imaginary parts
    if use_gpu:
        def uwrap(m): return xp.array(unwrap_phase(m.get()))
    else:
        def uwrap(m): return unwrap_phase(m)

    # loop over all dimensions except the last two
    nextra_shape = eimgs_ft.shape[:-2]
    nextra = np.prod(nextra_shape)
    for ii in range(nextra):
        ind = np.unravel_index(ii, nextra_shape)
        psi_rytov[ind] += 1j * uwrap(phase_diff[ind])

    # regularization
    psi_rytov[abs(eimgs_bg) < regularization] = 0

    # Fourier transform
    psi_rytov_ft = ft(psi_rytov)

    return psi_rytov_ft


def get_scattered_field(eimgs_ft: array,
                        eimgs_bg_ft: array,
                        regularization: float) -> array:
    """
    Compute estimate of scattered electric field with regularization. This function only operates on the
    last two dimensions of the array

    @param eimgs_ft: array of size n0 x ... x nm x ny x nx
    @param eimgs_bg_ft: broadcastable to same size as eimgs_ft
    @param regularization:
    @return efield_scattered_ft: scattered field of same size as eimgs_ft
    """
    # todo: could also define this sensibly for the rytov case. In that case, would want to take rytov phase shift
    # and shift it back to the correct place in phase space ... since scattered field Es = E_o * psi_rytov is the approximation

    if isinstance(eimgs_ft, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    def ift(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    def ft(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    # compute scattered field in real space
    holograms_bg = ift(xp.asarray(eimgs_bg_ft))
    holograms = ift(xp.asarray(eimgs_ft))
    efield_scattered = (holograms - holograms_bg) / (xp.abs(holograms_bg) + regularization)

    # Fourier transform
    efield_scattered_ft_raw = ft(efield_scattered)

    return efield_scattered_ft_raw


# holograms
def unmix_hologram(img: array,
                   dxy: float,
                   fmax_int: float,
                   fx_ref: np.ndarray,
                   fy_ref: np.ndarray,
                   apodization: array = 1) -> array:
    """
    Given an off-axis hologram image, determine the electric field represented

    @param img: n1 x ... x n_{-3} x n_{-2} x n_{-1} array
    @param dxy: pixel size
    @param fmax_int: maximum frequency where intensity OTF has support
    @param fx_ref:
    @param fy_ref:
    @param apodization:
    @return efield_ft:
    """

    if isinstance(img, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    apodization = xp.asarray(apodization)

    # FT of image
    img_ft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(img * apodization, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

    ny, nx = img_ft.shape[-2:]

    # get frequency data
    fxs = xp.fft.fftshift(xp.fft.fftfreq(nx, dxy))
    fys = xp.fft.fftshift(xp.fft.fftfreq(ny, dxy))
    fxfx, fyfy = xp.meshgrid(fxs, fys)
    ff_perp = np.sqrt(fxfx ** 2 + fyfy ** 2)

    # compute efield
    efield_ft = tools.translate_ft(img_ft, fx_ref, fy_ref, drs=(dxy, dxy))
    efield_ft[..., ff_perp > fmax_int / 2] = 0

    return efield_ft


# tomographic reconstruction
def get_fmax(no: float,
             na_detection: float,
             na_excitation: float,
             wavelength: float) -> np.ndarray:
    """
    Maximum frequencies measurable in ODT image
    @param no: index of refraction
    @param na_detection:
    @param na_excitation:
    @param wavelength:
    @return (fx_max, fy_max, fz_max):
    """
    alpha = np.arcsin(na_detection / no)
    beta = np.max(na_excitation / no)

    # maximum frequencies present in ODT
    fxy_max = (na_detection + no * np.sin(beta)) / wavelength
    fz_max = no / wavelength * np.max([1 - np.cos(alpha), 1 - np.cos(beta)])

    fmax = np.array([fxy_max, fxy_max, fz_max])

    return fmax


def get_reconstruction_sampling(no: float,
                                na_det: float,
                                na_exc: float,
                                wavelength: float,
                                dxy: float,
                                img_size: tuple[int],
                                z_fov: float,
                                dz_sampling_factor: float = 1.,
                                dxy_sampling_factor: float = 1.) -> (tuple[float], tuple[int]):
    """
    Get information about pixel grid scattering potential will be reconstructed on

    @param no: background index of refraction
    @param na_det: numerical aperture of detection objective
    @param na_exc: maximum excitation numerical aperture (i.e. corresponding to the steepest input beam and not nec.
    the objective).
    @param wavelength: wavelength
    @param dxy: camera pixel size
    @param img_size: (ny, nx) size of hologram images
    @param z_fov: field-of-view in z-direction
    @param dz_sampling_factor: z-spacing as a fraction of the nyquist limit
    @param dxy_sampling_factor: xy-spacing as a fraction of nyquist limit
    @return (dz_sp, dxy_sp, dxy_sp), (nz_sp, ny_sp, nx_sp):
    """
    ny, nx = img_size

    alpha = np.arcsin(na_det / no)
    beta = np.arcsin(na_exc / no)

    # maximum frequencies present in ODT
    fxy_max = (na_det + na_exc) / wavelength
    fz_max = no / wavelength * np.max([1 - np.cos(alpha), 1 - np.cos(beta)])

    # ##################################
    # generate real-space sampling from Nyquist sampling
    # ##################################
    dxy_v = dxy_sampling_factor * 0.5 / fxy_max
    dz_v = dz_sampling_factor * 0.5 / fz_max
    drs = (dz_v, dxy_v, dxy_v)

    # ##################################
    # get size from FOV
    # ##################################
    x_fov = nx * dxy  # um
    nx_v = int(np.ceil(x_fov / dxy_v) + 1)
    if np.mod(nx_v, 2) == 0:
        nx_v += 1

    y_fov = ny * dxy
    ny_v = int(np.ceil(y_fov / dxy_v) + 1)
    if np.mod(ny_v, 2) == 0:
        ny_v += 1

    nz_v = int(np.ceil(z_fov / dz_v) + 1)
    if np.mod(nz_v, 2) == 0:
        nz_v += 1

    n_size = (nz_v, ny_v, nx_v)

    return drs, n_size


def get_coords(drs: list[float],
               nrs: list[int],
               expand: bool = False) -> tuple[np.ndarray]:
    """
    Compute spatial coordinates

    @param drs: (dz, dy, dx)
    @param nrs: (nz, ny, nx)
    @param expand: if False then return z, y, x as 1D arrays, otherwise return as 3D arrays
    @return: coords (z, y, x)
    """
    coords = [dr * (np.arange(nr) - (nr // 2)) for dr, nr in zip(drs, nrs)]

    if expand:
        coords = np.meshgrid(*coords, indexing="ij")

    return coords


def fwd_model_linear(beam_fx,
                     beam_fy,
                     beam_fz,
                     no: float,
                     na_det: float,
                     wavelength: float,
                     e_shape: tuple[int],
                     drs_e: tuple[float],
                     v_shape: tuple[int],
                     drs_v: tuple[float],
                     mode: str = "born",
                     interpolate: bool = False):
    """
    Forward model from scattering potential v to imaged electric field after interacting with object

    @param beam_fx: nbeams Normalized to no/wavelength
    @param beam_fy:
    @param beam_fz:
    @param no: index of refraction
    @param na_det: detection numerical aperture
    @param wavelength:
    @param e_shape: (ny, nx), shape of scattered fields
    @param drs_e: (dy, dx) pixel size of scattered field
    @param v_shape: (nz, ny, nx) shape of scattering potential
    @param drs_v: (dz, dy, dx) pixel size of scattering potential
    @param mode: "born" or "rytov"
    @param interpolate: use trilinear interpolation or not
    @return model:
    @return (data, row_index, column_index): raw data
    """

    ny, nx = e_shape
    dy, dx = drs_e

    nimgs = len(beam_fx)

    # ##################################
    # get frequencies of electric field images and make broadcastable to shape (nimgs, ny, nx)
    # ##################################
    fx = np.expand_dims(fft.fftshift(fft.fftfreq(nx, dx)), axis=(0, 1))
    fy = np.expand_dims(fft.fftshift(fft.fftfreq(ny, dy)), axis=(0, 2))

    # ##################################
    # set sampling of 3D scattering potential
    # ##################################
    nz_v, ny_v, nx_v = v_shape
    v_size = np.prod(v_shape)
    dz_v, dy_v, dx_v = drs_v

    fx_v = fft.fftshift(fft.fftfreq(nx_v, dx_v))
    fy_v = fft.fftshift(fft.fftfreq(ny_v, dy_v))
    fz_v = fft.fftshift(fft.fftfreq(nz_v, dz_v))
    dfx_v = fx_v[1] - fx_v[0]
    dfy_v = fy_v[1] - fy_v[0]
    dfz_v = fz_v[1] - fz_v[0]

    # ##################################
    # we an equation that looks like:
    # V(Fx, Fy, Fz) = 2*i * (2*pi*fz) * Es(fxs, fys)
    # where V is the scattering potential and E is an approximation of the scattered field
    # use the notation Fx, Fy, Fz to give the frequencies in the 3D scattering potential
    # and fxs, fys the corresponding frequencies in the scattered field
    # for each combination fxs, fys and beam angle, find indices into scattering potential which correspond
    # to each (Fourier space) point in the image
    # in reconstruction, implicitly assume that Es(fx, fy) are in the right place, so have to change coordinates
    # to ensure this is true
    # ##################################
    if mode == "born":
        # ##################################
        # F(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * Es(fx, fy)
        # ##################################

        # logical array, which frqs in detection NA
        detectable = np.sqrt(fx ** 2 + fy ** 2)[0] <= (na_det / wavelength)
        detectable = np.tile(detectable, [nimgs, 1, 1])

        #
        fz = np.tile(get_fz(fx, fy, no, wavelength), [nimgs, 1, 1])

        # construct frequencies where we have data about the 3D scattering potentials
        # frequencies of the sample F = f - no/lambda * beam_vec
        Fx, Fy, Fz = np.broadcast_arrays(fx - np.expand_dims(beam_fx, axis=(1, 2)),
                                         fy - np.expand_dims(beam_fy, axis=(1, 2)),
                                         fz - np.expand_dims(beam_fz, axis=(1, 2))
                                         )
        # if don't copy, then elements of F's are reference to other elements.
        Fx = np.array(Fx, copy=True)
        Fy = np.array(Fy, copy=True)
        Fz = np.array(Fz, copy=True)

        # indices into the final scattering potential
        # taking advantage of the fact that the final scattering potential indices have FFT structure
        zind = Fz / dfz_v + nz_v // 2
        yind = Fy / dfy_v + ny_v // 2
        xind = Fx / dfx_v + nx_v // 2
    elif mode == "rytov":
        # F(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * psi_s(fx - n/lambda * nx, fy - n/lambda * ny)
        # F(Fx, Fy, Fz) = 2*i * (2*pi*fz) * psi_s(Fx, Fy)
        # so want to change variables and take (Fx, Fy) -> (fx, fy)
        # But have one problem: (Fx, Fy, Fz) do not form a normalized vector like (fx, fy, fz)
        # so although we can use fx, fy to stand in, we need to calculate the new z-component
        # Fz_rytov = np.sqrt( (n/lambda)**2 - (Fx + n/lambda * nx)**2 - (Fy + n/lambda * ny)**2) - n/lambda * nz
        # fz = Fz + n/lambda * nz
        Fx = fx
        Fy = fy

        # helper frequencies for calculating fz
        fx_rytov = Fx + np.expand_dims(beam_fx, axis=(1, 2))
        fy_rytov = Fy + np.expand_dims(beam_fy, axis=(1, 2))

        fz = get_fz(fx_rytov,
                    fy_rytov,
                    no,
                    wavelength)

        Fz = fz - np.expand_dims(beam_fz, axis=(1, 2))

        # take care of frequencies which do not contain signal
        detectable = (fx_rytov ** 2 + fy_rytov ** 2) <= (na_det / wavelength) ** 2

        # indices into the final scattering potential
        zind = Fz / dfz_v + nz_v // 2
        yind = Fy / dfy_v + ny_v // 2
        xind = Fx / dfx_v + nx_v // 2

        zind, yind, xind = [np.array(a, copy=True) for a in np.broadcast_arrays(zind, yind, xind)]
    else:
        raise ValueError(f"'mode' must be 'born' or 'rytov' but was '{mode:s}'")


    # build forward model as sparse matrix
    # E(k) = model * V(k)
    # where V is made into a vector by ravelling
    # and the scattered fields are first stacked then ravelled
    # use csr for fast right mult L.dot(v)
    # use csc for fast left mult w.dot(L)
    if interpolate:
        # trilinear interpolation scheme ... needs 8 points in cube around point of interest
        z0 = np.floor(zind).astype(int)
        z1 = z0 + 1
        y0 = np.floor(yind).astype(int)
        y1 = y0 + 1
        x0 = np.floor(xind).astype(int)
        x1 = x0 + 1

        # find indices in bounds
        to_use = np.logical_and.reduce((z0 >= 0, z1 < nz_v,
                                        y0 >= 0, y1 < ny_v,
                                        x0 >= 0, x1 < nx_v,
                                        detectable))
        # todo: add in the points this misses where only option is to round
        # todo: could do this by adding to_use per coordinate ... and then normalizing the rows of the matrix
        # todo: but what if miss point in E array entirely?
        # indices into V for each coordinate
        inds = [(z0, y0, x0),
                (z1, y0, x0),
                (z0, y1, x0),
                (z1, y1, x0),
                (z0, y0, x1),
                (z1, y0, x1),
                (z0, y1, x1),
                (z1, y1, x1)
                ]

        # no denominators since using units where coordinate step is 1
        interp_weights = [(z1 - zind) * (y1 - yind) * (x1 - xind),
                          (zind - z0) * (y1 - yind) * (x1 - xind),
                          (z1 - zind) * (yind - y0) * (x1 - xind),
                          (zind - z0) * (yind - y0) * (x1 - xind),
                          (z1 - zind) * (y1 - yind) * (xind - x0),
                          (zind - z0) * (y1 - yind) * (xind - x0),
                          (z1 - zind) * (yind - y0) * (xind - x0),
                          (zind - z0) * (yind - y0) * (xind - x0)]

        # row_index -> indices into E vector
        row_index = np.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]
        row_index = np.tile(row_index, 8)

        # column_index -> indices into V vector
        inds_to_use = [[i[to_use] for i in inow] for inow in inds]
        zinds_to_use, yinds_to_use, xinds_to_use = [np.concatenate(i) for i in list(zip(*inds_to_use))]

        column_index = np.ravel_multi_index(tuple((zinds_to_use, yinds_to_use, xinds_to_use)), v_shape)

        # construct sparse matrix values
        interp_weights_to_use = np.concatenate([w[to_use] for w in interp_weights])

        data = interp_weights_to_use / (2 * 1j * (2 * np.pi * np.tile(fz[to_use], 8))) * dx_v * dy_v * dz_v / (dx * dy)

    else:
        # find indices in bounds
        to_use = np.logical_and.reduce((zind >= 0, zind < nz_v,
                                        yind >= 0, yind < ny_v,
                                        xind >= 0, xind < nx_v,
                                        detectable))

        inds_round = (np.round(zind[to_use]).astype(int),
                      np.round(yind[to_use]).astype(int),
                      np.round(xind[to_use]).astype(int))

        # row index = position in E
        row_index = np.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]

        # column index = position in V
        column_index = np.ravel_multi_index(inds_round, v_shape)

        # matrix values
        data = np.ones(len(row_index)) / (2 * 1j * (2 * np.pi * fz[to_use])) * dx_v * dy_v * dz_v / (dx * dy)

    # construct sparse matrix
    # E(k) = model * V(k)
    # column index = position in V
    # row index = position in E
    model = sp.csr_matrix((data, (row_index, column_index)), shape=(nimgs * ny * nx, v_size))

    return model, (data, row_index, column_index)


def reconstruction(efield_fts: array,
                   beam_fx,
                   beam_fy,
                   beam_fz,
                   no: float,
                   na_det: float,
                   wavelength: float,
                   dxy: float,
                   drs_v: tuple[float],
                   v_shape: tuple[int],
                   regularization: float = 0.1,
                   mode: str = "rytov",
                   no_data_value: float = np.nan) -> array:
    """
    Given a set of holograms obtained using ODT, put the hologram information back in the correct locations in
    Fourier space

    @param efield_fts: The exact definition of efield_fts depends on whether "born" or "rytov" mode is used.
    Any points in efield_fts which are NaN will be ignored. efield_fts can have an arbitrary number of leading
    singleton dimensions, but must have at least three dimensions.
    i.e. it should have shape 1 x ... x 1 x nimgs x ny x nx
    @param beam_fx: beam frqs must satify each is [vx, vy, vz] and vx**2 + vy**2 + vz**2 = n**2 / wavelength**2.
     Divide these into three arguments to make this function easier to use with dask map_blocks()
    @param beam_fy:
    @param beam_fz:
    @param no: background index of refraction
    @param na_det: detection numerical aperture
    @param wavelength:
    @param dxy: pixel size
    @param drs_v: (dz_v, dy_v, dx_v) pixel size for scattering potential reconstruction grid
    @param v_shape: (nz_v, ny_v, nx_v) grid size for scattering potential reconstruction grid
    @param regularization: regularization factor
    @param mode: "born" or "rytov"
    @param no_data_value:
    @return v_ft:
    @return drs: full coordinate grid can be obtained from get_coords
    """

    if isinstance(efield_fts, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    efield_fts = xp.asarray(efield_fts)

    n_leading_dims = efield_fts.ndim - 3
    efield_fts = efield_fts.squeeze(axis=tuple(range(n_leading_dims)))
    nimgs, ny, nx = efield_fts.shape

    # ###########################
    # put information back in Fourier space
    # ###########################
    model, (data, row_index, col_index) = fwd_model_linear(beam_fx,
                                                           beam_fy,
                                                           beam_fz,
                                                           no,
                                                           na_det,
                                                           wavelength,
                                                           (ny, nx),
                                                           (dxy, dxy),
                                                           v_shape,
                                                           drs_v,
                                                           mode=mode,
                                                           interpolate=False)

    data = xp.asarray(data)
    row_index = xp.asarray(row_index)
    col_index = xp.asarray(col_index)

    # recover indices into V and E
    v_ind = xp.unravel_index(col_index, v_shape)
    e_ind = xp.unravel_index(row_index, (nimgs, ny, nx))

    # put information in correct space in Fourier space
    v_ft = xp.zeros(v_shape, dtype=complex)
    num_pts_all = xp.zeros(v_shape, dtype=int)

    for ii in range(nimgs):
        # check we don't have duplicate indices ... otherwise need to do something else ...
        # cind_unique_angle = np.unique(cind[ii][to_use_ind[ii]])
        # if len(cind_unique_angle) != np.sum(to_use_ind[ii]):
        #     # approach would be, get array mapping all elements to unique elements, using options for np.unique()
        #     # and the cind's
        #     # average these and keep track of number
        #     # next, convert unique elements back to 3D indices and use the below code
        #     raise NotImplementedError("reconstruction only implemented for one angle mapping")

        this_angle = e_ind[0] == ii
        vinds_angle = (v_ind[0][this_angle], v_ind[1][this_angle], v_ind[2][this_angle])
        einds_angle = (e_ind[0][this_angle], e_ind[1][this_angle], e_ind[2][this_angle])
        data_angle = data[this_angle]

        # assuming at most one point for each ... otherwise haveproblems
        # since using DFT's instead of FT's have to adjust the normalization
        # FT ~ DFT * dr1 * ... * drn
        v_ft[vinds_angle] += efield_fts[einds_angle] / data_angle

        num_pts_all[vinds_angle] += 1

    # average over angles/images
    no_data = num_pts_all == 0
    is_data = xp.logical_not(no_data)

    # todo: want to weight regularization against OTF
    v_ft[is_data] = v_ft[is_data] / (num_pts_all[is_data] + regularization)
    v_ft[no_data] = no_data_value

    # expand to match original dimensions
    v_ft = xp.expand_dims(v_ft, axis=tuple(range(n_leading_dims)))

    return v_ft


def apply_n_constraints(v_ft: array,
                        no: float,
                        wavelength: float,
                        n_iterations: int = 100,
                        beta: float = 0.5,
                        use_raar: bool = True,
                        require_real_part_greater_bg: bool = False,
                        print_info: bool = False) -> array:
    """
    Apply constraints on the scattering potential and the index of refraction using iterative projection

    constraint 1: scattering potential FT must match data at points where we have information
    constraint 2: real(n) >= no and imag(n) >= 0

    @param v_ft: 3D fourier transform of scattering potential. This array should have nan values where the array
     values are unknown.
    @param no: background index of refraction
    @param wavelength: wavelength in um
    @param n_iterations: number of iterations
    @param beta:
    @param bool use_raar: whether to use the Relaxed-Averaged-Alternating Reflection algorithm
    @param require_real_part_greater_bg:
    @param print_info:
    @return v_ft_out:
    """

    if isinstance(v_ft, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    def ft(m): return xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(m, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
    def ift(m): return xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(m, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))

    # scattering_potential masked with nans where no information
    v_ft = np.array(v_ft, copy=True)
    sp_data = np.array(v_ft, copy=True)

    if not np.any(np.isnan(v_ft)):
        raise ValueError("sp_ft contained no NaN's, so there is no information to infer")

    no_data = np.isnan(v_ft)
    is_data = np.logical_not(no_data)
    v_ft[no_data] = 0

    # do iterations
    tstart = time.perf_counter()
    for ii in range(n_iterations):
        if print_info:
            print(f"constraint iteration {ii + 1:d}/{n_iterations:d},"
                  f" elapsed time = {time.perf_counter() - tstart:.2f}s", end="\r")
            if ii == (n_iterations - 1):
                print("")

        # ############################
        # ensure n is physical
        # ############################
        # todo: can do imaginary part with the scattering potential instead
        sp = ift(v_ft)
        n = get_n(sp, no, wavelength)

        if require_real_part_greater_bg:
            # real part must be >= no
            correct_real = np.real(n) < no
            n[correct_real] = no + 1j * n[correct_real].imag

        # imaginary part must be >= 0
        correct_imag = n.imag < 0
        n[correct_imag] = n[correct_imag].real + 0*1j

        sp_ps = get_scattering_potential(n, no, wavelength)
        sp_ps_ft = ft(sp_ps)

        if use_raar:
            sp_ft_pm = np.array(v_ft, copy=True)
            sp_ft_pm[is_data] = sp_data[is_data]

            # ############################
            # projected Ps * Pm
            # ############################
            sp_ft_ps_pm = np.array(sp_ft_pm, copy=True)
            sp_ps_pm = ift(sp_ft_ps_pm)
            n_ps_pm = get_n(sp_ps_pm, no, wavelength)

            if require_real_part_greater_bg:
                # real part must be >= no
                correct_real = n_ps_pm.real < no
                n_ps_pm[correct_real] = no + n_ps_pm[correct_real].imag

            # imaginary part must be >= 0
            correct_imag = n_ps_pm.imag < 0
            n_ps_pm[correct_imag] = n_ps_pm[correct_imag].real + 0 * 1j

            sp_ps_pm = get_scattering_potential(n_ps_pm, no, wavelength)
            sp_ps_pm_ft = ft(sp_ps_pm)

            # update
            v_ft = beta * v_ft - beta * sp_ps_ft + (1 - 2 * beta) * sp_ft_pm + 2 * beta * sp_ps_pm_ft
        else:
            # ############################
            # projected Pm * Ps
            # ############################
            sp_ft_pm_ps = np.array(sp_ps_ft, copy=True)
            sp_ft_pm_ps[is_data] = sp_data[is_data]

            # update
            v_ft = sp_ft_pm_ps

    return v_ft


# fit frequencies
def fit_ref_frq(img_ft: np.ndarray,
                dxy: float,
                fmax_int: float,
                search_rad_fraction: float = 1,
                npercentiles: int = 50,
                filter_size=0,
                dilate_erode_footprint_size: int = 10,
                show_figure: bool = False):
    """
    Determine the hologram reference frequency from a single imaged, based on the regions in the hologram beyond the
    maximum imaging frequency that have information. These are expected to be circles centered around the reference
    frequency.

    The fitting strategy is this
    (1) determine a threshold value for which points have signal in the image. To do this, first make a plot of
    thresholds versus percentiles. This should look like two piecewise lines
    (2) after thresholding the image, fit to circles.

    Note: when the beam angle is non-zero, the dominant tomography frequency component will not be centered
    on this circle, but will be at position f_ref - f_beam
    @param img_ft:
    @param dxy:
    @param fmax_int:
    @param search_rad_fraction:
    @param npercentiles:
    @param filter_size:
    @param dilate_erode_footprint_size:
    @param show_figure:
    @return results, circ_dbl_fn, figh: results["fit_params"] = [cx, cy, radius]
    """
    ny, nx = img_ft.shape

    # filter image
    img_ft = gaussian_filter(img_ft, (filter_size, filter_size))

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(nx, dxy))
    dfx = fxs[1] - fxs[0]
    fys = fft.fftshift(fft.fftfreq(ny, dxy))
    dfy = fys[1] - fys[0]
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff_perp = np.sqrt(fxfx**2 + fyfy**2)

    extent_fxy = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx,
                  fys[0] - 0.5 * dfy, fys[-1] + 0.5 * dfy]

    # #########################
    # find threshold using expected volume of area above threshold
    # #########################
    # only search outside of this
    frad_search = search_rad_fraction * fmax_int
    search_region = ff_perp > frad_search

    frq_area = (fxfx[0, -1] - fxfx[0, 0]) * (fyfy[-1, 0] - fyfy[0, 0])
    expected_area = 2 * (np.pi * (0.5 * fmax_int)**2) / (frq_area - np.pi * frad_search**2)
    # thresh = np.percentile(np.abs(img_ft_bin[search_region]), 100 * (1 - expected_area))

    # find thresholds for different percentiles and look or plateau like behavior
    percentiles = np.linspace(0, 99, npercentiles)
    thresh_all = np.percentile(np.abs(img_ft[search_region]), percentiles)

    init_params_thresh = [0, thresh_all[0],
                          (thresh_all[-1] - thresh_all[-2]) / (percentiles[-1] - percentiles[-2]),
                          100 * (1 - expected_area)]
    results_thresh = fit.fit_model(thresh_all, lambda p: fit.line_piecewise(percentiles, p), init_params_thresh)

    thresh_ind = np.argmin(np.abs(percentiles - results_thresh["fit_params"][-1]))
    thresh = thresh_all[thresh_ind]

    # masked image
    img_ft_ref_mask = np.logical_and(np.abs(img_ft) > thresh, ff_perp > frad_search)

    # dilate and erode
    footprint = np.ones((dilate_erode_footprint_size, dilate_erode_footprint_size), dtype=bool)
    # dilate to remove holes
    img_ft_ref_mask = maximum_filter(img_ft_ref_mask, footprint=footprint)
    # remove to get back to original size
    img_ft_ref_mask = minimum_filter(img_ft_ref_mask, footprint=footprint)

    # #########################
    # define fitting function and get initial guesses
    # #########################

    def circ_dbl_fn(x, y, p):
        p = np.array([p[0], p[1], p[2], 1, 0, np.sqrt(dfx * dfy)])
        p2 = np.array(p, copy=True)
        p2[0] *= -1
        p2[1] *= -1
        circd = fit.circle(x, y, p) + fit.circle(x, y, p2)
        circd[circd > 1] = 1
        return circd

    # guess based on maximum pixel value. This actually gives f_ref - f_beam, but should be a close enough starting point
    guess_ind_1d = np.argmax(np.abs(img_ft) * (fyfy <= 0) * (ff_perp > fmax_int))
    guess_ind = np.unravel_index(guess_ind_1d, img_ft.shape)

    # do fitting
    # init_params = [np.mean(fxfx_bin[img_ft_ref_mask]), np.mean(fyfy_bin[img_ft_ref_mask]), 0.5 * fmax_int]
    init_params = [fxfx[guess_ind], fyfy[guess_ind], 0.5 * fmax_int]
    results = fit.fit_model(img_ft_ref_mask, lambda p: circ_dbl_fn(fxfx, fyfy, p), init_params)

    # #########################
    # plot
    # #########################
    if show_figure:
        figh = plt.figure(figsize=(16, 8))
        grid = figh.add_gridspec(2, 3)

        fp_ref = results["fit_params"]
        ax = figh.add_subplot(grid[0, 0])
        ax.set_title("img ft")
        ax.imshow(np.abs(img_ft), norm=PowerNorm(gamma=0.1), cmap='bone', extent=extent_fxy, origin="lower")
        ax.plot(fp_ref[0], fp_ref[1], 'kx')
        ax.plot(-fp_ref[0], -fp_ref[1], 'kx')
        ax.add_artist(Circle((fp_ref[0], fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((-fp_ref[0], -fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((0, 0), radius=fmax_int, facecolor="none", edgecolor="k"))

        ax = figh.add_subplot(grid[0, 1])
        ax.set_title("img ft binned")
        ax.imshow(np.abs(img_ft), norm=PowerNorm(gamma=0.1), cmap='bone', extent=extent_fxy, origin="lower")
        ax.plot(fp_ref[0], fp_ref[1], 'kx')
        ax.plot(-fp_ref[0], -fp_ref[1], 'kx')
        ax.add_artist(Circle((fp_ref[0], fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((-fp_ref[0], -fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((0, 0), radius=fmax_int, facecolor="none", edgecolor="k"))

        ax = figh.add_subplot(grid[0, 2])
        ax.set_title("img ft binned mask")
        ax.imshow(img_ft_ref_mask, norm=PowerNorm(gamma=0.1), cmap='bone', extent=extent_fxy, origin="lower")
        ax.plot(fp_ref[0], fp_ref[1], 'kx')
        ax.plot(-fp_ref[0], -fp_ref[1], 'kx')
        ax.add_artist(Circle((fp_ref[0], fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((-fp_ref[0], -fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((0, 0), radius=fmax_int, facecolor="none", edgecolor="k"))

        ax = figh.add_subplot(grid[1, 0])
        ax.plot(percentiles, thresh_all, 'rx')
        ax.plot(percentiles, fit.line_piecewise(percentiles, results_thresh["fit_params"]))
        ax.set_xlabel("percentile")
        ax.set_ylabel("threshold (counts)")
        ax.set_title('threshold = %.0f' % results_thresh["fit_params"][-1])
    else:
        figh = None

    return results, circ_dbl_fn, figh


# plotting functions
def plot_scattered_angle(img_efield_ft,
                         img_efield_bg_ft,
                         img_efield_scatt_ft,
                         beam_frq,
                         frq_ref,
                         fmax_int: float,
                         dxy: float,
                         title: str = "",
                         figsize=(18, 10),
                         gamma: float = 0.1,
                         **kwargs):
    """
    Plot diagnostic of ODT image and background image
    
    Allows img and ft to be passed in or calculated ....

    @param img_efield_ft:
    @param img_efield_bg_ft:
    @param img_efield_scatt_ft:
    @param beam_frq:
    @param frq_ref:
    @param fmax_int:
    @param dxy:
    @param title:
    @param figsize:
    @param gamma:
    @return figh:
    """

    # real-space coordinates
    ny, nx = img_efield_ft.shape
    x, y = get_coords((dxy, dxy), (nx, ny))
    extent_xy = [x[0] - 0.5 * dxy, x[-1] + 0.5 * dxy,
                 y[-1] + 0.5 * dxy, y[0] - 0.5 * dxy]

    # frequency-space coordinates
    fx = fft.fftshift(fft.fftfreq(nx, dxy))
    fy = fft.fftshift(fft.fftfreq(ny, dxy))
    dfy = fy[1] - fy[0]
    dfx = fx[1] - fx[0]
    extent_fxfy = [fx[0] - 0.5 * dfx, fx[-1] + 0.5 * dfx,
                   fy[-1] + 0.5 * dfy, fy[0] - 0.5 * dfy]

    # ##############################
    # compute real-space electric field
    # ##############################
    img_efield_ft_no_nan = np.array(img_efield_ft, copy=True)
    img_efield_ft_no_nan[np.isnan(img_efield_ft_no_nan)] = 0

    img_efield_bg_ft_no_nan = np.array(img_efield_bg_ft, copy=True)
    img_efield_bg_ft_no_nan[np.isnan(img_efield_bg_ft_no_nan)] = 0

    img_efield_scatt_ft_no_nan = np.array(img_efield_scatt_ft, copy=True)
    img_efield_scatt_ft_no_nan[np.isnan(img_efield_scatt_ft_no_nan)] = 0

    # remove carrier frequency so can easily see phase shifts
    exp_fact = np.exp(-1j * 2*np.pi * (beam_frq[0] * np.expand_dims(x, axis=0) +
                                       beam_frq[1] * np.expand_dims(y, axis=1)))

    img_efield_shift = fft.fftshift(fft.ifft2(fft.ifftshift(img_efield_ft_no_nan))) * exp_fact
    img_efield_shift_bg = fft.fftshift(fft.ifft2(fft.ifftshift(img_efield_bg_ft_no_nan))) * exp_fact
    img_efield_shift_scatt = fft.fftshift(fft.ifft2(fft.ifftshift(img_efield_scatt_ft_no_nan))) * exp_fact

    # ##############################
    # plot
    # ##############################
    figh = plt.figure(figsize=figsize, **kwargs)
    figh.suptitle(title)
    grid = figh.add_gridspec(ncols=9, width_ratios=[8, 1, 1] * 3,
                             nrows=3, hspace=0.5, wspace=0.5)


    labels = ["E_{shifted}", "E_{bg,shifted}", "E_{scatt,shifted}"]
    fields_r = [img_efield_shift, img_efield_shift_bg, img_efield_shift_scatt]
    vmin_e = np.percentile(np.abs(img_efield_shift).ravel(), 0.1)
    vmax_e = np.percentile(np.abs(img_efield_shift).ravel(), 99.9)
    # vmin_scat = np.percentile(np.abs(img_efield_shift_scatt).ravel(), 0.1)
    # vmax_scat = np.percentile(np.abs(img_efield_shift_scatt).ravel(), 99.9)
    vmin_scat = 0
    vmax_scat = 1.25
    vmin_r = [vmin_e, vmin_e, vmin_scat]
    vmax_r = [vmax_e, vmax_e, vmax_scat]
    fields_ft = [img_efield_ft, img_efield_bg_ft, img_efield_scatt_ft]

    flims = [[-fmax_int, fmax_int],
             [-fmax_int, fmax_int],
             [-fmax_int, fmax_int]]
    plot_pts = [beam_frq[:2]] * 3
    for ii, label in enumerate(labels):

        ax1 = figh.add_subplot(grid[0, 3*ii])
        im = ax1.imshow(np.abs(fields_r[ii]), cmap="bone", vmin=vmin_r[ii], vmax=vmax_r[ii], extent=extent_xy)
        ax1.set_title("$|%s(r)|$" % label)
        ax1.set_xlabel("x-position ($\mu m$)")
        ax1.set_ylabel("y-position ($\mu m$)")

        ax4 = figh.add_subplot(grid[0, 3*ii + 1])
        plt.colorbar(im, cax=ax4)

        ax2 = figh.add_subplot(grid[1, 3*ii])
        im = ax2.imshow(np.angle(fields_r[ii]), cmap="RdBu", vmin=-np.pi, vmax=np.pi, extent=extent_xy)
        ax2.set_title("$ang[%s(r)]$" % label)
        ax2.set_xticks([])
        ax2.set_yticks([])
        if ii == (len(labels) - 1):
            ax4 = figh.add_subplot(grid[1, 3 * ii + 1])
            plt.colorbar(im, cax=ax4)


        ax3 = figh.add_subplot(grid[2, 3 * ii])
        im = ax3.imshow(np.abs(fields_ft[ii]), norm=PowerNorm(gamma=gamma), cmap="bone", extent=extent_fxfy)
        ax3.plot(plot_pts[ii][0], plot_pts[ii][1], 'r.', fillstyle="none")
        ax3.set_xlabel("$f_x$ (1 / $\mu m$)")
        ax3.set_ylabel("$f_y$ (1 / $\mu m$)")
        ax3.set_title("$|E(f)|$")
        ax3.set_xlim(flims[ii])
        ax3.set_ylim(np.flip(flims[ii]))

    return figh


def plot_odt_sampling(frqs: np.ndarray,
                      na_detect: float,
                      na_excite: float,
                      ni: float,
                      wavelength: float,
                      figsize=(30, 8)):
    """
    Illustrate the region of frequency space which is obtained using the plane waves described by frqs

    @param frqs: nfrqs x 2 array of [[fx0, fy0], [fx1, fy1], ...]
    @param na_detect: detection NA
    @param na_excite: excitation NA
    @param ni: index of refraction of medium that samle is immersed in. This may differ from the immersion medium
    of the objectives
    @param wavelength:
    @param figsize:
    @return:
    """
    frq_norm = ni / wavelength
    alpha_det = np.arcsin(na_detect / ni)

    if na_excite / ni < 1:
        alpha_exc = np.arcsin(na_excite / ni)
    else:
        # if na_excite is immersion objective and beam undergoes TIR at interface for full NA
        alpha_exc = np.pi/2

    fzs = get_fz(frqs[:, 0], frqs[:, 1], ni, wavelength)
    frqs_3d = np.concatenate((frqs, np.expand_dims(fzs, axis=1)), axis=1)


    figh = plt.figure(figsize=figsize)
    figh.suptitle("Frequency support diagnostic")
    grid = figh.add_gridspec(nrows=1, ncols=4)

    # ########################
    # kx-kz plane
    # ########################
    ax = figh.add_subplot(grid[0, 0])
    ax.set_title("$k_x-k_z$ projection")
    ax.axis("equal")

    # plot centers
    ax.plot(-frqs_3d[:, 0], -frqs_3d[:, 2], 'k.', label="beam frqs")

    # plot arcs
    for ii in range(len(frqs_3d)):
        if ii == 0:
            kwargs = {"label": "frq support"}
        else:
            kwargs = {}
        ax.add_artist(Arc((-frqs_3d[ii, 0], -frqs_3d[ii, 2]), 2 * frq_norm, 2 * frq_norm, angle=90,
                          theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="k", **kwargs))

    # draw arcs for the extremal angles
    fx_edge = na_excite / wavelength
    fz_edge = np.sqrt((ni / wavelength)**2 - fx_edge**2)

    ax.plot(-fx_edge, -fz_edge, 'r.')
    ax.plot(fx_edge, -fz_edge, 'r.')

    ax.add_artist(Arc((-fx_edge, -fz_edge), 2 * frq_norm, 2 * frq_norm, angle=90,
                      theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi,
                      edgecolor="r", label="extremal frequency data"))
    ax.add_artist(Arc((fx_edge, -fz_edge), 2 * frq_norm, 2 * frq_norm, angle=90,
                      theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="r"))

    # draw arc showing possibly positions of centers
    ax.add_artist(Arc((0, 0), 2 * frq_norm, 2 * frq_norm, angle=-90,
                      theta1=-alpha_exc * 180 / np.pi, theta2=alpha_exc * 180 / np.pi,
                      edgecolor="b", label="allowed beam frqs"))

    ax.set_ylim([-2 * frq_norm, 2 * frq_norm])
    ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
    ax.set_xlabel("$f_x$ (1/$\mu m$)")
    ax.set_ylabel("$f_z$ (1/$\mu m$)")

    plt.legend()

    # ########################
    # ky-kz plane
    # ########################
    ax = figh.add_subplot(grid[0, 1])
    ax.set_title("$k_y-k_z$ projection")
    ax.axis("equal")

    # plot centers
    ax.plot(-frqs_3d[:, 1], -frqs_3d[:, 2], 'k.')

    # plot arcs
    for ii in range(len(frqs_3d)):
        ax.add_artist(Arc((-frqs_3d[ii, 1], -frqs_3d[ii, 2]), 2 * frq_norm, 2 * frq_norm, angle=90,
                          theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="k"))

    # draw arcs for the extremal angles
    fy_edge = na_excite / wavelength
    fz_edge = np.sqrt((ni / wavelength)**2 - fy_edge**2)

    ax.plot(-fy_edge, -fz_edge, 'r.')
    ax.plot(fy_edge, -fz_edge, 'r.')

    ax.add_artist(Arc((-fy_edge, -fz_edge), 2 * frq_norm, 2 * frq_norm, angle=90,
                      theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="r"))
    ax.add_artist(Arc((fy_edge, -fz_edge), 2 * frq_norm, 2 * frq_norm, angle=90,
                      theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="r"))

    # draw arc showing possibly positions of centers
    ax.add_artist(Arc((0, 0), 2 * frq_norm, 2 * frq_norm, angle=-90,
                      theta1=-alpha_exc * 180 / np.pi, theta2=alpha_exc * 180 / np.pi, edgecolor="b"))


    ax.set_ylim([-2 * frq_norm, 2 * frq_norm])
    ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
    ax.set_xlabel("$f_y$ (1/$\mu m$)")
    ax.set_ylabel("$f_z$ (1/$\mu m$)")

    # ########################
    # kx-ky plane
    # ########################
    ax = figh.add_subplot(grid[0, 2])
    ax.set_title("$k_x-k_y$ projection")
    ax.axis("equal")

    ax.plot(-frqs_3d[:, 0], -frqs_3d[:, 1], 'k.')
    for ii in range(len(frqs_3d)):
        ax.add_artist(Circle((-frqs_3d[ii, 0], -frqs_3d[ii, 1]),
                             na_detect / wavelength, fill=False, color="k"))

    ax.add_artist(Circle((0, 0), na_excite / wavelength, fill=False, color="b"))
    ax.add_artist(Circle((0, 0), (na_excite + na_detect) / wavelength, fill=False, color="r"))


    # size = 1.5 * (na_excite + na_detect) / wavelength
    # ax.set_ylim([-size, size])
    # ax.set_xlim([-size, size])
    ax.set_ylim([-2 * frq_norm, 2 * frq_norm])
    ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
    ax.set_xlabel("$f_x$ (1/$\mu m$)")
    ax.set_ylabel("$f_y$ (1/$\mu m$)")

    # ########################
    # 3D
    # ########################
    ax = figh.add_subplot(grid[0, 3], projection="3d")
    ax.set_title("3D projection")
    # ax.axis("equal")

    fx = fy = np.linspace(-na_detect / wavelength, na_detect / wavelength, 100)
    fxfx, fyfy = np.meshgrid(fx, fy)
    ff = np.sqrt(fxfx**2 + fyfy**2)
    fmax = na_detect / wavelength

    fxfx[ff > fmax] = np.nan
    fyfy[ff > fmax] = np.nan
    fzfz = np.sqrt((ni / wavelength)**2 - fxfx**2 - fyfy**2)


    # kx0, ky0, kz0
    fxyz0 = np.stack((fxfx, fyfy, fzfz), axis=-1)
    for ii in range(len(frqs_3d)):
        ax.plot_surface(fxyz0[..., 0] - frqs_3d[ii, 0], fxyz0[..., 1] - frqs_3d[ii, 1], fxyz0[..., 2] - frqs_3d[ii, 2], alpha=0.3)


    ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
    ax.set_ylim([-2 * frq_norm, 2 * frq_norm])
    ax.set_zlim([-1, 1]) # todo: set based on na's

    ax.set_xlabel("$f_x$ (1/$\mu m$)")
    ax.set_ylabel("$f_y$ (1/$\mu m$)")
    ax.set_zlabel("$f_z$ (1/$\mu m$)")

    return figh


def display_tomography_recon(recon_fname: str,
                             raw_data_fname: Optional[str] = None,
                             show_raw: bool = True,
                             show_raw_ft: bool = False,
                             show_v_fft: bool = False,
                             show_efields: bool = False,
                             compute: bool = False,
                             time_axis: int = 1,
                             time_range: Optional[list[int]] = None,
                             block_while_display: bool = True):
    """
    Display reconstruction results and (optionally) raw data in Napari

    @param recon_fname: refractive index reconstruction stored in zarr file
    @param raw_data_fname: raw data stored in zar file
    @param show_raw:
    @param show_raw_ft:
    @param show_v_fft:
    @param show_efields:
    @param block_while_display:
    @return:
    """

    import napari

    if raw_data_fname is not None:
        raw_data = zarr.open(raw_data_fname, "r")
    else:
        show_raw = False

    # load data
    img_z = zarr.open(recon_fname, "r")
    if not hasattr(img_z, "efield_bg_ft") or not hasattr(img_z, "efield_scattered_ft") or not hasattr(img_z, "efields_ft"):
        show_efields = False

    dxy = img_z.attrs["camera_path_attributes"]["dx_um"]
    proc_roi = img_z.attrs["processing roi"]
    ny = proc_roi[1] - proc_roi[0]
    nx = proc_roi[3] - proc_roi[2]

    try:
        cam_roi = img_z.attrs["camera_path_attributes"]["camera_roi"]
        ny_raw = cam_roi[1] - cam_roi[0]
        nx_raw = cam_roi[3] - cam_roi[2]
    except KeyError:
        ny_raw = ny
        nx_raw = nx

    dz_v = img_z.attrs["dz"]
    dxy_v = img_z.attrs["dx"]
    nz_sp = img_z.n.shape[-4]
    ny_sp = img_z.n.shape[-2]
    nx_sp = img_z.n.shape[-1]
    npatterns = len(img_z.attrs["beam_frqs"])
    wavelength = img_z.attrs["wavelength"]
    no = img_z.attrs["no"]

    try:
        tau_tv = img_z.attrs["tv_tau"]
    except KeyError:
        tau_tv = img_z.attrs["reconstruction_settings"]["tau_tv"]

    try:
        tau_lasso = img_z.attrs["l1_tau"]
    except KeyError:
        tau_lasso = img_z.attrs["reconstruction_settings"]["tau_lasso"]

    # load affine xforms
    # Napari is using convention (y, x) whereas I'm using (x, y), so need to swap these dimensions in affine xforms
    swap_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    affine_recon2cam = np.array(img_z.attrs["affine_xform_recon_2_raw_camera_roi"])

    # load images
    # plot index of refraction
    n_extra_dims = img_z.n.ndim - 4
    slices = []
    for ii in range(n_extra_dims):
        if ii == time_axis:
            if time_range is not None:
                slices.append(slice(time_range[0], time_range[1]))
            else:
                slices.append(slice(None))
        else:
            slices.append(slice(None))
    slices = tuple(slices)

    slices_n = slices + (slice(None), slice(None), slice(None), slice(None))

    n = da.from_zarr(img_z.n).rechunk((1,) * n_extra_dims + (1, 1, ny_sp, nx_sp))[slices_n]
    n_r = n[..., 0, :, :]
    n_im = n[..., 1, :, :]

    # broadcasting
    dummy_shape_array = da.from_array(np.zeros((1,) * n_extra_dims + (npatterns, nz_sp, 1, 1)))

    if compute:
        print("loading n")
        with ProgressBar():
            c = dask.compute([n_r, n_im])
            n_r, n_im = c[0]
            n_im_stack, n_r_stack, _ = np.broadcast_arrays(np.expand_dims(n_im, axis=-4),
                                                           np.expand_dims(n_r, axis=-4),
                                                           dummy_shape_array.compute())
    else:
        n_im_stack, n_r_stack, _ = da.broadcast_arrays(da.expand_dims(n_im, axis=-4),
                                                       da.expand_dims(n_r, axis=-4),
                                                       dummy_shape_array)


    if show_raw:
        slices_img = slices + (slice(None), slice(None), slice(None))

        imgs = da.from_zarr(raw_data.cam2.odt)[slices_img]
        imgs_raw_ft = da.fft.fftshift(da.fft.fft2(da.fft.ifftshift(imgs, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        if compute:
            print("loading raw images")
            with ProgressBar():
                c = dask.compute([imgs, imgs_raw_ft])
                imgs, img_raw_ft = c[0]

                ims_raw_stack, img_raw_ft_stack, _ = np.broadcast_arrays(np.expand_dims(imgs, axis=-3),
                                                                         np.expand_dims(imgs_raw_ft, axis=-3),
                                                                         dummy_shape_array.compute())

        else:
            ims_raw_stack, img_raw_ft_stack, _ = da.broadcast_arrays(da.expand_dims(imgs, axis=-3),
                                                                     da.expand_dims(imgs_raw_ft, axis=-3),
                                                                     dummy_shape_array)


    if show_efields:
        csize = list(img_z.efield_scattered_ft.chunks)
        csize[-3] = img_z.efield_scattered_ft.shape[-3]

        slices_e = slices + (slice(None), slice(None), slice(None))

        # scattered field
        escatt_load = da.from_zarr(img_z.efield_scattered_ft).rechunk(csize)[slices_e]
        escatt = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(escatt_load, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        # measured field
        e_load_ft = da.from_zarr(img_z.efields_ft).rechunk(csize)[slices_e]
        e = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(e_load_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        # background field
        ebg_load_ft = da.from_zarr(img_z.efield_bg_ft).rechunk(csize)[slices_e]
        ebg = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(ebg_load_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        # rytov phi
        try:
            erytov_ft = da.from_zarr(img_z.phi_rytov_ft).rechunk(csize)[slices_e]
            erytov = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(erytov_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
        except:
            erytov = da.zeros(e.shape, chunks=e.chunksize)

        # compute electric field power
        efield_power = da.mean(da.abs(e), axis=(-1, -2), keepdims=True)

        if compute:
            print("loading electric fields")
            with ProgressBar():
                c = dask.compute([escatt, e, ebg, erytov])
                escatt, e, ebg, erytov = c[0]

                efield_power_stack, _ = np.broadcast_arrays(np.expand_dims(efield_power, axis=-3), dummy_shape_array.compute())

            escatt_stack, estack, ebg_stack, erytov_stack, _ = np.broadcast_arrays(np.expand_dims(escatt, axis=-3),
                                                                                   np.expand_dims(e, axis=-3),
                                                                                   np.expand_dims(ebg, axis=-3),
                                                                                   np.expand_dims(erytov, axis=-3),
                                                                                   dummy_shape_array)

        else:
            efield_power_stack, _ = da.broadcast_arrays(da.expand_dims(efield_power, axis=-3), dummy_shape_array)

            escatt_stack, estack, ebg_stack, erytov_stack, _ = da.broadcast_arrays(da.expand_dims(escatt, axis=-3),
                                                                                   da.expand_dims(e, axis=-3),
                                                                                   da.expand_dims(ebg, axis=-3),
                                                                                   da.expand_dims(erytov, axis=-3),
                                                                                   dummy_shape_array)

    # ######################
    # create viewer
    # ######################
    viewer = napari.Viewer(title=str(recon_fname))

    # ######################
    # raw data
    # ######################
    # todo: maybe better to rewrite with da.brodcast_arrays() instead of tiling
    if show_raw:
        viewer.add_image(ims_raw_stack, scale=(dz_v / dxy_v, 1, 1),
                         # translate=(0, tm_set.nx * tm_set.dxy),
                         name="raw images", contrast_limits=[0, 4096])

        if show_raw_ft:
            # imgs_raw_ft = da.fft.fftshift(da.fft.fft2(da.fft.ifftshift(imgs, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
            # img_raw_ft_stack = da.stack([imgs_raw_ft] * nz_sp, axis=-3)
            viewer.add_image(da.abs(img_raw_ft_stack), scale=(dz_v / dxy_v, 1, 1),
                             name="raw images ft",
                             gamma=0.2,
                             translate=(ny_raw, 0))

    # ######################
    # reconstructed index of refraction
    # ######################

    # for convenience of affine xforms, keep xy in pixels
    viewer.add_image(n_im_stack,
                     scale=(dz_v / dxy_v, 1, 1),
                     name="n.imaginary",
                     affine=swap_xy.dot(affine_recon2cam.dot(swap_xy)),
                     contrast_limits=[0, 0.05],
                     visible=False)

    viewer.add_image(n_r_stack - no,
                     scale=(dz_v / dxy_v, 1, 1),
                     name=f"n-no, tv={tau_tv:.3f}, l1={tau_lasso:.3f}",
                     affine=swap_xy.dot(affine_recon2cam.dot(swap_xy)),
                     contrast_limits=[0, 0.05])


    # ######################
    # display phase shifts and power on points layer
    # ######################
    # add (1, 1) for xy dimensions
    coords = np.meshgrid(*[np.arange(d) for d in n_r_stack.shape[:-2] + (1, 1)], indexing="ij")
    coords = [c.ravel() for c in coords]
    points = np.stack(coords, axis=1)

    dphi = da.from_zarr(img_z.phase_shifts)[slices + (slice(None), slice(None), slice(None))]
    dphi_stack = da.stack([dphi] * nz_sp, axis=-3)

    if show_efields:
        epower = efield_power_stack.compute().ravel()
    else:
        epower = np.zeros(dphi_stack.shape).ravel() * np.nan

    viewer.add_points(points,
                      features={"dphi": dphi_stack.compute().ravel(),
                                "epower": epower},
                      text={"string": "phi={dphi:.3f}, epower={epower:.3f}",
                            "size": 10,
                            "color": "red"
                            },
                      scale=(1,) * (points.shape[-1] - 3) + (dz_v / dxy_v, 1, 1)
                      )

    # ######################
    # show ROI in image
    # ######################
    if show_raw:
        # draw so that points inside rectangle are in the ROI. Points under rectangle are not
        proc_roi_rect = np.array([[[proc_roi[0] - 1, proc_roi[2] - 1],
                                   [proc_roi[0] - 1, proc_roi[3]],
                                   [proc_roi[1], proc_roi[3]],
                                   [proc_roi[1], proc_roi[2] - 1]
                                   ]])
        viewer.add_shapes(proc_roi_rect, shape_type="polygon", name="processing ROI", edge_width=1,
                          edge_color=[1, 0, 0, 1], face_color=[0, 0, 0, 0])

    # ######################
    # Fts
    # ######################
    if show_v_fft:
        # plot Fourier transform of scattering potential
        n_full = n_r + 1j * n_im

        v = da.map_blocks(get_scattering_potential, n_full, img_z.attrs["no"], wavelength, dtype=complex)
        v_ft = da.fft.fftshift(da.fft.fftn(da.fft.ifftshift(v, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))
        v_ft_stack = da.stack([v_ft] * npatterns, axis=-4)

        v_ft_start = da.abs(da.from_zarr(img_z.v_ft_start))
        v_ft_start_stack = da.stack([v_ft_start] * npatterns, axis=-4)

        viewer.add_image(da.abs(v_ft_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="|v ft|", gamma=0.1, visible=False)
        viewer.add_image(da.abs(v_ft_start_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="|v ft| start", gamma=0.1, visible=False)

    # ######################
    # electric fields
    # ######################
    if show_efields:
        # E scattered field
        viewer.add_image(da.abs(escatt_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="|e scatt|", contrast_limits=[0, 1.2],
                         translate=(0, nx_raw))

        #
        viewer.add_image(da.angle(escatt_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="angle(e scatt)", contrast_limits=[-np.pi, np.pi],
                         colormap="PiYG",
                         translate=(ny, nx_raw))

        # offset = np.array([[0] * (points.shape[-1] - 2) + [-10, nx_raw + nx // 2]])
        # viewer.add_points(points + offset,
        #                   features={"dphi": np.zeros(len(points)),},
        #                   text={"string": "{dphi:.0f} E scattered",
        #                         "size": 10,
        #                         "color": "red"
        #                         },
        #                   scale=(1,) * (points.shape[-1] - 3) + (dz_v / dxy_v, 1, 1)
        #                   )

        # rytov
        viewer.add_image(da.abs(erytov_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="|e scatt|", contrast_limits=[0, 1.2],
                         translate=(0, nx_raw + nx))

        viewer.add_image(da.angle(erytov_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="angle(e rytov)", contrast_limits=[-np.pi, np.pi],
                         colormap="PiYG",
                         translate=(ny, nx_raw + nx))

        # offset = np.array([[0] * (points.shape[-1] - 2) + [-10, nx_raw + nx + nx // 2]])
        # viewer.add_points(points + offset,
        #                   features={"dphi": np.zeros(len(points)), },
        #                   text={"string": "{dphi:.0f} E Rytov",
        #                         "size": 10,
        #                         "color": "red"
        #                         },
        #                   scale=(1,) * (points.shape[-1] - 3) + (dz_v / dxy_v, 1, 1)
        #                   )

        # measured field
        viewer.add_image(da.abs(estack), scale=(dz_v / dxy_v, 1, 1),
                         name="|e|", contrast_limits=[0, 500],
                         translate=(0, nx_raw + 2*nx))

        viewer.add_image(da.angle(estack), scale=(dz_v / dxy_v, 1, 1),
                         name="angle(e)", contrast_limits=[-np.pi, np.pi],
                         colormap="PiYG",
                         translate=(ny, nx_raw + 2*nx))

        # offset = np.array([[0] * (points.shape[-1] - 2) + [-10, nx_raw + 2*nx + nx // 2]])
        # viewer.add_points(points + offset,
        #                   features={"dphi": np.zeros(len(points)), },
        #                   text={"string": "{dphi:.0f} E",
        #                         "size": 10,
        #                         "color": "red"
        #                         },
        #                   scale=(1,) * (points.shape[-1] - 3) + (dz_v / dxy_v, 1, 1)
        #                   )

        # background field
        viewer.add_image(da.abs(ebg_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="|e bg|", contrast_limits=[0, 500],
                         translate=(0, nx_raw + 3 * nx))

        viewer.add_image(da.angle(ebg_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="angle(e bg)", contrast_limits=[-np.pi, np.pi],
                         colormap="PiYG",
                         translate=(ny, nx_raw + 3 * nx))

        # offset = np.array([[0] * (points.shape[-1] - 2) + [-10, nx_raw + 3*nx + nx // 2]])
        # viewer.add_points(points + offset,
        #                   features={"dphi": np.zeros(len(points)), },
        #                   text={"string": "{dphi:.0f} E background",
        #                         "size": 10,
        #                         "color": "red"
        #                         },
        #                   scale=(1,) * (points.shape[-1] - 3) + (dz_v / dxy_v, 1, 1)
        #                   )

        # phase difference between measured and bg
        pd = da.mod(da.angle(estack) - da.angle(ebg_stack), 2 * np.pi)
        pd_even = pd - 2*np.pi * (pd > np.pi)
        viewer.add_image(pd_even, scale=(dz_v / dxy_v, 1, 1),
                         name="angle(e) - angle(e bg)", contrast_limits=[-0.1, 0.1],
                         colormap="PiYG",
                         translate=(ny, 0)
                         )

    viewer.dims.axis_labels = ["position", "time", "", "", "pattern", "z", "y", "x"]
    # set to first position
    viewer.dims.set_current_step(axis=0, value=0)
    # set to first time
    viewer.dims.set_current_step(axis=1, value=0)

    # block until closed by user
    viewer.show(block=block_while_display)

    return viewer
