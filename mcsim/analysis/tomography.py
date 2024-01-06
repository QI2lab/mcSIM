"""
Tools for reconstructing optical diffraction tomography (ODT) data using either the Born approximation,
 Rytov approximation, or multislice (paraxial) beam propogation method (BPM) and a FISTA solver. The primary
 reconstruction tasks are carried out with the tomography class
"""
import time
import datetime
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Sequence
from functools import partial
import numpy as np
from numpy import fft
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.signal.windows import tukey, hann
from skimage.restoration import unwrap_phase
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask_image.ndfilters import convolve as dconvolve
# plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Arc, Rectangle
# saving/loading
import zarr
# custom tools
from localize_psf import fit, rois, camera, affine
from mcsim.analysis.analysis_tools import translate_ft
from mcsim.analysis.field_prop import frqs2angles, get_fzs
from mcsim.analysis import field_prop
from mcsim.analysis.phase_unwrap import phase_unwrap as weighted_phase_unwrap
from mcsim.analysis.optimize import Optimizer, soft_threshold, tv_prox, to_cpu
from mcsim.analysis.fft import ft3, ift3, ft2, ift2

_gpu_available = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as sp_gpu
    import cupyx.scipy.fft as fft_gpu
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
        Reconstruct optical diffraction tomography data

        :param imgs_raw: n1 x n2 x ... x nm x npatterns x ny x nx. Data intensity images
        :param wavelength: wavelength in um
        :param no: background index of refraction
        :param na_detection:
        :param na_excitation:
        :param dxy: pixel size in um
        :param reference_frq_guess: [fx, fy] hologram reference frequency
        :param hologram_frqs_guess: npatterns x nmulti x 2 array
        :param imgs_raw_bg: background intensity images. If no background images are provided, then a time
        average of imgs_raw will be used as the background
        :param phase_offsets: phase shifts between images and corresponding background images
        :param axes_names: names of first m + 1 axes
        :param verbose:
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

        # todo: don't allow different multiplex for different patterns
        # list of length npatterns, with each entry [1 x 1 ... x 1 x nmultiplex x 2]
        # with potential different nmultiplex for each pattern
        self.hologram_frqs = hologram_frqs_guess
        self.hologram_frqs_bg = None
        self.nmax_multiplex = np.max([f.shape[0] for f in self.hologram_frqs])

        # physical parameters
        self.wavelength = wavelength
        self.no = no
        self.na_detection = na_detection
        self.na_excitation = na_excitation
        self.fmax = self.na_detection / self.wavelength

        # correction parameters
        # self.phase_offsets = phase_offsets
        self.phase_params = None
        self.phase_params_bg = None
        self.translations = None
        self.translations_bg = None

        # electric fields
        self.holograms_ft = None
        self.holograms_ft_bg = None

        # coordinates of ground truth image
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
                               roi_size_pix: int = 11,
                               save_dir: Optional[str] = None,
                               use_fixed_frequencies: bool = False,
                               fit_on_gpu: bool = False):
        """
        Estimate hologram frequencies from raw images.
        Guess values need to be within a few pixels for this to succeed. Can easily achieve this accuracy by
        looking at FFT
        :param roi_size_pix: ROI size (in pixels) to use for frequency fitting
        :param save_dir:
        :param use_fixed_frequencies: determine single set of frequencies for all data/background images
        :param fit_on_gpu: do fitting on GPU with gpufit. Otherwise use CPU
        :return:
        """
        self.reconstruction_settings.update({"roi_size_pix": roi_size_pix})
        self.reconstruction_settings.update({"use_fixed_frequencies": use_fixed_frequencies})

        # fitting logic
        # todo: re-implement!
        if not use_fixed_frequencies:
            slices_bg = tuple([slice(None) for _ in range(self.nextra_dims)])
        else:
            slices_bg = tuple([slice(0, 1) for _ in range(self.nextra_dims)])

        fit_data_imgs = not use_fixed_frequencies or self.use_average_as_background
        fit_bg_imgs = not self.use_average_as_background

        # create array for ref frqs
        # NOTE only works for equal multiplex for all images!
        hologram_frqs_guess = np.stack(self.hologram_frqs, axis=0)

        saving = save_dir is not None
        if saving:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # frequencies
        fxs = self.fxs
        fys = self.fys
        dfx = fxs[1] - fxs[0]
        dfy = fys[1] - fys[0]

        apodization = np.outer(hann(self.ny),
                               hann(self.nx))

        # get rois
        cx_pix = np.round((hologram_frqs_guess[..., 0] - fxs[0]) / dfx).astype(int)
        cy_pix = np.round((hologram_frqs_guess[..., 1] - fys[0]) / dfy).astype(int)
        c_guess = np.stack((cy_pix, cx_pix), axis=-1)

        rois_all = rois.get_centered_rois(c_guess,
                                          [roi_size_pix, roi_size_pix],
                                          min_vals=(0, 0),
                                          max_vals=(self.ny, self.nx)
                                          )

        xx, yy = np.meshgrid(range(roi_size_pix), range(roi_size_pix))

        # cut rois
        def cut_rois(img: array,
                     block_id=None):
            img_ft = ft2(img * apodization)

            npatt, ny, nx = img_ft.shape[-3:]
            nroi = rois_all.shape[1]
            roi_out = np.zeros(img_ft.shape[:-3] + (npatt, nroi, roi_size_pix, roi_size_pix))
            for ii in range(npatt):
                roi_out[..., ii, :, :, :] = abs(np.stack(rois.cut_roi(rois_all[ii], img_ft[..., ii, :, :]), axis=-3))

            return roi_out

        if fit_data_imgs:
            rois_cut = da.map_blocks(cut_rois,
                                     self.imgs_raw,
                                     drop_axis=(-1, -2),
                                     new_axis=(-1, -2, -3),
                                     chunks=self.imgs_raw.chunksize[:-2] + (self.nmax_multiplex, roi_size_pix, roi_size_pix),
                                     dtype=float,
                                     meta=np.array((), dtype=float))
        else:
            rois_cut = None

        if fit_bg_imgs:
            rois_cut_bg = da.map_blocks(cut_rois,
                                        self.imgs_raw_bg,
                                        drop_axis=(-1, -2),
                                        new_axis=(-1, -2, -3),
                                        chunks=self.imgs_raw_bg.chunksize[:-2] + (self.nmax_multiplex, roi_size_pix, roi_size_pix),
                                        dtype=float,
                                        meta=np.array((), dtype=float))[slices_bg]
        else:
            rois_cut_bg = None

        def fit_rois_cpu(img_rois,
                         model=fit.gauss2d_symm(),
                         # model=fit.gauss2d()
                         ):

            centers = np.zeros(img_rois.shape[:-2] + (2,))

            n_loop = np.prod(img_rois.shape[:-2])
            for ii in range(n_loop):
                ind = np.unravel_index(ii, img_rois.shape[:-2])
                rgauss = model.fit(img_rois[ind],
                                   (yy, xx),
                                   init_params=None,
                                   guess_bounds=True)
                centers[ind] = rgauss["fit_params"][1:3]

            return centers

        def fit_rois_gpu(rois_cut):
            data_shape = (np.prod(rois_cut.shape[:-2]), roi_size_pix ** 2)
            data = rois_cut.astype(np.float32).compute().reshape(data_shape)

            init_params = np.zeros((data_shape[0], 5), dtype=np.float32)

            # amplitude
            init_params[:, 0] = data.max(axis=-1)

            # center
            imax = np.argmax(data, axis=-1)
            iy_max, ix_max = np.unravel_index(imax, (roi_size_pix, roi_size_pix))
            init_params[:, 1] = ix_max
            init_params[:, 2] = iy_max

            # size
            init_params[:, 3] = 1
            init_params[:, 4] = data.min(axis=-1)

            fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data,
                                                                     None,
                                                                     gf.ModelID.GAUSS_2D,
                                                                     init_params,
                                                                     tolerance=1e-8,
                                                                     max_number_iterations=100,
                                                                     estimator_id=gf.EstimatorID.LSE)
            cx = fit_params[:, 1].reshape(rois_cut.shape[:-2])
            cy = fit_params[:, 2].reshape(rois_cut.shape[:-2])

            return cx, cy, fit_params, init_params, fit_states

        if not fit_on_gpu:
            if fit_data_imgs:
                centers = da.map_blocks(fit_rois_cpu,
                                        rois_cut,
                                        drop_axis=(-1, -2),
                                        new_axis=(-1),
                                        chunks=(rois_cut.chunksize[:-2] + (2,)),
                                        dtype=float,
                                        meta=np.array((), dtype=float)).compute()
                cx = centers[..., 0]
                cy = centers[..., 1]

            if fit_bg_imgs:
                centers_bg = da.map_blocks(fit_rois_cpu,
                                           rois_cut_bg,
                                           drop_axis=(-1, -2),
                                           new_axis=(-1),
                                           chunks=(rois_cut_bg.chunksize[:-2] + (2,)),
                                           dtype=float,
                                           meta=np.array((), dtype=float)).compute()
                cx_bg = centers_bg[..., 0]
                cy_bg = centers_bg[..., 1]

        else:
            # todo: maybe detect fits that failed and redo?
            if fit_data_imgs:
                cx, cy, fit_params, init_params, fit_states = fit_rois_gpu(rois_cut)

            if fit_bg_imgs:
                cx_bg, cy_bg, _, _, _ = fit_rois_gpu(rois_cut_bg)

        # final frequencies
        if fit_data_imgs:
            frqs_hologram = np.stack((cx * dfx + fxs[rois_all[..., 2]],
                                      cy * dfy + fys[rois_all[..., 0]]), axis=-1)
        else:
            frqs_hologram = None

        if fit_bg_imgs:
            frqs_hologram_bg = np.stack((cx_bg * dfx + fxs[rois_all[..., 2]],
                                         cy_bg * dfy + fys[rois_all[..., 0]]), axis=-1)
        else:
            frqs_hologram_bg = None

        if self.use_average_as_background:
            frqs_hologram_bg = frqs_hologram

        if use_fixed_frequencies:
            frqs_hologram = frqs_hologram_bg

        self.hologram_frqs = [frqs_hologram[..., ii, :, :] for ii in range(self.npatterns)]
        self.hologram_frqs_bg = [frqs_hologram_bg[..., ii, :, :] for ii in range(self.npatterns)]

        # optionally plot
        def plot(img,
                 frqs_holo,
                 frqs_guess=None,
                 rois_all=None,
                 prefix="",
                 figsize=(20, 10),
                 block_id=None):

            img_ft = ft2(img).squeeze()
            if img_ft.ndim != 2:
                raise ValueError()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                figh = plt.figure(figsize=figsize)

            ax = figh.add_subplot(1, 2, 1)
            extent_f = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx,
                        fys[-1] + 0.5 * dfy, fys[0] - 0.5 * dfy]

            ax.set_title("$|I(f)|$")
            ax.imshow(np.abs(img_ft),
                      extent=extent_f,
                      norm=PowerNorm(gamma=0.1),
                      cmap="bone")
            if frqs_guess is not None:
                ax.plot(frqs_guess[:, 0], frqs_guess[:, 1], 'gx')
            ax.plot(frqs_holo[..., 0], frqs_holo[..., 1], 'rx')

            if rois_all is not None:
                for roi in rois_all:
                    ax.add_artist(Rectangle((fxs[roi[2]], fys[roi[0]]),
                                            fxs[roi[3] - 1] - fxs[roi[2]],
                                            fys[roi[1] - 1] - fys[roi[0]],
                                            edgecolor='k',
                                            fill=False))

            ax.set_xlabel("$f_x$ (1/$\mu m$)")
            ax.set_ylabel("$f_y$ (1/$\mu m$)")

            ax = figh.add_subplot(1, 2, 2)
            roi = rois_all[0]
            iroi = rois.cut_roi(roi, img_ft)[0]

            extent_f_roi = [fxs[roi[2]] - 0.5 * dfx,
                            fxs[roi[3] - 1] + 0.5 * dfx,
                            fys[roi[1] - 1] + 0.5 * dfy,
                            fys[roi[0]] - 0.5 * dfy]

            ax.imshow(np.abs(iroi),
                      extent=extent_f_roi,
                      cmap="bone")

            if frqs_guess is not None:
                ax.plot(frqs_guess[0, 0], frqs_guess[0, 1], 'gx')
            ax.plot(frqs_holo[0, 0], frqs_holo[0, 1], 'rx')

            ax.set_xlabel("$f_x$ (1/$\mu m$)")
            ax.set_ylabel("$f_y$ (1/$\mu m$)")

            figh.savefig(Path(save_dir, f"{prefix:s}=hologram_frq_diagnostic.png"))
            plt.close(figh)

        slice_start = tuple([slice(0, 1) for _ in range(self.nextra_dims)])
        axes_start = tuple(range(self.nextra_dims))

        if saving:
            if fit_data_imgs:
                iraw = self.imgs_raw[slice_start].squeeze(axis=axes_start)
                # h = self.hologram_frqs[slice_start].squeeze(axis=axes_start)
                h = [f[slice_start].squeeze(axis=axes_start) for f in self.hologram_frqs]

                delayed = []
                for ii in range(self.npatterns):
                    delayed.append(dask.delayed(plot)(iraw[ii],
                                                      h[ii],
                                                      frqs_guess=hologram_frqs_guess[ii],
                                                      prefix=f"{ii:d}",
                                                      rois_all=rois_all[ii]))
                dask.compute(*delayed)

            if fit_bg_imgs:
                iraw_bg = self.imgs_raw_bg[slice_start].squeeze(axis=axes_start)
                # h = self.hologram_frqs[slice_start].squeeze(axis=axes_start)
                hbg = [f[slice_start].squeeze(axis=axes_start) for f in self.hologram_frqs_bg]

                delayed = []
                for ii in range(self.npatterns):
                    delayed.append(dask.delayed(plot)(iraw_bg[ii],
                                                      hbg[ii],
                                                      frqs_guess=hologram_frqs_guess[ii],
                                                      prefix=f"{ii:d}",
                                                      rois_all=rois_all[ii]))
                dask.compute(*delayed)

    def estimate_reference_frq(self,
                               mode: str = "average",
                               frq: Optional[array] = None,
                               ):
        """
        Estimate hologram reference frequency
        :param mode: if "fit" fit the residual speckle pattern to try and estimate the reference frequency.
         If "average" take the average of self.hologram_frqs as the reference frequency.
         If "set" use the frequency argument to set the value
        :param frq: set reference frequency to this value
        :return figure: a diagnostic figure is generated if mode is "fit", otherwise None is returned
        """
        
        self.reconstruction_settings.update({"reference_frequency_mode": mode})
        figh_ref_frq = None

        if mode == "average":
            frq_ref = np.mean(np.concatenate(self.hologram_frqs, axis=-2), axis=-2)
            frq_ref_bg = np.mean(np.concatenate(self.hologram_frqs_bg, axis=-2), axis=-2)
        elif mode == "set":
            if frq is None:
                raise ValueError("mode was 'set' so a reference frequency must be supplied")
            frq_ref = frq
            frq_ref_bg = frq

        elif mode == "fit":
            raise NotImplementedError("mode 'fit' not implemented after multiplexing code update")

            # load one slice of background data to get frequency reference. Load the first slice along all dimensions
            slices = tuple([slice(0, 1)] * self.nextra_dims + [slice(None)] * 3)
            imgs_frq_cal = np.squeeze(self.imgs_raw_bg[slices])

            imgs_frq_cal_ft = ft2(imgs_frq_cal)
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

        else:
            raise ValueError(f"'mode' must be '{mode:s}' but must be 'fit' or 'average'")

        self.reference_frq = frq_ref
        self.reference_frq_bg = frq_ref_bg

        return figh_ref_frq

    def get_beam_frqs(self):
        """
        Get beam incident beam frequencies from hologram frequencies and reference frequency

        :return beam_frqs: list (length n_patterns) with each element an array of size N1 x N2 ... x Nm x 3
        """

        bxys = [f - np.expand_dims(self.reference_frq, axis=-2) for f in self.hologram_frqs]
        bzs = [get_fzs(bxy[..., 0], bxy[..., 1], self.no, self.wavelength) for bxy in bxys]
        beam_frqs = [np.stack((bxy[..., 0], bxy[..., 1], bz), axis=-1) for bxy, bz in zip(bxys, bzs)]

        return beam_frqs

    def find_affine_xform_to_frqs(self,
                                  offsets: list[np.ndarray],
                                  save_dir: Optional[str] = None,
                                  dmd_size: Optional[tuple[int]] = None):
        """
        Fit affine transformation between device and measured frequency space.

        This could be between frequencies displayed on DMD and measured frequency space (DMD in imaging plane)
        or between mirror positions on DMD and frequency space (DMD in Fourier plane)

        :param offsets:
        :param save_dir:
        :param dmd_size:
        :return:
        """

        if dmd_size is not None:
            ny_dmd, nx_dmd = dmd_size

        centers_dmd = np.concatenate(offsets, axis=0)
        mean_hologram_frqs = np.concatenate([np.mean(f, axis=tuple(range(self.nextra_dims))) for f in self.hologram_frqs], axis=0)
        # mean_hologram_frqs = np.mean(self.hologram_frqs, axis=tuple(range(self.nextra_dims)))
        # mean_hologram_frqs = np.concatenate([hf for hf in mean_hologram_frqs], axis=0)

        mean_ref_frq = np.mean(self.reference_frq, axis=tuple(range(self.nextra_dims)))

        beam_frqs = np.concatenate([np.mean(f, axis=tuple(range(self.nextra_dims))) for f in self.get_beam_frqs()], axis=0)

        # fit affine transformation
        if len(mean_hologram_frqs) > 6:
            xform_dmd2frq, _, _, _ = affine.fit_xform_points_ransac(centers_dmd,
                                                                    mean_hologram_frqs,
                                                                    dist_err_max=0.1,
                                                                    niterations=100)
        elif len(mean_hologram_frqs) > 3:
            # no point in RANSAC if not enough points to invert transformation
            xform_dmd2frq, _ = affine.fit_xform_points(centers_dmd, mean_hologram_frqs)
        else:
            return None

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
        if dmd_size is not None:
            south = np.zeros((nx_dmd, 2))
            south[:, 0] = np.arange(nx_dmd) - (nx_dmd // 2)
            south[:, 1] = 0 - (ny_dmd // 2)

            north = np.zeros((nx_dmd, 2))
            north[:, 0] = np.arange(nx_dmd) - (nx_dmd // 2)
            north[:, 1] = ny_dmd - (ny_dmd // 2)

            east = np.zeros((ny_dmd, 2))
            east[:, 0] = nx_dmd - (nx_dmd // 2)
            east[:, 1] = np.arange(ny_dmd) - (ny_dmd // 2)

            west = np.zeros((ny_dmd, 2))
            west[:, 0] = 0 - (nx_dmd // 2)
            west[:, 1] = np.arange(ny_dmd) - (ny_dmd // 2)

            dmd_boundary = np.concatenate((south, north, east, west), axis=0)
            dmd_boundry_freq = affine.xform_points(dmd_boundary, xform_dmd2frq)

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

        if dmd_size is not None:
            ax.plot(dmd_boundry_freq[:, 0], dmd_boundry_freq[:, 1], 'k.')

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

        if dmd_size is not None:
            ax.plot(dmd_boundry_freq[:, 0] - mean_ref_frq[0], dmd_boundry_freq[:, 1] - mean_ref_frq[1], 'k.')

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
                        fourier_mask: Optional[np.ndarray] = None,
                        fit_phases: bool = False,
                        fit_translations: bool = False,
                        translation_thresh: float = 1/30,
                        apodization: Optional[np.ndarray] = None,
                        use_gpu: bool = False):
        """
        Unmix and preprocess holograms

        Note that this only depends on reference frequencies, and not on determined hologram frequencies

        :param bg_average_axes: axes to average along when producing background images
        :param fourier_mask: regions where this mask is True will be set to 0 during hologram unmixing
        :param fit_phases: whether to fit phase differences between image and background holograms
        :param fit_translations:
        :param translation_thresh:
        :param apodization: if None use tukey apodization with alpha = 0.1. To use no apodization set equal to 1
        :param use_gpu:
        :return:
        """
        self.reconstruction_settings.update({"fit_translations": fit_translations})

        # default values
        self.translations = np.zeros((1,) * self.nextra_dims + (self.npatterns, 1, 1, 2), dtype=float)
        self.translations_bg = np.zeros_like(self.translations)
        self.phase_params = np.ones((1,) * self.nextra_dims + (self.npatterns, 1, 1), dtype=complex)
        self.phase_params_bg = np.ones_like(self.phase_params)

        # slice used as reference for computing phase shifts/translations/etc
        # if we are going to average along a dimension (i.e. if it is in bg_average_axes) then need to use
        # single slice as background for that dimension.
        ref_slice = tuple([slice(0, 1) if a in bg_average_axes else slice(None) for a in range(self.nextra_dims)] +
                       [slice(None)] * 3)

        # set up
        if use_gpu and _gpu_available:
            xp = cp
        else:
            xp = np

        # #########################
        # get electric field from holograms
        # #########################
        if apodization is None:
            apodization = xp.outer(tukey(self.ny, alpha=0.1),
                                   tukey(self.nx, alpha=0.1))
        else:
            apodization = xp.asarray(apodization)

        # make broadcastable to same size as raw images so can use with dask array
        ref_frq_da = da.from_array(np.expand_dims(self.reference_frq, axis=(-2, -3, -4)),
                                   chunks=self.imgs_raw.chunksize[:-2] + (1, 1, 2)
                                   )

        if self.use_average_as_background:
            ref_frq_bg_da = ref_frq_da
        else:
            ref_frq_bg_da = da.from_array(np.expand_dims(self.reference_frq_bg, axis=(-2, -3, -4)),
                                          chunks=self.imgs_raw_bg.chunksize[:-2] + (1, 1, 2)
                                          )

        if fourier_mask is None:
            masks_da = None
        else:
            # ensure masks has same dims and chunks as imgs_raw
            masks_da = da.from_array(xp.array(fourier_mask),
                                     chunks=(1,) * (fourier_mask.ndim - 2) + (self.ny, self.nx))

        holograms_ft = da.map_blocks(unmix_hologram,
                                     self.imgs_raw,
                                     self.dxy,
                                     2*self.fmax,
                                     ref_frq_da[..., 0],
                                     ref_frq_da[..., 1],
                                     mask=masks_da,
                                     apodization=apodization,
                                     dtype=complex)

        # #########################
        # get background electric field from holograms
        # #########################
        if self.use_average_as_background:
            holograms_ft_bg = holograms_ft
        else:
            holograms_ft_bg = da.map_blocks(unmix_hologram,
                                            self.imgs_raw_bg,
                                            self.dxy,
                                            2*self.fmax,
                                            ref_frq_bg_da[..., 0],
                                            ref_frq_bg_da[..., 1],
                                            mask=masks_da,
                                            apodization=apodization,
                                            dtype=complex)

        # compute reference
        # holo_ft_ref = holograms_ft_bg[ref_slice].compute()

        # #########################
        # fit translations between signal and background electric fields
        # #########################
        if fit_translations:
            print("computing translations")

            holograms_abs_ft = da.map_blocks(_ft_abs,
                                             holograms_ft,
                                             dtype=complex)

            if self.use_average_as_background:
                holograms_abs_ft_bg = holograms_abs_ft
            else:
                holograms_abs_ft_bg = da.map_blocks(_ft_abs,
                                                    holograms_ft_bg,
                                                    dtype=complex)

            # fit phase ramp in holograms ft
            self.translations = da.map_blocks(fit_phase_ramp,
                                              holograms_abs_ft,
                                              holograms_abs_ft_bg[ref_slice],
                                              self.dxy,
                                              thresh=translation_thresh,
                                              dtype=float,
                                              new_axis=-1,
                                              chunks=holograms_abs_ft.chunksize[:-2] + (1, 1, 2)).compute()

            # correct translations
            def translate(e_ft, dxs, dys, fxs, fys):
                e_ft_out = xp.array(e_ft, copy=True)

                fx_bcastable = xp.expand_dims(fxs, axis=(-3, -2))
                fy_bcastable = xp.expand_dims(fys, axis=(-3, -1))

                e_ft_out *= np.exp(2*np.pi * 1j * (fx_bcastable * dxs +
                                                   fy_bcastable * dys))

                return e_ft_out

            dr_chunks = holograms_ft.chunksize[:-2] + (1, 1)
            dxs = da.from_array(self.translations[..., 0], chunks=dr_chunks)
            dys = da.from_array(self.translations[..., 1], chunks=dr_chunks)
            holograms_ft = da.map_blocks(translate,
                                         holograms_ft,
                                         dxs,
                                         dys,
                                         self.fxs,
                                         self.fys,
                                         dtype=complex,
                                         meta=xp.array((), dtype=complex))

            if self.use_average_as_background:
                self.translations_bg = self.translations
                holograms_ft_bg = holograms_ft
            else:
                self.translations_bg = da.map_blocks(fit_phase_ramp,
                                                     holograms_abs_ft_bg,
                                                     holograms_abs_ft_bg[ref_slice],
                                                     self.dxy,
                                                     thresh=translation_thresh,
                                                     dtype=float,
                                                     new_axis=-1,
                                                     chunks=holograms_abs_ft.chunksize[:-2] + (1, 1, 2)).compute()

                dxs_bg = da.from_array(self.translations_bg[..., 0], chunks=dr_chunks)
                dys_bg = da.from_array(self.translations_bg[..., 1], chunks=dr_chunks)

                holograms_ft_bg = da.map_blocks(translate,
                                                holograms_ft_bg,
                                                dxs_bg,
                                                dys_bg,
                                                self.fxs,
                                                self.fys,
                                                dtype=complex,
                                                meta=xp.array((), dtype=complex))

        # #########################
        # determine phase offsets for background electric field, relative to initial slice
        # for each angle so we can average this together to produce a single "background" image
        # #########################
        print("computing background phase shifts")
        if fit_phases:
            self.phase_params_bg = da.map_blocks(get_global_phase_shifts,
                                                 holograms_ft_bg,
                                                 holograms_ft_bg[ref_slice],  # reference slices
                                                 dtype=complex,
                                                 chunks=holograms_ft_bg.chunksize[:-2] + (1, 1)
                                                 ).compute()

        # #########################
        # determine background electric field
        # #########################
        print("computing background electric field")
        holograms_ft_bg_comp = da.mean(holograms_ft_bg * self.phase_params_bg,
                                       axis=bg_average_axes,
                                       keepdims=True)

        self.holograms_ft_bg = da.from_array(holograms_ft_bg_comp.compute(),
                                             chunks=holograms_ft_bg.chunksize)

        # #########################
        # determine phase offsets between electric field and background
        # #########################
        print("computing phase offsets")
        if self.use_average_as_background:
            self.phase_params = self.phase_params_bg
        else:
            if fit_phases:
                self.phase_params = da.map_blocks(get_global_phase_shifts,
                                                  holograms_ft,
                                                  self.holograms_ft_bg,
                                                  dtype=complex,
                                                  chunks=holograms_ft.chunksize[:-2] + (1, 1),
                                                  ).compute()

        self.holograms_ft = holograms_ft * self.phase_params

        return self.holograms_ft, self.holograms_ft_bg

    def reconstruct_n(self,
                      efields_ft: array,
                      efields_bg_ft: array,
                      drs_n: tuple[float],
                      n_size: tuple[int],
                      mode: str = "rytov",
                      scattered_field_regularization: float = 50.,
                      realspace_mask: Optional[np.ndarray] = None,
                      step: float = 1e-5,
                      use_gpu: bool = False,
                      n_guess: Optional[array] = None,
                      cam_roi: Optional[list] = None,
                      data_roi: Optional[list] = None,
                      use_weighted_phase_unwrap: bool = False,
                      e_fwd_out: Optional[array] = None,
                      e_scatt_out: Optional[array] = None,
                      n_start_out: Optional[array] = None,
                      **kwargs) -> (array, tuple, dict):

        """
        Reconstruct refractive index using one of a several different models

        :param efields_ft: typically derived from unmix_holograms()
        :param efields_bg_ft: typically derived from unmix_holograms()
        :param drs_n: (dz, dy, dx) voxel size of reconstructed refractive index
        :param n_size: (nz, ny, nx) shape of reconstructred refractive index # todo: make n_shape
        :param mode: "born", "rytov", "bpm", or "ssnp"
        :param scattered_field_regularization: regularization used in computing scattered field
          or Rytov phase. Regions where background electric field is smaller than this value will be suppressed                
        :param realspace_mask: indicate which parts of image to exclude/keep
        :param step: ignored if mode is "born" or "rytov"
        :param use_gpu:
        :param n_guess:
        :param cam_roi:
        :param data_roi:
        :param use_weighted_phase_unwrap:
        :param e_fwd_out: Zarr array. If provided, compute and write the predicted forward electric field
          from the inferred refractive index. chunk-size should be (1, ..., 1, ny, nx) for fastests saving
        :param e_scatt_out:
        :param n_start_out:
        :param **kwargs: passed through to both the constructor and the run() method of the optimizer.
          These are used to e.g. set the strength of TV regularization, the number of iterations, etc.
          Optimizer, RIOptimizer, and classes inheriting from RIOptimizer for more details
        :return n, drs_n, xforms:
        """

        if use_gpu and _gpu_available:
            xp = cp
        else:
            xp = np

        # ############################
        # get beam frequencies
        # ############################
        beam_frqs = self.get_beam_frqs()
        mean_beam_frqs = [np.mean(f, axis=tuple(range(self.nextra_dims))) for f in beam_frqs]

        # convert to array ... for images which don't have enough multiplexed frequencies, replaced by inf
        nmax_multiplex = np.max([len(f) for f in mean_beam_frqs])
        mean_beam_frqs_arr = np.ones((nmax_multiplex, self.npatterns, 3), dtype=float) * np.inf
        for aaa in range(self.npatterns):
            mean_beam_frqs_arr[:, aaa, :] = mean_beam_frqs[aaa]

        # beam frequencies with multiplexed freqs raveled
        mean_beam_frqs_no_multi = np.zeros([self.npatterns * nmax_multiplex, 3])
        for ii in range(self.npatterns):
            for jj in range(nmax_multiplex):
                mean_beam_frqs_no_multi[ii * nmax_multiplex + jj, :] = mean_beam_frqs_arr[jj, ii]

        # ############################
        # check arrays are chunked by volume
        # ############################
        # todo: rechunk here if necessary ...
        if self.holograms_ft.chunksize[-3] != self.npatterns:
            raise ValueError("")

        if self.holograms_ft_bg.chunksize[-3] != self.npatterns:
            raise ValueError("")

        # ############################
        # compute information we need for reconstructions e.g. linear models and dz_final
        # ############################
        # generate ATF ... ultimately want to do this based on pupil function defined in init
        fx_atf = xp.fft.fftshift(xp.fft.fftfreq(n_size[-1], drs_n[-1]))
        fy_atf = xp.fft.fftshift(xp.fft.fftfreq(n_size[-2], drs_n[-2]))
        atf = (xp.sqrt(fx_atf[None, :] ** 2 + fy_atf[:, None] ** 2) <= self.fmax).astype(complex)
        fxfx, fyfy = xp.meshgrid(fx_atf, fy_atf)

        apodization_n = xp.outer(xp.asarray(tukey(n_size[-2], alpha=0.1)),
                                 xp.asarray(tukey(n_size[-1], alpha=0.1)))

        if mode == "born" or mode == "rytov":
            optimizer = mode
            dz_final = None
            # affine transformation from reconstruction coordinates to pixel indices
            # for reconstruction, using FFT induced coordinates, i.e. zero is at array index (ny // 2, nx // 2)
            # for matrix, using image coordinates (0, 1, ..., N - 1)
            # note, due to order of operations -n//2 =/= - (n//2) when nx is odd
            xform_recon_pix2coords = affine.params2xform([drs_n[-1], 0, -(n_size[-1] // 2) * drs_n[-1],
                                                          drs_n[-2], 0, -(n_size[-2] // 2) * drs_n[-2]])

            linear_model = fwd_model_linear(mean_beam_frqs_arr[..., 0],
                                            mean_beam_frqs_arr[..., 1],
                                            mean_beam_frqs_arr[..., 2],
                                            self.no,
                                            self.na_detection,
                                            self.wavelength,
                                            (self.ny, self.nx),
                                            (self.dxy, self.dxy),
                                            n_size,
                                            drs_n,
                                            mode=mode,
                                            interpolate=True,
                                            use_gpu=use_gpu)

            if self.verbose:
                tstart_step = time.perf_counter()
                print("estimating step size")

            mguess = LinearScatt(np.empty((1, self.ny, self.nx)),
                                 linear_model, self.no, self.wavelength, None, None, None)
            step = mguess.guess_step()

            if self.verbose:
                print(f"estimated in {time.perf_counter() - tstart_step:.2f}s")

        elif mode == "bpm" or mode == "ssnp":
            if mode == "bpm":
                optimizer = BPM
            elif mode == "ssnp":
                optimizer = SSNP

            if not (self.ny / n_size[1]).is_integer():
                raise ValueError()
            if not (self.nx / n_size[2]).is_integer():
                raise ValueError()

            nbin_y = int(self.ny // n_size[1])
            nbin_x = int(self.nx // n_size[2])

            if drs_n[1] != self.dxy * nbin_x or drs_n[2] != self.dxy * nbin_y:
                raise ValueError()

            # position z=0 in the middle of the volume, accounting for the fact
            # the efields and n are shifted by dz/2
            dz_final = -drs_n[0] * ((n_size[0] - 1) - n_size[0] // 2 + 0.5)

            # affine transformation from reconstruction coordinates to pixel indices
            # coordinates in finer coordinates
            xb = camera.bin((xp.arange(self.nx) - (self.nx // 2)) * self.dxy,
                            [nbin_x], mode="mean")
            yb = camera.bin((xp.arange(self.ny) - (self.ny // 2)) * self.dxy,
                            [nbin_y], mode="mean")

            xform_recon_pix2coords = affine.params2xform([drs_n[-1], 0, float(xb[0]),
                                                          drs_n[-2], 0, float(yb[0])])

        else:
            raise ValueError(f"mode must be ..., but was {mode:s}")

        # general info
        if self.verbose:
            print(f"computing index of refraction for {np.prod(self.imgs_raw.shape[:-3]):d} images "
                  f"using mode {mode:s}.\n"
                  f"Image size = {self.npatterns} x {self.ny:d} x {self.nx:d},\n"
                  f"reconstruction size = {n_size[0]:d} x {n_size[1]:d} x {n_size[2]:d}")

        # #############################
        # initial guess
        # #############################
        if self.verbose:
            tstart_linear_model = time.perf_counter()

        if n_guess is not None:
            linear_model_invert = None
        else:
            linear_model_invert = fwd_model_linear(mean_beam_frqs_no_multi[..., 0],
                                                   mean_beam_frqs_no_multi[..., 1],
                                                   mean_beam_frqs_no_multi[..., 2],
                                                   self.no,
                                                   self.na_detection,
                                                   self.wavelength,
                                                   (self.ny, self.nx),
                                                   (self.dxy, self.dxy),
                                                   n_size,
                                                   drs_n,
                                                   mode="born" if mode == "born" else "rytov",
                                                   interpolate=False,
                                                   use_gpu=use_gpu)

        if self.verbose:
            print(f"Generated linear model for initial guess in {time.perf_counter() - tstart_linear_model:.2f}s")

        # #############################
        # reconstruction
        # #############################
        def recon(efields_ft,
                  efields_bg_ft,
                  beam_frqs,
                  rmask,
                  dxy,
                  no,
                  wavelength,
                  na_detection,
                  dz_final,
                  atf,
                  apod,
                  step,
                  optimizer,
                  verbose,
                  n_guess=None,
                  e_fwd_out=None,
                  e_scatt_out=None,
                  n_start_out=None,
                  block_id=None):

            nextra_dims = efields_ft.ndim - 3
            dims = tuple(range(nextra_dims))
            efields_ft = efields_ft.squeeze(axis=dims)
            efields_bg_ft = efields_bg_ft.squeeze(axis=dims)
            nimgs, ny, nx = efields_ft.shape

            if block_id is None:
                block_ind = None
            else:
                block_ind = block_id[:nextra_dims]

            if use_gpu:
                efields_ft = xp.asarray(efields_ft)
                efields_bg_ft = xp.asarray(efields_bg_ft)

                if rmask is not None:
                    rmask = xp.asarray(rmask)

            # #######################
            # get initial guess
            # #######################
            if optimizer == "born":
                scatt_fn = get_scattered_field
            else:
                scatt_fn = partial(get_rytov_phase, use_weighted_unwrap=use_weighted_phase_unwrap)

            if n_guess is not None:
                v_fts_start = ft3(get_v(xp.asarray(n_guess), no, wavelength), no_cache=True)
            else:
                tstart_scatt = time.perf_counter()
                if nmax_multiplex == 1:
                    efield_scattered = scatt_fn(ift2(efields_ft),
                                                ift2(efields_bg_ft),
                                                scattered_field_regularization)
                    efield_scattered_ft = ft2(efield_scattered)
                else:
                    e_unmulti = xp.zeros((nimgs * nmax_multiplex, ny, nx), dtype=complex)
                    ebg_unmulti = xp.zeros_like(e_unmulti)

                    for ii in range(nimgs):
                        for jj in range(nmax_multiplex):
                            dists = np.linalg.norm(mean_beam_frqs_arr[:, ii, :2] -
                                                   mean_beam_frqs_arr[jj, ii, :2], axis=1)
                            min_dist = 0.5 * np.min(dists[dists > 0.])
                            # min_dist = f_radius_factor * na_detection / wavelength

                            mask = xp.sqrt((fxfx - mean_beam_frqs_arr[jj, ii, 0]) ** 2 +
                                           (fyfy - mean_beam_frqs_arr[jj, ii, 1]) ** 2) > min_dist
                            e_unmulti[ii * nmax_multiplex + jj] = ift2(cut_mask(efields_ft[ii], mask))
                            ebg_unmulti[ii * nmax_multiplex + jj] = ift2(cut_mask(efields_bg_ft[ii], mask))

                    # compute scattered field
                    efield_scattered = scatt_fn(e_unmulti,
                                                ebg_unmulti,
                                                scattered_field_regularization)
                    efield_scattered_ft = ft2(efield_scattered)
                    del e_unmulti
                    del ebg_unmulti

                    # set regions we don't want to use to nans
                    # todo: testing ... I don't really think this is needed?
                    # if optimizer == "born":
                    #     raise NotImplementedError("demultiplexing not implemented for mode 'born'")
                    # else:
                    #     efield_scattered_ft[:, xp.sqrt(fxfx**2 + fyfy**2) > f_radius_factor * na_detection / wavelength] = np.nan

                # optionally export scattered field
                try:
                    if e_scatt_out is not None:
                        e_scatt_out[block_ind] = to_cpu(efield_scattered)

                    del efield_scattered
                except NameError:
                    pass

                v_fts_start = inverse_model_linear(efield_scattered_ft,
                                                   linear_model_invert,
                                                   n_size,
                                                   no_data_value=0.)

                if verbose:
                    print(f"computing scattered field took {time.perf_counter() - tstart_scatt:.2f}s")

            # ############
            # optionally write out starting refractive index
            # ############
            if n_start_out is not None:
                if self.verbose:
                    print("saving n_start")
                n_start_out[block_ind] = to_cpu(get_n(ift3(v_fts_start, no_cache=True), no, wavelength))

            print(f"starting inference")
            if use_gpu and verbose:
                print(f"gpu memory usage = {cp.get_default_memory_pool().used_bytes() / 1e9:.2f}GB")
                print(cp.fft.config.get_plan_cache())

            if optimizer == "born" or optimizer == "rytov":
                model = LinearScatt(efield_scattered_ft,
                                    linear_model,
                                    no,
                                    wavelength,
                                  (dxy, dxy),
                                    drs_n,
                                    v_fts_start.shape,
                                    **kwargs
                                    )

                results = model.run(v_fts_start,
                                    step=step,
                                    verbose=verbose,
                                    **kwargs
                                    )
                n = get_n(ift3(results["x"]), no, wavelength)

            else:
                # delete variables we no longer need
                try:
                    del efield_scattered_ft
                except NameError:
                    pass

                n_start = get_n(ift3(v_fts_start, no_cache=True), no, wavelength)
                del v_fts_start

                efields = camera.bin(ift2(efields_ft), [nbin_y, nbin_x], mode="mean")
                del efields_ft

                efields_bg = camera.bin(ift2(efields_bg_ft), [nbin_y, nbin_x], mode="mean")
                del efields_bg_ft

                model = optimizer(efields,
                                  efields_bg,
                                  beam_frqs[0] if beam_frqs.shape[0] == 1 else None,
                                  no,
                                  wavelength,
                                  drs_e=None,
                                  drs_n=drs_n,
                                  shape_n=n_start.shape,
                                  dz_final=dz_final,
                                  atf=atf,
                                  apodization=apod,
                                  mask=rmask,
                                  **kwargs)

                results = model.run(n_start,
                                    step=step,
                                    verbose=verbose,
                                    **kwargs
                                    )
                n = results["x"]

            # ################
            # optionally compute predicated e_field based on n
            # ################
            if e_fwd_out is not None:
                tstart_efwd = time.perf_counter()
                if self.verbose:
                    print("computing forward model")

                if optimizer == "born" or optimizer == "rytov":
                    e_fwd_out[block_ind] = to_cpu(ift2(model.fwd_model(ft3(get_v(n, no, wavelength)))))
                else:
                    slices = (slice(0, 1), slice(-1, None), slice(None), slice(None))  # [0, -1, :, :]

                    tstart_efwd = time.perf_counter()
                    for ii in range(nimgs):
                        ind_now = block_ind + (ii,)
                        e_fwd_out[ind_now] = to_cpu(model.fwd_model(n.squeeze(), inds=[ii])[slices]).squeeze()

                if self.verbose:
                    print(f"computed forward model in {time.perf_counter() - tstart_efwd:.2f}s")

            if self.verbose and use_gpu:
                print(f"gpu memory usage = {cp.get_default_memory_pool().used_bytes() / 1e9:.2f}GB")
                print(cp.fft.config.get_plan_cache())

            # todo: reshape here?
            return to_cpu(n).reshape((1,) * nextra_dims + n_size)

        # #######################
        # get refractive index
        # #######################
        n = da.map_blocks(recon,
                          efields_ft, # data
                          efields_bg_ft, # background
                          mean_beam_frqs_arr,
                          realspace_mask, # masks
                          self.dxy,
                          self.no,
                          self.wavelength,
                          self.na_detection,
                          dz_final,
                          atf,
                          apodization_n,
                          step,
                          optimizer,
                          self.verbose,
                          n_guess=n_guess,
                          e_fwd_out=e_fwd_out,
                          e_scatt_out=e_scatt_out,
                          n_start_out=n_start_out,
                          chunks=(1,) * self.nextra_dims + n_size,
                          dtype=complex,
                          )

        self.reconstruction_settings.update({"mode": mode,
                                             "scattered_field_regularization": scattered_field_regularization,
                                             "dz_final": dz_final,
                                             # "nbin": (nbin_y, nbin_x),
                                             "step": step,
                                             "use_gpu": use_gpu,
                                             }
                                            )
        self.reconstruction_settings.update(kwargs)

        # ############################
        # construct affine tranforms between reconstructed data and camera pixels
        # ############################

        # affine transformation from camera ROI coordinates to pixel indices
        xform_raw_roi_pix2coords = affine.params2xform([self.dxy, 0, -(n_size[-1] // 2) * self.dxy,
                                                        self.dxy, 0, -(n_size[-2] // 2) * self.dxy])

        # composing these two transforms gives affine from recon pixel indices to
        # recon pix inds -> recon coords = ROI coordinates -> ROI pix inds
        xform_recon2raw_roi = np.linalg.inv(xform_raw_roi_pix2coords).dot(xform_recon_pix2coords)

        # store all transforms
        xforms = {"xform_recon_pix2coords": xform_recon_pix2coords,
                  "affine_xform_recon_2_raw_process_roi": xform_recon2raw_roi}

        if data_roi is not None:
            xforms["processing roi"] = np.asarray(data_roi)
            # transform from reconstruction processing roi to camera roi
            odt_recon_roi = deepcopy(data_roi)
            xform_process_roi_to_cam_roi = affine.params2xform([1, 0, odt_recon_roi[2],
                                                                1, 0, odt_recon_roi[0]])
            xform_odt_recon_to_cam_roi = xform_process_roi_to_cam_roi.dot(xform_recon2raw_roi)

            xforms.update({"affine_xform_recon_2_raw_camera_roi": xform_odt_recon_to_cam_roi,})
            if cam_roi is not None:
                xforms["camera roi"] = np.asarray(cam_roi)

                # transform from camera roi to uncropped chip
                xform_cam_roi_to_full = affine.params2xform([1, 0, cam_roi[2],
                                                             1, 0, cam_roi[0]])
                xform_odt_recon_to_full = xform_cam_roi_to_full.dot(xform_process_roi_to_cam_roi)
                xforms.update({"affine_xform_recon_2_raw_camera": xform_odt_recon_to_full})

        return n, xforms, atf

    def plot_translations(self,
                          index: tuple[int],
                          time_axis: int = 1,
                          figsize: tuple[float] = (30., 8.),
                          **kwargs):

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

        # add slices for multiplexed dimension and pattern, y, x dimensions x coord
        slices = tuple(slices)
        squeeze_axes = tuple([ii for ii in range(self.nextra_dims) if ii != time_axis]) + (-2, -3)

        # ####################################
        # hologram frequencies
        # ####################################
        translations = self.translations[slices].squeeze(axis=squeeze_axes)
        translations_bg = self.translations_bg[slices].squeeze(axis=squeeze_axes)

        # plot
        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle(f"index={index}\ntranslation versus time")

        # plot frequency differences
        ax = figh.add_subplot(1, 2, 1)
        ax.plot(translations[..., 0], '.-', label="sig")
        ax.plot(translations_bg[..., 0], label="bg")
        ax.set_xlabel("time step")
        ax.set_ylabel("x-position (um)")
        ax.set_title("x-position")

        ax = figh.add_subplot(1, 2, 2)
        ax.plot(translations[..., 1], '.-', label="sig")
        ax.plot(translations_bg[..., 1], label="bg")
        ax.set_xlabel("time step")
        ax.set_ylabel("y-position (um)")
        ax.set_title("y-position")

        return figh

    def plot_frqs(self,
                  index: tuple[int],
                  time_axis: int = 1,
                  figsize: tuple[float] = (30., 8.),
                  **kwargs):
        """

        :param index: should be of length self.nextra_dims - 1. Index along these axes, but ignoring whichever
        axes is the time axis. So e.g. if the axis are position x time x z x parameter then time_axis = 1 and the index
        could be (2, 1, 0) which would selection position 2, z 1, parameter 0.
        :param time_axis:
        :param figsize:
        :param kwargs: passed through to matplotlib.pyplot.figure
        :return:
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
        ax.plot(np.linalg.norm(hgram_frq_diffs, axis=-1) / self.dfx, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("(frequency - mean) / dfx")
        ax.set_title("hologram frequency deviation amplitude")
        ax.legend([f"{ii:d}" for ii in range(self.npatterns)])

        # plot angles
        angles_unwrapped = np.unwrap(np.angle(hgram_frq_diffs[..., 0] + 1j * hgram_frq_diffs[..., 1]))

        ax = figh.add_subplot(1, 3, 2)
        ax.plot(angles_unwrapped, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("angle (rad)")
        ax.set_title("hologram frequency deviation rotation")

        # plot mean frequency differences
        ax = figh.add_subplot(1, 3, 3)
        ax.plot(np.linalg.norm(ref_frq_diffs, axis=-1) / self.dfx, '.-')
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

        :param index: should be of length self.nextra_dims - 1. Index along these axes, but ignoring whichever
        axes is the time axis. So e.g. if the axis are position x time x z x parameter then time_axis = 1 and the index
        could be (2, 1, 0) which would selection position 2, z 1, parameter 0.
        :param time_axis:
        :param figsize:
        :param kwargs: passed through to matplotlib.pyplot.figure
        :return: figh
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
        ph = np.unwrap(np.angle(self.phase_params[slices].squeeze(axis=squeeze_axes)), axis=0)
        ph_bg = np.unwrap(np.angle(self.phase_params_bg[slices].squeeze(axis=squeeze_axes)), axis=0)

        # plot
        figh2 = plt.figure(figsize=figsize, **kwargs)
        figh2.suptitle(f"index={index}\nphase variation versus time")

        ax = figh2.add_subplot(1, 2, 1)
        ax.set_title("Phases")
        ax.plot(ph, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("phase (rad)")

        ax = figh2.add_subplot(1, 2, 2)
        ax.set_title("Background phases")
        ax.plot(ph_bg, '.-')
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

        :param index:
        :param time_axis:
        :param figsize:
        :param kwargs:
        :return figh, epower, epower_bg:
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

        e_powers_rms = da.sqrt(da.mean(da.abs(self.holograms_ft) ** 2, axis=(-1, -2))) / np.prod(self.holograms_ft.shape[-2:])
        e_powers_rms_bg = da.sqrt(da.mean(da.abs(self.holograms_ft_bg) ** 2, axis=(-1, -2))) / np.prod(self.holograms_ft_bg.shape[-2:])

        # get slice of phases
        with ProgressBar():
            computed = dask.compute([e_powers_rms[slices].squeeze(axis=squeeze_axes),
                                     e_powers_rms_bg[slices].squeeze(axis=squeeze_axes)],
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
        ax.plot(epowers, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("|E|")

        ax.set_ylim([0, None])

        ax = figh2.add_subplot(2, 2, 2)
        ax.set_title("|Ebg| RMS average")
        ax.plot(epowers_bg, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("|E|")

        ax.set_ylim([0, None])

        ax = figh2.add_subplot(2, 2, 3)
        ax.set_title("E fit amplitude")
        ax.plot(amp, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("amp")

        ax.set_ylim([0, None])

        ax = figh2.add_subplot(2, 2, 4)
        ax.set_title("Ebg bit amplitude")
        ax.plot(amp_bg, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("amplitude")

        ax.set_ylim([0, None])

        return figh2, epowers, epowers_bg

    def show_image(self,
                   index: Optional[tuple[int]] = None,
                   figsize: tuple[float] = (35., 15.),
                   gamma: float = 0.1,
                   **kwargs) -> matplotlib.figure.Figure:
        """
        display raw image and holograms

        :param index: index of image to display. Should be of length self.nextra_dims + 1
        :param figsize:
        :param gamma: gamma to be used when display fourier transforms
        :return figh: figure handle
        """

        if index is None:
            index = (0,) * (self.nextra_dims + 1)

        extent = [self.x[0] - 0.5 * self.dxy, self.x[-1] + 0.5 * self.dxy,
                  self.y[-1] + 0.5 * self.dxy, self.y[0] - 0.5 * self.dxy]

        extent_f = [self.fxs[0] - 0.5 * self.dfx, self.fxs[-1] + 0.5 * self.dxy,
                    self.fys[-1] + 0.5 * self.dfy, self.fys[0] - 0.5 * self.dfy]

        # ######################
        # plot
        # ######################
        img_now = to_cpu(self.imgs_raw[index].compute())
        img_ft = ft2(img_now)

        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle(f"{index}, {self.axes_names}")
        grid = figh.add_gridspec(nrows=4, height_ratios=[1, 0.1, 1, 0.1],
                                 ncols=6, wspace=0.2)

        # ######################
        # raw image
        # ######################
        ax = figh.add_subplot(grid[0, 0])
        im = ax.imshow(img_now, extent=extent, cmap="bone")
        ax.set_title("$I(r)$")

        ax = figh.add_subplot(grid[1, 0])
        plt.colorbar(im, cax=ax, location="bottom")

        # ######################
        # raw image ft
        # ######################
        ax = figh.add_subplot(grid[2, 0])
        im = ax.imshow(np.abs(img_ft), extent=extent_f,
                       norm=PowerNorm(gamma=gamma), cmap="bone")
        ax.set_title("$|I(f)|$")
        # todo: optionally plot reference frequency and hologram frequencies (if available)

        ax = figh.add_subplot(grid[3, 0])
        plt.colorbar(im, cax=ax, location="bottom")

        # ######################
        # hologram
        # ######################
        try:
            holo_ft = to_cpu(self.holograms_ft[index].compute())
            holo = ift2(holo_ft)

            ax = figh.add_subplot(grid[0, 1])
            ax.set_title("$E(f)$")
            im = ax.imshow(np.abs(holo_ft), extent=extent_f,
                           norm=PowerNorm(gamma=gamma), cmap="bone")
            vmin_ef, vmax_ef = im.get_clim()
            ax.set_xlim([-self.fmax, self.fmax])
            ax.set_ylim([self.fmax, -self.fmax])

            ax = figh.add_subplot(grid[1, 1])
            plt.colorbar(im, cax=ax, location="bottom")

            # ######################
            # real-space hologram
            # ######################
            ax = figh.add_subplot(grid[0, 2])
            ax.set_title("$E(r)$")
            im = ax.imshow(np.abs(holo), extent=extent, cmap="bone", vmin=0)
            _, vmax_e = im.get_clim()

            ax = figh.add_subplot(grid[1, 2])
            plt.colorbar(im, cax=ax, location="bottom")

            # ######################
            # real-space hologram phase
            # ######################
            ax = figh.add_subplot(grid[0, 3])
            ax.set_title("angle $[E(r)]$")
            im = ax.imshow(np.angle(holo), extent=extent, cmap="RdBu", vmin=-np.pi, vmax=np.pi)

            ax = figh.add_subplot(grid[1, 3])
            plt.colorbar(im, cax=ax, location="bottom")

        except TypeError as e:
            print(e)

        # ######################
        # hologram background
        # ######################
        try:
            index_bg = tuple([v if self.holograms_ft.shape[ii] != 0 else 0 for ii, v in enumerate(index)])

            holo_ft_bg = to_cpu(self.holograms_ft_bg[index_bg].compute())
            holo_bg = ift2(holo_ft_bg)

            ax = figh.add_subplot(grid[2, 1])
            ax.set_title("$E_{bg}(f)$")
            im = ax.imshow(np.abs(holo_ft_bg), extent=extent_f,
                           norm=PowerNorm(gamma=gamma, vmin=vmin_ef, vmax=vmax_ef), cmap="bone")
            ax.set_xlim([-self.fmax, self.fmax])
            ax.set_ylim([self.fmax, -self.fmax])

            # ######################
            # real-space hologram
            # ######################
            ax = figh.add_subplot(grid[2, 2])
            ax.set_title("$E_{bg}(r)$")
            im = ax.imshow(np.abs(holo_bg), extent=extent, cmap="bone", vmin=0, vmax=vmax_e)

            # ######################
            # real-space hologram phase
            # ######################
            ax = figh.add_subplot(grid[2, 3])
            ax.set_title("angle $[E_{bg}(r)]$")
            im = ax.imshow(np.angle(holo_bg), extent=extent, cmap="RdBu", vmin=-np.pi, vmax=np.pi)

            # ######################
            # |E(r) - Ebg(r)|
            # ######################
            ax = figh.add_subplot(grid[0, 4])
            ax.set_title("$|E(r) - E_{bg}(r)|$")
            im = ax.imshow(np.abs(holo - holo_bg), extent=extent, cmap="bone", vmin=0)

            ax = figh.add_subplot(grid[1, 4])
            plt.colorbar(im, cax=ax, location="bottom")

            # ######################
            # |E(r)| - |Ebg(r)|
            # ######################
            ax = figh.add_subplot(grid[2, 4])
            ax.set_title("$|E(r)| - |E_{bg}(r)|$")

            im = ax.imshow(np.abs(holo) - np.abs(holo_bg), extent=extent, cmap="RdBu")
            # ensure symmetric
            vmin_now, vmax_now = im.get_clim()
            vmin = np.min([vmin_now, -vmax_now])
            vmax = np.max([-vmin_now, vmax_now])
            im.set_clim([vmin, vmax])

            ax = figh.add_subplot(grid[3, 4])
            plt.colorbar(im, cax=ax, location="bottom")

            # ######################
            # ang[E(r)] - ang[Ebg(r)]
            # ######################
            ax = figh.add_subplot(grid[0, 5])
            ax.set_title("$ang[E(r)] - ang[E_{bg}(r)]$")

            ang_diff = np.mod(np.angle(holo) - np.angle(holo_bg), 2*np.pi)
            ang_diff[ang_diff > np.pi] -= 2*np.pi

            im = ax.imshow(ang_diff, extent=extent, cmap="RdBu", vmin=-np.pi, vmax=np.pi)

            ax = figh.add_subplot(grid[1, 5])
            plt.colorbar(im, cax=ax, location="bottom")

        except Exception as e:
            print(e)

        return figh


# FFT idioms
def _ft_abs(m): return ft2(abs(ift2(m)))


def cut_mask(img: array,
             mask: array,
             mask_val: float = 0) -> array:
    """
    Mask points in image and set to a given value

    :param img: image
    :param mask: At any position where mask is True, replace the value of the image with mask_val
    :param mask_val:
    :return img_masked:
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
def get_global_phase_shifts(imgs: array,
                            ref_imgs: array,
                            thresh: Optional[float] = None) -> np.ndarray:
    """
    Given a stack of images and a reference, determine the phase shifts between images, such that
    imgs * A*np.exp(1j * phase_shift) ~ img_ref

    :param imgs: n0 x n1 x ... x n_{-2} x n_{-1} array
    :param ref_imgs: reference images. Should be broadcastable to same size as imgs.
    :param thresh: only consider points in images where both abs(imgs) and abs(ref_imgs) > thresh
    :return fit_params:
    """

    if isinstance(imgs, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    ref_imgs = xp.asarray(ref_imgs)

    # broadcast images and references images to same shapes
    imgs, ref_imgs = xp.broadcast_arrays(imgs, ref_imgs)

    # loop over all dimensions except the last two, which are the y and x dimensions respectively
    loop_shape = imgs.shape[:-2]

    fit_params = xp.zeros(loop_shape + (1, 1), dtype=complex)
    nfits = np.prod(loop_shape).astype(int)
    for ii in range(nfits):
        ind = np.unravel_index(ii, loop_shape)

        if thresh is None:
            A = xp.expand_dims(imgs[ind].ravel(), axis=1)
            B = ref_imgs[ind].ravel()
        else:
            mask = xp.logical_and(np.abs(imgs[ind]) > thresh, xp.abs(ref_imgs[ind]) > thresh)
            A = xp.expand_dims(imgs[ind][mask], axis=1)
            B = ref_imgs[ind][mask]

        fps, _, _, _ = xp.linalg.lstsq(A, B, rcond=None)
        fit_params[ind] = fps

    return fit_params


def fit_phase_ramp(imgs_ft: array,
                   ref_imgs_ft: array,
                   dxy: float,
                   thresh: float = 1/30):
    """
    Given a stack of images and reference images, determine the phase ramp parameters relating them

    :param imgs_ft: n0 x n1 x ... x n_{-2} x n_{-1} array
    :param ref_imgs_ft: reference images. Should be broadcastable to same size as imgs.
    :param dxy:
    :param thresh:
    :return fit_params:
    """

    if isinstance(imgs_ft, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    ref_imgs_ft = xp.asarray(ref_imgs_ft)

    # broadcast images and references images to same shapes
    imgs_ft, ref_imgs_ft = xp.broadcast_arrays(imgs_ft, ref_imgs_ft)

    #
    ny, nx = imgs_ft.shape[-2:]
    fx = xp.fft.fftshift(xp.fft.fftfreq(nx, dxy))
    fy = xp.fft.fftshift(xp.fft.fftfreq(ny, dxy))
    fxfx, fyfy = xp.meshgrid(fx, fy)

    # looper over all dimensions except the last two, which are the y and x dimensions respectively
    loop_shape = imgs_ft.shape[:-2]

    fit_params = xp.zeros(loop_shape + (1, 1, 2), dtype=float)
    nfits = np.prod(loop_shape).astype(int)
    for ii in range(nfits):
        ind = np.unravel_index(ii, loop_shape)

        ref_val = np.max(np.abs(ref_imgs_ft)) * thresh
        # mask = xp.logical_and(xp.abs(imgs_ft[ind]) >= ref_val)
        mask = xp.abs(imgs_ft[ind]) >= ref_val

        imgs_ft_mask = imgs_ft[ind][mask]
        fxfx_mask = fxfx[mask]
        fyfy_mask = fyfy[mask]
        ref_imgs_ft_mask = ref_imgs_ft[ind][mask]

        def fnh(p): return ref_imgs_ft_mask * np.exp(-2 * np.pi * 1j * (fxfx_mask * p[0] + fyfy_mask * p[1]))

        results = fit.fit_model(imgs_ft_mask,
                                fnh,
                                init_params=np.array([0., 0.]),
                                function_is_complex=True
                                )

        fit_params[ind] = results["fit_params"]

    return fit_params


def fit_ref_frq(img_ft: np.ndarray,
                dxy: float,
                fmax_int: float,
                search_rad_fraction: float = 1,
                npercentiles: int = 50,
                filter_size=0,
                dilate_erode_footprint_size: int = 10,
                show_figure: bool = False):
    """
    Determine the hologram reference frequency from a single image, based on the regions in the hologram beyond the
    maximum imaging frequency that have information. These are expected to be circles centered around the reference
    frequency.

    The fitting strategy is this
    (1) determine a threshold value for which points have signal in the image. To do this, first make a plot of
    thresholds versus percentiles. This should look like two piecewise lines
    (2) after thresholding the image, fit to circles.

    Note: when the beam angle is non-zero, the dominant tomography frequency component will not be centered
    on this circle, but will be at position f_ref - f_beam
    :param img_ft:
    :param dxy:
    :param fmax_int:
    :param search_rad_fraction:
    :param npercentiles:
    :param filter_size:
    :param dilate_erode_footprint_size:
    :param show_figure:
    :return results, circ_dbl_fn, figh: results["fit_params"] = [cx, cy, radius]
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


# convert between index of refraction and scattering potential
def get_n(v: array,
          no: float,
          wavelength: float) -> array:
    """
    convert from the scattering potential to the index of refraction

    :param v: scattering potential V(r) = - (2*np.pi / lambda)^2 * (n(r)^2 - no^2)
    :param no: background index of refraction
    :param wavelength: wavelength
    :return n: refractive index
    """
    if isinstance(v, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    k = 2 * np.pi / wavelength
    n = xp.sqrt(-v / k ** 2 + no ** 2)
    return n


def get_v(n: array,
          no: float,
          wavelength: float) -> array:
    """
    Convert from the index of refraction to the scattering potential

    :param n:
    :param no:
    :param wavelength:
    :return:
    """
    v = - (2 * np.pi / wavelength) ** 2 * (n**2 - no**2)
    return v


def get_rytov_phase(eimgs: array,
                    eimgs_bg: array,
                    regularization: float = 0.,
                    use_weighted_unwrap: bool = True) -> array:
    """
    Compute rytov phase from field and background field. The Rytov phase is \psi_s(r) where
    U_total(r) = exp[\psi_o(r) + \psi_s(r)]
    where U_o(r) = exp[\psi_o(r)] is the unscattered field

    We calculate \psi_s(r) = log | U_total(r) / U_o(r)| + 1j * unwrap[angle(U_total) - angle(U_o)]

    :param eimgs: n0 x n1 ... x nm x ny x nx
    :param eimgs_bg: broadcastable to same size as eimgs
    :param regularization: regularization value. Any pixels where the background
      exceeds this value will be set to zero
    :param use_weighted_unwrap:
    :return psi_rytov:
    """

    use_gpu = isinstance(eimgs, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
    else:
        xp = np

    eimgs = xp.asarray(eimgs)
    eimgs_bg = xp.asarray(eimgs_bg)

    # broadcast arrays
    eimgs, eimgs_bg = xp.broadcast_arrays(eimgs, eimgs_bg)

    # output values
    phase_diff = xp.mod(xp.angle(eimgs) - xp.angle(eimgs_bg), 2 * np.pi)
    # convert phase difference from interval [0, 2*np.pi) to [-np.pi, np.pi)
    phase_diff[phase_diff >= np.pi] -= 2 * np.pi

    # set real parts
    psi_rytov = xp.log(abs(eimgs) / (abs(eimgs_bg))) + 1j * 0

    # set imaginary parts
    # loop over all dimensions except the last two
    nextra_shape = eimgs.shape[:-2]
    nextra = np.prod(nextra_shape)
    for ii in range(nextra):
        ind = np.unravel_index(ii, nextra_shape)

        if use_weighted_unwrap:
            weights = xp.abs(eimgs_bg[ind])
        else:
            weights = None

        psi_rytov[ind] += 1j * weighted_phase_unwrap(phase_diff[ind],
                                                     weight=weights)

    # regularization
    psi_rytov[abs(eimgs_bg) < regularization] = 0

    return psi_rytov


def get_scattered_field(holograms: array,
                        holograms_bg: array,
                        regularization: float = 0.) -> array:
    """
    Compute estimate of scattered electric field with regularization. This function only operates on the
    last two dimensions of the array

    :param holograms: array of size n0 x ... x nm x ny x nx
    :param holograms_bg: broadcastable to same size as eimgs_ft
    :param regularization:
    :return efield_scattered_ft: scattered field of same size as eimgs_ft
    """
    # todo: could also define this sensibly for the rytov case. In that case, would want to take rytov phase shift
    #   and shift it back to the correct place in phase space ... since scattered field Es = E_o * psi_rytov is the approximation

    # compute scattered field in real space
    efield_scattered = (holograms - holograms_bg) / (abs(holograms_bg) + regularization)

    return efield_scattered


# holograms
def unmix_hologram(img: array,
                   dxy: float,
                   fmax_int: float,
                   fx_ref: np.ndarray,
                   fy_ref: np.ndarray,
                   apodization: array = 1,
                   mask: Optional[array] = None) -> array:
    """
    Given an off-axis hologram image, determine the electric field

    :param img: n1 x ... x n_{-3} x n_{-2} x n_{-1} array
    :param dxy: pixel size
    :param fmax_int: maximum frequency where intensity OTF has support
    :param fx_ref: x-component of hologram reference frequency
    :param fy_ref: y-component of hologram reference frequency
    :param apodization: apodization window applied to real-space image before Fourier transformation
    :param mask:
    :return efield_ft: hologram electric field
    """

    if isinstance(img, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    apodization = xp.asarray(apodization)

    # FT of image
    img_ft = ft2(img * apodization)

    # get frequency data
    ny, nx = img_ft.shape[-2:]
    fxs = xp.fft.fftshift(xp.fft.fftfreq(nx, dxy))
    fys = xp.fft.fftshift(xp.fft.fftfreq(ny, dxy))
    ff_perp = np.sqrt(fxs[None, :] ** 2 + fys[:, None] ** 2)

    # compute efield
    efield_ft = translate_ft(img_ft, fx_ref, fy_ref, drs=(dxy, dxy))
    efield_ft[..., ff_perp > fmax_int / 2] = 0.

    # optionally cut mask
    if mask is not None:
        efield_ft = cut_mask(efield_ft, mask, mask_val=0.)

    return efield_ft


# tomographic reconstruction
def get_fmax(no: float,
             na_detection: float,
             na_excitation: float,
             wavelength: float) -> np.ndarray:
    """
    Maximum frequencies measurable in ODT image

    :param no: index of refraction
    :param na_detection:
    :param na_excitation:
    :param wavelength:
    :return (fx_max, fy_max, fz_max):
    """
    alpha = np.arcsin(na_detection / no)
    beta = np.max(na_excitation / no)

    # maximum frequencies present in ODT
    fxy_max = (na_detection + no * np.sin(beta)) / wavelength
    fz_max = no / wavelength * np.max([1 - np.cos(alpha), 1 - np.cos(beta)])

    fmax = np.array([fxy_max, fxy_max, fz_max])

    return fmax


def get_reconstruction_nyquist_sampling(no: float,
                                        na_det: float,
                                        na_exc: float,
                                        wavelength: float,
                                        fov: Sequence[int],
                                        sampling_factors: Sequence[float] = (1., 1., 1.)
                                        ) -> (tuple[float], tuple[int]):
    """
    Helper function for computing Nyquist sampled grid size for scattering potential based on incoming beam angles.
    Note that this may not always be the ideal grid size, especially for multislice methods where the accuracy
    of the beam propagation depends on the choice of voxel size

    :param no: background index of refraction
    :param na_det: numerical aperture of detection objective
    :param na_exc: maximum excitation numerical aperture (i.e. corresponding to the steepest input beam and not nec.
    the objective).
    :param wavelength: wavelength
    :param fov: (nz, ny, nx) field of view voxels
    :param sampling_factors: (sz, sy, sx) spacing as a fraction of the nyquist limit
    :return (dz_sp, dxy_sp, dxy_sp), (nz_sp, ny_sp, nx_sp):
    """
    alpha = np.arcsin(na_det / no)
    beta = np.arcsin(na_exc / no)

    # maximum frequencies present in ODT
    fn_max = np.array([no / wavelength * np.max([1 - np.cos(alpha), 1 - np.cos(beta)]),
                       (na_det + na_exc) / wavelength,
                       (na_det + na_exc) / wavelength])

    # generate real-space sampling from Nyquist sampling
    drs = 0.5 * np.asarray(sampling_factors) / fn_max

    # get size from FOV
    n_size = np.ceil(fov / drs).astype(int)
    # ensure odd
    n_size += 1 - np.mod(n_size, 2)

    return tuple(drs), tuple(n_size)


def fwd_model_linear(beam_fx: array,
                     beam_fy: array,
                     beam_fz: array,
                     no: float,
                     na_det: float,
                     wavelength: float,
                     e_shape: tuple[int],
                     drs_e: tuple[float],
                     v_shape: tuple[int],
                     drs_v: tuple[float],
                     mode: str = "born",
                     interpolate: bool = False,
                     use_gpu: bool = False) -> csr_matrix:
    """
    Forward model from scattering potential v(k) to imaged electric field E(k) after interacting with object.
    Assumes plane wave illumination and linear scattering model (Born or Rytov)

    # todo: replace na_det with coherent transfer function

    :param beam_fx: beam frequencies. Either of size npatterns or nmultiplex x npatterns. If not all patterns
      use the same degree of multiplexing, then nmultiplex should be the maximum degree of multiplexing for all patterns
      and the extra frequences associated with patterns with a lower degree of multiplexing can be set to np.inf
    :param beam_fy:
    :param beam_fz:
    :param no: background index of refraction
    :param na_det: detection numerical aperture
    :param wavelength:
    :param e_shape: (ny, nx), shape of scattered fields
    :param drs_e: (dy, dx) pixel size of scattered field
    :param v_shape: (nz, ny, nx) shape of scattering potential
    :param drs_v: (dz, dy, dx) pixel size of scattering potential
    :param mode: "born" or "rytov"
    :param interpolate: use trilinear interpolation or nearest-neighbor
    :param use_gpu: usually doesn't make sense for one-off construction of matrix
    :return model: sparse csr matrix describing mapping from scattering potential to electric field
    """

    if beam_fx.ndim == 2:
        # support multiplex beam angles
        nmultiplex = beam_fx.shape[0]

        # for each degree of multiplexing, construct one matrix
        models = []
        for ii in range(nmultiplex):
            m = fwd_model_linear(beam_fx[ii],
                                 beam_fy[ii],
                                 beam_fz[ii],
                                 no,
                                 na_det,
                                 wavelength,
                                 e_shape,
                                 drs_e,
                                 v_shape,
                                 drs_v,
                                 mode,
                                 interpolate,
                                 use_gpu)
            models.append(m)

        # add models to get multiplexed model
        model = sum(models)
    elif 1 == 2:
        # todo: want to support looping over frequencies instead of running them in parallel
        # todo: but may have problems later anyways with holding all the fields I want in memory in that case
        nfrqs = beam_fx.shape[0]
        models = []
        for ii in range(nfrqs):
            m = fwd_model_linear(beam_fx[ii][None, :],
                                 beam_fy[ii][None, :],
                                 beam_fz[ii][None, :],
                                 no,
                                 na_det,
                                 wavelength,
                                 e_shape,
                                 drs_e,
                                 v_shape,
                                 drs_v,
                                 mode,
                                 interpolate,
                                 use_gpu)
            models.append(m)

        # here need to correct indices and combine to one matrix
        raise NotImplementedError()

    else:
        if use_gpu and _gpu_available:
            xp = cp
            spm = sp_gpu
        else:
            xp = np
            spm = sp

        beam_fx = xp.asarray(beam_fx)
        beam_fy = xp.asarray(beam_fy)
        beam_fz = xp.asarray(beam_fz)

        ny, nx = e_shape
        dy, dx = drs_e
        nimgs = len(beam_fx)

        # ##################################
        # get frequencies of electric field images and make broadcastable to shape (nimgs, ny, nx)
        # ##################################
        fx = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(nx, dx)), axis=(0, 1))
        fy = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(ny, dy)), axis=(0, 2))

        # ##################################
        # set sampling of 3D scattering potential
        # ##################################
        nz_v, ny_v, nx_v = v_shape
        v_size = np.prod(v_shape)
        dz_v, dy_v, dx_v = drs_v

        fx_v = xp.fft.fftshift(xp.fft.fftfreq(nx_v, dx_v))
        fy_v = xp.fft.fftshift(xp.fft.fftfreq(ny_v, dy_v))
        fz_v = xp.fft.fftshift(xp.fft.fftfreq(nz_v, dz_v))
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
            # V(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * Es(fx, fy)
            # ##################################

            # atf = xp.tile(atf, [nimgs, 1, 1])

            # logical array, which frqs in detection NA
            detectable = xp.sqrt(fx ** 2 + fy ** 2)[0] <= (na_det / wavelength)
            detectable = xp.tile(detectable, [nimgs, 1, 1])

            fz = xp.tile(get_fzs(fx, fy, no, wavelength), [nimgs, 1, 1])

            # construct frequencies where we have data about the 3D scattering potentials
            # frequencies of the sample F = f - no/lambda * beam_vec
            Fx, Fy, Fz = xp.broadcast_arrays(fx - xp.expand_dims(beam_fx, axis=(1, 2)),
                                             fy - xp.expand_dims(beam_fy, axis=(1, 2)),
                                             fz - xp.expand_dims(beam_fz, axis=(1, 2))
                                             )
            # if don't copy, then elements of F's are reference to other elements.
            Fx = xp.array(Fx, copy=True)
            Fy = xp.array(Fy, copy=True)
            Fz = xp.array(Fz, copy=True)

            # indices into the final scattering potential
            # taking advantage of the fact that the final scattering potential indices have FFT structure
            zind = Fz / dfz_v + nz_v // 2
            yind = Fy / dfy_v + ny_v // 2
            xind = Fx / dfx_v + nx_v // 2
        elif mode == "rytov":
            # V(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * psi_s(fx - n/lambda * nx, fy - n/lambda * ny)
            # V(Fx, Fy, Fz) = 2*i * (2*pi*fz) * psi_s(Fx, Fy)
            # so want to change variables and take (Fx, Fy) -> (fx, fy)
            # But have one problem: (Fx, Fy, Fz) do not form a normalized vector like (fx, fy, fz)
            # so although we can use fx, fy to stand in, we need to calculate the new z-component
            # Fz_rytov = np.sqrt( (n/lambda)**2 - (Fx + n/lambda * nx)**2 - (Fy + n/lambda * ny)**2) - n/lambda * nz
            # fz = Fz + n/lambda * nz
            Fx = fx
            Fy = fy

            # helper frequencies for calculating fz
            fx_rytov = Fx + xp.expand_dims(beam_fx, axis=(1, 2))
            fy_rytov = Fy + xp.expand_dims(beam_fy, axis=(1, 2))

            fz = get_fzs(fx_rytov,
                         fy_rytov,
                         no,
                         wavelength)

            Fz = fz - xp.expand_dims(beam_fz, axis=(1, 2))

            # atf at f_rytov
            # basis for applying the ATF in rytov case ... psi is a frequency shifted
            # version of scattered field, hence should nominally see effect of atf
            # atf = xp.tile(atf, [nimgs, 1, 1])
            # atf = translate_ft(atf, beam_fx, beam_fy, drs=(dy, dx))

            # take care of frequencies which do not contain signal
            detectable = (fx_rytov ** 2 + fy_rytov ** 2) <= (na_det / wavelength) ** 2

            # indices into the final scattering potential
            zind = Fz / dfz_v + nz_v // 2
            yind = Fy / dfy_v + ny_v // 2
            xind = Fx / dfx_v + nx_v // 2

            zind, yind, xind = [xp.array(a, copy=True) for a in xp.broadcast_arrays(zind, yind, xind)]
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
            z0 = xp.floor(zind).astype(int)
            z1 = z0 + 1
            y0 = xp.floor(yind).astype(int)
            y1 = y0 + 1
            x0 = xp.floor(xind).astype(int)
            x1 = x0 + 1

            # find indices in bounds
            # note: reduce not supported by cupy
            tzd = xp.logical_and(xp.logical_and(z0 >= 0, z1 < nz_v), detectable)
            txy = xp.logical_and(xp.logical_and(y0 >= 0, y1 < ny_v), xp.logical_and(x0 >= 0, x1 < nx_v))
            to_use = xp.logical_and(tzd, txy)

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
            row_index = xp.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]
            row_index = xp.tile(row_index, 8)

            # column_index -> indices into V vector
            inds_to_use = [[i[to_use] for i in inow] for inow in inds]
            zinds_to_use, yinds_to_use, xinds_to_use = [xp.concatenate(i) for i in list(zip(*inds_to_use))]

            column_index = xp.ravel_multi_index(tuple((zinds_to_use, yinds_to_use, xinds_to_use)), v_shape)

            # construct sparse matrix values
            interp_weights_to_use = xp.concatenate([w[to_use] for w in interp_weights])

            # since using DFT's instead of FT's have to adjust the normalization
            # FT ~ DFT * dr1 * ... * drn
            data = interp_weights_to_use / (2 * 1j * (2 * np.pi * xp.tile(fz[to_use], 8))) * dx_v * dy_v * dz_v / (dx * dy)

            # if atf is not None:
            #     data *= xp.tile(atf[to_use], 8)

        else:
            # find indices in bounds. CuPy does not support reduce()
            tzd = xp.logical_and(xp.logical_and(zind >= 0, zind < nz_v), detectable)
            txy = xp.logical_and(xp.logical_and(yind >= 0, yind < ny_v), xp.logical_and(xind >= 0, xind < nx_v))
            to_use = xp.logical_and(tzd, txy)

            inds_round = (xp.round(zind[to_use]).astype(int),
                          xp.round(yind[to_use]).astype(int),
                          xp.round(xind[to_use]).astype(int))

            # row index = position in E
            row_index = xp.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]

            # column index = position in V
            column_index = xp.ravel_multi_index(inds_round, v_shape)

            # matrix values
            # since using DFT's instead of FT's have to adjust the normalization
            # FT ~ DFT * dr1 * ... * drn
            data = xp.ones(len(row_index)) / (2 * 1j * (2 * np.pi * fz[to_use])) * dx_v * dy_v * dz_v / (dx * dy)

            # if atf is not None:
            #     data *= atf[to_use]

        # construct sparse matrix
        # E(k) = model * V(k)
        # column index = position in V
        # row index = position in E
        model = spm.csr_matrix((data, (row_index, column_index)), shape=(nimgs * ny * nx, v_size))

    return model


def inverse_model_linear(efield_fts: array,
                         model: csr_matrix,
                         v_shape: tuple[int],
                         regularization: float = 0.,
                         no_data_value: float = np.nan) -> array:
    """
    Given a set of holograms obtained using ODT, put the hologram information back in the correct locations in
    Fourier space

    :param efield_fts: The exact definition of efield_fts depends on whether "born" or "rytov" mode is used.
    Any points in efield_fts which are NaN will be ignored. efield_fts can have an arbitrary number of leading
    singleton dimensions, but must have at least three dimensions.
    i.e. it should have shape 1 x ... x 1 x nimgs x ny x nx
    :param model: forward model matrix. Generated from fwd_model_linear(). Should have interpolate=False
    :param v_shape:
    :param regularization: regularization factor
    :param no_data_value: value of any points in v_ft where no data is available
    :return v_ft:
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
    model = model.tocoo()
    data = xp.asarray(model.data)
    col_index = xp.asarray(model.col)
    row_index = xp.asarray(model.row)

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

        # ignore any nans in electric field
        # todo: can I do this without first constructing einds_angle?
        einds_angle = (e_ind[0][this_angle], e_ind[1][this_angle], e_ind[2][this_angle])
        is_not_nan_angle = xp.logical_not(xp.isnan(efield_fts[einds_angle]))

        # final indices
        einds_angle = (e_ind[0][this_angle][is_not_nan_angle],
                       e_ind[1][this_angle][is_not_nan_angle],
                       e_ind[2][this_angle][is_not_nan_angle]
                       )
        vinds_angle = (v_ind[0][this_angle][is_not_nan_angle],
                       v_ind[1][this_angle][is_not_nan_angle],
                       v_ind[2][this_angle][is_not_nan_angle]
                       )
        data_angle = data[this_angle][is_not_nan_angle]

        # assuming at most one point for each ... otherwise have problems
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


# plotting functions
def plot_odt_sampling(frqs: np.ndarray,
                      na_detect: float,
                      na_excite: float,
                      no: float,
                      wavelength: float,
                      **kwargs) -> matplotlib.figure.Figure:
    """
    Illustrate the region of frequency space which is obtained using the plane waves described by frqs

    :param frqs: nfrqs x 2 array of [[fx0, fy0], [fx1, fy1], ...]
    :param na_detect: detection NA
    :param na_excite: excitation NA
    :param ni: index of refraction of medium that sample is immersed in.
    :param wavelength:
    :param kwargs: passed through to figure
    :return figh:
    """
    frq_norm = no / wavelength
    alpha_det = np.arcsin(na_detect / no)

    if na_excite / no < 1:
        alpha_exc = np.arcsin(na_excite / no)
    else:
        # if na_excite is immersion objective and beam undergoes TIR at interface for full NA
        alpha_exc = np.pi/2

    fzs = get_fzs(frqs[:, 0], frqs[:, 1], no, wavelength)
    frqs_3d = np.concatenate((frqs, np.expand_dims(fzs, axis=1)), axis=1)

    figh = plt.figure(**kwargs)
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
    fz_edge = np.sqrt((no / wavelength)**2 - fx_edge**2)

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
    fz_edge = np.sqrt((no / wavelength)**2 - fy_edge**2)

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
    fzfz = np.sqrt((no / wavelength)**2 - fxfx**2 - fyfy**2)

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
                             raw_data_component: str = "cam2/odt",
                             show_raw: bool = True,
                             show_efields: bool = False,
                             compute: bool = True,
                             time_axis: int = 1,
                             time_range: Optional[list[int]] = None,
                             phase_lim: float = np.pi,
                             n_lim: tuple[float] = (0., 0.05),
                             e_lim: tuple[float] = (0., 500.),
                             escatt_lim: tuple[float] = (-5., 5.),
                             block_while_display: bool = True,
                             real_cmap="bone",
                             phase_cmap="RdBu",
                             scale_z: bool = True):
    """
    Display reconstruction results and (optionally) raw data in Napari

    :param recon_fname: refractive index reconstruction stored in zarr file
    :param raw_data_fname: raw data stored in zar file
    :param raw_data_component:
    :param show_raw:
    :param show_efields:
    :param compute:
    :param time_axis:
    :param time_range:
    :param phase_lim:
    :param block_while_display:
    :param real_cmap:
    :param phase_cmap:
    :return: viewer
    """

    import napari

    if raw_data_fname is not None:
        raw_data = zarr.open(raw_data_fname, "r")
    else:
        show_raw = False

    # load data
    img_z = zarr.open(recon_fname, "r")
    if not hasattr(img_z, "efield_bg_ft") or not hasattr(img_z, "efields_ft"):
        show_efields = False

    # raw data sizes
    # dxy_cam = img_z.attrs["camera_path_attributes"]["dx_um"]
    proc_roi = img_z.attrs["processing roi"]
    ny = proc_roi[1] - proc_roi[0]
    nx = proc_roi[3] - proc_roi[2]

    # try:
    #     cam_roi = img_z.attrs["camera_path_attributes"]["camera_roi"]
    #     ny_raw = cam_roi[1] - cam_roi[0]
    #     nx_raw = cam_roi[3] - cam_roi[2]
    # except KeyError:
    #     ny_raw = ny
    #     nx_raw = nx

    drs_n = img_z.attrs["dr"]
    n_axis_names = img_z.attrs["dimensions"]
    # wavelength = img_z.attrs["wavelength"]
    no = img_z.attrs["no"]

    # load affine xforms
    # Napari is using convention (y, x) whereas I'm using (x, y), so need to swap these dimensions in affine xforms
    swap_xy = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    try:
        affine_recon2cam_xy = np.array(img_z.attrs["affine_xform_recon_2_raw_camera_roi"])
    except KeyError:
        affine_recon2cam_xy = affine.params2xform([1, 0, 0, 1, 0, 0])
    affine_recon2cam = swap_xy.dot(affine_recon2cam_xy.dot(swap_xy))
    affine_cam2recon = np.linalg.inv(affine_recon2cam)

    # ######################
    # prepare n
    # ######################
    show_n = hasattr(img_z, "n")
    if show_n:
        n_extra_dims = img_z.n.ndim - 3
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

        slices_n = slices + (slice(None), slice(None), slice(None))

        n = da.expand_dims(da.from_zarr(img_z.n)[slices_n], axis=-4)
        if compute:
            print("loading n")
            with ProgressBar():
                n = n.compute()

        if hasattr(img_z, "n_start"):
            n_start = da.expand_dims(da.from_zarr(img_z.n_start)[slices_n], axis=-4)
            if compute:
                with ProgressBar():
                    n_start = n_start.compute()

    else:
        n = no * np.ones(1)
        n_start = no * np.ones(1)

    n_real = n.real - no
    n_imag = n.imag
    n_start_real = n_start.real - no
    n_start_imag = n_start.imag

    # ######################
    # prepare raw images
    # ######################
    slices_raw = slices + (slice(None), slice(None), slice(None))
    if show_raw:
        imgs = da.expand_dims(da.from_zarr(raw_data[raw_data_component])[slices_raw], axis=-3)

        if compute:
            print("loading raw images")
            with ProgressBar():
                imgs = imgs.compute()
    else:
        imgs = np.ones(1)
        imgs_raw_ft = np.ones(1)

    # ######################
    # prepare electric fields
    # ######################
    if show_efields:
        # measured field
        e_load_ft = da.expand_dims(da.from_zarr(img_z.efields_ft), axis=-3)
        e = da.map_blocks(ift2, e_load_ft, dtype=complex)

        # background field
        ebg_load_ft = da.expand_dims(da.from_zarr(img_z.efield_bg_ft), axis=-3)
        ebg = da.map_blocks(ift2, ebg_load_ft, dtype=complex)

        # compute electric field power
        efield_power = da.mean(da.abs(e), axis=(-1, -2), keepdims=True)

        if compute:
            print("loading electric fields")
            with ProgressBar():
                c = dask.compute([e, ebg, efield_power])
                e, ebg, efield_power = c[0]
    else:
        e = np.ones(1)
        ebg = np.ones(1)

    e_abs = da.abs(e)
    e_angle = da.angle(e)
    ebg_abs = da.abs(ebg)
    ebg_angle = da.angle(ebg)
    e_ebg_abs_diff = e_abs - ebg_abs

    pd = da.mod(da.angle(e) - da.angle(ebg), 2 * np.pi)
    e_ebg_phase_diff = pd - 2 * np.pi * (pd > np.pi)

    # ######################
    # scattered fields
    # ######################
    if hasattr(img_z, "escatt"):
        escatt = da.expand_dims(da.from_zarr(img_z.escatt), axis=-3)
        if compute:
            print('loading escatt')
            with ProgressBar():
                escatt = escatt.compute()

        escatt_real = da.real(escatt)
        escatt_imag = da.imag(escatt)
    else:
        escatt_real = np.ones(1)
        escatt_imag = np.ones(1)

    # ######################
    # simulated forward fields
    # ######################
    show_efields_fwd = show_efields and hasattr(img_z, "efwd")

    if show_efields_fwd:
        e_fwd = da.expand_dims(da.from_zarr(img_z.efwd), axis=-3)

        if compute:
            print("loading fwd electric fields")
            with ProgressBar():
                e_fwd = e_fwd.compute()

        e_fwd_abs = da.abs(e_fwd)
        e_fwd_angle = da.angle(e_fwd)

        efwd_ebg_abs_diff = e_fwd_abs - ebg_abs

        pd = da.mod(da.angle(e_fwd) - da.angle(ebg), 2 * np.pi)
        efwd_ebg_phase_diff = pd - 2 * np.pi * (pd > np.pi)

    else:
        e_fwd_abs = np.ones(1)
        e_fwd_angle = np.ones(1)
        efwd_ebg_abs_diff = np.ones(1)
        efwd_ebg_phase_diff = np.ones(1)

    # ######################
    # broadcasting
    # NOTE: cannot operate on these arrays after broadcasting otherwise memory use will explode
    # broadcasting does not cause memory size expansion, but in-place operations later will
    # ######################
    if compute:
        n_extra_dims = n_real.ndim - 4
        nz = n_real.shape[-3]

        if show_efields:
            npatt = e_abs.shape[-4]
        else:
            npatt = 1

        bcast_shape = (1,) * n_extra_dims + (npatt, nz) + (1, 1)

        # broadcast refractive index arrays
        bcast_shape_n = np.broadcast_shapes(n_real.shape, bcast_shape)
        n_real = np.broadcast_to(n_real, bcast_shape_n)
        n_imag = np.broadcast_to(n_imag, bcast_shape_n)
        n_start_real = np.broadcast_to(n_start_real, bcast_shape_n)
        n_start_imag = np.broadcast_to(n_start_imag, bcast_shape_n)

        # broadcast raw images
        bcast_shape_raw = np.broadcast_shapes(imgs.shape, bcast_shape)
        imgs = np.broadcast_to(imgs, bcast_shape_raw)

        # broadcast electric fields
        bcast_shape_e = np.broadcast_shapes(e_abs.shape, bcast_shape)
        e_abs = np.broadcast_to(e_abs, bcast_shape_e)
        e_angle = np.broadcast_to(e_angle, bcast_shape_e)
        ebg_abs = np.broadcast_to(ebg_abs, bcast_shape_e)
        ebg_angle = np.broadcast_to(ebg_angle, bcast_shape_e)
        e_ebg_abs_diff = np.broadcast_to(e_ebg_abs_diff, bcast_shape_e)
        e_ebg_phase_diff = np.broadcast_to(e_ebg_phase_diff, bcast_shape_e)
        e_fwd_bas = np.broadcast_to(e_fwd_abs, bcast_shape_e)
        e_fwd_angle = np.broadcast_to(e_fwd_angle, bcast_shape_e)
        efwd_ebg_abs_diff = np.broadcast_to(efwd_ebg_abs_diff, bcast_shape_e)
        efwd_ebg_phase_diff = np.broadcast_to(efwd_ebg_phase_diff, bcast_shape_e)

        # this can be a differetn size due to multiplexing
        bcast_root_scatt = (1,) * n_extra_dims + (1, nz, 1, 1)
        bcast_shape_scatt = np.broadcast_shapes(escatt_real.shape, bcast_root_scatt)
        escatt_real = np.broadcast_to(escatt_real, bcast_shape_scatt)
        escatt_imag = np.broadcast_to(escatt_imag, bcast_shape_scatt)

    # ######################
    # create viewer
    # ######################
    viewer = napari.Viewer(title=str(recon_fname))

    if scale_z:
        scale = (drs_n[0] / drs_n[1], 1, 1)
    else:
        scale = (1, 1, 1)

    # ######################
    # raw data
    # ######################
    if show_raw:
        viewer.add_image(imgs,
                         scale=scale,
                         name="raw images",
                         colormap=real_cmap,
                         affine=affine_cam2recon,
                         contrast_limits=[0, 4096])

        # processed ROI
        proc_roi_rect = np.array([[[0 - 1, 0 - 1],
                                   [0 - 1, nx],
                                   [ny, nx],
                                   [ny, 0 - 1]
                                   ]])

        viewer.add_shapes(proc_roi_rect,
                          shape_type="polygon",
                          name="processing ROI",
                          edge_width=1,
                          edge_color=[1, 0, 0, 1],
                          face_color=[0, 0, 0, 0])

    # ######################
    # reconstructed index of refraction
    # ######################

    if show_n:
        # for convenience of affine xforms, keep xy in pixels
        viewer.add_image(n_start_imag,
                         scale=scale,
                         name=f"n start.imaginary",
                         contrast_limits=n_lim,
                         colormap=real_cmap,
                         visible=False)

        viewer.add_image(n_start_real,
                         scale=scale,
                         name=f"n start - no",
                         colormap=real_cmap,
                         contrast_limits=n_lim)

        viewer.add_image(n_imag,
                         scale=scale,
                         name=f"n.imaginary",
                         contrast_limits=n_lim,
                         colormap=real_cmap,
                         visible=False)

        viewer.add_image(n_real,
                         scale=scale,
                         name=f"n-no",
                         colormap=real_cmap,
                         contrast_limits=n_lim)

    # ######################
    # electric fields
    # ######################
    if show_efields:
        # field amplitudes
        viewer.add_image(ebg_abs,
                         scale=scale,
                         name="|e bg|",
                         contrast_limits=e_lim,
                         colormap=real_cmap,
                         translate=[0, nx])

        if show_efields_fwd:
            viewer.add_image(e_fwd_bas,
                            scale=scale,
                            name="|E fwd|",
                            contrast_limits=e_lim,
                            colormap=real_cmap,
                            translate=[0, nx])

        viewer.add_image(e_abs,
                         scale=scale,
                         name="|e|",
                         contrast_limits=e_lim,
                         colormap=real_cmap,
                         translate=[0, nx])

        # field phases
        viewer.add_image(ebg_angle,
                         scale=scale,
                         name="angle(e bg)",
                         contrast_limits=[-np.pi, np.pi],
                         colormap=phase_cmap,
                         translate=[ny, nx])

        if show_efields_fwd:
            viewer.add_image(e_fwd_angle,
                             scale=scale,
                             name="ange(E fwd)",
                             contrast_limits=[-np.pi, np.pi],
                             colormap=phase_cmap,
                             translate=[ny, nx])

        viewer.add_image(e_angle,
                         scale=scale,
                         name="angle(e)",
                         contrast_limits=[-np.pi, np.pi],
                         colormap=phase_cmap,
                         translate=[ny, nx])

        # difference of absolute values
        if show_efields_fwd:
            viewer.add_image(efwd_ebg_abs_diff,
                             scale=scale,
                             name="|e fwd| - |e bg|",
                             contrast_limits=[-e_lim[1], e_lim[1]],
                             colormap=phase_cmap,
                             translate=[0, 2*nx])

        viewer.add_image(e_ebg_abs_diff,
                         scale=scale,
                         name="|e| - |e bg|",
                         contrast_limits=[-e_lim[1], e_lim[1]],
                         colormap=phase_cmap,
                         translate=[0, 2*nx])

        # difference of phases
        if show_efields_fwd:
            viewer.add_image(efwd_ebg_phase_diff,
                             scale=scale,
                             name="angle(e fwd) - angle(e bg)",
                             contrast_limits=[-phase_lim, phase_lim],
                             colormap=phase_cmap,
                             translate=[ny, 2*nx]
                             )

        viewer.add_image(e_ebg_phase_diff,
                         scale=scale,
                         name="angle(e) - angle(e bg)",
                         contrast_limits=[-phase_lim, phase_lim],
                         colormap=phase_cmap,
                         translate=[ny, 2*nx]
                         )

        viewer.add_image(escatt_real,
                         scale=scale,
                         name="Re(e scatt)",
                         contrast_limits=escatt_lim,
                         colormap=phase_cmap,
                         translate=[ny, 0]
                         )

        viewer.add_image(escatt_imag,
                         scale=scale,
                         name="Im(e scatt)",
                         contrast_limits=escatt_lim,
                         colormap=phase_cmap,
                         translate=[ny, 0]
                         )

        # label
        translations = np.array([[0, nx],
                                 [ny, nx],
                                 [0, 2*nx],
                                 [ny, 2*nx],
                                 [ny, 0]
                                 ])
        ttls = ["|E|",
                "ang(E)",
                "|E| - |Ebg|",
                "angle(e) - angle(e bg)",
                "E scatt"]

        viewer.add_points(translations + np.array([ny / 10, nx / 2]),
                          features={"names": ttls},
                          text={"string": "{names:s}",
                                "size": 30,
                                "color": "red"},
                          )

    viewer.dims.axis_labels = n_axis_names[:n_extra_dims + 1] + ["pattern", "z", "y", "x"]

    # set to first position
    viewer.dims.set_current_step(axis=0, value=0)
    # set to first time
    viewer.dims.set_current_step(axis=1, value=0)

    # block until closed by user
    viewer.show(block=block_while_display)

    return viewer

def get_2d_projections(n: np.ndarray,
                       z_to_xy_ratio: float = 1,
                       use_slice: bool = False,
                       n_pix_sep: int = 5
                       ) -> np.ndarray:
    """
    Generate an image showing 3 orthogonal projections from a 3D array.

    :param n: 3D array
    :param z_to_xy_ratio: pixel size ratio dz/dxy
    :param use_slice: use the central slice. If False, max project
    :param n_pix_sep: number of blank pixels between projections
    :return img: 2D image showing projections
    """

    nz, ny, nx = n.shape

    if use_slice:
        iz = nz // 2
        iy = ny // 2
        ix = nx // 2
        n_xy = n[iz]
        n_yz_before_xform = n[:, :, ix].transpose()
        n_xz_before_xform = n[:, iy, :]
    else:
        # max projection
        n_xy = np.max(n, axis=0)
        n_yz_before_xform = np.max(n, axis=2).transpose()
        n_xz_before_xform = np.max(n, axis=1)

    ny_img = ny + n_pix_sep + int(np.ceil(nz * z_to_xy_ratio))
    nx_img = nx + n_pix_sep + int(np.ceil(nz * z_to_xy_ratio))

    img = np.zeros((ny_img, nx_img))

    # xy slice
    img[:ny, :nx] = n_xy

    xx, yy = np.meshgrid(range(nx_img), range(ny_img))

    # yz slice
    xform_yz = affine.params2xform([z_to_xy_ratio, 0, nx + n_pix_sep, 1, 0, 0])
    n_yz = affine.xform_mat(n_yz_before_xform,
                            xform_yz,
                            (xx[:ny, nx + n_pix_sep:], yy[:ny, nx + n_pix_sep:]))
    img[:ny, nx + n_pix_sep:] = n_yz

    xform_xz = affine.params2xform([1, 0, 0, z_to_xy_ratio, 0, ny + n_pix_sep])
    n_xz = affine.xform_mat(n_xz_before_xform,
                            xform_xz,
                            (xx[ny + n_pix_sep:, :nx],
                             yy[ny + n_pix_sep:, :nx]))
    img[ny + n_pix_sep:, :nx] = n_xz

    img[np.isnan(img)] = 0

    return img

def get_color_projection(n: np.ndarray,
                         contrast_limits=(0, 1),
                         mask: Optional[np.ndarray] = None,
                         cmap="turbo") -> (np.ndarray, np.ndarray):
    """
    Given a 3D refractive index distribution, take the max-z projection and color code the results
    by height. For each xy position, only consider the voxel along z with the maximum value.
    Display this in the final array in a color based on the height where that voxel was.

    :param n: refractive index array of size n0 x ... x nm x nz x ny x nx
    :param contrast_limits: (nmin, nmax)
    :param mask: only consider points where mask value is True
    :param cmap: matplotlib colormap
    :return: n_proj, colors
    """

    if mask is None:
        maxz_args = np.argmax(n, axis=-3)
    else:
        maxz_args = np.argmax(n * mask, axis=-3)

    nz, _, _ = n.shape[-3:]
    shape = list(n.shape + (3,))
    shape[-4] = 1

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, nz))

    n_proj = np.zeros(shape, dtype=float)
    for ii in range(nz):
        to_use = maxz_args == ii

        intensity = (n[..., ii, :, :][to_use] - contrast_limits[0]) / ((contrast_limits[1] - contrast_limits[0]))
        intensity[intensity < 0] = 0
        intensity[intensity > 1] = 1

        n_proj[np.expand_dims(to_use, axis=-3), :] = np.expand_dims(intensity, axis=-1) * colors[ii, :3][None, :]

    return n_proj, colors

class RIOptimizer(Optimizer):

    def __init__(self,
                 e_measured: array,
                 e_measured_bg: array,
                 beam_frqs: array,
                 no: float,
                 wavelength: float,
                 drs_e: tuple[float],
                 drs_n: tuple[float],
                 shape_n: tuple[int],
                 dz_final: float = 0.,
                 atf: Optional[array] = None,
                 apodization: Optional[array] = None,
                 mask: Optional[array] = None,
                 tau_tv_real: float = 0,
                 tau_tv_imag: float = 0,
                 tau_l1_real: float = 0,
                 tau_l1_imag: float = 0,
                 use_imaginary_constraint: bool = False,
                 use_real_constraint: bool = False,
                 max_imaginary_part: float = np.inf,
                 efield_cost_factor: float = 1.,
                 **kwargs
                 ):
        """

        :param e_measured:
        :param e_measured_bg:
        :param beam_frqs:
        :param no:
        :param wavelength:
        :param drs_e:
        :param drs_n:
        :param shape_n:
        :param dz_final:
        :param atf: amplitude (coherent) transfer function
        :param apodization:
        :param tau_tv_real: weight for TV proximal operator applied to real-part
        :param tau_tv_imag: weight for TV proximal operator applied to imaginary-part
        :param tau_l1_real:
        :param tau_l1_imag:
        :param use_imaginary_constraint: enforce im(n) > 0
        :param use_real_constraint: enforce re(n) > no
        """

        super(RIOptimizer, self).__init__()

        self.e_measured = e_measured
        self.e_measured_bg = e_measured_bg
        self.beam_frqs = beam_frqs
        self.n_samples = self.e_measured.shape[-3]
        self.no = no
        self.wavelength = wavelength
        self.drs_e = drs_e
        self.drs_n = drs_n
        self.shape_n = shape_n
        self.dz_final = dz_final
        self.atf = atf
        self.apodization = apodization

        self.efield_cost_factor = float(efield_cost_factor)
        if self.efield_cost_factor > 1 or self.efield_cost_factor < 0:
            raise ValueError(f"efield_cost_factor must be between 0 and 1, but was {self.efield_cost_factor}")

        if mask is not None:
            if mask.dtype.kind != "b":
                raise ValueError("mask must be `None` or a boolean array")
        self.mask = mask

        self.set_prox(tau_tv_real,
                      tau_tv_imag,
                      tau_l1_real,
                      tau_l1_imag,
                      use_imaginary_constraint,
                      use_real_constraint,
                      max_imaginary_part)

    def set_prox(self,
                 tau_tv_real: float = 0,
                 tau_tv_imag: float = 0,
                 tau_l1_real: float = 0,
                 tau_l1_imag: float = 0,
                 use_imaginary_constraint: bool = False,
                 use_real_constraint: bool = False,
                 max_imaginary_part: float = np.inf
                 ):

        self.prox_parameters = {"tau_tv_real": float(tau_tv_real),
                                "tau_tv_imag": float(tau_tv_imag),
                                "tau_l1_real": float(tau_l1_real),
                                "tau_l1_imag": float(tau_l1_imag),
                                "use_imaginary_constraint": bool(use_imaginary_constraint),
                                "use_real_constraint": bool(use_real_constraint),
                                "max_imaginary_part": float(max_imaginary_part)
                                }

    def prox(self, x, step):

        # todo: is one order better than another for L1 and TV?

        # ###########################
        # TV proximal operators
        # ###########################

        # note cucim TV implementation requires ~10x memory as array does
        if self.prox_parameters["tau_tv_real"] != 0:
            x_real = tv_prox(x.real, self.prox_parameters["tau_tv_real"] * step)
        else:
            x_real = x.real

        if self.prox_parameters["tau_tv_imag"] != 0:
            x_imag = tv_prox(x.imag, self.prox_parameters["tau_tv_imag"] * step)
        else:
            x_imag = x.imag

        # ###########################
        # L1 proximal operators (softmax)
        # ###########################
        if self.prox_parameters["tau_l1_real"] != 0:
            x_real = soft_threshold(self.prox_parameters["tau_l1_real"] * step, x_real - self.no) + self.no

        if self.prox_parameters["tau_l1_imag"] != 0:
            x_imag = soft_threshold(self.prox_parameters["tau_l1_imag"] * step, x_imag)

        # ###########################
        # projection constraints
        # ###########################
        if self.prox_parameters["use_imaginary_constraint"]:
            x_imag[x_imag < 0] = 0

        if self.prox_parameters["max_imaginary_part"] != np.inf:
            x_imag[x_imag > self.prox_parameters["max_imaginary_part"]] = self.prox_parameters["max_imaginary_part"]

        if self.prox_parameters["use_real_constraint"]:
            x_real[x_real < self.no] = self.no

        return x_real + 1j * x_imag


    def cost(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        # todo: how to unify these?
        e_fwd = self.fwd_model(x, inds=inds)

        costs = 0
        if self.efield_cost_factor > 0:
            if self.mask is None:
                costs += self.efield_cost_factor * 0.5 * (abs(e_fwd[:, -1, :, :] - self.e_measured[inds]) ** 2).mean(axis=(-1, -2))
            else:
                costs += self.efield_cost_factor * 0.5 * (abs(e_fwd[:, -1, self.mask] - self.e_measured[inds][:, self.mask]) ** 2).mean(axis=-1)

        if (1 - self.efield_cost_factor) > 0:
            if self.mask is None:
                costs += (1 - self.efield_cost_factor) * 0.5 * (abs(abs(e_fwd[:, -1]) - abs(self.e_measured[inds])) ** 2).mean(axis=(-1, -2))
            else:
                costs += (1 - self.efield_cost_factor) * 0.5 * (abs(abs(e_fwd[:, -1, self.mask]) - abs(self.e_measured[inds][:, self.mask])) ** 2).mean(axis=-1)

        return costs



class LinearScatt(RIOptimizer):
    def __init__(self,
                 eft: array,
                 model: csr_matrix,
                 no: float,
                 wavelength: float,
                 drs_e: tuple[float],
                 drs_n: tuple[float],
                 shape_n: tuple[int],
                 **kwargs
                 ):
        """
        Born and Rytov optimizer.

        :param eft: n_angles x ny x nx array Fourier-transform of the electric field.
          This will be either the scattered field or the Rytov phase depending on the linear model chosen.
        :param model: The matrix relating the measured field and the scattering potential. This should be generated
          with fwd_model_linear(). Note that the effect of beam frqs and the atf is incorporated in model
        :param no:
        :param wavelength:
        :param drs_e:
        :param drs_n:
        :param shape_n:
        :param **kwargs: parameters passed through to RIOptimizer, including parameters for proximal operator
        """

        # note that have made different choices for LinearScatt optimizers vs. the multislice
        # but still inheriting from RIOptimizer for convenience in constructing proximal operator
        super(LinearScatt, self).__init__(eft,
                                          None,
                                          None,
                                          no,
                                          wavelength,
                                          drs_e,
                                          drs_n,
                                          shape_n,
                                          None,
                                          None,
                                          None,
                                          **kwargs)

        if self.efield_cost_factor != 1:
            raise NotImplementedError(f"Linear scattering models only support self.efield_cost_factor=1, "
                                      f"but values was {self.efield_cost_factor:.3f}")

        if isinstance(self.e_measured, cp.ndarray) and _gpu_available:
            self.model = sp_gpu.csr_matrix(model)
        else:
            self.model = model

    def fwd_model(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        # todo: not using this in gradient/cost because need to manipulate models there

        if isinstance(self.model, sp_gpu.csr_matrix) and _gpu_available:
            spnow = sp_gpu
        else:
            spnow = sp

        ny, nx = self.e_measured.shape[-2:]
        models = [self.model[slice(ny*nx*ii, ny*nx*(ii + 1)), :] for ii in inds]
        nind = len(inds)

        # first division is average
        # second division converts Fourier space to real-space sum
        # factor of 0.5 in cost function killed by derivative factor of 2
        efwd = spnow.vstack(models).dot(x.ravel()).reshape([nind, ny, nx])

        return efwd

    def gradient(self, x, inds=None):

        if inds is None:
            inds = list(range(self.n_samples))

        if isinstance(self.model, sp_gpu.csr_matrix) and _gpu_available:
            spnow = sp_gpu
            xp = cp
        else:
            spnow = sp
            xp = np

        ny, nx = self.e_measured.shape[-2:]
        models = [self.model[slice(ny*nx*ii, ny*nx*(ii + 1)), :] for ii in inds]
        nind = len(inds)

        # first division is average
        # second division converts Fourier space to real-space sum
        # factor of 0.5 in cost function killed by derivative factor of 2
        efwd = spnow.vstack(models).dot(x.ravel()).reshape([nind, ny, nx])
        dc_dm = (efwd - self.e_measured[inds]) / (ny * nx) / (ny * nx)

        dc_dv = xp.stack([((dc_dm[ii].conj()).ravel()[None, :] * m.tocsc()).conj().reshape(x.shape)
                          for ii, m in enumerate(models)], axis=0)

        return dc_dv

    def cost(self, x, inds=None):

        if isinstance(self.model, sp_gpu.csr_matrix):
            spnow = sp_gpu
        else:
            spnow = sp

        ny, nx = self.e_measured.shape[-2:]
        if inds is None:
            model = self.model
            ninds = self.n_samples
        else:
            model = spnow.vstack([self.model[slice(ny * nx * ii, ny * nx * (ii + 1)), :] for ii in inds])
            ninds = len(inds)

        efwd = model.dot(x.ravel()).reshape([ninds, ny, nx])

        return 0.5 * (abs(efwd - self.e_measured[inds]) ** 2).mean(axis=(-1, -2)) / (nx * ny)

    def guess_step(self, x=None):
        ny, nx = self.e_measured.shape[-2:]

        if isinstance(self.model, sp_gpu.csr_matrix):
            u, s, vh = sp.linalg.svds(self.model.get(), k=1, which='LM')
        else:
            u, s, vh = sp.linalg.svds(self.model, k=1, which='LM')

        lipschitz_estimate = s ** 2 / (self.n_samples * ny * nx) / (ny * nx)
        return float(1 / lipschitz_estimate)

    def prox(self, x, step):
        # convert from V to n
        n = get_n(ift3(x), self.no, self.wavelength)
        # apply proximal operator on n
        n_prox = super(LinearScatt, self).prox(n, step)

        return ft3(get_v(n_prox, self.no, self.wavelength))

    def run(self, x_start, **kwargs):
        return super(LinearScatt, self).run(x_start, **kwargs)



class BPM(RIOptimizer):
    """
    Beam propagation method (BPM)
    """

    def __init__(self,
                 e_measured: array,
                 e_measured_bg: array,
                 beam_frqs: array,
                 no: float,
                 wavelength: float,
                 drs_e: tuple[float],
                 drs_n: tuple[float],
                 shape_n: tuple[int],
                 dz_final: float = 0.,
                 atf: Optional[array] = None,
                 apodization: Optional[array] = None,
                 mask: Optional[array] = None,
                 **kwargs
                 ):
        """
        Suppose we have a 3D grid with nz voxels along the propagation direction. We define the electric field
        at the points before and after each voxel, and in an additional plane to account for the imaging. So we have
        nz + 2 electric field planes.

        :param e_measured: measured electric fields
        :param e_measured_bg: measured background electric fields
        :param beam_frqs: n_pattern x 3 array. If provided, modiefied BPM with extra cosine obliquity factor.
          will be used
        :param no: background index of refraction
        :param wavelength: wavelength of light
        :param drs_e: (dy, dx) of the electric field
        :param drs_n: (dz, dy, dx) of the refractive index
        :param shape_n: (nz, ny, nx)
        :param dz_final: distance to propagate field after last surface
        :param atf: coherent transfer function
        :param apodization: apodization used during FFTs
        :param mask: 2D array. Where true, these spatial pixels will be included in the cost function. Where false,
          they will not
        """

        super(BPM, self).__init__(e_measured,
                                  e_measured_bg,
                                  beam_frqs,
                                  no,
                                  wavelength,
                                  drs_e,
                                  drs_n,
                                  shape_n,
                                  dz_final,
                                  atf=atf,
                                  apodization=apodization,
                                  mask=mask,
                                  **kwargs)

        # include cosine oblique factor
        if self.beam_frqs is not None:
            self.thetas, _ = frqs2angles(self.beam_frqs, self.no, self.wavelength)
        else:
            self.thetas = np.zeros((self.n_samples,))

        # backpropagation distance to compute starting field
        self.dz_back = -np.array([float(self.dz_final) + float(self.drs_n[0]) * self.shape_n[0]])

    def fwd_model(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        e_start = self.get_estart(inds=inds)

        # forward propagation
        e_fwd = field_prop.propagate_bpm(e_start,
                                         x,
                                         self.no,
                                         self.drs_n,
                                         self.wavelength,
                                         self.dz_final,
                                         atf=self.atf,
                                         apodization=self.apodization,
                                         thetas=self.thetas[inds])

        return e_fwd

    def gradient(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        e_fwd = self.fwd_model(x, inds=inds)

        # back propagation ... build the gradient from this to save memory
        dtemp = 0
        if self.efield_cost_factor > 0:
            dtemp += self.efield_cost_factor * (e_fwd[:, -1, :, :] - self.e_measured[inds])

        if (1 - self.efield_cost_factor) > 0:
            dtemp += (1 - self.efield_cost_factor) * (abs(e_fwd[:, -1, :, :]) - abs(self.e_measured[inds])) * e_fwd[:, -1, :, :] / abs(e_fwd[:, -1, :, :])

        if self.mask is not None:
            dtemp *= self.mask

        dc_dn = field_prop.backpropagate_bpm(dtemp,
                                             x,
                                             self.no,
                                             self.drs_n,
                                             self.wavelength,
                                             self.dz_final,
                                             atf=self.atf,
                                             apodization=self.apodization,
                                             thetas=self.thetas[inds])[:, 1:-1, :, :]

        del dtemp

        # cost function gradient
        if isinstance(x, cp.ndarray) and _gpu_available:
            xp = cp
        else:
            xp = np

        thetas = xp.asarray(self.thetas[inds])
        if thetas.ndim == 1:
            thetas = xp.expand_dims(thetas, axis=(-1, -2, -3))

        dz = self.drs_n[0]
        ny, nx = x.shape[-2:]
        if self.mask is None:
            denom = ny * nx
        else:
            denom = self.mask.sum()

        dc_dn *= -1j * (2 * np.pi / self.wavelength) * dz / xp.cos(thetas) / denom

        # conjugate in place to avoid using extra GPU memory
        xp.conjugate(e_fwd, out=e_fwd)
        dc_dn *= e_fwd[:, 1:-1, :, :]

        return dc_dn

    def get_estart(self, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        return field_prop.propagate_homogeneous(self.e_measured_bg[inds],
                                                self.dz_back,
                                                self.no,
                                                self.drs_n[1:],
                                                self.wavelength)[..., 0, :, :]


class SSNP(RIOptimizer):
    def __init__(self,
                 e_measured: array,
                 e_measured_bg: array,
                 beam_frqs: array,
                 no: float,
                 wavelength: float,
                 drs_e: tuple[float],
                 drs_n: tuple[float],
                 shape_n: tuple[int],
                 dz_final: float,
                 atf: Optional[array] = None,
                 apodization: Optional[array] = None,
                 mask: Optional[array] = None,
                 **kwargs
                 ):

        super(SSNP, self).__init__(e_measured,
                                   e_measured_bg,
                                   beam_frqs,
                                   no,
                                   wavelength,
                                   drs_e,
                                   drs_n,
                                   shape_n,
                                   dz_final=dz_final,
                                   atf=atf,
                                   apodization=apodization,
                                   mask=mask,
                                   **kwargs)

        # distance to backpropagate to get starting field
        self.dz_back = -np.array([float(self.dz_final) + float(self.drs_n[0]) * self.shape_n[0]])

        # compute kzs, which need to get starting field derivative
        dz, dy, dx = self.drs_n
        ny, nx = self.e_measured.shape[-2:]

        fx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
        fy = np.fft.fftshift(np.fft.fftfreq(ny, dy))
        fxfx, fyfy = np.meshgrid(fx, fy)

        if isinstance(self.e_measured, cp.ndarray) and _gpu_available:
            xp = cp
        else:
            xp = np

        kz = xp.asarray(2 * np.pi * get_fzs(fxfx, fyfy, self.no, self.wavelength))
        kz[xp.isnan(kz)] = 0
        self.kz = kz


    def phi_fwd(self, x, inds=None):
        """
        Return the forward model field and the forward model derivative. Since we do not measure the derivative,
        we do not consider it part of our forward models

        :param x:
        :param inds:
        :return:
        """
        if inds is None:
            inds = list(range(self.n_samples))

            # initial field
        e_start, de_dz_start = self.get_estart(inds=inds)

        # forward propagation
        phi_fwd = field_prop.propagate_ssnp(e_start,
                                            de_dz_start,
                                            x,
                                            self.no,
                                            self.drs_n,
                                            self.wavelength,
                                            self.dz_final,
                                            atf=self.atf,
                                            apodization=self.apodization)
        return phi_fwd


    def fwd_model(self, x, inds=None):
        return self.phi_fwd(x, inds=inds)[..., 0]


    def gradient(self, x, inds=None):
        if isinstance(x, cp.ndarray) and _gpu_available:
            xp = cp
        else:
            xp = np

        if inds is None:
            inds = list(range(self.n_samples))

        phi_fwd = self.phi_fwd(x, inds=inds)

        # back propagation
        # this is the backpropagated field, but we will eventually transform it into the gradient
        # do things this way to reduce memory overhead

        # back propagation ... build the gradient from this to save memory
        dtemp = 0
        if self.efield_cost_factor > 0:
            dtemp += self.efield_cost_factor * (phi_fwd[inds, -1, :, :, 0] - self.e_measured[inds])

        if (1 - self.efield_cost_factor) > 0:
            dtemp += (1 - self.efield_cost_factor) * (abs(phi_fwd[:, -1, :, :, 0]) - abs(self.e_measured[inds])) * phi_fwd[:, -1, :, :, 0] / abs(phi_fwd[:, -1, :, :, 0])

        if self.mask is not None:
            dtemp *= self.mask

        dc_dn = field_prop.backpropagate_ssnp(dtemp,
                                              x,
                                              self.no,
                                              self.drs_n,
                                              self.wavelength,
                                              self.dz_final,
                                              atf=self.atf,
                                              apodization=self.apodization)[:, 1:-1, :, :, 1]

        del dtemp

        # cost function gradient
        ny, nx = x.shape[-2:]
        dz = self.drs_n[0]
        ko = 2 * np.pi / self.wavelength
        # from phi_back, take derivative part
        # from phi_fwd, take the electric field part
        if self.mask is None:
            denom = ny * nx
        else:
            denom = self.mask.sum()

        dc_dn *= (-2 * ko ** 2 * dz * x.conj()) / denom

        # conjugate in place to avoid using extra memory
        xp.conjugate(phi_fwd, out=phi_fwd)
        dc_dn *= phi_fwd[:, :-2, :, :, 0]

        return dc_dn

    def get_estart(self, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        # initial field
        e_start = field_prop.propagate_homogeneous(self.e_measured_bg[inds],
                                                   self.dz_back,
                                                   self.no,
                                                   self.drs_n[1:],
                                                   self.wavelength)[..., 0, :, :]
        # assume initial field is foward propagating only
        de_dz_start = ift2(1j * self.kz * ft2(e_start))

        return e_start, de_dz_start
