"""
Tools for reconstructing optical diffraction tomography (ODT) data using either the Born approximation,
 Rytov approximation, or multislice (paraxial) beam propogation method (BPM) and a FISTA solver. The primary
 reconstruction tasks are carried out with the tomography class
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Arc, Rectangle
# saving/loading
import zarr
# custom tools
from localize_psf import fit, rois, camera, affine
import mcsim.analysis.analysis_tools as tools
from mcsim.analysis import field_prop
from mcsim.analysis.field_prop import _ft2, _ift2

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
coo_matrix = Union[sp.coo_matrix, sp_gpu.coo_matrix]


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
        :param hologram_frqs_guess: list of length npatterns, where each entry is an array of frequencies contained
         in the given pattern. These arrays may be of different sizes, i.e. different patterns may contain a different
          number of frequencies. But each array should be of size n_i x 2
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

        # correction parameters
        self.phase_offsets = phase_offsets
        self.phase_offsets_bg = None
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
                               use_fixed_frequencies: bool = False):
        """
        Estimate hologram frequencies from raw images.
        Guess values need to be within a few pixels for this to succeed. Can easily achieve this accuracy by
        looking at FFT
        :param roi_size_pix: ROI size (in pixels) to use for frequency fitting
        :param save_dir:
        :param use_gpufit: do fitting on GPU with gpufit. Otherwise use CPU
        :param use_fixed_frequencies: determine single set of frequencies for all data/background images
        :return:
        """
        self.reconstruction_settings.update({"roi_size_pix": roi_size_pix})
        self.reconstruction_settings.update({"use_fixed_frequencies": use_fixed_frequencies})

        saving = save_dir is not None

        if saving:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # frequency data associated with images
        def get_hologram_frqs(img: array,
                              fx_guesses,
                              fy_guesses,
                              fxs: array,
                              fys: array,
                              use_guess_init_params: bool,
                              saving: bool,
                              roi_size_pix: int,
                              prefix: str = "",
                              figsize: tuple[float] = (20., 10.),
                              block_id=None):
            """

            :param img:
            :param fx_guesses:
            :param fy_guesses:
            :param use_guess_init_params: if True use guess not only to find an initial ROI but also as the initial
            guess in the fit
            :param saving:
            :param roi_size_pix:
            :param prefix:
            :param block_id: can use this to figure out what blcok we are in when running with dask.array map_blocks()
            :return:
            """

            fxfx, fyfy = np.meshgrid(fxs, fys)
            dfx = fxs[1] - fxs[0]
            dfy = fys[1] - fys[0]
            extent_f = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx,
                        fys[-1] + 0.5 * dfy, fys[0] - 0.5 * dfy]

            nextra_dims = img.ndim - 2

            if img.shape != (1,) * nextra_dims + img.shape[-2:]:
                raise NotImplementedError("not implemented for this shape chunk")

            img = img.squeeze(axis=tuple(range(nextra_dims)))
            apodization = np.outer(hann(img.shape[-2]), hann(img.shape[-1]))
            img_ft = _ft2(img * apodization).squeeze()

            # this will get block id when function is run using map_blocks
            if block_id is None:
                block_id = (0, )

            n_multiplex_this_pattern = len(fx_guesses)
            model = fit.gauss2d()
            rois_all = []
            frqs_holo = np.zeros((n_multiplex_this_pattern, 2))
            frqs_guess = np.zeros((n_multiplex_this_pattern, 2))
            for ii in range(n_multiplex_this_pattern):
                fx_guess = fx_guesses[ii].squeeze()
                fy_guess = fy_guesses[ii].squeeze()

                # ROI centered on frequency guess
                cx_pix = np.argmin(np.abs(fx_guess - fxs))
                cy_pix = np.argmin(np.abs(fy_guess - fys))
                roi2 = rois.get_centered_rois([cy_pix, cx_pix],
                                              [roi_size_pix, roi_size_pix],
                                              min_vals=[0, 0],
                                              max_vals=img.shape[-2:])[0]

                # trim ROIs
                ft2 = np.abs(rois.cut_roi(roi2, img_ft)[0])
                fxfx_roi = rois.cut_roi(roi2, fxfx)[0]
                fyfy_roi = rois.cut_roi(roi2, fyfy)[0]

                extent_f_roi = [fxfx_roi[0, 0] - 0.5 * dfx, fxfx_roi[0, -1] + 0.5 * dfx,
                                fyfy_roi[-1, 0] + 0.5 * dfy, fyfy_roi[0, 0] - 0.5 * dfy]

                # fit
                if use_guess_init_params:
                    fx_ip_guess = fx_guess
                    fy_ip_guess = fy_guess
                else:
                    max_ind = np.unravel_index(np.argmax(ft2), ft2.shape)
                    fx_ip_guess = fxfx_roi[max_ind]
                    fy_ip_guess = fyfy_roi[max_ind]

                init_params = [np.max(ft2),
                               fx_ip_guess,
                               fy_ip_guess,
                               dfx * 3,
                               dfy * 3,
                               0,
                               0]

                rgauss = model.fit(ft2,
                                   (fyfy_roi, fxfx_roi),
                                   init_params=init_params,
                                   fixed_params=[False, False, False, False, False, False, True],
                                   guess_bounds=True)

                rois_all.append(roi2)
                # frqs_guess[ii] = init_params[1:3]
                frqs_guess[ii] = np.array([fx_guess, fy_guess])
                frqs_holo[ii] = rgauss["fit_params"][1:3]

            # optional: plot results
            if saving and np.all(np.array(block_id) == 0):
                figh = plt.figure(figsize=figsize)

                ax = figh.add_subplot(1, 2, 1)
                ax.set_title("$|I(f)|$")
                ax.imshow(np.abs(img_ft), extent=extent_f,
                          norm=PowerNorm(gamma=0.1), cmap="bone")
                ax.plot(frqs_guess[:, 0], frqs_guess[:, 1], 'gx')
                ax.plot(frqs_holo[:, 0], frqs_holo[:, 1], 'rx')
                for roi in rois_all:
                    ax.add_artist(Rectangle((fxs[roi[2]], fys[roi[0]]),
                                             fxs[roi[3] - 1] - fxs[roi[2]],
                                             fys[roi[1] - 1] - fys[roi[0]],
                                             edgecolor='k',
                                             fill=False))
                ax.set_xlabel("$f_x$ (1/$\mu m$)")
                ax.set_ylabel("$f_y$ (1/$\mu m$)")

                # plot one roi
                ax = figh.add_subplot(1, 2, 2)
                ax.set_title("ROI")
                ax.imshow(ft2, extent=extent_f_roi,
                          norm=PowerNorm(gamma=0.1), cmap="bone")
                ax.plot(fx_guess, fy_guess, 'gx')
                ax.plot(frqs_holo[-1, 0], frqs_holo[-1, 1], 'rx')
                ax.set_xlabel("$f_x$ (1/$\mu m$)")
                ax.set_ylabel("$f_y$ (1/$\mu m$)")

                # figh.savefig(Path(save_dir, f"{prefix:s}=pattern_{ii:d}=multiplex_holo_frq_diagnostic.png"))
                figh.savefig(Path(save_dir, f"{prefix:s}=hologram_frq_diagnostic.png"))
                plt.close(figh)


            return np.expand_dims(frqs_holo, axis=tuple(range(nextra_dims)))

        # ############################
        # fit hologram frequencies for ALL chunks
        # ############################
        delayed_frqs = []
        delayed_frqs_bg = []

        if not use_fixed_frequencies:
            slices_bg = tuple([slice(None) for _ in range(self.nextra_dims)])
        else:
            slices_bg = tuple([slice(0, 1) for _ in range(self.nextra_dims)])

        # todo: test this logic
        fit_data_imgs = not use_fixed_frequencies or self.use_average_as_background
        fit_bg_imgs = not self.use_average_as_background

        # loop over patterns
        for ii in range(self.npatterns):
            n_multiplex_this_pattern = len(self.hologram_frqs[ii])

            if fit_data_imgs:
                delayed_frqs.append(da.map_blocks(get_hologram_frqs,
                                                  self.imgs_raw[..., ii, :, :],
                                                  self.hologram_frqs[ii][:, 0],
                                                  self.hologram_frqs[ii][:, 1],
                                                  self.fxs,
                                                  self.fys,
                                                  use_guess_init_params=False,
                                                  saving=saving,
                                                  roi_size_pix=roi_size_pix,
                                                  prefix=str(ii),
                                                  dtype=float,
                                                  chunks=(1,) * self.nextra_dims + (n_multiplex_this_pattern, 2),
                                                  drop_axis=(-1, -2),
                                                  new_axis=(-1, -2)
                                                  ))

            if fit_bg_imgs:
                delayed_frqs_bg.append(da.map_blocks(get_hologram_frqs,
                                         self.imgs_raw_bg[slices_bg][..., ii, :, :],
                                         self.hologram_frqs[ii][:, 0],
                                         self.hologram_frqs[ii][:, 1],
                                         self.fxs,
                                         self.fys,
                                         use_guess_init_params=False,
                                         saving=saving,
                                         roi_size_pix=roi_size_pix,
                                         prefix=f"background_{ii:d}",
                                         dtype=float,
                                         chunks=(1,) * self.nextra_dims + (n_multiplex_this_pattern, 2),
                                         drop_axis=(-1, -2),
                                         new_axis=(-1, -2)
                                         ))

        # do frequency calibration
        print(f"calibrating {self.npatterns:d} images,"
              f" multiplexing an average of {np.mean([f.shape[-2] for f in self.hologram_frqs]):.1f} plane waves")

        if self.verbose:
            with ProgressBar():
                frqs_hologram, frqs_hologram_bg = dask.compute([delayed_frqs, delayed_frqs_bg])[0]
        else:
            frqs_hologram, frqs_hologram_bg = dask.compute([delayed_frqs, delayed_frqs_bg])[0]

        if self.use_average_as_background:
            frqs_hologram_bg = frqs_hologram

        if use_fixed_frequencies:
            frqs_hologram = frqs_hologram_bg

        self.hologram_frqs = frqs_hologram
        self.hologram_frqs_bg = frqs_hologram_bg

    def estimate_reference_frq(self,
                               mode: str = "fit",
                               save_dir: Optional[str] = None):
        """
        Estimate hologram reference frequency
        :param mode: if "fit" fit the residual speckle pattern to try and estimate the reference frequency.
         If "average" take the average of self.hologram_frqs as the reference frequency
        :param save_dir:
        :return:
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
            frq_ref = np.mean(np.concatenate(self.hologram_frqs, axis=-2), axis=-2)
            frq_ref_bg = np.mean(np.concatenate(self.hologram_frqs_bg, axis=-2), axis=-2)
        else:
            raise ValueError(f"'mode' must be '{mode:s}' but must be 'fit' or 'average'")

        self.reference_frq = frq_ref
        self.reference_frq_bg = frq_ref_bg

    def get_beam_frqs(self):
        """
        Get beam incident beam frequencies from hologram frequencies and reference frequency

        :return beam_frqs: array of size N1 x N2 ... x Nm x 3
        """

        bxys = [f - np.expand_dims(self.reference_frq, axis=-2) for f in self.hologram_frqs]
        bzs = [field_prop.get_fzs(bxy[..., 0], bxy[..., 1], self.no, self.wavelength) for bxy in bxys]
        beam_frqs = [np.stack((bxy[..., 0], bxy[..., 1], bz), axis=-1) for bxy, bz in zip(bxys, bzs)]

        return beam_frqs

    def find_affine_xform_to_frqs(self,
                                  offsets: list[np.ndarray],
                                  save_dir: Optional[str] = None):
        """
        Fit affine transformation between device and measured frequency space.

        This could be between frequencies displayed on DMD and measured frequency space (DMD in imaging plane)
        or between mirror positions on DMD and frequency space (DMD in Fourier plane)

        :param offsets:
        :param save_dir:
        :return:
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
                        fourier_mask: Optional[np.ndarray] = None,
                        fit_phases: bool = False,
                        correct_amplitudes: bool = True,
                        fit_translations: bool = False,
                        apodization: Optional[np.ndarray] = None,
                        use_gpu: bool = False):
        """
        Unmix and preprocess holograms

        Note that this only depends on reference frequencies, and not on determined hologram frequencies

        :param bg_average_axes: axes to average along when producing background images
        :param kernel_size: when averaging time points to generate a background image, this is the number of
        time points that will be used
        :param fourier_mask: area to be cut out of holograms
        :param fit_phases: whether to fit phase differences between image and background holograms
        :param correct_amplitudes:
        :param apodization: if None use tukey apodization with alpha = 0.1. To use no apodization set equal to 1
        :param use_gpu:
        :return:
        """

        self.reconstruction_settings.update({"background_average_kernel_size": kernel_size})
        self.reconstruction_settings.update({"correct_amplitudes": correct_amplitudes})
        self.reconstruction_settings.update({"fit_translations": fit_translations})

        if use_gpu and _gpu_available:
            xp = cp
        else:
            xp = np

        if apodization is None:
            apodization = np.outer(tukey(self.ny, alpha=0.1), tukey(self.nx, alpha=0.1))

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
        if fourier_mask is None:
            holograms_ft = holograms_ft_raw
        else:
            # ensure masks has same dims and chunks as holograms_ft_raw
            masks_da = da.from_array(xp.expand_dims(xp.array(fourier_mask),
                                                    axis=tuple(range(0, self.imgs_raw.ndim - fourier_mask.ndim))),
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

            if fourier_mask is None:
                holograms_ft_bg = holograms_ft_raw_bg
            else:
                # ensure masks has same dims and chunks as holograms_ft_raw
                masks_da_bg = da.from_array(xp.expand_dims(xp.array(fourier_mask),
                                                           axis=tuple(range(0, self.imgs_raw.ndim - fourier_mask.ndim))),
                                            chunks=holograms_ft_raw_bg.chunksize)
                holograms_ft_bg = da.map_blocks(cut_mask,
                                                holograms_ft_raw_bg,
                                                masks_da_bg,
                                                mask_val=0,
                                                dtype=complex)

        # #########################
        # optionally fit translations
        # #########################
        if not fit_translations:
            self.translations = np.zeros(holograms_ft.shape[:-2] + (1, 1, 2))
            self.translations_bg = np.zeros(holograms_ft_bg.shape[:-2] + (1, 1, 2))
        else:
            print("computing translations")
            def ft_abs(m): return _ft2(abs(_ift2(m)))

            holograms_abs_ft = da.map_blocks(ft_abs,
                                             holograms_ft,
                                             dtype=complex)

            if self.use_average_as_background:
                holograms_abs_ft_bg = holograms_abs_ft
            else:
                holograms_abs_ft_bg = da.map_blocks(ft_abs,
                                                    holograms_ft_bg,
                                                    dtype=complex)

            # if we are going to average along a dimension (i.e. if it is in bg_average_axes) then need to use
            # single slice as background for that dimension.
            translation_ref_slice = tuple([slice(0, 1) if a in bg_average_axes else slice(None) for a in range(self.nextra_dims)] +
                                          [slice(None)] * 3)

            # fit phase ramp in holograms ft
            self.translations = da.map_blocks(fit_phase_ramp,
                                              holograms_abs_ft,
                                              holograms_abs_ft_bg[translation_ref_slice],
                                              self.dxy,
                                              dtype=float,
                                              new_axis=-1,
                                              chunks=holograms_abs_ft.chunksize[:-2] + (1, 1, 2)).compute()

            if self.use_average_as_background:
                self.translations_bg = self.translations
            else:
                self.translations_bg = da.map_blocks(fit_phase_ramp,
                                                     holograms_abs_ft_bg,
                                                     holograms_abs_ft_bg[translation_ref_slice],
                                                     self.dxy,
                                                     dtype=float,
                                                     new_axis=-1,
                                                     chunks=holograms_abs_ft.chunksize[:-2] + (1, 1, 2)).compute()

            # correct translations
            fx_bcastable = np.expand_dims(self.fxs, axis=list(range(self.nextra_dims)) + [-3, -2])
            fy_bcastable = np.expand_dims(self.fys, axis=list(range(self.nextra_dims)) + [-3, -1])
            factor = da.exp(2 * np.pi * 1j * (fx_bcastable * self.translations[..., 0] +
                                              fy_bcastable * self.translations[..., 1]))

            holograms_ft *= factor

            if self.use_average_as_background:
                holograms_ft_bg = holograms_ft
            else:
                factor_bg = da.exp(2 * np.pi * 1j * (fx_bcastable * self.translations_bg[..., 0] +
                                                     fy_bcastable * self.translations_bg[..., 1]))
                holograms_ft_bg = holograms_ft_bg * factor_bg


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
        print("computing background electric field")
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

    def reconstruct_n(self,
                      mode: str = "rytov",
                      scattered_field_regularization: float = 50,
                      reconstruction_regularizer: float = 0.1,
                      dxy_sampling_factor: float = 1.,
                      dz_sampling_factor: float = 1.,
                      z_fov: float = 20,
                      nbin: int = 1,
                      realspace_mask: Optional[np.ndarray] = None,
                      step: float = 1e-5,
                      use_gpu: bool = False,
                      use_constant_n_start: bool = False,
                      **kwargs):

        """
        Reconstruct refractive index using one of a several different models

        :param mode: "born", "rytov", "bpm", or "ssnp"
        :param scattered_field_regularization: regularization used in computing scattered field
          or Rytov phase
        :param reconstruction_regularizer:
        :param dxy_sampling_factor:
        :param dz_sampling_factor:
        :param z_fov: z-field of view in microns
        :param nbin: for BPM, bin image by this factor
        :param realspace_mask: indicate which parts of image to exclude/keep
        :param step: ignored if mode is "born" or "rytov"
        :param use_gpu:
        :param **kwargs: passed through to both the constructor and the run() method of the optimizer.
          These are used to e.g. set the strength of TV regularization, the number of iterations, etc.
          Optimizer, RIOptimizer, and classes inheriting from RIOptimizer for more details
        :return n, drs_n, xform_recon_pix2coords:
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

        # ############################
        # check arrays are chunked by volume
        # ############################
        # todo: rechunk here if necessary ...
        if self.holograms_ft.chunksize[-3] != self.npatterns:
            raise ValueError("")

        if self.holograms_ft_bg.chunksize[-3] != self.npatterns:
            raise ValueError("")

        # ############################
        # get size information
        # ############################
        # todo: set pixel size directly
        drs_ideal, size_ideal = get_reconstruction_nyquist_sampling(self.no,
                                                                    self.na_detection,
                                                                    self.na_excitation,
                                                                    self.wavelength,
                                                                    self.dxy,
                                                                    self.holograms_ft.shape[-2:],
                                                                    z_fov,
                                                                    dz_sampling_factor=dz_sampling_factor,
                                                                    dxy_sampling_factor=dxy_sampling_factor)

        # generate ATF ... ultimately want to do this based on pupil function defined in init
        e_size = (self.holograms_ft.shape[-2] // nbin, self.holograms_ft.shape[-1] // nbin)
        fx_atf = xp.fft.fftshift(xp.fft.fftfreq(e_size[-1], self.dxy * nbin))[None, :]
        fy_atf = xp.fft.fftshift(xp.fft.fftfreq(e_size[-2], self.dxy * nbin))[:, None]
        atf = (xp.sqrt(fx_atf ** 2 + fy_atf ** 2) <= self.fmax).astype(complex)

        if mode == "born" or mode == "rytov":
            if nbin != 1:
                raise NotImplementedError(f"nbin={nbin:d}, but only nbin=1 is supported for mode {mode:s}")

            dz_final = None
            drs_n = drs_ideal
            n_size = size_ideal

            # affine transformation from reconstruction coordinates to pixel indices
            # for reconstruction, using FFT induced coordinates, i.e. zero is at array index (ny // 2, nx // 2)
            # for matrix, using image coordinates (0, 1, ..., N - 1)
            # note, due to order of operations -n//2 =/= - (n//2) when nx is odd
            xform_recon_pix2coords = affine.params2xform([drs_n[-1], 0, -(n_size[-1] // 2) * drs_n[-1],
                                                          drs_n[-2], 0, -(n_size[-2] // 2) * drs_n[-2]])
            optimizer = mode

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

            if use_gpu:
                u, s, vh = sp.linalg.svds(linear_model.get(), k=1, which='LM')
            else:
                u, s, vh = sp.linalg.svds(linear_model, k=1, which='LM')
            lipschitz_est = float(s) ** 2 / (self.npatterns * self.ny * self.nx) / (self.ny * self.nx)
            step = 1 / lipschitz_est

        elif mode == "bpm" or mode == "ssnp":
            # after binning
            n_size = (size_ideal[0], self.holograms_ft.shape[-2] // nbin, self.holograms_ft.shape[-1] // nbin)
            drs_n = (drs_ideal[0], self.dxy * nbin, self.dxy * nbin)
            # position z=0 in the middle of the volume, accounting for the fact the efields and n are shifted by dz/2
            dz_final = -drs_n[0] * ((n_size[0] - 1) - n_size[0] // 2 + 0.5)

            # affine transformation from reconstruction coordinates to pixel indices
            # coordinates in finer coordinates
            # todo: check this?
            x = (xp.arange(self.nx) - (self.nx // 2)) * self.dxy
            y = (xp.arange(self.ny) - (self.ny // 2)) * self.dxy
            xb = camera.bin(x, [nbin], mode="mean")
            yb = camera.bin(y, [nbin], mode="mean")

            xform_recon_pix2coords = affine.params2xform([drs_n[-1], 0, float(xb[0]),
                                                          drs_n[-2], 0, float(yb[0])])

            if mode == "bpm":
                optimizer = BPM
            elif mode == "ssnp":
                optimizer = SSNP
        else:
            raise ValueError(f"mode must be ..., but was {mode:s}")

        # general info
        if self.verbose:
            print(f"computing index of refraction for {np.prod(self.imgs_raw.shape[:-3]):d} images "
                  f"using mode {mode:s}. "
                  f"Image size = {self.npatterns} x {self.ny:d} x {self.nx:d}, "
                  f"reconstruction size = {n_size[0]:d} x {n_size[1]:d} x {n_size[2]:d}")

        # model for computing guess
        if mode == "born":
            guess_mode = "born"
        else:
            guess_mode = "rytov"

        if self.verbose:
            tstart_linear_model = time.perf_counter()

        linear_model_no_interp = fwd_model_linear(mean_beam_frqs_arr[..., 0],
                                                  mean_beam_frqs_arr[..., 1],
                                                  mean_beam_frqs_arr[..., 2],
                                                  self.no,
                                                  self.na_detection,
                                                  self.wavelength,
                                                  (self.ny, self.nx),
                                                  (self.dxy, self.dxy),
                                                  n_size,
                                                  drs_n,
                                                  mode=guess_mode,
                                                  interpolate=False,
                                                  use_gpu=use_gpu)

        if self.verbose:
            print(f"Generated linear model for initial guess in {time.perf_counter() - tstart_linear_model:.2f}s")

        # apodization
        apodization_n = xp.outer(xp.asarray(tukey(n_size[-2], alpha=0.1)),
                                 xp.asarray(tukey(n_size[-1], alpha=0.1)))

        def recon(efields_ft,
                  efields_bg_ft,
                  beam_frqs,
                  rmask,
                  dxy,
                  no,
                  wavelength,
                  dz_final,
                  atf,
                  apod,
                  step,
                  optimizer,
                  verbose,
                  block_id=None):
            # todo: might be cleaner to compute starting points and scattered fields in this function
            #  disadvantage is harder to access from outside ... although still easy to compute

            # mempool = cp.get_default_memory_pool()
            # pinned_mempool = cp.get_default_pinned_memory_pool()
            # # memory_start = mempool.used_bytes()
            # memory_start = 0
            # mempool.free_all_blocks()
            # print(f"used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB, memory pool = {mempool.total_bytes() / 1e9:.3f}GB")
            # # fourier plan cache
            # cache = cp.fft.config.get_plan_cache()
            # cache.show_info()

            if use_gpu:
                # ensure not holding large FFT planes in the cache
                # todo: caching 2D FFT's not a problem, but caching the 3D FFT can be!
                cache = cp.fft.config.get_plan_cache()
                cache.set_size(0)
                # cache.set_memsize(0)

            nextra_dims = efields_ft.ndim - 3
            dims = tuple(range(nextra_dims))
            efields_ft = efields_ft.squeeze(axis=dims)
            efields_bg_ft = efields_bg_ft.squeeze(axis=dims)

            if use_gpu:
                efields_ft = xp.asarray(efields_ft)
                efields_bg_ft = xp.asarray(efields_bg_ft)

            if optimizer == "born":
                efield_scattered_ft = _ft2(get_scattered_field(_ift2(efields_ft),
                                                               _ift2(efields_bg_ft),
                                                               scattered_field_regularization))
            else:
                efield_scattered_ft = _ft2(get_rytov_phase(_ift2(efields_ft),
                                                           _ift2(efields_bg_ft),
                                                           scattered_field_regularization))

            # initial guess
            if use_constant_n_start:
                v_fts_start = xp.zeros(n_size, dtype=complex)
            else:
                v_fts_start = inverse_model_linear(efield_scattered_ft,
                                                   linear_model_no_interp,
                                                   n_size,
                                                   regularization=reconstruction_regularizer,
                                                   no_data_value=0.)

            if optimizer == "born" or optimizer == "rytov":
                opt = LinearScatt(efield_scattered_ft,
                                  linear_model,
                                  no,
                                  wavelength,
                                  (dxy, dxy),
                                  drs_n,
                                  v_fts_start.shape,
                                  **kwargs
                                  )

                results = opt.run(v_fts_start,
                                  step=step,
                                  verbose=verbose,
                                  **kwargs
                                  )

                n = get_n(_ift3(results["x"]), no, wavelength).reshape((1,) * nextra_dims + v_fts_start.shape[-3:])

            else:
                # delete variables we no longer need
                del efield_scattered_ft

                # bin if desired
                n_start = get_n(_ift3(v_fts_start), no, wavelength)
                del v_fts_start

                efields = camera.bin(_ift2(efields_ft), [nbin, nbin], mode="mean")
                del efields_ft

                efields_bg = camera.bin(_ift2(efields_bg_ft), [nbin, nbin], mode="mean")
                del efields_bg_ft

                if beam_frqs.shape[0] == 1:
                    bf = beam_frqs[0]
                else:
                    bf = None

                model = optimizer(efields,
                                  efields_bg,
                                  bf,
                                  no,
                                  wavelength,
                                  drs_e=None,
                                  drs_n=drs_n,
                                  shape_n=n_start.shape,
                                  dz_final=dz_final,
                                  atf=atf,
                                  apodization=apod,
                                  **kwargs)

                results = model.run(n_start,
                                    step=step,
                                    verbose=verbose,
                                    **kwargs
                                    )

                n = results["x"].reshape((1,) * nextra_dims + n_start.shape[-3:])

            # newline to preserve optimizer printout
            if verbose:
                print("")

            return _to_cpu(n)

        # #######################
        # get refractive index
        # #######################
        n = da.map_blocks(recon,
                          self.holograms_ft,  # data
                          self.holograms_ft_bg, # background
                          mean_beam_frqs_arr,
                          realspace_mask, # masks
                          self.dxy,
                          self.no,
                          self.wavelength,
                          dz_final,
                          atf,
                          apodization_n,
                          step,
                          optimizer,
                          self.verbose,
                          chunks=(1,) * self.nextra_dims + n_size,
                          dtype=complex,
                          )


        self.reconstruction_settings.update({"mode": mode,
                                             "scattered_field_regularization": scattered_field_regularization,
                                             "reconstruction_regularizer": reconstruction_regularizer,
                                             "dxy_sampling_factor": dxy_sampling_factor,
                                             "dz_sampling_factor": dz_sampling_factor,
                                             "z_fov": z_fov,
                                             "dz_final": dz_final,
                                             "nbin": nbin,                                             
                                             "use_gpu": use_gpu,
                                             "use_contant_n_start": use_constant_n_start
                                             }
                                            )
        self.reconstruction_settings.update(kwargs)

        return n, drs_n, xform_recon_pix2coords

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
        ax.plot(translations[..., 1], label="sig")
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
        ph = np.unwrap(self.phase_offsets[slices].squeeze(axis=squeeze_axes), axis=0)
        ph_bg = np.unwrap(self.phase_offsets_bg[slices].squeeze(axis=squeeze_axes), axis=0)

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

        def to_numpy(arr):
            if isinstance(arr, cp.ndarray) and _gpu_available:
                return arr.get()
            else:
                return arr

        # ######################
        # plot
        # ######################
        img_now = to_numpy(self.imgs_raw[index].compute())
        img_ft = _ft2(img_now)

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
            holo_ft = to_numpy(self.holograms_ft[index].compute())
            holo = _ift2(holo_ft)

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

        except AttributeError as e:
            print(e)

        # ######################
        # hologram background
        # ######################
        try:
            index_bg = tuple([v if self.holograms_ft.shape[ii] != 0 else 0 for ii, v in enumerate(index)])

            holo_ft_bg = to_numpy(self.holograms_ft_bg[index_bg].compute())
            holo_bg = _ift2(holo_ft_bg)

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

        except AttributeError as e:
            print(e)

        return figh


# FFT idioms
def _ft3(m):
    if isinstance(m, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    return xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(m, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))


def _ift3(m):
    if isinstance(m, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    return xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(m, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))


def _to_cpu(m):
    if _gpu_available and isinstance(m, cp.ndarray):
        return m.get()
    else:
        return m

def soft_threshold(t: float,
                   x: array) -> array:
    """
    Softmax function, which is the proximal operator for the LASSO (L1 regularization) problem

    :param t: softmax parameter
    :param x: array to take softmax of
    :return x_out:
    """
    # x_out = np.array(x, copy=True)
    x_out = x.copy()
    x_out[x > t] -= t
    x_out[x < -t] += t
    x_out[abs(x) <= t] = 0

    return x_out


def cut_mask(img: array,
             mask: array,
             mask_val: float = 0) -> array:
    """
    Mask points in image and set to a given value. Designed to be used with dask.array map_blocks()

    :param img:
    :param mask:
    :param mask_val: At any position where mask is True, replace the value of img with this value
    :return: img_masked
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
def frqs2angles(frqs: np.ndarray,
                no: float,
                wavelength: float):
    """
    Convert from frequency vectors to angle vectors. Frequency vectors should be normalized to no / wavelength
    :param frqs: (fx, fy, fz), expect |frqs| = no / wavelength
    :param no: background index of refraction
    :param wavelength:
    :return: theta, phi
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
    :param no:
    :param wavelength:
    :param theta:
    :param phi:
    :return:
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
            # mask = xp.ones(imgs[ind].shape, dtype=bool)
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

    :param imgs: n0 x n1 x ... x n_{-2} x n_{-1} array
    :param ref_imgs: reference images. Should be broadcastable to same size as imgs.
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
                    regularization: float = 0.) -> array:
    """
    Compute rytov phase from field and background field. The Rytov phase is \psi_s(r) where
    U_total(r) = exp[\psi_o(r) + \psi_s(r)]
    where U_o(r) = exp[\psi_o(r)] is the unscattered field

    We calculate \psi_s(r) = log | U_total(r) / U_o(r)| + 1j * unwrap[angle(U_total) - angle(U_o)]

    :param efields_ft: n0 x n1 ... x nm x ny x nx
    :param efields_bg_ft: broadcastable to same size as eimgs
    :param float regularization: regularization value. Any pixels where the background
    exceeds this value will be set to zero
    :return psi_rytov:
    """

    use_gpu = isinstance(eimgs, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
        def uwrap(m): return xp.array(unwrap_phase(m.get()))
    else:
        xp = np
        def uwrap(m): return unwrap_phase(m)

    eimgs = xp.asarray(eimgs)
    eimgs_bg = xp.asarray(eimgs_bg)

    # broadcast arrays
    # eimgs_bg = _ift2(efields_bg_ft)
    # eimgs = _ift2(efields_ft)
    eimgs, eimgs_bg = xp.broadcast_arrays(eimgs, eimgs_bg)

    # output values
    phase_diff = np.mod(xp.angle(eimgs) - xp.angle(eimgs_bg), 2 * np.pi)
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
        psi_rytov[ind] += 1j * uwrap(phase_diff[ind])

    # regularization
    psi_rytov[abs(eimgs_bg) < regularization] = 0

    return psi_rytov


def get_scattered_field(holograms: array,
                        holograms_bg: array,
                        regularization: float = 0.) -> array:
    """
    Compute estimate of scattered electric field with regularization. This function only operates on the
    last two dimensions of the array

    :param eimgs_ft: array of size n0 x ... x nm x ny x nx
    :param eimgs_bg_ft: broadcastable to same size as eimgs_ft
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
                   apodization: array = 1) -> array:
    """
    Given an off-axis hologram image, determine the electric field

    :param img: n1 x ... x n_{-3} x n_{-2} x n_{-1} array
    :param dxy: pixel size
    :param fmax_int: maximum frequency where intensity OTF has support
    :param fx_ref: x-component of hologram reference frequency
    :param fy_ref: y-component of hologram reference frequency
    :param apodization: apodization window applied to real-space image before Fourier transformation
    :return efield_ft: hologram electric field
    """

    if isinstance(img, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    apodization = xp.asarray(apodization)

    # FT of image
    img_ft = _ft2(img * apodization)

    # get frequency data
    ny, nx = img_ft.shape[-2:]
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
                                        dxy: float,
                                        img_size: tuple[int],
                                        z_fov: float,
                                        dz_sampling_factor: float = 1.,
                                        dxy_sampling_factor: float = 1.) -> (tuple[float], tuple[int]):
    """
    Helper function for computing Nyquist sampled grid size for scattering potential based on incoming beam angles.
    Note that this may not always be the ideal grid size, especially for multislice methods where the accuracy
    of the beam propagation depends on the choice of voxel size

    :param no: background index of refraction
    :param na_det: numerical aperture of detection objective
    :param na_exc: maximum excitation numerical aperture (i.e. corresponding to the steepest input beam and not nec.
    the objective).
    :param wavelength: wavelength
    :param dxy: camera pixel size
    :param img_size: (ny, nx) size of hologram images
    :param z_fov: field-of-view in z-direction
    :param dz_sampling_factor: z-spacing as a fraction of the nyquist limit
    :param dxy_sampling_factor: xy-spacing as a fraction of nyquist limit
    :return (dz_sp, dxy_sp, dxy_sp), (nz_sp, ny_sp, nx_sp):
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

            fz = xp.tile(field_prop.get_fzs(fx, fy, no, wavelength), [nimgs, 1, 1])

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

            fz = field_prop.get_fzs(fx_rytov,
                                    fy_rytov,
                                    no,
                                    wavelength)

            Fz = fz - xp.expand_dims(beam_fz, axis=(1, 2))

            # atf at f_rytov
            # basis for applying the ATF in rytov case ... psi is a frequency shifted
            # version of scattered field, hence should nominally see effect of atf
            # atf = xp.tile(atf, [nimgs, 1, 1])
            # atf = tools.translate_ft(atf, beam_fx, beam_fy, drs=(dy, dx))

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
                         regularization: float = 0.1,
                         no_data_value: float = np.nan) -> array:
    """
    Given a set of holograms obtained using ODT, put the hologram information back in the correct locations in
    Fourier space

    :param efield_fts: The exact definition of efield_fts depends on whether "born" or "rytov" mode is used.
    Any points in efield_fts which are NaN will be ignored. efield_fts can have an arbitrary number of leading
    singleton dimensions, but must have at least three dimensions.
    i.e. it should have shape 1 x ... x 1 x nimgs x ny x nx
    :param model: forward model matrix. Generated from fwd_model_linear(). Should have interpolate=False
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
        vinds_angle = (v_ind[0][this_angle], v_ind[1][this_angle], v_ind[2][this_angle])
        einds_angle = (e_ind[0][this_angle], e_ind[1][this_angle], e_ind[2][this_angle])
        data_angle = data[this_angle]

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
                      ni: float,
                      wavelength: float,
                      **kwargs) -> matplotlib.figure.Figure:
    """
    Illustrate the region of frequency space which is obtained using the plane waves described by frqs

    :param frqs: nfrqs x 2 array of [[fx0, fy0], [fx1, fy1], ...]
    :param na_detect: detection NA
    :param na_excite: excitation NA
    :param ni: index of refraction of medium that samle is immersed in. This may differ from the immersion medium
    of the objectives
    :param wavelength:
    :param kwargs: passed through to figure
    :return figh:
    """
    frq_norm = ni / wavelength
    alpha_det = np.arcsin(na_detect / ni)

    if na_excite / ni < 1:
        alpha_exc = np.arcsin(na_excite / ni)
    else:
        # if na_excite is immersion objective and beam undergoes TIR at interface for full NA
        alpha_exc = np.pi/2

    fzs = field_prop.get_fzs(frqs[:, 0], frqs[:, 1], ni, wavelength)
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
                             phase_lim: float = 0.1,
                             block_while_display: bool = True):
    """
    Display reconstruction results and (optionally) raw data in Napari

    :param recon_fname: refractive index reconstruction stored in zarr file
    :param raw_data_fname: raw data stored in zar file
    :param show_raw:
    :param show_raw_ft:
    :param show_v_fft:
    :param show_efields:
    :param block_while_display:
    :return:
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

    try:
        dz_v = img_z.attrs["dz"]
        dxy_v = img_z.attrs["dx"]
    except KeyError:
        dz_v, dxy_v, _ = img_z.attrs["dr"]

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

        v = da.map_blocks(get_v, n_full, img_z.attrs["no"], wavelength, dtype=complex)
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
                         name="|e rytov|", contrast_limits=[0, 1.2],
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

        viewer.add_image(da.abs(estack) - da.abs(ebg_stack), scale=(dz_v / dxy_v, 1, 1),
                         name="|e| - |e bg|", contrast_limits=[-500, 500],
                         colormap="twilight_shifted",
                         translate=(0, nx_raw + 4*nx))

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
                         name="angle(e) - angle(e bg)", contrast_limits=[-phase_lim, phase_lim],
                         colormap="PiYG",
                         translate=(ny, 0)
                         )

        label_pts = np.array([[0, nx_raw],
                              [ny, nx_raw],
                              [0, nx_raw + nx],
                              [ny, nx_raw + nx],
                              [0, nx_raw + 2 * nx],
                              [ny, nx_raw + 2 * nx],
                              [0, nx_raw + 3 * nx],
                              [ny, nx_raw + 3 * nx],
                              [0, nx_raw + 4 * nx],
                              [ny, 0]
                              ]) + np.array([ny / 10, nx / 2])
        ttls = ["|e scatt|", "angle(e scatt)",
                "|e rytov|", "angle(e rytov)",
                "|e|", "angle(e)",
                "|e bg|", "angle(e bg)",
                "|e| - |e bg|",
                "angle(e) - angle(e bg)"]

        viewer.add_points(label_pts,
                          features={"names": ttls},
                          text={"string": "{names:s}",
                                "size": 30,
                                "color": "red"},
                          )

    viewer.dims.axis_labels = ["position", "time", "", "", "pattern", "z", "y", "x"]
    # set to first position
    viewer.dims.set_current_step(axis=0, value=0)
    # set to first time
    viewer.dims.set_current_step(axis=1, value=0)

    # block until closed by user
    viewer.show(block=block_while_display)

    return viewer


class Optimizer():
    def __init__(self):
        self.n_samples = None
        self.prox_parameters = {}

    def cost(self, x, inds=None):
        pass

    def gradient(self, x, inds=None):
        pass

    def test_gradient(self, x, jind=0, inds=None, dx=1e-5):
        """

        :param x: point to compute gradient at
        :param jind: 1D index into x to compute gradient at
        :param inds: samples to compute gradient at
        :param dx: gradient step size
        :return grad, grad_numerical:
        """

        use_gpu = isinstance(x, cp.ndarray) and _gpu_available
        if use_gpu:
            xp = cp
        else:
            xp = np

        # in case x is multi dimensional, unravel
        xind = np.unravel_index(jind, x.shape)

        # compute gradient numerically
        x1 = xp.array(x, copy=True)
        x1[xind] -= 0.5 * dx
        c1 = self.cost(x1, inds=inds)

        x2 = xp.array(x, copy=True)
        x2[xind] += 0.5 * dx
        c2 = self.cost(x2, inds=inds)

        gn = (c2 - c1) / dx

        # if x is complex, compute "complex gradient"
        if xp.iscomplexobj(x):
            x1c = xp.array(x, copy=True)
            x1c[xind] -= 0.5 * dx * 1j
            c1c = self.cost(x1c, inds=inds)

            x2c = xp.array(x, copy=True)
            x2c[xind] += 0.5 * dx * 1j
            c2c = self.cost(x2c, inds=inds)

            gn = gn + (c2c - c1c) / dx * 1j

        # compute gradient
        slices = [slice(None)] + [slice(i, i+1) for i in xind]
        g = self.gradient(x, inds=inds)[tuple(slices)].ravel()

        return g, gn

    def prox(self, x):
        pass

    def guess_step(self, x):
        pass

    def run(self,
            x_start: array,
            step: float,
            max_iterations: int = 100,
            use_fista: bool = True,
            stochastic_descent: bool = True,
            nmax_stochastic_descent: int = np.inf,
            verbose: bool = False,
            compute_cost: bool = False,
            compute_all_costs: bool = False,
            line_search: bool = False,
            line_search_factor: float = 0.5,
            xtol: float = 1e-8,
            **kwargs) -> dict:

        """
        Proximal gradient descent on model starting from initial guess

        :param x_start: initial guess
        :param step: step-size
        :param max_iterations:
        :param use_fista:
        :param stochastic_descent: either select random subset of samples to use at each time step or use
          average of all samples at each time-step
        :param nmax_stochastic_descent: maximum size of random subset
        :param verbose: print iteration info
        :param compute_cost: optionally compute and store the cost. This can make optimization slower
        :param line_search: use line search to shrink step-size as necessary
        :param line_search_factor: factor to shrink step-size if line-search determines step too large
        :param xtol: TODO: stop when change in x is small
        :return results: dictionary containing results
        """

        use_gpu = isinstance(x_start, cp.ndarray) and _gpu_available
        if use_gpu:
            xp = cp
        else:
            xp = np

        if nmax_stochastic_descent is None or nmax_stochastic_descent > self.n_samples:
            nmax_stochastic_descent = self.n_samples

        # ###################################
        # initialize
        # ###################################
        results = {"n_samples": self.n_samples,
                   "step_size": step,
                   "niterations": max_iterations,
                   "use_fista": use_fista,
                   "use_gpu": use_gpu,
                   "x_init": _to_cpu(xp.array(x_start, copy=True)),
                   "prox_parameters": self.prox_parameters,
                   }

        timing = {"iteration": np.zeros(0),
                  "grad": np.zeros(0),
                  "prox":  np.zeros(0),
                  "update":  np.zeros(0),
                  "cost":  np.zeros(0),
                  }

        tstart = time.perf_counter()
        costs = np.zeros((max_iterations + 1, self.n_samples)) * np.nan
        steps = np.ones(max_iterations) * step
        line_search_iters = np.ones(max_iterations, dtype=int)
        q_last = 1
        x = xp.array(x_start, copy=True)

        for ii in range(max_iterations):
            # select which subsets of views/angles to use
            if stochastic_descent:
                # select random subset of angles
                num = random.sample(range(1, nmax_stochastic_descent + 1), 1)[0]
                inds = random.sample(range(self.n_samples), num)
            else:
                # use all angles
                inds = list(range(self.n_samples))

            # ###################################
            # compute cost
            # ###################################
            tstart_err = time.perf_counter()

            if compute_cost:
                if compute_all_costs:
                    costs[ii] = _to_cpu(self.cost(x))
                else:
                    costs[ii, inds] = _to_cpu(self.cost(x, inds=inds))

            timing["cost"] = np.concatenate((timing["cost"], np.array([time.perf_counter() - tstart_err])))

            # ###################################
            # proximal gradient descent
            # ###################################
            tstart_grad = time.perf_counter()

            liters = 0
            if not line_search:
                # ###################################
                # compute gradient
                # ###################################
                x -= steps[ii] * xp.mean(self.gradient(x, inds=inds), axis=0)

                timing["grad"] = np.concatenate((timing["grad"], np.array([time.perf_counter() - tstart_grad])))

                # ###################################
                # prox operator
                # ###################################
                tstart_prox = time.perf_counter()
                y = self.prox(x)

                timing["prox"] = np.concatenate((timing["prox"], np.array([time.perf_counter() - tstart_prox])))

            else:
                # gradient and cost at current point
                gx = xp.mean(self.gradient(x, inds=inds), axis=0)
                cx = xp.mean(self.cost(x, inds=inds), axis=0)

                # grab costs, since computing anyways
                costs[ii, inds] = _to_cpu(cx)
                timing["grad"] = np.concatenate((timing["grad"], np.array([time.perf_counter() - tstart_grad])))

                tstart_prox = time.perf_counter()

                # initialize line-search
                liters += 1
                if ii != 0:
                    steps[ii] = steps[ii - 1]
                y = self.prox(x - steps[ii] * gx)

                def lipschitz_condition_violated(y, cx, gx):
                    cy = xp.mean(self.cost(y, inds=inds), axis=0)
                    return cy > cx + xp.sum(gx.real * (y - x).real + gx.imag * (y - x).imag) + 0.5 / steps[ii] * xp.linalg.norm(y - x)**2

                # iterate ... at each point check if we violate Lipschitz continuous gradient condition
                while lipschitz_condition_violated(y, cx, gx):
                    steps[ii] *= line_search_factor
                    y = self.prox(x - steps[ii] * gx)
                    liters += 1

                # not exclusively prox ... but good enough for now
                timing["prox"] = np.concatenate((timing["prox"], np.array([time.perf_counter() - tstart_prox])))

            line_search_iters[ii] = liters

            # ###################################
            # update step
            # ###################################
            tstart_update = time.perf_counter()

            q_now = 0.5 * (1 + np.sqrt(1 + 4 * q_last ** 2))
            if ii == 0 or ii == (max_iterations - 1) or not use_fista:
                x = y
            else:
                x = y + (q_last - 1) / q_now * (y - y_last)

            # update for next gradient-descent/FISTA iteration
            q_last = q_now
            # todo: do I need to make a copy?
            y_last = y

            timing["update"] = np.concatenate((timing["update"], np.array([time.perf_counter() - tstart_update])))
            timing["iteration"] = np.concatenate((timing["iteration"], np.array([time.perf_counter() - tstart_err])))

            # print information
            if verbose:
                print(
                    f"iteration {ii + 1:d}/{max_iterations:d},"
                    f" cost={np.nanmean(costs[ii]):.3g},"
                    f" step={steps[ii]:.3g},"
                    f" line search iters={line_search_iters[ii]:d},"
                    f" grad={timing['grad'][ii]:.3f}s,"
                    f" prox={timing['prox'][ii]:.3f}s,"                                        
                    f" cost={timing['cost'][ii]:.3f}s,"
                    f" iter={timing['iteration'][ii]:.3f}s,"
                    f" total={time.perf_counter() - tstart:.3f}s",
                    end="\r")

        # compute final cost
        if compute_cost:
            if compute_all_costs:
                costs[ii + 1] = _to_cpu(self.cost(x))
            else:
                costs[ii + 1, inds] = _to_cpu(self.cost(x, inds=inds))

        # store results
        results.update({"timing": timing,
                        "costs": costs,
                        "steps": steps,
                        "line_search_iterations": line_search_iters,
                        "x": x})

        return results


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
                 tau_tv_real: float = 0,
                 tau_tv_imag: float = 0,
                 tau_l1_real: float = 0,
                 tau_l1_imag: float = 0,
                 use_imaginary_constraint: bool = False,
                 use_real_constraint: bool = False,
                 max_imaginary_part: float = np.inf,
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
        :param use_real_constraint:  enforce re(n) > no
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

    def prox(self, x):
        if isinstance(x, cp.ndarray) and _gpu_available:
            denoise_tv = denoise_tv_chambolle_gpu
        else:
            denoise_tv = denoise_tv_chambolle

        # ###########################
        # TV proximal operators
        # ###########################

        # note cucim TV implementation requires ~10x memory as array does
        if self.prox_parameters["tau_tv_real"] != 0:
            x_real = denoise_tv(x.real,
                                weight=self.prox_parameters["tau_tv_real"],
                                channel_axis=None)
        else:
            x_real = x.real

        if self.prox_parameters["tau_tv_imag"] != 0:
            x_imag = denoise_tv(x.imag,
                                weight=self.prox_parameters["tau_tv_imag"],
                                channel_axis=None)
        else:
            x_imag = x.imag

        # ###########################
        # L1 proximal operators (softmax)
        # ###########################
        if self.prox_parameters["tau_l1_real"] != 0:
            x_real = soft_threshold(self.prox_parameters["tau_l1_real"], x_real - self.no) + self.no

        if self.prox_parameters["tau_l1_imag"] != 0:
            x_imag = soft_threshold(self.prox_parameters["tau_l1_imag"], x_imag)

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

        :param eft: Fourier-transform of the electric field. This will be either the scattered field or the
          Rytov phase depending on the linear model chosen.
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

        if isinstance(self.e_measured, cp.ndarray) and _gpu_available:
            self.model = sp_gpu.csr_matrix(model)
        else:
            self.model = model

    def gradient(self, x, inds=None):

        if isinstance(self.model, sp_gpu.csr_matrix) and _gpu_available:
            spnow = sp_gpu
            xp = cp
        else:
            spnow = sp
            xp = np

        ny, nx = self.e_measured.shape[-2:]
        if inds is None:
            inds = list(range(self.n_samples))

        models = [self.model[slice(ny*nx*ii, ny*nx*(ii + 1)), :] for ii in inds]
        nind = len(inds)

        # first division is average
        # second division converts Fourier space to real-space sum
        # factor of 0.5 in cost function killed by derivative factor of 2
        efwd = spnow.vstack(models).dot(x.ravel()).reshape([nind, ny, nx])
        dc_dm = (efwd - self.e_measured[inds]) / (ny * nx) / (ny * nx)

        dc_dv = xp.stack([((dc_dm[ii].conj()).ravel()[None, :] * m.tocsc()).conj().reshape(x.shape) for ii, m in enumerate(models)], axis=0)

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

        # return (abs(model.dot(x.ravel()) - self.e_measured[inds].ravel()) ** 2).reshape([ninds, ny, nx]).mean(axis=(-1, -2)) / 2 / (nx * ny)
        return 0.5 * (abs(efwd - self.e_measured[inds]) ** 2).mean(axis=(-1, -2)) / (nx * ny)

    def guess_step(self, x=None):
        ny, nx = self.e_measured.shape[-2:]

        if isinstance(self.model, sp_gpu.csr_matrix):
            u, s, vh = sp.linalg.svds(self.model.get(), k=1, which='LM')
        else:
            u, s, vh = sp.linalg.svds(self.model, k=1, which='LM')

        lipschitz_estimate = s ** 2 / (self.n_samples * ny * nx) / (ny * nx)
        return float(1 / lipschitz_estimate)

    def prox(self, x):
        # convert from V to n
        n = get_n(_ift3(x), self.no, self.wavelength)
        # apply proximal operator on n
        n_prox = super(LinearScatt, self).prox(n)

        return _ft3(get_v(n_prox, self.no, self.wavelength))

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
                 mask: array = None,
                 use_modified_bpm: bool = True,
                 **kwargs
                 ):
        """
        Suppose we have a 3D grid with nz voxels along the propagation direction. We define the electric field
        at the points before and after each voxel, and in an additional plane to account for the imaging. So we have
        nz + 2 electric field planes.

        :param e_measured: measured electric fields
        :param e_measured_bg: measured background electric fields
        :param beam_frqs:
        :param no: background index of refraction
        :param wavelength: wavelength of light
        :param drs_e: (dy, dx) of the electric field
        :param drs_n: (dz, dy, dx) of the refractive index
        :param shape_n: (nz, ny, nx)
        :param dz_final: distance to propagate field after last surface
        :param atf: coherent transfer function
        :param apodization: apodization used during FFTs
        :param use_modified_bpm: allow use of modified BPM with extra cosine obliquity factor. Note that this
          option will only be used if there is no multiplexing of pattern angle. The degree of multiplexing
          is determined by looking at the beam_frqs
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
                                  **kwargs)

        self.mask = mask

        # include cosine oblique factor
        self.use_modified_bpm = use_modified_bpm

        # cosines
        if self.beam_frqs is not None and self.use_modified_bpm:
            self.thetas, _ = frqs2angles(self.beam_frqs, self.no, self.wavelength)
        else:
            self.thetas = np.zeros((self.n_samples,))

        # starting field
        nz = self.shape_n[0]
        self.dz_back = -np.array([float(self.dz_final) + float(self.drs_n[0]) * nz])

    def gradient(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        e_start = field_prop.propagate_homogeneous(self.e_measured_bg[inds],
                                                   self.dz_back,
                                                   self.no,
                                                   self.drs_n[1:],
                                                   self.wavelength)[..., 0, :, :]

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

        # back propagation ... build the gradient from this to save memory
        dc_dn = field_prop.backpropagate_bpm(e_fwd[:, -1, :, :] - self.e_measured[inds],
                                              x,
                                              self.no,
                                              self.drs_n,
                                              self.wavelength,
                                              self.dz_final,
                                              atf=self.atf,
                                              apodization=self.apodization,
                                              thetas=self.thetas[inds])[:, 1:-1, :, :]

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
        dc_dn *= -1j * (2 * np.pi / self.wavelength) * dz / xp.cos(thetas) / (nx * ny)
        dc_dn *= e_fwd[:, 1:-1, :, :].conj()

        return dc_dn

    def cost(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        e_start = field_prop.propagate_homogeneous(self.e_measured_bg[inds],
                                                   self.dz_back,
                                                   self.no,
                                                   self.drs_n[1:],
                                                   self.wavelength)[..., 0, :, :]

        e_fwd = field_prop.propagate_bpm(e_start,
                                         x,
                                         self.no,
                                         self.drs_n,
                                         self.wavelength,
                                         self.dz_final,
                                         atf=self.atf,
                                         apodization=self.apodization,
                                         thetas=self.thetas[inds])

        costs = 0.5 * (abs(e_fwd[:, -1, :, :] - self.e_measured[inds]) ** 2).mean(axis=(-1, -2))

        return costs


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
                                   **kwargs)

        self.mask = mask

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

        kz = xp.asarray(2 * np.pi * field_prop.get_fzs(fxfx, fyfy, self.no, self.wavelength))
        kz[xp.isnan(kz)] = 0
        self.kz = kz

    def gradient(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        # initial field
        e_start = field_prop.propagate_homogeneous(self.e_measured_bg[inds],
                                                   self.dz_back,
                                                   self.no,
                                                   self.drs_n[1:],
                                                   self.wavelength)[..., 0, :, :]
        # assume initial field is foward propagating only
        de_dz_start = _ift2(1j * self.kz * _ft2(e_start))

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

        # back propagation
        # this is the backpropagated field, but we will eventually transform it into the gradient
        # do things this way to reduce memory overhead
        dc_dn = field_prop.backpropagate_ssnp(phi_fwd[inds, -1, :, :, 0] - self.e_measured[inds],
                                              x,
                                              self.no,
                                              self.drs_n,
                                              self.wavelength,
                                              self.dz_final,
                                              atf=self.atf,
                                              apodization=self.apodization)[:, 1:-1, :, :, 1]

        # cost function gradient
        ny, nx = x.shape[-2:]
        dz = self.drs_n[0]
        ko = 2 * np.pi / self.wavelength
        # from phi_back, take derivative part
        # from phi_fwd, take the electric field part
        dc_dn *= (-2 * ko ** 2 * dz * x.conj()) / nx / ny
        dc_dn *= phi_fwd[:, :-2, :, :, 0].conj()

        return dc_dn


    def cost(self, x, inds=None):
        if inds is None:
            inds = list(range(self.n_samples))

        # initial field
        e_start = field_prop.propagate_homogeneous(self.e_measured_bg[inds],
                                                   self.dz_back,
                                                   self.no,
                                                   self.drs_n[1:],
                                                   self.wavelength)[..., 0, :, :]
        # assume initial field is foward propagating only
        de_dz_start = _ift2(1j * self.kz * _ft2(e_start))

        e_fwd = field_prop.propagate_ssnp(e_start,
                                          de_dz_start,
                                          x,
                                          self.no,
                                          self.drs_n,
                                          self.wavelength,
                                          self.dz_final,
                                          atf=self.atf,
                                          apodization=self.apodization)[..., 0]

        costs = 0.5 * (abs(e_fwd[:, -1, :, :] - self.e_measured[inds]) ** 2).mean(axis=(-1, -2))

        return costs
