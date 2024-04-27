"""
Tools for reconstructing optical diffraction tomography (ODT) data using either the Born approximation,
Rytov approximation, or multislice (paraxial) beam propagation method (BPM) and a FISTA solver. The primary
reconstruction tasks are carried out with the tomography class
"""
from time import perf_counter
import datetime
from warnings import catch_warnings, simplefilter
from copy import deepcopy
from typing import Union, Optional
from collections.abc import Sequence
from inspect import getfullargspec
# numerical tools
import numpy as np
from numpy.linalg import norm, inv
from numpy.fft import fftshift, fftfreq
from scipy.signal.windows import tukey, hann
# parallelization
from dask.config import set as dask_cfg_set
from dask import delayed
from dask import compute as dcompute
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
# plotting
import matplotlib.pyplot as plt
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Arc, Rectangle
# saving/loading
from pathlib import Path
import zarr
from numcodecs import Zlib, packbits
from numcodecs.abc import Codec
# custom tools
from localize_psf.camera import bin
from localize_psf.fit import fit_model, gauss2d_symm
from localize_psf.rois import get_centered_rois, cut_roi
from localize_psf.affine import (params2xform,
                                 xform2params,
                                 fit_xform_points_ransac,
                                 fit_xform_points,
                                 xform_points,
                                 xform_mat)
from mcsim.analysis.phase_unwrap import phase_unwrap as weighted_phase_unwrap
from mcsim.analysis.optimize import to_cpu
from mcsim.analysis.fft import ft3, ift3, ft2, ift2, translate_ft
from mcsim.analysis.field_prop import (get_n,
                                       get_v,
                                       get_fzs,
                                       fwd_model_linear,
                                       inverse_model_linear,
                                       LinearScatt,
                                       BPM,
                                       SSNP)

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import pygpufit.gpufit as gf
except ImportError:
    gf = None

# CPU/GPU arrays
if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


class Tomography:

    models = {"born": LinearScatt,
              "rytov": LinearScatt,
              "bpm": BPM,
              "ssnp": SSNP
              }

    store_name = "refractive_index.zarr"

    def __init__(self,
                 imgs_raw: da.array,
                 wavelength: float,
                 no: float,
                 na_detection: float,
                 na_excitation: float,
                 dxy: float,
                 drs_n: Sequence[float, float, float],
                 n_shape: Sequence[int, int, int],
                 model: str = "rytov",
                 scattered_field_regularization: float = 50.,
                 use_weighted_phase_unwrap: bool = False,
                 n_guess: Optional[array] = None,
                 use_fixed_holo_frequencies: bool = False,
                 hologram_frqs_guess: Optional[Sequence[np.ndarray]] = None,
                 reference_frq: Optional[np.ndarray] = None,
                 freq_roi_size_pix: int = 11,
                 imgs_raw_bg: Optional[da.array] = None,
                 phase_offsets: Optional[np.ndarray] = None,
                 axes_names: Optional[Sequence[str]] = None,
                 verbose: bool = True,
                 save_dir: Optional[Union[Path, str]] = None,
                 realspace_mask: Optional[array] = None,
                 cam_roi: Optional[Sequence[int, int, int, int]] = None,
                 data_roi: Optional[Sequence[int, int, int, int]] = None,
                 gain: float = 1.,
                 offset: float = 0.,
                 bg_average_axes: Optional[tuple[int]] = None,
                 fit_phases: bool = False,
                 fit_translations: bool = False,
                 translation_thresh: float = 1 / 30,
                 apodization: Optional[np.ndarray] = None,
                 save_auxiliary_fields: bool = False,
                 ):
        """
        Reconstruct optical diffraction tomography (ODT) data

        :param imgs_raw: Data intensity images with shape n0 x ... x nm x npatterns x ny x nx
        :param wavelength: wavelength in um
        :param no: background index of refraction
        :param na_detection: numerical aperture of the detection objective
        :param na_excitation: numerical aperture for the maximum excitation pattern
        :param dxy: pixel size in um
        :param drs_n: (dz, dy, dx) voxel size of reconstructed refractive index
        :param n_shape: (nz, ny, nx) shape of reconstructed refractive index # todo: make n_shape
        :param model: "born", "rytov", "bpm", or "ssnp"
        :param scattered_field_regularization: regularization used in computing scattered field
          or Rytov phase. Regions where background electric field is smaller than this value will be suppressed
        :param use_weighted_phase_unwrap: whether to use a weighted or unweighted phase unwrapping algorithm when
          computing the Rytov phase. This may not be used, depending on the model chosen and if a value for
          n_guess is provided
        :param n_guess:
        :param use_fixed_holo_frequencies: determine single set of frequencies for all data/background images
        :param hologram_frqs_guess: npatterns x nmulti x 2 array
        :param reference_frq: [fx, fy] hologram reference frequency with shape n0 x ... x nm x 2
        :param freq_roi_size_pix: ROI size (in pixels) to use for frequency fitting
        :param imgs_raw_bg: background intensity images. If no background images are provided, then a time
          average of imgs_raw will be used as the background
        :param phase_offsets: phase shifts between images and corresponding background images
        :param axes_names: names of first m + 1 axes
        :param verbose: whether to print progress information
        :param save_dir:
        :param realspace_mask: indicate which parts of image to exclude/keep
        :param cam_roi: ROI camera chip has been cropped to wrt to the entire camera chip
        :param data_roi: ROI the data has been cropped to after camera acquistion
        :param gain:
        :param offset:
        :param bg_average_axes: axes to average along when producing background images
        :param fit_phases: whether to fit phase differences between image and background holograms
        :param fit_translations: whether to fit the spatial translation between the data and reference images
        :param translation_thresh:
        :param apodization: if None use tukey apodization with alpha = 0.1. To use no apodization set equal to 1
        :param save_auxiliary_fields:
        """
        self.verbose = verbose
        self.tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

        # ########################
        # save directory and zarr backing
        # ########################
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True)

            self.store = zarr.open(self.save_dir / self.store_name, "a")
        else:
            self.save_dir = None
            self.store = zarr.open()

        # ########################
        # physical parameters
        # ########################
        self.wavelength = wavelength
        self.no = no
        self.na_detection = na_detection
        self.na_excitation = na_excitation
        self.fmax = self.na_detection / self.wavelength
        self.gain = gain
        self.offset = offset

        # ########################
        # RI data
        # ########################
        self.drs_n = drs_n
        self.n_shape = n_shape
        self.n_guess = n_guess
        self.reconstruction_settings = None

        if model not in self.models.keys():
            raise ValueError(f"model must be one of {self.models.keys()}, but was {model:s}")
        self.model = model

        # ########################
        # ROI and transformation info
        # ########################
        self.xform_dict = None
        self.data_roi = data_roi
        self.cam_roi = cam_roi

        # ########################
        # images
        # ########################
        # convert to photons
        # todo: ensure that are dask arrays?
        self.imgs_raw = (imgs_raw.astype(float) - offset) / gain
        self.imgs_raw_bg = None
        if imgs_raw_bg is not None:
            self.imgs_raw_bg = (imgs_raw_bg.astype(float) - offset) / gain

        self.npatterns, self.ny, self.nx = imgs_raw.shape[-3:]
        self.nextra_dims = imgs_raw.ndim - 3

        # ########################
        # information about nd-array axes
        # ########################
        if axes_names is None:
            self.dimensions = [f"i{ii:d}" for ii in range(self.imgs_raw.ndim - 2)]
        else:
            self.dimensions = axes_names

        self.use_average_as_background = self.imgs_raw_bg is None

        if bg_average_axes is None:
            bg_average_axes = ()
        self.bg_average_axes = bg_average_axes

        # ########################
        # beam frequencies
        # ########################
        # reference frequency
        # shape = n0 x ... x nm x 2 where
        if reference_frq is None:
            self.reference_frq = None
            self.use_fixed_ref = False
        else:
            self.reference_frq = np.array(reference_frq) + np.zeros(self.imgs_raw.shape[:-3] + (2,))
            self.use_fixed_ref = True
        self.reference_frq_bg = self.reference_frq

        # todo: don't allow different multiplex for different patterns
        # list of length npatterns, with each entry [1 x 1 ... x 1 x nmultiplex x 2]
        # with potential different nmultiplex for each pattern
        self.freq_roi_size_pix = freq_roi_size_pix
        self.use_fixed_holo_frequencies = use_fixed_holo_frequencies
        self.hologram_frqs = hologram_frqs_guess

        if self.hologram_frqs is None:
            self.nmax_multiplex = 1
        else:
            self.nmax_multiplex = np.max([f.shape[0] for f in self.hologram_frqs])

        self.hologram_frqs_bg = None

        # ########################
        # correction parameters
        # ########################
        # self.phase_offsets = phase_offsets
        self.fit_phases = fit_phases
        self.fit_translations = fit_translations
        self.translation_thresh = translation_thresh
        self.translations = np.zeros((1,) * self.nextra_dims + (self.npatterns, 1, 1, 2), dtype=float)
        self.translations_bg = np.zeros_like(self.translations)
        self.phase_params = np.ones((1,) * self.nextra_dims + (self.npatterns, 1, 1), dtype=complex)
        self.phase_params_bg = np.ones_like(self.phase_params)

        self.save_auxiliary_fields = save_auxiliary_fields

        # electric fields
        self.efields_ft = None
        self.efield_bg_ft = None

        # scattered field settings
        self.scattered_field_regularization = scattered_field_regularization
        self.use_weighted_phase_unwrap = use_weighted_phase_unwrap

        # ########################
        # coordinates
        # ########################
        self.dxy = dxy
        self.x = (np.arange(self.nx) - (self.nx // 2)) * dxy
        self.y = (np.arange(self.ny) - (self.ny // 2)) * dxy
        self.fxs = fftshift(fftfreq(self.nx, self.dxy))
        self.fys = fftshift(fftfreq(self.ny, self.dxy))
        self.dfx = self.fxs[1] - self.fxs[0]
        self.dfy = self.fys[1] - self.fys[0]

        # ########################
        # other arrays
        # ########################
        self.realspace_mask = realspace_mask
        if apodization is None:
            apodization = np.outer(tukey(self.ny, alpha=0.1),
                                   tukey(self.nx, alpha=0.1))
        self.apodization = apodization

        self.ctf = np.expand_dims(np.sqrt(self.fxs[None, :]**2 +
                                          self.fys[:, None]**2) <= self.fmax,
                                  axis=tuple(range(self.nextra_dims)) + (-3,))

    @classmethod
    def load_file(cls,
                  location: Union[str, Path, zarr.hierarchy.Group]):
        """
        Instantiated class from zarr store

        :param location:
        :return:
        """

        if isinstance(location, (Path, str)):
            z = zarr.open(location, "r")
        else:
            z = location

        spec = getfullargspec(cls.__init__)

        # ###########################
        # get arguments for construction
        # ###########################
        # get any kwargs from zarr attributes
        kwargs = {a: z.attrs[a] if a in z.attrs.keys() else None
                  for a in spec.args[1:]}

        # get any other kwargs from arrays
        for k, v in kwargs.items():
            if v is None and k in z.array_keys():
                kwargs[k] = np.array(z[k])

        # correct any attributes that were not saved
        kwargs["imgs_raw"] = da.zeros((1,) * z.attrs["nextra_dims"] +
                                      (z.attrs["npatterns"], z.attrs["ny"], z.attrs["nx"]))


        # ###########################
        # instantiate
        # ###########################
        inst = cls(**kwargs)

        # ###########################
        # load other attributes which are not set during __init__()
        # ###########################
        dask_fields = ["efields_ft", "efield_bg_ft"]
        for name, arr in z.arrays():
            if hasattr(inst, name) and getattr(inst, name) is None:
                if name in dask_fields:
                    load_arr = da.from_zarr(arr)
                else:
                    load_arr = np.array(arr)
                setattr(inst, name, load_arr)

        for k, v in z.attrs.items():
            if hasattr(inst, k) and getattr(inst, k) is None:
                setattr(inst, k, v)

        return inst

    @staticmethod
    def prepare_img_data(data_dir: Union[Path, str],
                         zarr_pattern: str,
                         component: str,
                         roi: Optional[Sequence[int]] = None,
                         slice_axis: Optional[int] = None,
                         n_every: int = 1,
                         bg_axis: Optional[int] = None,
                         bg_index: Optional[int] = None
                         ):
        """

        :param data_dir: directory to search for raw data
        :param zarr_pattern: file pattern passed to glob when searching for zarr array
        :param component: zarr component storing image data
        :param roi: [ystart, yend, xstart, xend]
        :param slice_axis: take a limited amount of data from this axis
        :param n_every: take images from slice_axis with this spacing
        :param bg_axis: take background image from this axis
        :param bg_index: use this index along bg_axis to get image
        :return imgs, imgs_bg:
        """

        fname_raw = list(data_dir.glob(zarr_pattern))[0]

        # load data
        zraw = zarr.open(fname_raw, "r")
        zc = zraw[component]

        # define data slice to reconstruct
        slices = [slice(0, None, n_every) if ax == slice_axis
                  else slice(None)
                  for ax in range(zc.ndim)]
        if roi is not None:
            slices[-2] = slice(roi[0], roi[1])
            slices[-1] = slice(roi[2], roi[3])
        slices = tuple(slices)

        # load and crop images
        imgs = da.from_zarr(zc, chunks=(1,) * (zc.ndim - 3) + zc.shape[-3:])[slices]

        # if background is a position slice of imgs use slice so preserve shape of array
        if bg_axis is not None:
            bg_axis = bg_axis % zc.ndim
            bg_index = bg_index % zc.shape[bg_axis]

            fg_slice1 = tuple([slice(0, bg_index) if ax == bg_axis
                               else slice(None)
                               for ax in range(zc.ndim)]
                              )
            fg_slice2 = tuple([slice(bg_index + 1, None) if ax == bg_axis
                               else slice(None)
                               for ax in range(zc.ndim)]
                              )

            bg_slice = tuple([slice(bg_index, bg_index + 1) if ax == bg_axis
                              else slice(None)
                              for ax in range(zc.ndim)]
                             )

            # split into foreground and background
            imgs_bg = imgs[bg_slice]
            imgs = da.concatenate((imgs[fg_slice1], imgs[fg_slice2]), axis=bg_axis)
        else:
            imgs_bg = None

        return zraw, zc, imgs, imgs_bg

    def save(self,
             attributes: Optional[dict] = None,
             compressor: Codec = Zlib()):
        """
        Save information from instance to zarr

        :param attributes:
        :param compressor:
        :return:
        """

        if attributes is None:
            attributes = {}

        # handle lists of arrays
        stackable_keys = ["hologram_frqs",
                          "hologram_frqs_bg"
                          ]
        for k in stackable_keys:
            self.store.array(k,
                             np.stack(getattr(self, k), axis=-3),
                             compressor=compressor,
                             dtype=float)

        self.store.array("beam_frqs",
                         np.stack(self.get_beam_frqs(), axis=-3),
                         compressor=compressor,
                         dtype=float)

        # save everything else
        for dictionary in [self.__dict__, attributes]:
            for k, v in dictionary.items():
                if k in stackable_keys:
                    continue
                try:
                    if isinstance(v, array):
                        if v.size > 10:
                            if v.dtype == bool:
                                c = packbits.PackBits()
                            else:
                                c = compressor

                            # todo: ensure reasonable chunk size
                            self.store.array(k,
                                             to_cpu(v),
                                             compressor=c,
                                             dtype=v.dtype)
                        else:
                            self.store.attrs[k] = to_cpu(v).tolist()

                    elif isinstance(v, (int, float, str, bool, dict, tuple)) or v is None:
                        self.store.attrs[k] = v
                    elif isinstance(v, Path):
                        self.store.attrs[k] = str(v)
                    elif isinstance(v, list):
                        if len(v) > 30:  # don't store long lists
                            raise TypeError()
                        self.store.attrs[k] = v

                except TypeError as e:
                    print(f"{k:s} {e}")

    def save_projections(self,
                         compressor: Codec = Zlib()):
        """
        Store orthogonal projections of the refractive index

        :param compressor:
        :return:
        """

        future = []
        for axis, label in zip([-3, -2, -1],
                               ["z", "y", "x"]):
            future.append(da.max(da.real(da.from_zarr(self.store.n)), axis=axis).to_zarr(self.store.store.path,
                                                                                         component=f"n_max{label:s}",
                                                                                         compute=False,
                                                                                         compressor=compressor)
                          )
        with ProgressBar():
            dcompute(*future)

    def estimate_hologram_frqs(self,
                               save: bool = False,
                               fit_on_gpu: bool = False,
                               processes: bool = True,
                               n_workers: int = 3,
                               threads_per_worker: int = 1,
                               worker_saturation: float = 1.0) -> None:
        """
        Estimate hologram frequencies from raw images.
        Guess values need to be within a few pixels for this to succeed. Can easily achieve this accuracy by
        looking at FFT

        :param save:
        :param fit_on_gpu: do fitting on GPU with gpufit. Otherwise, use CPU.
        :param processes:
        :param n_workers:
        :param threads_per_worker:
        :param worker_saturation:
        :return:
        """

        # https://distributed.dask.org/en/latest/scheduling-policies.html#adjusting-or-disabling-queuing
        dask_cfg_set({"distributed.scheduler.worker-saturation": worker_saturation})
        cluster = LocalCluster(processes=processes,
                               n_workers=n_workers,
                               threads_per_worker=threads_per_worker)
        client = Client(cluster)
        if self.verbose:
            print(cluster.dashboard_link)

        # fitting logic
        # todo: re-implement!
        if not self.use_fixed_holo_frequencies:
            slices_bg = tuple([slice(None) for _ in range(self.nextra_dims)])
        else:
            slices_bg = tuple([slice(0, 1) for _ in range(self.nextra_dims)])

        fit_data_imgs = not self.use_fixed_holo_frequencies or self.use_average_as_background
        fit_bg_imgs = not self.use_average_as_background

        # create array for ref frqs
        # NOTE only works for equal multiplex for all images!
        hologram_frqs_guess = np.stack(self.hologram_frqs, axis=0)

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

        rois_all = get_centered_rois(c_guess,
                                     [self.freq_roi_size_pix,
                                      self.freq_roi_size_pix],
                                     min_vals=(0, 0),
                                     max_vals=(self.ny, self.nx)
                                     )

        xx, yy = np.meshgrid(range(self.freq_roi_size_pix),
                             range(self.freq_roi_size_pix))

        # cut rois
        def cut_rois(img: array,
                     roi_size_pix: int,
                     block_id=None,):
            img_ft = ft2(img * apodization)

            npatt, ny, nx = img_ft.shape[-3:]
            nroi = rois_all.shape[1]
            roi_out = np.zeros(img_ft.shape[:-3] + (npatt, nroi, roi_size_pix, roi_size_pix))
            for ii in range(npatt):
                roi_out[..., ii, :, :, :] = abs(np.stack(cut_roi(rois_all[ii],
                                                                 img_ft[..., ii, :, :]), axis=-3))

            return roi_out

        rois_cut = None
        if fit_data_imgs:
            rois_cut = da.map_blocks(cut_rois,
                                     self.imgs_raw,
                                     self.freq_roi_size_pix,
                                     drop_axis=(-1, -2),
                                     new_axis=(-1, -2, -3),
                                     chunks=self.imgs_raw.chunksize[:-2] +
                                            (self.nmax_multiplex,
                                             self.freq_roi_size_pix,
                                             self.freq_roi_size_pix),
                                     dtype=float,
                                     meta=np.array((), dtype=float))

        rois_cut_bg = None
        if fit_bg_imgs:
            rois_cut_bg = da.map_blocks(cut_rois,
                                        self.imgs_raw_bg,
                                        self.freq_roi_size_pix,
                                        drop_axis=(-1, -2),
                                        new_axis=(-1, -2, -3),
                                        chunks=self.imgs_raw_bg.chunksize[:-2] +
                                               (self.nmax_multiplex,
                                                self.freq_roi_size_pix,
                                                self.freq_roi_size_pix),
                                        dtype=float,
                                        meta=np.array((), dtype=float))[slices_bg]

        def fit_rois_cpu(img_rois,
                         model=gauss2d_symm(),
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

        def fit_rois_gpu(rois_cut,
                         roi_size_pix: int):
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
            if fit_data_imgs:
                cx, cy, fit_params, init_params, fit_states = fit_rois_gpu(rois_cut,
                                                                           self.freq_roi_size_pix)

            if fit_bg_imgs:
                cx_bg, cy_bg, _, _, _ = fit_rois_gpu(rois_cut_bg,
                                                     self.freq_roi_size_pix)

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

        if self.use_fixed_holo_frequencies:
            frqs_hologram = frqs_hologram_bg

        self.hologram_frqs = [frqs_hologram[..., ii, :, :] for ii in range(self.npatterns)]
        self.hologram_frqs_bg = [frqs_hologram_bg[..., ii, :, :] for ii in range(self.npatterns)]
        if not self.use_fixed_ref:
            self.reference_frq = np.mean(np.concatenate(self.hologram_frqs, axis=-2), axis=-2)
            self.reference_frq_bg = np.mean(np.concatenate(self.hologram_frqs_bg, axis=-2), axis=-2)

        # #########################
        # optionally plot
        # #########################
        def plot(img,
                 frqs_holo,
                 frqs_guess=None,
                 rois_all=None,
                 save_dir=None,
                 prefix="",
                 figsize=(20, 10),
                 block_id=None):

            img_ft = ft2(img).squeeze()
            if img_ft.ndim != 2:
                raise ValueError()

            with catch_warnings():
                simplefilter("ignore")
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
            iroi = cut_roi(roi, img_ft)[0]

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

            if save_dir is not None:
                figh.savefig(Path(save_dir, f"{prefix:s}=hologram_frq_diagnostic.png"))
                plt.close(figh)

        slice_start = tuple([slice(0, 1) for _ in range(self.nextra_dims)])
        axes_start = tuple(range(self.nextra_dims))

        if save and self.save_dir is not None:
            frq_dir = self.save_dir / f"frq_fits"
            frq_dir.mkdir(exist_ok=True)

            with rc_context({'interactive': False,
                             'backend': "agg"}):

                if fit_data_imgs:
                    iraw = self.imgs_raw[slice_start].squeeze(axis=axes_start)
                    h = [f[slice_start].squeeze(axis=axes_start) for f in self.hologram_frqs]

                    dcompute(*[delayed(plot)(iraw[ii],
                                           h[ii],
                                           save_dir=frq_dir,
                                           frqs_guess=hologram_frqs_guess[ii],
                                           prefix=f"{ii:d}",
                                           rois_all=rois_all[ii])
                               for ii in range(self.npatterns)]
                             )

                if fit_bg_imgs:
                    iraw_bg = self.imgs_raw_bg[slice_start].squeeze(axis=axes_start)
                    hbg = [f[slice_start].squeeze(axis=axes_start) for f in self.hologram_frqs_bg]

                    dcompute([delayed(plot)(iraw_bg[ii],
                                           hbg[ii],
                                           save_dir=frq_dir,
                                           frqs_guess=hologram_frqs_guess[ii],
                                           prefix=f"{ii:d}",
                                           rois_all=rois_all[ii])
                              for ii in range(self.npatterns)]
                             )

            client.profile(filename=frq_dir / f"frequency_fitting_dask_profile.html")

        client.close()
        cluster.close()
        del cluster, client

    def get_beam_frqs(self) -> list[array]:
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
                                  dmd_size: Optional[Sequence[int, int]] = None,
                                  save: bool = False,
                                  interactive: bool = False) -> np.ndarray:
        """
        Fit affine transformation between device and measured frequency space.

        This could be between frequencies displayed on DMD and measured frequency space (DMD in imaging plane)
        or between mirror positions on DMD and frequency space (DMD in Fourier plane)

        :param offsets:
        :param dmd_size: (ny, nx)
        :param save:
        :param interactive:
        :return xform_dmd2frq:
        """

        if dmd_size is not None:
            ny_dmd, nx_dmd = dmd_size

        centers_dmd = np.concatenate(offsets, axis=0)
        mean_hologram_frqs = np.concatenate([np.mean(f, axis=tuple(range(self.nextra_dims)))
                                             for f in self.hologram_frqs], axis=0)
        # mean_hologram_frqs = np.mean(self.hologram_frqs, axis=tuple(range(self.nextra_dims)))
        # mean_hologram_frqs = np.concatenate([hf for hf in mean_hologram_frqs], axis=0)

        mean_ref_frq = np.mean(self.reference_frq, axis=tuple(range(self.nextra_dims)))

        beam_frqs = np.concatenate([np.mean(f, axis=tuple(range(self.nextra_dims)))
                                    for f in self.get_beam_frqs()], axis=0)

        # fit affine transformation
        if len(mean_hologram_frqs) > 6:
            xform_dmd2frq, _, _, _ = fit_xform_points_ransac(centers_dmd,
                                                             mean_hologram_frqs,
                                                             dist_err_max=0.1,
                                                             niterations=100)
        elif len(mean_hologram_frqs) > 3:
            # no point in RANSAC if not enough points to invert transformation
            xform_dmd2frq, _ = fit_xform_points(centers_dmd, mean_hologram_frqs)
        else:
            return None

        # # map pupil positions to frequency
        frqs_from_pupil = xform_points(centers_dmd, xform_dmd2frq)
        # estimate frequency of reference beam from affine transformation and previous calibration information
        frq_dmd_center = xform_points(np.array([[0, 0]]), xform_dmd2frq)[0]

        # also get inverse transform and map frequencies to pupil (DMD positions)
        xform_frq2dmd = inv(xform_dmd2frq)
        centers_pupil_from_frq = xform_points(mean_hologram_frqs, xform_frq2dmd)
        center_pupil_frq_ref = xform_points(mean_hologram_frqs, xform_frq2dmd)[0]
        # center_pupil_frq_ref = affine.xform_points(np.expand_dims(mean_hologram_frqs, axis=0), xform_frq2dmd)[0]

        # map maximum pupil frequency circle to DMD space
        circle_thetas = np.linspace(0, 2 * np.pi, 1001)
        frqs_pupil_boundary = self.fmax * np.stack((np.cos(circle_thetas), np.sin(circle_thetas)),
                                                     axis=1) + np.expand_dims(mean_ref_frq, axis=0)
        centers_dmd_fmax = xform_points(frqs_pupil_boundary, xform_frq2dmd)
        rmax_dmd_mirrors = np.max(norm(centers_dmd_fmax, axis=1))

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
            dmd_boundary_freq = xform_points(dmd_boundary, xform_dmd2frq)

        # check sign of frequency reference is consistent with affine transform
        assert norm(frq_dmd_center + mean_ref_frq) >= norm(frq_dmd_center - mean_ref_frq)

        # ##############################
        # plot data
        # ##############################
        context = {}
        if not interactive:
            context['interactive'] = False
            context['backend'] = "agg"

        with rc_context(context):
            xform_params = xform2params(xform_dmd2frq)

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
            ax.plot(centers_pupil_from_frq[..., 0],
                    centers_pupil_from_frq[..., 1],
                    'rx',
                    label="fit hologram frequencies")
            ax.plot(centers_dmd[:, 0],
                    centers_dmd[:, 1],
                    'b.',
                    label="mirror positions")
            ax.plot(0, 0, 'g+', label="DMD center")
            ax.plot(center_pupil_frq_ref[0],
                    center_pupil_frq_ref[1],
                    "m3",
                    label="reference freq")
            ax.plot(centers_dmd_fmax[:, 0],
                    centers_dmd_fmax[:, 1],
                    'k',
                    label="pupil")
            ax.set_xlim([-rmax_dmd_mirrors, rmax_dmd_mirrors])
            ax.set_ylim([-rmax_dmd_mirrors, rmax_dmd_mirrors])
            ax.legend(bbox_to_anchor=(0.2, 1.1))
            ax.set_xlabel("x-position (mirrors)")
            ax.set_ylabel("y-position (mirrors)")

            ax = figh.add_subplot(grid[0, 1])
            ax.axis("scaled")
            ax.set_title("Raw frequencies")

            if dmd_size is not None:
                ax.plot(dmd_boundary_freq[:, 0], dmd_boundary_freq[:, 1], 'k.')

            ax.plot(mean_hologram_frqs[..., 0], mean_hologram_frqs[..., 1], 'rx')
            ax.plot(frqs_from_pupil[..., 0], frqs_from_pupil[..., 1], 'b.')
            ax.plot(frq_dmd_center[0], frq_dmd_center[1], 'g+')
            ax.plot(mean_ref_frq[0], mean_ref_frq[1], "m3")
            ax.add_artist(Circle(mean_ref_frq,
                                 radius=self.fmax,
                                 facecolor="none",
                                 edgecolor="k"))
            ax.set_xlim([-self.fmax + mean_ref_frq[0], self.fmax + mean_ref_frq[0]])
            ax.set_ylim([-self.fmax + mean_ref_frq[1], self.fmax + mean_ref_frq[1]])
            ax.set_xlabel("$f_x$ (1/$\mu m$)")
            ax.set_ylabel("$f_y$ (1/$\mu m$)")

            ax = figh.add_subplot(grid[0, 2])
            ax.axis("scaled")
            ax.set_title("Frequencies - reference frequency")

            if dmd_size is not None:
                ax.plot(dmd_boundary_freq[:, 0] - mean_ref_frq[0],
                        dmd_boundary_freq[:, 1] - mean_ref_frq[1],
                        'k.')

            ax.plot(beam_frqs[..., 0], beam_frqs[..., 1], 'rx')
            ax.plot(frqs_from_pupil[..., 0] - mean_ref_frq[0],
                    frqs_from_pupil[..., 1] - mean_ref_frq[1],
                    'b.')
            ax.plot(frq_dmd_center[0] - mean_ref_frq[0],
                    frq_dmd_center[1] - mean_ref_frq[1],
                    'g+')
            ax.plot(0, 0, 'm3')
            ax.add_artist(Circle((0, 0),
                                 radius=self.fmax,
                                 facecolor="none",
                                 edgecolor="k"))
            ax.set_xlim([-self.fmax, self.fmax])
            ax.set_ylim([-self.fmax, self.fmax])
            ax.set_xlabel("$f_x$ (1/$\mu m$)")
            ax.set_ylabel("$f_y$ (1/$\mu m$)")

            if save and self.save_dir is not None:
                figh.savefig(self.save_dir / f"frequency_mapping.png")
                plt.close(figh)

        return xform_dmd2frq

    def unmix_holograms(self,
                        use_gpu: bool = False,
                        processes: bool = False,
                        compressor: Codec = Zlib()):
        """
        Unmix and preprocess holograms.

        :param use_gpu:
        :return:
        """

        cluster = LocalCluster(processes=processes)
        client = Client(cluster)
        if self.verbose:
            print(cluster.dashboard_link)

        # slice used as reference for computing phase shifts/translations/etc.
        # if we are going to average along a dimension (i.e. if it is in bg_average_axes) then need to use
        # single slice as background for that dimension.
        ref_slice = tuple([slice(0, 1) if a in self.bg_average_axes else slice(None)
                           for a in range(self.nextra_dims)] +
                          [slice(None)] * 3)

        # set up
        if use_gpu and cp:
            xp = cp
        else:
            xp = np

        # #########################
        # get electric field from holograms
        # #########################
        apodization = xp.asarray(self.apodization)

        # make broadcastable to same size as raw images so can use with dask array
        ref_frq_da = da.from_array(np.expand_dims(self.reference_frq, axis=(-2, -3, -4)),
                                   chunks=self.imgs_raw.chunksize[:-2] + (1, 1, 2)
                                   )
        ref_frq_bg_da = da.from_array(np.expand_dims(self.reference_frq_bg, axis=(-2, -3, -4)),
                                      chunks=self.imgs_raw_bg.chunksize[:-2] + (1, 1, 2)
                                      )

        holograms_ft = da.map_blocks(unmix_hologram,
                                     self.imgs_raw,
                                     self.dxy,
                                     2*self.fmax,
                                     ref_frq_da[..., 0],
                                     ref_frq_da[..., 1],
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
                                            apodization=apodization,
                                            dtype=complex)

        # #########################
        # fit translations between signal and background electric fields
        # #########################
        if self.fit_translations:
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
                                              thresh=self.translation_thresh,
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
            holograms_ft = da.map_blocks(translate,
                                         holograms_ft,
                                         da.from_array(self.translations[..., 0], chunks=dr_chunks),
                                         da.from_array(self.translations[..., 1], chunks=dr_chunks),
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
                                                     thresh=self.translation_thresh,
                                                     dtype=float,
                                                     new_axis=-1,
                                                     chunks=holograms_abs_ft.chunksize[:-2] + (1, 1, 2)).compute()

                holograms_ft_bg = da.map_blocks(translate,
                                                holograms_ft_bg,
                                                da.from_array(self.translations_bg[..., 0], chunks=dr_chunks),
                                                da.from_array(self.translations_bg[..., 1], chunks=dr_chunks),
                                                self.fxs,
                                                self.fys,
                                                dtype=complex,
                                                meta=xp.array((), dtype=complex))

        # #########################
        # determine phase offsets for background electric field, relative to initial slice
        # for each angle, so we can average this together to produce a single "background" image
        # #########################
        print("computing background phase shifts")
        if self.fit_phases:
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
                                       axis=self.bg_average_axes,
                                       keepdims=True)

        holograms_ft_bg = da.from_array(holograms_ft_bg_comp.compute(),
                                        chunks=holograms_ft_bg.chunksize)

        # #########################
        # determine phase offsets between electric field and background
        # #########################
        print("computing phase offsets")
        if self.use_average_as_background:
            self.phase_params = self.phase_params_bg
        else:
            if self.fit_phases:
                self.phase_params = da.map_blocks(get_global_phase_shifts,
                                                  holograms_ft,
                                                  holograms_ft_bg,
                                                  dtype=complex,
                                                  chunks=holograms_ft.chunksize[:-2] + (1, 1),
                                                  ).compute()

        holograms_ft = holograms_ft * self.phase_params

        # #########################
        # compute
        # #########################
        future = [da.map_blocks(to_cpu,
                          holograms_ft,
                                dtype=complex).to_zarr(self.store.store.path,
                                                       component="efields_ft",
                                                       compute=False,
                                                       compressor=compressor),
                  da.map_blocks(to_cpu,
                          holograms_ft_bg,
                                dtype=complex).to_zarr(self.store.store.path,
                                                       component="efield_bg_ft",
                                                       compute=False,
                                                       compressor=compressor)]
        dcompute(*future)

        self.efields_ft = da.from_zarr(self.store["efields_ft"])
        self.efield_bg_ft = da.from_zarr(self.store["efield_bg_ft"])

        if self.save_dir is not None:
            client.profile(filename=self.save_dir / f"{self.tstamp:s}_preprocessing_dask_profile.html")

        client.close()
        cluster.close()
        del cluster
        del client

    def reconstruct_n(self,
                      step: float = 1e-5,
                      max_iterations: int = 30,
                      use_gpu: bool = False,
                      print_fft_cache: bool = False,
                      compressor: Codec = Zlib(),
                      processes: bool = False,
                      n_workers: int = 1,
                      **reconstruction_kwargs) -> (array, tuple, dict):

        """
        Reconstruct refractive index using one of a several different models

        Additional keyword arguments are passed through to both the constructor and the run() method of
        the optimizer. These are used to e.g. set the strength of TV regularization, the number of iterations, etc.
        See Optimizer, RIOptimizer, and classes inheriting from RIOptimizer for more details.

        :param step: ignored if mode is "born" or "rytov"
        :param max_iterations:
        :param use_gpu:
        :param print_fft_cache: optionally print memory usage of GPU FFT cache at each iteration
        :param compressor:
        :param processes:
        :param n_workers:
        """
        if use_gpu and cp:
            xp = cp
        else:
            xp = np

        self.reconstruction_settings = deepcopy(reconstruction_kwargs)
        self.reconstruction_settings.update({"step": step,
                                             "max_iterations": max_iterations})

        if self.save_auxiliary_fields:
            chunk_shape = (1,) * (self.efields_ft.ndim - 2) + self.efields_ft.shape[-2:]
            e_fwd_out = self.store.create("efwd",
                                          shape=self.efields_ft.shape,
                                          chunks=chunk_shape,
                                          compressor=compressor,
                                          dtype=complex)

            e_scatt_out_shape = list(self.efields_ft.shape)
            e_scatt_out_shape[-3] *= self.nmax_multiplex
            e_scatt_out = self.store.create("escatt",
                                            shape=e_scatt_out_shape,
                                            chunks=chunk_shape,
                                            compressor=compressor,
                                            dtype=complex)

            cost_shape = self.efields_ft.shape[:-3] + (max_iterations + 1, self.npatterns)
            cost_chunk = (1,) * (self.efields_ft.ndim - 3) + cost_shape[-2:]
            costs_out = self.store.create("costs",
                                            shape=cost_shape,
                                            chunks=cost_chunk,
                                            compressor=compressor,
                                            dtype=float)

            step_shape = self.efields_ft.shape[:-3] + (max_iterations,)
            step_chunk = (1,) * (self.efields_ft.ndim - 3) + step_shape[-1:]
            steps_out = self.store.create("steps",
                                          shape=step_shape,
                                          chunks=step_chunk,
                                          compressor=compressor,
                                          dtype=float)

            if self.n_guess is None:
                n_start_out = self.store.create("n_start",
                                                shape=self.efields_ft.shape[:-3] + self.n_shape,
                                                compressor=compressor,
                                                dtype=complex)
            else:
                n_start_out = None
                self.store.array("n_start",
                                 np.expand_dims(self.n_guess, axis=list(range(self.nextra_dims))),
                                 compressor=compressor,
                                 dtype=complex)

        else:
            e_fwd_out = None
            e_scatt_out = None
            costs_out = None
            steps_out = None
            n_start_out = None

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
        if self.efields_ft.chunksize[-3] != self.npatterns:
            raise ValueError("")

        if self.efield_bg_ft.chunksize[-3] != self.npatterns:
            raise ValueError("")

        # ############################
        # compute information we need for reconstructions e.g. linear models and dz_final
        # ############################
        # todo: want this based on pupil function defined in init
        fx_atf = xp.fft.fftfreq(self.n_shape[-1], self.drs_n[-1])
        fy_atf = xp.fft.fftfreq(self.n_shape[-2], self.drs_n[-2])
        atf = (xp.sqrt(fx_atf[None, :] ** 2 +
                       fy_atf[:, None] ** 2) <= self.fmax).astype(complex)

        apodization_n = xp.outer(xp.asarray(tukey(self.n_shape[-2], alpha=0.1)),
                                 xp.asarray(tukey(self.n_shape[-1], alpha=0.1)))

        if self.model == "born" or self.model == "rytov":
            optimizer = self.model
            # affine transformation from reconstruction coordinates to pixel indices
            # for reconstruction, using FFT induced coordinates, i.e. zero is at array index (ny // 2, nx // 2)
            # for matrix, using image coordinates (0, 1, ..., N - 1)
            # note, due to order of operations -n//2 =/= - (n//2) when nx is odd
            xform_recon_pix2coords = params2xform([self.drs_n[-1], 0, -(self.n_shape[-1] // 2) * self.drs_n[-1],
                                                   self.drs_n[-2], 0, -(self.n_shape[-2] // 2) * self.drs_n[-2]])

            linear_model = fwd_model_linear(mean_beam_frqs_arr[..., 0],
                                            mean_beam_frqs_arr[..., 1],
                                            mean_beam_frqs_arr[..., 2],
                                            self.no,
                                            self.na_detection,
                                            self.wavelength,
                                            (self.ny, self.nx),
                                            (self.dxy, self.dxy),
                                            self.n_shape,
                                            self.drs_n,
                                            mode=self.model,
                                            interpolate=True,
                                            use_gpu=use_gpu)

            if self.verbose:
                tstart_step = perf_counter()
                print("estimating step size")

            mguess = LinearScatt(np.empty((1, self.ny, self.nx)),
                                 linear_model,
                                 self.no,
                                 self.wavelength,
                                 None,
                                 None,
                                 None)
            step = mguess.guess_step()

            if self.verbose:
                print(f"estimated in {perf_counter() - tstart_step:.2f}s")

        elif self.model == "bpm" or self.model == "ssnp":
            if self.model == "bpm":
                optimizer = BPM
            elif self.model == "ssnp":
                optimizer = SSNP

            if not (self.ny / self.n_shape[1]).is_integer():
                raise ValueError()
            if not (self.nx / self.n_shape[2]).is_integer():
                raise ValueError()

            nbin_y = int(self.ny // self.n_shape[1])
            nbin_x = int(self.nx // self.n_shape[2])

            if self.drs_n[1] != self.dxy * nbin_x or self.drs_n[2] != self.dxy * nbin_y:
                raise ValueError()

            # affine transformation from reconstruction coordinates to pixel indices
            # coordinates in finer coordinates
            xb = bin((xp.arange(self.nx) - (self.nx // 2)) * self.dxy,
                     [nbin_x], mode="mean")
            yb = bin((xp.arange(self.ny) - (self.ny // 2)) * self.dxy,
                     [nbin_y], mode="mean")

            xform_recon_pix2coords = params2xform([self.drs_n[-1], 0, float(xb[0]),
                                                   self.drs_n[-2], 0, float(yb[0])])

        if self.verbose:
            print(f"computing index of refraction for {int(np.prod(self.imgs_raw.shape[:-3])):d} images "
                  f"using model {self.model:s}.\n"
                  f"Image size = {self.npatterns} x {self.ny:d} x {self.nx:d},\n"
                  f"reconstruction size = {self.n_shape[0]:d} x {self.n_shape[1]:d} x {self.n_shape[2]:d}")

        # #############################
        # initial guess
        # #############################
        if self.verbose:
            tstart_linear_model = perf_counter()

        if self.n_guess is not None:
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
                                                   self.n_shape,
                                                   self.drs_n,
                                                   mode="born" if self.model == "born" else "rytov",
                                                   interpolate=False,
                                                   use_gpu=use_gpu)

        if self.verbose:
            print(f"Generated linear model for initial guess in {perf_counter() - tstart_linear_model:.2f}s")

        # #############################
        # reconstruction
        # #############################
        n_size = self.n_shape
        drs_n = self.drs_n
        scattered_field_regularization = self.scattered_field_regularization
        use_weighted_phase_unwrap = self.use_weighted_phase_unwrap

        def recon(efields_ft,
                  efields_bg_ft,
                  beam_frqs,
                  rmask,
                  dxy,
                  no,
                  wavelength,
                  na_detection,
                  atf,
                  apod,
                  step,
                  optimizer,
                  verbose,
                  print_fft_cache,
                  n_guess=None,
                  e_fwd_out=None,
                  e_scatt_out=None,
                  n_start_out=None,
                  block_id=None):

            nextra_dims = efields_ft.ndim - 3
            dims = tuple(range(nextra_dims))
            nimgs, ny, nx = efields_ft.shape[-3:]

            efields_ft = xp.asarray(efields_ft.squeeze(axis=dims))
            efields_bg_ft = xp.asarray(efields_bg_ft.squeeze(axis=dims))

            if rmask is not None:
                rmask = xp.asarray(rmask)

            if block_id is None:
                block_ind = None
                label = ""
            else:
                block_ind = block_id[:nextra_dims]
                label = f"{block_id} "

            # #######################
            # get initial guess
            # #######################
            if n_guess is not None:
                v_fts_start = ft3(get_v(xp.asarray(n_guess), no, wavelength), no_cache=True)
                if optimizer == "born" or optimizer == "rytov":
                    efield_scattered_ft = ft2(get_scattered_field(ift2(efields_ft),
                                                                  ift2(efields_bg_ft),
                                                                  scattered_field_regularization,
                                                                  use_born=optimizer == "born",
                                                                  use_weighted_unwrap=use_weighted_phase_unwrap))
            else:
                # todo: can I get rid of this if else statement?
                tstart_scatt = perf_counter()
                if nmax_multiplex == 1:
                    efield_scattered = get_scattered_field(ift2(efields_ft),
                                                           ift2(efields_bg_ft),
                                                           scattered_field_regularization,
                                                           use_born=optimizer == "born",
                                                           use_weighted_unwrap=use_weighted_phase_unwrap)
                else:
                    fx_shift = xp.fft.fftshift(xp.fft.fftfreq(n_size[-1], drs_n[-1]))
                    fy_shift = xp.fft.fftshift(xp.fft.fftfreq(n_size[-2], drs_n[-2]))
                    fxfx, fyfy = xp.meshgrid(fx_shift, fy_shift)

                    # demultiplexed fields
                    efield_scattered = xp.zeros((nimgs * nmax_multiplex, ny, nx), dtype=complex)
                    for ii in range(nimgs):
                        for jj in range(nmax_multiplex):
                            dists = norm(mean_beam_frqs_arr[:, ii, :2] -
                                         mean_beam_frqs_arr[jj, ii, :2], axis=1)
                            min_dist = 0.5 * np.min(dists[dists > 0.])

                            mask = xp.sqrt((fxfx - mean_beam_frqs_arr[jj, ii, 0]) ** 2 +
                                           (fyfy - mean_beam_frqs_arr[jj, ii, 1]) ** 2) > min_dist
                            efield_scattered[ii * nmax_multiplex + jj] = (
                                get_scattered_field(ift2(cut_mask(efields_ft[ii], mask)),
                                                    ift2(cut_mask(efields_bg_ft[ii], mask)),
                                                    scattered_field_regularization,
                                                    use_born=optimizer == "born",
                                                    use_weighted_unwrap=use_weighted_phase_unwrap))

                # compute scattered field
                efield_scattered_ft = ft2(efield_scattered)

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
                    print(f"computing scattered field took {perf_counter() - tstart_scatt:.2f}s")

            # ############
            # optionally write out starting refractive index
            # ############
            if n_start_out is not None:
                if verbose:
                    print("saving n_start", end="\r")
                    tstart_nstart = perf_counter()

                n_start_out[block_ind] = to_cpu(get_n(ift3(v_fts_start, no_cache=True), no, wavelength))

                if verbose:
                    print(f"saved n_start in {perf_counter() - tstart_nstart:.2f}s")

            if verbose:
                print(f"starting inference", end="\r")

            if use_gpu and print_fft_cache:
                print(f"gpu memory usage before inference = {cp.get_default_memory_pool().used_bytes() / 1e9:.2f}GB")
                print(cp.fft.config.get_plan_cache())

            if optimizer == "born" or optimizer == "rytov":
                model = LinearScatt(no,
                                    wavelength,
                                    (dxy, dxy),
                                    drs_n,
                                    v_fts_start.shape,
                                    efield_scattered_ft,
                                    linear_model,
                                    **reconstruction_kwargs
                                    )

                results = model.run(v_fts_start,
                                    step=step,
                                    verbose=verbose,
                                    label=label,
                                    max_iterations=max_iterations,
                                    **reconstruction_kwargs
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

                efields = bin(ift2(efields_ft), [nbin_y, nbin_x], mode="mean")
                del efields_ft

                efields_bg = bin(ift2(efields_bg_ft), [nbin_y, nbin_x], mode="mean")
                del efields_bg_ft

                model = optimizer(no,
                                  wavelength,
                                  None,
                                  drs_n,
                                  n_start.shape,
                                  efields,
                                  efields_bg,
                                  beam_frqs=beam_frqs[0] if beam_frqs.shape[0] == 1 else None,
                                  atf=atf,
                                  apodization=apod,
                                  mask=rmask,
                                  **reconstruction_kwargs
                                  )

                results = model.run(n_start,
                                    step=step,
                                    verbose=verbose,
                                    label=label,
                                    max_iterations=max_iterations,
                                    **reconstruction_kwargs
                                    )
                n = results["x"]

            # ################
            # optionally store costs
            # ################
            if costs_out is not None:
                if verbose:
                    print("storing costs")
                costs_out[block_ind] = to_cpu(results["costs"])

            if steps_out is not None:
                if verbose:
                    print("storing steps")
                steps_out[block_ind] = to_cpu(results["steps"])

            # ################
            # optionally compute predicated e_field based on n
            # ################
            if e_fwd_out is not None:
                tstart_efwd = perf_counter()
                if verbose:
                    print("computing forward model")

                if optimizer == "born" or optimizer == "rytov":
                    e_fwd_out[block_ind] = to_cpu(ift2(model.fwd_model(ft3(get_v(n, no, wavelength)))))
                else:
                    slices = (slice(0, 1), slice(-1, None), slice(None), slice(None))  # [0, -1, :, :]

                    tstart_efwd = perf_counter()
                    for ii in range(nimgs):
                        ind_now = block_ind + (ii,)
                        e_fwd_out[ind_now] = to_cpu(model.fwd_model(n.squeeze(), inds=[ii])[slices]).squeeze()

                if verbose:
                    print(f"computed forward model in {perf_counter() - tstart_efwd:.2f}s")

            if use_gpu and print_fft_cache:
                print(f"gpu memory usage after inference = {cp.get_default_memory_pool().used_bytes() / 1e9:.2f}GB")
                print(cp.fft.config.get_plan_cache())

            return to_cpu(n).reshape((1,) * nextra_dims + n_size)

        # #######################
        # get refractive index
        # #######################
        n = da.map_blocks(recon,
                          self.efields_ft,  # data
                          self.efield_bg_ft,  # background
                          mean_beam_frqs_arr,
                          self.realspace_mask,  # masks
                          self.dxy,
                          self.no,
                          self.wavelength,
                          self.na_detection,
                          atf,
                          apodization_n,
                          step,
                          optimizer,
                          self.verbose,
                          print_fft_cache,
                          n_guess=self.n_guess,
                          e_fwd_out=e_fwd_out,
                          e_scatt_out=e_scatt_out,
                          n_start_out=n_start_out,
                          chunks=(1,) * self.nextra_dims + self.n_shape,
                          dtype=complex,
                          )

        cluster = LocalCluster(processes=processes,
                               n_workers=n_workers,
                               threads_per_worker=1)
        client = Client(cluster)
        print(cluster.dashboard_link)

        dcompute([n.to_zarr(self.store.store.path,
                            component="n",
                            compute=False,
                            compressor=compressor)])  # compute

        # save profile to see what takes the most time
        client.profile(filename=self.save_dir / f"{self.tstamp:s}_reconstruction_profile.html")


        # ############################
        # construct affine tranforms between reconstructed data and camera pixels
        # todo: this could go in __init__() except for the fact xform_recon_pix2coords comes from reconstruct_n()
        # ############################
        # affine transformation from camera ROI coordinates to pixel indices
        xform_raw_roi_pix2coords = params2xform([self.dxy, 0, -(self.n_shape[-1] // 2) * self.dxy,
                                                 self.dxy, 0, -(self.n_shape[-2] // 2) * self.dxy])

        # composing these two transforms gives affine from recon pixel indices to
        # recon pix inds -> recon coords = ROI coordinates -> ROI pix inds
        xform_recon2raw_roi = inv(xform_raw_roi_pix2coords).dot(xform_recon_pix2coords)

        xform_odt_recon_to_cam_roi = None
        xform_odt_recon_to_full = None
        if self.data_roi is not None:
            # transform from reconstruction processing roi to camera roi
            odt_recon_roi = deepcopy(self.data_roi)
            xform_process_roi_to_cam_roi = params2xform([1, 0, odt_recon_roi[2],
                                                         1, 0, odt_recon_roi[0]])
            xform_odt_recon_to_cam_roi = xform_process_roi_to_cam_roi.dot(xform_recon2raw_roi)

            if self.cam_roi is not None:
                # todo: is there a problem with this transform?
                # transform from camera roi to uncropped chip
                xform_cam_roi_to_full = params2xform([1, 0, self.cam_roi[2],
                                                      1, 0, self.cam_roi[0]])
                xform_odt_recon_to_full = xform_cam_roi_to_full.dot(xform_process_roi_to_cam_roi)

        # store all transforms in JSON serializable form
        self.xform_dict = {"xform_recon_pix2coords": np.asarray(xform_recon_pix2coords).tolist(),
                           "affine_xform_recon_2_raw_process_roi": np.asarray(xform_recon2raw_roi).tolist(),
                           "affine_xform_recon_2_raw_camera_roi": np.asarray(xform_odt_recon_to_cam_roi).tolist(),
                           "affine_xform_recon_2_raw_camera": np.asarray(xform_odt_recon_to_full).tolist(),
                           "processing roi": np.asarray(self.data_roi).tolist(),
                           "camera roi": np.asarray(self.cam_roi).tolist()
                           }

    def plot_translations(self,
                          index: tuple[int],
                          time_axis: int = 1,
                          figsize: Sequence[float, float] = (30., 8.),
                          **kwargs) -> Figure:

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
                  index: Sequence[int],
                  time_axis: int = 1,
                  figsize: Sequence[float, float] = (30., 8.),
                  **kwargs) -> Figure:
        """

        :param index: should be of length self.nextra_dims - 1. Index along these axes, but ignoring whichever
          axes is the time axis. So e.g. if the axis are position x time x z x parameter then time_axis = 1 and
          the index could be (2, 1, 0) which would selection position 2, z 1, parameter 0.
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
        ref_frq_diffs = (self.reference_frq - np.mean(self.reference_frq,
                                                      axis=1,
                                                      keepdims=True))[ref_slices].squeeze(squeeze_axes)

        # plot
        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle(f"index={index}\nfrequency variation versus time")

        # plot frequency differences
        ax = figh.add_subplot(1, 3, 1)
        ax.plot(norm(hgram_frq_diffs, axis=-1) / self.dfx, '.-')
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
        ax.plot(norm(ref_frq_diffs, axis=-1) / self.dfx, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("(frequency norm - mean) / dfx")
        ax.set_title("reference frequency deviation amplitude")

        return figh

    def plot_phases(self,
                    index: tuple[int],
                    time_axis: int = 1,
                    figsize: Sequence[float, float] = (30., 8.),
                    **kwargs) -> Figure:
        """
        Plot phase drift fits versus time for a single slice.
        Additional key word arguments are passed through to plt.figure()

        :param index: should be of length self.nextra_dims - 1. Index along these axes, but ignoring whichever
          axes is the time axis. So e.g. if the axis are position x time x z x parameter then time_axis = 1
          and the index could be (2, 1, 0) which would selection position 2, z 1, parameter 0.
        :param time_axis:
        :param figsize:
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
                    figsize: Sequence[float, float] = (30., 8.),
                    **kwargs
                    ) -> (Figure, np.ndarray, np.ndarray):
        """
        Plot hologram intensity. Additional keyword arguments are passed through to plt.figure()

        :param index:
        :param time_axis:
        :param figsize:
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

        e_powers_rms = (da.sqrt(da.mean(da.abs(self.efields_ft) ** 2, axis=(-1, -2))) /
                        np.prod(self.efields_ft.shape[-2:]))
        e_powers_rms_bg = (da.sqrt(da.mean(da.abs(self.efield_bg_ft) ** 2, axis=(-1, -2))) /
                           np.prod(self.efield_bg_ft.shape[-2:]))

        # get slice of phases
        with ProgressBar():
            computed = dcompute([e_powers_rms[slices].squeeze(axis=squeeze_axes),
                                e_powers_rms_bg[slices].squeeze(axis=squeeze_axes)],
                                scheduler="threads")

        epowers, epowers_bg = computed[0]
        epowers = to_cpu(epowers)
        epowers_bg = to_cpu(epowers_bg)

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

        return figh2, epowers, epowers_bg

    def plot_costs(self,
                   index: tuple[int] = None,
                   **kwargs) -> Figure:

        # todo: implement index
        figh_cost = plt.figure(**kwargs)
        ax = figh_cost.add_subplot(1, 1, 1)

        if hasattr(self.store, "costs"):
            carr = np.array(self.store.costs)[0, 0, 0, 0].transpose()

            ax.plot(carr, '.')
            ax.set_xlabel("iteration")
            ax.set_ylabel("cost")

        return figh_cost

    def plot_steps(self,
                   index: tuple[int] = None,
                   **kwargs):
        # todo: implement index
        figh_step = plt.figure(**kwargs)
        ax = figh_step.add_subplot(1, 1, 1)

        if hasattr(self.store, "steps"):
            # plot steps
            sarr = np.array(self.store.steps)[0, 0, 0, 0]

            ax.plot(sarr, '.')
            ax.set_xlabel("iteration")
            ax.set_ylabel("step-size")

        return figh_step

    def plot_diagnostics(self,
                         index: tuple[int],
                         time_axis: int,
                         interactive: bool = False,
                         save: bool = True,
                         **kwargs):

        context = {}
        if not interactive:
            context['interactive'] = False
            context['backend'] = "agg"

        with rc_context(context):
            # frequencies
            figh_frq = self.plot_frqs(index,
                                      time_axis=time_axis,
                                      **kwargs)

            # plot phases
            figh_ph = self.plot_phases(index,
                                       time_axis=time_axis,
                                       **kwargs)

            # plot translations
            figh_xl = self.plot_translations(index,
                                             time_axis=time_axis,
                                             **kwargs)

            # plot sampling
            figh_sampling = self.plot_odt_sampling(figsize=(15, 5),
                                                   **kwargs)

            # costs
            figh_costs = self.plot_costs(index)

            # step
            figh_steps = self.plot_steps(index)

        if save:
            figh_frq.savefig(self.save_dir / "hologram_frequency_stability.png")
            figh_ph.savefig(self.save_dir / "phase_stability.png")
            figh_xl.savefig(self.save_dir / "registration.png")
            figh_sampling.savefig(self.save_dir / "fourier_sampling.png")
            figh_costs.savefig(self.save_dir / "costs.png")
            figh_steps.savefig(self.save_dir / "steps.png")

        if not interactive:
            plt.close(figh_frq)
            plt.close(figh_ph)
            plt.close(figh_xl)
            plt.close(figh_sampling)
            plt.close(figh_costs)
            plt.close(figh_costs)

    def plot_odt_sampling(self,
                          index: Optional[tuple[int]] = None,
                          **kwargs) -> Figure:
        """
        Illustrate the region of frequency space which is obtained using the plane waves described by frqs

        :param index:
        :param kwargs: passed through to figure
        :return figh:
        """
        # todo: take slice argument
        # nfrqs x 2 array of [[fx0, fy0], [fx1, fy1], ...]
        frqs = np.stack(self.get_beam_frqs(), axis=-3)[0, 0, 0, 0, :, 0][:, (0, 1)]

        frq_norm = self.no / self.wavelength
        alpha_det = np.arcsin(self.na_detection / self.no)

        if self.na_excitation / self.no < 1:
            alpha_exc = np.arcsin(self.na_excitation / self.no)
        else:
            # if na_excite is immersion objective and beam undergoes TIR at interface for full NA
            alpha_exc = np.pi / 2

        fzs = get_fzs(frqs[:, 0], frqs[:, 1], self.no, self.wavelength)
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
            ax.add_artist(Arc((-frqs_3d[ii, 0], -frqs_3d[ii, 2]),
                              2 * frq_norm,
                              2 * frq_norm,
                              angle=90,
                              theta1=-alpha_det * 180 / np.pi,
                              theta2=alpha_det * 180 / np.pi,
                              edgecolor="k",
                              **kwargs))

        # draw arcs for the extremal angles
        fx_edge = self.na_excitation / self.wavelength
        fz_edge = np.sqrt((self.no / self.wavelength) ** 2 - fx_edge ** 2)

        ax.plot(-fx_edge, -fz_edge, 'r.')
        ax.plot(fx_edge, -fz_edge, 'r.')

        ax.add_artist(Arc((-fx_edge, -fz_edge),
                          2 * frq_norm,
                          2 * frq_norm,
                          angle=90,
                          theta1=-alpha_det * 180 / np.pi,
                          theta2=alpha_det * 180 / np.pi,
                          edgecolor="r",
                          label="extremal frequency data"))
        ax.add_artist(Arc((fx_edge, -fz_edge),
                          2 * frq_norm,
                          2 * frq_norm,
                          angle=90,
                          theta1=-alpha_det * 180 / np.pi,
                          theta2=alpha_det * 180 / np.pi,
                          edgecolor="r"))

        # draw arc showing possibly positions of centers
        ax.add_artist(Arc((0, 0),
                          2 * frq_norm,
                          2 * frq_norm,
                          angle=-90,
                          theta1=-alpha_exc * 180 / np.pi,
                          theta2=alpha_exc * 180 / np.pi,
                          edgecolor="b",
                          label="allowed beam frqs"))

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
            ax.add_artist(Arc((-frqs_3d[ii, 1], -frqs_3d[ii, 2]),
                              2 * frq_norm,
                              2 * frq_norm,
                              angle=90,
                              theta1=-alpha_det * 180 / np.pi,
                              theta2=alpha_det * 180 / np.pi,
                              edgecolor="k"))

        # draw arcs for the extremal angles
        fy_edge = self.na_excitation / self.wavelength
        fz_edge = np.sqrt((self.no / self.wavelength) ** 2 - fy_edge ** 2)

        ax.plot(-fy_edge, -fz_edge, 'r.')
        ax.plot(fy_edge, -fz_edge, 'r.')

        ax.add_artist(Arc((-fy_edge, -fz_edge),
                          2 * frq_norm,
                          2 * frq_norm,
                          angle=90,
                          theta1=-alpha_det * 180 / np.pi,
                          theta2=alpha_det * 180 / np.pi,
                          edgecolor="r"))
        ax.add_artist(Arc((fy_edge, -fz_edge),
                          2 * frq_norm,
                          2 * frq_norm,
                          angle=90,
                          theta1=-alpha_det * 180 / np.pi,
                          theta2=alpha_det * 180 / np.pi,
                          edgecolor="r"))

        # draw arc showing possibly positions of centers
        ax.add_artist(Arc((0, 0),
                          2 * frq_norm,
                          2 * frq_norm,
                          angle=-90,
                          theta1=-alpha_exc * 180 / np.pi,
                          theta2=alpha_exc * 180 / np.pi,
                          edgecolor="b"))

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
                                 self.na_detection / self.wavelength,
                                 fill=False,
                                 color="k"))

        ax.add_artist(Circle((0, 0),
                             self.na_excitation / self.wavelength,
                             fill=False,
                             color="b"))
        ax.add_artist(Circle((0, 0),
                             (self.na_excitation + self.na_detection) / self.wavelength,
                             fill=False,
                             color="r"))

        ax.set_ylim([-2 * frq_norm, 2 * frq_norm])
        ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
        ax.set_xlabel("$f_x$ (1/$\mu m$)")
        ax.set_ylabel("$f_y$ (1/$\mu m$)")

        # ########################
        # 3D
        # ########################
        ax = figh.add_subplot(grid[0, 3], projection="3d")
        ax.set_title("3D projection")

        fx = fy = np.linspace(-self.na_detection / self.wavelength,
                              self.na_detection / self.wavelength, 100)
        fxfx, fyfy = np.meshgrid(fx, fy)
        ff = np.sqrt(fxfx ** 2 + fyfy ** 2)
        fmax = self.na_detection / self.wavelength

        fxfx[ff > fmax] = np.nan
        fyfy[ff > fmax] = np.nan
        fzfz = get_fzs(fxfx, fyfy, self.no, self.wavelength)

        # kx0, ky0, kz0
        fxyz0 = np.stack((fxfx, fyfy, fzfz), axis=-1)
        for ii in range(len(frqs_3d)):
            ax.plot_surface(fxyz0[..., 0] - frqs_3d[ii, 0],
                            fxyz0[..., 1] - frqs_3d[ii, 1],
                            fxyz0[..., 2] - frqs_3d[ii, 2],
                            alpha=0.3)

        ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
        ax.set_ylim([-2 * frq_norm, 2 * frq_norm])
        ax.set_zlim([-1, 1])  # todo: set based on na's

        ax.set_xlabel("$f_x$ (1/$\mu m$)")
        ax.set_ylabel("$f_y$ (1/$\mu m$)")
        ax.set_zlabel("$f_z$ (1/$\mu m$)")

        return figh

    def plot_image(self,
                   index: Optional[tuple[int]] = None,
                   gamma: float = 0.1,
                   figsize: tuple[float, float] = (35., 15.),
                   **kwargs) -> Figure:
        """
        display raw image and holograms

        :param index: index of image to display. Should be of length self.nextra_dims + 1
        :param gamma: gamma to be used when display fourier transforms
        :param figsize:
        :return figh: figure handle
        """

        if index is None:
            index = (0,) * (self.nextra_dims + 1)

        extent = [self.x[0] - 0.5 * self.dxy,
                  self.x[-1] + 0.5 * self.dxy,
                  self.y[-1] + 0.5 * self.dxy,
                  self.y[0] - 0.5 * self.dxy]

        extent_f = [self.fxs[0] - 0.5 * self.dfx,
                    self.fxs[-1] + 0.5 * self.dxy,
                    self.fys[-1] + 0.5 * self.dfy,
                    self.fys[0] - 0.5 * self.dfy]

        # ######################
        # plot
        # ######################
        img_now = to_cpu(self.imgs_raw[index].compute())
        img_ft = ft2(img_now)

        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle(f"{index}, {self.dimensions}")
        grid = figh.add_gridspec(nrows=4,
                                 height_ratios=[1, 0.1, 1, 0.1],
                                 ncols=6,
                                 wspace=0.2)

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
            holo_ft = to_cpu(self.efields_ft[index].compute())
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
            index_bg = tuple([v if self.efields_ft.shape[ii] != 0 else 0 for ii, v in enumerate(index)])

            holo_ft_bg = to_cpu(self.efield_bg_ft[index_bg].compute())
            holo_bg = ift2(holo_ft_bg)

            ax = figh.add_subplot(grid[2, 1])
            ax.set_title("$E_{bg}(f)$")
            im = ax.imshow(np.abs(holo_ft_bg),
                           extent=extent_f,
                           norm=PowerNorm(gamma=gamma, vmin=vmin_ef, vmax=vmax_ef),
                           cmap="bone")
            ax.set_xlim([-self.fmax, self.fmax])
            ax.set_ylim([self.fmax, -self.fmax])

            # ######################
            # real-space hologram
            # ######################
            ax = figh.add_subplot(grid[2, 2])
            ax.set_title("$E_{bg}(r)$")
            im = ax.imshow(np.abs(holo_bg),
                           extent=extent,
                           cmap="bone",
                           vmin=0,
                           vmax=vmax_e)

            # ######################
            # real-space hologram phase
            # ######################
            ax = figh.add_subplot(grid[2, 3])
            ax.set_title("angle $[E_{bg}(r)]$")
            im = ax.imshow(np.angle(holo_bg),
                           extent=extent,
                           cmap="RdBu",
                           vmin=-np.pi,
                           vmax=np.pi)

            # ######################
            # |E(r) - Ebg(r)|
            # ######################
            ax = figh.add_subplot(grid[0, 4])
            ax.set_title("$|E(r) - E_{bg}(r)|$")
            im = ax.imshow(np.abs(holo - holo_bg),
                           extent=extent,
                           cmap="bone",
                           vmin=0)

            ax = figh.add_subplot(grid[1, 4])
            plt.colorbar(im, cax=ax, location="bottom")

            # ######################
            # |E(r)| - |Ebg(r)|
            # ######################
            ax = figh.add_subplot(grid[2, 4])
            ax.set_title("$|E(r)| - |E_{bg}(r)|$")

            im = ax.imshow(np.abs(holo) - np.abs(holo_bg),
                           extent=extent,
                           cmap="RdBu")
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

            im = ax.imshow(ang_diff,
                           extent=extent,
                           cmap="RdBu",
                           vmin=-np.pi,
                           vmax=np.pi)

            ax = figh.add_subplot(grid[1, 5])
            plt.colorbar(im, cax=ax, location="bottom")

        except Exception as e:
            print(e)

        return figh


def _ft_abs(m: array) -> array: return ft2(abs(ift2(m)))


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
    if cp and isinstance(img, cp.ndarray):
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

    if cp and isinstance(imgs, cp.ndarray):
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
            a = xp.expand_dims(imgs[ind].ravel(), axis=1)
            b = ref_imgs[ind].ravel()
        else:
            mask = xp.logical_and(np.abs(imgs[ind]) > thresh,
                                  xp.abs(ref_imgs[ind]) > thresh)
            a = xp.expand_dims(imgs[ind][mask], axis=1)
            b = ref_imgs[ind][mask]

        fps, _, _, _ = xp.linalg.lstsq(a, b, rcond=None)
        fit_params[ind] = fps

    return fit_params


def fit_phase_ramp(imgs_ft: array,
                   ref_imgs_ft: array,
                   dxy: float,
                   thresh: float = 1/30,
                   init_params: Optional[array] = None) -> np.ndarray:
    """
    Given a stack of images and reference images, determine the phase ramp parameters relating them

    :param imgs_ft: n0 x n1 x ... x n_{-2} x n_{-1} array
    :param ref_imgs_ft: reference images. Should be broadcastable to same size as imgs.
    :param dxy:
    :param thresh:
    :param init_params:
    :return fit_params:
    """

    if cp and isinstance(imgs_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    ref_imgs_ft = xp.asarray(ref_imgs_ft)

    # broadcast images and references images to same shapes
    imgs_ft, ref_imgs_ft = xp.broadcast_arrays(imgs_ft, ref_imgs_ft)

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
        mask = xp.abs(imgs_ft[ind]) >= ref_val

        imgs_ft_mask = imgs_ft[ind][mask]
        fxfx_mask = fxfx[mask]
        fyfy_mask = fyfy[mask]
        ref_imgs_ft_mask = ref_imgs_ft[ind][mask]

        def fnh(p): return ref_imgs_ft_mask * np.exp(-2 * np.pi * 1j * (fxfx_mask * p[0] + fyfy_mask * p[1]))

        if init_params is None:
            ip = np.array([0., 0.])
        else:
            ip = init_params[ind]

        results = fit_model(imgs_ft_mask,
                            fnh,
                            init_params=ip,
                            function_is_complex=True
                            )

        fit_params[ind] = results["fit_params"]

    return fit_params


def get_scattered_field(efields: array,
                        efields_bg: array,
                        regularization: float = 0.,
                        use_born: bool = False,
                        use_weighted_unwrap: bool = True) -> array:
    """
    Compute estimate of scattered field in real space with regularization. Depending on the mode, this
    will either be an actual estimate of the scattered field, or an estimate of the Rytov phase

    The Rytov phase, \psi_s(r), is defined by

    .. math::

      U_t(r) &= \\exp \\left[\\psi_o(r) + \\psi_s(r) \\right]

      U_o(r) &= \\exp \\left[\\psi_o(r) \\right]

    We estimate it from

    .. math::

      \\psi_s(r) = \\log \\left \\vert \\frac{U_t(r)}{U_o(r)} \\right \\vert + i \\text{unwrap} \\left[\\text{angle}(U_t) - \\text{angle}(U_o) \\right]


    :param efields: array of size n0 x ... x nm x ny x nx
    :param efields_bg: broadcastable to same size as efields
    :param regularization: limit the influence of any pixels where |efields| < regularization
    :param use_born: whether to use the Born or Rytov model.
    :param use_weighted_unwrap: whether to use phase wrap routine where confidence is weighted by the
      electric field amplitude
    :return scattered: scattered field or Rytov phase of same size as efields
    """
    if use_born:
        scattered = (efields - efields_bg) / (abs(efields_bg) + regularization)
    else:
        if cp and isinstance(efields, cp.ndarray):
            xp = cp
        else:
            xp = np

        efields = xp.asarray(efields)
        efields_bg = xp.asarray(efields_bg)
        efields, efields_bg = xp.broadcast_arrays(efields, efields_bg)

        # output values
        phase_diff = xp.mod(xp.angle(efields) - xp.angle(efields_bg), 2 * np.pi)
        # convert phase difference from interval [0, 2*np.pi) to [-np.pi, np.pi)
        phase_diff[phase_diff >= np.pi] -= 2 * np.pi

        # set real parts of Rytov phase
        scattered = xp.log(abs(efields) / (abs(efields_bg))) + 1j * 0

        # set imaginary parts
        # loop over all dimensions except the last two
        nextra_shape = efields.shape[:-2]
        nextra = int(np.prod(nextra_shape))
        for ii in range(nextra):
            ind = np.unravel_index(ii, nextra_shape)

            if use_weighted_unwrap:
                weights = xp.abs(efields_bg[ind])
            else:
                weights = None

            scattered[ind] += 1j * weighted_phase_unwrap(phase_diff[ind],
                                                         weight=weights)

        scattered[abs(efields_bg) < regularization] = 0

    return scattered


# holograms
def unmix_hologram(img: array,
                   dxy: float,
                   fmax_int: float,
                   fx_ref: np.ndarray,
                   fy_ref: np.ndarray,
                   apodization: Optional[array] = None,
                   mask: Optional[array] = None) -> array:
    """
    Given an off-axis hologram image, determine the electric field.
    (1) Fourier transform the raw hologram
    (2) shift the hologram region near f_ref to the center of the image
    (3) set regions beyond fmax to zero

    :param img: n1 x ... x n_{-3} x n_{-2} x n_{-1} array
    :param dxy: pixel size
    :param fmax_int: maximum frequency where intensity OTF has support = 2*na / wavelength
    :param fx_ref: x-component of hologram reference frequency
    :param fy_ref: y-component of hologram reference frequency
    :param apodization: apodization window applied to real-space image before Fourier transformation
    :param mask: set regions of hologram where mask is True to zero, even if they are within the bandpass
    :return efield_ft: hologram electric field in Fourier space
    """

    if cp and isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    if apodization is None:
        apodization = 1.
    apodization = xp.asarray(apodization)

    # get frequency data
    ny, nx = img.shape[-2:]
    fxs = xp.fft.fftshift(xp.fft.fftfreq(nx, dxy))
    fys = xp.fft.fftshift(xp.fft.fftfreq(ny, dxy))
    ff_perp = np.sqrt(fxs[None, :] ** 2 + fys[:, None] ** 2)

    # compute efield
    img_ft = ft2(img * apodization)
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
                                        fov: Sequence[int, int, int],
                                        sampling_factors: Sequence[float, float, float] = (1., 1., 1.)
                                        ) -> (tuple[float, float, float], tuple[int, int, int]):
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


def display_tomography_recon(location: Union[str, Path, zarr.hierarchy.Group],
                             raw_data_fname: Optional[Union[str, Path, zarr.hierarchy.Group]] = None,
                             raw_data_component: str = "cam2/odt",
                             show_raw: bool = True,
                             show_scattered_fields: bool = False,
                             show_efields: bool = False,
                             compute: bool = True,
                             time_axis: int = 1,
                             time_range: Optional[list[int]] = None,
                             phase_lim: float = np.pi,
                             n_lim: tuple[float, float] = (0., 0.05),
                             e_lim: tuple[float, float] = (0., 500.),
                             escatt_lim: tuple[float, float] = (-5., 5.),
                             block_while_display: bool = True,
                             real_cmap="bone",
                             phase_cmap="RdBu",
                             scale_z: bool = True):
    """
    Display reconstruction results and (optionally) raw data in Napari

    :param location: refractive index reconstruction stored in zarr file
    :param raw_data_fname: raw data stored in zar file
    :param raw_data_component:
    :param show_raw:
    :param show_scattered_fields:
    :param show_efields:
    :param compute:
    :param time_axis:
    :param time_range:
    :param phase_lim:
    :param n_lim:
    :param e_lim:
    :param escatt_lim:
    :param block_while_display:
    :param real_cmap:
    :param phase_cmap:
    :param scale_z:
    :return: viewer
    """

    import napari

    # optionally load raw data
    if raw_data_fname is not None:
        if isinstance(raw_data_fname, (Path, str)):
            raw_data = zarr.open(raw_data_fname, "r")
        else:
            raw_data = raw_data_fname
    else:
        show_raw = False

    # load data
    if isinstance(location, (Path, str)):
        img_z = zarr.open(location, "r")
    else:
        img_z = location

    if not hasattr(img_z, "efield_bg_ft") or not hasattr(img_z, "efields_ft"):
        show_efields = False

    if "dr" in img_z.attrs.keys():
        drs_n = img_z.attrs["dr"]
    elif "drs_n" in img_z.attrs.keys():
        drs_n = img_z.attrs["drs_n"]
    else:
        raise ValueError()

    n_axis_names = img_z.attrs["dimensions"]
    no = img_z.attrs["no"]

    try:
        data_slice = tuple([slice(s[0], s[1], s[2]) for s in img_z.attrs["data_slice"]])
    except KeyError:
        data_slice = None

    # load affine xforms
    # Napari uses convention (y, x) whereas I'm using (x, y),
    # so need to swap these dimensions in affine xforms
    swap_xy = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    try:
        affine_recon2cam_xy = np.array(img_z.attrs["affine_xform_recon_2_raw_camera_roi"])
    except KeyError:
        affine_recon2cam_xy = params2xform([1, 0, 0, 1, 0, 0])
    affine_recon2cam = swap_xy.dot(affine_recon2cam_xy.dot(swap_xy))
    affine_cam2recon = inv(affine_recon2cam)

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
            n_start = no * np.ones(1)

    else:
        n = no * np.ones(1)
        n_start = no * np.ones(1)

    n_real = n.real - no
    n_imag = n.imag
    n_start_real = n_start.real - no
    n_start_imag = n_start.imag

    ny, nx = n_real.shape[-2:]

    # ######################
    # prepare raw images
    # ######################
    if show_raw:
        if show_n:
            slices_raw = slices + (slice(None), slice(None), slice(None))
        else:
            slices_raw = tuple([slice(None)] * raw_data[raw_data_component].ndim)

        if data_slice is None:
            data_slice = tuple([slice(None)] * raw_data[raw_data_component].ndim)

        dc = da.from_zarr(raw_data[raw_data_component])
        imgs = da.expand_dims(dc[data_slice][slices_raw], axis=-3)

        if compute:
            print("loading raw images")
            with ProgressBar():
                imgs = imgs.compute()
    else:
        imgs = np.ones(1)

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
                c = dcompute([e, ebg, efield_power])
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
    if show_scattered_fields and hasattr(img_z, "escatt"):
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
        elif not show_efields and show_raw:
            npatt = imgs.shape[-4]
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

        # this can be a different size due to multiplexing
        bcast_root_scatt = (1,) * n_extra_dims + (1, nz, 1, 1)
        bcast_shape_scatt = np.broadcast_shapes(escatt_real.shape, bcast_root_scatt)
        escatt_real = np.broadcast_to(escatt_real, bcast_shape_scatt)
        escatt_imag = np.broadcast_to(escatt_imag, bcast_shape_scatt)
        print('finished broadcasting')

    # ######################
    # create viewer
    # ######################
    viewer = napari.Viewer(title=str(img_z.store.path))

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

    # ######################
    # reconstructed index of refraction
    # ######################
    if show_n:
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


def compare_recons(fnames: Sequence[Union[str, Path]],
                   vmax: float = 0.05,
                   compute: bool = True,
                   block_while_display: bool = True,
                   verbose: bool = False):
    """

    :param fnames:
    :param vmax:
    :param compute:
    :param block_while_display:
    :param verbose:
    :return viewer:
    """

    import napari

    if isinstance(fnames, (Path, str)):
        fnames = [fnames]

    swap_xy = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])

    v = napari.Viewer()
    tstart = perf_counter()
    for f in fnames:
        if verbose:
            print(f"loading {str(f):s}, elapsed time = {perf_counter() - tstart:.2f}s")

        znow = zarr.open(f, "r")
        affine_now = swap_xy.dot(np.array(znow.attrs["affine_xform_recon_2_raw_camera_roi"]).dot(swap_xy))
        no = znow.attrs["no"]
        dz_now = znow.attrs["dr"][0]
        try:
            dz_refocus_now = znow.attrs["reconstruction_settings"]["dz_refocus"]
        except KeyError:
            dz_refocus_now = 0.

        if znow.n.shape[-3] == 2:
            if compute:
                n_now = np.array(znow.n[..., 0, :, :]) - no
            else:
                n_now = da.from_zarr(znow.n[..., 0, :, :]) - no
        else:
            if compute:
                n_now = np.array(znow.n).real - no
            else:
                n_now = da.from_zarr(znow.n).real - no

        nz = znow.n.shape[-3]
        zs = (np.arange(nz) - nz // 2) * dz_now + dz_refocus_now

        v.add_image(n_now,
                    scale=(dz_now, 1, 1),
                    translate=(zs[0], 0, 0),
                    affine=affine_now,
                    name=f"{f.parent.name:s} n real",
                    contrast_limits=[0, vmax],
                    colormap="bone"
                    )

    v.dims.set_current_step(axis=0, value=0)
    v.dims.set_current_step(axis=1, value=0)
    v.show(block=block_while_display)

    return v


def parse_time(dt: float) -> (tuple, str):
    """
    Parse a time difference in seconds into days, hours, minutes and seconds

    :param dt: time difference in seconds
    :return data tuple, time string:
    """
    days = int(dt // (24 * 60 * 60))
    hours = int(dt // (60 * 60) - days * 24)
    mins = int(dt // 60 - hours * 60 - days * 24 * 60)
    secs = dt - mins * 60 - hours * 60 * 60 - days * 24 * 60 * 60
    tstr = f"{days:02d} days, {hours:02d}h:{mins:02d}m:{secs:04.1f}s"

    return (days, hours, mins, secs), tstr
