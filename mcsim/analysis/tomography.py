"""
Tools for reconstructing optical diffraction tomography (ODT) data using either the Born approximation,
Rytov approximation, or multislice (paraxial) beam propagation method (BPM) and a FISTA solver. The primary
reconstruction tasks are carried out with the tomography class
"""
from time import perf_counter
import datetime
from warnings import catch_warnings, simplefilter, warn
from typing import Union, Optional
from collections.abc import Sequence
from inspect import getfullargspec
# numerical tools
import numpy as np
from numpy.linalg import norm, inv
from numpy.fft import fftshift, fftfreq
from scipy.signal.windows import tukey, hann
# parallelization
from tqdm import tqdm
import joblib
from joblib.externals.loky import get_reusable_executor
from dask.config import set as dask_cfg_set
from dask import delayed
import dask
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
                                 fit_xform_points_ransac,
                                 fit_xform_points,
                                 xform_points)
from mcsim.analysis.phase_unwrap import phase_unwrap as weighted_phase_unwrap
from mcsim.analysis.optimize import Optimizer, to_cpu, soft_threshold
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
    import napari
except ImportError:
    napari = None

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
                 fit_phase_profile: bool = False,
                 phase_profile_l1: float = 1e3,
                 apodization: Optional[np.ndarray] = None,
                 save_auxiliary_fields: bool = False,
                 compressor: Codec = Zlib(),
                 save_float32: bool = False,
                 step: float = 1.,
                 **reconstruction_kwargs,
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
        :param compressor: by default use Zlib(), which is a good combination of fast/reasonable compression ratio.
          For better compression, use bz2.BZ2(), although this is much slower.
        :param save_float32:
        :param step:
        :param reconstruction_kwargs: Additional keyword arguments are passed through to both the constructor
          and the run() method of the optimizer. These are used to e.g. set the strength of TV regularization,
          the number of iterations, etc. See Optimizer, RIOptimizer, and classes inheriting from RIOptimizer
          for more details.
        """
        self.verbose = verbose
        self.tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

        # ########################
        # save directory and zarr backing
        # ########################
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True, parents=True)
            self.dask_tmp_dir = self.save_dir
            self.store = zarr.open(self.save_dir / self.store_name, "a")
        else:
            self.save_dir = None
            self.store = zarr.open()
            self.dask_tmp_dir = None

        self.compressor = compressor
        self.save_float32 = bool(save_float32)

        self.timing = {"fit_frequency_time": None,
                       "unmix_hologram_time": None,
                       "plot_diagnostics_time": None,
                       "reconstruction_time": None,
                       "mips_processing_time": None
                       }

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

        if model not in self.models.keys():
            raise ValueError(f"model must be one of {self.models.keys()}, but was {model:s}")
        self.model = model

        # ########################
        # images
        # ########################
        if not isinstance(imgs_raw, da.core.Array):
            raise ValueError(f"imgs_raw should be a dask array, but was {type(imgs_raw)}")

        if imgs_raw.chunksize[-2:] != imgs_raw.shape[-2:]:
            raise ValueError(f"imgs_raw chunksize along last two dimensions should match array size, but"
                             f"{imgs_raw.chunksize[-2:]} != {imgs_raw.shape[-2:]}")

        # convert to photons
        self.imgs_raw = (imgs_raw.astype(float) - offset) / gain
        self.npatterns, self.ny, self.nx = imgs_raw.shape[-3:]
        self.extra_shape = imgs_raw.shape[:-3]
        self.nextra_dims = imgs_raw.ndim - 3

        # ########################
        # background images
        # ########################
        self.imgs_raw_bg = None
        if imgs_raw_bg is not None:
            if not isinstance(imgs_raw_bg, da.core.Array):
                raise ValueError(f"imgs_raw_bg should be a dask array, but was {type(imgs_raw)}")

            if imgs_raw_bg.chunksize[-2:] != imgs_raw_bg.shape[-2:]:
                raise ValueError(f"imgs_raw_bg chunksize along last three dimensions should match array size, but"
                                 f"{imgs_raw_bg.chunksize[-2:]} != {imgs_raw_bg.shape[-2:]}")

            try:
                np.broadcast_shapes(self.imgs_raw.shape, imgs_raw_bg.shape)
            except ValueError:
                raise ValueError("foreground and background image shapes are not compatible")

            self.imgs_raw_bg = (imgs_raw_bg.astype(float) - offset) / gain

        # ########################
        # information about nd-array axes
        # ########################
        if axes_names is None:
            self.dimensions = [f"i{ii:d}" for ii in range(self.imgs_raw.ndim - 2)]
        else:
            self.dimensions = axes_names

        self.use_average_as_background = self.imgs_raw_bg is None

        if bg_average_axes is None or not self.use_average_as_background:
            bg_average_axes = ()
        self.bg_average_axes = bg_average_axes

        # ########################
        # beam frequencies
        # ########################
        # reference frequency
        # shape = n0 x ... x nm x 1 x 1 x 1 x 2 so either component is broadcastable with imgs
        # todo: check shape
        if reference_frq is None:
            self.reference_frq = None
            self.use_fixed_ref = False
        else:
            # todo: coerce to correct shape
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
            self.nmax_multiplex = int(np.max([f.shape[0] for f in self.hologram_frqs]))

        self.hologram_frqs_bg = None

        # ########################
        # correction parameters
        # parameter arrays should be broadcastable to same size as image arrays
        # ########################
        # self.phase_offsets = phase_offsets
        # logic
        self.fit_phases = fit_phases
        self.fit_translations = fit_translations
        self.translation_thresh = translation_thresh
        self.fit_phase_profile = fit_phase_profile
        self.phase_profile_l1 = phase_profile_l1
        self.save_auxiliary_fields = save_auxiliary_fields
        # arrays
        self.translations = np.zeros((1,) * self.nextra_dims + (self.npatterns, 1, 1, 2), dtype=float)
        self.translations_bg = np.zeros_like(self.translations)
        self.phase_params = np.ones((1,) * self.nextra_dims + (self.npatterns, 1, 1), dtype=complex)
        self.phase_params_bg = np.ones_like(self.phase_params)

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

        self.ctf = (np.sqrt(self.fxs[None, :] ** 2 + self.fys[:, None] ** 2) <= self.fmax).astype(complex)

        # ########################
        # values passed through to RI reconstruction
        # ########################
        self.step_start = step
        self.reconstruction_settings = reconstruction_kwargs

        # ########################
        # prepare zarr arrays to store results
        # ########################
        self.store.create("efields_ft",
                          shape=self.imgs_raw.shape,
                          chunks=(1,) * self.nextra_dims + self.imgs_raw.shape[-3:],
                          compressor=self.compressor,
                          dtype=np.complex64 if self.save_float32 else complex)

        
        bg_shape_ref = self.imgs_raw_bg.shape if self.imgs_raw_bg is not None else self.imgs_raw.shape
        self.store.create("efield_bg_ft",
                          shape=tuple([s if ii not in self.bg_average_axes
                                       else 1
                                       for ii, s in enumerate(bg_shape_ref)]),
                          chunks=(1,) * self.nextra_dims + self.imgs_raw.shape[-3:],
                          compressor=self.compressor,
                          dtype=np.complex64 if self.save_float32 else complex)

        self.store.create("phase_correction_profile",
                          shape=self.imgs_raw.shape[:-3] + (1, self.ny, self.nx),
                          chunks=(1,) * self.nextra_dims + (1, self.ny, self.nx),
                          compressor=self.compressor,
                          dtype=np.complex64 if self.save_float32 else complex)


        if not hasattr(self.store, "n"):
            self.store.create("n",
                              shape=self.imgs_raw.shape[:-3] + self.n_shape,
                              chunks=(1,) * self.nextra_dims + self.n_shape,
                              compressor=self.compressor,
                              dtype=np.complex64 if self.save_float32 else complex
                              )

        if not hasattr(self.store, "efwd"):
            self.store.create("efwd",
                              shape=self.imgs_raw.shape,
                              chunks=(1,) * (self.imgs_raw.ndim - 2) + (self.ny, self.nx),
                              compressor=self.compressor,
                              dtype=np.complex64 if self.save_float32 else complex)

        if not hasattr(self.store, "escatt"):
            self.store.create("escatt",
                              shape=self.imgs_raw.shape[:-3] + (self.nmax_multiplex, self.ny, self.nx),
                              chunks=(1,) * (self.imgs_raw.ndim - 2) + (self.ny, self.nx),
                              compressor=self.compressor,
                              dtype=complex)

        if not hasattr(self.store, "n_start"):
            if self.n_guess is None:
                self.store.create("n_start",
                                  shape=self.imgs_raw.shape[:-3] + self.n_shape,
                                  chunks=(1,) * self.nextra_dims + self.n_shape,
                                  compressor=self.compressor,
                                  dtype=np.complex64 if self.save_float32 else complex)
            else:
                self.store.array("n_start",
                                 np.expand_dims(self.n_guess, axis=list(range(self.nextra_dims))),
                                 compressor=self.compressor,
                                 dtype=np.complex64 if self.save_float32 else complex)

        if "max_iterations" not in self.reconstruction_settings.keys():
            raise ValueError()

        if not hasattr(self.store, "costs"):
            self.store.create("costs",
                              shape=self.imgs_raw.shape[:-3] +
                                    (self.reconstruction_settings["max_iterations"] + 1, self.npatterns),
                              chunks=(1,) * (self.imgs_raw.ndim - 3) +
                                     (self.reconstruction_settings["max_iterations"] + 1, self.npatterns),
                              compressor=self.compressor,
                              dtype=float)

        if not hasattr(self.store, "steps"):
            self.store.create("steps",
                              shape=self.imgs_raw.shape[:-3] +
                                    (self.reconstruction_settings["max_iterations"],),
                              chunks=(1,) * (self.imgs_raw.ndim - 3) +
                                     (self.reconstruction_settings["max_iterations"],),
                              compressor=self.compressor,
                              dtype=float)

        # ########################
        # ROI and affine transformations
        # ########################
        # for reconstruction, using FFT induced coordinates, i.e. zero is at array index (ny // 2, nx // 2)
        # for matrix, using image coordinates (0, 1, ..., N - 1)
        # note, due to order of operations -n//2 =/= - (n//2) when nx is odd
        if self.model in ["born", "rytov"]:
            affine_xform_recon_pix2coords = params2xform([self.drs_n[-2], 0, -(self.n_shape[-2] // 2) * self.drs_n[-2],
                                                          self.drs_n[-1], 0, -(self.n_shape[-1] // 2) * self.drs_n[-1]])
        else:
            nbin_y = 1
            nbin_x = 1

            if (self.drs_n[1] != self.dxy * nbin_x or
                self.drs_n[2] != self.dxy * nbin_y):
                raise ValueError()

            # affine transformation from reconstruction coordinates to pixel indices
            # coordinates in finer coordinates
            xb = bin(self.x, [nbin_x], mode="mean")
            yb = bin(self.y, [nbin_y], mode="mean")

            affine_xform_recon_pix2coords = params2xform([self.drs_n[-2], 0, float(yb[0]),
                                                          self.drs_n[-1], 0, float(xb[0])])

        # ############################
        # construct affine tranformations
        # ############################
        # transform from reconstruction coordinates in um to pixel indices
        xform_raw_roi_pix2coords = params2xform([self.dxy, 0, -(self.n_shape[-2] // 2) * self.dxy,
                                                 self.dxy, 0, -(self.n_shape[-1] // 2) * self.dxy])

        # transform from recon pixels to data roi pixels
        xform_recon2raw_roi = inv(xform_raw_roi_pix2coords).dot(affine_xform_recon_pix2coords)

        xform_odt_recon_to_cam_roi = None
        xform_odt_recon_to_full = None
        if data_roi is not None:
            # transform from data roi pixels to camera roi pixels
            xform_process_roi_to_cam_roi = params2xform([1, 0, data_roi[0],
                                                         1, 0, data_roi[2]])
            xform_odt_recon_to_cam_roi = xform_process_roi_to_cam_roi.dot(xform_recon2raw_roi)

            if cam_roi is not None:
                # transform from camera roi to full camera chip
                xform_cam_roi_to_full = params2xform([1, 0, cam_roi[0],
                                                      1, 0, cam_roi[2]])
                xform_odt_recon_to_full = xform_cam_roi_to_full.dot(xform_process_roi_to_cam_roi)

        # store all transforms in JSON serializable form
        self.xform_dict = {"affine_xform_recon_pix2coords": np.asarray(affine_xform_recon_pix2coords).tolist(),
                           "affine_xform_recon_2_raw_data_roi": np.asarray(xform_recon2raw_roi).tolist(),
                           "affine_xform_recon_2_raw_camera_roi": np.asarray(xform_odt_recon_to_cam_roi).tolist(),
                           "affine_xform_recon_2_raw_camera": np.asarray(xform_odt_recon_to_full).tolist(),
                           "data_roi": np.array(data_roi).tolist(),
                           "camera_roi": np.array(cam_roi).tolist(),
                           "coordinate_order": "yx"
                           }


    @classmethod
    def load_file(cls,
                  location: Union[str, Path, zarr.hierarchy.Group]):
        """
        Instantiate class from zarr store

        :param location:
        :return instance:
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
        if hasattr(z, "efields_ft"):
            kwargs["imgs_raw"] = da.zeros(z.efields_ft.shape,
                                          chunks=z.efields_ft.chunks,
                                          dtype=float)
        else:
            kwargs["imgs_raw"] = da.zeros((1,) * z.attrs["nextra_dims"] +
                                          (z.attrs["npatterns"], z.attrs["ny"], z.attrs["nx"]),
                                          chunks=(1,) * z.attrs["nextra_dims"] +
                                          (z.attrs["npatterns"], z.attrs["ny"], z.attrs["nx"])
                                          )

        # load kwargs passed through to other functions
        kwargs.update(z.attrs["reconstruction_settings"])

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

        # ###########################
        # convert some types as needed
        # ###########################
        inst.hologram_frqs = [inst.hologram_frqs[..., ii, :, :] for ii in range(inst.hologram_frqs.shape[-3])]
        inst.hologram_frqs_bg = [inst.hologram_frqs_bg[..., ii, :, :] for ii in range(inst.hologram_frqs_bg.shape[-3])]
        inst.n_shape = tuple(inst.n_shape)

        return inst

    @staticmethod
    def prepare_img_data(data_dir: Union[Path, str],
                         zarr_pattern: str,
                         component: str,
                         roi: Optional[Sequence[int]] = None,
                         slice_axis: Optional[int] = None,
                         n_every: int = 1,
                         bg_axis: Optional[int] = None,
                         bg_index: Optional[int] = None,
                         one_chunk_per_volume: bool = True,
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
        if one_chunk_per_volume:
            imgs = da.from_zarr(zc, chunks=(1,) * (zc.ndim - 3) + zc.shape[-3:])[slices]
        else:
            imgs = da.from_zarr(zc, chunks=(1,) * (zc.ndim - 2) + zc.shape[-2:])[slices]

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
             attributes: Optional[dict] = None):
        """
        Save information from instance to zarr file, except for dask arrays. NumPy arrays are
        stored as new zarr arrays. Other data types are stored as zarr attributes

        :param attributes: dictionary of additional items to be saved as zarr arrays or attributes
        :return:
        """

        if attributes is None:
            attributes = {}

        # store beam frequencies
        try:
            stack = np.stack(self.get_beam_frqs(), axis=-3)
        except np.AxisError:
            stack = None

        self.store.array("beam_frqs",
                         stack,
                         compressor=self.compressor,
                         dtype=float)

        # store everything else
        for dictionary in [self.__dict__, attributes]:
            for k, v in dictionary.items():
                if k in ["hologram_frqs", "hologram_frqs_bg"]:
                    try:
                        stack = np.stack(getattr(self, k), axis=-3)
                    except TypeError:
                        stack = None

                    self.store.array(k,
                                     stack,
                                     compressor=self.compressor,
                                     dtype=float)
                    continue

                try:
                    if isinstance(v, array):
                        if v.size > 10:
                            # todo: ensure reasonable chunk size
                            self.store.array(k,
                                             to_cpu(v),
                                             compressor=packbits.PackBits() if v.dtype == bool else self.compressor,
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
                    elif isinstance(v, (dask.array.Array, Codec, zarr.hierarchy.Group)):
                        pass
                    else:
                        if self.verbose:
                            print(f"did not save key {k:s} of unsupported type {type(v)}")

                except TypeError as e:
                    print(f"{k:s} {e}")

    def save_projections(self,
                         overwrite: bool = True,
                         **kwargs
                         ):
        """
        Store orthogonal projections of the refractive index

        :param overwrite: whether to overwrite max-z projections if they already exist
        :return:
        """

        tstart_proj = perf_counter()

        future = []
        for axis, label in zip([-3, -2, -1],
                               ["z", "y", "x"]):
            future.append(da.max(da.real(da.from_zarr(self.store.n)), axis=axis).to_zarr(self.store.store.path,
                                                                                         component=f"n_max{label:s}",
                                                                                         compute=False,
                                                                                         compressor=self.compressor,
                                                                                         overwrite=overwrite)
                          )

        cluster = LocalCluster(**kwargs)
        client = Client(cluster)
        dask.compute(*future)
        del client
        del cluster

        self.timing["mips_processing_time"] = perf_counter() - tstart_proj
        if self.verbose:
            print(f"MIPs time: {parse_time(self.timing['mips_processing_time'])[1]:s}")


    def estimate_hologram_frqs(self,
                               save: bool = False,
                               processes: bool = True,
                               n_workers: int = -1,
                               use_gpu: bool=cp is not None,
                               ) -> None:
        """
        Estimate hologram frequencies from raw images. The current value of self.hologram_frqs() is used as the guess.
        Guess values usually need to be within a few pixels for fitting to succeed. This accuracy is usually
        achievable by looking at the Fourier transform of the hologram. To display the Fourier transform,
        use the function plot_image()

        :param save:
        :param processes:
        :param n_workers:
        :param use_gpu:
        :return:
        """

        if save and self.save_dir is not None:
            frq_dir = self.save_dir / f"frq_fits"
            frq_dir.mkdir(exist_ok=True)

        tstart_est_frqs = perf_counter()

        # todo: why not combine this function with unmix_holograms()?
        def fit_rois_cpu(img: array,
                         frqs_guess: array,
                         roi_size_pix,
                         fx: array,
                         fy: array,
                         fmax: float,
                         model=gauss2d_symm(),
                         save_dir=None,
                         prefix: str = "",
                         figsize: Sequence[float, float] = (20., 10.),
                         gamma: float = 0.2,
                         use_gpu: bool = True,
                         block_id=None,
                         block_info=None,
                         **fig_kwargs,
                         ):

            if isinstance(img, dask.array.Array):
                img = img.compute()

            # grab info about dimensions
            nextra_dims = img.ndim - 3
            pattern_offset = block_id[-3]
            nmulti = frqs_guess.shape[-2]
            npatterns, ny, nx = img.shape[-3:]

            # fourier transform image
            xp = cp if use_gpu and cp else np
            img_ft = to_cpu(ft2(xp.asarray(img) *
                                xp.asarray(hann(ny)[:, None]) *
                                xp.asarray(hann(nx)[None, :])))

            # get ROIs
            dfx = fx[1] - fx[0]
            dfy = fy[1] - fy[0]
            c_guess = np.stack((np.round((frqs_guess[..., 1] - fy[0]) / dfy),
                                np.round((frqs_guess[..., 0] - fx[0]) / dfx)
                                ),
                               axis=-1).astype(int)

            rois_all = get_centered_rois(c_guess,
                                         [roi_size_pix, roi_size_pix],
                                         min_vals=(0, 0),
                                         max_vals=(self.ny, self.nx)
                                         )

            # fit centers
            centers = np.zeros(img.shape[:-2] + (nmulti, 2,), dtype=float)
            for ii in range(np.prod(centers.shape[:-1])):
                ind = np.unravel_index(ii, centers.shape[:-1])
                ind_img = ind[:-1]
                roi = rois_all[ind[-2:]]

                img_roi = abs(np.stack(cut_roi(roi, img_ft[ind_img]), axis=-3))[0]

                fxfx, fyfy = np.meshgrid(fx[roi[2]:roi[3]], fy[roi[0]:roi[1]])

                rgauss = model.fit(img_roi,
                                   (fyfy, fxfx),
                                   init_params=None,
                                   guess_bounds=True)
                centers[ind] = rgauss["fit_params"][1:3]

            # plot centers
            if save_dir is not None and np.all(np.array(block_id[:-3]) == 0):
                with catch_warnings():
                    simplefilter("ignore")
                    with rc_context({'interactive': False,
                                     'backend': "agg"}):

                        for aa in range(npatterns):
                            figh = plt.figure(figsize=figsize, **fig_kwargs)
                            ax = figh.add_subplot(1, 1, 1)
                            ax.set_title("$|I(f)|$")
                            im = ax.imshow(np.abs(img_ft).squeeze(axis=tuple(range(nextra_dims)))[aa],
                                           extent=[fx[0] - 0.5 * dfx,
                                                   fx[-1] + 0.5 * dfx,
                                                   fy[-1] + 0.5 * dfy,
                                                   fy[0] - 0.5 * dfy],
                                           norm=PowerNorm(gamma=gamma),
                                           cmap="bone")
                            plt.colorbar(im)

                            fy_min = fy[rois_all[aa, :, 0].min()] - fmax
                            fy_max = fy[rois_all[aa, :, 1].max()] + fmax
                            fx_min = fx[rois_all[aa, :, 2].min()] - fmax
                            fx_max = fx[rois_all[aa, :, 3].max()] + fmax
                            ax.set_xlim([fx_min, fx_max])
                            ax.set_ylim([fy_max, fy_min])

                            # plot fit and guess frqs
                            ax.plot(frqs_guess[aa, :, 0].squeeze(),
                                    frqs_guess[aa, :, 1].squeeze(),
                                    'gx')
                            ax.plot(centers.squeeze(axis=tuple(range(nextra_dims)))[aa, :, 0],
                                    centers.squeeze(axis=tuple(range(nextra_dims)))[aa, :, 1],
                                    'rx')

                            for bb in range(nmulti):
                                roi = rois_all[aa, bb]
                                ax.add_artist(Rectangle((fx[roi[2]] - 0.5 * dfx, fy[roi[0]] - 0.5 * dfy),
                                                        fx[roi[3] - 1] - fx[roi[2]],
                                                        fy[roi[1] - 1] - fy[roi[0]],
                                                        edgecolor='k',
                                                        fill=False))

                            ax.set_xlabel("$f_x$ (1/$\\mu m$)")
                            ax.set_ylabel("$f_y$ (1/$\\mu m$)")

                            if save_dir is not None:
                                figh.savefig(Path(save_dir, f"{prefix:s}pattern={aa + pattern_offset:d}=hologram_frq_diagnostic.png"))
                                plt.close(figh)

            del img_ft

            return centers

        # #####################################
        # calibrate frequencies
        # #####################################
        hologram_frqs_guess = np.stack(self.hologram_frqs, axis=0)

        if self.use_fixed_holo_frequencies:
            slices_fit = tuple([slice(0, 1) for _ in range(self.nextra_dims)])
        else:
            slices_fit = tuple([slice(None) for _ in range(self.nextra_dims)])

        frqs_hologram = None
        if not self.use_fixed_holo_frequencies or self.use_average_as_background:
            print("fitting foreground frequencies")
            r = map_blocks_joblib(fit_rois_cpu,
                                  self.imgs_raw[slices_fit],
                                  hologram_frqs_guess,
                                  self.freq_roi_size_pix,
                                  self.fxs,
                                  self.fys,
                                  self.fmax,
                                  use_gpu=use_gpu,
                                  save_dir=frq_dir if save and self.use_average_as_background else None,
                                  n_workers=n_workers,
                                  processes=processes,
                                  chunks=self.imgs_raw.chunksize[:-2] + (None, None),
                                  )
            frqs_hologram = np.stack(r, axis=0).reshape(self.imgs_raw.shape[:-2] + (self.nmax_multiplex, 2))

        frqs_hologram_bg = None
        if not self.use_average_as_background:
            print("fitting background frequencies")
            r = map_blocks_joblib(fit_rois_cpu,
                                  self.imgs_raw_bg[slices_fit],
                                  hologram_frqs_guess,
                                  self.freq_roi_size_pix,
                                  self.fxs,
                                  self.fys,
                                  self.fmax,
                                  use_gpu=use_gpu,
                                  prefix="bg_",
                                  save_dir=frq_dir if save else None,
                                  n_workers=n_workers,
                                  processes=processes,
                                  chunks=self.imgs_raw_bg.chunksize[:-2] + (None, None),
                                  )
            frqs_hologram_bg = np.stack(r, axis=0).reshape(self.imgs_raw_bg.shape[:-2] + (self.nmax_multiplex, 2))

        get_reusable_executor().shutdown(wait=True)

        if self.use_average_as_background:
            frqs_hologram_bg = frqs_hologram

        if self.use_fixed_holo_frequencies:
            frqs_hologram = frqs_hologram_bg

        # self.hologram_frqs = frqs_hologram
        # self.hologram_frqs_bg = frqs_hologram_bg
        # if not self.use_fixed_ref:
        #     self.reference_frq = np.mean(self.hologram_frqs, axis=(-2, -3))
        #     self.reference_frq_bg = np.mean(self.hologram_frqs_bg, axis=(-2, -3))
        self.hologram_frqs = [frqs_hologram[..., ii, :, :] for ii in range(self.npatterns)]
        self.hologram_frqs_bg = [frqs_hologram_bg[..., ii, :, :] for ii in range(self.npatterns)]
        if not self.use_fixed_ref:
            self.reference_frq = np.expand_dims(np.mean(np.concatenate(self.hologram_frqs,
                                                                       axis=-2),
                                                        axis=-2),
                                                axis=(-2, -3, -4))
            self.reference_frq_bg = np.expand_dims(np.mean(np.concatenate(self.hologram_frqs_bg,
                                                                          axis=-2),
                                                           axis=-2),
                                                   axis=(-2, -3, -4))

        # store and print timing information
        self.timing["fit_frequency_time"] = perf_counter() - tstart_est_frqs
        if self.verbose:
            print(f"calibration time: {parse_time(self.timing['fit_frequency_time'])[1]:s}")

    def get_beam_frqs(self) -> list[np.ndarray]:
        """
        Get beam incident beam frequencies from hologram frequencies and reference frequency

        :return beam_frqs: list (length n_patterns) with each element an array of size N1 x N2 ... x Nm x 3
        """

        bxys = [f - self.reference_frq.squeeze(axis=(-2, -3)) for f in self.hologram_frqs]
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

        mean_ref_frq = np.mean(self.reference_frq.squeeze(axis=(-2, -3, -4)), axis=tuple(range(self.nextra_dims)))

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

        # map pupil positions to frequency
        frqs_from_pupil = xform_points(centers_dmd, xform_dmd2frq)
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

        # DMD boundary
        if dmd_size is not None:
            south = np.zeros((nx_dmd, 2))
            south[:, 0] = np.arange(nx_dmd)            

            north = np.zeros((nx_dmd, 2))
            north[:, 0] = np.arange(nx_dmd)
            north[:, 1] = ny_dmd - 1

            east = np.zeros((ny_dmd, 2))
            east[:, 0] = nx_dmd - 1
            east[:, 1] = np.arange(ny_dmd)

            west = np.zeros((ny_dmd, 2))            
            west[:, 1] = np.arange(ny_dmd)

            dmd_boundary = np.concatenate((south, north, east, west), axis=0)
            dmd_boundary_freq = xform_points(dmd_boundary, xform_dmd2frq)

        # ##############################
        # plot data
        # ##############################
        context = {}
        if not interactive:
            context['interactive'] = False
            context['backend'] = "agg"

        with rc_context(context):            
            figh = plt.figure(figsize=(20, 8))
            grid = figh.add_gridspec(1, 2)
            figh.suptitle("Mapping from pupil (DMD surface) to hologram frequencies (in object space)\n"
                          f"Reference freq = ({mean_ref_frq[0]:.3f}, {mean_ref_frq[1]:.3f}) $1/\\mu m$,"
                          f" central mirror = ({center_pupil_frq_ref[0]:.1f}, {center_pupil_frq_ref[1]:.1f})")

            # plot in reference space
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
            ax.plot(center_pupil_frq_ref[0],
                    center_pupil_frq_ref[1],
                    "m3",
                    label="reference freq")
            ax.plot(centers_dmd_fmax[:, 0],
                    centers_dmd_fmax[:, 1],
                    'k',
                    label="pupil")
            
            xmax_bound = max([np.max(centers_dmd[:, 0]), np.max(centers_pupil_from_frq[..., 0]), np.max(centers_dmd_fmax[:, 0])])
            xmin_bound = min([np.min(centers_dmd[:, 0]), np.min(centers_pupil_from_frq[..., 0]), np.min(centers_dmd_fmax[:, 0])])
            ymax_bound = max([np.max(centers_dmd[:, 1]), np.max(centers_pupil_from_frq[..., 1]), np.max(centers_dmd_fmax[:, 1])])
            ymin_bound = min([np.min(centers_dmd[:, 1]), np.min(centers_pupil_from_frq[..., 1]), np.min(centers_dmd_fmax[:, 1])])
            ax.set_xlim([xmin_bound, xmax_bound])
            ax.set_ylim([ymin_bound, ymax_bound])
            ax.legend(bbox_to_anchor=(0.2, 1.1))
            ax.set_xlabel("x-position (mirrors)")
            ax.set_ylabel("y-position (mirrors)")

            # plot in frequency space
            ax = figh.add_subplot(grid[0, 1])
            ax.axis("scaled")
            ax.set_title("Raw frequencies")

            if dmd_size is not None:
                ax.plot(dmd_boundary_freq[:, 0], dmd_boundary_freq[:, 1], 'k.')

            ax.plot(mean_hologram_frqs[..., 0], mean_hologram_frqs[..., 1], 'rx')
            ax.plot(frqs_from_pupil[..., 0], frqs_from_pupil[..., 1], 'b.')
            ax.plot(mean_ref_frq[0], mean_ref_frq[1], "m3")
            ax.add_artist(Circle(mean_ref_frq,
                                 radius=self.fmax,
                                 facecolor="none",
                                 edgecolor="k"))
            ax.set_xlim([-self.fmax + mean_ref_frq[0], self.fmax + mean_ref_frq[0]])
            ax.set_ylim([self.fmax + mean_ref_frq[1], -self.fmax + mean_ref_frq[1]])
            ax.set_xlabel("$f_x$ (1/$\\mu m$)")
            ax.set_ylabel("$f_y$ (1/$\\mu m$)")

            if save and self.save_dir is not None:
                figh.savefig(self.save_dir / f"frequency_mapping.png")
                plt.close(figh)

        return xform_dmd2frq

    def unmix_holograms_v2(self,
                           processes: bool = False,
                           n_workers: int = -1,
                           use_gpu: bool = False,
                           ):

        tstart_holos = perf_counter()

        def calibrate(imgs,
                      hft_ref,
                      fx_ref,
                      fy_ref,
                      dxy,
                      fmax,
                      threshold,
                      apodization=None,
                      eft_out=None,
                      phase_corr_out=None,
                      fit_phases=True,
                      fit_translations=True,
                      average_axes=None,
                      use_gpu=False,
                      block_id=None,
                      block_info=None,
                      phase_corr_kwargs=None,
                      ):

            if isinstance(imgs, dask.array.Array):
                imgs = imgs.compute()

            xp = cp if use_gpu and cp else np
            n_used_dims = imgs.ndim - np.min([ii for ii, s in enumerate(imgs.shape) if s != 1])
            if n_used_dims != 2 and n_used_dims != 3:
                raise ValueError("calibrate only supports arrays with 2 or 3 non-singleton dimensions")

            squeeze_ax = tuple(range(imgs.ndim - n_used_dims))

            hft_ref = xp.asarray(hft_ref.squeeze(axis=squeeze_ax))

            # unmix holograms
            hft = unmix_hologram(xp.asarray(imgs.squeeze(axis=squeeze_ax)),
                                 dxy,
                                 2*fmax,
                                 fx_ref.squeeze(axis=squeeze_ax),
                                 fy_ref.squeeze(axis=squeeze_ax),
                                 apodization)

            # translation correction
            if fit_translations:
                translations = fit_phase_ramp(ft2(abs(ift2(hft))),
                                              ft2(abs(ift2(hft_ref))),
                                              dxy,
                                              threshold)

                fx_bcastable = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(hft.shape[-1], dxy)), axis=-2)
                fy_bcastable = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(hft.shape[-2], dxy)), axis=-1)
                hft *= np.exp(2 * np.pi * 1j * (fx_bcastable * translations[..., 0] +
                                                fy_bcastable * translations[..., 1]))
            else:
                translations = np.zeros(hft.shape[:-2] + (1, 1, 2))

            # global phase correction
            if fit_phases:
                phase_params = get_global_phase_shifts(hft, hft_ref)
                hft *= phase_params
            else:
                phase_params = np.ones(hft.shape[:-2] + (1, 1), dtype=complex)

            # other phase correction
            if phase_corr_out is not None:
                if n_used_dims != 3:
                    raise ValueError("Can only run PhaseCorr if each chunk includes all angles")

                pc = PhaseCorr(ift2(hft),
                               ift2(hft_ref),
                               **phase_corr_kwargs)
                rpc = pc.run(xp.ones(hft.shape[-2:], dtype=complex),
                             **phase_corr_kwargs
                             )
                phase_prof = rpc["x"]
                phase_corr_out[block_id[:-n_used_dims] + (0,)] = to_cpu(phase_prof)
            else:
                phase_prof = 1

            # save results
            hft_out = to_cpu(ft2(phase_prof * ift2(hft)))
            if average_axes is None:
                eft_out[block_id[:-n_used_dims]] = hft_out
                hft_out = None

            if use_gpu:
                cp.fft.config.get_plan_cache().clear()

            return to_cpu(translations), to_cpu(phase_params), hft_out, block_info

        # choose background image
        if self.use_average_as_background:
            imgs_raw_bg = self.imgs_raw
        else:
            imgs_raw_bg = self.imgs_raw_bg

        # get reference holograms
        ref_slice = tuple([slice(0, 1) if a in self.bg_average_axes
                           else slice(None)
                           for a in range(self.nextra_dims)])

        hft_ref = unmix_hologram(imgs_raw_bg[ref_slice].compute(),
                                 self.dxy,
                                 2*self.fmax,
                                 self.reference_frq_bg[..., 0][ref_slice],
                                 self.reference_frq_bg[..., 1][ref_slice],
                                 self.apodization)

        # correct background
        print("correcting background")
        rbg = map_blocks_joblib(calibrate,
                                imgs_raw_bg,
                                hft_ref,
                                self.reference_frq_bg[..., 0],
                                self.reference_frq_bg[..., 1],
                                self.dxy,
                                self.fmax,
                                threshold=self.translation_thresh,
                                apodization=self.apodization,
                                eft_out=self.store.efield_bg_ft,
                                fit_phases=self.fit_phases,
                                fit_translations=self.fit_translations,
                                average_axes=self.bg_average_axes,
                                use_gpu=use_gpu,
                                chunks=imgs_raw_bg.chunksize,
                                n_workers=n_workers,
                                processes=processes,
                                return_generator=True,
                                )
        ts = []
        ps = []
        ef_bg_ft_temp = np.zeros(self.store.efield_bg_ft.shape, dtype=complex)
        nbg = np.prod([d for ii, d in enumerate(self.extra_shape) if ii in self.bg_average_axes])
        for data in rbg:
            t, p, img, block_info = data
            ts.append(t)
            ps.append(p)
            slice_now = tuple([slice(a[0], a[1]) for a in block_info[0]['array-location']])
            ef_bg_ft_temp[slice_now] += img / nbg

        # t, p = zip(*rbg)
        self.translations_bg = np.stack(ts, axis=0).reshape(imgs_raw_bg.shape[:-2] + (1, 1, 2))
        self.phase_params_bg = np.stack(ps, axis=0).reshape(imgs_raw_bg.shape[:-2] + (1, 1))
        self.store.efield_bg_ft[:] = ef_bg_ft_temp

        # correct foreground
        print("correcting foreground")
        # todo: more logic for using average as own background?
        r = map_blocks_joblib(calibrate,
                              self.imgs_raw,
                              da.from_zarr(self.store.efield_bg_ft),
                              self.reference_frq[..., 0],
                              self.reference_frq[..., 1],
                              self.dxy,
                              self.fmax,
                              threshold=self.translation_thresh,
                              apodization=self.apodization,
                              eft_out=self.store.efields_ft,
                              phase_corr_out=self.store.phase_correction_profile if self.fit_phase_profile else None,
                              fit_phases=self.fit_phases,
                              fit_translations=self.fit_translations,
                              use_gpu=use_gpu,
                              phase_corr_kwargs={"tau_l1": self.phase_profile_l1,
                                                 "escale": 40.,
                                                 "fit_magnitude": True,
                                                 "step": 1e5,
                                                 "max_iterations": 10,
                                                 "line_search_iter_limit": 3,
                                                 "n_batch": self.reconstruction_settings["n_batch"],
                                                 "compute_batch_grad_parallel": True,
                                                 },
                              chunks=((1,) * self.nextra_dims + (self.npatterns, self.ny, self.nx))
                                      if self.fit_phase_profile else imgs_raw_bg.chunksize,
                              n_workers=n_workers,
                              processes=processes
                              )

        t, p, _, _ = zip(*r)
        self.translations = np.stack(t, axis=0).reshape(self.imgs_raw.shape[:-2] + (1, 1, 2))
        self.phase_params = np.stack(p, axis=0).reshape(self.imgs_raw.shape[:-2] + (1, 1))

        get_reusable_executor().shutdown(wait=True)

        self.timing["unmix_hologram_time"] = perf_counter() - tstart_holos
        if self.verbose:
            print(f"unmixed holograms and fit phases in {parse_time(self.timing['unmix_hologram_time'])[1]}")

    def unmix_holograms(self,
                        use_gpu: bool = False,
                        processes: bool = False,
                        **kwargs):
        """
        Unmix and preprocess holograms. Additional kwargs are passed through to dask LocalCluster()

        :param use_gpu:
        :param processes:
        :return:
        """
        tstart_holos = perf_counter()

        def _ft_abs(m: array) -> array:
            return ft2(abs(ift2(m)))

        def translate(e_ft, dxs, dys, fxs, fys):
            e_ft_out = xp.array(e_ft, copy=True)

            fx_bcastable = xp.expand_dims(fxs, axis=(-3, -2))
            fy_bcastable = xp.expand_dims(fys, axis=(-3, -1))

            e_ft_out *= np.exp(2 * np.pi * 1j * (fx_bcastable * dxs +
                                                 fy_bcastable * dys))

            return e_ft_out

        def correct_phase_profile(eft, ebg_ft, block_id=None):
            nextra_dims = eft.ndim - 3
            extra_dims = tuple(range(eft.ndim - 3))

            phase_corr_kwargs = {"tau_l1": self.phase_profile_l1,
                                 "escale": 40.,
                                 "fit_magnitude": True,
                                 "step": 1e5,
                                 "max_iterations": 10,
                                 "line_search_iter_limit": self.reconstruction_settings["n_batch"],
                                 "n_batch": 3,
                                 "compute_batch_grad_parallel": True,
                                 }

            pc = PhaseCorr(ift2(eft.squeeze(extra_dims)),
                           ift2(ebg_ft.squeeze(extra_dims)),
                           **phase_corr_kwargs)
            rpc = pc.run(xp.ones(eft.shape[-2:], dtype=complex),
                         **phase_corr_kwargs
                         )

            return rpc["x"].reshape((1,) * nextra_dims + (1,) + eft.shape[-2:])

        xp = cp if use_gpu and cp else np

        with LocalCluster(processes=processes, **kwargs) as cluster, Client(cluster) as client:
            if self.verbose:
                print(cluster.dashboard_link)

            if self.use_average_as_background:
                imgs_raw_bg = self.imgs_raw
            else:
                imgs_raw_bg = self.imgs_raw_bg

            # get electric field from holograms
            ref_frq_bg_da = da.from_array(self.reference_frq_bg,
                                          chunks=imgs_raw_bg.chunksize[:-2] + (1, 1, 2)
                                          )

            hft_bg = da.map_blocks(unmix_hologram,
                                            imgs_raw_bg,
                                            self.dxy,
                                            2*self.fmax,
                                            ref_frq_bg_da[..., 0],
                                            ref_frq_bg_da[..., 1],
                                            apodization=self.apodization,
                                            dtype=complex)

            # slice used as reference for computing phase shifts/translations/etc.
            # if we are going to average along a dimension (i.e. if it is in bg_average_axes) then need to use
            # single slice as background for that dimension.
            ref_slice = tuple([slice(0, 1) if a in self.bg_average_axes
                               else slice(None)
                               for a in range(self.nextra_dims)])

            # #########################
            # compute background
            # #########################
            if self.fit_translations:
                # fit translations between signal and background electric fields
                holograms_abs_ft_bg = da.map_blocks(_ft_abs, hft_bg, dtype=complex)
                self.translations_bg = da.map_blocks(fit_phase_ramp,
                                                     holograms_abs_ft_bg,
                                                     holograms_abs_ft_bg[ref_slice],
                                                     self.dxy,
                                                     thresh=self.translation_thresh,
                                                     dtype=float,
                                                     new_axis=-1,
                                                     chunks=holograms_abs_ft_bg.chunksize[:-2] + (1, 1, 2)).compute()

                hft_bg = da.map_blocks(translate,
                                       hft_bg,
                                       da.from_array(self.translations_bg[..., 0], chunks=hft_bg.chunksize[:-2] + (1, 1)),
                                       da.from_array(self.translations_bg[..., 1], chunks=hft_bg.chunksize[:-2] + (1, 1)),
                                       self.fxs,
                                       self.fys,
                                       dtype=complex,
                                       meta=xp.array((), dtype=complex))

            # determine phase offsets for background electric field, relative to initial slice
            # for each angle, so we can average this together to produce a single "background" image
            if self.fit_phases:
                self.phase_params_bg = da.map_blocks(get_global_phase_shifts,
                                                     hft_bg,
                                                     hft_bg[ref_slice],  # reference slices
                                                     dtype=complex,
                                                     chunks=hft_bg.chunksize[:-2] + (1, 1)
                                                     ).compute()
                hft_bg = hft_bg * self.phase_params_bg

            # determine background electric field
            print("computing background electric field")
            hft_bg_comp = da.mean(hft_bg,
                                  axis=self.bg_average_axes,
                                  keepdims=True)

            print("computing background")
            hft_bg_arr = hft_bg_comp.compute()
            self.store.efield_bg_ft[:] = hft_bg_arr
            hft_bg_avg = da.from_array(hft_bg_arr,
                                       chunks=hft_bg_comp.chunksize)


            # #########################
            # compute foreground
            # #########################
            if self.use_average_as_background:
                hft = hft_bg
                self.translations = self.translations_bg
                self.phase_params = self.phase_params_bg
            else:
                ref_frq_da = da.from_array(self.reference_frq,
                                           chunks=self.imgs_raw.chunksize[:-2] + (1, 1, 2)
                                           )
                hft = da.map_blocks(unmix_hologram,
                                             self.imgs_raw,
                                             self.dxy,
                                             2*self.fmax,
                                             ref_frq_da[..., 0],
                                             ref_frq_da[..., 1],
                                             apodization=self.apodization,
                                             dtype=complex)

                if self.fit_translations:
                    # fit phase ramp in holograms ft
                    holograms_abs_ft = da.map_blocks(_ft_abs, hft, dtype=complex)
                    self.translations = da.map_blocks(fit_phase_ramp,
                                                      holograms_abs_ft,
                                                      _ft_abs(hft_bg_avg),
                                                      # holograms_abs_ft_bg[ref_slice],
                                                      self.dxy,
                                                      thresh=self.translation_thresh,
                                                      dtype=float,
                                                      new_axis=-1,
                                                      chunks=holograms_abs_ft.chunksize[:-2] + (1, 1, 2)).compute()

                    # correct translations
                    hft = da.map_blocks(translate,
                                        hft,
                                        da.from_array(self.translations[..., 0],
                                                      chunks=hft.chunksize[:-2] + (1, 1)),
                                        da.from_array(self.translations[..., 1],
                                                      chunks=hft.chunksize[:-2] + (1, 1)),
                                        self.fxs,
                                        self.fys,
                                        dtype=complex,
                                        meta=xp.array((), dtype=complex))

                if self.fit_phases:
                    self.phase_params = da.map_blocks(get_global_phase_shifts,
                                                      hft,
                                                      hft_bg_avg,
                                                      dtype=complex,
                                                      chunks=hft.chunksize[:-2] + (1, 1),
                                                      ).compute()
                    hft = hft * self.phase_params

            if self.fit_phase_profile:
                phase_prof = da.map_blocks(correct_phase_profile,
                                           hft,
                                           hft_bg_avg,
                                           drop_axis=(-3),
                                           new_axis=(-3),
                                           dtype=complex,
                                           meta=xp.array((), dtype=complex),
                                           )
                hft = ft2(phase_prof * ift2(hft))
            else:
                phase_prof = da.ones(1)

            # #########################
            # compute
            # #########################
            # todo: want to define these arrays in init ... but can't get to_zarr() to work then
            future = [da.map_blocks(to_cpu,
                              hft,
                                    dtype=np.complex64 if self.save_float32 else complex,
                                    ).to_zarr(self.store.store.path,
                                              component="efields_ft",
                                              compute=False,
                                              compressor=self.compressor,
                                              overwrite=True),
                      da.map_blocks(to_cpu,
                                    phase_prof,
                                    dtype=np.complex64 if self.save_float32 else complex,
                                    ).to_zarr(self.store.store.path,
                                              component="phase_correction_profile",
                                              compute=False,
                                              compressor=self.compressor,
                                              overwrite=True)
                      ]
            dask.compute(*future)

            if self.save_dir is not None:
                client.profile(filename=self.save_dir / f"{self.tstamp:s}_preprocessing_dask_profile.html")

            self.timing["unmix_hologram_time"] = perf_counter() - tstart_holos
            if self.verbose:
                print(f"unmixed holograms and fit phases in {parse_time(self.timing['unmix_hologram_time'])[1]}")

    def reconstruct_n(self,
                      use_gpu: bool = False,
                      print_fft_cache: bool = False,
                      processes: bool = False,
                      n_workers: int = 1,
                      **kwargs) -> (array, tuple, dict):

        """
        Reconstruct refractive index using one of a several different models Additional keyword arguments are
        passed through to the dask scheduler

        :param use_gpu:
        :param print_fft_cache: optionally print memory usage of GPU FFT cache at each iteration
        :param processes:
        :param n_workers:
        """

        tstart_recon = perf_counter()

        if use_gpu and cp:
            xp = cp
        else:
            xp = np

        self.efields_ft = da.from_zarr(self.store.efields_ft,
                                       chunks=(1,) * self.nextra_dims + (self.npatterns, self.ny, self.nx))
        self.efield_bg_ft = da.from_zarr(self.store.efield_bg_ft,
                                         chunks=(1,) * self.nextra_dims + (self.npatterns, self.ny, self.nx))

        # ############################
        # get beam frequencies
        # ############################
        beam_frqs = self.get_beam_frqs()
        mean_beam_frqs = [np.mean(f, axis=tuple(range(self.nextra_dims))) for f in beam_frqs]

        # convert to array ... for images which don't have enough multiplexed frequencies, replaced by inf
        mean_beam_frqs_arr = np.ones((self.nmax_multiplex, self.npatterns, 3), dtype=float) * np.inf
        for aaa in range(self.npatterns):
            mean_beam_frqs_arr[:, aaa, :] = mean_beam_frqs[aaa]

        # beam frequencies with multiplexed freqs raveled
        mean_beam_frqs_no_multi = np.zeros([self.npatterns * self.nmax_multiplex, 3])
        for ii in range(self.npatterns):
            for jj in range(self.nmax_multiplex):
                mean_beam_frqs_no_multi[ii * self.nmax_multiplex + jj, :] = mean_beam_frqs_arr[jj, ii]

        # ############################
        # compute information we need for reconstructions
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
        else:
            optimizer = self.models[self.model]

        if self.verbose:
            print(f"computing index of refraction for {int(np.prod(self.imgs_raw.shape[:-3])):d} images "
                  f"using model {self.model:s}.\n"
                  f"image size = {self.npatterns} x {self.ny:d} x {self.nx:d},\n"
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
                  n_size,
                  drs_n,
                  scattered_field_regularization,
                  use_weighted_phase_unwrap,
                  reconstruction_kwargs,
                  n_guess=None,
                  e_fwd_out=None,
                  e_scatt_out=None,
                  n_start_out=None,
                  n_out=None,
                  costs_out=None,
                  steps_out=None,
                  block_id=None):

            nextra_dims = efields_ft.ndim - 3
            dims = tuple(range(nextra_dims))
            nimgs, ny, nx = efields_ft.shape[-3:]

            nmax_multiplex = np.max([len(f) for f in beam_frqs])

            if isinstance(efields_ft, dask.array.Array):
                efields_ft = efields_ft.compute()
            if isinstance(efields_bg_ft, dask.array.Array):
                efields_bg_ft = efields_bg_ft.compute()

            efields_ft = xp.asarray(efields_ft.squeeze(axis=dims))
            efields_bg_ft = xp.asarray(efields_bg_ft.squeeze(axis=dims))

            if rmask is not None:
                rmask = xp.asarray(rmask)

            if block_id is None:
                block_ind = None
                label = ""
            else:
                block_ind = block_id[:nextra_dims]
                label = f"{np.stack(block_id).tolist()} "

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

                efields = bin(ift2(efields_ft), [1, 1], mode="mean")
                del efields_ft

                efields_bg = bin(ift2(efields_bg_ft), [1, 1], mode="mean")
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
                                    **reconstruction_kwargs
                                    )
                n = results["x"]

            # ################
            # store auxilliary info
            # ################
            tstart_aux = perf_counter()
            if costs_out is not None:
                costs_out[block_ind] = to_cpu(results["costs"])

            if steps_out is not None:
                steps_out[block_ind] = to_cpu(results["steps"])

            if e_fwd_out is not None:

                if optimizer == "born" or optimizer == "rytov":
                    e_fwd_out[block_ind] = to_cpu(ift2(model.fwd_model(ft3(get_v(n, no, wavelength)))))
                else:
                    slices = (slice(0, 1), slice(-1, None), slice(None), slice(None))  # [0, -1, :, :]
                    for ii in range(nimgs):
                        ind_now = block_ind + (ii,)
                        e_fwd_out[ind_now] = to_cpu(model.fwd_model(n.squeeze(), inds=[ii])[slices]).squeeze()

            if verbose:
                print(f"stored auxilliary info in {perf_counter() - tstart_aux:.2f}s")

            if use_gpu and print_fft_cache:
                print(f"gpu memory usage after inference = {cp.get_default_memory_pool().used_bytes() / 1e9:.2f}GB")
                print(cp.fft.config.get_plan_cache())


            # ################
            # store n
            # ################
            if n_out is not None:
                n_out[block_ind] = to_cpu(n)
            else:
                return to_cpu(n).reshape((1,) * nextra_dims + n_size)

        # #######################
        # get refractive index
        # #######################
        map_blocks_joblib(recon,
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
                          self.step_start,
                          optimizer,
                          self.verbose,
                          print_fft_cache,
                          self.n_shape,
                          self.drs_n,
                          self.scattered_field_regularization,
                          self.use_weighted_phase_unwrap,
                          self.reconstruction_settings,
                          n_guess=self.n_guess,
                          e_fwd_out=self.store.efwd if self.save_auxiliary_fields else None,
                          e_scatt_out=self.store.escatt if self.save_auxiliary_fields else None,
                          n_start_out=self.store.n_start if self.save_auxiliary_fields and
                                                            self.n_guess is None else None,
                          costs_out=self.store.costs,
                          steps_out=self.store.steps,
                          n_out=self.store.n,
                          chunks=(1,) * self.nextra_dims + (None, None, None),
                          n_workers=n_workers,
                          processes=processes,
                          verbose=False,
                          )

        get_reusable_executor().shutdown(wait=True)
        self.timing["reconstruction_time"] = perf_counter() - tstart_recon
        if self.verbose:
            print(f"recon time: {parse_time(self.timing['reconstruction_time'])[1]:s}")


    def plot_translations(self,
                          time_axis: int = 1,
                          index: Optional[tuple[int]] = None,
                          figsize: Sequence[float, float] = (30., 8.),
                          **kwargs) -> Figure:
        """

        :param time_axis:
        :param index:
        :param figsize:
        :param kwargs:
        :return figh:
        """

        if index is None:
            index = (0,) * (self.nextra_dims - 1)

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
        ax.plot(translations[..., 0] / self.dxy, '.-', label="sig")
        ax.plot(translations_bg[..., 0] / self.dxy, label="bg")
        ax.set_xlabel("time step")
        ax.set_ylabel("x-position / dxy")
        ax.set_title("x-position")

        ax = figh.add_subplot(1, 2, 2)
        ax.plot(translations[..., 1] / self.dxy, '.-', label="sig")
        ax.plot(translations_bg[..., 1] / self.dxy, label="bg")
        ax.set_xlabel("time step")
        ax.set_ylabel("y-position / dxy")
        ax.set_title("y-position")

        return figh

    def plot_frqs(self,
                  time_axis: int,
                  index: Optional[Sequence[int]] = None,
                  figsize: Sequence[float, float] = (30., 8.),
                  **kwargs) -> Figure:
        """

        :param time_axis:
        :param index: should be of length self.nextra_dims - 1. Index along these axes, but ignoring whichever
          axes is the time axis. So e.g. if the axis are position x time x z x parameter then time_axis = 1 and
          the index could be (2, 1, 0) which would selection position 2, z 1, parameter 0.
        :param figsize:
        :param kwargs: passed through to matplotlib.pyplot.figure
        :return:
        """

        if index is None:
            index = (0,) * (self.nextra_dims - 1)

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
        hologram_frqs_mean = [np.mean(f, axis=time_axis, keepdims=True) for f in self.hologram_frqs]
        hgram_frq_diffs = [(f - g)[slices].squeeze(axis=squeeze_axes)
                           for f, g in zip(self.hologram_frqs, hologram_frqs_mean)]
        # stack all hologram frqs
        hgram_frq_diffs = np.concatenate(hgram_frq_diffs, axis=1)

        # shape = ntimes x 2
        rfrq_sq = self.reference_frq.squeeze(axis=(-2, -3, -4))
        ref_frq_diffs = (rfrq_sq - np.mean(rfrq_sq,
                                           axis=time_axis,
                                           keepdims=True))[ref_slices].squeeze(squeeze_axes)

        # plot
        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle(f"index={index}\nfrequency variation versus time")

        # plot frequency differences
        ax = figh.add_subplot(1, 2, 1)
        ax.plot(norm(hgram_frq_diffs, axis=-1) / self.dfx, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("|f - f_mean| / dfx")
        ax.set_title("hologram frequency deviation amplitude")
        ax.legend([f"{ii:d}" for ii in range(self.npatterns)])

        # plot mean frequency differences
        ax = figh.add_subplot(1, 2, 2)
        ax.plot(norm(ref_frq_diffs, axis=-1) / self.dfx, '.-')
        ax.set_xlabel("time step")
        ax.set_ylabel("|fref - fref_mean| / dfx")
        ax.set_title("reference frequency deviation amplitude")

        return figh

    def plot_phases(self,
                    time_axis: int = 1,
                    index: Optional[tuple[int]] = None,
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

        if index is None:
            index = (0,) * (self.nextra_dims - 1)

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

    def plot_diagnostics(self,
                         time_axis: int,
                         index: Optional[tuple[int]] = None,
                         interactive: bool = False,
                         save: bool = True,
                         **kwargs):
        """
        Plot all diagnostics

        :param time_axis:
        :param index:
        :param interactive:
        :param save:
        :param kwargs:
        :return:
        """

        tstart_plot = perf_counter()

        context = {}
        if not interactive:
            context['interactive'] = False
            context['backend'] = "agg"

        with rc_context(context):
            # frequencies
            figh_frq = self.plot_frqs(time_axis=time_axis,
                                      index=index,
                                      **kwargs)

            # plot phases
            figh_ph = self.plot_phases(time_axis=time_axis,
                                       index=index,
                                       **kwargs)

            # plot translations
            figh_xl = self.plot_translations(time_axis=time_axis,
                                             index=index,
                                             **kwargs)

        if save:
            figh_frq.savefig(self.save_dir / "hologram_frequency_stability.png")
            figh_ph.savefig(self.save_dir / "phase_stability.png")
            figh_xl.savefig(self.save_dir / "registration.png")

        if not interactive:
            plt.close(figh_frq)
            plt.close(figh_ph)
            plt.close(figh_xl)

        self.timing["plot_diagnostics_time"] = perf_counter() - tstart_plot
        if self.verbose:
            print(f"plotted diagnostics in: {parse_time(self.timing['plot_diagnostics_time'])[1]:s}")

    def plot_odt_sampling(self,
                          index: Optional[tuple[int]] = None,
                          **kwargs) -> Figure:
        """
        Illustrate the region of frequency space which is obtained using the plane waves described by frqs

        :param index: same length as nextra_dims
        :param kwargs: passed through to figure
        :return figh:
        """

        if index is None:
            index = (0,) * self.nextra_dims

        if len(index) != self.nextra_dims:
            raise ValueError(f"len(index) was {len(index):d} != {self.nextra_dims:d}")

        # todo: take slice argument
        # nfrqs x 2 array of [[fx0, fy0], [fx1, fy1], ...]
        frqs = np.stack(self.get_beam_frqs(), axis=-3)[index][:, 0, (0, 1)]

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
        ax.set_xlabel("$f_x$ (1/$\\mu m$)")
        ax.set_ylabel("$f_z$ (1/$\\mu m$)")

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
        ax.set_xlabel("$f_y$ (1/$\\mu m$)")
        ax.set_ylabel("$f_z$ (1/$\\mu m$)")

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
        ax.set_xlabel("$f_x$ (1/$\\mu m$)")
        ax.set_ylabel("$f_y$ (1/$\\mu m$)")

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

        ax.set_xlabel("$f_x$ (1/$\\mu m$)")
        ax.set_ylabel("$f_y$ (1/$\\mu m$)")
        ax.set_zlabel("$f_z$ (1/$\\mu m$)")

        return figh

    def plot_image(self,
                   index: Optional[tuple[int]] = None,
                   gamma: float = 0.1,
                   figsize: tuple[float, float] = (35., 15.),
                   **kwargs) -> Figure:
        """
        display raw image and holograms

        :param index: index of image to display. Should be of length self.nextra_dims + 1, where the last index
          indicates the pattern to display
        :param gamma: gamma to be used when display fourier transforms
        :param figsize:
        :return figh: figure handle
        """

        if index is None:
            index = (0,) * (self.nextra_dims + 1)

        # if didn't give a pattern index, add it
        if len(index) == self.nextra_dims:
            index += (0, )

        if len(index) != (self.nextra_dims + 1):
            raise ValueError(f"len(index) was {len(index):d} != {self.nextra_dims + 1:d}")

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

        ff = np.sqrt(self.fxs[None, :]**2 + self.fys[:, None]**2)
        max_ind = np.unravel_index(np.argmax(img_ft * (ff > 2*self.fmax)), ff.shape)

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
        ax.set_title(f"$I(r)$")

        ax = figh.add_subplot(grid[1, 0])
        plt.colorbar(im, cax=ax, location="bottom")

        # ######################
        # raw image ft
        # ######################
        ax = figh.add_subplot(grid[2, 0])
        im = ax.imshow(np.abs(img_ft), extent=extent_f,
                       norm=PowerNorm(gamma=gamma), cmap="bone")
        ax.set_title(f"$|I(f)|$, guess fx={self.fxs[max_ind[1]]:.3f}, fy={self.fys[max_ind[0]]:.3f}")
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

    def display_tomography_recon(self, **kwargs):
        """
        Display reconstruction in napari

        :param kwargs:
        :return:
        """
        return display_tomography_recon(self.store,
                                        **kwargs)


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

    :param imgs_ft: n0 x n1 x ... x n_{-2} x n_{-1} array. This function operates along the
     last two axes.
    :param ref_imgs_ft: reference images. Should be broadcastable to same size as imgs.
    :param dxy: pixel size
    :param thresh: Ignore points in imgs_ft which have magnitude less than this fraction of
     max(|ref_imgs_ft|)
    :param init_params: [fx, fy]
    :return fit_params: [fx, fy]
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

        # must be on CPU to run
        imgs_ft_mask = to_cpu(imgs_ft[ind][mask])
        fxfx_mask = to_cpu(fxfx[mask])
        fyfy_mask = to_cpu(fyfy[mask])
        ref_imgs_ft_mask = to_cpu(ref_imgs_ft[ind][mask])

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

        fit_params[ind] = xp.asarray(results["fit_params"])

    return fit_params


def get_scattered_field(efields: array,
                        efields_bg: array,
                        regularization: float = 0.,
                        use_born: bool = False,
                        use_weighted_unwrap: bool = True) -> array:
    """
    Compute estimate of scattered field in real space with regularization. Depending on the mode, this
    will either be an actual estimate of the scattered field, or an estimate of the Rytov phase

    The Rytov phase, \\psi_s(r), is defined by

    .. math::

      U_t(r) &= \\exp \\left[\\psi_o(r) + \\psi_s(r) \\right]

      U_o(r) &= \\exp \\left[\\psi_o(r) \\right]

    We estimate it from

    .. math::

      \\psi_s(r) = \\log \\left \\vert \\frac{U_t(r)}{U_o(r)} \\right \\vert + i \\text{unwrap} \\left[\\text{angle}(U_t) - \\text{angle}(U_o) \\right]


    :param efields: array of size n0 x ... x nm x ny x nx
    :param efields_bg: broadcastable to same size as efields
    :param regularization: limit the influence of any pixels where abs(efields) < regularization
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
                   apodization: Optional[array] = None) -> array:
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
    :return efield_ft: hologram electric field in Fourier space
    """

    if cp and isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    if apodization is None:
        apodization = 1.
    apodization = xp.asarray(apodization)

    # compute efield
    efield_ft = translate_ft(ft2(img * apodization),
                             fx_ref,
                             fy_ref,
                             drs=(dxy, dxy))

    # clear frequencies beyond bandpass
    ny, nx = img.shape[-2:]
    fxs = xp.fft.fftshift(xp.fft.fftfreq(nx, dxy))
    fys = xp.fft.fftshift(xp.fft.fftfreq(ny, dxy))
    ff_perp = np.sqrt(fxs[None, :] ** 2 + fys[:, None] ** 2)
    efield_ft[..., ff_perp > fmax_int / 2] = 0.

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
                             show_n3d: bool = True,
                             show_n_aux: bool = False,
                             show_mips: bool = False,
                             show_raw: bool = False,
                             show_scattered_fields: bool = False,
                             show_efields: bool = False,
                             show_efields_fwd: bool = False,
                             show_phase_correction_profiles: bool = False,
                             compute: bool = True,
                             slices: Optional[tuple] = None,
                             data_slice: Optional[tuple] = None,
                             phase_lim: float = np.pi,
                             n_lim: tuple[float, float] = (0., 0.05),
                             e_lim: tuple[float, float] = (0., 500.),
                             escatt_lim: tuple[float, float] = (-5., 5.),
                             n_cmap="gray_r",
                             real_cmap="bone",
                             phase_cmap="RdBu",
                             xshift_pix: Optional[int] = None,
                             yshift_pix: Optional[int] = None,
                             prefix: str = "",
                             viewer=None,
                             block_while_display: bool = True,
                             **kwargs):
    """
    Display reconstruction results and (optionally) raw data in Napari

    :param location: refractive index reconstruction stored in zarr file
    :param raw_data_fname: raw data stored in zarr file
    :param raw_data_component:
    :param show_n3d:
    :param show_n_aux:
    :param show_mips:
    :param show_raw:
    :param show_scattered_fields:
    :param show_efields:
    :param show_efields_fwd:
    :param compute:
    :param slices: slice n others fields using this tuple of slices. Should be of length n.ndim - 3
    :param data_slice:
    :param phase_lim: phase_min_max
    :param n_lim: (nmin, nmax)
    :param e_lim: (emin, emax)
    :param escatt_lim: (emin, emax)
    :param real_cmap: color map to use for intensity-like images
    :param phase_cmap: color map to use for phase-like images
    :param xshift_pix: shift images laterally to view electric fields and n simultaneously by this many pixels
    :param yshift_pix:
    :param prefix:
    :param viewer: display on this viewer, otherwise create a new one
    :param block_while_display:
    :return: viewer
    """

    if napari is None:
        warn("napari was not successfully imported")
        return None

    # ##############################
    # optionally load raw data
    # ##############################
    if raw_data_fname is not None:
        if isinstance(raw_data_fname, (Path, str)):
            raw_data = zarr.open(raw_data_fname, "r")
        else:
            raw_data = raw_data_fname
    else:
        show_raw = False

    # ##############################
    # load data
    # ##############################
    if isinstance(location, (Path, str)):
        img_z = zarr.open(location, "r")
    else:
        img_z = location

    if "dr" in img_z.attrs.keys():
        drs_n = img_z.attrs["dr"]
    elif "drs_n" in img_z.attrs.keys():
        drs_n = img_z.attrs["drs_n"]
    else:
        raise ValueError("could not identify RI voxel sizes")

    n_axis_names = img_z.attrs["dimensions"]
    no = img_z.attrs["no"]

    # ##############################
    # get size info
    # ##############################
    n_extra_dims = img_z.n.ndim - 3
    nz, ny, nx = img_z.n.shape[-3:]
    npatterns = img_z.attrs["npatterns"]

    if yshift_pix is None:
        yshift_pix = ny

    if xshift_pix is None:
        xshift_pix = nx

    # NOTE: do not operate on these arrays after broadcasting otherwise memory use will explode
    # broadcasting does not cause memory size expansion, but in-place operations later will
    bcast_shape = [1,] * n_extra_dims + [npatterns, nz, 1, 1]

    if not show_raw and not show_efields and not show_efields_fwd:
        bcast_shape[-4] = 1

    if not show_n3d:
        bcast_shape[-3] = 1

    bcast_shape = tuple(bcast_shape)

    # ##############################
    # z refocusing
    # ##############################
    try:
        dz_refocus = img_z.attrs["reconstruction_settings"]["dz_refocus"]
    except KeyError:
        dz_refocus = 0.
    zs = (np.arange(nz) - nz // 2) * drs_n[0] + dz_refocus

    if show_n3d:
        zoffset = float(zs[0])
    else:
        zoffset = 0.

    # ##############################
    # raw data slice corresponding to reconstructed data
    # ##############################
    if data_slice is None:
        try:
            data_slice = tuple([slice(s[0], s[1], s[2]) for s in img_z.attrs["data_slice"]])
        except KeyError:
            data_slice = None

    # ##############################
    # select slices of reconstruction to display
    # ##############################
    if slices is None:
        slices = tuple([slice(None) for _ in range(n_extra_dims)])
    # add last 3 dims
    slices = slices + (slice(None), slice(None), slice(None))

    # ##############################
    # load affine xforms
    # ##############################
    order_is_yx = (hasattr(img_z.attrs["xform_dict"], "coordinate_order") and
                   img_z.attrs["xform_dict"]["coordinate_order"] == "yx")

    if order_is_yx:
        affine_recon2cam = np.array(img_z.attrs["xform_dict"]["affine_xform_recon_2_raw_camera_roi"])
    else:
        # Napari uses convention (y, x) whereas I'm using (x, y),
        # so need to swap these dimensions in affine xforms
        swap_xy = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])
        try:
            affine_recon2cam_xy = np.array(img_z.attrs["xform_dict"]["affine_xform_recon_2_raw_camera_roi"])
        except KeyError:
            try:
                affine_recon2cam_xy = np.array(img_z.attrs["affine_xform_recon_2_raw_camera_roi"])
            except KeyError:
                affine_recon2cam_xy = params2xform([1, 0, 0, 1, 0, 0])

        try:
            affine_recon2cam = swap_xy.dot(affine_recon2cam_xy.dot(swap_xy))
        except TypeError:
            affine_recon2cam = params2xform([1, 0, 0, 1, 0, 0])

    with dask_cfg_set(scheduler='threads',
                      **{'array.slicing.split_large_chunks': False}):
        cluster = LocalCluster(processes=False)
        client = Client(cluster)
        # ######################
        # prepare n
        # ######################
        if show_n3d:
            print("loading n")
            n = da.expand_dims(da.from_zarr(img_z.n)[slices], axis=-4)
            n_real = n.real - no
            n_imag = n.imag
        else:
            n_real = np.zeros((1, 1))
            n_imag = np.zeros_like(n_real)

        if show_n3d and hasattr(img_z, "n_start"):
            slices_nstart = tuple([s if img_z.n_start.shape[ii] != 1 else slice(None) for ii, s in enumerate(slices)])
            n_start = da.expand_dims(da.from_zarr(img_z.n_start)[slices_nstart], axis=-4)
            n_start_real = n_start.real - no
            n_start_imag = n_start.imag
        else:
            n_start_real = np.zeros((1, 1))
            n_start_imag = np.zeros_like(n_start_real)

        if compute:
            with ProgressBar():
                n_real, n_imag = dask.compute([n_real, n_imag])[0]
                n_start_real, n_start_imag = dask.compute([n_start_real, n_start_imag])[0]

            # broadcast
            bcast_shape_n = np.broadcast_shapes(n_real.shape, bcast_shape)
            n_real = np.broadcast_to(n_real, bcast_shape_n)
            n_imag = np.broadcast_to(n_imag, bcast_shape_n)
            n_start_real = np.broadcast_to(n_start_real, bcast_shape_n)
            n_start_imag = np.broadcast_to(n_start_imag, bcast_shape_n)

        # ######################
        # Re(n) MIPs
        # ######################
        if (not hasattr(img_z, "n_maxz") or
            not hasattr(img_z, "n_maxy") or
            not hasattr(img_z, "n_maxx")):
            show_mips = False

        if show_mips:
            n_maxz = da.expand_dims(da.from_zarr(img_z.n_maxz)[slices[:-1]] - no, axis=(-3, -4))
            n_maxy = da.expand_dims(da.from_zarr(img_z.n_maxy)[slices[:-1]] - no, axis=(-3, -4))
            n_maxx = da.flip(da.swapaxes(da.expand_dims(da.from_zarr(img_z.n_maxx)[slices[:-1]] - no,
                                                        axis=(-3, -4)),
                                         -1, -2),
                             axis=-1)
        else:
            n_maxz = np.zeros((1, 1))
            n_maxy = np.zeros_like(n_maxz)
            n_maxx = np.zeros_like(n_maxz)

        if compute:
            with ProgressBar():
                n_maxz, n_maxy, n_maxx = dask.compute([n_maxz, n_maxy, n_maxx])[0]

            n_maxz = np.broadcast_to(n_maxz,
                                     np.broadcast_shapes(n_maxz.shape, bcast_shape))
            n_maxy = np.broadcast_to(n_maxy,
                                     np.broadcast_shapes(n_maxy.shape, bcast_shape))
            n_maxx = np.broadcast_to(n_maxx,
                                     np.broadcast_shapes(n_maxx.shape, bcast_shape))

        # ######################
        # prepare raw images
        # ######################
        if show_raw:
            print("loading raw images")
            if data_slice is None:
                data_slice = tuple([slice(None)] * raw_data[raw_data_component].ndim)

            dc = da.from_zarr(raw_data[raw_data_component])
            imgs = da.expand_dims(dc[data_slice][slices], axis=-3)
        else:
            imgs = np.ones((1, 1))

        if compute:
            with ProgressBar():
                imgs = dask.compute(imgs)[0]

            # broadcast raw images
            bcast_shape_raw = np.broadcast_shapes(imgs.shape, bcast_shape)
            imgs = np.broadcast_to(imgs, bcast_shape_raw)

        # ######################
        # prepare electric fields
        # ######################
        if not hasattr(img_z, "efield_bg_ft") or not hasattr(img_z, "efields_ft"):
            show_efields = False

        if show_efields:
            print("loading electric fields")
            # measured field
            e_load_ft = da.expand_dims(da.from_zarr(img_z.efields_ft)[slices], axis=-3)
            e = da.map_blocks(ift2, e_load_ft, dtype=complex)
            e_abs = da.abs(e)
            e_angle = da.angle(e)

            # background field
            ebg_shape = img_z.efield_bg_ft.shape
            slices_bg = tuple([s if ebg_shape[ii] != 1 else slice(None) for ii, s in enumerate(slices)])

            ebg_load_ft = da.expand_dims(da.from_zarr(img_z.efield_bg_ft)[slices_bg], axis=-3)
            ebg = da.map_blocks(ift2, ebg_load_ft, dtype=complex)
            ebg_abs = da.abs(ebg)
            ebg_angle = da.angle(ebg)

            e_ebg_abs_diff = e_abs - ebg_abs
            e_ebg_phase_diff = da.mod(e_angle - ebg_angle, 2 * np.pi)
            e_ebg_phase_diff -= 2 * np.pi * (e_ebg_phase_diff > np.pi)
        else:
            e_abs = np.zeros((1, 1))
            e_angle = np.zeros_like(e_abs)
            ebg_abs = np.zeros_like(e_abs)
            ebg_angle = np.zeros_like(e_abs)
            e_ebg_abs_diff = np.zeros_like(e_abs)
            e_ebg_phase_diff = np.zeros_like(e_abs)

        if compute:
            with ProgressBar():
                e_abs, e_angle, ebg_abs, ebg_angle, e_ebg_abs_diff, e_ebg_phase_diff = \
                    dask.compute([e_abs, e_angle, ebg_abs, ebg_angle, e_ebg_abs_diff, e_ebg_phase_diff])[0]

            # broadcast electric fields
            bcast_shape_e = np.broadcast_shapes(e_abs.shape, bcast_shape)
            e_abs = np.broadcast_to(e_abs, bcast_shape_e)
            e_angle = np.broadcast_to(e_angle, bcast_shape_e)
            ebg_abs = np.broadcast_to(ebg_abs, bcast_shape_e)
            ebg_angle = np.broadcast_to(ebg_angle, bcast_shape_e)
            e_ebg_abs_diff = np.broadcast_to(e_ebg_abs_diff, bcast_shape_e)
            e_ebg_phase_diff = np.broadcast_to(e_ebg_phase_diff, bcast_shape_e)

        # ######################
        # scattered fields
        # ######################
        if show_scattered_fields and hasattr(img_z, "escatt"):
            print('loading escatt')
            escatt = da.expand_dims(da.from_zarr(img_z.escatt)[slices], axis=-3)
            escatt_real = da.real(escatt)
            escatt_imag = da.imag(escatt)
        else:
            escatt_real = np.zeros((1, 1))
            escatt_imag = np.zeros_like(escatt_real)

        if compute:
            with ProgressBar():
                escatt_real, escatt_imag = dask.compute([escatt_real, escatt_imag])[0]

            # this can be a different size due to multiplexing
            bcast_root_scatt = (1,) * n_extra_dims + (1, nz, 1, 1)
            bcast_shape_scatt = np.broadcast_shapes(escatt_real.shape, bcast_root_scatt)
            escatt_real = np.broadcast_to(escatt_real, bcast_shape_scatt)
            escatt_imag = np.broadcast_to(escatt_imag, bcast_shape_scatt)

        # ######################
        # simulated forward fields
        # ######################
        if show_efields_fwd and hasattr(img_z, "efwd"):
            print("loading fwd electric fields")
            e_fwd = da.expand_dims(da.from_zarr(img_z.efwd)[slices], axis=-3)
            e_fwd_abs = da.abs(e_fwd)
            e_fwd_angle = da.angle(e_fwd)

            # need to slice bg because may have already broadcast
            efwd_ebg_abs_diff = e_fwd_abs - ebg_abs[..., :, slice(0, 1), :, :]
            efwd_ebg_phase_diff = da.mod(e_fwd_angle - ebg_angle[..., :, slice(0, 1), :, :], 2 * np.pi)
            efwd_ebg_phase_diff -= - 2 * np.pi * (efwd_ebg_phase_diff > np.pi)
        else:
            e_fwd_abs = np.zeros((1, 1))
            e_fwd_angle = np.zeros_like(e_fwd_abs)
            efwd_ebg_abs_diff = np.zeros_like(e_fwd_abs)
            efwd_ebg_phase_diff = np.zeros_like(e_fwd_abs)

        if compute:
            with ProgressBar():
                e_fwd_abs, e_fwd_angle, efwd_ebg_abs_diff, efwd_ebg_phase_diff = \
                dask.compute([e_fwd_abs, e_fwd_angle, efwd_ebg_abs_diff, efwd_ebg_phase_diff])[0]

            # broadcast
            bcast_shape_efwd = np.broadcast_shapes(e_fwd_abs.shape, bcast_shape)
            e_fwd_abs = np.broadcast_to(e_fwd_abs, bcast_shape_efwd)
            e_fwd_angle = np.broadcast_to(e_fwd_angle, bcast_shape_efwd)
            efwd_ebg_abs_diff = np.broadcast_to(efwd_ebg_abs_diff, bcast_shape_efwd)
            efwd_ebg_phase_diff = np.broadcast_to(efwd_ebg_phase_diff, bcast_shape_efwd)

        # ######################
        # phase correction
        # ######################
        if (hasattr(img_z, "phase_correction_profile") and
            show_phase_correction_profiles and
            img_z.phase_correction_profile.size != 1):
            pcorr = da.expand_dims(da.from_zarr(img_z.phase_correction_profile)[slices], axis=-3)
            pcorr_abs = da.abs(pcorr)
            pcorr_angle = da.angle(pcorr)
        else:
            pcorr_abs = np.ones((1, 1))
            pcorr_angle = np.ones_like(pcorr_abs)

        if compute:
            pcorr_abs, pcorr_angle = dask.compute([pcorr_abs, pcorr_angle])[0]

            bcast_shape_pcorr = np.broadcast_shapes(pcorr_abs.shape, bcast_shape)
            pcorr_abs = np.broadcast_to(pcorr_abs, bcast_shape_pcorr)
            pcorr_angle = np.broadcast_to(pcorr_angle, bcast_shape_pcorr)

        # ######################
        # create viewer
        # ######################
        if viewer is None:
            viewer = napari.Viewer(title=str(img_z.store.path),
                                   **kwargs)

        # for convenience of affine xforms, keep xy scale in pixels
        scale = (drs_n[0] / drs_n[1], 1, 1)
        zoffset /= drs_n[1]

        # ######################
        # raw data
        # ######################
        if show_raw:
            viewer.add_image(imgs,
                             scale=scale,
                             translate=(zoffset, 0, 0),
                             colormap=real_cmap,
                             contrast_limits=[0, 4096],
                             name=f"{prefix:s}raw images",
                             )

        # ######################
        # electric fields
        # ######################
        if show_efields:
            # field amplitudes
            viewer.add_image(ebg_abs,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, 0, xshift_pix],
                             contrast_limits=e_lim,
                             colormap=real_cmap,
                             name=f"{prefix:s}|e bg|",
                             )

            if show_efields_fwd:
                viewer.add_image(e_fwd_abs,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=[zoffset, 0, xshift_pix],
                                 contrast_limits=e_lim,
                                 colormap=real_cmap,
                                 name=f"{prefix:s}|E fwd|",
                                 )

            viewer.add_image(e_abs,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, 0, xshift_pix],
                             contrast_limits=e_lim,
                             colormap=real_cmap,
                             name=f"{prefix:s}|e|",
                             )

            # field phases
            viewer.add_image(ebg_angle,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, yshift_pix, xshift_pix],
                             contrast_limits=[-np.pi, np.pi],
                             colormap=phase_cmap,
                             name=f"{prefix:s}angle(e bg)",
                             )

            if show_efields_fwd:
                viewer.add_image(e_fwd_angle,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=[zoffset, yshift_pix, xshift_pix],
                                 contrast_limits=[-np.pi, np.pi],
                                 colormap=phase_cmap,
                                 name=f"{prefix:s}angle(E fwd)",
                                 )

            viewer.add_image(e_angle,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, yshift_pix, xshift_pix],
                             contrast_limits=[-np.pi, np.pi],
                             colormap=phase_cmap,
                             name=f"{prefix:s}angle(e)",
                             )

            # difference of absolute values
            if show_efields_fwd:
                viewer.add_image(efwd_ebg_abs_diff,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=[zoffset, 0, 2 * xshift_pix],
                                 contrast_limits=[-e_lim[1], e_lim[1]],
                                 colormap=phase_cmap,
                                 name=f"{prefix:s}|e fwd| - |e bg|",
                                 )

            viewer.add_image(e_ebg_abs_diff,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, 0, 2 * xshift_pix],
                             contrast_limits=[-e_lim[1], e_lim[1]],
                             colormap=phase_cmap,
                             name=f"{prefix:s}|e| - |e bg|",
                             )

            # difference of phases
            if show_efields_fwd:
                viewer.add_image(efwd_ebg_phase_diff,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=[zoffset, yshift_pix, 2*xshift_pix],
                                 contrast_limits=[-phase_lim, phase_lim],
                                 colormap=phase_cmap,
                                 name=f"{prefix:s}angle(e fwd) - angle(e bg)",
                                 )

            viewer.add_image(e_ebg_phase_diff,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, yshift_pix, 2 * xshift_pix],
                             contrast_limits=[-phase_lim, phase_lim],
                             colormap=phase_cmap,
                             name=f"{prefix:s}angle(e) - angle(e bg)",
                             )

            if show_scattered_fields:
                viewer.add_image(escatt_real,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=[zoffset, yshift_pix, 0],
                                 contrast_limits=escatt_lim,
                                 colormap=phase_cmap,
                                 name=f"{prefix:s}Re(e scatt)",
                                 )

                viewer.add_image(escatt_imag,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=[zoffset, yshift_pix, 0],
                                 contrast_limits=escatt_lim,
                                 colormap=phase_cmap,
                                 name=f"{prefix:s}Im(e scatt)",
                                 )

        # ######################
        # phase correction
        # ######################
        if show_phase_correction_profiles:
            viewer.add_image(pcorr_abs,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, yshift_pix, 0],
                             contrast_limits=[0., 1.],
                             colormap=real_cmap,
                             name=f"{prefix:s}abs(P)",
                             )

            viewer.add_image(pcorr_angle,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=[zoffset, yshift_pix, 0],
                             contrast_limits=[-phase_lim, phase_lim],
                             colormap=phase_cmap,
                             name=f"{prefix:s}angle(P)",
                             )

        # ######################
        # reconstructed index of refraction
        # ######################
        if show_n3d:
            if show_n_aux:
                viewer.add_image(n_start_imag,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=(zoffset, 0, 0),
                                 contrast_limits=n_lim,
                                 colormap=real_cmap,
                                 visible=False,
                                 name=f"{prefix:s}n start.imaginary",
                                 )

                viewer.add_image(n_start_real,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=(zoffset, 0, 0),
                                 colormap=n_cmap,
                                 contrast_limits=n_lim,
                                 visible=False,
                                 name=f"{prefix:s}n start - no",
                                 )

                viewer.add_image(n_imag,
                                 scale=scale,
                                 affine=affine_recon2cam,
                                 translate=(zoffset, 0, 0),
                                 colormap=real_cmap,
                                 contrast_limits=n_lim,
                                 visible=False,
                                 name=f"{prefix:s}n.imaginary",
                                 )

            viewer.add_image(n_real,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=(zoffset, 0, 0),
                             colormap=n_cmap,
                             contrast_limits=n_lim,
                             name=f"{prefix:s}n-no",
                             )

        if show_mips:
            viewer.add_image(n_maxz,
                             scale=scale,
                             affine=affine_recon2cam,
                             translate=(zoffset, 0, 0),
                             contrast_limits=n_lim,
                             colormap=n_cmap,
                             name=f"{prefix:s}n max z"
                             )

            # xz
            viewer.add_image(n_maxy,
                             scale=(scale[0], drs_n[0] / drs_n[1], scale[2]),
                             affine=affine_recon2cam,
                             translate=(zoffset, ny, 0),
                             contrast_limits=n_lim,
                             colormap=n_cmap,
                             name=f"{prefix:s}n max y"
                             )

            # yz
            viewer.add_image(n_maxx,
                             scale=(scale[0], scale[1], drs_n[0] / drs_n[1]),
                             affine=affine_recon2cam,
                             translate=(zoffset, 0, -nz * drs_n[0] / drs_n[1]),
                             contrast_limits=n_lim,
                             colormap=n_cmap,
                             name=f"{prefix:s}n max x"
                             )

    # processed ROI
    if show_raw:
        viewer.add_shapes(np.array([[
                                    [0 - 1, 0 - 1],
                                    [0 - 1, nx],
                                    [ny, nx],
                                    [ny, 0 - 1]
                                    ]]),
                          shape_type="polygon",
                          affine=affine_recon2cam,
                          name=f"{prefix:s}processing ROI",
                          edge_width=3,
                          edge_color=[1, 0, 0, 1],
                          face_color=[0, 0, 0, 0])

    # correct labels for broadcasting
    viewer.dims.axis_labels = n_axis_names[:-3] + ["pattern", "z", "y", "x"]
    # set to start positions
    viewer.dims.set_current_step(axis=range(n_extra_dims + 1),
                                 value=[0] * (n_extra_dims + 1))

    # block until closed by user
    viewer.show(block=block_while_display)

    return viewer


def compare_recons(fnames: Sequence[Union[str, Path]],
                   labels: Optional[Sequence] = None,
                   verbose: bool = True,
                   **kwargs):
    """

    :param fnames: sequence of file paths
    :param labels: sequence of names to identify the files
    :param verbose: whether to print information as load files
    :return viewer:
    """
    v = napari.Viewer()

    if isinstance(fnames, (Path, str)):
        fnames = [fnames]

    if labels is None:
        labels = [f"{ii:d} " for ii in range(len(fnames))]

    if len(labels) != len(fnames):
        raise ValueError(f"len(labels) = {len(labels):d} not equal to "
                         f"len(fnames) = {len(fnames):d}")

    for ii, fn in enumerate(fnames):
        if verbose:
            print(fn)

        display_tomography_recon(fn,
                                 viewer=v,
                                 prefix=labels[ii],
                                 block_while_display=ii == (len(fnames) - 1),
                                 **kwargs)

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


class PhaseCorr(Optimizer):
    def __init__(self,
                 e,
                 ebg,
                 tau_l1: float = 0,
                 escale: float = 40.,
                 fit_magnitude: bool = True,
                 **kwargs):
        super(PhaseCorr, self).__init__(e.shape[-3],
                                        prox_parameters={"tau_l1": float(tau_l1),
                                                         "fit_magnitude": bool(fit_magnitude)})

        self.e = e
        self.ebg = ebg
        self.escale = float(escale)

    def gradient(self,
                 x: array,
                 inds: Optional[Sequence[int]] = None) -> array:
        if inds is None:
            inds = list(range(self.n_samples))

        einds = self.e[inds]

        g = ((einds * x - self.ebg[inds]) * einds.conj() /
             (np.abs(self.ebg[inds])**2 + self.escale**2))

        return g

    def cost(self,
             x: array,
             inds: Optional[Sequence[int]] = None) -> array:
        if inds is None:
            inds = list(range(self.n_samples))

        c = 0.5 * ((abs(self.e[inds] * x - self.ebg[inds])**2) /
                   (abs(self.ebg[inds])**2 + self.escale**2)).sum(axis=(-1, -2))
        return c

    def prox(self,
             x: array,
             step: float = 1.) -> array:

        if cp and isinstance(x, cp.ndarray):
            xp = cp
        else:
            xp = np

        # apply l1 to FT
        if self.prox_parameters["tau_l1"] != 0:
            ft_x = ft2(x)
            ft_prox = (soft_threshold(self.prox_parameters["tau_l1"], abs(ft_x)) *
                       xp.exp(1j * xp.angle(ft_x)))

            y = ift2(ft_prox)
            if not self.prox_parameters["fit_magnitude"]:
                y = xp.exp(1j * xp.angle(y))

        else:
            y = xp.array(x, copy=True)

        return y

def map_blocks_joblib(fn,
                      *args,
                      n_workers: int = -1,
                      processes: bool = True,
                      chunks: Optional[Sequence[int]] = None,
                      verbose: bool = True,
                      return_generator: bool = False,
                      **kwargs) -> list:
    """
    Processes function along blocks of a chunked array with joblib. This is intended to be a replacement for
    dask.array.map_blocks()

    :param fn: function to call on blocks
    :param args: first argument should be the main array to map blocks over. Will also map over subsequent arrays, if
    they are broadcastable with args[0]
    :param n_workers: number of joblib workers
    :param processes: whether to use processes or threads
    :param chunks: size of chunks
    :param verbose: whether to display a progress bar
    :param kwargs: should not be arrays
    :return results:
    """

    fullsize = args[0].shape
    nchunks_dims = np.array([f / c if c is not None
                             else 1
                             for c, f in zip(chunks, fullsize)]).astype(int)
    nchunks = np.prod(nchunks_dims)

    def get_block_ind(chunk_ind): return np.unravel_index(chunk_ind, nchunks_dims)

    def slicer(a, chunk_ind):
        if isinstance(a, (np.ndarray, dask.array.Array)) or (cp and isinstance(a, cp.ndarray)):
            block_ind = get_block_ind(chunk_ind)
            slices = [slice(c * s, (c + 1) * s) if s is not None
                      else slice(None)
                      for c, s in zip(block_ind, chunks)]

            # if a is not full dimensionality, only slice last dimensions
            slices_dim = slices[-a.ndim:]

            # don't slice along dimension if we can broadcast
            slices_dim = [s if d != 1
                          else slice(None)
                          for s, d in zip(slices_dim, a.shape)]

            return a[tuple(slices_dim)]
        else:
            return a

    # todo: tqdm only prints out when chunks start processing
    inds = range(nchunks)
    if verbose:
        inds = tqdm(inds)

    # ########################
    # detect and pass special keyword arguments for block logic
    # ########################
    spec = getfullargspec(fn)
    # add block id
    if "block_id" in kwargs.keys():
        raise ValueError("detected special key 'block_id' in kwargs. This is not allowed.")

    if "block_id" in spec.args:
        def bid(ind): return {"block_id": get_block_ind(ind)}
    else:
        def bid(ind): return {}

    if "block_info" in kwargs.keys():
        raise ValueError("detected special key 'block_info' in kwargs. This is not allowed.")

    if "block_info" in spec.args:
        def binfo(ind):
            block_id = get_block_ind(ind)
            array_loc = [(b * s // n, (b + 1) * s // n) for b, n, s in zip(block_id, nchunks_dims, fullsize)]
            return {"block_info": {0: {"shape": fullsize,
                                       "num-chunks": nchunks_dims,
                                       "chunk-location": block_id,
                                       "array-location": array_loc
                                       }
                                    }
                    }
    else:
        def binfo(ind): return {}

    # todo: pass input_chunks or derive from array (through chunksize attribute?) and separately (output) chunks?
    results = (joblib.Parallel(n_jobs=n_workers,
                               prefer="processes" if processes else "threads",
                               return_as="generator" if return_generator else "list")
                 (joblib.delayed(fn)(*[slicer(a, ind) for a in args],
                                     **bid(ind), # pass block id if function accepts it
                                     **binfo(ind), # pass block info
                                     **kwargs)
                                     for ind in inds)
              )

    return results
