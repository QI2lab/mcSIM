"""
Tools for reconstructiong optical diffraction tomography (ODT) data
"""
import time
import datetime
from pathlib import Path
import numpy as np
from numpy import fft
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from scipy.signal.windows import tukey
from skimage.restoration import unwrap_phase, denoise_tv_chambolle
import dask
import dask.array as da
from dask.diagnostics import ProgressBar, ResourceProfiler, Profiler, visualize
# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.patches import Circle, Arc
from matplotlib.cm import ScalarMappable
#
import localize_psf.fit as fit
import mcsim.analysis.sim_reconstruction as sim
import mcsim.analysis.analysis_tools as tools
from field_prop import prop_ft_medium

_cupy_available = True
try:
    import cupy as cp
except ImportError:
    _cupy_available = False


class tomography:
    def __init__(self,
                 imgs_raw,
                 wavelength,
                 no,
                 na_detection,
                 na_excitation,
                 dxy,
                 reference_frq_guess,
                 hologram_frqs_guess=None,
                 imgs_raw_bg=None,
                 phase_offsets=None,
                 axes_names=None):
        """
        Object to reconstruct optical diffraction tomography data

        Philosophy: only add one of Fourier transform/real space representation of fields and etc. as class attributes.
        Others should be recalculated if needed.

        @param imgs_raw: n1 x n2 x ... x nm x npatterns x ny x nx. Data intensity images
        @param wavelength: wavelength in um
        @param no: background index of refraction
        @param na_detection:
        @param na_excitation:
        @param dxy: pixel size in um
        @param reference_frq_guess: [fx, fy] hologram reference frequency
        @param hologram_frqs_guess: npatterns x 2 array of hologram frequency guesses
        @param imgs_raw_bg: background intensity images
        @param phase_offsets: phase shifts between images and corresponding background images
        @param axes_names: names of first m + 1 axes
        """

        self.tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

        # image dimensions
        self.imgs_raw = imgs_raw
        self.imgs_raw_bg = imgs_raw_bg
        self.npatterns, self.ny, self.nx = imgs_raw.shape[-3:]
        self.nextra_dims = imgs_raw.ndim - 3

        if axes_names is None:
            self.axes_names = [f"i{ii:d}" for ii in range(self.imgs_raw.ndim - 2)]
        else:
            self.axes_names = axes_names

        # hologram frequency info
        self.reference_frq = np.array(reference_frq_guess) # 2
        self.hologram_frqs = np.array(hologram_frqs_guess) # npatterns x 2

        # physical parameters
        self.wavelength = wavelength
        self.no = no
        self.na_detection = na_detection
        self.na_excitation = na_excitation
        self.fmax = self.na_detection / self.wavelength

        # phase shifts
        self.phase_offsets = phase_offsets
        self.phase_offsets_bg = None

        # generate coordinates
        self.dxy = dxy
        self.x = (np.arange(self.nx) - self.nx // 2) * dxy
        self.y = (np.arange(self.ny) - self.ny // 2) * dxy
        self.fxs = fft.fftshift(fft.fftfreq(self.nx, self.dxy))
        self.fys = fft.fftshift(fft.fftfreq(self.ny, self.dxy))
        self.fxfx, self.fyfy = np.meshgrid(self.fxs, self.fys)
        self.dfx = self.fxs[1] - self.fxs[0]
        self.dfy = self.fys[1] - self.fys[0]

        # pupil
        self.pupil_mask = np.sqrt(self.fxfx**2 + self.fyfy**2) <= self.fmax  # where pupil allowed to be non-zero
        self.pupil = self.pupil_mask.astype(complex) # value of pupil function

        # settings
        self.use_gpu = False
        self.reconstruction_settings = {} # keys will differ depending on recon mode

    def use_slice_as_background(self, axis, index):
        raise NotImplementedError()
        self.imgs_raw_bg = self.imgs_raw

    def estimate_hologram_frqs(self, save_dir=None):
        """
        Estimate hologram frequencies form raw image
        @param save_dir:
        @return:
        """
        saving = save_dir is not None

        if saving:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # ############################
        # calibrate pattern frequencies
        # ############################

        # load one slice of background data to get frequency reference. Load the first slice along all dimensions
        slices = tuple([slice(0, 1)] * self.nextra_dims + [slice(None)] * 3)
        imgs_frq_cal = np.squeeze(self.imgs_raw_bg[slices])

        # frequency data associated with images
        fxfx, fyfy = np.meshgrid(self.fxs, self.fys)

        def get_hologram_frqs(img, ii):
            img_ref_ft = fft.fftshift(fft.fft2(fft.ifftshift(img)))

            # define allowed search area
            ff_diff_ref = np.abs((fxfx - self.reference_frq[0]) ** 2 + (fyfy - self.reference_frq[1]) ** 2)
            allowed_search_mask = np.logical_and(ff_diff_ref < 1.25 * self.fmax,
                                                 ff_diff_ref >= 0.5 * np.linalg.norm(self.hologram_frqs[ii] - self.reference_frq))

            # fit frequency
            frq_holo, _, _ = sim.fit_modulation_frq(img_ref_ft, img_ref_ft, self.dxy, mask=allowed_search_mask, roi_pix_size=50)

            if saving:
                figh = sim.plot_correlation_fit(img_ref_ft, img_ref_ft, frq_holo, self.dxy)
                figh.savefig(Path(save_dir, f"{ii:d}_angle_correlation_diagnostic.png"))
                plt.close(figh)

            return frq_holo

        # do frequency calibration
        print(f"calibrating {self.npatterns:d} pattern frequencies")

        # create list of tasks
        results = []
        for ii, im in enumerate(imgs_frq_cal):
            results.append(dask.delayed(get_hologram_frqs)(im, ii))

        with ProgressBar():
            r = dask.compute(*results)
        frqs_hologram = np.array(r)

        # for hologram interference frequencies, use frequency closer to guess value
        frq_dists_ref = np.linalg.norm(frqs_hologram - np.expand_dims(self.reference_frq, axis=0), axis=1)
        frq_dists_neg_ref = np.linalg.norm(frqs_hologram + np.expand_dims(self.reference_frq, axis=0), axis=1)
        frqs_hologram[frq_dists_neg_ref < frq_dists_ref] = -frqs_hologram[frq_dists_neg_ref < frq_dists_ref]

        #return frqs_hologram
        self.hologram_frqs = frqs_hologram

    def estimate_reference_frq(self, save_dir=None):
        """
        Estimate hologram reference frequency by looking at residual speckle
        @param save_dir:
        @return:
        """
        saving = save_dir is not None

        # load one slice of background data to get frequency reference. Load the first slice along all dimensions
        slices = tuple([slice(0, 1)] * self.nextra_dims + [slice(None)] * 3)
        imgs_frq_cal = np.squeeze(self.imgs_raw_bg[slices])

        imgs_frq_cal_ft = fft.fftshift(fft.fft2(fft.ifftshift(imgs_frq_cal, axes=(-1, -2)), axes=(-1, -2)),
                                       axes=(-1, -2))
        imgs_frq_cal_ft_abs_mean = np.mean(np.abs(imgs_frq_cal_ft), axis=0)

        results, circ_dbl_fn, figh_ref_frq = fit_ref_frq(imgs_frq_cal_ft_abs_mean, self.dxy, 2*self.fmax, show_figure=saving)
        frq_ref = results["fit_params"][:2]

        # flip sign if necessary to ensure close to guess
        if np.linalg.norm(frq_ref - self.reference_frq) > np.linalg.norm(frq_ref + self.reference_frq):
            frq_ref = -frq_ref

        if saving:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

            figh_ref_frq.savefig(Path(save_dir, "frequency_reference_diagnostic.png"))

        self.reference_frq = frq_ref

    def get_beam_frqs(self):
        """

        @return beam_frqs:
        """

        bxys = self.hologram_frqs - np.expand_dims(self.reference_frq, axis=0)
        bzs = get_fz(bxys[:, 0], bxys[:, 1], self.no, self.wavelength)
        # x, y, z
        beam_frqs = np.concatenate((bxys, np.expand_dims(bzs, axis=1)), axis=1)

        return beam_frqs

    def unmix_holograms(self, mask=None, bg_average_axes=(0,), fit_phases=False, apodization=None):
        """
        Unmix and preprocess holograms
        @param mask: area to be cut out of hologrms
        @param bg_average_axes: axes to average along when producing background images
        @param fit_phases: whether or not to fit phase differences between image and background holograms
        @param apodization: if None use tukey apodization with alpha = 0.1. To use no apodization set equal to 1
        @return:
        """

        if apodization is None:
            apodization = np.expand_dims(np.expand_dims(tukey(self.nx, alpha=0.1), axis=0) *
                                         np.expand_dims(tukey(self.ny, alpha=0.1), axis=1),
                                         axis=tuple(range(self.nextra_dims + 1)))

        # #########################
        # unmix holograms
        # #########################
        holograms_ft_raw = da.map_blocks(unmix_hologram, self.imgs_raw, self.dxy, 2*self.fmax, self.reference_frq,
                                         self.use_gpu, apodization=apodization, dtype=complex)

        if mask is None:
            holograms_ft = holograms_ft_raw
        else:
            holograms_ft = da.map_blocks(cut_mask, holograms_ft_raw, mask, dtype=complex)

        # #########################
        # background holograms
        # #########################
        if self.imgs_raw_bg is None:
            holograms_ft_bg = self.holograms_ft
        else:
            holograms_ft_raw_bg = da.map_blocks(unmix_hologram, self.imgs_raw_bg, self.dxy, 2*self.fmax, self.reference_frq,
                                                self.use_gpu, apodization=apodization, dtype=complex)

            if mask is None:
                holograms_ft_bg = holograms_ft_raw_bg
            else:
                holograms_ft_bg = da.map_blocks(cut_mask, holograms_ft_raw_bg, mask, dtype=complex)

        # #########################
        # determine background phases
        # #########################
        if fit_phases:
            # optionally determine background phase offsets
            with ProgressBar():
                # take one slice from all axes which will be averaged. Otherwise, must keep the dimensions
                slices = tuple([slice(0, 1) if a in bg_average_axes else slice(None) for a in range(self.nextra_dims)] +
                               [slice(None)] * 3)

                hologram_ft_ref = holograms_ft_bg[slices]

                phase_offsets_bg = da.map_blocks(get_global_phase_shifts, holograms_ft_bg, hologram_ft_ref, dtype=float,
                                                 chunks=(1,) * self.nextra_dims + (1, 1, 1)).compute()
        else:
            phase_offsets_bg = np.ones((1,) * self.nextra_dims + (self.npatterns, 1, 1))

        self.phase_offsets_bg = phase_offsets_bg
        self.holograms_ft_bg = da.mean(holograms_ft_bg * da.exp(1j * self.phase_offsets_bg), axis=bg_average_axes, keepdims=True)

        # #########################
        # determine phase offsets
        # #########################
        if self.imgs_raw_bg is None:
            phase_offsets = self.phase_offsets_bg
        else:
            if fit_phases:
                with ProgressBar():
                    phase_offsets = da.map_blocks(get_global_phase_shifts, holograms_ft, self.holograms_ft_bg, dtype=float,
                                                  chunks=(1,) * self.nextra_dims + (1, 1, 1)).compute()
            else:
                phase_offsets = np.ones((1,) * self.nextra_dims + (self.npatterns, 1, 1))

        self.phase_offsets = phase_offsets
        self.holograms_ft = holograms_ft * da.exp(1j * self.phase_offsets)


    def refocus(self, dz):
        # todo: do I need to rename variable instead of redefining?
        self.holograms_ft = da.map_blocks(prop_ft_medium, self.holograms_ft, self.dxy, dz, self.wavelength, self.no,
                                   axis=(-2, -1),
                                   dtype=complex)

        self.holograms_ft_bg = da.map_blocks(prop_ft_medium, self.holograms_ft_bg, self.dxy, dz, self.wavelength, self.no,
                                   axis=(-2, -1),
                                   dtype=complex)


    def set_scattered_field(self, scattered_field_regularization=10, mask=None):
        """
        The scattered field is only used directly in the Born reconstruction.

        But separate the logic for computing it because can also be useful in displaying interferograms
        @param scattered_field_regularization:
        @param mask:
        @return:
        """

        # compute scattered field in real-space
        holograms_bg = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(self.holograms_ft_bg, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
        holograms = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(self.holograms_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        efield_scattered = (holograms - holograms_bg) / (da.abs(holograms_bg) + scattered_field_regularization)

        # cut out any regions we don't want in Fourier space
        if mask is None:
            mask = np.zeros((self.ny, self.nx), dtype=bool)

        efield_scattered_ft_raw = da.fft.fftshift(da.fft.fft2(da.fft.ifftshift(efield_scattered, axes=(-1, -2)),
                                                              axes=(-1, -2)), axes=(-1, -2))

        # region to cut out of scattered field
        combined_mask = np.logical_or(np.logical_not(self.pupil_mask), mask)

        if np.any(combined_mask):
            self.efield_scattered_ft = da.map_blocks(cut_mask,
                                                     efield_scattered_ft_raw,
                                                     combined_mask,
                                                     self.fmax,
                                                     dtype=complex)
        else:
            self.efield_scattered_ft = efield_scattered_ft_raw

    def reconstruct_born(self, mode="naive", **kwargs):
        """

        @param mode: 'naive' or 'fista'
        @param kwargs:
        @return:
        """

        scattered_field_regularization = kwargs["scattered_field_regularization"]
        niters = kwargs["niters"]
        reconstruction_regularizer = kwargs["reconstruction_regularizer"]
        dxy_sampling_factor = kwargs["dxy_sampling_factor"]
        dz_sampling_factor = kwargs["dz_sampling_factor"]
        z_fov = kwargs["z_fov"]
        mask = kwargs["mask"]

        imaginary_constraint = True
        real_constraint = False
        # only for FISTA approach
        tv = True
        tau = 0.02

        # ############################
        # compute scattered field
        # ############################
        self.set_scattered_field(scattered_field_regularization=scattered_field_regularization,
                                 mask=mask)


        # rechunk erecon_ft since must operate on all patterns at once
        # get sampling so can determine new chunk sizes
        (dz_sp, dxy_sp, _), (nz_sp, ny_sp, nx_sp) = \
            get_reconstruction_sampling(self.no,
                                        self.na_detection,
                                        self.get_beam_frqs(),
                                        self.wavelength,
                                        self.dxy,
                                        self.holograms_ft.shape[-2:],
                                        z_fov,
                                        dz_sampling_factor=dz_sampling_factor,
                                        dxy_sampling_factor=dxy_sampling_factor)

        new_chunks = list(self.efield_scattered_ft.chunksize)
        new_chunks[-3] = self.npatterns

        # ############################
        # scattering potential from measured data
        # ############################
        def recon(erecon_ft, no_data_value):
            # only works on arrays with arbitrariy number of singleton dimensions and then npattern x ny x nx
            dsizes = erecon_ft.shape[:-3]
            if np.prod(dsizes) != 1:
                raise ValueError(
                    f"erecon_ft was wrong shape. Must have any number of leading singleton dimensions and then be"
                    f"npatterns x ny x nx shape = {erecon_ft.shape}")

            v_ft, drs = reconstruction(da.squeeze(erecon_ft), self.get_beam_frqs(), self.no,
                                       self.na_detection, self.wavelength, self.dxy, z_fov=z_fov,
                                       regularization=reconstruction_regularizer,
                                       dz_sampling_factor=dz_sampling_factor,
                                       dxy_sampling_factor=dxy_sampling_factor, mode="born",
                                       no_data_value=no_data_value)

            return np.expand_dims(v_ft, axis=list(range(len(dsizes))))


        if mode == "fista":
            v_fts = da.map_blocks(recon, da.rechunk(self.efield_scattered_ft, chunks=new_chunks), 0, dtype=complex,
                                  chunks=(1,) * self.nextra_dims + (nz_sp, ny_sp, nx_sp))

            # settings
            fista = True
            steps_per_tv = 1

            # define forward model
            model, _ = fwd_model_linear(self.get_beam_frqs(),
                                                self.no,
                                                self.na_detection,
                                                self.wavelength,
                                                (self.ny, self.nx),
                                                (self.dxy, self.dxy),
                                                (nz_sp, ny_sp, nx_sp),
                                                (dz_sp, dxy_sp, dxy_sp),
                                                mode="born",
                                                interpolate=True)

            model_csc = model.tocsc(copy=True)

            # set step size
            u, s, vh = sp.linalg.svds(model, k=1, which='LM')
            # since lipschitz constant of model which is multiplied by...
            step = 1 / s * (self.npatterns * self.ny * self.nx) * (self.ny * self.nx)

            def fista_recon(v_ft_start, e_measured_ft):


                v_ft_start = np.squeeze(v_ft_start)
                # v_ft_start = fft.fftshift(fft.fftn(fft.ifftshift(np.zeros(v_ft_start.shape, dtype=complex))))
                e_measured_ft = np.squeeze(e_measured_ft)
                def cost(v_ft):
                    # mean cost per (real-space) pixel
                    # divide by nxy**2 again because using FT's and want mean cost in real space
                    # recall that \sum_r |f(r)|^2 = 1/N \sum_k |f(k)|^2
                    # NOTE: to get total cost, must take mean over patterns
                    costs = np.mean(np.reshape(np.abs(model.dot(v_ft.ravel()) - e_measured_ft.ravel()) ** 2,
                                               [self.npatterns, self.ny, self.nx]),
                                    axis=(-1, -2)) / 2 / self.ny / self.ny
                    return costs

                def grad(v_ft):
                    # first division is average
                    # second division converts Fourier space to real-space sum
                    dc_dm = (model.dot(v_ft.ravel()) - e_measured_ft.ravel()) / (self.npatterns * self.ny * self.nx) / (self.ny * self.nx)
                    dc_dv = (dc_dm[np.newaxis, :].conj() * model_csc)[0].conj()
                    return dc_dv

                # initialize
                tstart = time.perf_counter()
                costs = np.zeros((niters + 1, self.npatterns))
                v_ft = v_ft_start.ravel()
                q_last = 1
                costs[0] = cost(v_ft)

                for ii in range(niters):
                    # gradient descent
                    tstart_grad = time.perf_counter()
                    dc_dv = grad(v_ft)

                    v_ft -= step * dc_dv

                    tend_grad = time.perf_counter()

                    # FT so can apply TV
                    tstart_fft = time.perf_counter()

                    if self.use_gpu:
                        v = cp.asnumpy(
                            cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(cp.asarray(v_ft.reshape(v_ft.shape))))))
                    else:
                        v = fft.fftshift(fft.ifftn(fft.ifftshift(v_ft.reshape(v_ft.shape))))

                    tend_fft = time.perf_counter()

                    # apply TV proximity operators
                    tstart_tv = time.perf_counter()
                    if tv and ii % steps_per_tv == 0:
                        v_real = denoise_tv_chambolle(v.real, tau)
                        v_imag = denoise_tv_chambolle(v.imag, tau)
                    else:
                        v_real = v.real
                        v_imag = v.imag

                    tend_tv = time.perf_counter()

                    # apply projection onto constraints
                    tstart_constraints = time.perf_counter()

                    if imaginary_constraint:
                        v_imag[v_imag > 0] = 0

                    if real_constraint:
                        # actually ... this is no longer right if there is any imaginary part
                        v_real[v_real > 0] = 0

                    tend_constraints = time.perf_counter()

                    # ft back
                    tstart_fft_back = time.perf_counter()

                    if self.use_gpu:
                        v_ft_prox = cp.asnumpy(
                            cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(cp.asarray(v_real + 1j * v_imag)))).ravel())
                    else:
                        v_ft_prox = fft.fftshift(fft.fftn(fft.ifftshift(v_real + 1j * v_imag))).ravel()

                    tend_fft_back = time.perf_counter()

                    # update step
                    tstart_update = time.perf_counter()

                    q_now = 0.5 * (1 + np.sqrt(1 + 4 * q_last ** 2))

                    if ii == 0 or ii == (niters - 1) or not fista:
                        v_ft = v_ft_prox
                    else:
                        v_ft = v_ft_prox + (q_last - 1) / q_now * (v_ft_prox - v_ft_prox_last)

                    tend_update = time.perf_counter()

                    # compute errors
                    tstart_err = time.perf_counter()

                    costs[ii + 1] = cost(v_ft)

                    tend_err = time.perf_counter()

                    # update for next step
                    q_last = q_now
                    v_ft_prox_last = v_ft_prox

                    # print information
                    tend_iter = time.perf_counter()

                    # debug
                    # print(
                    #     f"iteration {ii + 1:d}/{niters:d}, cost={np.mean(costs[ii + 1]):.3g},"
                    #     f" grad={tend_grad - tstart_grad:.2f}s, fft={tend_fft - tstart_fft:.2f}s,"
                    #     f" TV={tend_tv - tstart_tv:.2f}s, projection={tend_constraints - tstart_constraints:.2f}s,"
                    #     f" fft={tend_fft_back - tstart_fft_back:.2f}s, update={tend_update - tstart_update:.2f}s,"
                    #     f" err={tend_err - tstart_err:.2f}s, iter={tend_iter - tstart_grad:.2f}s,"
                    #     f" total={time.perf_counter() - tstart:.2f}s", end="\r")

                v_out_ft = v_ft.reshape((1,) * self.nextra_dims + (nz_sp, ny_sp, nx_sp))

                # debugging
                # v_out = fft.fftshift(fft.ifftn(fft.ifftshift(v_out_ft)))
                # n_out = get_n(v_out, self.no, self.wavelength)
                #
                # plot_n3d(np.squeeze(v_out_ft), self.no, self.wavelength, title="n out")
                #
                # e_out_ft = model.dot(v_out_ft.ravel()).reshape((self.npatterns, self.ny, self.nx))
                # e_out_ft[:, np.logical_not(self.pupil_mask)] = 0
                # e_out = fft.fftshift(fft.ifft2(fft.ifftshift(e_out_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
                #
                # etest = fft.fftshift(fft.ifft2(fft.ifftshift(np.squeeze(self.efield_scattered_ft[0].compute()), axes=(-1, -2)),
                #               axes=(-1, -2)), axes=(-1, -2))
                # figh1, figh2 = compare_escatt(e_out, etest, self.get_beam_frqs(), self.dxy, ttl="E output",
                #                               figsize=(20, 10))

                return v_out_ft



            efield_scat_rechunk = self.efield_scattered_ft.rechunk((1,) * self.nextra_dims + (self.npatterns, self.ny, self.nx))
            v_ft_out = da.map_blocks(fista_recon,
                                     v_fts,
                                     efield_scat_rechunk,
                                     dtype=complex,
                                     chunks=(1,) * self.nextra_dims + (nz_sp, ny_sp, nx_sp))



        else:
            v_fts = da.map_blocks(recon,
                                  da.rechunk(self.efield_scattered_ft, chunks=new_chunks),
                                  np.nan,
                                  dtype=complex,
                                  chunks=(1,) * self.nextra_dims + (nz_sp, ny_sp, nx_sp))

            # ############################
            # fill in fourier space with constraint algorithm
            # ############################
            v_ft_out = da.map_blocks(apply_n_constraints,
                                     v_fts,
                                     self.no,
                                     self.wavelength,
                                     n_iterations=niters,
                                     beta=0.5,
                                     use_raar=False,
                                     use_gpu=False,
                                     dtype=complex)

        v_out = da.fft.fftshift(da.fft.ifftn(da.fft.ifftshift(v_ft_out, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))

        n = da.map_blocks(get_n,
                          v_out,
                          self.no,
                          self.wavelength,
                          dtype=complex)

        return n, (dz_sp, dxy_sp, dxy_sp)

    def reconstruct_rytov(self, mode="naive", **kwargs):
        rytov_regul = kwargs["rytov_regularizer"]
        niters = kwargs["niters"]
        reconstruction_regularizer = kwargs["reconstruction_regularizer"]
        dxy_sampling_factor = kwargs["dxy_sampling_factor"]
        dz_sampling_factor = kwargs["dz_sampling_factor"]
        z_fov = kwargs["z_fov"]

        # ############################
        # compute rytov phase
        # ############################
        holograms_bg = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(self.holograms_ft_bg, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
        holograms = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(self.holograms_ft, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        phi_rytov = da.map_blocks(get_rytov_phase, holograms, holograms_bg, rytov_regul, dtype=complex)
        phi_rytov_ft = da.fft.fftshift(da.fft.fftn(da.fft.ifftshift(phi_rytov, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        # ############################
        # do reconstruction
        # ############################
        def recon(erecon_ft):
            # only works on arrays with arbitrariy number of singleton dimensions and then npattern x ny x nx
            dsizes = erecon_ft.shape[:-3]
            if np.prod(dsizes) != 1:
                raise ValueError(
                    f"erecon_ft was wrong shape. Must have any number of leading singleton dimensions and then be"
                    f"npatterns x ny x nx shape = {erecon_ft.shape}")

            v_ft, drs = reconstruction(da.squeeze(erecon_ft), self.get_beam_frqs(), self.no,
                                       self.na_detection, self.wavelength, self.dxy, z_fov=z_fov,
                                       regularization=reconstruction_regularizer,
                                       dz_sampling_factor=dz_sampling_factor,
                                       dxy_sampling_factor=dxy_sampling_factor, mode="rytov")

            return np.expand_dims(v_ft, axis=list(range(len(dsizes))))

        # rechunk erecon_ft since must operate on all patterns at once
        # get sampling so can determine new chunk sizes
        (dz_sp, dxy_sp, _), (nz_sp, ny_sp, nx_sp) = \
            get_reconstruction_sampling(self.no,
                                        self.na_detection,
                                        self.get_beam_frqs(),
                                        self.wavelength,
                                        self.dxy,
                                        self.holograms_ft.shape[-2:],
                                        z_fov,
                                        dz_sampling_factor=dz_sampling_factor,
                                        dxy_sampling_factor=dxy_sampling_factor)

        new_chunks = list(phi_rytov_ft.chunksize)
        new_chunks[-3] = self.npatterns
        v_fts = da.map_blocks(recon, da.rechunk(phi_rytov_ft, chunks=new_chunks), dtype=complex,
                              chunks=(1,) * self.nextra_dims + (nz_sp, ny_sp, nx_sp))

        # ############################
        # fill in fourier space with constraint algorithm
        # ############################
        v_ft_infer = da.map_blocks(apply_n_constraints,
                                   v_fts,
                                   self.no,
                                   self.wavelength,
                                   n_iterations=niters,
                                   beta=0.5,
                                   use_raar=False,
                                   use_gpu=False,
                                   require_real_part_greater_bg=False,
                                   dtype=complex)

        v_ft_out = da.fft.fftshift(da.fft.ifftn(da.fft.ifftshift(v_ft_infer, axes=(-3, -2, -1)), axes=(-3, -2, -1)),
                                   axes=(-3, -2, -1))

        n_infer_out = da.map_blocks(get_n,
                                    v_ft_out,
                                    self.no,
                                    self.wavelength,
                                    dtype=complex)

        return n_infer_out, (dz_sp, dxy_sp, dxy_sp)

    def get_powers(self):
        # compute powers
        powers_rms = da.sqrt(da.mean(da.abs(self.imgs_raw) ** 2, axis=(-1, -2)))
        e_powers_rms = da.sqrt(da.mean(da.abs(self.holograms_ft) ** 2, axis=(-1, -2))) / np.prod(self.holograms_ft.shape[-2:])

        return powers_rms, e_powers_rms

    def get_powers_bg(self):
        # compute powers
        powers_rms_bg = da.sqrt(da.mean(da.abs(self.imgs_raw_bg) ** 2, axis=(-1, -2)))
        e_powers_rms_bg = da.sqrt(da.mean(da.abs(self.holograms_ft_bg) ** 2, axis=(-1, -2))) / np.prod(self.holograms_ft_bg.shape[-2:])

        return powers_rms_bg, e_powers_rms_bg

    def plot_interferograms(self, nmax_to_plot, save_dir=None, scattered_field_regularization=10):
        """
        Plot nmax interferograms

        @param nmax_to_plot:
        @param save_dir:
        @param axes_names:
        @return:
        """
        if not hasattr(self, "efield_scattered_ft"):
            self.set_scattered_field(scattered_field_regularization=scattered_field_regularization)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)


        nmax_to_plot = np.min([nmax_to_plot, np.prod(self.holograms_ft.shape[:-2])])

        beam_frqs = self.get_beam_frqs()
        theta, phi = get_angles(beam_frqs, self.no, self.wavelength)

        def plot_interferograms(index, save_dir):
            unraveled_inds = np.unravel_index(index, self.holograms_ft.shape[:-2])

            desc = ""
            for ii in range(len(unraveled_inds)):
                desc += f"_{self.axes_names[ii]:s}={unraveled_inds[ii]:d}"

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


def cut_mask(img, mask, mask_val=0):
    mask = np.expand_dims(mask, axis=list(range(img.ndim - 2)))

    img_masked = np.array(img, copy=True)
    img_masked[mask] = mask_val
    return img_masked


def get_angular_spectrum_kernel(dz, wavelength, no, shape, drs):
    """

    @param dz:
    @param wavelength:
    @param no:
    @param shape:
    @param drs:
    @return:
    """
    k = 2*np.pi / wavelength
    ny, nx = shape
    dy, dx = drs

    fx = fft.fftshift(fft.fftfreq(nx, dx))
    fy = fft.fftshift(fft.fftfreq(ny, dy))
    fxfx, fyfy = np.meshgrid(fx, fy)

    with np.errstate(invalid="ignore"):
        kernel = np.exp(1j * dz * np.sqrt((k * no)**2 - (2*np.pi * fxfx)**2 - (2*np.pi * fyfy)**2))
        kernel[np.isnan(kernel)] = 0

    return kernel


def propagate_field(efield_start, n_stack, no, drs, wavelength, use_gpu=_cupy_available):
    """
    Propagate electric field through medium with index of refraction n(x, y, z) using the projection approximation.

    @param efield_start: ny x nx array
    @param n_stack: nz x ny x nx array
    @param no: background index of refraction
    @param drs: (dz, dy, dx)
    @param wavelength: wavelength in same units as drs
    @return efield: nz x ny x nx electric field
    """
    n_stack = np.atleast_3d(n_stack)

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n_stack.shape

    prop_kernel = get_angular_spectrum_kernel(dz, wavelength, no, n_stack.shape[1:], drs[1:])
    # apodization = np.expand_dims(tukey(nxy, alpha=0.1), axis=0) * np.expand_dims(tukey(nxy, alpha=0.1), axis=1)
    apodization = 1

    # ifftshift these to eliminate doing an fftshift every time
    prop_factor = fft.ifftshift(prop_kernel * apodization)

    # do simulation
    efield = np.zeros((nz, ny, nx), dtype=complex)
    efield[0] = efield_start
    if use_gpu:
        prop_factor = cp.array(prop_factor)
        # note: also tried creating efield on GPU and putting n_stack() on the gpu, but then function took much longer
        # dominated by time to transfer efield back to CPU

    for ii in range(nz - 1):
        # projection approximation
        # propagate through background medium using angular spectrum method
        # then accumulate extra phases in real space
        if use_gpu:
            enow = cp.asarray(efield[ii])
            etemp1 = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fft2(cp.fft.ifftshift(enow)) * prop_factor)) # k-space propagation
            efield[ii + 1] = cp.asnumpy(etemp1 * cp.exp(1j * k * dz * (cp.asarray(n_stack[ii]) - no)))  # real space phase
        else:
            efield[ii + 1] = fft.fftshift(fft.ifft2(
                                 fft.fft2(fft.ifftshift(efield[ii])) * prop_factor  # k-space propagation
                                 )) * \
                                 np.exp(1j * k * dz * (n_stack[ii] - no))  # real space phase


    return efield


# helper functions
def get_fz(fx, fy, no, wavelength):
    """
    Get z-component of frequency given fx, fy

    @param frqs_2d: nfrqs x 2
    @param no: index of refraction
    @param wavelength: wavelength
    @return frqs_3d:
    """

    with np.errstate(invalid="ignore"):
        fzs = np.sqrt(no**2 / wavelength ** 2 - fx**2 - fy**2)

    return fzs


def get_angles(frqs, no, wavelength):
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


def angles2frqs(no, wavelength, theta, phi):
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


def get_global_phase_shifts(imgs, ref_imgs, thresh=None):
    """
    Given a stack of images and a reference, determine the phase shifts between images, such that
    imgs * np.exp(1j * phase_shift) ~ img_ref

    @param imgs: n0 x n1 x ... x n_{-2} x n_{-1} array
    @param ref_imgs: n_{-m} x ... x n_{-2} x n_{-1} array
    @param thresh: only consider points in images where both abs(imgs) and abs(ref_imgs) > thresh
    @return phase_shifts:
    """
    # broadcast images and references images to same shapes
    imgs, ref_imgs = np.broadcast_arrays(imgs, ref_imgs)

    # looper over all dimensions except the last two, which are the y and x dimensions respectively
    loop_shape = imgs.shape[:-2]
    phase_shifts = np.zeros(loop_shape + (1, 1))
    nfits = np.prod(loop_shape).astype(int)
    for ii in range(nfits):
        ind = np.unravel_index(ii, loop_shape)

        if thresh is None:
            mask = np.ones(imgs[ind].shape, dtype=bool)
        else:
            mask = np.logical_and(np.abs(imgs[ind]) > thresh, np.abs(ref_imgs) > thresh)

        # todo: could add a more sophisticated fitting function if seemed useful
        def fn(p): return np.abs(imgs[ind] * np.exp(1j * p[0]) - ref_imgs[ind])[mask].ravel()
        # def jac(p): return [(1j * imgs[ind] * np.exp(1j * p[0])).ravel()]
        # s1 = np.mean(imgs[ii])
        # s2 = np.mean(ref_imgs[ii])
        # def fn(p): return np.abs(s1 * np.exp(1j * p[0]) - s2)
        results = fit.fit_least_squares(fn, [0])
        phase_shifts[ind] = results["fit_params"]

    return phase_shifts


# convert between index of refraction and scattering potential
def get_n(scattering_pot, no, wavelength):
    """
    convert from the scattering potential to the index of refraction
    @param scattering_pot: F(r) = - (2*np.pi / lambda)^2 * (n(r)^2 - no^2)
    @param no: background index of refraction
    @param wavelength: wavelength
    @return n:
    """
    k = 2 * np.pi / wavelength
    n = np.sqrt(-scattering_pot / k ** 2 + no ** 2)
    return n


def get_scattering_potential(n, no, wavelength):
    """
    Convert from the index of refraction to the scattering potential

    @param n:
    @param no:
    @param wavelength:
    @return:
    """
    k = 2 * np.pi / wavelength
    sp = - k ** 2 * (n**2 - no**2)
    return sp


def get_rytov_phase(eimgs, eimgs_bg, regularization):
    """
    Compute rytov phase from field and background field. The Rytov phase is \psi_s(r) where
    U_total(r) = exp[\psi_o(r) + \psi_s(r)]
    where U_o(r) = exp[\psi_o(r)] is the unscattered field

    We calculate \psi_s(r) = log | U_total(r) / U_o(r)| + 1j * unwrap[angle(U_total) - angle(U_o)]

    @param eimgs: npatterns x ny x nx
    @param eimgs_bg: same size as eimgs
    @param float regularization: regularization value
    @return psi_rytov:
    """

    if eimgs.ndim < 3:
        raise ValueError("eimgs must be at least 3D")

    if eimgs_bg.ndim < 3:
        raise ValueError("eimgs_bg must be at least 3D")

    # output values
    psi_rytov = np.zeros(eimgs.shape, dtype=complex)

    # broadcast arrays
    eimgs, eimgs_bg = np.broadcast_arrays(eimgs, eimgs_bg)

    loop_shape = eimgs.shape[:-2]
    nloop = np.prod(loop_shape)
    for ii in range(nloop):
        ind = np.unravel_index(ii, loop_shape)

        # convert phase difference from interval [0, 2*np.pi) to [-np.pi, np.pi)
        phase_diff = np.mod(np.angle(eimgs[ind]) - np.angle(eimgs_bg[ind]), 2 * np.pi)
        phase_diff[phase_diff >= np.pi] -= 2 * np.pi

        # get rytov phase change
        # psi_rytov[aa] = np.log(np.abs(eimg[aa]) / (np.abs(eimg_bg[aa]) + delta)) + 1j * unwrap_phase(phase_diff)
        psi_rytov[ind] = np.log(np.abs(eimgs[ind]) / (np.abs(eimgs_bg[ind]))) + 1j * unwrap_phase(phase_diff)
        psi_rytov[ind][np.abs(eimgs_bg[ind]) < regularization] = 0

    return psi_rytov


# holograms
def unmix_hologram(img: np.ndarray, dxy: float, fmax_int: float, frq_ref: np.ndarray,
                   use_gpu: bool=_cupy_available, apodization=1) -> np.ndarray:
    """
    Given an off-axis hologram image, determine the electric field represented

    @param img: n1 x ... x n_{-3} x n_{-2} x n_{-1} array
    @param dxy: pixel size
    @param fmax_int: maximum frequency where intensity OTF has support
    @param frq_ref: reference frequency (fx, fy)
    @return:
    """
    # FT of image
    if not use_gpu:
        img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img * apodization, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    else:
        img_ft = cp.asnumpy(cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(img * apodization, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1)))
    ny, nx = img_ft.shape[-2:]

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(nx, dxy))
    fys = fft.fftshift(fft.fftfreq(ny, dxy))
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff_perp = np.sqrt(fxfx ** 2 + fyfy ** 2)

    # compute efield
    efield_ft = tools.translate_ft(img_ft, frq_ref[..., 0], frq_ref[..., 1], drs=(dxy, dxy), use_gpu=use_gpu)
    efield_ft[..., ff_perp > fmax_int / 2] = 0

    return efield_ft


# tomographic reconstruction
def get_fmax(no, na_detection, na_excitation, wavelength):
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


def get_reconstruction_sampling(no, na_det, beam_frqs, wavelength, dxy, img_size, z_fov,
                                dz_sampling_factor=1., dxy_sampling_factor=1.):
    """
    Get information about pixel grid scattering potential will be reconstructed on

    @param no: background index of refraction
    @param na_det: numerical aperture of detection objective
    @param beam_frqs:
    @param wavelength: wavelength
    @param dxy: camera pixel size
    @param img_size: (ny, nx) size of hologram images
    @param z_fov: field-of-view in z-direction
    @param dz_sampling_factor: z-spacing as a fraction of the nyquist limit
    @param dxy_sampling_factor: xy-spacing as a fraction of nyquist limit
    @return (dz_sp, dxy_sp, dxy_sp), (nz_sp, ny_sp, nx_sp), (fz_max, fxy_max):
    """
    ny, nx = img_size

    # todo: let's just use beta based on na ... can always provide an effective na if desired
    theta, _ = get_angles(beam_frqs, no, wavelength)
    alpha = np.arcsin(na_det / no)
    beta = np.max(theta)

    # maximum frequencies present in ODT
    fxy_max = (na_det + no * np.sin(beta)) / wavelength
    fz_max = no / wavelength * np.max([1 - np.cos(alpha), 1 - np.cos(beta)])

    # ##################################
    # generate real-space sampling from Nyquist sampling
    # ##################################
    dxy_v = dxy_sampling_factor * 0.5 * 1 / fxy_max
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


def get_coords(drs, nrs, expand=False):
    """
    Compute spatial coordinates

    @param drs: (dz, dy, dx)
    @param nrs: (nz, ny, nx)
    @param expand: if False then return z, y, x as 1D arrays, otherwise return as 3D arrays
    @return: coords (z, y, x)
    """
    coords = [dr * (np.arange(nr) - nr // 2) for dr, nr in zip(drs, nrs)]

    if expand:
        coords = np.meshgrid(*coords, indexing="ij")

    return coords


def fwd_model_linear(beam_frqs, no, na_det, wavelength,
                     e_shape: tuple[int],
                     drs_e: tuple[float],
                     v_shape: tuple[int],
                     drs_v: tuple[float],
                     mode: str = "born",
                     interpolate: bool = False):
    """
    Forward model from scattering potential v to imaged electric field after interacting with object

    @param beam_frqs: (nbeams, 3). Normalized to no/wavelength
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
    # todo: can I combine this logic with reconstruct() in some way that I don't duplicate so much code?

    ny, nx = e_shape
    dy, dx = drs_e

    nimgs = len(beam_frqs)

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
        Fx, Fy, Fz = np.broadcast_arrays(fx - np.expand_dims(beam_frqs[:, 0], axis=(1, 2)),
                                         fy - np.expand_dims(beam_frqs[:, 1], axis=(1, 2)),
                                         fz - np.expand_dims(beam_frqs[:, 2], axis=(1, 2))
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
        fx_rytov = Fx + np.expand_dims(beam_frqs[:, 0], axis=(1, 2))
        fy_rytov = Fy + np.expand_dims(beam_frqs[:, 1], axis=(1, 2))

        fz = get_fz(fx_rytov,
                    fy_rytov,
                    no,
                    wavelength)

        Fz = fz - np.expand_dims(beam_frqs[:, 2], axis=(1, 2))

        # take care of frequencies which do not contain signal
        detectable = (fx_rytov ** 2 + fy_rytov ** 2) <= (na_det / wavelength) ** 2

        # indices into the final scattering potential
        zind = Fz / dfz_v + nz_v // 2
        yind = Fy / dfy_v + ny_v // 2
        xind = Fx / dfx_v + nx_v // 2

        zind, yind, xind = np.broadcast_arrays(zind, yind, xind)
        zind = np.array(zind, copy=True)
        yind = np.array(yind, copy=True)
        xind = np.array(xind, copy=True)
    else:
        raise ValueError(f"'mode' must be 'born' or 'rytov' but was '{mode:s}'")


    # build forward model as sparse matrix
    # E(k) = model * V(k)
    # where V is made into a vector by ravelling
    # and the scattered fields are first stacked then ravelled
    # use csr for fast right mult L.dot(v)
    # use csc for fast left mult w.dot(L)
    if interpolate:
        # trilinear interpolation scheme ... needs points before and after
        z1 = np.floor(zind).astype(int)
        z2 = np.ceil(zind).astype(int)
        y1 = np.floor(yind).astype(int)
        y2 = np.ceil(yind).astype(int)
        x1 = np.floor(xind).astype(int)
        x2 = np.ceil(xind).astype(int)

        # find indices in bounds
        to_use = np.logical_and.reduce((z1 >= 0, z2 < nz_v,
                                        y1 >= 0, y2 < ny_v,
                                        x1 >= 0, x2 < nx_v,
                                        detectable))
        # todo: add in the points this misses where only option is to round

        inds = [(z1, y1, x1),
                (z2, y1, x1),
                (z1, y2, x1),
                (z2, y2, x1),
                (z1, y1, x2),
                (z2, y1, x2),
                (z1, y2, x2),
                (z2, y2, x2)
                ]
        interp_weights = [(zind - z1) * (yind - y1) * (xind - x1),
                          (z2 - zind) * (yind - y1) * (xind - x1),
                          (zind - z1) * (y2 - yind) * (xind - x1),
                          (z2 - zind) * (y2 - yind) * (xind - x1),
                          (zind - z1) * (yind - y1) * (x2 - xind),
                          (z2 - zind) * (yind - y1) * (x2 - xind),
                          (zind - z1) * (y2 - yind) * (x2 - xind),
                          (z2 - zind) * (y2 - yind) * (x2 - xind)]

        # row_index -> indices into E vector
        row_index = np.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]
        row_index = np.tile(row_index, 8)

        # column_index -> indices into V vector
        inds_to_use = [[i[to_use] for i in inow] for inow in inds]
        zinds_to_use, yinds_to_use, xinds_to_use = list(zip(*inds_to_use))
        zinds_to_use = np.concatenate(zinds_to_use)
        yinds_to_use = np.concatenate(yinds_to_use)
        xinds_to_use = np.concatenate(xinds_to_use)

        column_index = np.ravel_multi_index(tuple((zinds_to_use, yinds_to_use, xinds_to_use)), v_shape)

        # construct sparse matrix values
        interp_weights_to_use = np.concatenate([w[to_use] for w in interp_weights])

        fz_stack = np.tile(fz[to_use], 8)

        data = interp_weights_to_use / (2 * 1j * (2 * np.pi * fz_stack)) * dx_v * dy_v * dz_v / (dx * dy)

    else:
        # find indices in bounds
        to_use = np.logical_and.reduce((zind >= 0, zind < nz_v,
                                        yind >= 0, yind < ny_v,
                                        xind >= 0, xind < nx_v,
                                        detectable))

        # round coordinates to nearest values
        zind_round = np.round(zind).astype(int)
        yind_round = np.round(yind).astype(int)
        xind_round = np.round(xind).astype(int)
        inds_round = (zind_round, yind_round, xind_round)

        # construct sparse matrix non-zero coords

        # row_index = position in E
        row_index = np.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]

        # columns = position in V
        inds_to_use = tuple([i[to_use] for i in inds_round])
        column_index = np.ravel_multi_index(inds_to_use, v_shape)

        # matrix values
        data = np.ones(len(row_index)) / (2 * 1j * (2 * np.pi * fz[to_use])) * dx_v * dy_v * dz_v / (dx * dy)

    # construct sparse matrix
    # E(k) = model * V(k)
    model = sp.csr_matrix((data, (row_index, column_index)), shape=(nimgs * ny * nx, v_size))

    return model, (data, row_index, column_index)


def reconstruction(efield_fts, beam_frqs, no, na_det, wavelength,
                   dxy: float,
                   z_fov: float = 10.,
                   regularization: float = 0.1,
                   dz_sampling_factor: float = 1,
                   dxy_sampling_factor: float = 1,
                   mode: str = "born",
                   no_data_value: float = np.nan):
    """
    Given a set of holograms obtained using ODT, put the hologram information back in the correct locations in
    Fourier space

    Note: coordinates can be obtained from ...

    @param efield_fts: The exact definition of efield_fts depends on whether "born" or "rytov" mode is used.
    Any points in efield_fts which are NaN will be ignored
    @param beam_frqs: nimgs x 3, where each is [vx, vy, vz] and vx**2 + vy**2 + vz**2 = n**2 / wavelength**2
    @param no: background index of refraction
    @param na_det: detection numerical aperture
    @param wavelength:
    @param dxy: pixel size
    @param z_fov: z-field of view
    @param regularization: regularization factor
    @param dz_sampling_factor: fraction of Nyquist sampling factor to use
    @param dxy_sampling_factor:
    @param mode: "born" or "rytov"
    @return v_ft:
    @return drs: full coordinate grid can be obtained from get_coords
    """

    nimgs, ny, nx = efield_fts.shape[-3:]

    # ##################################
    # set sampling of 3D scattering potential
    # ##################################
    drs, (nz_v, ny_v, nx_v) = get_reconstruction_sampling(no,
                                                             na_det,
                                                             beam_frqs,
                                                             wavelength,
                                                             dxy,
                                                             (ny, nx),
                                                             z_fov,
                                                             dz_sampling_factor=dz_sampling_factor,
                                                             dxy_sampling_factor=dxy_sampling_factor)

    # ###########################
    # put information back in Fourier space
    # ###########################
    v_shape = (nz_v, ny_v, nx_v)

    model, (data, row_index, col_index) = fwd_model_linear(beam_frqs, no, na_det, wavelength,
                                                           (ny, nx),
                                                           (dxy, dxy),
                                                           v_shape,
                                                           drs,
                                                           mode=mode,
                                                           interpolate=False)

    # recover indices into V and E
    v_ind = np.unravel_index(col_index, v_shape)
    e_ind = np.unravel_index(row_index, (nimgs, ny, nx))


    # put information in correct space in Fourier space
    v_ft = np.zeros(v_shape, dtype=complex)
    num_pts_all = np.zeros(v_shape, dtype=int)

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
    is_data = np.logical_not(no_data)

    v_ft[is_data] = v_ft[is_data] / (num_pts_all[is_data] + regularization)
    v_ft[no_data] = no_data_value

    return v_ft, drs


def apply_n_constraints(sp_ft, no, wavelength, n_iterations=100, beta=0.5, use_raar=True,
                        require_real_part_greater_bg=False, use_gpu=_cupy_available, print_info=False):
    """
    Apply constraints on the scattering potential and the index of refraction using iterative projection

    constraint 1: scattering potential FT must match data at points where we information
    constraint 2: real(n) >= no and imag(n) >= 0

    @param sp_ft: 3D fourier transform of scattering potential. This array should have nan values where the array
     values are unknown.
    @param no: background index of refraction
    @param wavelength: wavelength in um
    @param n_iterations: number of iterations
    @param beta:
    @param bool use_raar: whether or not to use the Relaxed-Averaged-Alternating Reflection algorithm
    @return scattering_pot_ft:
    """
    # scattering_potential masked with nans where no information
    sp_ft = np.array(sp_ft, copy=True)
    sp_data = np.array(sp_ft, copy=True)

    if not np.any(np.isnan(sp_ft)):
        raise ValueError("sp_ft contained no NaN's, so there is no information to infer")

    no_data = np.isnan(sp_ft)
    is_data = np.logical_not(no_data)
    sp_ft[no_data] = 0

    # try smoothing image first ...
    # todo: is this useful?
    # sp_ft = gaussian_filter(sp_ft, (4, 4, 4))

    tstart = time.perf_counter()
    for ii in range(n_iterations):
        if print_info:
            print("constraint iteration %d/%d, elapsed time = %0.2fs" % (ii + 1, n_iterations, time.perf_counter() - tstart), end="\r")
            if ii == (n_iterations - 1):
                print("")

        # ############################
        # ensure n is physical
        # ############################

        # todo: can do imaginary part with the scattering potential instead

        if not use_gpu:
            sp = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
        else:
            sp = cp.asnumpy(cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(sp_ft, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))
        n = get_n(sp, no, wavelength)

        if require_real_part_greater_bg:
            # real part must be >= no
            correct_real = np.real(n) < no
            n[correct_real] = no + 1j * np.imag(n[correct_real])

        # imaginary part must be >= 0
        correct_imag = np.imag(n) < 0
        n[correct_imag] = np.real(n[correct_imag]) + 0*1j

        sp_ps = get_scattering_potential(n, no, wavelength)
        if not use_gpu:
            sp_ps_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
        else:
            sp_ps_ft = cp.asnumpy(cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(sp_ps, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))

        if use_raar:
            sp_ft_pm = np.array(sp_ft, copy=True)
            sp_ft_pm[is_data] = sp_data[is_data]

            # ############################
            # projected Ps * Pm
            # ############################
            sp_ft_ps_pm = np.array(sp_ft_pm, copy=True)
            if not use_gpu:
                sp_ps_pm = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
            else:
                sp_ps_pm = cp.asnumpy(cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(sp_ft_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))
            n_ps_pm = get_n(sp_ps_pm, no, wavelength)

            if require_real_part_greater_bg:
                # real part must be >= no
                correct_real = np.real(n_ps_pm) < no
                n_ps_pm[correct_real] = no + np.imag(n_ps_pm[correct_real])

            # imaginary part must be >= 0
            correct_imag = np.imag(n_ps_pm) < 0
            n_ps_pm[correct_imag] = np.real(n_ps_pm[correct_imag]) + 0 * 1j

            sp_ps_pm = get_scattering_potential(n_ps_pm, no, wavelength)
            if not use_gpu:
                sp_ps_pm_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
            else:
                sp_ps_pm_ft = cp.asnumpy(cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(sp_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))

            # update
            sp_ft = beta * sp_ft - beta * sp_ps_ft + (1 - 2 * beta) * sp_ft_pm + 2 * beta * sp_ps_pm_ft
        else:
            # ############################
            # projected Pm * Ps
            # ############################
            sp_ft_pm_ps = np.array(sp_ps_ft, copy=True)
            sp_ft_pm_ps[is_data] = sp_data[is_data]

            # update
            sp_ft = sp_ft_pm_ps

    return sp_ft


# fit frequencies
def fit_ref_frq(img_ft, dxy, fmax_int, search_rad_fraction=1, npercentiles=50, filter_size=0, show_figure=False):
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
    @param img:
    @param dxy:
    @param fmax_int:
    @param search_rad_fraction:
    @param npercentiles:
    @return results, circ_dbl_fn, figh: results["fit_params"] = [cx, cy, radius]
    """
    ny, nx = img_ft.shape
    # fourier transforms
    # img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img)))
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
    @param img_efield_scattered_ft:
    @param beam_frq:
    @param frq_ref:
    @param fmax_int:
    @param fcoords:
    @param dxy:
    @param title:
    @return figh:
    """

    # real-space coordinates
    ny, nx = img_efield_ft.shape
    x = (np.arange(nx) - nx // 2) * dxy
    y = (np.arange(ny) - ny // 2) * dxy
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



def plot_odt_sampling(frqs, na_detect, na_excite, ni, wavelength, figsize=(30, 8)):
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

    # pp, tt = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, alpha_det, 30))
    # pp = pp.ravel()
    # tt = tt.ravel()

    # kx0, ky0, kz0
    # fxyz0 = np.stack((np.cos(pp) * np.sin(tt), np.sin(pp) * np.sin(tt), np.cos(tt)), axis=1) * ni / wavelength
    fxyz0 = np.stack((fxfx, fyfy, fzfz), axis=-1)

    # ax.plot(fxyz0[:, 0], fxyz0[:, 1], 'r', zs=fxyz0[:, 2] - ni/wavelength)
    for ii in range(len(frqs_3d)):
        # ax.plot(fxyz0[..., 0] - frqs_3d[ii, 0], fxyz0[..., 1] - frqs_3d[ii, 1], 'k', zs=fxyz0[..., 2] - frqs_3d[ii, 2], alpha=0.3)
        ax.plot_surface(fxyz0[..., 0] - frqs_3d[ii, 0], fxyz0[..., 1] - frqs_3d[ii, 1], fxyz0[..., 2] - frqs_3d[ii, 2], alpha=0.3)


    ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
    ax.set_ylim([-2 * frq_norm, 2 * frq_norm])
    ax.set_zlim([-1, 1]) # todo: set based on na's

    ax.set_xlabel("$f_x$ (1/$\mu m$)")
    ax.set_ylabel("$f_y$ (1/$\mu m$)")
    ax.set_zlabel("$f_z$ (1/$\mu m$)")

    return figh


def plot_n3d(n, no, coords=None, title=""):
    """
    Plot 3D index of refraction
    @param v_ft: 3D Fourier transform of scattering potential
    @param no: background index of refraction
    @param coords: (x, y, z)
    @return:
    """

    nz_v, ny_v, nx_v = n.shape

    # ####################
    # real-space coordinates
    # ####################
    if coords is None:
        x_v = np.arange(nx_v) - nx_v // 2
        y_v = np.arange(ny_v) - ny_v // 2
        z_v = np.arange(nz_v) - nz_v // 2
    else:
        x_v, y_v, z_v = coords

    dxy_v = x_v[1] - x_v[0]

    extent_v_xy = [x_v[0] - 0.5 * dxy_v, x_v[-1] + 0.5 * dxy_v,
                   y_v[0] - 0.5 * dxy_v, y_v[-1] + 0.5 * dxy_v]

    # ####################
    # fourier-space coordinates
    # ####################
    fx = fft.fftshift(fft.fftfreq(nx_v, dxy_v))
    dfx_sp = fx[1] - fx[0]
    fy = fft.fftshift(fft.fftfreq(ny_v, dxy_v))
    dfy_sp = fy[1] - fy[0]

    extent_fxy = [fx[0] - 0.5 * dfx_sp, fx[-1] + 0.5 * dfx_sp,
                  fy[0] - 0.5 * dfy_sp, fy[-1] + 0.5 * dfy_sp]

    # FT
    n_ft = fft.fftshift(fft.fftn(fft.ifftshift(n)))

    vmax_real = 1.5 * np.percentile((np.real(n) - no), 99.99)
    if vmax_real <= 0:
        vmax_real = 1e-12

    vmax_imag = 1.5 * np.percentile(np.imag(n), 99.99)
    if vmax_imag <= 0:
        vmax_imag = 1e-12

    vmax_n_ft = 1.5 * np.percentile(np.abs(n_ft), 99.99)
    if vmax_n_ft <= 0:
        vmax_n_ft = 1e-12

    # plots
    fmt_fn = lambda x: "%0.6f" % x

    figh = plt.figure(figsize=(16, 8))
    figh.suptitle(f"Index of refraction, {title:s}")
    grid = figh.add_gridspec(4, nz_v + 1)

    for ii in range(nz_v):
        ax = figh.add_subplot(grid[0, ii])
        ax.set_title("%0.1fum" % z_v[ii])
        im = ax.imshow(np.real(n[ii]) - no,
                       vmin=-vmax_real, vmax=vmax_real, cmap="RdBu",
                       origin="lower", extent=extent_v_xy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])

        if ii == 0:
            ax.set_ylabel("real(n) - no")

        ax = figh.add_subplot(grid[1, ii])
        im = ax.imshow(np.imag(n[ii]),
                       vmin=-vmax_imag, vmax=vmax_imag, cmap="RdBu",
                       origin="lower", extent=extent_v_xy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("imag(n)")

        ax = figh.add_subplot(grid[2, ii])
        im = ax.imshow(np.abs(n_ft[ii]), cmap="copper", norm=PowerNorm(gamma=0.1, vmin=0, vmax=vmax_n_ft),
                       origin="lower", extent=extent_fxy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|n(f)|")

        # ax = figh.add_subplot(grid[3, ii])
        # im = ax.imshow(np.abs(v_ft[ii]), cmap="copper", norm=PowerNorm(gamma=0.1, vmin=0, vmax=vmax_sp_ft),
        #                origin="lower", extent=extent_fxy)
        # im.format_cursor_data = fmt_fn
        # if ii == 0:
        #     ax.set_ylabel("|F(f)|")
        # ax.set_xticks([])
        # ax.set_yticks([])

    ax = figh.add_subplot(grid[0, nz_v])
    ax.axis("off")
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax_real, vmax=vmax_real), cmap="RdBu"), ax=ax)

    ax = figh.add_subplot(grid[1, nz_v])
    ax.axis("off")
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax_imag, vmax=vmax_imag), cmap="RdBu"), ax=ax)

    return figh


def compare_v3d(v_ft, v_ft_gt, ttl="", figsize=(35, 20), gamma=0.1, nmax_columns=21):
    """
    Display slices of scattering potential in 3D. Show both real-space and Fourier-space

    @param v_ft:
    @param v_ft_gt:
    @param ttl:
    @param figsize:
    @param gamma: gamma applied to FT images only
    @param nmax_columns: maximum number of columns per figure. If number of slices of v_ft exceeds this, create more
    figures to display
    @return fighs: list of figures displaying real-space scattering potential
    @return fighs_ft:
    """

    fighs = []
    fighs_ft = []

    v = fft.fftshift(fft.ifftn(fft.ifftshift(v_ft)))
    v_gt = fft.fftshift(fft.ifftn(fft.ifftshift(v_ft_gt)))

    # set display limits
    vminr = np.min([1.2 * np.min(v_gt.real), -0.01])
    vmaxr = 0
    vmini = np.min([1.2 * np.min(v_gt.imag), -0.1])
    vmaxi = -vmini
    vmink = 0
    vmaxk = 1.2 * np.max(np.abs(v_ft_gt))

    # create figures
    nmax_columns = np.min([nmax_columns, v.shape[0]])
    nfigs = int(np.ceil(v.shape[0] / nmax_columns))

    for ii in range(nfigs):
        figh = plt.figure(figsize=figsize)
        figh.suptitle(f"{ttl:s} ({ii + 1:d}/{nfigs:d})")
        grid = figh.add_gridspec(nrows=6, ncols=nmax_columns + 1, width_ratios=[1] * nmax_columns + [0.1])

        fighs.append(figh)

    for ii in range(v.shape[0]):
        fig_num = ii % nmax_columns
        col = ii - (fig_num * nmax_columns)
        figh = fighs[nfigs]

        ax = figh.add_subplot(grid[0, col])
        im = ax.imshow(v[ii].real, vmin=vminr, vmax=vmaxr, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Re(v)")

        if col == (nmax_columns - 1):
            ax = figh.add_subplot(grid[0, col + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[1, col])
        im = ax.imshow(v_gt[ii].real, vmin=vminr, vmax=vmaxr, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Re(v gt)")

        if col == (nmax_columns - 1):
            ax = figh.add_subplot(grid[1, col + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[2, col])
        im = ax.imshow(v[ii].real - v_gt[ii].real, vmin=vminr, vmax=-vminr, cmap="PiYG")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Re(v - v gt)")

        if col == (nmax_columns - 1):
            ax = figh.add_subplot(grid[2, col + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[3, col])
        im = ax.imshow(v[ii].imag, vmin=vmini, vmax=vmaxi, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Im(v)")

        if col == (nmax_columns - 1):
            ax = figh.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[4, ii])
        im = ax.imshow(v_gt[ii].imag, vmin=vmini, vmax=vmaxi, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Im(v gt)")

        if col == (nmax_columns - 1):
            ax = figh.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[5, ii])
        im = ax.imshow(v[ii].imag - v_gt[ii].imag, vmin=vmini, vmax=vmaxi, cmap="PiYG")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Im(v - v gt)")

        if col == (nmax_columns - 1):
            ax = figh.add_subplot(grid[5, ii + 1])
            plt.colorbar(im, cax=ax)

    # k-space
    for ii in range(nfigs):
        figh_ft = plt.figure(figsize=figsize)
        figh_ft.suptitle(f"{ttl:s} k-space ({ii + 1:d}/{nfigs:d})")
        grid = figh_ft.add_gridspec(nrows=5, ncols=v.shape[0] + 1, width_ratios=[1] * v.shape[0] + [0.1])

        fighs_ft.append(figh_ft)

    for ii in range(v.shape[0]):
        fig_num = ii % nmax_columns
        col = ii - (fig_num * nmax_columns)
        figh_ft = fighs_ft[nfigs]

        ax = figh_ft.add_subplot(grid[0, col])
        im = ax.imshow(np.abs(v_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=vmink, vmax=vmaxk), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("|V(k)|")

        if col == (nmax_columns - 1):
            ax = figh_ft.add_subplot(grid[0, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[1, col])
        im = ax.imshow(np.abs(v_ft_gt[ii]), norm=PowerNorm(gamma=gamma, vmin=vmink, vmax=vmaxk), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("|V gt(k)|")

        if col == (nmax_columns - 1):
            ax = figh_ft.add_subplot(grid[1, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[2, col])
        im = ax.imshow(np.abs(v_ft[ii] - v_ft_gt[ii]), norm=PowerNorm(gamma=gamma, vmin=vmink, vmax=vmaxk), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("|V - V gt(k)|")

        if col == (nmax_columns - 1):
            ax = figh_ft.add_subplot(grid[2, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[3, col])
        im = ax.imshow(np.angle(v_ft[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Angle(V(k))")

        if col == (nmax_columns - 1):
            ax = figh_ft.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[4, col])
        im = ax.imshow(np.angle(v_ft_gt[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Angle(V gt(k))")

        if col == (nmax_columns - 1):
            ax = figh_ft.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

    return fighs, fighs_ft


def compare_escatt(e, e_gt, beam_frqs, dxy, ttl="", figsize=(35, 20), gamma=0.1):
    """

    @param e:
    @param e_gt:
    @param beam_frqs:
    @param dxy:
    @param ttl:
    @param figsize:
    @return:
    """
    nbeams, ny, nx = e.shape

    x, y = np.meshgrid((np.arange(nx) - nx // 2) * dxy,
                       (np.arange(ny) - ny // 2) * dxy)

    e_gt_ft = fft.fftshift(fft.fft2(fft.fftshift(e_gt, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    e_ft = fft.fftshift(fft.fft2(fft.fftshift(e, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    # display limits
    vmaxe = 1.2 * np.max(np.abs(e_gt))
    vmaxek = 1.2 * np.max(np.abs(e_gt_ft))

    # compare electric fields
    figh = plt.figure(figsize=figsize)
    figh.suptitle(f"{ttl:s} real space")
    grid = figh.add_gridspec(nrows=5, ncols=nbeams + 1, width_ratios=[1] * nbeams + [0.1])
    for ii in range(nbeams):
        shift_fact = np.exp(-1j * 2*np.pi * (beam_frqs[ii, 0] * x + beam_frqs[ii, 1] * y))

        ax = figh.add_subplot(grid[0, ii])
        im = ax.imshow(np.abs(e[ii]), vmin=0, vmax=vmaxe, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E(r)|")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[0, ii + 1])
            plt.colorbar(im, cax=ax)


        ax = figh.add_subplot(grid[1, ii])
        im = ax.imshow(np.abs(e_gt[ii]), vmin=0, vmax=vmaxe, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E gt(r)|")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[1, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[2, ii])
        im = ax.imshow(np.abs(e[ii] - e_gt[ii]), vmin=0, vmax=vmaxe, cmap="Greens")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E - E gt(r)|")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[2, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[3, ii])
        im = ax.imshow(np.angle(e[ii] * shift_fact), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E(r)) shifted")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[4, ii])
        im = ax.imshow(np.angle(e_gt[ii] * shift_fact), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E gt(r)) shifted")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

    # k-space
    figh_ft = plt.figure(figsize=figsize)
    figh_ft.suptitle(f"{ttl:s} k-space")
    grid = figh_ft.add_gridspec(nrows=5, ncols=nbeams + 1, width_ratios=[1] * nbeams + [0.1])
    for ii in range(nbeams):
        ax = figh_ft.add_subplot(grid[0, ii])
        im = ax.imshow(np.abs(e_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmaxek), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E(k)|")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[0, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[1, ii])
        im = ax.imshow(np.abs(e_gt_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmaxek), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E gt(k)|")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[1, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[2, ii])
        im = ax.imshow(np.abs(e_ft[ii] - e_gt_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmaxek), cmap="Greens")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E - E gt(k)|")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[2, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[3, ii])
        im = ax.imshow(np.angle(e_ft[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E(k))")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[4, ii])
        im = ax.imshow(np.angle(e_gt_ft[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E gt(k))")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

    return figh, figh_ft