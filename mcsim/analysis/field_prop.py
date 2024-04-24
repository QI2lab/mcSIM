"""
Numerical beam propagation through homogeneous and inhomogeneous media

Here we assume complex electric fields are phasors coming from the convention exp(ikr - iwt)
see e.g. get_angular_spectrum_kernel()
This is natural when working with discrete Fourier transforms
"""

from typing import Union, Optional
from collections.abc import Sequence
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from mcsim.analysis.fft import ft2, ift2, ft3, ift3
from mcsim.analysis.optimize import Optimizer, soft_threshold, tv_prox

try:
    import cupy as cp
    import cupyx.scipy.sparse as sp_gpu
except ImportError:
    cp = None
    sp_gpu = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray

if sp_gpu:
    csr_matrix = Union[sp.csr_matrix, sp_gpu.csr_matrix]
else:
    csr_matrix = sp.csr_matrix


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
    if cp and isinstance(v, cp.ndarray):
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
    :return v:
    """
    v = - (2 * np.pi / wavelength) ** 2 * (n**2 - no**2)
    return v


def frqs2angles(frqs: array,
                magnitude: float = 1.) -> (array, array):
    """
    Convert from frequency vectors to angle vectors

    :param frqs: N x 3 array with order (fx, fy, fz).
      Frequency vectors should be normalized such that
      norm(frqs, axis=1) = no / wavelength
    :param magnitude: norm of frequency vectors. For frequency (in "hertz") this is  no/wavelength
      while for angular frequency (in "radians") this is 2*np.pi * no / wavelength
    :return: theta, phi
    """

    if cp and isinstance(frqs, cp.ndarray):
        xp = cp
    else:
        xp = np

    frqs = xp.atleast_2d(frqs)

    # magnitude = (no / wavelength)
    with np.errstate(invalid="ignore"):
        theta = xp.array(np.arccos(xp.dot(frqs, xp.array([0, 0, 1])) / magnitude))
        theta[xp.isnan(theta)] = 0
        phi = xp.angle(frqs[..., 0] + 1j * frqs[..., 1])
        phi[xp.isnan(phi)] = 0

    return theta, phi


def angles2frqs(theta: array,
                phi: array,
                magnitude: float = 1.) -> array:
    """
    Get frequency vector from angles. Inverse function for frqs2angles()

    :param theta:
    :param phi:
    :param magnitude: e.g. no/wavelength
    :return frqs:
    """

    if cp and isinstance(theta, cp.ndarray):
        xp = cp
    else:
        xp = np

    phi = xp.asarray(phi)

    # magnitude = no / wavelength
    fz = magnitude * xp.cos(theta)
    fy = magnitude * xp.sin(theta) * xp.sin(phi)
    fx = magnitude * xp.sin(theta) * xp.cos(phi)
    f = xp.stack((fx, fy, fz), axis=1)

    return f


def get_fzs(fx: array,
            fy: array,
            no: float,
            wavelength: float) -> array:
    """
    Get z-component of frequency given fx, fy. For frequencies where the resulting wave
    is evanescent, return NaN values.

    :param fx: fx and fy should be broadcastable to the same size
    :param fy:
    :param no: index of refraction
    :param wavelength: wavelength
    :return fzs:
    """

    if cp and isinstance(fx, cp.ndarray):
        xp = cp
    else:
        xp = np
    fy = xp.asarray(fy)

    with np.errstate(invalid="ignore"):
        fzs = xp.sqrt(no**2 / wavelength ** 2 - fx**2 - fy**2)

    return fzs


def get_angular_spectrum_kernel(fx: array,
                                fy: array,
                                dz: float,
                                wavelength: float,
                                no: float) -> array:
    """
    Get the angular spectrum/plane wave expansion kernel for propagating an electric field by distance dz.
    Here we assume a propagating plane wave has the form Re{\\exp[i(kz - \\omega t)]}. That is,
    phasors carry the implicit time dependence \\exp[-i \\omega t].

    Typically, this kernel is useful when working with DFT's. In that case, frequencies might be generated like this

    >>> from numpy.fft import fftshift, fftfreq
    >>> nxy = 101
    >>> dxy = 0.1
    >>> fx = fftshift(fftfreq(nxy, dxy))[None, :]
    >>> fy = fftshift(fftfreq(nxy, dxy))[:, None]
    >>> k = get_angular_spectrum_kernel(fx, fy, dz=0.1, wavelength=0.532, no=1.333)

    :param fx: x-spatial frequencies (1/cycles)
    :param fy: y-spatial frequencies
    :param dz: propagation distance along the optical axis
    :param wavelength:
    :param no: background refractive index
    :return kernel:
    """

    if cp and isinstance(fx, cp.ndarray):
        xp = cp
    else:
        xp = np
    fy = xp.asarray(fy)

    fzs = get_fzs(fx, fy, no, wavelength)
    allowed = xp.logical_not(xp.isnan(fzs))
    kernel = xp.zeros(fzs.shape, dtype=complex)
    kernel[allowed] = xp.exp(1j * dz * 2*np.pi * fzs[allowed])

    return kernel


def propagation_kernel(fx: array,
                       fy: array,
                       dz: float,
                       wavelength: float,
                       no: float) -> array:
    """
    Propagation kernel for field represented by field value and derivative. Note that this can alternatively be
    understood as a field which contains both forward and backward propagating components

    :param fx:
    :param fy:
    :param dz:
    :param wavelength:
    :param no:
    :return kernel:
    """

    if cp and isinstance(fx, cp.ndarray):
        xp = cp
    else:
        xp = np

    kzs = 2*np.pi*get_fzs(fx, fy, no, wavelength)
    allowed = xp.logical_not(xp.isnan(kzs))
    kernel = xp.zeros(kzs.shape + (2, 2), dtype=float)
    with np.errstate(invalid="ignore"):
        kz_allowed = kzs[allowed]
        kernel[allowed, 0, 0] = xp.cos(kz_allowed * dz)
        kernel[allowed, 0, 1] = xp.sin(kz_allowed * dz) / kz_allowed
        kernel[allowed, 1, 0] = -kz_allowed * xp.sin(kz_allowed * dz)
        kernel[allowed, 1, 1] = xp.cos(kz_allowed * dz)

    return kernel


def forward_backward_proj(fx: array,
                          fy: array,
                          wavelength: float,
                          no: float) -> array:
    """
    matrix converting from (phi, dphi/dz) -> (phi_f, phi_b) representation

    :param fx:
    :param fy:
    :param wavelength:
    :param no:
    :return kernel:
    """
    if cp and isinstance(fx, cp.ndarray):
        xp = cp
    else:
        xp = np

    kzs = 2*np.pi*get_fzs(fx, fy, no, wavelength)
    allowed = xp.logical_not(xp.isnan(kzs))
    kernel = xp.zeros(kzs.shape + (2, 2), dtype=complex)
    with np.errstate(invalid="ignore"):
        kz_allowed = kzs[allowed]

        kernel[..., 0, 0] = 0.5
        kernel[allowed, 0, 1] = -0.5 * 1j / kz_allowed
        kernel[..., 1, 0] = 0.5
        kernel[allowed, 1, 1] = 0.5 * 1j / kz_allowed

    return kernel


def field_deriv_proj(fx: array,
                     fy: array,
                     wavelength: float,
                     no: float) -> array:
    """
    matrix converting from (phi, dphi/dz) -> (phi_f, phi_b) representation

    :param fx:
    :param fy:
    :param wavelength:
    :param no:
    :return projector:
    """
    if cp and isinstance(fx, cp.ndarray):
        xp = cp
    else:
        xp = np

    kzs = 2*np.pi*get_fzs(fx, fy, no, wavelength)
    allowed = xp.logical_not(xp.isnan(kzs))
    kernel = xp.zeros(kzs.shape + (2, 2), dtype=complex)
    with np.errstate(invalid="ignore"):
        kz_allowed = kzs[allowed]

        kernel[..., 0, 0] = 1
        kernel[..., 0, 1] = 1
        kernel[allowed, 1, 0] = 1j * kz_allowed
        kernel[allowed, 1, 1] = -1j * kz_allowed

    return kernel


def propagate_homogeneous(efield_start: array,
                          zs: Union[float, array],
                          no: float,
                          drs: Sequence[float, float],
                          wavelength: float,
                          adjoint: bool = False) -> array:
    """
    Propagate the Fourier transform of an optical field a distance z through a medium with homogeneous index
    of refraction n using the angular spectrum method

    :param efield_start: electric field to be propagated. n0 x ... x nm x ny x nx array
    :param zs: z-positions to propagate of size nz
    :param no: background refractive index
    :param drs: (dy, dx) pixel size
    :param wavelength: wavelength in the same units as drs and zs
    :param adjoint: if True, perform the adjoint operation instead of beam propagation
    :return efield_prop: propagated electric field of shape no x ... x nm x nz x ny x nx
    """

    if isinstance(efield_start, cp.ndarray):
        xp = cp
    else:
        xp = np

    zs = np.atleast_1d(zs)
    dy, dx = drs

    # prepare output array
    nz = len(zs)
    ny, nx = efield_start.shape[-2:]
    new_size = efield_start.shape[:-2] + (nz, ny, nx)
    efield_ft_prop = xp.zeros(new_size, dtype=complex)

    # frequency grid
    fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
    fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)

    # propagation
    if not adjoint:
        efield_start_ft = ft2(efield_start, shift=False)
    else:
        efield_start_ft = ift2(efield_start, adjoint=True, shift=False)

    for ii in range(len(zs)):
        # construct propagation kernel
        kernel = get_angular_spectrum_kernel(fx, fy, zs[ii], wavelength, no)

        if adjoint:
            xp.conjugate(kernel, out=kernel)

        # propagate field with kernel
        efield_ft_prop[..., ii, :, :] = efield_start_ft * kernel

    if not adjoint:
        efield_prop = ift2(efield_ft_prop, shift=False)
    else:
        efield_prop = ft2(efield_ft_prop, adjoint=True, shift=False)

    return efield_prop


def fwd_model_linear(beam_fx: array,
                     beam_fy: array,
                     beam_fz: array,
                     no: float,
                     na_det: float,
                     wavelength: float,
                     e_shape: Sequence[int, int],
                     drs_e: Sequence[float, float],
                     v_shape: Sequence[int, int, int],
                     drs_v: Sequence[float, float, float],
                     mode: str = "born",
                     interpolate: bool = False,
                     use_gpu: bool = False) -> csr_matrix:
    """
    Forward model from scattering potential v(k) to imaged electric field E(k) after interacting with object.
    Assumes plane wave illumination and linear scattering model (Born or Rytov)

    :param beam_fx: beam frequencies. Either of size npatterns or nmultiplex x npatterns. If not all patterns
      use the same degree of multiplexing, then nmultiplex should be the maximum degree of multiplexing for all patterns
      and the extra frequencies associated with patterns with a lower degree of multiplexing can be set to np.inf
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
    # todo: replace na_det with coherent transfer function?

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
        if cp and use_gpu:
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
            # if we don't copy, then the elements of F's are reference to other elements.
            Fx = xp.array(Fx, copy=True)
            Fy = xp.array(Fy, copy=True)
            Fz = xp.array(Fz, copy=True)

            # indices into the final scattering potential
            # taking advantage of the fact that the final scattering potential indices have FFT structure
            zind = Fz / dfz_v + nz_v // 2
            yind = Fy / dfy_v + ny_v // 2
            xind = Fx / dfx_v + nx_v // 2
        elif mode == "rytov":
            # V(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) =
            # 2*i * (2*pi*fz) * psi_s(fx - n/lambda * nx, fy - n/lambda * ny)
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
            data = (interp_weights_to_use / (2 * 1j * (2 * np.pi * xp.tile(fz[to_use], 8))) *
                    dx_v * dy_v * dz_v / (dx * dy))

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

        # construct sparse matrix
        # E(k) = model * V(k)
        # column index = position in V
        # row index = position in E
        model = spm.csr_matrix((data, (row_index, column_index)), shape=(nimgs * ny * nx, v_size))

    return model


def inverse_model_linear(efield_fts: array,
                         model: csr_matrix,
                         v_shape: Sequence[int, int, int],
                         regularization: float = 0.,
                         no_data_value: float = np.nan) -> array:
    """
    Given a set of holograms obtained using ODT, put the hologram information back in the correct locations in
    Fourier space

    :param efield_fts: The exact definition of efield_fts depends on whether "born" or "rytov" mode is used.
      Any points in efield_fts which are NaN will be ignored. efield_fts can have an arbitrary number of leading
      singleton dimensions, but must have at least three dimensions.
      i.e. it should have shape 1 x ... x 1 x nimgs x ny x nx
    :param model: forward model matrix. Generated from fwd_model_linear(), which should have interpolate=False
    :param v_shape:
    :param regularization: regularization factor
    :param no_data_value: value of any points in v_ft where no data is available
    :return v_ft:
    """

    if cp and isinstance(efield_fts, cp.ndarray):
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


class RIOptimizer(Optimizer):

    def __init__(self,
                 no: float,
                 wavelength: float,
                 drs_e: Sequence[float, float],
                 drs_n: Sequence[float, float, float],
                 shape_n: Sequence[int, int, int],
                 e_measured: Optional[array] = None,
                 e_measured_bg: Optional[array] = None,
                 beam_frqs: Optional[array] = None,
                 dz_refocus: float = 0.,
                 atf: Optional[array] = None,
                 apodization: Optional[array] = None,
                 mask: Optional[array] = None,
                 tau_tv_real: float = 0.,
                 tau_tv_imag: float = 0.,
                 tau_l1_real: float = 0.,
                 tau_l1_imag: float = 0.,
                 use_imaginary_constraint: bool = False,
                 use_real_constraint: bool = False,
                 max_imaginary_part: float = np.inf,
                 efield_cost_factor: float = 1.,
                 **kwargs
                 ):
        """

        :param no: background index of refraction
        :param wavelength: wavelength of coherent light. All other spatial quantities should use the same
          units as wavelength (typically um or nm)
        :param drs_e: (dy, dx) electric field pixel size
        :param drs_n: (dz, dy, dx) refractive index voxel size
        :param shape_n: (nz, ny, nx) refractive index array shape
        :param e_measured: measured electric field of shape npatterns x ny x nx
        :param e_measured_bg: measured electric field backgrounds
        :param beam_frqs: beam frequencies, of shape npatterns x 3
        :param dz_refocus:
        :param atf: amplitude (coherent) transfer function
        :param apodization: apodization applied during 2D FFT's
        :param tau_tv_real: weight for TV proximal operator applied to real-part
        :param tau_tv_imag: weight for TV proximal operator applied to imaginary-part
        :param tau_l1_real:
        :param tau_l1_imag:
        :param use_imaginary_constraint: enforce im(n) > 0
        :param use_real_constraint: enforce re(n) > no
        :param max_imaginary_part:
        :param efield_cost_factor:
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
        self.dz_refocus = dz_refocus

        if atf is None:
            atf = 1.
        self.atf = atf

        if apodization is None:
            apodization = 1.
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
                 tau_tv_real: float = 0.,
                 tau_tv_imag: float = 0.,
                 tau_l1_real: float = 0.,
                 tau_l1_imag: float = 0.,
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

    def prox(self,
             x: array,
             step: float) -> array:

        x_real = x.real
        x_imag = x.imag

        # ###########################
        # L1 proximal operators (softmax)
        # ###########################
        if self.prox_parameters["tau_l1_real"] != 0:
            x_real = soft_threshold(self.prox_parameters["tau_l1_real"] * step, x_real - self.no) + self.no

        if self.prox_parameters["tau_l1_imag"] != 0:
            x_imag = soft_threshold(self.prox_parameters["tau_l1_imag"] * step, x_imag)

        # ###########################
        # TV proximal operators
        # ###########################
        # note cucim TV implementation requires ~10x memory as array does
        if self.prox_parameters["tau_tv_real"] != 0:
            x_real = tv_prox(x_real, self.prox_parameters["tau_tv_real"] * step)

        if self.prox_parameters["tau_tv_imag"] != 0:
            x_imag = tv_prox(x_imag, self.prox_parameters["tau_tv_imag"] * step)

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

    def cost(self,
             x: array,
             inds: Optional[Sequence[int]] = None) -> array:
        if inds is None:
            inds = list(range(self.n_samples))

        # todo: how to unify these?
        e_fwd = self.fwd_model(x, inds=inds)[:, -1]

        costs = 0
        if self.efield_cost_factor > 0:
            if self.mask is None:
                costs += (self.efield_cost_factor * 0.5 *
                          (abs(e_fwd - self.e_measured[inds]) ** 2).sum(axis=(-1, -2))
                          )
            else:
                costs += (self.efield_cost_factor * 0.5 *
                          (abs(e_fwd[:, self.mask] - self.e_measured[inds][:, self.mask]) ** 2).sum(axis=-1)
                          )

        if (1 - self.efield_cost_factor) > 0:
            if self.mask is None:
                costs += ((1 - self.efield_cost_factor) * 0.5 *
                          (abs(abs(e_fwd) - abs(self.e_measured[inds])) ** 2).sum(axis=(-1, -2)))
            else:
                costs += ((1 - self.efield_cost_factor) * 0.5 *
                          (abs(abs(e_fwd[:, self.mask]) - abs(self.e_measured[inds][:, self.mask])) ** 2).sum(axis=-1))

        return costs

    def propagate(self,
                  efield_start: array,
                  n: array) -> array:
        pass

    def backpropagate(self,
                      efield_start: array,
                      n: array) -> array:
        pass


class LinearScatt(RIOptimizer):
    def __init__(self,
                 no: float,
                 wavelength: float,
                 drs_e: Sequence[float, float],
                 drs_n: Sequence[float, float, float],
                 shape_n: Sequence[int, int, int],
                 eft: array,
                 model: csr_matrix,
                 **kwargs
                 ):
        """
        Born and Rytov optimizer.

        :param no:
        :param wavelength:
        :param drs_e:
        :param drs_n:
        :param shape_n:
        :param eft: n_angles x ny x nx array Fourier-transform of the electric field.
          This will be either the scattered field or the Rytov phase depending on the linear model chosen.
        :param model: The matrix relating the measured field and the scattering potential. This should be generated
          with fwd_model_linear(). Note that the effect of beam frqs and the atf is incorporated in model
        :param **kwargs: parameters passed through to RIOptimizer, including parameters for proximal operator
        """

        # note that have made different choices for LinearScatt optimizers vs. the multislice
        # but still inheriting from RIOptimizer for convenience in constructing proximal operator
        super(LinearScatt, self).__init__(no,
                                          wavelength,
                                          drs_e,
                                          drs_n,
                                          shape_n,
                                          eft,
                                          **kwargs)

        if self.dz_refocus != 0.:
            raise NotImplementedError(f"LinearScatt models do not yet support dz_refocus != 0")

        if self.efield_cost_factor != 1:
            raise NotImplementedError(f"Linear scattering models only support self.efield_cost_factor=1, "
                                      f"but value was {self.efield_cost_factor:.3f}")

        if cp and isinstance(self.e_measured, cp.ndarray):
            self.model = sp_gpu.csr_matrix(model)
        else:
            self.model = model

    def fwd_model(self,
                  x: array,
                  inds: Optional[Sequence[int]] = None) -> array:
        if inds is None:
            inds = list(range(self.n_samples))

        # todo: not using this in gradient/cost because need to manipulate models there

        if cp and isinstance(self.model, sp_gpu.csr_matrix):
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

    def gradient(self,
                 x: array,
                 inds: Optional[Sequence[int]] = None) -> array:

        if inds is None:
            inds = list(range(self.n_samples))

        if sp_gpu and isinstance(self.model, sp_gpu.csr_matrix):
            spnow = sp_gpu
            xp = cp
        else:
            spnow = sp
            xp = np

        ny, nx = self.e_measured.shape[-2:]
        models = [self.model[slice(ny*nx*ii, ny*nx*(ii + 1)), :] for ii in inds]
        nind = len(inds)

        # division by ny*nx converts Fourier space to real-space sum
        # factor of 0.5 in cost function killed by derivative factor of 2
        efwd = spnow.vstack(models).dot(x.ravel()).reshape([nind, ny, nx])
        dc_dm = (efwd - self.e_measured[inds]) / (ny * nx)

        dc_dv = xp.stack([((dc_dm[ii].conj()).ravel()[None, :] * m.tocsc()).conj().reshape(x.shape)
                          for ii, m in enumerate(models)], axis=0)

        return dc_dv

    def cost(self,
             x: array,
             inds: Optional[Sequence[int]] = None) -> array:

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

        return 0.5 * (abs(efwd - self.e_measured[inds]) ** 2).sum(axis=(-1, -2)) / (nx * ny)

    def guess_step(self,
                   x: Optional[array] = None) -> float:
        ny, nx = self.e_measured.shape[-2:]

        if sp_gpu and isinstance(self.model, sp_gpu.csr_matrix):
            u, s, vh = svds(self.model.get(), k=1, which='LM')
        else:
            u, s, vh = svds(self.model, k=1, which='LM')

        lipschitz_estimate = s ** 2 / (self.n_samples * ny * nx)
        return float(1 / lipschitz_estimate)

    def prox(self,
             x: array,
             step: float) -> array:
        # convert from V to n
        n = get_n(ift3(x), self.no, self.wavelength)
        # apply proximal operator on n
        n_prox = super(LinearScatt, self).prox(n, step)

        return ft3(get_v(n_prox, self.no, self.wavelength))

    def run(self,
            x_start: array,
            **kwargs) -> dict:
        return super(LinearScatt, self).run(x_start, **kwargs)


class BPM(RIOptimizer):
    """
    Beam propagation method (BPM). Propagate electric field through medium with index of refraction n(x, y, z)
    using the projection approximation, which is paraxial. That is, first propagate through the background
    medium using the angular spectrum method, and then include the effect of the inhomogeneous refractive
    index in the projection approximation
    """

    def __init__(self,
                 no: float,
                 wavelength: float,
                 drs_e: Sequence[float, float],
                 drs_n: Sequence[float, float, float],
                 shape_n: Sequence[int, int, int],
                 e_measured: Optional[array] = None,
                 e_measured_bg: Optional[array] = None,
                 **kwargs
                 ):
        """
        Suppose we have a 3D grid with nz voxels along the propagation direction. We define the electric field
        at the points before and after each voxel, and in an additional plane to account for the imaging. So we have
        nz + 2 electric field planes.

        :param no: background index of refraction
        :param wavelength: wavelength of light
        :param drs_e: (dy, dx) of the electric field
        :param drs_n: (dz, dy, dx) of the refractive index
        :param shape_n: (nz, ny, nx)
        :param atf: coherent transfer function
        :param apodization: apodization used during FFTs
        :param mask: 2D array. Where true, these spatial pixels will be included in the cost function. Where false,
          they will not
        :param e_measured: measured electric fields
        :param e_measured_bg: measured background electric fields
        :param beam_frqs: n_pattern x 3 array. If provided, modified BPM with extra cosine obliquity factor.
          will be used
        """

        super(BPM, self).__init__(no,
                                  wavelength,
                                  drs_e,
                                  drs_n,
                                  shape_n,
                                  e_measured,
                                  e_measured_bg,
                                  **kwargs)

        # include cosine obliquity factor
        if self.beam_frqs is not None:
            thetas, _ = frqs2angles(self.beam_frqs, self.no / self.wavelength)
        else:
            thetas = np.zeros(self.n_samples)
        self.thetas = thetas

        # distance to propagate beam after last RI voxel
        # if dz_refocus=0, this is the center of the volume
        self.dz_final = (-float(self.drs_n[0] * ((self.shape_n[0] - 1) - self.shape_n[0] // 2 + 0.5)) -
                         self.dz_refocus)

        # backpropagation distance to compute starting field
        self.dz_back = np.array([-float(self.dz_final) -
                                 float(self.drs_n[0]) * self.shape_n[0]
                                 ])

    def propagate(self,
                  efield_start: array,
                  n: array,
                  thetas: Optional[array] = None) -> array:
        """
        Propagate electric field through medium with index of refraction n(x, y, z) using the projection approximation,
        which is paraxial. That is, first propagate through the background medium using the angular spectrum method,
        and then include the effect of the inhomogeneous refractive index in the projection approximation

        :param efield_start: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
        :param n: nz x ny x nx array
        :param thetas: assuming a plane wave input, this is the angle between the plane wave propagation direction
          and the optical axis. This provides a better approximation for the phase shift of the beam through a
          refractive index layer
        :return efield: n0 x ... x nm x nz x ny x nx electric field. Each slice of the array stores the electric field
          on each side of pixels, plus the electric field at the imaging plane. So if there are nz pixels,
          there are nz + 2 planes
        """

        if cp and isinstance(efield_start, cp.ndarray):
            xp = cp
        else:
            xp = np

        if thetas is None:
            thetas = xp.zeros(efield_start.shape[:-2] + (1, 1))
        thetas = xp.asarray(thetas)

        if thetas.ndim == 1:
            thetas = xp.expand_dims(thetas, axis=(-1, -2))

        # ensure arrays of correct type
        efield_start = xp.asarray(efield_start)
        n = xp.asarray(n)
        atf = xp.asarray(self.atf)
        apodization = xp.asarray(self.apodization)

        dz, dy, dx = self.drs_n
        nz, ny, nx = n.shape

        # Fourier space propagation operators
        fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
        fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)
        prop_kernel = xp.asarray(get_angular_spectrum_kernel(fx, fy, dz, self.wavelength, self.no))
        kernel_img = xp.asarray(get_angular_spectrum_kernel(fx, fy, self.dz_final, self.wavelength, self.no))

        # ##########################
        # propagate
        # ##########################
        efield = xp.zeros(efield_start.shape[:-2] + (nz + 2, ny, nx),
                          dtype=complex)
        efield[..., 0, :, :] = efield_start
        for ii in range(nz):
            efield[..., ii + 1, :, :] = (ift2(ft2(efield[..., ii, :, :] * apodization, shift=False) *
                                              prop_kernel, shift=False) *
                                         xp.exp(1j * (2 * np.pi / self.wavelength) * dz *
                                                (n[ii] - self.no) / xp.cos(thetas)))

        # propagate to imaging plane
        efield[..., -1, :, :] = ift2(ft2(efield[..., -2, :, :], shift=False) * kernel_img * atf, shift=False)

        return efield

    def backpropagate(self,
                      efield_end: array,
                      n: array,
                      thetas: Optional[array] = None) -> array:
        """
        Apply the adjoint operation to propagate(). This is adjoint in the sense that for any pair of fields
        a, b we have
        np.sum(a.conj() * propagate_inhomogeneous(b)) = np.sum(backpropagate_inhomogeneous(a).conj() * b)

        :param efield_end: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
        :param n: nz x ny x nx array
        :param thetas: assuming a plane wave input, this is the angle between the plane wave propagation direction
          and the optical axis. This provides a better approximation for the phase shift of the beam through a
          refractive index layer
        :return efield: n0 x ... x nm x nz x ny x nx electric field
        """

        if cp and isinstance(efield_end, cp.ndarray):
            xp = cp
        else:
            xp = np

        efield_end = xp.asarray(efield_end)
        n = xp.asarray(n)
        atf = xp.asarray(self.atf)
        apodization = xp.asarray(self.apodization)

        if thetas is None:
            thetas = xp.zeros(efield_end.shape[:-2] + (1, 1))
        thetas = xp.asarray(thetas)

        if thetas.ndim == 1:
            thetas = xp.expand_dims(thetas, axis=(-1, -2))

        dz, dy, dx = self.drs_n
        nz, ny, nx = n.shape

        # frequency grid
        fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
        fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)

        # propagate
        out_shape = efield_end.shape[:-2] + (nz + 2, ny, nx)

        efield = xp.zeros(out_shape, dtype=complex)
        efield[..., -1, :, :] = efield_end

        prop_kernel = xp.asarray(get_angular_spectrum_kernel(fx, fy, dz, self.wavelength, self.no))

        # propagate from imaging plane back to last plane
        kernel_img = xp.asarray(get_angular_spectrum_kernel(fx, fy, self.dz_final, self.wavelength, self.no))
        efield[..., -2, :, :] = ft2(ift2(efield[..., -1, :, :], adjoint=True, shift=False) *
                                    kernel_img.conj() *
                                    xp.conj(atf),
                                    adjoint=True, shift=False)

        for ii in range(nz - 1, -1, -1):
            efield[..., ii, :, :] = ft2(ift2(efield[..., ii + 1, :, :] *
                                             xp.exp(1j * (2 * np.pi / self.wavelength) * dz *
                                                    (n[ii] - self.no) / xp.cos(thetas)).conj(),
                                             adjoint=True, shift=False) *
                                        prop_kernel.conj(),
                                        adjoint=True, shift=False) * xp.conj(apodization)

        return efield

    def fwd_model(self,
                  x: array,
                  inds: Optional[Sequence[int]] = None) -> array:
        if inds is None:
            inds = list(range(self.n_samples))

        e_start = self.get_estart(inds=inds)
        return self.propagate(e_start, x, thetas=self.thetas[inds])

    def gradient(self,
                 x: array,
                 inds: Optional[Sequence[int]] = None) -> array:
        if inds is None:
            inds = list(range(self.n_samples))

        e_fwd = self.fwd_model(x, inds=inds)

        # back propagation ... build the gradient from this to save memory
        dtemp = 0
        if self.efield_cost_factor > 0:
            dtemp += (self.efield_cost_factor *
                      (e_fwd[:, -1, :, :] - self.e_measured[inds]))

        if (1 - self.efield_cost_factor) > 0:
            dtemp += ((1 - self.efield_cost_factor) *
                      (abs(e_fwd[:, -1, :, :]) - abs(self.e_measured[inds])) *
                      e_fwd[:, -1, :, :] / abs(e_fwd[:, -1, :, :]))

        if self.mask is not None:
            dtemp *= self.mask

        dc_dn = self.backpropagate(dtemp, x, thetas=self.thetas[inds])[:, 1:-1, :, :]
        del dtemp

        # cost function gradient
        if cp and isinstance(x, cp.ndarray):
            xp = cp
        else:
            xp = np

        thetas = xp.asarray(self.thetas[inds])
        if thetas.ndim == 1:
            thetas = xp.expand_dims(thetas, axis=(-1, -2, -3))

        dc_dn *= -1j * (2 * np.pi / self.wavelength) * self.drs_n[0] / xp.cos(thetas)

        # conjugate in place to avoid using extra GPU memory
        xp.conjugate(e_fwd, out=e_fwd)
        dc_dn *= e_fwd[:, 1:-1, :, :]

        return dc_dn

    def get_estart(self,
                   inds: Optional[Sequence[int]] = None) -> array:
        if inds is None:
            inds = list(range(self.n_samples))

        return propagate_homogeneous(self.e_measured_bg[inds],
                                     self.dz_back,
                                     self.no,
                                     self.drs_n[1:],
                                     self.wavelength)[..., 0, :, :]


class SSNP(RIOptimizer):
    def __init__(self,
                 no: float,
                 wavelength: float,
                 drs_e: Sequence[float, float],
                 drs_n: Sequence[float, float, float],
                 shape_n: Sequence[int, int, int],
                 e_measured: Optional[array] = None,
                 e_measured_bg: Optional[array] = None,
                 **kwargs
                 ):

        super(SSNP, self).__init__(no,
                                   wavelength,
                                   drs_e,
                                   drs_n,
                                   shape_n,
                                   e_measured,
                                   e_measured_bg,
                                   **kwargs)

        # todo: maybe get rid of dz_final and dz_back and only have dz_refocus?
        # distance to propagate beam after last RI voxel
        # if dz_refocus=0, this is the center of the volume
        self.dz_final = (-float(self.drs_n[0] * ((self.shape_n[0] - 1) - self.shape_n[0] // 2 + 0.5)) -
                         self.dz_refocus)

        # distance to backpropagate to get starting field
        self.dz_back = np.array([-float(self.dz_final) -
                                 float(self.drs_n[0]) * self.shape_n[0]])

        # compute kzs, which need to get starting field derivative
        if cp and isinstance(self.e_measured, cp.ndarray):
            xp = cp
        else:
            xp = np

        dz, dy, dx = self.drs_n
        ny, nx = self.e_measured.shape[-2:]
        kz = xp.asarray(2 * np.pi * get_fzs(xp.fft.fftfreq(nx, dx)[None, :],
                                            xp.fft.fftfreq(ny, dy)[:, None],
                                            self.no,
                                            self.wavelength))
        kz[xp.isnan(kz)] = 0
        self.kz = kz

    def propagate(self,
                  efield_start: array,
                  de_dz_start: array,
                  n: array) -> array:
        """

        :param efield_start:
        :param de_dz_start:
        :param n:
        :return phi:
        """

        if cp and isinstance(efield_start, cp.ndarray):
            xp = cp
        else:
            xp = np

        n = xp.asarray(n)
        atf = xp.asarray(self.atf)
        apodization = xp.asarray(self.apodization)
        # expand over phi dims + extra dim
        atf = xp.expand_dims(xp.asarray(xp.atleast_2d(atf)), axis=(-1, -2))
        apodization = xp.expand_dims(xp.atleast_2d(xp.asarray(apodization)), axis=(-1, -2))

        dz, dy, dx = self.drs_n
        nz, ny, nx = n.shape

        # construct propagation operators we will need
        fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
        fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)
        p = propagation_kernel(fx, fy, dz, self.wavelength, self.no)
        p_img = propagation_kernel(fx, fy, self.dz_final, self.wavelength, self.no)
        fb_proj = forward_backward_proj(fx, fy, self.wavelength, self.no)[..., slice(0, 1), :]

        # add extra dimension at the end for broadcasting during matmult
        out_shape = efield_start.shape[:-2] + (nz + 2, ny, nx, 2, 1)
        yx_axes = (-3, -4)
        phi = xp.zeros(out_shape, dtype=complex)
        # initialize
        phi[..., 0, :, :, 0, 0] = xp.asarray(efield_start)
        phi[..., 0, :, :, 1, 0] = xp.asarray(de_dz_start)
        # z-step propagation
        for ii in range(nz):
            # refractive index matrix
            q = xp.ones((n.shape[1:]) + (2, 2), dtype=complex)
            q[..., 0, 1] = 0
            q[..., 1, 0] = (2 * np.pi / self.wavelength) ** 2 * (self.no ** 2 - n[ii] ** 2) * dz

            phi[..., ii + 1, :, :, :, :] = ift2(
                xp.matmul(p, ft2(xp.matmul(q, phi[..., ii, :, :, :, :] * apodization),
                                 axes=yx_axes, shift=False)
                          ),
                axes=yx_axes, shift=False)

        # propagate to imaging plane and apply coherent transfer function
        # the last element of phi is fundamentally different from the others because fb_proj changes the basis
        # so this is (phi_f, phi_b) whereas the others are (phi, dphi / dz)
        phi[..., -1, :, :, 0, :] = ift2(xp.matmul(fb_proj, atf * xp.matmul(p_img,
                                                                           ft2(phi[..., -2, :, :, :, :],
                                                                               axes=yx_axes, shift=False))),
                                        axes=yx_axes, shift=False)[..., 0, :]

        # strip off extra dim
        return phi[..., 0]

    def backpropagate(self,
                      efield_end: array,
                      n: array) -> array:
        """
        This acts with the adjoint operators required when taking the gradient

        :param efield_end:
        :param n:
        """

        use_gpu = cp and isinstance(efield_end, cp.ndarray)
        if use_gpu:
            xp = cp
        else:
            xp = np

        n = xp.asarray(n)
        atf = xp.asarray(self.atf)
        apodization = xp.asarray(self.apodization)
        # expand over phi dims + extra dim
        atf = xp.expand_dims(xp.asarray(xp.atleast_2d(atf)), axis=(-1, -2))
        apodization = xp.expand_dims(xp.atleast_2d(xp.asarray(apodization)), axis=(-1, -2))

        dz, dy, dx = self.drs_n
        nz, ny, nx = n.shape

        # construct field propagators
        fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
        fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)
        p = propagation_kernel(fx, fy, dz, self.wavelength, self.no)
        p_img = propagation_kernel(fx, fy, self.dz_final, self.wavelength, self.no)

        p_adj = p.conj().swapaxes(-1, -2)
        p_img_adj = p_img.conj().swapaxes(-1, -2)

        fb_proj_adj = forward_backward_proj(fx,
                                            fy,
                                            self.wavelength,
                                            self.no).conj()[..., slice(0, 1), :].swapaxes(-1, -2)

        # add extra dimension at the end for broadcasting during matmult
        out_shape = efield_end.shape[:-2] + (nz + 2, ny, nx, 2, 1)
        yx_axes = (-3, -4)
        phi = xp.zeros(out_shape, dtype=complex)
        phi[..., -1, :, :, 0, 0] = xp.asarray(efield_end)

        # adjoint of imaging/final prop operation
        fadj = ft2(xp.matmul(p_img_adj,
                             xp.conj(atf) *
                             xp.matmul(fb_proj_adj,
                                       ift2(phi[..., -1, :, :, slice(0, 1), :],
                                            axes=yx_axes,
                                            adjoint=True,
                                            shift=False)
                                       )
                             ),
                   axes=yx_axes,
                   adjoint=True,
                   shift=False)

        # last propagation also
        phi[..., -2, :, :, :, :] = ft2(xp.matmul(p_adj,
                                                 ift2(fadj,
                                                      axes=yx_axes,
                                                      adjoint=True,
                                                      shift=False)
                                                 ),
                                       axes=yx_axes,
                                       adjoint=True,
                                       shift=False)
        # loop backwards through z-stack
        for ii in range(nz - 1, -1, -1):
            q_adj = xp.ones((n.shape[1:]) + (2, 2), dtype=complex)
            q_adj[..., 0, 1] = (2 * np.pi / self.wavelength) ** 2 * (self.no ** 2 - n[ii] ** 2).conj() * dz
            q_adj[..., 1, 0] = 0

            phi[..., ii, :, :, :, :] = ft2(xp.matmul(p_adj,
                                                     ift2(xp.matmul(q_adj,
                                                                    phi[..., ii + 1, :, :, :, :]) *
                                                          xp.conj(apodization),
                                                          axes=yx_axes,
                                                          adjoint=True,
                                                          shift=False)),
                                           axes=yx_axes,
                                           adjoint=True,
                                           shift=False)

        # strip off extra dim
        return phi[..., 0]

    def phi_fwd(self,
                x: array,
                inds: Optional[Sequence[int]] = None) -> array:
        """
        Return the forward model field and the forward model derivative. Since we do not measure the derivative,
        we do not consider it part of our forward models

        :param x:
        :param inds:
        :return:
        """
        if inds is None:
            inds = list(range(self.n_samples))

        e_start, de_dz_start = self.get_estart(inds=inds)
        return self.propagate(e_start, de_dz_start, x)

    def fwd_model(self,
                  x: array,
                  inds: Optional[Sequence[int]] = None) -> array:
        return self.phi_fwd(x, inds=inds)[..., 0]

    def gradient(self,
                 x: array,
                 inds: Optional[Sequence[int]] = None) -> array:
        if cp and isinstance(x, cp.ndarray):
            xp = cp
        else:
            xp = np

        if inds is None:
            inds = list(range(self.n_samples))

        phi_fwd = self.phi_fwd(x, inds=inds)

        # this is the backpropagated field, but we will eventually transform it into the gradient
        # do things this way to reduce memory overhead
        dtemp = 0
        if self.efield_cost_factor > 0:
            dtemp += (self.efield_cost_factor *
                      (phi_fwd[inds, -1, :, :, 0] - self.e_measured[inds]))

        if (1 - self.efield_cost_factor) > 0:
            dtemp += ((1 - self.efield_cost_factor) *
                      (abs(phi_fwd[:, -1, :, :, 0]) - abs(self.e_measured[inds])) *
                      phi_fwd[:, -1, :, :, 0] / abs(phi_fwd[:, -1, :, :, 0]))

        if self.mask is not None:
            dtemp *= self.mask

        # from phi_back, take derivative part
        dc_dn = self.backpropagate(dtemp, x)[:, 1:-1, :, :, 1]
        del dtemp
        dc_dn *= -2 * (2 * np.pi / self.wavelength) ** 2 * self.drs_n[0] * x.conj()

        # conjugate in place to avoid using extra memory
        xp.conjugate(phi_fwd, out=phi_fwd)
        # from phi_fwd, take the electric field part
        dc_dn *= phi_fwd[:, :-2, :, :, 0]

        return dc_dn

    def get_estart(self,
                   inds: Optional[Sequence[int]] = None) -> (array, array):
        if inds is None:
            inds = list(range(self.n_samples))

        e_start = propagate_homogeneous(self.e_measured_bg[inds],
                                        self.dz_back,
                                        self.no,
                                        self.drs_n[1:],
                                        self.wavelength)[..., 0, :, :]
        # assume initial field is forward propagating only
        de_dz_start = ift2(1j * self.kz * ft2(e_start, shift=False), shift=False)

        return e_start, de_dz_start
