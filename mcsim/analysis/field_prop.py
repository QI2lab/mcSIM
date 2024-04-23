"""
Numerical beam propagation through homogeneous and inhomogeneous media

Here we assume complex electric fields are phasors coming from the convention exp(ikr - iwt)
see e.g. get_angular_spectrum_kernel()
This is natural when working with discrete Fourier transforms
"""

from typing import Union, Optional
from collections.abc import Sequence
import numpy as np
from mcsim.analysis.fft import ft2, ift2

try:
    import cupy as cp
except ImportError:
    cp = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


# spatial frequency helper functions
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

        # todo: want to do this?
        # theta = np.atleast_1d(np.arcsin(wavelength / n * np.linalg.norm(frq_2d, axis=-1)))
        # ensure disallowed points return nans
        # disallowed = np.linalg.norm(frq_2d, axis=-1) > n / wavelength
        # phi[disallowed] = np.nan
        # theta[disallowed] = np.nan

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
        kernel = get_angular_spectrum_kernel(fx,
                                             fy,
                                             zs[ii],
                                             wavelength,
                                             no)

        if adjoint:
            xp.conjugate(kernel, out=kernel)

        # propagate field with kernel
        efield_ft_prop[..., ii, :, :] = efield_start_ft * kernel

    if not adjoint:
        efield_prop = ift2(efield_ft_prop, shift=False)
    else:
        efield_prop = ft2(efield_ft_prop, adjoint=True, shift=False)

    return efield_prop


def propagate_bpm(efield_start: array,
                  n: array,
                  no: float,
                  drs: Sequence[float, float, float],
                  wavelength: float,
                  dz_final: float = 0.,
                  atf: Optional[array] = None,
                  apodization: Optional[array] = None,
                  thetas: Optional[array] = None) -> array:
    """
    Propagate electric field through medium with index of refraction n(x, y, z) using the projection approximation,
    which is paraxial. That is, first propagate through the background medium using the angular spectrum method,
    and then include the effect of the inhomogeneous refractive index in the projection approximation

    :param efield_start: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
    :param n: nz x ny x nx array
    :param no: background index of refraction
    :param drs: (dz, dy, dx) voxel size in same units as wavelength. efield is assumed to have the same dy, dx
    :param wavelength: wavelength in same units as drs
    :param dz_final: the distance to propagate the field after the last plane
    :param atf: coherent transfer function
    :param apodization:
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

    efield_start = xp.asarray(efield_start)
    n = xp.asarray(n)

    # ensure other arguments of correct type
    if atf is None:
        atf = 1.
    atf = xp.asarray(atf)

    if apodization is None:
        apodization = 1.
    apodization = xp.asarray(apodization)

    if thetas is None:
        thetas = xp.zeros(efield_start.shape[:-2] + (1, 1))
    thetas = xp.asarray(thetas)

    if thetas.ndim == 1:
        thetas = xp.expand_dims(thetas, axis=(-1, -2))

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    # frequency grid
    fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
    fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(fx, fy, dz, wavelength, no))

    # propagate
    out_shape = efield_start.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[..., 0, :, :] = efield_start
    for ii in range(nz):
        efield[..., ii + 1, :, :] = (ift2(ft2(efield[..., ii, :, :] * apodization, shift=False) *
                                          prop_kernel, shift=False) *
                                     xp.exp(1j * k * dz * (n[ii] - no) / xp.cos(thetas)))

    # propagate to imaging plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(fx, fy, dz_final, wavelength, no))
    efield[..., -1, :, :] = ift2(ft2(efield[..., -2, :, :], shift=False) * kernel_img * atf, shift=False)

    return efield


def backpropagate_bpm(efield_end: array,
                      n: array,
                      no: float,
                      drs: Sequence[float, float, float],
                      wavelength: float,
                      dz_final: float = 0.,
                      atf: Optional[array] = None,
                      apodization: Optional[array] = None,
                      thetas: Optional[array] = None) -> array:
    """
    Apply the adjoint operation to propagate_inhomogeneous(). This is adjoint in the sense that for any pair of fields
    a, b we have
    np.sum(a.conj() * propagate_inhomogeneous(b)) = np.sum(backpropagate_inhomogeneous(a).conj() * b)

    :param efield_end: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
    :param n: nz x ny x nx array
    :param no: background index of refraction
    :param drs: (dz, dy, dx) voxel size in same units as wavelength. efield is assumed to have the same dy, dx
    :param wavelength: wavelength in same units as drs
    :param dz_final:
    :param atf:
    :param apodization:
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

    # ensure other arguments of correct type
    if atf is None:
        atf = 1.
    atf = xp.asarray(atf)

    if apodization is None:
        apodization = 1.
    apodization = xp.asarray(apodization)

    if thetas is None:
        thetas = xp.zeros(efield_end.shape[:-2] + (1, 1))
    thetas = xp.asarray(thetas)

    if thetas.ndim == 1:
        thetas = xp.expand_dims(thetas, axis=(-1, -2))

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    # frequency grid
    fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
    fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)

    # propagate
    out_shape = efield_end.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[..., -1, :, :] = efield_end

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(fx, fy, dz, wavelength, no))

    # propagate from imaging plane back to last plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(fx, fy, dz_final, wavelength, no))
    efield[..., -2, :, :] = ft2(ift2(efield[..., -1, :, :], adjoint=True, shift=False) *
                                kernel_img.conj() *
                                xp.conj(atf),
                                adjoint=True, shift=False)

    for ii in range(nz - 1, -1, -1):
        efield[..., ii, :, :] = ft2(ift2(efield[..., ii + 1, :, :] *
                                         xp.exp(1j * k * dz * (n[ii] - no) / xp.cos(thetas)).conj(),
                                         adjoint=True, shift=False) *
                                    prop_kernel.conj(),
                                    adjoint=True, shift=False) * xp.conj(apodization)

    return efield


def propagate_ssnp(efield_start: array,
                   de_dz_start: array,
                   n: array,
                   no: float,
                   drs: Sequence[float, float, float],
                   wavelength: float,
                   dz_final: float = 0.,
                   atf: Optional[array] = None,
                   apodization: Optional[array] = None) -> array:
    """

    :param efield_start:
    :param de_dz_start:
    :param n:
    :param no:
    :param drs: (dz, dy, dx)
    :param wavelength:
    :param dz_final:
    :param atf:
    :param apodization:
    :return phi:
    """

    use_gpu = cp and isinstance(efield_start, cp.ndarray)
    if use_gpu:
        xp = cp
    else:
        xp = np

    if atf is None:
        atf = 1.

    if apodization is None:
        apodization = 1.

    n = xp.asarray(n)
    # expand over phi dims + extra dim
    atf = xp.expand_dims(xp.asarray(xp.atleast_2d(atf)), axis=(-1, -2))
    apodization = xp.expand_dims(xp.atleast_2d(xp.asarray(apodization)), axis=(-1, -2))

    ko = 2 * np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
    fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)

    # construct operators we will need
    p = propagation_kernel(fx, fy, dz, wavelength, no)
    p_img = propagation_kernel(fx, fy, dz_final, wavelength, no)
    fb_proj = forward_backward_proj(fx, fy, wavelength, no)[..., slice(0, 1), :]

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
        q[..., 1, 0] = ko**2 * (no**2 - n[ii]**2) * dz

        phi[..., ii + 1, :, :, :, :] = ift2(xp.matmul(p, ft2(xp.matmul(q, phi[..., ii, :, :, :, :] * apodization),
                                                             axes=yx_axes, shift=False)
                                                      ),
                                            axes=yx_axes, shift=False)

    # propagate to imaging plane and apply coherent transfer function
    # the last element of phi is fundamentally different than the others because fb_proj changes the basis
    # so this is (phi_f, phi_b) whereas the others are (phi, dphi / dz)
    phi[..., -1, :, :, 0, :] = ift2(xp.matmul(fb_proj, atf * xp.matmul(p_img,
                                                                       ft2(phi[..., -2, :, :, :, :],
                                                                           axes=yx_axes, shift=False))),
                                    axes=yx_axes, shift=False)[..., 0, :]

    # strip off extra dim
    return phi[..., 0]


def backpropagate_ssnp(efield_end: array,
                       n: array,
                       no: float,
                       drs: Sequence[float, float, float],
                       wavelength: float,
                       dz_final: float = 0.,
                       atf: Optional[array] = None,
                       apodization: Optional[array] = None) -> array:
    """
    This acts with the adjoint operators required when taking the gradient

    :param efield_end:
    :param n:
    :param no:
    :param drs:
    :param wavelength:
    :param dz_final:
    :param atf:
    :param apodization:
    :return efield_back:
    """

    use_gpu = cp and isinstance(efield_end, cp.ndarray)
    if use_gpu:
        xp = cp
    else:
        xp = np

    if atf is None:
        atf = 1.

    if apodization is None:
        apodization = 1.

    n = xp.asarray(n)
    # expand over phi dims + extra dim
    atf = xp.expand_dims(xp.asarray(xp.atleast_2d(atf)), axis=(-1, -2))
    apodization = xp.expand_dims(xp.atleast_2d(xp.asarray(apodization)), axis=(-1, -2))

    ko = 2 * np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    fx = xp.expand_dims(xp.fft.fftfreq(nx, dx), axis=0)
    fy = xp.expand_dims(xp.fft.fftfreq(ny, dy), axis=1)

    # construct field propagators
    p = propagation_kernel(fx, fy, dz, wavelength, no)
    p_img = propagation_kernel(fx, fy, dz_final, wavelength, no)

    p_adj = p.conj().swapaxes(-1, -2)
    p_img_adj = p_img.conj().swapaxes(-1, -2)

    # add extra dimension at the end for broadcasting during matmult
    out_shape = efield_end.shape[:-2] + (nz + 2, ny, nx, 2, 1)
    yx_axes = (-3, -4)
    phi = xp.zeros(out_shape, dtype=complex)
    phi[..., -1, :, :, 0, 0] = xp.asarray(efield_end)

    #
    fb_proj_adj = forward_backward_proj(fx, fy, wavelength, no).conj()[..., slice(0, 1), :].swapaxes(-1, -2)

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
        q_adj[..., 0, 1] = ko**2 * (no**2 - n[ii]**2).conj() * dz
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
