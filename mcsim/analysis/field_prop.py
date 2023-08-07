"""
Numerical beam propagation through homogeneous and inhomogeneous media
"""

from typing import Union, Optional
import numpy as np

_gpu_available = True
try:
    import cupy as cp
except ImportError:
    cp = np
    _gpu_available = False

array = Union[np.ndarray, cp.ndarray]


# Fourier transform recipes
def _ft2(m, axes=(-1, -2)):
    if isinstance(m, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=axes), axes=axes), axes=axes)


def _ift2(m, axes=(-1, -2)):
    if isinstance(m, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=axes), axes=axes), axes=axes)


# adjoint operations are FT or IFT with exponential conjugated, so would be swapping FT and IFT except for normalization
# changing normalization to "forward" instead of the default "backwards" is all else we need
def _ft2_adj(m, axes=(-1, -2)):
    if isinstance(m, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=axes), norm="forward", axes=axes), axes=axes)


def _ift2_adj(m, axes=(-1, -2)):
    if isinstance(m, cp.ndarray) and _gpu_available:
        xp = cp
    else:
        xp = np

    return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=axes), norm="forward", axes=axes), axes=axes)


def get_fzs(fx: array,
            fy: array,
            no: float,
            wavelength: float) -> array:
    """
    Get z-component of frequency given fx, fy

    :param fx: nfrqs
    :param fy:
    :param no: index of refraction
    :param wavelength: wavelength
    :return fzs:
    """

    # todo: should support gpu

    with np.errstate(invalid="ignore"):
        fzs = np.sqrt(no**2 / wavelength ** 2 - fx**2 - fy**2)

    return fzs


def get_angular_spectrum_kernel(dz: float,
                                wavelength: float,
                                no: float,
                                shape: tuple[int],
                                drs: tuple[float],
                                use_gpu: bool = False) -> array:
    """
    Get the angular spectrum/plane wave expansion kernel for propagating an electric field by distance dz

    Here we assume a propagating plane wave has the form Re{exp[i(kz - \omega t)]}. That is, phasors carry the implicit
    time dependence exp[-i \omega t].

    :param dz:
    :param wavelength:
    :param no:
    :param shape:
    :param drs:
    :return kernel:
    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    k = 2*np.pi / wavelength
    ny, nx = shape
    dy, dx = drs

    fx = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(nx, dx)), axis=0)
    fy = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(ny, dy)), axis=1)

    # todo: use get_fzs()
    kernel = xp.zeros((ny, nx), dtype=complex)
    with np.errstate(invalid="ignore"):
        arg = (k * no)**2 - (2*np.pi * fx)**2 - (2*np.pi * fy)**2
        allowed = arg >= 0
        kernel[allowed] = xp.exp(1j * dz * xp.sqrt(arg[allowed]))

    return kernel


def propagation_kernel(dz: float,
                       wavelength: float,
                       no: float,
                       shape: tuple[int],
                       drs: tuple[float],
                       use_gpu: bool = False) -> array:
    """
    Propagation kernel for field represented by field value and derivative. Note that this can alternatively be
    understood as a field which contains both forward and backward propagating components

    :param dz:
    :param wavelength:
    :param no:
    :param shape:
    :param drs:
    :param use_gpu:
    :return kernel:
    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    k = 2*np.pi / wavelength
    ny, nx = shape
    dy, dx = drs

    fx = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(nx, dx)), axis=0)
    fy = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(ny, dy)), axis=1)

    # todo: use get fzs
    kernel = xp.zeros((ny, nx, 2, 2), dtype=float)
    with np.errstate(invalid="ignore"):
        arg = (k * no) ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2
        allowed = arg >= 0
        kz_allowed = xp.sqrt(arg[allowed])

        kernel[allowed, 0, 0] = xp.cos(kz_allowed * dz)
        kernel[allowed, 0, 1] = xp.sin(kz_allowed * dz) / kz_allowed
        kernel[allowed, 1, 0] = -kz_allowed * xp.sin(kz_allowed * dz)
        kernel[allowed, 1, 1] = xp.cos(kz_allowed * dz)

    return kernel


def forward_backward_proj(wavelength: float,
                          no: float,
                          shape: tuple[int],
                          drs: tuple[float],
                          use_gpu: bool = False) -> array:
    """
    matrix converting from (phi, dphi/dz) -> (phi_f, phi_b) representation

    :param wavelength:
    :param no:
    :param shape:
    :param drs:
    :param use_gpu:
    :return:
    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    k = 2 * np.pi / wavelength
    ny, nx = shape
    dy, dx = drs

    fx = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(nx, dx)), axis=0)
    fy = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(ny, dy)), axis=1)

    # todo: use get_fzs
    kernel = xp.zeros((ny, nx, 2, 2), dtype=complex)
    with np.errstate(invalid="ignore"):
        arg = (k * no) ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2
        allowed = arg >= 0
        kz_allowed = xp.sqrt(arg[allowed])

        kernel[..., 0, 0] = 0.5
        kernel[allowed, 0, 1] = -0.5 * 1j / kz_allowed
        kernel[..., 1, 0] = 0.5
        kernel[allowed, 1, 1] = 0.5 * 1j / kz_allowed

    return kernel


def field_deriv_proj(wavelength: float,
                          no: float,
                          shape: tuple[int],
                          drs: tuple[float],
                          use_gpu: bool = False) -> array:
    """
    matrix converting from (phi, dphi/dz) -> (phi_f, phi_b) representation

    :param wavelength:
    :param no:
    :param shape:
    :param drs:
    :param use_gpu:
    :return:
    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    k = 2 * np.pi / wavelength
    ny, nx = shape
    dy, dx = drs

    fx = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(nx, dx)), axis=0)
    fy = xp.expand_dims(xp.fft.fftshift(xp.fft.fftfreq(ny, dy)), axis=1)

    # todo: use get_fzs
    kernel = xp.zeros((ny, nx, 2, 2), dtype=complex)
    with np.errstate(invalid="ignore"):
        arg = (k * no) ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2
        allowed = arg >= 0
        kz_allowed = xp.sqrt(arg[allowed])

        kernel[..., 0, 0] = 1
        kernel[..., 0, 1] = 1
        kernel[allowed, 1, 0] = 1j * kz_allowed
        kernel[allowed, 1, 1] = -1j * kz_allowed

    return kernel


def propagate_homogeneous(efield_start: array,
                          zs: array,
                          no: float,
                          drs: list[float],
                          wavelength: float) -> array:
    """
    Propagate the Fourier transform of an optical field a distance z through a medium with homogeneous index
    of refraction n using the angular spectrum method

    :param efield_start: n0 x ... x nm x ny x nx array
    :param zs: array of z-positions to compute
    :param no: homogeneous refractive index
    :param drs: (dy, dx)
    :param wavelength: wavelength in the same units as drs and zs
    :return efield_prop: propagated electric field of shape no x ... x nm x nz x ny x nx
    """

    use_gpu = isinstance(efield_start, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
    else:
        xp = np

    nz = len(zs)
    ny, nx = efield_start.shape[-2:]

    efield_start_ft = _ft2(efield_start)

    new_size = efield_start.shape[:-2] + (nz, ny, nx)
    efield_ft_prop = xp.zeros(new_size, dtype=complex)
    for ii in range(len(zs)):
        # construct propagation kernel
        kernel = get_angular_spectrum_kernel(zs[ii], wavelength, no, (ny, nx), drs=drs, use_gpu=use_gpu)
        # propagate field with kernel
        efield_ft_prop[..., ii, :, :] = efield_start_ft * kernel

    efield_prop = _ift2(efield_ft_prop)

    return efield_prop


def propagate_bpm(efield_start: array,
                  n: array,
                  no: float,
                  drs: tuple[float],
                  wavelength: float,
                  dz_final: float = 0.,
                  atf: Optional[array] = None,
                  apodization: Optional[array] = None,
                  thetas: Optional[array] = None) -> array:
    """
    Propagate electric field through medium with index of refraction n(x, y, z) using the projection approximation,
    which is paraxial. That is, first propagate through the background medium using the angular spectrum method,
    and then include the effect of the inhomogeneous refractive index in the projection approximation

    Array stores electric field on each side of pixels, plus the electric field at the imaging plane. So if
    there are nz pixels, there are nz + 2 planes

    :param efield_start: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
    :param n: nz x ny x nx array
    :param no: background index of refraction
    :param drs: (dz, dy, dx) voxel size in same units as wavelength. efield is assumed to have the same dy, dx
    :param wavelength: wavelength in same units as drs
    :param apodization:
    :param thetas: assuming a plane wave input, this is the angle between the plane wave propagation direction
      and the optical axis. This provides a better approximation for the phase shift of the beam through a
      refractive index layer
    :return efield: n0 x ... x nm x nz x ny x nx electric field
    """

    use_gpu = isinstance(efield_start, cp.ndarray) and _gpu_available

    if use_gpu:
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

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu))

    # propagate
    out_shape = efield_start.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[..., 0, :, :] = efield_start
    for ii in range(nz):
        efield[..., ii + 1, :, :] = _ift2(_ft2(efield[..., ii, :, :]) * prop_kernel * apodization) * xp.exp(1j * k * dz * (n[ii] - no) / xp.cos(thetas))

    # propagate to imaging plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu))
    efield[..., -1, :, :] = _ift2(_ft2(efield[..., -2, :, :]) * kernel_img * atf)

    return efield


def backpropagate_bpm(efield_end: array,
                      n: array,
                      no: float,
                      drs: tuple[float],
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
    :param apodization:
    :param thetas: assuming a plane wave input, this is the angle between the plane wave propagation direction
      and the optical axis. This provides a better approximation for the phase shift of the beam through a
      refractive index layer
    :return efield: n0 x ... x nm x nz x ny x nx electric field
    """

    use_gpu = isinstance(efield_end, cp.ndarray) and _gpu_available
    if use_gpu:
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
        thetas = xp.zeros(efield_start.shape[:-2] + (1, 1))
    thetas = xp.asarray(thetas)

    if thetas.ndim == 1:
        thetas = xp.expand_dims(thetas, axis=(-1, -2))

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    # propagate
    out_shape = efield_end.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[..., -1, :, :] = efield_end

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu))

    # propagate from imaging plane back to last plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu))
    efield[..., -2, :, :] = _ft2_adj(_ift2_adj(efield[..., -1, :, :]) * kernel_img.conj() * xp.conj(atf))

    for ii in range(nz - 1, -1, -1):
        efield[..., ii, :, :] = _ft2_adj(_ift2_adj(efield[..., ii + 1, :, :] * xp.exp(1j * k * dz * (n[ii] - no) / xp.cos(thetas)).conj()) * prop_kernel.conj() * xp.conj(apodization))

    return efield


def propagate_ssnp(efield_start: array,
                   de_dz_start: array,
                   n: array,
                   no: float,
                   drs: tuple[float],
                   wavelength: float,
                   dz_final: float = 0.,
                   atf: Optional[array] = None,
                   apodization: Optional[array] = None):

    use_gpu = isinstance(efield_start, cp.ndarray) and _gpu_available
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

    ko = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    # construct field propagators
    p = propagation_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu=use_gpu)
    p_img = propagation_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu)

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

        phi[..., ii + 1, :, :, :, :] = _ift2(xp.matmul(p, (_ft2(xp.matmul(q, phi[..., ii, :, :, :, :]), axes=yx_axes)) * apodization), axes=yx_axes)

    # propagate to imaging plane and aply coherent transfer function
    fb_proj = forward_backward_proj(wavelength, no, n.shape[1:], drs[1:], use_gpu=use_gpu)[..., slice(0, 1), :]

    # note that the last element of phi is fundamentally different than the other sbecause fb_proj changes the basis
    # so this is (phi_f, phi_b) wherease the others are (phi, dphi / dz)
    phi[..., -1, :, :, 0, :] = _ift2(xp.matmul(fb_proj, atf * xp.matmul(p_img, _ft2(phi[..., -2, :, :, :, :], axes=yx_axes))), axes=yx_axes)[..., 0, :]

    # strip off extra dim
    return phi[..., 0]


def backpropagate_ssnp(efield_end: array,
                       n: array,
                       no: float,
                       drs: tuple[float],
                       wavelength: float,
                       dz_final: float = 0.,
                       atf: Optional[array] = None,
                       apodization: Optional[array] = None):
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
    :return:
    """

    use_gpu = isinstance(efield_end, cp.ndarray) and _gpu_available
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

    # construct field propagators
    p = propagation_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu=use_gpu)
    p_img = propagation_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu)

    p_adj = p.conj().swapaxes(-1, -2)
    p_img_adj = p_img.conj().swapaxes(-1, -2)

    # add extra dimension at the end for broadcasting during matmult
    out_shape = efield_end.shape[:-2] + (nz + 2, ny, nx, 2, 1)
    yx_axes = (-3, -4)
    phi = xp.zeros(out_shape, dtype=complex)
    phi[..., -1, :, :, 0, 0] = xp.asarray(efield_end)

    #
    fb_proj_adj = forward_backward_proj(wavelength, no, n.shape[1:], drs[1:], use_gpu=use_gpu).conj()[..., slice(0, 1), :].swapaxes(-1, -2)

    # adjoint of imaging/final prop operation
    fadj = _ft2_adj(xp.matmul(p_img_adj, xp.conj(atf) * xp.matmul(fb_proj_adj, _ift2_adj(phi[..., -1, :, :, slice(0, 1), :], axes=yx_axes))), axes=yx_axes)

    # last propagation also
    phi[..., -2, :, :, :, :] = _ft2_adj(xp.matmul(p_adj, _ift2_adj(fadj, axes=yx_axes)), axes=yx_axes)
    # loop backwards through z-stack
    for ii in range(nz - 1, -1, -1):
        q_adj = xp.ones((n.shape[1:]) + (2, 2), dtype=complex)
        q_adj[..., 0, 1] = ko**2 * (no**2 - n[ii]**2).conj() * dz
        q_adj[..., 1, 0] = 0

        phi[..., ii, :, :, :, :] = _ft2_adj(xp.matmul(p_adj, _ift2_adj(xp.matmul(q_adj, phi[..., ii + 1, :, :, :, :]), axes=yx_axes)), axes=yx_axes)

    # strip off extra dim
    return phi[..., 0]
