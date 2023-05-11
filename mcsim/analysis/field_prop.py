"""
Numerical beam propagation
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

def get_angular_spectrum_kernel(dz: float,
                                wavelength: float,
                                no: float,
                                shape: tuple[int],
                                drs: tuple[float],
                                use_gpu: bool = False) -> array:
    """
    Get the angular spectrum/plane wave expansion kernel for propagating an electric field by distance dz

    :param dz:
    :param wavelength:
    :param no:
    :param shape:
    :param drs:
    :return:
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

    kernel = xp.zeros((ny, nx), dtype=complex)
    with np.errstate(invalid="ignore"):
        arg = (k * no)**2 - (2*np.pi * fx)**2 - (2*np.pi * fy)**2
        allowed = arg >= 0
        kernel[allowed] = xp.exp(1j * dz * xp.sqrt(arg[allowed]))

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
    :param zs:
    :param no:
    :param drs: (dy, dx)
    :param wavelength:
    :return efield_prop: no x ... x nm x nz x ny x nx array
    """

    use_gpu = isinstance(efield_start, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
    else:
        xp = np

    def ft(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    def ift(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    nz = len(zs)
    ny, nx = efield_start.shape[-2:]


    efield_start_ft = ft(efield_start)

    new_size = efield_start.shape[:-2] + (nz, ny, nx)
    efield_ft_prop = xp.zeros(new_size, dtype=complex)
    for ii in range(len(zs)):
        # construct propagation kernel
        kernel = get_angular_spectrum_kernel(zs[ii], wavelength, no, (ny, nx), drs=drs, use_gpu=use_gpu)
        # propagate field with kernel
        efield_ft_prop[..., ii, :, :] = efield_start_ft * kernel

    efield_prop = ift(efield_ft_prop)

    return efield_prop


def propagate_inhomogeneous(efield_start: array,
                            n: array,
                            no: float,
                            drs: tuple[float],
                            wavelength: float,
                            dz_final: float = 0.,
                            atf: Optional[array] = None,
                            apodization: Optional[array] = None,
                            model: str = "bpm") -> array:
    """
    Propagate electric field through medium with index of refraction n(x, y, z) using the projection approximation,
    which is paraxial. That is, first propagate through the background medium using the angular spectrum method,
    and then include the effect of the inhomogeneous refractive index in the projection approximation

    Array stores electric field on each side of pixels, plus the electric field at the imaging plane. So if
    there are nz pixels, there are nz + 2 planes

    :param efield_start: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
    :param n: nz x ny x nx array
    :param no: background index of refraction
    :param drs: (dz, dy, dx)
    :param wavelength: wavelength in same units as drs
    :param apodization:
    :param model:
    :return efield: n0 x ... x nm x nz x ny x nx electric field
    """

    use_gpu = isinstance(efield_start, cp.ndarray) and _gpu_available

    if use_gpu:
        xp = cp
    else:
        xp = np

    if atf is None:
        atf = 1.

    if apodization is None:
        apodization = 1.

    if model != "bpm":
        raise NotImplementedError(f"only model={model:s} is implemented")

    efield_start = xp.asarray(efield_start)
    n = xp.asarray(n)
    atf = xp.asarray(atf)
    apodization = xp.asarray(apodization)

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu))

    def ft(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    def ift(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    # propagate
    out_shape = efield_start.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[..., 0, :, :] = efield_start
    for ii in range(nz):
        efield[..., ii + 1, :, :] = ift(ft(efield[..., ii, :, :]) * prop_kernel * apodization) * xp.exp(1j * k * dz * (n[ii] - no))

    # propagate to imaging plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu))
    efield[..., -1, :, :] = ift(ft(efield[..., -2, :, :]) * kernel_img * atf)

    return efield


def backpropagate_inhomogeneous(efield_end: array,
                                n: array,
                                no: float,
                                drs: tuple[float],
                                wavelength: float,
                                dz_final: float = 0.,
                                atf: Optional[array] = None,
                                apodization: Optional[array] = None,
                                model: str = "bpm") -> array:
    """
    Apply the adjoint operation to propagate_inhomogeneous(). This is adjoint in the sense that for any pair of fields
    a, b we have
    np.sum(a.conj() * propagate_inhomogeneous(b)) = np.sum(backpropagate_inhomogeneous(a).conj() * b)

    :param efield_end: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
    :param n: nz x ny x nx array
    :param no: background index of refraction
    :param drs: (dz, dy, dx)
    :param wavelength: wavelength in same units as drs
    :param apodization:
    :param model:
    :return efield: n0 x ... x nm x nz x ny x nx electric field
    """

    if atf is None:
        atf = 1.

    if apodization is None:
        apodization = 1.

    if model != "bpm":
        raise NotImplementedError(f"only model={model:s} is implemented")

    use_gpu = isinstance(efield_end, cp.ndarray) and _gpu_available
    if use_gpu:
        xp = cp
    else:
        xp = np

    efield_end = xp.asarray(efield_end)
    n = xp.asarray(n)
    atf = xp.asarray(atf)
    apodization = xp.asarray(apodization)

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    # adjoint operations are FT or IFT with exponential conjugated, so would be swapping FT and IFT except for normalization
    # changing normalization to "forward" instead of the default "backwards" is all else we need
    def ft_adj(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), norm="forward", axes=(-1, -2)), axes=(-1, -2))
    def ift_adj(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), norm="forward", axes=(-1, -2)), axes=(-1, -2))

    # propagate
    out_shape = efield_end.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[..., -1, :, :] = efield_end

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu))

    # propagate from imaging plane back to last plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu))
    # efield[-1] = ift(ft(efield[-2]) * kernel_img * atf)
    efield[..., -2, :, :] = ft_adj(ift_adj(efield[..., -1, :, :]) * kernel_img.conj() * xp.conj(atf))

    for ii in range(nz - 1, -1, -1):
        efield[..., ii, :, :] = ft_adj(ift_adj(efield[..., ii + 1, :, :] * xp.exp(1j * k * dz * (n[ii] - no)).conj()) * prop_kernel.conj() * xp.conj(apodization))

    return efield
