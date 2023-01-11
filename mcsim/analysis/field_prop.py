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
    @param dz:
    @param wavelength:
    @param no:
    @param shape:
    @param drs:
    @return:
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


def propagate_homogeneous(efield: array,
                          zs: array,
                          no: float,
                          drs: list[float],
                          wavelength: float,
                          dz_final: float = 0.,
                          atf: Optional[array] = None,
                          apodization: Union[array, float] = 1.,
                          axis=(-2, -1)) -> array:
    """
    Propagate the Fourier transform of an optical field a distance z through a medium with homogeneous index
    of refraction n using the angular spectrum method
    @param efield:
    @param zs:
    @param no:
    @param drs: (dy, dx)
    @param wavelength:
    @param dz_final:
    @param atf:
    @param apodization:
    @param axis: y and x dimension respectively
    @return efield_ft_prop: coords = (fy, fx)
    """
    use_gpu = isinstance(efield, cp.ndarray) and _gpu_available

    if use_gpu:
        xp = cp
    else:
        xp = np

    def ft(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=axis), axes=axis), axes=axis)
    def ift(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=axis), axes=axis), axes=axis)

    nz = len(zs)
    ny = efield.shape[axis[0]]
    nx = efield.shape[axis[1]]

    # find dimensions to expand kernel along
    axs = [a % efield.ndim for a in axis]
    dims_to_expand = tuple([d for d in range(efield.ndim) if d not in axs])

    efield_ft = ft(efield * apodization)
    efield_ft_prop = xp.zeros((nz + 1, ny, nx), dtype=complex)
    for ii in range(len(zs)):
        # construct propagation kernel
        kernel = get_angular_spectrum_kernel(zs[ii], wavelength, no, (ny, nx), drs=drs, use_gpu=use_gpu)
        # propagate field with kernel
        efield_ft_prop[ii] = efield_ft * xp.expand_dims(kernel, dims_to_expand)

    # propagate to imaging plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(dz_final, wavelength, no, (ny, nx), drs=drs, use_gpu=use_gpu))
    efield_ft_prop[-1] = efield_ft_prop[-2] * kernel_img * atf

    efield_prop = ift(efield_ft_prop)

    return efield_prop


def propagate_inhomogeneous(efield_start: array,
                            n: array,
                            no: float,
                            drs: tuple[float],
                            wavelength: float,
                            dz_final: float = 0.,
                            atf: Optional[array] = None,
                            apodization: Union[array, float] = 1.,
                            model:str = "bpm") -> array:
    """
    Propagate electric field through medium with index of refraction n(x, y, z) using the projection approximation,
    which is paraxial. That is, first propagate through the background medium using the angular spectrum method,
    and then include the effect of the inhomogeneous refractive index in the projection approximation

    Array stores electric field on each side of pixels, plus the electric field at the imaging plane. So if
    there are nz pixels, there are nz + 2 planes

    @param efield_start: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
    @param n: nz x ny x nx array
    @param no: background index of refraction
    @param drs: (dz, dy, dx)
    @param wavelength: wavelength in same units as drs
    @param apodization:
    @param model:
    @return efield: nz x ny x nx electric field
    """

    if atf is None:
        atf = 1.

    use_gpu = isinstance(efield_start, cp.ndarray) and _gpu_available

    if use_gpu:
        xp = cp
    else:
        xp = np

    efield_start = xp.asarray(efield_start)
    n = xp.asarray(n)
    atf = xp.asarray(atf)

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu))

    def ft(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    def ift(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    # propagate
    out_shape = efield_start.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[0] = efield_start
    for ii in range(nz):
        efield[ii + 1] = ift(ft(efield[ii]) * prop_kernel * apodization) * xp.exp(1j * k * dz * (n[ii] - no))

    # propagate to imaging plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu))
    efield[-1] = ift(ft(efield[-2]) * kernel_img * atf)

    return efield

def backpropagate_inhomogeneous(efield_imaged: array,
                                n: array,
                                no: float,
                                drs: tuple[float],
                                wavelength: float,
                                dz_final: float = 0.,
                                atf: Optional[array] = None,
                                apodization: Union[array, float] = 1.,
                                model: str = "bpm") -> array:
    """
    Apply the adjoint operation to propagate_inhomogeneous(). This is adjoint in the sense that for any pair of fields
    a, b we have
    np.sum(a.conj() * propagate_inhomogeneous(b)) = np.sum(backpropagate_inhomogeneous(a).conj() * b)

    @param efield_imaged: n0 x ... x nm x ny x nx NumPy or CuPy array. If CuPy array, run computation on GPU
    @param n: nz x ny x nx array
    @param no: background index of refraction
    @param drs: (dz, dy, dx)
    @param wavelength: wavelength in same units as drs
    @param apodization:
    @param model:
    @return efield: nz x ny x nx electric field
    """

    if atf is None:
        atf = 1.

    use_gpu = isinstance(efield_imaged, cp.ndarray) and _gpu_available

    if use_gpu:
        xp = cp
    else:
        xp = np

    efield_imaged = xp.asarray(efield_imaged)
    n = xp.asarray(n)
    atf = xp.asarray(atf)

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n.shape

    # adjoint operations are FT or IFT with exponential conjugated, so would be swapping FT and IFT except for normalization
    # changing normalization to "forward" instead of the default "backwards" is all else we need
    def ft_adj(m): return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(m, axes=(-1, -2)), norm="forward", axes=(-1, -2)), axes=(-1, -2))
    def ift_adj(m): return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(m, axes=(-1, -2)), norm="forward", axes=(-1, -2)), axes=(-1, -2))

    # propagate
    out_shape = efield_imaged.shape[:-2] + (nz + 2, ny, nx)

    efield = xp.zeros(out_shape, dtype=complex)
    efield[-1] = efield_imaged

    prop_kernel = xp.asarray(get_angular_spectrum_kernel(dz, wavelength, no, n.shape[1:], drs[1:], use_gpu))

    # propagate from imaging plane back to last plane
    kernel_img = xp.asarray(get_angular_spectrum_kernel(dz_final, wavelength, no, n.shape[1:], drs[1:], use_gpu))
    # efield[-1] = ift(ft(efield[-2]) * kernel_img * atf)
    efield[-2] = ft_adj(ift_adj(efield[-1]) * kernel_img.conj() * xp.conj(atf))

    for ii in range(nz - 1, -1, -1):
        efield[ii] = ft_adj(ift_adj(efield[ii + 1] * xp.exp(1j * k * dz * (n[ii] - no)).conj()) * prop_kernel.conj() * xp.conj(apodization))

    return efield
