"""
CPU/GPU agnostic FFT functions using our preferred idioms

Specifically, the ifftshift/fftshift patterns are chosen such that the natural spatial coordinates are
(range(n) - (n // 2)) * dr along a dimension of size n
and the frequency coordinates are found from
fftshift(fftfreq(n, dr))
"""

from typing import Union
import numpy as np
import numpy.fft as fft_cpu

_gpu_available = True
try:
    import cupy as cp
    import cupyx.scipy.fft as fft_gpu
except ImportError:
    cp = np
    fft_gpu = None
    _gpu_available = False

array = Union[np.ndarray, cp.ndarray]

# ######################
# 2D Fourier transform recipes
# ######################
def ft2(m: array, axes: tuple[int] = (-1, -2), plan=None, no_cache: bool = False) -> array:
    """
    2D FFT idiom assuming the center of our coordinate system is near the center of our array. Specifically,
    the spatial coordinates are range(n) - (n // 2) along a dimension of size n

    :param m: array to perform Fourier transform on
    :param axes: axes to perform Fourier transform on
    :param plan: CuPy FFT plan. This has no effect if running on the CPU. If a plan is passed through, then
      no plan will be cached
    :param no_cache:
    :return:
    """
    if isinstance(m, cp.ndarray) and _gpu_available:
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.fft2(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.fft2(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


def ift2(m: array, axes: tuple[int] = (-1, -2), plan=None, no_cache: bool = False) -> array:
    """
    Inverse function for ft2()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return:
    """
    if isinstance(m, cp.ndarray) and _gpu_available:
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.ifft2(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.ifft2(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


def ft2_adj(m: array, axes: tuple[int] = (-1, -2), plan=None, no_cache: bool = False) -> array:
    """
    Adjoint to 2D Fourier transform. This operation is the adjoint in the sense that for any two
    images w and v the following inner products are equal:
    <w, ft(v)> = <ft_adj(w), v>

    adjoint operations are FT or IFT with exponential conjugated, so would be swapping FT and IFT except for normalization
    changing normalization to "forward" instead of the default "backwards" is all else we need

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return:
    """
    if isinstance(m, cp.ndarray) and _gpu_available:
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.ifft2(fft_gpu.ifftshift(m, axes=axes), norm="forward", axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.ifft2(fft_cpu.ifftshift(m, axes=axes), norm="forward", axes=axes), axes=axes)


def ift2_adj(m: array, axes: tuple[int] = (-1, -2), plan=None, no_cache: bool = False) -> array:
    """
    Adjoint to ift2()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return:
    """
    if isinstance(m, cp.ndarray) and _gpu_available:
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.fft2(fft_gpu.ifftshift(m, axes=axes), norm="forward", axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.fft2(fft_cpu.ifftshift(m, axes=axes), norm="forward", axes=axes), axes=axes)


# ######################
# 3D Fourier transform recipes
# ######################
def ft3(m: array, axes: tuple[int] = (-1, -2, -3), plan=None, no_cache: bool = False) -> array:
    """
    3D FFT idiom assuming the center of our coordinate system is near the center of our array. Specifically,
    the spatial coordinates are range(n) - (n // 2) along a dimension of size n

    :param m: array to perform Fourier transform on
    :param axes: axes to perform Fourier transform on
    :param plan: CuPy FFT plan. This has no effect if running on the CPU. If a plan is passed through, then
      no plan will be cached
    :param no_cache:
    :return:
    """
    if isinstance(m, cp.ndarray) and _gpu_available:
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)

        return fft_gpu.fftshift(fft_gpu.fftn(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.fftn(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


def ift3(m: array, axes: tuple[int] = (-1, -2, -3), plan=None, no_cache: bool = False) -> array:
    """
    Inverse to ft3()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return:
    """
    if isinstance(m, cp.ndarray) and _gpu_available:
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)

        return fft_gpu.fftshift(fft_gpu.ifftn(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.ifftn(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)
