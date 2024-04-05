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

try:
    import cupy as cp
    import cupyx.scipy.fft as fft_gpu
except ImportError:
    cp = None
    fft_gpu = None
    _gpu_available = False

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


# ######################
# 2D Fourier transform recipes
# ######################
def ft2(m: array,
        axes: tuple[int, int] = (-1, -2),
        plan=None,
        no_cache: bool = False) -> array:
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
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.fft2(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.fft2(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


def ift2(m: array,
         axes: tuple[int, int] = (-1, -2),
         plan=None,
         no_cache: bool = False) -> array:
    """
    Inverse function for ft2()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return:
    """
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.ifft2(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.ifft2(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


def ft2_adj(m: array,
            axes: tuple[int, int] = (-1, -2),
            plan=None,
            no_cache: bool = False) -> array:
    """
    Adjoint to 2D Fourier transform. This operation is the adjoint in the sense that for any two
    images w and v the following inner products are equal:
    <w, ft(v)> = <ft_adj(w), v>

    adjoint operations are FT or IFT with exponential conjugated, so would be swapping FT and IFT except for
    normalization changing normalization to "forward" instead of the default "backwards" is all else we need

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return ft2_adj:
    """
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.ifft2(fft_gpu.ifftshift(m, axes=axes), norm="forward", axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.ifft2(fft_cpu.ifftshift(m, axes=axes), norm="forward", axes=axes), axes=axes)


def ift2_adj(m: array,
             axes: tuple[int, int] = (-1, -2),
             plan=None,
             no_cache: bool = False) -> array:
    """
    Adjoint to ift2()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return ift2_adj:
    """
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        return fft_gpu.fftshift(fft_gpu.fft2(fft_gpu.ifftshift(m, axes=axes), norm="forward", axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.fft2(fft_cpu.ifftshift(m, axes=axes), norm="forward", axes=axes), axes=axes)


def irft2(m: array,
          axes: tuple[int, int] = (-1, -2)) -> array:
    """
    2D inverse real Fourier transform which can use either CPU or GPU

    :param m:
    :param axes:
    :return ft:
    """
    # note: at first had self.use_gpu here, but then next map_blocks fn took ~1 minute to run!
    # so think I should avoid calls to self
    # todo: should adhere more closely to ft2
    if cp and isinstance(m, cp.ndarray):
        xp = cp
    else:
        xp = np

    # irfft2 ~2X faster than ifft2
    # # Have to keep full -2 axis because need two full quadrants for complete fft info
    one_sided = xp.fft.ifftshift(m, axes=axes)[..., :m.shape[-1] // 2 + 1]

    # note: for irfft2 must match shape and axes, so order important
    result = xp.fft.fftshift(xp.fft.irfft2(one_sided, s=m.shape[-2:], axes=(-2, -1)), axes=(-1, -2))

    return result


# ######################
# 3D Fourier transform recipes
# ######################
def ft3(m: array,
        axes: tuple[int, int, int] = (-1, -2, -3),
        plan=None,
        no_cache: bool = False) -> array:
    """
    3D FFT idiom assuming the center of our coordinate system is near the center of our array. Specifically,
    the spatial coordinates are range(n) - (n // 2) along a dimension of size n

    :param m: array to perform Fourier transform on
    :param axes: axes to perform Fourier transform on
    :param plan: CuPy FFT plan. This has no effect if running on the CPU. If a plan is passed through, then
      no plan will be cached
    :param no_cache:
    :return ft3:
    """
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)

        return fft_gpu.fftshift(fft_gpu.fftn(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.fftn(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


def ift3(m: array,
         axes: tuple[int, int, int] = (-1, -2, -3),
         plan=None,
         no_cache: bool = False) -> array:
    """
    Inverse to ft3()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return ift3:
    """
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)

        return fft_gpu.fftshift(fft_gpu.ifftn(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.ifftn(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


# ######################
# ND Fourier transform recipes
# ######################
def conj_transpose_fft(img_ft: array,
                       axes: tuple[int, int] = (-1, -2)) -> array:
    """
    Given img_ft(f), return a new array
    img_new_ft(f) := conj(img_ft(-f))

    :param img_ft:
    :param axes: axes on which to perform the transformation
    """

    if cp and isinstance(img_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    # convert axes to positive number
    axes = np.mod(np.array(axes), img_ft.ndim)

    # flip and conjugate
    img_ft_ct = xp.flip(xp.conj(img_ft), axis=tuple(axes))

    # for odd FFT size, can simply flip the array to take f -> -f
    # for even FFT size, have on more negative frequency than positive frequency component.
    # by flipping array, have put the negative frequency components on the wrong side of the array
    # (i.e. where the positive frequency components are)
    # so must roll array to put them back on the right side
    to_roll = [a for a in axes if np.mod(img_ft.shape[a], 2) == 0]
    img_ft_ct = xp.roll(img_ft_ct,
                        shift=[1] * len(to_roll),
                        axis=tuple(to_roll))

    return img_ft_ct
