"""
CPU/GPU agnostic FFT functions using our preferred idioms
"""

from typing import Union, Optional
import numpy as np
import numpy.fft as fft_cpu

try:
    import cupy as cp
    import cupyx.scipy.fft as fft_gpu
except ImportError:
    cp = None
    fft_gpu = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


def _noshift(arr: array, **kwargs) -> array:
    return arr


def ft2(m: array,
        axes: tuple[int, int] = (-1, -2),
        plan: Optional = None,
        no_cache: bool = False,
        shift: bool = True,
        adjoint: bool = False) -> array:
    """
    2D FFT which can run on the CPU/GPU and handle fftshifting appropriately

    :param m: array to perform Fourier transform on
    :param axes: axes to perform Fourier transform on
    :param plan: CuPy FFT plan. This has no effect if running on the CPU. If a plan is passed through, then
      no plan will be cached
    :param no_cache: Avoid caching FFT plans by generating a plan based on the input array
    :param shift: whether to shift the arrays to move zero index from/to the center.
      When True, the spatial coordinates are for each pixel are range(n) - (n // 2) along a dimension of size n
      When False, the spatial coordinates are range(n)
    :param adjoint: perform the adjoint operation instead, in the sense that <w, op(v)> = <adj(w), v>.
      Adjoint operations are FT or IFT with exponential conjugated, so would be swapping FT and IFT except for
      normalization changing normalization to "forward" instead of the default "backwards" is all else we need
    :return ft2: Fourier transform. Frequencies can be found with fftshift(fftfreq(n, dr)) when if shift is True,
      or fftfreq(n, dr) otherwise.
    """
    # todo: replace with ftn
    kwargs = {}
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)
        kwargs["plan"] = plan
        fft = fft_gpu
    else:
        fft = fft_cpu

    if adjoint:
        kwargs["norm"] = "forward"
        op = fft.ifft2
    else:
        op = fft.fft2

    if shift:
        fftshift = fft.fftshift
        ifftshift = fft.ifftshift
    else:
        fftshift = _noshift
        ifftshift = _noshift

    return fftshift(op(ifftshift(m, axes=axes), axes=axes, **kwargs), axes=axes)


def ift2(m: array,
         axes: tuple[int, int] = (-1, -2),
         plan: Optional = None,
         no_cache: bool = False,
         shift: bool = True,
         adjoint: bool = False) -> array:
    """
    Inverse function for ft2()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :param shift:
    :param adjoint:
    :return ift2:
    """
    kwargs = {}
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m, axes=axes)
        kwargs["plan"] = plan
        fft = fft_gpu
    else:
        fft = fft_cpu

    if adjoint:
        kwargs["norm"] = "forward"
        op = fft.fft2
    else:
        op = fft.ifft2

    if shift:
        fftshift = fft.fftshift
        ifftshift = fft.ifftshift
    else:
        fftshift = _noshift
        ifftshift = _noshift

    return fftshift(op(ifftshift(m, axes=axes), axes=axes, **kwargs), axes=axes)


def irft2(m: array,
          axes: tuple[int, int] = (-2, -1),
          plan: Optional = None,
          no_cache: bool = False,
          shift: bool = True,
          adjoint: bool = False) -> array:
    """
    2D inverse real Fourier transform which can use either CPU or GPU

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :param shift:
    :param adjoint:
    :return ft:
    """
    kwargs = {}
    if cp and isinstance(m, cp.ndarray):
        fft = fft_gpu
        if no_cache and plan is None:
            raise NotImplementedError("fft plan logic not implemented")
            plan = fft_gpu.get_fft_plan(m)
    else:
        fft = fft_cpu

    if adjoint:
        raise NotImplementedError("adjoint operation not implemented")
    else:
        op = fft.irfft2

    if shift:
        fftshift = fft.fftshift
        ifftshift = fft.ifftshift
    else:
        fftshift = _noshift
        ifftshift = _noshift

    # irfft2 ~2X faster than ifft2
    # todo: slice last axis named in axes
    one_sided = ifftshift(m, axes=axes)[..., :m.shape[-1] // 2 + 1]

    # note: for irfft2 must match shape and axes, so order important
    result = fftshift(op(one_sided, s=m.shape[-2:], axes=axes, **kwargs), axes=axes)

    return result


def ft3(m: array,
        axes: tuple[int, int, int] = (-1, -2, -3),
        plan: Optional = None,
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
    # todo: replace with ftn
    if cp and isinstance(m, cp.ndarray):
        if no_cache and plan is None:
            plan = fft_gpu.get_fft_plan(m)

        return fft_gpu.fftshift(fft_gpu.fftn(fft_gpu.ifftshift(m, axes=axes), axes=axes, plan=plan), axes=axes)
    else:
        return fft_cpu.fftshift(fft_cpu.fftn(fft_cpu.ifftshift(m, axes=axes), axes=axes), axes=axes)


def ift3(m: array,
         axes: tuple[int, int, int] = (-1, -2, -3),
         plan: Optional = None,
         no_cache: bool = False) -> array:
    """
    Inverse to ft3()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :return ift3:
    """
    # todo: replace with iftn
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
    :return img_ft_ct:
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
