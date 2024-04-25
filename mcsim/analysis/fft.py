"""
CPU/GPU agnostic FFT functions using our preferred idioms
"""

from typing import Union, Optional
from collections.abc import Sequence
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
    return ftn(m, axes=axes, plan=plan, no_cache=no_cache, shift=shift, adjoint=adjoint)


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
    return iftn(m, axes=axes, plan=plan, no_cache=no_cache, shift=shift, adjoint=adjoint)


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
        no_cache: bool = False,
        shift: bool = True,
        adjoint: bool = False) -> array:
    """
    3D FFT

    :param m: array to perform Fourier transform on
    :param axes: axes to perform Fourier transform on
    :param plan: CuPy FFT plan. This has no effect if running on the CPU. If a plan is passed through, then
      no plan will be cached
    :param no_cache:
    :param shift:
    :param adjoint:
    :return ft3:
    """
    return ftn(m, axes=axes, plan=plan, no_cache=no_cache, shift=shift, adjoint=adjoint)


def ift3(m: array,
         axes: tuple[int, int, int] = (-1, -2, -3),
         plan: Optional = None,
         no_cache: bool = False,
         shift: bool = True,
         adjoint: bool = False) -> array:
    """
    Inverse to ft3()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :param shift:
    :param adjoint:
    :return ift3:
    """
    return iftn(m, axes=axes, plan=plan, no_cache=no_cache, shift=shift, adjoint=adjoint)


def ftn(m: array,
        axes: tuple[int],
        plan: Optional = None,
        no_cache: bool = False,
        shift: bool = True,
        adjoint: bool = False) -> array:
    """
    nD FFT which can run on the CPU/GPU and handle fftshifting appropriately

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
    :return ft: Fourier transform. Frequencies can be found with fftshift(fftfreq(n, dr)) when if shift is True,
      or fftfreq(n, dr) otherwise.
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
        op = fft.ifftn
    else:
        op = fft.fftn

    if shift:
        fftshift = fft.fftshift
        ifftshift = fft.ifftshift
    else:
        fftshift = _noshift
        ifftshift = _noshift

    return fftshift(op(ifftshift(m, axes=axes), axes=axes, **kwargs), axes=axes)


def iftn(m: array,
         axes: tuple[int],
         plan: Optional = None,
         no_cache: bool = False,
         shift: bool = True,
         adjoint: bool = False) -> array:
    """
    Inverse function for ftn()

    :param m:
    :param axes:
    :param plan:
    :param no_cache:
    :param shift:
    :param adjoint:
    :return ift:
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
        op = fft.fftn
    else:
        op = fft.ifftn

    if shift:
        fftshift = fft.fftshift
        ifftshift = fft.ifftshift
    else:
        fftshift = _noshift
        ifftshift = _noshift

    return fftshift(op(ifftshift(m, axes=axes), axes=axes, **kwargs), axes=axes)


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


def translate_ft(img_ft: array,
                 fx: array,
                 fy: array,
                 drs: Optional[Sequence[float, float]] = None) -> array:
    """
    Given img_ft(f), return the translated function
    img_ft_shifted(f) = img_ft(f + shift_frq)
    using the FFT shift relationship, img_ft(f + shift_frq) = F[ exp(-2*pi*i * shift_frq * r) * img(r) ]
    This is an approximation to the Whittaker-Shannon interpolation formula which can be performed using only FFT's.
    In this sense, it is exact for band-limited functions.
    The total amount of memory used by this function is roughly set the fft routine, and is about 3x the size of
    the input image

    :param img_ft: NumPy or CuPy array representing the fourier transform of an image with
      frequency origin centered as if using fftshift. Shape n1 x n2 x ... x n_{-3} x ny x nx
      Shifting is done along the last two axes. If this is a CuPy array, routine will be run on the GPU
    :param fx: array of x-shift frequencies
      fx and fy should either be broadcastable to the same size as img_ft, or they should be of size
      n_{-m} x ... x n_{-3} where images along dimensions -m, ..., -3 are shifted in parallel
    :param fy: array of y-shift frequencies
    :param drs: (dy, dx) pixel size (sampling rate) of real space image in directions.
    :return: shifted images, same size as img_ft
    """

    if cp and isinstance(img_ft, cp.ndarray):
        xp = cp
    else:
        xp = np

    if img_ft.ndim < 2:
        raise ValueError("img_ft must be at least 2D")

    n_extra_dims = img_ft.ndim - 2
    ny, nx = img_ft.shape[-2:]

    fx = xp.asarray(fx)
    fy = xp.asarray(fy)

    if fx.shape != fy.shape:
        raise ValueError(f"fx and fy must have same shape, but had shapes {fx.shape} and {fy.shape}")

    shapes_broadcastable = True
    try:
        _ = xp.broadcast(fx, img_ft)
    except ValueError:
        shapes_broadcastable = False

    # otherwise make shapes broadcastable if possible
    if not shapes_broadcastable:
        ndim_extra_frq = fx.ndim
        if fx.shape != img_ft.shape[-2 - ndim_extra_frq:-2]:
            raise ValueError(f"fx and shift_frq have incompatible shapes {fx.shape} and {img_ft.shape}")

        axis_expand_frq = tuple(list(range(n_extra_dims - fx.ndim)) + [-2, -1])
        fx = xp.expand_dims(fx, axis=axis_expand_frq)
        fy = xp.expand_dims(fy, axis=axis_expand_frq)

    if xp.all(fx == 0) and xp.all(fy == 0):
        return xp.array(img_ft, copy=True)
    else:
        if drs is None:
            drs = (1, 1)
        dy, dx = drs

        # must use symmetric frequency representation to do shifting correctly AND only works perfectly is size off
        x = xp.expand_dims(xp.fft.fftfreq(nx) * nx * dx,
                           axis=tuple(range(n_extra_dims + 1)))
        # todo: replace with this second option which is more intuitive
        # x = xp.expand_dims(xp.fft.ifftshift(np.arange(nx) - nx // 2) * dx, axis=axis_expand_x)

        y = xp.expand_dims(xp.fft.fftfreq(ny) * ny * dy,
                           axis=tuple(range(n_extra_dims)) + (-1,))
        # y = xp.expand_dims(xp.fft.ifftshift(np.arange(ny) - ny // 2) * dy, axis=axis_expand_y)

        exp_factor = xp.exp(-1j * 2 * np.pi * (fx * x + fy * y))

        # FT shift theorem to approximate the Whittaker-Shannon interpolation formula,
        # 1. shift frequencies in img_ft so zero frequency is in corner using ifftshift
        # 2. inverse ft
        # 3. multiply by exponential factor
        # 4. take fourier transform, then shift frequencies back using fftshift
        img_ft_shifted = xp.fft.fftshift(
                         xp.fft.fft2(xp.asarray(exp_factor) *
                         xp.fft.ifft2(xp.fft.ifftshift(img_ft, axes=(-1, -2)),
                                      axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        return img_ft_shifted


def translate_im(img: array,
                 xshift: np.ndarray,
                 yshift: np.ndarray,
                 drs: Sequence[float] = (1, 1)) -> array:
    """
    Translate img(y,x) to img(y+yo, x+xo) using FFT. This approach is exact for band-limited functions.
    e.g. suppose the pixel spacing dx = 0.05 um, and we want to shift the image by 0.0366 um,
    then dx = 0.05 and shift = [0, 0.0366]

    :param img: NumPy or CuPy array, size ny x nx. If CuPy array will run on GPU
    :param xshift: in same units as drs
    :param yshift:
    :param drs: (dy, dx) pixel size of image along y- and x-directions
    :return img_shifted:
    """

    ny, nx = img.shape[-2:]
    dy, dx = drs

    xshift_mod = xshift / dx / nx
    yshift_mod = yshift / dy / ny

    return translate_ft(img, xshift_mod, yshift_mod, drs=(1., 1.))
