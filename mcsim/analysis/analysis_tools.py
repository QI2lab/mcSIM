"""
Miscellaneous helper functions. Many are written to run on either the CPU or the GPU depending on if the arrays
passed to them are NumPy or CuPy arrays
"""

import numpy as np
from typing import Union, Optional
import localize_psf.rois as rois

_cupy_available = True
try:
    import cupy as cp
except ImportError:
    cp = np
    _cupy_available = False

array = Union[np.ndarray, cp.ndarray]


# geometry tools
def get_peak_value(img: array,
                   x: array,
                   y: array,
                   peak_coord: np.ndarray,
                   peak_pixel_size: int = 1) -> array:
    """
    Estimate value for a peak that is not precisely aligned to the pixel grid by performing a weighted average
    over neighboring pixels, based on how much these overlap with a rectangular area surrounding the peak.
    The size of this rectangular area is set by peak_pixel_size, given in integer multiples of a pixel.

    :param img: array of size n0 x n1 ... x ny x nx. This function operates on the last two dimensions of the array
    :param x: 1D array representing x-coordinates of images.
    :param y: 1D array representing y-coordinates of image
    :param peak_coord: peak coordinates [px, py]
    :param peak_pixel_size: number of pixels (along each direction) to sum to get peak value
    :return: estimated value of the peak
    """

    if isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    px, py = peak_coord

    # frequency coordinates
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)

    # find closest pixel
    ix = int(np.argmin(np.abs(px - x)))
    iy = int(np.argmin(np.abs(py - y)))

    # get ROI around pixel for weighted averaging
    roi = rois.get_centered_rois([iy, ix], [3 * peak_pixel_size, 3 * peak_pixel_size])[0]
    img_roi = rois.cut_roi(roi, img)[0]

    xx_roi = xp.expand_dims(rois.cut_roi(roi, xx)[0], axis=tuple(range(img_roi.ndim - 2)))
    yy_roi = xp.expand_dims(rois.cut_roi(roi, yy)[0], axis=tuple(range(img_roi.ndim - 2)))

    weights = pixel_overlap(xp.array([[py, px]]),
                            xp.stack((yy_roi.ravel(), xx_roi.ravel()), axis=1),
                            [peak_pixel_size * dy, peak_pixel_size * dx],
                            [dy, dx]) / (dx * dy)

    _, weights = xp.broadcast_arrays(img_roi, weights.reshape(xx_roi.shape))

    peak_value = xp.average(img_roi, weights=weights, axis=(-1, -2))

    return peak_value


def pixel_overlap(centers1: array,
                  centers2: array,
                  lens1: list[float],
                  lens2: Optional[list[float]] = None) -> array:
    """
    Calculate overlap of two nd-square pixels. The pixels go from coordinates
    centers[ii] - 0.5 * lens[ii] to centers[ii] + 0.5 * lens[ii].

    :param centers1: Array of size ncenters x ndims. coordinates define centers of first pixel along each dimension.
    :param centers2: Broadcastable to same size as centers1
    :param lens1: list of pixel 1 sizes along each dimension
    :param lens2: list of pixel 2 sizes along each dimension
    :return: overlap area of pixels
    """

    if isinstance(centers1, cp.ndarray):
        xp = cp
    else:
        xp = np

    centers1 = xp.array(centers1)
    centers2 = xp.array(centers2)
    centers1, centers2 = xp.broadcast_arrays(centers1, centers2)

    lens1 = np.expand_dims(xp.array(lens1), axis=tuple(range(centers1.ndim - 1)))

    if lens2 is None:
        lens2 = lens1

    lens2 = xp.array(lens2)

    # compute overlaps
    lower_edge = xp.max(xp.stack((centers1 - 0.5 * lens1, centers2 - 0.5 * lens2), axis=0), axis=0)
    upper_edge = xp.min(xp.stack((centers1 + 0.5 * lens1, centers2 + 0.5 * lens2), axis=0), axis=0)
    overlaps = upper_edge - lower_edge
    overlaps[overlaps < 0] = 0
    volume_overlap = xp.prod(overlaps, axis=-1)

    return volume_overlap


# translating images
def translate_pix(img: array,
                  shifts: tuple[float],
                  dr: tuple[float] = (1, 1),
                  axes: tuple[int] = (-2, -1),
                  wrap: bool = True,
                  pad_val: float = 0) -> (array, list[int]):
    """
    Translate image by given number of pixels with several different boundary conditions. If the shifts are sx, sy,
    then the image will be shifted by sx/dx and sy/dy. If these are not integers, they will be rounded to the closest
    integer.

    i.e. given img(y, x) return img(y + sy, x + sx). For example, a positive value for sx will shift the image
    to the left.

    :param img: image to translate
    :param shifts: distance to translate along each axis (s1, s2, ...). If these are not integers, then they will be
      rounded to the closest integer value.
    :param dr: size of pixels along each axis (dr1, dr2, ...)
    :param axes: identify the axes being wrapped, (a1, a2, ...)
    :param bool wrap: if true, pixels on the boundary are shifted across to the opposite side. If false, these
      parts of the array are padded with pad_val
    :param pad_val: value to pad portions of the image that would wrap around. Only used if wrap is False
    :return img_shifted, pix_shifts:
    """

    if isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    # make sure axes positive
    axes = np.mod(np.array(axes), img.ndim)

    # convert pixel shifts to integers
    shifts_pix = np.array([int(np.round(-s / d)) for s, d in zip(shifts, dr)])

    # only need to operate on img if pixel shift is not zero
    if np.any(shifts_pix != 0):
        # roll arrays. If wrap is True, this is all we need to do
        for s, axis in zip(shifts_pix, axes):
            img = xp.roll(img, s, axis=axis)

        if wrap:
            pass
        else:
            # set parts of axes that have wrapped around to zero
            for s, axis in zip(shifts_pix, axes):

                if s >= 0:
                    slices = tuple([slice(0, img.shape[ii]) if ii != axis else slice(0, s) for ii in range(img.ndim)])
                else:
                    slices = tuple([slice(0, img.shape[ii]) if ii != axis else
                                    slice(s + img.shape[axis], img.shape[axis]) for ii in range(img.ndim)])

                img[slices] = pad_val

    return img, shifts_pix


def translate_im(img: array,
                 xshift: np.ndarray,
                 yshift: np.ndarray,
                 drs: tuple[float] = (1, 1)) -> array:
    """
    Translate img(y,x) to img(y+yo, x+xo) using FFT. This approach is exact for band-limited functions.

    e.g. suppose the pixel spacing dx = 0.05 um and we want to shift the image by 0.0366 um,
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


def translate_ft(img_ft: array,
                 fx: np.ndarray,
                 fy: np.ndarray,
                 drs: Optional[list[float, float]] = None) -> array:
    """
    Given img_ft(f), return the translated function
    img_ft_shifted(f) = img_ft(f + shift_frq)
    using the FFT shift relationship, img_ft(f + shift_frq) = F[ exp(-2*pi*i * shift_frq * r) * img(r) ]
    This is an approximation to the Whittaker-Shannon interpolation formula which can be performed using only FFT's.
    In this sense, it is exact for band-limited functions.

    If an array with more than 2 dimensions is passed in, then the shift will be applied to the last two dimensions

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

    if isinstance(img_ft, cp.ndarray) and _cupy_available:
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

    # check if shapes are broadcastable
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

        # exponential phase ramp. Should be broadcastable to shape of img_ft
        axis_expand_frq = tuple(list(range(n_extra_dims - fx.ndim)) + [-2, -1])
        fx = xp.expand_dims(fx, axis=axis_expand_frq)
        fy = xp.expand_dims(fy, axis=axis_expand_frq)

    # do translations
    if xp.all(fx == 0) and xp.all(fy == 0):
        return xp.array(img_ft, copy=True)
    else:
        # 1. shift frequencies in img_ft so zero frequency is in corner using ifftshift
        # 2. inverse ft
        # 3. multiply by exponential factor
        # 4. take fourier transform, then shift frequencies back using fftshift
        if drs is None:
            drs = (1, 1)
        dy, dx = drs

        # must use symmetric frequency representation to do shifting correctly!
        # we are using the FT shift theorem to approximate the Whittaker-Shannon interpolation formula,
        # but we get an extra phase if we don't use the symmetric rep. AND only works perfectly if size odd
        # we do not apply an fftshift, so we don't have to apply an intermediate shift in our FFTs later
        axis_expand_x = tuple(range(n_extra_dims + 1))
        x = xp.expand_dims(xp.fft.fftfreq(nx) * nx * dx, axis=axis_expand_x)
        # todo: replace with this second option which is more intuitive
        # x = xp.expand_dims(xp.fft.ifftshift(np.arange(nx) - nx // 2) * dx, axis=axis_expand_x)

        axis_expand_y = tuple(range(n_extra_dims)) + (-1,)
        y = xp.expand_dims(xp.fft.fftfreq(ny) * ny * dy, axis=axis_expand_y)
        # y = xp.expand_dims(xp.fft.ifftshift(np.arange(ny) - ny // 2) * dy, axis=axis_expand_y)

        exp_factor = xp.exp(-1j * 2 * np.pi * (fx * x + fy * y))

        img_ft_shifted = xp.fft.fftshift(
                         xp.fft.fft2(xp.asarray(exp_factor) *
                         xp.fft.ifft2(xp.fft.ifftshift(img_ft, axes=(-1, -2)),
                                      axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        return img_ft_shifted
