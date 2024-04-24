"""
Miscellaneous helper functions. Many are written to run on either the CPU or the GPU depending on if the arrays
passed to them are NumPy or CuPy arrays
"""

import numpy as np
from typing import Union, Optional
from collections.abc import Sequence
from localize_psf.rois import get_centered_rois, cut_roi

try:
    import cupy as cp
except ImportError:
    cp = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


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

    if cp and isinstance(img, cp.ndarray):
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
    roi = get_centered_rois([iy, ix], [3 * peak_pixel_size, 3 * peak_pixel_size])[0]
    img_roi = cut_roi(roi, img)[0]

    xx_roi = xp.expand_dims(cut_roi(roi, xx)[0], axis=tuple(range(img_roi.ndim - 2)))
    yy_roi = xp.expand_dims(cut_roi(roi, yy)[0], axis=tuple(range(img_roi.ndim - 2)))

    weights = pixel_overlap(xp.array([[py, px]]),
                            xp.stack((yy_roi.ravel(), xx_roi.ravel()), axis=1),
                            [peak_pixel_size * dy, peak_pixel_size * dx],
                            [dy, dx]) / (dx * dy)

    _, weights = xp.broadcast_arrays(img_roi, weights.reshape(xx_roi.shape))

    peak_value = xp.average(img_roi, weights=weights, axis=(-1, -2))

    return peak_value


def pixel_overlap(centers1: array,
                  centers2: array,
                  lens1: Sequence[float],
                  lens2: Optional[Sequence[float]] = None) -> array:
    """
    Calculate overlap of two nd-rectangular pixels. The pixels go from coordinates
    centers[ii] - 0.5 * lens[ii] to centers[ii] + 0.5 * lens[ii].

    :param centers1: Array of size ncenters x ndims. coordinates define centers of first pixel along each dimension.
    :param centers2: Broadcastable to same size as centers1
    :param lens1: pixel 1 sizes along each dimension
    :param lens2: pixel 2 sizes along each dimension
    :return: overlap area of pixels
    """

    if cp and isinstance(centers1, cp.ndarray):
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
    lower_edge = xp.max(xp.stack((centers1 - 0.5 * lens1,
                                  centers2 - 0.5 * lens2), axis=0), axis=0)
    upper_edge = xp.min(xp.stack((centers1 + 0.5 * lens1,
                                  centers2 + 0.5 * lens2), axis=0), axis=0)
    overlaps = upper_edge - lower_edge
    overlaps[overlaps < 0] = 0
    volume_overlap = xp.prod(overlaps, axis=-1)

    return volume_overlap


# translating images
def translate_pix(img: array,
                  shifts: Sequence[float],
                  dr: Sequence[float] = (1, 1),
                  axes: Sequence[int] = (-2, -1),
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

    if cp and isinstance(img, cp.ndarray):
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
