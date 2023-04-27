"""
Miscellaneous helper functions. Many are written to run on the GPU if the array passed to them are CuPy arrays
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

def azimuthal_avg(img: np.ndarray,
                  dist_grid: np.ndarray,
                  bin_edges: np.ndarray,
                  weights: Optional[np.ndarray] = None):
    """
    Take azimuthal average of img. All points which have a dist_grid value lying
    between successive bin_edges will be averaged. Points are considered to lie within a bin
    if their value is strictly smaller than the upper edge, and greater than or equal to the lower edge.
    :param img: 2D image
    :param dist_grid:
    :param bin_edges:
    :param weights:

    :return az_avg:
    :return sdm:
    :return dist_mean:
    :return dist_sd:
    :return npts_bin:
    :return masks:
    """

    # there are many possible approaches for doing azimuthal averaging.
    # Naive way: for each mask az_avg = np.mean(img[mask])
    # also can do using scipy.ndimage.mean(img, labels=masks, index=np.arange(0, n_bins)
    # scipy approach is slightly slower than np.bincount.
    # Naive approach ~ factor of 2 slower.

    if weights is None:
        weights = np.ones(img.shape)

    n_bins = len(bin_edges) - 1
    # build masks. initialize with integer value that does not conflict with any of our bins
    masks = np.ones((img.shape[0], img.shape[1]), dtype=int) * n_bins
    for ii in range(n_bins):
        # create mask
        bmin = bin_edges[ii]
        bmax = bin_edges[ii + 1]
        mask = np.logical_and(dist_grid < bmax, dist_grid >= bmin)
        masks[mask] = ii

    # get indices to use during averaging. Exclude any nans in img, and exclude points outside of any bin
    to_use_inds = np.logical_and(np.logical_not(np.isnan(img)), masks < n_bins)
    npts_bin = np.bincount(masks[to_use_inds])

    # failing to correct for case where some points are not contained in any bins. These had the same bin index as
    # the first bin, which caused problems!
    # nan_inds = np.isnan(img)
    # npts_bin = np.bincount(masks.ravel(), np.logical_not(nan_inds).ravel())
    # set any points with nans to zero, and these will be ignored by averaging due to above correction of npts_bin
    # img[nan_inds] = 0
    # dist_grid[nan_inds] = 0
    # az_avg = np.bincount(masks.ravel(), img.ravel())[0:-1] / npts_bin
    # sd = np.sqrt(np.bincount(masks.ravel(), img.ravel() ** 2) / npts_bin - az_avg ** 2) * np.sqrt(npts_bin / (npts_bin - 1))
    # dist_mean = np.bincount(masks.ravel(), dist_grid.ravel()) / npts_bin
    # dist_sd = np.sqrt(np.bincount(masks.ravel(), dist_grid.ravel() ** 2) / npts_bin - dist_mean ** 2) * np.sqrt(npts_bin / (npts_bin - 1))

    # do azimuthal averaging
    az_avg = np.bincount(masks[to_use_inds], img[to_use_inds].real) / npts_bin + \
             1j * np.bincount(masks[to_use_inds], img[to_use_inds].imag) / npts_bin
    # correct variance for unbiased estimator. (of course still biased for sd)
    # todo: correct to handle complex numbers appropriately
    sd = np.sqrt(np.bincount(masks[to_use_inds], np.abs(img[to_use_inds]) ** 2) / npts_bin - np.abs(az_avg) ** 2) * np.sqrt(npts_bin / (npts_bin - 1))
    dist_mean = np.bincount(masks[to_use_inds], dist_grid[to_use_inds]) / npts_bin
    dist_sd = np.sqrt(np.bincount(masks[to_use_inds], dist_grid[to_use_inds] ** 2) / npts_bin - dist_mean ** 2) * np.sqrt(npts_bin / (npts_bin - 1))

    # pad to match expected size given number of bin edges provided
    n_occupied_bins = npts_bin.size
    extra_zeros = np.zeros(n_bins - n_occupied_bins)
    if n_occupied_bins < n_bins:
        npts_bin = np.concatenate((npts_bin, extra_zeros), axis=0)
        az_avg = np.concatenate((az_avg, extra_zeros * np.nan), axis=0)
        sd = np.concatenate((sd, extra_zeros * np.nan), axis=0)
        dist_mean = np.concatenate((dist_mean, extra_zeros * np.nan), axis=0)
        dist_sd = np.concatenate((dist_sd, extra_zeros * np.nan), axis=0)

    # alternate approach with scipy.ndimage functions. 10-20% slower in my tests
    # az_avg = ndimage.mean(img, labels=masks,  index=np.arange(0, n_bins))
    # sd = ndimage.standard_deviation(img, labels=masks, index=np.arange(0, n_bins))
    # dist_mean = ndimage.mean(dist_grid, labels=masks, index=np.arange(0, n_bins))
    # dist_sd = ndimage.standard_deviation(dist_grid, labels=masks, index=np.arange(0, n_bins))
    # npts_bin = ndimage.sum(np.ones(img.shape), labels=masks, index=np.arange(0, n_bins))

    sdm = sd / np.sqrt(npts_bin)

    return az_avg, sdm, dist_mean, dist_sd, npts_bin, masks


def elliptical_grid(params: np.ndarray,
                    xx: np.ndarray,
                    yy: np.ndarray,
                    units: str = 'mean') -> np.ndarray:
    """
    Get elliptical `distance' grid for use with azimuthal averaging. These `distances' will be the same for points lying
    on ellipses with the parameters specified by params.

    Ellipse equation is (x - cx) ^ 2 / A ^ 2 + (y - cy) ^ 2 / B ^ 2 = 1
    Define d_A  = sqrt((x - cx) ^ 2 + (y - cy) ^ 2 * (A / B) ^ 2)...which is the
    Define d_B  = sqrt((x - cx) ^ 2 * (B / A) ^ 2 + (y - cy) ^ 2) = (B / A) * d_A
    Define d_AB = sqrt((x - cx) ^ 2 * (B / A) + (y - cy) ^ 2 * (A / B)) = sqrt(B / A) * d_A
    for a given ellipse, d_A is the distance along the A axis, d_B along the B
    axis, and d_AB along 45 deg axis.i.e.d_A(x, y) gives the length of the A
    axis of an ellipse with the given axes A and B that contains (x, y).

    :param params: [cx, cy, aspect_ratio, theta]. aspect_ratio = wy/wx. theta is the rotation angle of the x-axis of the
    ellipse measured CCW from the x-axis of the coordinate system
    :param xx: x-coordinates to compute grid on
    :param yy: y-coordinates to compute grid on
    :param units: 'mean', 'major', or 'minor'
    :return:
    """

    cx = params[0]
    cy = params[1]
    aspect_ratio = params[2]
    theta = params[3]

    distance_grid = np.sqrt(
        ((xx - cx) * np.cos(theta) - (yy - cy) * np.sin(theta))**2 +
        ((yy - cy) * np.cos(theta) + (xx - cx) * np.sin(theta))**2 * aspect_ratio**2)

    if aspect_ratio < 1:
        if units == 'minor':
            pass  # if aspect ratio < 1 we are already in 'minor' units.
        elif units == 'major':
            distance_grid = distance_grid / aspect_ratio
        elif units == 'mean':
            distance_grid = distance_grid / np.sqrt(aspect_ratio)
        else:
            raise ValueError(f"'units' must be 'minor', 'major', or 'mean', but was '{units:s}'")
    else:
        if units == 'minor':
            distance_grid = distance_grid / aspect_ratio
        elif units == 'major':
            pass  # if aspect ratio > 1 we are already in 'major' units
        elif units == 'mean':
            distance_grid = distance_grid / np.sqrt(aspect_ratio)
        else:
            raise ValueError(f"'units' must be 'minor', 'major', or 'mean', but was '{units:s}'")

    return distance_grid


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

    :return peak_value: estimated value of the peak
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
    ix = np.argmin(np.abs(px - x))
    iy = np.argmin(np.abs(py - y))

    # get ROI around pixel for weighted averaging
    roi = rois.get_centered_roi([iy, ix], [3 * peak_pixel_size, 3 * peak_pixel_size])
    img_roi = rois.cut_roi(roi, img)

    xx_roi = xp.expand_dims(rois.cut_roi(roi, xx), axis=tuple(range(img_roi.ndim - 2)))
    yy_roi = xp.expand_dims(rois.cut_roi(roi, yy), axis=tuple(range(img_roi.ndim - 2)))


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
    :return overlaps: overlap area of pixels
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
                 shift: tuple[float],
                 drs: tuple[float] = (1, 1)) -> array:
    """
    Translate img(y,x) to img(y+yo, x+xo) using FFT. This approach is exact for band-limited functions.

    e.g. suppose the pixel spacing dx = 0.05 um and we want to shift the image by 0.0366 um,
    then dx = 0.05 and shift = [0, 0.0366]

    :param img: NumPy or CuPy array, size ny x nx. If CuPy array will run on GPU
    :param shift: [yo, xo], in same units as pixels
    :param drs: (dy, dx) pixel size of image along y- and x-directions
    :return img_shifted:
    """

    # todo: use same approach as translate_ft() to make this work with nD arrays only operating along last two dims

    if img.ndim != 2:
        raise ValueError("img must be 2D")

    if isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    img = xp.asarray(img)
    ny, nx = img.shape

    dy, dx = drs

    # must use symmetric frequency representation to do correctly!
    # we are using the FT shift theorem to approximate the Nyquist-Whittaker interpolation formula,
    # but we get an extra phase if we don't use the symmetric rep. AND only works perfectly if size odd
    fx = xp.asarray(xp.fft.fftfreq(nx, dx))
    fy = xp.asarray(xp.fft.fftfreq(ny, dy))
    fxfx, fyfy = xp.meshgrid(fx, fy)

    # 1. ft
    # 2. multiply by exponential factor
    # 3. inverse ft
    exp_factor = xp.exp(1j * 2 * np.pi * (shift[0] * fyfy + shift[1] * fxfx))
    img_shifted = xp.fft.fftshift(xp.fft.ifft2(xp.asarray(exp_factor) * xp.fft.fft2(xp.fft.ifftshift(img))))

    return img_shifted


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

    :param img_ft: NumPy or CuPy array representing the fourier transform of an image with
     frequency origin centered as if using fftshift. Shape n1 x n2 x ... x n_{-2} x n_{-1}.
      Shifting is done along the last two axes. If this is a CuPy array, routine will be run on the GPU
    :param fx: array of x-shift frequencies
    fx and fy should either be broadcastable to the same size as img_ft, or they should be of size
    n_{-m} x ... x n_{-3} x 2 where images along dimensions -m, ..., -3 are shifted in parallel
    :param fy:
    :param drs: (dy, dx) pixel size (sampling rate) of real space image in directions.

    :return img_ft_shifted: shifted images, same size as img_ft
    """

    use_gpu = isinstance(img_ft, cp.ndarray) and _cupy_available

    if use_gpu:
        xp = cp
        # avoid issues like https://github.com/cupy/cupy/issues/6355
        cp.fft._cache.PlanCache(memsize=0)
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
        axis_expand_frq = list(range(n_extra_dims - fx.ndim)) + [-2, -1]
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
        # x = np.expand_dims(fft.ifftshift(np.arange(nx) - nx // 2) * dx, axis=axis_expand_x)

        axis_expand_y = tuple(range(n_extra_dims)) + (-1,)
        y = xp.expand_dims(xp.fft.fftfreq(ny) * ny * dy, axis=axis_expand_y)
        # y = np.expand_dims(fft.ifftshift(np.arange(ny) - ny // 2) * dy, axis=axis_expand_y)

        exp_factor = xp.exp(-1j * 2 * np.pi * (fx * x + fy * y))

        img_ft_shifted = xp.fft.fftshift(
                         xp.fft.fft2(xp.asarray(exp_factor) *
                         xp.fft.ifft2(xp.fft.ifftshift(img_ft, axes=(-1, -2)),
                                      axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        if use_gpu:
            cache = cp.fft.config.get_plan_cache()
            cache.clear()

        return img_ft_shifted
