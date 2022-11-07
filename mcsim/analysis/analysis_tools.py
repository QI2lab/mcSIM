"""
Miscellaneous helper functions
"""

import numpy as np
import localize_psf.rois as rois

_cupy_available = True
try:
    import cupy as cp
except ImportError:
    _cupy_available = False


def azimuthal_avg(img: np.ndarray,
                  dist_grid: np.ndarray,
                  bin_edges: np.ndarray,
                  weights: np.ndarray = None):
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

    # there are many possible approaches for doing azimuthal averaging. Naive way: for each mask az_avg = np.mean(img[mask])
    # also can do using scipy.ndimage.mean(img, labels=masks, index=np.arange(0, n_bins). scipy approach is slightly slower
    # than np.bincount. Naive approach ~ factor of 2 slower.

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
    az_avg = np.bincount(masks[to_use_inds], img[to_use_inds]) / npts_bin
    # correct variance for unbiased estimator. (of course still biased for sd)
    sd = np.sqrt(np.bincount(masks[to_use_inds], img[to_use_inds] ** 2) / npts_bin - az_avg ** 2) * np.sqrt(npts_bin / (npts_bin - 1))
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
            pass # if aspect ratio < 1 we are already in 'minor' units.
        elif units == 'major':
            distance_grid = distance_grid / aspect_ratio
        elif units == 'mean':
            distance_grid = distance_grid / np.sqrt(aspect_ratio)
        else:
            raise ValueError("'units' must be 'minor', 'major', or 'mean', but was '%s'" % units)
    else:
        if units == 'minor':
            distance_grid = distance_grid / aspect_ratio
        elif units == 'major':
            pass # if aspect ratio > 1 we are already in 'major' units
        elif units == 'mean':
            distance_grid = distance_grid / np.sqrt(aspect_ratio)
        else:
            raise ValueError("'units' must be 'minor', 'major', or 'mean', but was '%s'" % units)

    return distance_grid


# geometry tools
def get_peak_value(img: np.ndarray,
                   x: np.ndarray,
                   y: np.ndarray,
                   peak_coord: np.ndarray,
                   peak_pixel_size: int = 1):
    """
    Estimate value for a peak that is not precisely aligned to the pixel grid by performing a weighted average
    over neighboring pixels, based on how much these overlap with a rectangular area surrounding the peak.
    The size of this rectangular area is set by peak_pixel_size, given in integer multiples of a pixel.

    :param img: image containing peak
    :param x: x-coordinates of image
    :param y: y-coordinates of image
    :param peak_coord: peak coordinate [px, py]
    :param peak_pixel_size: number of pixels (along each direction) to sum to get peak value
    :return peak_value: estimated value of the peak
    """
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
    img_roi = img[roi[0]:roi[1], roi[2]:roi[3]]
    xx_roi = xx[roi[0]:roi[1], roi[2]:roi[3]]
    yy_roi = yy[roi[0]:roi[1], roi[2]:roi[3]]

    # estimate value from weighted average of pixels in ROI, based on overlap with pixel area centered at [px, py]
    weights = np.zeros(xx_roi.shape)
    for ii in range(xx_roi.shape[0]):
        for jj in range(xx_roi.shape[1]):
            weights[ii, jj] = pixel_overlap([py, px], [yy_roi[ii, jj], xx_roi[ii, jj]],
                                            [peak_pixel_size * dy, peak_pixel_size * dx], [dy, dx]) / (dx * dy)

    peak_value = np.average(img_roi, weights=weights)

    return peak_value


def pixel_overlap(centers1: np.ndarray,
                  centers2: np.ndarray,
                  lens1: np.ndarray,
                  lens2: np.ndarray=None):
    """
    Calculate overlap of two nd-square pixels. The pixels go from coordinates
    centers[ii] - 0.5 * lens[ii] to centers[ii] + 0.5 * lens[ii].

    :param centers1: list of coordinates defining centers of first pixel along each dimension
    :param centers2: list of coordinates defining centers of second pixel along each dimension
    :param lens1: list of pixel 1 sizes along each dimension
    :param lens2: list of pixel 2 sizes along each dimension
    :return overlaps: overlap area of pixels
    """

    centers1 = np.atleast_1d(centers1).ravel()
    centers2 = np.atleast_1d(centers2).ravel()
    lens1 = np.atleast_1d(lens1).ravel()

    if lens2 is None:
        lens2 = lens1

    lens2 = np.atleast_1d(lens2).ravel()

    overlaps = []
    for c1, c2, l1, l2 in zip(centers1, centers2, lens1, lens2):
        if np.abs(c1 - c2) >= 0.5*(l1 + l2):
            overlaps.append(0)
        else:
            # ensure whichever pixel has leftmost edge is c1
            if (c1 - 0.5 * l1) > (c2 - 0.5 * l2):
                c1, c2 = c2, c1
                l1, l2 = l2, l1
            # by construction left start of overlap is c2 - 0.5*l2
            # end is either c2 + 0.5 * l2 OR c1 + 0.5 * l1
            lstart = c2 - 0.5 * l2
            lend = np.min([c2 + 0.5 * l2, c1 + 0.5 * l1])
            overlaps.append(np.max([lend - lstart, 0]))

    return np.prod(overlaps)


# translating images
def translate_pix(img: np.ndarray,
                  shifts: tuple[float],
                  dr: tuple[float] = (1, 1),
                  axes: tuple[int] = (-2, -1),
                  wrap: bool = True,
                  pad_val: float = 0) -> (np.ndarray, list[int]):
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

    # make sure axes positive
    axes = np.mod(axes, img.ndim)

    # convert pixel shifts to integers
    shifts_pix = np.array([int(np.round(-s / d)) for s, d in zip(shifts, dr)])

    # only need to operate on img if pixel shift is not zero
    if np.any(shifts_pix != 0):
        # roll arrays. If wrap is True, this is all we need to do
        for s, axis in zip(shifts_pix, axes):
            img = np.roll(img, s, axis=axis)

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


def translate_im(img: np.ndarray,
                 shift: tuple[float],
                 drs: tuple[float] = (1, 1),
                 use_gpu: bool = _cupy_available) -> np.ndarray:
    """
    Translate img(y,x) to img(y+yo, x+xo) using FFT. This approach is exact for band-limited functions.

    e.g. suppose the pixel spacing dx = 0.05 um and we want to shift the image by 0.0366 um,
    then dx = 0.05 and shift = [0, 0.0366]

    :param img: NumPy array, size ny x nx
    :param shift: [yo, xo], in same units as pixels
    :param drs: (dy, dx) pixel size of image along y- and x-directions
    :param use_gpu: run on GPU using CuPy. NOTE: result will be returned as a CuPy array and caller must
    convert to NumPy array with get() method or etc.
    :return img_shifted:
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")

    if use_gpu:
        xp = cp
    else:
        xp = np

    img = xp.array(img)
    ny, nx = img.shape

    dy, dx = drs

    # must use symmetric frequency representation to do correctly!
    # we are using the FT shift theorem to approximate the Nyquist-Whittaker interpolation formula,
    # but we get an extra phase if we don't use the symmetric rep. AND only works perfectly if size odd
    fx = xp.array(xp.fft.fftfreq(nx, dx))
    fy = xp.array(xp.fft.fftfreq(ny, dy))
    fxfx, fyfy = xp.meshgrid(fx, fy)

    # 1. ft
    # 2. multiply by exponential factor
    # 3. inverse ft
    exp_factor = xp.exp(1j * 2 * np.pi * (shift[0] * fyfy + shift[1] * fxfx))
    img_shifted = xp.fft.fftshift(xp.fft.ifft2(xp.array(exp_factor) * xp.fft.fft2(xp.fft.ifftshift(img))))

    return img_shifted


def translate_ft(img_ft: np.ndarray,
                 fx: np.ndarray,
                 fy: np.ndarray,
                 drs: list[float, float] = None,
                 use_gpu: bool = _cupy_available) -> np.ndarray:
    """
    Given img_ft(f), return the translated function
    img_ft_shifted(f) = img_ft(f + shift_frq)
    using the FFT shift relationship, img_ft(f + shift_frq) = F[ exp(-2*pi*i * shift_frq * r) * img(r) ]

    This is an approximation to the Whittaker-Shannon interpolation formula which can be performed using only FFT's.
    In this sense, it is exact for band-limited functions.

    If an array with more than 2 dimensions is passed in, then the shift will be applied to the last two dimensions

    :param img_ft: array representing the fourier transform of an image with frequency origin centered as if using fftshift.
    Shape n1 x n2 x ... x n_{-2} x n_{-1}. Shifting is done along the last two axes.
    :param fx: array of x-shift frequencies
    fx and fy should either be broadcastable to the same szie as img_ft, or they should be of size
    n_{-m} x ... x n_{-3} x 2 where images along dimensions -m, ..., -3 are shifted in parallel
    :param fy:
    :param drs: (dy, dx) pixel size (sampling rate) of real space image in directions.
    :param use_gpu: perform Fourier transforms on GPU

    :return img_ft_shifted: shifted images, same size as img_ft
    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    if img_ft.ndim < 2:
        raise ValueError("img_ft must be at least 2D")

    n_extra_dims = img_ft.ndim - 2
    ny, nx = img_ft.shape[-2:]

    fx = xp.array(fx, copy=True)
    fy = xp.array(fy, copy=True)

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
                         xp.fft.fft2(xp.array(exp_factor) *
                         xp.fft.ifft2(xp.fft.ifftshift(img_ft, axes=(-1, -2)),
                                      axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        return img_ft_shifted
