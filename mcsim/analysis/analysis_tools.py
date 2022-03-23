"""
Miscellaneous helper functions. As I collect many relating to a certain area, these can be split into topic
specific modules.
"""

import numpy as np
import scipy.sparse as sp
from scipy import fft

# _cupy_available = False
_cupy_available = True
try:
    import cupy as cp
except ImportError:
    _cupy_available = False

import localize_psf.rois as rois

# image processing
def azimuthal_avg(img, dist_grid, bin_edges, weights=None):
    """
    Take azimuthal average of img. All points which have a dist_grid value lying
    between successive bin_edges will be averaged. Points are considered to lie within a bin
    if their value is strictly smaller than the upper edge, and greater than or equal to the lower edge.
    :param np.array img: 2D image
    :param np.array dist_grid:
    :param np.array or list bin_edges:
    :param np.array weights:

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
    masks = np.ones((img.shape[0], img.shape[1]), dtype=np.int) * n_bins
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


def elliptical_grid(params, xx, yy, units='mean'):
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


def bin(img, bin_size, mode='sum'):
    """
    bin image by combining adjacent pixels

    In 1D, this is a straightforward problem. The image is a vector,
    I = (I[0], I[1], ..., I[nx-1])
    and the binning operator is a nx/nx_bin x nx matrix
    M = [[1, 1, ..., 1, 0, ..., 0, 0, ..., 0]
         [0, 0, ..., 0, 1, ..., 1, 0, ..., 0]
         ...
         [0, ...,              0,  1, ..., 1]]
    which has a tensor product structure, which is intuitive because we are operating on each run of x points independently.
    M = identity(nx/nx_bin) \prod ones(1, nx_bin)
    the binned image is obtained from matrix multiplication
    Ib = M * I

    In 2D, this situation is very similar. Here we take the image to be a row stacked vector
    I = (I[0, 0], I[0, 1], ..., I[0, nx-1], I[1, 0], ..., I[ny-1, nx-1])
    the binning operator is a (nx/nx_bin)*(ny/ny_bin) x nx*ny matrix which has a tensor product structure.

    This time the binning matrix has dimension (nx/nx_bin * ny/ny_bin) x (nx * ny)
    The top row starts with nx_bin 1's, then zero until position nx, and then ones until position nx + nx_bin.
    This pattern continues, with nx_bin 1's starting at jj*nx for jj = 0,...,ny_bin-1. The second row follows a similar
    pattern, but shifted by nx_bin pixels
    M = [[1, ..., 1, 0, ..., 0, 1, ..., 1, 0,...]
         [0, ..., 0, 1, ..., 1, ...
    Again, this has tensor product structure. Notice that the first (nx/nx_bin) x nx entries are the same as the 1D case
    and the whole matrix is constructed from blocks of these.
    M = [identity(ny/ny_bin) \prod ones(1, ny_bin)] \prod  [identity(nx/nx_bin) \prod ones(1, nx_bin)]

    Again, Ib = M*I

    Probably this pattern generalizes to higher dimensions!

    :param img: image to be binned
    :param nbin: [ny_bin, nx_bin] where these must evenly divide the size of the image
    :param mode: either 'sum' or 'mean'
    :return:
    """
    # todo: could also add ability to bin in this direction. Maybe could simplify function by allowing binning in
    # arbitrary dimension (one mode), with another mode to bin only certain dimensions and leave others untouched.
    # actually probably don't need to distinguish modes, this can be done by looking at bin_size.
    # still may need different implementation for the cases, as no reason to flatten entire array to vector if not
    # binning. But maybe this is not really a performance hit anyways with the sparse matrices?

    # if three dimensional, bin each image
    if img.ndim == 3:
        ny_bin, nx_bin = bin_size
        nz, ny, nx = img.shape

        # size of image after binning
        nx_s = int(nx / nx_bin)
        ny_s = int(ny / ny_bin)

        m_binned = np.zeros((nz, ny_s, nx_s))
        for ii in range(nz):
            m_binned[ii, :] = bin(img[ii], bin_size, mode=mode)

    # bin 2D image
    elif img.ndim == 2:
        ny_bin, nx_bin = bin_size
        ny, nx = img.shape

        if ny % ny_bin != 0 or nx % nx_bin != 0:
            raise ValueError('bin size must evenly divide image size.')

        # size of image after binning
        nx_s = int(nx/nx_bin)
        ny_s = int(ny/ny_bin)

        # matrix which performs binning operation on row stacked matrix
        # need to use sparse matrices to bin even moderately sized images
        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        bin_mat_y = sp.kron(sp.identity(ny_s), np.ones((1, ny_bin)))
        bin_mat_xy = sp.kron(bin_mat_y, bin_mat_x)

        # row stack img. img.ravel() = [img[0, 0], img[0, 1], ..., img[0, nx-1], img[1, 0], ...]
        m_binned = bin_mat_xy.dot(img.ravel()).reshape([ny_s, nx_s])

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / (nx_bin * ny_bin)
        else:
            raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    # 1D "image"
    elif img.ndim == 1:

        nx_bin = bin_size[0]
        nx = img.size

        if nx % nx_bin != 0:
            raise ValueError('bin size must evenly divide image size.')
        nx_s = int(nx / nx_bin)

        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        m_binned = bin_mat_x.dot(img)

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / nx_bin
        else:
            raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    else:
        raise ValueError("Only 1D, 2D, or 3D arrays allowed")

    return m_binned


# resampling functions
def duplicate_pix(img, nx=2, ny=2):
    """
    Resample image by expanding each pixel into a rectangle of ny x nx identical pixels

    :param img: image to resample
    :param nx:
    :param ny:
    :return:
    """
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise TypeError('nx and ny must be ints')

    block = np.ones((ny, nx))
    img_resampled = np.kron(img, block)

    return img_resampled


def duplicate_pix_ft(img_ft, mx=2, my=2, centered=True):
    """
    Resample the Fourier transform of image. In real space, this operation corresponds to replacing each pixel with a
    myxmx square of identical pixels. Note that this is often NOT the desired resampling behavior if you e.g. have
    an image. In that case you should use expand_fourier_sp() instead.


    In Fourier space we take advantage of the following connection. Let a be the original sequence and
    b[2n] = b[2n+1] = a[n]
    a[k] = \sum_{n=0}^{N-1} a[n] * exp(-2*pi*i/N * k * n)
    b[k] = \sum_{n=0}^{N-1} a[n] * {exp[-2*pi*i/(2N) * k * 2n] + exp[-2*pi*i/(2N) * k * 2n]}
         =  (1 + exp(-2*pi*i*k/(2N)) a[k]
    For N-1 < k < 2N, we are free to replace k by k-N

    This generalizes to
    b[k] = a[k] * \sum_{l=0}^{m-1} exp(-2*pi*i*k/(mN) * l)
         = a[k] * (1 - exp(-2*pi*i*k/N)) / (1 - exp(-2*pi*i*k/(mN)))

    :param img_ft:
    :param centered: If False, treated as raw output of fft.fft2. If true, treated as fft.fftshift(fft.fft2())
    :return:
    """

    ny, nx = img_ft.shape

    if centered:
        img_ft = fft.ifftshift(img_ft)

    kxkx, kyky = np.meshgrid(range(nx*mx), range(ny*my))

    phase_x = np.exp(-1j*2*np.pi * kxkx / (nx*mx))
    factor_x = (1 - phase_x**mx) / (1 - phase_x)
    # at kx or ky = 0 these give indeterminate forms
    factor_x[kxkx == 0] = mx

    phase_y = np.exp(-1j*2*np.pi * kyky / (ny*my))
    factor_y = (1 - phase_y**my) / (1 - phase_y)
    factor_y[kyky == 0] = my

    img_ft_resampled = factor_x * factor_y * np.tile(img_ft, (my, mx))

    if centered:
        img_ft_resampled = fft.fftshift(img_ft_resampled)

    return img_ft_resampled


def resample_bandlimited(img, mag=(2, 2)):
    """
    Expand real-space imaging while keeping fourier content constant

    :param img: nD image
    :param mag:

    :return img_resampled:
    """
    img_resampled = fft.fftshift(fft.ifft2(resample_bandlimited_ft(fft.fftshift(fft.fft2(fft.ifftshift(img))), mag)))
    return img_resampled


def resample_bandlimited_ft(img_ft, mag=(2, 2)):
    """
    Expand image by factors of mx and my while keeping Fourier content constant.

    Let the initial (real space) array be a_{ij} and the final be b_{ij}.
    If a has odd sizes, b_{2i-1,2j-1} = a_{i,j}
    If a has even sizes, b_{2i, 2j} = a_{i,j}
    This choice is dictated by the ``natural'' FFT position values, and it ensures that the zero positions of b and a
    give the same value.

    NOTE: the expanded FT function is normalized so that the realspace values will match after an inverse FFT,
    thus the corresponding Fourier space components will have the relationship b_k = a_k * b.size / a.size

    :param img_ft: frequency space representation of image, arranged according to the natural FFT representation.
    e.g. img_ft = fftshift(fft2(ifftshift(img))).
    :param mag: (my, mx)

    :return img_ft_expanded:
    """
    # todo: add axes argument, so will only resample some axes

    if isinstance(mag, int):
        mag = [mag]

    if not np.all([isinstance(m, int) for m in mag]):
        raise ValueError("mx and my must both be integers")

    if np.all([m == 1 for m in mag]):
        return img_ft

    if not np.all([m == 2 for m in mag]):
        raise NotImplementedError("not implemented for any expansion except factor of 2")

    # new method, works for arbitrary sized array
    # don't need frequencies, but useful for checking using proper points in arrays
    # frq_old = [get_fft_frqs(n) for n in img_ft.shape]
    # frq_new = [get_fft_frqs(n * m, dt=1/m) for n, m in zip(img_ft.shape, mag)]
    # center frequency for FFT of odd or even size is at position n//2
    ind_start = [(m * n) // 2 - n // 2 for n, m in zip(img_ft.shape, mag)]

    slice_obj = tuple([slice(istart, istart + n, 1) for istart, n in zip(ind_start, img_ft.shape)])
    img_ft_exp = np.zeros([n * m for n, m in zip(img_ft.shape, mag)], dtype=np.complex)
    img_ft_exp[slice_obj] = img_ft

    # if initial array was even it had an unpaired negative frequency, but its pair is present in the larger array
    # this negative frequency was at -N/2, so this enters the IFT for a_n as a_(k=-N/2) * exp(2*np.pi*i * -n/2)
    # not that exp(2*np.pi*i * -k/2) = exp(2*np.pi*i * k/2), so this missing frequency doesn't matter for a
    # however, when we construct b, in the IFT for b_n we now have b_(k=-N/2) * exp(2*np.pi*i * -n/4)
    # Since we are supposing N is even, we must satisfy
    # b_(2n) = a_n -> b_(k=-L/2) + b_(k=L/2) = a_(k=-L/2)
    # Further, we want to ensure that b is real if a is real, which implies
    # b_(k=-N/2) = 0.5 * a(k=-N/2)
    # b_(k= N/2) = 0.5 * a(k=-N/2)
    # no complex conjugate is required for b_(k=N/2). If a_n is real, then a(k=-N/2) must also be real.
    #
    # consider the 2D case. We have an unfamiliar condition required to make a real
    # a_(ky=-N/2, kx) = conj(a_(ky=-N/2, -kx))
    # recall -N/2 <-> N/2 to make this more familiar
    # for b_(n, m) we have b_(ky=-N/2, kx) * exp(2*np.pi*i * -n/4) * exp(2*np.pi*i * kx*m/(fx*N))
    # to ensure all b_(n, m) are real we must enforce
    # b_(ky=N/2, kx) = conj(b(ky=-N/2, -kx))
    # b_(ky, kx=N/2) = conj(b(-ky, kx=-N/2))
    # on the other hand, to enforce b_(2n, 2m) = a_(n, m)
    # a(ky=-N/2,  kx) = b(ky=-N/2,  kx) + b(ky=N/2,  kx)
    # a(ky=-N/2, -kx) = b(ky=-N/2, -kx) + b(ky=N/2, -kx) = b^*(ky=-N/2, kx) + b^*(ky=N/2, kx)
    # but this second equation doesn't give us any more information than the real condition above
    # the easiest way to do this is...
    # b(ky=+/- N/2, kx) = 0.5 * a(ky=-N/2, kx)
    # for the edges, the conditions are
    # b(ky=+/- N/2, kx=+/- N/2) = 0.25 * a(ky=kx=-N/2)
    # b(ky=+/- N/2, kx=-/+ N/2) = 0.25 * a(ky=kx=-N/2)

    for ii in range(img_ft.ndim):
        slice_obj = [slice(None, None)] * img_ft.ndim
        slice_obj[ii] = slice(ind_start[ii], ind_start[ii] + 1)

        val = img_ft_exp[tuple(slice_obj)]
        img_ft_exp[tuple(slice_obj)] *= 0.5

        slice_obj[ii] = slice(ind_start[ii] + img_ft.shape[ii], ind_start[ii] + img_ft.shape[ii] + 1)
        img_ft_exp[tuple(slice_obj)] = val

    # correct normalization so real-space values of expanded array match real-space values of initial array
    img_ft_exp = np.prod(mag) * img_ft_exp

    return img_ft_exp

# geometry tools
def get_peak_value(img, x, y, peak_coord, peak_pixel_size=1):
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


def pixel_overlap(centers1, centers2, lens1, lens2=None):
    """
    Calculate overlap of two nd-square pixels. The pixels go from coordinates
    centers[ii] - 0.5 * lens[ii] to centers[ii] + 0.5 * lens[ii].

    :param centers1: list of coordinates defining centers of first pixel along each dimension
    :param centers2: list of coordinates defining centers of second pixel along each dimension
    :param lens1: list of pixel 1 sizes along each dimension
    :param lens2: list of pixel 2 sizes along each dimension
    :return overlaps: overlap area of pixels
    """

    if not isinstance(centers1, list):
        centers1 = [centers1]

    if not isinstance(centers2, list):
        centers2 = [centers2]

    if not isinstance(lens1, list):
        lens1 = [lens1]

    if lens2 is None:
        lens2 = lens1

    if not isinstance(lens2, list):
        lens2 = [lens2]

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


def segment_intersect(start1, end1, start2, end2):
    """
    Get intersection point of two 2D line segments
    :param start1: [x, y]
    :param end1:
    :param start2:
    :param end2:
    :return:
    """
    # solve S1 * (1-t) + e1 * t = S2 * (1-r) * e2 * r
    # phrase this as roi_size matrix problem:
    # S1 - S2 = [[e1_x - s1_x, e2_x - s2_x]; [e1_y - s1_y, e2_y - s2_y]] * [t; r]
    start1 = np.asarray(start1)
    end1 = np.asarray(end1)
    start2 = np.asarray(start2)
    end2 = np.asarray(end2)

    try:
        # solve system of equations by inverting matrix
        M = np.array([[start1[0] - end1[0], end2[0] - start2[0]],
                      [start1[1] - end1[1], end2[1] - start2[1]]])
        vs = np.linalg.inv(M).dot(np.asarray([[start1[0] - start2[0]], [start1[1] - start2[1]]]))
    except np.linalg.LinAlgError:
        return None

    t = vs[0][0]
    r = vs[1][0]

    # check within bounds
    if t<=1 and t>=0 and r<=1 and r>=0:
        return start1 * (1-t) + end1 * t
    else:
        return None


def nearest_point_on_line(line_point, line_unit_vec, pt):
    """
    Find the shortest distance between a line and a point of interest.

    Parameterize line by v(t) = line_point + line_unit_vec * t

    :param line_point: a point on the line
    :param line_unit_vec: unit vector giving the direction of the line
    :param pt: the point of interest

    :return nearest_pt: the nearest point on the line to the point of interest
    :return dist: the minimum distance between the line and the point of interest.
    """
    if np.linalg.norm(line_unit_vec) != 1:
        raise ValueError("line_unit_vec norm != 1")
    tmin = np.sum((pt - line_point) * line_unit_vec)
    nearest_pt = line_point + tmin * line_unit_vec
    dist = np.sqrt(np.sum((nearest_pt - pt)**2))

    return nearest_pt, dist


def get_linecut(img, start_coord, end_coord, width):
    """
    Get data along a 1D line from img

    todo: would like the option to resample along the new coordinates? Otherwise the finite width
    can lead to artifacts
    :param img: 2D numpy array
    :param start_coord: [xstart, ystart], where the upper left pixel of the array is at [0, 0]
    :param end_coord: [xend, yend]
    :param width: width of cut
    :return xcut: coordinate along the cut (in pixels)
    :return cut: values along the cut
    """
    xstart, ystart = start_coord
    xend, yend = end_coord

    angle = np.arctan( (yend - ystart) / (xend - xstart))

    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    xrot = (xx - xstart) * np.cos(angle) + (yy - ystart) * np.sin(angle)
    yrot = (yy - ystart) * np.cos(angle) - (xx - xstart) * np.sin(angle)

    # line goes from (0,0) to (xend - xstart)
    mask = np.ones(img.shape)
    mask[yrot > 0.5 * width] = 0
    mask[yrot < -0.5 * width] = 0
    mask[xrot < 0] = 0
    xrot_end = (xend - xstart) * np.cos(angle) + (yend - ystart) * np.sin(angle)
    mask[xrot > xrot_end] = 0

    xcut = xrot[mask != 0]
    cut = img[mask != 0]

    # sort by coordinate
    inds = np.argsort(xcut)
    xcut = xcut[inds]
    cut = cut[inds]

    return xcut, cut


# working with regions of interest
def get_extent(y, x, origin="lower"):
    """
    Get extent required for plotting arrays using imshow in real coordinates. The resulting list can be
    passed directly to imshow using the extent keyword.

    Here we assume the values y and x are equally spaced and describe the center coordinates of each pixel

    :param y: equally spaced y-coordinates
    :param x: equally spaced x-coordinates
    :param origin: "lower" or "upper" depending on if the y-origin is at the lower or upper edge of the image
    :return extent: [xstart, xend, ystart, yend]
    """
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    if origin == "lower":
        extent = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[-1] + 0.5 * dy, y[0] - 0.5 * dy]
    elif origin == "upper":
        extent = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[0] - 0.5 * dy, y[-1] + 0.5 * dy]
    else:
        raise ValueError("origin must be 'lower' or 'upper' but was '%s'" % origin)

    return extent


def map_intervals(vals, from_intervals, to_intervals):
    """
    Given value v in interval [a, b], find the corresponding value in the interval [c, d]

    :param vals: list of vals [v1, v2, v3, ..., vn]
    :param from_intervals: list of intervals containing start values [[a1, b1], [a2, b2], ..., [an, bn]]
    :param to_intervals: list of intervals containing end valus [[c1, d1], [c2, d2], ..., [cn, dn]]
    :return:
    """

    # todo: maybe move to affine.py?
    if not isinstance(vals, list):
        vals = [vals]

    if not isinstance(from_intervals[0], list):
        from_intervals = [from_intervals]

    if not isinstance(to_intervals[0], list):
        to_intervals = [to_intervals]

    vals_out = []
    for v, i1, i2 in zip(vals, from_intervals, to_intervals):
        vals_out.append( (v - i1[0]) * (i2[1] - i2[0]) / (i1[1] - i1[0])  + i2[0])

    return vals_out


# fft tools
def get_fft_frqs(length, dt=1, centered=True, mode='symmetric'):
    """
    Get frequencies associated with FFT, ordered from largest magnitude negative to largest magnitude positive.

    We are always free to take f -> f + n*dt for any integer n, which allows us to transform between the 'symmetric'
    and 'positive' frequency representations.

    If fftshift=False, the natural frequency representation is the positive one, with
    f = [0, ..., L-1]/(dt*L)

    If fftshift=True, the natural frequency representation is the symmetric one
    If x = [0, ..., L-1], then
    for even length sequences, (L-1) = 2*N+1, and we have one more negative frequency than positive frequency:
    f = [-(N+1), ..., 0, ..., N]/(dt*L) = [-L/2, ..., 0, ..., (L-2)/2]
    and for odd length sequences, (L-1) = 2*N, and we have an equal number of negative and positive frequencies.
    f = [    -N, ..., 0, ..., N]/(dt*L) = [-(L-1)/2, ..., 0, ..., (L-1)/2]
    i.e. for sequences of even length, we have one more negative frequency than we have positive frequencies.


    :param length: length of sample
    :param dt: spacing between samples
    :param centered: Bool. Controls the order in which fequencies are returned. If true, return
    frequencies in the order corresponding to fftshift(fft(fn)), i.e. with origin in the center of the array.
    If false, origin is at the edge.
    :param mode: 'symmetric' or 'positive'. Controls which frequencies are repoted as postive/negative.
    If 'positive', return positive representation of all frequencies. If 'symmetric', return frequencies larger
    than length//2 as negative
    :return:
    """
    # todo: deprecated since almost identical functionality from fft.fftfreq() in combination with fft.fftshift()

    # generate symmetric, fftshifted frequencies
    if np.mod(length, 2) == 0:
        n = int((length - 2) / 2)
        frqs = np.arange(-(n+1), n+1) / length / dt
    else:
        n = int((length - 1) / 2)
        frqs = np.arange(-n, n+1) / length / dt

    # ifftshift if necessary
    if centered:
        pass
    else:
        # convert from origin at center to origin at edge
        frqs = fft.ifftshift(frqs)

    # shift back to positive if necessary
    if mode == 'symmetric':
        pass
    elif mode == 'positive':
        frqs[frqs < 0] = frqs[frqs < 0] + 1 / dt
    else:
        raise ValueError("mode must be 'symmetric' or 'positive', but was '%s'" % mode)

    return frqs


def get_fft_pos(length, dt=1, centered=True, mode='symmetric'):
    """
    Get position coordinates for use with fast fourier transforms (fft's) using one of several different conventions.

    With the default arguments, will return the appropriate coordinates for the idiom
    array_ft = fftshift(fft2(ifftshift(array)))

    We are always free to change the position by a multiple of the overall length, i.e. x -> x + n*L for n an integer.

    if centered=False,
    pos = [0, 1, ..., L-1] * dt

    if centered=True, then for a sequence of length L, we have
    [- ceil( (L-1)/2), ..., 0, ..., floor( (L-1)/2)]
    which is symmetric for L odd, and has one more positive coordinate for L even

    :param length: length of array
    :param dt: spacing between points
    :param centered: controls the order in which frequencies are returned.
    :param mode: "positive" or "symmetric": control which frequencies are reported as positive vs. negative

    :return pos: list of positions
    """

    # symmetric, centered frequencies
    pos = np.arange(-np.ceil(0.5 * (length - 1)), np.floor(0.5 * (length - 1)) + 1)

    if mode == 'symmetric':
        pass
    elif mode == 'positive':
        pos[pos < 0] = pos[pos < 0] + length
    else:
        raise ValueError("mode must be 'symmetric' or 'positive', but was '%s'" % mode)

    if centered:
        pass
    else:
        pos = fft.ifftshift(pos)

    pos = pos * dt

    return pos


def get_spline_fn(x1, x2, y1, y2, dy1, dy2):
    """

    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param dy1:
    :param dy2:
    :return spline_fn:
    :return spline_deriv:
    :return coeffs:
    """
    # s(x) = a * x**3 + b * x**2 + c * x + d
    # vec = mat * [[a], [b], [c], [d]]
    vec = np.array([[y1], [dy1], [y2], [dy2]])
    mat = np.array([[x1**3, x1**2, x1, 1],
                   [3*x1**2, 2*x1, 1, 0],
                   [x2**3, x2**2, x2, 1],
                   [3*x2**2, 2*x2, 1, 0]])
    coeffs = np.linalg.inv(mat).dot(vec)

    fn = lambda x: coeffs[0, 0] * x**3 + coeffs[1, 0] * x ** 2 + coeffs[2, 0] * x + coeffs[3, 0]
    dfn = lambda x: 3 * coeffs[0, 0] * x**2 + 2 * coeffs[1, 0] * x + coeffs[2, 0]
    return fn, dfn, coeffs


# translating images
def translate_pix(img, shifts, dr=(1, 1), axes=(-2, -1), wrap=True, pad_val=0):
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
                    slices = tuple([slice(0, img.shape[ii]) if ii != axis else slice(s + img.shape[axis], img.shape[axis]) for ii in range(img.ndim)])

                img[slices] = pad_val

    return img, shifts_pix


def translate_im(img, shift, dx=1, dy=None, use_gpu=_cupy_available):
    """
    Translate img(y,x) to img(y+yo, x+xo) using FFT. This approach is exact for band-limited functions.

    e.g. suppose the pixel spacing dx = 0.05 um and we want to shift the image by 0.0366 um,
    then dx = 0.05 and shift = [0, 0.0366]

    :param img: NumPy array, size ny x nx
    :param shift: [yo, xo], in same units as pixels
    :param dx: pixel size of image along x-direction
    :param dy: pixel size of image along y-direction
    :return img_shifted:
    """
    if img.ndim != 2:
        raise ValueError("img must be 2D")

    if dy is None:
        dy = dx
    # todo: take (dx, dy) as argument instead

    # must use symmetric frequency representation to do correctly!
    # we are using the FT shift theorem to approximate the Nyquist-Whittaker interpolation formula,
    # but we get an extra phase if we don't use the symmetric rep. AND only works perfectly if size odd
    fx = get_fft_frqs(img.shape[1], dt=dx, centered=False, mode='symmetric')
    fy = get_fft_frqs(img.shape[0], dt=dy, centered=False, mode='symmetric')
    fxfx, fyfy = np.meshgrid(fx, fy)

    # 1. ft
    # 2. multiply by exponential factor
    # 3. inverse ft
    exp_factor = np.exp(1j * 2 * np.pi * (shift[0] * fyfy + shift[1] * fxfx))
    if not use_gpu:
        img_shifted = fft.fftshift(fft.ifft2(exp_factor * fft.fft2(fft.ifftshift(img))))
    else:
        img_shifted = cp.asnumpy(cp.fft.fftshift(cp.fft.ifft2(cp.array(exp_factor) * cp.fft.fft2(cp.fft.ifftshift(img)))))

    return img_shifted


def translate_ft(img_ft, shift_frq, drs=None, apodization=None, use_gpu=_cupy_available):
    """
    Given img_ft(f), return the translated function
    img_ft_shifted(f) = img_ft(f + shift_frq)
    using the FFT shift relationship, img_ft(f + shift_frq) = F[ exp(-2*pi*i * shift_frq * r) * img(r) ]

    This is an approximation to the Whittaker-Shannon interpolation formula which can be performed using only FFT's.
    In this sense, it is exact for band-limited functions.

    If an array with more than 2 dimensions is passed in, then the shift will be applied to the last two dimensions

    :param img_ft: fourier transform, with frequencies centered using fftshift
    :param list[float] shift_frq: [fx, fy]. Frequency in hertz (i.e. angular frequency is k = 2*pi*f)
    :param list[float] drs: pixel size (sampling rate) of real space image in directions. For 2D, (dy, dx)
    :param apodization: can be applied to the Fourier transform images
    :param bool use_gpu:

    :return img_ft_shifted:
    """
    # todo: accept arbitrary dimension arrays. Would want to give an argument telling which dimensions to shift along

    if img_ft.ndim < 2:
        raise ValueError("img_ft must be at least 2D")
    shift_frq = np.array(shift_frq)

    if np.all(shift_frq == 0):
        return np.array(img_ft, copy=True)
    else:
        # 1. shift frequencies in img_ft so zero frequency is in corner using ifftshift
        # 2. inverse ft
        # 3. multiply by exponential factor
        # 4. take fourier transform, then shift frequencies back using fftshift
        if drs is None:
            drs = (1, 1)
        dy, dx = drs

        ndims = img_ft.ndim
        ny, nx = img_ft.shape[-2:]
        # must use symmetric frequency representation to do shifting correctly!
        # we are using the FT shift theorem to approximate the Whittaker-Shannon interpolation formula,
        # but we get an extra phase if we don't use the symmetric rep. AND only works perfectly if size odd
        # we do not apply an fftshift, so we don't have to apply an intermediate shift in our FFTs later
        # x = get_fft_pos(nx, dx, centered=False, mode='symmetric')
        # y = get_fft_pos(ny, dy, centered=False, mode='symmetric')
        x = fft.fftfreq(nx) * nx * dx
        y = fft.fftfreq(ny) * ny * dy

        # exponential phase ramp
        axis_expand_x = list(range(ndims - 2 + 1))
        axis_expand_y = list(range(ndims - 2)) + [-1]
        exp_factor = np.exp(-1j * 2 * np.pi * (shift_frq[0] * np.expand_dims(x, axis=axis_expand_x) +
                                               shift_frq[1] * np.expand_dims(y, axis=axis_expand_y)))

        if apodization is None:
            apodization = 1

        if not use_gpu:
            # ifft2(ifftshift(img_ft)) = ifftshift(img)
            img_ft_shifted = fft.fftshift(fft.fft2(exp_factor *
                                                   fft.ifft2(fft.ifftshift(img_ft * apodization, axes=(-1, -2)), axes=(-1, -2)),
                                                   axes=(-1, -2)), axes=(-1, -2))
        else:
            img_ft_shifted = cp.asnumpy(cp.fft.fftshift(
                                        cp.fft.fft2(cp.array(exp_factor) *
                                        cp.fft.ifft2(cp.fft.ifftshift(img_ft * apodization, axes=(-1, -2)), axes=(-1, -2)),
                                                    axes=(-1, -2)), axes=(-1, -2)))

        return img_ft_shifted


def conj_transpose_fft(img_ft, axes=(-1, -2)):
    """
    Given img_ft(f), return a new array
    img_new_ft(f) := conj(img_ft(-f))

    :param img_ft:
    :param axes: axes on which to perform the transformation
    """

    # convert axes to positive number
    axes = np.mod(axes, img_ft.ndim)

    # flip and conjugate
    img_ft_ct = np.flip(np.conj(img_ft), axis=axes)

    # for odd FFT size, can simply flip the array to take f -> -f
    # for even FFT size, have on more negative frequency than positive frequency component.
    # by flipping array, have put the negative frequency components on the wrong side of the array
    # (i.e. where the positive frequency components are)
    # so must roll array to put them back on the right side
    to_roll = [a for a in axes if np.mod(img_ft.shape[a], 2) == 0]
    img_ft_ct = np.roll(img_ft_ct, shift=[1] * len(to_roll), axis=to_roll)

    return img_ft_ct


def rfft2fft(img_rft):
    """
    Convert rfft2 representation to fft2
    """

    nx = img_rft.shape[0]
    
    img_fft = np.zeros((nx, nx), dtype=np.complex)
    img_fft[: img_rft.shape[0], : img_rft.shape[1]] = img_rft
    img_fft = fft.fftshift(img_fft)
    test = conj_transpose_fft(img_fft)
    img_fft[:, :nx // 2] = test[:, :nx // 2]
    
    return img_fft


def shannon_whittaker_interp(pts, fn_vals, dt=1):
    """
    Get function value between sampling points using Shannon-Whittaker interpolation formula.

    :param pts: point to find interpolated function
    :param fn_vals: function at points n * dt
    :param dt: sampling rate
    :return fn_interp: fn(pts)
    """
    ns = np.arange(len(fn_vals))
    fn_interp = np.zeros(pts.shape)
    for ii in range(pts.size):
        ind = np.unravel_index(ii, pts.shape)
        fn_interp[ind] = np.sum(fn_vals * sinc(np.pi * (pts[ind] - ns * dt) / dt))

    return fn_interp


def shannon_whittaker_interp2d(pts, fn_vals, drs):
    # todo: combine 1D and 2D functions to nD function
    ns = np.expand_dims(np.arange(fn_vals.shape[0]), axis=1)
    ms = np.expand_dims(np.arange(fn_vals.shape[1]), axis=0)

    fn_interp = np.zeros(pts.shape)
    for ii in range(pts.size):
        ind = np.unravel_index(ii, pts.shape)
        fn_interp[ind] = np.sum(fn_vals *
                                sinc(np.pi * (pts[ind] - ns * drs[0]) / drs[0]) *
                                sinc(np.pi * (pts[ind] - ms * drs[1]) / drs[1]))

    return fn_interp


def sinc(x):
    """
    sinc(x) = sin(x) / x
    """
    val = np.sin(x) / x
    val[x == 0] = 1
    return val
