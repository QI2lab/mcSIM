"""
Functions for estimating PSF's from images of fluorescent beads (z-stacks or single planes). Useful for generating
experimental PSF's from the average of many beads and fitting 2D and 3D PSF models to beads. Also includes tools for
working withpoint-spread functions and optical transfer functions more generally.

The primary functions that will be called by an external script are, find_beads(), autofit_psfs(), and display_autofit().
"""
import os
import timeit
import pickle
import copy

import numpy as np
import scipy.ndimage as ndi
import scipy.special as sp
import scipy.integrate
import scipy.interpolate
from scipy import fft
import skimage.feature
import skimage.filters
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm, LinearSegmentedColormap, Normalize

import joblib
from functools import partial

import psfmodels as psfm
import localize
import fit
import analysis_tools as tools

def blur_img_otf(ground_truth, otf):
    """
    Blur image with OTF

    :param ground_truth:
    :param otf: optical transfer function evalated at the FFT frequencies (with f=0 near the center of the array)

    :return img_blurred:
    """
    gt_ft = fft.fftshift(fft.fft2(fft.ifftshift(ground_truth)))
    img_blurred = fft.fftshift(fft.ifft2(fft.ifftshift(gt_ft * otf))).real

    return img_blurred

def blur_img_psf(ground_truth, psf):
    """
    Blur image with PSF

    :param ground_truth:
    :param psf: point-spread function
    # todo: allow PSF of different size than image
    """
    otf, _ = psf2otf(psf)

    return blur_img_otf(ground_truth, otf)

# model OTF function
def symm_fn_1d_to_2d(arr, fs, fmax, npts):
    """
    Convert a 1D function which is symmetric wrt to the radial variable to a 2D matrix. Useful helper function when
    computing PSFs from 2D OTFs
    :param arr:
    :param fs:
    :param df:
    :return:
    """

    ny = 2 * npts + 1
    nx = 2 * npts + 1
    # fmax = fs.max()
    dx = 1 / (2 * fmax)
    dy = dx

    not_nan = np.logical_not(np.isnan(arr))

    fxs = tools.get_fft_frqs(nx, dx)
    fys = tools.get_fft_frqs(ny, dy)
    fmag = np.sqrt(fxs[None, :]**2 + fys[:, None]**2)
    to_interp = np.logical_and(fmag >= fs[not_nan].min(), fmag <= fs[not_nan].max())


    arr_out = np.zeros((ny, nx), dtype=arr.dtype)
    arr_out[to_interp] = scipy.interpolate.interp1d(fs[not_nan], arr[not_nan])(fmag[to_interp])

    return arr_out, fxs, fys

# should rewrite these to handle 1D/3D functions
def otf2psf(otf, dfs=1):
    """
    Compute the point-spread function from the optical transfer function
    :param otf: otf, as a 1D, 2D or 3D array. Assumes that f=0 is near the center of the array, and frequency are
    arranged by the FFT convention
    :param dfs: (dfz, dfy, dfx), (dfy, dfx), or (dfx). If only a single number is provided, will assume these are the
    same
    :return psf, coords: where coords = (z, y, x)
    """

    if isinstance(dfs, (int, float)) and otf.ndim > 1:
        dfs = [dfs] * otf.ndim

    if len(dfs) != otf.ndim:
        raise ValueError("dfs length must be otf.ndim")

    shape = otf.shape
    drs = np.array([1 / (df * n) for df, n in zip(shape, dfs)])
    coords = [tools.get_fft_pos(n, dt=dr) for n, dr in zip(shape, drs)]

    psf = fft.fftshift(fft.ifftn(fft.ifftshift(otf))).real

    return psf, coords

def psf2otf(psf, drs=1):
    """
    Compute the optical transfer function from the point-spread function

    :param psf: psf, as a 1D, 2D or 3D array. Assumes that r=0 is near the center of the array, and positions
    are arranged by the FFT convention
    :param drs: (dz, dy, dx), (dy, dx), or (dx). If only a single number is provided, will assume these are the
    same
    :return otf, coords: where coords = (fz, fy, fx)
    """

    if isinstance(drs, (int, float)) and psf.ndim > 1:
        drs = [drs] * psf.ndim

    if len(drs) != psf.ndim:
        raise ValueError("drs length must be psf.ndim")

    shape = psf.shape
    coords = [tools.get_fft_frqs(n, dt=dr) for n, dr in zip(shape, drs)]

    otf = fft.fftshift(fft.fftn(fft.ifftshift(psf)))

    return otf, coords

def circ_aperture_pupil(fx, fy, na, wavelength):
    fmax = 0.5 / (0.5 * wavelength / na)

    nx = len(fx)
    ny = len(fy)
    ff = np.sqrt(fx[None, :]**2 + fy[:, None]**2)

    pupil = np.ones((ny, nx))
    pupil[ff > fmax] = 0

    return pupil

def circ_aperture_otf(fx, fy, na, wavelength):
    """
    OTF for roi_size circular aperture

    :param fx:
    :param fy:
    :param na: numerical aperture
    :param wavelength: in um
    :return:
    """
    # maximum frequency imaging system can pass
    fmax = 1 / (0.5 * wavelength / na)

    # freq data
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    ff = np.asarray(np.sqrt(fx**2 + fy**2))

    with np.errstate(invalid='ignore'):
        # compute otf
        otf = np.asarray(2 / np.pi * (np.arccos(ff / fmax) - (ff / fmax) * np.sqrt(1 - (ff / fmax)**2)))
        otf[ff > fmax] = 0

    return otf

def otf_coherent2incoherent(otf_c, dx=None, wavelength=0.5, ni=1.5, defocus_um=0, fx=None, fy=None):
    """
    Get incoherent transfer function from autocorrelation of coherent transfer function
    :param otf_c:
    :param dx:
    :param wavelength:
    :param ni:
    :param defocus_um:
    :param fx:
    :param fy:
    :return:
    """
    ny, nx = otf_c.shape

    if fx is None:
        fx = tools.get_fft_frqs(nx, dt=dx)
    if fy is None:
        fy = tools.get_fft_frqs(ny, dt=dx)

    if defocus_um != 0:

        if dx is None or wavelength is None or ni is None:
            raise TypeError("if defocus != 0, dx, wavelength, ni must be provided")

        k = 2*np.pi / wavelength * ni
        defocus_fn = np.exp(1j * defocus_um * np.sqrt(np.array(k**2 - (2 * np.pi)**2 * (fx[None, :]**2 + fy[:, None]**2), dtype=np.complex)))
    else:
        defocus_fn = 1

    otf_c_defocus = otf_c * defocus_fn
    # if even number of frequencies, we must translate otf_c by one so that f and -f match up
    otf_c_minus_conj = np.roll(np.roll(np.flip(otf_c_defocus, axis=(0, 1)), np.mod(ny + 1, 2), axis=0), np.mod(nx + 1, 2), axis=1).conj()

    otf_inc = scipy.signal.fftconvolve(otf_c_defocus, otf_c_minus_conj, mode='same') / np.sum(np.abs(otf_c) ** 2)
    return otf_inc, otf_c_defocus

# helper functions for converting between NA and peak widths
def na2fwhm(na, wavelength):
    """
    Convert numerical aperture to full-width at half-maximum, assuming an Airy-function PSF

    :param na: numerical aperture
    :param wavelength:
    :return fwhm: in same units as wavelength
    """
    fwhm = (1.6163399561827614) / np.pi * wavelength / na
    return fwhm

def fwhm2na(wavelength, fwhm):
    """

    @param wavelength:
    @param fwhm:
    @return:
    """
    na = (1.6163399561827614) / np.pi * wavelength / fwhm
    return na

def na2sxy(na, wavelength):
    """
    Convert numerical aperture to standard deviation, assuming the numerical aperture and the sigma
    are related as in the Airy function PSF
    :param na:
    :param wavelength:
    :return:
    """
    fwhm = na2fwhm(na, wavelength)
    sigma = 1.49686886 / 1.6163399561827614 / 2 * fwhm
    #2 * sqrt{2*log(2)} * sigma = 0.5 * wavelength / NA
    # sigma = na2fwhm(na, wavelength) / (2*np.sqrt(2 * np.log(2)))
    return sigma


def sxy2na(wavelength, sigma):
    """

    @param wavelength:
    @param sigma:
    @return:
    """
    fwhm = 2 * 1.6163399561827614 / 1.49686886 * sigma
    # fwhm = na2fwhm(na, wavelength)
    # fwhm = sigma * (2*np.sqrt(2 * np.log(2)))
    return fwhm2na(wavelength, fwhm)


# different PSF model functions
def gaussian2d_psf(x, y, p, sf=1):
    """
    2D Gaussian approximation to airy function. Matches well for equal peak intensity, but then area will not match.
    :param x:
    :param y:
    :param p: [A, cx, cy, NA, bg]
    :param wavelength:
    :return:
    """

    return gaussian3d_pixelated_psf_v2(x, y, np.array([0]), [p[0], p[1], p[2], 0, p[3], 1, p[4]], sf=sf, angles=(0., 0., 0.))


def gaussian3d_pixelated_psf_v2(x, y, z, dc, p, sf=3, angles=(0., 0., 0.)):
    """
    Gaussian function, accounting for image pixelation in the xy plane.

    vectorized, i.e. can rely on obeying broadcasting rules for x,y,z

    :param x:
    :param y:
    :param z: coordinates of z-planes to evaluate function at
    :param dc: pixel size
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :param angle: orientation of pixel to resample
    :return:
    """

    if not isinstance(sf, int):
        raise ValueError("sf must be an integer")

    # oversample points in pixel
    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dc, sf=sf, euler_angles=angles)

    # calculate psf at oversampled points
    psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4] ** 2
                   -(yy_s - p[2]) ** 2 / 2 / p[4] ** 2
                   -(zz_s - p[3]) ** 2 / 2 / p[5] ** 2)

    # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
    psf = p[0] * np.mean(psf_s, axis=-1) + p[-1]

    return psf


def gaussian3d_pixelated_psf_jac(x, y, z, dc, p, sf, angles=(0., 0., 0.)):
    """
    Jacobian of gaussian3d_pixelated_psf()

    :param x:
    :param y:
    :param z: coordinates of z-planes to evaluate function at
    :param dc: pixel size
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :return:
    """


    # oversample points in pixel
    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dc, sf=sf, euler_angles=angles)

    psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4] ** 2
                   -(yy_s - p[2]) ** 2 / 2 / p[4] ** 2
                   -(zz_s - p[3]) ** 2 / 2 / p[5] ** 2)

    # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
    # psf = p[0] * psf_sum + p[-1]

    bcast_shape = (x + y + z).shape
    # [A, cx, cy, cz, sxy, sz, bg]
    jac = [np.mean(psf_s, axis=-1),
           p[0] * np.mean(2 * (xx_s - p[1]) / 2 / p[4]**2 * psf_s, axis=-1),
           p[0] * np.mean(2 * (yy_s - p[2]) / 2 / p[4]**2 * psf_s, axis=-1),
           p[0] * np.mean(2 * (zz_s - p[3]) / 2/ p[5]**2 * psf_s, axis=-1),
           p[0] * np.mean((2 / p[4]**3 * (xx_s - p[1])**2 / 2 +
                           2 / p[4]**3 * (yy_s - p[2])**2 / 2) * psf_s, axis=-1),
           p[0] * np.mean(2 / p[5]**3 * (zz_s - p[3])**2 / 2 * psf_s, axis=-1),
           np.ones(bcast_shape)]

    return jac


def gaussian_lorentzian_psf(x, y, z, p, wavelength):
    """

    @param x:
    @param y:
    @param z:
    @param p: [A, cx, cy, cz, NA, bg, FWHM z]
    @param wavelength:
    @return:
    """
    sigma_xy = 0.22 * wavelength / p[4]

    lor_factor = 1 + (z - p[3]) ** 2 / p[6] ** 2
    norm = 2*np.pi * lor_factor * sigma_xy**2
    val = p[0] * np.exp(- ((x - p[1])**2 + (y - p[2])**2) / (2 * sigma_xy**2 * lor_factor)) / norm + p[5]

    return val


def gaussian_lorentzian_psf_jac(x, y, z, p, wavelength):
    """

    @param x:
    @param y:
    @param z:
    @param p:
    @param wavelength:
    @return:
    """
    sigma_xy = 0.22 * wavelength / p[4]

    lor_factor = 1 + (z - p[3]) ** 2 / p[6] ** 2
    norm = 2 * np.pi * lor_factor * sigma_xy ** 2

    factor = np.exp(- ((x - p[1]) ** 2 + (y - p[2]) ** 2) / (2 * sigma_xy ** 2 * lor_factor)) / norm

    # todo: finish
    jac = [factor,
           0,
           0,
           0,
           0,
           np.ones(),
           0]

    return jac


def born_wolf_psf(x, y, z, p, wavelength, ni, sf=1):
    """
    Born-wolf PSF function evaluated using Airy function if in-focus, and axial function if along the axis.
    Otherwise evaluated using numerical integration.

    :param x: in um
    :param y: in um
    :param z: in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param wavelength: in um
    :param ni: index of refraction
    :return:
    """
    if sf != 1:
        raise NotImplementedError("Only implemented for sf=1")

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x, y, z = np.broadcast_arrays(x, y, z)

    k = 2 * np.pi / wavelength
    rr = np.sqrt((x - p[1]) ** 2 + (y - p[2]) ** 2)

    psfs = np.zeros(rr.shape) * np.nan
    is_in_focus = (z == p[3])
    is_on_axis = (rr == 0)

    # ################################
    # evaluate in-focus portion using airy function, which is much faster than integrating
    # ################################
    def airy_fn(rho):
        val = p[0] * 4 * np.abs(sp.j1(rho * k * p[4]) / (rho * k * p[4]))**2 + p[5]
        val[rho == 0] = p[0] + p[4]
        return val

    with np.errstate(invalid="ignore"):
        psfs[is_in_focus] = airy_fn(rr[is_in_focus])

    # ################################
    # evaluate on axis portion using exact expression
    # ################################
    def axial_fn(z):
        val = p[0] * 4 * (2 * ni ** 2) / (k ** 2 * p[4] ** 4 * (z - p[3]) ** 2) * \
               (1 - np.cos(0.5 * k * (z - p[3]) * p[4] ** 2 / ni)) + p[5]
        val[z == p[3]] = p[0] + p[5]
        return val

    with np.errstate(invalid="ignore"):
        psfs[is_on_axis] = axial_fn(z[is_on_axis])

    # ################################
    # evaluate out of focus portion using integral
    # ################################
    if not np.all(is_in_focus):

        def integrand(rho, r, z):
            return rho * sp.j0(k * r * p[4] * rho) * np.exp(-1j * k * (z - p[3]) * p[4]**2 * rho**2 / (2 * ni))

        # like this approach because allows rr, z, etc. to have arbitrary dimension
        already_evaluated = np.logical_or(is_in_focus, is_in_focus)
        for ii, (r, zc, already_eval) in enumerate(zip(rr.ravel(), z.ravel(), already_evaluated.ravel())):
            if already_eval:
                continue

            int_real = scipy.integrate.quad(lambda rho: integrand(rho, r, zc).real, 0, 1)[0]
            int_img = scipy.integrate.quad(lambda rho: integrand(rho, r, zc).imag, 0, 1)[0]

            coords = np.unravel_index(ii, rr.shape)
            psfs[coords] = p[0] * 4 * (int_real ** 2 + int_img **2) + p[5]

    return psfs


# utility functions
def oversample_pixel(x, y, z, ds, sf=3, euler_angles=(0., 0., 0.)):
    """
    Suppose we have a set of pixels centered at points given by x, y, z. Generate sf**2 points in this pixel equally
    spaced about the center. Allow the pixel to be orientated in an arbitrary direction with respect to the coordinate
    system. The pixel rotation is described by the Euler angles (psi, theta, phi), where the pixel "body" frame
    is a square with xy axis orientated along the legs of the square with z-normal to the square

    :param x:
    :param y:
    :param z:
    :param ds: pixel size
    :param sf: sample factor
    :param euler_angles: [phi, theta, psi] where phi and theta are the polar angles describing the normal of the pixel,
    and psi describes the rotation of the pixel about its normal
    :return:

    """
    # generate new points in pixel, each of which is centered about an equal area of the pixel, so summing them is
    # giving an approximation of the integral
    if sf > 1:
        pts = np.arange(1 / (2*sf), 1 - 1 / (2*sf), 1 / sf) - 0.5
        xp, yp = np.meshgrid(ds * pts, ds * pts)
        zp = np.zeros(xp.shape)

        # rotate points to correct position using normal vector
        # for now we will fix x, but lose generality
        mat = fit.euler_mat(*euler_angles)
        result = mat.dot(np.concatenate((xp.ravel()[None, :], yp.ravel()[None, :], zp.ravel()[None, :]), axis=0))
        xs, ys, zs = result

        # now must add these to each point x, y, z
        xx_s = x[..., None] + xs[None, ...]
        yy_s = y[..., None] + ys[None, ...]
        zz_s = z[..., None] + zs[None, ...]
    else:
        xx_s = np.expand_dims(x, axis=-1)
        yy_s = np.expand_dims(y, axis=-1)
        zz_s = np.expand_dims(z, axis=-1)

    return xx_s, yy_s, zz_s


# main functions for dealing with PSF
def get_psf_coords(ns, drs):
    """
    Get centered coordinates for PSFmodels style PSF's from step size and number of coordinates
    :param ns: list of number of points
    :param drs: list of step sizes
    :return coords: list of coordinates [[c1, c2, c3, ...], ...]
    """
    return [d * (np.arange(n) - n // 2) for n, d in zip(ns, drs)]


def model_psf(nx, dxy, z, p, wavelength, ni, sf=1, model='vectorial', **kwargs):
    """
    Wrapper function for evaluating different PSF models where only the coordinate grid information is given
    (i.e. nx and dxy) and not the actual coordinates. The real coordinates can be obtained using get_psf_coords().

    The size of these functions is parameterized by the numerical aperture, instead of the sigma or other size
    parameters that are only convenient for some specific models of the PSF

    For vectorial or gibson-lanni PSF's, this wraps the functions in the psfmodels package
    (https://pypi.org/project/psfmodels/) with added ability to shift the center of these functions away from the
    center of the ROI, which is useful when fitting them.

    For 'gaussian', it wraps the gaussian3d_pixelated_psf() function. More details about the relationship between
    the Gaussian sigma and the numerical aperture can be found here: https://doi.org/10.1364/AO.46.001819

    todo: need to implement index of refraction?

    :param nx: number of points to be sampled in x- and y-directions
    :param dxy: pixel size in um
    :param z: z positions in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param wavelength: wavelength in um
    :param model: 'gaussian', 'gibson-lanni', or 'vectorial'. 'gibson-lanni' relies on the psfmodels function
    scalar_psf(), while 'vectorial' relies on the psfmodels function vectorial_psf()
    :param kwargs: keywords passed through to vectorial_psf() or scalar_psf()
    :return:
    """
    if 'NA' in kwargs.keys():
        raise ValueError("'NA' is not allowed to be passed as a named parameter. It is specified in p.")

    model_params = {'NA': p[4], 'sf': sf}
    model_params.update(kwargs)

    if model == 'vectorial':
        if sf != 1:
            raise NotImplementedError('vectorial model not implemented for sf=/=1')
        psf_norm = psfm.vectorial_psf(0, 1, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = psfm.vectorial_psf(z - p[3], nx, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = p[0] / psf_norm * ndi.shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]
    elif model == 'gibson-lanni':
        if sf != 1:
            raise NotImplementedError('gibson-lanni model not implemented for sf=/=1')
        psf_norm = psfm.scalar_psf(0, 1, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = psfm.scalar_psf(z - p[3], nx, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = p[0] / psf_norm * ndi.shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]
    elif model == "born-wolf":
        if sf != 1:
            raise NotImplementedError('gibson-lanni model not implemented for sf=/=1')

        y, x, = get_psf_coords([nx, nx], [dxy, dxy])
        y = np.expand_dims(y, axis=(0, 2))
        x = np.expand_dims(x, axis=(0, 1))
        z = np.expand_dims(np.array(z, copy=True), axis=(1, 2))

        val = born_wolf_psf(x, y, z, p, wavelength, ni, sf=sf)
    elif model == 'gaussian':
        # Gaussian approximation to PSF. Matches well for equal peak intensity, but some deviations in area.
        # See https://doi.org/10.1364/AO.46.001819 for more details.
        # sigma_xy = 0.22 * lambda / NA.
        # This comes from equating the FWHM of the Gaussian and the airy function.
        # FWHM = 2 * sqrt{2*log(2)} * sigma ~ 0.51 * wavelength / NA
        # transform NA to sigmas
        p_gauss = [p[0], p[1], p[2], p[3],
                   0.22 * wavelength / p[4],
                   np.sqrt(6) / np.pi * ni * wavelength / p[4] ** 2,
                   p[5]]

        y, x, = get_psf_coords([nx, nx], [dxy, dxy])
        y = np.expand_dims(y, axis=(0, 2))
        x = np.expand_dims(x, axis=(0, 1))
        z = np.expand_dims(np.array(z, copy=True), axis=(1, 2))

        # normalize so that peak amplitude is actually
        psf_norm = gaussian3d_pixelated_psf_v2(p[1], p[2], p[3], dxy, p_gauss, sf, angles=(0., 0., 0.)) - p[5]
        val = p[0] / psf_norm * (gaussian3d_pixelated_psf_v2(x, y, z, dxy, p_gauss, sf, angles=(0., 0., 0.)) - p[5]) + p[5]
    else:
        raise ValueError("model must be 'gibson-lanni', 'vectorial', or 'gaussian' but was '%s'" % model)

    return val


def fit_pixelated_psfmodel(img, dxy, dz, wavelength, ni, sf=1, model='vectorial',
                           init_params=None, fixed_params=None, sd=None, bounds=None):
    """
    3D non-linear least squares fit using one of the point spread function models from psfmodels package.

    The x/y coordinates are assumed to match the convention of get_coords(), i.e. they are (arange(nx) / nx//2) * d

    # todo: make sure ni implemented correctly. if want to use different ni, have to be careful because this will shift the focus position away from z=0
    # todo: make sure oversampling (sf) works correctly with all functions

    :param img: Nz x Ny x Nx image stack
    :param dxy: dx and dy in um
    :param dz: dz in um
    :param init_params: [A, cx, cy, cz, NA, bg]
    :param sd: standard deviations if img is derived from averaged pictures
    :param wavelength: wavelength in um
    :param model: 'gaussian', 'gibson-lanni', or 'vectorial'

    :return result:
    :return fit_fn:
    """

    # get coordinates
    z, y, x = get_psf_coords(img.shape, [dz, dxy, dxy])

    # check size
    nz, ny, nx = img.shape
    if not ny == nx:
        raise ValueError('x- and y-size of img must be equal')

    if not np.mod(nx, 2) == 1:
        raise ValueError('x- and y-size of img must be odd')

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 6
    else:
        init_params = copy.deepcopy(init_params)

    # use default values for any params that are None
    if np.any([ip is None for ip in init_params]):
        # exclude nans
        to_use = np.logical_not(np.isnan(img))

        bg = np.mean(img[to_use].ravel())
        A = np.max(img[to_use].ravel()) - bg

        cz, cy, cx = fit.get_moments(img, order=1, coords=[z, y, x])

        # iz = np.argmin(np.abs(z - cz))
        # m2y, m2x = tools.get_moments(img[iz], order=2, coords=[y, x])
        # sx = np.sqrt(m2x - cx ** 2)
        # sy = np.sqrt(m2y - cy ** 2)
        # # from gaussian approximation
        # na_guess = 0.22 * wavelength / np.sqrt(sx * sy)
        na_guess = 1

        ip_default = [A, cx, cy, cz, na_guess, bg]

        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    # set bounds
    if bounds is None:
        # allow for 2D fitting by making z bounds eps apart
        if z.min() == z.max():
            zlow = z.min() - 1e-12
            zhigh = z.max() + 1e-12
        else:
            zlow = z.min()
            zhigh = z.max()

        # NA must be <= index of refraction
        bounds = ((0, x.min(), y.min(), zlow, 0, -np.inf),
                  (np.inf, x.max(), y.max(), zhigh, ni, np.inf))

    # do fitting
    model_fn = lambda z, nx, dxy, p: model_psf(nx, dxy, z, p, wavelength, ni, sf=sf, model=model)

    result = fit.fit_model(img, lambda p: model_fn(z, nx, dxy, p), init_params,
                           fixed_params=fixed_params, sd=sd, bounds=bounds, jac='3-point', x_scale='jac')

    # model function at fit parameters
    fit_fn = lambda z, nx, dxy: model_fn(z, nx, dxy, result['fit_params'])

    return result, fit_fn


def plot_psfmodel_fit(imgs, dx, dz, wavelength, ni, sf, fit_params, model='vectorial',
                      gamma=1, figsize=(18, 10), save_dir=None, label='', **kwargs):
    """
    Plot data and fit obtained from fit_psfmodel().

    Multiple different fits can be plotted if fit_params, chi_sqrs, cov, and model are provided as lists.

    :param ni:
    :param imgs: 3D image stack
    :param dx: pixel size in um
    :param dz: space between z-planes in um
    :param fits: list of fit dictionary objects to be plotted.
    :param model: 'vectorial', 'gibson-lanni', 'born-wolf', or 'gaussian'
    :param figsize:
    :param save_dir: if not None, then a png of figure will be saved in the provided directory
    :param label: label to add to the start of the file name, if saving
    :param kwargs: additional keyword arguments are passed to plt.figure()
    :return:
    """

    # get coordinates
    nz, ny, nx = imgs.shape
    z, y, x, = get_psf_coords([nz, ny, nx], [dz, dx, dx])

    # other useful coordinate info
    extent_xy = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[-1] + 0.5 * dx, y[0] - 0.5 * dx]
    extent_xz = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx, z[-1] + 0.5 * dz, z[0] - 0.5 * dz]
    extent_zy = [z[0] - 0.5 * dz, z[-1] + 0.5 * dz, y[-1] + 0.5 * dx, y[0] - 0.5 * dx]

    zc_pix = np.argmin(np.abs(z))
    yc_pix = np.argmin(np.abs(y))
    xc_pix = np.argmin(np.abs(x))

    # fit function
    imgs_fit = model_psf(nx, dx, z, fit_params, wavelength, ni, sf, model=model)

    #
    not_nan = np.logical_not(np.isnan(imgs))
    vmin_img = np.percentile(imgs[not_nan], 0.1)
    vmax_img = np.percentile(imgs[not_nan], 99.99)

    #
    # if use_same_scale:
    #     vmin_fit = vmin_img
    #     vmax_fit = vmax_img
    # else:
    #     vmin_fit = np.percentile(imgs_fit, 0.1)
    #     vmax_fit = np.percentile(imgs_fit, 99.99)

    figh = plt.figure(figsize=figsize, **kwargs)
    grid = plt.GridSpec(2, 4, wspace=0.5, hspace=0.5)

    strs = "%s, %s, sf=%d\nNA=%0.3f, zc=%0.3fum, yc=%0.3fum, xc=%0.3fum, amp=%0.2f, bg=%0.2f" % \
            (label, model, sf, fit_params[4], fit_params[3], fit_params[2], fit_params[1],
             fit_params[0], fit_params[5])
    plt.suptitle(strs)

    # XY-plane
    ax = plt.subplot(grid[0, 1])
    ax.imshow(imgs[zc_pix], extent=extent_xy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Data")

    ax = plt.subplot(grid[0, 3])
    ax.imshow(imgs_fit[zc_pix], extent=extent_xy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Fit")

    # XZ-plane
    ax = plt.subplot(grid[1, 1])
    ax.imshow(imgs[:, yc_pix], extent=extent_xz, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("x ($\mu m$)")
    ax.set_ylabel("z ($\mu m$)")

    ax = plt.subplot(grid[1, 3])
    ax.imshow(imgs_fit[:, yc_pix], extent=extent_xz, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("x ($\mu m$)")
    ax.set_ylabel("z ($\mu m$)")

    # YZ-plane
    ax = plt.subplot(grid[0, 0])
    ax.imshow(np.transpose(imgs[:, :, xc_pix]), extent=extent_zy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("z ($\mu m$)")
    ax.set_ylabel("y ($\mu m$)")

    ax = plt.subplot(grid[0, 2])
    ax.imshow(np.transpose(imgs_fit[:, :, xc_pix]), extent=extent_zy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("z ($\mu m$)")
    ax.set_ylabel("y ($\mu m$)")

    # XY cuts
    ax = plt.subplot(grid[1, 0])
    ax.plot(np.sqrt(np.expand_dims(x, axis=0)**2 + np.expand_dims(y, axis=1)**2).ravel(), imgs[zc_pix].ravel(), 'g.')
    ax.plot(y.ravel(), imgs[zc_pix, :, xc_pix], 'b.')
    ax.plot(x.ravel(), imgs[zc_pix, yc_pix, :], 'k.')
    ax.plot(y.ravel(), imgs_fit[zc_pix, :, xc_pix], 'b')
    ax.plot(x.ravel(), imgs_fit[zc_pix, yc_pix, :], 'k')
    ax.set_xlabel("xy-position ($\mu m$)")
    ax.set_ylabel("amplitude")
    ax.legend(["all", "y-cut", "x-cut"])

    # z cuts
    ax = plt.subplot(grid[1, 2])
    ax.plot(z.ravel(), imgs[:, yc_pix, xc_pix], 'b.')
    ax.plot(z.ravel(), imgs_fit[:, yc_pix, xc_pix], 'b')
    ax.set_xlabel("z-position ($\mu m$)")
    ax.set_ylabel("amplitude")

    # optional saving
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        figh.savefig(os.path.join(save_dir, "%s.png" % label))
        plt.close(figh)

    return figh


# get real PSF
def get_exp_psf(imgs, coords, centers, roi_size, backgrounds=None):
    """
    Get experimental psf from imgs and the results of autofit_psfs

    :param imgs:z-stack of images
    :param coords: (z, y, x) of full image. Must be broadcastable to full image size
    :param fit_params: n x 6 array, where each element gives [A, cx, cy, cz, NA, bg]
    Use the center positions, background, and amplitude. # todo: can I get rid of the amplitude?
    :param roi_size: must be odd

    :return psf_mean:
    :return psf_sdm:
    """
    z, y, x, = coords
    dz = z[1, 0, 0] - z[0, 0, 0]
    dx = x[0, 0, 1] - x[0, 0, 0]

    # centers = np.stack((fit_params[:, 3], fit_params[:, 2], fit_params[:, 1]), axis=1)

    # set up array to hold psfs
    nrois = len(centers)
    if backgrounds is None:
        backgrounds = np.zeros(nrois)

    psf_shifted = np.zeros((nrois, roi_size[0], roi_size[1], roi_size[2])) * np.nan
    # coordinates
    z_psf, y_psf, x_psf = get_psf_coords(roi_size, [dz, dx, dx])
    zc_pix_psf = np.argmin(np.abs(z_psf))
    yc_pix_psf = np.argmin(np.abs(y_psf))
    xc_pix_psf = np.argmin(np.abs(x_psf))

    # loop over rois and shift psfs so they are centered
    for ii in range(nrois):
        # get closest pixels to center
        xc_pix = np.argmin(np.abs(x - centers[ii, 2]))
        yc_pix = np.argmin(np.abs(y - centers[ii, 1]))
        zc_pix = np.argmin(np.abs(z - centers[ii, 0]))

        # cut roi from image
        roi_unc = tools.get_centered_roi((zc_pix, yc_pix, xc_pix), roi_size)
        roi = tools.get_centered_roi((zc_pix, yc_pix, xc_pix), roi_size, min_vals=[0, 0, 0], max_vals=imgs.shape)
        img_roi = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

        zroi = z[roi[0]:roi[1], :, :]
        yroi = y[:, roi[2]:roi[3], :]
        xroi = x[:, :, roi[4]:roi[5]]

        cx_pix_roi = (roi[5] - roi[4]) // 2
        cy_pix_roi = (roi[3] - roi[2]) // 2
        cz_pix_roi = (roi[1] - roi[0]) // 2

        xshift_pix = (xroi[0, 0, cx_pix_roi] - centers[ii, 2]) / dx
        yshift_pix = (yroi[0, cy_pix_roi, 0] - centers[ii, 1]) / dx
        zshift_pix = (zroi[cz_pix_roi, 0, 0] - centers[ii, 0]) / dz

        # get coordinates
        img_roi_shifted = ndi.shift(np.array(img_roi, dtype=float), [zshift_pix, yshift_pix, xshift_pix], mode="constant", cval=-1)
        img_roi_shifted[img_roi_shifted == -1] = np.nan

        # put into array in appropriate positions
        zstart = zc_pix_psf - cz_pix_roi
        zend = zstart + (roi[1] - roi[0])
        ystart = yc_pix_psf - cy_pix_roi
        yend = ystart + (roi[3] - roi[2])
        xstart = xc_pix_psf - cx_pix_roi
        xend = xstart + (roi[5] - roi[4])

        psf_shifted[ii, zstart:zend, ystart:yend, xstart:xend] = img_roi_shifted - backgrounds[ii]

    with np.errstate(divide='ignore', invalid='ignore'):
        psf_mean = np.nanmean(psf_shifted, axis=0)

    # the above doesn't do a good enough job of normalizing PSF
    max_val = np.nanmax(psf_mean[psf_mean.shape[0]//2])
    psf_mean = psf_mean / max_val

    # get otf
    otf_mean, ks = psf2otf(psf_mean, drs=(dz, dx, dx))
    kz, ky, kx = ks

    return psf_mean, (z_psf, y_psf, x_psf), otf_mean, (kz, ky, kx)


# main fitting function
def autofit_psfs(imgs, psf_roi_size, dx, dz, wavelength, ni=1.5, model='vectorial', sf=1,
                 threshold=100, min_spot_sep=(3, 3),
                 filter_sigma_small=(1, 0.5, 0.5), filter_sigma_large=(3, 5, 5),
                 sigma_bounds=((1, 1), (10, 10)), roi_size_loc=(13, 21, 21), fit_amp_thresh=100,
                 num_localizations_to_plot=5, psf_percentiles=(20, 5), plot=True, gamma=0.5, save_dir=None,
                 figsize=(18, 10), **kwargs):

    """
    Find isolated points, fit PSFs, and report data. This is the main function of this module

    :param imgs: nz x nx x ny image
    :param psf_roi_size: [nz, ny, nx]
    :param float dx: pixel size in um
    :param float dz: z-plane spacing in um
    :param float wavelength: wavelength in um
    :param float ni: index of refraction of medium
    :param str model: "vectorial", "gibson-lanni", "born-wolf", or "gaussian"
    :param int sf: sampling factor for oversampling
    :param float threshold: threshold pixel value to identify peaks. Note: this is applied to the filtered image,
    and is not directly compraable to the values in imgs
    :param min_spot_sep: (sz, sxy) minimum spot separation between different beads, in pixels
    :param filter_sigma_small: (sz, sy, sx) sigmas of Gaussian filter used to smooth image, in pixels
    :param filter_sigma_large: (sz, sy, sx) sigmas of Gaussian filter used to removed background, in pixels
    :param roi_size_loc: (sz, sy, sx) size of ROI to used in localization, in pixels
    :param float fit_amp_thresh: only consider spots which have fit values larger tha this amplitude
    :param **kwargs: passed through to figures

    :return:
    """

    saving = False
    if save_dir is not None:
        saving = True

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # ###################################
    # convert spot-finding parameters from pixels to distnace units
    # ###################################
    min_spot_sep = list(copy.deepcopy(min_spot_sep))
    min_spot_sep[0] = min_spot_sep[0] * dz
    min_spot_sep[1] = min_spot_sep[1] * dx

    filter_sigma_small = list(copy.deepcopy(filter_sigma_small))
    filter_sigma_small[0] = filter_sigma_small[0] * dz
    filter_sigma_small[1] = filter_sigma_small[1] * dx
    filter_sigma_small[2] = filter_sigma_small[2] * dx

    filter_sigma_large = list(copy.deepcopy(filter_sigma_large))
    filter_sigma_large[0] = filter_sigma_large[0] * dz
    filter_sigma_large[1] = filter_sigma_large[1] * dx
    filter_sigma_large[2] = filter_sigma_large[2] * dx

    sigma_bounds = list(copy.deepcopy(sigma_bounds))
    sigma_bounds[0] = list(sigma_bounds[0])
    sigma_bounds[1] = list(sigma_bounds[1])
    sigma_bounds[0][0] = sigma_bounds[0][0] * dz
    sigma_bounds[0][1] = sigma_bounds[0][1] * dx
    sigma_bounds[1][0] = sigma_bounds[1][0] * dz
    sigma_bounds[1][1] = sigma_bounds[1][1] * dx

    roi_size_loc = list(copy.deepcopy(roi_size_loc))
    roi_size_loc[0] = roi_size_loc[0] * dz
    roi_size_loc[1] = roi_size_loc[1] * dx
    roi_size_loc[2] = roi_size_loc[2] * dx

    # ###################################
    # do localization
    # ###################################
    x, y, z = localize.get_coords(imgs.shape, dx, dz)

    coords, fit_params, init_params, rois, to_keep, conditions, condition_names, filter_settings = localize.localize_beads(
        imgs, dx, dz, threshold, roi_size_loc, filter_sigma_small, filter_sigma_large,
        min_spot_sep, sigma_bounds, fit_amp_thresh, fit_dist_max_err=(np.inf, np.inf), dist_boundary_min=(0, 0),
        use_gpu_filter=False)

    # ###################################
    # plot individual localizations
    # ###################################
    ind_to_plot = np.arange(len(to_keep), dtype=np.int)[to_keep][:num_localizations_to_plot]
    results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
        joblib.delayed(localize.plot_gauss_roi)(fit_params[ind], rois[ind], imgs, coords, init_params[ind], figsize=figsize,
                                                prefix="localization_roi_%d" % ind, save_dir=save_dir)
        for ind in ind_to_plot
    )

    # ###################################
    # plot fit statistics
    # ###################################
    if plot:
        figh = plot_fit_stats(fit_params[to_keep], figsize=figsize, **kwargs)

        if saving:
            fname = os.path.join(save_dir, "fit_stats.png")
            figh.savefig(fname)
            plt.close(figh)

    # ###################################
    # get and plot experimental PSFs
    # ###################################
    nps = len(psf_percentiles)
    psfs_real = np.zeros((nps,) + tuple(psf_roi_size))
    otfs_real = np.zeros(psfs_real.shape, dtype=np.complex)
    for ii in range(len(psf_percentiles)):
        # only keep smallest so many percent of spots
        sigma_max = np.percentile(fit_params[:, 4][to_keep],  psf_percentiles[ii])
        to_use = np.logical_and(to_keep, fit_params[:, 4] <= sigma_max)

        # get centers
        centers = np.stack((fit_params[:, 3][to_use],
                            fit_params[:, 2][to_use],
                            fit_params[:, 1][to_use]), axis=1)

        # find experiment psf/otf
        psfs_real[ii], psf_coords, otfs_real[ii], otf_coords = get_exp_psf(imgs, (z, y, x), centers, psf_roi_size,
                                                                           backgrounds=fit_params[:, 5][to_use])

        results, _ = fit_pixelated_psfmodel(psfs_real[ii], dx, dz, wavelength, ni, sf, model=model)
        fit_params_real = results["fit_params"]

        if plot:
            figh = plot_psfmodel_fit(psfs_real[ii], dx, dz, wavelength, ni, sf, fit_params_real, model=model,
                                     gamma=gamma, figsize=figsize, label="smallest %d percent" % psf_percentiles[ii])

            if saving:
                fname = os.path.join(save_dir, "experimental_psf_smallest_%0.2f.png" % psf_percentiles[ii])
                figh.savefig(fname)
                plt.close(figh)

    # ###################################
    # plot localization positions
    # ###################################
    if plot:
        centers = np.stack((fit_params[:, 3][to_keep] / dz,
                            fit_params[:, 2][to_keep] / dx,
                            fit_params[:, 1][to_keep] / dx), axis=1)

        figh = plot_bead_locations(imgs, centers, title="Max intensity projection and NA from 2D fit versus position",
                            weights=fit_params[:, 4], cbar_labels=["sigma"], figsize=figsize, **kwargs)

        if saving:
            fname = os.path.join(save_dir, "sigma_versus_position.png")
            figh.savefig(fname)
            plt.close(figh)

    # todo: return all data

    return None

# other display functions
def plot_fit_stats(fit_params, figsize=(18, 10), **kwargs):
    """
    Plot statistics for a list of fit result dictionaries

    :param fit_params: N x 6 list of localization fit parameters
    :param kwargs: passed to plt.figure()
    :return figh: figure handle
    """

    # fit parameter summary
    figh = plt.figure(figsize=figsize, **kwargs)
    plt.suptitle("Localization fit parameter summary")
    grid = plt.GridSpec(2, 2, hspace=1, wspace=0.5)

    # amplitude vs sxy
    ax = plt.subplot(grid[0, 0])
    ax.plot(fit_params[:, 4], fit_params[:, 0], '.')
    ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax.set_ylabel("amp")

    # sxy vs sz
    ax = plt.subplot(grid[0, 1])
    ax.plot(fit_params[:, 4], fit_params[:, 5], '.')
    ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax.set_ylabel(r"$\sigma_{z}$ ($\mu m$)")

    # sxy vs bg
    ax = plt.subplot(grid[1, 1])
    ax.plot(fit_params[:, 4], fit_params[:, 6], '.')
    ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax.set_ylabel(r"$\sigma_{z}$ ($\mu m$)")

    return figh


def plot_bead_locations(imgs, center_lists, title="", color_lists=None, legend_labels=None, weights=None,
                        cbar_labels=None, vlims_percentile=(0.01, 99.99), **kwargs):
    """
    Plot center locations on 2D image or max projection of 3D image

    # todo: replace some of the more complicated plotting functions with this one ... maybe add another argument
    # weights which gives intensity of each color

    :param imgs: np.array either 3D or 2D. Dimensions order Z, Y, X
    :param center_lists: [center_array_1, center_array_2, ...] where each center_array is a numpy array of size N_i x 3
    consisting of triples of center values giving cz, cy, cx
    :param color_lists: list of colors for each series to be plotted in
    :param legend_labels: labels for each series
    :param weights: list of arrays [w_1, ..., w_n], with w_i the same size as center_array_i, giving the intensity of
    the color to be plotted
    :return:
    """

    if not isinstance(center_lists, list):
        center_lists = [center_lists]
    nlists = len(center_lists)

    if color_lists is None:
        cmap = plt.cm.get_cmap('hsv')
        color_lists = []
        for ii in range(nlists):
            color_lists.append(cmap(ii / nlists))

    if legend_labels is None:
        legend_labels = list(map(lambda x: "series #" + str(x) + " %d pts" % len(center_lists[x]), range(nlists)))

    if weights is None:
        weights = [np.ones(len(cs)) for cs in center_lists]

    if cbar_labels is None:
        cbar_labels = ["" for cs in center_lists]

    if imgs.ndim == 3:
        img_max_proj = np.nanmax(imgs, axis=0)
    else:
        img_max_proj = imgs

    figh = plt.figure(**kwargs)
    plt.suptitle(title)

    # plot image
    vmin = np.percentile(img_max_proj, vlims_percentile[0])
    vmax = np.percentile(img_max_proj, vlims_percentile[1])

    plt.imshow(img_max_proj, vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap("bone"))

    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Image intensity (counts)")

    # plot centers
    for ii in range(nlists):
        cmap_color = LinearSegmentedColormap.from_list("test", [[0., 0., 0.], color_lists[ii]])
        plt.scatter(center_lists[ii][:, 2], center_lists[ii][:, 1], facecolor='none', marker='o',
                    edgecolor=cmap_color(weights[ii] / np.nanmax(weights[ii])))

        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=np.nanmax(weights[ii])), cmap=cmap_color))
        cbar.ax.set_ylabel(cbar_labels[ii])

    plt.legend(legend_labels)

    return figh
