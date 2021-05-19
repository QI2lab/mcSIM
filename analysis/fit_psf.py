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
import fit
import analysis_tools as tools

# model OTF function
def symm1d_to_2d(arr, fs, fmax, npts):
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

def otf2psf(otf, dfx=1, dfy=1):
    """

    :param dfx:
    :param dfy:
    :param otf:
    :return:
    """
    ny, nx = otf.shape
    psf = fft.fftshift(fft.ifft2(fft.ifftshift(otf))).real
    dx = 1 / (dfx * nx)
    dy = 1 / (dfy * ny)
    xs = tools.get_fft_pos(nx, dt=dx)
    ys = tools.get_fft_pos(ny, dt=dy)

    return psf, xs, ys

def psf2otf(psf, dx=1, dy=1):
    ny, nx = psf.shape
    otf = fft.fftshift(fft.fft2(fft.ifftshift(psf)))
    fxs = tools.get_fft_pos(nx, dt=dx)
    fys = tools.get_fft_pos(ny, dt=dy)

    return otf, fxs, fys

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
    Get incoherent transfer function from autocorrelation of coherent transfer functino
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

    :param na: numerical aperture
    :param wavelength:
    :return fwhm: in same units as wavelength
    """
    return 0.5 * wavelength / na

def na2sigma(na, wavelength):
    """

    :param na:
    :param wavelength:
    :return:
    """
    #2 * sqrt{2*log(2)} * sigma = 0.5 * wavelength / NA
    sigma = na2fwhm(na, wavelength) / (2*np.sqrt(2 * np.log(2)))
    return sigma

def fwhm2na(wavelength, fwhm):
    return 0.5 * wavelength / fwhm

def sigma2na(wavelength, sigma):
    fwhm = sigma * (2*np.sqrt(2 * np.log(2)))
    return fwhm2na(wavelength, fwhm)

# different PSF model functions
def gaussian2d_psf(x, y, p, wavelength):
    """
    2D Gaussian approximation to airy function. Matches well for equal peak intensity, but then area will not match.
    :param x:
    :param y:
    :param p: [A, cx, cy, NA, bg]
    :param wavelength:
    :return:
    """
    # ni does not matter for 2D gaussian, so may as well set to 1
    return gaussian3d_psf(x, y, np.zeros(x.shape), [p[0], p[1], p[2], 0, p[3], p[4]], wavelength, 1.)

def gaussian3d_psf(x, y, z, p, wavelength, ni):
    """
    Gaussian approximation to PSF. Matches well for equal peak intensity, but some deviations in area.
    See https://doi.org/10.1364/AO.46.001819 for more details.

    sigma_xy = 0.22 * lambda / NA.
    This comes from equating the FWHM of the Gaussian and the airy function.
    FWHM = 2 * sqrt{2*log(2)} * sigma = 0.5 * wavelength / NA

    :param x: x-coordinate in um
    :param y: in um
    :param z: in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param wavelength: wavelength in um
    :param ni: refractive index
    :return:
    """

    # factor = 0.5 / (2 * np.sqrt(2 * np.log(2)))
    sigma_xy = 0.22 * wavelength / p[4]
    sigma_z = np.sqrt(6) / np.pi * ni * wavelength / p[4] ** 2
    val = p[0] * np.exp(- (x - p[1])**2 / (2 * sigma_xy**2) -
                          (y - p[2])**2 / (2 * sigma_xy**2) -
                          (z - p[3])**2 / (2 * sigma_z**2)) + p[5]

    return val

def gaussian3d_psf_jac(x, y, z, p, wavelength, ni):
    """
    Jacobean of gaussian3d_psf
    """
    sigma_xy = 0.22 * wavelength / p[4]
    sigma_z = np.sqrt(6) / np.pi * ni * wavelength / p[4] ** 2

    exp = np.exp(-(x - p[1])**2 / (2 * sigma_xy**2) -
                (y - p[2])**2 / (2 * sigma_xy**2) -
                (z - p[3])**2 / (2 * sigma_z**2))

    bcast_shape = (x + y + z).shape

    jac = [exp,
           p[0] * exp * 2 * (x - p[1]) / (2 * sigma_xy**2),
           p[0] * exp * 2 * (y - p[2]) / (2 * sigma_xy**2),
           p[0] * exp * 2 * (z - p[3]) / (2 * sigma_z**2),
           p[0] * exp * (-(x - p[1])**2 / sigma_xy**2 / p[4] +
                         -(y - p[2])**2 / sigma_xy**2 / p[4] +
                         -(z - p[3])**2 / sigma_z**2 * 2 / p[4]),
           np.ones(bcast_shape)]

    return jac

def gaussian3d_pixelated_psf(nx, dx, z, p, wavelength, ni, sf=3):
    """
    Gaussian function, accounting for image pixelation in the xy plane. This function mimics the style of the
    PSFmodels functions.

    :param dx: pixel size in um
    :param nx: number of pixels (must be odd)
    :param z: coordinates of z-planes to evaluate function at
    :param p: [A, cx, cy, cz, NA, bg]
    :param wavelength: in um
    :param ni: refractive index
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :return:
    """

    if np.mod(nx, 2) == 0:
        raise ValueError("number of xy pixels must be odd, but was %d." % nx)

    if isinstance(z, (int, float, list)):
        z = np.array([z])

    if not isinstance(sf, int):
        raise ValueError("sf must be an integer")

    # we will add amplitude and background back in at the end
    pt = [1, p[1], p[2], p[3], p[4], 0]

    y, x, = get_psf_coords([nx, nx], [dx, dx])
    y = np.expand_dims(y, axis=(0, 2))
    x = np.expand_dims(x, axis=(0, 1))
    z = np.expand_dims(np.array(z, copy=True), axis=(1, 2))

    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dx, sf=sf)

    psf = np.mean(gaussian3d_psf(xx_s, yy_s, zz_s, pt, wavelength, ni), axis=-1)

    # get psf norm
    xx_sc, yy_sc, zz_sc = oversample_pixel(np.array([0]), np.array([0]), np.array([0]), dx, sf=sf)
    psf_norm = np.mean(gaussian3d_psf(xx_sc, yy_sc, zz_sc, [1, 0, 0, 0, p[4], 0], wavelength, ni), axis=-1)

    # psf normalized to one
    psf = p[0] * psf / psf_norm + p[5]

    return psf

def gaussian3d_pixelated_psf_jac(nx, dx, z, p, wavelength, ni, sf=3):
    if np.mod(nx, 2) == 0:
        raise ValueError("number of xy pixels must be odd, but was %d." % nx)

    if isinstance(z, (int, float, list)):
        z = np.array([z])

    if not isinstance(sf, int):
        raise ValueError("sf must be an integer")

    # we will need this value later
    psf = gaussian3d_pixelated_psf(nx, dx, z, p, wavelength, ni, sf)

    # we will add amplitude and background back in at the end
    pt = [1, p[1], p[2], p[3], p[4], 0]

    y, x, = get_psf_coords([nx, nx], [dx, dx])
    y = np.expand_dims(y, axis=(0, 2))
    x = np.expand_dims(x, axis=(0, 1))
    z = np.expand_dims(np.array(z, copy=True), axis=(1, 2))

    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dx, sf=sf)

    jac = gaussian3d_psf_jac(xx_s, yy_s, zz_s, pt, wavelength, ni)
    jac = [np.mean(j, axis=-1) for j in jac]

    # get psf norm
    xx_sc, yy_sc, zz_sc = oversample_pixel(np.array([0]), np.array([0]), np.array([0]), dx, sf=sf)
    psf_norm = np.squeeze(np.mean(gaussian3d_psf(xx_sc, yy_sc, zz_sc, [1, 0, 0, 0, p[4], 0], wavelength, ni), axis=-1))
    psf_norm_jac = gaussian3d_psf_jac(xx_sc, yy_sc, zz_sc, [1, 0, 0, 0, p[4], 0], wavelength, ni)
    psf_norm_jac = np.squeeze(np.mean(psf_norm_jac[4], axis=-1))

    # modify jacobean for psf normalization, which is: psf = p[0] * psf / psf_norm + p[5]
    jac[0] = jac[0] / psf_norm
    jac[1] = jac[1] * p[0] / psf_norm
    jac[2] = jac[2] * p[0] / psf_norm
    jac[3] = jac[3] * p[0] / psf_norm
    jac[4] = jac[4] * p[0] / psf_norm - (psf - p[5]) * psf_norm_jac / psf_norm

    return jac

def airy_fn(x, y, p, wavelength):
    """
    i.e. Born-Wolf in focus intensity PSF. Helper function for born_wolf_psf
    :param x:
    :param y:
    :param p: [amplitude, cx, cy, NA, bg]
    :param wavelength: in um
    :return:
    """

    # construct radial coordinates
    rho = np.sqrt((x - p[1]) ** 2 + (y - p[2]) ** 2)
    k = 2 * np.pi / wavelength
    # evaluate airy function
    val = np.square(np.abs(np.divide(sp.j1(rho * k * p[3]), rho * k * p[3])))
    # normalize by value at zero
    with np.errstate(divide='ignore', invalid='ignore'):
        val_zero = np.square(np.abs(np.divide(sp.j1(1e-3 * 2*np.pi * p[3]), 1e-3 * 2*np.pi * p[3])))

    # add amplitude and offset
    val = p[0] * val / val_zero + p[4]
    # handle case where argument is zero
    val[rho == 0] = p[0] + p[4]

    return val

def born_wolf_axial_fn(z, p, wavelength, ni):
    """
    Axial profile for (x,y) = (0, 0) Born-Wolf model.

    :param z:
    :param p: [A, cx, cy, cz, NA, bg]. cx and cy have no effect on this function
    :param wavelength:
    :param ni: index of refraction in object space
    :return:
    """
    na = p[4]
    cz = p[3]
    k = 2*np.pi / wavelength
    val = 4 * (2 * ni**2) / (k**2 * na**4 * (z - cz)**2) * (1 - np.cos(0.5 * k * (z - cz) * na**2 / ni))
    # correct singularity at zero
    val[z - cz == 0] = 1

    # add on amplitude/offset
    val = p[0] * val + p[5]

    return val

def born_wolf_psf(x, y, z, p, wavelength, ni):
    """
    Born-wolf PSF function from numerical integration.
    :param x: in um
    :param y: in um
    :param z: in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param wavelength: in um
    :param ni: index of refraction
    :return:

    # todo: one thing we are ignoring is the finite pixel size. Could imagine integrating each pixel, instead of determining value at center
    # todo: tried using hankel transform from hankel PyPI package, but this turns out to be even slower than direct integration...
    """
    k = 2 * np.pi / wavelength
    rr = np.sqrt((x - p[1]) ** 2 + (y - p[2]) ** 2)

    psfs = np.zeros(rr.shape)

    # in focus portion
    is_in_focus = (z - p[3]) == 0
    psfs[is_in_focus] = airy_fn(x[is_in_focus], y[is_in_focus], [p[0], p[1], p[2], p[4], p[5]], wavelength)

    # out of focus portion
    if not np.all(is_in_focus):
        integrand = lambda rho, r, z: rho * sp.j0(k * r * p[4] * rho) * \
                                      np.exp(-1j * k * (z - p[3]) * p[4]**2 * rho**2 / (2 * ni))

        # like this approach because allows rr, z, etc. to have arbitrary dimension
        for ii, (r, zc, ifocus) in enumerate(zip(rr.ravel(), z.ravel(), is_in_focus.ravel())):
            if ifocus:
                continue
            int_real = scipy.integrate.quad(lambda rho: integrand(rho, r, zc).real, 0, 1)[0]
            int_img = scipy.integrate.quad(lambda rho: integrand(rho, r, zc).imag, 0, 1)[0]

            coords = np.unravel_index(ii, rr.shape)
            psfs[coords] = np.square(np.abs(int_real + 1j * int_img))
            # todo: change to this. But want to test...
            # psfs[coords] = int_real ** 2 + int_img **2

        # normalize so height above background is p[0]
        # int_zero_real = scipy.integrate.quad(lambda rho: integrand(rho, 0, 0).real, 0, 1)[0]
        # int_zero_img = scipy.integrate.quad(lambda rho: integrand(rho, 0, 0).imag, 0, 1)[0]
        # psf_zero = np.square(np.abs(int_zero_real + 1j * int_zero_img))
        psf_zero = 0.25

        #vals = p[0] * psfs / psf_zero + p[5]
        psfs[np.logical_not(is_in_focus)] = p[0] * psfs[np.logical_not(is_in_focus)] / psf_zero + p[5]

    return psfs

def model_psf(nx, dxy, z, p, wavelength, ni, sf=1, model='vectorial', **kwargs):
    """
    Wrapper function for evaluating different psfmodels. For vectorial or gibson-lanni PSF's, this wraps the functions
    in PSFmodels. For gaussian, it wraps the gaussian3d_pixelated_psf() function.

    todo: need to implement index of refraction?

    :param nx: number of points to be sampled in x- and y-directions
    :param dxy: pixel size in um
    :param z: z positions in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param wavelength: wavelength in um
    :param model: 'gaussian', 'gibson-lanni', or 'vectorial'
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
    elif model == 'gaussian':
        val = gaussian3d_pixelated_psf(nx, dxy, z, p, wavelength, ni, sf)
    else:
        raise ValueError("model must be 'gibson-lanni', 'vectorial', or 'gaussian' but was '%s'" % model)

    return val

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

    return xx_s, yy_s, zz_s

# fitting functions
def fit_cont_psfmodel(img, wavelength, ni, model='gaussian',
                      init_params=None, fixed_params=None, sd=None, xx=None, yy=None, zz=None, bounds=None):
    """
    Fit PSF functions not accounting for pixelation. i.e. the value of these functions is their value at the center
    of a given pixel.

    :param img: img to be fit
    :param wavelength: wavelength in um
    :param init_params: [A, cx, cy, cz, NA, bg]. If not set, will guess reasonable values. Can also use a mixture of
    externally provided values and default guesses. If any elements of inti_params are None, then will use default values.
    :param fixed_params: list of boolean values, same size as init_params.
    :param sd: img uncertainties, same size as img
    :param xx: x coordinates of image, same size as img
    :param yy: y coordinates
    :param zz: z coordinates
    :param bounds: tuple of tuples. First tuple is lower bounds, second is upper bounds
    :param ni: index of refraction
    :param model: 'gaussian' for 'born-wolf'

    :return results: dictionary of fit results
    :return fit_fn: function returning PSF for arguments x, y, z
    """

    # get default coordinates
    if xx is None or yy is None or zz is None:
        nz, ny, nx = img.shape
        yy, zz, xx = np.meshgrid(range(ny), range(nz), range(nx))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 6
    else:
        init_params = copy.deepcopy(init_params)

    # use default values for any params that are None
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(img))
        bg = np.mean(img[to_use].ravel())
        A = np.max(img[to_use].ravel()) - bg
        cz, cy, cx = fit.get_moments(img, order=1, coords=[zz[:, 0, 0], yy[0, :, 0], xx[0, 0, :]])

        # iz = np.argmin(np.abs(zz[:, 0, 0] - cz))
        # m2y, m2x = tools.get_moments(img, order=2, coords=[yy[0, :, 0], xx[0, 0, :]])
        # sx = np.sqrt(m2x - cx ** 2)
        # sy = np.sqrt(m2y - cy ** 2)
        # # from gaussian approximation
        # na_guess = 0.22 * wavelength / np.sqrt(sx * sy)
        na_guess = 1

        ip_default = [A, cx, cy, cz, na_guess, bg]

        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    # get default bounds
    if bounds is None:
        bounds = ((0, xx.min(), yy.min(), zz.min(), 0, -np.inf),
                  (np.inf, xx.max(), yy.max(), zz.max(), ni, np.inf))

    # select fitting model and do fitting
    if model == 'gaussian':
        model_fn = lambda x, y, z, p: gaussian3d_psf(x, y, z, p, wavelength, ni)
        jac_fn = lambda x, y, z, p: gaussian3d_psf_jac(x, y, z, p, wavelength, ni)

        result = fit.fit_model(img, lambda p: model_fn(xx, yy, zz, p), init_params,
                               fixed_params=fixed_params, sd=sd, bounds=bounds, model_jacobian=jac_fn)

    elif model == 'born-wolf':
        model_fn = lambda x, y, z, p: born_wolf_psf(x, y, z, p, wavelength, ni)

        result = fit.fit_model(img, lambda p: model_fn(xx, yy, zz, p), init_params,
                               fixed_params=fixed_params, sd=sd, bounds=bounds)
    else:
        raise ValueError("model must be 'gaussian' or 'born-wolf' but was '%s'" % model)

    # model function at fit parameters
    pfit = result['fit_params']
    fit_fn = lambda x, y, z: model_fn(x, y, z, pfit)

    return result, fit_fn

def fit_pixelated_psfmodel(img, dxy, dz, wavelength, ni, sf=1, model='vectorial',
                           init_params=None, fixed_params=None, sd=None, bounds=None):
    """
    3D non-linear least squares fit using one of the point spread function models from psfmodels package.

    The x/y coordinates are assumed to match the convention of get_coords(), i.e. they are (arange(nx) / nx//2) * d

    # todo: make sure ni implemented correctly. if want to use different ni, have to be careful because this will shift the focus position away from z=0
    # todo: make sure oversampling (sf) works correctly with all functions

    :param img: Nz x Ny x Nx image stack
    :param dxy: dx and dy in microns
    :param dz: dz in microns
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
    if model == "gaussian":
        model_fn = lambda z, nx, dxy, p: model_psf(nx, dxy, z, p, wavelength, ni, sf=sf, model=model)
        result = fit.fit_model(img, lambda p: model_fn(z, nx, dxy, p), init_params,
                               fixed_params=fixed_params, sd=sd, bounds=bounds,
                               model_jacobian= lambda p: gaussian3d_pixelated_psf_jac(nx, dxy, z, p, wavelength, ni, sf=sf))
        # result = fit.fit_model(img, lambda p: model_fn(z, nx, dxy, p), init_params,
        #                        fixed_params=fixed_params, sd=sd, bounds=bounds, jac='3-point', x_scale='jac')

    elif model == 'vectorial' or model =='gibson-lanni':
        model_fn = lambda z, nx, dxy, p: model_psf(nx, dxy, z, p, wavelength, ni, sf=sf, model=model)

        result = fit.fit_model(img, lambda p: model_fn(z, nx, dxy, p), init_params,
                               fixed_params=fixed_params, sd=sd, bounds=bounds, jac='3-point', x_scale='jac')
    else:
        raise ValueError("Model should be 'vectorial', 'gibson-lanni', or 'gaussian', but was '%s'" % model)


    # model function at fit parameters
    fit_fn = lambda z, nx, dxy: model_fn(z, nx, dxy, result['fit_params'])

    return result, fit_fn

# localize using radial symmetry
def localize2d(img, mode="radial-symmetry"):
    """
    Perform 2D localization using the radial symmetry approach of https://doi.org/10.1038/nmeth.2071

    :param img: 2D image of size ny x nx
    :param mode: 'radial-symmetry' or 'centroid'

    :return xc:
    :return yc:
    """
    if img.ndim != 2:
        raise ValueError("img must be a 2D array, but was %dD" % img.ndim)

    ny, nx = img.shape
    x = np.arange(nx)
    y = np.arange(ny)

    if mode == "centroid":
        xc = np.sum(img * x[None, :]) / np.sum(img)
        yc = np.sum(img * y[:, None]) / np.sum(img)
    elif mode == "radial-symmetry":
        # gradients taken at point between four pixels, i.e. (xk, yk) = (j + 0.5, i + 0.5)
        # using the Roberts cross operator
        yk = 0.5 * (y[:-1] + y[1:])
        xk = 0.5 * (x[:-1] + x[1:])
        # gradients along 45 degree rotated directions
        grad_uk = img[1:, 1:] - img[:-1, :-1]
        grad_vk = img[1:, :-1] - img[:-1, 1:]
        grad_xk = 1 / np.sqrt(2) * (grad_uk - grad_vk)
        grad_yk = 1 / np.sqrt(2) * (grad_uk + grad_vk)
        with np.errstate(invalid="ignore", divide="ignore"):
            # slope of the gradient at this point
            mk = grad_yk / grad_xk
            mk[np.isnan(mk)] = np.inf

            # compute weights by (1) increasing weight where gradient is large and (2) decreasing weight for points far away
            # from the centroid (as small slope errors can become large as the line is extended to the centroi)
            # approximate distance between (xk, yk) and (xc, yc) by assuming (xc, yc) is centroid of the gradient
            grad_norm = np.sqrt(grad_xk**2 + grad_yk**2)
            centroid_grad_norm_x = np.sum(xk[None, :] * grad_norm) / np.sum(grad_norm)
            centroid_grad_norm_y = np.sum(yk[:, None] * grad_norm) / np.sum(grad_norm)
            dk_centroid = np.sqrt((yk[:, None] - centroid_grad_norm_y)**2 + (xk[None, :] - centroid_grad_norm_x)**2)
            # weights
            wk = grad_norm**2 / dk_centroid


            # def chi_sqr(xc, yc):
            #     val = ((yk[:, None] - yc) - mk * (xk[None, :] - xc))**2 / (mk**2 + 1) * wk
            #     val[np.isinf(mk)] = (np.tile(xk[None, :], [yk.size, 1])[np.isinf(mk)] - xc)**2
            #     return np.sum(val)

            # line passing through through (xk, yk) with slope mk is y = yk + mk*(x - xk)
            # minimimum distance of points (xc, yc) is dk**2 = [(yk - yc) - mk*(xk -xc)]**2 / (mk**2 + 1)
            # must handle the case mk -> infinity separately. In this case dk**2 -> (xk - xc)**2
            # minimize chi^2 = \sum_k dk**2 * wk
            # minimizing for xc, yc gives a matrix equation
            # [[A, B], [C, D]] * [[xc], [yc]] = [[E], [F]]
            # in case the slope is infinite, need to take the limit of the sum manually
            summand_a = -mk ** 2 * wk / (mk ** 2 + 1)
            summand_a[np.isinf(mk)] = wk[np.isinf(mk)]
            A = np.sum(summand_a)

            summand_b = mk * wk / (mk**2 + 1)
            summand_b[np.isinf(mk)] = 0
            B = np.sum(summand_b)
            C = -B

            D = np.sum(wk / (mk**2 + 1))

            summand_e = (mk * wk * (yk[:, None] - mk * xk[None, :])) / (mk**2 + 1)
            summand_e[np.isinf(mk)] = - (wk * xk[None, :])[np.isinf(mk)]
            E = np.sum(summand_e)

            summand_f = (yk[:, None] - mk * xk[None, :]) * wk / (mk**2 + 1)
            summand_f[np.isinf(mk)] = 0
            F = np.sum(summand_f)


            xc = (D * E - B * F) / (A*D - B*C)
            yc = (-C * E + A * F) / (A*D - B*C)
    else:
        raise ValueError("mode must be 'centroid' or 'radial-symmetry', but was '%s'" % mode)

    return xc, yc

def localize3d(img, mode="radial-symmetry"):
    """
    Perform 3D localization using an extension of the radial symmetry approach of https://doi.org/10.1038/nmeth.2071

    :param img: 3D image of size nz x ny x nx
    :param mode: 'radial-symmetry' or 'centroid'

    :return xc:
    :return yc:
    :return zc:
    """
    if img.ndim != 3:
        raise ValueError("img must be a 3D array, but was %dD" % img.ndim)

    nz, ny, nx = img.shape
    x = np.arange(nx)[None, None, :]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[:, None, None]

    if mode == "centroid":
        xc = np.sum(img * x) / np.sum(img)
        yc = np.sum(img * y) / np.sum(img)
        zc = np.sum(img * z) / np.sum(img)
    elif mode == "radial-symmetry":
        yk = 0.5 * (y[:, :-1, :] + y[:, 1:, :])
        xk = 0.5 * (x[:, :, :-1] + x[:, :, 1:])
        zk = 0.5 * (z[:-1] + z[1:])
        coords = (zk, yk, xk)

        # take a cube of 8 voxels, and compute gradients at the center, using the four pixel diagonals that pass
        # through the center
        grad_n1 = img[1:, 1:, 1:] - img[:-1, :-1, :-1]
        n1 = np.array([1, 1, 1]) / np.sqrt(3) # vectors go [nz, ny, nx]
        grad_n2 = img[1:, :-1, 1:] - img[:-1, 1:, :-1]
        n2 = np.array([1, -1, 1]) / np.sqrt(3)
        grad_n3 = img[1:, :-1, :-1] - img[:-1, 1:, 1:]
        n3 = np.array([1, -1, -1]) / np.sqrt(3)
        grad_n4 = img[1:, 1:, :-1] - img[:-1, :-1, 1:]
        n4 = np.array([1, 1, -1]) / np.sqrt(3)

        # compute the gradient xyz components
        # 3 unknowns and 4 eqns, so use pseudo-inverse to optimize overdetermined system
        mat = np.concatenate((n1[None, :], n2[None, :], n3[None, :], n4[None, :]), axis=0)
        gradk = np.linalg.pinv(mat).dot(
            np.concatenate((grad_n1.ravel()[None, :], grad_n2.ravel()[None, :],
                            grad_n3.ravel()[None, :], grad_n4.ravel()[None, :]), axis=0))
        gradk = np.reshape(gradk, [3, zk.size, yk.size, xk.size])

        # compute weights by (1) increasing weight where gradient is large and (2) decreasing weight for points far away
        # from the centroid (as small slope errors can become large as the line is extended to the centroi)
        # approximate distance between (xk, yk) and (xc, yc) by assuming (xc, yc) is centroid of the gradient
        grad_norm = np.sqrt(np.sum(gradk**2, axis=0))
        centroid_gns = np.array([np.sum(zk * grad_norm), np.sum(yk * grad_norm), np.sum(xk * grad_norm)]) / np.sum(grad_norm)
        dk_centroid = np.sqrt((zk - centroid_gns[0]) ** 2 + (yk - centroid_gns[1]) ** 2 + (xk - centroid_gns[2]) ** 2)
        # weights
        wk = grad_norm ** 2 / dk_centroid

        # in 3D, parameterize a line passing through point Po along normal n by
        # V(t) = Pk + n * t
        # distance between line and point Pc minimized at
        # tmin = -\sum_{i=1}^3 (Pk_i - Pc_i) / \sum_i n_i^2
        # dk^2 = \sum_k \sum_i (Pk + n * tmin - Pc)^2
        # again, we want to minimize the quantity
        # chi^2 = \sum_k dk^2 * wk
        # so we take the derivatives of chi^2 with respect to Pc_x, Pc_y, and Pc_z, which gives a system of linear
        # equations, which we can recast into a matrix equation
        # np.array([[A, B, C], [D, E, F], [G, H, I]]) * np.array([[Pc_z], [Pc_y], [Pc_x]]) = np.array([[J], [K], [L]])
        nk = gradk / np.linalg.norm(gradk, axis=0)

        # def chi_sqr(xc, yc, zc):
        #     cs = (zc, yc, xc)
        #     chi = 0
        #     for ii in range(3):
        #         chi += np.sum((coords[ii] + nk[ii] * (cs[jj] - coords[jj]) - cs[ii]) ** 2 * wk)
        #     return chi

        # build 3x3 matrix from above
        mat = np.zeros((3, 3))
        for ll in range(3): # rows of matrix
            for ii in range(3): # columns of matrix
                if ii == ll:
                    mat[ll, ii] += np.sum(-wk * (nk[ii] * nk[ll] - 1))
                else:
                    mat[ll, ii] += np.sum(-wk * nk[ii] * nk[ll])

                for jj in range(3): # internal sum
                    if jj == ll:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * (nk[jj] * nk[ll] - 1))
                    else:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * nk[jj] * nk[ll])

        # build vector from above
        vec = np.zeros((3, 1))
        coord_sum = zk * nk[0] + yk * nk[1] + xk * nk[2]
        for ll in range(3): # sum over J, K, L
            for ii in range(3): # internal sum
                if ii == ll:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * (nk[ii] * nk[ll] - 1) * wk)
                else:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * nk[ii] * nk[ll] * wk)

        # invert matrix
        zc, yc, xc = np.linalg.inv(mat).dot(vec)
    else:
        raise ValueError("mode must be 'centroid' or 'radial-symmetry', but was '%s'" % mode)

    return xc, yc, zc

# utility functions
def get_psf_coords(ns, drs):
    """
    Get centered coordinates for PSFmodels style PSF's from step size and number of coordinates
    :param ns: list of number of points
    :param drs: list of step sizes
    :return coords: list of coordinates [[c1, c2, c3, ...], ...]
    """
    return [d * (np.arange(n) - n // 2) for n, d in zip(ns, drs)]

def min_dist(pt, pts):
    """
    minimum distance between roi_size point and roi_size collection of points
    :param pt: 1 x 3 array
    :param pts: N x 3 array
    :return min_dist:
    """
    dists = np.sqrt(np.sum(np.square(pts - pt), axis=1))
    dists = dists[dists > 0]
    return dists.min()

def get_strehl_ratio(design_na, fit_na, wavelength, model='born-wolf'):
    """
    Calculate theoretical strehl ratio using NA.
    :param design_na:
    :param fit_na:
    :param wavelength:
    :param model:
    :return:
    """
    nx = 301
    dx = 0.01

    # TODO: not sure this actually makes sense for gibson-lanni and vectorial models?
    if model == 'gibson-lanni':
        psf_design = psfm.scalar_psf(0, nx, dx, wvl=wavelength, params={'NA': design_na}, normalize=1)
        psf_fit = psfm.scalar_psf(0, nx, dx, wvl=wavelength, params={'NA': fit_na}, normalize=1)
    elif model == 'vectorial':
        psf_design = psfm.vectorial_psf(0, nx, dx, wvl=wavelength, params={'NA': design_na}, normalize=1)
        psf_fit = psfm.vectorial_psf(0, nx, dx, wvl=wavelength, params={'NA': fit_na}, normalize=1)
    elif model == 'born-wolf':
        #x = dx * (np.arange(nx) - nx//2)
        x = get_psf_coords(nx, dx)
        xx, yy = np.meshgrid(x, x)
        psf_design = airy_fn(xx, yy, [1, 0, 0, design_na, 0], wavelength=wavelength)[None, :, :]
        psf_fit = airy_fn(xx, yy, [1, 0, 0, fit_na, 0], wavelength=wavelength)[None, :, :]
    else:
        raise ValueError("model must be 'born-wolf', 'gibson-lanni', or 'vectorial', but was %s" % model)

    psf_design_area = dx ** 2 * np.sum(np.sum(psf_design[0]))
    psf_fit_area = dx ** 2 * np.sum(np.sum(psf_fit[0]))
    # this identity holds if PSF's are normalized to have peak 1
    strehl_ratio = psf_design_area / psf_fit_area

    return strehl_ratio

# plotting functions
def plot_psf3d(imgs, dx, dz, wavelength, ni, fits,
               imgs_unc=None, model='vectorial', sfs=None, figsize=(20, 10),
               save_dir=None, label='', close_after_saving=True, **kwargs):
    """
    Plot data and fit obtained from fit_psfmodel().

    Multiple different fits can be plotted if fit_params, chi_sqrs, cov, and model are provided as lists.

    :param ni:
    :param imgs: 3D image stack
    :param dx: pixel size in um
    :param dz: space between z-planes in um
    :param fits: list of fit dictionary objects to be plotted.
    :param imgs_unc: image uncertainties (same size as 3D image stack). May also be None, in which case not used
    :param model: 'vectorial', 'gibson-lanni', 'born-wolf', or 'gaussian'
    :param figsize:
    :param save_dir: if not None, then a png of figure will be saved in the provided directory
    :param label: label to add to the start of the file name, if saving
    :param kwargs: additional keyword arguments are passed to plt.figure()
    :return:
    """

    if not isinstance(fits, list):
        fits = [fits]
    nfits = len(fits)

    if sfs is None:
        sfs = nfits * [1]

    if not isinstance(model, list):
        model = [model]

    # unpack fit parameters
    fit_params = np.asarray([f['fit_params'] for f in fits])
    chi_sqr = np.asarray([f['chi_squared'] for f in fits]).transpose()
    cov = np.asarray([f['covariance'] for f in fits])

    # get coordinates
    nz, ny, nx = imgs.shape
    z, y, x, = get_psf_coords([nz, ny, nx], [dz, dx, dx])

    vmin = np.min([fp[5] - 0.1 * fp[0] for fp in fit_params])
    vmax = np.max([1.4 * fp[0] + vmin for fp in fit_params])

    cx3d = [fp[1] for fp in fit_params]
    cy3d = [fp[2] for fp in fit_params]
    # pixel centers based on first fit
    cx_pix3d = np.argmin(np.abs(cx3d[0] - x))
    cy_pix3d = np.argmin(np.abs(cy3d[0] - y))

    fit_img3d = []
    fit_cut = []
    for fp, m, cy, sf in zip(fit_params, model, cy3d, sfs):
        cy_pix_interp = np.argmin(np.abs(cy - x))

        if m == 'gibson-lanni' or m == 'vectorial' or m == 'gaussian':
            v = model_psf(nx, dx, z, fp, wavelength, ni, sf=sf, model=m)
            fit_img3d.append(v)
            fit_cut.append(np.squeeze(v[:, cy_pix_interp, :]))
        elif m == 'born-wolf':
            yy, zz, xx = np.meshgrid(y, z, x)
            fit_img3d.append(born_wolf_psf(xx, yy, zz, fp, wavelength, ni))
            xxi, zzi = np.meshgrid(x, z)
            fit_cut.append(np.squeeze(born_wolf_psf(xxi, cy * np.ones(xxi.shape), zzi, fp, wavelength, ni)))
        else:
            raise ValueError("model must be 'gaussian', 'born-wolf', 'gibson-lanni', or 'vectorial', but was '%s'." % m)

    strs = ['%s, sf=%d, NA=%0.3f(%.0f), chi sq=%0.3f, zc=%0.3f(%.0f)um, xc=%0.3f(%.0f)um, yc=%0.3f(%.0f)um,' \
            ' amp=%0.2f(%.0f), bg=%0.2f(%.0f)' % \
            (m, sf, fp[4], np.sqrt(cv[4, 4]) * 1e3, csq,
             fp[3], np.sqrt(cv[3, 3]) * 1e3, fp[1], np.sqrt(cv[1, 1]) * 1e3,
             fp[2], np.sqrt(cv[2, 2]) * 1e3, fp[0], np.sqrt(cv[0, 0]) * 1e2,
             fp[5], np.sqrt(cv[5, 5]) * 1e2)
            for m, fp, csq, cv, sf in zip(model, fit_params, chi_sqr, cov, sfs)]
    leg = ['%s %0.2f' % (m, fp[4]) for m, fp in zip(model, fit_params)]

    # plot z-planes from 3D fit
    figh_list = []
    fig_names = []
    ncols = 10
    nfigs = np.ceil(nz / ncols)
    nrows = 4
    fig_index = 0
    for ii in range(nz):

        column_ind = np.mod(ii, ncols)
        if column_ind == 0:
            fig_index = fig_index + 1
            figh = plt.figure(figsize=figsize, **kwargs)
            figh_list.append(figh)
            fig_names.append("%s_3d%s_fig%d" % (label, model[0], fig_index))

            stitle = '%d/%d ' % (fig_index, nfigs) + "\n".join(strs)
            plt.suptitle(stitle)

        # plot fit
        ax1 = plt.subplot(nrows, ncols, column_ind + 1)
        plt.imshow(fit_img3d[0][ii, :, :], vmin=vmin, vmax=vmax, cmap='bone')
        plt.title('z=%0.3fum' % z[ii])
        if column_ind == 0:
            plt.ylabel('fit')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.yticks([])
        plt.xticks([])

        # plot data
        ax2 = plt.subplot(nrows, ncols, ncols + column_ind + 1)
        plt.imshow(imgs[ii, :, :], vmin=vmin, vmax=vmax, cmap='bone')
        if column_ind == 0:
            plt.ylabel('img')
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.yticks([])
        plt.xticks([])

        ax3 = plt.subplot(nrows, ncols, 2 * ncols + column_ind + 1)
        plt.imshow(imgs[ii, :, :] - fit_img3d[0][ii, :, :], cmap='bone')
        if column_ind == 0:
            plt.ylabel('img - fit')
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.yticks([])
        plt.xticks([])

        # plot this one centered so x and y can appear on same plot
        ax4 = plt.subplot(nrows, ncols, 3 * ncols + column_ind + 1)
        for cx, cut in zip(cx3d, fit_cut):
            plt.plot(x - cx, cut[ii, :], '.-')

        if imgs_unc is not None:
            yerr_xx = imgs_unc[ii, cy_pix3d, :]
            yerr_yy = imgs_unc[ii, :, cx_pix3d]
        else:
            yerr_xx = np.zeros(imgs[ii, cy_pix3d, :].shape)
            yerr_yy = np.zeros(imgs[ii, :, cx_pix3d].shape)

        plt.errorbar(x - cx3d[0], imgs[ii, cy_pix3d, :], yerr=yerr_xx, fmt='b.')
        plt.errorbar(y - cy3d[0], imgs[ii, :, cx_pix3d], yerr=yerr_yy, fmt='k.')
        plt.ylim([vmin, vmax])

        plt.xlabel('position (um)')
        if column_ind == 0:
            plt.ylabel('ADU')
            plt.legend(leg)
        else:
            plt.setp(ax4.get_yticklabels(), visible=False)

    # plot XZ/YZ planes
    figh = plt.figure(constrained_layout=True, figsize=figsize, **kwargs)
    plt.suptitle("PSF XZ/YZ planes")
    nrows = nfits + 1
    ncols = 9
    spec = plt.GridSpec(ncols=ncols, nrows=nrows, figure=figh)

    gamma = 0.6
    extent = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx, z[-1] + 0.5 * dz, z[0] - 0.5 * dz]

    # ax = plt.subplot(nrows, ncols, 1)
    ax = figh.add_subplot(spec[0, 0:2])
    plt.imshow(imgs[:, cy_pix3d, :], vmin=vmin, vmax=vmax, cmap='bone', extent=extent)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel('Z (um)')
    plt.xlabel('X (um)')
    plt.title('XZ plane')

    # ax = plt.subplot(nrows, ncols, 2)
    ax = figh.add_subplot(spec[0, 2:4])
    plt.imshow(imgs[:, cy_pix3d, :], cmap='bone', extent=extent, norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))
    plt.title("XZ power norm")

    ax = figh.add_subplot(spec[0, 4:6])
    plt.imshow(imgs[:, :, cx_pix3d], vmin=vmin, vmax=vmax, cmap='bone', extent=extent)
    plt.ylabel('Experiment\nZ ($\mu$m)')
    plt.xlabel('Y (um)')
    plt.title('YZ plane')

    ax = figh.add_subplot(spec[0, 6:8])
    plt.imshow(imgs[:, :, cx_pix3d], cmap='bone', extent=extent, norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))
    plt.title('YZ power norm')

    for ii in range(nfits):
        # normal scale
        ax = figh.add_subplot(spec[ii + 1, 0:2])
        im = plt.imshow(fit_img3d[ii][:, cy_pix3d, :], vmin=vmin, vmax=vmax, cmap='bone', extent=extent)
        plt.ylabel('%s, sf=%d, NA=%0.3f\nz ($\mu$m)' % (model[ii], sfs[ii], fit_params[ii][4]))
        if ii < (nfits - 1):
            plt.setp(ax.get_xticklabels(), visible=False)

        # power law scaled, to emphasize smaller features
        ax = figh.add_subplot(spec[ii + 1, 2:4])
        plt.imshow(fit_img3d[ii][:, cy_pix3d, :], cmap='bone', extent=extent, norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))
        if ii < (nfits - 1):
            plt.setp(ax.get_xticklabels(), visible=False)

        # other cut
        ax = figh.add_subplot(spec[ii + 1, 6:8])
        im = plt.imshow(fit_img3d[ii][:, :, cx_pix3d], cmap='bone', extent=extent,
                        norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))
        if ii < (nfits - 1):
            plt.setp(ax.get_xticklabels(), visible=False)

        ax = figh.add_subplot(spec[ii + 1, 4:6])
        im = plt.imshow(fit_img3d[ii][:, :, cx_pix3d], vmin=vmin, vmax=vmax, cmap='bone', extent=extent)
        if ii < (nfits - 1):
            plt.setp(ax.get_xticklabels(), visible=False)

    # figh.subplots_adjust(right=0.8)
    # cbar_ax = figh.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = figh.add_subplot(spec[:, 8])
    figh.colorbar(im, cax=cbar_ax)

    figh_list.append(figh)
    fig_names.append("%s_XZ_YZ_psf" % label)

    # optional saving
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for f, fn in zip(figh_list, fig_names):
            f.savefig(os.path.join(save_dir, fn + ".png"))

            if not close_after_saving:
                plt.close(f)

    return figh_list, fig_names

# get real PSF
def get_exp_psf(imgs, dx, dz, fit_params, rois):
    """
    Get experimental psf from imgs and the results of autofit_psfs

    :param imgs:z-stack of images
    :param dx: pixel size (um)
    :param dz: spacing between image planes (um)
    :param fit_params: n x 6 array, where each element gives [A, cx, cy, cz, NA, bg]
    Use the center positions, background, and amplitude. # todo: can I get rid of the amplitude?
    :param rois: regions of interest

    :return psf_mean:
    :return psf_sdm:
    """

    # fit_params = np.asarray([f['fit_params'] for f in fit_results])

    # get size
    nroi = rois[0][3] - rois[0][2]

    # maximum z size
    nzroi = np.max([r[1] - r[0] for r in rois])
    zpsf = get_psf_coords([nzroi], [dz])[0]
    izero = np.argmin(np.abs(zpsf))
    if not np.abs(zpsf[izero]) < 1e-10:
        raise ValueError("z coordinates do not include zero")

    psf_shifted = np.zeros((len(rois), nzroi, nroi, nroi)) * np.nan
    # loop over rois and shift psfs so they are centered
    for ii, (fp, roi) in enumerate(zip(fit_params, rois)):
        # get roi
        img_roi = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

        # get coordinates
        nz = img_roi.shape[0]
        z, y, x = get_psf_coords(img_roi.shape, [dz, dx, dx])
        izero_shift = np.argmin(np.abs(z))

        # find closest z-plane
        cz = fp[3]
        cx = fp[1]
        cy = fp[2]

        # 3D PSF, shift to be centered
        # any points not present replaced by nans
        psf_temp = ndi.shift(np.array(img_roi, dtype=np.float), [-cz / dz, -cy / dx, -cx / dx], mode='constant', cval=-1)
        psf_temp[psf_temp == -1] = np.nan

        for jj in range(nz):
            ind_z = (izero - izero_shift) + jj
            if ind_z < 0 or ind_z >= nzroi:
                continue
            psf_shifted[ii, ind_z] = psf_temp[jj]

        psf_shifted[ii] = (psf_shifted[ii] - fp[5])

    with np.errstate(divide='ignore', invalid='ignore'):
        not_nan = np.logical_not(np.isnan(psf_shifted))
        norm = np.sum(fit_params[:, 0][:, None, None, None] * not_nan, axis=0)

        psf_mean = np.nansum(psf_shifted, axis=0) / norm

        psf_sd = np.sqrt(np.nansum((psf_shifted / fit_params[:, 0][:, None, None, None] - psf_mean)**2, axis=0) /
                         np.sum(not_nan, axis=0))
        psf_sdm = psf_sd / np.sqrt(np.sum(not_nan, axis=0))

    # the above doesn't do a good enough job of normalizing PSF
    max_val = np.nanmax(psf_mean[psf_mean.shape[0]//2])
    psf_mean = psf_mean / max_val
    psf_sdm = psf_sdm / max_val

    return psf_mean, psf_sdm

def export_psf_otf(imgs, dx, dz, wavelength, ni, rois, fit3ds, fit2ds, expected_na=None,
                   percentiles=(20, 15, 10, 5, 2.5), figsize=(20, 10), **kwargs):
    """
    Plot 2D PSF and OTF and fit PSF to several simple models.

    :param ni:
    :param psf: ny x nx array
    :param dx: pixel size in um
    :param dz: plane separation in um (not currently used...)
    :param wavelength: wavelength in um
    :param expected_na: expected numerical aperture
    :param psf_unc: point spread function standard deviation (if created from average of several spots).
    :return:
    """
    fighs = []
    fignames = []

    fit_params_2d = np.asarray([f['fit_params'] for f in fit2ds])

    exp_psfs = []
    exp_psfs_sd = []
    fits2d = []
    fits3d = []

    for p in percentiles:
        na_min = np.percentile(fit_params_2d[:, 4], 100 - p)
        fit3ds_psf, fit2ds_psf, rois_psf = zip(*[[f3, f2, r] for f3, f2, r in zip(fit3ds, fit2ds, rois)
                                                 if f2['fit_params'][4] >= na_min])
        fit_params = np.asarray([f["fit_params"] for f in fit3ds_psf])
        psf, psf_sd = get_exp_psf(imgs, dx, dz, fit_params, rois_psf)
        exp_psfs.append(psf)
        exp_psfs_sd.append(psf_sd)

        # get coordinates
        nz, ny, nx = psf.shape

        # Fit to 2D PSFs
        z, y, x = get_psf_coords(psf.shape, [dz, dx, dx])
        iz = np.argmin(np.abs(z))
        ixz = np.argmin(np.abs(x))
        psf2d = psf[iz]
        psf2d_sd = psf_sd[iz]

        bounds = ((0, -4*dx, -4*dx, -1e-12, 0.2, -np.inf),
                  (np.inf, 4*dx, 4*dx, 1e-12, ni, np.inf))

        init_params = [None, 0, 0, 0, None, None]
        fixed_params = [False, False, False, True, False, False]

        # gaussian 2D fit
        sf = 3
        result_g, fn_g2 = fit_pixelated_psfmodel(psf2d[None, :, :], dx, dz, wavelength, ni, sf=sf, model='gaussian',
                                                 init_params=init_params, fixed_params=fixed_params, bounds=bounds)
        # correct cz for 3D plotting
        result_g['fit_params'][3] = z[iz]
        pfit_g2d = result_g['fit_params']
        cov_g2 = result_g['covariance']

        fits2d.append(result_g)

        # vectorial 2D fit
        result_v2, fn_v2 = fit_pixelated_psfmodel(psf2d[None, :, :], dx, dz, wavelength, ni, model='vectorial',
                                                  init_params=init_params, fixed_params=fixed_params, bounds=bounds)
        result_v2['fit_params'][3] = z[iz]
        cov_v2 = result_v2['covariance']
        pfit_v2 = result_v2['fit_params']

        # fit with expected_na fixed to get fit to compare against
        if expected_na is not None:
            pfit_exp = copy.deepcopy(pfit_v2)
            pfit_exp[4] = expected_na
            # fixed_ps = [False, False, False, True, True, False]
            fixed_ps = [True, False, False, True, True, False]
            result_exp_na, ffn_exp_na = fit_pixelated_psfmodel(psf2d[None, :, :], dx, dz, wavelength, ni,
                                                               model='vectorial', init_params=pfit_exp,
                                                               fixed_params=fixed_ps,
                                                               bounds=bounds)
            vexp = ffn_exp_na(np.array([0]), x.size, dx)[0]



        # Fit and plot 3D vectorial PSF model
        result_v, fn_v = fit_pixelated_psfmodel(psf, dx, dz, wavelength, ni, model='vectorial')
        fits3d.append(result_v)

        figs_v, fig_names_v = plot_psf3d(psf, dx, dz, wavelength, ni, [result_v, result_g, result_v2], imgs_unc=psf_sd,
                                         model=['vectorial', 'gaussian', 'vectorial'], sfs=[1, sf, 1], figsize=figsize)
        for ii in range(len(fig_names_v)):
            fig_names_v[ii] = "psf_otf_%.1f_percentile_" % p + fig_names_v[ii]

        # plot 2D fits
        figh = plt.figure(figsize=figsize, **kwargs)
        gauss_fwhm = na2fwhm(pfit_g2d[4], wavelength)

        plt.suptitle('Smallest %0.1f percent of beads, %d points with 2D fit NA >= %0.3f\n'
                     'Gauss NA = %0.2f(%.0f), Vectorial = %0.2f(%.0f)\nGauss FWHM=%0.0f(%.0f)nm' %
                     (p, len(rois_psf), na_min, pfit_g2d[4], np.sqrt(cov_g2[4, 4]) * 1e2,
                      pfit_v2[4], np.sqrt(cov_v2[4, 4]) * 1e2,
                      gauss_fwhm * 1e3, gauss_fwhm * 1e3 / pfit_g2d[4] * np.sqrt(cov_g2[4, 4])))

        plt.subplot(2, 2, 1)
        plt.imshow(psf2d, cmap='bone')
        plt.title('psf')

        # plot data
        plt.subplot(2, 2, 3)
        # plot zero
        cmin = np.min([np.min(x) * np.sqrt(2), np.min(x) * np.sqrt(2)])
        cmax = np.max([np.max(y) * np.sqrt(2), np.max(x) * np.sqrt(2)])
        ph1, = plt.plot([cmin, cmax], [0, 0], 'k')

        # plot fits
        ph2, = plt.plot(y, fn_g2(0, nx, dx)[0, ixz, :], '--', marker='o', color='r')
        ph3, = plt.plot(y, fn_v2(0, x.size, dx)[0, ixz, :], marker='s', color='orange')
        if expected_na is not None:
            ph4, = plt.plot(y, vexp[ixz, :], '--', marker='>', color='g')

        # plot data
        ph5 = plt.errorbar(x, psf2d[ixz, :], yerr=psf2d_sd[ixz, :], fmt='.', color='k')
        ph6 = plt.errorbar(y, psf2d[:, ixz], yerr=psf2d_sd[:, ixz], fmt='.', color='b')
        ph7 = plt.errorbar(np.sqrt(2)/2 * (x + y), np.diag(psf2d), yerr=np.diag(psf2d_sd), fmt='.', color='m')
        ph8 = plt.errorbar(np.sqrt(2)/2 * (x + y), np.diag(np.rot90(psf2d)), yerr=np.diag(np.rot90(psf2d_sd)),
                     fmt='.', color='blueviolet')

        plt.xlabel('position (um)')
        if expected_na is None:
            plt.legend([ph2, ph3, ph5, ph6, ph7, ph8], ['gauss na=%0.2f' % pfit_g2d[4],
                        'vectorial na=%0.2f' % pfit_g2d[4],
                        'x', 'y', 'x+y', 'x-y'])
        else:
            plt.legend([ph2, ph3, ph4, ph5, ph6, ph7, ph8], ['gauss na=%0.2f' % pfit_g2d[4],
                        'vectorial na=%0.2f' % pfit_g2d[4],
                        'vectorial na=%0.2f' % expected_na, 'x', 'y', 'x+y', 'x-y'])

        # OTF
        # correct problematic PSF points before calculating mtf
        psf2d[np.isnan(psf2d)] = 0
        psf2d[psf2d < 0] = 0
        mtf = np.abs(fft.fftshift(fft.fft2(fft.ifftshift(psf2d))))
        fx = tools.get_fft_frqs(mtf.shape[0], dt=dx)
        fy = fx
        dfx = fx[1] - fx[0]

        ax = plt.subplot(2, 2, 2)

        extent = [fx[0] - 0.5*dfx, fx[-1] + 0.5*dfx, fy[-1] + 0.5*dfx, fy[0] - 0.5*dfx]
        plt.imshow(mtf, extent=extent, cmap='bone')

        if expected_na is not None:
            fmax = 1 / (wavelength / 2 / expected_na)
            circ = Circle((0, 0), radius=fmax, color='r', fill=0)
            ax.add_artist(circ)

        plt.title('otf')

        # 1D OTF plot
        ax = plt.subplot(2, 2, 4)

        if expected_na is not None:
            finterp = np.linspace(np.min(fx) * np.sqrt(2), np.max(fx) * np.sqrt(2), 500)
            otf_ideal = circ_aperture_otf(finterp, 0, expected_na, wavelength) * mtf[ixz, ixz]
            plt.plot(finterp, otf_ideal, '-')

        plt.plot(fx, mtf[ixz, :], '.')
        plt.plot(fy, mtf[:, ixz], '.')
        plt.plot(np.sqrt(2)/2 * (fx + fy), np.diag(mtf), '.')
        plt.plot(np.sqrt(2)/2 * (fx + fy), np.diag(np.rot90(mtf)), '.')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if expected_na is not None:
            plt.plot([fmax, fmax], [ylim[0], ylim[1]], 'k')
            plt.plot([-fmax, -fmax], [ylim[0], ylim[1]], 'k')

        # zero line
        plt.plot([xlim[0], xlim[1]], [0, 0], 'k--')

        if expected_na is None:
            plt.legend(['x', 'y', 'x+y', 'x-y'])
        else:
            plt.legend(['airy otf na=%0.2f' % expected_na, 'x', 'y', 'x+y', 'x-y'])

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

        plt.xlabel('fx (1/um)')

        fighs.append(figh)
        fignames.append('psf_otf_%.1f_percentile' % p)

        fighs += figs_v
        fignames += fig_names_v

    data = {'experimental_psfs': np.asarray(exp_psfs), 'experimental_psfs_sd': np.asarray(exp_psfs_sd),
            'experimental_psfs_fit2d': fits2d, 'experimental_psfs_fits3d': fits3d,
            'experimental_psf_percentiles': percentiles}

    return fighs, fignames, data

# automatically process image
def find_candidate_beads(img, min_distance=1, abs_thresh_std=1., max_thresh=-np.inf, max_num_peaks=np.inf,
                         mode="max_filter"):
    """
    Find candidate beads in image. Based on function from mesoSPIM-PSFanalysis

    :param img: 2D or 3D image
    :param filter_xy_pix: standard deviation of Gaussian filter applied to image in xy plane before peak finding
    :param filter_z_pix:
    :param min_distance: minimum allowable distance between peaks
    :param abs_thresh_std: absolute threshold for identifying peaks, as a multiple of the image standard deviation
    :param abs_thresh: absolute threshold, in raw counts. If both abs_thresh_std and abs_thresh are provided, the
    maximum value will be used
    :param max_num_peaks: maximum number of peaks to find.

    :return centers: np.array([[cz, cy, cx], ...])
    """

    # gaussian filter to smooth image before peak finding
    # if img.ndim == 3:
    #     filter_sds = [filter_z_pix, filter_xy_pix, filter_xy_pix]
    # elif img.ndim == 2:
    #     filter_sds = filter_xy_pix
    # else:
    #     raise ValueError("img should be a 2 or 3 dimensional array, but was %d dimensional" % img.ndim)
    #
    # smoothed = skimage.filters.gaussian(img, filter_sds, output=None, mode='nearest', cval=0,
    #                                     multichannel=None, preserve_range=True)

    # set threshold value
    abs_threshold = np.max([img.mean() + abs_thresh_std * img.std(), max_thresh])

    if mode == "max_filter":
        centers = skimage.feature.peak_local_max(img, min_distance=min_distance, threshold_abs=abs_threshold,
                                                 exclude_border=False, num_peaks=max_num_peaks)
    elif mode == "threshold":
        ispeak = img > abs_threshold
        # get indices of points above threshold
        coords = np.meshgrid(*[range(img.shape[ii]) for ii in range(img.ndim)], indexing="ij")
        centers = np.concatenate([c[ispeak][:, None] for c in coords], axis=1)
    else:
        raise ValueError("mode must be 'max_filter', or 'threshold', but was '%s'" % mode)

    return centers

# todo: good candidate for numba or computing in chunks...
def combine_nearby_beads(centers, min_xy_dist, min_z_dist, mode="average", weights=None):
    """
    Combine multiple peaks above threshold into reduced set, where assume all peaks separated by no more than
    min_xy_dist and min_z_dist come from the same feature.

    :param centers:
    :param min_xy_dist:
    :param min_z_dist:
    :param mode:
    :param weights:
    :return:
    """
    centers_unique = np.array(centers, copy=True)
    inds = np.arange(len(centers), dtype=np.int)

    if weights is None:
        weights = np.ones(len(centers_unique))

    counter = 0
    while 1:
        # compute distances to all other beads
        z_dists = np.abs(centers_unique[counter][0] - centers_unique[:, 0])
        xy_dists = np.sqrt((centers_unique[counter][1] - centers_unique[:, 1]) ** 2 +
                           (centers_unique[counter][2] - centers_unique[:, 2]) ** 2)

        # beads which are close enough we will combine
        combine = np.logical_and(z_dists < min_z_dist, xy_dists < min_xy_dist)
        if mode == "average":
            # centers_unique[counter] = np.nanmean(centers_unique[combine], axis=0, dtype=np.float)
            denom = np.nansum(np.logical_not(np.isnan(np.sum(centers_unique[combine], axis=1))) * weights[combine])
            centers_unique[counter] = np.nansum(centers_unique[combine] * weights[combine][:, None], axis=0, dtype=np.float) / denom
            weights[counter] = denom
        elif mode == "keep-one":
            pass
        else:
            raise ValueError("mode must be 'average' or 'keep-one', but was '%s'" % mode)

        # remove all points from list except for one representative
        combine[counter] = False

        inds = inds[np.logical_not(combine)]
        centers_unique = centers_unique[np.logical_not(combine)]
        weights = weights[np.logical_not(combine)]

        counter += 1
        if counter >= len(centers_unique):
            break

    return centers_unique, inds

def find_beads(imgs, imgs_sd=None, roi_size_pix=(1, 1, 1), min_distance_to_keep=1., thresh_std=1., thresh_abs=-np.inf,
               max_sep_assume_one_peak=2., min_sigma_pix=0.7, max_sigma_pix=5., max_percent_asymmetry=0.5,
               min_fit_amp=100., max_off_center_fraction=0.25, max_num_peaks=np.inf, filter_image=False,
               filter_small_sigmas=(0, 1, 1), filter_large_sigmas=None):
    """
    Identify beads in a 3D image. This is done in three steps.

    1. The 2D planes of the image are smoothed and a peak finding algorithm is applied, which returns up to a user
    specified number of the largest peaks. For best results, this number should be somewhat larger than the total number
    of expected beads in the image.
    2. Each peak candidate is fit to a Gaussian function, and peaks with implausible fit parameters are excluded.
    3. Remaining beads are excluded if more than falls within the same region of interest.

    :param imgs: nz x nx x ny image
    :param imgs_sd: standard deviations, assuming img is an average of other images or can estimate uncertainty from
    camera read noise, or etc. If not used, set to None
    :param dx: pixel size in um
    :param dz: z-plane spacing in um
    :param roi_size_pix: ROI size in real units (nz_um, ny_um, nx_um)
    :param min_sigma_pix: only points with gaussian standard deviation larger than this value will be considered
    so we can avoid e.g. hot pixels
    :param float min_distance_to_keep: minimum distance between peaks, in pixels
    :param float thresh_std:
    :param int max_num_peaks: maximum number of peaks to find
    :param bool filter_image: boolean. whether or not to remove background from image before peak finding.

    :return rois: list of regions of interest [[zstart, zend, ystart, yend, xstart, xend], ...] as coordinates in imgs
    :return centers: list of centers [[cz, cy, cx], ....] as coordinates in imgs
    :return fit_params: nroi x 7, where each row is of the form [A, cx, cy, sx, sy, bg]. The coordinates sx are given
    relative to the region of interest. So the center for the bead is at (cx + x_roi_start, cy + y_roi_start)
    """

    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)

    # ##############################
    # get ROI sizes
    # ##############################
    # todo: make in pixels instead of um
    # nx_roi = np.round(roi_size_um[2] / dx)
    # if np.mod(nx_roi, 2) == 0:
    #     nx_roi = nx_roi + 1
    #
    # # using square ROI
    # ny_roi = nx_roi
    #
    # # don't care if z-roi size is odd
    # nz_roi = np.round(roi_size_um[0] / dz)
    nz_roi, ny_roi, nx_roi = roi_size_pix

    if not nx_roi % 2 == 1:
        # raise ValueError("nx_roi must be odd, but was %d." % nx_roi)
        nx_roi += 1

    if not ny_roi % 2 == 1:
        # raise ValueError("ny_roi must be odd, but was %d." % ny_roi)
        ny_roi += 1

    if not nz_roi % 2 == 1:
        # raise ValueError("ny_roi must be odd, but was %d." % nz_roi)
        nz_roi += 1



    # ##############################
    # filter image
    # ##############################
    if filter_image:
        if filter_large_sigmas is not None:
            imgs_filtered = skimage.filters.difference_of_gaussians(imgs, filter_small_sigmas, filter_large_sigmas,
                                                                    mode='nearest', cval=0, multichannel=None)
        else:
            imgs_filtered = skimage.filters.gaussian(imgs, filter_small_sigmas, mode="nearest", cval=0,
                                                     multichannel=None, preserve_range=True)
    else:
        imgs_filtered = imgs

    # ##############################
    # find plausible peaks
    # ##############################
    centers_all = find_candidate_beads(imgs_filtered, min_distance=min_distance_to_keep, abs_thresh_std=thresh_std,
                                       max_thresh=thresh_abs, max_num_peaks=max_num_peaks)
    print("Found %d candidates" % len(centers_all))
    centers = copy.deepcopy(centers_all)

    # get ROI's for each peak
    rois = np.array([tools.get_centered_roi(c, [nz_roi, ny_roi, nx_roi], min_vals=[0, 0, 0], max_vals=np.array(imgs.shape)) for c in centers])

    # discard rois that get cropped by the edge of the image
    nys = rois[:, 3] - rois[:, 2]
    nxs = rois[:, 5] - rois[:, 4]
    away_from_edge = np.logical_and(nys == ny_roi, nxs == nx_roi)
    rois = rois[away_from_edge]
    centers = centers[away_from_edge]
    print("%d candidates not too close to edge" % len(centers))

    # ##############################
    # fit 2D center to gaussian to determine if plausible candidates
    # ##############################
    # centers within rois
    c_rois = np.array([tools.full2roi(c, roi) for c, roi in zip(centers, rois)])

    # do fitting in parallel
    if imgs_sd is not None:
        results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                joblib.delayed(fit.fit_gauss2d)(imgs[centers[ii, 0], rois[ii, 2]:rois[ii, 3], rois[ii, 4]:rois[ii, 5]],
                                                sd=imgs_sd[centers[ii, 0], rois[ii, 2]:rois[ii, 3], rois[ii, 4]:rois[ii, 5]],
                                                init_params=[None, c_rois[ii, 2], c_rois[ii, 1], 0.5, 0.5, 0, 0])
                                                for ii in range(len(rois))
                )
    else:
        results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
            joblib.delayed(fit.fit_gauss2d)(imgs[centers[ii, 0], rois[ii, 2]:rois[ii, 3], rois[ii, 4]:rois[ii, 5]],
                                            init_params=[None, c_rois[ii, 2], c_rois[ii, 1], 0.5, 0.5, 0, 0])
            for ii in range(len(rois))
        )

    results = list(zip(*results))[0]
    fitp = np.asarray([r['fit_params'] for r in results])

    # exclude peaks if too little weight
    big_enough = fitp[:, 0] >= min_fit_amp

    # exclude points that are not very symmetric
    asymmetry = np.abs(fitp[:, 3] - fitp[:, 4]) / (0.5 * (fitp[:, 3] + fitp[:, 4]))
    is_symmetric = asymmetry < max_percent_asymmetry

    # exclude points too far from center of ROI
    on_center = np.logical_and(np.abs(fitp[:, 1] - 0.5 * nx_roi) / nx_roi < max_off_center_fraction,
                               np.abs(fitp[:, 2] - 0.5 * nx_roi) / nx_roi < max_off_center_fraction)
    # exclude points if sigma too large
    not_too_big = np.logical_and(fitp[:, 3] < max_sigma_pix, fitp[:, 4] < max_sigma_pix)
    # exclude points if sigma too small
    not_too_small = np.logical_and(fitp[:, 3] > min_sigma_pix, fitp[:, 4] > min_sigma_pix)

    # combine all conditions and reduce centers/rois
    to_use = np.logical_and.reduce((is_symmetric, on_center, not_too_big, not_too_small, big_enough))

    centers = centers[to_use]
    rois = rois[to_use]
    fitp = fitp[to_use]
    results = [r for r, keep in zip(results, to_use) if keep]
    print("%d candidates with plausible fit parameters" % len(centers))

    # ##############################
    # replace centers by fit values
    # ##############################
    centers = np.array(centers, dtype=np.float)
    centers[:, 2] = fitp[:, 1] + rois[:, 4]
    centers[:, 1] = fitp[:, 2] + rois[:, 2]

    # ##############################
    # find sets of peaks so close together probably fitting the same peak. Only keep one each
    # ##############################
    if len(centers) > 1:
        _, inds_to_keep = combine_nearby_beads(centers, max_sep_assume_one_peak, max_sep_assume_one_peak, mode="keep-one")
        centers = centers[inds_to_keep]
        rois = rois[inds_to_keep]
        fitp = fitp[inds_to_keep]
        results = [results[ii] for ii in inds_to_keep]

    print("%d points remain after removing likely duplicates" % len(centers))

    # ##############################
    # discards points too close together (only look at 2D distance)
    # ##############################
    if len(centers) > 1:
        min_dists = np.array([min_dist(c[1:], centers[:, 1:]) for c in centers])
        dists_ok = min_dists > min_distance_to_keep

        rois = rois[dists_ok]
        centers = centers[dists_ok]
        fitp = fitp[dists_ok]
        results = [r for r, keep in zip(results, dists_ok) if keep]

    print("%d candidates separated by > %0.2f pix" % (len(centers), min_distance_to_keep))

    return rois, centers, fitp, results, imgs_filtered, centers_all

def fit_roi(center, roi, dx, dz, wavelength, ni, sf, imgs, imgs_sd, model):
    """
    Fit region of interest in image stack to a psf model function

    :param center: (cz, cy, cx)
    :param roi: [zstart, zend, ystart, yend, xstart, xend]
    :param dx: pixel size in um
    :param dz: z-plane spacing in um
    :param wavelength: wavelength in um
    :param imgs: images stack
    :param imgs_sd: may also be None, in which case is ignored
    :param model:
    :return fit2d:
    :return fit3d:
    """

    # get roi
    sub_img = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    if imgs_sd is not None:
        sub_img_sd = imgs_sd[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    else:
        sub_img_sd = None

    # get coordinates
    z, y, x = get_psf_coords(sub_img.shape, [dz, dx, dx])

    # find z-plane closest to fit center and fit gaussian to get initial parameters for model fit
    c_roi = tools.full2roi(center, roi)
    cz_roi_int = int(np.round(c_roi[0]))
    # result, _ = fit_pixelated_psfmodel(sub_img[cz_roi_int][None, :, :], dx, dz, wavelength, ni=ni, sf=sf, model='gaussian',
    #                                    init_params=[None, None, None, 0, None, None],
    #                                    fixed_params=[False, False, False, True, False, False])
    # result['fit_params'][3] = z[cz_roi_int]

    # full 3D PSF fit
    # na cannot exceed index of refraction...
    cz = z[cz_roi_int]
    bounds = ((0, x.min(), y.min(), np.max([cz - 2.5 * float(dz), z.min()]), 0.2, -np.inf),
              (np.inf, x.max(), y.max(), np.min([cz + 2.5 * float(dz), z.max()]), ni, np.inf))

    # fit 3D psf model
    # todo: add sf
    # init_params = result['fit_params']
    init_params = [None, None, None, cz, None, None]
    fit3d, _ = fit_pixelated_psfmodel(sub_img, dx, dz, wavelength, ni=ni, model=model, init_params=init_params,
                                      sd=sub_img_sd, bounds=bounds)

    # fit 2D gaussian to closest slice to 3D fit center
    izc = np.argmin(np.abs(z - fit3d['fit_params'][3]))
    fit2d, _ = fit_pixelated_psfmodel(sub_img[izc][None, :, :], dx, dz, wavelength, ni=ni, sf=sf, model='gaussian',
                                      init_params=[None, None, None, 0, None, None],
                                      fixed_params=[False, False, False, True, False, False])
    fit2d['fit_params'][3] = z[izc]

    return fit2d, fit3d

# main fitting function
def autofit_psfs(imgs, imgs_sd, dx, dz, wavelength, ni=1.5, model='vectorial', sf=3, window_size_um=(1, 1, 1), **kwargs):
    """
    Find isolated points, fit PSFs, and report data. This is the main function of this module

    :param imgs: nz x nx x ny image
    :param imgs_sd: standard deviations, assuming img is an average of other images
    :param float dx: pixel size in um
    :param float dz: z-plane spacing in um
    :param float wavelength: wavelength in um
    :param **kwargs: passed through to find_beads(). See that function for allowed keyword arguments
    :return centers:
    :return rois:
    :return fit2ds:
    :return fit3ds:
    """

    # ensure these are floats, not numpy arrays
    dz = float(copy.deepcopy(dz))
    dx = float(copy.deepcopy(dx))
    wavelength = float(copy.deepcopy(wavelength))

    sz_um, sy_um, sx_um = window_size_um
    sz_pix = int(np.round(sz_um / dz))
    sy_pix = int(np.round(sy_um / dx))
    sx_pix = int(np.round(sx_um / dx))
    if sy_pix != sx_pix:
        raise ValueError("window_size_um must give the same size window in x and y directions, but gave %0.2f and %0.2f" % (sy_um, sx_um))

    # todo: to account for images with fluctuating background, might want to segment the image and then apply some stages of this?
    rois, centers, _, _, _, _ = find_beads(imgs, imgs_sd, roi_size_pix=(sz_pix, sy_pix, sx_pix), **kwargs)

    # ##############################
    # fit each peak to psf model in parallel using joblib
    # ##############################
    if len(rois) == 0:
        centers = []
        rois = [[]]
        fit2ds = None
        fit3ds = None
    else:
        fit_roi_partial = partial(fit_roi, dx=dx, dz=dz, wavelength=wavelength, ni=ni, sf=sf, imgs=imgs, imgs_sd=imgs_sd, model=model)
        print("starting PSF fitting for %d ROI's" % len(rois))
        results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                  joblib.delayed(fit_roi_partial)(centers[ii], rois[ii]) for ii in range(len(centers)))
        # results = []
        # for ii in range(len(centers)):
        #     r = fit_roi(centers[ii], rois[ii], dx=dx, dz=dz, wavelength=wavelength, ni=ni, sf=sf, imgs=imgs, imgs_sd=imgs_sd, model=model)
        #     results.append(r)

        # unpack results
        results = list(zip(*results))
        fit3ds = results[1]
        fit2ds = results[0]

    return centers, rois, fit2ds, fit3ds

# main display function
def display_autofit(imgs, imgs_unc, dx, dz, wavelength, ni, rois, centers, fit2ds, fit3ds, model='vectorial',
                    expected_na=None, psf_percentiles=(50, 20, 15, 10, 5, 2.5), summary_only=False, num_psfs_to_display=20,
                    save_dir=None, figsize=(20, 10), **kwargs):
    """
    display results of autofit_psfs

    :param ni:
    :param dx: pixel size in um
    :param dz: z plane separation in um
    :param imgs: full 3D image
    :param imgs_unc:
    :param fit3ds: list of lists [[A, cx, cy, cz, NA, bg], ... ]
    :param rois: [[zstart, zend, ystart, yend, xstart, xend], ...]
    :param model: 'gibson-lanni', 'vectorial', or 'gaussian'
    :param summary_only: if True, plot only summary graphs and do not plot individual PSF's
    :param num_psfs_to_display: maximum number of PSF's to plot
    :param save_dir: directory to save results. If None, results will not be saved.
    :param figsize:
    :param kwargs: parameters which will be passed to figures
    :return:
    """

    fighs = []
    fig_names = []

    if save_dir is not None:
        save_dir = tools.get_unique_name(save_dir, mode='dir')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # unpack fit results
    fp3d = np.asarray([f['fit_params'] for f in fit3ds])
    fp2d = np.asarray([f['fit_params'] for f in fit2ds])

    figh1 = plot_fit_stats(fit3ds, fit2ds, figsize, **kwargs)

    fighs.append(figh1)
    fig_names.append(['NA_summary'])
    if save_dir is not None:
        fname = 'NA_summary.png'
        figh1.savefig(os.path.join(save_dir, fname))

    # plot points
    # figh2 = plot_bead_locations(imgs, centers, fit3ds, fit2ds, figsize, **kwargs)
    figh2 = plot_bead_locations(imgs, centers, title="Max intensity projection and NA from 2D fit versus position",
                                weights=[ft["fit_params"][4] for ft in fit2ds], cbar_labels=["NA"], figsize=figsize, **kwargs)

    fighs.append(figh2)
    fig_names.append(['NA_vs_position'])
    if save_dir is not None:
        fname = 'NA_vs_position.png'
        figh2.savefig(os.path.join(save_dir, fname), dpi=400)

    # psf averages
    figs_psf, fig_names_psf, exp_psf_data = export_psf_otf(imgs, dx, dz, wavelength, ni, rois, fit3ds, fit2ds,
                                                           expected_na=expected_na, percentiles=psf_percentiles,
                                                           figsize=figsize, **kwargs)

    # collect fit data
    data = {'fit3ds': fit3ds, 'rois': rois, 'centers': centers, 'psf_model_3d': model, 'fit2ds': fit2ds,
            'wavelength': wavelength, 'dx': dx, 'dz': dz, 'ni': ni}
    data.update(exp_psf_data)

    if save_dir is not None:
        fpath = os.path.join(save_dir, 'data.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)

    # save figures
    if save_dir is not None:
        for f, n in zip(figs_psf, fig_names_psf):
            f.savefig(os.path.join(save_dir, '%s.png' % n))
            plt.close(f)
    else:
        fighs = fighs + figs_psf
        fig_names = fig_names + fig_names_psf


    if not summary_only:
        n_to_plot = np.min([len(fit3ds), num_psfs_to_display])
        print("plotting %d PSF fits..." % n_to_plot)
        # plot all ROI fits
        inds = np.flip(np.argsort(fp2d[:, 4]))[:n_to_plot]

        # do plotting in parallel
        if imgs_unc is not None:
            results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                joblib.delayed(plot_psf3d)(
                    imgs[rois[ii][0]:rois[ii][1], rois[ii][2]:rois[ii][3], rois[ii][4]:rois[ii][5]],
                    dx, dz, wavelength, ni, [fit3ds[ii], fit2ds[ii]],
                    imgs_unc=imgs_unc[rois[ii][0]:rois[ii][1], rois[ii][2]:rois[ii][3], rois[ii][4]:rois[ii][5]],
                    model=[model, 'gaussian'], sfs=[1, 3],
                    figsize=figsize, save_dir=save_dir, close_after_saving=False,
                    label='na=%0.3f_cx=%d_cy=%d_cz=%d' % (fp2d[ii, 4], centers[ii, 2], centers[ii, 1], centers[ii, 0]))
                for ii in inds
            )

        else:
            results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                joblib.delayed(plot_psf3d)(
                    imgs[rois[ii][0]:rois[ii][1], rois[ii][2]:rois[ii][3], rois[ii][4]:rois[ii][5]],
                    dx, dz, wavelength, ni, [fit3ds[ii], fit2ds[ii]],
                    model=[model, 'gaussian'], sfs=[1, 3],
                    figsize=figsize, save_dir=save_dir, close_after_saving=False,
                    label='na=%0.3f_cx=%d_cy=%d_cz=%d' % (
                        fp2d[ii, 4], centers[ii, 2], centers[ii, 1], centers[ii, 0]))
                for ii in inds
            )

        if save_dir is None:
            fighs_psf, fig_names_psf = zip(*results)
            fighs = fighs + fighs_psf
            fig_names = fig_names + fig_names_psf

    return fighs, fig_names, data

def plot_fit_stats(fit3ds, fit2ds, figsize, **kwargs):
    """
    Plot statistics for a list of fit result dictionaries

    :param fit3ds: list of dictionary objects, or None
    :param fit2ds: list of dictionary objects, or None
    :param figsize:
    :param kwargs: passed to plt.figure()
    :return figh: figure handle
    """

    # unpack fit results we will need
    fp3d = np.asarray([f['fit_params'] for f in fit3ds])
    chisq_3d = np.squeeze(np.asarray([[f['chi_squared'] for f in fit3ds]]))
    covs3d = np.asarray([f['covariance'] for f in fit3ds])
    # 2d
    fp2d = np.asarray([f['fit_params'] for f in fit2ds])
    chisq_2d = np.squeeze(np.asarray([[f['chi_squared'] for f in fit2ds]]))
    covs_2d = np.asarray([f['covariance'] for f in fit2ds])


    # fit parameter summary
    figh = plt.figure(figsize=figsize, **kwargs)
    plt.suptitle("PSF fit parameter summary")
    grid = plt.GridSpec(2, 4, hspace=1, wspace=0.5)

    # 3D PSF fits
    if fit3ds is not None:
        # histogram of NAs
        plt.subplot(grid[0, 0])
        # edges = np.linspace(0, 1.5, 20)
        edges = np.arange(0, 1.6, 0.05)
        hcenters = 0.5 * (edges[1:] + edges[:-1])
        na_h, _ = np.histogram(fp3d[:, 4], edges)
        plt.plot(hcenters, na_h, '.-')

        plt.xlabel('NA')
        plt.ylabel('#')
        plt.title('3D fits: histogram of NAs')

        plt.subplot(grid[0, 1])
        plt.errorbar(fp3d[:, 4], chisq_3d, xerr=np.sqrt(covs3d[:, 4, 4]), fmt='o')
        plt.xlim([-0.05, 1.55])
        plt.ylim([0.9 * chisq_3d.min(), 1.1 * chisq_3d.max()])

        plt.xlabel('NA')
        plt.ylabel('chi sqr')
        plt.title('3D fits: chi sqr vs. NA')

        plt.subplot(grid[0, 2])
        plt.errorbar(fp3d[:, 4], fp3d[:, 0], xerr=np.sqrt(covs3d[:, 4, 4]), yerr=np.sqrt(covs3d[:, 0, 0]), fmt='o')
        plt.xlim([-0.05, 1.55])
        plt.ylim([0, 1.2 * fp3d[:, 0].max()])

        plt.xlabel('NA')
        plt.ylabel('Amplitude')
        plt.title('3D fits: NA vs. amp')

        plt.subplot(grid[0, 3])
        plt.errorbar(fp3d[:, 4], fp3d[:, 5], xerr=np.sqrt(covs3d[:, 4, 4]), yerr=np.sqrt(covs3d[:, 5, 5]), fmt='o')
        plt.xlim([-0.05, 1.55])
        max_diff = fp3d[:, 5].max() - fp3d[:, 5].min()
        plt.ylim([fp3d[:, 5].min() - 0.1 * max_diff, fp3d[:, 5].max() + 0.1 * max_diff])

        plt.xlabel('NA')
        plt.ylabel('Background')
        plt.title('3D fits: NA vs. bg')

    # 2D
    if fit2ds is not None:
        plt.subplot(grid[1, 0])
        na_h2d, _ = np.histogram(fp2d[:, 4], edges)
        plt.plot(hcenters, na_h2d, '.-')

        plt.xlabel('NA')
        plt.ylabel('#')
        plt.title('2D fit: histogram of NAs')

        plt.subplot(grid[1, 1])
        plt.errorbar(fp2d[:, 4], chisq_2d, xerr=np.sqrt(covs_2d[:, 4, 4]), fmt='o')
        plt.xlim([-0.05, 1.55])
        plt.ylim([0.9 * chisq_2d.min(), 1.1 * chisq_2d.max()])

        plt.xlabel('NA')
        plt.ylabel('chi sqr')
        plt.title('2D fits: chi sqr vs. NA')

        plt.subplot(grid[1, 2])
        plt.errorbar(fp2d[:, 4], fp2d[:, 0], xerr=np.sqrt(covs_2d[:, 4, 4]), yerr=np.sqrt(covs_2d[:, 0, 0]), fmt='o')
        plt.xlim([-0.05, 1.55])
        plt.ylim([0, 1.2 * fp2d[:, 0].max()])

        plt.xlabel('NA')
        plt.ylabel('Amplitude')
        plt.title('2D fit: NA vs. amp')

        plt.subplot(grid[1, 3])
        plt.errorbar(fp2d[:, 4], fp2d[:, 5], xerr=np.sqrt(covs_2d[:, 4, 4]), yerr=np.sqrt(covs_2d[:, 5, 5]), fmt='o')
        plt.xlim([-0.05, 1.55])
        max_diff = fp2d[:, 5].max() - fp2d[:, 5].min()
        plt.ylim([fp2d[:, 5].min() - 0.1 * max_diff, fp2d[:, 5].max() + 0.1 * max_diff])

        plt.xlabel('NA')
        plt.ylabel('Background')
        plt.title('2D fit: NA vs. bg')

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

# test PSF model functions and fits
def compare_psfs_models(wavelength=0.635, na=1.35, ni=1.5, dxy=0.065, nx=51, dz=0.1, nz=21, figsize=(20, 10), **kwargs):
    """
    compare various PSF model functions at different sets of parameters

    :param wavelength:
    :param na:
    :param dxy:
    :param nx:
    :param dz:
    :param nz:
    :return:
    """

    # get coordinates
    ny = nx
    z, y, x = get_psf_coords([nz, ny, nx], [dz, dxy, dxy])
    yy, zz, xx = np.meshgrid(y, z, x)

    # set parameters
    params = [1, 0, 0, 0, na, 0]

    # vectorial psf
    psf_vec = model_psf(nx, dxy, z, params, wavelength, ni, model='vectorial')
    # gibson-lanni
    psf_gl = model_psf(nx, dxy, z, params, wavelength, ni, model='gibson-lanni')
    # born-wolf
    psf_wolf = born_wolf_psf(xx, yy, zz, params, wavelength, ni)
    # gauss
    psf_gauss = gaussian3d_psf(xx, yy, zz, params, wavelength, ni)
    # gauss sampled
    psf_gauss_sampled = gaussian3d_pixelated_psf(nx, dxy, z, params, wavelength, ni, sf=3)

    nrows = np.floor(np.sqrt(nz))
    ncols = np.ceil(nz / nrows)
    fig = plt.figure(figsize=figsize, **kwargs)
    for ii in range(nz):
        ax = plt.subplot(nrows, ncols, ii+1)
        plt.plot(x, psf_vec[ii, nx//2, :])
        plt.plot(x, psf_gl[ii, nx//2, :], '.-')
        plt.plot(x, psf_wolf[ii, nx//2, :], '.-')
        plt.plot(x, psf_gauss[ii, nx//2, :], '.-')
        plt.plot(x, psf_gauss_sampled[ii, nx//2, :], '.-')
        plt.plot([x[0], x[-1]], [0, 0], 'k--')
        plt.ylim([-0.05, 1.2])

        # plt.xlabel('position (um)')
        # plt.title('%0.3fum' % z[ii])
        ax.text(0.25, 0.9, '%0.3fum' % z[ii], horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        if ii == 0:
            plt.legend(['vectorial', 'gibson-lanni', 'born-wolf', 'gauss', 'gauss pixilated'])
            # plt.legend(['vectorial', 'gauss pixilated'])

    plt.suptitle('PSF comparison, NA=%0.2f, wavelength=%0.1fnm, dx=%0.03fum' % (na, wavelength*1000, dxy))

    return fig

def test_fits(wavelength=0.635, na=1.35, ni=1.5, dx=0.065, nx=19, dz=0.1, nz=11, model='gaussian'):
    """
    Test fit functions generating synthetic data, adding noise, and fitting
    :param wavelength: in nm
    :param na:
    :param dxy:
    :param nx:
    :param dz:
    :param nz:
    :param model: 'vectorial', 'gibson-lanni', 'born-wolf', 'gaussian', 'pixelated-gaussian'
    :return:
    """

    # setup
    noise_sigma = 50
    p_gtruth = [375, 1.7 * dx, -1.8 * dx, 0.8 * dz, na, 120]

    # get coordinates
    z, y, x = get_psf_coords([nz, nx, nx], [dz, dx, dx])
    yy, zz, xx = np.meshgrid(y, z, x)

    # noise
    noise = noise_sigma * np.random.randn(xx.shape[0], xx.shape[1], xx.shape[2])

    # different fit options
    if model == 'gaussian':
        img_gtruth = gaussian3d_psf(xx, yy, zz, p_gtruth, wavelength, ni)
        img_noisy = img_gtruth + noise
        result, ffn = fit_cont_psfmodel(img_noisy, wavelength, ni, model='gaussian', xx=xx, yy=yy, zz=zz)
    elif model == 'sampled-gaussian':
        img_gtruth = gaussian3d_pixelated_psf(nx, dx, z, p_gtruth, wavelength, ni, sf=3)
        img_noisy = img_gtruth + noise
        result, ffn = fit_pixelated_psfmodel(img_noisy, dx, dz, wavelength, ni, sf=3, model='gaussian')
    elif model == 'born-wolf':
        img_gtruth = born_wolf_psf(xx, yy, zz, p_gtruth, wavelength, ni)
        img_noisy = img_gtruth + noise
        result, ffn = fit_cont_psfmodel(img_noisy, wavelength, ni, model='born-wolf', xx=xx, yy=yy, zz=zz)
    elif model == 'gibson-lanni':
        img_gtruth = model_psf(nx, dx, z, p_gtruth, wavelength, ni, model='gibson-lanni')
        img_noisy = img_gtruth + noise
        result, ffn = fit_pixelated_psfmodel(img_noisy, dx, dz, wavelength, ni, model='gibson-lanni')
    elif model == 'vectorial':
        img_gtruth = model_psf(nx, dx, z, p_gtruth, wavelength, ni, model='vectorial')
        img_noisy = img_gtruth + noise
        result, ffn = fit_pixelated_psfmodel(img_noisy, dx, dz, wavelength, ni, model='vectorial')
    else:
        raise ValueError("model must be 'gaussian', 'sampled-gaussian', 'born-wolf', 'gibson-lanni', or 'vectorial', but was %s" % model)
    fitp = result['fit_params']
    chi_sqr = result['chi_squared']
    cov = result['covariance']

    data = {'dx': dx, 'nx': nx, 'dz': dz, 'nz': nz, 'wavelength': wavelength, 'model': model,
            'noise sigma': noise_sigma, 'params ground truth': p_gtruth,
            'fit parameters': fitp, 'reduced chi squared': chi_sqr, 'covariances': cov,
            'ground truth image': img_gtruth, 'noisy image': img_noisy}

    figs, fnames = plot_psf3d(img_noisy, dx, dz, wavelength, ni, fitp, chi_sqr, model=model, cov=cov)

    pnames = ['A', 'cx', 'cy', 'cz', 'na', 'bg']
    print('truth, fit, uncertainty, err')
    print('-----------------------------')
    for ii in range(fitp.size):
        print('%s: %0.3f, %0.3f, %0.3f, %0.3g' % (pnames[ii], p_gtruth[ii], fitp[ii], np.sqrt(cov[ii, ii]), p_gtruth[ii] - fitp[ii]))

    return figs, fnames, data

def compare_psf_eval_time(nreps=5):
    """
    Compare evaluation time for different psf models
    :param nreps:
    :return:
    """

    # setup
    setup = '''
    import numpy as np
    import psfmodels as psfm 
    import fit_psf

    na = 1.3
    ni = 1.5
    wavelength = 0.662 #um
    nz = 17
    dz = 0.1
    nx = 25
    dx = 0.11

    z = dz * (np.arange(nz) - nz//2)
    x = dx * (np.arange(nx) - nx//2)
    yy, zz, xx = np.meshgrid(x, z, x)
    '''

    # psf models
    tgl = timeit.Timer(stmt='psf_gl = psfm.scalar_psf(z, nx, dx, wvl=wavelength, params={"NA":na})', setup=setup).repeat(
        repeat=nreps, number=1)
    tv = timeit.Timer(stmt='psf_v = psfm.vectorial_psf(z, nx, dx, wvl=wavelength, params={"NA":na})', setup=setup).repeat(
        repeat=nreps, number=1)

    stmat = '''
    params = [1, 0, 0, 0, na, 0]
    psf_bw = fit_psf.born_wolf_psf(xx, yy, zz, params, wavelength=wavelength, ni=ni)
    '''
    tbw = timeit.Timer(stmt=stmat, setup=setup).repeat(repeat=nreps, number=1)

    stmt = '''
    params = [1, 0, 0, na, 0]
    psf_airy = fit_psf.airy_fn(xx, yy, params, wavelength=wavelength)
    '''
    tairy = timeit.Timer(stmt=stmt, setup=setup).repeat(repeat=nreps, number=1)

    print('min time Gibson-Lanni = %0.9f' % np.min(tgl))
    print('min time Vectorial = %0.9f' % np.min(tv))
    print('min time Born-Wolf numerical integration = %0.9f' % np.min(tbw))
    print('min time Airy function = %0.9f' % np.min(tairy))
