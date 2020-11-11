"""
Fit point spread functions using a variety of models
"""
import os
import timeit
import pickle
import copy

import numpy as np
from scipy import fft
import scipy.ndimage as ndi
import scipy.special as sp
import scipy.integrate
import scipy.interpolate
from scipy import fft
from skimage.feature import peak_local_max
import skimage.filters

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm
import matplotlib.gridspec
import matplotlib.cm
import joblib
from functools import partial

import psfmodels as psfm

import analysis_tools as tools
# from . import analysis_tools as tools

def get_background(img, npix):
    """
    Use polynomial fit across subregions of image to model background.

    :param img: a 2D image, size ny, nx
    :param npix: size of image blocks to do background fitting on
    :return bg:
    """

    ny, nx = img.shape
    xx, yy = np.meshgrid(range(nx), range(ny))
    bg = np.zeros((ny, nx))

    for ii in range(int(np.ceil(ny/npix))):
        for jj in range(int(np.ceil(nx/npix))):
            xstart = jj * npix
            xend = np.min([(jj+1) * npix, nx])
            ystart = ii * npix
            yend = np.min([(ii+1) * npix, ny])

            # linear lest squares
            # dependent variable matrix
            x = xx[ystart:yend, xstart:xend].ravel()[:, None]
            y = yy[ystart:yend, xstart:xend].ravel()[:, None]
            one_vec = np.ones(y.shape)
            Xmat = np.concatenate((x, y, one_vec), axis=1)
            # data vector
            d = img[ystart:yend, xstart:xend].ravel()[:, None]
            # solve
            coeffs, _, _, _ = np.linalg.lstsq(Xmat, d)
            bg[ystart:yend, xstart:xend] = coeffs[0, 0] * xx[ystart:yend, xstart:xend] + \
                                           coeffs[1, 0] * yy[ystart:yend, xstart:xend] + \
                                           coeffs[2, 0]

    return bg

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
            raise Exception("if defocus != 0, dx, wavelength, ni must be provided")

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
def gaussian_psf(x, y, p, wavelength):
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
        raise Exception("pixel size must be odd.")

    # we will add amplitude and background back in at the end
    pt = [1, p[1], p[2], p[3], p[4], 0]

    y, x, = get_coords([nx * sf, nx * sf], [dx / sf, dx / sf])
    yy, zz, xx = np.meshgrid(y, z, x)
    psf_exp = gaussian3d_psf(xx, yy, zz, pt, wavelength, ni)

    # sum points in pixel
    psf = tools.bin(psf_exp, [sf, sf], mode='mean')

    # get psf norm
    ycs, xcs = get_coords([sf, sf], [dx / sf, dx / sf])
    yyc, zzc, xxc = np.meshgrid(ycs, 0, xcs)
    psf_norm = np.mean(gaussian3d_psf(xxc, yyc, zzc, [1, 0, 0, 0, p[4], 0], wavelength, ni))

    # psf normalized to one
    psf = p[0] * psf / psf_norm + p[5]

    return psf

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
    General functiion for evaluating different psfmodels. For vectorial or gibson-lanni PSF's, this wraps the functions
    in PSFmodels. For gaussian, it wraps the gaussian3d_pixelated_psf() function.

    todo: need to implement index of refraction? Not sure this matters...

    :param nx: number of points to be sampled in x- and y-directions
    :param dxy: pixel size in um
    :param z: z positions in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param wavelength: wavelength in um
    :param model: 'gaussian', 'gibson-lanni', or 'vectorial'
    :return:
    """
    if 'NA' in kwargs.keys():
        raise Exception("'NA' is not allowed to be passed as a named parameter. It is specified in p.")

    model_params = {'NA': p[4], 'sf': sf}
    model_params.update(kwargs)

    if model == 'vectorial':
        if sf != 1:
            raise Exception('vectorial model not implemented for sf=/=1')
        psf_norm = psfm.vectorial_psf(0, 1, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = psfm.vectorial_psf(z - p[3], nx, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = p[0] / psf_norm * ndi.shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]
    elif model == 'gibson-lanni':
        if sf != 1:
            raise Exception('gibson-lanni model not implemented for sf=/=1')
        psf_norm = psfm.scalar_psf(0, 1, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = psfm.scalar_psf(z - p[3], nx, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = p[0] / psf_norm * ndi.shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]
    elif model == 'gaussian':
        val = gaussian3d_pixelated_psf(nx, dxy, z, p, wavelength, ni, sf)
    else:
        raise Exception("model must be 'gibson-lanni', 'vectorial', or 'gaussian' but was '%s'" % model)

    return val

# pupil and phase retrieval
class pupil():
    def __init__(self, nx, dx, wavelength, na, psf, zs=np.array([0]), n=1.5, mag=100, mode='abbe'):
        """

        :param nx:
        :param dx:
        :param wavelength:
        :param na:
        :param psf:
        :param zs:
        :param n: index of refraction in object space
        :param mag: magnification
        :param mode: 'abbe' or 'herschel'
        """
        self.nx = nx
        self.dx = dx
        self.wavelength = wavelength
        self.n = n
        self.mag = mag
        self.na = na
        self.psf = psf
        self.zs = zs
        if self.zs.size != self.psf.shape[0]:
            raise Exception("first dimension of psf does not match z size")
        # efield max frequency
        self.fmax = self.na / self.wavelength

        #
        self.x = self.dx * (np.arange(self.nx) - self.nx // 2)
        self.y = self.x

        # frequency data
        self.fx = tools.get_fft_frqs(self.nx, self.dx)
        self.fy = tools.get_fft_frqs(self.nx, self.dx)
        self.fxfx, self.fyfy = np.meshgrid(self.fx, self.fy)
        self.ff = np.sqrt(self.fxfx**2 + self.fyfy**2)

        # we must divide this by the magnification, because we are working in image space (i.e. we typically write
        # all of our integrals after making the transforming x-> x/(M*n) and kx -> kx*M*n, which makes the dimensions
        # of image space the same as object space
        kperp = np.sqrt((2 * np.pi * self.fxfx) ** 2 + (2 * np.pi * self.fyfy) ** 2) / self.mag
        keff = 2 * np.pi / self.wavelength
        self.sin_tp = kperp / keff
        self.cos_tp = np.sqrt(1 - self.sin_tp**2)
        if mode == 'abbe':
            self.apodization = self.cos_tp ** (-1/2) * (1 - (self.mag / self.n * self.sin_tp)**2) ** (-1/4)
        elif mode == 'herschel':
            # 1 / cos(theta')
            self.apodization = self.cos_tp ** (-1)
        elif mode == 'none':
            self.apodization = 1
        else:
            raise Exception("mode must be 'abbe' or 'herschel', but was '%s'" % mode)

        # initialize pupil with random phases
        # absorb apodization into pupil
        self.pupil = self.apodization * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(nx, nx)))
        self.pupil[self.ff > self.fmax] = 0

        self.iteration = 0
        self.mean_err = []

        # use norm so we can keep the pupil magnitude fixed at 1, no matter what the normalization of the PSF is.
        # based on fact we expect sum_f |g(f)|^2 / N = sum_r |g(r)|^2 = sum_r psf(r)
        # this norm is not changed by any operation
        # actually not totally true, because the pupil cropping operation can change it...
        #self.norm = np.sqrt(np.sum(np.abs(self.pupil)**2) / self.nx**2 / np.sum(self.psf))
        self.norm = 1

    def get_defocus(self, z):
        """

        :param z:
        :return:
        """
        # kz = (self.mag**2 * self.n) * (2*np.pi / self.wavelength) * self.cos_tp
        kz = (self.mag ** 2 / self.n) * (2 * np.pi / self.wavelength) * self.cos_tp
        kz[np.isnan(kz)] = 0
        return np.exp(-1j * kz * z)

    def get_amp_psf(self, z, normalize=True):
        """

        :param z:
        :return:
        """
        # recall that iffshift is the inverse of fftshift. Since we have centered pupil with fftshift, now
        # need to uncenter it.
        amp_psf = fft.fftshift(fft.ifft2(fft.ifftshift(self.pupil * self.get_defocus(z))))
        if normalize:
            amp_psf = amp_psf / self.norm
        return amp_psf

    def get_pupil(self, amp_psf, normalize=True):
        """

        :param amp_psf:
        :return:
        """
        pupil = fft.fftshift(fft.fft2(fft.ifftshift(amp_psf)))
        if normalize:
            pupil = pupil * self.norm
        return pupil

    def iterate_pupil(self):
        """

        :return:
        """

        # get amplitude psf
        psf_e = np.zeros(self.psf.shape, dtype=np.complex)
        psf_e_new = np.zeros(self.psf.shape, dtype=np.complex)
        pupils_new_phase = np.zeros(self.psf.shape, dtype=np.complex)
        for ii in range(self.zs.size):
            psf_e[ii] = self.get_amp_psf(self.zs[ii])
            # new amplitude psf from phase of transformed pupil and amplitude of measured psf
            psf_e_new[ii] = np.sqrt(self.psf[ii]) * np.exp(1j * np.angle(psf_e[ii]))
            # weird issue with numpy square root for very small positive numbers
            # think probably related to https://github.com/numpy/numpy/issues/11448
            psf_e_new[ii][np.isnan(psf_e_new[ii])] = 0

            # get new pupil by transforming, then undoing defocus
            xform = self.get_pupil(psf_e_new[ii]) * self.get_defocus(self.zs[ii]).conj()
            pupils_new_phase[ii] = np.angle(xform)

        # get error
        self.mean_err.append(np.nanmean(np.abs(np.abs(psf_e)**2 - self.psf) / np.nanmax(self.psf)))

        # pupil_new = scipy.ndimage.gaussian_filter(pupil_new_mag, sigma=1) * np.exp(1j * pupil_new_phase)
        phase = np.angle(np.mean(np.exp(1j * pupils_new_phase), axis=0))
        self.pupil = np.abs(self.pupil) * np.exp(1j * phase)
        # this should already be enforced by the initial pupil, but can't hurt
        self.pupil[self.ff > self.fmax] = 0
        self.iteration += 1

    def show_current_pupil(self):

        extent_real = [self.x[0] - 0.5*self.dx, self.x[-1] + 0.5*self.dx, self.y[-1] + 0.5*self.dx, self.y[0] - 0.5*self.dx]

        df = self.fx[1] - self.fx[0]
        extent_ft = [self.fx[0] - 0.5*df, self.fx[-1] + 0.5*df, self.fy[-1] + 0.5*df, self.fy[0] - 0.5*df]

        #
        psf_e = np.zeros(self.psf.shape, dtype=np.complex)
        for ii in range(self.zs.size):
            psf_e[ii] = self.get_amp_psf(self.zs[ii])

        psf_i = np.abs(psf_e) ** 2

        figh = plt.figure()
        plt.suptitle('iteration = %d' % self.iteration)
        nrows = 3
        ncols = self.zs.size

        for ii in range(ncols):
            plt.subplot(nrows, ncols, ii + 1)
            plt.imshow(self.psf[ii] / np.nanmax(self.psf), extent=extent_real)
            plt.title('PSF / max at z=%0.3fum' % self.zs[ii])

            plt.subplot(nrows, ncols, ncols + ii + 1)
            plt.imshow(psf_i[ii] / np.nanmax(psf_i), extent=extent_real)
            plt.title('PSF from pupil / max')

            plt.subplot(nrows, ncols, 2*ncols + ii + 1)
            plt.imshow((self.psf[ii] - psf_i[ii]) / np.nanmax(self.psf))
            plt.title('(PSF - PSF from pupil) / max(psf')
            plt.colorbar()

        figh = plt.figure()
        nrows = 2
        ncols = 3
        zind = np.argmin(np.abs(self.zs))

        plt.subplot(nrows, ncols, 1)
        plt.imshow(np.abs(psf_e[zind]), extent=extent_real)
        plt.title('PSF amp, magnitude')

        plt.subplot(nrows, ncols, 4)
        plt.imshow(np.angle(psf_e[zind]) / np.pi, vmin=-1, vmax=1, extent=extent_real)
        plt.title('PSF phase (pi)')

        plt.subplot(nrows, ncols, 2)
        plt.imshow(np.abs(self.pupil), extent=extent_ft)
        plt.xlim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.ylim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.title('Pupil, magnitude')

        plt.subplot(nrows, ncols, 5)
        phase = np.unwrap(np.angle(self.pupil))
        phase[self.ff > self.fmax] = np.nan

        plt.imshow(phase / np.pi, vmin=-1, vmax=1, extent=extent_ft)
        plt.xlim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.ylim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.title('Pupil, phase/pi')

        plt.subplot(nrows, ncols, 3)
        plt.semilogy(self.mean_err)
        plt.xlabel('iteration')
        plt.ylabel('mean PSF err / max(PSF)')

# pixelation function
def pixelate_model(nx, dxy, z, p, wavelength, fn, ni=1.5, sf=1):
    # todo:
    pass

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
    :return:
    """

    # select fitting model
    if model == 'gaussian':
        model_fn = lambda x, y, z, p: gaussian3d_psf(x, y, z, p, wavelength, ni)
    elif model == 'born-wolf':
        model_fn = lambda x, y, z, p: born_wolf_psf(x, y, z, p, wavelength, ni)
    else:
        raise Exception("model must be 'gaussian' or 'born-wolf' but was '%s'" % model)

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
        cz, cy, cx = tools.get_moments(img, order=1, coords=[zz[:, 0, 0], yy[0, :, 0], xx[0, 0, :]])

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

    # fitting
    result = tools.fit_model(img, lambda p: model_fn(xx, yy, zz, p), init_params,
                             fixed_params=fixed_params, sd=sd, bounds=bounds)

    # model function at fit parameters
    pfit = result['fit_params']
    chi_sq = result['chi_squared']
    cov = result['covariance']
    fit_fn = lambda x, y, z: model_fn(x, y, z, pfit)

    #return pfit, fit_fn, chi_sq, cov
    return result, fit_fn

def fit_pixelated_psfmodel(img, dxy, dz, wavelength, ni, sf=1, model='vectorial',
                           init_params=None, fixed_params=None, sd=None, bounds=None):
    """
    3D non-linear least squares fit using one of the point spread function models from psfmodels package.

    # todo: make sure ni implemented correctly. if want to use different ni, have to be careful because this will
    # shift the focus position away from z=0
    # todo: make over sampling setable and ensure this works correctly with all functions
    # todo: pixelated version of gaussian and born-wolf (b-w will be very slow!)

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
    z, y, x = get_coords(img.shape, [dz, dxy, dxy])

    # check size
    nz, ny, nx = img.shape
    if not ny == nx:
        raise Exception('x- and y-size of img must be equal')

    if not np.mod(nx, 2) == 1:
        raise Exception('x- and y-size of img must be odd')

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

        cz, cy, cx = tools.get_moments(img, order=1, coords=[z, y, x])

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

    # fit function
    if model == 'vectorial' or model =='gibson-lanni' or model=='gaussian':
        model_fn = lambda z, nx, dxy, p: model_psf(nx, dxy, z, p, wavelength, ni, sf=sf, model=model)
    else:
        raise Exception("Model should be 'vectorial', 'gibson-lanni', or 'gaussian', but was '%s'" % model)

    # fitting
    result = tools.fit_model(img, lambda p: model_fn(z, nx, dxy, p), init_params,
                             fixed_params=fixed_params, sd=sd, bounds=bounds, jac='3-point', x_scale='jac')

    # model function at fit parameters
    fit_fn = lambda z, nx, dxy: model_fn(z, nx, dxy, result['fit_params'])

    return result, fit_fn

# utility functions
def get_coords(ns, drs):
    """
    Get centered coordinates from step size and number of coordinates
    :param ns: list of number of points
    :param drs: list of step sizes
    :return coords: list of coordinates [[c1, c2, c3, ...], ...]
    """
    return [d * (np.arange(n) - n //2) for n, d in zip(ns, drs)]

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
        x = get_coords(nx, dx)
        xx, yy = np.meshgrid(x, x)
        psf_design = airy_fn(xx, yy, [1, 0, 0, design_na, 0], wavelength=wavelength)[None, :, :]
        psf_fit = airy_fn(xx, yy, [1, 0, 0, fit_na, 0], wavelength=wavelength)[None, :, :]
    else:
        raise Exception("model must be 'born-wolf', 'gibson-lanni', or 'vectorial', but value was %s" % model)

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
    z, y, x, = get_coords([nz, ny, nx], [dz, dx, dx])

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
            raise Exception("model must be 'gaussian', 'born-wolf', 'gibson-lanni', or 'vectorial', but was '%s'." % m)

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
    spec = matplotlib.gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=figh)

    gamma = 0.1
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
    plt.imshow(imgs[:, cy_pix3d, :], vmin=vmin, vmax=vmax, cmap='bone', extent=extent, norm=PowerNorm(gamma=gamma))
    plt.title("XZ power norm")

    ax = figh.add_subplot(spec[0, 4:6])
    plt.imshow(imgs[:, :, cx_pix3d], vmin=vmin, vmax=vmax, cmap='bone', extent=extent)
    plt.ylabel('Z (um)')
    plt.xlabel('Y (um)')
    plt.title('YZ plane')

    ax = figh.add_subplot(spec[0, 6:8])
    plt.imshow(imgs[:, :, cx_pix3d], vmin=vmin, vmax=vmax, cmap='bone', extent=extent, norm=PowerNorm(gamma=gamma))
    plt.title('YZ power norm')

    for ii in range(nfits):
        # normal scale
        ax = figh.add_subplot(spec[ii + 1, 0:2])
        im = plt.imshow(fit_img3d[ii][:, cy_pix3d, :], vmin=vmin, vmax=vmax, cmap='bone', extent=extent)
        plt.title('%s, sf=%d, NA=%0.3f' % (model[ii], sfs[ii], fit_params[ii][4]))
        if ii < (nfits - 1):
            plt.setp(ax.get_xticklabels(), visible=False)

        # power law scaled, to emphasize smaller features
        ax = figh.add_subplot(spec[ii + 1, 2:4])
        plt.imshow(fit_img3d[ii][:, cy_pix3d, :], vmin=vmin, vmax=vmax, cmap='bone', extent=extent, norm=PowerNorm(gamma=gamma))
        if ii < (nfits - 1):
            plt.setp(ax.get_xticklabels(), visible=False)

        # other cut
        ax = figh.add_subplot(spec[ii + 1, 6:8])
        im = plt.imshow(fit_img3d[ii][:, :, cx_pix3d], vmin=vmin, vmax=vmax, cmap='bone', extent=extent,
                        norm=PowerNorm(gamma=gamma))
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
def get_exp_psf(imgs, dx, dz, fit3ds, rois, model='vectorial'):
    """
    Get experimental psf from imgs and the results of autofit_psfs

    :param imgs:z-stack of images
    :param dx: pixel size (um)
    :param dz: spacing between image planes (um)
    :param fit_params: n x 6 array, [A, cx, cy, cz, NA, bg]
    :param rois: regions of interest

    :return psf_mean:
    :return psf_sd:
    """

    fit_params = np.asarray([f['fit_params'] for f in fit3ds])
    chi_sqrs = np.asarray([f['chi_squared'] for f in fit3ds]).transpose()

    # get size
    nroi = rois[0][3] - rois[0][2]

    # maximum z size
    nzroi = np.max([r[1] - r[0] for r in rois])
    zpsf = get_coords([nzroi], [dz])[0]
    izero = np.argmin(np.abs(zpsf))
    if not np.abs(zpsf[izero]) < 1e-10:
        raise Exception("z coordinates do not include zero")

    psf_shifted = np.zeros((len(rois), nzroi, nroi, nroi)) * np.nan
    weights = np.zeros((len(rois)))
    # loop over rois and shift psfs so they are centered
    for ii, (fp, roi) in enumerate(zip(fit_params, rois)):
        # get roi
        img_roi = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

        # get coordinates
        nz = img_roi.shape[0]
        z, y, x = get_coords(img_roi.shape, [dz, dx, dx])
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

        # subtract background
        # old way, but do not like because will not average noise in background nicely to zero. Essentially relying on
        # having most spots with similar amplitudes
        # psf_shifted[ii] = (psf_shifted[ii] - fp[5]) / fp[0]
        # not sure that chi squared is such a useful thing. But assuming all fits look very similar, probably ok.
        # weights[ii] = fp[0] / chi_sqrs[ii]

        psf_shifted[ii] = (psf_shifted[ii] - fp[5])
        # weights[ii] = 1

    # weighted averaging
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     psf_weighted = np.array([p * w for p, w in zip(psf_shifted, weights)])
    #     weights_nan = np.transpose(np.tile(weights, [nzroi, nroi, nroi, 1]), [3, 0, 1, 2])
    #     weights_nan[np.isnan(psf_weighted)] = np.nan
    #     psf_mean = np.nansum(psf_weighted, axis=0) / np.nansum(weights_nan, axis=0)
    #
    #     # reliability weighted standard error
    #     v1 = np.nansum(weights_nan, axis=0)
    #     v2 = np.nansum(weights_nan ** 2, axis=0)
    #     numerator = np.asarray([w * (pshift - psf_mean)**2 for w, pshift in zip(weights, psf_shifted)])
    #     psf_sd = np.nansum(numerator, axis=0) / (v1 - v2/v1)

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
        psf, psf_sd = get_exp_psf(imgs, dx, dz, fit3ds_psf, rois_psf)
        exp_psfs.append(psf)
        exp_psfs_sd.append(psf_sd)

        # get coordinates
        nz, ny, nx = psf.shape

        # Fit to 2D PSFs
        z, y, x = get_coords(psf.shape, [dz, dx, dx])
        iz = np.argmin(np.abs(z))
        ixz = np.argmin(np.abs(x))
        psf2d = psf[iz]
        psf2d_sd = psf_sd[iz]

        bounds = ((0, -4*dx, -4*dx, -1e-12, 0.2, -np.inf),
                  (np.inf, 4*dx, 4*dx, 1e-12, ni, np.inf))

        init_params = [None, None, None, 0, None, None]
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
            circ = matplotlib.patches.Circle((0, 0), radius=fmax, color='r', fill=0)
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
def find_candidate_beads(img, filter_xy_pix=1, filter_z_pix=0.5, min_distance=1, abs_thresh_std=1, max_num_peaks=300):
    """
    Find candidate beads in image. Based on function from mesoSPIM-PSFanalysis
    :return:
    """

    # filter to limit of NA
    #res = 0.5 * wavelength / NA
    # half because this is std, not full width
    # filter_size_pix = 0.5 * res/dx
    #filter_size_pix = 1

    # todo: think might be better to do some filtering in z direction also
    if img.ndim == 3:
        smoothed = skimage.filters.gaussian(img, [filter_z_pix, filter_xy_pix, filter_xy_pix],
                                            output=None, mode='nearest', cval=0,
                                            multichannel=None, preserve_range=True)

        # if img.shape[0] > 1:
        #     exclude_border = (True, True, True)
        # else:
        #     exclude_border = (False, True, True)

    else:
        smoothed = skimage.filters.gaussian(img, filter_xy_pix, output=None, mode='nearest', cval=0,
                                            multichannel=None, preserve_range=True)
        # exclude_border = (0, 1, 1)

    abs_threshold = smoothed.mean() + abs_thresh_std * img.std()
    centers = peak_local_max(smoothed, min_distance=min_distance, threshold_abs=abs_threshold,
                             exclude_border=False, num_peaks=max_num_peaks)
    return centers

def find_beads(imgs, imgs_sd, dx, dz, window_size_um=(1, 1, 1), min_sigma_pix=0.7,
               filter_xy_pix=1, filter_z_pix=0, min_distance=1, abs_thresh_std=1, max_sep_assume_one_peak=2,
               max_percent_asymmetry=0.5,
               max_num_peaks=np.inf, remove_background=False, require_isolated=True):
    """
    Identify beads in a 3D image. This is done in three steps.

    1. The 2D planes of the image are smoothed and a peak finding algorithm is applied, which returns up to a user
    specified number of the largest peaks. For best results, this number should be somewhat larger than the total number
    of expected beads in the image.
    2. Each peak candidate is fit to a Gaussian function, and peaks with implausible fit parameters are excluded.
    3. Remaining beads are excluded if more than falls within the same region of interest.

    :param imgs: nz x nx x ny image
    :param imgs_sd: standard deviations, assuming img is an average of other images, otherwise can be none
    :param dx: pixel size in um
    :param dz: z-plane spacing in um
    :param window_size_um: ROI size in real units (nz_um, ny_um, nx_um)
    :param min_sigma_pix: only points with gaussian standard deviation larger than this value will be considered
    so we can avoid e.g. hot pixels
    :param filter_xy_pix: standard deviation of gaussian filter applied in x- and y-directions
    :param filter_z_pix: standard deviation of gaussian filter applied in z-direction
    :param min_distance: minimum distance between peaks, in pixels
    :param float abs_thresh_std:
    :param int max_num_peaks: maximum number of peaks to find
    :param bool remove_background: boolean. whether or not to remove background from image before peak finding.
    :return rois: list of regions of interest [[zstart, zend, ystart, yend, xstart, xend], ...] as coordinates in imgs
    :return centers: list of centers [[cz, cy, cx], ....] as coordinates in imgs
    :return fit_params: nroi x 7, where each row is of the form [A, cx, cy, sx, sy, bg]. The coordinates sx are given
    relative to the region of interest. So the center for the bead is at (cx + x_roi_start, cy + y_roi_start)
    """

    if imgs.ndim == 2:
        imgs = imgs[None, :, :]

    _, ny, nx = imgs.shape

    # ##############################
    # get ROI sizes
    # ##############################
    nx_roi = np.round(window_size_um[2] / dx)
    if np.mod(nx_roi, 2) == 0:
        nx_roi = nx_roi + 1

    # using square ROI
    ny_roi = nx_roi

    # don't care if z-roi size is odd
    nz_roi = np.round(window_size_um[0] / dz)

    # ##############################
    # remove background
    # ##############################
    if remove_background:
        # print("Removing background prior to peakfinding")
        # get_bg_partial = partial(get_background, npix=200)
        # bgs = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
        #     joblib.delayed(get_bg_partial)(imgs[ii]) for ii in range(imgs.shape[0])
        # )
        # imgs = imgs - np.array(bgs)
        filter_pix_bg = nx_roi
        bg = skimage.filters.gaussian(imgs, [0, filter_pix_bg, filter_pix_bg],
                                            output=None, mode='nearest', cval=0,
                                            multichannel=None, preserve_range=True)
        imgs = imgs - bg
    else:
        bg = 0


    # todo: maybe useful to set filter_size_pix based on expected NA?
    # ##############################
    # find plausible peaks
    # ##############################
    centers = find_candidate_beads(imgs, filter_xy_pix=filter_xy_pix, filter_z_pix=filter_z_pix,
                                   min_distance=min_distance, abs_thresh_std=abs_thresh_std, max_num_peaks=max_num_peaks)
    print("Found %d candidates" % len(centers))

    # get ROI's for each peak
    rois = np.array([tools.crop_roi(tools.get_centered_roi(c, [nz_roi, ny_roi, nx_roi]), imgs.shape) for c in centers])

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
                joblib.delayed(tools.fit_gauss)(imgs[centers[ii, 0], rois[ii, 2]:rois[ii, 3], rois[ii, 4]:rois[ii, 5]],
                                                sd=imgs_sd[centers[ii, 0], rois[ii, 2]:rois[ii, 3], rois[ii, 4]:rois[ii, 5]],
                                                init_params=[None, c_rois[ii, 2], c_rois[ii, 1], 0.5, 0.5, 0, 0])
                                                for ii in range(len(rois))
                )
    else:
        results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
            joblib.delayed(tools.fit_gauss)(imgs[centers[ii, 0], rois[ii, 2]:rois[ii, 3], rois[ii, 4]:rois[ii, 5]],
                                            init_params=[None, c_rois[ii, 2], c_rois[ii, 1], 0.5, 0.5, 0, 0])
            for ii in range(len(rois))
        )

    results = list(zip(*results))[0]
    gauss_fitp = np.asarray([r['fit_params'] for r in results])
    chi_sqrs = np.asarray([r['chi_squared'] for r in results])

    # exclude peaks if too little weight
    min_peak_amp = np.std(imgs) * abs_thresh_std
    big_enough = gauss_fitp[:, 0] >= min_peak_amp

    # exclude points that are not very symmetric
    asymmetry = np.abs(gauss_fitp[:, 3] - gauss_fitp[:, 4]) / (0.5 * (gauss_fitp[:, 3] + gauss_fitp[:, 4]))
    is_symmetric = asymmetry < max_percent_asymmetry

    # exclude points too far from center of ROI
    max_off_center_fractional = 0.25
    on_center = np.logical_and(np.abs(gauss_fitp[:, 1] - 0.5 * nx_roi) / nx_roi < max_off_center_fractional,
                               np.abs(gauss_fitp[:, 2] - 0.5 * nx_roi) / nx_roi < max_off_center_fractional)
    # exclude points if sigma too large
    max_sigma = nx_roi / 2
    not_too_big = np.logical_and(gauss_fitp[:, 3] < max_sigma,
                                 gauss_fitp[:, 4] < max_sigma)
    # exclude points if sigma too small
    not_too_small = np.logical_and(gauss_fitp[:, 3] > min_sigma_pix, gauss_fitp[:, 4] > min_sigma_pix)

    # filter on chi squares
    # todo: removed this because seems to fail mostly for bright peaks.
    # chi_sq_ok = chi_sqrs < np.mean(chi_sqrs) + 2 * np.std(chi_sqrs)

    # combine all conditions and reduce centers/rois
    geometry_ok = np.logical_and(is_symmetric, on_center)
    size_ok = np.logical_and(not_too_big, not_too_small)
    to_use = np.logical_and(geometry_ok, size_ok)
    to_use = np.logical_and(to_use, big_enough)
    # to_use = np.logical_and(to_use, chi_sq_ok)

    centers = centers[to_use]
    rois = rois[to_use]
    gauss_fitp = gauss_fitp[to_use]
    print("%d candidates with plausible fit parameters" % len(centers))

    # ##############################
    # find sets of peaks so close together probably fitting the same peak. Only keep one each
    # ##############################
    to_use = np.ones(len(centers), dtype=np.bool)
    inds = np.arange(len(centers))
    for ii in range(len(centers)):
        if not to_use[ii]:
            continue

        d2ds = np.linalg.norm(centers[ii, 1:] - centers[:, 1:], axis=1)
        inds_very_close = inds[d2ds < max_sep_assume_one_peak]
        # only keep index with maximum amplitudes for fit
        ind_to_keep = inds_very_close[np.argmax(gauss_fitp[inds_very_close, 0])]
        to_use[inds_very_close] = False
        to_use[ind_to_keep] = True

    centers = centers[to_use]
    rois = rois[to_use]
    fit_params = gauss_fitp[to_use]
    print("%d points remain after removing likely duplicates" % len(centers))

    # ##############################
    # discards points too close together (only look at 2D distance)
    # ##############################
    if require_isolated:
        min_sep = window_size_um[2] / dx
        min_dists = np.array([min_dist(c[1:], centers[:, 1:]) for c in centers])
        dists_ok = min_dists > min_sep
        rois = rois[dists_ok]
        centers = centers[dists_ok]
        fit_params = fit_params[dists_ok]
        print("%d isolated candidates" % len(centers))

    return rois, centers, fit_params, bg

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
    z, y, x = get_coords(sub_img.shape, [dz, dx, dx])

    # find z-plane closest to fit center and fit gaussian to get initial parameters for model fit
    c_roi = tools.full2roi(center, roi)
    result, _ = fit_pixelated_psfmodel(sub_img[c_roi[0]][None, :, :], dx, dz, wavelength, ni=ni, sf=sf, model='gaussian',
                                       init_params=[None, None, None, 0, None, None],
                                       fixed_params=[False, False, False, True, False, False])
    result['fit_params'][3] = z[c_roi[0]]

    # full 3D PSF fit
    # na cannot exceed index of refraction...
    cz = z[c_roi[0]]
    bounds = ((0, x.min(), y.min(), np.max([cz - 2.5 * float(dz), z.min()]), 0.2, -np.inf),
              (np.inf, x.max(), y.max(), np.min([cz + 2.5 * float(dz), z.max()]), ni, np.inf))

    # fit 3D psf model
    # todo: add sf
    init_params = result['fit_params']
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
def autofit_psfs(imgs, imgs_sd, dx, dz, wavelength, ni=1.5, model='vectorial', sf=3, **kwargs):
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

    # todo: to account for images with fluctuating background, might want to segment the image and then apply some stages of this?
    rois, centers, _, _ = find_beads(imgs, imgs_sd, dx, dz, **kwargs)

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
    figh2 = plot_bead_locations(imgs, centers, fit3ds, fit2ds, figsize, **kwargs)

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
    grid = plt.GridSpec(2, 4)

    # 3D PSF fits
    if fit3ds is not None:
        # histogram of NAs
        plt.subplot(grid[0, 0])
        edges = np.linspace(0, 1.5, 20)
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

def plot_bead_locations(imgs, centers, fit3ds, fit2ds, figsize=(20, 10), **kwargs):
    """
    Plot bead locations against maximum intensity projection

    :param imgs:
    :param centers:
    :param fit3ds: if None, will not plot
    :param fit2ds: if None, will not plot
    :param figsize:
    :param kwargs:
    :return:
    """

    figh = plt.figure(figsize=figsize, **kwargs)

    plt.title("Maximum intensity projection and NA map")

    max_int_proj = np.max(imgs, axis=0)
    vmin = np.percentile(max_int_proj, 0.05)
    vmax = np.percentile(max_int_proj, 99.99)

    plt.imshow(max_int_proj, vmin=vmin, vmax=vmax, cmap='bone')

    # plot circles with colormap representing NA. Reds for 3D NA, blues for 2D NA
    if fit3ds:
        fp3d = np.asarray([f['fit_params'] for f in fit3ds])

        cmap = matplotlib.cm.get_cmap('Reds')
        plt.scatter(centers[:, 2], centers[:, 1], marker='o', facecolor='none',
                    edgecolors=cmap(fp3d[:, 4] / np.max(fp3d[:, 4])))
        cb1 = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(fp3d[:, 4])), cmap=cmap))
        cb1.set_label("3D fits NA")

    if fit2ds:
        fp2d = np.asarray([f['fit_params'] for f in fit2ds])

        cmap = matplotlib.cm.get_cmap('Blues')
        plt.scatter(centers[:, 2], centers[:, 1], marker='s', facecolor='none',
                    edgecolors=cmap(fp2d[:, 4] / np.max(fp2d[:, 4])))
        cb2 = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(fp2d[:, 4])), cmap=cmap))
        cb2.set_label('2D fits NA')

    if not fit3ds and not fit2ds:
        plt.scatter(centers[:, 2], centers[:, 1], marker='o', facecolor='none',
                    edgecolors='red')

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
    z, y, x = get_coords([nz, ny, nx], [dz, dxy, dxy])
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
    z, y, x = get_coords([nz, nx, nx], [dz, dx, dx])
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
        raise Exception("model must be 'gaussian', 'sampled-gaussian', 'born-wolf', 'gibson-lanni', or 'vectorial', but was %s" % model)
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

if __name__ == "__main__":
    pass
