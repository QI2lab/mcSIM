"""
Tools for reconstructiong optical diffraction tomography (ODT) data
"""
import time
import numpy as np
from numpy import fft
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.patches import Circle, Arc
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interpn
from skimage.restoration import unwrap_phase
import localize_psf.fit as fit
import mcsim.analysis.analysis_tools as tools

_cupy_available = True
try:
    import cupy as cp
except ImportError:
    _cupy_available = False


def get_angular_spectrum_kernel(dz, wavelength, no, shape, drs):
    """

    @param dz:
    @param wavelength:
    @param no:
    @param shape:
    @param drs:
    @return:
    """
    k = 2*np.pi / wavelength
    ny, nx = shape
    dy, dx = drs

    fx = fft.fftshift(fft.fftfreq(nx, dx))
    fy = fft.fftshift(fft.fftfreq(ny, dy))
    fxfx, fyfy = np.meshgrid(fx, fy)

    with np.errstate(invalid="ignore"):
        kernel = np.exp(1j * dz * np.sqrt((k * no)**2 - (2*np.pi * fxfx)**2 - (2*np.pi * fyfy)**2))
        kernel[np.isnan(kernel)] = 0

    return kernel


def propagate_field(efield_start, n_stack, no, drs, wavelength, use_gpu=_cupy_available):
    """
    Propagate electric field through medium with index of refraction n(x, y, z) using the projection approximation.

    @param efield_start: ny x nx array
    @param n_stack: nz x ny x nx array
    @param no: background index of refraction
    @param drs: (dz, dy, dx)
    @param wavelength: wavelength in same units as drs
    @return efield: nz x ny x nx electric field
    """
    n_stack = np.atleast_3d(n_stack)

    k = 2*np.pi / wavelength
    dz, dy, dx = drs
    nz, ny, nx = n_stack.shape

    prop_kernel = get_angular_spectrum_kernel(dz, wavelength, no, n_stack.shape[1:], drs[1:])
    # apodization = np.expand_dims(tukey(nxy, alpha=0.1), axis=0) * np.expand_dims(tukey(nxy, alpha=0.1), axis=1)
    apodization = 1

    # ifftshift these to eliminate doing an fftshift every time
    prop_factor = fft.ifftshift(prop_kernel * apodization)

    # do simulation
    efield = np.zeros((nz, ny, nx), dtype=complex)
    efield[0] = efield_start
    if use_gpu:
        prop_factor = cp.array(prop_factor)
        # note: also tried creating efield on GPU and putting n_stack() on the gpu, but then function took much longer
        # dominated by time to transfer efield back to CPU

    for ii in range(nz - 1):
        # projection approximation
        # propagate through background medium using angular spectrum method
        # then accumulate extra phases in real space
        if use_gpu:
            enow = cp.asarray(efield[ii])
            etemp1 = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fft2(cp.fft.ifftshift(enow)) * prop_factor)) # k-space propagation
            efield[ii + 1] = cp.asnumpy(etemp1 * cp.exp(1j * k * dz * (cp.asarray(n_stack[ii]) - no)))  # real space phase
        else:
            efield[ii + 1] = fft.fftshift(fft.ifft2(
                                 fft.fft2(fft.ifftshift(efield[ii])) * prop_factor  # k-space propagation
                                 )) * \
                                 np.exp(1j * k * dz * (n_stack[ii] - no))  # real space phase


    return efield


# helper functions
def get_fz(fx, fy, no, wavelength):
    """
    Get z-component of frequency given fx, fy

    @param frqs_2d: nfrqs x 2
    @param no: index of refraction
    @param wavelength: wavelength
    @return frqs_3d:
    """

    with np.errstate(invalid="ignore"):
        fzs = np.sqrt(no**2 / wavelength ** 2 - fx**2 - fy**2)

    return fzs


def get_angles(frqs, no, wavelength):
    """
    Convert from frequency vectors to angle vectors. Frequency vectors should be normalized to no / wavelength
    @param frqs: (fx, fy, fz), expect |frqs| = no / wavelength
    @param no: background index of refraction
    @param wavelength:
    @return: theta, phi
    """
    frqs = np.atleast_2d(frqs)

    with np.errstate(invalid="ignore"):
        theta = np.array(np.arccos(np.dot(frqs, np.array([0, 0, 1])) / (no / wavelength)))
        theta[np.isnan(theta)] = 0
        phi = np.angle(frqs[:, 0] + 1j * frqs[:, 1])
        phi[np.isnan(phi)] = 0

    return theta, phi


def angles2frqs(no, wavelength, theta, phi):
    """
    Get frequency vector from angles
    @param no:
    @param wavelength:
    @param theta:
    @param phi:
    @return:
    """
    fz = no / wavelength * np.cos(theta)
    fy = no / wavelength * np.sin(theta) * np.sin(phi)
    fx = no / wavelength * np.sin(theta) * np.cos(phi)
    f = np.stack((fx, fy, fz), axis=1)

    return f


def get_global_phase_shifts(imgs, ref_imgs, thresh=None):
    """
    Given a stack of images and a reference, determine the phase shifts between images, such that
    imgs * np.exp(1j * phase_shift) ~ img_ref

    @param imgs: n0 x n1 x ... x n_{-2} x n_{-1} array
    @param ref_imgs: n_{-m} x ... x n_{-2} x n_{-1} array
    @param thresh: only consider points in images where both abs(imgs) and abs(ref_imgs) > thresh
    @return phase_shifts:
    """
    # broadcast images and references images to same shapes
    imgs, ref_imgs = np.broadcast_arrays(imgs, ref_imgs)

    ny, nx = imgs.shape[-2:]

    # looper over all dimensions except the last two, which are the y and x dimensions respectively
    loop_shape = imgs.shape[:-2]
    phase_shifts = np.zeros(loop_shape + (1, 1))
    nfits = np.prod(loop_shape).astype(int)
    for ii in range(nfits):
        ind = np.unravel_index(ii, loop_shape)

        if thresh is None:
            mask = np.ones(imgs[ind].shape, dtype=bool)
        else:
            mask = np.logical_and(np.abs(imgs[ind]) > thresh, np.abs(ref_imgs) > thresh)

        # todo: could add a more sophisticated fitting function if seemed useful
        def fn(p): return np.abs(imgs[ind] * np.exp(1j * p[0]) - ref_imgs[ind])[mask].ravel()
        # def jac(p): return [(1j * imgs[ind] * np.exp(1j * p[0])).ravel()]
        # s1 = np.mean(imgs[ii])
        # s2 = np.mean(ref_imgs[ii])
        # def fn(p): return np.abs(s1 * np.exp(1j * p[0]) - s2)
        results = fit.fit_least_squares(fn, [0])
        phase_shifts[ind] = results["fit_params"]

    return phase_shifts


# convert between index of refraction and scattering potential
def get_n(scattering_pot, no, wavelength):
    """
    convert from the scattering potential to the index of refraction
    @param scattering_pot: F(r) = - (2*np.pi / lambda)^2 * (n(r)^2 - no^2)
    @param no: background index of refraction
    @param wavelength: wavelength
    @return n:
    """
    k = 2 * np.pi / wavelength
    n = np.sqrt(-scattering_pot / k ** 2 + no ** 2)
    return n


def get_scattering_potential(n, no, wavelength):
    """
    Convert from the index of refraction to the scattering potential

    @param n:
    @param no:
    @param wavelength:
    @return:
    """
    k = 2 * np.pi / wavelength
    sp = - k ** 2 * (n**2 - no**2)
    return sp


def get_rytov_phase(eimgs, eimgs_bg, regularization):
    """
    Compute rytov phase from field and background field. The Rytov phase is \psi_s(r) where
    U_total(r) = exp[\psi_o(r) + \psi_s(r)]
    where U_o(r) = exp[\psi_o(r)] is the unscattered field

    @param eimgs: npatterns x ny x nx
    @param eimgs_bg: same size as eimgs
    @param float regularization: regularization value
    @return psi_rytov:
    """

    if eimgs.ndim < 3:
        raise ValueError("eimgs must be at least 3D")

    if eimgs_bg.ndim < 3:
        raise ValueError("eimgs_bg must be at least 3D")

    # output values
    psi_rytov = np.zeros(eimgs.shape, dtype=complex)

    # broadcast arrays
    eimgs, eimgs_bg = np.broadcast_arrays(eimgs, eimgs_bg)

    loop_shape = eimgs.shape[:-2]
    nloop = np.prod(loop_shape)
    for ii in range(nloop):
        ind = np.unravel_index(ii, loop_shape)

        # convert phase difference from interval [0, 2*np.pi) to [-np.pi, np.pi)
        phase_diff = np.mod(np.angle(eimgs[ind]) - np.angle(eimgs_bg[ind]), 2 * np.pi)
        # phase_diff[phase_diff >= np.pi] = phase_diff[phase_diff >= np.pi] - 2 * np.pi
        phase_diff[phase_diff >= np.pi] -= 2 * np.pi

        # get rytov phase change
        # psi_rytov[aa] = np.log(np.abs(eimg[aa]) / (np.abs(eimg_bg[aa]) + delta)) + 1j * unwrap_phase(phase_diff)
        psi_rytov[ind] = np.log(np.abs(eimgs[ind]) / (np.abs(eimgs_bg[ind]))) + 1j * unwrap_phase(phase_diff)
        psi_rytov[ind][np.abs(eimgs_bg[ind]) < regularization] = 0

    return psi_rytov


# holograms
def unmix_hologram(img: np.ndarray, dxy: float, fmax_int: float, frq_ref: np.ndarray,
                   use_gpu: bool=_cupy_available) -> np.ndarray:
    """
    Given an off-axis hologram image, determine the electric field represented

    @param img: n1 x ... x n_{-3} x n_{-2} x n_{-1} array
    @param dxy: pixel size
    @param fmax_int: maximum frequency where intensity OTF has support
    @param frq_ref: reference frequency (fx, fy)
    @return:
    """
    # FT of image
    if not use_gpu:
        img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    else:
        img_ft = cp.asnumpy(cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(img, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1)))
    ny, nx = img_ft.shape[-2:]

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(nx, dxy))
    fys = fft.fftshift(fft.fftfreq(ny, dxy))
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff_perp = np.sqrt(fxfx ** 2 + fyfy ** 2)

    # compute efield
    efield_ft = tools.translate_ft(img_ft, frq_ref[..., 0], frq_ref[..., 1], drs=(dxy, dxy), use_gpu=use_gpu)
    efield_ft[..., ff_perp > fmax_int / 2] = 0

    return efield_ft


# tomographic reconstruction
def get_reconstruction_sampling(no, na_det, beam_frqs, wavelength, dxy, img_size, z_fov,
                                dz_sampling_factor=1., dxy_sampling_factor=1.):
    """
    Get information about pixel grid scattering potential will be reconstructed on

    @param no: background index of refraction
    @param na_det: numerical aperture of detection objective
    @param beam_frqs:
    @param wavelength: wavelength
    @param dxy: camera pixel size
    @param img_size: (ny, nx) size of hologram images
    @param z_fov: field-of-view in z-direction
    @param dz_sampling_factor: z-spacing as a fraction of the nyquist limit
    @param dxy_sampling_factor: xy-spacing as a fraction of nyquist limit
    @return (dz_sp, dxy_sp, dxy_sp), (nz_sp, ny_sp, nx_sp), (fz_max, fxy_max):
    """
    ny, nx = img_size

    # ##################################
    # set sampling of 3D scattering potential
    # ##################################
    theta, _ = get_angles(beam_frqs, no, wavelength)
    alpha = np.arcsin(na_det / no)
    beta = np.max(theta)

    # maximum frequencies present in ODT
    fxy_max = (na_det + no * np.sin(beta)) / wavelength
    fz_max = no / wavelength * np.max([1 - np.cos(alpha), 1 - np.cos(beta)])

    # generate real-space sampling from Nyquist sampling
    dxy_sp = dxy_sampling_factor * 0.5 * 1 / fxy_max
    dz_sp = dz_sampling_factor * 0.5 / fz_max

    x_fov = nx * dxy  # um
    nx_sp = int(np.ceil(x_fov / dxy_sp) + 1)
    if np.mod(nx_sp, 2) == 0:
        nx_sp += 1

    y_fov = ny * dxy
    ny_sp = int(np.ceil(y_fov / dxy_sp) + 1)
    if np.mod(ny_sp, 2) == 0:
        ny_sp += 1

    nz_sp = int(np.ceil(z_fov / dz_sp) + 1)
    if np.mod(nz_sp, 2) == 0:
        nz_sp += 1

    return (dz_sp, dxy_sp, dxy_sp), (nz_sp, ny_sp, nx_sp), (fz_max, fxy_max)


def get_coords(drs, nrs, expand=False):
    """
    Compute spatial coordinates

    @param drs: (dz, dy, dx)
    @param nrs: (nz, ny, nx)
    @param expand: if False then return z, y, x as 1D arrays, otherwise return as 3D arrays
    @return: coords (z, y, x)
    """
    coords = [dr * (np.arange(nr) - nr // 2) for dr, nr in zip(drs, nrs)]

    if expand:
        coords = np.meshgrid(*coords, indexing="ij")

    return coords


def fwd_model_linear(beam_frqs, no, na_det, wavelength, img_shape, dxy, dxy_sp, dz_sp, v, mode="born", interpolate=False):
    """

    @param beam_frqs:
    @param no:
    @param na_det:
    @param wavelength:
    @param img_shape:
    @param dxy:
    @param dxy_sp:
    @param dz_sp:
    @param v:
    @param mode:
    @return:
    """
    ny, nx = img_shape

    nimgs = len(beam_frqs)

    # ##################################
    # get frequencies of initial images and make broadcastable to shape (nimgs, ny, nx)
    # ##################################

    # frequency of initial images
    fx = np.expand_dims(fft.fftshift(fft.fftfreq(nx, dxy)), axis=(0, 1))
    fy = np.expand_dims(fft.fftshift(fft.fftfreq(ny, dxy)), axis=(0, 2))
    fz = get_fz(fx, fy, no, wavelength)

    # logical array, which frqs in detection NA
    detectable = np.sqrt(fx ** 2 + fy ** 2)[0] <= (na_det / wavelength)
    detectable = np.tile(detectable, [nimgs, 1, 1])

    # scattering potential frqs
    nz_sp, ny_sp, nx_sp = v.shape

    fx_sp = fft.fftshift(fft.fftfreq(nx_sp, dxy_sp))
    fy_sp = fft.fftshift(fft.fftfreq(ny_sp, dxy_sp))
    fz_sp = fft.fftshift(fft.fftfreq(nz_sp, dz_sp))
    dfx_sp = fx_sp[1] - fx_sp[0]
    dfy_sp = fy_sp[1] - fy_sp[0]
    dfz_sp = fz_sp[1] - fz_sp[0]

    # ##################################
    # find indices in scattering potential per image
    # use the notation Fx, Fy, Fz to give the frequencies in the 3D scattering potential
    # ##################################
    if mode == "born":
        # ##################################
        # F(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * Es(fx, fy)
        # ##################################

        # construct frequencies where we have data about the 3D scattering potentials
        # frequencies of the sample F = f - no/lambda * beam_vec
        Fx, Fy, Fz = np.broadcast_arrays(fx - np.expand_dims(beam_frqs[:, 0], axis=(1, 2)),
                                         fy - np.expand_dims(beam_frqs[:, 1], axis=(1, 2)),
                                         fz - np.expand_dims(beam_frqs[:, 2], axis=(1, 2))
                                         )
        # if don't copy, then elements of F's are reference to other elements.
        Fx = np.array(Fx, copy=True)
        Fy = np.array(Fy, copy=True)
        Fz = np.array(Fz, copy=True)

        # indices into the final scattering potential
        # taking advantage of the fact that the final scattering potential indices have FFT structure
        zind = Fz / dfz_sp + nz_sp // 2
        yind = Fy / dfy_sp + ny_sp // 2
        xind = Fx / dfx_sp + nx_sp // 2
    elif mode == "rytov":
        # F(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * psi_s(fx - n/lambda * nx, fy - n/lambda * ny)
        # F(Fx, Fy, Fz) = 2*i * (2*pi*fz) * psi_s(Fx, Fy)
        # so want to change variables and take (Fx, Fy) -> (fx, fy)
        # But have one problem: (Fx, Fy, Fz) do not form a normalized vector like (fx, fy, fz)
        # so although we can use fx, fy to stand in, we need to calculate the new z-component
        # Fz_rytov = np.sqrt( (n/lambda)**2 - (Fx + n/lambda * nx)**2 - (Fy + n/lambda * ny)**2) - n/lambda * nz
        # fz = Fz + n/lambda * nz
        Fx_rytov = fx
        Fy_rytov = fy

        fx_rytov = Fx_rytov + np.expand_dims(beam_frqs[:, 0], axis=(1, 2))
        fy_rytov = Fy_rytov + np.expand_dims(beam_frqs[:, 1], axis=(1, 2))
        fz_rytov = get_fz(fx_rytov, fy_rytov, no, wavelength)

        Fz_rytov = fz_rytov - np.expand_dims(beam_frqs[:, 2], axis=(1, 2))

        # indices into the final scattering potential
        zind = Fz_rytov / dfz_sp + nz_sp // 2
        yind = Fy_rytov / dfy_sp + ny_sp // 2
        xind = Fx_rytov / dfx_sp + nx_sp // 2

        zind, yind, xind = np.broadcast_arrays(zind, yind, xind)
        zind = np.array(zind, copy=True)
        yind = np.array(yind, copy=True)
        xind = np.array(xind, copy=True)
    else:
        raise ValueError("'mode' must be 'born' or 'rytov' but was '%s'" % mode)

    inds_raw = (zind, yind, xind)


    # find indices in bounds
    to_use = np.logical_and.reduce((zind >= 0, zind < nz_sp,
                                    yind >= 0, yind < ny_sp,
                                    xind >= 0, xind < nx_sp,
                                    detectable))

    # build forward model as sparse matrix
    # E = M*V
    # where V is made into a vector by ravelling
    # and the scattered fields are first stacked then ravelled
    # use csr for fast right mult L.dot(v)
    # use csc for fast left mult w.dot(L)
    if interpolate:
        # trilinear interpolation scheme ... needs points before and after
        z1 = np.floor(zind).astype(int)
        z2 = np.ceil(zind).astype(int)
        y1 = np.floor(yind).astype(int)
        y2 = np.ceil(yind).astype(int)
        x1 = np.floor(xind).astype(int)
        x2 = np.ceil(xind).astype(int)

        # find indices in bounds
        to_use = np.logical_and.reduce((z1 >= 0, z2 < nz_sp,
                                        y1 >= 0, y2 < ny_sp,
                                        x1 >= 0, x2 < nx_sp,
                                        detectable))
        # todo: add in the points this misses where only option is to round

        inds = [(z1, y1, x1), (z2, y1, x1),
                (z1, y2, x1), (z2, y2, x1),
                (z1, y1, x2), (z2, y1, x2),
                (z1, y2, x2), (z2, y2, x2)
                ]
        interp_weights = [(zind - z1) * (yind - y1) * (xind - x1),
                          (z2 - zind) * (yind - y1) * (xind - x1),
                          (zind - z1) * (y2 - yind) * (xind - x1),
                          (z2 - zind) * (y2 - yind) * (xind - x1),
                          (zind - z1) * (yind - y1) * (x2 - xind),
                          (z2 - zind) * (yind - y1) * (x2 - xind),
                          (zind - z1) * (y2 - yind) * (x2 - xind),
                          (z2 - zind) * (y2 - yind) * (x2 - xind)]

        # rows -> indices into E vector
        rows = np.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]
        rows = np.tile(rows, 8)

        # cols -> indices into V vector
        inds_to_use = [[i[to_use] for i in inow] for inow in inds]
        zinds_to_use, yinds_to_use, xinds_to_use = list(zip(*inds_to_use))
        zinds_to_use = np.concatenate(zinds_to_use)
        yinds_to_use = np.concatenate(yinds_to_use)
        xinds_to_use = np.concatenate(xinds_to_use)

        cols = np.ravel_multi_index(tuple((zinds_to_use, yinds_to_use, xinds_to_use)), v.shape)

        # construct sparse matrix values
        interp_weights_to_use = np.concatenate([w[to_use] for w in interp_weights])
        fz_stack = np.tile(np.tile(fz, [nimgs, 1, 1])[to_use], 8)
        data = interp_weights_to_use / (2 * 1j * (2 * np.pi * fz_stack)) * dxy_sp * dxy_sp * dz_sp / (dxy ** 2)

    else:
        # round coordinates to nearest values
        zind_round = np.round(zind).astype(int)
        yind_round = np.round(yind).astype(int)
        xind_round = np.round(xind).astype(int)
        inds_round = (zind_round, yind_round, xind_round)

        # construct sparse matrix non-zero coords
        rows = np.arange(nimgs * ny * nx, dtype=int).reshape([nimgs, ny, nx])[to_use]

        inds_to_use = tuple([i[to_use] for i in inds_round])
        cols = np.ravel_multi_index(inds_to_use, v.shape)

        # construct sparse matrix values
        fz_stack = np.tile(fz, [nimgs, 1, 1])
        data = np.ones(len(rows)) / (2 * 1j * (2 * np.pi * fz_stack[to_use])) * dxy_sp * dxy_sp * dz_sp / (dxy ** 2)

    # construct sparse matrix
    model = sp.csr_matrix((data, (rows, cols)), shape=(nimgs * ny * nx, v.size))

    return model


def reconstruction(efield_fts, beam_frqs, no, na_det, wavelength, dxy, z_fov=10, reg=0.1, dz_sampling_factor=1,
                   dxy_sampling_factor=1, mode="born", use_interpolation=False):
    """
    Given a set of holograms obtained using ODT, put the hologram information back in the correct locations in
    Fourier space

    @param efield_fts: The exact definition of efield_fts depends on whether "born" or "rytov" mode is used
    @param beam_frqs: nimgs x 3, where each is [vx, vy, vz] and vx**2 + vy**2 + vz**2 = n**2 / wavelength**2
    @param no: background index of refraction
    @param na_det: detection numerical aperture
    @param wavelength:
    @param dxy: pixel size
    @param z_fov: z-field of view
    @param reg: regularization factor
    @param dz_sampling_factor: fraction of Nyquist sampling factor to use
    @param dxy_sampling_factor:
    @param mode: "born" or "rytov"
    @return sp_ft, sp_ft_imgs, coords, fcoords:
    """

    nimgs, ny, nx = efield_fts.shape[-3:]

    # ##################################
    # get frequencies of initial images and make broadcastable to shape (nimgs, ny, nx)
    # ##################################
    fx = np.expand_dims(fft.fftshift(fft.fftfreq(nx, dxy)), axis=(0, 1))
    fy = np.expand_dims(fft.fftshift(fft.fftfreq(ny, dxy)), axis=(0, 2))
    fz = get_fz(fx, fy, no, wavelength)

    # ##################################
    # set sampling of 3D scattering potential
    # ##################################
    (dz_sp, dxy_sp, _), (nz_sp, ny_sp, nx_sp), _ = get_reconstruction_sampling(no, na_det, beam_frqs, wavelength,
                                                                               dxy, (ny, nx), z_fov,
                                                                               dz_sampling_factor, dxy_sampling_factor)

    x_sp = dxy_sp * (np.arange(nx_sp) - nx_sp // 2)
    y_sp = dxy_sp * (np.arange(ny_sp) - ny_sp // 2)
    z_sp = dz_sp * (np.arange(nz_sp) - nz_sp // 2)

    fx_sp = fft.fftshift(fft.fftfreq(nx_sp, dxy_sp))
    fy_sp = fft.fftshift(fft.fftfreq(ny_sp, dxy_sp))
    fz_sp = fft.fftshift(fft.fftfreq(nz_sp, dz_sp))
    dfx_sp = fx_sp[1] - fx_sp[0]
    dfy_sp = fy_sp[1] - fy_sp[0]
    dfz_sp = fz_sp[1] - fz_sp[0]

    # ##################################
    # find indices in scattering potential per image
    # use the notation Fx, Fy, Fz to give the frequencies in the 3D scattering potential
    # ##################################
    if mode == "born":
        # ##################################
        # construct frequencies where we have data about the 3D scattering potentials
        # frequencies of the sample F = f - no/lambda * beam_vec
        # ##################################
        Fx, Fy, Fz = np.broadcast_arrays(fx - np.expand_dims(beam_frqs[:, 0], axis=(1, 2)),
                                         fy - np.expand_dims(beam_frqs[:, 1], axis=(1, 2)),
                                         fz - np.expand_dims(beam_frqs[:, 2], axis=(1, 2))
                                         )
        # if don't copy, then elements of F's are reference to other elements.
        # otherwise later when set some elements to nans, many that we don't expect would become nans also
        Fx = np.array(Fx, copy=True)
        Fy = np.array(Fy, copy=True)
        Fz = np.array(Fz, copy=True)

        # take care of frequencies which do not contain signal
        not_detectable = (fx ** 2 + fy ** 2) > (na_det / wavelength) ** 2
        not_detectable = np.tile(not_detectable, [nimgs, 1, 1])

        # F(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * Es(fx, fy)
        # indices into the final scattering potential
        # taking advantage of the fact that the final scattering potential indices have FFT structure
        zind = (np.round(Fz / dfz_sp) + nz_sp // 2).astype(int)
        yind = (np.round(Fy / dfy_sp) + ny_sp // 2).astype(int)
        xind = (np.round(Fx / dfx_sp) + nx_sp // 2).astype(int)
    elif mode == "rytov":
        # F(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * psi_s(fx - n/lambda * nx, fy - n/lambda * ny)
        # F(Fx, Fy, Fz) = 2*i * (2*pi*fz) * psi_s(Fx, Fy)
        # so want to change variables and take (Fx, Fy) -> (fx, fy)
        # But have one problem: (Fx, Fy, Fz) do not form a normalized vector like (fx, fy, fz)
        # so although we can use fx, fy to stand in, we need to calculate the new z-component
        # Fz_rytov = np.sqrt( (n/lambda)**2 - (Fx + n/lambda * nx)**2 - (Fy + n/lambda * ny)**2) - n/lambda * nz
        # fz = Fz + n/lambda * nz
        Fx_rytov = fx
        Fy_rytov = fy

        fx_rytov = Fx_rytov + np.expand_dims(beam_frqs[:, 0], axis=(1, 2))
        fy_rytov = Fy_rytov + np.expand_dims(beam_frqs[:, 1], axis=(1, 2))
        fz_rytov = get_fz(fx_rytov, fy_rytov, no, wavelength)

        Fz_rytov = fz_rytov - np.expand_dims(beam_frqs[:, 2], axis=(1, 2))

        # take care of frequencies which do not contain signal
        not_detectable = (fx_rytov ** 2 + fy_rytov ** 2) > (na_det / wavelength) ** 2

        # indices into the final scattering potential
        zind = (np.round(Fz_rytov / dfz_sp) + nz_sp // 2).astype(int)
        yind = (np.round(Fy_rytov / dfy_sp) + ny_sp // 2).astype(int)
        xind = (np.round(Fx_rytov / dfx_sp) + nx_sp // 2).astype(int)

        zind, yind, xind = np.broadcast_arrays(zind, yind, xind)
        zind = np.array(zind, copy=True)
        yind = np.array(yind, copy=True)
        xind = np.array(xind, copy=True)
    else:
        raise ValueError("'mode' must be 'born' or 'rytov' but was '%s'" % mode)

    # only use those within bounds
    # when things work right, these should be the same for all images ... so maybe don't really need 3D array
    # this is correct NA check for Born, but maybe not for Rytov approx?
    to_use_ind = np.logical_and.reduce((zind >= 0, zind < nz_sp,
                                        yind >= 0, yind < ny_sp,
                                        xind >= 0, xind < nx_sp,
                                        np.logical_not(not_detectable),
                                        np.logical_not(np.isnan(efield_fts))))

    # convert nD indices to 1D indices so can easily check if points are unique
    cind = -np.ones(zind.shape, dtype=int)
    cind[to_use_ind] = np.ravel_multi_index((zind[to_use_ind].astype(int).ravel(),
                                             yind[to_use_ind].astype(int).ravel(),
                                             xind[to_use_ind].astype(int).ravel()), (nz_sp, ny_sp, nx_sp))

    # actually can do a little better than pixel rounding. Can interpolate to get appropriate fx, fy frequencies
    # idea: Fx and Fy give the exact frequencies within the scattered field data, but once we find the corresponding
    # pixel in the scaattering potential, there is a subpixel offset
    # define Fx_on_pix and Fy_on_pix which give the exact frequencies in the scattering potential. Then we can
    # interpolate the scattered field data to get this point.
    # we still have some pixel error in Fz, and solving this would require interpolating among multiple angles
    # which we don't want to do because then we are no longer working on a grid ...
    Fx_on_pix = dfx_sp * (xind - nx_sp // 2)
    Fy_on_pix = dfy_sp * (yind - ny_sp // 2)

    f_unshift_ft = np.zeros(efield_fts.shape, dtype=complex)
    for ii in range(nimgs):
        if use_interpolation:
            if mode == "rytov":
                raise NotImplementedError("Current interpolation implementation does nto work with mode 'rytov'")

            # only interpolate at points we are going to use
            interp_fcoords = np.stack((Fy_on_pix[ii, to_use_ind[ii]] + beam_frqs[ii, 1],
                                       Fx_on_pix[ii, to_use_ind[ii]] + beam_frqs[ii, 0]), axis=1)

            # also want corrected fz
            fz_temp = get_fz(interp_fcoords[:, 1], interp_fcoords[:, 0], no, wavelength)

            # do interpolation
            f_unshift_ft[ii, to_use_ind[ii]] = interpn((fy[0, :, 0], fx[0, 0, :]),
                                                       efield_fts[ii],
                                                       interp_fcoords,
                                                       method="linear") * 2 * 1j * (2 * np.pi * fz_temp)
        else:
            # no interp mode
            if mode == "born":
                f_unshift_ft[ii] = 2 * 1j * (2 * np.pi * fz[0]) * efield_fts[ii]
            elif mode == "rytov":
                f_unshift_ft[ii] = 2 * 1j * (2 * np.pi * fz_rytov[ii]) * efield_fts[ii]


    # one reconstruction per image
    # spot_ft_imgs = np.zeros((nimgs, nz_sp, ny_sp, nx_sp), dtype=complex) * np.nan
    spot_ft = np.zeros((nz_sp, ny_sp, nx_sp), dtype=complex)
    num_pts = np.zeros((nimgs, nz_sp, ny_sp, nx_sp), dtype=np.int8)
    for ii in range(nimgs):
        # check we don't have duplicate indices ... otherwise need to do something else ...
        cind_unique_angle = np.unique(cind[ii][to_use_ind[ii]])
        if len(cind_unique_angle) != np.sum(to_use_ind[ii]):
            # approach would be, get array mapping all elements to unique elements, using options for np.unique()
            # and the cind's
            # average these and keep track of number
            # next, convert unique elements back to 3D indices and use the below code
            raise NotImplementedError("reconstruction only implemented for one angle mapping")

        # assuming at most one point for each ...
        inds_angle = (zind[ii][to_use_ind[ii]], yind[ii][to_use_ind[ii]], xind[ii][to_use_ind[ii]])
        # since using DFT's instead of FT's have to adjust the normalization. Recall that FT ~ DFT * dr1 * ... * drn
        spot_ft[inds_angle] += f_unshift_ft[ii, to_use_ind[ii]] * (dxy * dxy) / (dxy_sp * dxy_sp * dz_sp)
        num_pts[ii][inds_angle] = 1

    # average over angles/images
    num_pts_all = np.sum(num_pts, axis=0)
    no_data = num_pts_all == 0
    is_data = np.logical_not(no_data)

    spot_ft[is_data] = spot_ft[is_data] / (num_pts_all[is_data] + reg)
    spot_ft[no_data] = np.nan

    # real space and fourier space coordinates
    fcoords = (fx_sp, fy_sp, fz_sp)
    coords = (x_sp, y_sp, z_sp)

    # todo: get rid of coords since do not use them

    return spot_ft, coords, fcoords


def apply_n_constraints(sp_ft, no, wavelength, n_iterations=100, beta=0.5, use_raar=True,
                        require_real_part_greater_bg=False, use_gpu=_cupy_available, print_info=False):
    """
    Iterative apply constraints on the scattering potential and the index of refraction

    constraint 1: scattering potential FT must match data at points where we information
    constraint 2: real(n) >= no and imag(n) >= 0

    @param sp_ft: 3D fourier transform of scattering potential. This array should have nan values where the array
     values are unknown.
    @param no: background index of refraction
    @param wavelength: wavelength in um
    @param n_iterations: number of iterations
    @param beta:
    @param bool use_raar: whether or not to use the Relaxed-Averaged-Alternating Reflection algorithm
    @return scattering_pot_ft:
    """
    # scattering_potential masked with nans where no information
    sp_ft = np.array(sp_ft, copy=True)
    sp_data = np.array(sp_ft, copy=True)

    if not np.any(np.isnan(sp_ft)):
        raise ValueError("sp_ft contained no NaN's, so there is no information to infer")

    no_data = np.isnan(sp_ft)
    is_data = np.logical_not(no_data)
    sp_ft[no_data] = 0

    # try smoothing image first ...
    # todo: is this useful?
    # sp_ft = gaussian_filter(sp_ft, (4, 4, 4))

    tstart = time.perf_counter()
    for ii in range(n_iterations):
        if print_info:
            print("constraint iteration %d/%d, elapsed time = %0.2fs" % (ii + 1, n_iterations, time.perf_counter() - tstart), end="\r")
            if ii == (n_iterations - 1):
                print("")
        # ############################
        # ensure n is physical
        # ############################
        if not use_gpu:
            sp = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
        else:
            sp = cp.asnumpy(cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(sp_ft, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))
        n = get_n(sp, no, wavelength)

        if require_real_part_greater_bg:
            # real part must be >= no
            correct_real = np.real(n) < no
            n[correct_real] = no + np.imag(n[correct_real])

        # imaginary part must be >= 0
        correct_imag = np.imag(n) < 0
        n[correct_imag] = np.real(n[correct_imag]) + 0*1j

        sp_ps = get_scattering_potential(n, no, wavelength)
        if not use_gpu:
            sp_ps_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
        else:
            sp_ps_ft = cp.asnumpy(cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(sp_ps, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))

        if use_raar:
            sp_ft_pm = np.array(sp_ft, copy=True)
            sp_ft_pm[is_data] = sp_data[is_data]

            # ############################
            # projected Ps * Pm
            # ############################
            sp_ft_ps_pm = np.array(sp_ft_pm, copy=True)
            if not use_gpu:
                sp_ps_pm = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
            else:
                sp_ps_pm = cp.asnumpy(cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(sp_ft_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))
            n_ps_pm = get_n(sp_ps_pm, no, wavelength)

            if require_real_part_greater_bg:
                # real part must be >= no
                correct_real = np.real(n_ps_pm) < no
                n_ps_pm[correct_real] = no + np.imag(n_ps_pm[correct_real])

            # imaginary part must be >= 0
            correct_imag = np.imag(n_ps_pm) < 0
            n_ps_pm[correct_imag] = np.real(n_ps_pm[correct_imag]) + 0 * 1j

            sp_ps_pm = get_scattering_potential(n_ps_pm, no, wavelength)
            if not use_gpu:
                sp_ps_pm_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1))
            else:
                sp_ps_pm_ft = cp.asnumpy(cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(sp_ps_pm, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)))

            # update
            sp_ft = beta * sp_ft - beta * sp_ps_ft + (1 - 2 * beta) * sp_ft_pm + 2 * beta * sp_ps_pm_ft
        else:
            # ############################
            # projected Pm * Ps
            # ############################
            sp_ft_pm_ps = np.array(sp_ps_ft, copy=True)
            sp_ft_pm_ps[is_data] = sp_data[is_data]

            # update
            sp_ft = sp_ft_pm_ps

    return sp_ft


# fit frequencies
def fit_ref_frq(img_ft, dxy, fmax_int, search_rad_fraction=1, npercentiles=50, filter_size=0, show_figure=False):
    """
    Determine the hologram reference frequency from a single imaged, based on the regions in the hologram beyond the
    maximum imaging frequency that have information. These are expected to be circles centered around the reference
    frequency.

    The fitting strategy is this
    (1) determine a threshold value for which points have signal in the image. To do this, first make a plot of
    thresholds versus percentiles. This should look like two piecewise lines
    (2) after thresholding the image, fit to circles.

    Note: when the beam angle is non-zero, the dominant tomography frequency component will not be centered
    on this circle, but will be at position f_ref - f_beam
    @param img:
    @param dxy:
    @param fmax_int:
    @param search_rad_fraction:
    @param npercentiles:
    @return results, circ_dbl_fn, figh: results["fit_params"] = [cx, cy, radius]
    """
    ny, nx = img_ft.shape
    # fourier transforms
    # img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img)))
    # filter image
    img_ft = gaussian_filter(img_ft, (filter_size, filter_size))

    # get frequency data
    fxs = fft.fftshift(fft.fftfreq(nx, dxy))
    dfx = fxs[1] - fxs[0]
    fys = fft.fftshift(fft.fftfreq(ny, dxy))
    dfy = fys[1] - fys[0]
    fxfx, fyfy = np.meshgrid(fxs, fys)
    ff_perp = np.sqrt(fxfx**2 + fyfy**2)

    extent_fxy = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx,
                  fys[0] - 0.5 * dfy, fys[-1] + 0.5 * dfy]

    # #########################
    # find threshold using expected volume of area above threshold
    # #########################
    # only search outside of this
    frad_search = search_rad_fraction * fmax_int
    search_region = ff_perp > frad_search

    frq_area = (fxfx[0, -1] - fxfx[0, 0]) * (fyfy[-1, 0] - fyfy[0, 0])
    expected_area = 2 * (np.pi * (0.5 * fmax_int)**2) / (frq_area - np.pi * frad_search**2)
    # thresh = np.percentile(np.abs(img_ft_bin[search_region]), 100 * (1 - expected_area))

    # find thresholds for different percentiles and look or plateau like behavior
    percentiles = np.linspace(0, 99, npercentiles)
    thresh_all = np.percentile(np.abs(img_ft[search_region]), percentiles)

    init_params_thresh = [0, thresh_all[0],
                          (thresh_all[-1] - thresh_all[-2]) / (percentiles[-1] - percentiles[-2]),
                          100 * (1 - expected_area)]
    results_thresh = fit.fit_model(thresh_all, lambda p: fit.line_piecewise(percentiles, p), init_params_thresh)

    thresh_ind = np.argmin(np.abs(percentiles - results_thresh["fit_params"][-1]))
    thresh = thresh_all[thresh_ind]

    # masked image
    img_ft_ref_mask = np.logical_and(np.abs(img_ft) > thresh, ff_perp > frad_search)

    # #########################
    # define fitting function and get initial guesses
    # #########################

    def circ_dbl_fn(x, y, p):
        p = np.array([p[0], p[1], p[2], 1, 0, np.sqrt(dfx * dfy)])
        p2 = np.array(p, copy=True)
        p2[0] *= -1
        p2[1] *= -1
        circd = fit.circle(x, y, p) + fit.circle(x, y, p2)
        circd[circd > 1] = 1
        return circd

    # guess based on maximum pixel value. This actually gives f_ref - f_beam, but should be a close enough starting point
    guess_ind_1d = np.argmax(np.abs(img_ft) * (fyfy <= 0) * (ff_perp > fmax_int))
    guess_ind = np.unravel_index(guess_ind_1d, img_ft.shape)

    # do fitting
    # init_params = [np.mean(fxfx_bin[img_ft_ref_mask]), np.mean(fyfy_bin[img_ft_ref_mask]), 0.5 * fmax_int]
    init_params = [fxfx[guess_ind], fyfy[guess_ind], 0.5 * fmax_int]
    results = fit.fit_model(img_ft_ref_mask, lambda p: circ_dbl_fn(fxfx, fyfy, p), init_params)

    # #########################
    # plot
    # #########################
    if show_figure:
        figh = plt.figure(figsize=(16, 8))
        grid = figh.add_gridspec(2, 3)

        fp_ref = results["fit_params"]
        ax = figh.add_subplot(grid[0, 0])
        ax.set_title("img ft")
        ax.imshow(np.abs(img_ft), norm=PowerNorm(gamma=0.1), cmap='bone', extent=extent_fxy, origin="lower")
        ax.plot(fp_ref[0], fp_ref[1], 'kx')
        ax.plot(-fp_ref[0], -fp_ref[1], 'kx')
        ax.add_artist(Circle((fp_ref[0], fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((-fp_ref[0], -fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((0, 0), radius=fmax_int, facecolor="none", edgecolor="k"))

        ax = figh.add_subplot(grid[0, 1])
        ax.set_title("img ft binned")
        ax.imshow(np.abs(img_ft), norm=PowerNorm(gamma=0.1), cmap='bone', extent=extent_fxy, origin="lower")
        ax.plot(fp_ref[0], fp_ref[1], 'kx')
        ax.plot(-fp_ref[0], -fp_ref[1], 'kx')
        ax.add_artist(Circle((fp_ref[0], fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((-fp_ref[0], -fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((0, 0), radius=fmax_int, facecolor="none", edgecolor="k"))

        ax = figh.add_subplot(grid[0, 2])
        ax.set_title("img ft binned mask")
        ax.imshow(img_ft_ref_mask, norm=PowerNorm(gamma=0.1), cmap='bone', extent=extent_fxy, origin="lower")
        ax.plot(fp_ref[0], fp_ref[1], 'kx')
        ax.plot(-fp_ref[0], -fp_ref[1], 'kx')
        ax.add_artist(Circle((fp_ref[0], fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((-fp_ref[0], -fp_ref[1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
        ax.add_artist(Circle((0, 0), radius=fmax_int, facecolor="none", edgecolor="k"))

        ax = figh.add_subplot(grid[1, 0])
        ax.plot(percentiles, thresh_all, 'rx')
        ax.plot(percentiles, fit.line_piecewise(percentiles, results_thresh["fit_params"]))
        ax.set_xlabel("percentile")
        ax.set_ylabel("threshold (counts)")
        ax.set_title('threshold = %.0f' % results_thresh["fit_params"][-1])
    else:
        figh = None

    return results, circ_dbl_fn, figh


# plotting functions
def plot_scattered_angle(img_efield_ft, img_efield_bg_ft, img_efield_scatt_ft,
                         beam_frq, frq_ref, fmax_int, dxy,
                         title="", figsize=(18, 10), gamma=0.1,
                         **kwargs):
    """
    Plot diagnostic of ODT image and background image
    
    Allows img and ft to be passed in or calculated ....

    @param img_efield_ft:
    @param img_efield_bg_ft:
    @param img_efield_scattered_ft:
    @param beam_frq:
    @param frq_ref:
    @param fmax_int:
    @param fcoords:
    @param dxy:
    @param title:
    @return figh:
    """

    # real-space coordinates
    ny, nx = img_efield_ft.shape
    x = (np.arange(nx) - nx // 2) * dxy
    y = (np.arange(ny) - ny // 2) * dxy
    extent_xy = [x[0] - 0.5 * dxy, x[-1] + 0.5 * dxy,
                 y[-1] + 0.5 * dxy, y[0] - 0.5 * dxy]

    # frequency-space coordinates
    fx = fft.fftshift(fft.fftfreq(nx, dxy))
    fy = fft.fftshift(fft.fftfreq(ny, dxy))
    dfy = fy[1] - fy[0]
    dfx = fx[1] - fx[0]
    extent_fxfy = [fx[0] - 0.5 * dfx, fx[-1] + 0.5 * dfx,
                   fy[-1] + 0.5 * dfy, fy[0] - 0.5 * dfy]

    # ##############################
    # compute real-space electric field
    # ##############################
    img_efield_ft_no_nan = np.array(img_efield_ft, copy=True)
    img_efield_ft_no_nan[np.isnan(img_efield_ft_no_nan)] = 0

    img_efield_bg_ft_no_nan = np.array(img_efield_bg_ft, copy=True)
    img_efield_bg_ft_no_nan[np.isnan(img_efield_bg_ft_no_nan)] = 0

    img_efield_scatt_ft_no_nan = np.array(img_efield_scatt_ft, copy=True)
    img_efield_scatt_ft_no_nan[np.isnan(img_efield_scatt_ft_no_nan)] = 0

    # remove carrier frequency so can easily see phase shifts
    exp_fact = np.exp(-1j * 2*np.pi * (beam_frq[0] * np.expand_dims(x, axis=0) +
                                       beam_frq[1] * np.expand_dims(y, axis=1)))

    img_efield_shift = fft.fftshift(fft.ifft2(fft.ifftshift(img_efield_ft_no_nan))) * exp_fact
    img_efield_shift_bg = fft.fftshift(fft.ifft2(fft.ifftshift(img_efield_bg_ft_no_nan))) * exp_fact
    img_efield_shift_scatt = fft.fftshift(fft.ifft2(fft.ifftshift(img_efield_scatt_ft_no_nan))) * exp_fact

    # ##############################
    # plot
    # ##############################
    figh = plt.figure(figsize=figsize, **kwargs)
    figh.suptitle(title)
    grid = figh.add_gridspec(ncols=9, width_ratios=[8, 1, 1] * 3,
                             nrows=3, hspace=0.5, wspace=0.5)


    labels = ["E_{shifted}", "E_{bg,shifted}", "E_{scatt,shifted}"]
    fields_r = [img_efield_shift, img_efield_shift_bg, img_efield_shift_scatt]
    vmin_e = np.percentile(np.abs(img_efield_shift).ravel(), 0.1)
    vmax_e = np.percentile(np.abs(img_efield_shift).ravel(), 99.9)
    vmin_scat = np.percentile(np.abs(img_efield_shift_scatt).ravel(), 0.1)
    vmax_scat = np.percentile(np.abs(img_efield_shift_scatt).ravel(), 99.9)
    vmin_r = [vmin_e, vmin_e, vmin_scat]
    vmax_r = [vmax_e, vmax_e, vmax_scat]
    fields_ft = [img_efield_ft, img_efield_bg_ft, img_efield_scatt_ft]

    flims = [[-fmax_int, fmax_int],
             [-fmax_int, fmax_int],
             [-fmax_int, fmax_int]]
    plot_pts = [beam_frq[:2]] * 3
    for ii, label in enumerate(labels):

        ax1 = figh.add_subplot(grid[0, 3*ii])
        im = ax1.imshow(np.abs(fields_r[ii]), cmap="bone", vmin=vmin_r[ii], vmax=vmax_r[ii], extent=extent_xy)
        ax1.set_title("$|%s(r)|$" % label)
        ax1.set_xlabel("x-position ($\mu m$)")
        ax1.set_ylabel("y-position ($\mu m$)")

        ax4 = figh.add_subplot(grid[0, 3*ii + 1])
        plt.colorbar(im, cax=ax4)

        ax2 = figh.add_subplot(grid[1, 3*ii])
        im = ax2.imshow(np.angle(fields_r[ii]), cmap="RdBu", vmin=-np.pi, vmax=np.pi, extent=extent_xy)
        ax2.set_title("$ang[%s(r)]$" % label)
        ax2.set_xticks([])
        ax2.set_yticks([])
        if ii == (len(labels) - 1):
            ax4 = figh.add_subplot(grid[1, 3 * ii + 1])
            plt.colorbar(im, cax=ax4)


        ax3 = figh.add_subplot(grid[2, 3 * ii])
        im = ax3.imshow(np.abs(fields_ft[ii]), norm=PowerNorm(gamma=gamma), cmap="bone", extent=extent_fxfy)
        ax3.plot(plot_pts[ii][0], plot_pts[ii][1], 'r.', fillstyle="none")
        ax3.set_xlabel("$f_x$ (1 / $\mu m$)")
        ax3.set_ylabel("$f_y$ (1 / $\mu m$)")
        ax3.set_title("$|E(f)|$")
        ax3.set_xlim(flims[ii])
        ax3.set_ylim(np.flip(flims[ii]))

    return figh



def plot_odt_sampling(frqs, na_detect, na_excite, ni, wavelength, figsize=(16, 8)):
    """
    Illustrate the region of frequency space which is obtained using the plane waves described by frqs

    @param frqs: nfrqs x 2 array of [[fx0, fy0], [fx1, fy1], ...]
    @param na_detect: detection NA
    @param na_excite: excitation NA
    @param ni: index of refraction of medium that samle is immersed in. This may differ from the immersion medium
    of the objectives
    @param wavelength:
    @param figsize:
    @return:
    """
    frq_norm = ni / wavelength
    alpha_det = np.arcsin(na_detect / ni)

    if na_excite / ni < 1:
        alpha_exc = np.arcsin(na_excite / ni)
    else:
        # if na_excite is immersion objective and beam undergoes TIR at interface for full NA
        alpha_exc = np.pi/2

    fzs = get_fz(frqs[:, 0], frqs[:, 1], ni, wavelength)
    frqs_3d = np.concatenate((frqs, np.expand_dims(fzs, axis=1)), axis=1)


    figh = plt.figure(figsize=figsize)
    figh.suptitle("red = maximum extent of frequency info\n"
                 "blue = maximum extent of centers")
    grid = figh.add_gridspec(1, 2)

    # ########################
    # kx-kz plane
    # ########################
    ax = figh.add_subplot(grid[0, 0])
    ax.axis("equal")

    # plot centers
    ax.plot(-frqs_3d[:, 0], -frqs_3d[:, 2], 'k.')

    # plot arcs
    for ii in range(len(frqs_3d)):
        ax.add_artist(Arc((-frqs_3d[ii, 0], -frqs_3d[ii, 2]), 2 * frq_norm, 2 * frq_norm, angle=90,
                          theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="k"))

    # draw arcs for the extremal angles
    fx_edge = na_excite / wavelength
    fz_edge = np.sqrt((ni / wavelength)**2 - fx_edge**2)

    ax.plot(-fx_edge, -fz_edge, 'r.')
    ax.plot(fx_edge, -fz_edge, 'r.')

    ax.add_artist(Arc((-fx_edge, -fz_edge), 2 * frq_norm, 2 * frq_norm, angle=90,
                      theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="r"))
    ax.add_artist(Arc((fx_edge, -fz_edge), 2 * frq_norm, 2 * frq_norm, angle=90,
                      theta1=-alpha_det * 180 / np.pi, theta2=alpha_det * 180 / np.pi, edgecolor="r"))

    # draw arc showing possibly positions of centers
    ax.add_artist(Arc((0, 0), 2 * frq_norm, 2 * frq_norm, angle=-90,
                      theta1=-alpha_exc * 180 / np.pi, theta2=alpha_exc * 180 / np.pi, edgecolor="b"))


    ax.set_xlim([-2 * frq_norm, 2 * frq_norm])
    ax.set_ylim([-2 * frq_norm, 2*frq_norm])
    ax.set_xlabel("$f_x$ (1/$\mu m$)")
    ax.set_ylabel("$f_z$ (1/$\mu m$)")

    # ########################
    # kx-ky plane
    # ########################
    ax = figh.add_subplot(grid[0, 1])
    ax.axis("equal")

    ax.plot(-frqs_3d[:, 0], -frqs_3d[:, 1], 'k.')
    for ii in range(len(frqs_3d)):
        ax.add_artist(Circle((-frqs_3d[ii, 0], -frqs_3d[ii, 1]),
                             na_detect / wavelength, fill=False, color="k"))

    ax.add_artist(Circle((0, 0), na_excite / wavelength, fill=False, color="b"))
    ax.add_artist(Circle((0, 0), (na_excite + na_detect) / wavelength, fill=False, color="r"))

    size = 1.1 * (na_excite + na_detect) / wavelength
    ax.set_xlim([-size, size])
    ax.set_ylim([-size, size])
    ax.set_xlabel("$f_x$ (1/$\mu m$)")
    ax.set_ylabel("$f_y$ (1/$\mu m$)")

    return figh


def plot_n3d(sp_ft, no, wavelength, coords, title=""):
    """
    Plot 3D index of refraction
    @param sp_ft: 3D Fourier transform of scattering potential
    @param no: background index of refraction
    @param wavelength:
    @param coords: (x, y, z)
    @return:
    """

    # work with coordinates
    x_sp, y_sp, z_sp = coords
    nx_sp = len(x_sp)
    ny_sp = len(y_sp)
    nz_sp = len(z_sp)
    dxy_sp = x_sp[1] - x_sp[0]
    dz_sp = z_sp[1] - z_sp[0]

    fx = fft.fftshift(fft.fftfreq(nx_sp, dxy_sp))
    dfx_sp = fx[1] - fx[0]
    fy = fft.fftshift(fft.fftfreq(ny_sp, dxy_sp))
    dfy_sp = fy[1] - fy[0]

    extent_sp_xy = [x_sp[0] - 0.5 * dxy_sp, x_sp[-1] + 0.5 * dxy_sp, y_sp[0] - 0.5 * dxy_sp, y_sp[-1] + 0.5 * dxy_sp]
    extent_sp_xz = [x_sp[0] - 0.5 * dxy_sp, x_sp[-1] + 0.5 * dxy_sp, z_sp[0] - 0.5 * dz_sp, z_sp[-1] + 0.5 * dz_sp]
    extent_sp_yz = [y_sp[0] - 0.5 * dxy_sp, y_sp[-1] + 0.5 * dxy_sp, z_sp[0] - 0.5 * dz_sp, z_sp[-1] + 0.5 * dz_sp]


    extent_fxy = [fx[0] - 0.5 * dfx_sp, fx[-1] + 0.5 * dfx_sp, fy[0] - 0.5 * dfy_sp, fy[-1] + 0.5 * dfy_sp]

    # get need quantities
    sp_ft_nonan = np.array(sp_ft, copy=True)
    sp_ft_nonan[np.isnan(sp_ft_nonan)] = 0
    sp = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft_nonan)))

    n_recon = get_n(sp, no, wavelength)
    n_recon_ft = fft.fftshift(fft.fftn(fft.ifftshift(n_recon)))

    vmax_real = 1.5 * np.percentile((np.real(n_recon) - no), 99.99)
    if vmax_real <=0:
        vmax_real = 1e-12

    vmax_imag = 1.5 * np.percentile(np.imag(n_recon), 99.99)
    if vmax_imag <= 0:
        vmax_imag = 1e-12

    vmax_n_ft = 1.5 * np.percentile(np.abs(n_recon_ft), 99.99)
    if vmax_n_ft <= 0:
        vmax_n_ft = 1e-12

    not_nan = np.logical_not(np.isnan(sp_ft))
    vmax_sp_ft = 1.5 * np.percentile(np.abs(sp_ft[not_nan]), 99)

    # plots
    fmt_fn = lambda x: "%0.6f" % x

    figh = plt.figure(figsize=(16, 8))
    figh.suptitle("Index of refraction, %s" % title)
    grid = figh.add_gridspec(4, nz_sp + 1)

    for ii in range(nz_sp):
        ax = figh.add_subplot(grid[0, ii])
        ax.set_title("%0.1fum" % z_sp[ii])
        im = ax.imshow(np.real(n_recon[ii]) - no, vmin=-vmax_real, vmax=vmax_real, cmap="RdBu", origin="lower", extent=extent_sp_xy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])

        if ii == 0:
            ax.set_ylabel("real(n) - no")

        ax = figh.add_subplot(grid[1, ii])
        im = ax.imshow(np.imag(n_recon[ii]), vmin=-vmax_imag, vmax=vmax_imag, cmap="RdBu", origin="lower", extent=extent_sp_xy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("imag(n)")

        ax = figh.add_subplot(grid[2, ii])
        im = ax.imshow(np.abs(n_recon_ft[ii]), cmap="copper", norm=PowerNorm(gamma=0.1, vmin=0, vmax=vmax_n_ft),
                       origin="lower", extent=extent_fxy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|n(f)|")

        ax = figh.add_subplot(grid[3, ii])
        im = ax.imshow(np.abs(sp_ft[ii]), cmap="copper", norm=PowerNorm(gamma=0.1, vmin=0, vmax=vmax_sp_ft),
                       origin="lower", extent=extent_fxy)
        im.format_cursor_data = fmt_fn
        if ii == 0:
            ax.set_ylabel("|F(f)|")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = figh.add_subplot(grid[0, nz_sp])
    ax.axis("off")
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax_real, vmax=vmax_real), cmap="RdBu"), ax=ax)

    ax = figh.add_subplot(grid[1, nz_sp])
    ax.axis("off")
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax_imag, vmax=vmax_imag), cmap="RdBu"), ax=ax)

    return figh


def compare_v3d(v_ft, v_ft_gt, ttl="", figsize=(35, 20), gamma=0.1):
    """

    @param v_ft:
    @param v_ft_gt:
    @param ttl:
    @param figsize:
    @return:
    """

    v = fft.fftshift(fft.ifftn(fft.ifftshift(v_ft)))
    v_gt = fft.fftshift(fft.ifftn(fft.ifftshift(v_ft_gt)))

    # set display limits
    vminr = 1.2 * np.min(v_gt.real)
    vmaxr = 0
    vmini = -0.1
    vmaxi = 0.1
    vmink = 0
    vmaxk = 1.2 * np.max(np.abs(v_ft_gt))

    # display
    figh = plt.figure(figsize=figsize)
    figh.suptitle(f"{ttl:s}")
    grid = figh.add_gridspec(nrows=6, ncols=v.shape[0] + 1, width_ratios=[1] * v.shape[0] + [0.1])

    for ii in range(v.shape[0]):
        ax = figh.add_subplot(grid[0, ii])
        im = ax.imshow(v[ii].real, vmin=vminr, vmax=vmaxr, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Re(v)")

        if ii == (v.shape[0] - 1):
            ax = figh.add_subplot(grid[0, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[1, ii])
        im = ax.imshow(v_gt[ii].real, vmin=vminr, vmax=vmaxr, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Re(v gt)")

        if ii == (v.shape[0] - 1):
            ax = figh.add_subplot(grid[1, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[2, ii])
        im = ax.imshow(v[ii].real - v_gt[ii].real, vmin=vminr, vmax=-vminr, cmap="PiYG")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Re(v - v gt)")

        if ii == (v.shape[0] - 1):
            ax = figh.add_subplot(grid[2, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[3, ii])
        im = ax.imshow(v[ii].imag, vmin=vmini, vmax=vmaxi, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Im(v)")

        if ii == (v.shape[0] - 1):
            ax = figh.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[4, ii])
        im = ax.imshow(v_gt[ii].imag, vmin=vmini, vmax=vmaxi, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Im(v gt)")

        if ii == (v.shape[0] - 1):
            ax = figh.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[5, ii])
        im = ax.imshow(v[ii].imag - v_gt[ii].imag, vmin=vmini, vmax=vmaxi, cmap="PiYG")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Im(v - v gt)")

        if ii == (v.shape[0] - 1):
            ax = figh.add_subplot(grid[5, ii + 1])
            plt.colorbar(im, cax=ax)

    # k-space
    figh_ft = plt.figure(figsize=figsize)
    figh_ft.suptitle(f"{ttl:s} k-space")
    grid = figh_ft.add_gridspec(nrows=6, ncols=v.shape[0] + 1, width_ratios=[1] * v.shape[0] + [0.1])

    for ii in range(v.shape[0]):
        ax = figh_ft.add_subplot(grid[0, ii])
        im = ax.imshow(np.abs(v_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=vmink, vmax=vmaxk), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|V(k)|")

        if ii == (v.shape[0] - 1):
            ax = figh_ft.add_subplot(grid[0, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[1, ii])
        im = ax.imshow(np.abs(v_ft_gt[ii]), norm=PowerNorm(gamma=gamma, vmin=vmink, vmax=vmaxk), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|V gt(k)|")

        if ii == (v.shape[0] - 1):
            ax = figh_ft.add_subplot(grid[1, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[2, ii])
        im = ax.imshow(np.abs(v_ft[ii] - v_ft_gt[ii]), norm=PowerNorm(gamma=gamma, vmin=vmink, vmax=vmaxk), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|V - V gt(k)|")

        if ii == (v.shape[0] - 1):
            ax = figh_ft.add_subplot(grid[2, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[3, ii])
        im = ax.imshow(np.angle(v_ft[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Angle(V(k))")

        if ii == (v.shape[0] - 1):
            ax = figh_ft.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[4, ii])
        im = ax.imshow(np.angle(v_ft_gt[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("Angle(V gt(k))")

        if ii == (v.shape[0] - 1):
            ax = figh_ft.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

    return figh, figh_ft


def compare_escatt(e, e_gt, beam_frqs, dxy, ttl="", figsize=(35, 20), gamma=0.1):
    """

    @param e:
    @param e_gt:
    @param beam_frqs:
    @param dxy:
    @param ttl:
    @param figsize:
    @return:
    """
    nbeams, ny, nx = e.shape

    x, y = np.meshgrid((np.arange(nx) - nx // 2) * dxy,
                       (np.arange(ny) - ny // 2) * dxy)

    e_gt_ft = fft.fftshift(fft.ifft2(fft.fftshift(e_gt, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    e_ft = fft.fftshift(fft.ifft2(fft.fftshift(e, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

    # display limits
    vmaxe = 1.2 * np.max(np.abs(e_gt))
    vmaxek = 1.2 * np.max(np.abs(e_gt_ft))

    # compare electric fields
    figh = plt.figure(figsize=figsize)
    figh.suptitle(f"{ttl:s} real space")
    grid = figh.add_gridspec(nrows=5, ncols=nbeams + 1, width_ratios=[1] * nbeams + [0.1])
    for ii in range(nbeams):
        shift_fact = np.exp(-1j * 2*np.pi * (beam_frqs[ii, 0] * x + beam_frqs[ii, 1] * y))

        ax = figh.add_subplot(grid[0, ii])
        im = ax.imshow(np.abs(e[ii]), vmin=0, vmax=vmaxe, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E(r)|")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[0, ii + 1])
            plt.colorbar(im, cax=ax)


        ax = figh.add_subplot(grid[1, ii])
        im = ax.imshow(np.abs(e_gt[ii]), vmin=0, vmax=vmaxe, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E gt(r)|")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[1, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[2, ii])
        im = ax.imshow(np.abs(e[ii] - e_gt[ii]), vmin=-vmaxe/10, vmax=vmaxe/10, cmap="PiYG")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E - E gt(r)|")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[2, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[3, ii])
        im = ax.imshow(np.angle(e[ii] * shift_fact), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E(r)) shifted")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh.add_subplot(grid[4, ii])
        im = ax.imshow(np.angle(e_gt[ii] * shift_fact), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E gt(r)) shifted")

        if ii == (nbeams - 1):
            ax = figh.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

    # k-space
    figh_ft = plt.figure(figsize=figsize)
    figh_ft.suptitle(f"{ttl:s} k-space")
    grid = figh_ft.add_gridspec(nrows=5, ncols=nbeams + 1, width_ratios=[1] * nbeams + [0.1])
    for ii in range(nbeams):
        ax = figh_ft.add_subplot(grid[0, ii])
        im = ax.imshow(np.abs(e_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmaxek), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E(k)|")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[0, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[1, ii])
        im = ax.imshow(np.abs(e_gt_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmaxek), cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E gt(k)|")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[1, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[2, ii])
        im = ax.imshow(np.abs(e_ft[ii] - e_gt_ft[ii]), norm=PowerNorm(gamma=gamma, vmin=-vmaxek, vmax=vmaxek), cmap="PiYG")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|E - E gt(k)|")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[2, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[3, ii])
        im = ax.imshow(np.angle(e_ft[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E(k))")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[3, ii + 1])
            plt.colorbar(im, cax=ax)

        ax = figh_ft.add_subplot(grid[4, ii])
        im = ax.imshow(np.angle(e_gt_ft[ii]), vmin=-np.pi, vmax=np.pi, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("ang(E gt(k))")

        if ii == (nbeams - 1):
            ax = figh_ft.add_subplot(grid[4, ii + 1])
            plt.colorbar(im, cax=ax)

    return figh, figh_ft