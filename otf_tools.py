"""
Compute experimental OTF of microscope, based on a series of DMD pictures taken a range of angles/pattern spacings.

Using known affine transformation as a starting point, identify peaks in fluorescence image. These must be normalized
to the DC component of the fluorescence image (laser intensity may change between frames) and to the size of the fourier
component at the frequency in the given pattern.

Actually, we need to the fourier transform of the intensity for our DMD pattern properly bandlimited for the effect
of the imaging system on the coherent light.
"""

import time
# import sys
import numpy as np
from scipy import fft
import scipy.signal
import scipy.optimize

import matplotlib.colors
from matplotlib.colors import PowerNorm
import matplotlib.patches
import matplotlib.pyplot as plt

import dmd_patterns
import simulate_dmd
import affine
import analysis_tools as tools
import sim_reconstruction
import fit_psf

def get_all_fourier_exp(imgs, frq_vects_theory, roi, pixel_size_um, fmax_img, use_guess_frqs=True, peak_pix=2, bg=100):
    """

    :param imgs: nimgs x ny x nx
    :param vects: nimgs x nvecs1 x nvecs2 x 2
    :param float pixel_size_um:
    :param bool use_guess_frqs: if True, use guess frequencies computed from frq_vects_theory, if False use fitting
    procedure to find peak
    :param int peak_pix: number of pixels to use when calculating peak. Typically 2.
    :param float bg:

    :return intensity:
    :return intensity_unc:
    :return frq_vects_expt:
    """
    nimgs = frq_vects_theory.shape[0]
    n1_vecs = frq_vects_theory.shape[1]
    n2_vecs = frq_vects_theory.shape[2]

    intensity = np.zeros(frq_vects_theory.shape[:-1], dtype=np.complex) * np.nan
    intensity_unc = np.zeros(intensity.shape) * np.nan

    # apodization, 2D window from broadcasting
    nx_roi = roi[3] - roi[2]
    ny_roi = roi[1] - roi[0]
    window = scipy.signal.windows.hann(nx_roi)[None, :] * scipy.signal.windows.hann(ny_roi)[:, None]

    # generate frequency data for image FT's
    fxs = tools.get_fft_frqs(nx_roi, pixel_size_um)
    dfx = fxs[1] - fxs[0]
    fys = tools.get_fft_frqs(ny_roi, pixel_size_um)
    dfy = fys[1] - fys[0]

    if imgs.shape[0] == nimgs:
        multiple_images = True
    elif imgs.shape[0] == 1:
        multiple_images = False
        icrop = imgs[0, roi[0]:roi[1], roi[2]:roi[3]]

        img = icrop - bg
        img[img < 0] = 1e-6

        img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img * window)))
        noise_power = sim_reconstruction.get_noise_power(img_ft, fxs, fys, fmax_img)
    else:
        raise Exception()

    frq_vects_expt = np.zeros(frq_vects_theory.shape)
    tstart = time.process_time()
    for ii in range(nimgs):
        tnow = time.process_time()
        print("%d/%d, elapsed time = %0.2fs" % (ii + 1, nimgs, tnow - tstart))
        # sys.stdout.write("\033[F")
        # sys.stdout.flush()

        if multiple_images:
            # subtract background and crop to ROI
            # img = img[0, roi[0]:roi[1], roi[2]:roi[3]] - bg
            # img[img < 0] = 1e-6
            icrop = imgs[ii, roi[0]:roi[1], roi[2]:roi[3]]
            img = icrop - bg
            img[img < 0] = 1e-6

            # fft
            img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img * window)))
            # get noise
            noise_power = sim_reconstruction.get_noise_power(img_ft, fxs, fys, fmax_img)

        # minimimum separation between reciprocal lattice vectors
        vnorms = np.linalg.norm(frq_vects_theory[ii], axis=2)
        min_sep = np.min(vnorms[vnorms > 0])

        # get experimental weights of fourier components
        for aa in range(n1_vecs):
            for bb in range(n2_vecs):

                frq_vects_expt[ii, aa, bb] = frq_vects_theory[ii, aa, bb]

                # only do fitting if peak size exceeds tolerance, and only fit one of a peak and its compliment
                if aa < bb or vnorms[aa, bb] > fmax_img:
                    continue

                # ensure we don't get (0, 0) in our ROI
                if vnorms[aa, bb] == 0:
                    size_away_zero = np.inf
                else:
                    size_away_zero = int(np.floor(vnorms[aa, bb] / dfx / np.sqrt(2) - 1))

                roi_half_size = int(
                    np.min([3, int(np.floor(min_sep / dfx / np.sqrt(2))) - 1, size_away_zero]))

                # get experimental frequency
                if roi_half_size < 1 or use_guess_frqs:
                    # if can't get large enough ROI, then use our guess
                    pass
                else:
                    # fit real fourier component in image space
                    frq_vects_expt[ii, aa, bb], mask = sim_reconstruction.fit_modulation_frq(
                        img_ft, img_ft, sim_options, frq_guess=frq_vects_theory[ii, aa, bb],
                        roi_pix_size=roi_half_size)

                try:
                    # get peak value and phase
                    intensity[ii, aa, bb] = tools.get_peak_value(img_ft, fxs, fys, frq_vects_expt[ii, aa, bb],
                                                                         peak_pixel_size=peak_pix)

                    intensity_unc[ii, aa, bb] = np.sqrt(noise_power) * peak_pix ** 2

                    # handle complimentary point with aa > bb
                    aa_neg = n1_vecs - 1 - aa
                    bb_neg = n2_vecs - 1 - bb
                    intensity[ii, aa_neg, bb_neg] = intensity[ii, aa, bb].conj()
                    intensity_unc[ii, aa_neg, bb_neg] = intensity_unc[ii, aa, bb]

                except:
                    pass

    return intensity, intensity_unc, frq_vects_expt

def get_all_fourier_thry(vas, vbs, nmax, nphases, phase_index, dmd_size):
    """
    Calculate theory intensity/electric field fourier components

    :param vas:
    :param vbs:
    :param roi:
    :param affine_xform:
    :param pixel_size_um:
    :param nmax:
    :param nphases:
    :param phase_index:
    :param dmd_size:
    :param use_blaze_correction:
    :param dmd_params:
    :return:
    """
    npatterns = len(vas)

    norders = 2 * nmax + 1
    efield_theory = np.zeros((npatterns, norders, norders), dtype=np.complex) * np.nan
    frq_vects_dmd = np.zeros((npatterns, norders, norders, 2))

    tstart = time.process_time()
    for ii in range(npatterns):
        tnow = time.process_time()
        print("%d/%d, elapsed time = %0.2fs" % (ii + 1, npatterns, tnow - tstart))
        # sys.stdout.write("\033[F")
        # sys.stdout.flush()
        va = vas[ii]
        vb = vbs[ii]

        # recp_vec_a, recp_vec_b = dmd_patterns.get_reciprocal_vects(va, vb)
        unit_cell, xcell, ycell = dmd_patterns.get_sim_unit_cell(va, vb, nphases)

        # get expected values
        efield_theory[ii], ns, ms, frq_vects_dmd[ii] = \
            dmd_patterns.get_efield_fourier_components(unit_cell, xcell, ycell, va, vb, nphases, phase_index,
                                                       dmd_size=dmd_size, nmax=nmax)

        # change normalization from 1 being maximum possible fourier component to 1 being DC component
        efield_theory[ii] = efield_theory[ii] / np.nansum(unit_cell) * np.nansum(unit_cell >= 0)

    return efield_theory, ns, ms, frq_vects_dmd

def get_intensity_fourier_thry(efields, frq_vects_dmd, roi, affine_xform, fmax_efield_ex, pixel_size_um,
                               use_blaze_correction=False, dmd_params=None):
    """

    :param efields: nimgs x nvecs1 x nvecs2, electric field Fourier components from DMD pattern, with no blaze condition corrections.
    :param frq_vects_dmd: frequency vectors in 1/mirrors in DMD space
    :param roi: region of interest within image
    :param affine_xform: affine transformation connecting image space and DMD space (using pixels as coordinates on both ends)
    :param fmax_efield_ex: maximum electric field frequency that can pass throught he imaging system from the DMD to sample, in 1/mirrors
    :param pixel_size_um: camera pixel size in um
    :param bool use_blaze_correction: whether or not to use blaze correction
    :param dmd_params: {"wavelength", "gamma", "wx", "wy", "theta_ins": [tx, ty], "theta_outs": [tx, ty]}

    :return intensity_theory:
    :return frq_vects_cam: frequency vectors in camera space in 1/pixels
    :return frq_vects_um: frequency vectors in camera space in 1/microns
    """

    frq_vects_cam = np.zeros(frq_vects_dmd.shape)
    intensity_phases = np.zeros(efields.shape) * np.nan
    intensity_theory = np.zeros(efields.shape, dtype=np.complex) * np.nan

    # get new affine xform accounting for ROI
    xform_roi = affine.xform_shift_center(affine_xform, cimg_new=(roi[2], roi[0]))

    if use_blaze_correction:
        # todo: is frequency in the right units?
        def otf(fx, fy):
            return simulate_dmd.blaze_envelope(dmd_params["wavelength"], dmd_params["gamma"], dmd_params["wx"],
                                               dmd_params["wy"], dmd_params["theta_ins"][0], dmd_params["theta_ins"][1],
                                               dmd_params["theta_outs"][0] + dmd_params["wavelength"] * fx,
                                               dmd_params["theta_outs"][1] + dmd_params["wavelength"] * fy)
    else:
        def otf(fx, fy):
            return np.sqrt(fx ** 2 + fy ** 2) <= fmax_efield_ex

    for ii in range(frq_vects_cam.shape[0]):
        otf_now = otf(frq_vects_dmd[ii, ..., 0], frq_vects_dmd[ii, ..., 1])

        intensity_theory[ii] = dmd_patterns.get_int_fc(efields[ii] * otf_now)
        intensity_theory[ii] = intensity_theory[ii] / np.abs(intensity_theory[ii]).max()

        frq_vects_cam[ii, ..., 0], frq_vects_cam[ii, ..., 1], intensity_phases[ii] = \
            affine.xform_sinusoid_params(frq_vects_dmd[ii, ..., 0], frq_vects_dmd[ii, ..., 1], np.angle(intensity_theory[ii]), xform_roi)

    # correct frequency vectors in camera space to be in real units
    frq_vects_um = frq_vects_cam / pixel_size_um

    # vecs[ii, jj] = ns[ii] * recp_va[:, 0] + ms[jj] * recp_vb[:, 0]
    # vecs_xformed[ii, jj, 0], vecs_xformed[ii, jj, 1], _ = \
    #     affine.xform_sinusoid_params(vecs[ii, jj, 0], vecs[ii, jj, 1], 0, affine_xform)

    return intensity_theory, intensity_phases, frq_vects_cam, frq_vects_um

def fit_phase_diff(phase_th, phase_expt, frqs):
    def phase_diff_fn(phi1, phi2):
        diff = np.array(np.mod(phi1 - phi2, 2*np.pi))
        to_flip = np.abs(diff - 2*np.pi) < np.abs(diff)
        diff[to_flip] = diff[to_flip] - 2*np.pi
        return diff

    def phase_xform_fn(phi, fx, fy, p): return np.mod(phi + 2 * np.pi * fx * p[0] + 2 * np.pi * fy * p[1], 2 * np.pi)

    def min_fn(p): return np.nansum(np.abs(phase_diff_fn(phase_xform_fn(phase_th, frqs[:, 0], frqs[:, 1], p), phase_expt)))

    results = scipy.optimize.minimize(min_fn, [0, 0])

    figh = plt.figure()
    plt.plot(np.linalg.norm(frqs, axis=-1), phase_xform_fn(phase_th, frqs[:, 0], frqs[:, 1], results["x"]), '.')
    plt.plot(np.linalg.norm(frqs, axis=-1), np.mod(phase_th, 2*np.pi), '.')
    plt.plot(np.linalg.norm(frqs, axis=-1), np.mod(phase_expt, 2*np.pi), 'x')
    plt.legend(["th fit", "th", "expt"])

    return figh, results


def plot_pattern(img, va, vb, frq_vects, fmax_img, pixel_size_um, dmd_size, affine_xform, roi, nphases, phase_index, fmax_in=None,
                 peak_int_exp=None, peak_int_exp_unc=None, peak_int_theory=None, otf=None, otf_unc=None, figsize=(20, 10)):
    """
    plot image and affine xformed pattern it corresponds to
    :param img: image ny x nx
    :param va: [vx, vy]
    :param vb: [vx, vy]
    :param frq_vects: vecs1 x nvecs2 x 2. in 1/um
    :param float fmax_img: in 1/um
    :param float pixel_size_um:
    :param dmd_size: []
    :param affine_xform: affine transformation between DMD space and camera space
    :param roi: [ystart, yend, xstart, xend] region of interest in image
    :param nphases: number of phaseshifts used to generate the DMD pattern. Needed in addition to va/vb to specify pattern
    :param phase_index:  index of phaseshift. Needed in addition to va, vb, nphases to specify pattern

    :return fig_handle:
    """

    # generate DMD pattern
    pattern, _ = dmd_patterns.get_sim_pattern(dmd_size, va, vb, nphases, phase_index)

    # crop image
    img_roi = img[roi[0]:roi[1], roi[2]:roi[3]]
    ny, nx = img_roi.shape

    # transform pattern using affine transformation
    xform_roi = affine.xform_shift_center(affine_xform, cimg_new=(roi[2], roi[0]))
    img_coords = np.meshgrid(range(nx), range(ny))
    pattern_xformed = affine.affine_xform_mat(pattern, xform_roi, img_coords, mode="interp")

    # get fourier transform of image
    fxs = tools.get_fft_frqs(nx, pixel_size_um)
    dfx = fxs[1] - fxs[0]
    fys = tools.get_fft_frqs(ny, pixel_size_um)
    dfy = fys[1] - fys[0]

    extent = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx, fys[-1] + 0.5 * dfy, fys[0] - 0.5 * dfy]


    window = scipy.signal.windows.hann(nx)[None, :] * scipy.signal.windows.hann(ny)[:, None]
    img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img_roi * window)))

    # plot results
    figh = plt.figure(figsize=figsize)
    grid = plt.GridSpec(2, 6)

    period = dmd_patterns.get_sim_period(va, vb)
    angle = dmd_patterns.get_sim_angle(va, vb)

    plt.suptitle("period=%0.3f mirrors, angle=%0.2fdeg\n"
                 "va=(%d, %d); vb=(%d, %d)" % (period, angle, va[0], va[1], vb[0], vb[1]))

    plt.subplot(grid[0, 0:2])
    plt.imshow(img_roi)
    plt.title('image')

    plt.subplot(grid[0, 2:4])
    plt.imshow(pattern_xformed)
    plt.title('pattern after affine xform')

    ax = plt.subplot(grid[0, 4:6])
    plt.title('image FFT')

    plt.imshow(np.abs(img_ft) ** 2, norm=PowerNorm(gamma=0.1), extent=extent)

    # to_plot = np.logical_not(np.isnan(intensity_exp_norm))
    nmax = int(np.round((frq_vects.shape[1] - 1) * 0.5))
    plt.scatter(frq_vects[..., 0].ravel(), frq_vects[..., 1].ravel(), facecolor='none', edgecolor='r')
    plt.scatter(-frq_vects[..., 0].ravel(), -frq_vects[..., 1].ravel(), facecolor='none', edgecolor='m')
    # plt.scatter(frq_vects[nmax, nmax + 1, 0], frq_vects[nmax, nmax + 1, 1], facecolor="none", edgecolor='k')
    # plt.scatter(frq_vects[nmax, nmax - 1, 0], frq_vects[nmax, nmax - 1, 1], facecolor="none", edgecolor='k')
    # plt.scatter(frq_vects[nmax, nmax + 2, 0], frq_vects[nmax, nmax + 2, 1], facecolor="none", edgecolor='k')
    # plt.scatter(frq_vects[nmax, nmax - 2, 0], frq_vects[nmax, nmax - 2, 1], facecolor="none", edgecolor='k')

    circ = matplotlib.patches.Circle((0, 0), radius=fmax_img, color='k', fill=0, ls='--')
    ax.add_artist(circ)

    if fmax_in is not None:
        circ2 = matplotlib.patches.Circle((0, 0), radius=(fmax_in/2), color='r', fill=0, ls='--')
        ax.add_artist(circ2)


    plt.xlim([-fmax_img, fmax_img])
    plt.ylim([fmax_img, -fmax_img])

    ax = plt.subplot(grid[1, :2])
    plt.title("peaks amp expt/theory")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("Intensity")

    plt.plot([fmax_img, fmax_img], [0, 1], 'k')

    phs = []
    legend_entries = []
    fmags = np.linalg.norm(frq_vects, axis=-1).ravel()

    if peak_int_theory is not None:
        ph, = ax.plot(fmags, np.abs(peak_int_theory).ravel() / np.nanmax(np.abs(peak_int_theory)), '.')
        phs.append(ph)
        legend_entries.append("theory")

    if peak_int_exp is not None:
        if peak_int_exp_unc is None:
            peak_int_exp_unc = np.zeros(peak_int_exp.shape)

        ph = ax.errorbar(fmags, np.abs(peak_int_exp).ravel() / np.nanmax(np.abs(peak_int_exp)),
                         yerr=peak_int_exp_unc.ravel() / np.nanmax(np.abs(peak_int_exp)), fmt='x')
        phs.append(ph)
        legend_entries.append("experiment")


    ax.set_ylim([1e-4, 1.2])
    ax.set_xlim([-0.1 * fmax_img, 1.1 * fmax_img])
    ax.set_yscale('log')

    plt.legend(phs, legend_entries)


    # plot phase
    ax = plt.subplot(grid[1, 2:4])
    plt.title("peaks phase expt/theory")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("Intensity")

    plt.plot([fmax_img, fmax_img], [-np.pi, np.pi], 'k')

    phs = []
    legend_entries = []
    fmags = np.linalg.norm(frq_vects, axis=-1).ravel()

    if peak_int_theory is not None:
        ph, = ax.plot(fmags, np.angle(peak_int_theory).ravel(), '.')
        phs.append(ph)
        legend_entries.append("theory")

    if peak_int_exp is not None:
        ph, = ax.plot(fmags, np.angle(peak_int_exp).ravel(), 'x')
        phs.append(ph)
        legend_entries.append("experiment")

    ax.set_xlim([-0.1 * fmax_img, 1.1 * fmax_img])

    plt.legend(phs, legend_entries)

    # plot otf
    ax = plt.subplot(grid[1, 4:])
    plt.title("otf")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("otf")

    ax.plot([0, fmax_img], [0, 0], 'k')
    ax.plot([fmax_img, fmax_img], [0, 1], 'k')
    if otf is not None:
        if otf_unc is None:
            otf_unc = np.zeros(otf.shape)
        ax.errorbar(fmags, np.abs(otf).ravel(), yerr=otf_unc.ravel(), fmt='.')

    ax.set_ylim([-0.05, 1.2])
    ax.set_xlim([-0.1 * fmax_img, 1.1 * fmax_img])

    return figh

def plot_otf(frq_vects, fmax_img, otf, otf_unc=None, figsize=(20, 10)):
    """
    Plot complete OTF
    :param frq_vects:
    :param fmax_img:
    :param otf:
    :param figsize:
    :return:
    """
    if otf_unc is None:
        otf_unc = np.zeros(otf.shape)

    fmag = np.linalg.norm(frq_vects, axis=-1).ravel()

    fmag_interp = np.linspace(0, fmag.max(), 1000)
    # only care about fmax value, so create na/wavelength that give us this
    na = 1
    wavelength = 2 * na / fmax_img
    otf_ideal = fit_psf.circ_aperture_otf(fmag_interp, 0, na, wavelength)


    figh = plt.figure(figsize=figsize)
    grid = plt.GridSpec(2, 6)

    # 1D otf
    ax = plt.subplot(grid[0, :3])
    ylim = [-0.05, 1.2]
    plt.title("otf mag")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("otf")

    plt.plot(fmag_interp, otf_ideal, 'k')
    plt.errorbar(fmag, np.abs(otf).ravel(), yerr=otf_unc.ravel(), fmt='.')
    xlim = ax.get_xlim()
    plt.plot(xlim, [0, 0], 'k')
    plt.plot([fmax_img, fmax_img], ylim, 'k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 1D log scale
    ax = plt.subplot(grid[1, :3])
    plt.title("otf mag (log scale)")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("otf")

    plt.plot(fmag_interp, otf_ideal, 'k')
    plt.errorbar(fmag, np.abs(otf).ravel(), yerr=otf_unc.ravel(), fmt='.')
    xlim = ax.get_xlim()
    ax.plot([fmax_img, fmax_img], ylim, 'k')
    ax.set_xlim(xlim)
    ax.set_ylim([1e-4, 1.2])

    ax.set_yscale('log')


    # 2D otf
    ax = plt.subplot(grid[0, 3:6])
    plt.title("2D otf (log scale)")
    plt.xlabel("fx (1/um)")
    plt.ylabel("fy (1/um)")
    clims = [1e-3, 1]

    frqs_pos = np.array(frq_vects, copy=True)
    y_is_neg = frq_vects[..., 1] < 0
    frqs_pos[y_is_neg] = -frqs_pos[y_is_neg]

    plt.plot([-fmax_img, fmax_img], [0, 0], 'k')
    plt.scatter(frqs_pos[..., 0].ravel(), frqs_pos[..., 1].ravel(),
                c=np.log10(np.abs(otf).ravel()),
                norm=matplotlib.colors.Normalize(vmin=np.log10(clims[0]), vmax=np.log10(clims[1])))
    cb = plt.colorbar()
    plt.clim(np.log10(clims))

    circ = matplotlib.patches.Circle((0, 0), radius=fmax_img, color='k', fill=0, ls='-')
    ax.add_artist(circ)
    ax.set_xlim([-fmax_img, fmax_img])
    ax.set_ylim([-0.05 * fmax_img, fmax_img])





    # plot phase
    ax = plt.subplot(grid[1, 3:6])
    plt.title("phase")
    plt.xlabel("Frequency (1/um)")
    plt.ylabel("phase")

    ax.plot(fmag, np.angle(otf).ravel(), '.')

    ylims = [-np.pi - 0.1, np.pi + 0.1]
    ax.set_ylim(ylims)

    return figh