"""
Tools for reconstructiong optical diffraction tomography (ODT) data
"""
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import time
import fit

def get_fz(fx, fy, ni, wavelength):
    """
    Get z-component of frequency given fx, fy

    @param frqs_2d: nfrqs x 2
    @param ni: index of refraction
    @param wavelength: wavelength
    @return frqs_3d:
    """

    with np.errstate(invalid="ignore"):
        fzs = np.sqrt(ni**2 / wavelength ** 2 - fx**2 - fy**2)

    return fzs


def get_angles(frqs, no, wavelength):
    with np.errstate(invalid="ignore"):
        theta = np.arccos(np.dot(frqs, np.array([0, 0, 1])) / (no / wavelength))
        theta[np.isnan(theta)] = 0
        phi = np.angle(frqs[:, 0] + 1j * frqs[:, 1])
        phi[np.isnan(phi)] = 0

    return theta, phi


def get_global_phase_shifts(imgs, ref_ind):
    """
    Given a stack of images and a reference, determine the phase shifts between images, such that
    imgs * np.exp(1j * phase_shift) ~ img_ref

    @param imgs:
    @param ref_ind:
    @return phase_shifts:
    """
    nimgs = imgs.shape[0]

    phase_shifts = np.zeros(nimgs)
    # loop over non-reference images
    inds_list = list(range(nimgs))
    inds_list.remove(ref_ind)
    for ii in inds_list:
        def fn(p): return np.abs(imgs[ii] * np.exp(1j * p[0]) - imgs[ref_ind]).ravel()
        results = fit.fit_least_squares(fn, [0])
        phase_shifts[ii] = results["fit_params"]

    return phase_shifts


def get_n(scattering_pot, no, wavelength):
    """
    convert from the scattering potential to the index of refraction
    @param scattering_pot:
    @param no:
    @param wavelength:
    @return:
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


def reconstruction(efield_fts, beam_frqs, ni, na_det, wavelength, dxy, z_fov=10, reg=0.1):
    """

    @param efield_fts:
    @param beam_frqs:
    @param ni:
    @param na_det:
    @param wavelength:
    @param dxy:
    @param z_fov:
    @param reg:
    @return:
    """
    nimgs, ny, nx = efield_fts.shape

    # ##################################
    # get frequencies of initial images and make broadcastable to shape (nimgs, ny, nx)
    # ##################################
    fx = np.expand_dims(fft.fftshift(fft.fftfreq(nx, dxy)), axis=(0, 1))
    fy = np.expand_dims(fft.fftshift(fft.fftfreq(ny, dxy)), axis=(0, 2))
    fz = get_fz(fx, fy, ni, wavelength)

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

    freq_nan = np.isnan(Fz)
    to_use_data = np.logical_not(freq_nan)

    Fx[freq_nan] = np.nan
    Fy[freq_nan] = np.nan

    # ##################################
    # set sampling of 3D scattering potential
    # ##################################
    theta, _ = get_angles(beam_frqs, ni, wavelength)
    alpha = np.arcsin(na_det / ni)
    beta = np.max(theta)

    f_perp_max = (na_det + ni * np.sin(beta)) / wavelength
    fz_max = ni / wavelength * (1 - np.cos(alpha))
    dxy_sp = 0.5 * 1 / f_perp_max
    dz_sp = 0.5 * 1 / fz_max

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

    x_sp = dxy_sp * (np.arange(nx_sp) - nx_sp // 2)
    y_sp = dxy_sp * (np.arange(ny_sp) - ny_sp // 2)
    z_sp = dz_sp * (np.arange(nz_sp) - nz_sp // 2)

    fx_sp = fft.fftshift(fft.fftfreq(nx_sp, dxy_sp))
    fy_sp = fft.fftshift(fft.fftfreq(ny_sp, dxy_sp))
    fz_sp = fft.fftshift(fft.fftfreq(nz_sp, dz_sp))
    dfx_sp = fx_sp[1] - fx_sp[0]
    dfy_sp = fy_sp[1] - fy_sp[0]
    dfz_sp = fz_sp[1] - fz_sp[0]

    fzfz_sp, fyfy_sp, fxfx_sp = np.meshgrid(fz_sp, fy_sp, fx_sp, indexing="ij")

    # ##################################
    # define method for mapping pixels
    # ##################################

    def kernel(dx, dy, dz):
        return np.logical_and.reduce((np.abs(dx) <= dfx_sp / 2, np.abs(dy) <= dfy_sp / 2, np.abs(dz) <= dfz_sp / 2))

    # todo: could also think about looping over only pixel positions present in data

    # one reconstruction per image
    tstart = time.perf_counter()

    sp_ft_imgs = np.zeros((nimgs, nz_sp, ny_sp, nx_sp), dtype=complex) * np.nan
    num_pts = np.zeros((nimgs, nz_sp, ny_sp, nx_sp), dtype=int)
    for ii in range(nimgs):
        # F(kx - k*nx, ky - k * ny, kz - k * nz) = 2 * 1j * kz * E(kx, ky))
        f_unshift_ft = 2 * 1j * (2 * np.pi * fz[0]) * efield_fts[ii]

        # expand to nz x nx x ndata points
        f_unshift_ft_exp = np.expand_dims(f_unshift_ft[to_use_data[ii]], axis=(0, 1))

        # loop over y-direction, but do x/z in parallel. Much faster than looping over all pixels
        for jj in range(ny_sp):
            print("img %d/%d, y-coord %d/%d, elapsed time = %0.2fs" %
                  (ii + 1, nimgs, jj + 1, ny_sp, time.perf_counter() - tstart), end="\r")
            if jj == (ny_sp - 1):
                print("")

            # expand array so that is nz x nx x ndata points
            dist_xs = np.expand_dims(fxfx_sp[:, jj], axis=-1) - np.expand_dims(Fx[ii, to_use_data[ii]], axis=(0, 1))
            dist_ys = np.expand_dims(fyfy_sp[:, jj], axis=-1) - np.expand_dims(Fy[ii, to_use_data[ii]], axis=(0, 1))
            dist_zs = np.expand_dims(fzfz_sp[:, jj], axis=-1) - np.expand_dims(Fz[ii, to_use_data[ii]], axis=(0, 1))
            kern = kernel(dist_xs, dist_ys, dist_zs)

            # sum over data points axis
            # also must correct fact Fourier transform normalization is effectively different for different sized arrays
            sp_ft_imgs[ii, :, jj] = np.sum(kern * f_unshift_ft_exp, axis=-1) * (nx_sp * ny_sp * nz_sp) / (ny * nx)
            # track number of points for later averaging
            num_pts[ii, :, jj] = np.sum(kern, axis=-1)

    sp_ft = np.sum(sp_ft_imgs, axis=0) / (np.sum(num_pts, axis=0) + reg)

    fcoords = (fx_sp, fy_sp, fz_sp)
    coords = (x_sp, y_sp, z_sp)

    return sp_ft, sp_ft_imgs, coords, fcoords


def apply_n_constraints(scattering_pot_ft, no, wavelength, n_iterations=100, beta=0.5, use_raar=True):
    """
    Iterative apply constraints

    @param scattering_pot_ft: 3D fourier transform of scattering potential
    @param no: background index of refraction
    @param wavelength:
    @param n_iterations:
    @param beta:
    @param bool use_raar: whether or not to use the Relaxed-Averaged-Alternating Reflection algorithm
    @return scattering_pot_ft:
    """
    # scattering_potential masked with nans where no information
    scattering_pot_ft = np.array(scattering_pot_ft, copy=True)
    sp_data = np.array(scattering_pot_ft, copy=True)

    no_data = np.isnan(scattering_pot_ft)
    is_data = np.logical_not(no_data)
    scattering_pot_ft[no_data] = 0 #np.exp(1j * np.random.rand(*scattering_potential_ft[no_data].shape))

    for ii in range(n_iterations):
        # ############################
        # ensure n is physical
        # ############################
        sp = fft.fftshift(fft.ifftn(fft.ifftshift(scattering_pot_ft)))
        n = get_n(sp, no, wavelength)

        # real part must be >= 1
        correct_real = np.real(n) < 1
        n[correct_real] = 1 + np.imag(n[correct_real])

        # imaginary part must be >= 0
        correct_imag = np.imag(n) < 0
        n[correct_imag] = np.real(n[correct_imag]) + 0*1j

        sp_ps = get_scattering_potential(n, no, wavelength)
        sp_ps_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps)))

        # ############################
        # ensure img matches data
        # ############################
        sp_ft_pm = np.array(scattering_pot_ft, copy=True)
        sp_ft_pm[is_data] = sp_data[is_data]

        # ############################
        # projected Pm * Ps
        # ############################
        sp_ft_pm_ps = np.array(sp_ps_ft, copy=True)
        sp_ft_pm_ps[is_data] = sp_data[is_data]

        # ############################
        # projected Ps * Pm
        # ############################
        sp_ft_ps_pm = np.array(sp_ft_pm, copy=True)
        sp_ps_pm = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft_ps_pm)))
        n_ps_pm = get_n(sp_ps_pm, no, wavelength)

        # real part must be >= 1
        correct_real = np.real(n_ps_pm) < 1
        n_ps_pm[correct_real] = 1 + np.imag(n_ps_pm[correct_real])

        # imaginary part must be >= 0
        correct_imag = np.imag(n_ps_pm) < 0
        n_ps_pm[correct_imag] = np.real(n_ps_pm[correct_imag]) + 0 * 1j

        sp_ps_pm = get_scattering_potential(n_ps_pm, no, wavelength)
        sp_ps_pm_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps_pm)))

        # ############################
        # update
        # ############################
        if use_raar:
            scattering_pot_ft = beta * scattering_pot_ft - beta * sp_ps_ft + (1 - 2 * beta) * sp_ft_pm + 2 * beta * sp_ps_pm_ft
        else:
            scattering_pot_ft = sp_ft_pm_ps

    return scattering_pot_ft


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
    plt.suptitle("red = maximum extent of frequency info\n"
                 "blue = maximum extent of centers")
    grid = plt.GridSpec(1, 2)

    # ########################
    # kx-kz plane
    # ########################
    ax = plt.subplot(grid[0, 0])
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
    ax = plt.subplot(grid[0, 1])
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
