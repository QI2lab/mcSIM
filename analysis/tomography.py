"""
Tools for reconstructiong optical diffraction tomography (ODT) data
"""
import numpy as np
from numpy import fft
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.patches import Circle, Arc
from matplotlib.cm import ScalarMappable
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
    """
    Convert from frequency vectors to angle vectors
    @param frqs:
    @param no:
    @param wavelength:
    @return:
    """
    with np.errstate(invalid="ignore"):
        theta = np.arccos(np.dot(frqs, np.array([0, 0, 1])) / (no / wavelength))
        theta[np.isnan(theta)] = 0
        phi = np.angle(frqs[:, 0] + 1j * frqs[:, 1])
        phi[np.isnan(phi)] = 0

    return theta, phi


def get_global_phase_shifts(imgs, ref_imgs):
    """
    Given a stack of images and a reference, determine the phase shifts between images, such that
    imgs * np.exp(1j * phase_shift) ~ img_ref

    @param imgs:
    @param ref_ind:
    @return phase_shifts:
    """
    nimgs = imgs.shape[0]

    # if using only single ref image ...
    if ref_imgs.ndim == (imgs.ndim - 1):
        ref_imgs = np.expand_dims(ref_imgs, axis=0)

    ref_imgs, imgs = np.broadcast_arrays(ref_imgs, imgs)

    tstart = time.perf_counter()
    phase_shifts = np.zeros(nimgs)
    for ii in range(nimgs):
        print("computing phase shift %d/%d, elapsed time = %0.2fs" % (ii + 1, nimgs, time.perf_counter() - tstart), end="\r")
        def fn(p): return np.abs(imgs[ii] * np.exp(1j * p[0]) - ref_imgs[ii]).ravel()
        # s1 = np.mean(imgs[ii])
        # s2 = np.mean(ref_imgs[ii])
        # def fn(p): return np.abs(s1 * np.exp(1j * p[0]) - s2)
        results = fit.fit_least_squares(fn, [0])
        phase_shifts[ii] = results["fit_params"]
    print("")

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


def reconstruction(efield_fts, beam_frqs, ni, na_det, wavelength, dxy, z_fov=10, reg=0.1, dz_sampling_factor=1,
                   mode="born"):
    """

    @param efield_fts: The exact definition of efield_fts depends on whether "born" or "rytov" mode is used
    @param beam_frqs: nimgs x 3, where each is [vx, vy, vz] and vx**2 + vy**2 + vz**2 = n**2 / wavelength**2
    @param ni: background index of refraction
    @param na_det: detection numerical aperture
    @param wavelength:
    @param dxy: pixel size
    @param z_fov: z-field of view
    @param reg: regularization factor
    @param dz_sampling_factor: fraction of Nyquist sampling factor to use
    @param mode: "born" or "rytov"
    @return sp_ft, sp_ft_imgs, coords, fcoords:
    """
    nimgs, ny, nx = efield_fts.shape

    # ##################################
    # get frequencies of initial images and make broadcastable to shape (nimgs, ny, nx)
    # ##################################
    fx = np.expand_dims(fft.fftshift(fft.fftfreq(nx, dxy)), axis=(0, 1))
    fy = np.expand_dims(fft.fftshift(fft.fftfreq(ny, dxy)), axis=(0, 2))
    fz = get_fz(fx, fy, ni, wavelength)

    not_detectable = (fx**2 + fy**2) > na_det / wavelength
    fz[not_detectable] = np.nan

    # ##################################
    # set sampling of 3D scattering potential
    # ##################################
    theta, _ = get_angles(beam_frqs, ni, wavelength)
    alpha = np.arcsin(na_det / ni)
    beta = np.max(theta)

    # maximum frequencies present in ODT
    f_perp_max = (na_det + ni * np.sin(beta)) / wavelength
    fz_max = ni / wavelength * np.max([1 - np.cos(alpha), 1 - np.cos(beta)])

    # generate realspace sampling from Nyquist sampling
    dxy_sp = 0.5 * 1 / f_perp_max
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
    # ##################################

    # find indices into reconstruction from rounding
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

        freq_nan = np.isnan(Fz)
        Fx[freq_nan] = np.nan
        Fy[freq_nan] = np.nan

        # F(fx - n/lambda * nx, fy - n/lambda * ny, fz - n/lambda * nz) = 2*i * (2*pi*fz) * Es(fx, fy)
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
        with np.errstate(invalid="ignore"):
            fz_rytov = np.sqrt((ni / wavelength)**2 - fx_rytov**2 - fy_rytov**2)
        Fz_rytov = fz_rytov - np.expand_dims(beam_frqs[:, 2], axis=(1, 2))

        zind = (np.round(Fz_rytov / dfz_sp) + nz_sp // 2).astype(int)
        yind = (np.round(Fy_rytov / dfy_sp) + ny_sp // 2).astype(int)
        xind = (np.round(Fx_rytov / dfx_sp) + nx_sp // 2).astype(int)

        zind, yind, xind = np.broadcast_arrays(zind, yind, xind)
        zind = np.array(zind, copy=True)
        yind = np.array(yind, copy=True)
        xind = np.array(xind, copy=True)
    else:
        raise ValueError("'mode' must be 'born' or 'rytov' but was '%s'" % mode)

    # only use those within bounds. will have some negative because not checking against NA
    to_use_ind = np.logical_and.reduce((zind >= 0, zind < nz_sp,
                                        yind >= 0, yind < ny_sp,
                                        xind >= 0, xind < nx_sp))

    # convert nD indices to 1D indices so can easily check if points are unique
    cind = -np.ones(zind.shape, dtype=int)
    cind[to_use_ind] = np.ravel_multi_index((zind[to_use_ind].astype(int).ravel(),
                                             yind[to_use_ind].astype(int).ravel(),
                                             xind[to_use_ind].astype(int).ravel()), (nz_sp, ny_sp, nx_sp))

    # one reconstruction per image
    tstart = time.perf_counter()

    sp_ft_imgs = np.zeros((nimgs, nz_sp, ny_sp, nx_sp), dtype=complex) * np.nan
    num_pts = np.zeros((nimgs, nz_sp, ny_sp, nx_sp), dtype=int)
    for ii in range(nimgs):
        print("reconstructing angle %d/%d, elapsed time = %0.2fs" % (ii + 1, nimgs, time.perf_counter() - tstart), end="\r")
        if ii == (nimgs - 1):
            print("")

        if mode == "born":
            f_unshift_ft = 2 * 1j * (2 * np.pi * fz[0]) * efield_fts[ii]
        elif mode == "rytov":
            f_unshift_ft = 2 * 1j * (2 * np.pi * fz_rytov[ii]) * efield_fts[ii]

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
        sp_ft_imgs[ii][inds_angle] = f_unshift_ft[to_use_ind[ii]] * (dxy * dxy) / (dxy_sp * dxy_sp * dz_sp)
        num_pts[ii][inds_angle] = 1

    # average over angles/images
    num_pts_all = np.sum(num_pts, axis=0)
    no_data = num_pts_all == 0

    sp_ft = np.nansum(sp_ft_imgs, axis=0) / (num_pts_all + reg)
    sp_ft[no_data] = np.nan


    fcoords = (fx_sp, fy_sp, fz_sp)
    coords = (x_sp, y_sp, z_sp)

    return sp_ft, sp_ft_imgs, coords, fcoords


def apply_n_constraints(sp_ft, no, wavelength, n_iterations=100, beta=0.5, use_raar=True):
    """
    Iterative apply constraints on the scattering potential and the index of refraction

    constraint 1: scattering potential FT must match data at points where we information
    constraint 2: real(n) >= no and imag(n) >= 0

    @param sp_ft: 3D fourier transform of scattering potential
    @param no: background index of refraction
    @param wavelength:
    @param n_iterations:
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
    sp_ft[no_data] = 0 #np.exp(1j * np.random.rand(*scattering_potential_ft[no_data].shape))

    # try smoothing image first ...
    sp_ft = gaussian_filter(sp_ft, (2, 2, 2))

    for ii in range(n_iterations):
        # ############################
        # ensure n is physical
        # ############################
        sp = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft)))
        n = get_n(sp, no, wavelength)

        # real part must be >= no
        correct_real = np.real(n) < no
        n[correct_real] = no + np.imag(n[correct_real])

        # imaginary part must be >= 0
        correct_imag = np.imag(n) < 0
        n[correct_imag] = np.real(n[correct_imag]) + 0*1j

        sp_ps = get_scattering_potential(n, no, wavelength)
        sp_ps_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps)))

        # ############################
        # ensure img matches data
        # ############################
        sp_ft_pm = np.array(sp_ft, copy=True)
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

        # real part must be >= no
        correct_real = np.real(n_ps_pm) < no
        n_ps_pm[correct_real] = no + np.imag(n_ps_pm[correct_real])

        # imaginary part must be >= 0
        correct_imag = np.imag(n_ps_pm) < 0
        n_ps_pm[correct_imag] = np.real(n_ps_pm[correct_imag]) + 0 * 1j

        sp_ps_pm = get_scattering_potential(n_ps_pm, no, wavelength)
        sp_ps_pm_ft = fft.fftshift(fft.fftn(fft.ifftshift(sp_ps_pm)))

        # ############################
        # update
        # ############################
        if use_raar:
            sp_ft = beta * sp_ft - beta * sp_ps_ft + (1 - 2 * beta) * sp_ft_pm + 2 * beta * sp_ps_pm_ft
        else:
            sp_ft = sp_ft_pm_ps

    return sp_ft


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

def plot_n3d(sp_ft, no, wavelength, coords, title=""):
    """
    Plot 3D index of refraction
    @param sp_ft:
    @param no:
    @param wavelength:
    @param coords:
    @return:
    """

    # work with coordinates
    x_sp, y_sp, z_sp  = coords
    nz_sp = len(z_sp)
    dxy_sp = x_sp[1] - x_sp[0]
    dz_sp = z_sp[1] - z_sp[0]

    extent_sp_xy = [x_sp[0] - 0.5 * dxy_sp, x_sp[-1] + 0.5 * dxy_sp, y_sp[0] - 0.5 * dxy_sp, y_sp[-1] + 0.5 * dxy_sp]
    extent_sp_xz = [x_sp[0] - 0.5 * dxy_sp, x_sp[-1] + 0.5 * dxy_sp, z_sp[0] - 0.5 * dz_sp, z_sp[-1] + 0.5 * dz_sp]
    extent_sp_yz = [y_sp[0] - 0.5 * dxy_sp, y_sp[-1] + 0.5 * dxy_sp, z_sp[0] - 0.5 * dz_sp, z_sp[-1] + 0.5 * dz_sp]

    # get need quantities
    sp_ft = np.array(sp_ft, copy=True)
    sp_ft[np.isnan(sp_ft)] = 0
    sp = fft.fftshift(fft.ifftn(fft.ifftshift(sp_ft)))

    n_recon = get_n(sp, no, wavelength)
    n_recon_ft = fft.fftshift(fft.fftn(fft.ifftshift(n_recon)))

    vmax_real = 1.5 * np.max(np.real(n_recon) - no)
    vmax_imag = 1.5 * np.max(np.imag(n_recon))
    vmax_n_ft = 1.5 * np.max(np.abs(n_recon_ft))
    vmax_sp_ft = 1.5 * np.max(np.abs(sp_ft))

    # plots
    fmt_fn = lambda x: "%0.6f" % x

    figh = plt.figure(figsize=(16, 8))
    plt.suptitle("Index of refraction, %s" % title)
    grid = plt.GridSpec(4, nz_sp + 1)

    for ii in range(nz_sp):
        ax = plt.subplot(grid[0, ii])
        ax.set_title("%0.1fum" % z_sp[ii])
        im = ax.imshow(np.real(n_recon[ii]) - no, vmin=-vmax_real, vmax=vmax_real, cmap="RdBu", origin="lower", extent=extent_sp_xy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])

        if ii == 0:
            ax.set_ylabel("real(n) - no")

        ax = plt.subplot(grid[1, ii])
        im = ax.imshow(np.imag(n_recon[ii]), vmin=-vmax_imag, vmax=vmax_imag, cmap="RdBu", origin="lower", extent=extent_sp_xy)
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("imag(n)")

        ax = plt.subplot(grid[2, ii])
        im = ax.imshow(np.abs(n_recon_ft[ii]), cmap="bone", norm=PowerNorm(gamma=0.1, vmin=0, vmax=vmax_n_ft), origin="lower")
        im.format_cursor_data = fmt_fn
        ax.set_xticks([])
        ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel("|n(f)|")

        ax = plt.subplot(grid[3, ii])
        im = ax.imshow(np.abs(sp_ft[ii]), cmap="bone", norm=PowerNorm(gamma=0.1, vmin=0, vmax=vmax_sp_ft), origin="lower")
        im.format_cursor_data = fmt_fn
        if ii == 0:
            ax.set_ylabel("|F(f)|")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = plt.subplot(grid[0, nz_sp])
    ax.axis("off")
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax_real, vmax=vmax_real), cmap="RdBu"), ax=ax)

    ax = plt.subplot(grid[1, nz_sp])
    ax.axis("off")
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax_imag, vmax=vmax_imag), cmap="RdBu"), ax=ax)

    return figh