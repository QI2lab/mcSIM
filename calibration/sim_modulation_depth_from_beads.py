"""
Determine SIM modulation depth from multicolor imaging of fluorescence microspheres/beads. The modulation depth can
be estimated from the amplitudes of the beads in each SIM picture using an optical sectioning SIM reconstruction
type approach.

This script handles doing this with multiple directories, where each directory contains multidimensional data which
may include color channels, time series, and z-scans. It does not handle different xy-positions.
"""
import datetime
import numpy as np
import os
from pathlib import Path
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.signal
import scipy.optimize
import zarr

import mcsim.analysis.sim_reconstruction as sim
import localize_psf.affine as affine
import localize_psf.rois as roi_fns
import localize_psf.localize as localize


tstamp = datetime.datetime.now().strftime("%Y_%m_%d_%H;%M;%S")

# data files
root_path = Path(r"G://2022_04_14")
data_dirs = [root_path / "04_200nm_beads_sim_mod_depth"]
# color channel information
ignore_color = [False, False, False]
min_fit_amp = [5000, 1000, 300]
bead_radii = 0.5 * np.array([0.2, 0.2, 0.2])
nrois_to_plot = 0

# set parameters
dxy = 0.065  # um
figsize = (21, 9)
save_results = True
close_fig_after_saving = True
nangles = 3
nphases = 3
# roi/filtering parameters in um
roi_size = (0, 1.5, 1.5)  # (sz, sy, sx) in um
filters_small_sigma = (0.1, 0.1, 0.1)  # (sz, sy, sx) in um
filters_large_sigma = (5, 5, 5)  # (sz, sy, sx) in um
min_boundary_distance = (1, 1)  # (dz, dxy) in um
sigma_bounds = ((0, 0.05), (3, 3))
min_spot_sep = (0, 1) # (dz, dxy) in um

# turn off interactive plotting
plt.ioff()
plt.switch_backend("agg")


# ################################
# load affine xform data
# ################################
affine_paths = ["2021-04-13_12;49;32_affine_xform_red_z=0.pkl",
                "2021-04-13_12;49;32_affine_xform_blue_z=0.pkl",
                "2021-04-13_12;49;32_affine_xform_green_z=0.pkl"]
affine_paths = [root_path / d for d in affine_paths]
affine_xforms = []
for p in affine_paths:
    with open(p, "rb") as f:
        dat = pickle.load(f)
    affine_xforms.append(dat["affine_xform"])

# ################################
# load pattern data
# ################################
# todo: load in correct order
pattern_paths = [r"period=6.0_nangles=3\wavelength=635nm\sim_patterns_period=7.98_nangles=3.pkl",
                 r"period=6.0_nangles=3\wavelength=473nm\sim_patterns_period=6.01_nangles=3.pkl",
                 r"period=6.0_nangles=3\wavelength=532nm\sim_patterns_period=6.82_nangles=3.pkl"]


pattern_paths = [root_path / d for d in pattern_paths]
pattern_dat = []
xformed_angles = []
periods = []
frqs = np.zeros((len(ignore_color), nangles, 2))
phases = np.zeros((len(ignore_color), nangles, nphases))
for ii, p in enumerate(pattern_paths):
    with open(p, "rb") as f:
        dat = pickle.load(f)

    pattern_dat.append(dat)

    fxt, fyt, phit = affine.xform_sinusoid_params(dat["frqs"][:, 0], dat["frqs"][:, 1],
                                                  dat["phases"], affine_xforms[ii])
    frqs[ii, ..., 0] = 2 * fxt / dxy
    frqs[ii, ..., 1] = 2 * fyt / dxy
    phases[ii] = 2 * phit

    xformed_angles.append(np.angle(fxt + 1j * fyt))
    periods.append(0.5 / np.sqrt(fxt**2 + fyt**2) * dxy)

# ################################
# loop over data dirs and estimate modulation depth
# ################################
for d in data_dirs:
    file_path = list(d.glob("*.zarr"))[0]
    data = zarr.open(file_path, "r")

    imgs = np.array(data.sim)
    ntimes, nz, ncolors, npatterns, ny, nx = imgs.shape
    imgs = imgs.reshape([ntimes, nz, ncolors, nangles, nphases, ny, nx])
    dz = data.attrs["dz_um"]
    zs = data.attrs["z_position_um"]
    exposure_t = data.sim.attrs["exposure_time_ms"]
    channels = data.sim.attrs["channels"]

    # metadata, dims, summary = mm_io.parse_mm_metadata(d)
    #
    # nxy = dims['position']
    # ntimes = dims['time']
    # ncolors = dims['channel']
    # nz = dims['z']
    # exposure_t = metadata["Exposure-ms"][0]
    #     zs = np.unique(metadata["ZPositionUm"])

    iz_center = nz // 2
    # if len(zs) > 1:
    #     dz = zs[1] - zs[0]
    # else:
    #     dz = 1.

    # make save directory
    if save_results:
        save_dir = Path(d, f"{tstamp:s}_sim_modulation_depth")
        save_dir.mkdir(exist_ok=True)


    # #################################################
    # fit images for all times/z-planes/channels/angles/phases only
    # #################################################
    # each entry in these lists will have size nfits x ntimes x nz x nangles x nphases x other axes
    rois = [[]] * ncolors
    fps = [[]] * ncolors
    # fit parameters for central z-index and first time, each entry size nfits x nangles
    fps_start = [[]] * ncolors

    tstart = time.perf_counter()
    for ic in range(ncolors):
        # #################################################
        # load first image and identify spots, which will be used in all subsequent analysis
        # #################################################
        if ignore_color[ic]:
            continue
        print("started initial fitting for channel %d" % ic)

        # img_inds = list(range(nignored_frames, nignored_frames + 9))
        # # grab first image and average over angles/phases to get all beads
        # img_first = np.mean(mm_io.read_mm_dataset(metadata, time_indices=0, z_indices=iz_center,
        #                                           user_indices={"UserChannelIndex": ic, "UserSimIndex": img_inds}),
        #                     axis=0)

        # take widefield image
        img_middle_wf = np.mean(imgs[0, nz//2, ic], axis=(0, 1))

        # identify beads in first image
        coords, fps_temp, ips_temp, rois_temp,\
        to_keep, conditions, condition_names, filter_settings,\
        fit_states, chi_sqrs, niters, imgs_filtered = \
            localize.localize_beads(img_middle_wf, dxy, dz, min_fit_amp[ic], roi_size=roi_size,
                                    filter_sigma_small=filters_small_sigma, filter_sigma_large=filters_large_sigma,
                                    min_spot_sep=min_spot_sep,
                                    dist_boundary_min=min_boundary_distance, sigma_bounds=sigma_bounds,
                                    fit_amp_min=min_fit_amp[ic])
        z, y, x = coords
        zz, yy, xx = np.broadcast_arrays(*(z, y, x))

        if not np.any(to_keep):
            raise ValueError("no beads found...")

        rois[ic] = rois_temp[to_keep]
        fps_start[ic] = fps_temp[to_keep]

        # define array to hold later fit results
        fps[ic] = np.zeros((len(rois[ic]), ntimes, nz, nangles, nphases, 7))

        # plot localizations
        centers_temp_all_pix = np.stack((fps_temp[:, 3],
                                         fps_temp[:, 2],
                                         fps_temp[:, 1]), axis=-1)
        centers_temp_pix = np.stack((fps_start[ic][:, 3],
                                     fps_start[ic][:, 2],
                                     fps_start[ic][:, 1]), axis=-1)
        figh1 = localize.plot_bead_locations(img_middle_wf, [centers_temp_all_pix, centers_temp_pix],
                                             coords=(yy[0], xx[0]),
                                             legend_labels=["all fits", "reliable fits"],
                                             weights=[fps_temp[:, 4], fps_start[ic][:, 4]],
                                             cbar_labels=[r"$\sigma_{xy} (\mu m)$", r"$\sigma_{xy} (\mu m)$"],
                                             vlims_percentile=(0.001, 99.99), gamma=0.5,
                                             title="spot centers, channel %d = %s, threshold=%.0f" % (ic, channels[ic], min_fit_amp[ic]),
                                             figsize=figsize)

        if save_results:
            figh1.savefig(Path(save_dir, f"bead_centers_channel={ic:d}.png"))
        if close_fig_after_saving:
            plt.close(figh1)

        # plot fits if desired
        for aaa in range(nrois_to_plot):
            localize.plot_gauss_roi(fps_start[ic][aaa], rois[ic][aaa], np.expand_dims(img_middle_wf, axis=0),
                                    coords,
                                    #init_params=fps_start[ic][aaa],
                                    prefix=f"reference_roi_ic={ic:d}_iz={iz_center:d}_it={0:d}_roi={aaa:d}_",
                                    save_dir=save_dir)

        # #################################################
        # do fitting for all other z-slices/times/angles/phases using same ROI's with fixed centers
        # #################################################
        for iz in range(nz):
            for it in range(ntimes):
                for ia in range(nangles):
                    #sim_inds = list(range(nignored_frames + ia * nphases, nignored_frames + (ia + 1) * nphases))
                    for ip in range(nphases):
                        print(f"{it+1:d}/{ntimes:d} times,"
                              f" {iz + 1:d}/{nz:d} zs,"
                              f" {ic + 1:d}/{ncolors:d} colors,"
                              f"{ia+1:d}/{nangles:d} angles,"
                              f" {ip+1:d}/{nphases:d} phases,"
                              f" t elapsed={time.perf_counter() - tstart:.2f}")

                        # imgs = mm_io.read_mm_dataset(metadata, time_indices=it, z_indices=iz,
                        #                              user_indices={"UserChannelIndex": ic,
                        #                                            "UserSimIndex": sim_inds[ip]})

                        imgs_now = np.expand_dims(imgs[it, iz, ic, ia, ip], axis=0)

                        # get ROI's to fit spots
                        img_rois = [roi_fns.cut_roi(r, imgs_now) for r in rois[ic]]
                        coords_rois = [[roi_fns.cut_roi(r, z), roi_fns.cut_roi(r, y), roi_fns.cut_roi(r, x)]
                                       for r in rois[ic]]
                        coords_rois = list(zip(*coords_rois))
                        fixed_params = np.zeros((7), dtype=bool)
                        fixed_params[1:4] = True
                        fixed_params[-1] = True

                        # do fitting
                        # todo: fitting on gpu not working ... not sure if the problem is the fixed parameters or something else
                        fps[ic][:, it, iz, ia, ip, :], fit_states, chi_sqrs, niters, fit_t = \
                            localize.fit_gauss_rois(img_rois, coords_rois, fps_start[ic], fixed_params=fixed_params, use_gpu=False) #, use_gpu=False #, debug=True)

                        for aaa in range(nrois_to_plot):
                            localize.plot_gauss_roi(fps[ic][aaa, it, iz, ia, ip], rois[ic][aaa], imgs_now,
                                                    coords,
                                                    init_params=fps_start[ic][aaa],
                                                    prefix=f"roi_ic={ic:d}_iz={iz:d}_it={it:d}_ia={ia:d}_ip={ip:d}_roi={aaa:d}_",
                                                    save_dir=save_dir)

    # ####################################
    # compute modulation depth statistics
    # ####################################
    # parameters extracted from fits, size nfits x ntimes x nz x nangles
    # these combine the phase images, either necessarily as we need all three to estimate the modulation depth
    # or we take an average to get the best estimate of the parameter
    fps_mean = [[]] * ncolors
    m_ests = [[]] * ncolors
    cos_phis = [[]] * ncolors

    for ic in range(ncolors):
        if ignore_color[ic]:
            continue
        # mean value of parameters over phase axis
        fps_mean[ic] = np.mean(fps[ic], axis=-2)

        # estimate modulation depth and amplitude
        m_ests[ic] = sim.sim_optical_section(fps[ic][..., 0], axis=-1) / fps_mean[ic][..., 0]

        # if any amplitudes were negative, m_est will not be any good, so throw it away
        amp_is_neg = fps[ic][..., 0] < 0
        m_ests[ic][np.any(amp_is_neg, axis=-1)] = np.nan

        # cos phis
        # cos_phis[ic] = (fps_mean[ic][..., 0] / np.expand_dims(fps_mean[ic][..., 0], axis=-1) - 1) /\
        #                np.expand_dims(m_ests[ic], axis=-1)

    # ####################################
    # plot results for each z/t/angle
    # ####################################
    tstart = time.perf_counter()
    for iz in range(nz):
        for it in range(ntimes):

            # todo: think better to have one plot per color
            print("plotting iz %d/%d, it %d/%d, elapsed t=%0.2fs" %
                  (iz + 1, nz, it + 1, ntimes, time.perf_counter() - tstart))

            # figure plotting all data for a given time point
            figh = plt.figure(figsize=figsize)
            figh.suptitle(f"{d}\nz={iz:d}, time={it:d}, exposure={exposure_t:.1f}ms")

            nplots_per_angle = 3
            grid = figh.add_gridspec(ncolors, nplots_per_angle * nangles, wspace=0.1, hspace=0.5)

            for ic in range(ncolors):
                if ignore_color[ic]:
                    continue

                mod_depth_bead_size_correction = sim.correct_modulation_for_bead_size(bead_radii[ic], 1 / (periods[ic][ia]))

                for ia in range(nangles):
                    # calculate modulation depth statistics
                    m_temp = m_ests[ic][:, it, iz, ia]
                    m_allowed = m_temp[np.logical_and(m_temp > 0, m_temp < 1)]
                    mean_depth = np.mean(m_allowed)
                    std_depth = np.std(m_allowed)
                    bin_edges = np.linspace(0, 1.25, 50)
                    ms_hist, bin_edges = np.histogram(m_temp, bins=bin_edges)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                    # amplitude statistics
                    # amp_temp = np.mean(amp_ests[ic][:, it, iz], axis=-1)
                    amp_temp = np.mean(fps_mean[ic][:, it, iz, :, 0], axis=-1)
                    mean_amp = np.mean(amp_temp)
                    med_amp = np.median(amp_temp)
                    std_amp = np.std(amp_temp)

                    # sigma statistics
                    sigma_means_temp = np.mean(fps[ic][:, it, iz, ia, :, 4], axis=-1)
                    med_sigma = np.median(sigma_means_temp)

                    imgs_now_wf = np.mean(imgs[it, iz, ic], axis=(0, 1))

                    centers_temp = np.stack((fps[ic][:, it, iz, ia, 0, 3],
                                             fps[ic][:, it, iz, ia, 0, 2],
                                             fps[ic][:, it, iz, ia, 0, 1]), axis=-1)

                    figh2 = localize.plot_bead_locations(imgs_now_wf, centers_temp, weights=m_ests[ic][:, it, iz, ia],
                                                         coords=(yy[0], xx[0]),
                                                         title="modulation depth versus position ic=%d = %s, iz=%d, it=%d, ia=%d\n"
                                                               "m = %0.3f +/- %0.3f\nadjusted=%0.3f" %
                                                               (ic, channels[ic], iz, it, ia, mean_depth, std_depth,
                                                                mean_depth / mod_depth_bead_size_correction),
                                                         cbar_labels=["modulation depth"],
                                                         vlims_percentile=(0.001, 99.99), gamma=0.5, figsize=figsize)

                    if save_results:
                        figh2.savefig(Path(save_dir, f"beads_ic={ic:d}_angle={ia:d}_time={it:d}_z={iz:d}.png"))
                    if close_fig_after_saving:
                        plt.close(figh2)

                    # make paramter distribution figure active again...
                    plt.figure(figh.number)

                    # histogram of modulation depths
                    ax = plt.subplot(grid[ic, ia * nplots_per_angle])
                    ax.set_title("mean=%0.3f(%.0f)\npeak=%0.3f" %
                                 (mean_depth / mod_depth_bead_size_correction,
                                  std_depth * 1e3 / mod_depth_bead_size_correction,
                                  bin_centers[np.argmax(ms_hist)]))

                    ax.plot(bin_centers, ms_hist)

                    ax.set_xlim([-0.05, 1.15])
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    if ia == 0:
                        ax.set_ylabel(f"{channels[ic]}\nangle={ia:d}")
                    else:
                        ax.set_ylabel("angle=%d" % ia)

                    # modulation depths versus amplitude
                    ax = plt.subplot(grid[ic, ia * nplots_per_angle + 1])
                    ax.set_title("amp\n med %0.2f" % (med_amp))

                    ax.plot([-0.05, 1.15], [med_amp, med_amp], 'b')
                    ax.plot([-0.05, 1.15], [0, 0], 'k')
                    ax.plot(m_temp, amp_temp, '.')

                    ax.set_xlim([-0.05, 1.15])
                    ax.set_ylim([-10, 2 * np.percentile(amp_temp, 90)])
                    ax.set_yticks([])
                    ax.set_yticklabels([])

                    # size versus amplitude
                    ax = plt.subplot(grid[ic, ia * nplots_per_angle + 2])
                    ax.set_title("sigma\nmed %0.2f" % (med_sigma))

                    ax.plot([-0.05, 1.15], [med_sigma, med_sigma], 'b')
                    ax.plot([-0.05, 1.15], [0, 0], 'k')
                    ax.plot(m_temp, sigma_means_temp, '.')

                    ax.set_xlim([-0.05, 1.15])
                    ax.set_ylim([-0.1, 2 * np.percentile(sigma_means_temp, 90)])
                    ax.set_yticks([])
                    ax.set_yticklabels([])

            if save_results:
                figh.savefig(Path(save_dir, f"modulation_estimate_z={iz:d}_time={it:d}.png"))
            if close_fig_after_saving:
                plt.close(figh)

    # ###############################
    # focus and maximum modulation vs z
    # ###############################
    if nz > 1:
        # window function for smoothing data versus z-position
        # ensure window size is positive and keeps at least 5 data points
        nwindow = np.max([np.min([nz - 5, 7]), 1])
        window = scipy.signal.hann(nwindow)
        window = window / window.sum()

        # arrays for storing results
        mod_max_zs = [[]] * ncolors
        focus_zs = [[]] * ncolors
        for ic in range(ncolors):
            mod_max_zs[ic] = np.zeros((len(rois[ic]), ntimes, nangles))
            focus_zs[ic] = np.zeros((len(rois[ic]), ntimes, nangles))

        for it in range(ntimes):
            for ic in range(ncolors):
                if ignore_color[ic]:
                    continue

                figh = plt.figure(figsize=figsize)
                plt.suptitle("focus and modulation max versus z, ic=%d = %s, it = %d" % (ic, channels[ic], it))
                grid = plt.GridSpec(2, 2 * nangles, hspace=0.5, wspace=0.5)

                for ia in range(nangles):
                    sigma_to_plot = np.array(fps_mean[ic][:, it, :, ia, 4], copy=True)
                    amp_to_plot = np.array(fps_mean[ic][:, it, :, ia, 0], copy=True)
                    m_to_plot = np.array(m_ests[ic][:, it, :, ia], copy=True)

                    # loop over ROI's and fit position of minimum sigmas and maximum modulation depth
                    for ii in range(len(rois[ic])):
                        # ############################
                        # fit modulation depth vs z
                        # smooth modulation depth vs z function and then take derivative
                        # ############################
                        # remove nans, as these will mess up interpolation
                        not_nan = np.logical_not(np.isnan(m_to_plot[ii]))

                        # m_smoothed = np.convolve(m_to_plot[ii][not_nan], window, mode="valid")
                        # zs_smoothed_m = np.convolve(zs[not_nan], window, mode="valid")
                        m_smoothed = np.convolve(m_to_plot[ii][not_nan], window, mode="same") / \
                                     np.convolve(np.ones(m_to_plot[ii][not_nan].shape), window, mode="same")
                        zs_smoothed_m = np.convolve(zs[not_nan], window, mode="same") / \
                                        np.convolve(np.ones(m_to_plot[ii][not_nan].shape), window, mode="same")

                        zd = 0.5 * (zs_smoothed_m[:-1] + zs_smoothed_m[1:])
                        dzs = zs_smoothed_m[1:] - zs_smoothed_m[:-1]

                        m_deriv = (m_smoothed[1:] - m_smoothed[:-1]) / dzs

                        max_guess = np.argmax(0.5 * (m_smoothed[1:] + m_smoothed[:-1]))

                        # interpolate zero
                        def fn(x): return np.interp(x, zd, m_deriv)
                        result = scipy.optimize.root_scalar(fn, x0=zd[max_guess] - dz, x1=zd[max_guess] + dz)

                        if result.root < zd[0] or result.root > zd[-1]:
                            mod_max_zs[ic][ii, it, ia] = np.nan
                        else:
                            mod_max_zs[ic][ii, it, ia] = result.root

                        # ############################
                        # fit sigma versus z
                        # ############################
                        # sig_smoothed = np.convolve(sigma_to_plot[ii], window, mode="valid")
                        # zs_smoothed_sig = np.convolve(zs, window, mode="valid")
                        sig_smoothed = np.convolve(sigma_to_plot[ii], window, mode="same") / \
                                       np.convolve(np.ones(sigma_to_plot[ii].shape), window, mode="same")
                        zs_smoothed_sig = np.convolve(zs, window, mode="same") / \
                                          np.convolve(np.ones(zs.shape), window, mode="same")

                        sid_deriv = (sig_smoothed[1:] - sig_smoothed[:-1]) / dz
                        zd = 0.5 * (zs_smoothed_sig[:-1] + zs_smoothed_sig[1:])

                        min_guess = np.argmax(fps_mean[ic][ii, it, :, ia, 0])
                        zmin_guess = zs[min_guess]
                        def fn(x): return np.interp(x, zd, sid_deriv)
                        result = scipy.optimize.root_scalar(fn, x0=zmin_guess - dz, x1=zmin_guess + dz)

                        if result.root < zd[0] or result.root > zd[-1]:
                            focus_zs[ic][ii, it, ia] = np.nan
                        else:
                            focus_zs[ic][ii, it, ia] = result.root

                        # ############################
                        # fit mean amp versus z
                        # ############################
                        # amp_smoothed = np.convolve(amp_to_plot[ii], window, mode="valid")
                        # zs_smoothed_amp = np.convolve(zs, window, mode="valid")
                        amp_smoothed = np.convolve(amp_to_plot[ii], window, mode="same") / \
                                       np.convolve(np.ones(amp_to_plot[ii].shape), window, mode="same")
                        zs_smoothed_amp = np.convolve(zs, window, mode="same") / \
                                          np.convolve(np.ones(zs.shape), window, mode="same")

                        amp_deriv = (amp_smoothed[1:] - amp_smoothed[:-1]) / dz
                        zd = 0.5 * (zs_smoothed_sig[:-1] + zs_smoothed_sig[1:])

                        min_guess = np.argmax(fps_mean[ic][ii, it, :, ia, 0])
                        zmin_guess = zs[min_guess]

                        def fn(x): return np.interp(x, zd, amp_deriv)

                        result = scipy.optimize.root_scalar(fn, x0=zmin_guess - dz, x1=zmin_guess + dz)

                        if result.root < zd[0] or result.root > zd[-1]:
                            focus_zs[ic][ii, it, ia] = np.nan
                        else:
                            focus_zs[ic][ii, it, ia] = result.root

                        # diagnostic plots
                        plot_z_diagnostic = True
                        if plot_z_diagnostic:
                            # annoying to deal with a second figure while on is already open...
                            figh_diagnostic = plt.figure(figsize=figsize)
                            plt.suptitle("channel = %d, angle = %d, time = %d, roi = %d" % (ic, ia, it, ii))
                            g = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)

                            ax = plt.subplot(g[0, 0])
                            ax.plot(zs, sigma_to_plot[ii], 'rx')
                            ax.plot(zs_smoothed_sig, sig_smoothed, 'r.')
                            ax.plot([focus_zs[ic][ii, it, ia], focus_zs[ic][ii, it, ia]], [sig_smoothed.min(), sig_smoothed.max()], 'r')
                            ax.legend(["raw", "smoothed"])
                            ax.set_title("sigma, peak = %0.2f $\mu m$" % focus_zs[ic][ii, it, ia])
                            ax.set_ylabel("sigma (pix)")
                            ax.set_xlabel("z ($\mu$m)")

                            ax = plt.subplot(g[0, 1])
                            ax.plot(zs, m_to_plot[ii], 'bx')
                            ax.plot(zs_smoothed_m, m_smoothed, 'b.')
                            ax.plot([mod_max_zs[ic][ii, it, ia], mod_max_zs[ic][ii, it, ia]], [0, 1], 'b')
                            ax.legend(["raw", "smoothed"])
                            ax.set_title("mod depth, peak = %0.2f $\mu m$" % mod_max_zs[ic][ii, it, ia])
                            ax.set_ylabel("mod depth")
                            ax.set_xlabel("z ($\mu$m)")

                            ax = plt.subplot(g[1, 0])
                            ax.plot(zs, fps[ic][ii, it, :, ia, :, 0], 'x')
                            ax.plot(zs_smoothed_amp, amp_smoothed, '.')
                            ax.plot([focus_zs[ic][ii, it, ia], focus_zs[ic][ii, it, ia]],
                                    [np.min(amp_to_plot[ii]), np.max(amp_to_plot[ii])])
                            ax.set_title('amplitude (all angles), peak = %0.2f $\mu m$' % focus_zs[ic][ii, it, ia])
                            ax.set_ylabel("amplitude (pix)")
                            ax.set_xlabel("z ($\mu$m)")
                            ax.set_ylim([-50, np.max(amp_to_plot[ii]) * 1.2])

                            ax = plt.subplot(g[1, 1])
                            ax.plot(zs, fps[ic][ii, it, :, ia, :, -2], 'x-')
                            ax.set_title('background (all angles)')
                            ax.set_ylabel("background (pix)")
                            ax.set_xlabel("z ($\mu$m)")

                            if save_results:

                                figh_diagnostic.savefig(Path(save_dir, f"focus_diagnostic_color={ic:d}_angle={ia:d}_roi={ii:d}.png"))
                            if close_fig_after_saving:
                                plt.close(figh_diagnostic)

                    # ############################################
                    # find focus gradient using linear least squares: f(x,y) = C0 + C1*X + C2*Y
                    # ############################################
                    cx = fps_mean[ic][:, it, iz, ia, 1]
                    cy = fps_mean[ic][:, it, iz, ia, 2]

                    zf = focus_zs[ic][:, it, ia]
                    focus_not_nan = np.logical_not(np.isnan(zf))

                    A = np.concatenate((np.ones((np.sum(focus_not_nan), 1)), cx[focus_not_nan][:, None], cy[focus_not_nan][:, None]), axis=1)
                    B = zf[focus_not_nan]
                    lsq_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                    zf_fit = lsq_params[0] + lsq_params[1] * cx + lsq_params[2] * cy

                    xy_focus_angle = np.arctan(lsq_params[2] / lsq_params[1])
                    z_focus_angle = np.arctan(np.sqrt(lsq_params[1] ** 2 + lsq_params[2] ** 2) / dxy)

                    # ############################################
                    # find mod gradient
                    # ############################################
                    mzf = mod_max_zs[ic][:, it, ia]
                    mzf_not_nan = np.logical_not(np.isnan(mzf))

                    A = np.concatenate((np.ones((np.sum(mzf_not_nan), 1)), cx[mzf_not_nan][:, None], cy[mzf_not_nan][:, None]), axis=1)
                    B = mzf[mzf_not_nan]
                    lsq_params_m, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                    mzf_fit = lsq_params_m[0] + lsq_params_m[1] * cx + lsq_params_m[2] * cy

                    xy_m_angle = np.arctan(lsq_params_m[2] / lsq_params_m[1])
                    z_m_angle = np.arctan(np.sqrt(lsq_params_m[1] ** 2 + lsq_params_m[2] ** 2) / dxy)

                    # plot results versus z
                    cmap = plt.cm.get_cmap('Reds')
                    zmin = zs.min()
                    zmax = zs.max()

                    # focus back to this figure...
                    plt.figure(figh.number)

                    # plot focus angle shift
                    ax = plt.subplot(grid[0, ia * 2])
                    c = cmap( (focus_zs[ic][:, it, ia] - zmin) / (zmax - zmin))
                    ax.scatter(cx, cy, marker='o', facecolor=c)
                    # plt.clim(np.nanmin(focus_zs[ic][:, it, ia]), zs.max())
                    # cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap))
                    ax.set_ylim(ax.get_ylim()[::-1])
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_title("focus, experiment\nangle = %d" % ia)

                    ax = plt.subplot(grid[0, ia * 2 + 1])
                    ax.set_title("focus fit\n" + r"$\theta_{xy}$=%0.1fdeg, $\theta_z$=%0.3fdeg" %
                                 (xy_focus_angle * 180 / np.pi, z_focus_angle * 180 / np.pi))
                    c_fit = cmap((zf_fit - zmin) / (zmax - zmin))
                    ax.scatter(cx, cy, marker='o', facecolor=c_fit)
                    if ia == (nangles - 1):
                        cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=zmin, vmax=zmax), cmap=cmap))
                        cb1.set_label("Position ($\mu m$)", rotation=270)

                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_ylim(ax.get_ylim()[::-1])

                    # plot m angle shift
                    ax = plt.subplot(grid[1, ia * 2])
                    c = cmap((mod_max_zs[ic][:, it, ia] - zmin) / (zmax - zmin))
                    ax.scatter(cx, cy, marker='o', facecolor=c)
                    # plt.clim(np.nanmin(focus_zs[ic][:, it, ia]), zs.max())
                    # cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap))
                    ax.set_ylim(ax.get_ylim()[::-1])
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_title("mod max, experiment")

                    ax = plt.subplot(grid[1, ia * 2 + 1])
                    ax.set_title("mod max fit\n" + r"$\theta_{xy}$=%0.1fdeg, $\theta_z$=%0.3fdeg" %
                                 (xy_m_angle * 180 / np.pi, z_m_angle * 180 / np.pi))
                    c_fit = cmap((mzf_fit - zmin) / (zmax - zmin))
                    ax.scatter(cx, cy, marker='o', facecolor=c_fit)
                    ax.set_ylim(ax.get_ylim()[::-1])
                    ax.set_yticks([])
                    ax.set_xticks([])
                    if ia == (nangles - 1):
                        cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=zmin, vmax=zmax), cmap=cmap))
                        cb1.set_label("Position ($\mu m$)", rotation=270)

                if save_results:
                    figh.savefig(Path(save_dir, f"focus_color={ic:d}_time={it:d}.png"))
                if close_fig_after_saving:
                    plt.close(figh)

                # ############################################
                # plot peak modulation and amplitude
                # ############################################

                # todo:
                # figh = plt.figure()
                # ax = plt.subplot(1, 2, 1)

                # todo: calculate modulation depth for each
                # c = 0
                # ax.scatter(cx, cy, marker='o', facecolor=c)
                #
                # cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cmap))
                # ax.set_ylim(ax.get_ylim()[::-1])
                # ax.set_yticks([])
                # ax.set_xticks([])
                # ax.set_title("modulation maximum")
