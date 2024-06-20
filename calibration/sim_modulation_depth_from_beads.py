"""
Determine SIM modulation depth from multicolor imaging of fluorescence microspheres/beads. The modulation depth can
be estimated from the amplitudes of the beads in each SIM picture using an optical sectioning SIM reconstruction
type approach.

This script handles doing this with multiple directories, where each directory contains multidimensional data which
may include color channels, time series, and z-scans. It does not handle different xy-positions.
"""
import datetime
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import zarr
import mcsim.analysis.sim_reconstruction as sim
from localize_psf import affine, localize, fit_psf

# ################################
# data files and options
# ################################
data_dirs = [Path(r"F:\2024_06_18\004_red_sim_cal")]

# channel dependent settings

# red/blue/green with 0.2um beads
# ignore_color = [False, False, False]
# min_fit_amp = [100, 30, 100]
# bead_radii = 0.5 * np.array([0.2, 0.2, 0.2])

# red/green with 0.2um beads
# ignore_color = [False, False]
# min_fit_amp = [100, 100]
# bead_radii = 0.5 * np.array([0.2, 0.2])

# blue/green with 505/515nm 0.1um beads
# ignore_color = [False, False]
# min_fit_amp = [1000, 30]
# bead_radii = 0.5 * np.array([0.1, 0.1])

# blue/green
# ignore_color = [False, False]
# min_fit_amp = [200, 200]
# bead_radii = 0.5 * np.array([0.1, 0.1])

# red beads
ignore_color = [False]
min_fit_amp = [1000]
bead_radii = 0.5 * np.array([0.2])

# blue
# ignore_color = [False]
# min_fit_amp = [300]
# bead_radii = 0.5 * np.array([0.1])

# channel independent setings
use_gpu = True
figsize = (30, 9)
save_results = True
# nrois_to_plot = 5 # 5
nrois_to_plot = 0

# localization filtering parameters
correct_mod_for_bead_size = True
roi_size = (0, 1.5, 1.5)  # (sz, sy, sx) in um
filters_small_sigma = (0.1, 0.1, 0.1)  # (sz, sy, sx) in um
filters_large_sigma = (5, 5, 5)  # (sz, sy, sx) in um
min_boundary_distance = (0, 1)  # (dz, dxy) in um
sigma_bounds = ((0, 0.05), (3, 0.2))
min_spot_sep = (0, 1) # (dz, dxy) in um

model = fit_psf.gaussian3d_psf_model()
nmodel_param = model.nparams

# ################################
# start processing
# ################################
tstamp = datetime.datetime.now().strftime("%Y_%m_%d_%H;%M;%S")

# turn off interactive plotting
plt.ioff()
plt.switch_backend("agg")

# helper function
def get_key(code, fit_states_key):
    msg = "message not found"
    for k, v in fit_states_key.items():
        if code == v:
            msg = k
            break
    return msg

# ################################
# loop over data dirs and estimate modulation depth
# ################################
for d in data_dirs:
    file_path = list(d.glob("*.zarr"))[0]
    data = zarr.open(file_path, "r")
    channels = data.cam1.attrs["channels"]

    # arrs = [a for name, a in data.cam1.arrays() if name[:3] == "sim"]
    arrs = [data.cam1[f"sim_{ch:s}"] for ch in channels]
    ncolors = len(arrs)

    # get axes sizes and names
    axes_names = arrs[0].attrs["dimensions"]
    nt_fast, npos, ntimes, nz, nparams, npatterns, ny, nx = arrs[0].shape
    nphases = arrs[0].attrs["nphases"]
    nangles = arrs[0].attrs["nangles"]

    dxy = data.cam1.attrs["dx_um"]
    dz = data.attrs["dz_um"]
    zs = np.array(data.attrs["z_position_um"])
    exposure_t = data.cam1.attrs["exposure_time_ms"]

    # load affine xforms
    affine_xforms = [np.array(xform) for xform in data.cam1.attrs["affine_transformations"]]

    # load pattern data
    frqs = np.zeros((ncolors, nangles, 2))
    phases = np.zeros((ncolors, nangles, nphases))
    for ii in range(ncolors):
        frqs_raw = np.array(arrs[ii].attrs["frqs"])[::nphases]
        phases_raw = np.array(arrs[ii].attrs["phases"]).reshape((nangles, nphases))
        fxt, fyt, phit = affine.xform_sinusoid_params(frqs_raw[:, 0],
                                                      frqs_raw[:, 1],
                                                      phases_raw,
                                                      affine_xforms[ii])
        frqs[ii, ..., 0] = 2 * fxt / dxy
        frqs[ii, ..., 1] = 2 * fyt / dxy
        phases[ii] = 2 * phit

    periods = 1 / np.linalg.norm(frqs, axis=-1)

    # load images
    nextra_dims = arrs[0].ndim - 3
    imgs = [np.array(a).reshape(a.shape[:nextra_dims] + (nangles, nphases, ny, nx)) for a in arrs]

    shape_extra = imgs[0].shape[:nextra_dims]
    ntot_extra = np.prod(shape_extra)

    # make save directory
    if save_results:
        save_dir = Path(d, f"{tstamp:s}_sim_modulation_depth")
        save_dir.mkdir(exist_ok=True)

        roi_save_dir = save_dir / "roi_figures"
        roi_save_dir.mkdir(exist_ok=True)


    # #################################################
    # prepare variables to hold results
    # #################################################
    # each entry in these lists will have size nfits x n_other_axes ... x nangles x nphases
    rois = [[]] * ncolors
    fps = [[]] * ncolors
    # derived parameters
    fps_mean = [[]] * ncolors
    m_ests = [[]] * ncolors
    cos_phis = [[]] * ncolors
    mod_depth_bead_size_correction = np.zeros((ncolors, nangles))

    # fit parameters for central z-index and first time, each entry size nfits x n_other_axes ... x nangles
    fps_start = [[]] * ncolors
    ips_start = [[]] * ncolors

    tstart = time.perf_counter()
    for ic in range(ncolors):
        # #################################################
        # load first image and identify spots, which will be used in all subsequent analysis
        # #################################################
        if ignore_color[ic]:
            continue
        print(f"started initial fitting for channel {ic:d}")

        # take widefield image
        img_middle_wf = np.max(imgs[ic].mean(axis=(-3, -4)), axis=tuple(range(nextra_dims)))

        zz, yy, xx = localize.get_coords((1,) + img_middle_wf.shape, (1, dxy, dxy), broadcast=True)
        coords = localize.get_coords((1,) + img_middle_wf.shape, (1, dxy, dxy), broadcast=False)
        filter = localize.get_param_filter(coords,
                                           fit_dist_max_err=(np.inf, np.inf),
                                           min_spot_sep=min_spot_sep,
                                           sigma_bounds=sigma_bounds,
                                           amp_bounds=(min_fit_amp[ic], np.inf),
                                           dist_boundary_min=min_boundary_distance)

        localize.prepare_rois(img_middle_wf[None, :, :], coords, np.array([[0, 1, 0, 1, 0, 1]]))

        # filter = localize.no_filter()

        # mask DMD area
        # check which points are within DMD area
        # todo: need to update for new camera orientation
        # cpix = np.stack((fps_temp[:, 1] / dxy, fps_temp[:, 2] / dxy), axis=1)
        # cpix_dmd = affine.xform_points(cpix, np.linalg.inv(affine_xforms[ic]))
        # nx_dmd = data.dmd_data.attrs["dmd_nx"]
        # ny_dmd = data.dmd_data.attrs["dmd_ny"]
        #
        # in_dmd_area = np.logical_and.reduce((cpix_dmd[:, 0] >= 0, cpix_dmd[:, 0] <= nx_dmd,
        #                                      cpix_dmd[:, 1] >= 0, cpix_dmd[:, 1] <= ny_dmd))
        in_dmd_area = np.ones(img_middle_wf.shape, dtype=bool)

        # identify beads in first image
        _, fit_results, imgs_filtered = \
            localize.localize_beads_generic(img_middle_wf,
                                            (dz, dxy, dxy),
                                            min_fit_amp[ic],
                                            roi_size=roi_size,
                                            filter_sigma_small=filters_small_sigma,
                                            filter_sigma_large=filters_large_sigma,
                                            min_spot_sep=min_spot_sep,
                                            filter=filter,
                                            model=model,
                                            use_gpu_fit=use_gpu,
                                            mask=in_dmd_area)

        fps_temp = fit_results["fit_params"]
        ips_temp = fit_results["init_params"]
        rois_temp = fit_results["rois"]
        to_keep = fit_results["to_keep"]

        if not np.any(to_keep):
            raise ValueError("no beads found...")


        # ROIs we will use going forward
        rois[ic] = rois_temp[to_keep]
        fps_start[ic] = fps_temp[to_keep]
        ips_start[ic] = ips_temp[to_keep]

        # define array to hold later fit results
        fps[ic] = np.zeros(shape_extra + (nangles, nphases, len(rois[ic]), nmodel_param))

        # ##############
        # plot detected bead positions along with amplitudes and sigmas
        figh1 = plt.figure(figsize=figsize)
        figh1.suptitle(f"detected spots\n"
                       f"channel {ic:d} = {channels[ic]:s}, threshold={min_fit_amp[ic]:.0f}")

        width_ratios1 = [1, 0.05, 0.05, 0.05] * 2
        wspace1 = 0.2 / np.mean(width_ratios1)
        grid1 = figh1.add_gridspec(nrows=1, ncols=len(width_ratios1), width_ratios=width_ratios1, wspace=wspace1)
        axes1 = []
        for ii in range(len(width_ratios1)):
            axes1.append(figh1.add_subplot(grid1[0, ii]))

        localize.plot_bead_locations(img_middle_wf,
                                             center_lists=[fps_temp[:, (3, 2, 1)], fps_start[ic][:, (3, 2, 1)]],
                                             coords=(yy[0], xx[0]),
                                             legend_labels=["all fits", "reliable fits"],
                                             weights=[fps_temp[:, 4], fps_start[ic][:, 4]],
                                             cbar_labels=[r"$\sigma_{xy} (\mu m)$"] * 2,
                                             color_limits=[[np.percentile(fps_temp[:, 4], 5), sigma_bounds[1][1]]] * 2,
                                             color_lists=["bone", "hsv"],
                                             vlims_percentile=(1, 99.99),
                                             gamma=0.5,
                                             title=f"sigmas",
                                             axes=axes1[0:4],
                                             figsize=figsize)

        localize.plot_bead_locations(img_middle_wf,
                                             center_lists=[fps_temp[:, (3, 2, 1)], fps_start[ic][:, (3, 2, 1)]],
                                             legend_labels=["all fits", "reliable fits"],
                                             weights=[fps_temp[:, 0], fps_start[ic][:, 0]],
                                             coords=(yy[0], xx[0]),
                                             cbar_labels=["amplitude (ADU)"] * 2,
                                             color_limits=[[0, np.percentile(fps_start[ic][:, 0], 95)]] * 2,
                                             color_lists=["bone", "hsv"],
                                             vlims_percentile=(1, 99.99),
                                             gamma=0.5,
                                             title=f"amplitudes",
                                             axes=axes1[4:],
                                             figsize=figsize)

        if save_results:
            figh1.savefig(save_dir / f"beads_initial_channel={ic:d}.png")
            plt.close(figh1)


        # plot fits if desired
        for aaa in range(np.min([nrois_to_plot, len(fps_start[ic])])):
            localize.plot_fit_roi(fps_start[ic][aaa],
                                  rois[ic][aaa],
                                  np.expand_dims(img_middle_wf, axis=0),
                                  coords,
                                  init_params=ips_start[ic][aaa],
                                  prefix=f"reference_roi={aaa:d}_ic={ic:d}_iz={nz//2:d}_it={0:d}_",
                                  save_dir=roi_save_dir)

        # #################################################
        # do fitting for all other images
        # #################################################
        for ccc in range(ntot_extra):
            for ia in range(nangles):
                for ip in range(nphases):
                    # grab current indices
                    inds_extra = np.unravel_index(ccc, shape_extra)
                    inds_now = inds_extra + (ia, ip)
                    imgs_now = np.expand_dims(imgs[ic][inds_now], axis=0)

                    # id
                    extra_names = "_".join([f"{n:s}={v:d}" for n, v in zip(axes_names[:nextra_dims], inds_now)])
                    id = f"channel={ic:d}_{extra_names:s}_angle={ia:d}_phase={ip:d}"

                    print(f"fitting {id.replace('_', ', '):s},"
                          f" elapsed time={time.perf_counter() - tstart:.2f}s", end="\r")

                    # fix parameters to match initial fits
                    fixed_params = np.zeros(model.nparams, dtype=bool)
                    fixed_params[1:] = True

                    # do fitting
                    img_rois, coords_rois, roi_sizes = localize.prepare_rois(imgs_now, coords, rois[ic])
                    fit_results = localize.fit_rois(img_rois,
                                                    coords_rois,
                                                    roi_sizes,
                                                    fps_start[ic],
                                                    fixed_params=fixed_params,
                                                    use_gpu=use_gpu,
                                                    model=model)

                    # unpack results
                    fps[ic][inds_now] = fit_results["fit_params"]
                    chi_sqrs = fit_results["chi_sqrs"]
                    niters = fit_results["niters"]
                    fit_states = fit_results["fit_states"]
                    fit_states_key = fit_results["fit_states_key"]

                    # plot ROI's as diagnostics
                    for aaa in range(np.min([nrois_to_plot, len(fps_start[ic])])):
                        str = f"ROI={aaa:d}, {id.replace('_', '', ''):s}\n" \
                              f"fit iters={niters[aaa]:d} " \
                              f"with result '{get_key(fit_states[aaa], fit_states_key):s}', and " \
                              f"chi squared = {chi_sqrs[aaa]:.1g}"

                        localize.plot_fit_roi(fps[ic][inds_now][aaa],
                                              rois[ic][aaa],
                                              imgs_now,
                                              coords,
                                              init_params=fps_start[ic][aaa],
                                              string=str,
                                              prefix=f"roi={aaa:d}_{id:s}_",
                                              save_dir=roi_save_dir)

            print("")

        # ####################################
        # compute modulation depth using the phase axis
        # ####################################
        # estimate modulation depth and amplitude
        mean_amps = np.mean(fps[ic][..., 0], axis=-2)
        m_ests[ic] = sim.sim_optical_section(fps[ic][..., 0], axis=-2) / mean_amps

        # if any amplitudes were negative, m_est will not be any good, so throw it away
        amp_is_neg = np.any(fps[ic][..., 0] < 0, axis=-2)
        m_ests[ic][amp_is_neg] = np.nan

        for ia in range(nangles):
            if correct_mod_for_bead_size:
                mod_depth_bead_size_correction[ic, ia] = sim.correct_modulation_for_bead_size(bead_radii[ic], 1 / periods[ic][ia])
            else:
                mod_depth_bead_size_correction[ic, ia] = 1.

        # ####################################
        # plot results for each data set
        # ####################################
        tstart = time.perf_counter()
        for ccc in range(ntot_extra):
            inds_now = np.unravel_index(ccc, shape_extra)
            imgs_now = np.expand_dims(imgs[ic][inds_now], axis=0)

            # id
            extra_names = "_".join([f"{n:s}={v:d}" for n, v in zip(axes_names[:nextra_dims], inds_now)])
            id = f"channel={ic:d}_{extra_names:s}"

            print(f"plotting {id.replace('_', ', '):s},"
                  f" elapsed time={time.perf_counter() - tstart:.2f}s")

            # ###########################
            # plot modulation depth for  all angles
            # ###########################
            figh = plt.figure(figsize=(20, 9))
            figh.suptitle(f"channel={channels[ic]:s}\n"
                          f"{id.replace('_', ', '):s}\n"
                          f"exposure={exposure_t:.1f}ms")

            nplots_per_angle = 3
            grid = figh.add_gridspec(nrows=nangles,
                                     ncols=nplots_per_angle,
                                     wspace=0.3,
                                     hspace=0.5)

            # spatial map
            figh2 = plt.figure(figsize=figsize)
            figh2.suptitle(f"modulation depth, channel={channels[ic]:s}\n"
                           f"{id.replace('_', ', '):s}\n"
                           f"exposure={exposure_t:.1f}ms")

            width_ratios = [1, 0.05, 0.05] * nangles
            wspace = 0.2 / np.mean(width_ratios)
            grid2 = figh2.add_gridspec(nrows=1,
                                       ncols=3 * nangles,
                                       width_ratios=[1, 0.05, 0.05] * nangles,
                                       wspace=wspace
                                       )

            axes2 = []
            for ii in range(nangles * 3):
                axes2.append(figh2.add_subplot(grid2[0, ii]))


            for ia in range(nangles):
                id_angle = f"channel={ic:d}_{extra_names:s}_angle={ia:d}"
                inds_angle = inds_now + (ia, )

                # ##########################
                # calculate modulation depth statistics
                # ##########################
                m_temp = m_ests[ic][inds_angle] / mod_depth_bead_size_correction[ic, ia]
                m_allowed = m_temp[np.logical_and(m_temp > 0, m_temp < 1)]
                mean_depth = np.mean(m_allowed)
                med_depth = np.median(m_allowed)
                std_depth = np.std(m_allowed)

                bin_edges = np.linspace(0, 1.25, 50)
                ms_hist, bin_edges = np.histogram(m_temp, bins=bin_edges)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                # amplitude statistics
                amp_temp = np.mean(fps[ic][inds_angle][:, :, 0], axis=-2)
                med_amp = np.median(amp_temp)

                # sigma statistics
                sigma_means_temp = np.mean(fps[ic][inds_angle][:, :, 4], axis=-2)
                med_sigma = np.median(sigma_means_temp)

                imgs_now_wf = np.mean(imgs[ic][inds_angle], axis=0)
                centers_temp = fps[ic][inds_angle + (0, )][:, (3, 2, 1)]

                # ##########################
                # plot on main plot
                # ##########################

                # histogram of modulation depths
                ax = figh.add_subplot(grid[ia, 0])
                ax.set_title(f"mean={mean_depth:.3f}({std_depth * 1e3:.0f})\n"
                             f"peak={bin_centers[np.argmax(ms_hist)]:.3f}")

                ax.plot(bin_centers, ms_hist)
                ax.axvline(med_depth, c='b')
                ax.set_xlim([-0.05, 1.15])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel(f"angle={ia:d}")

                # modulation depths versus amplitude
                ax = figh.add_subplot(grid[ia, 1])
                ax.set_title(f"amp\n med {med_amp:.1f}")
                ax.axhline(med_amp, c='b')
                ax.axhline(0, c='k')
                ax.axvline(med_depth, c='b')
                ax.plot(m_temp, amp_temp, '.')
                ax.set_xlim([-0.05, 1.15])
                ax.set_ylim([-10, 2 * np.percentile(amp_temp, 90)])

                # size versus amplitude
                ax = figh.add_subplot(grid[ia, 2])
                ax.set_title(f"sigma\nmed {med_sigma:.02f}")
                ax.axhline(med_sigma, c='b')
                ax.axhline(0, c='k')
                ax.axvline(med_depth, c='b')
                ax.plot(m_temp, sigma_means_temp, '.')
                ax.set_xlim([-0.05, 1.15])
                ax.set_ylim([-0.1, 2 * np.percentile(sigma_means_temp, 90)])

                # ##########################
                # plot modulation depths per angle
                # ##########################
                localize.plot_bead_locations(imgs_now_wf,
                                             centers_temp,
                                             weights=m_ests[ic][inds_angle],
                                             coords=(yy[0], xx[0]),
                                             title=f"angle={ia:d}\n"
                                                   f"m={mean_depth:.3f}({std_depth * 1e3:.0f})",
                                             cbar_labels=["modulation depth"],
                                             color_limits=[[0, 1]],
                                             color_lists=["hsv"],
                                             vlims_percentile=(1, 99.99),
                                             gamma=0.5,
                                             axes=axes2[ia * nangles: (ia + 1) * nangles],
                                             figsize=figsize)

            if save_results:
                figh.savefig(Path(save_dir, f"modulation_statistics_{id:s}.png"))
                plt.close(figh)

                figh2.savefig(save_dir / f"modulation_vs_position_{id:s}.png")
                plt.close(figh2)