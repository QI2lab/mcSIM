"""
Functions for calibrating noise and gain map of sCMOS camera, doing denoising of imaging data, or producing
simulated camera data
"""
import time
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

# camera calibration
def get_pixel_statistics(imgs, description="", moments=(1, 2), corr_dists=None):
    """
    Estimate camera readout noise from a set of images taken in zero light conditions

    :param imgs: numpy or dask image array. If you have a set of individual tiff files, such an array can be produced
    with dask_image, e.g.
    >>> imgs = dask_image.imread.imread("*.tif")
    :param moments list of moments to calculate: by default only the first and second moment. But if also compute the
     3rd and 4th can estimate the uncertainty in the variance
    :param corr_dists: list of correlator distances (along x, y, and t) to compute correlations

    :return data: dictionary storing data.
    Keys have the form "moment_n" for the nth moment and
    "corr_r_dist=n" where r = x, y, or t and n is the correlation step size
    """

    if corr_dists is not None and np.any([not isinstance(cd, int) for cd in corr_dists]):
        raise ValueError(f"All corr_dists values must be integers, but were {corr_dists}")


    tstart = time.process_time()

    nimgs, ny, nx = imgs.shape

    # compute moments
    results_arrays = [(imgs.astype(float)**m).mean(axis=0) for m in moments]
    results_names = [f"moment_{m:d}" for m in moments]

    # compute correlators
    if corr_dists is None:
        ncorrs = 0
    else:
        ncorrs = len(corr_dists)

        if isinstance(imgs, np.ndarray):
            roll_fn = np.roll
        else:
            roll_fn = da.roll

        for cd in corr_dists:
            # todo: should I compute the connected correlators instead? i.e. <I(x) I(x+dx)> - <I(x)><I(x + dx)>
            # for x- and y- correlations wrap arounds don't matter in the sense they don't obscure any information
            results_arrays.append((imgs.astype(float) * roll_fn(imgs, cd, axis=2).astype(float)).mean(axis=0))
            results_names.append(f"corr_x_dist={cd:d}")

            results_arrays.append((imgs.astype(float) * roll_fn(imgs, cd, axis=1).astype(float)).mean(axis=0))
            results_names.append(f"corr_y_dist={cd:d}")

            # cut-off first cd-images which are correlations with "wrapped around" images
            # todo: technically don't have the right averages for computing the connected correlator bc would want
            # means of all images except the first/last
            results_arrays.append((imgs.astype(float) * roll_fn(imgs, cd, axis=0).astype(float))[cd:].mean(axis=0))
            results_names.append(f"corr_t_dist={cd:d}")

    # compute with dask if img is a dask array
    if not isinstance(imgs, np.ndarray):
        print(f"computing {len(moments):d} moments and {3*ncorrs:d} correlators")
        with ProgressBar():
            results_arrays = dask.compute(*results_arrays)

    # store results in dictionary object
    data = {"nimgs": nimgs,
            "description": description,
            "processing_time_s": time.perf_counter() - tstart}

    for mn, ma in zip(results_names, results_arrays):
        data.update({mn: ma})

    # calculate "uncertainty" in variance, if computed moments 1, 2, 3, and 4
    # variance of sample variance. See e.g. https://mathworld.wolfram.com/SampleVarianceDistribution.html
    if np.all([f"moment_{m:d}" in results_names for m in (1, 2, 3, 4)]):
        moment2_central = data["moment_2"] - data["moment_1"]**2
        moment4_central = data["moment_4"] \
                          - 4 * data["moment_1"] * data["moment_3"] + \
                          6 * data["moment_1"]**2 * data["moment_2"] \
                          - 3 * data["moment_1"]**4
        var_sample_var = (nimgs - 1)**2 / nimgs**3 * moment4_central - \
                         (nimgs - 1) * (nimgs - 3) * moment2_central**2 / nimgs**3

        data.update({"variance of sample variance": var_sample_var})

    return data


def get_gain_map(dark_means, dark_vars, light_means, light_vars, max_mean=np.inf):
    """

    @param dark_means: array of size ny x nx
    @param dark_vars: array of size ny x nx
    @param light_means: list of arrays of size ni x ny x nx
    @param light_vars: list of arrays of size ni x ny x nx
    @param max_mean: ignore data points with means larger than this value (to avoid issues with saturation)
    @return gains, dark_means, dark_vars:
    """

    # minimize over g_ij \sum_k ( (nu_ij - v_ij) - g_ij * (Dk_ij - o_ij) )^2
    # recast this as min || A^t - B^t * g||^2
    # g = pinv(B^t) * A^t
    ny, nx = dark_means.shape
    gains = np.zeros(dark_means.shape)
    for ii in range(ny):
        for jj in range(nx):
            to_use = light_means[:, ii, jj] <= max_mean

            ms_pix = (light_means[:, ii, jj] - dark_means[ii, jj])[to_use]
            vs_pix = (light_vars[:, ii, jj] - dark_vars[ii, jj])[to_use]

            gains[ii, jj] = np.linalg.pinv(np.expand_dims(ms_pix, axis=1)).dot(np.expand_dims(vs_pix, axis=1))

    return gains, dark_means, dark_vars


def plot_noise_stats(noise_data, nbins=600, figsize=(20, 10)):
    """
       Display gain curve for single pixel vs. illumination data
       :param dict dark_data: {"means", "variances"} fields are ny x nx arrays
       :param dict light_data: {"means", "means_unc", "variances", "variances_uncertainty"}
       fields  store nillum x ny x nx arrays
       :param gains: ny x nx array of gains
       :param nbins: number of bins for histograms
       :param figsize:
       :return:
       """

    # todo: deprecate
    offsets = noise_data['means']
    vars = noise_data['variances']

    plot_correlators = False
    if "corr_x" in noise_data.keys():
        plot_correlators = True
        cx = noise_data['corr_x'] - offsets * np.roll(offsets, 1, axis=1)
        cy = noise_data['corr_y'] - offsets * np.roll(offsets, 1, axis=0)
        ct = noise_data['corr_t'] - offsets * offsets


    # ############################################
    # parameter histograms
    # ############################################

    offs_start = np.percentile(offsets, 0.1) - 1
    offs_end = np.percentile(offsets, 99.5)
    bin_edges_offs = np.linspace(offs_start, offs_end, nbins + 1)
    hmeans, _ = np.histogram(offsets.ravel(), bin_edges_offs)
    bin_centers_offs = 0.5 * (bin_edges_offs[:-1] + bin_edges_offs[1:])

    vars_start = 0  # np.percentile(vars, 0.1) - 1
    vars_end = np.percentile(vars, 99.5)
    bin_edges_vars = np.linspace(vars_start, vars_end, nbins + 1)
    hvars, _ = np.histogram(vars.ravel(), bin_edges_vars)
    bin_centers_vars = 0.5 * (bin_edges_vars[:-1] + bin_edges_vars[1:])

    if plot_correlators:
        vcx_start = np.percentile(cx, 0.5)
        vcx_end = np.percentile(cx, 99.5)
        bin_edges_cx = np.linspace(vcx_start, vcx_end, nbins + 1)
        hcx, _ = np.histogram(cx.ravel(), bin_edges_cx)
        bin_centers_cx = 0.5 * (bin_edges_cx[:-1] + bin_edges_cx[1:])

        vcy_start = np.percentile(cy, 0.5)
        vcy_end = np.percentile(cy, 99.5)
        bin_edges_cy = np.linspace(vcy_start, vcy_end, nbins + 1)
        hcy, _ = np.histogram(cy.ravel(), bin_edges_cy)
        bin_centers_cy = 0.5 * (bin_edges_cy[:-1] + bin_edges_cy[1:])

        vct_start = np.percentile(ct, 0.5)
        vct_end = np.percentile(ct, 99.5)
        bin_edges_ct = np.linspace(vct_start, vct_end, nbins + 1)
        hct, _ = np.histogram(ct.ravel(), bin_edges_ct)
        bin_centers_ct = 0.5 * (bin_edges_ct[:-1] + bin_edges_ct[1:])

    fn1 = "camera_params_histograms"
    figh1 = plt.figure(figsize=figsize)
    grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.3)
    plt.suptitle("Camera parameter histograms")

    ax2 = plt.subplot(grid[0, 0])
    ax2.plot(bin_centers_offs, hmeans)
    ax2.set_xlabel('offsets (ADU)')
    ax2.set_title('means')

    ax3 = plt.subplot(grid[0, 1])
    ax3.plot(bin_centers_vars, hvars)
    ax3.set_xlabel('variances (ADU^2)')
    ax3.set_title('variances')

    if plot_correlators:
        ax4 = plt.subplot(grid[1, 0])
        ax4.plot(bin_centers_cx, hcx)
        ax4.set_xlabel("correlation")
        ax4.set_title("<I(x, y)*I(x - 1, y)>_c")

        ax5 = plt.subplot(grid[1, 1])
        ax5.plot(bin_centers_cy, hcy)
        ax5.set_xlabel("y-correlation")
        ax5.set_title("<I(x, y)*I(x, y - 1)>_c")

        ax6 = plt.subplot(grid[1, 2])
        ax6.plot(bin_centers_ct, hct)
        ax6.set_xlabel("t-correlation")
        ax6.set_title("<I(x, y, t)*I(x, y, t - 1)>_c")

    # ############################################
    # param maps
    # ############################################
    fn2 = "camera_params_maps"
    figh2 = plt.figure(figsize=figsize)
    grid2 = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.3)
    plt.suptitle("Camera parameter maps")

    ax1 = plt.subplot(grid2[0, 0])
    vmin = np.percentile(offsets, 2)
    vmax = np.percentile(offsets, 98)
    im1 = ax1.imshow(offsets, vmin=vmin, vmax=vmax)
    ax1.set_title("offsets")
    figh2.colorbar(im1)

    ax2 = plt.subplot(grid2[0, 1])
    vmin = np.percentile(vars, 2)
    vmax = np.percentile(vars, 98)
    im2 = ax2.imshow(vars, vmin=vmin, vmax=vmax)
    ax2.set_title("variances")
    figh2.colorbar(im2)

    if plot_correlators:
        ax3 = plt.subplot(grid2[1, 0])
        vmin_temp = np.percentile(cx, 1)
        vmax_temp = np.percentile(cx, 99.5)
        vmin = np.min([vmin_temp, -vmax_temp])
        vmax = np.max([-vmin_temp, vmax_temp])

        im3 = ax3.imshow(cx, vmin=vmin, vmax=vmax)
        ax3.set_title("x-correlator")
        figh2.colorbar(im3)

        ax4 = plt.subplot(grid2[1, 1])
        vmin_temp = np.percentile(cy, 1)
        vmax_temp = np.percentile(cy, 99.5)
        vmin = np.min([vmin_temp, -vmax_temp])
        vmax = np.max([-vmin_temp, vmax_temp])

        im4 = ax4.imshow(cy, vmin=vmin, vmax=vmax)
        ax4.set_title("y-correlator")
        figh2.colorbar(im4)

        ax5 = plt.subplot(grid2[1, 2])
        vmin_tem = np.percentile(ct, 1)
        vmax_temp = np.percentile(ct, 99.5)
        vmin = np.min([vmin_temp, -vmax_temp])
        vmax = np.max([-vmin_temp, vmax_temp])

        im5 = ax5.imshow(ct, vmin=vmin, vmax=vmax)
        ax5.set_title("t-correlator")
        figh2.colorbar(im5)

    fighs = [figh1, figh2]
    fig_names = [fn1, fn2]

    return fighs, fig_names


def plot_camera_noise_results(offsets, dark_vars, light_means, light_vars, gains,
                              light_means_err=None, light_vars_err=None, nbins=600, figsize=(20, 10)):
    """
    Display gain curve for single pixel vs. illumination data
    :param offsets: ny x nx
    :param dark_vars: ny x nx
    :param light_means: ni x ny x nx
    :param light_vars: ni x ny x nx
    :param gains: ny x nx array of gains
    :param nbins: number of bins for histograms
    :param figsize:
    :return:
    """

    if light_means_err is None:
        light_means_err = np.zeros(light_means.shape) * np.nan

    if light_vars_err is None:
        light_vars_err = np.zeros(light_vars.shape) * np.nan

    # for comparison with read noise stats in electrons
    std_es = np.sqrt(dark_vars) / gains

    # ############################################
    # parameter histograms
    # ############################################

    gain_start = np.percentile(gains, 0.1) * 0.8
    gain_end = np.percentile(gains, 99.1) * 1.2
    bin_edges = np.linspace(gain_start, gain_end, nbins + 1)
    hgains, _ = np.histogram(gains.ravel(), bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    offs_start = np.percentile(offsets, 0.1) - 1
    offs_end = np.percentile(offsets, 99.5)
    bin_edges_offs = np.linspace(offs_start, offs_end, nbins + 1)
    hmeans, _ = np.histogram(offsets.ravel(), bin_edges_offs)
    bin_centers_offs = 0.5 * (bin_edges_offs[:-1] + bin_edges_offs[1:])

    vars_start = 0 #np.percentile(vars, 0.1) - 1
    vars_end = np.percentile(dark_vars, 99.5)
    bin_edges_vars = np.linspace(vars_start, vars_end, nbins + 1)
    hvars, _ = np.histogram(dark_vars.ravel(), bin_edges_vars)
    bin_centers_vars = 0.5 * (bin_edges_vars[:-1] + bin_edges_vars[1:])

    # for comparison with read noise stats in electrons
    sds_es_start = 0 #np.percentile(std_es, 0.1)
    sds_es_end = np.percentile(std_es, 99.5)
    bin_edges_stds_es = np.linspace(sds_es_start, sds_es_end, nbins + 1)
    h_std_es, _ = np.histogram(std_es.ravel(), bin_edges_stds_es)
    bin_centers_stds_es = 0.5 * (bin_edges_stds_es[:-1] + bin_edges_stds_es[1:])

    med_std_es = np.median(std_es)
    rms_es = np.sqrt(np.mean(std_es**2))

    fn1 = "camera_params_histograms"
    figh1 = plt.figure(figsize=figsize)
    grid = figh1.add_gridspec(nrows=2, ncols=2, wspace=0.2, hspace=0.3)
    figh1.suptitle("Camera parameter histograms")

    ax1 = figh1.add_subplot(grid[0, 0])
    ax1.plot(bin_centers, hgains)
    ax1.set_xlabel('gains (ADU/e)')
    ax1.set_ylabel('counts')
    ax1.set_title('histogram of pixel gains, median=%0.2f ADU/e' % (np.median(gains.ravel())))

    ax2 = figh1.add_subplot(grid[0, 1])
    ax2.plot(bin_centers_offs, hmeans)
    ax2.set_xlabel('offsets (ADU)')
    ax2.set_title('dark mean, median=%0.2f ADU' % (np.median(offsets.ravel())))

    ax3 = figh1.add_subplot(grid[1, 0])
    ax3.plot(bin_centers_vars, hvars)
    ax3.set_xlabel('variances (ADU^2)')
    ax3.set_title('dark variances, median=%0.2f ADU^2' % (np.median(dark_vars.ravel())))

    ax4 = figh1.add_subplot(grid[1, 1])
    ph1, = ax4.plot(bin_centers_stds_es, h_std_es)
    ylims = ax4.get_ylim()
    ph2, = ax4.plot([med_std_es, med_std_es], ylims)
    ph3, = ax4.plot([rms_es, rms_es], ylims)
    ax4.set_ylim(ylims)
    ax4.legend([ph2, ph3], ["median = %0.2f" % med_std_es, "rms = %0.2f" % rms_es])

    ax4.set_xlabel('standard deviation (electrons)')
    ax4.set_title('standard dev (electrons)')

    # ############################################
    # param maps
    # ############################################
    fn2 = "camera_params_maps"
    figh2 = plt.figure(figsize=figsize)
    grid2 = figh2.add_gridspec(nrows=2, ncols=2, wspace=0.2, hspace=0.3)
    figh2.suptitle("Camera parameter maps")

    ax1 = figh2.add_subplot(grid2[0, 0])
    vmin = np.percentile(offsets, 2)
    vmax = np.percentile(offsets, 98)
    im1 = ax1.imshow(offsets, vmin=vmin, vmax=vmax)
    ax1.set_title("offsets (ADU)")
    figh2.colorbar(im1)

    ax2 = figh2.add_subplot(grid2[0, 1])
    vmin = np.percentile(dark_vars, 2)
    vmax = np.percentile(dark_vars, 98)
    im2 = ax2.imshow(dark_vars, vmin=vmin, vmax=vmax)
    ax2.set_title("variances (ADU^2)")
    figh2.colorbar(im2)

    ax3 = figh2.add_subplot(grid2[1, 0])
    vmin = np.percentile(gains, 2)
    vmax = np.percentile(gains, 98)
    im3 = ax3.imshow(gains, vmin=vmin, vmax=vmax)
    ax3.set_title("gains (ADU/e)")
    figh2.colorbar(im3)

    ax4 = figh2.add_subplot(grid2[1, 1])
    im4 = ax4.imshow(std_es, vmin=0, vmax=np.percentile(std_es, 98))
    ax4.set_title("read noise SD (e)")
    figh2.colorbar(im4)

    # ############################################
    # example gain fits
    # ############################################
    nrows = 4
    ncols = 4

    # choose some random pixels to plot
    ninds = nrows * ncols
    xinds = np.random.randint(0, gains.shape[1], size=ninds)
    yinds = np.random.randint(0, gains.shape[0], size=ninds)

    # choose pixels with largest/smallest gains to plot
    sorted_inds = np.argsort(gains.ravel())
    yinds_small, xinds_small = np.unravel_index(sorted_inds[: ninds], gains.shape)
    yinds_large, xinds_large = np.unravel_index(sorted_inds[-ninds:], gains.shape)

    # interpolated mean values
    minterp = np.linspace(0, light_means.max(), 100)
    var_vmin = 0
    var_vmax = np.percentile(light_vars, 99.9)
    mean_vmin = 0
    mean_vmax = minterp[-1] * 1.2

    # plot gain fits
    figs_pix_eg = [[]]*3
    figs_pix_eg_names = ["camera_gain_fitting_examples", "camera_gain_fitting_smallest_gains", "camera_gain_fitting_largest_gains"]
    for ll, (yis, xis, fn) in enumerate(zip([yinds, yinds_small, yinds_large], [xinds, xinds_small, xinds_large], figs_pix_eg_names)):
        figh = plt.figure(figsize=figsize)
        grid3 = figh.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.5, wspace=0.3)
        figh.suptitle(fn.replace("_", " "))

        for ii in range(ninds):
            aa, bb = np.unravel_index(ii, (nrows, ncols))

            # create subplot
            ax = figh.add_subplot(grid3[aa, bb])
            # grab pixel coordinates
            y = yis[ii]
            x = xis[ii]

            # plot dark frames
            ax.plot(offsets[y, x], dark_vars[y, x], 'b.')
            # plot light frames
            ax.errorbar(light_means[:, y, x], light_vars[:, y, x], fmt='.', xerr=light_means_err[:, y, x], yerr=light_vars_err[:, y, x])
            # plot fit
            ax.plot(minterp + offsets[y, x], minterp * gains[y, x] + dark_vars[y, x])
            ax.set_xlim([mean_vmin, mean_vmax])
            ax.set_ylim([var_vmin, var_vmax])

            ax.set_title("(%d,%d), g=%0.2f, O=%0.2f, v=%0.2f" % (y, x, gains[y, x], offsets[y, x], dark_vars[y, x]))
            if bb == 0:
                ax.set_ylabel('variance (ADU^2)')
            if aa == (ncols - 1):
                ax.set_xlabel('mean (ADU)')

        figs_pix_eg[ll] = figh

    fighs = [figh1, figh2] + figs_pix_eg
    fns = [fn1, fn2] + figs_pix_eg_names

    return fighs, fns

