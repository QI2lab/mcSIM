"""
Functions for calibrating noise and gain maps of sCMOS cameras using the method of https://doi.org/10.1038/nmeth.2488
"""
import datetime
from time import perf_counter
from collections.abc import Sequence
from typing import Optional, Union
import numpy as np
from numpy.random import randint
import dask.array as da
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    import cupy as cp
except ImportError:
    cp = None

if cp:
    array = Union[np.ndarray, cp.ndarray, da.core.Array]
else:
    array = Union[np.ndarray, da.core.Array]


def get_pixel_statistics(imgs: array,
                         moments: Sequence[int] = (1,),
                         corr_dists: Optional[Sequence[int]] = None,
                         metadata: Optional[dict] = None,
                         n_bootstrap: Optional[int] = None) -> dict:
    """
    Compute pixel statistics for a stack of CMOS camera images

    :param imgs: numpy or dask image array. If you have a set of individual tiff files, such an array can be produced
      with dask_image, e.g.
      >>> import dask_image
      >>> imgs = dask_image.imread.imread("*.tif")
      >>> stats = get_pixel_statistics(imgs)
    :param moments: list of moments to calculate. By default, compute the first and second moment
    :param corr_dists: list of correlator distances (along x, y) to compute correlations
    :param metadata: dictionary of information to add to output dictionary
    :param n_bootstrap: number of bootstrap iterations to compute to determine the uncertainty in the different moments
    :return data: dictionary storing data. Keys have the form "moment_n" for the nth moment and
     "corr_r_dist=n" where r = x, y, or t and n is the correlation step size
    """

    if corr_dists:
        corr_dists = np.asarray(corr_dists, dtype=int)

    if isinstance(imgs, np.ndarray):
        roll_fn = np.roll
    elif cp and isinstance(imgs, cp.ndarray):
        roll_fn = cp.roll
    else:
        roll_fn = da.roll

    def _compute(arr):
        if isinstance(arr, da.core.Array):
            with ProgressBar():
                arr = arr.compute()
        return arr

    # start processing
    tstart = perf_counter()
    nimgs, _, _ = imgs.shape

    # permutations to use for bootstrap
    if not n_bootstrap:
        permutations = [tuple(range(nimgs))]
    else:
        permutations = [tuple(randint(nimgs, size=nimgs)) for _ in range(n_bootstrap)]

    # initialize array to store data
    data = {"nimgs": nimgs,
            "n_bootstrap": n_bootstrap,
            "tstamp": datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
            }

    # compute using bootstrap to estimate uncertainty
    for ii, p in enumerate(permutations):
        print(f"computing permutation {ii+1:d}/{len(permutations):d} in "
              f"{perf_counter() - tstart:.2f}s", end="\r")
        imgsp = imgs[p, :, :].astype(float)

        # compute moments
        for m in moments:
            name = f"moment_{m:d}"
            name_unc = f"{name:s}_sd"
            if ii == 0:
                data[name] = 0
                data[name_unc] = 0

            moment_now = _compute((imgsp ** m).mean(axis=0))
            data[name] += moment_now / n_bootstrap
            data[name_unc] += moment_now**2 / n_bootstrap
            del moment_now

            if ii == len(permutations) - 1:
                data[name_unc] = np.sqrt(n_bootstrap / (n_bootstrap - 1) * (data[name_unc] - data[name]**2))

        # variance
        # when bootstrapping, suppose the above moments for each iteration are
        # m^{(l)}_i with mean over bootstrapped samples m^{(l)}
        # var_i = m^{(2)}_i - m^{(1)}_i^2
        # var = m^{(2)} - < m^{(1)}_i^2 >, which we cannot compute from moments alone
        name = "variance"
        name_unc = f"{name:s}_sd"
        if ii == 0:
            data[name] = 0
            data[name_unc] = 0

        var_now = _compute((imgsp**2).mean(axis=0)) - _compute(imgsp.mean(axis=0))**2
        data[name] += var_now / n_bootstrap
        data[name_unc] += var_now**2 / n_bootstrap

        if ii == len(permutations) - 1:
            data[name_unc] = np.sqrt(n_bootstrap / (n_bootstrap - 1) * (data[name_unc] - data[name] ** 2))

        # compute distance correlations
        if corr_dists:
            for cd in corr_dists:
                for ax in [1, 2]:
                    name = f"corr_{ax:d}_dist={cd:d}"
                    name_unc = f"{name:s}_sd"
                    if ii == 0:
                        data[name] = 0
                        data[name_unc] = 0

                    corr = _compute((imgsp * roll_fn(imgsp, cd, axis=ax)).mean(axis=0))
                    data[name] += corr / n_bootstrap
                    data[name_unc] += corr**2 / n_bootstrap
                    del corr

    data["processing_time_s"] = perf_counter() - tstart
    if metadata is not None:
        data.update(metadata)

    # calculate "uncertainty" in variance, if computed moments 1, 2, 3, and 4
    # variance of sample variance. See e.g. https://mathworld.wolfram.com/SampleVarianceDistribution.html
    # if np.all([f"moment_{m:d}" in results_names for m in (1, 2, 3, 4)]):
    #     moment2_central = data["moment_2"] - data["moment_1"]**2
    #     moment4_central = data["moment_4"] \
    #                       - 4 * data["moment_1"] * data["moment_3"] + \
    #                       6 * data["moment_1"]**2 * data["moment_2"] \
    #                       - 3 * data["moment_1"]**4
    #     var_sample_var = (nimgs - 1)**2 / nimgs**3 * moment4_central - \
    #                      (nimgs - 1) * (nimgs - 3) * moment2_central**2 / nimgs**3
    #
    #     data.update({"variance of sample variance": var_sample_var})

    return data


def get_gain_map(means: np.ndarray,
                 vars: np.ndarray,
                 vars_sd: Optional[np.ndarray] = None,
                 max_mean: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute camera pixel gains by finding the best-fit line describing the variance as a function of the mean

    .. math::

    g = \text{argmin} \sum_k \left | \left(\text{var}_k - \text{var}_\text{bg} \right) - g * \left( \text{mean}_k - \text{offset} \right) \right |^2

    :param means: array of size ni x ny x nx. The first entry should represent the background values
    :param vars: array of size ni x ny x nx
    :param vars_sd: estimate standard deviation of sample variance. This can be estimated e.g. using bootstrap methods
    :param max_mean: ignore data points with means larger than this value to avoid issues with saturation
    :return gains, gains_sd, offsets, background_var:
    """
    if vars_sd is None:
        vars_sd = np.ones_like(vars)

    dm = (means[1:] - means[0]) / vars_sd[1:]
    dv = (vars[1:] - vars[0]) / vars_sd[1:]

    # remove points too close to saturation
    not_use = means[1:] > max_mean
    dm[not_use] = np.nan

    # compute gain using penrose-moore pseudoinverse.
    # Easy to directly compute
    gains = np.nansum(dm / np.nansum(dm**2, axis=0) * dv, axis=0)
    gains_sd = np.sqrt(1 / np.nansum(dm**2, axis=0))

    return gains, gains_sd, means[0], vars[0]


def plot_camera_noise_results(gains: np.ndarray,
                              means: np.ndarray,
                              vars: np.ndarray,
                              gains_err: Optional[np.ndarray] = None,
                              means_err: Optional[np.ndarray] = None,
                              vars_err: Optional[np.ndarray] = None,
                              nbins: int = 600,
                              **kwargs) -> tuple[list[Figure], list[str]]:
    """
    Display gain curve for single pixel vs. illumination data.
    Additional keyword arguments are passed through to plt.figure()

    :param gains: ny x nx
    :param means: nill x ny x nx
    :param vars: nill x ny x nx
    :param gains_err:
    :param gains: estimated uncertainty (standard dev) of each gain value
    :param means_err: estimated uncertainty (standard dev) of each mean value
    :param vars_err: estimated uncertainty (standard dev) of each variance value
    :param nbins: number of bins for histograms
    :return figures, figure_names:
    """

    if gains_err is None:
        gains_err = np.zeros_like(gains)

    if means_err is None:
        means_err = np.zeros(means.shape) * np.nan

    if vars_err is None:
        vars_err = np.zeros(vars.shape) * np.nan

    # ############################################
    # parameter histograms
    # ############################################
    # gain
    bin_edges = np.linspace(np.percentile(gains, 0.1) * 0.8,
                            np.percentile(gains, 99.1) * 1.2,
                            nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hgains, _ = np.histogram(gains.ravel(), bin_edges)

    # offset
    bin_edges_offs = np.linspace(np.percentile(means[0], 0.1) - 1,
                                 np.percentile(means[0], 99.5),
                                 nbins + 1)
    bin_centers_offs = 0.5 * (bin_edges_offs[:-1] + bin_edges_offs[1:])
    hmeans, _ = np.histogram(means[0].ravel(), bin_edges_offs)

    # variance
    bin_edges_vars = np.linspace(0,
                                 np.percentile(vars[0], 99.5),
                                 nbins + 1)
    bin_centers_vars = 0.5 * (bin_edges_vars[:-1] + bin_edges_vars[1:])
    hvars, _ = np.histogram(vars[0].ravel(), bin_edges_vars)

    # read-noise in electrons
    with np.errstate(divide="ignore"):
        std_es = np.sqrt(vars[0]) / gains

    bin_edges_stds_es = np.linspace(0,
                                    np.percentile(std_es, 99.5),
                                    nbins + 1)
    bin_centers_stds_es = 0.5 * (bin_edges_stds_es[:-1] + bin_edges_stds_es[1:])
    h_std_es, _ = np.histogram(std_es.ravel(), bin_edges_stds_es)

    med_std_es = np.median(std_es)
    not_inf = np.logical_not(np.isinf(std_es))
    rms_es = np.sqrt(np.mean(std_es[not_inf]**2))

    # figure 1
    fn1 = "camera_params_summary"
    figh1 = plt.figure(**kwargs)
    figh1.suptitle("Camera parameter summary")
    grid = figh1.add_gridspec(nrows=2,
                              hspace=0.1,
                              ncols=4,
                              wspace=0.1,
                              left=0.02,
                              right=0.98
                              )

    # offsets
    offset_median = np.median(means[0].ravel())
    offset_sd = np.std(means[0].ravel())
    offset_unc_median = np.median(means_err[0].ravel())

    ax1 = figh1.add_subplot(grid[0, 0])
    im1 = ax1.imshow(means[0],
                     vmin=np.percentile(means[0], 2),
                     vmax=np.percentile(means[0], 98))
    ax1.set_title("offsets (ADU)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    figh1.colorbar(im1)

    ax2 = figh1.add_subplot(grid[1, 0])
    ax2.plot(bin_centers_offs, hmeans)
    ax2.axvline(offset_median, c="k")
    ax2.set_ylabel("Histogram (arb)")
    ax2.set_yticks([])
    ax2.set_xlabel('offsets (ADU)')
    ax2.set_title(f'median={offset_median:.2f} ADU\n'
                  f'sd={offset_sd:.2f} ADU\n'
                  f'median unc={offset_unc_median:.2f} ADU')

    # variances
    var_median = np.median(vars[0].ravel())
    var_sd = np.std(vars[0].ravel())
    var_unc_median = np.median(vars_err[0].ravel())

    ax2 = figh1.add_subplot(grid[0, 1])
    im2 = ax2.imshow(vars[0],
                     vmin=np.percentile(vars[0], 2),
                     vmax=np.percentile(vars[0], 98))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("variances (ADU^2)")
    figh1.colorbar(im2)

    ax3 = figh1.add_subplot(grid[1, 1])
    ax3.plot(bin_centers_vars, hvars)
    ax3.axvline(var_median, c='k')
    ax3.set_yticks([])
    ax3.set_xlabel('variance (ADU^2)')
    ax3.set_title(f'median={var_median:.2f} ADU^2\n'
                  f'sd={var_sd:.2f} ADU^2\n'
                  f'median unc={var_unc_median:.2f} ADU^2')

    # read noise
    ax4 = figh1.add_subplot(grid[0, 2])
    im4 = ax4.imshow(std_es,
                     vmin=0,
                     vmax=np.percentile(std_es, 98))
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title('readout noise sd (e)')
    figh1.colorbar(im4)

    ax4 = figh1.add_subplot(grid[1, 2])
    ax4.plot(bin_centers_stds_es, h_std_es)
    ax4.axvline(med_std_es,
                c='k',
                label=f"median = {med_std_es:.2f}e")
    ax4.axvline(rms_es,
                c='r',
                label=f"rms = {rms_es:.2f}e")
    ax4.set_yticks([])
    ax4.set_xlabel('standard deviation (e)')
    ax4.legend()

    # gain
    gain_med = np.median(gains.ravel())
    gain_sd = np.std(gains.ravel())
    gain_unc_median = np.median(gains_err.ravel())

    ax3 = figh1.add_subplot(grid[0, 3])
    im3 = ax3.imshow(gains,
                     vmin=np.percentile(gains, 2),
                     vmax=np.percentile(gains, 98))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("gains (ADU/e)")
    figh1.colorbar(im3)

    ax1 = figh1.add_subplot(grid[1, 3])
    ax1.plot(bin_centers,
             hgains)
    ax1.axvline(gain_med, c="k")
    ax1.set_xlabel('gains (ADU/e)')
    ax1.set_yticks([])
    ax1.set_title(f'median={gain_med:.2f} ADU/e\n'
                  f'sd={gain_sd:.2f} ADU/e\n'
                  f'unc median={gain_unc_median:.2f} ADU/e')

    # ############################################
    # correlations in params
    # ############################################
    fn2 = "parameter_correlations"

    figh2 = plt.figure(**kwargs)
    figh2.suptitle("Parameter correlations")
    grid2 = figh2.add_gridspec(ncols=3,
                               nrows=1)

    ax1 = figh2.add_subplot(grid2[0, 0])
    ax1.plot(means[0].ravel(), gains.ravel(), '.')
    ax1.axhline(gain_med, c='k')
    ax1.axvline(offset_median, c='k')
    ax1.set_xlabel("offsets (ADU)")
    ax1.set_ylabel("gains (ADU/e)")

    ax2 = figh2.add_subplot(grid2[0, 1])
    ax2.plot(means[0].ravel(), vars[0].ravel(), '.')
    ax2.axhline(var_median, c='k')
    ax2.axvline(offset_median, c='k')
    ax2.set_xlabel("offsets (ADU)")
    ax2.set_ylabel("vars (ADU^2)")

    ax3 = figh2.add_subplot(grid2[0, 2])
    ax3.plot(gains.ravel(), vars[0].ravel(), '.')
    ax3.axhline(var_median, c='k')
    ax3.axvline(gain_med, c='k')
    ax3.set_xlabel("gain (ADU/e)")
    ax3.set_ylabel("vars (ADU^2)")

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
    minterp = np.linspace(0, means[1:].max(), 100)
    var_vmin = -0.05 * np.mean(vars)
    var_vmax = np.percentile(vars[1:], 99.99)
    mean_vmin = -0.05 * means.mean()
    mean_vmax = minterp[-1] * 1.2

    # plot gain fits
    figs_pix_eg = [[]]*3
    figs_pix_eg_names = ["camera_gain_fitting_examples",
                         "camera_gain_fitting_smallest_gains",
                         "camera_gain_fitting_largest_gains"]
    use_lims = [True, True, False]
    for ll, (yis, xis, fn, ul) in enumerate(zip([yinds, yinds_small, yinds_large],
                                                [xinds, xinds_small, xinds_large],
                                                figs_pix_eg_names,
                                                use_lims)):
        figh = plt.figure(**kwargs)
        figh.suptitle(fn.replace("_", " "))
        grid3 = figh.add_gridspec(nrows=nrows,
                                  ncols=ncols,
                                  hspace=0.5,
                                  wspace=0.3)

        for ii in range(ninds):
            aa, bb = np.unravel_index(ii, (nrows, ncols))
            # grab pixel coordinates
            y = yis[ii]
            x = xis[ii]

            # create subplot
            ax = figh.add_subplot(grid3[aa, bb])
            ax.set_title(f"({y:d},{x:d})")

            # plot frames
            ax.errorbar(means[:, y, x],
                        vars[:, y, x],
                        fmt='.',
                        xerr=means_err[:, y, x],
                        yerr=vars_err[:, y, x])
            # plot fit
            lc = "orange"
            ax.fill_between(minterp + means[0, y, x],
                            minterp * (gains[y, x] - gains_err[y, x]) + vars[0, y, x],
                            minterp * (gains[y, x] + gains_err[y, x]) + vars[0, y, x],
                            color=lc,
                            alpha=0.5)

            ax.plot(minterp + means[0, y, x],
                    minterp * gains[y, x] + vars[0, y, x],
                    color=lc,
                    label=f"g={gains[y, x]:.2f}({gains_err[y, x]*1e2:.0f})\n"
                          f"o={means[0, y, x]:.1f}({means_err[0, y, x]*1e1:.0f})\n"
                          f"v={vars[0, y, x]:.0f}({vars_err[0, y, x]:.0f})")
            if ul:
                ax.set_xlim([mean_vmin, mean_vmax])
                ax.set_ylim([var_vmin, var_vmax])
            ax.legend(frameon=False)

            if bb == 0:
                ax.set_ylabel('variance (ADU^2)')
            if aa == (ncols - 1):
                ax.set_xlabel('mean (ADU)')

        figs_pix_eg[ll] = figh

    fighs = [figh1, figh2] + figs_pix_eg
    fns = [fn1, fn2] + figs_pix_eg_names

    return fighs, fns
