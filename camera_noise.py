import os
import numpy as np
import pickle
import time
from PIL import Image
import matplotlib.pyplot as plt
import tiffile

def adc2photons(img, gain_map, bg_map):
    """
    Convert ADC counts to photon number
    :param img:
    :param gain_map:
    :param bg_map:
    :return:
    """

    # subtraction
    photons = (img - bg_map) / gain_map

    # set anything less than zero to machine precision
    photons[photons <= 0] = np.finfo(float).eps
    return photons

def estimate_camera_noise(files, description=""):
    """
    Estimate camera readout noise from a set of images taken in zero light conditions
    :param list[str] files: list of file names
    :return data: dictionary storing relevant data, including means, variances, and higher order moments
    """

    tstart = time.process_time()

    # number of images to analyze at a time (if using TIFFs storing multiple images)
    n_chunk_size = 50

    # read first image to get size
    # if OME TIFF this will try to read the entire stack if don't set multifile
    imgs = tiffile.imread(files[0], multifile=False)

    nimgs = imgs.shape[0]

    means = np.mean(imgs, axis=0)
    mean_sqrs = np.zeros(means.shape)
    mean_cubes = np.zeros(means.shape)
    mean_fourths = np.zeros(means.shape)

    # loop to avoid casting entire
    for ii in range(0, nimgs, n_chunk_size):
        tnow = time.process_time()
        print("%d, elapsed time=%0.2fs" % (ii, tnow - tstart))
        img_float = np.asarray(imgs[ii : ii + n_chunk_size], dtype=np.float)
        mean_sqrs += np.sum(img_float**2, axis=0) / nimgs
        mean_cubes += np.sum(img_float**3, axis=0) / nimgs
        mean_fourths += np.sum(img_float**4, axis=0) / nimgs



    # mean_cubes = np.mean(np.asarray(imgs, dtype=np.float)**3, axis=0)
    # mean_fourths = np.mean(np.asarray(imgs, dtype=np.float)**4, axis=0)
    for ii, f in enumerate(files[1:]):
        tnow = time.process_time()
        print("finished %d/%d, elapsed time=%0.2fs" % (ii + 1, len(files), tnow - tstart))
        imgs = tiffile.imread(f, multifile=False)
        nimgs_current = imgs.shape[0]

        means = means * (nimgs / (nimgs + nimgs_current)) + \
                np.mean(imgs, axis=0) * (nimgs_current / (nimgs + nimgs_current))

        # loop to avoid casting entire
        mean_sqrs_temp = np.zeros(means.shape)
        mean_cubes_temp = np.zeros(means.shape)
        mean_fourths_temp = np.zeros(means.shape)
        for jj in range(0, nimgs_current, n_chunk_size):
            img_float = np.asarray(imgs[jj: jj + n_chunk_size], dtype=np.float)
            mean_sqrs_temp += np.sum(img_float ** 2, axis=0) / nimgs_current
            mean_cubes_temp += np.sum(img_float ** 3, axis=0) / nimgs_current
            mean_fourths_temp += np.sum(img_float ** 4, axis=0) / nimgs_current

        mean_sqrs = mean_sqrs * (nimgs / (nimgs + nimgs_current)) + \
                    mean_sqrs_temp * (nimgs_current / (nimgs + nimgs_current))
        mean_cubes = mean_cubes * (nimgs / (nimgs + nimgs_current)) + \
                     mean_cubes_temp * (nimgs_current / (nimgs + nimgs_current))
        mean_fourths = mean_fourths * (nimgs / (nimgs + nimgs_current)) + \
                       mean_fourths_temp * (nimgs_current / (nimgs + nimgs_current))

        nimgs += nimgs_current

    vars = mean_sqrs - means**2
    # variance of sample variance. See e.g. https://mathworld.wolfram.com/SampleVarianceDistribution.html
    mean_sqrs_central = mean_sqrs - means**2
    mean_fourth_central = mean_fourths - 4 * means * mean_cubes + 6 * means**2 * mean_sqrs -3 * means**4
    var_sample_var = (nimgs - 1)**2 / nimgs**3 * mean_fourth_central - \
                     (nimgs - 1) * (nimgs - 3) * mean_sqrs_central**2 / nimgs**3

    data = {"means": means, "means_unc": np.sqrt(vars) / np.sqrt(nimgs),
            "variances": vars, "variances_uncertainty": np.sqrt(var_sample_var),
            "mean_sqrs": mean_sqrs, "mean_cubes": mean_cubes, "mean_fourths": mean_fourths,
            "nimgs": nimgs,
            "description": description}

    return data

def get_gain_map(dark_data, light_data):
    """
    Get gain map from raw picture data
    :param dict dark_data: {"means", "variances"}.
    :param dict light_data: {"means": means, "variances": vars}. means and vars are n_illum x ny x nx arrays

    :return gains:
    :return offsets:
    :return variances:
    """
    # import data
    dark_means = dark_data['means']
    dark_vars = dark_data['variances']

    light_means = light_data["means"]
    light_vars = light_data["variances"]

    # subtract background
    ms = light_means - dark_means[None, :, :]
    vs = light_vars - dark_vars[None, :, :]

    # minimize over g_ij \sum_k ( (nu_ij - v_ij) - g_ij * (Dk_ij - o_ij) )^2
    # recast this as min || A^t - B^t * g||^2
    # g = pinv(B^t) * A^t
    ny, nx = dark_means.shape
    gains = np.zeros(dark_means.shape)
    for ii in range(ny):
        for jj in range(nx):
            gains[ii, jj] = np.linalg.pinv(ms[:, ii, jj][:, None]).dot(vs[:, ii, jj][:, None])

    return gains, dark_means, dark_vars

def plot_camera_noise_results(dark_data, light_data, gains, nbins=600, figsize=(20, 10)):
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

    offsets = dark_data['means']
    vars = dark_data['variances']

    light_means = light_data["means"]
    light_means_err = light_data["means_unc"]
    light_vars = light_data["variances"]
    light_vars_err = light_data["variances_uncertainty"]

    # ############################################
    # parameter histograms
    # ############################################

    bin_edges = np.linspace(1, 3, nbins + 1)
    hgains, _ = np.histogram(gains.ravel(), bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    offs_start = np.percentile(offsets, 0.1) - 1
    offs_end = np.percentile(offsets, 99.5)
    bin_edges_offs = np.linspace(offs_start, offs_end, nbins + 1)
    hmeans, _ = np.histogram(offsets.ravel(), bin_edges_offs)
    bin_centers_offs = 0.5 * (bin_edges_offs[:-1] + bin_edges_offs[1:])

    vars_start = 0 #np.percentile(vars, 0.1) - 1
    vars_end = np.percentile(vars, 99.5)
    bin_edges_vars = np.linspace(vars_start, vars_end, nbins + 1)
    hvars, _ = np.histogram(vars.ravel(), bin_edges_vars)
    bin_centers_vars = 0.5 * (bin_edges_vars[:-1] + bin_edges_vars[1:])

    # for comparison with read noise stats in electrons
    std_es = np.sqrt(vars) / gains

    sds_es_start = 0 #np.percentile(std_es, 0.1)
    sds_es_end = np.percentile(std_es, 99.5)
    bin_edges_stds_es = np.linspace(sds_es_start, sds_es_end, nbins + 1)
    h_std_es, _ = np.histogram(std_es.ravel(), bin_edges_stds_es)
    bin_centers_stds_es = 0.5 * (bin_edges_stds_es[:-1] + bin_edges_stds_es[1:])

    med_std_es = np.median(std_es)
    rms_es = np.sqrt(np.mean(std_es**2))

    fn1 = "camera_params_histograms"
    figh1 = plt.figure(figsize=figsize)
    grid = plt.GridSpec(nrows=2, ncols=2, wspace=0.2, hspace=0.3)
    plt.suptitle("Camera parameter histograms")

    ax1 = plt.subplot(grid[0, 0])
    ax1.plot(bin_centers, hgains)
    ax1.set_xlabel('gains (ADU/e)')
    ax1.set_ylabel('counts')
    ax1.set_title('histogram of pixel gains')

    ax2 = plt.subplot(grid[0, 1])
    ax2.plot(bin_centers_offs, hmeans)
    ax2.set_xlabel('offsets (ADU)')
    ax2.set_title('dark mean')

    ax3 = plt.subplot(grid[1, 0])
    ax3.plot(bin_centers_vars, hvars)
    ax3.set_xlabel('variances (ADU^2)')
    ax3.set_title('dark variances')

    ax4 = plt.subplot(grid[1, 1])
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
    grid2 = plt.GridSpec(nrows=2, ncols=2, wspace=0.2, hspace=0.3)
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

    ax3 = plt.subplot(grid2[1, 0])
    vmin = np.percentile(gains, 2)
    vmax = np.percentile(gains, 98)
    im3 = ax3.imshow(gains, vmin=vmin, vmax=vmax)
    ax3.set_title("gains")
    figh2.colorbar(im3)

    # ############################################
    # example gain fits
    # ############################################

    # choose some random pixels to plot
    ninds = 16
    xinds = np.random.randint(0, gains.shape[1], size=ninds)
    yinds = np.random.randint(0, gains.shape[0], size=ninds)

    # choose pixels with largest/smallest gains to plot
    sorted_inds = np.argsort(gains.ravel())
    yinds_small, xinds_small = np.unravel_index(sorted_inds[: ninds], gains.shape)
    yinds_large, xinds_large = np.unravel_index(sorted_inds[-ninds:], gains.shape)

    # values after subtracting background
    ms = light_means - offsets[None, :, :]
    vs = light_vars - vars[None, :, :]
    minterp = np.linspace(0, ms.max(), 100)

    figs_pix_eg = [[]]*3
    figs_pix_eg_names = ["camera_gain_fitting_examples", "camera_gain_fitting_smallest_gains", "camera_gain_fitting_largest_gains"]

    for ll, (yis, xis, fn) in enumerate(zip([yinds, yinds_small, yinds_large], [xinds, xinds_small, xinds_large], figs_pix_eg_names)):
        figs_pix_eg[ll] = plt.figure(figsize=figsize)
        grid3 = plt.GridSpec(4, 4, hspace=0.5, wspace=0.3)
        plt.suptitle(fn.replace("_", " "))

        for ii in range(ninds):
            ax = plt.subplot(grid3[ii])
            y = yis[ii]
            x = xis[ii]
            ax.errorbar(ms[:, y, x], vs[:, y, x], fmt='.', xerr=light_means_err[:, y, x], yerr=light_vars_err[:, y, x])
            ax.plot(minterp, minterp * gains[y, x])

            ax.set_title("(%d,%d), g=%0.2f" % (y, x, gains[y, x]))
            if ii == 0:
                ax.set_ylabel('variance')
                ax.set_xlabel('mean ADU counts')


    fighs = [figh1, figh2] + figs_pix_eg
    fns = [fn1, fn2] + figs_pix_eg_names

    return fighs, fns

def export_camera_params(offsets, variances, gains, id="", save_dir=''):
    """
    Export camera parameters as three separate tif files (one for gains, offsets, variances), and also as pickled file
    :param offsets:
    :param variances:
    :param gains:
    :param id:
    :param save_dir:
    :return:
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname = os.path.join(save_dir, "%s_camera_parameters.pkl" % id)
    data = {'gains': gains, 'offsets': offsets, 'vars': variances}
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    # also save to tiff file
    im = Image.fromarray(gains)
    im.save(os.path.join(save_dir, "%s_gain_map.tif" % id))

    im = Image.fromarray(offsets)
    im.save(os.path.join(save_dir, "%s_offset_map.tif" % id))

    im = Image.fromarray(variances)
    im.save(os.path.join(save_dir, "%s_variance_map.tif" % id))

if __name__ == "__main__":
    pass
