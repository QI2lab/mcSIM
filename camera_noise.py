"""
Functions for calibrating noise and gain map of sCMOS camera, doing denoising of imaging data, or producing
simulated camera data
"""
import os
import time
import warnings
import numpy as np
import pickle
from scipy import fft
from PIL import Image
import matplotlib.pyplot as plt

import tiffile
import analysis_tools as tools
import fit_psf as psf


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

def estimate_camera_noise(files, description="", calculate_correlations=False, n_chunk_size=50):
    """
    Estimate camera readout noise from a set of images taken in zero light conditions
    :param list[str] files: list of file names
    :param calculate_correlations:
    :param n_chunk_size: number of images from single file to analyze at once, if using TIFFs storing multiple images
    :return data: dictionary storing relevant data, including means, variances, and higher order moments
    """

    tstart = time.process_time()

    # read first image to get size
    # if OME TIFF this will try to read the entire stack if don't set multifile
    imgs = tiffile.imread(files[0], multifile=False)
    _, ny, nx = imgs.shape

    means = np.zeros((ny, nx))
    mean_sqrs = np.zeros(means.shape)
    mean_cubes = np.zeros(means.shape)
    mean_fourths = np.zeros(means.shape)
    corrsx = np.zeros(means.shape)
    corrsy = np.zeros(means.shape)
    corrst = np.zeros(means.shape)

    nimgs_prior = 0
    nimgs_t_corr = 0
    for ii, f in enumerate(files):
        tnow = time.process_time()
        print("started file %d/%d, elapsed time=%0.2fs" % (ii + 1, len(files), tnow - tstart))

        # don't reread first image
        if ii > 0:
            imgs = tiffile.imread(f, multifile=False)

        nimgs_current = imgs.shape[0]

        # loop to avoid casting entire array to float
        mean_sqrs_temp = np.zeros(means.shape)
        mean_cubes_temp = np.zeros(means.shape)
        mean_fourths_temp = np.zeros(means.shape)
        corrsx_temp = np.zeros(means.shape)
        corrsy_temp = np.zeros(means.shape)
        corrst_temp = np.zeros(means.shape)
        niterations_current = int(np.ceil(nimgs_current / n_chunk_size))
        for jj in range(0, nimgs_current, n_chunk_size):
            if ii == 0:
                tnow = time.process_time()
                print("chunk %d/%d, elapsed time=%0.2fs" % (jj / n_chunk_size + 1, niterations_current, tnow - tstart))
            img_float = np.asarray(imgs[jj: jj + n_chunk_size], dtype=np.float)
            mean_sqrs_temp += np.sum(img_float ** 2, axis=0) / nimgs_current
            mean_cubes_temp += np.sum(img_float ** 3, axis=0) / nimgs_current
            mean_fourths_temp += np.sum(img_float ** 4, axis=0) / nimgs_current

            if calculate_correlations:
                corrsx_temp += np.sum(img_float * np.roll(img_float, 1, axis=2), axis=0) / nimgs_current
                corrsy_temp += np.sum(img_float * np.roll(img_float, 1, axis=1), axis=0) / nimgs_current
                corrst_temp += np.sum(img_float[1:] * np.roll(img_float, 1, axis=0)[1:], axis=0) / (nimgs_current - niterations_current)

        # combine current images with all previously averaged images
        w1 = nimgs_prior / (nimgs_prior + nimgs_current)
        w2 = nimgs_current / (nimgs_prior + nimgs_current)

        means = means * w1 + np.mean(imgs, axis=0) * w2
        mean_sqrs = mean_sqrs * w1 + mean_sqrs_temp * w2
        mean_cubes = mean_cubes * w1 + mean_cubes_temp * w2
        mean_fourths = mean_fourths * w1 + mean_fourths_temp * w2
        corrsx = corrsx * w1 + corrsx_temp * w2
        corrsy = corrsy * w1 + corrsy_temp * w2

        w1t = nimgs_t_corr / (nimgs_t_corr + nimgs_current - niterations_current)
        w2t = (nimgs_current - niterations_current) / (nimgs_t_corr + nimgs_current - niterations_current)
        corrst = corrst * w1t + corrst_temp * w2t

        nimgs_prior += nimgs_current
        nimgs_t_corr += nimgs_current - niterations_current

    vars = mean_sqrs - means**2
    # variance of sample variance. See e.g. https://mathworld.wolfram.com/SampleVarianceDistribution.html
    mean_sqrs_central = mean_sqrs - means**2
    mean_fourth_central = mean_fourths - 4 * means * mean_cubes + 6 * means**2 * mean_sqrs -3 * means**4
    var_sample_var = (nimgs_prior - 1)**2 / nimgs_prior**3 * mean_fourth_central - \
                     (nimgs_prior - 1) * (nimgs_prior - 3) * mean_sqrs_central**2 / nimgs_prior**3

    data = {"means": means, "means_unc": np.sqrt(vars) / np.sqrt(nimgs_prior),
            "variances": vars, "variances_uncertainty": np.sqrt(var_sample_var),
            "mean_sqrs": mean_sqrs, "mean_cubes": mean_cubes, "mean_fourths": mean_fourths,
            "nimgs": nimgs_prior,
            "description": description}

    if calculate_correlations:
        data.update({"corr_x": corrsx, "corr_y": corrsy, "corr_t": corrst})

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
    ax1.set_title('histogram of pixel gains, median=%0.2f' % (np.median(gains.ravel())))

    ax2 = plt.subplot(grid[0, 1])
    ax2.plot(bin_centers_offs, hmeans)
    ax2.set_xlabel('offsets (ADU)')
    ax2.set_title('dark mean, median=%0.2f' % (np.median(offsets.ravel())))

    ax3 = plt.subplot(grid[1, 0])
    ax3.plot(bin_centers_vars, hvars)
    ax3.set_xlabel('variances (ADU^2)')
    ax3.set_title('dark variances, median=%0.2f' % (np.median(vars.ravel())))

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
            # ax.errorbar(ms[:, y, x], vs[:, y, x], fmt='.', xerr=light_means_err[:, y, x], yerr=light_vars_err[:, y, x])
            ax.plot([offsets[y, x], offsets[y, x]], [0, 1.2 * light_vars[:, y, x].max()], 'k')
            ax.plot([0, (ms + offsets[y, x]).max()], [vars[y, x], vars[y, x]], 'k')
            ax.errorbar(light_means[:, y, x], light_vars[:, y, x], fmt='.', xerr=light_means_err[:, y, x], yerr=light_vars_err[:, y, x])
            ax.plot(minterp + offsets[y, x], minterp * gains[y, x] + vars[y, x])

            ax.set_title("(%d,%d), g=%0.2f, O=%0.2f, v=%0.2f" % (y, x, gains[y, x], offsets[y, x], vars[y, x]))
            if ii == 0:
                ax.set_ylabel('variance (ADU^2)')
                ax.set_xlabel('mean (ADU)')


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

    # save to tiff file
    if gains is not None:
        im = Image.fromarray(gains)
        im.save(os.path.join(save_dir, "%s_gain_map.tif" % id))

    if offsets is not None:
        im = Image.fromarray(offsets)
        im.save(os.path.join(save_dir, "%s_offset_map.tif" % id))

    if variances is not None:
        im = Image.fromarray(variances)
        im.save(os.path.join(save_dir, "%s_variance_map.tif" % id))

    # pickle
    fname = os.path.join(save_dir, "%s_camera_parameters.pkl" % id)
    data = {'gains': gains, 'offsets': offsets, 'vars': variances}
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def simulated_img(ground_truth, max_photons, cam_gains, cam_offsets, cam_readout_noise_sds,
                  pix_size, otf=None, na=1.3, wavelength=0.5, photon_shot_noise=True, bin_size=1):
    """
    Convert ground truth image (with values between 0-1) to simulated camera image, including the effects of
    photon shot noise and camera readout noise.

    :param ground_truth: Relative intensity values of image
    :param max_photons: Mean photons emitted by ber of photons will be different than expected. Furthermore, due to
    the "blurring" of the point spread function and possible binning of the image, no point in the image
     may realize "max_photons"
    :param cam_gains: gains at each camera pixel
    :param cam_offsets: offsets of each camera pixel
    :param cam_readout_noise_sds: standard deviation characterizing readout noise at each camera pixel
    :param pix_size: pixel size of ground truth image in ums. Note that the pixel size of the output image will be
    pix_size * bin_size
    :param otf: optical transfer function. If None, use na and wavelength to set values
    :param na: numerical aperture. Only used if otf=None
    :param wavelength: wavelength in microns. Only used if otf=None
    :param photon_shot_noise: turn on/off photon shot-noise
    :param bin_size: bin pixels before applying Poisson/camera noise. This is to allow defining a pattern on a
    finer pixel grid.

    :return img:
    :return snr:
    :return max_photons_real:
    """
    if np.any(ground_truth > 1) or np.any(ground_truth < 0):
        warnings.warn('ground_truth image values should be in the range [0, 1] for max_photons to be correct')

    img_size = ground_truth.shape

    # get OTF
    if otf is None:
        fx = tools.get_fft_frqs(img_size[1], pix_size)
        fy = tools.get_fft_frqs(img_size[0], pix_size)
        otf = psf.circ_aperture_otf(fx[None, :], fy[:, None], na, wavelength)

    # blur image with otf/psf
    # todo: maybe should add an "imaging forward model" function to fit_psf.py and call it here.
    gt_ft = fft.fftshift(fft.fft2(fft.ifftshift(ground_truth)))
    img_blurred = max_photons * fft.fftshift(fft.ifft2(fft.ifftshift(gt_ft * otf))).real
    img_blurred[img_blurred < 0] = 0

    # resample image by binning
    img_blurred = tools.bin(img_blurred, (bin_size, bin_size), mode='sum')

    max_photons_real = img_blurred.max()

    # add shot noise
    if photon_shot_noise:
        img_shot_noise = np.random.poisson(img_blurred)
    else:
        img_shot_noise = img_blurred

    # add camera noise and convert from photons to ADU
    readout_noise = np.random.randn(img_shot_noise.shape[0], img_shot_noise.shape[1]) * cam_readout_noise_sds

    img = cam_gains * img_shot_noise + readout_noise + cam_offsets

    # signal to noise ratio
    sig = cam_gains * img_blurred
    # assuming photon number large enough ~gaussian
    noise = np.sqrt(cam_readout_noise_sds**2 + cam_gains**2 * img_blurred)
    snr = sig / noise

    return img, snr, max_photons_real

def denoise(img, gains, offsets, vars):
    pass

