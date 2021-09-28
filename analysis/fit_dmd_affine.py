"""
Determine affine transformation mapping DMD space (object space) to camera space (image space).
The affine transformation (in homogeneous coordinates) is represented by a matrix,
[[xi], [yi], [1]] = T * [[xo], [yo], [1]]

Given a function defined on object space, g(xo, yo), we can define a corresponding function on image space
gi(xi, yi) = g(T^{-1} [[xi], [yi], [1]])
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

import affine
import rois
import fit

# helper functions for dealing with an array of dots projected on a flat background
def fit_pattern_peaks(img, centers, centers_init, indices_init, roi_size, chi_squared_relative_max=1.5,
                      max_position_err=0.1, img_sd=None, debug=False):
    """
    Fit peak of fluorescence pattern

    :param np.array img:
    :param centers:
    :param list[list[float]] centers_init: [[cy1, cx1], [cy2, cx2], ...] must supply at least three centers.
    Given one initial center, the other two should be shifted by one index along each direction of the DMD
    :param indices_init: indices corresponding to centers init
    :param int roi_size:
    :param chi_squared_relative_max: fits with chi squared values larger than this factor * the chi squared of the
     initial guess points will be ignored
    :param max_position_err: points where fits have larger relative position error than this value will be ignored
    :param img_sd:
    :param bool debug:
    """

    if img_sd is None:
        img_sd = np.ones(img.shape)

    indices_init = np.array(indices_init, copy=True)

    # indices for dmd_centers from mask
    inds_a = np.arange(centers.shape[0])
    inds_b = np.arange(centers.shape[1])
    ibib, iaia = np.meshgrid(inds_b, inds_a)

    # to store fitting results
    fps = np.zeros((centers.shape[0], centers.shape[1], 7))
    chisqs = np.zeros((centers.shape[0], centers.shape[1]))

    # fit initial dmd_centers
    for ii in range(len(centers_init)):
        roi = rois.get_centered_roi(centers_init[ii], [roi_size, roi_size])

        cell = img[roi[0]:roi[1], roi[2]:roi[3]]
        cell_sd = img_sd[roi[0]:roi[1], roi[2]:roi[3]]
        xx, yy = np.meshgrid(range(roi[2], roi[3]), range(roi[0], roi[1]))
        result, fit_fn = fit.fit_gauss2d(cell, sd=cell_sd, xx=xx, yy=yy)
        pfit = result['fit_params']
        chi_sq = result['chi_squared']

        fps[indices_init[ii, 0], indices_init[ii, 1], :] = pfit
        chisqs[indices_init[ii, 0], indices_init[ii, 1]] = chi_sq

        if debug:
            dx = xx[0, 1] - xx[0, 0]
            dy = yy[1, 0] - yy[0, 0]
            extent = (xx[0, 0] - 0.5 * dx, xx[0, -1] + 0.5 * dx,
                      yy[-1, 0] + 0.5 * dy, yy[0, 0] - 0.5 * dy)

            vmin = pfit[5]
            vmax = pfit[5] + pfit[0] * 1.2

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.imshow(cell, vmin=vmin, vmax=vmax, extent=extent)
            ax1.scatter(pfit[1], pfit[2], c='r', marker='x')
            ax2.imshow(fit_fn(xx, yy), vmin=vmin, vmax=vmax, extent=extent)
            ax3.imshow(cell - fit_fn(xx, yy), extent=(xx[0, 0], xx[0, -1], yy[-1, 0], yy[0, 0]))
            ax4.imshow(img, vmin=vmin, vmax=vmax)
            rec = Rectangle((roi[2], roi[0]), roi[3] - roi[2], roi[1] - roi[0], color='white', fill=0)
            ax4.add_artist(rec)
            plt.suptitle('(ia, ib) = (%d, %d), chi sq=%0.2f' % (indices_init[ii][0], indices_init[ii][1], chi_sq))

    # guess initial vec_a, vec_b
    #iia = np.array(indices_init)
    ias1, ias2 = np.meshgrid(indices_init[:, 0], indices_init[:, 0])
    adiffs = ias1 - ias2
    ibs1, ibs2 = np.meshgrid(indices_init[:, 1], indices_init[:, 1])
    bdiffs = ibs1 - ibs2
    xdiffs = fps[ias1, ibs1, 1] - fps[ias2, ibs2, 1]
    ydiffs = fps[ias1, ibs1, 2] - fps[ias2, ibs2, 2]

    to_use_a = np.logical_and(adiffs != 0, bdiffs == 0)
    vec_a_ref = np.array([np.mean(xdiffs[to_use_a] / adiffs[to_use_a]), np.mean(ydiffs[to_use_a] / adiffs[to_use_a])])

    to_use_b = np.logical_and(adiffs == 0, bdiffs != 0)
    vec_b_ref = np.array([np.mean(xdiffs[to_use_b] / bdiffs[to_use_b]), np.mean(ydiffs[to_use_b] / bdiffs[to_use_b])])

    # set maximum chi-square value that we will consider a 'good fit'
    # chi_sq_max = np.nanmean(chisqs[chisqs != 0]) * 1.5
    # the first three fits must succeed
    chi_sq_max = np.nanmax(chisqs[chisqs != 0]) * chi_squared_relative_max

    # loop over points
    # estimate position by guessing vec_a and vec_b from nearest pairs
    while np.sum(chisqs != 0) < chisqs.size:
        successful_fits = np.logical_and(chisqs > 0, chisqs < chi_sq_max)
        completed_fits = (chisqs != 0)

        # find nearest point to already fitted points, only including successful fits
        ia_fitted = iaia[successful_fits]
        ib_fitted = ibib[successful_fits]
        # use broadcasting to minimize distance sum
        dists = np.asarray(
            np.sum(np.square(iaia[:, :, None] - ia_fitted) + np.square(ibib[:, :, None] - ib_fitted), axis=2),
            dtype=np.float)
        # exclude points already considered
        dists[completed_fits] = np.nan
        # find minimum
        ind = np.nanargmin(dists)
        ind_tuple = np.unravel_index(ind, chisqs.shape)

        # also find nearest successfully fitted point for later use
        dists = np.asarray((iaia - ind_tuple[0]) ** 2 + (ibib - ind_tuple[1]) ** 2, dtype=np.float)
        dists[np.logical_not(successful_fits)] = np.nan
        nearest_ind = np.nanargmin(dists)
        nearest_ind_tuple = np.unravel_index(nearest_ind, chisqs.shape)

        # find nearest fitted pairs
        pair_fitted_a = np.logical_and(successful_fits[1:, :], successful_fits[:-1, :])
        # get coordinates for pairs
        bb, aa = np.meshgrid(inds_b, 0.5 * (inds_a[1:] + inds_a[:-1]))
        # find pair with minimum distance
        dists = (aa - ind_tuple[0]) ** 2 + (bb - ind_tuple[1]) ** 2
        dists[pair_fitted_a != 1] = np.nan
        ind_pair_a = np.nanargmin(dists)
        ind_pair_tuple_a = np.unravel_index(ind_pair_a, aa.shape)
        amin = aa[ind_pair_tuple_a]
        a1 = int(np.floor(amin))
        a2 = int(np.ceil(amin))
        bmin = bb[ind_pair_tuple_a]
        # get estimated a vector
        vec_a_guess = fps[a2, bmin, 1:3] - fps[a1, bmin, 1:3]

        # similarly for b
        pair_fitted_b = np.logical_and(successful_fits[:, 1:], successful_fits[:, :-1])
        bb, aa = np.meshgrid(0.5 * (inds_b[1:] + inds_b[:-1]), inds_a)
        # find pair with minimum distance
        dists = (aa - ind_tuple[0]) ** 2 + (bb - ind_tuple[1]) ** 2
        dists[pair_fitted_b != 1] = np.nan
        ind_pair_b = np.nanargmin(dists)
        ind_pair_tuple_b = np.unravel_index(ind_pair_b, aa.shape)
        amin = aa[ind_pair_tuple_b]
        bmin = bb[ind_pair_tuple_b]
        b1 = int(np.floor(bmin))
        b2 = int(np.ceil(bmin))
        # get estimated b vector
        vec_b_guess = fps[amin, b2, 1:3] - fps[amin, b1, 1:3]

        # guess point
        diff_a = ind_tuple[0] - nearest_ind_tuple[0]
        diff_b = ind_tuple[1] - nearest_ind_tuple[1]
        center_guess = fps[nearest_ind_tuple[0], nearest_ind_tuple[1], 1:3] + \
                       vec_a_guess * diff_a + vec_b_guess * diff_b
        xc = int(center_guess[0])
        yc = int(center_guess[1])

        # get roi
        roi = rois.get_centered_roi([yc, xc], [roi_size, roi_size])

        xstart = int(roi[2])
        xend = int(roi[3])
        ystart = int(roi[0])
        yend = int(roi[1])

        # do fitting if end points are reasonable
        end_points_ok = xstart >= 0 and xend < img.shape[1] and ystart >= 0 and yend < img.shape[0]
        if end_points_ok:
            xx, yy = np.meshgrid(range(xstart, xend), range(ystart, yend))
            cell = img[ystart:yend, xstart:xend]
            cell_sd = img_sd[ystart:yend, xstart:xend]

            result, fit_fn = fit.fit_gauss2d(cell, sd=cell_sd, xx=xx, yy=yy)
            pfit = result['fit_params']
            chi_sq = result['chi_squared']

            # check if actual center was within max_position_err of the distance we expected
            vec_ref = diff_a * vec_a_ref + diff_b * vec_b_ref
            center_err = np.linalg.norm(np.array([pfit[1], pfit[2]]) - np.array([xc, yc])) / np.linalg.norm(vec_ref)

            if center_err > max_position_err:
                fps[ind_tuple[0], ind_tuple[1], :] = np.nan
                chisqs[ind_tuple] = np.nan

            else:
                fps[ind_tuple[0], ind_tuple[1], :] = pfit
                chisqs[ind_tuple] = chi_sq

                if debug and np.sum(chisqs != 0) < 20:
                    vmin = pfit[5]
                    vmax = pfit[5] + pfit[0] * 1.2

                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                    ax1.imshow(cell, vmin=vmin, vmax=vmax, extent=(xx[0, 0], xx[0, -1], yy[-1, 0], yy[0, 0]))
                    ax1.scatter(pfit[1], pfit[2], c='r', marker='x')
                    ax2.imshow(fit_fn(xx, yy), vmin=vmin, vmax=vmax, extent=(xx[0, 0], xx[0, -1], yy[-1, 0], yy[0, 0]))
                    ax3.imshow(cell - fit_fn(xx, yy), extent=(xx[0, 0], xx[0, -1], yy[-1, 0], yy[0, 0]))
                    ax4.imshow(img, vmin=vmin, vmax=vmax)
                    rec = Rectangle((xstart, ystart), xend - xstart, yend - ystart, color='white', fill=0)
                    ax4.add_artist(rec)
                    plt.suptitle('(ia, ib) = (%d, %d), chi sq=%0.2f' % (ind_tuple[0], ind_tuple[1], chi_sq))

        else:
            fps[ind_tuple[0], ind_tuple[1], :] = np.nan
            chisqs[ind_tuple] = np.nan

    # final succesful fits
    successful_fits = np.logical_and(chisqs > 0, chisqs < chi_sq_max)

    return fps, chisqs, successful_fits


def get_affine_xform(fps, succesful_fits, dmd_centers):
    """
    Determine affine transformation from DMD to image space using the results of fit_pattern_peaks
    :param fps: fit parameters
    :param succesful_fits:
    :param dmd_centers:
    :return affine_xform:
    """
    # points to use for estimating affine transformation. Only include good fits.
    # using pixels as coordinates, instead of real distance
    xcam = fps[:, :, 1]
    ycam = fps[:, :, 2]
    xcam = xcam[succesful_fits]
    ycam = ycam[succesful_fits]

    xdmd = dmd_centers[:, :, 0]
    ydmd = dmd_centers[:, :, 1]
    xdmd = xdmd[succesful_fits]
    ydmd = ydmd[succesful_fits]

    # estimate affine transformation
    out_pts = [(x, y) for x, y in zip(xcam, ycam)]
    in_pts = [(x, y) for x, y in zip(xdmd, ydmd)]
    affine_xform, _ = affine.fit_xform_points(in_pts, out_pts)

    return affine_xform


def plot_affine_summary(img, mask, fps, chisqs, succesful_fits, dmd_centers,
                        affine_xform, options, figsize=(16, 12)):
    """
    Plot results of DMD affine transformation fitting using results of fit_pattern_peaks and get_affine_xform

    :param np.array img:
    :param np.array mask:
    :param np.array fps: fit parameters for each peak ny x nx x 7 array
    :param chisqs: chi squared statistic for each peak
    :param bool succesful_fits: boolean giving which fits we consider successful
    :param dmd_centers: ny x nx x 2, center positiosn on DMD
    :param affine_xform: affine transformation
    :param options: {'cam_pix', 'dmd_pix', 'dmd2cam_mag_expected', 'cam_mag'}

    :return fig: figure handle to summary figure
    """
    if "cam_mag" not in options.keys():
        options.update({"cam_mag": np.nan})


    fps = np.array(fps, copy=True)
    # ensure longer sigma is first
    to_swap = fps[:, :, 4] > fps[:, :, 3]
    sigma_max = np.array(fps[to_swap, 4], copy=True)
    sigma_min = np.array(fps[to_swap, 3], copy=True)

    fps[to_swap, 3] = sigma_max
    fps[to_swap, 4] = sigma_min
    fps[to_swap, 6] += np.pi / 2

    # extract useful info from fits
    xcam = fps[:, :, 1]
    ycam = fps[:, :, 2]
    xcam = xcam[succesful_fits]
    ycam = ycam[succesful_fits]
    amps = fps[:, :, 0]
    bgs = fps[:, :, 5]
    sigma_mean_cam = np.sqrt(fps[:, :, 3] * fps[:, :, 4])
    sigma_asymmetry_cam = np.abs(fps[:, :, 3] - fps[:, :, 4]) / (0.5 * (fps[:, :, 3] + fps[:, :, 4]))
    with np.errstate(invalid="ignore"):
        # angles differing by pi represent same ellipse
        angles = np.mod(fps[:, :, 6], np.pi)

    xdmd = dmd_centers[:, :, 0]
    ydmd = dmd_centers[:, :, 1]
    xdmd = xdmd[succesful_fits]
    ydmd = ydmd[succesful_fits]

    # extract parameters from options
    pixel_correction_factor = options['cam_pix'] / options['dmd_pix']
    expected_mag = options['dmd2cam_mag_expected']

    # get affine paramaters from xform
    affine_params = affine.xform2params(affine_xform)

    # transform DMD points and mask to image space
    homog_coords = np.concatenate((xdmd[None, :], ydmd[None, :], np.ones((1, xdmd.size))), axis=0)
    dmd_coords_xformed = affine_xform.dot(homog_coords)
    xdmd_xform = dmd_coords_xformed[0, :]
    ydmd_xform = dmd_coords_xformed[1, :]

    ny, nx = mask.shape
    xc_dmd = nx / 2
    yc_dmd = ny / 2
    xc_dmd_cam, yc_dmd_cam, _ = affine_xform.dot(np.array([[xc_dmd], [yc_dmd], [1]]))

    nvec = 100
    x_xvec, y_xvec, _ = affine_xform.dot(np.array([[xc_dmd + nvec], [yc_dmd], [1]]))
    x_yvec, y_yvec, _ = affine_xform.dot(np.array([[xc_dmd], [yc_dmd + nvec], [1]]))

    # residual position error
    residual_dist_err = np.zeros((fps.shape[0], fps.shape[1])) * np.nan
    residual_dist_err[succesful_fits] = np.sqrt((xdmd_xform - xcam)**2 + (ydmd_xform - ycam)**2)

    # get mask transformed to image space
    img_coords = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    mask_xformed = affine.xform_mat(mask, affine_xform, img_coords, mode='nearest')

    # summarize results
    vmin_img = np.min(bgs[succesful_fits])
    vmax_img = vmin_img + np.max(amps[succesful_fits]) * 1.2

    # plot results
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(6, 4)

    plt.subplot(grid[:2, 0])
    no_nans = chisqs.ravel()[np.logical_not(np.isnan(chisqs.ravel()))]
    vmin = np.percentile(no_nans, 1)
    vmax = np.percentile(no_nans, 90)
    plt.imshow(chisqs, vmin=vmin, vmax=vmax)
    plt.title('$\chi^2$')
    plt.colorbar()

    plt.subplot(grid[:2, 1])
    no_nans = residual_dist_err.ravel()[np.logical_not(np.isnan(residual_dist_err.ravel()))]
    vmax = np.percentile(no_nans, 90)
    plt.imshow(residual_dist_err, vmin=0, vmax=vmax)
    plt.title('position error (pix)')
    plt.colorbar()

    plt.subplot(grid[2:4, 0])
    no_nans = sigma_mean_cam.ravel()[np.logical_not(np.isnan(sigma_mean_cam.ravel()))]
    sigma_mean_median = np.median(no_nans)
    vmin = np.percentile(no_nans, 1)
    vmax = np.percentile(no_nans, 90)
    plt.imshow(sigma_mean_cam, vmin=vmin, vmax=vmax)
    sigma_m = sigma_mean_median * options["cam_pix"] / options["cam_mag"]
    plt.title('$\sqrt{\sigma_x \sigma_y}$, median=%0.2f\n$\sigma$=%0.1fnm, FWHM=%0.1fnm' %
              (sigma_mean_median, sigma_m * 1e9, sigma_m * 2 *  np.sqrt(2 * np.log(2)) * 1e9))
    plt.colorbar()

    plt.subplot(grid[2:4, 1])
    no_nans = amps.ravel()[np.logical_not(np.isnan(amps.ravel()))]
    amp_median = np.median(no_nans)
    vmin = np.percentile(no_nans, 1)
    vmax = np.percentile(no_nans, 90)
    plt.imshow(amps, vmin=vmin, vmax=vmax)
    plt.title('amps median=%0.0f' % amp_median)
    plt.colorbar()

    plt.subplot(grid[4:6, 0])
    no_nans = sigma_asymmetry_cam.ravel()[np.logical_not(np.isnan(sigma_asymmetry_cam.ravel()))]
    sigma_asym_median = np.median(no_nans)
    plt.imshow(sigma_asymmetry_cam, vmin=0, vmax=1)
    plt.title('$\sigma$ asym median=%0.2f' % sigma_asym_median)
    plt.colorbar()

    plt.subplot(grid[4:6, 1])
    no_nans = angles.ravel()[np.logical_not(np.isnan(angles.ravel()))]
    median_angle = np.median(no_nans * 180 / np.pi)
    plt.imshow(angles * 180 /np.pi, vmin=0, vmax=180)
    plt.title('angle, median=%0.1f$^\deg$' % median_angle)
    plt.colorbar()

    plt.subplot(grid[:3, 2:])
    plt.imshow(img, vmin=vmin_img, vmax=vmax_img)
    plt.plot(xcam, ycam, 'rx')
    plt.plot(xdmd_xform, ydmd_xform, 'y1')
    plt.plot(xc_dmd_cam, yc_dmd_cam, 'mx')
    plt.plot([xc_dmd_cam, x_xvec], [yc_dmd_cam, y_xvec], 'm')
    plt.plot([xc_dmd_cam, x_yvec], [yc_dmd_cam, y_yvec], 'm')
    plt.legend(["fit points", "affine xform"])

    plt.subplot(grid[3:, 2:])
    plt.imshow(mask_xformed)
    plt.plot(xc_dmd_cam, yc_dmd_cam, c='m', marker='x')
    plt.plot([xc_dmd_cam, x_xvec], [yc_dmd_cam, y_xvec], 'm')
    plt.plot([xc_dmd_cam, x_yvec], [yc_dmd_cam, y_yvec], 'm')
    plt.title('dmd mask xformed to img space')

    plt.suptitle('theta_x=%.2fdeg, mx=%0.3f, cx=%0.1f, pixel corrected mx=%.3f\n'
                 'theta_y=%.2fdeg, my=%0.3f, cy=%0.1f, pixel corrected my=%0.3f\n'
                 'expected mag=%0.3f'
                 % (affine_params[1] * 180 / np.pi, affine_params[0], affine_params[2], affine_params[0] * pixel_correction_factor,
                    affine_params[4] * 180 / np.pi, affine_params[3], affine_params[5], affine_params[3] * pixel_correction_factor,
                    expected_mag))

    return fig


# main function for estimating affine transform
def estimate_xform(img, mask, pattern_centers, centers_init, indices_init, options, roi_size=25,
                   export_fname="affine_xform", plot=True, export_dir=None, **kwargs):
    """
    Estimate affine transformation from DMD space to camera image space from an image.

    :param img: single image to analyze
    :param mask: DMD image that is being imaged
    :param pattern_centers: pattern centers in DMD coordinates
    :param centers_init: [[cy1, cx1], [cy2, cx2], [cy3, cx3]] in image space
    :param indices_init: [[ia1, ib1], [ia2, ib2], [ia3, ib3]]. ia indices are along the y-direction of the DMD pattern,
    starting with 0 which is the topmost when plotted. ib indices are long the x-direction starting with 0 which is the
    leftmost when plotted
    :param options: {'cam_pix', 'dmd_pix', 'dmd2cam_mag_expected'}. Distances are
    in meters.
    :param roi_size: size of ROI to fit each peak
    :param export_fname: file name (not including extension) to use when saving results
    :param export_dir: directory to save results
    :return data: affine trasnformation data
    :return fig: figure handle to summary data
    """

    # fit points
    fps, chisqs, succesful_fits = fit_pattern_peaks(img, pattern_centers, centers_init, indices_init,
                                                    roi_size, img_sd=None, debug=False, **kwargs)
    # get affine xforms
    affine_xform = get_affine_xform(fps, succesful_fits, pattern_centers)

    # export data
    data = {'affine_xform': affine_xform, 'pattern_centers': pattern_centers, 'fit_params': fps,
            'fit_params_description': "[amp, cx, cy, sigma_x, sigma_y, theta, bg]", "chi_squareds": chisqs}
    data.update(options)
    if export_dir is not None:
        fpath = os.path.join(export_dir, export_fname + ".pkl")
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)

    # plot data
    if not plot:
        fig = None
    else:
        fig = plot_affine_summary(img, mask, fps, chisqs, succesful_fits, pattern_centers, affine_xform, options)

        if export_dir is not None:
            fig_fpath = os.path.join(export_dir, export_fname + ".png")
            fig.savefig(fig_fpath)

        plt.show()

    return data, fig
