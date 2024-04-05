"""
Determine affine transformation mapping DMD space (object space) to camera space (image space) by projecting
an array of spots onto a relatively flat fluorescent background (e.g. a thin layer of fluorescent dye)

The affine transformation is represented by a matrix :math:`T` which relates object
space coordinates :math:`r_o = (x_o, y_o)` to image space coordinates :math:`r_i = (x_i, y_i)`,

.. math::

  \\begin{pmatrix}
  x_i\\\\
  y_i\\\\
  1
  \\end{pmatrix}
  = T
  \\begin{pmatrix}
  xo\\\\
  yo\\\\
  1
  \\end{pmatrix}

Given a function defined on object space, :math:`g(x_o, y_o)`, we can define a corresponding function on image space
:math:`\\tilde{g}(r_i) = g(T^{-1} r_i)`
"""

from typing import Optional, Union
from collections.abc import Sequence
import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm
from localize_psf import affine, rois, fit


def get_affine_fit_pattern(dmd_size: Sequence[int, int],
                           radii: Sequence[float] = (1., 1.5, 2.),
                           corner_size: int = 4,
                           point_spacing: int = 61,
                           mark_sep: int = 15):
    """
    Create DMD patterns of a sparse 2D grid of points all with the same radius. This is useful for determining the
    affine transformation between the DMD and the camera

    :param dmd_size: [nx, ny]
    :param radii: list of radii of spots for affine patterns.
     If more than one, more than one pattern will be generated.
    :param corner_size: size of blcosk indicating corners
    :param point_spacing: spacing between points
    :param mark_sep: separation between inversion/flip markers near center
    :return patterns, radii, centers: centers in format [[cx, cy]]
    """
    if isinstance(radii, (float, int)):
        radii = [radii]

    nx, ny = dmd_size

    # set spacing between points. Does not necessarily need to divide Nx and Ny
    xc = (point_spacing - 1) / 2
    yc = (point_spacing - 1) / 2

    cxs = np.arange(xc, nx, point_spacing)
    cys = np.arange(yc, ny, point_spacing)

    cxcx, cycy = np.meshgrid(cxs, cys)
    centers = np.concatenate((cxcx[:, :, None], cycy[:, :, None]), axis=2)

    patterns = []
    for r in radii:
        one_pt = np.zeros((point_spacing, point_spacing))
        xx, yy = np.meshgrid(range(one_pt.shape[1]), range(one_pt.shape[0]))
        rr = np.sqrt(np.square(xx - xc) + np.square(yy - yc))
        one_pt[rr < r] = 1

        mask = np.tile(one_pt, [int(np.ceil(ny / one_pt.shape[0])), int(np.ceil(nx / one_pt.shape[1]))])
        mask = mask[:ny, :nx]

        # add corners
        mask[:corner_size, :corner_size] = 1
        mask[:corner_size, -corner_size:] = 1
        mask[-corner_size:, :corner_size] = 1
        mask[-corner_size:, -corner_size:] = 1

        # add various markers to fix orientation

        # two edges
        mask[:1, :] = 1
        mask[:, :1] = 1

        # marks near center
        cx = nx // 2
        cy = ny // 2

        # block displaced along x-axis
        xstart1 = cx - mark_sep
        xend1 = xstart1 + corner_size
        ystart1 = cy - corner_size//2
        yend1 = ystart1 + corner_size
        mask[ystart1:yend1, xstart1:xend1] = 1

        # second block along x-axis
        xstart4 = cx - 2 * mark_sep
        xend4 = xstart4 + corner_size
        ystart4 = ystart1
        yend4 = yend1
        mask[ystart4:yend4, xstart4:xend4] = 1

        # central block
        xstart2 = cx - corner_size//2
        xend2 = xstart2 + corner_size
        ystart2 = cy - mark_sep
        yend2 = ystart2 + corner_size
        mask[ystart2:yend2, xstart2:xend2] = 1

        # block displaced along y-axis
        xstart3 = cx - corner_size//2
        xend3 = xstart3 + corner_size
        ystart3 = cy - corner_size//2
        yend3 = ystart3 + corner_size
        mask[ystart3:yend3, xstart3:xend3] = 1

        patterns.append(mask)

    patterns = np.asarray(patterns).astype(bool)

    return patterns, radii, centers


def fit_pattern_peaks(img: np.ndarray,
                      centers,
                      centers_init: Sequence[Sequence[float]],
                      indices_init: Sequence[Sequence[int]],
                      roi_size: int,
                      chi_squared_relative_max: float = np.inf,
                      max_position_err: float = 0.1,
                      img_sd: Optional[np.ndarray] = None,
                      debug: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Fit peaks of fluorescence image corresponding to affine calibration pattern.

    Affine calibration pattern can be obtained with dmd.get_affine_fit_pattern() function

    :param img: Fluorescence image of calibration pattern
    :param centers: spot positions on DMD
    :param centers_init: [[cy1, cx1], [cy2, cx2], ...] guess position of centers in the fluorescence image.
      Must supply at least three centers. Given one initial center, the other two should be shifted by one index
      along each direction of the DMD
    :param indices_init: indices corresponding to centers init in the calibration pattern
    :param roi_size: ROI size in pixels used in fitting
    :param chi_squared_relative_max: fits with chi squared values larger than this factor * the chi squared of the
      initial guess points will be ignored
    :param max_position_err: points where fits have larger relative position error than this value will be ignored
    :param img_sd:
    :param bool debug:
    :return:
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

    # #############################
    # fit initial centers
    # #############################
    for ii in range(len(centers_init)):
        roi = rois.get_centered_rois(centers_init[ii], [roi_size, roi_size])[0]

        cell = img[roi[0]:roi[1], roi[2]:roi[3]]
        cell_sd = img_sd[roi[0]:roi[1], roi[2]:roi[3]]
        xx, yy = np.meshgrid(range(roi[2], roi[3]), range(roi[0], roi[1]))

        gauss_model = fit.gauss2d()
        result = gauss_model.fit(cell, (yy, xx), init_params=None, sd=cell_sd)
        def fit_fn(x, y): return gauss_model.model((y, x), result["fit_params"])
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
            fig.suptitle('(ia, ib) = (%d, %d), chi sq=%0.2f' % (indices_init[ii][0], indices_init[ii][1], chi_sq))

    # #############################
    # guess spacing between spots, i.e. vec_a, vec_b, from initial fits
    # #############################
    # iia = np.array(indices_init)
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
    # the first three fits must succeed
    chi_sq_max = np.nanmax(chisqs[chisqs != 0]) * chi_squared_relative_max

    # loop over points
    # estimate position by guessing vec_a and vec_b from nearest pairs
    while np.sum(chisqs != 0) < chisqs.size:
        with np.errstate(invalid="ignore"):
            successful_fits = np.logical_and(chisqs > 0, chisqs < chi_sq_max)
            completed_fits = (chisqs != 0)

        # find nearest point to already fitted points, only including successful fits
        ia_fitted = iaia[successful_fits]
        ib_fitted = ibib[successful_fits]
        # use broadcasting to minimize distance sum
        dists = np.asarray(
            np.sum(np.square(iaia[:, :, None] - ia_fitted) + np.square(ibib[:, :, None] - ib_fitted), axis=2),
            dtype=float)
        # exclude points already considered
        dists[completed_fits] = np.nan
        # find minimum
        ind = np.nanargmin(dists)
        ind_tuple = np.unravel_index(ind, chisqs.shape)

        # also find nearest successfully fitted point for later use
        dists = np.asarray((iaia - ind_tuple[0]) ** 2 + (ibib - ind_tuple[1]) ** 2, dtype=float)
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
        roi = rois.get_centered_rois([yc, xc], [roi_size, roi_size])[0]

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

            gauss_model = fit.gauss2d()
            # init_params = gauss_model.estimate_parameters(cell, (yy, xx))
            result = gauss_model.fit(cell, (yy, xx), init_params=None, sd=cell_sd)
            def fit_fn(x, y): return gauss_model.model((y, x), result["fit_params"])
            # result, fit_fn = fit.fit_gauss2d(cell, sd=cell_sd, xx=xx, yy=yy)
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
                    fig.suptitle('(ia, ib) = (%d, %d), chi sq=%0.2f' % (ind_tuple[0], ind_tuple[1], chi_sq))

        else:
            fps[ind_tuple[0], ind_tuple[1], :] = np.nan
            chisqs[ind_tuple] = np.nan

    # final succesful fits
    with np.errstate(invalid="ignore"):
        successful_fits = np.logical_and(chisqs > 0, chisqs < chi_sq_max)

    return fps, chisqs, successful_fits


def plot_affine_summary(img: np.ndarray,
                        mask: np.ndarray,
                        fps: np.ndarray,
                        chisqs: np.ndarray,
                        succesful_fits: np.ndarray,
                        dmd_centers: np.ndarray,
                        affine_xform: np.ndarray,
                        options: dict,
                        indices_init: Optional[Sequence[Sequence[int]]] = None,
                        vmin_percentile: float = 5,
                        vmax_percentile: float = 99.9,
                        gamma: float = 1.,
                        **kwargs) -> matplotlib.figure.Figure:
    """
    Plot results of DMD affine transformation fitting using results of fit_pattern_peaks()

    :param img:
    :param mask:
    :param fps: fit parameters for each peak ny x nx x 7 array
    :param chisqs: chi squared statistic for each peak
    :param bool succesful_fits: boolean giving which fits we consider successful
    :param dmd_centers: ny x nx x 2, center positiosn on DMD
    :param affine_xform: affine transformation
    :param options: {'cam_pix', 'dmd_pix', 'dmd2cam_mag_expected', 'cam_mag'}
    :param indices_init:
    :param vmin_percentile:
    :param vmax_percentile:
    :param gamma: gamma used in displaying image
    :param kwargs: passed through to figure
    :return fig: figure handle to summary figure
    """

    if "cam_mag" not in options.keys():
        options.update({"cam_mag": np.nan})

    fps = np.array(fps, copy=True)
    # ensure longer sigma is first
    with np.errstate(invalid="ignore"):
        to_swap = fps[:, :, 4] > fps[:, :, 3]
    sigma_max = np.array(fps[to_swap, 4], copy=True)
    sigma_min = np.array(fps[to_swap, 3], copy=True)

    fps[to_swap, 3] = sigma_max
    fps[to_swap, 4] = sigma_min
    fps[to_swap, 6] += np.pi / 2

    # extract useful info from fits
    xcam = fps[succesful_fits, 1]
    ycam = fps[succesful_fits, 2]
    amps = fps[:, :, 0]
    bgs = fps[:, :, 5]
    with np.errstate(invalid="ignore"):
        sigma_mean_cam = np.sqrt(fps[:, :, 3] * fps[:, :, 4])
        sigma_asymmetry_cam = np.abs(fps[:, :, 3] - fps[:, :, 4]) / (0.5 * (fps[:, :, 3] + fps[:, :, 4]))
        # angles differing by pi represent same ellipse
        angles = np.mod(fps[:, :, 6], np.pi)

    # extract parameters from options
    pixel_correction_factor = options['cam_pix'] / options['dmd_pix']
    expected_mag = options['dmd2cam_mag_expected']

    # get affine parematers from xform
    affine_params = affine.xform2params(affine_xform)

    # transform DMD points and mask to image space
    dmd_coords_xform = affine.xform_points(dmd_centers[succesful_fits, :], affine_xform)
    xdmd_xform = dmd_coords_xform[:, 0]
    ydmd_xform = dmd_coords_xform[:, 1]

    # DMD axes image space
    nvec = 100
    ny, nx = mask.shape
    xc_dmd = nx // 2
    yc_dmd = ny // 2

    dmd_axes = np.array([[xc_dmd, yc_dmd],
                         [xc_dmd + nvec, yc_dmd],
                         [xc_dmd, yc_dmd],
                         [xc_dmd, yc_dmd + nvec]])

    dmd_axes_cam = affine.xform_points(dmd_axes, affine_xform)

    # residual position error
    residual_dist_err = np.zeros((fps.shape[0], fps.shape[1])) * np.nan
    residual_dist_err[succesful_fits] = np.sqrt((xdmd_xform - xcam)**2 + (ydmd_xform - ycam)**2)

    # get mask transformed to image space
    img_coords = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    mask_xformed = affine.xform_mat(mask, affine_xform, img_coords, mode='nearest')

    # #####################################
    # plot results
    # #####################################
    fig = plt.figure(**kwargs)
    grid = fig.add_gridspec(nrows=6,
                            ncols=5, width_ratios=[1, 1, 1, 1, 1])
    cmap = "inferno"

    # chi squareds
    ax = fig.add_subplot(grid[:2, 0])
    ax.set_title('$\chi^2$')
    no_nans = chisqs.ravel()[np.logical_not(np.isnan(chisqs.ravel()))]
    vmin = np.percentile(no_nans, 1)
    vmax = np.percentile(no_nans, 90)
    im = ax.imshow(chisqs, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im)

    # position errors
    ax = fig.add_subplot(grid[:2, 1])
    ax.set_title('position error (pix)')

    no_nans = residual_dist_err.ravel()[np.logical_not(np.isnan(residual_dist_err.ravel()))]
    vmax = np.percentile(no_nans, 90)
    im = ax.imshow(residual_dist_err, vmin=0, vmax=vmax, cmap=cmap)
    plt.colorbar(im)

    # sigmas
    ax = fig.add_subplot(grid[2:4, 0])
    no_nans = sigma_mean_cam.ravel()[np.logical_not(np.isnan(sigma_mean_cam.ravel()))]
    sigma_mean_median = np.median(no_nans)
    vmin = np.percentile(no_nans, 1)
    vmax = np.percentile(no_nans, 90)
    im = ax.imshow(sigma_mean_cam, vmin=vmin, vmax=vmax, cmap=cmap)

    sigma_m = sigma_mean_median * options["cam_pix"] / options["cam_mag"]
    ax.set_title('$\sqrt{\sigma_x \sigma_y}$, median=%0.2f pix'
                 '\n$\sigma$=%0.1fnm, FWHM=%0.1fnm' %
                 (sigma_mean_median, sigma_m * 1e9, sigma_m * 2 * np.sqrt(2 * np.log(2)) * 1e9))
    plt.colorbar(im)

    # amplitudes
    ax = fig.add_subplot(grid[2:4, 1])
    no_nans = amps.ravel()[np.logical_not(np.isnan(amps.ravel()))]
    amp_median = np.median(no_nans)
    vmin = np.percentile(no_nans, 1)
    vmax = np.percentile(no_nans, 90)
    im = ax.imshow(amps, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('amps median=%0.0f' % amp_median)
    plt.colorbar(im)

    # sigma asymmetry
    ax = fig.add_subplot(grid[4:6, 0])
    no_nans = sigma_asymmetry_cam.ravel()[np.logical_not(np.isnan(sigma_asymmetry_cam.ravel()))]
    sigma_asym_median = np.median(no_nans)
    im = ax.imshow(sigma_asymmetry_cam, vmin=0, vmax=1, cmap=cmap)
    ax.set_title('$\sigma$ asym median=%0.2f' % sigma_asym_median)
    plt.colorbar(im)

    # angles
    ax = fig.add_subplot(grid[4:6, 1])
    no_nans = angles.ravel()[np.logical_not(np.isnan(angles.ravel()))]
    median_angle = np.median(no_nans * 180 / np.pi)
    im = ax.imshow(angles * 180 / np.pi, vmin=0, vmax=180, cmap=cmap)
    ax.set_title('angle, median=%0.1f$^\deg$' % median_angle)
    plt.colorbar(im)

    # raw image with fit points overlayed
    ax = fig.add_subplot(grid[:3, 2:4])
    ax.set_title('raw image and fits')

    im = ax.imshow(img,
                   norm=PowerNorm(vmin=np.percentile(img, vmin_percentile),
                                  vmax=np.percentile(img, vmax_percentile),
                                  gamma=gamma),
                   cmap="bone")
    ax.plot(xcam, ycam, 'rx', label="fit points")
    ax.plot(xdmd_xform, ydmd_xform, 'y1', label="affine xform")
    # ax.plot(xc_dmd_cam, yc_dmd_cam, 'mx')
    # ax.plot([xc_dmd_cam, x_xvec], [yc_dmd_cam, y_xvec], 'm', label="dmd axes")
    # ax.plot([xc_dmd_cam, x_yvec], [yc_dmd_cam, y_yvec], 'm')
    ax.plot(dmd_axes_cam[:2, 0], dmd_axes_cam[:2, 1], 'm', label="dmd axes")
    ax.plot(dmd_axes_cam[2:, 0], dmd_axes_cam[2:, 1], 'm')
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im)

    # ax = fig.add_subplot(grid[:3, 4])
    # plt.colorbar(im, cax=ax)

    # dmd image transformed
    ax = fig.add_subplot(grid[3:, 2:4])
    ax.set_title('DMD pattern (camera space)')
    im = ax.imshow(mask_xformed, cmap="bone")
    # ax.plot(xc_dmd_cam, yc_dmd_cam, c='m', marker='x')
    ax.plot(dmd_axes_cam[:2, 0], dmd_axes_cam[:2, 1], 'm')
    ax.plot(dmd_axes_cam[2:, 0], dmd_axes_cam[2:, 1], 'm')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im)

    # dmd image
    ax = fig.add_subplot(grid[3:, 4])
    ax.set_title("DMD pattern (DMD space)")
    im = ax.imshow(mask, cmap="bone")
    ax.plot(dmd_axes[:2, 0], dmd_axes[:2, 1], 'm')
    ax.plot(dmd_axes[2:, 0], dmd_axes[2:, 1], 'm')
    if indices_init is not None:
        dmd_centers_init = np.array([dmd_centers[ind[0], ind[1], :] for ind in indices_init])
        ax.plot(dmd_centers_init[:, 0], dmd_centers_init[:, 1], 'gx', label="centers init")
        ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle('theta_x=%.2fdeg, mx=%0.3f, cx=%0.1f, pixel corrected mx=%.3f\n'
                 'theta_y=%.2fdeg, my=%0.3f, cy=%0.1f, pixel corrected my=%0.3f\n'
                 'expected mag=%0.3f'
                 % (affine_params[1] * 180 / np.pi, affine_params[0], affine_params[2], affine_params[0] * pixel_correction_factor,
                    affine_params[4] * 180 / np.pi, affine_params[3], affine_params[5], affine_params[3] * pixel_correction_factor,
                    expected_mag))

    return fig


# main function for estimating affine transform
def estimate_xform(img: np.ndarray,
                   mask: np.ndarray,
                   pattern_centers,
                   centers_init: Sequence[Sequence[float]],
                   indices_init: Sequence[Sequence[float]],
                   options: dict,
                   roi_size: int = 25,
                   export_fname: str = "affine_xform",
                   plot: bool = True,
                   export_dir: Optional[Union[str, Path]] = None,
                   debug: bool = False,
                   vmin_percentile: float = 5.,
                   vmax_percentile: float = 99.,
                   gamma: float = 1.,
                   figsize: Sequence[float, float] = (16., 12.),
                   **kwargs) -> (dict, matplotlib.figure.Figure):
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
    :param plot:
    :param export_dir: directory to save results
    :param debug:
    :param vmin_percentile:
    :param vmax_percentile:
    :param gamma:
    :param figsize:
    :return (data, figure): affine trasnformation data and figure handle
    """

    # fit points
    fps, chisqs, succesful_fits = fit_pattern_peaks(img,
                                                    pattern_centers,
                                                    centers_init,
                                                    indices_init,
                                                    roi_size,
                                                    img_sd=None,
                                                    debug=debug,
                                                    **kwargs)

    # affine_xform = get_affine_xform(fps, succesful_fits, pattern_centers)
    # get affine xform
    # using pixels as coordinates, instead of real distance
    out_pts = fps[succesful_fits, 1:3]
    in_pts = pattern_centers[succesful_fits, 0:2]
    affine_xform, _ = affine.fit_xform_points(in_pts, out_pts)

    # export data
    data = {'affine_xform': affine_xform.tolist(),
            'pattern_centers': pattern_centers.tolist(),
            'fit_params': fps.tolist(),
            'fit_params_description': "[amp, cx, cy, sigma_x, sigma_y, theta, bg]",
            "chi_squareds": chisqs.tolist()}
    data.update(options)

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(exist_ok=True)

        fpath = Path(export_dir) / f"{export_fname:s}.json"
        with open(fpath, "w") as f:
            json.dump(data, f, indent="\t")

    # plot data
    if not plot:
        fig = None
    else:
        fig = plot_affine_summary(img,
                                  mask,
                                  fps,
                                  chisqs,
                                  succesful_fits,
                                  pattern_centers,
                                  affine_xform,
                                  options,
                                  indices_init=indices_init,
                                  vmin_percentile=vmin_percentile,
                                  vmax_percentile=vmax_percentile,
                                  gamma=gamma,
                                  figsize=figsize)

        if export_dir is not None:
            fig.savefig(Path(export_dir) / f"{export_fname:s}.png")

        plt.show()

    return data, fig
