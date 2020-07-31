"""
Determine affine transformation mapping DMD space (object space) to camera space (image space).
The affine transformation (in homogeneous coordinates) is represented by a matrix,
[[xi], [yi], [1]] = T * [[xo], [yo], [1]]

Given a function defined on object space, g(xo, yo), we can define a corresponding function on image space
gi(xi, yi) = g(T^{-1} [[xi], [yi], [1]])
"""

import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.path
from scipy import optimize
import scipy.interpolate
import pickle

import analysis_tools as tools
# from . import analysis_tools as tools


# affine matrix parameterizations
def xform2params(affine_mat):
    """
    Parametrize affine transformation in terms of rotation angles, magnifications, and offsets.
    T = Mx * cos(tx), -My * sin(ty), vx
        Mx * sin(tx),  My * cos(ty), vy
           0        ,    0        , 1

    Both theta_x and theta_y are measured CCW from the x-axis

    :param np.array affine_mat:

    :return list[float]: [mx, theta_x, vx, my, theta_y, vy]
    """
    # get offsets
    vx = affine_mat[0, -1]
    vy = affine_mat[1, -1]

    # get rotation and scale for x-axis
    theta_x = np.angle(affine_mat[0, 0] + 1j * affine_mat[1, 0])
    mx = np.nanmean([affine_mat[0, 0] / np.cos(theta_x), affine_mat[1, 0] / np.sin(theta_x)])

    # get rotation and scale for y-axis
    theta_y = np.angle(affine_mat[1, 1] - 1j * affine_mat[0, 1])
    my = np.nanmean([affine_mat[1, 1] / np.cos(theta_y), -affine_mat[0, 1] / np.sin(theta_y)])

    return [mx, theta_x, vx, my, theta_y, vy]


def inv_xform2params(affine_mat_inv):
    """
    Compute parameters of affine transform from the inverse matrix

    T = Mx * cos(tx), -My * sin(ty), vx
        Mx * sin(tx),  My * cos(ty), vy
           0        ,    0        , 1

    T^{-1} = |1/Mx,    0,  0 |   | cos(ty)/F, sin(ty)/F, 0|    |1, 0, -vx|
             |  0 ,  1/My, 0 | * |-sin(tx)/F, cos(tx)/F, 0|  * |0, 1, -vy|
             |  0,     0 , 1 |   |     0    ,     0    , 1|    |0, 0,  0|

           = cos(ty)/(Mx*cos(tx-ty), sin(ty)/(My*cos(tx-ty)), -( vx*cos(ty) + vy*sin(ty))/(Mx*cos(tx-ty))
            -sin(tx)/(My*cos(tx-ty), cos(tx)/(My*cos(tx-ty)), -(-Vx*sin(tx) + vy*cos(tx))/(My*cos(tx-ty))
                     0             ,             0          ,                  1

    :param np.array affine_mat_inv:

    :return list[float]: [mx, theta_x, vx, my, theta_y, vy]
    """

    # todo: why not just invert the matrix and apply xform2params?

    # this works as long as np.cos(theta_x - theta_y) > 0
    theta_x = np.angle(affine_mat_inv[1, 1] - 1j * affine_mat_inv[1, 0])
    theta_y = np.angle(affine_mat_inv[0, 0] + 1j * affine_mat_inv[0, 1])
    if np.cos(theta_x - theta_y) < 0:
        theta_x = theta_x + np.pi
        theta_y = theta_y + np.pi

    mx = 1 / (np.cos(theta_x - theta_y) * np.nanmean([affine_mat_inv[0, 0] / np.cos(theta_y),
                                                      affine_mat_inv[0, 1] / np.sin(theta_y)]))
    my = 1 / (np.cos(theta_x - theta_y) * np.nanmean([-affine_mat_inv[1, 0] / np.sin(theta_x),
                                                      affine_mat_inv[1, 1] / np.cos(theta_x)]))

    # invert the matrix we've found so far, and whats left is just the offsets
    mat = params2xform([mx, theta_x, 0, my, theta_y, 0])
    shift_mat = mat.dot(affine_mat_inv)
    vx = -shift_mat[0, 2]
    vy = -shift_mat[1, 2]

    return [mx, theta_x, vx, my, theta_y, vy]


def params2xform(params):
    """
    Construct affine transformation from parameters. Inverse function for xform2params()

    T = Mx * cos(tx), -My * sin(ty), vx
        Mx * sin(tx),  My * cos(ty), vy
           0        ,    0        , 1

    :param list[float] params: [Mx, theta_x, vx ,My, theta_y, vy]

    :return np.array affine_xform:
    """
    # read parameters
    mx = params[0]
    theta_x = params[1]
    vx = params[2]
    my = params[3]
    theta_y = params[4]
    vy = params[5]

    # construct affine xform
    affine_xform = np.array([[mx * np.cos(theta_x), -my * np.sin(theta_y), vx],
                             [mx * np.sin(theta_x),  my * np.cos(theta_y), vy],
                             [0                   ,  0                   ,  1]])

    return affine_xform


# transform functions/matrices under action of affine transformation
def affine_xform_mat(mat, xform, img_coords, mode='nearest'):
    """
    Given roi_size matrix defined on object space coordinates, i.e. M[yo, xo], calculate corresponding matrix at image
    space coordinates, M'[yi, xi] = M[ T^{-1} * [xi, yi] ]

    Object coordinates are assumed to be [0, ..., nx-1] and [0, ..., ny-1]

    # todo: change out_shape to a coordinate argument so I can directly generate e.g. a region of interest or etc.

    :param np.array mat: matrix in object space
    :param np.array xform: affine transformation which takes object space coordinates as input, [yi, xi] = T * [xo, yo]
    :param img_coords: (xi, yi) coordinates where the transformed matrix is evaluated.
    :param str mode: 'nearest' or 'interp'. 'interp' will produce better results if e.g. looking at phase content after
    affine transformation.

    :return mat_out: matrix in image space
    """

    # if obj_coords is None:
    #     xo = np.arange(mat.shape[1])
    #     yo = np.arange(mat.shape[0])
    # else:
    #     xo, yo = obj_coords
    xo = np.arange(mat.shape[1])
    yo = np.arange(mat.shape[0])

    # xi, yi = img_coords
    # xixi, yiyi = np.meshgrid(xi, yi)
    xixi, yiyi = img_coords
    mat_out = np.zeros(xixi.shape)
    # xixi, yiyi = np.meshgrid(range(out_shape[1]), range(out_shape[0]))

    coords_i_aug = np.concatenate((xixi.ravel()[None, :], yiyi.ravel()[None, :], np.ones((1, xixi.size))), axis=0)
    # get corresponding object space coordinates
    coords_o = np.linalg.inv(xform).dot(coords_i_aug)[:2, :]
    xos = np.reshape(coords_o[0, :], xixi.shape)
    yos = np.reshape(coords_o[1, :], xixi.shape)

    if mode == 'nearest':
        # xos = np.round(xos)
        # yos = np.round(yos)

        # only use points with coords in image
        to_use_x = np.logical_and(xos >= np.min(xo), xos <= np.max(xo))
        to_use_y = np.logical_and(yos >= np.min(yo), yos < np.max(yo))
        to_use = np.logical_and(to_use_x, to_use_y)

        # find closest point in matrix to each output point
        # inds_y = tuple([np.argmin(np.abs(y - yo)) for y in yos[to_use]])
        # inds_x = tuple([np.argmin(np.abs(x - xo)) for x in xos[to_use]])

        # inds_y = tuple([int(np.round(y)) for y in yos[to_use]])
        # inds_x = tuple([int(np.round(x)) for x in xos[to_use]])

        inds_y = np.array(np.round(yos[to_use]), dtype=np.int)
        inds_x = np.array(np.round(xos[to_use]), dtype=np.int)

        inds = (tuple(inds_y), tuple(inds_x))
        mat_out[to_use] = mat[inds]

    elif mode == 'interp':
        # only use points with coords in image
        to_use_x = np.logical_and(xos >= 0, xos < mat.shape[1])
        to_use_y = np.logical_and(yos >= 0, yos < mat.shape[0])
        to_use = np.logical_and(to_use_x, to_use_y)

        mat_out = scipy.interpolate.RectBivariateSpline(xo, yo, mat.transpose()).ev(xos, yos)
        mat_out[np.logical_not(to_use)] = 0
    else:
        raise Exception("'mode' must be 'nearest' or 'interp' but was %s" % mode)

    return mat_out


def xform_fn(fn, xform, out_coords):
    """
    Given a function on object space, evaluate the corresponding image space function at out_coords

    :param fn: function on object space. fn(x, y)
    :param xform: affine transformation matrix which takes points in object space to points in image space
    :param out_coords: (x_img, y_img) coordinates in image space. x_img and y_img must be the same size

    :return img: function evaluated at desired image space coordinates
    """

    x_img, y_img = out_coords
    xform_inv = np.linalg.inv(xform)

    coords_i = np.concatenate((x_img.ravel()[None, :], y_img.ravel()[None, :], np.ones((1, y_img.size))), axis=0)
    coords_o = xform_inv.dot(coords_i)
    x_o = coords_o[0]
    y_o = coords_o[1]

    img = np.reshape(fn(x_o, y_o), x_img.shape)

    return img


# modify affine xform
def xform_shift_center(xform, cobj_new=None, cimg_new=None):
    """
    Modify affine transform for coordinate shift in object or image space.

    Useful e.g. for changing region of interest

    Ro_new = Ro_old - Co
    Ri_new = Ri_old - Ci

    :param xform:
    :param cobj_new: [cox, coy]
    :param cimg_new: [cix, ciy]
    :return:
    """

    xform = copy.deepcopy(xform)

    if cobj_new is None:
        cobj_new = [0, 0]
    cox, coy = cobj_new

    xform[0, 2] = xform[0, 2] + xform[0, 0] * cox + xform[0, 1] * coy
    xform[1, 2] = xform[1, 2] + xform[1, 0] * cox + xform[1, 1] * coy

    if cimg_new is None:
        cimg_new = [0, 0]
    cix, ciy = cimg_new

    xform[0, 2] = xform[0, 2] - cix
    xform[1, 2] = xform[1, 2] - ciy

    return xform


# transform sinusoid parameters for coordinate shifts
def phase_edge2fft(frq, phase, img_shape, dx=1):
    """

    :param list[float] or np.array frq:
    :param float phase:
    :param tuple or list img_shape:
    :param float dx:

    :return phase_fft:
    """

    xft = tools.get_fft_pos(img_shape[1], dt=dx, centered=True, mode="symmetric")
    yft = tools.get_fft_pos(img_shape[0], dt=dx, centered=True, mode="symmetric")
    phase_fft = xform_phase_translation(frq[0], frq[1], phase, [-xft[0], -yft[0]])

    return phase_fft


def phase_fft2edge(frq, phase, img_shape, dx=1):
    """

    :param list[float] or np.array frq:
    :param float phase:
    :param tuple or list img_shape:
    :param float dx:

    :return phase_edge:
    """

    xft = tools.get_fft_pos(img_shape[1], dt=dx, centered=True, mode="symmetric")
    yft = tools.get_fft_pos(img_shape[0], dt=dx, centered=True, mode="symmetric")
    phase_edge = xform_phase_translation(frq[0], frq[1], phase, [xft[0], yft[0]])

    return phase_edge


def xform_phase_translation(fx, fy, phase, shifted_center):
    """
    Transform sinusoid phase based on translating coordinate center. If we make the transformation,
    x' = x - cx
    y' = y - cy
    then the phase transforms
    phase' = phase + 2*pi * (fx * cx + fy * cy)

    :param float fx: x-component of frequency
    :param float fy: y-component of frequency
    :param float phase:
    :param list[float] shifted_center: shifted center in initial coordinates, [cx, cy]

    :return phase_shifted:
    """

    cx, cy = shifted_center
    phase_shifted = np.mod(phase + 2*np.pi * (fx * cx + fy * cy), 2*np.pi)
    return phase_shifted


# transform sinusoid parameters under full affine transformation
def xform_sinusoid_params(fx_obj, fy_obj, phi_obj, affine_mat):
    """
    Given a sinusoid function of object space,
    cos[2pi f_x * x_o + 2pi f_y * y_o + phi],
    find the frequency and phase parameters for the corresponding function on image space,
    cos[2pi f_xi * x_i + 2pi f_yi * yi + phi_i]

    :param float fx_obj: x-component of frequency in object space
    :param float fy_obj: y-component of frequency in object space
    :param float phi_obj: phase in object space
    :param np.array affine_mat: affine transformation homogeneous coordinate matrix transforming
     points in object space to image space

    :return fx_img: x-component of frequency in image space
    :return fy_img: y-component of frequency in image space
    :return phi_img: phase in image space
    """
    affine_inv = np.linalg.inv(affine_mat)
    fx_img = fx_obj * affine_inv[0, 0] + fy_obj * affine_inv[1, 0]
    fy_img = fx_obj * affine_inv[0, 1] + fy_obj * affine_inv[1, 1]
    phi_img = np.mod(phi_obj + 2 * np.pi * fx_obj * affine_inv[0, 2] + 2 * np.pi * fy_obj * affine_inv[1, 2], 2 * np.pi)

    return fx_img, fy_img, phi_img


def xform_sinusoid_params_roi(fx, fy, phase, object_size, img_roi, affine_mat,
                              input_origin="fft", output_origin="fft"):
    """
    Transform sinusoid parameter from object space to a region of interest in image space.

    # todo: would it be more appropriate to put this function in sim_reconstruction.py?

    This is an unfortunately complicated function because we have five coordinate systems to worry about
    o: object space coordinates with origin at the corner of the DMD pattern
    o': object space coordinates assumed by fft functions
    i: image space coordinates, with origin at corner of the camera
    r: roi coordinates with origin at the edge of the roi
    r': roi coordinates, with origin near the center of the roi (coordinates for fft)
    The frequencies don't care about the coordinate origin, but the phase does

    :param float fx: x-component of frequency in object space
    :param float fy: y-component of frequency in object space
    :param float phase: phase of pattern in object space coordinates system o or o'.
    :param list[int] object_size: [sy, sx], size of object space, required to define origin of o'
    :param list[int] img_roi: [ystart, yend, xstart, xend], region of interest in image space. Note: this region does not include
    the pixels at yend and xend! In coordinates with integer values the pixel centers, it is the area
    [ystart - 0.5*dy, yend-0.5*dy] x [xstart -0.5*dx, xend - 0.5*dx]
    :param np.array affine_mat: affine transformation matrix, which takes points from o -> i
    :param str input_origin: "fft" if phase is provided in coordinate system o', or "edge" if provided in coordinate sysem o
    :param str output_origin: "fft" if output phase should be in coordinate system r' or "edge" if in coordinate system r

    :return fx_xform: x-component of frequency in coordinate system r'
    :return fy_xform: y-component of frequency in coordinates system r'
    :return phi_xform: phase in coordinates system r or r' (depending on the value of output_origin)
    """

    if input_origin == "fft":
        phase_o = phase_fft2edge([fx, fy], phase, object_size, dx=1)
        # xft = tools.get_fft_pos(object_size[1])
        # yft = tools.get_fft_pos(object_size[0])
        # phase_o = xform_phase_translation(fx, fy, phase, [xft[0], yft[0]])
    elif input_origin == "edge":
        phase_o = phase
    else:
        raise Exception("input origin must be 'fft' or 'edge' but was '%s'" % input_origin)

    # affine transformation, where here we take coordinate origins at the corners
    fx_xform, fy_xform, phase_i = xform_sinusoid_params(fx, fy, phase_o, affine_mat)

    if output_origin == "edge":
        phase_r = xform_phase_translation(fx_xform, fy_xform, phase_i, [img_roi[2], img_roi[0]])
        phase_xform = phase_r
    elif output_origin == "fft":
        # transform so that phase is relative to center of ROI
        ystart, yend, xstart, xend = img_roi

        x_rp = tools.get_fft_pos(xend - xstart, dt=1, centered=True, mode="symmetric")
        y_rp = tools.get_fft_pos(yend - ystart, dt=1, centered=True, mode="symmetric")

        # origin of rp-coordinate system, written in the i-coordinate system
        cx = xstart - x_rp[0]
        cy = ystart - y_rp[0]

        phase_rp = xform_phase_translation(fx_xform, fy_xform, phase_i, [cx, cy])
        phase_xform = phase_rp
    else:
        raise Exception("output_origin must be 'fft' or 'edge' but was '%s'" % output_origin)

    return fx_xform, fy_xform, phase_xform


# deprecate this fn in favor of xform_sinusoid_params_roi()
def get_roi_sinusoid_params(roi, fs, phi, dr=None):
    """
    choosing a region of interest (ROI) amounts to making a coordinate transform

    # todo: probably better to write this more generally for a change of center, rather than narrowly as roi shift

    :param list[int] roi: [r1_start, r1_end, r2_start, r2_end, r3_start, ..., rn_end]
    :param fs: [f1, f2, ..., fn]
    :param float phi:
    :param list[float] dr: [dr1, dr2, ..., drn]
    :return:
    """

    if dr is None:
        dr = [1] * len(fs)

    if isinstance(dr, (int, float)):
        dr = [dr]

    if len(dr) == 1 and len(fs) > 1:
        dr = dr * len(fs)

    phi_roi = phi
    for ii in range(len(fs)):
        phi_roi = phi_roi + 2 * np.pi * fs[ii] * roi[2*ii] * dr[ii]

    return np.mod(phi_roi, 2*np.pi)


# fit affine transformation
def fit_affine_xform_points(from_pts, to_pts):
    """
    Solve for affine transformation T = [[A, b], [0, ..., 0, 1]], satisfying
    to_pts = A * from_pts + b, or
    to_pts_aug = T * from_pts_aug
    Put this in roi_size form where Gaussian elimination is applicable by taking transpose of this,
    from_pts_aug^t * T^t = to_pts_aug^t

    Based on a`function <https://elonen.iki.fi/code/misc-notes/affine-fit/>` written by Jarno Elonen
    <elonen@iki.fi> in 2007 (Placed in Public Domain),
    which was in turn based on the paper "Fitting affine and orthogonal transformations between
    two sets of points" Mathematical Communications 9 27-34 (2004) by Helmuth Späth, available
    `here <https://hrcak.srce.hr/712>`

    :param list[list[float]] from_pts: list of lists [[x1, y1, ...], [x2, y2, ...]] or array, where each column represents roi_size point
    :param to_pts:
    :return soln:

    :return np.array affine_mat:
    """

    # input and output points as arrays
    q = np.asarray(from_pts).transpose()
    p = np.asarray(to_pts).transpose()

    # augmented points
    ones_row = np.ones((1, q.shape[1]))
    q_aug = np.concatenate((q, ones_row), axis=0)
    p_aug = np.concatenate((p, ones_row), axis=0)

    # solve using gaussian elimination. soln = [A, b]
    c = q_aug.dot(p.transpose())
    d = q_aug.dot(q_aug.transpose())
    soln = np.linalg.solve(d, c).transpose()
    # construct affine matrix in unprojected space
    btm_row = np.concatenate((np.zeros((1, soln.shape[1] - 1)), np.ones((1, 1))), axis=1)
    affine_mat = np.concatenate((soln, btm_row), axis=0)

    # todo: equivalently can solve like ... are there any negative consequences to this?
    # affine_mat = np.linalg.solve(q_aug.transpose(), p_aug.transpose()).transpose()

    return soln, affine_mat


def fit_affine_xform_mask(img, mask, init_params=None):
    """
    Fit affine transformation by comparing img with transformed images of mask
    :param img:
    :param mask:
    :param init_params:

    :return np.array pfit:
    """

    if init_params is None:
        init_params = [1, 0, 0, 0, 1, 0]

    raise Exception("Function not finished!")
    # todo: need to binarize img
    # todo: OR maybe better idea: look at cross correlation and maximize this
    xform_fn = lambda p: np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1]])

    # err_fn = lambda p: img.ravel() - affine_xform_mat(mask, xform_fn(p), img.shape, mode='nearest').ravel()
    # fit_dict = optimize.least_squares(err_fn, init_params)
    img_sum = np.sum(img)

    img_coords = np.meshgrid(range(img.shape[1], img.shape[0]))
    min_fn = lambda p: -np.sum(img.ravel() * affine_xform_mat(mask, xform_fn(p), img_coords, mode='interp').ravel()) / \
                       img_sum / np.sum(affine_xform_mat(mask, xform_fn(p), img_coords, mode='interp'))

    fit_dict = optimize.minimize(min_fn, init_params)
    pfit = fit_dict['x']

    return pfit


def fit_barrel_distortion(from_pts, to_pts):
    """

    :param from_pts:
    :param to_pts:
    :return:
    """
    raise Exception("todo: not implemented")


def fit_pattern_peaks(img, centers, centers_init, indices_init, roi_size, img_sd=None, debug=False):
    """
    Fit peak of fluorescence pattern

    :param np.array img:
    :param centers:
    :param centers_init:
    :param indices_init:
    :param roi_size:
    :param img_sd:
    :param bool debug:
    """

    if img_sd is None:
        img_sd = np.ones(img.shape)

    # indices for dmd_centers from mask
    inds_a = np.arange(centers.shape[0])
    inds_b = np.arange(centers.shape[1])
    ibib, iaia = np.meshgrid(inds_b, inds_a)

    # to store fitting results
    fps = np.zeros((centers.shape[0], centers.shape[1], 7))
    chisqs = np.zeros((centers.shape[0], centers.shape[1]))

    # fit initial dmd_centers
    for ii in range(len(centers_init)):
        roi = tools.get_centered_roi(centers_init[ii], [roi_size, roi_size])
        xstart = roi[2]
        xend = roi[3]
        ystart = roi[0]
        yend = roi[1]

        cell = img[ystart:yend, xstart:xend]
        cell_sd = img_sd[ystart:yend, xstart:xend]
        xx, yy = np.meshgrid(range(roi[2], roi[3]), range(roi[0], roi[1]))
        result, fit_fn = tools.fit_gauss(cell, sd=cell_sd, xx=xx, yy=yy)
        pfit = result['fit_params']
        chi_sq = result['chi_squared']

        fps[indices_init[ii][0], indices_init[ii][1], :] = pfit
        chisqs[indices_init[ii][0], indices_init[ii][1]] = chi_sq

        if debug:
            vmin = pfit[5]
            vmax = pfit[5] + pfit[0] * 1.2

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.imshow(cell, vmin=vmin, vmax=vmax, extent=(xx[0, 0], xx[0, -1], yy[-1, 0], yy[0, 0]))
            ax1.scatter(pfit[1], pfit[2], c='r', marker='x')
            ax2.imshow(fit_fn(xx, yy), vmin=vmin, vmax=vmax, extent=(xx[0, 0], xx[0, -1], yy[-1, 0], yy[0, 0]))
            ax3.imshow(cell - fit_fn(xx, yy), extent=(xx[0, 0], xx[0, -1], yy[-1, 0], yy[0, 0]))
            ax4.imshow(img, vmin=vmin, vmax=vmax)
            rec = matplotlib.patches.Rectangle((xstart, ystart), xend - xstart, yend - ystart, color='white', fill=0)
            ax4.add_artist(rec)
            plt.suptitle('(ia, ib) = (%d, %d), chi sq=%0.2f' % (indices_init[ii][0], indices_init[ii][1], chi_sq))

    # set maximum chi-square value that we will consider a 'good fit'
    # chi_sq_max = np.nanmean(chisqs[chisqs != 0]) * 1.5
    # the first three fits must succeed
    chi_sq_max = np.nanmax(chisqs[chisqs != 0]) * 1.5

    # loop over points
    # estimate position by guessing vec_a and vec_b from nearest pairs
    while np.sum(chisqs != 0) < chisqs.size:
        successful_fits = np.logical_and(chisqs > 0, chisqs < chi_sq_max)
        completed_fits = (chisqs != 0)

        # find nearest point to already fitted points, only including succesful fits
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
        # get estimated a vector
        vec_b_guess = fps[amin, b2, 1:3] - fps[amin, b1, 1:3]

        # guess point
        diff_a = ind_tuple[0] - nearest_ind_tuple[0]
        diff_b = ind_tuple[1] - nearest_ind_tuple[1]
        center_guess = fps[nearest_ind_tuple[0], nearest_ind_tuple[1], 1:3] + \
                       vec_a_guess * diff_a + vec_b_guess * diff_b
        xc = int(center_guess[0])
        yc = int(center_guess[1])

        # get roi
        roi = tools.get_centered_roi([yc, xc], [roi_size, roi_size])

        xstart = int(roi[2])
        xend = int(roi[3])
        ystart = int(roi[0])
        yend = int(roi[1])

        # do fitting if end points are reasonable
        if xstart >= 0 and xend < img.shape[1] and ystart >= 0 and yend < img.shape[0]:
            xx, yy = np.meshgrid(range(xstart, xend), range(ystart, yend))
            cell = img[ystart:yend, xstart:xend]
            cell_sd = img_sd[ystart:yend, xstart:xend]

            result, fit_fn = tools.fit_gauss(cell, sd=cell_sd, xx=xx, yy=yy)
            pfit = result['fit_params']
            chi_sq = result['chi_squared']

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
                rec = matplotlib.patches.Rectangle((xstart, ystart), xend - xstart, yend - ystart, color='white',
                                                   fill=0)
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
    _, affine_xform = fit_affine_xform_points(in_pts, out_pts)

    return affine_xform


def plot_affine_summary(img, mask, fps, chisqs, succesful_fits, dmd_centers, affine_xform, options):
    """
    Plot results of DMD affine transformation fitting using results of fit_pattern_peaks and get_affine_xform

    :param np.array img:
    :param np.array mask:
    :param fps:
    :param chisqs:
    :param bool succesful_fits: boolean giving which fits we consider successful
    :param dmd_centers:
    :param affine_xform:
    :param options:

    :return fig: figure handle to summary figure
    """

    xcam = fps[:, :, 1]
    ycam = fps[:, :, 2]
    xcam = xcam[succesful_fits]
    ycam = ycam[succesful_fits]
    amps = fps[:, :, 0]
    bgs = fps[:, :, 5]
    sigma_mean_cam = np.sqrt(fps[:, :, 3] * fps[:, :, 4])
    sigma_asymmetry_cam = (fps[:, :, 3] - fps[:, :, 4]) / (fps[:, :, 3] + fps[:, :, 4]) * 2
    angles = np.mod(fps[:, :, 6], 2*np.pi)

    xdmd = dmd_centers[:, :, 0]
    ydmd = dmd_centers[:, :, 1]
    xdmd = xdmd[succesful_fits]
    ydmd = ydmd[succesful_fits]

    # extract parameters from options
    pixel_correction_factor = options['cam_pix'] / options['dmd_pix']
    expected_mag = options['dmd2cam_mag_expected']

    # get affine paramaters from xform
    affine_params = xform2params(affine_xform)

    # transform DMD points and mask to image space
    homog_coords = np.concatenate((xdmd[None, :], ydmd[None, :], np.ones((1, xdmd.size))), axis=0)
    dmd_coords_xformed = affine_xform.dot(homog_coords)
    xdmd_xform = dmd_coords_xformed[0, :]
    ydmd_xform = dmd_coords_xformed[1, :]

    # residual position error
    residual_dist_err = np.zeros((fps.shape[0], fps.shape[1])) * np.nan
    residual_dist_err[succesful_fits] = np.sqrt((xdmd_xform - xcam)**2 + (ydmd_xform - ycam)**2)

    # get mask transformed to image space
    img_coords = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    mask_xformed = affine_xform_mat(mask, affine_xform, img_coords, mode='nearest')

    # summarize results
    vmin = np.min(bgs[succesful_fits])
    vmax = vmin + np.max(amps[succesful_fits]) * 1.2

    # plot results
    fig = plt.figure(figsize=(16, 12))
    grid = plt.GridSpec(4, 4)

    plt.subplot(grid[0, 0])
    plt.imshow(chisqs)
    plt.title('chi squared values')
    plt.colorbar()

    plt.subplot(grid[0, 1])
    plt.imshow(residual_dist_err)
    plt.title('residual position error')
    plt.colorbar()

    plt.subplot(grid[1, 0])
    plt.imshow(sigma_mean_cam)
    plt.title('sigma (geometric mean)')
    plt.colorbar()

    plt.subplot(grid[1, 1])
    plt.imshow(amps)
    plt.title('amplitudes')
    plt.colorbar()

    plt.subplot(grid[2, 0])
    plt.imshow(sigma_asymmetry_cam)
    plt.title('sigma asymmetry')
    plt.colorbar()

    plt.subplot(grid[2, 1])
    plt.imshow(angles)
    plt.title('rotation angle')
    plt.colorbar()

    plt.subplot(grid[:2, 2:])
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.scatter(xcam, ycam, c='r', marker='x')
    plt.scatter(xdmd_xform, ydmd_xform, c='y', marker='1')
    plt.title('red=fit points, yellow=dmd xformed points')

    plt.subplot(grid[3:, :2])
    plt.imshow(mask)
    plt.title('dmd mask')

    plt.subplot(grid[2:, 2:])
    plt.imshow(mask_xformed)
    plt.title('dmd mask xformed to img space')

    plt.suptitle('theta_x=%.0fdeg, pixel corrected Mag_x=%.2f\ntheta_y=%.0fdeg, Mag_y=%0.2f\n expected mag=%0.2f'
                 % (affine_params[1] * 180 / np.pi, affine_params[0] * pixel_correction_factor,
                    affine_params[4] * 180 / np.pi, affine_params[3] * pixel_correction_factor,
                    expected_mag))

    return fig


# main function for estimating affine transform
def estimate_xform(img, mask, pattern_centers, centers_init, indices_init, options, roi_size=25,
                   export_fname="affine_xform", export_dir=None):
    """
    Estimate affine transformation from DMD space to camera image space from an image.

    :param img: single image to analyze
    :param mask: DMD image that is being imaged
    :param pattern_centers: pattern centers in DMD coordinates
    :param centers_init: [[cy1, cx1], [cy2, cx2], [cy3, cx3]] in image space
    :param indices_init: [[ia1, ib1], [ia2, ib2], [ia3, ib3]]. ia indices are along the y-direction of the DMD pattern,
    starting with 0 which is the topmost when plotted. ib indices are long the x-direction starting with 0 which is the
    leftmost when plotted
    :param options: {'tube_lens_f', 'cam_pix', 'dmd_pix', 'dmd2cam_mag_expected', 'objective_mag', 'na'}. Distances are
    in meters.
    :param roi_size: size of ROI to fit each peak
    :param export_fname: file name (not including extension) to use when saving results
    :param export_dir: directory to save results
    :return data: affine trasnformation data
    :return fig: figure handle to summary data
    """

    # fit points
    fps, chisqs, succesful_fits = fit_pattern_peaks(img, pattern_centers, centers_init, indices_init,
                                                    roi_size, img_sd=None, debug=False)
    # get affine xforms
    affine_xform = get_affine_xform(fps, succesful_fits, pattern_centers)

    # plot data
    data = {'affine_xform': affine_xform}
    fig = plot_affine_summary(img, mask, fps, chisqs, succesful_fits, pattern_centers, affine_xform, options)

    if export_dir is not None:
        fig_fpath = os.path.join(export_dir, export_fname + ".png")
        fig.savefig(fig_fpath)

        fpath = os.path.join(export_dir, export_fname + ".pkl")
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)

    plt.show()

    return data, fig


if __name__ == '__main__':
    pass
