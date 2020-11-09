"""
Tools for simulating diffraction of digital mirror device (DMD)
"""
# from . import analysis_tools as tools
import analysis_tools as tools

import os
import numpy as np
import scipy.optimize
import pickle
import joblib
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle

# main simulation function and important auxiliary functions
def simulate_dmd(pattern, wavelength, gamma_on, gamma_off, dx, dy, wx, wy,
                 tx_in, ty_in, txs_out, tys_out, is_coherent=True):
    """
    Simulate plane wave diffracted from a digital mirror device (DMD). We assume that the body of the device is in the xy
    plane with the negative z- unit vector defining the plane's normal. We suppose the device has rectangular pixels with sides
    parallel to the x- and y-axes. We further suppose a given pixel (centered at (0,0))
    swivels about the vector n = [1, 1, 0]/sqrt(2) by angle gamma, i.e. the direction x-y is the most interesting one.

    We simulate the device for incoming rays along vector
    a = [tan(tx_in), tan(ty_in), 1] / sqrt{tan^2(tx_in) + tan^2(ty_in) + 1}
    tx_in is the angle between the components of a projected into the xz plane and the positive z unit vector
    and outgoing rays along vector
    b = [tan(tx_out), tan(ty_out), -1] / sqrt{tan^2(tx_out) + tan^2(ty_out) + 1}
    tx_out is the angle between the components of b projected into the xz plane and the negative z unit vector.
    Using different conventions for a and b preserves the law of reflection as tx_in = tx_out.

    For more information on the simulation, see doi:10.1101/797670
 
    :param pattern: an NxM array. Dimensions of the DMD are determined from this. As usual, the upper left hand corner
    if this array represents the smallest x- and y- values
    :param gamma_on: wavelength. Can choose any units as long as consistent with
          dx, dy, wx, and wy. 
    :param gamma_on: DMD mirror angle in radians 
    :param gamma_off: 
    :param dx: spacing between DMD pixels in the x-direction. Same units as wavelength. 
    :param dy: spacing between DMD pixels in the y-direction. Same units as wavelength. 
    :param wx: width of mirrors in the x-direction. Must be < dx. 
    :param wy: width of mirrors in the y-direction. Must be < dy. 
    :param tx_in: input angle projected in the xz-plane (radians) 
    :param ty_in: input angle projected in the yz-plane (radians)
    :param txs_out: array of arbitrary size specifying output angles in the xz-plane
           to compute diffraction pattern at. Note the difference in how incoming and outgoing angles are defined.
    :param tys_out: 
    :param is_coherent: boolean. If True, treat light as coherent.
    :return: 
    """

    # axis to rotate mirror about
    # n = [1, 1, 0]/sqrt(2);
    # rotation by angle gamma about unit vector (nx, ny, nz)
    # Rn = @(gamma, nx, ny, nz)...
    #     [nx^2 * (1-cos(gamma)) + cos(gamma),...
    #     nx*ny * (1-cos(gamma)) - nz * sin(gamma),...
    #     nx*nz * (1-cos(gamma)) + ny * sin(gamma);...
    #     nx*ny * (1-cos(gamma)) + nz * sin(gamma),...
    #     ny^2  * (1-cos(gamma)) + cos(gamma),...
    #     ny*nz * (1-cos(gamma)) - nx * sin(gamma);...
    #     nx*nz * (1-cos(gamma)) - ny * sin(gamma),...
    #     ny*nz * (1-cos(gamma)) + nx * sin(gamma),...
    #     nz^2 *  (1-cos(gamma)) + cos(gamma)];

    # check input arguments are sensible
    if not np.all(np.logical_or(pattern == 0, pattern == 1)):
        raise Exception('pattern must be binary. All entries should be 0 or 1.')

    if dx < wx or dy < wy:
        raise Exception('w must be <= d.')

    if txs_out.size != tys_out.size:
        raise Exception('txs_out and tys_out should be the same size')

    # k-vector for wavelength
    k = 2*np.pi/wavelength

    ny, nx = pattern.shape
    mxmx, mymy = np.meshgrid(range(nx), range(ny))

    # difference between incoming and outgoing unit vectors in terms of angles (a-b)
    amb_fn = lambda tx_a, ty_a, tx_b, ty_b: get_unit_vector(tx_a, ty_a, 'in') - get_unit_vector(tx_b, ty_b, 'out')

    # want to do integral
    #   \int ds dt exp[ ik Rn*(s,t,0) \cdot (a-b)]
    # = \int ds dt exp[ ik * (A_+*s + A_-*t)]
    # the result is a product of sinc functions (up to a phase)
    mirror_int_fn = lambda gamma, amb: wx * wy \
                    * sinc_fn(0.5 * k * wx * blaze_condition_fn(gamma, amb, 'plus')) \
                    * sinc_fn(0.5 * k * wy * blaze_condition_fn(gamma, amb, 'minus'))

    # phases for each mirror
    phase_fn = lambda mx, my, amb_x, amb_y: np.exp(1j * k * (dx * mx * amb_x + dy * my * amb_y))

    # full e-field info
    efields = np.zeros(tys_out.shape, dtype=np.complex)
    # info from diffraction pattern (i.e. sum without mirror integral)
    diffraction_efield = np.zeros(efields.shape, dtype=np.complex)
    # info from mirror integral
    sinc_efield_on = np.zeros(efields.shape, dtype=np.complex)
    sinc_efield_off = np.zeros(efields.shape, dtype=np.complex)

    # loop over arbitrary sized array using single index.
    for ii in range(tys_out.size):
        ii_subs = np.unravel_index(ii, tys_out.shape)

        # incoming minus outgoing unit vectors
        amb = amb_fn(tx_in, ty_in, txs_out[ii_subs], tys_out[ii_subs])

        # get envelope functions for ON and OFF states
        sinc_efield_on[ii_subs] = mirror_int_fn(gamma_on, amb)
        sinc_efield_off[ii_subs] = mirror_int_fn(gamma_off, amb)

        # phases for each pixel
        phases = phase_fn(mxmx, mymy, amb[0], amb[1])

        mask_phases = np.zeros((ny, nx), dtype=np.complex)
        mask_phases[pattern == 0] = sinc_efield_off[ii_subs]
        mask_phases[pattern == 1] = sinc_efield_on[ii_subs]
        mask_phases = mask_phases * phases

        # coherent case: add electric fields
        # incoherent case: add intensities
        if is_coherent:
            efields[ii_subs] = np.sum(mask_phases)
            diffraction_efield[ii_subs] = np.sum(phases)
        else:
            efields[ii_subs] = np.sqrt(np.sum(abs(mask_phases)**2))
            diffraction_efield[ii_subs] = np.sqrt(np.sum(abs(phases)**2))

    return efields, sinc_efield_on, sinc_efield_off, diffraction_efield

def blaze_envelope(wavelength, gamma, wx, wy, tx_in, ty_in, tx_out, ty_out):
    """
    Compute normalized blaze envelope function. Envelope function has value 1 where the blaze condition is satisfied.

    :param wavelength: wavelength of light. Units are arbitrary, but must be the same for wavelength, wx, and wy
    :param gamma: in radians
    :param wx: mirror width in x-direction
    :param wy: mirror width in y-direction
    :param tx_in: arbitrary shape
    :param ty_in: same shape as tx_in
    :param tx_out: same shape as tx_in
    :param ty_out: same shape as tx_in

    :return envelope: same shape as tx_in
    """

    k = 2*np.pi / wavelength
    amb = get_unit_vector(tx_in, ty_in, 'in') - get_unit_vector(tx_out, ty_out, 'out')

    envelope = sinc_fn(0.5 * k * wx * blaze_condition_fn(gamma, amb, 'plus')) \
             * sinc_fn(0.5 * k * wy * blaze_condition_fn(gamma, amb, 'minus'))
    return envelope

def blaze_condition_fn(gamma, amb, mode='plus'):
    """
    Return the dimensionsless part of the sinc function argument which determines the Blaze condition, which we refer
    to as A_+ and A_-

    E = (diffraction from different mirrors) x w**2 * sinc(0.5 * k * w * A_+) * sinc(0.5 * k * w * A_-)

    A_\pm = 0.5*(1 \pm cos(gamma)) * (a-b)_x + 0.5*(1 \mp cos(gamma)) * (a-b)_y \mp sin(gamma)/sqrt(2) * (a-b)_z

    :param gamma: angle micro-mirror normal makes with device normal
    :param amb: incoming unit vector - outgoing unit vector, [vx, vy, vz]. Will also accept a matrix of shape
    n0 x n1 x ... x 3
    :param mode: 'plus' or 'minus'
    :return A:
    """
    if mode == 'plus':
        A = 0.5 * (1 + np.cos(gamma)) * amb[..., 0] + \
            0.5 * (1 - np.cos(gamma)) * amb[..., 1] - \
            1 / np.sqrt(2) * np.sin(gamma) * amb[..., 2]
    elif mode == 'minus':
        A = 0.5 * (1 - np.cos(gamma)) * amb[..., 0] + \
            0.5 * (1 + np.cos(gamma)) * amb[..., 1] + \
            1 / np.sqrt(2) * np.sin(gamma) * amb[..., 2]
    else:
        raise Exception("mode must be 'plus' or 'minus', but was '%s'" % mode)
    return A

def sinc_fn(x):
    """
    Unnormalized sinc function, sinc(x) = sin(x) / x

    :param x:
    :return sinc(x):
    """
    # x = np.asarray(x)
    x = np.atleast_1d(x)
    with np.errstate(divide='ignore'):
        y = np.asarray(np.sin(x) / x)
    y[x == 0] = 1
    return y

# convert between xyz and 123 coords
def xyz2mirror(vx, vy, vz, gamma):
    v1 = np.cos(gamma) / np.sqrt(2) * (vx - vy) - np.sin(gamma) * vz
    v2 = 1 / np.sqrt(2) * (vx + vy)
    v3 = np.sin(gamma) / np.sqrt(2) * (vx - vy) + np.cos(gamma) * vz
    return v1, v2, v3

def mirror2xyz(v1, v2, v3, gamma):
    vx = np.cos(gamma) / np.sqrt(2) * v1 + 1 / np.sqrt(2) * v2 + np.sin(gamma) / np.sqrt(2) * v3
    vy = -np.cos(gamma) / np.sqrt(2) * v1 + 1 / np.sqrt(2) * v2 - np.sin(gamma) / np.sqrt(2) * v3
    vz = -np.sin(gamma) * v1 + np.cos(gamma) * v3
    return vx, vy, vz

# functions for converting between different angular representations
def angle2xy(tp, tm):
    """
    Convert angle projections along the x and y axis to angle projections along the p=(x+y)/sqrt(2) and m=(x-y)/sqrt(2)
    axis. The mode argument is required

    :param tp:
    :param tm:
    :param mode:
    :return:
    """

    tx = np.arctan((np.tan(tp) + np.tan(tm)) / np.sqrt(2))
    ty = np.arctan((np.tan(tp) - np.tan(tm)) / np.sqrt(2))

    return tx, ty

def angle2pm(tx, ty):
    """
    Convert angle projections along the the p=(x+y)/sqrt(2) and m=(x-y)/sqrt(2) to x and y axes.

    :param tx:
    :param ty:
    :param mode:
    :return:
    """

    tm = np.arctan((np.tan(tx) - np.tan(ty)) / np.sqrt(2))
    tp = np.arctan((np.tan(tx) + np.tan(ty)) / np.sqrt(2))

    return tp, tm

def txty2polar(tx, ty):
    """
    Convert between v = [tan(tx), tan(ty), 1] / sqrt(tan(tx)**2 + tan(ty)**2 + 1)
    and v = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]

    these are related by
    tan(tx) = tan(theta) * cos(phi)
    tan(ty) = tan(theta) * sin(phi)

    :param tx: in radians
    :param ty: in radians
    :return theta: in radians
    :return phi: in radians
    """
    if tx == 0 and ty == 0:
        # phi indeterminate in this case
        theta = 0
        phi = 0
    elif tx == 0 and ty != 0:
        phi1 = np.pi/2
        phi2 = 3*np.pi/2

        theta1 = np.arctan(np.tan(ty) / np.sin(phi1))
        theta2 = np.arctan(np.tan(ty) / np.sin(phi2))

        # only one of these will be in the interval [0, pi/2)
        if theta1 >= 0 and theta1 <= np.pi/2:
            phi = phi1
            theta = theta1
        else:
            phi = phi2
            theta = theta2

    elif tx != 0 and ty == 0:
        phi1 = 0
        phi2 = np.pi

        theta1 = np.artcan(np.tan(tx) / np.cos(phi1))
        theta2 = np.arctan(np.tan(tx) / np.cos(phi2))

        if theta1 >= 0 and theta1 <= np.pi/2:
            phi = phi1
            theta = theta1
        else:
            phi = phi2
            theta = theta2

    else:
        phi = np.arctan(np.tan(ty) / np.tan(tx))
        theta = np.arctan(np.tan(tx) / np.cos(phi))

    # check everything worked
    assert np.abs(np.tan(tx) - np.tan(theta) * np.cos(phi)) < 1e-10
    assert np.abs(np.tan(ty) - np.tan(theta) * np.sin(phi)) < 1e-10

    return theta, phi

def polar2txty(theta, phi):
    """
    Convert between v = [tan(tx), tan(ty), 1] / sqrt(tan(tx)**2 + tan(ty)**2 + 1)
    and v = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]

    these are related by
    tan(tx) = tan(theta) * cos(phi)
    tan(ty) = tan(theta) * sin(phi)

    :param theta: in radians
    :param phi: in radians
    :return tx: in radians
    :return ty: in radians
    """

    # arctan returns angles in range [-pi/2, pi/2], which is what we want
    tx = np.arctan(np.tan(theta) * np.cos(phi))
    ty = np.arctan(np.tan(theta) * np.sin(phi))
    return tx, ty

def uvector2txty(vx, vy, vz):
    """
    Convert unit vector from components to theta_x, theta_y representation. Inverse function for get_unit_vector()
    :param vx:
    :param vy:
    :param vz:
    :return:
    """
    norm_factor = np.abs(1 / vz)
    tx = np.arctan(vx * norm_factor)
    ty = np.arctan(vy * norm_factor)

    return tx, ty

def get_unit_vector(tx, ty, mode='in'):
    """
    Get incoming or outgoing unit vector of light propogation parametrized by angles tx and ty

    Let a represent an incoming vector, and b and outgoing one. Then we paraemtrize these by
    a = az * [tan(tx_a), tan(ty_a), 1]
    b = |bz| * [tan(tb_x), tan(tb_y), -1]
    choosing negative z component for outgoing vectors is effectively taking a different
    conventions for the angle between b and the z axis (compared with a and
    the z-axis). We do this so that e.g. the law of reflection would give
    theta_a = theta_b, instead of theta_a = -theta_b, which would hold if we
    defined everything symmetrically.

    :param tx: arbitrary size
    :param ty: same size as tx
    :param mode: "in" or "out" depending on whether representing a vector pointing in the positive or negative z-direction

    :return uvec: unit vectors, array of size tx.size x 3
    """
    tx = np.atleast_1d(tx)
    ty = np.atleast_1d(ty)
    norm = np.sqrt(np.tan(tx)**2 + np.tan(ty)**2 + 1)
    if mode == 'in':
        ux = np.tan(tx)
        uy = np.tan(ty)
        uz = np.ones(tx.shape)
    elif mode == 'out':
        ux = np.tan(tx)
        uy = np.tan(ty)
        uz = -np.ones(tx.shape)
    else:
        raise Exception("mode must be 'in' or 'out', but was '%s'" % mode)

    uvec = np.concatenate((ux[..., None], uy[..., None], uz[..., None]), axis=-1) / norm[..., None]

    return uvec

# todo: clean up these functions. probably several can combine
# # utility functions for solving blaze + diffraction conditions
def solve_max_diffraction_order(wavelength, d, gamma):
    """
    Find the maximum and minimum diffraction orders consistent with given parameters and the blaze condition

    :param wavelength: wavelength of light
    :param d: mirror pitch (in same units as wavelength)
    :param gamma: mirror angle
    :return nmax: maximum index of diffraction order
    :return nmin: minimum index of diffraction order
    """

    # # solution for maximum order
    # theta_a_opt = np.arctan( (np.cos(2*gamma) - 1) / np.sin(2*gamma))
    #
    # # this should = sqrt(2) * lambda / d * n when diffraction condition is satisfied
    # f = lambda t: np.sin(t) - np.sin(t - 2*gamma)
    #
    # # must also check end points for possible extrema
    # # ts in range [-np.pi/2 np.pi/2]
    # if gamma > 0:
    #     fopts = [f(-np.pi/2 + 2*gamma), f(np.pi/2), f(theta_a_opt)]
    # elif gamma <= 0:
    #     fopts = [f(-np.pi / 2), f(np.pi / 2 + 2*gamma), f(theta_a_opt)]
    #
    # # find actually extrema
    # fmin = np.min(fopts)
    # fmax = np.max(fopts)
    #
    # nmax = np.floor(fmax * d / np.sqrt(2) / wavelength)
    # nmin = np.ceil(fmin * d / np.sqrt(2) / wavelength)

    if gamma > 0:
        nmax = int(np.floor(d / wavelength * np.sqrt(2) * np.sin(gamma)))
        nmin = 1
    if gamma <= 0:
        nmax = -1
        nmin = int(np.ceil(d / wavelength * np.sqrt(2) * np.sin(gamma)))

    return nmax, nmin

def solve_1color_1d(wavelength, d, gamma, n):
    """
    # todo: superseded by solve_combined_condition()
    Solve for theta_in satisfying diffraction condition and blaze angle for a given diffraction order (if possible).

    (1) theta_in - theta_out = 2*gamma
    (2) sin(theta_in) - sin(theta_out) = sqrt(2) * wavelength / d * n
    Which we can reduce to
    sin(theta_in) [1 - cos(2*gamma)] - cos(theta_in) * sin(2*gamma) = sqrt(2) * wavelength / d * n

    Solve this by first finding the maximum of sin(theta_in) - sin(theta_in - 2*gamma), then we can have 0, 1, or 2
    solutions. One can be between [tin_min, tin_opt], and the other can be between [tin_opt, tin_max]

    :param wavelength: wavelength of light
    :param d: mirror pitch (in same units as wavelength)
    :param gamma: angle mirror normal makes with DMD body normal
    :param n: diffraction order index
    :return tins: list of solutions for incoming angle
    :return touts: list of solutions for outgoing angle
    """
    # point where derivative of sin(tin) - sin(tin - 2*gamma) = 0
    tin_opt = np.arctan((1 - np.cos(2 * gamma)) / np.sin(2 * gamma))
    # tin and tout must be in range [-np.pi/2, np.pi/2]

    tin_min = np.max([-np.pi/2, -np.pi/2 + 2*gamma])
    tin_max = np.min([np.pi/2, np.pi/2 + 2*gamma])
    # can have on root for smaller values and one root for larger values

    tout = lambda tin: tin - 2 * gamma
    #fn = lambda tin: np.abs(np.sin(tin) - np.sin(tout(tin)) - np.sqrt(2) * wavelength / d * n)
    fn = lambda tin: np.sin(tin) - np.sin(tout(tin)) - np.sqrt(2) * wavelength / d * n

    try:
        r1 = scipy.optimize.root_scalar(fn, bracket=(tin_opt, tin_max), xtol=1e-5)
        root1 = r1.root
    except ValueError:
        root1 = None

    try:
        r2 = scipy.optimize.root_scalar(fn, bracket=(tin_min, tin_opt), xtol=1e-5)
        root2 = r2.root
    except ValueError:
        root2 = None

    # solutions
    tins = [r for r in [root1, root2] if r is not None]
    touts = [ti - 2 * gamma for ti in tins]

    return tins, touts

def solve_diffraction_input_1d(theta_out, wavelength, d, order):
    """
    Find input angle corresponding to given diffraction output angle, for 1D case.

    Helper function which wraps solve_diffraction_input()

    :param theta_out: desired output angle (in radians)
    :param wavelength: wavelength of light
    :param d: mirror pitch (in same units as wavelength)
    :param order: index of diffraction order
    :return theta_in: input angle
    """
    # theta_in = np.arcsin(np.sin(theta_out) + np.sqrt(2) * wavelength / d * order)
    theta_out = np.atleast_1d(theta_out)
    tx_out, ty_out = angle2xy(np.zeros(theta_out.shape), theta_out)
    tx_in, ty_in = solve_diffraction_input(tx_out, ty_out, d, d, wavelength, (order, -order))
    _, theta_in = angle2pm(tx_in, ty_in)

    return theta_in

def solve_blaze(tp_in, tm_in, gamma):
    """
    Find the blaze condition for arbitrary input angle.

    :param tp_in: input angle in x+y plane in radians
    :param tm_in: input angle in x-y plane in radians
    :param gamma: DMD mirror angle in radians
    :return tp_out: output angles
    :return tm_out:
    """

    tx_in, ty_in = angle2xy(tp_in, tm_in)
    avec = get_unit_vector(tx_in, ty_in, "in")
    # convert to convenient coordinates and apply blaze
    a1, a2, a3 = xyz2mirror(avec[..., 0], avec[..., 1], avec[..., 2], gamma)
    bx, by, bz = mirror2xyz(a1, a2, -a3, gamma)

    # convert back to angles
    tx_out, ty_out = uvector2txty(bx, by, bz)
    tp_out, tm_out = angle2pm(tx_out, ty_out)

    return tp_out, tm_out

def solve_diffraction_input(tx_out, ty_out, dx, dy, wavelength, order):
    """
    Solve for diffraction input angle, given output angle

    :param tx_out:
    :param ty_out:
    :param dx:
    :param dy:
    :param wavelength:
    :param order:
    :return:
    """

    bvec = get_unit_vector(np.atleast_1d(tx_out), np.atleast_1d(ty_out), "out")
    ax = bvec[..., 0] + wavelength / dx * order[0]
    ay = bvec[..., 1] + wavelength / dy * order[1]
    az = np.sqrt(1 - ax**2 - ay**2)

    tx_in, ty_in = uvector2txty(ax, ay, az)

    return tx_in, ty_in

def solve_diffraction_output(tx_in, ty_in, dx, dy, wavelength, order):
    """
    Solve for diffraction output angle, given input angle

    :param tx_out:
    :param ty_out:
    :param dx:
    :param dy:
    :param wavelength:
    :param order: (nx, ny)
    :return:
    """

    avec = get_unit_vector(np.atleast_1d(tx_in), np.atleast_1d(ty_in), "in")
    bx = avec[..., 0] - wavelength / dx * order[0]
    by = avec[..., 1] - wavelength / dy * order[1]
    bz = -np.sqrt(1 - bx**2 - by**2)

    tx_out, ty_out = uvector2txty(bx, by, bz)

    return tx_out, ty_out

def solve_combined_condition(d, gamma, wavelength, order):
    """
    Return functions for the simultaneous blaze/diffraction condition solution as a function of a1

    :param d:
    :param gamma:
    :param wavelength:
    :param order:
    :return a_fn:
    :return b_fn:
    :return a1_bounds:

    """
    a3 = 1 / np.sqrt(2) / np.sin(gamma) * wavelength / d * order
    a1_bounds = (-np.sqrt(1 - a3**2), np.sqrt(1 - a3**2))
    # return positive solution
    def a2_positive_fn(a1): return np.sqrt(1 - a1**2 - a3**2)

    def ax_fn(a1, positive=True):
        a2 = a2_positive_fn(a1)
        if not positive:
            a2 = -a2
        return np.cos(gamma) / np.sqrt(2) * a1 + 1 / np.sqrt(2) * a2 + np.sin(gamma) / np.sqrt(2) * a3

    def ay_fn(a1, positive=True):
        a2 = a2_positive_fn(a1)
        if not positive:
            a2 = -a2
        return -np.cos(gamma) / np.sqrt(2) * a1 + 1 / np.sqrt(2) * a2 - np.sin(gamma) / np.sqrt(2) * a3

    def az_fn(a1): return -np.sin(gamma) * a1 + np.cos(gamma) * a3

    # b functions
    def bx_fn(a1, positive=True):
        a2 = a2_positive_fn(a1)
        if not positive:
            a2 = -a2
        return np.cos(gamma) / np.sqrt(2) * a1 + 1 / np.sqrt(2) * a2 - np.sin(gamma) / np.sqrt(2) * a3

    def by_fn(a1, positive=True):
        a2 = a2_positive_fn(a1)
        if not positive:
            a2 = -a2
        return -np.cos(gamma) / np.sqrt(2) * a1 + 1 / np.sqrt(2) * a2 + np.sin(gamma) / np.sqrt(2) * a3

    def bz_fn(a1):
        return -np.sin(gamma) * a1 - np.cos(gamma) * a3

    def a_fn(a1, positive=True): return(ax_fn(a1, positive), ay_fn(a1, positive), az_fn(a1))
    def b_fn(a1, positive=True): return(bx_fn(a1, positive), by_fn(a1, positive), bz_fn(a1))

    return a_fn, b_fn, a1_bounds

def solve_combined_onoff(d, gamma_on, wavelength_on, n_on, wavelength_off, n_off):
    """
    Solve overlap for two wavelengths, one incident on the ``on'' mirrors and the other on the
     ``off'' mirrors

    :param d: mirror pitch
    :param gamma_on: mirror angle in ON state in radians. Assume that gamma_off = -gamma_on
    :param wavelength_on: wavelength of light incident on ON mirrors. Must be in same units as d
    :param n_on: diffraction order for ON mirrors
    :param wavelength_off: wavelength of light incident on OFF mirrors. Must be in same units as d
    :param n_off: diffraction order for OFF mirrors

    :return b_vecs: output unit vectors. Two solution vectors, size 2 x 3
    :return a_vecs_on: input unit vectors for ON mirrors
    :return b_vecs_on: input unit vectors for OFF mirrors
    """

    b3_on = -1 / np.sqrt(2) / np.sin(gamma_on) * wavelength_on / d * n_on
    b3_off = 1 / np.sqrt(2) / np.sin(gamma_on) * wavelength_off / d * n_off

    # equate b_on and b_off, and solve for bz, bx, by
    # (1) b3_on + b3_off = 2 * cos(gamma) * bz
    # (2) b3_on - b3_off = np.sqrt(2) * np.sin(gamma) * (bx - by)
    bz = 0.5 / np.cos(gamma_on) * (b3_on + b3_off)

    # quadratic equation for bx from (2)
    c1 = 1
    c2 = -(b3_on - b3_off) / np.sqrt(2) / np.sin(gamma_on)
    c3 = 0.5 * (bz**2 + (b3_on - b3_off)**2 / 2 / np.sin(gamma_on)**2 - 1)

    bxs = np.array([0.5 * (-c2 + np.sqrt(c2**2 - 4 * c3)) / c1,
                    0.5 * (-c2 - np.sqrt(c2**2 - 4 * c3)) / c1])

    # apply eq. (2) again to get by (since lost information when we squared it to get quadratic eqn)
    bys = bxs - (b3_on - b3_off) / np.sqrt(2) / np.sin(gamma_on)

    # assemble b-vector
    b_vecs = np.array([[bxs[0], bys[0], bz], [bxs[1], bys[1], bz]])

    for ii in range(b_vecs.shape[0]):
        if np.any(np.isnan(b_vecs[ii])):
            b_vecs[ii, :] = np.nan

    # get input unit vectors
    a_vecs_on = np.zeros(b_vecs.shape)
    a_vecs_off = np.zeros(b_vecs.shape)
    for ii in range(b_vecs.shape[0]):
        b1_on, b2_on, b3_on = xyz2mirror(b_vecs[ii, 0], b_vecs[ii, 1], b_vecs[ii, 2], gamma_on)
        a1_on = b1_on
        a2_on = b2_on
        a3_on = -b3_on
        a_vecs_on[ii] = mirror2xyz(a1_on, a2_on, a3_on, gamma_on)

        b1_off, b2_off, b3_off = xyz2mirror(b_vecs[ii, 0], b_vecs[ii, 1], b_vecs[ii, 2], -gamma_on)
        a1_off = b1_off
        a2_off = b2_off
        a3_off = -b3_off
        a_vecs_off[ii] = mirror2xyz(a1_off, a2_off, a3_off, -gamma_on)

    return b_vecs, a_vecs_on, a_vecs_off

# 1D simulation in x-y plane (i.e. theta_x = -theta_y)
def plot_graphical_soln1d(d, gamma, wavelengths):
    """
    Plot graphical solution to Blaze condition + diffraction condition for 1D case

    :param d: mirror pitch
    :param gamma:
    :param wavelengths:
    :return fig: figure handle of resulting plot
    """

    if isinstance(wavelengths, float):
        wavelengths = [wavelengths]

    nmax = 5
    tmin = np.max([-np.pi/2, -np.pi/2 + 2 * gamma])
    tmax = np.min([np.pi/2 , np.pi/2 + 2 *gamma])
    ts = np.linspace(tmin, tmax, 1000)

    cmap = matplotlib.cm.get_cmap('Spectral')
    phs = []

    fig = plt.figure()
    pht = plt.plot(ts * 180/np.pi, np.sin(ts) - np.sin(ts - 2*gamma))
    phs.append(pht[0])
    for ii, wvl in enumerate(wavelengths):
        for n in np.arange(-nmax, nmax, 1):
            pht = plt.plot(np.array([ts.min(), ts.max()]) * 180/np.pi, [np.sqrt(2) * wvl / d * n, np.sqrt(2) * wvl / d * n],
                     color=cmap(ii / len(wavelengths)))
            if n == -nmax:
                phs.append(pht[0])

    plt.xlabel('tin (degrees)')
    plt.legend(phs, ['blaze condition'] + wavelengths)

    return fig

def simulate_1d(pattern, wavelengths, gamma_on, gamma_off, dx, dy, wx, wy,
                t45_ins, t45_out_offsets=None):
    """
    Simulate various colors of light incident on a DMD, assuming the DMD is oriented so that the mirrors swivel in
    the same plane the incident light travels in and that this plane makes a 45 degree angle with the principle axes
    of the DMD. For more detailed discussion DMD parameters see the function simulate_dmd()

    :param pattern: binary pattern of arbitrary size
    :param wavelengths: list of wavelengths to compute
    :param gamma_on: mirror angle in ON position, relative to the DMD normal
    :param gamma_off:
    :param dx: spacing between DMD pixels in the x-direction. Same units as wavelength.
    :param dy: spacing between DMD pixels in the y-direction. Same units as wavelength.
    :param wx: width of mirrors in the x-direction. Must be < dx.
    :param wy: width of mirrors in the y-direction. Must be < dy.
    :param t45_ins: input angles in the plane of incidence
    :return:
    """

    if isinstance(t45_ins, (float, int)):
        t45_ins = np.array([t45_ins])

    if t45_out_offsets is None:
        t45_out_offsets = np.linspace(-45, 45, 2400) * np.pi / 180

    if isinstance(wavelengths, float):
        wavelengths = [wavelengths]
    n_wavelens = len(wavelengths)

    # input angles
    tx_ins, ty_ins = angle2xy(0, t45_ins)

    # blaze condition
    # t45s_blaze_on = solve_blaze_condition_1d(t45_ins, gamma_on)
    # t45s_blaze_off = solve_blaze_condition_1d(t45_ins, gamma_off)
    _, t45s_blaze_on = solve_blaze(np.zeros(t45_ins.shape), t45_ins, gamma_on)
    _, t45s_blaze_off = solve_blaze(np.zeros(t45_ins.shape), t45_ins, gamma_off)

    # expected diffraction zero peak
    _, t45s_diff_zero = angle2pm(tx_ins, ty_ins)

    # variables to store simulation output data
    efields = np.zeros((len(t45_ins), len(t45_out_offsets), n_wavelens), dtype=np.complex)
    sinc_efield_on = np.zeros(efields.shape, dtype=np.complex)
    sinc_efield_off = np.zeros(efields.shape, dtype=np.complex)
    diffraction_efield = np.zeros(efields.shape, dtype=np.complex)
    t45s_out = np.zeros((len(t45_ins), len(t45_out_offsets)))
    txs_out = np.zeros((len(t45_ins), len(t45_out_offsets)))
    tys_out = np.zeros((len(t45_ins), len(t45_out_offsets)))
    # diffraction order predictions
    ndiff_orders = 10
    nxs = np.array(range(-ndiff_orders, ndiff_orders + 1))
    nys = -nxs
    diff_t45_out = np.zeros((len(t45_ins), n_wavelens, 2 * ndiff_orders + 1))
    # diff_n = np.zeros((2 * ndiff_orders + 1))

    # loop over input angles
    for kk in range(len(t45_ins)):
        # output angles track input angle
        t45s_out[kk] = t45s_blaze_on[kk] + t45_out_offsets
        txs_out[kk], tys_out[kk] = angle2xy(0, t45s_out[kk])

        ### do simulation
        for ii in range(n_wavelens):
            efields[kk, :, ii], sinc_efield_on[kk, :, ii], sinc_efield_off[kk, :, ii], diffraction_efield[kk, :, ii] \
             = simulate_dmd(pattern, wavelengths[ii], gamma_on, gamma_off, dx, dy, wx, wy,
                            tx_ins[kk], ty_ins[kk], txs_out[kk], tys_out[kk], is_coherent=True)

            # get diffraction orders. Orders we want are along the antidiagonal
            # diff_tx_out, diff_ty_out, diff_nx, diff_ny = \
            #     solve_diffraction_output(tx_ins[kk], ty_ins[kk], dx, dy, wavelengths[ii], ndiff_orders)
            diff_tx_out = np.zeros(len(nxs))
            diff_ty_out = np.zeros(len(nxs))
            for aa in range(len(nxs)):
                diff_tx_out[aa], diff_ty_out[aa] = solve_diffraction_output(tx_ins[kk], ty_ins[kk], dx, dy,
                                                                            wavelengths[ii], (nxs[aa], nys[aa]))
            # convert back to 1D angle
            _, diff_t45_out[kk, ii] = angle2pm(diff_tx_out, diff_ty_out)

        # intensity = np.abs(efields[kk])**2
        # sinc_int_on = np.abs(sinc_efield_on[kk])**2
        # sinc_int_off = np.abs(sinc_efield_off[kk])**2
        # diffraction_int = np.abs(diffraction_efield[kk])**2

    data = {'pattern': pattern, 'wavelengths': wavelengths,
            'gamma_on': gamma_on, 'gamma_off': gamma_off,
            'dx': dx, 'dy': dy, 'wx': wx, 'wy': wy,
            't45_ins': t45_ins, 'tx_ins': tx_ins, 'ty_ins': ty_ins,
            't45s_blaze_on': t45s_blaze_on, 't45s_blaze_off': t45s_blaze_off,
            't45s_out': t45s_out, 'txs_out': txs_out, 'tys_out': tys_out,
            'diff_t45_out': diff_t45_out, 'diff_nxs': nxs, 'diff_nys': nys,
            'efields': efields, 'sinc_efield_on': sinc_efield_on,
            'sinc_efield_off': sinc_efield_off, 'diffraction_efield': diffraction_efield}

    return data

def plot_1d_sim(data, colors=None, plot_log=False, saving=False, save_dir='dmd_simulation', figsize=(18, 14)):
    """
    :param data: output from
    :param colors:
    :param plot_log: boolean
    :param plot_each_angle: boolean
    :param saving: boolean
    :param save_dir: directory to save data and figure results in
    :param saving:
    :param save_dir:
    :param figsize:
    :return:
    """

    # save data
    if saving:
        # unique path
        save_dir = tools.get_unique_name(save_dir, mode='dir')
        # unique file name
        fname = os.path.join(save_dir, 'simulation_data.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

    # unpack data
    # todo: can probably do this using **dict somehow
    pattern = data['pattern']
    wavelengths = data['wavelengths']
    n_wavelens = len(wavelengths)
    gamma_on = data['gamma_on']
    gamma_off = data['gamma_off']
    dx = data['dx']
    dy = data['dy']
    wx = data['wx']
    wy = data['wy']
    t45_ins = data['t45_ins']
    tx_ins = data['tx_ins']
    ty_ins = data['ty_ins']
    t45s_blaze_on = data['t45s_blaze_on']
    t45s_blaze_off = data['t45s_blaze_off']
    t45s_out = data['t45s_out']
    efields = data['efields']
    sinc_efield_on = data['sinc_efield_on']
    sinc_efield_off = data['sinc_efield_off']
    diffraction_efield = data['diffraction_efield']
    diff_t45_out = data['diff_t45_out']
    diff_n = data['diff_nxs']
    iz = int((len(diff_n) - 1) / 2)

    # get colors if not provided
    if colors is None:
        cmap = matplotlib.cm.get_cmap('jet')
        colors = [cmap(ii / (n_wavelens - 1)) for ii in range(n_wavelens)]

    #decide how to scale plot
    if plot_log:
        scale_fn = lambda I: np.log10(I)
    else:
        scale_fn = lambda I: I

    figs = []
    fig_names = []

    for kk in range(len(t45_ins)):
        figh = plt.figure(figsize=figsize)
        grid = plt.GridSpec(2, 2, hspace=0.5)

        # title
        param_str = 'spacing = %0.2fum, w=%0.2fum, theta in = (%0.2f, %0.2f)deg = %0.2f deg (x-y) ' \
                    'gamma (on,off)=(%.1f, %.1f) deg\n theta blaze (on,off)=(%.2f, %.2f) deg in x-y dir' % \
                    (dx * 1e6, wx * 1e6,
                     tx_ins[kk] * 180 / np.pi, ty_ins[kk] * 180 / np.pi, t45_ins[kk] * 180 / np.pi,
                     gamma_on * 180 / np.pi, gamma_off * 180 / np.pi,
                     t45s_blaze_on[kk] * 180 / np.pi, t45s_blaze_off[kk] * 180 / np.pi)

        plt.suptitle(param_str)

        # legend
        leg = [str(l * 1e9) for l in wavelengths] + ['blaze on', 'blaze off']

        # ######################################
        # plot diffracted output field
        # ######################################
        ax = plt.subplot(grid[0, 0])
        phs = []

        for ii in range(n_wavelens):
            intensity = np.abs(efields[kk, :, ii])**2
            intensity_sinc_on = np.abs(sinc_efield_on[kk, :, ii]) ** 2

            im = np.argmax(np.abs(intensity))
            norm = intensity[im] / (intensity_sinc_on[im] / wx**2 / wy**2)

            ph = plt.plot(t45s_out[kk] * 180 / np.pi, scale_fn(intensity / norm), color=colors[ii])
            phs.append(ph[0])
            plt.plot(t45s_out[kk] * 180 / np.pi, scale_fn(intensity_sinc_on / (wx*wy)**2), color=colors[ii], ls=':')
            plt.plot(t45s_out[kk] * 180 / np.pi, scale_fn(np.abs(sinc_efield_off[kk, :, ii]) ** 2 / (wx*wy)**2), color=colors[ii], ls='--')

        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        ph = plt.plot([t45s_blaze_on[kk] * 180 / np.pi, t45s_blaze_on[kk] * 180 / np.pi], ylim, 'k:')
        phs.append(ph[0])
        ph = plt.plot([t45s_blaze_off[kk] * 180 / np.pi, t45s_blaze_off[kk] * 180 / np.pi], ylim, 'k--')
        phs.append(ph[0])

        for ii in range(n_wavelens):
            plt.plot(np.array([diff_t45_out[kk, ii], diff_t45_out[kk, ii]]) * 180 / np.pi, ylim, color=colors[ii], ls='-')
        plt.plot([diff_t45_out[kk, ii, iz] * 180 / np.pi, diff_t45_out[kk, ii, iz] * 180 / np.pi], ylim, 'm')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.xlim([t45s_blaze_on[kk] * 180 / np.pi - 7.5, t45s_blaze_on[kk] * 180 / np.pi + 7.5])
        plt.xlabel('theta 45 (deg)')
        plt.ylabel('intensity (arb)')
        plt.title('diffraction pattern')

        # ###########################
        # plot sinc functions and wider angular range
        # ###########################
        ax = plt.subplot(grid[0, 1])

        phs = []
        for ii in range(n_wavelens):
            ph = plt.plot(t45s_out[kk] * 180 / np.pi, scale_fn(np.abs(sinc_efield_on[kk, :, ii]/ wx / wy)**2), color=colors[ii], ls=':')
            plt.plot(t45s_out[kk] * 180 / np.pi, scale_fn(np.abs(sinc_efield_off[kk, :, ii] / wx / wy)**2), color=colors[ii], ls='--')
            phs.append(ph[0])

        # get xlim, ylim, set back to these at the end
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        # plot expected blaze conditions
        ph = plt.plot([t45s_blaze_on[kk] * 180 / np.pi, t45s_blaze_on[kk] * 180 / np.pi], ylim, 'k:')
        phs.append(ph[0])
        ph = plt.plot([t45s_blaze_off[kk] * 180 / np.pi, t45s_blaze_off[kk] * 180 / np.pi], ylim, 'k--')
        phs.append(ph[0])

        # plot expected diffraction conditions
        for ii in range(n_wavelens):
            plt.plot(np.array([diff_t45_out[kk, ii], diff_t45_out[kk, ii]]) * 180 / np.pi, ylim, color=colors[ii], ls='-')
        plt.plot([diff_t45_out[kk, ii, iz] * 180 / np.pi, diff_t45_out[kk, ii, iz] * 180 / np.pi], ylim, 'm')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.legend(phs, leg)
        plt.xlabel('theta 45 (deg)')
        plt.ylabel('intensity (arb)')
        plt.title('sinc envelopes')

        # show pattern
        plt.subplot(grid[1, 0])
        plt.imshow(pattern)

        plt.title('DMD pattern')
        plt.xlabel('mx')
        plt.ylabel('my')

        fname = 'dmd_sim_theta_in=%0.3fdeg' % (t45_ins[kk] * 180 / np.pi)
        fig_names.append(fname)
        figs.append(figh)

        if saving:
            fname = os.path.join(save_dir, fname + '.png')
            figh.savefig(fname)
            plt.close(figh)

    return figs, fig_names

# 2D simulation
def simulate_2d(pattern, wavelengths, gamma_on, gamma_off, dx, dy, wx, wy, tx_in, ty_in, tout_offsets=None):
    """
    Simulate light incident on a DMD to determine output diffraction pattern. See simulate_dmd() for more information.

    The output angles are simulated in parallel using joblib, but there is no parallelization for input angles/wavelengths.
    Generally one wants to simulate many output angles but only a few input angles/wavelengths.

    :param pattern:
    :param wavelengths:
    :param colors:
    :param gamma_on:
    :param gamma_off:
    :param dx:
    :param dy:
    :param wx:
    :param wy:
    :param tx_in:
    :param ty_in:
    :param tout_offsets: offsets from the blaze condition to solve problem
    :return:
    """

    if tout_offsets is None:
        tout_offsets = np.linspace(-25, 25, 50) * np.pi / 180

    if isinstance(wavelengths, float):
        wavelengths = [wavelengths]

    if isinstance(tx_in, (float, int)):
        tx_in  = np.array([tx_in])
    if isinstance(ty_in, (float, int)):
        ty_in = np.array([ty_in])

    n_wavelens = len(wavelengths)

    # input angles
    txtx_in, tyty_in = np.meshgrid(tx_in, ty_in)
    tptp_ins, tmtm_ins = angle2pm(txtx_in, tyty_in)

    # blaze condition (does not depend on wavelength)
    tp_blaze_on = np.zeros(tptp_ins.shape)
    tm_blaze_on = np.zeros(tptp_ins.shape)
    tp_blaze_off = np.zeros(tptp_ins.shape)
    tm_blaze_off = np.zeros(tptp_ins.shape)
    for ii in range(tptp_ins.size):
        ind = np.unravel_index(ii, tptp_ins.shape)
        tp_blaze_on[ind], tm_blaze_on[ind] = solve_blaze(tptp_ins[ind], tmtm_ins[ind], gamma_on)
        tp_blaze_off[ind], tm_blaze_off[ind] = solve_blaze(tptp_ins[ind], tmtm_ins[ind], gamma_off)
    tx_blaze_on, ty_blaze_on = angle2xy(tp_blaze_on, tm_blaze_on)
    tx_blaze_off, ty_blaze_off = angle2xy(tp_blaze_off, tm_blaze_off)

    # store results
    efields = np.zeros((n_wavelens, len(ty_in), len(tx_in), len(tout_offsets), len(tout_offsets)), dtype=np.complex)
    sinc_efield_on = np.zeros(efields.shape, dtype=np.complex)
    sinc_efield_off = np.zeros(efields.shape, dtype=np.complex)
    diffraction_efield = np.zeros(efields.shape, dtype=np.complex)
    tx_out = np.zeros((len(ty_in), len(tx_in), len(tout_offsets), len(tout_offsets)))
    ty_out = np.zeros((len(ty_in), len(tx_in), len(tout_offsets), len(tout_offsets)))
    # diffraction order predictions
    ndiff_orders = 7
    diff_tx_out = np.zeros((n_wavelens, len(ty_in), len(tx_in), 2*ndiff_orders + 1, 2*ndiff_orders + 1))
    diff_ty_out = np.zeros((n_wavelens, len(ty_in), len(tx_in), 2*ndiff_orders + 1, 2*ndiff_orders + 1))
    # diff_nx = np.zeros((2*ndiff_orders + 1, 2*ndiff_orders + 1))
    # diff_ny = np.zeros((2*ndiff_orders + 1, 2*ndiff_orders + 1))
    diff_nx, diff_ny = np.meshgrid(range(-ndiff_orders, ndiff_orders + 1), range(-ndiff_orders, ndiff_orders + 1))

    for ii in range(len(ty_in)):
        for jj in range(len(tx_in)):
            for kk in range(n_wavelens):
                # predicted diffraction orders
                # diff_tx_out[kk, ii, jj], diff_ty_out[kk, ii, jj], diff_nx, diff_ny = \
                #     solve_diffraction_output(tx_in[jj], ty_in[ii], dx, dy, wavelengths[kk], ndiff_orders)
                for aa in range(diff_nx.shape[0]):
                    for bb in range(diff_nx.shape[1]):
                        diff_tx_out[kk, ii, jj, aa, bb], diff_ty_out[kk, ii, jj, aa, bb] = \
                            solve_diffraction_output(tx_in[jj], ty_in[jj], dx, dy, wavelengths[kk],
                                                     order=(diff_nx[aa, bb], diff_ny[aa, bb]))


                # get output angles
                tx_out[ii, jj], ty_out[ii, jj] = np.meshgrid(tx_blaze_on[ii, jj] + tout_offsets,
                                                             ty_blaze_on[ii, jj] + tout_offsets)

                # simulate output angles in parallel
                simulate_partial = partial(simulate_dmd, pattern=pattern, wavelength=wavelengths[kk], gamma_on=gamma_on,
                                           gamma_off=gamma_off, dx=dx, dy=dy, wx=wx, wy=wy,
                                           tx_in=txtx_in[ii, jj], ty_in=tyty_in[ii, jj], is_coherent=1)

                results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                    joblib.delayed(simulate_partial)(txs_out=tx, tys_out=ty) for tx, ty in
                    zip(tx_out[ii, jj].ravel(), ty_out[ii, jj].ravel())
                )

                ef, sinc_ef_on, sinc_ef_off, diff_ef = zip(*results)
                efields[kk, ii, jj] = np.reshape(ef, tx_out.shape)
                sinc_efield_on[kk, ii, jj] = np.reshape(sinc_ef_on, tx_out.shape)
                sinc_efield_off[kk, ii, jj] = np.reshape(sinc_ef_off, tx_out.shape)
                diffraction_efield[kk, ii, jj] = np.reshape(diff_ef, tx_out.shape)

                # efields[kk, ii, jj], sinc_efield_on[kk, ii, jj],\
                # sinc_efield_off[kk, ii, jj], diffraction_efield[kk, ii, jj] = \
                #     simulate_dmd(pattern, wavelengths[kk], gamma_on, gamma_off, dx, dy, wx, wy,
                #                  txtx_in[ii, jj], tyty_in[ii, jj],
                #                  tx_out[ii, jj], ty_out[ii, jj], is_coherent=1)

    data = {'tx_ins': tx_in, 'ty_ins': ty_in,
            'tx_blaze_on': tx_blaze_on, 'ty_blaze_on': ty_blaze_on,
            'tx_blaze_off': tx_blaze_off, 'ty_blaze_off': ty_blaze_off,
            'pattern': pattern, 'wavelengths': wavelengths,
            'gamma_on': gamma_on, 'gamma_off': gamma_off,
            'dx': dx, 'dy': dy, 'wx': wx, 'wy': wy,
            'txs_out': tx_out, 'tys_out': ty_out,
            'efields': efields, 'sinc envelope ON efield': sinc_efield_on,
            'sinc envelope OFF efield': sinc_efield_off, 'diffraction efield': diffraction_efield,
            'diffraction tx outs': diff_tx_out, 'diffraction ty outs': diff_ty_out,
            'diffraction nx': diff_nx, 'diffraction ny': diff_ny}

    return data

def plot_2d_sim(data, saving=False, save_dir='dmd_simulation', figsize=(18, 14)):
    """
    Plot results from simulate_2d()

    :param data:
    :param saving:
    :param save_dir:
    :param figsize:
    :return:
    """

    if saving and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # physical parameters
    pattern = data['pattern']
    ny, nx = pattern.shape
    wavelengths = data['wavelengths']
    dx = data['dx']
    dy = data['dy']
    wx = data['wx']
    wy = data['wy']
    gamma_on = data['gamma_on']
    gamma_off = data['gamma_off']

    # input angles
    tx_in = data['tx_ins']
    ty_in = data['ty_ins']

    # blaze angles
    tx_blaze_on = data['tx_blaze_on']
    ty_blaze_on = data['ty_blaze_on']
    tx_blaze_off = data['tx_blaze_off']
    ty_blaze_off = data['ty_blaze_off']

    # output angles
    tx_out = data['txs_out']
    ty_out = data['tys_out']

    # simulation results
    intensity = np.abs(data['efields'])**2
    sinc_on = np.abs(data['sinc envelope ON efield'])**2
    sinc_off = np.abs(data['sinc envelope OFF efield'])**2
    diff = np.abs(data['diffraction efield'])**2

    # diffraction info
    diff_tx_out = data['diffraction tx outs']
    diff_ty_out = data['diffraction ty outs']
    iz = int((diff_tx_out.shape[-1] - 1) / 2)
    diff_nx = data['diffraction nx']
    diff_ny = data['diffraction ny']

    # plot results
    figs = []
    fig_names = []
    gamma = 0.1
    for kk in range(len(wavelengths)):
        for ii in range(len(ty_in)):
            for jj in range(len(tx_in)):
                tp45_in, tm45_in = angle2pm(tx_in[jj], ty_in[ii])

                param_str = 'lambda=%dnm, dx=%0.2fum, w=%0.2fum\n' \
                            'theta in =(%.2f, %.2f)deg xy =(%0.2f, %.2f)deg pm, ' \
                            'gamma (on,off)=(%.2f,%.2f) deg' % \
                            (int(wavelengths[kk] * 1e9), dx * 1e6, wx * 1e6,
                             tx_in[jj] * 180 / np.pi, ty_in[ii] * 180 / np.pi,
                             tp45_in * 180/np.pi, tm45_in * 180 / np.pi,
                             gamma_on * 180 / np.pi, gamma_off * 180 / np.pi)

                dtout = tx_out[ii, jj, 0, 1] - tx_out[ii, jj, 0, 0]
                extent = [(tx_out[ii, jj].min() - 0.5 * dtout) * 180/np.pi,
                          (tx_out[ii, jj].max() + 0.5 * dtout) * 180/np.pi,
                          (ty_out[ii, jj].max() + 0.5 * dtout) * 180/np.pi,
                          (ty_out[ii, jj].min() - 0.5 * dtout) * 180/np.pi]

                fig = plt.figure(figsize=figsize)
                fname = 'tx_in=%0.2f_ty_in=%0.2f_wl=%dnm.png' % (tx_in[jj], ty_in[ii], int(wavelengths[kk] * 1e9))

                figs.append(fig)
                fig_names.append(fname)

                nrows = 2
                ncols = 2
                plt.suptitle(param_str)

                # intensity patterns
                ax = plt.subplot(nrows, ncols, 1)
                # plt.imshow(intensity[kk, ii, jj] / intensity[kk, ii, jj].max(), extent=extent, norm=PowerNorm(gamma=gamma))
                plt.imshow(intensity[kk, ii, jj] / (dx*dy*nx*ny)**2, extent=extent,
                           norm=PowerNorm(gamma=gamma))
                # get xlim and ylim, we will want to keep these...
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # blaze condition
                circ = Circle((tx_blaze_on[ii, jj] * 180 / np.pi, ty_blaze_on[ii, jj] * 180 / np.pi),
                               radius=1, color='r', fill=0, ls='-')
                ax.add_artist(circ)
                circ2 = Circle((tx_blaze_off[ii, jj] * 180 / np.pi, ty_blaze_off[ii, jj] * 180 / np.pi),
                               radius=1, color='g', fill=0, ls='-')
                ax.add_artist(circ2)
                # diffraction peaks
                plt.scatter(diff_tx_out[kk, ii, jj] * 180 / np.pi, diff_ty_out[kk, ii, jj] * 180 / np.pi,
                            edgecolor='k', facecolor='none')
                # diffraction zeroth order
                plt.scatter(diff_tx_out[kk, ii, jj, iz, iz] * 180 / np.pi,
                            diff_ty_out[kk, ii, jj, iz, iz] * 180 / np.pi,
                            edgecolor='m', facecolor='none')


                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                plt.xlabel('theta x outgoing')
                plt.ylabel('theta y outgoing')
                plt.title('I / (wx*wy*nx*ny)**2 vs. output angle')

                # sinc envelopes from pixels
                ax = plt.subplot(nrows, ncols, 2)
                int_sinc = np.abs(sinc_on[kk, ii, jj]) ** 2
                plt.imshow(int_sinc / (wx*wy)**2, extent=extent, norm=PowerNorm(gamma=gamma))
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # blaze condition
                circ = Circle((tx_blaze_on[ii, jj] * 180 / np.pi, ty_blaze_on[ii, jj] * 180 / np.pi),
                              radius=1, color='r', fill=0, ls='-')
                ax.add_artist(circ)
                circ2 = Circle((tx_blaze_off[ii, jj] * 180 / np.pi, ty_blaze_off[ii, jj] * 180 / np.pi),
                               radius=1, color='g', fill=0, ls='-')
                ax.add_artist(circ2)
                # diffraction peaks
                plt.scatter(diff_tx_out[kk, ii, jj] * 180 / np.pi, diff_ty_out[kk, ii, jj] * 180 / np.pi,
                            edgecolor='k', facecolor='none')
                # diffraction zeroth order
                plt.scatter(diff_tx_out[kk, ii, jj, iz, iz] * 180 / np.pi,
                            diff_ty_out[kk, ii, jj, iz, iz] * 180 / np.pi,
                            edgecolor='m', facecolor='none')

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                plt.xlabel('theta x outgoing')
                plt.ylabel('theta y outgoing')
                plt.title('Sinc ON, normalized to peak efficiency = (wx*wy)**2')

                # diffraction patterns
                ax = plt.subplot(nrows, ncols, 3)
                plt.imshow(np.abs(diff[kk, ii, jj]) ** 2, extent=extent, norm=PowerNorm(gamma=gamma))
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # blaze condition
                circ = Circle((tx_blaze_on[ii, jj] * 180 / np.pi, ty_blaze_on[ii, jj] * 180 / np.pi),
                              radius=1, color='r', fill=0, ls='-')
                ax.add_artist(circ)
                circ2 = Circle((tx_blaze_off[ii, jj] * 180 / np.pi, ty_blaze_off[ii, jj] * 180 / np.pi),
                               radius=1, color='g', fill=0, ls='-')
                ax.add_artist(circ2)

                # diffraction peaks
                plt.scatter(diff_tx_out[kk, ii, jj] * 180/np.pi, diff_ty_out[kk, ii, jj] * 180/np.pi,
                            edgecolor='k', facecolor='none')
                # diffraction zeroth order
                plt.scatter(diff_tx_out[kk, ii, jj, iz, iz] * 180 / np.pi,
                            diff_ty_out[kk, ii, jj, iz, iz] * 180 / np.pi,
                            edgecolor='m', facecolor='none')

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                plt.xlabel('theta_x outgoing')
                plt.ylabel('theta_y outgoing')
                plt.title('diffraction pattern')

                plt.subplot(nrows, ncols, 4)
                plt.imshow(pattern)
                plt.title('DMD pattern')

                # xy cuts
                # plt.subplot(nrows, ncols, 4)
                # tb_45 = np.arctan(np.sqrt(2) * np.tan(tx_out[0, :]))
                #
                # tb_x_oversample = np.linspace(tx_out[0, :].min(), tx_out[0, :].max(), 1000)
                # tb_45_oversample = np.arctan(np.sqrt(2) * np.tan(tb_x_oversample))
                #
                # plt.plot(tb_45 * 180 / np.pi, scale_fn(np.diag(intensity)), 'b.-')
                # plt.plot([t45_diff, t45_diff] * 180 / np.pi, ylim, 'k')
                # plt.plot([tm45_on, tm45_on] * 180 / np.pi, ylim, 'r')
                # plt.plot([tm45_off, tm45_off] * 180 / np.pi, ylim, 'g')
                #
                # plt.xlabel('theta in 45 deg')
                # plt.ylabel('log10|I|')
                # plt.title('I along theta x = -theta y')

                if saving:
                    fpath = os.path.join(save_dir, fname)
                    fig.savefig(fpath)

    return figs, fig_names

def simulate_2d_angles(wavelengths, gamma, dx, dy, tx_ins, ty_ins):
    """
    Determine Blaze and diffraction angles in 2D

    :param wavelengths:
    :param gamma:
    :param dx:
    :param dy:
    :param tx_ins:
    :param ty_ins:
    :return:
    """

    if isinstance(wavelengths, float):
        wavelengths = [wavelengths]

    n_wavelens = len(wavelengths)
    max_orders = 15
    nxs, nys = np.meshgrid(range(-max_orders, max_orders + 1), range(-max_orders, max_orders + 1))
    nangles = 60

    txtx_in, tyty_in = np.meshgrid(tx_ins, ty_ins)
    tp_in, tm_in = angle2pm(txtx_in, tyty_in)

    tx_out_diff = np.zeros((n_wavelens, txtx_in.shape[0], txtx_in.shape[1], 2 * max_orders + 1, 2 * max_orders + 1))
    ty_out_diff = np.zeros(tx_out_diff.shape)
    min_cost = np.zeros(txtx_in.shape)
    nx_closest = np.zeros((n_wavelens, txtx_in.shape[0], txtx_in.shape[1]), dtype=np.int8)
    ny_closest = np.zeros(nx_closest.shape, dtype=np.int8)

    tp_outs = np.zeros(txtx_in.shape)
    tm_outs = np.zeros(tyty_in.shape)
    tx_out = np.zeros(tyty_in.shape)
    ty_out = np.zeros(tyty_in.shape)
    for ii in range(txtx_in.size):
        ind = np.unravel_index(ii, txtx_in.shape)
        tp_outs[ind], tm_outs[ind] = solve_blaze(tp_in[ind], tm_in[ind], gamma)
        tx_out[ind], ty_out[ind] = angle2xy(tp_outs[ind], tm_outs[ind])
        for jj in range(n_wavelens):
            # tx_out_temp, ty_out_temp, nxs, nys = solve_diffraction_output(txtx_in[ind], tyty_in[ind], d, d,
            #                                                               wavelengths[jj], max_orders)
            tx_out_temp = np.zeros(nxs.size)
            ty_out_temp = np.zeros(nys.size)
            for aa in range(nxs.size[0]):
                for bb in range(nys.size[1]):
                    tx_out_temp[aa, bb], ty_out_temp[aa, bb] = solve_diffraction_output(txtx_in[ind], tyty_in[ind],
                                                                                        dx, dy, wavelengths[jj],
                                                                                        order=(nxs[aa, bb], nys[aa, bb]))


            tx_out_diff[jj, ind[0], ind[1]] = tx_out_temp
            ty_out_diff[jj, ind[0], ind[1]] = ty_out_temp

            try:
                imin = np.nanargmin((tx_out_temp - tx_out[ind]) ** 2 + (ty_out_temp - ty_out[ind]) ** 2)
                indmin = np.unravel_index(imin, (2 * max_orders + 1, 2 * max_orders + 1))
                min_cost[ind] += (tx_out_temp[indmin[0], indmin[1]] - tx_out[ind]) ** 2 + \
                                 (ty_out_temp[indmin[0], indmin[1]] - ty_out[ind]) ** 2
                nx_closest[jj, ind[0], ind[1]] = int(nxs[indmin[1]])
                ny_closest[jj, ind[0], ind[1]] = int(nys[indmin[0]])
            except ValueError:
                min_cost[ind] = np.nan
                nx_closest[jj, ind[0], ind[1]] = -100
                ny_closest[jj, ind[0], ind[1]] = -100

            # find closest diffraction orders
            imin = np.argmin(tx_out_diff[jj, ind[0]])

    data = {'wavelengths': wavelengths, 'gamma': gamma, 'dx': dx, 'dy': dy,
            'tx_out_diff': tx_out_diff, 'ty_out_diff': ty_out_diff,
            'min_cost': min_cost,
            'nx_closest': nx_closest, 'ny_closest': ny_closest,
            'tp_outs': tp_outs, 'tm_outs': tm_outs, 'tx_out': tx_out, 'ty_out': ty_out,
            'tx_ins': tx_ins, 'ty_ins': ty_ins}
    return data

def display_2d_angles(data):
    """
    Plot graphical solutions to Blaze condition and diffraction condition obtained from simulate_2d_angles()
    :param **data:
    :return:
    """
    wavelengths = data['wavelengths']
    gamma = data['gamma']
    dx = data['dx']
    dy = data['dy']
    tx_out_diff = data['tx_out_diff']
    ty_out_diff = data['ty_out_diff']
    min_cost = data['min_cost']
    nx_closest = data['nx_closest']
    ny_closest = data['ny_closest']
    tp_outs = data['tp_outs']
    tm_outs = data['tm_outs']
    tx_out = data['tx_out']
    ty_out = data['ty_out']
    tx_ins = data['tx_ins']
    ty_ins = data['ty_ins']


    n_wavelens = len(wavelengths)
    max_orders = int((tx_out_diff.shape[-1] - 1) / 2)
    nangles = tx_ins.size

    figh1 = plt.figure()
    dtx = tx_ins[1] - tx_ins[0]
    dty = ty_ins[1] - ty_ins[0]
    extent = [(tx_ins[0] - 0.5 * dtx) * 180 / np.pi,
              (tx_ins[-1] + 0.5 * dtx) * 180 / np.pi,
              (ty_ins[-1] + 0.5 * dty) * 180 / np.pi,
              (ty_ins[0] - 0.5 * dty) * 180 / np.pi]
    plt.imshow(np.log(np.sqrt(min_cost) * 180 / np.pi), extent=extent)
    plt.colorbar()
    plt.xlabel('tx in (deg)')
    plt.ylabel('ty in (deg)')

    # display results
    cmap = matplotlib.cm.get_cmap('jet')
    colors = [cmap(ii / (n_wavelens - 1)) for ii in range(n_wavelens)]

    figh2, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    for jj in range(n_wavelens):
        ax.scatter(tx_out_diff[jj, 0, 0] * 180 / np.pi, ty_out_diff[jj, 0, 0] * 180 / np.pi,
                   edgecolor=colors[jj], facecolor='none')
    ax.scatter(tx_out[0, 0] * 180 / np.pi, ty_out[0, 0] * 180 / np.pi, color='r')
    ax.set_xlabel('tx out (deg)')
    ax.set_ylabel('ty out (deg)')
    ax.set_xlim([-90, 90])
    ax.set_ylim([-90, 90])

    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    tx_in_slider = matplotlib.widgets.Slider(ax_x, 'tx in index', 0, nangles - 1, valinit=0, valstep=1)
    ty_in_slider = matplotlib.widgets.Slider(ax_y, 'ty in index', 0, nangles - 1, valinit=0, valstep=1)

    def update(val):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()
        ix = int(tx_in_slider.val)
        iy = int(ty_in_slider.val)

        ax.plot([-90, 90], [90, -90], 'k')
        for jj in range(n_wavelens):
            ax.scatter(tx_out_diff[jj, iy, ix] * 180 / np.pi, ty_out_diff[jj, iy, ix] * 180 / np.pi,
                       edgecolor=colors[jj], facecolor='none')
        ax.scatter(tx_out[iy, ix] * 180 / np.pi, ty_out[iy, ix] * 180 / np.pi, color='r')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title('(tx, ty) = (%0.2f, %0.2f) deg' % (tx_ins[ix] * 180 / np.pi, ty_ins[iy] * 180 / np.pi))
        figh2.canvas.draw_idle()

    tx_in_slider.on_changed(update)
    ty_in_slider.on_changed(update)

    return figh1, figh2

if __name__ == "__main__":
    # example 2/4 color 1D/2D simulation

    # wavelengths = [405e-9, 473e-9, 532e-9, 635e-9]
    # colors = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0]]
    wavelengths = [473e-9, 532e-9, 635e-9]
    colors = [[0, 1, 1], [0, 1, 0], [1, 0, 0]]

    #tx_ins = np.array([30, 40]) * np.pi/180
    # tx_ins = np.array([74]) * np.pi/180
    # ty_ins = np.array([70]) * np.pi/180
    tx_ins = np.array([35]) * np.pi / 180
    ty_ins = np.array([30]) * np.pi/180
    _, t45_ins = angle2pm(tx_ins, ty_ins)
    # t45_ins = 45 * np.pi/180

    # pixel spacing (DMD pitch)
    dx = 7.56e-6 # meters
    dy = dx
    # pixel size, inferred from coverage fraction
    coverage_f = 0.92
    wx = np.sqrt(coverage_f) * dx
    wy = wx
    # on and off angle of mirrors from the normal plane of the DMD body
    gamma_on = 12 * np.pi/180
    gamma_off = -12 * np.pi/180

    # create pattern
    nx = 10
    ny = 10

    # set pattern for DMD to be simulated
    # pattern = np.ones((ny, nx))
    [xx, yy] = np.meshgrid(range(nx), range(ny))
    theta_pattern = -np.pi/4
    period = 3 * np.sqrt(2)
    #
    pattern = np.cos(2 * np.pi / period * (xx * np.cos(theta_pattern) + yy * np.sin(theta_pattern)))
    pattern[pattern < 0] = 0
    pattern[pattern > 0] = 1

    save_dir = 'data/dmd_simulation'

    # sample 1D simulation
    # data1d = simulate_1d(pattern, wavelengths, gamma_on, gamma_off, dx, dy, wx, wy, t45_ins)
    # plot_1d_sim(data1d, colors, saving=False, save_dir=save_dir)

    # sample 2D simulation
    data2d = simulate_2d(pattern, wavelengths, gamma_on, gamma_off, dx, dy, wx, wy, tx_ins, ty_ins,
                         tout_offsets=np.linspace(-25, 25, 150) * np.pi / 180)
    plot_2d_sim(data2d, saving=False, save_dir=save_dir)
