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

class dmd_simulation():
    # todo: work in progress.
    def __init__(self, gamma_on, gamma_off, dx, dy, wx, wy):
        self.gamma_on = gamma_on
        self.gamma_off = gamma_off
        self.dx = dx
        self.dy = dy
        self.wx = wx
        self.wy = wy

    def simulate_dmd_diffraction(self, wavelength, pattern, tx_in, ty_in, tx_out, ty_out):
        pass

# main simulation function and important auxiliary functions
def simulate_dmd(pattern, wavelength, gamma_on, gamma_off, dx, dy, wx, wy,
                 tx_in, ty_in, txs_out, tys_out, is_coherent=True):
    """
    Simulate light diffracted from a digital mirror device (DMD). We assume that the body of the device is in the xy
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
    Compute normalized blaze envelope function

    :param wavelength:
    :param gamma:
    :param wx:
    :param wy:
    :param tx_in:
    :param ty_in:
    :param tx_out:
    :param ty_out:
    :return:
    """

    k = 2*np.pi / wavelength
    amb = get_unit_vector(tx_in, ty_in, 'in') - get_unit_vector(tx_out, ty_out, 'out')

    val = sinc_fn(0.5 * k * wx * blaze_condition_fn(gamma, amb, 'plus')) \
        * sinc_fn(0.5 * k * wy * blaze_condition_fn(gamma, amb, 'minus'))
    return val

def blaze_condition_fn(gamma, amb, mode='plus'):
    """
    Return the dimensionsless part of the sinc function argument which determines the Blaze condition, which we refer
    to as A_+ and A_-

    E = (diffraction from different mirrors) x w**2 * sinc(0.5 * k * w * A_+) * sinc(0.5 * k * w * A_-)

    A_\pm = 0.5*(1 \pm cos(gamma)) * (a-b)_x + 0.5*(1 \mp cos(gamma)) * (a-b)_y \mp sin(gamma)/sqrt(2) * (a-b)_z

    :param gamma: angle micro-mirror normal makes with device normal
    :param amb: incoming unit vector - outgoing unit vector, [vx, vy, vz]
    :param mode: 'plus' or 'minus'
    :return A:
    """
    if mode == 'plus':
        A = 0.5 * (1 + np.cos(gamma)) * amb[0] + \
            0.5 * (1 - np.cos(gamma)) * amb[1] - \
            1 / np.sqrt(2) * np.sin(gamma) * amb[2]
    elif mode == 'minus':
        A = 0.5 * (1 - np.cos(gamma)) * amb[0] + \
            0.5 * (1 + np.cos(gamma)) * amb[1] + \
            1 / np.sqrt(2) * np.sin(gamma) * amb[2]
    else:
        raise Exception("mode must be 'plus' or 'minus', but was '%s'" % mode)
    return A

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

    :param tx:
    :param ty:
    :param mode: "in" or "out" depending on whether representing a vector pointing in the positive or negative z-direction
    :return uvec: unit vector
    """
    if mode == 'in':
        uvec = np.array([np.tan(tx), np.tan(ty), 1])
    elif mode == 'out':
        uvec = np.array([np.tan(tx), np.tan(ty), -1])
    else:
        raise Exception("mode must be 'in' or 'out', but was '%s'" % mode)

    uvec = uvec / np.linalg.norm(uvec)
    return uvec

def sinc_fn(x):
    """
    Sinc function

    :param x:
    :return sin(x) / x:
    """
    x = np.asarray(x)
    y = np.asarray(np.sin(x) / x)
    y[x == 0] = 1
    return y

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

def txty2theta_phi(tx, ty):
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

def theta_phi2txty(theta, phi):
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

# solve blaze condition and diffraction condition
def solve_blaze_condition_1d(tm_in, gamma):
    """
    Get blaze angle for tp_in = 0 and tm_in. For all other applications, use get_blaze_2d()

    :param tm_in:
    :param gamma:
    :return tm_out_blaze:
    """
    return tm_in - 2*gamma

def solve_blaze_condition(tp_in, tm_in, gamma):
    """
    Find the blaze condition for arbitrary input angle. Don't know of a closed form solution to this problem.

    :param tp_in: input angle in x+y plane in radians
    :param tm_in: input angle in x-y plane in radians
    :param gamma: DMD mirror angle in radians
    :return tp_out: output angles (in general there will be two solutions)
    :return tm_out:
    """
    #tp_in, tm_in = angle2pm(tx_in, ty_in)
    a = lambda tp, tm: get_unit_vector(tp, tm, mode='in')
    b = lambda tp, tm: get_unit_vector(tp, tm, mode='out')
    # a = lambda tp, tm: np.array([np.tan(tp), np.tan(tm), 1]) / np.sqrt(np.tan(tp)**2 + np.tan(tm)**2 + 1)
    # b = lambda tp, tm: np.array([np.tan(tp), np.tan(tm), -1]) / np.sqrt(np.tan(tp) ** 2 + np.tan(tm) ** 2 + 1)

    f1 = lambda tp, tm: a(tp_in, tm_in)[0] - b(tp, tm)[0]
    f2 = lambda tp, tm: (a(tp_in, tm_in)[1] - b(tp, tm)[1]) / (a(tp_in, tm_in)[2] - b(tp, tm)[2]) - np.tan(gamma)

    init_param = [tp_in, tm_in - 2*gamma]
    min_fn = lambda p: np.abs(f1(p[0], p[1])) + np.abs(f2(p[0], p[1]))
    param_min = scipy.optimize.fmin(min_fn, init_param, disp=False)

    if min_fn(param_min) > 1e-2:
        tp_out = np.nan
        tm_out = np.nan
    else:
        tp_out, tm_out = param_min

    return tp_out, tm_out

def solve_diffraction_condition(tx_in, ty_in, dx, dy, wavelength, max_order=4):
    """
    Solve for output diffraction peak angles, given input angles and wavelength.

    these occur where the following conditions are fulfilled:
    tan(tx_in) / norm_in - tan(tx_out) / norm_out = lambda/dx * nx
    tan(ty_in) / norm_in - tan(ty_out) / norm_out = lambda/dx * ny

    :param tx_in: theta x incoming, in radians
    :param ty_in:
    :param dx:
    :param dy:
    :param wavelength: in the same units as dx, dy
    :param max_order: will solve for all orders (i, j), for i and j in  [-max_order, ..., 0, ... max_order]
    :return:
    """

    # expected diffraction peaks
    # these functions are zero where diffraction condition is satisfied
    norm_in = np.sqrt(np.tan(tx_in) ** 2 + np.tan(ty_in) ** 2 + 1)
    cond_x = lambda tx_out, ty_out, nx: np.tan(tx_in) / norm_in - np.tan(tx_out) / np.sqrt(np.tan(tx_out)**2 + np.tan(ty_out)**2 + 1) - wavelength/dx * nx
    cond_y = lambda tx_out, ty_out, ny: np.tan(ty_in) / norm_in - np.tan(ty_out) / np.sqrt(np.tan(tx_out)**2 + np.tan(ty_out)**2 + 1) - wavelength/dy * ny

    # all the orders we want to solve for
    nxs = np.arange(-max_order, max_order+1)
    nys = np.arange(-max_order, max_order+1)
    tx_outs = np.zeros((2*max_order + 1, 2*max_order + 1))
    ty_outs = np.zeros((2*max_order + 1, 2*max_order + 1))

    for ii in range(len(nys)):
        for jj in range(len(nxs)):
            # these will be overwritten if there is a solution
            tx_out = np.nan
            ty_out = np.nan

            nx = nxs[ii]
            ny = nys[jj]

            cx = np.tan(tx_in) / norm_in - wavelength / dx * nx
            cy = np.tan(ty_in) / norm_in - wavelength / dy * ny
            cmat = np.array([[1 - cx**2, -cx**2], [-cy**2, 1 - cy**2]])

            try:
                tan_sqr_outs = np.linalg.solve(cmat, np.array([[cx**2], [cy**2]]))
                #tan_sqr_outs = np.linalg.inv(cmat).dot(np.array([[cx**2], [cy**2]]))

                # not guaranteed the signs are correct
                tx_out_abs = np.arctan(np.sqrt(tan_sqr_outs[0]))
                ty_out_abs = np.arctan(np.sqrt(tan_sqr_outs[1]))

                choices = [[tx_out_abs, ty_out_abs], [-tx_out_abs, ty_out_abs], [tx_out_abs, -ty_out_abs],
                           [-tx_out_abs, -ty_out_abs]]
                for c in choices:
                    if np.abs(cond_x(c[0], c[1], nx)) < 1e-12 and np.abs(cond_y(c[0], c[1], ny)) < 1e-12:
                        tx_out = c[0]
                        ty_out = c[1]
                        break
            except np.linalg.LinAlgError:
                pass

            tx_outs[ii, jj] = tx_out
            ty_outs[ii, jj] = ty_out

    return tx_outs, ty_outs, nxs, nys

# utility functions for solving blaze + diffraction conditions
def solve_max_diffraction_order(wavelength, d, gamma):
    """
    Find the maximum and minimum diffraction orders consistent with given parameters and the Blaze condition assuming
    1D situtation (tx = -ty)

    Diffraction condition in this case is:
    sin(theta_in) - sin(theta_out) = sqrt(2) * lambda / d * n

    :param wavelength: wavelength of light
    :param d: mirror pitch (in same units as wavelength)
    :param gamma: mirror angle
    :return nmax: maximum index of diffraction order
    :return nmin: minimum index of diffraction order
    """

    # solution for maximum order
    theta_a_opt = np.arctan( (np.cos(2*gamma) - 1) / np.sin(2*gamma))

    # this should = sqrt(2) * lambda / d * n when diffraction condition is satisfied
    f = lambda t: np.sin(t) - np.sin(t - 2*gamma)

    # must also check end points for possible extrema
    # ts in range [-np.pi/2 np.pi/2]
    if gamma > 0:
        fopts = [f(-np.pi/2 + 2*gamma), f(np.pi/2), f(theta_a_opt)]
    elif gamma <= 0:
        fopts = [f(-np.pi / 2), f(np.pi / 2 + 2*gamma), f(theta_a_opt)]

    # find actually extrema
    fmin = np.min(fopts)
    fmax = np.max(fopts)

    nmax = np.floor(fmax * d / np.sqrt(2) / wavelength)
    nmin = np.ceil(fmin * d / np.sqrt(2) / wavelength)

    return nmax, nmin

def solve_1color(wavelength, d, gamma, n):
    """
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

def solve_diffraction_input(theta_out, wavelength, d, order):
    """
    Find input angle corresponding to given diffraction output angle, for 1D case

    :param theta_out: desired output angle (in radians)
    :param wavelength: wavelength of light
    :param d: mirror pitch (in same units as wavelength)
    :param order: index of diffraction order
    :return theta_in: input angle
    """
    theta_in = np.arcsin(np.sin(theta_out) + np.sqrt(2) * wavelength / d * order)
    return theta_in

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

# 1D simulation in x-y plane (i.e. theta_x = -theta_y)
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
    t45s_blaze_on = solve_blaze_condition_1d(t45_ins, gamma_on)
    t45s_blaze_off = solve_blaze_condition_1d(t45_ins, gamma_off)

    # expected diffraction zero peak
    _, t45s_diff_zero = angle2pm(tx_ins, ty_ins)

    # arrays to store info about diffraction peaks
    peak_locs_all = np.zeros((len(t45_ins), 3, n_wavelens))
    peak_heights_all = np.zeros(peak_locs_all.shape)

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
    diff_t45_out = np.zeros((len(t45_ins), n_wavelens, 2 * ndiff_orders + 1))
    diff_n = np.zeros((2 * ndiff_orders + 1))

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
            diff_tx_out, diff_ty_out, diff_nx, diff_ny = \
                solve_diffraction_condition(tx_ins[kk], ty_ins[kk], dx, dy, wavelengths[ii], ndiff_orders)
            tp, tm = angle2pm(diff_tx_out, diff_ty_out)
            diff_t45_out[kk, ii] = np.diag(np.rot90(tm))
            diff_n = diff_nx

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
            'diff_t45_out': diff_t45_out, 'diff_n': diff_n,
            'efields': efields, 'sinc_efield_on': sinc_efield_on,
            'sinc_efield_off': sinc_efield_off, 'diffraction_efield': diffraction_efield}

    return data

def find_peaks_sim1d(data):
    """
    Plot peak positions for main and sidepeaks in SIM data.
    #todo: dumped code from simulate_1d() here during refactoring. Need to make work.
    :return:
    """

    # todo: unpack data

    n_wavelens = len(wavelengths)

    # find peaks, most naive method
    for ii in range(n_wavelens):

        # diffraction peaks
        # todo: get from diffraction fn now
        tout_peak_pos, peak_ht, peak_ind = find_peaks(diffraction_int[:, ii], t45s_out[kk])

        if tout_peak_pos.size != 0:
            # todo: when replace with diffraction condition, no need to do this filtering.
            # only keep peaks within certain height of max
            inds_keep = peak_ht > 0.01 * np.max(peak_ht)
            tout_peak_pos = tout_peak_pos[inds_keep]
            peak_ht = peak_ht[inds_keep]
            peak_ind = peak_ind[inds_keep]

            # closest diffraction peak to blaze angle
            closest_peak_ind = np.argmin(abs(tout_peak_pos - t45s_blaze_on[kk]))

            peak_locs_all[kk, 1, ii] = tout_peak_pos[closest_peak_ind]
            peak_heights_all[kk, 1, ii] = peak_ht[closest_peak_ind]

            # find side peaks
            tout_side_peaks, side_peak_ht, side_peak_ind = find_peaks(np.abs(efields[:, ii])**2, t45s_out[kk])
            inds_keep = side_peak_ht > 0.0001 * np.max(side_peak_ht)
            tout_side_peaks = tout_side_peaks[inds_keep]
            side_peak_ht = side_peak_ht[inds_keep]
            side_peak_ind = side_peak_ind[inds_keep]

            main_peak_ind = np.argmin(np.abs(tout_side_peaks - peak_locs_all[kk, 1, ii]))

            # save peak locs and heights
            try:
                peak_locs_all[kk, 0, ii] = tout_side_peaks[main_peak_ind - 1]
                peak_heights_all[kk, 0, ii] = side_peak_ht[main_peak_ind - 1]
            except:
                # if sidepeak index doesn't exist....
                peak_locs_all[kk, 0, ii] = np.nan
                peak_heights_all[kk, 0, ii] = np.nan

            try:
                peak_locs_all[kk, 2, ii] = tout_side_peaks[main_peak_ind + 1]
                peak_heights_all[kk, 2, ii] = side_peak_ht[main_peak_ind + 1]
            except:
                peak_locs_all[kk, 2, ii] = np.nan
                peak_heights_all[kk, 2, ii] = np.nan

        else:
            peak_locs_all[kk, :, ii] = np.nan
            peak_heights_all[kk, :, ii] = np.nan

    # find largest diffraction peak for each color

        # peak locations vs input angle
        # no point in plotting summary if we have only one angle

        figh = plt.figure(figsize=(16,12))
        plt.suptitle('DMD performance for multiple colors')

        nrows = 1
        ncols = 2

        ax = plt.subplot(nrows, ncols, 1)
        hs = []
        for ii in range(n_wavelens):
            h = plt.plot(t45_ins * 180 / np.pi, peak_locs_all[:, 1, ii] * 180 / np.pi, color=colors[ii])
            plt.plot(t45_ins * 180 / np.pi, peak_locs_all[:, [0, 2], ii] * 180 / np.pi, '--', color=colors[ii])
            hs.append(h)

        h1 = plt.plot(t45_ins * 180 / np.pi, t45s_blaze_on * 180 / np.pi, 'k:')
        # h2 = plt.plot(t45_ins * 180/np.pi, t45s_blaze_off * 180/np.pi, 'k--')
        h3 = plt.plot(t45_ins * 180 / np.pi, t45s_diff_zero * 180 / np.pi, 'm')
        hs = hs + [h1, h3]

        leg = [str(l * 1e9) for l in wavelengths] + ['blaze ON', '0th order']

        plt.legend(hs, leg)
        plt.xlabel('t45 in (deg)')
        plt.ylabel('t45 out (deg)')
        plt.title('peak locations vs input angle')

        # side peak imbalance
        ax = plt.subplot(nrows, ncols, 2)
        side_peak_imb = peak_heights_all[:, 0, :] / peak_heights_all[:, 2, :]

        side_peak_imb[side_peak_imb > 1] = 1 / side_peak_imb[side_peak_imb > 1]
        side_peak_imb = np.sqrt(side_peak_imb)

        for ii in range(n_wavelens):
            plt.plot(t45_ins * 180 / np.pi, side_peak_imb[:, ii], color=colors[ii])

        plt.legend([str(l * 1e9) for l in wavelengths])
        plt.xlabel('t45 in (deg)')
        plt.ylabel('imbalance')
        plt.title('side peak  electric field imbalance vs input angle')

def find_peaks(x, t):
    """
    Find peaks in 1D data
    # todo: test or delete

    :param x:
    :param t:
    :return:
    """

    if x.size != t.size:
        raise Exception('x and t must be the same size')

    # if x.shape[0] > 1 and t.shape[1] > 1:
    #     t = np.transpose(t)
    # elif x.shape[1] > 1 and t.shape[0] > 1:
    #     t = np.transpose(t)

    # take derivative
    dxdt = (x[1:] - x[:-1]) / (t[1:] - t[:-1])
    td = 0.5 * (t[1:] - t[:-1])

    # identify zero crossingings
    # bool_zc(ii) = 1 if zero crossing between dxdt(ii-1) and dxdt(ii)
    bool_zc = (dxdt > 0) * np.roll(dxdt < 0, 1) + \
              (dxdt < 0) * np.roll(dxdt > 0, 1)
    bool_zc[0] = 0 # to eliminate circular aspect of circshift.
    bool_zc = np.concatenate((bool_zc, np.array([0])))

    # for dxdt, td(ii) = 0.5(t(ii+1) + t(ii)), so if
    # bool_zc(ii) = 1, i.e. the zero crossing is definitely in
    #(td(ii-1), td(ii)) = ( 0.5*(t(ii) - t(ii-1)), 0.5*(t(ii+1) - t(ii))
    # then it must be at t(ii).
    # on the other hand, if we find an actual zero at td(ii), then either
    # t(ii+1) or t(ii) could be the peak. We will take t(ii) for ease.
    inds = np.arange(len(t))
    # inds_peaks = inds[bool_zc == 1]

    # peak data
    # locs = t[inds_peaks]
    # heights = x[inds_peaks]
    locs = t[bool_zc > 0]
    heights = x[bool_zc > 0]
    inds_peaks = inds[bool_zc > 0]

    return locs, heights, inds_peaks

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
    diff_n = data['diff_n']
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

        nrows = 2
        ncols = 2

        # title
        param_str = 'spacing = %0.2fum, w=%0.2fum \ntheta in = (%0.2f, %0.2f)deg = %0.2f deg (x-y) ' \
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
        ax = plt.subplot(nrows, ncols, 1)
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
        plt.title('diffraction pattern along theta_x = -theta_y')

        # ###########################
        # plot sinc functions and wider angular range
        # ###########################
        ax = plt.subplot(nrows, ncols, 2)

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
        plt.title('sinc envelopes along theta_x = -theta_y')

        # show pattern
        plt.subplot(nrows, ncols, 3)
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
        tp_blaze_on[ind], tm_blaze_on[ind] = solve_blaze_condition(tptp_ins[ind], tmtm_ins[ind], gamma_on)
        tp_blaze_off[ind], tm_blaze_off[ind] = solve_blaze_condition(tptp_ins[ind], tmtm_ins[ind], gamma_off)
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
    diff_nx = np.zeros((2*ndiff_orders + 1, 2*ndiff_orders + 1))
    diff_ny = np.zeros((2*ndiff_orders + 1, 2*ndiff_orders + 1))

    for ii in range(len(ty_in)):
        for jj in range(len(tx_in)):
            for kk in range(n_wavelens):
                # predicted diffraction orders
                diff_tx_out[kk, ii, jj], diff_ty_out[kk, ii, jj], diff_nx, diff_ny = \
                    solve_diffraction_condition(tx_in[jj], ty_in[ii], dx, dy, wavelengths[kk], ndiff_orders)

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

    if not os.path.exists(save_dir):
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
        tp_outs[ind], tm_outs[ind] = solve_blaze_condition(tp_in[ind], tm_in[ind], gamma)
        tx_out[ind], ty_out[ind] = angle2xy(tp_outs[ind], tm_outs[ind])
        for jj in range(n_wavelens):
            tx_out_temp, ty_out_temp, nxs, nys = solve_diffraction_condition(txtx_in[ind], tyty_in[ind], d, d,
                                                                             wavelengths[jj], max_orders)

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
    tx_ins = np.array([74]) * np.pi/180
    ty_ins = np.array([70]) * np.pi/180
    _, t45_ins = angle2pm(tx_ins, ty_ins)

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
