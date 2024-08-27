"""
Tools for dealing with Gaussian beams. Most importantly
(1) generate Gaussian beam electric field with gauss_efield()
(2) calculate Gauss beam parameters including waist, Rayleigh range, radius of curvature etc.
with get_q() and q2beam_params()
(3) Fit waist and position along beam from a sequence of w(z) data with fit_beam_radii()
"""

from typing import Optional, Union
from collections.abc import Sequence
import numpy as np
from localize_psf.fit import fit_model
from localize_psf.rotation import euler_mat_inv
from localize_psf.affine import xform_points

def gauss_efield(x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 params: Sequence[float]):
    """
    Compute the electric field of a Gaussian beam at position (x, y, z) assuming phasor convention exp(-iwt). If using
    the other phasor convention, you must take the complex conjugate the resulting field

    :param x: x-coordinate where electric field will be evaluated. x, y, and z coordinates should be broadcastable
     to the same shape
    :param y:
    :param z:
    :param params: [wo_x, wo_y, xc, yc, zc, wavelength, n, phi, theta, psi]
     where the beam is centered at (xc, yc, zc) and rotated by the Euler angles (phi, theta, psi)
    :return efield:
    """
    # unpack parameters
    wo_x, wo_y, xc, yc, zc, wavelength, n, phi, theta, psi = params
    k = 2*np.pi / wavelength

    # generate affine matrix which converts between rotated/unrotated coordinates
    # and also accounts for center shift
    emat = euler_mat_inv(phi, theta, psi)
    affine_mat = np.zeros((4, 4))
    affine_mat[:-1, :-1] = emat
    affine_mat[:-1, -1] = np.array([-xc, -yc, -zc])
    affine_mat[-1, -1] = 1

    # broadcast coordinates and do transformation
    xb, yb, zb = np.broadcast_arrays(x, y, z)
    bcast_shape = xb.shape
    xp, yp, zp = xform_points(np.stack((xb.ravel(), yb.ravel(), zb.ravel()), axis=1),
                              affine_mat).transpose()
    xp = xp.reshape(bcast_shape)
    yp = yp.reshape(bcast_shape)
    zp = zp.reshape(bcast_shape)

    # compute using complex beam parameter
    qx_zs = get_q(wo_x, zp, wavelength, 0, n)
    qx_o = get_q(wo_x, 0, wavelength, 0, n)
    qy_zs = get_q(wo_y, zp, wavelength, 0, n)
    qy_o = get_q(wo_y, 0, wavelength, 0, n)
    # rzs_x, wzs_sqr_x, wo_sqr_x, _, zr_x = q2beam_params(qx_zs, wavelength, n)

    # note that the qo/qz factors account for both the waist prefactor the Guoy phase
    efield = (np.sqrt(qx_o * qy_o / qx_zs / qy_zs) *
              np.exp(1j * k * n * xp**2 / (2*qx_zs)) *
              np.exp(1j * k * n * yp**2 / (2*qy_zs)) *
              np.exp(1j * k * n * zp)
              )

    return efield


def get_q(wo: np.ndarray,
          z: np.ndarray,
          wavelength: float,
          zc: np.ndarray = 0.,
          n: Optional[float] = 1.
          ):
    """
    Get the Gaussian complex beam parameter, which is defined by

    q(z) = (z - zc) - i * zr

    1/q(z) = 1 / R(z) + i * wavelength / pi n / w(z)^2

    assuming we use the phasor convention assuming phasor convention exp(-iwt). If using the other phasor convention,
    then take the complex conjugate of q

    :param z:
    :param p: [wo, zc, wavelength, n]
    :return:
    """
    zr = np.pi * wo ** 2 / wavelength * n
    q = (z - zc) - 1j * zr
    return q


def q2beam_params(qz,
                  wavelength: float,
                  n: Optional[float] = 1.):
    """
    Convert complex beam parameter to R(z) and w(z)^2

    :param qz: complex beam parameter
    :param wavelength:
    :return R(z), w(z)^2, w(o)^2, z, zr:
    """

    qz = np.atleast_1d(qz)
    z = np.real(qz)
    zr = -np.imag(qz)
    with np.errstate(all="ignore"):
        r = 1 / (1 / qz).real
        w_sqr = 1 / (1 / qz).imag * wavelength / np.pi / n
        wo_sqr = np.abs(z * (r / w_sqr) * wavelength**2 / np.pi**2)
        wo_sqr[np.isinf(r)] = w_sqr[np.isinf(r)]

    return r, w_sqr, wo_sqr, z, zr


def propagate_abcd(q,
                   abcd_mat: np.ndarray):
    """
    Gaussian propagation of q-parameter using ray transfer matrix

    :param q:
    :param abcd_mat:
    :return qn:
    """
    A = abcd_mat[0, 0]
    B = abcd_mat[0, 1]
    C = abcd_mat[1, 0]
    D = abcd_mat[1, 1]
    return (q * A + B) / (q * C + D)


def solve_waist(wz: float,
                z: float,
                wavelength: float,
                n: Optional[float] = 1.):
    """
    Compute waist (wo) given value 1/e^2 radius at distance, w(z). There are two possible solutions,
    one if the beam is focusing, the other if it is diverging.

    i.e. solve for wo
    w(z) = wo * np.sqrt(1 + (z / zr)**2)

    :param wz: 1/e^2 radius at distance=z
    :param z: distance where wz is given
    :param wavelength:
    :param n: index of refraction
    :return wos:
    """

    val = np.sqrt(1 - 4 * z ** 2 * wavelength ** 2 / np.pi ** 2 / n ** 2 / wz ** 4)
    wos = wz * np.sqrt(np.array([1 + val, 1 - val]) / 2)

    return wos


def solve_distance(wo: np.ndarray,
                   wz: np.ndarray,
                   wavelength: float,
                   n: Optional[float] = 1.):
    """
    Given a waist, wo, and beam radius, wz, compute the distance
    where w(z) = wz

    :param wo: waist
    :param wz: given beam radius
    :param wavelength:
    :param n:
    :return z: the value returned is always positive, but the focus can be either +/- z away depending on if the beam
     is focusing or diverging
    """
    z = np.sqrt((wz / wo) ** 2 - 1) * np.pi * wo ** 2 / wavelength * n**2
    return z


def fit_beam_radii(radii: np.ndarray,
                   zs: np.ndarray,
                   init_params: Sequence[float],
                   fixed_params: Sequence[bool] = (False, False, True, True),
                   sd: Optional[Sequence[float]] = None,
                   bounds=None):
    """
    Fit beam radii data to determine beam waist, waist position, and optionally wavelength. See fit_model() for more
    details on the various parameters

    :param radii:
    :param zs:
    :param init_params: [wo, zc, wavelength, n]
    :param fixed_params:
    :param sd:
    :param bounds:
    :return results, ffh:
    """

    def get_beam_radius(z, p):
        """
        Compute beam radius (i.e. radius where intensity drops to 1/e^2) a distance z from the beam-waist
        :param z:
        :param p: [wo, zc, wavelength, n]
        :return wz:
        """
        wo, zc, wavelength, n = p
        q = get_q(wo, z, wavelength, zc, n)
        return np.sqrt(q2beam_params(q, wavelength, n)[1])

    def get_wz_jacobian(z,
                        p):
        """
        p = [wo, zc, wavelength, n]
        """
        a = np.sqrt(1 + ((z - p[1]) / (np.pi * p[0] ** 2 / p[2] * p[3])) ** 2)
        b = (z - p[1]) * p[2] / np.pi ** 2 / p[3] ** 2 / p[0] ** 4

        jac = [a - 2 / a * b * (z - p[1]) * p[2],
               -p[0] / a * b * p[2],
               p[0] / a * b * (z - p[1]),
               -p[0] / a * b * (z - p[1]) * p[2] / p[3]]

        return jac

    results = fit_model(radii,
                        lambda p: get_beam_radius(zs, p),
                        init_params,
                        fixed_params=fixed_params,
                        sd=sd,
                        bounds=bounds,
                        model_jacobian=lambda p: get_wz_jacobian(zs, p)
                        )
    def ffh(z): return get_beam_radius(z, results["fit_params"])

    return results, ffh
