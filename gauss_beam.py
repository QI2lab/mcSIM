"""
Tools for dealing with Gaussian beams, including fitting
"""
import numpy as np
from scipy.integrate import quad
import fit

def beam_int(r, w):
    """
    Intensity of gaussian beam distance r from the center
    :param r: radial coordinate
    :param w: beam-waist (i.e. 1/e^2 distance)
    :return:
    """
    return np.exp(-2*r**2 / w**2) / (2 * np.pi * (w/2)**2)


def power_integral(number_of_waists):
    """
    Compute beam power contained within a certain number of waist lengths
    :param number_of_waists:
    :return:
    """
    int, err = quad(lambda r: beam_int(r, 1) * r, 0, number_of_waists)
    return 2*np.pi * int

def wo(wd, d, wavelen):
    """
    Compute waist given value 1/e^2 radius at distance d. Note, there are two possible solutions

    :param wd:
    :param d:
    :param wavelen:

    :return:
    """
    val = np.sqrt(1 - 4 * d ** 2 * wavelen ** 2 / np.pi ** 2 / wd ** 4)
    return wd * np.sqrt(np.array([1 + val, 1 - val]) / 2)

def wz(z, p):
    """
    Compute beam radius (i.e. radius where intensity drops to 1/e^2) a distance z from the beam-waist
    :param z:
    :param p: [wo, zc, wavelength]
    :return:
    """
    return p[0] * np.sqrt(1 + ((z - p[1]) / (np.pi * p[0]**2 / p[2]))**2)

def wz_jacobian(z, p):
    """
    jacobian of wz
    :param z:
    :param p:
    :return:
    """
    a = np.sqrt(1 + ((z - p[1]) / (np.pi * p[0] ** 2 / p[2])) ** 2)
    b = (z - p[1]) * p[2] / np.pi**2 / p[0]**4

    jac = [a - 2 / a * b * (z - p[1]) * p[2],
           -p[0] / a * b * p[2],
           p[0] / a * b * (z - p[1])]

    return jac


def fit_beam_radii(radii, zs, init_params, fixed_params=(False, False, True), sd=None, bounds=None):
    """
    Fit beam radii data to determine beam waist, waist position, and optionally wavelength. See fit_model() for more
    details on the various parameters

    :param radii:
    :param zs:
    :param init_params: [wo, zc, wavelength]
    :param fixed_params:
    :param sd:
    :param bounds:

    :return results:
    :return ffh:
    """
    results = fit.fit_model(radii, lambda p: wz(zs, p), init_params, fixed_params=fixed_params, sd=sd,
                            bounds=bounds, model_jacobian=lambda p: wz_jacobian(zs, p))
    def ffh(z): return wz(z, results["fit_params"])

    return results, ffh