"""
Compute electric field for a plane wave incident on a dielectric sphere.
"""
from typing import Union
from collections.abc import Sequence
import numpy as np
from miepython.miepython import _mie_An_Bn
from scipy.special import spherical_jn, spherical_yn, lpmv
from localize_psf.rotation import euler_mat, euler_mat_inv


try:
    import cupy as cp
    from cupyx.scipy.special import lpmv as lpmv_gpu

    # spherical bessel functions of the second kind using upward recursion
    yn_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void my_yn(const int n, const double* x, const int xsize, double* out, double* deriv) {
       int tid = blockDim.x * blockIdx.x + threadIdx.x;

       if (tid < xsize) {    
          // todo: handle x = 0
          //if (x[tid] == 0) {
          //       out[tid * n] = 0; // todo: how to set to infinity?
          //}
          //else  {}  

          // yo and y1
          out[tid * n] = -cos(x[tid]) / x[tid];
          out[tid * n + 1] = (out[tid * n] - sin(x[tid])) / x[tid];

          deriv[tid * n] = (sin(x[tid]) + cos(x[tid]) / x[tid]) / x[tid];
          deriv[tid * n + 1] = out[tid * n] - (2.0) * out[tid * n + 1] / x[tid];

          for (int ii = 2; ii < n; ii++) {
             out[tid * n + ii] = (2.0 * ii - 1.0) * out[tid * n + ii - 1] / x[tid] - out[tid * n + ii - 2]; 
             deriv[tid * n + ii] = out[tid * n + ii - 1] - (ii + 1.0) * out[tid * n + ii] / x[tid];
          }
       }
    }
    """, "my_yn")

    # spherical bessel functions of the first kind using downward recursion and Miller's algorithm
    # todo: prefer to estimate number of terms for given value x and deal with
    # for z < l, upward recursion is unstable, so use downward recursion and Miller's algorithm
    # for z > l, either recursion is stable. Use upward recursion
    # see discussion in e.g. https://doi.org/10.1007/978-3-642-61010-3 chapter "The Calculation of Spherical Bessel Functions and Coulomb Functions"
    # upward recursion is stable for z > l
    jn_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void my_jn(const int n, const int nmax, const double* x, const int xsize, double* out, double* deriv) {
       int tid = blockDim.x * blockIdx.x + threadIdx.x;       

       if (tid < xsize) {
            double j_0, j_1, dj_0;

            // initial values
            if (x[tid] == 0.0) {
                j_0 = 1.0;
                j_1 = 0.0;
                dj_0 = 0.0;
            }
            else {
                j_0 = sin(x[tid]) / x[tid];          
                j_1 = (sin(x[tid]) / x[tid] - cos(x[tid])) / x[tid];
                dj_0 = (cos(x[tid]) - sin(x[tid]) / x[tid]) / x[tid];
            }

            if (x[tid] == 0.0){
                out[tid * nmax] = j_0;
                out[tid * nmax + 1] = j_1;
                deriv[tid * nmax] = dj_0;
                deriv[tid * nmax + 1] = 1.0 / 3.0;
                
                for (int ii = 2; ii < n; ii++) {
                    out[tid * nmax + ii] = 0.0; 
                    deriv[tid * nmax + ii] = 0.0;
                }                 
            }
            else if (fabs(x[tid]) <= 0.1){ // downward recursion doesn't work well here                
                out[tid * nmax] = j_0;                
                deriv[tid * nmax] = dj_0;
                             
                double double_factorial = 1.0;                
                double xpow = 1.0; // x^{ii - 1}
                for (int ii = 1; ii < n; ii++) {
                    double_factorial = double_factorial * (2.0*ii + 1.0);
                    xpow = xpow * x[tid];
                    
                    if (ii == 1) {
                        out[tid * nmax + ii] = j_1;
                    }
                    else {
                        out[tid * nmax + ii] = xpow / double_factorial - \
                                               xpow * x[tid] * x[tid] / double_factorial / (2.0*ii + 3.0) / 2.0 + \
                                               xpow * x[tid] * x[tid] * x[tid] * x[tid] / double_factorial / (2.0*ii + 3.0) / (2.0*ii + 5.0) / 8.0 -\
                                               xpow * x[tid] * x[tid] * x[tid] * x[tid] * x[tid] * x[tid] / double_factorial / (2.0*ii + 3.0) / (2.0*ii + 5.0) / (2.0*ii + 7.0) / 48.0;
                    }
                    
                    deriv[tid * nmax + ii] = out[tid * nmax + ii - 1] - (ii + 1.0) / x[tid] * out[tid * nmax + ii];                    
                }
                                             
            }
            else if (x[tid] > n) { // upward recursion                                                             
                out[tid * nmax] = j_0;
                out[tid * nmax + 1] = j_1;
                deriv[tid * nmax] = dj_0;
                
                // first derivative iteration             
                deriv[tid * nmax + 1] = out[tid * nmax] - (2.0) * out[tid * nmax + 1] / x[tid];
                             
                // remaining iterations
                for (int ii = 2; ii < n; ii++) {
                    out[tid * nmax + ii] = (2.0 * ii - 1.0) * out[tid * nmax + ii - 1] / x[tid] - out[tid * nmax + ii - 2]; 
                    deriv[tid * nmax + ii] = out[tid * nmax + ii - 1] - (ii + 1.0) * out[tid * nmax + ii] / x[tid];
                }
            }
            else { // downward recursion using Miller's algorithm
                // start recursion at different points depending on x value. Choose starting point so any higher bessel functions are negligible here
                int nstart_miller = (int) x[tid] + (nmax - n);
                
                for (int ii=nstart_miller; ii<nmax; ii++) {
                    out[tid * nmax + ii] = 0.0;
                }                
                out[tid * nmax + nstart_miller] = 0.0;
                out[tid * nmax + nstart_miller - 1] = 1.0;                
                                             
                for (int ii=nstart_miller - 2; ii >= 0; ii--) {
                    out[tid * nmax + ii] = (2.0 * ii + 3.0) / x[tid] * out[tid * nmax + ii + 1] - out[tid * nmax + ii + 2];
                    
                    // avoid numerical issues if started with too large n            
                    //if (fabs(out[tid * nmax + ii]) > 1.0E20) {                                           
                    //    for (int jj=ii; jj < nmax; jj++) {
                    //        out[tid * nmax + jj] = 0.0;                            
                    //    }
                    //    out[tid * nmax + ii + 5] = 1.0;                        
                    //    ii += 5;       
                    //}
                }                                                             
                             
                // compute norm using known values of first few spherical bessel functions
                double norm;
                if (j_0 != 0.0 && j_1 != 0.0) {
                    norm = 0.5 * (out[tid * nmax] / j_0 + out[tid * nmax + 1] / j_1);
                }
                else if (j_0 == 0.0 && j_1 != 0.0) {
                    norm = out[tid * nmax + 1] / j_1;
                }
                else {
                    norm = out[tid * nmax] / j_0;
                }

                // normalize and compute derivatives
                for (int ii = 0; ii < nmax; ii++) {
                    out[tid * nmax + ii] /= norm;
                    if (ii == 0) {
                        deriv[tid * nmax + ii] = dj_0;
                    }
                    else {
                        deriv[tid * nmax + ii] = out[tid * nmax + ii - 1] - (ii + 1.0) / x[tid] * out[tid * nmax + ii];
                    }
                }
            }
       }
    }
    """, "my_jn")


    def yn(n: int,
           z: cp.ndarray,
           threads: int = 256):
        """
        Generate a sequence of spherical Bessel functions of the 2nd kind and their derivatives at coordinates z
        using upward recursion

        :param n: number of spherical Bessel functions to generate. The maximum function will be bessel_{n-1}(x)
        :param z: points to evaluate the Bessel functions at
        :param threads:
        :return yn, derivative:
        """
        out = cp.zeros((z.size, n), dtype=cp.double)
        derivative = cp.zeros((z.size, n), dtype=cp.double)
        z = z.astype(cp.double)

        blocks = int(np.ceil(z.size / threads))
        yn_kernel((blocks,), (threads,), (n, z, z.size, out, derivative))

        return out, derivative


    def jn(n: int,
           z: cp.ndarray,
           threads: int = 256,
           overhead: int = 40,):
        """
        Generate a sequence of spherical Bessel functions of the 1st kind and their derivatives at coordinates z
        using downward recursion

        :param n: number of spherical Bessel functions to generate. The maximum function will be bessel_{n-1}(x)
        :param z:
        :param threads:
        :param overhead: number of additional Bessel functions to include in calculation for downward recursion
        :return jn, derivative:
        """
        nmax = n + overhead
        out = cp.zeros((z.size, nmax), dtype=cp.double)
        derivative = cp.zeros((z.size, nmax), dtype=cp.double)
        z = z.astype(cp.double)

        blocks = int(np.ceil(z.size / threads))
        jn_kernel((blocks,), (threads,), (n, nmax, z, z.size, out, derivative))

        return out[:, :n], derivative[:, :n]

except ImportError:
    cp = None
    lpmv_gpu = None
    yn = None
    jn = None


if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


# helper function
def spherical_hn(n: int, z: np.ndarray) -> (np.ndarray, np.ndarray):
    hn = spherical_jn(n, z) + 1j * spherical_yn(n, z)
    dhn = spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True)
    return hn, dhn


def mie_efield(wavelength: float,
               no: float,
               radius: float,
               n_sphere: float,
               dxy: float,
               esize: Sequence[int, int],
               dz: float,
               beam_theta: float = 0.,
               beam_phi: float = 0.,
               beam_psi: float = 0.,
               use_gpu: bool = False,
               **kwargs) -> (array, array, array, array):
    """
    Use miepython to compute multipolar coefficients, and use either scipy special functions
    or GPU accelerated custom CuPy kernels to calculate fields. There are many good references
    for computing scattered electric fields from spherical particles using Mie theory.
    This code follows "Absorption and scattering of light by small particles" Bohren, 1983,
    https://doi.org/10.1364/AO.39.005117. Bohren uses the phase convention exp(ikx - iwt),
    as described in the start of ch 3.

    :param wavelength: wavelength of light
    :param no: refractive index of background medium
    :param radius: radius of sphere
    :param n_sphere: refractive index of sphere
    :param dxy: pixel size
    :param esize: (ny, nx) size of grid to calculate electric field on
    :param dz: distance away from sphere to calculate field
    :param beam_theta: euler angles describing beam incident direction
    :param beam_phi: euler angles describing beam incident direction
    :param beam_psi: euler angles describing beam incident direction
    :param use_gpu: whether to compute electric field arrays on the GPU
    :return exyz, exyz_o, exyz_in, exyz_in_o:
    """

    if cp and use_gpu:
        xp = cp
    else:
        xp = np

    k = 2 * np.pi / wavelength * no
    ny, nx = esize
    # #######################
    # define coordinates
    # #######################
    # plane of interest in cartesian coordinates
    xx_o, yy_o = np.meshgrid((xp.arange(nx) - (nx // 2)) * dxy,
                             (xp.arange(ny) - (ny // 2)) * dxy)
    zz_o = xp.zeros_like(xx_o) + dz

    # convert to coordinates in which incident beam
    # is x-polarized and travels along z-direction
    mat_inv = euler_mat_inv(beam_phi, beam_theta, beam_psi)
    mat = euler_mat(beam_phi, beam_theta, beam_psi)

    xx = mat_inv[0, 0] * xx_o + mat_inv[0, 1] * yy_o + mat_inv[0, 2] * zz_o
    yy = mat_inv[1, 0] * xx_o + mat_inv[1, 1] * yy_o + mat_inv[1, 2] * zz_o
    zz = mat_inv[2, 0] * xx_o + mat_inv[2, 1] * yy_o + mat_inv[2, 2] * zz_o

    # convert to spherical coordinates
    r = xp.sqrt(xx**2 + yy**2 + zz**2)
    theta = xp.arctan2(np.sqrt(xx**2 + yy**2), zz)
    phi = xp.arctan2(yy, xx)

    # incident beam
    ex_in = xp.exp(1j * k * zz)
    ey_in = xp.zeros_like(ex_in)
    ez_in = xp.zeros_like(ex_in)

    # #######################
    # scattering coefficients
    # use relative values for alpha and n_sphere
    # https://miepython.readthedocs.io/en/latest/01_basics.html
    # #######################
    # MiePython uses convention Im(n) < 0, so take conjugate
    an, bn = _mie_An_Bn(np.conj(n_sphere) / no, k * radius)
    an = an.conj()
    bn = bn.conj()

    # this gives same results
    # an, bn = Mie_ab(n_sphere / no, k * radius)

    an = xp.asarray(an[:, None, None])
    bn = xp.asarray(bn[:, None, None])

    ls = xp.arange(len(an)) + 1
    # hankel functions and derivatives we will need
    with np.errstate(invalid="ignore"):

        if not use_gpu:
            h_ns = xp.zeros((len(ls),) + xx.shape, dtype=complex)
            dh_ns = xp.zeros_like(h_ns)
            pi_ns = xp.zeros_like(h_ns)
            for ii in range(len(ls)):
                # todo: in my testing, this takes ~10x longer than anything else
                h_ns[ii], dh_ns[ii] = spherical_hn(ls[ii], k * r)

                # Bohren 4.46
                pi_ns[ii] = lpmv(1, ls[ii], xp.cos(theta)) / xp.sin(theta)

                # correct behavior at zero
                # todo: probably a better way if e.g. have derivative
                th_zero = theta == 0
                th_eff = 1e-4
                pi_ns[ii][th_zero] = lpmv(1, ls[ii], xp.cos(th_eff)) / xp.sin(th_eff)
        else:
            jns, djns = jn(len(ls) + 1, k * r, **kwargs)
            yns, dyns = yn(len(ls) + 1, k * r, **kwargs)

            jns = jns.transpose()[1:].reshape((len(ls),) + xx.shape)
            yns = yns.transpose()[1:].reshape((len(ls),) + xx.shape)
            djns = djns.transpose()[1:].reshape((len(ls),) + xx.shape)
            dyns = dyns.transpose()[1:].reshape((len(ls),) + xx.shape)

            h_ns = jns + 1j * yns
            dh_ns = djns + 1j * dyns

            pi_ns = xp.zeros_like(h_ns)
            for ii in range(len(ls)):
                pi_ns[ii] = lpmv_gpu(1, ls[ii], xp.cos(theta)) / xp.sin(theta)

                th_zero = theta == 0
                th_eff = 1e-4
                pi_ns[ii][th_zero] = lpmv_gpu(1, ls[ii], xp.cos(th_eff)) / xp.sin(th_eff)

    # compute tau_ns from Bohren 4.47
    tau_ns = ls[:, None, None] * xp.cos(theta) * pi_ns - \
             (ls[:, None, None] + 1) * xp.concatenate((xp.zeros((1, ny, nx)), pi_ns[:-1]), axis=0)

    # #######################
    # compute field
    # #######################

    # multiple components
    # Bohren 4.37 / just after 4.40
    en = (1j ** ls * (2*ls + 1) / (ls * (ls + 1)))[:, None, None]
    # Bohren 4.50
    m_o1n_r = 0
    m_o1n_th = xp.cos(phi) * pi_ns * h_ns
    m_o1n_ph = -xp.sin(phi) * tau_ns * h_ns
    n_e1n_r = (ls * (ls + 1))[:, None, None] * xp.cos(phi) * np.sin(theta) * pi_ns * h_ns / (k * r)
    n_e1n_th = xp.cos(phi) * tau_ns * (h_ns + (k*r) * dh_ns) / (k*r)
    n_e1n_ph = -xp.sin(phi) * pi_ns * (h_ns + (k*r) * dh_ns) / (k*r)

    # full field
    # Bohren 4.45
    e_r = xp.sum(en * (1j * an * n_e1n_r - bn * m_o1n_r), axis=0)
    e_th = xp.sum(en * (1j * an * n_e1n_th - bn * m_o1n_th), axis=0)
    e_ph = xp.sum(en * (1j * an * n_e1n_ph - bn * m_o1n_ph), axis=0)

    # #######################
    # convert efield from spherical to cartesian vectors
    # #######################
    sph_to_xyz = xp.stack([
                  xp.stack([xp.sin(theta) * xp.cos(phi), xp.cos(theta) * xp.cos(phi), -xp.sin(phi)], axis=0),
                  xp.stack([xp.sin(theta) * xp.sin(phi), xp.cos(theta) * xp.sin(phi),  xp.cos(phi)], axis=0),
                  xp.stack([xp.cos(theta),              -xp.sin(theta),                xp.zeros_like(theta)], axis=0),
                           ], axis=0)

    e_x = sph_to_xyz[0, 0] * e_r + sph_to_xyz[0, 1] * e_th + sph_to_xyz[0, 2] * e_ph
    e_y = sph_to_xyz[1, 0] * e_r + sph_to_xyz[1, 1] * e_th + sph_to_xyz[1, 2] * e_ph
    e_z = sph_to_xyz[2, 0] * e_r + sph_to_xyz[2, 1] * e_th + sph_to_xyz[2, 2] * e_ph

    # #######################
    # convert vector efields beam back to initial coordinates
    # #######################
    e_xo = mat[0, 0] * e_x + mat[0, 1] * e_y + mat[0, 2] * e_z
    e_yo = mat[1, 0] * e_x + mat[1, 1] * e_y + mat[1, 2] * e_z
    e_zo = mat[2, 0] * e_x + mat[2, 1] * e_y + mat[2, 2] * e_z

    exo_in = mat[0, 0] * ex_in + mat[0, 1] * ey_in + mat[0, 2] * ez_in
    eyo_in = mat[1, 0] * ex_in + mat[1, 1] * ey_in + mat[1, 2] * ez_in
    ezo_in = mat[2, 0] * ex_in + mat[2, 1] * ey_in + mat[2, 2] * ez_in

    # scattered fields
    # todo: why do I need a minus sign here?
    exyz = -xp.stack((e_x, e_y, e_z), axis=0)
    exyz_o = -xp.stack((e_xo, e_yo, e_zo), axis=0)
    # input fields
    exyz_in = xp.stack((ex_in, ey_in, ez_in), axis=0)
    exyz_in_o = xp.stack((exo_in, eyo_in, ezo_in), axis=0)

    return exyz, exyz_o, exyz_in, exyz_in_o
