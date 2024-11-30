"""
Tools for solving inverse problems using accelerated proximal gradient methods
"""

from warnings import catch_warnings, simplefilter
import numpy as np
from time import perf_counter
from random import sample
from typing import Union, Optional
from collections.abc import Sequence
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_chambolle


try:
    import cupy as cp
except ImportError:
    cp = None


try:
    from cucim.skimage.restoration import denoise_tv_chambolle as denoise_tv_chambolle_gpu
except ImportError:
    denoise_tv_chambolle_gpu = None

try:
    from cupyx.scipy.ndimage import median_filter as median_filter_gpu
except ImportError:
    median_filter_gpu = None


if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


def to_cpu(m: array) -> np.ndarray:
    """
    Ensure array is CPU/NumPy

    :param m:
    :return m_cpu:
    """
    if cp and isinstance(m, cp.ndarray):
        return m.get()
    else:
        return m


def soft_threshold(tau: float,
                   x: array) -> array:
    """
    Soft-threshold function, which is the proximal operator for the LASSO (L1 regularization) problem

    .. math::

      \\text{prox}_t(x) = \\text{argmin}_y \\left\\{ \\frac{1}{2} \\left|x - y \\right|_2^2 + t  \\left|y \\right|_1 \\right\\}

    :param tau: softmax parameter
    :param x: array to take softmax of
    :return x_out:
    """
    x_out = x.copy()
    x_out[x > tau] -= tau
    x_out[x < -tau] += tau
    x_out[abs(x) <= tau] = 0

    return x_out

def haar_wavelet_transform_3d(x: array, use_gpu: bool = True) -> Sequence[array]:
    """Compute the 3D Haar wavelet transform.

    Returns a tuple of (approximation, details along z, y, x, yz, xz, xy, xyz).

    :param x: 3D image to decompose into haar wavelets.
    :param use_gpu: Use GPU (Default = True).
    :return: Haar wavelet decomposition
    """

    xp = cp if use_gpu and cp else np

    scale = xp.sqrt(8)
    avg = (
        x[::2, ::2, ::2]
        + x[1::2, ::2, ::2]
        + x[::2, 1::2, ::2]
        + x[::2, ::2, 1::2]
        + x[1::2, 1::2, ::2]
        + x[1::2, ::2, 1::2]
        + x[::2, 1::2, 1::2]
        + x[1::2, 1::2, 1::2]
    ) / scale

    diff_z = (x[::2, ::2, ::2] - x[1::2, ::2, ::2]) / scale
    diff_y = (x[::2, ::2, ::2] - x[::2, 1::2, ::2]) / scale
    diff_x = (x[::2, ::2, ::2] - x[::2, ::2, 1::2]) / scale
    diff_yz = (x[::2, ::2, ::2] - x[1::2, 1::2, ::2]) / scale
    diff_xz = (x[::2, ::2, ::2] - x[1::2, ::2, 1::2]) / scale
    diff_xy = (x[::2, ::2, ::2] - x[::2, 1::2, 1::2]) / scale
    diff_xyz = (x[::2, ::2, ::2] - x[1::2, 1::2, 1::2]) / scale

    return avg, diff_z, diff_y, diff_x, diff_yz, diff_xz, diff_xy, diff_xyz


def inverse_haar_wavelet_transform_3d(
    avg: array,
    diff_z: array,
    diff_y: array,
    diff_x: array,
    diff_yz: array,
    diff_xz: array,
    diff_xy: array,
    diff_xyz: array,
    use_gpu: bool = True,
) -> array:
    """Compute the inverse 3D Haar wavelet transform.

    Reconstructs the original 3D signal from its wavelet coefficients.

    :param avg: Average term of Haar decomposition.
    :param diff_z: difference in z term of Haar wavelet decomposition.
    :param diff_y: difference in y term of Haar wavelet decomposition.
    :param diff_x: difference in x term of Haar wavelet decomposition.
    :param diff_yz: difference in yz term of Haar wavelet decomposition.
    :param diff_xz: difference in xz term of Haar wavelet decomposition.
    :param diff_xy: difference in xy term of Haar wavelet decomposition.
    :param diff_xyz: difference in xyz term of Haar wavelet decomposition.
    :param use_gpu: Use GPU (Default = True).
    :return x: inverse Haar wavelet transform. 
    """

    xp = cp if use_gpu and cp else np

    out_shape = (avg.shape[0] * 2, avg.shape[1] * 2, avg.shape[2] * 2)
    x = xp.zeros(out_shape, dtype=avg.dtype)

    x[::2, ::2, ::2] = (
        avg + diff_z + diff_y + diff_x + diff_yz + diff_xz + diff_xy + diff_xyz
    )
    x[1::2, ::2, ::2] = (
        avg - diff_z + diff_y + diff_x - diff_yz - diff_xz + diff_xy - diff_xyz
    )
    x[::2, 1::2, ::2] = (
        avg + diff_z - diff_y + diff_x - diff_yz + diff_xz - diff_xy - diff_xyz
    )
    x[1::2, 1::2, ::2] = (
        avg - diff_z - diff_y + diff_x + diff_yz - diff_xz - diff_xy + diff_xyz
    )
    x[::2, ::2, 1::2] = (
        avg + diff_z + diff_y - diff_x - diff_yz - diff_xz + diff_xy - diff_xyz
    )
    x[1::2, ::2, 1::2] = (
        avg - diff_z + diff_y - diff_x + diff_yz + diff_xz - diff_xy + diff_xyz
    )
    x[::2, 1::2, 1::2] = (
        avg + diff_z - diff_y - diff_x + diff_yz - diff_xz - diff_xy + diff_xyz
    )
    x[1::2, 1::2, 1::2] = (
        avg - diff_z - diff_y - diff_x - diff_yz + diff_xz + diff_xy - diff_xyz
    )

    return x


def parallel_tv_prox(img: array, lambda_reg: float, use_gpu: bool = True) -> array:
    """Fast parallel proximal algorithm for anisotropic total variation.

    :param img: 3D image to denoise.
    :param lambda_reg: Regularization parameter.
    :param use_gpu: Use GPU (Default = True).
    :return x: Denoised 3D image.
    """
    xp = cp if use_gpu and cp else np

    avg, diff_z, diff_y, diff_x, diff_yz, diff_xz, diff_xy, diff_xyz = (
        haar_wavelet_transform_3d(img)
    )

    # Shrinkage on the difference components
    diff_z = soft_threshold(xp.sqrt(2) * 8 * lambda_reg, diff_z)
    diff_y = soft_threshold(xp.sqrt(2) * 8 * lambda_reg, diff_y)
    diff_x = soft_threshold(xp.sqrt(2) * 8 * lambda_reg, diff_x)
    diff_yz = soft_threshold(xp.sqrt(2) * 8 * lambda_reg, diff_yz)
    diff_xz = soft_threshold(xp.sqrt(2) * 8 * lambda_reg, diff_xz)
    diff_xy = soft_threshold(xp.sqrt(2) * 8 * lambda_reg, diff_xy)
    diff_xyz = soft_threshold(xp.sqrt(2) * 8 * lambda_reg, diff_xyz)

    # Inverse Haar transform
    x = inverse_haar_wavelet_transform_3d(
        avg, diff_z, diff_y, diff_x, diff_yz, diff_xz, diff_xy, diff_xyz
    )

    return x

def tv_prox(x: array,
            tau: float,
            max_num_iter: int = 200,
            eps: float = 2e-4) -> array:
    """
    Apply TV proximal operator to x. Helper function which runs on CPU or GPU

    .. math::

      \\text{prox}_t(x) &= \\text{argmin}_y \\left\\{ \\frac{1}{2} \\left|x - y \\right|_2^2 + t  \\text{TV}(y) \\right\\}

      \\text{TV}(y) &= \\sum_{ij} \\sqrt{ \\left(y_{i+1,j} - y_{i,j}\\right)^2 + \\left(y_{i, j+1} - y_{i, j}\\right)^2}

    :param x: array to be denoised
    :param tau: strength of TV term compared with data fidelity term
    :param max_num_iter: see denoise_tv_chambolle() for more details. The default values used here
     are the same as in the cucim version of this function.
    :param eps:
    :return x_tv:
    """

    if cp and isinstance(x, cp.ndarray):
        if not denoise_tv_chambolle_gpu:
            raise ValueError("array x is a CuPy array, but a GPU compatible TV implementation was not"
                             " imported successfully. Convert x to NumPy array or correctly install GPU compatible TV")

        tv = denoise_tv_chambolle_gpu
    else:
        tv = denoise_tv_chambolle

    return tv(x, weight=tau, channel_axis=None, max_num_iter=max_num_iter, eps=eps)


def median_prox(x: array,
                size: Sequence[int]):
    """

    :param x:
    :param size:
    :return:
    """
    if cp and isinstance(x, cp.ndarray):
        if not median_filter_gpu:
            raise ValueError("array x is a CuPy array, but a GPU compatible median filter implementation was not"
                             " imported successfully.")
        f = median_filter_gpu
    else:
        f = median_filter

    return f(x, size=size)


class Optimizer:
    def __init__(self,
                 n_samples: int,
                 prox_parameters: Optional[dict] = None):
        self.n_samples = int(n_samples)
        self.prox_parameters = prox_parameters

    def fwd_model(self,
                  x: array,
                  inds: Optional[Sequence[int]] = None) -> array:
        """
        Apply the forward model to x, assuming the loss function can be interpreted as a difference
        between the forward model and the measured result

        :param x:
        :param inds:
        :return:
        """
        pass

    def fwd_model_adjoint(self,
                          y: array,
                          inds: Optional[Sequence[int]] = None) -> array:
        """
        Adjoint operator to forward model.

        :param y:
        :param inds:
        :return:
        """
        pass

    def cost(self,
             x: array,
             inds: Optional[Sequence[int]] = None) -> array:
        """
        Compute the data penalty part of cost/loss function at x.
        Compute only those components specified by inds

        :param x:
        :param inds:
        :return:
        """
        pass

    def gradient(self,
                 x: array,
                 inds: Optional[Sequence[int]] = None) -> array:
        """
        Compute the gradient of the loss function at x for the loss functions specified by inds

        :param x:
        :param inds:
        :return:
        """
        pass

    def test_gradient(self,
                      x: array,
                      jind: int = 0,
                      inds: Optional[Sequence[int]] = None,
                      dx: float = 1e-5) -> (array, array):
        """
        Numerically test the gradient computation at a single coordinate of x.

        :param x: point to compute gradient at
        :param jind: 1D index into x to compute gradient at
        :param inds: samples to compute gradient at
        :param dx: gradient step size
        :return grad, grad_numerical:
        """

        use_gpu = cp and isinstance(x, cp.ndarray)
        if use_gpu:
            xp = cp
        else:
            xp = np

        # in case x is multi dimensional, unravel
        xind = np.unravel_index(jind, x.shape)

        # compute gradient numerically
        x1 = xp.array(x, copy=True)
        x1[xind] -= 0.5 * dx
        c1 = self.cost(x1, inds=inds)

        x2 = xp.array(x, copy=True)
        x2[xind] += 0.5 * dx
        c2 = self.cost(x2, inds=inds)

        gn = (c2 - c1) / dx

        # if x is complex, compute "complex gradient"
        if xp.iscomplexobj(x):
            x1c = xp.array(x, copy=True)
            x1c[xind] -= 0.5 * dx * 1j
            c1c = self.cost(x1c, inds=inds)

            x2c = xp.array(x, copy=True)
            x2c[xind] += 0.5 * dx * 1j
            c2c = self.cost(x2c, inds=inds)

            gn = gn + (c2c - c1c) / dx * 1j

        # compute gradient
        slices = [slice(None)] + [slice(i, i+1) for i in xind]
        g = self.gradient(x, inds=inds)[tuple(slices)].ravel()

        return g, gn

    def prox(self,
             x: array,
             step: float) -> array:
        """
        Apply the proximal operator

        :param x:
        :param step:
        :return prox(x):
        """
        pass

    def guess_step(self,
                   x: Optional[array] = None) -> float:
        """
        Guess an appropriate step-size for gradient descent

        :param x:
        :return step_guess:
        """
        pass

    def _lipschitz_condition_violated(self,
                                      step: float,
                                      y: array,
                                      x: array,
                                      cx: float,
                                      gx: array,
                                      inds):
        xp = cp if cp and isinstance(x, cp.ndarray) else np
        cy = xp.mean(self.cost(y, inds=inds), axis=0)
        return (cy > cx + xp.sum(gx.real * (y - x).real +
                                 gx.imag * (y - x).imag) +
                0.5 / step * xp.linalg.norm(y - x)**2)

    def run(self,
            x_start: array,
            step: float,
            max_iterations: int = 100,
            use_fista: bool = True,
            n_batch: int = 1,
            compute_batch_grad_parallel: bool = True,
            verbose: bool = False,
            compute_cost: bool = False,
            compute_all_costs: bool = False,
            line_search_iter_limit: Optional[int] = 0,
            line_search_factor: float = 0.5,
            restart_line_search: bool = False,
            stop_on_nan: bool = True,
            xtol: float = 0.0,
            ftol: float = 0.0,
            gtol: float = 0.0,
            print_newline: bool = False,
            label: str = "",
            **kwargs) -> dict:

        """
        Proximal gradient descent on model starting from initial guess

        :param x_start: initial guess
        :param step: step-size
        :param max_iterations: maximum number of iterations
        :param use_fista: use momentum term in update step to accelerate convergence
        :param n_batch: number of samples to average gradient over at each iteration. If None, use all samples
        :param compute_batch_grad_parallel: compute all gradients in a single batch in a vectorized way
          advantageous for speed, but requires more memory
        :param verbose: print iteration info
        :param compute_cost: optionally compute and store the cost. This can make optimization slower
        :param compute_all_costs: compute costs for all samples, even those not in the current batch
        :param line_search_iter_limit: only run line search if loop iteration number is less than this number.
         To never run a line search, set to 0 (default). To always run a line search, set to None.
        :param line_search_factor: factor to shrink step-size if line-search determines step too large
        :param restart_line_search: if true, restart line search from initial value at each iteration. If false,
          restart line search from step-size determined at previous iteration
        :param stop_on_nan: stop if there are NaN's in x
        :param xtol: When norm(x[t] - x[t-1]) / norm(x[0]) < xtol, stop iteration
        :param ftol: stop iterating when cost function relative change < ftol. Not yet implemented
        :param gtol: stop iterating when gradient is less thn gtol. Not yet implemented
        :param print_newline:
        :param label: if verbose, print this information
        :return results: dictionary containing results
        """

        use_gpu = cp and isinstance(x_start, cp.ndarray)
        if use_gpu:
            xp = cp
            mempool = cp.get_default_memory_pool()
        else:
            xp = np

        if n_batch is None or n_batch > self.n_samples:
            n_batch = self.n_samples

        # ###################################
        # initialize
        # ###################################
        results = {"n_samples": self.n_samples,
                   "step_size": step,
                   "niterations": max_iterations,
                   "use_fista": use_fista,
                   "use_gpu": use_gpu,
                   "x_init": np.array(to_cpu(x_start), copy=True),
                   "prox_parameters": self.prox_parameters,
                   "stop_condition": "ok"
                   }

        timing = {"iteration": np.zeros(0),
                  "grad": np.zeros(0),
                  "prox":  np.zeros(0),
                  "update":  np.zeros(0),
                  "cost":  np.zeros(0),
                  }

        tstart = perf_counter()
        costs = np.zeros((max_iterations + 1, self.n_samples)) * np.nan
        steps = np.ones(max_iterations) * step
        xdiffs = np.ones(max_iterations) * np.nan
        line_search_iters = np.ones(max_iterations, dtype=int)
        q_last = 1
        x = xp.array(x_start, copy=True)

        for ii in range(max_iterations):
            # select batch indices
            inds = sample(range(self.n_samples), n_batch)

            if stop_on_nan:
                if xp.any(xp.isnan(x)):
                    results["stop_condition"] = "stopped on NaN"
                    break

            # ###################################
            # proximal gradient descent
            # ###################################

            ls_iters = 0
            if line_search_iter_limit is not None and ii >= line_search_iter_limit:
                # ###################################
                # compute cost
                # ###################################
                tstart_err = perf_counter()

                if compute_cost:
                    if compute_all_costs:
                        costs[ii] = to_cpu(self.cost(x))
                    else:
                        costs[ii, inds] = to_cpu(self.cost(x, inds=inds))

                timing["cost"] = np.concatenate((timing["cost"], np.array([perf_counter() - tstart_err])))

                # ###################################
                # compute gradient
                # ###################################
                tstart_grad = perf_counter()

                if compute_batch_grad_parallel:
                    x -= steps[ii] * xp.mean(self.gradient(x, inds=inds), axis=0)
                else:
                    grad_mean = 0
                    for inow in inds:
                        grad_mean += self.gradient(x, inds=[inow])[0] / n_batch

                    x -= steps[ii] * grad_mean

                timing["grad"] = np.concatenate((timing["grad"], np.array([perf_counter() - tstart_grad])))

                # ###################################
                # prox operator
                # ###################################
                tstart_prox = perf_counter()
                y = self.prox(x, steps[ii])

                timing["prox"] = np.concatenate((timing["prox"], np.array([perf_counter() - tstart_prox])))

            else:
                # cost at current point
                # always grab costs, since computing for line-search
                tstart_err = perf_counter()

                if compute_all_costs:
                    c_all = self.cost(x)
                    costs[ii] = to_cpu(c_all)
                    cx = xp.mean(c_all[inds], axis=0)
                else:
                    c_now = self.cost(x, inds=inds)
                    costs[ii, inds] = to_cpu(c_now)
                    cx = xp.mean(c_now, axis=0)

                timing["cost"] = np.concatenate((timing["cost"], np.array([perf_counter() - tstart_err])))

                # gradient at current point
                tstart_grad = perf_counter()
                if compute_batch_grad_parallel:
                    gx = xp.mean(self.gradient(x, inds=inds), axis=0)
                else:
                    gx = 0
                    for inow in inds:
                        gx += self.gradient(x, inds=[inow])[0] / n_batch

                timing["grad"] = np.concatenate((timing["grad"], np.array([perf_counter() - tstart_grad])))

                # line-search
                tstart_prox = perf_counter()

                # initialize line-search
                ls_iters += 1
                if ii != 0:
                    if restart_line_search:
                        steps[ii] = step
                    else:
                        steps[ii] = steps[ii - 1]

                y = self.prox(x - steps[ii] * gx, steps[ii])

                # reduce step until we don't violate Lipschitz continuous gradient condition
                while self._lipschitz_condition_violated(steps[ii], y, x, cx, gx, inds):
                    steps[ii] *= line_search_factor
                    y = self.prox(x - steps[ii] * gx, steps[ii])
                    ls_iters += 1

                # set subsequent step-sizes
                if ii == (line_search_iter_limit - 1):
                    steps[ii:] = steps[ii]

                # not exclusively prox
                timing["prox"] = np.concatenate((timing["prox"], np.array([perf_counter() - tstart_prox])))

            line_search_iters[ii] = ls_iters

            # ###################################
            # stop conditions
            # ###################################
            if ii == 0:
                ynorm = xp.sqrt(xp.sum(xp.abs(y)**2))
            else:
                xdiffs[ii] = to_cpu(xp.sqrt(xp.sum(xp.abs(y - y_last)**2)) / ynorm)

            if ftol != 0:
                raise NotImplementedError("ftol != 0 not implemented")

            if gtol != 0:
                raise NotImplementedError("gtol != 0 not implemented")

            stop = xdiffs[ii] < xtol or ii == (max_iterations - 1)

            # ###################################
            # update step
            # ###################################
            tstart_update = perf_counter()

            q = 0.5 * (1 + np.sqrt(1 + 4 * q_last ** 2))
            if ii == 0 or not use_fista or stop:
                # must copy, otherwise y_last will be updated with x, and xtol comparison will fail
                x = xp.array(y, copy=True)
            else:
                x = y + (q_last - 1) / q * (y - y_last)

            # update for next iteration
            q_last = q
            y_last = y

            timing["update"] = np.concatenate((timing["update"], np.array([perf_counter() - tstart_update])))
            timing["iteration"] = np.concatenate((timing["iteration"], np.array([perf_counter() - tstart_err])))

            # print information
            if verbose:
                with catch_warnings():
                    simplefilter("ignore", category=RuntimeWarning)

                    status = f"{label:s}iteration {ii + 1:d}/{max_iterations:d}," \
                             f" cost={np.nanmean(costs[ii]):.3g}," \
                             f" diff={xdiffs[ii]:.3g}," \
                             f" step={steps[ii]:.3g}," \
                             f" lsearch #={line_search_iters[ii]:d}," \
                             f" grad={timing['grad'][ii]:.3f}s," \
                             f" prox={timing['prox'][ii]:.3f}s," \
                             f" cost={timing['cost'][ii]:.3f}s," \
                             f" iter={timing['iteration'][ii]:.3f}s," \
                             f" total={perf_counter() - tstart:.3f}s"

                if use_gpu:
                    status += f", GPU={mempool.used_bytes()/1e9:.3}GB"

                if print_newline or stop:
                    end = "\n"
                else:
                    end = "\r"

                print(status, end=end)

            # ###################################
            # loop termination
            # ###################################
            if stop:
                break

        # compute final cost
        if compute_cost:
            if compute_all_costs:
                costs[ii + 1] = to_cpu(self.cost(x))
            else:
                costs[ii + 1, inds] = to_cpu(self.cost(x, inds=inds))

        # store results
        results.update({"timing": timing,
                        "costs": costs,
                        "steps": steps,
                        "xdiffs": xdiffs,
                        "line_search_iterations": line_search_iters,
                        "x": x})

        return results
