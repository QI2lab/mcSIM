"""
Tools for solving inverse problems using accelerated proximal gradient methods
"""

import numpy as np
import time
import random
from typing import Union
from skimage.restoration import denoise_tv_chambolle

_gpu_available = True
try:
    import cupy as cp
except:
    cp = np
    _gpu_available = False


_gpu_tv_available = True
try:
    from cucim.skimage.restoration import denoise_tv_chambolle as denoise_tv_chambolle_gpu
except:
    denoise_tv_chambolle_gpu = None

array = Union[np.ndarray, cp.ndarray]

def _to_cpu(m):
    """
    Ensure array is CPU/NumPy
    :param m:
    :return m_cpu:
    """
    if _gpu_available and isinstance(m, cp.ndarray):
        return m.get()
    else:
        return m


def soft_threshold(t: float,
                   x: array) -> array:
    """
    Soft-threshold function, which is the proximal operator for the LASSO (L1 regularization) problem

    x* = argmin{ 0.5 * |x - y|_2^2 + t |x - y|_1}

    :param t: softmax parameter
    :param x: array to take softmax of
    :return x_out:
    """
    x_out = x.copy()
    x_out[x > t] -= t
    x_out[x < -t] += t
    x_out[abs(x) <= t] = 0

    return x_out


def tv_prox(x: array,
            tau: float):
    """
    TV helper function which runs on CPU or GPU

    :param x:
    :param tau:
    :return x_tv:
    """

    if isinstance(x, cp.ndarray) and _gpu_available:
        if not _gpu_tv_available:
            raise ValueError("array x is a CuPy array, but a GPU compatible TV implementation was not"
                             " imported successfully. Convert x to NumPy array or correctly install GPU compatible TV")

        tv = denoise_tv_chambolle_gpu
    else:
        tv = denoise_tv_chambolle

    return tv(x, weight=tau, channel_axis=None)


class Optimizer():
    def __init__(self):
        self.n_samples = None
        self.prox_parameters = {}

    def fwd_model(self, x, inds=None):
        pass

    def fwd_model_adjoint(self, y, inds=None):
        pass

    def cost(self, x, inds=None):
        pass

    def gradient(self, x, inds=None):
        pass

    def test_gradient(self, x, jind=0, inds=None, dx=1e-5):
        """

        :param x: point to compute gradient at
        :param jind: 1D index into x to compute gradient at
        :param inds: samples to compute gradient at
        :param dx: gradient step size
        :return grad, grad_numerical:
        """

        use_gpu = isinstance(x, cp.ndarray) and _gpu_available
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

    def prox(self, x, step):
        pass

    def guess_step(self, x):
        pass

    def run(self,
            x_start: array,
            step: float,
            max_iterations: int = 100,
            use_fista: bool = True,
            stochastic_descent: bool = True,
            nmax_stochastic_descent: int = np.inf,
            verbose: bool = False,
            compute_cost: bool = False,
            compute_all_costs: bool = False,
            line_search: bool = False,
            line_search_factor: float = 0.5,
            stop_on_nan: bool = True,
            xtol: float = 1e-8,
            **kwargs) -> dict:

        """
        Proximal gradient descent on model starting from initial guess

        :param x_start: initial guess
        :param step: step-size
        :param max_iterations:
        :param use_fista:
        :param stochastic_descent: either select random subset of samples to use at each time step or use
          average of all samples at each time-step
        :param nmax_stochastic_descent: maximum size of random subset
        :param verbose: print iteration info
        :param compute_cost: optionally compute and store the cost. This can make optimization slower
        :param line_search: use line search to shrink step-size as necessary
        :param line_search_factor: factor to shrink step-size if line-search determines step too large
        :param xtol: TODO: stop when change in x is small
        :return results: dictionary containing results
        """

        use_gpu = isinstance(x_start, cp.ndarray) and _gpu_available
        if use_gpu:
            xp = cp
        else:
            xp = np

        if nmax_stochastic_descent is None or nmax_stochastic_descent > self.n_samples:
            nmax_stochastic_descent = self.n_samples

        # ###################################
        # initialize
        # ###################################
        results = {"n_samples": self.n_samples,
                   "step_size": step,
                   "niterations": max_iterations,
                   "use_fista": use_fista,
                   "use_gpu": use_gpu,
                   "x_init": _to_cpu(xp.array(x_start, copy=True)),
                   "prox_parameters": self.prox_parameters,
                   "stop_condition": "ok"
                   }

        timing = {"iteration": np.zeros(0),
                  "grad": np.zeros(0),
                  "prox":  np.zeros(0),
                  "update":  np.zeros(0),
                  "cost":  np.zeros(0),
                  }

        tstart = time.perf_counter()
        costs = np.zeros((max_iterations + 1, self.n_samples)) * np.nan
        steps = np.ones(max_iterations) * step
        line_search_iters = np.ones(max_iterations, dtype=int)
        q_last = 1
        x = xp.array(x_start, copy=True)

        for ii in range(max_iterations):
            # select which subsets of views/angles to use
            if stochastic_descent:
                # select random subset of angles
                num = random.sample(range(1, nmax_stochastic_descent + 1), 1)[0]
                inds = random.sample(range(self.n_samples), num)
            else:
                # use all angles
                inds = list(range(self.n_samples))

            # if any nans, break
            if xp.any(xp.isnan(x)):
                results["stop_condition"] = "stopped on NaN"
                break

            # ###################################
            # proximal gradient descent
            # ###################################

            liters = 0
            if not line_search:
                # ###################################
                # compute cost
                # ###################################
                tstart_err = time.perf_counter()

                if compute_cost:
                    if compute_all_costs:
                        costs[ii] = _to_cpu(self.cost(x))
                    else:
                        costs[ii, inds] = _to_cpu(self.cost(x, inds=inds))

                timing["cost"] = np.concatenate((timing["cost"], np.array([time.perf_counter() - tstart_err])))

                # ###################################
                # compute gradient
                # ###################################
                tstart_grad = time.perf_counter()

                x -= steps[ii] * xp.mean(self.gradient(x, inds=inds), axis=0)

                timing["grad"] = np.concatenate((timing["grad"], np.array([time.perf_counter() - tstart_grad])))

                # ###################################
                # prox operator
                # ###################################
                tstart_prox = time.perf_counter()
                y = self.prox(x, steps[ii])

                timing["prox"] = np.concatenate((timing["prox"], np.array([time.perf_counter() - tstart_prox])))

            else:
                # cost at current point
                # always grab costs, since computing anyways for line-search
                tstart_err = time.perf_counter()

                if compute_all_costs:
                    c_all = self.cost(x)
                    costs[ii, inds] = _to_cpu(c_all)
                    cx = xp.mean(c_all[inds], axis=0)
                else:
                    c_now = self.cost(x, inds=inds)
                    costs[ii, inds] = _to_cpu(c_now)
                    cx = xp.mean(c_now, axis=0)

                timing["cost"] = np.concatenate((timing["cost"], np.array([time.perf_counter() - tstart_err])))

                # gradient at current point
                tstart_grad = time.perf_counter()
                gx = xp.mean(self.gradient(x, inds=inds), axis=0)
                timing["grad"] = np.concatenate((timing["grad"], np.array([time.perf_counter() - tstart_grad])))

                # line-search
                tstart_prox = time.perf_counter()

                # initialize line-search
                liters += 1
                if ii != 0:
                    steps[ii] = steps[ii - 1]
                y = self.prox(x - steps[ii] * gx, steps[ii])

                def lipschitz_condition_violated(y, cx, gx):
                    cy = xp.mean(self.cost(y, inds=inds), axis=0)
                    return cy > cx + xp.sum(gx.real * (y - x).real +
                                            gx.imag * (y - x).imag) + \
                                            0.5 / steps[ii] * xp.linalg.norm(y - x)**2

                # iterate ... at each point check if we violate Lipschitz continuous gradient condition
                while lipschitz_condition_violated(y, cx, gx):
                    steps[ii] *= line_search_factor
                    y = self.prox(x - steps[ii] * gx, steps[ii])
                    liters += 1

                # not exclusively prox ... but good enough for now
                timing["prox"] = np.concatenate((timing["prox"], np.array([time.perf_counter() - tstart_prox])))

            line_search_iters[ii] = liters

            # ###################################
            # update step
            # ###################################
            tstart_update = time.perf_counter()

            q_now = 0.5 * (1 + np.sqrt(1 + 4 * q_last ** 2))
            if ii == 0 or ii == (max_iterations - 1) or not use_fista:
                x = y
            else:
                x = y + (q_last - 1) / q_now * (y - y_last)

            # update for next gradient-descent/FISTA iteration
            q_last = q_now
            y_last = y

            timing["update"] = np.concatenate((timing["update"], np.array([time.perf_counter() - tstart_update])))
            timing["iteration"] = np.concatenate((timing["iteration"], np.array([time.perf_counter() - tstart_err])))

            # print information
            if verbose:
                print(
                    f"iteration {ii + 1:d}/{max_iterations:d},"
                    f" cost={np.nanmean(costs[ii]):.3g},"
                    f" step={steps[ii]:.3g},"
                    f" line search iters={line_search_iters[ii]:d},"
                    f" grad={timing['grad'][ii]:.3f}s,"
                    f" prox={timing['prox'][ii]:.3f}s,"                                        
                    f" cost={timing['cost'][ii]:.3f}s,"
                    f" iter={timing['iteration'][ii]:.3f}s,"
                    f" total={time.perf_counter() - tstart:.3f}s",
                    end="\r")

        # compute final cost
        if compute_cost:
            if compute_all_costs:
                costs[ii + 1] = _to_cpu(self.cost(x))
            else:
                costs[ii + 1, inds] = _to_cpu(self.cost(x, inds=inds))

        # store results
        results.update({"timing": timing,
                        "costs": costs,
                        "steps": steps,
                        "line_search_iterations": line_search_iters,
                        "x": x})

        return results
