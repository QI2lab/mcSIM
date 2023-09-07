#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2020 Tobias A. de Jong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


A weighed phase unwrap algorithm implemented in pure Python

author: Tobias A. de Jong
Based on:
Ghiglia, Dennis C., and Louis A. Romero. 
"Robust two-dimensional weighted and unweighted phase unwrapping that uses 
fast transforms and iterative methods." JOSA A 11.1 (1994): 107-117.
URL: https://doi.org/10.1364/JOSAA.11.000107
and an existing MATLAB implementation:
https://nl.mathworks.com/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
Should maybe use a scipy conjugate descent.
"""

import numpy as np
from scipy.fft import dctn, idctn
import scipy.fft as fft_cpu

_gpu_available = True
try:
    import cupy as cp
    import cupyx.scipy.fft as fft_gpu
except ImportError:
    _gpu_available = False
    fft_gpu = fft_cpu

def phase_unwrap_ref(psi, weight, kmax=100):

    if isinstance(psi, cp.ndarray):
        xp = cp
        fft = fft_gpu
    else:
        xp = np
        fft = fft_cpu

    weight = xp.asarray(weight)

    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(xp.diff(psi, axis=1))
    dy = _wrapToPi(xp.diff(psi, axis=0))
    
    # multiply the vector b by weight square (W^T * W)
    WW = weight**2
    
    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = xp.minimum(WW[:,:-1], WW[:,1:])
    WWy = xp.minimum(WW[:-1,:], WW[1:,:])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = xp.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = xp.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = xp.linalg.norm(rk)

    # start the iteration
    eps = 1e-9
    k = 0
    phi = xp.zeros_like(psi)
    while (~xp.all(rk == 0.0)):
        zk = solvePoisson(rk)
        k += 1
        
        # equivalent to (rk*zk).sum()
        rkzksum = xp.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum / rkzkprevsum
            pk = zk + betak * pk

        # save the current value as the previous values
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / xp.tensordot(pk, Qpk)
        phi += alphak * pk
        rk -= alphak * Qpk

        # check the stopping conditions
        if ((k >= kmax) or (xp.linalg.norm(rk) < eps * normR0)):
            break

    return phi

def solvePoisson(rho):
    """Solve the poisson equation "P phi = rho" using DCT
    """
    if isinstance(rho, cp.ndarray):
        xp = cp
        fft = fft_gpu
    else:
        xp = np
        fft = fft_cpu

    dctRho = fft.dctn(rho)
    N, M = rho.shape
    I, J = xp.ogrid[0:N,0:M]
    with np.errstate(divide='ignore'):
        dctPhi = dctRho / 2 / (xp.cos(np.pi*I/M) + xp.cos(np.pi*J/N) - 2)
    dctPhi[0, 0] = 0 # handling the inf/nan value
    # now invert to get the result
    phi = fft.idctn(dctPhi)
    return phi

def solvePoisson_precomped(rho, scale):
    """Solve the poisson equation "P phi = rho" using DCT

    Uses precomputed scaling factors `scale`
    """
    if isinstance(rho, cp.ndarray):
        xp = cp
        fft = fft_gpu
    else:
        xp = np
        fft = fft_cpu

    dctPhi = fft.dctn(rho) / scale
    # now invert to get the result
    phi = fft.idctn(dctPhi, overwrite_x=True)
    return phi

def precomp_Poissonscaling(rho):
    if isinstance(rho, cp.ndarray):
        xp = cp
    else:
        xp = np

    N, M = rho.shape
    I, J = xp.ogrid[0:N,0:M]
    scale = 2 * (xp.cos(np.pi*I/M) + xp.cos(np.pi*J/N) - 2)
    # Handle the inf/nan value without a divide by zero warning:
    # By Ghiglia et al.:
    # "In practice we set dctPhi[0,0] = dctn(rho)[0, 0] to leave
    #  the bias unchanged"
    scale[0, 0] = 1. 
    return scale

def applyQ(p, WWx, WWy):
    """Apply the weighted transformation (A^T)(W^T)(W)(A) to 2D matrix p"""
    if isinstance(p, cp.ndarray):
        xp = cp
    else:
        xp = np

    WWx = xp.asarray(WWx)
    WWy = xp.asarray(WWy)

    # apply (A)
    dx = xp.diff(p, axis=1)
    dy = xp.diff(p, axis=0)

    # apply (W^T)(W)
    WWdx = WWx * dx
    WWdy = WWy * dy
    
    # apply (A^T)
    WWdx2 = xp.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = xp.diff(WWdy, axis=0, prepend=0, append=0)
    Qp = WWdx2 + WWdy2
    return Qp


def _wrapToPi(x):
    if isinstance(x, cp.ndarray):
        xp = cp
    else:
        xp = np

    r = xp.mod(x+np.pi, 2*np.pi) - np.pi
    return r

def phase_unwrap(psi, weight=None, kmax=100):
    """
    Unwrap the phase of an image psi given weights weight

    This function uses an algorithm described by Ghiglia and Romero
    and can either be used with or without weight array.
    It is especially suited to recover a unwrapped phase image
    from a (noisy) complex type image, where psi would be 
    the angle of the complex values and weight the absolute values
    of the complex image.
    """

    if isinstance(psi, cp.ndarray):
        xp = cp
        fft = fft_gpu
    else:
        xp = np
        fft = fft_cpu

    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(xp.diff(psi, axis=1))
    dy = _wrapToPi(xp.diff(psi, axis=0))
    
    # multiply the vector b by weight square (W^T * W)
    if weight is None:
        # Unweighed case. will terminate in 1 round
        WW = xp.ones_like(psi)
    else:
        weight = xp.asarray(weight)
        WW = weight**2
    
    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = xp.minimum(WW[:,:-1], WW[:,1:])
    WWy = xp.minimum(WW[:-1,:], WW[1:,:])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = xp.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = xp.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = xp.linalg.norm(rk)

    # start the iteration
    eps = 1e-9
    k = 0
    phi = xp.zeros_like(psi)
    scaling = precomp_Poissonscaling(rk)
    while (~xp.all(rk == 0.0)):
        zk = solvePoisson_precomped(rk, scaling)
        k += 1
        
        # equivalent to (rk*zk).sum()
        rkzksum = xp.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum / rkzkprevsum
            pk = zk + betak * pk

        # save the current value as the previous values
        
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / xp.tensordot(pk, Qpk)
        phi += alphak * pk
        rk -= alphak * Qpk

        # check the stopping conditions
        if ((k >= kmax) or (xp.linalg.norm(rk) < eps * normR0)):
            break

    return phi
