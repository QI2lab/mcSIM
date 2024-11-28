"""
Test routines that Mie field calculations rely on
"""
import unittest
import numpy as np
from scipy.special import spherical_jn, spherical_yn
from mcsim.analysis.mie_fields import jn, yn

try:
    import cupy as cp
except ValueError:
    cp = None


class TestMie(unittest.TestCase):

    def setUp(self):
        pass

    def test_jn_gpu(self):
        x = np.linspace(0, 400, 5001)
        ns = np.arange(300)

        jn_scipy = np.zeros((len(x), len(ns)))
        djn_scipy = np.zeros((len(x), len(ns)))        
        for ii in range(len(ns)):
            jn_scipy[:, ii] = spherical_jn(ns[ii], x)
            djn_scipy[:, ii] = spherical_jn(ns[ii], x, derivative=True)            

        jn_cp, djn_cp = jn(ns.max() + 1, cp.asarray(x), overhead=40)        
        jn_cp = jn_cp.get()
        djn_cp = djn_cp.get()    

        np.testing.assert_allclose(jn_scipy, jn_cp, atol=1e-8)
        np.testing.assert_allclose(djn_scipy, djn_cp, atol=1e-8)

    def test_yn_gpu(self):
        x = np.linspace(0, 400, 5001)
        ns = np.arange(300)
     
        yn_scipy = np.zeros((len(x), len(ns)))
        dyn_scipy = np.zeros((len(x), len(ns)))
        for ii in range(len(ns)):
            yn_scipy[:, ii] = spherical_yn(ns[ii], x)
            dyn_scipy[:, ii] = spherical_yn(ns[ii], x, derivative=True)
        
        yn_cp, dyn_cp = yn(ns.max() + 1, cp.asarray(x))        
        yn_cp = yn_cp.get()
        dyn_cp = dyn_cp.get()

        np.testing.assert_allclose(jn_scipy, jn_cp, atol=1e-8)
        np.testing.assert_allclose(djn_scipy, djn_cp, atol=1e-8)
