import unittest
import mcsim.analysis.tomography as tm
from mcsim.analysis.tomography import _ft3
import numpy as np
import cupy as cp

class TestPatterns(unittest.TestCase):

    def setUp(self):
        pass


    def test_born_grad(self):
        dxy = 0.1
        dz = 0.1
        nx = 960
        ny = 900
        nz = 100
        npattern = 11
        no = 1.333
        wavelength = 0.785
        na_det = 1

        fmax = no / wavelength
        fx = cp.fft.fftshift(cp.fft.fftfreq(nx, dxy))[None, :]
        fy = cp.fft.fftshift(cp.fft.fftfreq(ny, dxy))[:, None]

        # todo: include
        atf = (cp.sqrt(fx ** 2 + fy ** 2) <= fmax).astype(complex)

        e = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        # ebg = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        n = no + 0.1 * cp.random.rand(nz, ny, nx) + 1j * cp.random.rand(nz, ny, nx) * 1e-3
        v = tm.get_v(n, no, wavelength)
        vft = _ft3(v)

        theta = 25 * np.pi/ 180 * np.ones(npattern)
        phis = np.arange(npattern) / npattern * 2*np.pi
        beam_frqs = tm.angles2frqs(no, wavelength, theta, phis)

        model = tm.fwd_model_linear(beam_frqs[..., 0],
                                    beam_frqs[..., 1],
                                    beam_frqs[..., 2],
                                    no,
                                    na_det,
                                    wavelength,
                                    (ny, nx),
                                    (dxy, dxy),
                                    (nz, ny, nx),
                                    (dz, dxy, dxy),
                                    mode="born",
                                    interpolate=True,
                                    use_gpu=True)

        opt = tm.LinearScatt(e,
                             model,
                             no,
                             wavelength,
                             (dxy, dxy),
                             (dz, dxy, dxy),
                             (nz, ny, nx))

        jind = np.ravel_multi_index((nz // 2, ny // 2, nx // 2), vft.shape)
        g, gn = opt.test_gradient(vft, jind, inds=[0, 1])

        np.testing.assert_allclose(g.get(), gn.get(), atol=1e-8)

    def test_rytov_grad(self):
        dxy = 0.1
        dz = 0.1
        nx = 960
        ny = 900
        nz = 100
        npattern = 11
        no = 1.333
        wavelength = 0.785
        na_det = 1

        fmax = no / wavelength
        fx = cp.fft.fftshift(cp.fft.fftfreq(nx, dxy))[None, :]
        fy = cp.fft.fftshift(cp.fft.fftfreq(ny, dxy))[:, None]

        # todo: include
        atf = (cp.sqrt(fx ** 2 + fy ** 2) <= fmax).astype(complex)

        e = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        # ebg = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        n = no + 0.1 * cp.random.rand(nz, ny, nx) + 1j * cp.random.rand(nz, ny, nx) * 1e-3
        v = tm.get_v(n, no, wavelength)
        vft = _ft3(v)

        theta = 25 * np.pi/ 180 * np.ones(npattern)
        phis = np.arange(npattern) / npattern * 2*np.pi
        beam_frqs = tm.angles2frqs(no, wavelength, theta, phis)

        model = tm.fwd_model_linear(beam_frqs[..., 0],
                                    beam_frqs[..., 1],
                                    beam_frqs[..., 2],
                                    no,
                                    na_det,
                                    wavelength,
                                    (ny, nx),
                                    (dxy, dxy),
                                    (nz, ny, nx),
                                    (dz, dxy, dxy),
                                    mode="rytov",
                                    interpolate=True,
                                    use_gpu=True,)

        opt = tm.LinearScatt(e,
                             model,
                             no,
                             wavelength,
                             (dxy, dxy),
                             (dz, dxy, dxy),
                             (nz, ny, nx))

        jind = np.ravel_multi_index((nz // 2, ny // 2, nx // 2), vft.shape)
        g, gn = opt.test_gradient(vft, jind, inds=[0, 1])

        np.testing.assert_allclose(g.get(), gn.get(), atol=1e-8)

    def test_bpm_grad(self):
        dxy = 0.1
        dz = 0.1
        nx = 960
        ny = 900
        nz = 100
        npattern = 11
        no = 1.333
        wavelength = 0.785

        dz_final = -dz * ((nz - 1) - nz // 2 + 0.5)

        fmax = no / wavelength
        fx = cp.fft.fftshift(cp.fft.fftfreq(nx, dxy))[None, :]
        fy = cp.fft.fftshift(cp.fft.fftfreq(ny, dxy))[:, None]
        atf = (cp.sqrt(fx ** 2 + fy ** 2) <= fmax).astype(complex)

        e = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        ebg = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        n = no + 0.1 * cp.random.rand(nz, ny, nx) + 1j * cp.random.rand(nz, ny, nx) * 1e-3

        opt = tm.BPM(e,
                     ebg,
                     None,
                     no,
                     wavelength,
                     (dxy, dxy),
                     (dz, dxy, dxy),
                     (nz, ny, nx),
                     dz_final=dz_final,
                     atf=atf)

        jind = np.ravel_multi_index((nz//2, ny//2, nx//2), n.shape)
        g, gn = opt.test_gradient(n, jind, inds=[0, 1])

        np.testing.assert_allclose(g.get(), gn.get(), atol=1e-8)

    def test_ssnp_grad(self):
        dxy = 0.1
        dz = 0.1
        nx = 960
        ny = 900
        nz = 100
        npattern = 11
        no = 1.333
        wavelength = 0.785

        dz_final = -dz * ((nz - 1) - nz // 2 + 0.5)

        fmax = no / wavelength
        fx = cp.fft.fftshift(cp.fft.fftfreq(nx, dxy))[None, :]
        fy = cp.fft.fftshift(cp.fft.fftfreq(ny, dxy))[:, None]
        atf = (cp.sqrt(fx ** 2 + fy ** 2) <= fmax).astype(complex)

        e = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        ebg = cp.random.rand(npattern, ny, nx) + 1j * cp.random.rand(npattern, ny, nx)
        n = no + 0.1 * cp.random.rand(nz, ny, nx) + 1j * cp.random.rand(nz, ny, nx) * 1e-3

        opt = tm.SSNP(e,
                      ebg,
                      None,
                      no,
                      wavelength,
                      (dxy, dxy),
                      (dz, dxy, dxy),
                      (nz, ny, nx),
                      dz_final=dz_final,
                      atf=atf)

        jind = np.ravel_multi_index((nz // 2, ny // 2, nx // 2), n.shape)
        g, gn = opt.test_gradient(n, jind, inds=[0, 1])

        np.testing.assert_allclose(g.get(), gn.get(), atol=1e-8)
