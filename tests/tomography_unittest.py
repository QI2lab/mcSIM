import unittest
import numpy as np
from scipy.signal.windows import tukey
import mcsim.analysis.tomography as tm
from mcsim.analysis.field_prop import angles2frqs, propagate_homogeneous
from mcsim.analysis.fft import ft3, ft2, ift2
from mcsim.analysis.optimize import to_cpu

try:
    import cupy as cp
except ValueError:
    cp = None


class TestPatterns(unittest.TestCase):

    def setUp(self):
        pass

    def test_ft_adjoint(self):
        backends = [np]
        if cp:
            backends += [cp]

        for xp in backends:
            for nxy in [100, 101]:
                for shift in [True, False]:
                    v = xp.random.rand(nxy, nxy) + 1j * xp.random.rand(nxy, nxy)
                    w = xp.random.rand(nxy, nxy) + 1j * xp.random.rand(nxy, nxy)
                    op_v = ft2(v, shift=shift)
                    opadj_w = ft2(w, shift=shift, adjoint=True)

                    w_dot_op_v = to_cpu(xp.sum(xp.conj(w) * op_v))
                    opadj_w_dot_v = to_cpu(xp.sum(xp.conj(opadj_w) * v))

                    np.testing.assert_allclose(w_dot_op_v,
                                               opadj_w_dot_v,
                                               atol=1e-10)

    def test_ift_adjoint(self):
        backends = [np]
        if cp:
            backends += [cp]

        for xp in backends:
            for nxy in [100, 101]:
                for shift in [True, False]:
                    v = xp.random.rand(nxy, nxy) + 1j * xp.random.rand(nxy, nxy)
                    w = xp.random.rand(nxy, nxy) + 1j * xp.random.rand(nxy, nxy)
                    op_v = ift2(v, shift=shift)
                    opadj_w = ift2(w, shift=shift, adjoint=True)

                    w_dot_op_v = to_cpu(xp.sum(xp.conj(w) * op_v))
                    opadj_w_dot_v = to_cpu(xp.sum(xp.conj(opadj_w) * v))

                    np.testing.assert_allclose(w_dot_op_v,
                                               opadj_w_dot_v,
                                               atol=1e-10)

    def test_prop_adjoint(self):
        dxy = 0.1
        z = 5
        nxy = 100
        no = 1.333
        wlen = 0.532

        v = np.random.rand(nxy, nxy) + 1j * np.random.rand(nxy, nxy)
        w = np.random.rand(nxy, nxy) + 1j * np.random.rand(nxy, nxy)

        op_v = propagate_homogeneous(v, z, no, (dxy, dxy), wlen)
        opadj_w = propagate_homogeneous(w, z, no, (dxy, dxy), wlen, adjoint=True)

        w_dot_op_v = np.sum(np.conj(w) * op_v)
        opadj_w_dot_v = np.sum(np.conj(opadj_w) * v)

        np.testing.assert_allclose(w_dot_op_v,
                                   opadj_w_dot_v,
                                   atol=1e-10)

    def test_born_grad(self):

        if cp:
            xp = cp
        else:
            xp = np

        dxy = 0.1
        dz = 0.1
        nx = 960
        ny = 900
        nz = 100
        npattern = 11
        no = 1.333
        wavelength = 0.785
        na_det = 1

        e = xp.random.rand(npattern, ny, nx) + 1j * xp.random.rand(npattern, ny, nx)
        n = no + 0.1 * xp.random.rand(nz, ny, nx) + 1j * xp.random.rand(nz, ny, nx) * 1e-3
        v = tm.get_v(n, no, wavelength)
        vft = ft3(v)

        theta = 25 * np.pi / 180 * np.ones(npattern)
        phis = np.arange(npattern) / npattern * 2*np.pi
        beam_frqs = angles2frqs(theta, phis, no / wavelength)

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

        np.testing.assert_allclose(g.get(), gn.get(), rtol=1e-1)

    def test_rytov_grad(self):

        if cp:
            xp = cp
        else:
            xp = np

        dxy = 0.1
        dz = 0.1
        nx = 960
        ny = 900
        nz = 100
        npattern = 11
        no = 1.333
        wavelength = 0.785
        na_det = 1

        e = xp.random.rand(npattern, ny, nx) + 1j * xp.random.rand(npattern, ny, nx)
        n = no + 0.1 * xp.random.rand(nz, ny, nx) + 1j * xp.random.rand(nz, ny, nx) * 1e-3
        v = tm.get_v(n, no, wavelength)
        vft = ft3(v)

        theta = 25 * np.pi / 180 * np.ones(npattern)
        phis = np.arange(npattern) / npattern * 2*np.pi
        beam_frqs = angles2frqs(theta, phis, no / wavelength)

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

        np.testing.assert_allclose(g.get(), gn.get(), rtol=1e-1)

    def test_bpm_grad(self):

        if cp:
            xp = cp
        else:
            xp = np

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
        # fx = xp.fft.fftshift(xp.fft.fftfreq(nx, dxy))[None, :]
        # fy = xp.fft.fftshift(xp.fft.fftfreq(ny, dxy))[:, None]
        fx = xp.fft.fftfreq(nx, dxy)[None, :]
        fy = xp.fft.fftfreq(ny, dxy)[:, None]
        atf = (xp.sqrt(fx ** 2 + fy ** 2) <= fmax).astype(complex)

        apo = xp.asarray(np.outer(tukey(ny, alpha=0.1),
                                  tukey(nx, alpha=0.1)))

        e = xp.random.rand(npattern, ny, nx) + 1j * xp.random.rand(npattern, ny, nx)
        ebg = xp.random.rand(npattern, ny, nx) + 1j * xp.random.rand(npattern, ny, nx)
        n = no + 0.1 * xp.random.rand(nz, ny, nx) + 1j * xp.random.rand(nz, ny, nx) * 1e-3

        mask = xp.ones((ny, nx), dtype=bool)
        mask[ny // 2:, :] = False

        opt = tm.BPM(e,
                     ebg,
                     None,
                     no,
                     wavelength,
                     (dxy, dxy),
                     (dz, dxy, dxy),
                     (nz, ny, nx),
                     dz_final=dz_final,
                     atf=atf,
                     apodization=apo,
                     mask=mask,
                     efield_cost_factor=0.5)

        jind = np.ravel_multi_index((nz//2, ny//2, nx//2), n.shape)
        g, gn = opt.test_gradient(n, jind, inds=[0, 1])

        np.testing.assert_allclose(g.get(), gn.get(), rtol=1e-3)

    def test_ssnp_grad(self):
        if cp:
            xp = cp
        else:
            xp = np

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
        fx = xp.fft.fftfreq(nx, dxy)[None, :]
        fy = xp.fft.fftfreq(ny, dxy)[:, None]
        atf = (xp.sqrt(fx ** 2 + fy ** 2) <= fmax).astype(complex)

        apo = xp.asarray(np.outer(tukey(ny, alpha=0.1),
                                  tukey(nx, alpha=0.1)))

        e = xp.random.rand(npattern, ny, nx) + 1j * xp.random.rand(npattern, ny, nx)
        ebg = xp.random.rand(npattern, ny, nx) + 1j * xp.random.rand(npattern, ny, nx)
        n = no + 0.1 * xp.random.rand(nz, ny, nx) + 1j * xp.random.rand(nz, ny, nx) * 1e-3

        mask = xp.ones((ny, nx), dtype=bool)
        mask[ny // 2:, :] = False

        opt = tm.SSNP(e,
                      ebg,
                      None,
                      no,
                      wavelength,
                      (dxy, dxy),
                      (dz, dxy, dxy),
                      (nz, ny, nx),
                      dz_final=dz_final,
                      atf=atf,
                      apodization=apo,
                      mask=mask,
                      efield_cost_factor=0.5)

        jind = np.ravel_multi_index((nz // 2, ny // 2, nx // 2), n.shape)
        g, gn = opt.test_gradient(n, jind, inds=[0, 1])

        np.testing.assert_allclose(to_cpu(g),
                                   to_cpu(gn),
                                   rtol=1e-3)
