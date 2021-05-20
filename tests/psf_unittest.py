import unittest

import numpy as np

import fit_psf
import analysis_tools as tools

class Test_psf(unittest.TestCase):

    def setUp(self):
        pass

    def test_otf2psf(self):
        """
        Test otf2psf() by verifying that the ideal circular aperture otf obtained with circ_aperture_otf() produces
        the correct psf, obtained by airy_fn()
        :return:
        """
        na = 1.3
        wavelength = 0.465
        dx = 0.061
        nx = 101
        dy = dx
        ny = nx

        fxs = tools.get_fft_frqs(nx, dx)
        fys = tools.get_fft_frqs(ny, dy)
        dfx = fxs[1] - fxs[0]
        dfy = fys[1] - fys[0]

        otf = fit_psf.circ_aperture_otf(fxs[None, :], fys[:, None], na, wavelength)
        psf, (ys, xs) = fit_psf.otf2psf(otf, (dfy, dfx))
        psf = psf / psf.max()

        psf_true = fit_psf.airy_fn(xs[None, :], ys[:, None], [1, 0, 0, na, 0], wavelength)

        self.assertAlmostEqual(np.max(np.abs(psf - psf_true)), 0, 4)


if __name__ == "__main__":
    unittest.main()