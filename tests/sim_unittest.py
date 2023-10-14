"""
Tests for some important functions from sim_reconstruction.py
"""
import unittest
import numpy as np
from scipy import fft
import mcsim.analysis.sim_reconstruction as sim
import mcsim.analysis.analysis_tools as tools
from localize_psf.camera import bin


class TestSIM(unittest.TestCase):

    def setUp(self):
        pass

    def test_fit_modulation_frq(self):
        """
        Test fit_modulation_frq() function

        :return:
        """
        # set parameters
        dxy = 0.065
        wavelength = 0.5
        na = 1.3
        fmax = 1 / (0.5 * wavelength / na)
        nxy = 512

        # set frequency we will try to find
        f = 1 / 0.25
        angle = 30 * np.pi / 180
        frqs = [f * np.cos(angle), f * np.sin(angle)]
        phi = 0.2377747474

        # create sample image and pixel grid
        x = dxy * np.arange(nxy)
        y = x
        xx, yy = np.meshgrid(x, y)
        m = 1 + 0.5 * np.cos(2 * np.pi * (frqs[0] * xx + frqs[1] * yy) + phi)

        # Fourier transform and frequency grid
        mft = fft.fftshift(fft.fft2(fft.ifftshift(m)))
        fx = fft.fftshift(np.fft.fftfreq(nxy, dxy))
        dfx = fx[1] - fx[0]
        fy = fft.fftshift(np.fft.fftfreq(nxy, dxy))
        ff = np.sqrt(np.expand_dims(fx, axis=0)**2 + np.expand_dims(fy, axis=1)**2)

        # do fitting
        mask = ff > 0.6 * fmax
        frq_extracted, mask, _ = sim.fit_modulation_frq(mft, mft, dxy, mask, max_frq_shift=5 * dfx)

        self.assertAlmostEqual(np.abs(frq_extracted[0]), np.abs(frqs[0]), places=5)
        self.assertAlmostEqual(np.abs(frq_extracted[1]), np.abs(frqs[1]), places=5)

    def test_getphase_realspace(self):
        """
        Test get_phase_realspace() function

        :return:
        """

        # set parameters
        dx = 0.065
        nx = 2048
        f = 1 / 0.25
        angle = 30 * np.pi/180
        frqs = [f * np.cos(angle), f * np.sin(angle)]
        phi = 0.2377747474

        # create sample image with origin at edge
        x_edge = dx * np.arange(nx)
        y_edge = x_edge
        xx_edge, yy_edge = np.meshgrid(x_edge, y_edge)
        m_edge = 1 + 0.2 * np.cos(2 * np.pi * (frqs[0] * xx_edge + frqs[1] * yy_edge) + phi)

        phase_guess_edge = sim.get_phase_realspace(m_edge,
                                                   frqs,
                                                   dx,
                                                   phase_guess=0,
                                                   origin="edge")

        self.assertAlmostEqual(phi, float(phase_guess_edge), places=4)

        # create sample image with origin in center/i.e. using fft style coordinates
        # x_center = tools.get_fft_pos(nx, dx)
        x_center = (np.arange(nx) - nx // 2) * dx
        y_center = x_center
        xx_center, yy_center = np.meshgrid(x_center, y_center)
        m_center = 1 + 0.2 * np.cos(2 * np.pi * (frqs[0] * xx_center + frqs[1] * yy_center) + phi)

        phase_guess_center = sim.get_phase_realspace(m_center,
                                                     frqs,
                                                     dx,
                                                     phase_guess=0,
                                                     origin="center")

        self.assertAlmostEqual(phi, float(phase_guess_center), places=4)

    def test_get_phase_ft(self):
        """
        Test get_phase_ft() function, which guesses phase from value of image FT

        :return:
        """
        # set parameters
        dx = 0.065
        nx = 2048
        f = 1 / 0.25
        angle = 30 * np.pi / 180
        frqs = [f * np.cos(angle), f * np.sin(angle)]
        phi = 0.2377747474

        # create sample image with origin in center
        # x_center = tools.get_fft_pos(nx, dx)
        x_center = (np.arange(nx) - nx // 2) * dx
        y_center = x_center
        xx_center, yy_center = np.meshgrid(x_center, y_center)
        m_center = 1 + 0.2 * np.cos(2 * np.pi * (frqs[0] * xx_center + frqs[1] * yy_center) + phi)

        m_center_ft = fft.fftshift(fft.fft2(fft.ifftshift(m_center)))

        phase_guess_center = sim.get_phase_ft(m_center_ft, frqs, dx)

        self.assertAlmostEqual(phi, float(phase_guess_center), places=5)

    def test_get_phase_wicker(self):
        pass

    def test_get_simulated_sim_imgs(self):
        """
        Test that get_simulated_sim_imgs() returns images with the correct SIM parameters when used with different
        levels of binning

        @return:
        """
        nbins = [1, 2, 3, 4, 5, 6, 4, 5]
        nxs = [600, 600, 600, 600, 600, 600, 500, 625]

        dxy = 0.59
        for nbin, nx in zip(nbins, nxs):

            freq = np.array([1 / 25.6346, 1 / 25.77772])
            phi = 0.23626
            pattern = sim.get_sinusoidal_patterns(dxy * nbin,
                                                  (nx // nbin, nx // nbin),
                                                  freq,
                                                  [phi],
                                                  n_oversampled=nbin
                                                  )
            gt = bin(pattern, (nbin, nbin))

            # gt, _, _, _ = sim.get_simulated_sim_imgs(np.ones((nx, nx)),
            #                                          frqs=freq,
            #                                          phases=phi,
            #                                          mod_depths=[1.],
            #                                          gains=1,
            #                                          offsets=0,
            #                                          readout_noise_sds=0,
            #                                          pix_size=dxy,
            #                                          photon_shot_noise=False,
            #                                          nbin=nbin)

            gt = gt[0, 0, 0]
            gt_ft = fft.fftshift(fft.fft2(fft.ifftshift(gt)))

            # test phase
            phases_fit = sim.get_phase_realspace(gt, freq, nbin * dxy, origin="center")
            self.assertAlmostEqual(phi, float(phases_fit), places=3)

            # test frequency
            fx = fft.fftshift(fft.fftfreq(gt_ft.shape[0], nbin * dxy))
            dfx = fx[1] - fx[0]
            frq_guess = freq + np.random.uniform(-0.005, 0.005, 2)
            frqs_fit, _, result = sim.fit_modulation_frq(gt_ft,
                                                         gt_ft,
                                                         nbin * dxy,
                                                         frq_guess=frq_guess,
                                                         max_frq_shift=5 * dfx)
            np.testing.assert_allclose(frqs_fit, freq, atol=1e-5)

    def test_get_band_mixing_matrix(self):
        """
        Test that the real-space and Fourier-space pattern generation models agree.

        i.e. that D_i(x) = amp * (1 + m * cos(2*pi*f + phi_i)) * S(r)
        matches the result given using the fourier space matrix
        [[D_1(k)], [D_2(k)], [D_3(k)]] = M * [[S(k)], [S(k-p)], [S(k+p)]]

        :return:
        """

        # set values for SIM images
        dx = 0.065
        frqs = np.array([[3.5785512, 2.59801082]])
        phases = np.array([[0, 2 * np.pi / 3, 3 * np.pi / 3]])
        mods = np.array([0.85, 0.26, 0.19])
        amps = np.array([[1.11, 1.23, 0.87]])
        nangles, nphases = phases.shape

        # ground truth image
        ny = 512
        nx = ny
        gt = np.random.rand(ny, nx)

        # calculate sim patterns using real space method
        # x = tools.get_fft_pos(nx, dx)
        # y = tools.get_fft_pos(ny, dx)
        x = (np.arange(nx) - nx // 2) * dx
        y = (np.arange(ny) - ny // 2) * dx
        xx, yy = np.meshgrid(x, y)

        sim_rs = np.zeros((nangles, nphases, ny, nx))
        sim_rs_ft = np.zeros((nangles, nphases, ny, nx), dtype=complex)
        for ii in range(nangles):
            for jj in range(nphases):
                pattern = amps[ii, jj] * (1 + mods[ii] * np.cos(2*np.pi * (xx * frqs[ii, 0] + yy * frqs[ii, 1]) + phases[ii, jj]))
                sim_rs[ii, jj] = gt * pattern
                sim_rs_ft[ii, jj] = fft.fftshift(fft.fft2(fft.ifftshift(sim_rs[ii, jj])))

        # calculate SIM patterns using Fourier space method
        # frq shifted gt images
        gt_ft_shifted = np.zeros((nangles, nphases, ny, nx), dtype=complex)
        for ii in range(nangles):
            gt_ft_shifted[ii, 0] = fft.fftshift(fft.fft2(fft.ifftshift(gt)))
            gt_ft_shifted[ii, 1] = tools.translate_ft(gt_ft_shifted[ii, 0], -frqs[ii, 0], -frqs[ii, 1], drs=(dx, dx))
            gt_ft_shifted[ii, 2] = tools.translate_ft(gt_ft_shifted[ii, 0], frqs[ii, 0], frqs[ii, 1], drs=(dx, dx))

        sim_fs_ft = np.zeros(gt_ft_shifted.shape, dtype=complex)
        for ii in range(nangles):
            kmat = sim.get_band_mixing_matrix(phases[ii], mods[ii], amps[ii])
            # sim_fs_ft[ii] = sim.image_times_matrix(gt_ft_shifted[ii], kmat)
            for jj in range(nphases):
                sim_fs_ft[ii, jj] = np.sum(np.expand_dims(kmat[jj], axis=(-1, -2)) * gt_ft_shifted[ii], axis=0)

        sim_fs_rs = np.zeros(gt_ft_shifted.shape)
        for ii in range(nangles):
            for jj in range(nphases):
                sim_fs_rs[ii, jj] = fft.fftshift(fft.ifft2(fft.ifftshift(sim_fs_ft[ii, jj]))).real

        np.testing.assert_allclose(sim_fs_ft, sim_rs_ft, atol=1e-10)
        np.testing.assert_allclose(sim_fs_rs, sim_rs, atol=1e-12)

    def test_band_mixing_mat_jac(self):
        """
        test jacobian of band mixing matrix
        @return:
        """
        phases = [0, 2*np.pi/3 - 0.89243, 4*np.pi/3 + 0.236]
        amps = [0.78, 0.876, 0.276]
        m = 0.777
        params = np.array(phases + amps + [m])
        ds = 1e-8

        jac = sim.get_band_mixing_matrix_jac(phases, m, amps)

        jac_est = []
        def get_mat(p): return sim.get_band_mixing_matrix([p[0], p[1], p[2]], p[6], [p[3], p[4], p[5]])
        for ii in range(len(params)):
            params_temp = np.array(params, copy=True)
            params_temp[ii] -= ds
            jac_est.append(1 / ds * (get_mat(params) - get_mat(params_temp)))

        max_err = np.max([np.max(np.abs(jac[ii] - jac_est[ii])) for ii in range(len(params))])
        self.assertAlmostEqual(max_err, 0, places=7)

    def test_band_mixing_mat_inv(self):
        """
        test inverse mixing matrix gives correct result
        @return:
        """
        phases = [0, 2 * np.pi / 3 - 0.89243, 4 * np.pi / 3 + 0.236]
        amps = [0.78, 0.876, 0.276]
        m = 0.777

        mat = sim.get_band_mixing_matrix(phases, m, amps)
        mat_inv = sim.get_band_mixing_inv(phases, m, amps)

        np.testing.assert_allclose(mat.dot(mat_inv), np.identity(mat.shape[0]), atol=1e-10)

    def test_expand_fourier_sp(self):
        """
        Test expand_fourier_sp() function
        :return:
        """

        arr = np.array([[1, 2], [3, 4]])
        arr_ft = fft.fftshift(fft.fft2(fft.ifftshift(arr)))

        arr_ft_ex = sim.resample_bandlimited_ft(arr_ft, (2, 2), axes=(-1, -2))
        arr_ex = fft.fftshift(fft.ifft2(fft.ifftshift(arr_ft_ex)))

        self.assertTrue(np.array_equal(arr_ex.real, np.array([[1, 1.5, 2, 1.5],
                                                               [2, 2.5, 3, 2.5],
                                                               [3, 3.5, 4, 3.5],
                                                               [2, 2.5, 3, 2.5]])))

    def test_expand_fourier_sp_odd1d(self):
        """
        Test function with odd input size

        :return:
        """
        arr = np.random.rand(151)
        arr_ft = fft.fftshift(fft.fft(fft.ifftshift(arr)))

        arr_ex_ft = sim.resample_bandlimited_ft(arr_ft, (2,), axes=(-1,))
        arr_exp = fft.fftshift(fft.ifft(fft.ifftshift(arr_ex_ft))).real

        max_err = np.max(np.abs(arr_exp[1::2] - arr))
        self.assertTrue(max_err < 1e-14)

    def test_expand_fourier_sp_even1d(self):
        """
        test function with even input size

        :return:
        """
        arr = np.random.rand(100)
        arr_ft = fft.fftshift(fft.fft(fft.ifftshift(arr)))

        arr_ex_ft = sim.resample_bandlimited_ft(arr_ft, (2,), axes=(-1,))
        arr_exp = fft.fftshift(fft.ifft(fft.ifftshift(arr_ex_ft))).real

        max_err = np.max(np.abs(arr_exp[::2] - arr))
        self.assertTrue(max_err < 1e-14)

    def test_expand_fourier_sp_odd2d(self):
        """
        Test function with odd input size
        """
        arr = np.random.rand(151, 151)
        arr_ft = fft.fftshift(fft.fft2(fft.ifftshift(arr)))

        arr_ex_ft = sim.resample_bandlimited_ft(arr_ft, (2, 2), axes=(-1, -2))
        arr_exp = fft.fftshift(fft.ifft2(fft.ifftshift(arr_ex_ft))).real

        max_err = np.max(np.abs(arr_exp[1::2, 1::2] - arr))
        self.assertTrue(max_err < 1e-14)

    def test_expand_fourier_sp_even2d(self):
        """
        test function with even input size
        """
        arr = np.random.rand(100, 100)
        arr_ft = fft.fftshift(fft.fft2(fft.ifftshift(arr)))

        arr_ex_ft = sim.resample_bandlimited_ft(arr_ft, (2, 2), axes=(-1, -2))
        arr_exp = fft.fftshift(fft.ifft2(fft.ifftshift(arr_ex_ft))).real

        max_err = np.max(np.abs(arr_exp[::2, ::2] - arr))
        self.assertTrue(max_err < 1e-14)

    def test_translate_ft(self):
        """
        Test translate_ft() function
        :return:
        """

        img = np.random.rand(100, 100)
        img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img)))
        dx = 0.065

        fx = fft.fftshift(fft.fftfreq(img.shape[1], dx))
        fy = fft.fftshift(fft.fftfreq(img.shape[0], dx))
        df = fx[1] - fx[0]

        # x-shifting
        for n in range(1, 20):
            img_ft_shifted = tools.translate_ft(img_ft, n * df, 0, drs=(dx, dx))
            max_err = np.abs(img_ft_shifted[:, :-n] - img_ft[:, n:]).max()
            self.assertTrue(max_err < 1e-7)

        # y-shifting
        for n in range(1, 20):
            img_ft_shifted = tools.translate_ft(img_ft, 0, n * df, drs=(dx, dx))
            max_err = np.abs(img_ft_shifted[:-n, :] - img_ft[n:, :]).max()
            self.assertTrue(max_err < 1e-7)

        # x+y shifting
        for n in range(1, 20):
            img_ft_shifted = tools.translate_ft(img_ft, n * df, n * df, drs=(dx, dx))
            max_err = np.abs(img_ft_shifted[:-n, :-n] - img_ft[n:, n:]).max()
            self.assertTrue(max_err < 1e-7)

        # todo: also test approximately gives the right thing for partial pixel shifts (i.e. that the phases make sense)


if __name__ == "__main__":
    unittest.main()
