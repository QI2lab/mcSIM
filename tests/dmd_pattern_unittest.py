import unittest
import time
import numpy as np
from scipy.signal.windows import hann
from scipy import fft
from localize_psf import affine, rois
import mcsim.analysis.dmd_patterns as dmd
import mcsim.analysis.analysis_tools as tools


class TestPatterns(unittest.TestCase):

    def setUp(self):
        pass

    def test_pattern_main_phase_vs_phase_index(self):
        """
        Test get_sim_phase() on the main frequency component for one SIM pattern. Ensure works for all phase indices.

        Tests only a single pattern vector
        :return:
        """
        nx = 1920
        ny = 1080
        nphases = 3

        fx = fft.fftshift(fft.fftfreq(nx))
        fy = fft.fftshift(fft.fftfreq(ny))
        window = np.expand_dims(hann(nx), axis=0) * \
                 np.expand_dims(hann(ny), axis=1)

        va = [-3, 11]
        vb = [3, 12]
        # va = [-1, 117]
        # vb = [-6, 117]
        rva, rvb = dmd.get_reciprocal_vects(va, vb)

        for jj in range(nphases):
            phase = dmd.get_sim_phase(va, vb, nphases, jj, [nx, ny], origin="fft")
            pattern, _ = dmd.get_sim_pattern([nx, ny], va, vb, nphases, jj)
            pattern_ft = fft.fftshift(fft.fft2(fft.ifftshift(pattern * window)))

            peak_val = tools.get_peak_value(pattern_ft, fx, fy, rvb, 2)
            pattern_phase_est = np.mod(np.angle(peak_val), 2 * np.pi)

            # assert np.round(np.abs(pattern_phase_est - float(phase)), 3) == 0
            self.assertAlmostEqual(pattern_phase_est, float(phase), 3)

    def test_patterns_main_phase(self):
        """
        Test determination of DMD pattern phase by comparing with FFT phase for dominant frequency component
        of given pattern.

        This compares get_sim_phase() with directly producing a pattern via get_sim_pattern()
        and numerically determining the phase.

        Unlike test_phase_vs_phase_index(), test many different lattice vectors, but only a single phase index
        :return:
        """

        # dmd_size = [1920, 1080]
        dmd_size = [192, 108]
        nphases = 3
        phase_index = 0
        nx, ny = dmd_size

        # va_comps = np.arange(-15, 15)
        # vb_comps = np.arange(-30, 30, 3)
        va_comps = np.array([-15, -7, -3, 0, 4, 8, 17])
        vb_comps = np.array([-30, -15, -6, -3, 12, 18])
        # va_comps = np.array([1, 3, 10])
        # vb_comps = np.array([-3, 0, 3])

        # generate all lattice vectors
        vas_x, vas_y = np.meshgrid(va_comps, va_comps)
        vas = np.stack((vas_x.ravel(), vas_y.ravel()), axis=1)
        vas = vas[np.linalg.norm(vas, axis=1) != 0]

        vbs_x, vbs_y = np.meshgrid(vb_comps, vb_comps)
        vbs = np.stack((vbs_x.ravel(), vbs_y.ravel()), axis=1)
        vbs = vbs[np.linalg.norm(vbs, axis=1) != 0]

        tstart = time.perf_counter()
        for ii in range(len(vas)):
            for jj in range(len(vbs)):
                print(f"pattern {len(vbs) * ii + jj + 1:d}/{len(vas) * len(vbs):d},"
                      f" elapsed time = {time.perf_counter() - tstart:.2f}s")

                vec_a = vas[ii]
                vec_b = vbs[jj]

                try:
                    recp_va, recp_vb = dmd.get_reciprocal_vects(vec_a, vec_b)
                except:
                    continue

                # normal phase function
                phase = dmd.get_sim_phase(vec_a, vec_b, nphases, phase_index, dmd_size)

                # estimate phase from FFT
                pattern, _ = dmd.get_sim_pattern(dmd_size, vec_a, vec_b, nphases, phase_index)

                window = hann(nx)[None, :] * hann(ny)[:, None]
                pattern_ft = fft.fftshift(fft.fft2(fft.ifftshift(pattern * window)))
                fx = fft.fftshift(fft.fftfreq(nx))
                fy = fft.fftshift(fft.fftfreq(ny))

                try:
                    phase_direct = np.angle(tools.get_peak_value(pattern_ft, fx, fy, recp_vb, peak_pixel_size=2))
                except:
                    # recp_vb too close to edge of pattern
                    continue

                phase_diff = np.min([np.abs(phase - phase_direct), np.abs(phase - phase_direct - 2*np.pi)])

                # assert np.round(phase_diff, 1) == 0
                self.assertAlmostEqual(phase_diff, 0, 1)

    def test_pattern_all_phases(self):
        """
        Test that can predict phases for ALL reciprocal lattice vectors in pattern using the
        get_efield_fourier_components() function.

        Do this for a few lattice vectors
        :return:
        """
        dmd_size = [1920, 1080]
        nx, ny = dmd_size
        nphases = 3
        phase_index = 0

        # list of vectors to test
        vec_as = [[-7, 7], [-5, 5]]
        vec_bs = [[3, 9], [3, 9]]
        # vec_as = [[-1, 117]]
        # vec_bs = [[-6, 117]]

        # other unit vectors with larger unit cells that don't perfectly tile DMD do not work as well.
        # close, but larger errors
        # [12, -12]
        # [0, 18]

        for vec_a, vec_b in zip(vec_as, vec_bs):

            pattern, _, _, angles, frqs, periods, phases, recp_vects_a, recp_vects_b, min_leakage_angle = \
                dmd.vects2pattern_data(dmd_size, [vec_a], [vec_b], nphases=nphases)

            pattern = pattern[0, 0]
            unit_cell, xc, yc = dmd.get_sim_unit_cell(vec_a, vec_b, nphases)

            # get ft
            window = hann(nx)[None, :] * hann(ny)[:, None]
            pattern_ft = fft.fftshift(fft.fft2(fft.ifftshift(pattern * window)))
            fxs = fft.fftshift(fft.fftfreq(nx))
            dfx = fxs[1] - fxs[0]
            fys = fft.fftshift(fft.fftfreq(ny))
            dfy = fys[1] - fys[0]

            # get expected pattern components
            efield, ns, ms, vecs = dmd.get_efield_fourier_components(unit_cell, xc, yc, vec_a, vec_b,
                                                                     nphases, phase_index, dmd_size, nmax=40)
            # divide by size of DC component
            efield = efield / np.max(np.abs(efield))

            # get phase from fft
            efield_img = np.zeros(efield.shape, dtype=complex)
            for ii in range(len(ns)):
                for jj in range(len(ms)):
                    if np.abs(vecs[ii, jj][0]) > 0.5 or np.abs(vecs[ii, jj][1]) > 0.5:
                        efield_img[ii, jj] = np.nan
                        continue

                    try:
                        efield_img[ii, jj] = tools.get_peak_value(pattern_ft, fxs, fys, vecs[ii, jj], 2)
                    except:
                        efield_img[ii, jj] = np.nan

            # divide by size of DC component
            efield_img = efield_img / np.nanmax(np.abs(efield_img))

            # import matplotlib.pyplot as plt
            # from matplotlib.colors import PowerNorm
            # plt.figure()
            # fs = np.linalg.norm(vecs, axis=2)
            #
            # xlim = [-0.05, 1.2*np.max([fxs.max(), fys.max()])]
            # to_use = np.logical_and(np.logical_not(np.isnan(efield_img)), np.abs(efield) > 1e-8)
            #
            # plt.subplot(2, 2, 1)
            # plt.semilogy(fs[to_use], np.abs(efield_img[to_use]), 'r.')
            # plt.semilogy(fs[to_use], np.abs(efield[to_use]), 'bx')
            # plt.xlim(xlim)
            # plt.ylabel('amplitude')
            # plt.xlabel('Frq 1/mirrors')
            # plt.legend(['FFT', 'Prediction'])
            #
            # plt.subplot(2, 2, 2)
            # plt.plot(fs[to_use], np.abs(efield_img[to_use]), 'r.')
            # plt.plot(fs[to_use], np.abs(efield[to_use]), 'bx')
            # plt.xlim(xlim)
            # plt.ylabel('amplitude')
            # plt.xlabel('Frq 1/mirrors')
            #
            # plt.subplot(2, 2, 3)
            # plt.plot(fs[to_use], np.mod(np.angle(efield_img[to_use]), 2*np.pi), 'r.')
            # plt.plot(fs[to_use], np.mod(np.angle(efield[to_use]), 2*np.pi), 'bx')
            # plt.xlim(xlim)
            # plt.ylabel('phases')
            # plt.xlabel('Frq 1/mirrors')
            #
            # plt.subplot(2, 2, 4)
            # extent = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx, fys[-1] + 0.5 * dfy, fys[0] - 0.5 * dfy]
            # plt.imshow(np.abs(pattern_ft), extent=extent, norm=PowerNorm(gamma=0.1))
            #
            # assert np.round(np.nanmax(np.abs(efield_img - efield)), 12) == 0
            # self.assertAlmostEqual(np.nanmax(np.abs(efield_img - efield)), 0, 12)
            to_compare = np.logical_not(np.isnan(efield_img - efield))
            np.testing.assert_allclose(efield_img[to_compare], efield[to_compare], atol=1e-12)

    def test_affine_phase_xform(self):
        """
        Test transforming pattern and phases through affine xform


        :return:
        """

        # ##############################
        # define pattern
        # ##############################
        dmd_size = [1920, 1080]  # [nx, ny]
        nphases = 3
        phase_index = 0
        nmax = 40
        nx = 500
        ny = 500
        roi = rois.get_centered_roi([1024, 1024], [ny, nx])

        # vec_a = np.array([8, 17])
        # vec_b = np.array([3, -6])
        vec_a = [1, -117]
        vec_b = [6, -117]

        pattern, _ = dmd.get_sim_pattern(dmd_size, vec_a, vec_b, nphases, phase_index)
        unit_cell, xc, yc = dmd.get_sim_unit_cell(vec_a, vec_b, nphases)

        # ##############################
        # define affine matrix
        # ##############################
        # affine_mat = affine.params2xform([2, 15 * np.pi/180, 15, 1.7, -13 * np.pi/180, 3])
        affine_mat = np.array([[-1.01788979e+00, -1.04522661e+00, 2.66353915e+03],
                               [9.92641451e-01, -9.58516962e-01, 7.83771959e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # get affine matrix for our ROI
        affine_xform_roi = affine.xform_shift_center(affine_mat, cimg_new=(roi[2], roi[0]))

        # ###########################################
        # estimate phases/intensity after affine transformation using model
        ###########################################
        # transform pattern phase
        efields, ns, ms, vecs = dmd.get_efield_fourier_components(unit_cell, xc, yc, vec_a, vec_b, nphases, phase_index,
                                                                  dmd_size, nmax=nmax, origin="fft")
        efields = efields / np.max(np.abs(efields))

        vecs_xformed = np.zeros(vecs.shape)
        vecs_xformed[..., 0], vecs_xformed[..., 1], phases = \
            affine.xform_sinusoid_params_roi(vecs[..., 0], vecs[..., 1], np.angle(efields), pattern.shape, roi,
                                             affine_mat, input_origin="fft", output_origin="fft")
        efields_xformed = np.abs(efields) * np.exp(1j * phases)

        ###########################################
        # estimate phases/intensity after affine transformation numerically
        ###########################################
        # transform pattern to image space
        img_coords = np.meshgrid(range(nx), range(ny))
        # interpolation preserves phases but can distort Fourier components
        pattern_xform = affine.xform_mat(pattern, affine_xform_roi, img_coords, mode="interp")
        # taking nearest pixel does a better job with amplitudes, but can introduce fourier components that did not exist before
        # pattern_xform_nearest = affine.affine_xform_mat(pattern, affine_xform_roi, img_coords, mode="nearest")

        window = np.expand_dims(hann(nx), axis=0) * \
                 np.expand_dims(hann(ny), axis=1)

        pattern_ft = fft.fftshift(fft.fft2(fft.ifftshift(pattern_xform * window)))
        fx = fft.fftshift(fft.fftfreq(nx))
        fy = fft.fftshift(fft.fftfreq(ny))

        efields_direct = np.zeros(efields_xformed.shape, dtype=complex)
        for ii in range(efields.shape[0]):
            for jj in range(efields.shape[1]):
                if np.abs(vecs_xformed[ii, jj][0]) > 0.5 or np.abs(vecs_xformed[ii, jj][1]) > 0.5:
                    efields_direct[ii, jj] = np.nan
                else:
                    try:
                        efields_direct[ii, jj] = tools.get_peak_value(pattern_ft, fx, fy,
                                                                      vecs_xformed[ii, jj],
                                                                      peak_pixel_size=2)
                    except ZeroDivisionError:
                        efields_direct[ii, jj] = np.nan

        efields_direct = efields_direct / np.nanmax(np.abs(efields_direct))

        # compare results
        to_compare = np.logical_and(np.abs(efields_xformed) > 0.05, np.logical_not(np.isnan(efields_direct)))
        # test angles
        np.testing.assert_allclose(np.angle(efields_xformed[to_compare]),
                                   np.angle(efields_direct[to_compare]),
                                   atol=0.003)
        # test amplitudes
        np.testing.assert_allclose(np.abs(efields_xformed[to_compare]),
                                   np.abs(efields_direct[to_compare]),
                                   atol=0.07)
        np.testing.assert_allclose(np.abs(efields_xformed[to_compare]),
                                   np.abs(efields_direct[to_compare]),
                                   rtol=0.25)


if __name__ == "__main__":
    unittest.main()
