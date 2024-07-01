import unittest
from time import perf_counter
import numpy as np
from scipy.signal.windows import hann
from scipy.fft import fftshift, fftfreq
from localize_psf.affine import params2xform, xform_sinusoid_params, xform_mat
from localize_psf.rois import get_centered_rois
from mcsim.analysis.sim_reconstruction import get_peak_value
from mcsim.analysis.fft import ft2
import mcsim.analysis.dmd_patterns as dmd


class TestPatterns(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_sim_phase(self):
        """
        Test get_sim_phase() on the main frequency component for one SIM pattern. Ensure works for all phase indices.

        Tests only a single pattern vector
        :return:
        """
        nx = 1920
        ny = 1080
        nphases = 3

        fx = fftshift(fftfreq(nx))
        fy = fftshift(fftfreq(ny))
        window = np.outer(hann(ny), hann(nx))

        va = [-3, 11]
        vb = [3, 12]
        rva, rvb = dmd.get_reciprocal_vects(va, vb)
        rva = rva.ravel()
        rvb = rvb.ravel()

        for jj in range(nphases):
            phase = dmd.get_sim_phase(va, vb, nphases, jj, [nx, ny], use_fft_origin=True)
            pattern, _ = dmd.get_sim_pattern([nx, ny], va, vb, nphases, jj)
            pattern_ft = ft2(pattern * window)

            peak_val = get_peak_value(pattern_ft, fx, fy, rvb, 2)
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

        tstart = perf_counter()
        for ii in range(len(vas)):
            for jj in range(len(vbs)):
                print(f"pattern {len(vbs) * ii + jj + 1:d}/{len(vas) * len(vbs):d},"
                      f" elapsed time = {perf_counter() - tstart:.2f}s")

                vec_a = vas[ii]
                vec_b = vbs[jj]

                try:
                    recp_va, recp_vb = dmd.get_reciprocal_vects(vec_a, vec_b)
                    recp_va = recp_va.ravel()
                    recp_vb = recp_vb.ravel()
                except ValueError:
                    # linearly dependent vectors
                    continue

                # normal phase function
                phase = dmd.get_sim_phase(vec_a, vec_b, nphases, phase_index, dmd_size, use_fft_origin=True)

                # estimate phase from FFT
                pattern, _ = dmd.get_sim_pattern(dmd_size, vec_a, vec_b, nphases, phase_index)

                window = np.outer(hann(ny), hann(nx))
                pattern_ft = ft2(pattern * window)
                fx = fftshift(fftfreq(nx))
                fy = fftshift(fftfreq(ny))

                try:
                    phase_direct = np.angle(get_peak_value(pattern_ft, fx, fy, recp_vb, peak_pixel_size=2))
                except ZeroDivisionError:
                    # recp_vb too close to edge of pattern
                    continue

                phase_diff = np.min([np.abs(phase - phase_direct), np.abs(phase - phase_direct - 2*np.pi)])

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

            pattern, data = dmd.vects2pattern_data(dmd_size,
                                                   [vec_a],
                                                   [vec_b],
                                                   nphases=nphases)

            pattern = pattern[0, 0]
            unit_cell, xc, yc = dmd.get_sim_unit_cell(vec_a, vec_b, nphases)

            # get ft
            window = np.outer(hann(ny), hann(nx))
            pattern_ft = ft2(pattern * window)
            fxs = fftshift(fftfreq(nx))
            fys = fftshift(fftfreq(ny))
            dfx = fxs[1] - fxs[0]
            dfy = fys[1] - fys[0]

            # get expected pattern components
            efield, ns, ms, vecs = dmd.get_efield_fourier_components(unit_cell,
                                                                     xc,
                                                                     yc,
                                                                     vec_a,
                                                                     vec_b,
                                                                     nphases,
                                                                     phase_index,
                                                                     dmd_size,
                                                                     nmax=40)
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
                        efield_img[ii, jj] = get_peak_value(pattern_ft, fxs, fys, vecs[ii, jj], 2)
                    except ZeroDivisionError:
                        efield_img[ii, jj] = np.nan

            # divide by size of DC component
            efield_img = efield_img / np.nanmax(np.abs(efield_img))

            # compare
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
        nx = 502
        ny = 500
        roi = get_centered_rois([1024, 1024], [ny, nx])[0]

        # vec_a = np.array([8, 17])
        # vec_b = np.array([3, -6])
        vec_a = [1, -117]
        vec_b = [6, -117]

        pattern, _ = dmd.get_sim_pattern(dmd_size, vec_a, vec_b, nphases, phase_index)
        unit_cell, xc, yc = dmd.get_sim_unit_cell(vec_a, vec_b, nphases)

        # ##############################
        # define affine matrix
        # ##############################
        affine_mat = np.array([[-1.01788979e+00, -1.04522661e+00, 2.66353915e+03],
                               [9.92641451e-01, -9.58516962e-01, 7.83771959e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # get affine matrix for our ROI
        affine_xform_roi = params2xform([1, 0, -roi[2], 1, 0, -roi[0]]).dot(affine_mat)

        # ###########################################
        # estimate phases/intensity after affine transformation using model
        ###########################################
        # transform pattern phase
        efields, ns, ms, vecs = dmd.get_efield_fourier_components(unit_cell,
                                                                  xc, yc,
                                                                  vec_a,
                                                                  vec_b,
                                                                  nphases,
                                                                  phase_index,
                                                                  dmd_size,
                                                                  nmax=nmax,
                                                                  use_fft_origin=True)
        efields = efields / np.max(np.abs(efields))

        vecs_xformed = np.zeros(vecs.shape)

        xform_input2edge = params2xform([1, 0, (pattern.shape[1] // 2),
                                         1, 0, (pattern.shape[0] // 2)])
        xform_full2roi = params2xform([1, 0, -roi[2],
                                       1, 0, -roi[0]])
        xform_edge2output = params2xform([1, 0, -((roi[3] - roi[2]) // 2),
                                          1, 0, -((roi[1] - roi[0]) // 2)])
        xform_full = xform_edge2output.dot(xform_full2roi.dot(affine_mat.dot(xform_input2edge)))
        vecs_xformed[..., 0], vecs_xformed[..., 1], phases = xform_sinusoid_params(vecs[..., 0],
                                                                                   vecs[..., 1],
                                                                                   np.angle(efields),
                                                                                   xform_full)
        efields_xformed = np.abs(efields) * np.exp(1j * phases)

        ###########################################
        # estimate phases/intensity after affine transformation numerically
        ###########################################
        # todo: why does this still work? Expect second option to work instead
        # transform pattern to image space
        xx, yy = np.meshgrid(range(nx), range(ny))
        # interpolation preserves phases but can distort Fourier components
        pattern_xform = xform_mat(pattern,
                                  affine_xform_roi,
                                  (xx, yy),
                                  mode="interp")

        # swap_xy = np.array([[0, 1, 0],
        #                     [1, 0, 0],
        #                     [0, 0, 1]])
        # affine_xform_roi_yx = swap_xy.dot(affine_xform_roi.dot(swap_xy))
        # pattern_xform2 = xform_mat(pattern,
        #                            affine_xform_roi_yx,
        #                           (yy, xx),
        #                            mode="interp")

        # taking nearest pixel (i.e. mode = "nearest") does a better job with amplitudes,
        # but can introduce fourier components that did not exist before

        window = np.outer(hann(ny), hann(nx))
        pattern_ft = ft2(pattern_xform * window)
        fx = fftshift(fftfreq(nx))
        fy = fftshift(fftfreq(ny))

        efields_direct = np.zeros(efields_xformed.shape, dtype=complex)
        for ii in range(efields.shape[0]):
            for jj in range(efields.shape[1]):
                if np.abs(vecs_xformed[ii, jj][0]) > 0.5 or np.abs(vecs_xformed[ii, jj][1]) > 0.5:
                    efields_direct[ii, jj] = np.nan
                else:
                    try:
                        efields_direct[ii, jj] = get_peak_value(pattern_ft,
                                                                fx,
                                                                fy,
                                                                vecs_xformed[ii, jj],
                                                                peak_pixel_size=2)
                    except ZeroDivisionError:
                        efields_direct[ii, jj] = np.nan

        efields_direct = efields_direct / np.nanmax(np.abs(efields_direct))

        # compare results
        to_compare = np.logical_and(np.abs(efields_xformed) > 0.05,
                                    np.logical_not(np.isnan(efields_direct)))
        # test angles
        np.testing.assert_allclose(np.angle(efields_xformed[to_compare]),
                                   np.angle(efields_direct[to_compare]),
                                   atol=0.03)
        # test amplitudes
        np.testing.assert_allclose(np.abs(efields_xformed[to_compare]),
                                   np.abs(efields_direct[to_compare]),
                                   atol=0.07)
        np.testing.assert_allclose(np.abs(efields_xformed[to_compare]),
                                   np.abs(efields_direct[to_compare]),
                                   rtol=0.25)

    def test_reduce_basis(self):
        v1 = np.array([1, 0], dtype=int)
        v2 = np.array([1, 1], dtype=int)

        v1_red, v2_red = dmd.reduce_basis(v1, v2)
        v1_red = v1_red.ravel()
        v2_red = v2_red.ravel()

        v1_red_actual = np.array([1, 0], dtype=int)
        v2_red_actual = np.array([0, 1], dtype=int)

        np.testing.assert_allclose(v1_red_actual, v1_red, atol=1e-12)
        np.testing.assert_allclose(v2_red_actual, v2_red, atol=1e-12)

    def test_get_sim_unitcell(self):
        v1 = np.array([3, 0], dtype=int)
        v2 = np.array([0, 3], dtype=int)
        cell, x, y = dmd.get_sim_unit_cell(v1, v2, 3)

        cell_actual = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [0, 0, 0]])
        x_actual = np.array([0, 1, 2])
        y_actual = np.array([0, 1, 2])

        np.testing.assert_allclose(cell, cell_actual, atol=1e-12)
        np.testing.assert_allclose(x, x_actual, atol=1e-12)
        np.testing.assert_allclose(y, y_actual, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
