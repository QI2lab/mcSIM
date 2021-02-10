import unittest

import numpy as np
import scipy.signal
from scipy import fft

import sim_reconstruction as sim
import affine as affine
import analysis_tools as tools

class Test_affine(unittest.TestCase):

    def setUp(self):
        pass

    def test_xform_sinusoid_params(self):
        """
        test the xform_sinusoid_params() function by constructing sinusoid pattern and passing through an affine
        transformation. Compare the resulting frequency determined numerically with the resulting frequency determined
        from the initial frequency + affine parameters
        :return:
        """

        # define object space parameters
        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)

        # define affine transform
        xform = affine.params2xform([1.4296003114502853, 2.3693263411981396, 2671.39109,
                                     1.4270495211450602, 2.3144621088632635, 790.402632])

        # sinusoid parameter transformation
        fxi, fyi, phase_img = affine.xform_sinusoid_params(fobj[0], fobj[1], phase_obj, xform)
        fimg = np.array([fxi, fyi])

        # compared with phase from fitting image directly
        out_coords = np.meshgrid(range(2048), range(2048))
        img = affine.xform_fn(fn, xform, out_coords)

        phase_fit = float(sim.fit_phase_realspace(img, fimg, 1, origin="edge"))

        # todo: could also test frequencies if wanted...

        self.assertAlmostEqual(phase_img, phase_fit, 5)

    def test_xform_phase_translation(self):
        """
        Test function xform_phase_translation() function by defining sinusoid image and then translating. Compare numerically
        determined phase with that given by xform_phase_translation().
        :return:
        """
        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)

        # center of "new" coordinates in "old" coordinates
        # xn = xo - cx
        # yn = yo - cy
        cx = -100.62362
        cy = 0.3743743
        phase_xlated = affine.xform_phase_translation(fobj[0], fobj[1], phase_obj, [cx, cy])

        # fn of new coordinates
        # xo = xn + cx
        fn_xlated = lambda xn, yn: fn(xn + cx, yn + cy)

        x_new, y_new = np.meshgrid(range(500), range(500))
        img_new = fn_xlated(x_new, y_new)

        phase_xlated_test = float(sim.fit_phase_realspace(img_new, fobj, 1, origin="edge"))

        self.assertAlmostEqual(phase_xlated, phase_xlated_test, 3)

    def test_xform_phase_roi(self):
        """
        Test function xform_phase_translation() function by defining sinusoid image and then cropping. Compare numerically
        determined phase with that given by xform_phase_translation().
        :return:
        """
        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)
        xo, yo = np.meshgrid(range(500), range(500))
        img = fn(xo, yo)

        # get ROI phase from function
        # center of "new" coordinates in "old" coordinates
        # xn = xo - cx
        # yn = yo - cy
        roi = [30, 450, 100, 210]
        phase_roi = affine.xform_phase_translation(fobj[0], fobj[1], phase_obj, [roi[2], roi[0]])

        # determine ROI phase from fitting
        img_roi = img[roi[0]:roi[1], roi[2]:roi[3]]
        phase_roi_test = float(sim.fit_phase_realspace(img_roi, fobj, 1, origin="edge"))

        self.assertAlmostEqual(phase_roi, phase_roi_test, 8)

    def test_xform_sinusoid_params_roi(self):
        """
        Test function xform_sinusoid_params_roi() by constructing sinusoid pattern and passing through an affine
        transformation. Compare the resulting frequency determined numerically with the resulting frequency determined
        from the initial frequency + affine parameters
        :return:
        """
        # define object space parameters
        # roi_img = [0, 2048, 0, 2048]
        roi_img = [512, 788, 390, 871]

        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)

        # define affine transform
        xform = affine.params2xform([1.4296003114502853, 2.3693263411981396, 2671.39109,
                                     1.4270495211450602, 2.3144621088632635, 790.402632])

        # sinusoid parameter transformation
        fxi, fyi, phase_roi = affine.xform_sinusoid_params_roi(fobj[0], fobj[1], phase_obj, None, roi_img, xform,
                                                               input_origin="edge", output_origin="edge")
        fimg = np.array([fxi, fyi])

        # FFT phase
        _, _, phase_roi_ft = affine.xform_sinusoid_params_roi(fobj[0], fobj[1], phase_obj, None, roi_img, xform,
                                                              input_origin="edge", output_origin="fft")

        # compared with phase from fitting image directly
        out_coords = np.meshgrid(range(roi_img[2], roi_img[3]), range(roi_img[0], roi_img[1]))
        img = affine.xform_fn(fn, xform, out_coords)
        phase_fit_roi = float(sim.fit_phase_realspace(img, fimg, 1, phase_guess=phase_roi, origin="edge"))

        # phase FFT
        ny, nx = img.shape
        window = scipy.signal.windows.hann(nx)[None, :] * scipy.signal.windows.hann(ny)[:, None]
        img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img * window)))
        fx = tools.get_fft_frqs(nx, 1)
        fy = tools.get_fft_frqs(ny, 1)

        peak = tools.get_peak_value(img_ft, fx, fy, fimg, 2)
        phase_fit_roi_ft = np.mod(np.angle(peak), 2*np.pi)

        # accuracy is limited by frequency fitting routine...
        self.assertAlmostEqual(phase_roi, phase_fit_roi, 1)
        # probably limited by peak height finding routine
        self.assertAlmostEqual(phase_roi_ft, phase_fit_roi_ft, 3)

if __name__ == "__main__":
    unittest.main()
