import unittest
import mcsim.analysis.analysis_tools as tools
from scipy import fft
import numpy as np


class TestTools(unittest.TestCase):

    def setUp(self):
        pass

    def test_expand_fourier_sp(self):
        """
        Test expand_fourier_sp() function
        :return:
        """

        arr = np.array([[1, 2], [3, 4]])
        arr_ft = fft.fftshift(fft.fft2(fft.ifftshift(arr)))

        arr_ft_ex = tools.resample_bandlimited_ft(arr_ft, (2, 2))
        arr_ex = fft.fftshift(fft.ifft2(fft.ifftshift(arr_ft_ex)))

        self.assertTrue(np.array_equal(arr_ex.real, np.array([[1, 1.5, 2, 1.5],
                                                               [2, 2.5, 3, 2.5],
                                                               [3, 3.5, 4, 3.5],
                                                               [2, 2.5, 3, 2.5]])))

    def test_expand_fourier_sp_odd1d(self):
        """
        Test function with odd input size
        """
        arr = np.random.rand(151)
        arr_ft = fft.fftshift(fft.fft(fft.ifftshift(arr)))

        arr_ex_ft = tools.resample_bandlimited_ft(arr_ft, (2,))
        arr_exp = fft.fftshift(fft.ifft(fft.ifftshift(arr_ex_ft))).real

        max_err = np.max(np.abs(arr_exp[1::2] - arr))
        self.assertTrue(max_err < 1e-14)

    def test_expand_fourier_sp_even1d(self):
        """
        test function with even input size
        """
        arr = np.random.rand(100)
        arr_ft = fft.fftshift(fft.fft(fft.ifftshift(arr)))

        arr_ex_ft = tools.resample_bandlimited_ft(arr_ft, (2,))
        arr_exp = fft.fftshift(fft.ifft(fft.ifftshift(arr_ex_ft))).real

        max_err = np.max(np.abs(arr_exp[::2] - arr))
        self.assertTrue(max_err < 1e-14)

    def test_expand_fourier_sp_odd2d(self):
        """
        Test function with odd input size
        """
        arr = np.random.rand(151, 151)
        arr_ft = fft.fftshift(fft.fft2(fft.ifftshift(arr)))

        arr_ex_ft = tools.resample_bandlimited_ft(arr_ft, (2, 2))
        arr_exp = fft.fftshift(fft.ifft2(fft.ifftshift(arr_ex_ft))).real

        max_err = np.max(np.abs(arr_exp[1::2, 1::2] - arr))
        self.assertTrue(max_err < 1e-14)

    def test_expand_fourier_sp_even2d(self):
        """
        test function with even input size
        """
        arr = np.random.rand(100, 100)
        arr_ft = fft.fftshift(fft.fft2(fft.ifftshift(arr)))

        arr_ex_ft = tools.resample_bandlimited_ft(arr_ft, (2, 2))
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

        # fx = tools.get_fft_frqs(img.shape[1], dx)
        # fy = tools.get_fft_frqs(img.shape[0], dx)
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
