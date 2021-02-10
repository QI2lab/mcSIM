import numpy as np
import matplotlib.pyplot as plt
import analysis_tools as tools
from scipy import fft

class pupil2:
    def __init__(self, psf, zs, dx, wavelength, na, n, mag):

        self.beta = 0.5

        # experiment data
        self.psf = psf
        self.nz, self.ny, self.nx = psf.shape
        self.zs = zs
        self.dx = dx

        # system parameters
        self.wavelength = wavelength
        self.na = na
        self.n = n
        self.mag = mag
        self.fmax = self.na / self.wavelength

        # real-space coordinates
        self.x = tools.get_fft_frqs(self.nx, self.dx)
        self.y = tools.get_fft_frqs(self.ny, self.dx)

        # frequency data
        self.fx = tools.get_fft_frqs(self.nx, self.dx)
        self.fy = tools.get_fft_frqs(self.nx, self.dx)
        self.ff = np.sqrt(self.fx[None, :]**2 + self.fy[:, None]**2)
        # we must divide this by the magnification, because we are working in image space (i.e. we typically write
        # all of our integrals after making the transforming x-> x/(M*n) and kx -> kx*M*n, which makes the dimensions
        # of image space the same as object space
        self.kperp = (2*np.pi) * np.sqrt(self.fx[None, :]**2 + self.fy[:, None]**2) / self.mag
        self.kz = (2*np.pi) * np.sqrt((1 / self.wavelength)**2 - self.kperp)

        # self.sin_tp = kperp / keff
        # self.cos_tp = np.sqrt(1 - self.sin_tp**2)
        # if mode == 'abbe':
        #     self.apodization = self.cos_tp ** (-1/2) * (1 - (self.mag / self.n * self.sin_tp)**2) ** (-1/4)
        # elif mode == 'herschel':
        #     # 1 / cos(theta')
        #     self.apodization = self.cos_tp ** (-1)
        # elif mode == 'none':
        #     self.apodization = 1
        # else:
        #     raise Exception("mode must be 'abbe' or 'herschel', but was '%s'" % mode)

        # initialize pupil
        self.pupil = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(self.ny, self.nx)))

        # track algorithm
        self.iteration = 0
        self.mean_err = [np.abs(self.get_psf() - self.psf)]

    def project_to_psf(self, pupil, ind):
        """
        project pupil to give correct amplitudes for psf slices
        :param pupil:
        :param ind:
        :return:
        """
        psf_a = fft.fftshift(fft.fft2(fft.ifftshift(pupil * np.exp(1j * self.kz * self.zs[ind]))))
        pupil = fft.fftshift(fft.ifft2(fft.ifftshift(psf_a / np.abs(psf_a) * np.sqrt(self.psf[ind])))) * np.exp(-1j * self.kz * self.zs[ind])
        return pupil

    def project_support(self, pupil):
        """
        project pupil to have correct support
        :param pupil:
        :return:
        """
        pupil[self.ff > self.fmax] = 0
        return pupil

    def get_psf(self):
        """
        Get amplitude psf from pupil
        :return:
        """
        psf = np.zeros(self.psf.shape)
        for ii in range(len(self.zs)):
            psf = np.abs(fft.fftshift(fft.fft2(fft.ifftshift(pupil * np.exp(1j * self.kz * self.zs[ii])))))**2

        return psf

    def iterate(self):
        pupils = np.zeros(self.psf.shape)
        for ii in range(len(self.zs)):
            pupils[ii] = (1 - self.beta) * self.pupil + self.beta * self.project_support(self.project_to_psf(self.pupil, ii))

        self.pupil = np.mean(pupils, axis=0)
        self.iteration += 1
        self.mean_err.append(np.abs(self.psf - self.get_psf()))



# pupil and phase retrieval
class pupil():
    def __init__(self, nx, dx, wavelength, na, psf, zs=np.array([0]), n=1.5, mag=100, mode='abbe'):
        """

        :param nx:
        :param dx:
        :param wavelength:
        :param na:
        :param psf:
        :param zs:
        :param n: index of refraction in object space
        :param mag: magnification
        :param mode: 'abbe' or 'herschel'
        """
        self.nx = nx
        self.dx = dx
        self.wavelength = wavelength
        self.n = n
        self.mag = mag
        self.na = na
        self.psf = psf
        self.zs = zs
        if self.zs.size != self.psf.shape[0]:
            raise ValueError("first dimension of psf does not match z size")
        # efield max frequency
        self.fmax = self.na / self.wavelength

        #
        self.x = self.dx * (np.arange(self.nx) - self.nx // 2)
        self.y = self.x

        # frequency data
        self.fx = tools.get_fft_frqs(self.nx, self.dx)
        self.fy = tools.get_fft_frqs(self.nx, self.dx)
        self.fxfx, self.fyfy = np.meshgrid(self.fx, self.fy)
        self.ff = np.sqrt(self.fxfx**2 + self.fyfy**2)

        # we must divide this by the magnification, because we are working in image space (i.e. we typically write
        # all of our integrals after making the transforming x-> x/(M*n) and kx -> kx*M*n, which makes the dimensions
        # of image space the same as object space
        kperp = np.sqrt((2 * np.pi * self.fxfx) ** 2 + (2 * np.pi * self.fyfy) ** 2) / self.mag
        keff = 2 * np.pi / self.wavelength
        self.sin_tp = kperp / keff
        self.cos_tp = np.sqrt(1 - self.sin_tp**2)
        if mode == 'abbe':
            self.apodization = self.cos_tp ** (-1/2) * (1 - (self.mag / self.n * self.sin_tp)**2) ** (-1/4)
        elif mode == 'herschel':
            # 1 / cos(theta')
            self.apodization = self.cos_tp ** (-1)
        elif mode == 'none':
            self.apodization = 1
        else:
            raise ValueError("mode must be 'abbe' or 'herschel', but was '%s'" % mode)

        # initialize pupil with random phases
        # absorb apodization into pupil
        self.pupil = self.apodization * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(nx, nx)))
        self.pupil[self.ff > self.fmax] = 0

        self.iteration = 0
        self.mean_err = []

        # use norm so we can keep the pupil magnitude fixed at 1, no matter what the normalization of the PSF is.
        # based on fact we expect sum_f |g(f)|^2 / N = sum_r |g(r)|^2 = sum_r psf(r)
        # this norm is not changed by any operation
        # actually not totally true, because the pupil cropping operation can change it...
        #self.norm = np.sqrt(np.sum(np.abs(self.pupil)**2) / self.nx**2 / np.sum(self.psf))
        self.norm = 1

    def get_defocus(self, z):
        """

        :param z:
        :return:
        """
        # kz = (self.mag**2 * self.n) * (2*np.pi / self.wavelength) * self.cos_tp
        kz = (self.mag ** 2 / self.n) * (2 * np.pi / self.wavelength) * self.cos_tp
        kz[np.isnan(kz)] = 0
        return np.exp(-1j * kz * z)

    def get_amp_psf(self, z, normalize=True):
        """

        :param z:
        :return:
        """
        # recall that iffshift is the inverse of fftshift. Since we have centered pupil with fftshift, now
        # need to uncenter it.
        amp_psf = fft.fftshift(fft.ifft2(fft.ifftshift(self.pupil * self.get_defocus(z))))
        if normalize:
            amp_psf = amp_psf / self.norm
        return amp_psf

    def get_pupil(self, amp_psf, normalize=True):
        """

        :param amp_psf:
        :return:
        """
        pupil = fft.fftshift(fft.fft2(fft.ifftshift(amp_psf)))
        if normalize:
            pupil = pupil * self.norm
        return pupil

    def iterate_pupil(self):
        """

        :return:
        """

        # get amplitude psf
        psf_e = np.zeros(self.psf.shape, dtype=np.complex)
        psf_e_new = np.zeros(self.psf.shape, dtype=np.complex)
        pupils_new_phase = np.zeros(self.psf.shape, dtype=np.complex)
        for ii in range(self.zs.size):
            psf_e[ii] = self.get_amp_psf(self.zs[ii])
            # new amplitude psf from phase of transformed pupil and amplitude of measured psf
            psf_e_new[ii] = np.sqrt(self.psf[ii]) * np.exp(1j * np.angle(psf_e[ii]))
            # weird issue with numpy square root for very small positive numbers
            # think probably related to https://github.com/numpy/numpy/issues/11448
            psf_e_new[ii][np.isnan(psf_e_new[ii])] = 0

            # get new pupil by transforming, then undoing defocus
            xform = self.get_pupil(psf_e_new[ii]) * self.get_defocus(self.zs[ii]).conj()
            pupils_new_phase[ii] = np.angle(xform)

        # get error
        self.mean_err.append(np.nanmean(np.abs(np.abs(psf_e)**2 - self.psf) / np.nanmax(self.psf)))

        # pupil_new = scipy.ndimage.gaussian_filter(pupil_new_mag, sigma=1) * np.exp(1j * pupil_new_phase)
        phase = np.angle(np.mean(np.exp(1j * pupils_new_phase), axis=0))
        self.pupil = np.abs(self.pupil) * np.exp(1j * phase)
        # this should already be enforced by the initial pupil, but can't hurt
        self.pupil[self.ff > self.fmax] = 0
        self.iteration += 1

    def show_current_pupil(self):

        extent_real = [self.x[0] - 0.5*self.dx, self.x[-1] + 0.5*self.dx, self.y[-1] + 0.5*self.dx, self.y[0] - 0.5*self.dx]

        df = self.fx[1] - self.fx[0]
        extent_ft = [self.fx[0] - 0.5*df, self.fx[-1] + 0.5*df, self.fy[-1] + 0.5*df, self.fy[0] - 0.5*df]

        #
        psf_e = np.zeros(self.psf.shape, dtype=np.complex)
        for ii in range(self.zs.size):
            psf_e[ii] = self.get_amp_psf(self.zs[ii])

        psf_i = np.abs(psf_e) ** 2

        figh = plt.figure()
        plt.suptitle('iteration = %d' % self.iteration)
        nrows = 3
        ncols = self.zs.size

        for ii in range(ncols):
            plt.subplot(nrows, ncols, ii + 1)
            plt.imshow(self.psf[ii] / np.nanmax(self.psf), extent=extent_real)
            plt.title('PSF / max at z=%0.3fum' % self.zs[ii])

            plt.subplot(nrows, ncols, ncols + ii + 1)
            plt.imshow(psf_i[ii] / np.nanmax(psf_i), extent=extent_real)
            plt.title('PSF from pupil / max')

            plt.subplot(nrows, ncols, 2*ncols + ii + 1)
            plt.imshow((self.psf[ii] - psf_i[ii]) / np.nanmax(self.psf))
            plt.title('(PSF - PSF from pupil) / max(psf')
            plt.colorbar()

        figh = plt.figure()
        nrows = 2
        ncols = 3
        zind = np.argmin(np.abs(self.zs))

        plt.subplot(nrows, ncols, 1)
        plt.imshow(np.abs(psf_e[zind]), extent=extent_real)
        plt.title('PSF amp, magnitude')

        plt.subplot(nrows, ncols, 4)
        plt.imshow(np.angle(psf_e[zind]) / np.pi, vmin=-1, vmax=1, extent=extent_real)
        plt.title('PSF phase (pi)')

        plt.subplot(nrows, ncols, 2)
        plt.imshow(np.abs(self.pupil), extent=extent_ft)
        plt.xlim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.ylim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.title('Pupil, magnitude')

        plt.subplot(nrows, ncols, 5)
        phase = np.unwrap(np.angle(self.pupil))
        phase[self.ff > self.fmax] = np.nan

        plt.imshow(phase / np.pi, vmin=-1, vmax=1, extent=extent_ft)
        plt.xlim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.ylim([-1.2 * self.fmax, 1.2 * self.fmax])
        plt.title('Pupil, phase/pi')

        plt.subplot(nrows, ncols, 3)
        plt.semilogy(self.mean_err)
        plt.xlabel('iteration')
        plt.ylabel('mean PSF err / max(PSF)')