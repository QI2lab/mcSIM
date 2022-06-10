"""
Fitt phase curvature of hologram
"""
import numpy as np
from numpy import fft
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import tifffile
from pathlib import Path
import mcsim.analysis.analysis_tools as tools
import mcsim.analysis.sim_reconstruction as sim
import localize_psf.fit as fit

# fname = r"F:\2021_11_23\23_odt_align\23_odt_align_MMStack_Pos0.ome.tif"
# fname = r"F:\2021_11_23\24_odt_align\24_odt_align_MMStack_Pos0.ome.tif"
# fname = r"F:\2021_12_07\13_odt_focus_test\13_odt_focus_test_MMStack_Pos0.ome.tif"
# fname = Path(r"E:\2022_02_25\05_hologram.tif")
fname = Path(r"G:\2022_05_30\01_r=5_hologram.tif")

img = tifffile.imread(fname)
if img.ndim == 3:
    img = img[0]

img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img)))

ny, nx = img.shape

threshold = 300
# ################################
# physical parameters
# ################################
wavelength = 0.785 # um
k = 2*np.pi / wavelength
dp = 6.5
# mag = 50
mag = 60
dxy = dp / mag
# na = 0.55
na = 1
fmax_int = 1 / (0.5 * wavelength / na)

x = dxy * (np.arange(nx) - nx//2)
y = dxy * (np.arange(ny) - ny//2)
xx, yy = np.meshgrid(x, y)

fxs = tools.get_fft_frqs(nx, dxy)
dfx = fxs[1] - fxs[0]
fys = tools.get_fft_frqs(ny, dxy)
dfy = fys[1] - fys[0]

fxfx, fyfy = np.meshgrid(fxs, fys)
ff_perp = np.sqrt(fxfx**2 + fyfy**2)

guess_ind_1d = np.argmax(np.abs(img_ft) * (fyfy <= 0) * (ff_perp > fmax_int))
guess_ind = np.unravel_index(guess_ind_1d, img_ft.shape)

frq_guess = np.array([fxfx[guess_ind], fyfy[guess_ind]])
frq_fit, mask, _ = sim.fit_modulation_frq(img_ft, img_ft, dxy, frq_guess=frq_guess, roi_pix_size=50)
# sim.plot_correlation_fit(img_ft, img_ft, frq_fit, dxy, frqs_guess=frq_guess)

# shifted field
efield_ft_shift = tools.translate_ft(img_ft, frq_fit[0], frq_fit[1], drs=(dxy, dxy))
efield_ft_shift[ff_perp > fmax_int / 2] = 0

efield_shift = fft.fftshift(fft.ifft2(fft.ifftshift(efield_ft_shift)))

# unwrapped phase
phase_unwrapped = unwrap_phase(np.angle(efield_shift))

# fit defocus
to_fit_pix = np.abs(efield_shift) > threshold
def fn(p, x, y):
    xrot = (x - p[2]) * np.cos(p[5]) + (y - p[3]) * np.sin(p[5])
    yrot = -(x - p[2]) * np.sin(p[5]) + (y - p[3]) * np.cos(p[5])
    return -p[0] * xrot**2 - p[1] * yrot**2 + p[4]
# p[0] = k/(2*R)

results = fit.fit_model(phase_unwrapped[to_fit_pix], lambda p: fn(p, xx[to_fit_pix], yy[to_fit_pix]),
                        init_params=[0, 0, np.sum(xx * to_fit_pix) / np.sum(to_fit_pix),
                                     np.sum(yy * to_fit_pix) / np.sum(to_fit_pix),
                                     np.mean(phase_unwrapped[to_fit_pix]),
                                     0],
                        bounds=([-np.inf, -np.inf, np.min(x), np.min(y), -np.inf, -np.inf],
                                [np.inf, np.inf, np.max(x), np.max(y), np.inf, np.inf]))
fp = results["fit_params"]
# radius of curvature in um
rx = -0.5 * k / fp[0]
ry = -0.5 * k / fp[1]

# plot
extent_xy = [x[0] - 0.5 * dxy, x[-1] + 0.5 * dxy, y[-1] + 0.5 * dxy, y[0] - 0.5 * dxy]

figh = plt.figure(figsize=(16, 8))
figh.suptitle("object plane $R_x$ = %0.2fmm, $R_y$ = %0.2fmm\n"
             "image plane  $R_x$ = %0.3fm, $R_y$ = %0.3fm\n"
             "angle = %0.2fdeg" %
             (rx / 1e3, ry / 1e3, rx * dp**2 / dxy**2 / 1e6, ry * dp**2 / dxy**2 / 1e6, fp[-1] * 180/np.pi))
grid = figh.add_gridspec(nrows=2, ncols=15 + 3)

ax = figh.add_subplot(grid[0, :5])
ax.imshow(np.abs(efield_shift), vmin=0, cmap="bone", extent=extent_xy)
ax.set_title("efield")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[0, 6:11])
im = ax.imshow(np.angle(efield_shift), vmin=-np.pi, vmax=np.pi, cmap="RdBu", extent=extent_xy)
ax.plot(fp[2], fp[3], 'kx')
ax.set_title("wrapped phase")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[0, 11])
figh.colorbar(im, cax=ax)


# vmin = np.percentile(phase_unwrapped, 0.05)
# vmax = np.percentile(phase_unwrapped, 99.95)
vmin = -2*np.pi
vmax = 2*np.pi
ax = figh.add_subplot(grid[0, 12:17])
im = ax.imshow(phase_unwrapped - fp[-2], cmap="RdBu", vmin=vmin, vmax=vmax, extent=extent_xy)
ax.plot(fp[2], fp[3], 'kx')
ax.set_title("unwrapped phase")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[0, 17])
figh.colorbar(im, cax=ax)


ax = figh.add_subplot(grid[1, 6:11])
ax.imshow(to_fit_pix, cmap="bone", extent=extent_xy)
ax.set_title("fit mask")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[1, 12:17])
phase_fit_plot = fn(results["fit_params"], xx, yy)
ax.plot(fp[2], fp[3], 'kx')
phase_fit_plot[np.logical_not(to_fit_pix)] = np.nan

ax.imshow(phase_fit_plot - fp[-2], cmap="RdBu", vmin=vmin, vmax=vmax, extent=extent_xy)
ax.plot(fp[2], fp[3], 'kx')
ax.set_title("fit phase")
ax.set_xticks([])
ax.set_yticks([])