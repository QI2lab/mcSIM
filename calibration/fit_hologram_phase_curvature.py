"""
Fit frequency and phase curvature of hologram

Phase curvature = k * r^2 / (2*R)
Typically want R > 0 to indicate beam is diverging and R < 0 converging
BUT, since this is holography that only works if we choose the "correct" reference beam, instead of the negative one
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from skimage.restoration import unwrap_phase
import numpy as np
from numpy import fft
import tifffile
import zarr
from scipy.ndimage import maximum_filter, minimum_filter
from pathlib import Path
import mcsim.analysis.analysis_tools as tools
import mcsim.analysis.sim_reconstruction as sim
import localize_psf.fit as fit

# fname = r"F:\2021_11_23\23_odt_align\23_odt_align_MMStack_Pos0.ome.tif"
# fname = r"F:\2021_11_23\24_odt_align\24_odt_align_MMStack_Pos0.ome.tif"
# fname = r"F:\2021_12_07\13_odt_focus_test\13_odt_focus_test_MMStack_Pos0.ome.tif"
# fname = Path(r"E:\2022_02_25\05_hologram.tif")
# fname = Path(r"G:\2022_05_30\01_r=5_hologram.tif")
# fname = Path(r"I:\2022_08_21\01_odt_beam0_test\test_00.tif")
# fname = Path(r"I:\2022_08_24\01_odt_test\test_00.tif")
# fname = Path(r"I:\2022_08_24\16_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_24\17_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_24\18_ref_beam_test\test_1.tif")
# fname = Path(r"I:\2022_08_24\19_ref_beam_test\test_00.tif")
# fname = Path(r"I:\2022_08_24\20_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_24\21_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_24\22_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_24\23_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_24\24_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_24\25_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\01_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\02_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\03_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\04_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\05_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\06_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\07_ref_beam_test\test_000.tif")
# fname = Path(r"I:\2022_08_25\08_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\09_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\10_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\11_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\12_ref_beam_test\test_0.tif")
# fname = Path(r"I:\2022_08_25\13_ref_beam_test\test_0.tif")
fname = Path(r"I:\2022_08_25\14_ref_beam_test\test_0.tif")
img = tifffile.imread(fname)
if img.ndim == 3:
    img = img[0]

# fname = Path(r"G:\2022_06_26\26_phase_curvature_test\sim_odt.zarr")
# fname = Path(r"G:\2022_07_07\08_odt_phase_curvature\sim_odt.zarr")
# dat = zarr.open(fname, "r")
# img = np.array(dat.cam2.odt[0, 0, 0, 0, 0, 0])

# fname = Path(r"I:\2022_07_21\02_1um_beads\sim_odt.zarr")
# dat = zarr.open(fname, "r")
# img = np.array(dat.cam2.odt[-1, 0, 0, 0, 0, 0])


img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img)))

ny, nx = img.shape

threshold = 60
# ################################
# physical parameters
# ################################
wavelength = 0.785 # um
k = 2*np.pi / wavelength
# dp = 6.5
dp = 20
# mag = 60
mag = 60 * 3
dxy = dp / mag
na = 1
fmax_int = 1 / (0.5 * wavelength / na)

x = dxy * (np.arange(nx) - nx//2)
y = dxy * (np.arange(ny) - ny//2)
xx, yy = np.meshgrid(x, y)

fxs = fft.fftshift(fft.fftfreq(nx, dxy))
dfx = fxs[1] - fxs[0]
fys = fft.fftshift(fft.fftfreq(ny, dxy))
dfy = fys[1] - fys[0]

fxfx, fyfy = np.meshgrid(fxs, fys)
ff_perp = np.sqrt(fxfx**2 + fyfy**2)

guess_mask = np.logical_and(np.abs(fxfx) > dfx, np.abs(fyfy) > dfy)

guess_ind_1d = np.argmax(np.abs(img_ft) * (fyfy <= 0) * (ff_perp > fmax_int) * guess_mask)
guess_ind = np.unravel_index(guess_ind_1d, img_ft.shape)

frq_guess = np.array([fxfx[guess_ind], fyfy[guess_ind]])
frq_fit, mask, _ = sim.fit_modulation_frq(img_ft, img_ft, dxy, frq_guess=frq_guess, max_frq_shift=50 * dfx)
# sim.plot_correlation_fit(img_ft, img_ft, frq_fit, dxy, frqs_guess=frq_guess)

# plot hologram fit
sim.plot_correlation_fit(img_ft, img_ft, frq_fit, dxy, frqs_guess=frq_guess)

# shifted field
efield_ft_shift = tools.translate_ft(img_ft, frq_fit[0], frq_fit[1], drs=(dxy, dxy))
efield_ft_shift[ff_perp > fmax_int / 2] = 0

efield_shift = fft.fftshift(fft.ifft2(fft.ifftshift(efield_ft_shift)))

# unwrapped phase
amp = np.abs(efield_shift)
phase_unwrapped = unwrap_phase(np.angle(efield_shift))

# fit defocus
to_fit_pix = np.abs(efield_shift) > threshold
# dilate/erode to close holes
footprint = np.ones((10, 10), dtype=bool)
to_fit_pix = maximum_filter(to_fit_pix, footprint=footprint)
to_fit_pix = minimum_filter(to_fit_pix, footprint=footprint)

# exclude region around edges
edge_exclude_size = 20
to_fit_pix[:edge_exclude_size] = False
to_fit_pix[-edge_exclude_size:] = False
to_fit_pix[:, :edge_exclude_size] = False
to_fit_pix[:, -edge_exclude_size:] = False

# fit defocus phase
def defocus_phase_fn(p, x, y):
    """
    @param p: [k/Rx, k/Ry, cx, cy, offset, theta]
    @param x:
    @param y:
    @return phase:
    """
    xrot = (x - p[2]) * np.cos(p[5]) + (y - p[3]) * np.sin(p[5])
    yrot = -(x - p[2]) * np.sin(p[5]) + (y - p[3]) * np.cos(p[5])
    phase = 0.5 * p[0] * xrot**2 + 0.5 * p[1] * yrot**2 + p[4]
    return phase

def fit_fn(p): return defocus_phase_fn(p, xx[to_fit_pix], yy[to_fit_pix])


# tried fitting wais
# def waist_fn(p, x, y):
#     """
#
#     @param p: [., ., cx, cy, ., theta, amp, wx, wy, amp offset]
#     @param x:
#     @param y:
#     @return:
#     """
#     xrot = (x - p[2]) * np.cos(p[5]) + (y - p[3]) * np.sin(p[5])
#     yrot = -(x - p[2]) * np.sin(p[5]) + (y - p[3]) * np.cos(p[5])
#     amp = p[6] * np.exp(-xrot**2 / p[7]**2 - yrot**2 / p[8]**2) + p[9]
#     return amp

# def fit_fn(p):
#     vals = np.concatenate((defocus_phase_fn(p, xx[to_fit_pix], yy[to_fit_pix]),
#                            waist_fn(p, xx[to_fit_pix], yy[to_fit_pix])))
#     return vals

init_params = [0, 0,
               np.sum(xx * to_fit_pix) / np.sum(to_fit_pix),
               np.sum(yy * to_fit_pix) / np.sum(to_fit_pix),
               np.mean(phase_unwrapped[to_fit_pix]),
               0]
# init_params = [0,
#                0,
#                np.sum(xx * to_fit_pix) / np.sum(to_fit_pix),
#                np.sum(yy * to_fit_pix) / np.sum(to_fit_pix),
#                np.mean(phase_unwrapped[to_fit_pix]),
#                0,
#                np.mean(amp),
#                np.max(xx) / 2,
#                np.max(yy) / 2,
#                0]

# fixed_params = [False, False, False, False, False, False]
lbs = [-np.inf, -np.inf, np.min(xx[to_fit_pix]), np.min(yy[to_fit_pix]), -np.inf, -np.inf]
ubs = [np.inf, np.inf, np.max(xx[to_fit_pix]), np.max(yy[to_fit_pix]), np.inf, np.inf]
# lbs = [-np.inf, -np.inf, np.min(xx[to_fit_pix]), np.min(yy[to_fit_pix]), -np.inf, -np.inf, 0, 0, 0, 0]
# ubs = [ np.inf,  np.inf, np.max(xx[to_fit_pix]), np.max(yy[to_fit_pix]),  np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf]

fit_data = phase_unwrapped[to_fit_pix]
# fit_data = np.concatenate((phase_unwrapped[to_fit_pix], amp[to_fit_pix]))

results = fit.fit_model(fit_data,
                        fit_fn,
                        init_params=init_params,
                        bounds=(lbs, ubs))
fp = results["fit_params"]
# radius of curvature in um
rx = k / fp[0]
ry = k / fp[1]

# plot
extent_xy = [x[0] - 0.5 * dxy, x[-1] + 0.5 * dxy,
             y[-1] + 0.5 * dxy, y[0] - 0.5 * dxy]

figh = plt.figure(figsize=(16, 8))
figh.suptitle(f"{str(fname):s}\n"
              f"object plane $R_x$ = {rx / 1e3:.2f}mm, $R_y$ = {ry / 1e3:.2f}mm\n"
              f"image plane  $R_x$ = {rx * dp**2 / dxy**2 / 1e6:.3f}m,"
                          f" $R_y$ = {ry * dp**2 / dxy**2 / 1e6:.3f}m\n"
              f"angle = {fp[-1] * 180/np.pi:.2f}deg")
grid = figh.add_gridspec(nrows=2, ncols=9, width_ratios=[8, 1, 1]*3)

ax = figh.add_subplot(grid[0, 0])
im = ax.imshow(np.abs(efield_shift), vmin=0, cmap="bone", extent=extent_xy)
ax.set_title("efield")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[0, 1])
figh.colorbar(im, cax=ax)

ax = figh.add_subplot(grid[0, 3])
im = ax.imshow(np.angle(efield_shift), vmin=-np.pi, vmax=np.pi, cmap="RdBu", extent=extent_xy)
ax.plot(fp[2], fp[3], 'kx')
ax.set_title("wrapped phase")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[0, 4])
figh.colorbar(im, cax=ax)


# vmin = np.percentile(phase_unwrapped, 0.05)
# vmax = np.percentile(phase_unwrapped, 99.95)
vmin = -2*np.pi
vmax = 2*np.pi
ax = figh.add_subplot(grid[0, 6])
im = ax.imshow(phase_unwrapped - fp[4], cmap="RdBu", vmin=vmin, vmax=vmax, extent=extent_xy)
ax.plot(fp[2], fp[3], 'kx')
ax.set_title("unwrapped phase")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[0, 7])
figh.colorbar(im, cax=ax)


ax = figh.add_subplot(grid[1, 0])
ax.imshow(to_fit_pix, cmap="bone", extent=extent_xy)
ax.set_title("fit mask")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[1, 3])

pnow = phase_unwrapped - fp[4]
pnow[np.logical_not(to_fit_pix)] = np.nan

im = ax.imshow(pnow, vmin=vmin, vmax=vmax, cmap="RdBu", extent=extent_xy)
ax.plot(fp[2], fp[3], 'kx')
ax.set_title("masked unwrapped phase")
ax.set_xticks([])
ax.set_yticks([])

ax = figh.add_subplot(grid[1, 6])
phase_fit_plot = defocus_phase_fn(results["fit_params"], xx, yy)
ax.plot(fp[2], fp[3], 'kx')
phase_fit_plot[np.logical_not(to_fit_pix)] = np.nan

ax.imshow(phase_fit_plot - fp[4], cmap="RdBu", vmin=vmin, vmax=vmax, extent=extent_xy)
ax.plot(fp[2], fp[3], 'kx')

axis_len = 10
ax.plot([fp[2], fp[2] + axis_len * np.cos(fp[5])], [fp[3], fp[3] + axis_len * np.sin(fp[5])], 'k', label="x")
ax.plot([fp[2], fp[2] - axis_len * np.sin(fp[5])], [fp[3], fp[3] + axis_len * np.cos(fp[5])], 'm', label="y")

ax.set_title("fit phase")
ax.set_xticks([])
ax.set_yticks([])
ax.legend()