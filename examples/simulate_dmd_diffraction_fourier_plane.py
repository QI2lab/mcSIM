"""
Simulate the following situation:
The DMD displays a pattern. It is illuminated by a Gaussian beam, and the diffracted light is collected by
a lens. The light intensity profile is recorded in the back focal plane (BFP) of the lens.

Simulate DMD peak broadening including
1. real pattern
2. beam profile
3. PSF from first lens
"""
import numpy as np
from numpy.fft import fftshift, fftfreq
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
import mcsim.analysis.simulate_dmd as sdmd
from localize_psf.fit_psf import blur_img_psf

_cupy_available = True
try:
    import cupy as cp
except ImportError:
    _cupy_available = False

# ############################
# user settings
# ############################
use_gpu = True

if use_gpu and _cupy_available:
    xp = cp
else:
    xp = np

# ############################
# physical parameters
# ############################
wavelength = 0.785
k = 2*np.pi / wavelength

# ############################
# DMD parameters
# ############################
dmd = sdmd.DLP6500()
ny, nx = dmd.size

xx, yy = xp.meshgrid(xp.arange(nx) - nx // 2,
                     xp.arange(ny) - ny // 2)

fxs_dmd = fftshift(fftfreq(nx))
dfx_dmd = fxs_dmd[1] - fxs_dmd[0]
fys_dmd = fftshift(fftfreq(ny))
dfy_dmd = fys_dmd[1] - fys_dmd[0]
extent_fxy_dmd = [fxs_dmd[0] - 0.5 * dfx_dmd, fxs_dmd[-1] + 0.5 * dfx_dmd,
                  fys_dmd[0] - 0.5 * dfy_dmd, fys_dmd[-1] + 0.5 * dfy_dmd]

# ############################
# collection lens properties
# ############################
fl = 200e3  # lens focal length, mm
rl = 0.5 * 25.4e3  # lens aperture radius, mm
na = np.sin(np.arctan(rl/fl))
fmax_efield = na / wavelength

# ############################
# set pattern
# ############################
offsets = np.array([[0, 0], [0, 30]])
phase = 0
rad = 10
ang = -45 * np.pi/180

f1 = np.array([np.sin(-45 * np.pi / 180), np.cos(-45 * np.pi / 180)]) * 1/4 * np.sqrt(2)
frqs = np.expand_dims(f1, axis=0)

frqs, offsets = np.broadcast_arrays(frqs, offsets)

# define pattern
square_spot = False

pattern = xp.ones((ny, nx), dtype=bool)

for offset, frq in zip(offsets, frqs):
    # generate base pattern
    pattern_base = xp.round(np.cos(2 * np.pi * (xx * frq[0] + yy * frq[1]) + phase), 12)
    pattern_base[pattern_base <= 0] = 0
    pattern_base[pattern_base > 0] = 1
    pattern_base = 1 - pattern_base
    pattern_base = pattern_base.astype(bool)

    if square_spot:
        mask = np.logical_and(np.abs(xx - offset[1]) <= rad,
                              np.abs(yy - offset[0]) <= rad)
    else:
        mask = np.sqrt((xx - offset[1])**2 +
                       (yy - offset[0])**2) <= rad

    pattern[mask] = pattern_base[mask]

# phase errors
phase_err_size = 0
phase_errs = xp.random.normal(scale=phase_err_size, size=(ny, nx))

# set beam profile
w = 8e3
beam = xp.exp(-(xx**2 + yy**2) * dmd.dx**2 / w**2)

# ############################
# get output angles assuming perfect alignment for blue
# ############################
_, uvecs_out = sdmd.solve_1color_1d(0.465, dmd.dx, dmd.gamma_on, 4)
optical_axis = uvecs_out[1]
tp_oa, tm_oa = sdmd.uvector2tmtp(*optical_axis)

# find input angle for IR
order_ir = (-3, 3)
# calculate b_out so that the main diffraction order of our pattern lines up with the optical axis
uvec_in_ir = sdmd.solve_diffraction_input(optical_axis,
                                          dmd.dx,
                                          dmd.dy,
                                          wavelength,
                                          order_ir[0],
                                          order_ir[1],
                                          frqs[0][0],
                                          frqs[0][1])
main_ir_order_out = sdmd.solve_diffraction_output(uvec_in_ir,
                                                  dmd.dx,
                                                  dmd.dy,
                                                  wavelength,
                                                  order_ir[0],
                                                  order_ir[1])

tp, tm = sdmd.uvector2tmtp(*uvec_in_ir.ravel())
print("%.0fnm input angles:" % (wavelength * 1e3))
print("theta_p = %0.2fdeg" % (tp * 180/np.pi))
print("theta_m = %0.2fdeg" % (tm * 180/np.pi))
print("unit vector = (%0.3f, %0.3f, %0.3f)" % tuple(uvec_in_ir.ravel()))

tp_out_main, tm_out_main = sdmd.uvector2tmtp(*main_ir_order_out.ravel())
print("main output order:")
print("theta_p = %0.2fdeg" % (tp_out_main * 180/np.pi))
print("theta_m = %0.2fdeg" % (tm_out_main * 180/np.pi))
print("unit vector = (%0.3f, %0.3f, %0.3f)" % tuple(main_ir_order_out.ravel()))

# blaze angle out
uvec_blaze_off = sdmd.solve_blaze_output(uvec_in_ir, dmd.gamma_off, dmd.rot_axis_off)
tp_blaze, tm_blaze = sdmd.uvector2tmtp(*uvec_blaze_off.squeeze())

# sanity check ...
# carrier_uvec_out_check = sdmd.dmd_frq2uvec(main_ir_order_out, frq[0], frq[1], wavelength, dm, dm)
# assert np.linalg.norm(carrier_uvec_out_check - optical_axis) < 1e-12

# ############################
# get positions of carrier frequency spot
# ############################
bf_x, bf_y, bf_z = sdmd.dmd_frq2uvec(main_ir_order_out,
                                     frqs[0][0],
                                     frqs[0][1],
                                     wavelength,
                                     dmd.dx,
                                     dmd.dy)
bf_xp, bf_yp, bf_zp = sdmd.dmd_uvec2opt_axis_uvec(np.stack((bf_x, bf_y, bf_z), axis=-1),
                                                  optical_axis,
                                                  )
uvec_opt_axis_carrier = np.stack((bf_xp, bf_yp, bf_zp)).ravel()

xc_carrier = uvec_opt_axis_carrier[0] * fl
yc_carrier = uvec_opt_axis_carrier[1] * fl

# ############################
# get DFT positions
# ############################
efields_dft, _, _, sinc_on_dft, sinc_off_dft, bvecs_dft = dmd.simulate_pattern_dft(wavelength,
                                                                                   pattern,
                                                                                   beam * np.exp(1j * phase_errs),
                                                                                   uvec_in_ir,
                                                                                   order_ir)
efields_dft_no_err, _, _, _, _, _ = dmd.simulate_pattern_dft(wavelength,
                                                             pattern,
                                                             beam * xp.ones(pattern.shape),
                                                             uvec_in_ir,
                                                             order_ir)

xf_dft, yf_dft, _ = sdmd.dmd_uvec2opt_axis_uvec(bvecs_dft, optical_axis)
xf_dft *= fl
yf_dft *= fl

# sample at uniform grid in Fourier plane
sigma_eff_dmd = rad / np.sqrt(2 * np.log(2)) * dmd.dx
sigma_eff_fourier = fl * wavelength / (2*np.pi * sigma_eff_dmd)

rad_fourier_fov = 2 * np.sqrt(2 * np.log(2)) * sigma_eff_fourier
npts = 51

dxyf = (2 * rad_fourier_fov) / (npts - 1)
xf = (xp.arange(npts) - npts // 2) * dxyf
yf = (xp.arange(npts) - npts // 2) * dxyf
extent_xyf = [float(xf[0]) - 0.5 * dxyf, float(xf[-1]) + 0.5 * dxyf,
              float(yf[0]) - 0.5 * dxyf, float(yf[-1]) + 0.5 * dxyf]

xf, yf = xp.meshgrid(xf, yf)

# convert to unit vectors
bxps = xf / fl
byps = yf / fl
bzps = np.sqrt(1 - bxps**2 - byps**2)
opt_axis_uvecs = np.stack((bxps, byps, bzps), axis=-1)

# convert these back to unit vectors in DMD coordinate system
dmd_uvecs_out = np.stack(sdmd.opt_axis_uvec2dmd_uvec(opt_axis_uvecs, optical_axis), axis=-1)

# ############################
# do simulations
# ############################
efields, _, _ = dmd.simulate_pattern(wavelength,
                                     pattern,
                                     uvec_in_ir,
                                     dmd_uvecs_out,
                                     phase_errs=phase_errs,
                                     efield_profile=beam)

efields_interp = dmd.interpolate_pattern_data(wavelength,
                                              pattern,
                                              beam,
                                              uvec_in_ir,
                                              order_ir,
                                              dmd_uvecs_out)

# ############################
# ideal PSF of lens
# ############################
nyf, nxf = xf.shape
def pupil_fn(r): return r < rl


dxp = fl * wavelength / (nxf * dxyf)
dyp = fl * wavelength / (nyf * dxyf)
x_pupil = (xp.arange(nxf) - nxf // 2) * dxp
y_pupil = (xp.arange(nyf) - nyf // 2) * dyp
xxp, yyp = xp.meshgrid(x_pupil, y_pupil)

pupil = pupil_fn(xp.sqrt(xxp**2 + yyp**2))
psf_amp = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(pupil)))
psf_amp = psf_amp / xp.sqrt(xp.sum(xp.abs(psf_amp)**2))

fx_fourier_plane = xf / fl * wavelength
fy_fourier_plane = yf / fl * wavelength
ff_fourier_plane = xp.sqrt(fx_fourier_plane**2 + fy_fourier_plane**2)
efield_broad = blur_img_psf(efields * xp.exp(-1j * (2 * np.pi) ** 2 * (ff_fourier_plane ** 2) / (2 * k) * fl),
                            psf_amp) * xp.exp(1j * k * (xf ** 2 + yf ** 2) / (2 * fl))

# ############################
# move results off GPU that want to plot
# ############################
if use_gpu and _cupy_available:
    efields_dft = efields_dft.get()
    sinc_on_dft = sinc_on_dft.get()
    sinc_off_dft = sinc_off_dft.get()
    efields = efields.get()
    xf_dft = xf_dft.get()
    yf_dft = yf_dft.get()
    efield_broad = efield_broad.get()
    pattern = pattern.get()
    beam = beam.get()

# ############################
# plot results
# ############################
gamma_norm = 1

figh = plt.figure(figsize=(18, 8))
grid = figh.add_gridspec(nrows=2, hspace=0.5,
                         ncols=10, wspace=0.5, width_ratios=[1, 1, 0.1, 0.1, 1, 1, 0.1, 0.1, 1, 0.1])


figh.suptitle(f"Simulation of DMD pattern imaged by lens of f={fl * 1e-3:.1f}mm\n"
              f" input dir = ({uvec_in_ir[0, 0]:.3f}, {uvec_in_ir[0, 1]:.3f}, {uvec_in_ir[0, 2]:.3f});"
              f" $(\\theta_-, \\theta_+)$ = ({tm * 180 / np.pi:.2f}, {tp * 180 / np.pi:.2f})deg\n"
              f"optical axis = ({optical_axis[0]:.3f}, {optical_axis[1]:.3f}, {optical_axis[2]:.3f});"
              f" $(\\theta_-, \\theta_+)$ = ({tm_oa * 180 / np.pi:.2f}, {tp_oa * 180 / np.pi:.2f})deg\n"
              f" blaze direction = ({uvec_blaze_off[0, 0]:.3f}, {uvec_blaze_off[0, 1]:.3f}, {uvec_blaze_off[0, 2]:.3f});"
              f" $(\\theta_-, \\theta_+)$ = ({tm_blaze * 180 / np.pi:.2f}, {tp_blaze * 180 / np.pi:.2f})deg\n"
              f"phase error = {phase_err_size / np.pi:.3f}*pi")

# #####################
# electric field versus output angle
# #####################
ax = figh.add_subplot(grid[0, 0])
ax.imshow(np.abs(efields_dft)**2,
          origin="lower",
          extent=extent_fxy_dmd,
          norm=PowerNorm(gamma=0.2),
          cmap="bone")
ax.add_artist(Circle(frqs[0], radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frqs[0], radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("Electric field versus output angle\n"
             "$|E(b(f) - a)|^2$ from DFT")

ax = figh.add_subplot(grid[1, 0])
ax.imshow(np.angle(efields_dft), origin="lower", extent=extent_fxy_dmd,
           vmin=-np.pi, vmax=np.pi, cmap="RdBu")
ax.add_artist(Circle(frqs[0], radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frqs[0], radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("$ang[E(b(f) - a)]$ from DFT")

# #####################
# blaze envelopes
# #####################
ax = figh.add_subplot(grid[0, 1])
ax.imshow(np.abs(sinc_on_dft)**2 / dmd.dx**2 / dmd.dy**2,
          origin="lower",
          extent=extent_fxy_dmd,
          norm=PowerNorm(gamma=1, vmin=0, vmax=1),
          cmap="bone")
ax.add_artist(Circle(frqs[0], radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frqs[0], radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("Blaze envelope, on mirrors, $|E|^2$")

ax = figh.add_subplot(grid[1, 1])
im = ax.imshow(np.abs(sinc_off_dft)**2 / dmd.dx**2 / dmd.dy**2,
               origin="lower",
               extent=extent_fxy_dmd,
               norm=PowerNorm(gamma=1, vmin=0, vmax=1),
               cmap="bone")
ax.add_artist(Circle(frqs[0], radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frqs[0], radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("Blaze envelope, off mirrors, $|E|^2$")

ax = figh.add_subplot(grid[:, 2])
plt.colorbar(im, cax=ax)

# #####################
# electric fields in lens back focal plane
# #####################
vmax = 1.2 * np.max(np.abs(efields)**2)

ax = figh.add_subplot(grid[0, 4])
ax.imshow(np.abs(efields) ** 2,
          origin="lower",
          extent=extent_xyf,
          norm=PowerNorm(gamma=gamma_norm, vmin=0, vmax=vmax),
          cmap="bone")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
#ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("|E(r)|^2, no broadening")

ax = figh.add_subplot(grid[1, 4])
ax.imshow(np.angle(efields), origin="lower", extent=extent_xyf, vmin=-np.pi, vmax=np.pi, cmap="RdBu")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
#ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("ang(E(r))")

ax = figh.add_subplot(grid[0, 5])
im = ax.imshow(np.abs(efield_broad) ** 2,
               origin="lower",
               extent=extent_xyf,
               norm=PowerNorm(gamma=gamma_norm, vmin=0, vmax=vmax),
               cmap="bone")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
#ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("|E(r)|^2")

ax = figh.add_subplot(grid[1, 5])
im_ph = ax.imshow(np.angle(efield_broad), origin="lower", extent=extent_xyf, vmin=-np.pi, vmax=np.pi, cmap="RdBu")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
#ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("ang(E(r))")

ax = figh.add_subplot(grid[0, 6])
cb = plt.colorbar(im, cax=ax)
cb.set_label(f"gamma={gamma_norm:0.1f}")

ax = figh.add_subplot(grid[1, 6])
plt.colorbar(im_ph, cax=ax)

# #####################
# pattern
# #####################
ax = figh.add_subplot(grid[0, 8])
ax.set_title(f"DMD pattern, frq = ({frqs[0][0]:.3f}, {frqs[0][1]:.3f}) 1/mirrors\n"
             f"rad={rad:.1f}, angle={ang*180/np.pi:.1f}deg")
ax.imshow(pattern, origin="lower", cmap="bone")
ax.set_xlabel("x-position (mirrors)")
ax.set_ylabel("y-position (mirrors)")
ax.add_artist(Circle((nx // 2, ny // 2), radius=2, fill=False, color="r"))

xsize = np.min([3 * rad, nx // 2])
ysize = np.min([3 * rad, ny // 2])
ax.set_xlim([nx // 2 - xsize - 0.5, nx // 2 + xsize + 0.5])
ax.set_ylim([ny // 2 - ysize - 0.5, ny // 2 + ysize + 0.5])

# #####################
# pattern
# #####################
ax = figh.add_subplot(grid[1, 8])
ax.set_title(f"beam profile on DMD (|E|), waist={w*1e-3:.1f}mm")
im = ax.imshow(beam, origin="lower", cmap="bone")
ax.set_xlabel("x-position (mirrors)")
ax.set_ylabel("y-position (mirrors)")

ax = figh.add_subplot(grid[1, 9])
plt.colorbar(im, cax=ax)

plt.show()
