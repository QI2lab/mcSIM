"""
Simulate DMD peak broadening including
1. real pattern
2. beam profile
3. PSF from first lens
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import simulate_dmd as sdmd
import analysis_tools as tools
from numpy import fft
import fit_psf
from matplotlib.patches import Circle

wavelength = 0.785
k = 2*np.pi / wavelength

# DMD
dm = 7.56 # DMD pitch
gamma = 12 * np.pi/180
nx = 1920
ny = 1080 # 1080
xx, yy = np.meshgrid(range(nx), range(ny))

fxs_dmd = fft.fftshift(fft.fftfreq(nx))
dfx_dmd = fxs_dmd[1] - fxs_dmd[0]
fys_dmd = fft.fftshift(fft.fftfreq(ny))
dfy_dmd = fys_dmd[1] - fys_dmd[0]
extent_fxy_dmd = [fxs_dmd[0] - 0.5 * dfx_dmd, fxs_dmd[-1] + 0.5 * dfx_dmd,
                  fys_dmd[0] - 0.5 * dfy_dmd, fys_dmd[-1] + 0.5 * dfy_dmd]

# lens properties
fl = 200e3 # mm, lens focal length
rl = 0.5 * 25.4e3
na = np.sin(np.arctan(rl/fl))
fmax_efield = na / wavelength

# set pattern
cref = np.array([ny // 2 , nx // 2])
offset = np.array([0, 0])
phase = 0
# rad = np.inf
# rad = 5
rad = 300
# ang = 0 * np.pi/180
# frq = np.array([np.sin(ang), np.cos(ang)]) * 1/3
# frq = np.array([-1/4, 1/4])
# frq = np.array([0, 0])
# frq = -np.array([-1/4, 1/4])
frq = -np.array([-1/3, 1/3])

pattern_base = np.round(np.cos(2*np.pi * (xx * frq[0] + yy * frq[1]) + phase), 12)
pattern_base[pattern_base <= 0] = 0
pattern_base[pattern_base > 0] = 1
pattern_base = 1 - pattern_base
# pattern = np.ones((ny, nx), dtype=bool)

pattern = np.array(pattern_base, copy=True).astype(bool)
square_spot = False
if square_spot:
    pattern[np.abs(xx - cref[1] - offset[1]) > rad] = 1
    pattern[np.abs(yy - cref[0] - offset[0]) > rad] = 1
else:
    pattern[np.sqrt((xx - cref[1] - offset[1])**2 + (yy - cref[0] - offset[0])**2) > rad] = 1

# phase errors
phase_err_size = 0
phase_errs = np.random.normal(scale=phase_err_size, size=(ny, nx))
# phase_errs = np.abs(xx - nx//2) * np.pi / nx

# set beam profile
w = 8e3
beam = np.exp(-((xx - nx//2)**2 + (yy - ny//2)**2) * dm**2 / w**2)

# get output angles assuming perfect alignment for blue
_, uvecs_out = sdmd.solve_1color_1d(0.465, dm, gamma, 4)
optical_axis = uvecs_out[1]
tp_oa, tm_oa = sdmd.uvector2tmtp(*optical_axis)

# find input angle for IR
order_ir = (-3, 3)
# calculate b_out so that the main diffraction order of our pattern lines up with the optical axis
bx_out = optical_axis[0] - wavelength / dm * frq[0]
by_out = optical_axis[1] - wavelength / dm * frq[1]
bz_out = np.sqrt(1 - bx_out**2 - by_out**2)
main_ir_order_out = np.stack((bx_out, by_out, bz_out))

# uvec_in = sdmd.solve_diffraction_input(uvec_out, dm, dm, wavelength, order_ir)
uvec_in_ir = sdmd.solve_diffraction_input(main_ir_order_out, dm, dm, wavelength, order_ir)
tp, tm = sdmd.uvector2tmtp(*uvec_in_ir.ravel())
print("input angles:")
print("theta_p = %0.2fdeg" % (tp * 180/np.pi))
print("theta_m = %0.2fdeg" % (tm * 180/np.pi))

# blaze angle out
uvec_blaze_off = sdmd.solve_blaze_output(uvec_in_ir, -gamma)
tp_blaze, tm_blaze = sdmd.uvector2tmtp(*uvec_blaze_off.squeeze())

# sanity check ...
carrier_uvec_out_check = sdmd.dmd_frq2uvec(main_ir_order_out, frq[0], frq[1], wavelength, dm, dm)
assert np.linalg.norm(carrier_uvec_out_check - optical_axis) < 1e-12

# get positions of carrier frequency spot
uvec_opt_axis_carrier = np.array(sdmd.dmd_frq2opt_axis_uvec(frq[0], frq[1], main_ir_order_out, optical_axis, dm, dm, wavelength)).ravel()
xc_carrier = uvec_opt_axis_carrier[0] * fl
yc_carrier = uvec_opt_axis_carrier[1] * fl


# get DFT positions
efields_dft, _, _, sinc_on_dft, sinc_off_dft, bvecs_dft = sdmd.simulate_dmd_dft(pattern, beam * np.exp(1j * phase_errs), wavelength, gamma, -gamma, dm, dm, dm, dm, uvec_in_ir, order_ir)
efields_dft_no_err, _, _, _, _, _ = sdmd.simulate_dmd_dft(pattern, beam * np.ones(pattern.shape), wavelength, gamma, -gamma, dm, dm, dm, dm, uvec_in_ir, order_ir)
xf_dft, yf_dft, _ = sdmd.dmd_uvec2opt_axis_uvec(bvecs_dft, optical_axis)
xf_dft *= fl
yf_dft *= fl

# sample at uniform grid in Fourier plane
sigma_eff_dmd = rad / np.sqrt(2 * np.log(2)) * dm
sigma_eff_fourier = fl * wavelength / (2*np.pi * sigma_eff_dmd)

# rad_fourier_fov = 40
rad_fourier_fov = 2 * np.sqrt(2 * np.log(2)) * sigma_eff_fourier
npts = 51

dxyf = (2 * rad_fourier_fov) / (npts - 1)
xf = tools.get_fft_pos(npts, dxyf)
yf = tools.get_fft_pos(npts, dxyf)
extent_xyf = [xf[0] - 0.5 * dxyf, xf[-1] + 0.5 * dxyf, yf[0] - 0.5 * dxyf, yf[-1] + 0.5 * dxyf]

xf, yf = np.meshgrid(xf, yf)

# convert to unit vectors
bxps = xf / fl
byps = yf / fl
bzps = np.sqrt(1 - bxps**2 - byps**2)
opt_axis_uvecs = np.stack((bxps, byps, bzps), axis=-1)

# convert these back to unit vectors in DMD coordinate system
dmd_uvecs_out = np.stack(sdmd.opt_axis_uvec2dmd_uvec(opt_axis_uvecs, optical_axis), axis=-1)

# do simulations
efields, _, _, _ = sdmd.simulate_dmd(pattern, wavelength, gamma, -gamma, dm, dm, dm, dm, uvec_in_ir, dmd_uvecs_out,
                                     phase_errs=phase_errs, efield_profile=beam)

efields_interp = sdmd.interpolate_dmd_data(pattern, beam, wavelength, gamma, -gamma,
                                           dm, dm, dm, dm, uvec_in_ir, order_ir, dmd_uvecs_out)

# ideal PSF of lens
nyf, nxf = xf.shape
def pupil_fn(r): return r < rl
dxp = fl * wavelength / (nxf * dxyf)
dyp = fl * wavelength / (nyf * dxyf)
xp = tools.get_fft_pos(nxf, dxp)
yp = tools.get_fft_pos(nyf, dyp)
xxp, yyp = np.meshgrid(xp, yp)

pupil = pupil_fn(np.sqrt(xxp**2 + yyp**2))
psf_amp = fft.fftshift(fft.fft2(fft.ifftshift(pupil)))
psf_amp = psf_amp / np.sqrt(np.sum(np.abs(psf_amp)**2))

fx_fourier_plane = xf / fl * wavelength
fy_fourier_plane = yf / fl * wavelength
ff_fourier_plane = np.sqrt(fx_fourier_plane**2 + fy_fourier_plane**2)
efield_broad = fit_psf.blur_img_psf(efields * np.exp(-1j * (2*np.pi)**2 * (ff_fourier_plane**2) / (2*k) * fl),
                                    psf_amp) * np.exp(1j * k * (xf**2 + yf**2) / (2 * fl))

# plot results
gamma_norm = 1

figh = plt.figure(figsize=(18, 8))
grid = plt.GridSpec(2, 5, hspace=0.5, wspace=0.5)
plt.suptitle(("carrier freq = (%0.2f, %0.2f) 1/mirrors; input dir = (%0.3f, %0.3f, %0.3f);" +
             r" $\theta_-$, $\theta_+$" + " = (%0.2f, %0.2f)deg\n"
             "optical axis = (%0.3f, %0.3f, %0.3f);" + r" $\theta_-$, $\theta_+$" + " = (%0.2f, %0.2f);"
             " blaze direction = (%0.3f, %0.3f, %0.3f);" + r" $\theta_-$, $\theta_+$" + " = (%0.2f, %0.2f)\n"
             "radius=%0.1f, phase error = %0.3f*pi") %
             (frq[0], frq[1],
              uvec_in_ir[0, 0], uvec_in_ir[0, 1], uvec_in_ir[0, 2],
              tm * 180 / np.pi, tp * 180 / np.pi,
              optical_axis[0], optical_axis[1], optical_axis[2],
              tm_oa * 180 / np.pi, tp_oa * 180 / np.pi,
              uvec_blaze_off[0, 0], uvec_blaze_off[0, 1], uvec_blaze_off[0, 2],
              tm_blaze * 180 / np.pi, tp_blaze * 180 / np.pi,
              rad, phase_err_size / np.pi))
# norm = np.max(np.abs(efields_dft_no_err))**2

ax = plt.subplot(grid[0, 0])
plt.imshow(np.abs(efields_dft)**2, origin="lower", extent=extent_fxy_dmd,
           norm=PowerNorm(gamma=0.1), cmap="bone")
ax.add_artist(Circle(frq, radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frq, radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("$|E(b(f) - a)|^2$ from DFT")

ax = plt.subplot(grid[1, 0])
plt.imshow(np.angle(efields_dft), origin="lower", extent=extent_fxy_dmd,
           vmin=-np.pi, vmax=np.pi, cmap="RdBu")
ax.add_artist(Circle(frq, radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frq, radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("$ang[E(b(f) - a)]$ from DFT")

ax = plt.subplot(grid[0, 1])
plt.imshow(np.abs(sinc_on_dft)**2 / dm**4, origin="lower", extent=extent_fxy_dmd,
           norm=PowerNorm(gamma=0.1, vmin=0, vmax=1), cmap="bone")
ax.add_artist(Circle(frq, radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frq, radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("on envelope, $|E|^2$")

ax = plt.subplot(grid[1, 1])
plt.imshow(np.abs(sinc_off_dft)**2 / dm**4, origin="lower", extent=extent_fxy_dmd,
           norm=PowerNorm(gamma=1, vmin=0, vmax=1), cmap="bone")
ax.add_artist(Circle(frq, radius=0.01, fill=False, color="r"))
ax.add_artist(Circle(-frq, radius=0.01, fill=False, color="m"))
ax.set_xlabel("$f_x$ (1/mirrors)")
ax.set_ylabel("$f_y$ (1/mirrors)")
ax.set_title("off envelope, $|E|^2$")

ax = plt.subplot(grid[0, 2])
ax.imshow(np.abs(efields) ** 2, origin="lower", extent=extent_xyf, norm=PowerNorm(gamma=gamma_norm), cmap="bone")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("|E(r)|^2, no broadening (gamma=%0.1f)" % gamma_norm)

ax = plt.subplot(grid[1, 2])
ax.imshow(np.angle(efields), origin="lower", extent=extent_xyf, vmin=-np.pi, vmax=np.pi, cmap="RdBu")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("ang(E(r))")

ax = plt.subplot(grid[0, 3])
ax.imshow(np.abs(efield_broad) ** 2, origin="lower", extent=extent_xyf, norm=PowerNorm(gamma=gamma_norm), cmap="bone")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("|E(r)|^2")

ax = plt.subplot(grid[1, 3])
ax.imshow(np.angle(efield_broad), origin="lower", extent=extent_xyf, vmin=-np.pi, vmax=np.pi, cmap="RdBu")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.add_artist(Circle((xc_carrier, yc_carrier), radius=3*dxyf, fill=False, color="r"))
ax.plot(xf_dft.ravel(), yf_dft.ravel(), 'gx')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x-position, BFP (um)")
ax.set_ylabel("y-position, BFP (um)")
ax.set_title("ang(E(r))")


ax = plt.subplot(grid[0, 4])
ax.set_title("pattern")
ax.imshow(pattern, origin="lower", cmap="bone")
ax.set_xlabel("x-position (mirrors)")
ax.set_ylabel("y-position (mirrors)")

xsize = np.min([3 * rad, nx // 2])
ysize = np.min([3 * rad, ny // 2])
ax.set_xlim([nx // 2 - xsize - 0.5, nx // 2 + xsize + 0.5])
ax.set_ylim([ny // 2 - ysize - 0.5, ny // 2 + ysize + 0.5])


ax = plt.subplot(grid[1, 4])
ax.set_title("beam profile, |E|, w=%0.1fmm" % (w / 1e3))
ax.imshow(beam, origin="lower", cmap="bone")
ax.set_xlabel("x-position (mirrors)")
ax.set_ylabel("y-position (mirrors)")

