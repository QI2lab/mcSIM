"""
For a variety of wavelength, determine allowed input/output angles where the blaze condition and diffraction condition
can be satisfied simultaneously. These produce curves in 2D output angle space. Intersection of curves from different
wavelengths are locations where multicolor operation is possible.
"""
import numpy as np
import matplotlib.pyplot as plt
import mcsim.analysis.simulate_dmd as sdmd

# #########################################
# define wavelengths to calculate
# #########################################
# DMD physical data
gamma = 12 * np.pi/180
rot_axis = (1/np.sqrt(2), 1/np.sqrt(2), 0)
d = 7.56
# wavelength data
wavelengths = [0.465, 0.550, 0.635]
colors = ['b', 'g', 'r']
nc = len(wavelengths)

# number of points to sample curves
nangles = 3000 # must be even
npts = 4 * nangles

# diffraction orders
nlims = np.array([sdmd.get_diffraction_order_limits(wl, d, gamma, rot_axis) for wl in wavelengths])
nmax_all = nlims.max()
nmin_all = nlims.min()
ns_all = range(nmin_all, nmax_all + 1)
n_diff_orders = len(ns_all)

# #########################################
# find blaze solutions
# #########################################
a123s = np.zeros((n_diff_orders, nc, npts, 3)) * np.nan
axyzs = np.zeros((n_diff_orders, nc, npts, 3)) * np.nan
bxyzs = np.zeros((n_diff_orders, nc, npts, 3)) * np.nan
a_pts = np.linspace(-1, 1, nangles)

for jj in range(nc):
    wavelength = wavelengths[jj]
    nmin, nmax = sdmd.get_diffraction_order_limits(wavelength, d, gamma)

    # loop over diffraction orders
    for ii, n in enumerate(ns_all):
            if n < nmin or n > nmax:
                continue

            uvec_fn_ax, uvec_fn_ay = sdmd.solve_combined_condition(d, gamma, rot_axis, wavelength, (n, -n))

            # uvec_fn, a2lims = sdmd.solve_combined_condition(d, gamma, wavelength, n)
            # a2s = np.linspace(a2lims[0], a2lims[1], nangles)

            # axyzs[ii, jj, :npts//2], bxyzs[ii, jj, :npts//2] = uvec_fn(a2s, positive=True)
            # axyzs[ii, jj, -1:npts//2-1:-1], bxyzs[ii, jj, -1:npts // 2 - 1:-1] = uvec_fn(a2s, positive=False)

            at1, bt1, _ = uvec_fn_ax(a_pts, positive=True)
            at2, bt2, _ = uvec_fn_ax(a_pts, positive=False)
            at3, bt3, _ = uvec_fn_ay(a_pts, positive=True)
            at4, bt4, _ = uvec_fn_ay(a_pts, positive=False)

            a_temps = np.concatenate((at1, at2, at3, at4), axis=0)
            b_temps = np.concatenate((bt1, bt2, bt3, bt4), axis=0)

            not_nan = np.logical_not(np.any(np.isnan(a_temps), axis=1))
            ax_mid = np.mean(a_temps[not_nan, 0])
            ay_mid = np.mean(a_temps[not_nan, 1])
            isort = np.argsort(np.angle((a_temps[:, 0] - ax_mid) + 1j * (a_temps[:, 1] - ay_mid))[not_nan])

            axyzs[ii, jj, :len(isort)] = a_temps[not_nan][isort]
            bxyzs[ii, jj, :len(isort)] = b_temps[not_nan][isort]

# #########################################
# plot results
# #########################################
figh = plt.figure(figsize=(18, 14))
figh.suptitle("Allowed output angles satisfying the combined blaze and diffraction conditions\n"
             "parameterized by output unit vector $\hat{b}=(b_x, b_y, b_z)$")

ax = figh.add_subplot(1, 1, 1)
ax.axis("equal")

ph_on = []
ph_off = []
ax.plot(np.linspace(-1, 1, 300), -np.linspace(-1, 1, 300), 'k')
for jj in range(nc):
    ph = ax.plot(bxyzs[:, jj, :, 0].transpose(), bxyzs[:, jj, :, 1].transpose(), color=colors[jj])
    ph_on.append(ph[0])

    # use symmetry
    ph2 = ax.plot(axyzs[:, jj, :, 0].transpose(), axyzs[:, jj, :, 1].transpose(), '--', color=colors[jj])
    ph_off.append(ph2[0])
ax.plot(0, 0, 'kx')

strs_on = ["%dnm on, n=%d to %d" % (w*1e3, nlims[ii, 0], nlims[ii, 1]) for ii, w in enumerate(wavelengths)]
strs_off = ["%dnm off" % (w*1e3) for w in wavelengths]
ax.legend(ph_on + ph_off, strs_on + strs_off)
ax.set_xlabel("output unit-vector component $b_x$")
ax.set_ylabel("output unit-vector component $b_y$")

plt.show()
