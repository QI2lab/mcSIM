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

            # solve combined blaze/diffraction conditions
            ab_from_ax_fn, ab_from_ay_fn = sdmd.solve_combined_condition(d, gamma, rot_axis, wavelength, (n, -n))
            a_from_ax, b_from_ax = ab_from_ax_fn(a_pts)
            a_from_ay, b_from_ay = ab_from_ay_fn(a_pts)

            # combine solutions
            a_temps = np.concatenate((a_from_ax[0], a_from_ax[1], a_from_ay[0], a_from_ay[1]), axis=0)
            b_temps = np.concatenate((b_from_ax[0], b_from_ax[1], b_from_ay[0], b_from_ay[1]), axis=0)


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
    ph = ax.plot(bxyzs[:, jj, :, 0].transpose(),
                 bxyzs[:, jj, :, 1].transpose(),
                 color=colors[jj])
    ph_on.append(ph[0])

    # use symmetry
    ph2 = ax.plot(axyzs[:, jj, :, 0].transpose(),
                  axyzs[:, jj, :, 1].transpose(),
                  '--',
                  color=colors[jj])
    ph_off.append(ph2[0])
ax.plot(0, 0, 'kx')

strs_on = [f"{w*1e3:.0f}nm on, n={nlims[ii, 0]:d} to {nlims[ii, 1]:d}" for ii, w in enumerate(wavelengths)]
strs_off = [f"{w*1e3:.0f}nm off" for w in wavelengths]
ax.legend(ph_on + ph_off, strs_on + strs_off)
ax.set_xlabel("output unit-vector component $b_x$")
ax.set_ylabel("output unit-vector component $b_y$")

plt.show()
