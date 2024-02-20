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
dmd = sdmd.DLP6500()

# wavelength data
wavelengths = [0.465, 0.550, 0.635]
cmaps = ['Blues', 'Greens', 'Reds']
nc = len(wavelengths)

# diffraction orders
nlims = np.array([sdmd.get_diffraction_order_limits(wl,
                                                    dmd.dx,
                                                    dmd.gamma_on,
                                                    dmd.rot_axis_on)
                  for wl in wavelengths])
ns_all = range(nlims.min(), nlims.max() + 1)
n_diff_orders = len(ns_all)

# #########################################
# find blaze solutions
# #########################################

# number of points to sample curves
nangles = 3000  # must be even
npts = 4 * nangles
a_pts = np.linspace(-1, 1, nangles)

# accumulate solution vectors
axyzs = np.zeros((n_diff_orders, nc, npts, 3)) * np.nan
bxyzs = np.zeros((n_diff_orders, nc, npts, 3)) * np.nan
for jj, wavelength in enumerate(wavelengths):
    # loop over diffraction orders
    for ii, n in enumerate(ns_all):
        if n < nlims[jj, 0] or n > nlims[jj, 1]:
            continue

        # solve combined blaze/diffraction conditions
        ab_from_ax_fn, ab_from_ay_fn, _ = sdmd.solve_combined_condition(dmd.dx,
                                                                        dmd.gamma_on,
                                                                        dmd.rot_axis_on,
                                                                        wavelength,
                                                                     (n, -n))
        a_from_ax, b_from_ax = ab_from_ax_fn(a_pts)
        a_from_ay, b_from_ay = ab_from_ay_fn(a_pts)

        # combine solutions
        a_temps = np.concatenate((a_from_ax[0], a_from_ax[1], a_from_ay[0], a_from_ay[1]), axis=0)
        b_temps = np.concatenate((b_from_ax[0], b_from_ax[1], b_from_ay[0], b_from_ay[1]), axis=0)

        # sort by angle
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
              r"parameterized by output unit vector $\hat{b}=(b_x, b_y, b_z)$")

ax = figh.add_subplot(1, 1, 1)
ax.axis("equal")

ax.plot([-1, 1], [1, -1], 'k')
for jj in range(nc):
    for ii in range(n_diff_orders):
        if not np.all(np.isnan(bxyzs[ii, jj, :])):
            cmap = plt.get_cmap(cmaps[jj])
            counter = (ns_all[ii] - nlims[jj, 0] + 1) / (nlims[jj, 1] - nlims[jj, 0] + 1)
            color = cmap(counter)

            ph = ax.plot(bxyzs[ii, jj, :, 0].transpose(),
                         bxyzs[ii, jj, :, 1].transpose(),
                         color=color,
                         label=f"{wavelengths[jj]*1e3:.0f}nm on,"
                               f" n={ns_all[ii]:d}")

            # use symmetry
            ph2 = ax.plot(axyzs[ii, jj, :, 0].transpose(),
                          axyzs[ii, jj, :, 1].transpose(),
                          '--',
                          color=color,
                          label=f"{wavelengths[jj]*1e3:.0f}nm off, n={-ns_all[ii]:d}")
ax.plot(0, 0, 'kx')

ax.legend()
ax.set_xlabel("output unit-vector component $b_x$")
ax.set_ylabel("output unit-vector component $b_y$")

plt.show()
