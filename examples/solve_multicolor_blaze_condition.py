"""
For a variety of wavelength, determine allowed input/output angles where the blaze condition and diffraction condition
can be satisfied simultaneously. These produce curves in 2D output angle space. Intersection of curves from different
wavelengths are locations where multicolor operation is possible.
"""
import numpy as np
import matplotlib.pyplot as plt
import mcsim.analysis.simulate_dmd as simulate_dmd

# #########################################
# define wavelengths to calculate
# #########################################
# DMD physical data
gamma = 12 * np.pi/180
d = 7.56
# wavelength data
wavelengths = [0.405, 0.465, 0.580, 0.635]
colors = [(1, 0, 1), 'b', 'g', 'r']

# number of points to sample curves
nangles = 3000 # must be even
npts = 2 * nangles

# diffraction orders
nlims = np.array([simulate_dmd.get_diffraction_order_limits(wl, d, gamma) for wl in wavelengths])
nmax_all = nlims.max()
nmin_all = nlims.min()
ns_all = range(nmin_all, nmax_all + 1)

# #########################################
# find blaze solutions
# #########################################
a123s = np.zeros((len(ns_all), len(wavelengths), npts, 3)) * np.nan
axyzs = np.zeros((len(ns_all), len(wavelengths), npts, 3)) * np.nan
bxyzs = np.zeros((len(ns_all), len(wavelengths), npts, 3)) * np.nan
for jj in range(len(wavelengths)):
    wavelength = wavelengths[jj]
    nmin, nmax = simulate_dmd.get_diffraction_order_limits(wavelength, d, gamma)

    # loop over diffraction orders
    for ii, n in enumerate(ns_all):
            if n < nmin or n > nmax:
                continue

            afn, bfn, a2lims = simulate_dmd.solve_combined_condition(d, gamma, wavelength, n)
            a2s = np.linspace(a2lims[0], a2lims[1], nangles)
            axyzs[ii, jj, :npts//2, 0], axyzs[ii, jj, :npts//2, 1], axyzs[ii, jj, :npts//2, 2] = afn(a2s, positive=True)
            axyzs[ii, jj, -1:npts//2-1:-1, 0], axyzs[ii, jj, -1:npts//2-1:-1, 1], axyzs[ii, jj, -1:npts//2-1:-1, 2] = afn(a2s, positive=False)

            bxyzs[ii, jj, :npts//2, 0], bxyzs[ii, jj, :npts//2, 1], bxyzs[ii, jj, :npts//2, 2] = bfn(a2s, positive=True)
            bxyzs[ii, jj, -1:npts // 2 - 1:-1, 0], bxyzs[ii, jj, -1:npts // 2 - 1:-1, 1], bxyzs[ii, jj,-1:npts // 2 - 1:-1, 2] = bfn(a2s, positive=False)


a_tx, a_ty = simulate_dmd.uvector2txty(axyzs[..., 0], axyzs[..., 1], axyzs[..., 2])
b_tx, b_ty = simulate_dmd.uvector2txty(bxyzs[..., 0], bxyzs[..., 1], bxyzs[..., 2])

# #########################################
# plot results
# #########################################
plt.figure(figsize=(18, 14))
plt.suptitle("Allowed input and output angles for combined blaze and diffraction conditions\n"
             "parameterized by input unit vector a=(ax, ay, az) output unit vector b=(bx, by, bz)")

plt.subplot(1, 2, 2)
plt.title("output directions")
ph_on = []
ph_off = []
plt.plot(np.linspace(-1, 1, 300), -np.linspace(-1, 1, 300), 'k')
for jj in range(len(wavelengths)):
    ph = plt.plot(bxyzs[:, jj, :, 0].transpose(), bxyzs[:, jj, :, 1].transpose(), color=colors[jj])
    ph_on.append(ph[0])
    # use symmetry
    ph2 = plt.plot(axyzs[:, jj, :, 0].transpose(), axyzs[:, jj, :, 1].transpose(), '--', color=colors[jj])
    ph_off.append(ph2[0])
plt.plot(0, 0, 'kx')

strs_on = ["%dnm on" % (w*1e3) for w in wavelengths]
strs_off = ["%dnm off" % (w*1e3) for w in wavelengths]
plt.legend(ph_on + ph_off, strs_on + strs_off)
plt.xlabel("bx")
plt.ylabel("by")


plt.subplot(1, 2, 1)
plt.title("input directions")
for jj in range(len(wavelengths)):
    plt.plot(axyzs[:, jj, :, 0].transpose(), axyzs[:, jj, :, 1].transpose(), color=colors[jj])
    # use symmetry
    plt.plot(bxyzs[:, jj, :, 0].transpose(), bxyzs[:, jj, :, 1].transpose(), '--', color=colors[jj])

plt.xlabel("ax")
plt.ylabel("ay")
