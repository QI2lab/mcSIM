"""
For a variety of wavelength, determine allowed input/output angles where the blaze condition and diffraction condition
can be satisfied simultaneously. These produce curves in 2D output angle space. Intersection of curves from different
wavelengths are locations where multicolor operation is possible.
"""
import numpy as np
import matplotlib.pyplot as plt
import simulate_dmd

gamma = 12 * np.pi/180
wavelengths = [0.405, 0.473, 0.580, 0.635]
colors = [(1, 0, 1), 'b', 'g', 'r']
# wavelengths = [0.473, 0.532, 0.635]
# colors = ['b', 'g', 'r']
d = 7.56

nmin_all = int(np.ceil(-np.sqrt(2) * np.sin(gamma) / np.max(wavelengths) * d))
nmax_all = int(np.floor(np.sqrt(2) * np.sin(gamma) / np.min(wavelengths) * d))
ns_all = range(nmin_all, nmax_all + 1)

nangles = 2000 # must be even

a3s = np.zeros((len(ns_all), len(wavelengths), 2*nangles)) * np.nan
a2s = np.zeros(a3s.shape) * np.nan
a1s = np.zeros(a3s.shape) * np.nan
solns_tins = np.zeros((len(ns_all), len(wavelengths), 2)) * np.nan
solns_touts = np.zeros((len(ns_all), len(wavelengths), 2)) * np.nan
for jj in range(len(wavelengths)):
    wavelength = wavelengths[jj]

    if gamma >= 0:
        nmin = 1
        nmax = int(np.floor(np.sqrt(2) * np.sin(gamma) / wavelength * d))
    else:
        nmin = int(np.ceil(-np.sqrt(2) * np.sin(gamma) / wavelength * d))
        nmax = -1


    for ii in range(len(ns_all)):
            n = ns_all[ii]

            if n < nmin or n > nmax:
                continue

            ins, outs = simulate_dmd.solve_1color_1d(wavelength, d, gamma, n)
            tan_ins = np.tan(ins) / np.sqrt(2)
            tan_outs = np.tan(outs) / np.sqrt(2)

            if tan_ins.size != 0:
                solns_tins[ii, jj] = tan_ins / np.sqrt(1 + 2*tan_ins**2)
                solns_touts[ii, jj] = tan_outs / np.sqrt(1 + 2*tan_outs**2)

            a3s[ii, jj] = 1 / np.sqrt(2) / np.sin(gamma) * wavelength / d * n

            # ensure we get a2=0
            a2s_quarter = np.linspace(-np.sqrt(1 - a3s[ii, jj, 0] ** 2), 0, int(nangles/2))
            a2s_half = np.concatenate((a2s_quarter, np.flip(-a2s_quarter)), axis=0)
            a2s[ii, jj] = np.concatenate((a2s_half, np.flip(a2s_half)), axis=0)

            # a1s_half = np.linspace(-np.sqrt(1 - a3s[ii, jj, 0] ** 2), np.sqrt(1 - a3s[ii, jj, 0] ** 2), nangles)
            # a1s[ii, jj] = np.concatenate((a1s_half, a1s_half), axis=0)

            # a2s_half = np.sqrt(1 - a1s_half ** 2 - a3s[ii, jj, 0] ** 2)
            # a2s[ii, jj] = np.concatenate((-a2s_half, a2s_half), axis=0)

            # rounding to ensure ~0 number is not interpreted as negative number
            a1s_half = np.sqrt(np.round(1 - a2s_half ** 2 - a3s[ii, jj, 0] ** 2, 15))
            a1s[ii, jj] = np.concatenate((-a1s_half, np.flip(a1s_half)), axis=0)

b1s = a1s
b2s = a2s
b3s = -a3s

axs = np.cos(gamma) / np.sqrt(2) * a1s + 1 / np.sqrt(2) * a2s + np.sin(gamma) / np.sqrt(2) * a3s
ays = -np.cos(gamma) / np.sqrt(2) * a1s + 1 / np.sqrt(2) * a2s - np.sin(gamma) / np.sqrt(2) * a3s
azs = -np.sin(gamma) * a1s + np.cos(gamma) * a3s
# axs = np.cos(gamma) / np.sqrt(2) * a1s + 1 / np.sqrt(2) * a2s - np.sin(gamma) / np.sqrt(2) * a3s
# ays = -np.cos(gamma) / np.sqrt(2) * a1s + 1 / np.sqrt(2) * a2s + np.sin(gamma) / np.sqrt(2) * a3s
# azs = np.sin(gamma) * a1s + np.cos(gamma) * a3s

bxs = np.cos(gamma) / np.sqrt(2) * b1s + 1 / np.sqrt(2) * b2s + np.sin(gamma) / np.sqrt(2) * b3s
bys = -np.cos(gamma) / np.sqrt(2) * b1s + 1 / np.sqrt(2) * b2s - np.sin(gamma) / np.sqrt(2) * b3s
bzs = -np.sin(gamma) * b1s + np.cos(gamma) * b3s

a_tx, a_ty = simulate_dmd.uvector2txty(axs, ays, azs)
b_tx, b_ty = simulate_dmd.uvector2txty(bxs, bys, bzs)


plt.figure(figsize=(18, 14))
plt.suptitle("Allowed input and output angles for combined blaze and diffraction conditions\n"
             "parameterized by input unit vector a=(ax, ay, az) output unit vector b=(bx, by, bz)")

plt.subplot(1, 2, 2)
plt.title("output angles")
ph_on = []
ph_off = []
plt.plot(np.linspace(-1, 1, 300), -np.linspace(-1, 1, 300), 'k')
for jj in range(len(wavelengths)):
    ph = plt.plot(bxs[:, jj, :].transpose(), bys[:, jj, :].transpose(), color=colors[jj])
    ph_on.append(ph[0])
    # use symmetry
    ph2 = plt.plot(axs[:, jj, :].transpose(), ays[:, jj, :].transpose(), '--', color=colors[jj])
    ph_off.append(ph2[0])
plt.plot(0, 0, 'kx')

strs_on = ["%dnm on" % (w*1e3) for w in wavelengths]
strs_off = ["%dnm off" % (w*1e3) for w in wavelengths]
plt.legend(ph_on + ph_off, strs_on + strs_off)
plt.xlabel("bx")
plt.ylabel("by")


plt.subplot(1, 2, 1)
plt.title("input angles")
for jj in range(len(wavelengths)):
    plt.plot(axs[:, jj, :].transpose(), ays[:, jj, :].transpose(), color=colors[jj])
    # use symmetry
    plt.plot(bxs[:, jj, :].transpose(), bys[:, jj, :].transpose(), '--', color=colors[jj])

plt.xlabel("ax")
plt.ylabel("ay")

# figh = plt.figure(figsize=(18, 14))
# plt.suptitle("Allowed input and output angles for combined blaze and diffraction conditions\n"
#              "parameterized by output angle projections")
# for jj in range(len(wavelengths)):
#     plt.plot(b_tx[:, jj, :].transpose() * 180/np.pi, b_ty[:, jj, :].transpose() * 180/np.pi, color=colors[jj])
#
#     plt.plot(a_tx[:, jj, :].transpose() * 180 / np.pi, a_ty[:, jj, :].transpose() * 180 / np.pi, '--', color=colors[jj])