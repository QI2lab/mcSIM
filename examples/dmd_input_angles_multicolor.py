"""
Simulate a given DMD pattern for several wavelength and corresponding diffraction orders and input angles.
This script only considers output angles along the diagonal (tx=-ty). Also display information showing different
diffraction orders and the blaze condition.
"""
import numpy as np
import matplotlib.pyplot as plt
import simulate_dmd as sdmd

# define DMD values
# pixel spacing (DMD pitch)
dx = 7.56e-6 # meters
dy = dx
# pixel size, inferred from coverage fraction
coverage_f = 0.92
wx = np.sqrt(coverage_f) * dx
wy = wx
# on and off angle of mirrors from the normal plane of the DMD body
gamma_on = 12 * np.pi/180
gamma_off = -12 * np.pi/180
tout_offsets = np.linspace(-10, 10, 4800) * np.pi / 180
# colors
# we want to use the solution near theta_in=43deg, theta_out=19deg, n=4 for 465nm, n=3 for 635nm
wlens = [0.465e-6, 0.635e-6, 0.532e-6, 0.785e-6, 0.980e-6]
diff_orders = [4, 3, -4, -3, 2]
inverted = [False, False, True, True, False]
display_colors = ['b', 'r', 'g', 'y', 'k']
#base_period = np.sqrt(2) * 3
base_period = 5
periods = np.zeros(len(wlens))
# create small pattern (to broaden diffraction peaks
n_pattern = 50
xx, yy = np.meshgrid(range(n_pattern), range(n_pattern))

# get exact angles and do simulation
tins_exact = np.zeros(len(wlens))
simulation_data = []
for ii in range(len(wlens)):
    if inverted[ii]:
        gon, goff = gamma_off, gamma_on
    else:
        gon, goff = gamma_on, gamma_off

    if ii == 0:
        ti, to = sdmd.solve_1color_1d(wlens[ii], dx, gon, diff_orders[ii])
        tins_exact[ii] = ti[0]
        tout_exact = to[0]
    else:
        tins_exact[ii] = sdmd.solve_diffraction_input_1d(tout_exact, wlens[ii], dx, diff_orders[ii])
        # tin, tout = solve_1color(wlens[ii], dx, gon, diff_orders[ii])

    # 1D DMD simulations
    periods[ii] = base_period * wlens[ii] / wlens[0]
    sinusoid = np.cos(2 * np.pi * (xx - yy) / np.sqrt(2) / periods[ii])
    pattern = np.zeros(((n_pattern, n_pattern)))
    pattern[sinusoid >= 0] = 1
    #
    data_temp = sdmd.simulate_1d(pattern, [wlens[ii]], gon, goff, dx, dy, wx, wy, tins_exact[ii], t45_out_offsets=tout_offsets)
    simulation_data.append(data_temp)


# sample 1D simulation
figh = plt.figure()
plt.suptitle("output angle = %0.2f deg" % (tout_exact * 180/np.pi))

leg = []
hs = []
for ii in range(len(wlens)):
    data = simulation_data[ii]

    tout = data['t45s_out'].transpose()
    sinc = np.abs(data['sinc_efield_on'][0] / data['wx'] / data['wy']) ** 2
    sinc_off = np.abs(data['sinc_efield_off'][0] / data['wx'] / data['wy']) ** 2
    int = np.abs(data['efields'][0]) ** 2

    peak_angle = tout_exact + wlens[0] / (dx * base_period)
    ind = np.argmin(np.abs(peak_angle - tout))
    int_check = np.array(int, copy=True)
    int_check[np.arange(int.size) < (ind - 10)] = 0
    imax = np.argmax(int_check)
    int = int / int[imax] * sinc[imax]

    h, = plt.plot(tout * 180/np.pi, int, color=display_colors[ii])
    hs.append(h)
    h, = plt.plot(tout * 180/np.pi, sinc, color=display_colors[ii])
    hs.append(h)
    plt.plot(tout * 180 / np.pi, sinc_off, '--', color=display_colors[ii])
    plt.ylim([-0.05, 1.1])


    leg += ['%.0f order, tin = %0.2fdeg' % (wlens[ii]*1e9, tins_exact[ii] * 180/np.pi), '%.0f sinc envelope' % (wlens[ii]*1e9)]

plt.xlim([12, 28])
plt.legend(hs, leg)
plt.xlabel('Output angle in x-y plane (degrees)')
plt.ylabel('intensity (arb)')


