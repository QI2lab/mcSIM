"""
Simulate a given DMD pattern for several wavelength and corresponding diffraction orders and input directions.

We adopt the following coordinate system: suppose we have the DMD orientated above an optical table. Let the z-direction
be the normal to the DMD chip, facing outward from the chip. Let the x- and y-directions be along the principle axes
of the DMD. Suppose the DMD is rotated 45degrees about the z-axis so that the corner which is the origin is clisest to
the table. The x+y direction is the vertical direction pointing away from the table. The x-y direction is parallel
to the table.

The input direction is parameterized by the unit vector a = (ax, ay, az), where we must have az < 0. Similarly
the output direction is parameterized by unit vector b = (bx, by, bz), where we must have bz > 0.

Sometimes we will also refer to an angular parameterization of the input/output directions. In this case, the angles
are defined differently for the input and output.
a = (tan(tx), tan(ty), -1) / sqrt( tan(tx)**2 + tan(ty)**2 + 1)
b = (tan(tx), tan(ty), +1) / sqrt( tan(tx)**2 + tan(ty)**2 + 1)

This script only considers output directions along the diagonal (vx=-vy). Also display information showing different
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
diff_orders = [-4, -3, 4, 3, -2]
inverted = [False, False, True, True, False]
display_colors = ['b', 'r', 'g', 'y', 'k']
#base_period = np.sqrt(2) * 3
base_period = 5
periods = np.zeros(len(wlens))
# create small pattern (to broaden diffraction peaks
n_pattern = 50
xx, yy = np.meshgrid(range(n_pattern), range(n_pattern))

# get exact angles and do simulation
uvecs_in_exact = np.zeros((len(wlens), 3))
tins_exact = np.zeros(len(wlens))
simulation_data = []
for ii in range(len(wlens)):
    if inverted[ii]:
        gon, goff = gamma_off, gamma_on
    else:
        gon, goff = gamma_on, gamma_off

    if ii == 0:
        # ti, to = sdmd.solve_1color_1d(wlens[ii], dx, gon, diff_orders[ii])
        uvecs_in, uvecs_out = sdmd.solve_1color_1d(wlens[ii], dx, gon, diff_orders[ii])
        uvecs_in_exact[ii] = uvecs_in[1]
        uvec_out = uvecs_out[1]
        # ti, to = sdmd.solve_1color_1d(wlens[ii], dx, gon, diff_orders[ii])
        # tins_exact[ii] = ti[0]
        # tout_exact = to[0]

        tx_in, ty_in = sdmd.uvector2txty(*uvecs_in_exact[ii])
        tp_in, tm_in = sdmd.angle2pm(tx_in, ty_in)
        tins_exact[ii] = tm_in
        uvec_in_pm = np.array(sdmd.xyz2mpz(*uvecs_in_exact[ii]))

        tx_out, ty_out = sdmd.uvector2txty(*uvec_out)
        tp_out, tm_out = sdmd.angle2pm(tx_out, ty_out)
        tout_exact = tm_out
        uvec_out_pm = np.array(sdmd.xyz2mpz(*uvec_out))

        print("output angle, determined from %0.0fnm, order %d" % (wlens[ii] * 1e9, diff_orders[ii]))
        print("output angle (tx, ty) = (%0.2f, %0.2f)deg" % (tx_out * 180/np.pi, ty_out * 180/np.pi))
        print("output angle (tm, tp) = (%0.2f, %0.2f)deg" % (tout_exact * 180/np.pi, 0))
        print("(bx, by, bz) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_out.squeeze()))
        print("(bm, bp, bz) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_out_pm))

        print("input data for %.0fnm, order %d:" % (wlens[ii] * 1e9, diff_orders[ii]))
        print("input angle (tx, ty) = (%0.2f, %0.2f)deg" % (tx_in * 180/np.pi, ty_in * 180/np.pi))
        print("input angle (tm, tp) = (%0.2f, %0.2f)deg" % (tins_exact[ii] * 180/np.pi, 0))
        print("(ax, ay, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvecs_in_exact[ii].squeeze()))
        print("(am, ap, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_in_pm))
    else:
        uvecs_in_exact[ii] = sdmd.solve_diffraction_input(uvec_out, dx, dy, wlens[ii], (diff_orders[ii], -diff_orders[ii]))
        # tins_exact[ii] = sdmd.solve_diffraction_input_1d(tout_exact, wlens[ii], dx, diff_orders[ii])

        tx_in, ty_in = sdmd.uvector2txty(*uvecs_in_exact[ii])
        tp_in, tm_in = sdmd.angle2pm(tx_in, ty_in)
        tins_exact[ii] = tm_in
        uvec_in_pm = np.array(sdmd.xyz2mpz(*uvecs_in_exact[ii]))

        print("input data for %.0fnm, order %d:" % (wlens[ii] * 1e9, diff_orders[ii]))
        print("input angle (tx, ty) = (%0.2f, %0.2f)deg" % (tx_in * 180 / np.pi, ty_in * 180 / np.pi))
        print("input angle (tm, tp) = (%0.2f, %0.2f)deg" % (tins_exact[ii] * 180 / np.pi, 0))
        print("(ax, ay, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvecs_in_exact[ii].squeeze()))
        print("(am, ap, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_in_pm))


    # 1D DMD simulations
    periods[ii] = base_period * wlens[ii] / wlens[0]
    sinusoid = np.cos(2 * np.pi * (xx - yy) / np.sqrt(2) / periods[ii])
    pattern = np.zeros(((n_pattern, n_pattern)))
    pattern[sinusoid >= 0] = 1
    #
    data_temp = sdmd.simulate_1d(pattern, [wlens[ii]], gon, goff, dx, dy, wx, wy, tins_exact[ii], tm_out_offsets=tout_offsets)
    simulation_data.append(data_temp)


# sample 1D simulation
figh = plt.figure(figsize=(16, 8))
plt.suptitle("output angle = %0.2f deg\nunit vector (bx, by, bz) = (%0.4f, %0.4f, %0.4f)" %
             (tout_exact * 180/np.pi, uvec_out[0], uvec_out[1], uvec_out[2]))

leg = []
hs = []
for ii in range(len(wlens)):
    data = simulation_data[ii]

    _, tout = sdmd.uvector2tmtp(*data["uvecs_out"][0].transpose())
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

# plt.xlim([12, 28])
plt.xlim([tout_exact * 180/np.pi - 10, tout_exact * 180/np.pi + 10])
plt.legend(hs, leg)
plt.xlabel('Output angle in x-y plane (degrees)')
plt.ylabel('intensity (arb)')


