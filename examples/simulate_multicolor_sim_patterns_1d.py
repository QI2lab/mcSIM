"""
A simple 1D simulation of multicolor SIM patterns with periods designed to match the positions of the
different diffraction orders in the Fourier plane. The main diffraction order output angle is chosen
to satisfy the blaze condition for the first wavelength. The input angles for the other wavelengths are chosen
to match this output direction.

This simulation produces a visualization of the diffraction output and blaze envelopes. It captures the effect
that violations of the blaze condition have on the SIM order strengths for different wavelengths.

In this example, we focus on the blaze/diffraction solution for 465nm and the (4, -4) order with input angle
theta_in~43deg and output angle theta_out~19deg.

For a complete description of the geometry and coordinate systems, see the docstring for simulate_dmd.py
"""
import numpy as np
import matplotlib.pyplot as plt
import mcsim.analysis.simulate_dmd as sdmd

# define DMD pitch, mirror size, mirror angles, etc.
dx = 7.56e-6 # DMD pitch in meters
dy = dx
coverage_f = 0.92 # pixel size, inferred from coverage fraction
wx = np.sqrt(coverage_f) * dx
wy = wx
# on and off angle of mirrors from the normal plane of the DMD body
gamma_on = 12 * np.pi/180
gamma_off = -12 * np.pi/180
rot_axis = (1/np.sqrt(2), 1/np.sqrt(2), 0)
# output angles offset around the main output direction
tout_offsets = np.linspace(-10, 10, 4800) * np.pi / 180

# set up different wavelengths
wlens = [0.465e-6, 0.635e-6, 0.532e-6, 0.785e-6]
diff_orders = [4, 3, -4, -3, 2] # todo: find closest orders instead of presetting
inverted = [False, False, True, True]
display_colors = ['b', 'r', 'g', 'k']

# set up pattern information
base_period = 5 # period of pattern at the first wavelength listed above
periods = np.zeros(len(wlens))
# create small pattern (i.e. not using full DMD size) to broaden diffraction peaks and speed up calculation
n_pattern = 50
xx, yy = np.meshgrid(range(n_pattern), range(n_pattern))

# get exact input/ouput angles and do simulation
uvecs_in_exact = np.zeros((len(wlens), 3))
tins_exact = np.zeros(len(wlens))
simulation_data = []
for ii in range(len(wlens)):
    if inverted[ii]:
        gon, goff = gamma_off, gamma_on
    else:
        gon, goff = gamma_on, gamma_off

    # for the first/main wavelength, ensure we satisfy the blaze and diffraction conditions
    if ii == 0:
        uvecs_in, uvecs_out = sdmd.solve_1color_1d(wlens[ii], dx, gon, diff_orders[ii])
        uvecs_in_exact[ii] = uvecs_in[1]
        uvec_out = uvecs_out[1]

        tx_in, ty_in = sdmd.uvector2txty(*uvecs_in_exact[ii])
        tp_in, tm_in = sdmd.angle2pm(tx_in, ty_in)
        tins_exact[ii] = tm_in
        uvec_in_pm = np.array(sdmd.xyz2mpz(*uvecs_in_exact[ii]))

        tx_out, ty_out = sdmd.uvector2txty(*uvec_out)
        tp_out, tm_out = sdmd.angle2pm(tx_out, ty_out)
        uvec_out_pm = np.array(sdmd.xyz2mpz(*uvec_out))

        print("output angle, determined from %0.0fnm, order %d" % (wlens[ii] * 1e9, diff_orders[ii]))
        print("output angle (tx, ty) = (%0.2f, %0.2f)deg" % (tx_out * 180/np.pi, ty_out * 180/np.pi))
        print("output angle (tm, tp) = (%0.2f, %0.2f)deg" % (tm_out * 180/np.pi, tp_out))
        print("(bx, by, bz) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_out.squeeze()))
        print("(bm, bp, bz) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_out_pm))

        print("input data for %.0fnm, order %d:" % (wlens[ii] * 1e9, diff_orders[ii]))
        print("input angle (tx, ty) = (%0.2f, %0.2f)deg" % (tx_in * 180/np.pi, ty_in * 180/np.pi))
        print("input angle (tm, tp) = (%0.2f, %0.2f)deg" % (tins_exact[ii] * 180/np.pi, 0))
        print("(ax, ay, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvecs_in_exact[ii].squeeze()))
        print("(am, ap, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_in_pm))

    # for other wavelength, make sure the output direction is the same
    else:
        uvecs_in_exact[ii] = sdmd.solve_diffraction_input(uvec_out,
                                                          dx,
                                                          dy,
                                                          wlens[ii],
                                                          (diff_orders[ii], -diff_orders[ii]))
        tx_in, ty_in = sdmd.uvector2txty(*uvecs_in_exact[ii])
        tp_in, tm_in = sdmd.angle2pm(tx_in, ty_in)
        tins_exact[ii] = tm_in
        uvec_in_pm = np.array(sdmd.xyz2mpz(*uvecs_in_exact[ii]))

        print("input data for %.0fnm, order %d:" % (wlens[ii] * 1e9, diff_orders[ii]))
        print("input angle (tx, ty) = (%0.2f, %0.2f)deg" % (tx_in * 180 / np.pi, ty_in * 180 / np.pi))
        print("input angle (tm, tp) = (%0.2f, %0.2f)deg" % (tins_exact[ii] * 180 / np.pi, 0))
        print("(ax, ay, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvecs_in_exact[ii].squeeze()))
        print("(am, ap, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_in_pm))

    # generate pattern for this wavelength
    periods[ii] = base_period * wlens[ii] / wlens[0]
    sinusoid = np.cos(2 * np.pi * (xx - yy) / np.sqrt(2) / periods[ii])
    pattern = np.zeros(((n_pattern, n_pattern)))
    pattern[sinusoid >= 0] = 1

    # do 1D DMD simulations
    data_temp = sdmd.simulate_1d(pattern,
                                 [wlens[ii]],
                                 gon,
                                 rot_axis,
                                 goff,
                                 rot_axis,
                                 dx,
                                 dy,
                                 wx,
                                 wy,
                                 tins_exact[ii],
                                 tm_out_offsets=tout_offsets)
    simulation_data.append(data_temp)

# ##################################
# plot results of 1D simulations
# ##################################
figh = plt.figure(figsize=(16, 8))
str = r"Diffraction orders and blaze envelopes for beams diffracted along the $\hat{e}_- = \frac{x - y}{\sqrt{2}}$ direction" + \
      "\noutput direction, " + r"($\theta_-$, $\theta_+$)" + " = (%0.2f, %0.2f) deg; " + \
      r"($\theta_x$, $\theta_y$)" + " = (%0.2f, %0.2f) deg;" + \
      " $(b_x, b_y, b_z)$ = (%0.4f, %0.4f, %0.4f)"
figh.suptitle(str %
             (tm_out * 180/np.pi,
              tp_out * 180/np.pi,
              tx_out * 180/np.pi,
              ty_out * 180/np.pi,
              uvec_out[0],
              uvec_out[1],
              uvec_out[2]))
ax = figh.add_subplot(1, 1, 1)

leg = []
hs = []
for ii in range(len(wlens)):
    data = simulation_data[ii]

    _, tout = sdmd.uvector2tmtp(*data["uvecs_out"][0].transpose())
    sinc = np.abs(data['sinc_efield_on'][0] / data['wx'] / data['wy']) ** 2
    sinc_off = np.abs(data['sinc_efield_off'][0] / data['wx'] / data['wy']) ** 2
    int = np.abs(data['efields'][0]) ** 2

    peak_angle = tm_out + wlens[0] / (dx * base_period)
    ind = np.argmin(np.abs(peak_angle - tout))
    int_check = np.array(int, copy=True)
    int_check[np.arange(int.size) < (ind - 10)] = 0
    imax = np.argmax(int_check)
    int = int / int[imax] * sinc[imax]

    label = ("%.0fnm, $(n_x, n_y)$ = (%d, %d); " + r"$\theta_-$" + " = %0.2fdeg" + "\na=(%0.3f, %0.3f, %0.3f)") % \
            (wlens[ii]*1e9,
             diff_orders[ii],
             -diff_orders[ii],
             tins_exact[ii] * 180/np.pi,
             uvecs_in_exact[ii][0],
             uvecs_in_exact[ii][1],
             uvecs_in_exact[ii][2])

    h, = ax.plot(tout * 180/np.pi, int, color=display_colors[ii], label=label)
    hs.append(h)
    h, = ax.plot(tout * 180/np.pi, sinc, color=display_colors[ii])
    hs.append(h)
    ax.plot(tout * 180 / np.pi, sinc_off, '--', color=display_colors[ii])
    ax.set_ylim([-0.05, 1.1])

ax.set_xlim([tm_out * 180/np.pi - 10, tm_out * 180/np.pi + 10])
ax.legend()
ax.set_xlabel(r"$\theta_-$ (degrees)")
ax.set_ylabel("intensity (arb)")

plt.show()
