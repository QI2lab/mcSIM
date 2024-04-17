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

# load DMD model, or define your own
dmd = sdmd.DLP6500()  # NOTE: DMD object stores spatial information in um

# output angles offset around the main output direction
tout_offsets = np.linspace(-10, 10, 4800) * np.pi / 180

# set up different wavelengths
wlens = [0.465, 0.635, 0.532, 0.785]  # must be in um to match default DMD models
diff_orders = [4, 3, -4, -3]  # todo: find closest orders instead of presetting
inverted = [False, False, True, True]
display_colors = ['b', 'r', 'g', 'k']

# set up pattern information
base_period = 5  # period of pattern at the first wavelength listed above
periods = np.zeros(len(wlens))
# create small pattern (i.e. not using full DMD size) to broaden diffraction peaks and speed up calculation
n_pattern = 50
xx, yy = np.meshgrid(range(n_pattern), range(n_pattern))

# get exact input/ouput angles and do simulation
uvecs_in_exact = np.zeros((len(wlens), 3))
tins_exact = np.zeros(len(wlens))
blaze_deviations = np.zeros_like(tins_exact)
simulation_data = []
for ii in range(len(wlens)):
    if inverted[ii]:
        gamma_use = dmd.gamma_off
        rot_axis_use = dmd.rot_axis_off
    else:
        gamma_use = dmd.gamma_on
        rot_axis_use = dmd.rot_axis_on

    if ii == 0:
        # for the first/main wavelength, ensure we satisfy the blaze and diffraction conditions
        uvecs_in, uvecs_out = sdmd.solve_1color_1d(wlens[ii],
                                                   dmd.dx,
                                                   gamma_use,
                                                   diff_orders[ii])
        uvecs_in_exact[ii] = uvecs_in[1]
        uvec_out = uvecs_out[1]
        tx_out, ty_out = sdmd.uvector2txty(*uvec_out)
        tp_out, tm_out = sdmd.angle2pm(tx_out, ty_out)
        uvec_out_pm = np.array(sdmd.xyz2mpz(*uvec_out))

        print(f"output angle, determined from {wlens[ii] * 1e3:.0f}nm, order {diff_orders[ii]:d}")
        print(f"output angle (tx, ty) = ({tx_out * 180/np.pi:2f}, {ty_out * 180/np.pi:.2f})deg")
        print(f"output angle (tm, tp) = ({tm_out * 180/np.pi:.2f}, {tp_out * 180/np.pi:.2f})deg")
        print("(bx, by, bz) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_out.squeeze()))
        print("(bm, bp, bz) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_out_pm))
    else:
        # for other wavelength, make sure the output direction is the same
        uvecs_in_exact[ii] = sdmd.solve_diffraction_input(uvec_out,
                                                          dmd.dx,
                                                          dmd.dy,
                                                          wlens[ii],
                                                          diff_orders[ii],
                                                          -diff_orders[ii])

    tx_in, ty_in = sdmd.uvector2txty(*uvecs_in_exact[ii])
    tp_in, tm_in = sdmd.angle2pm(tx_in, ty_in)
    tins_exact[ii] = tm_in
    uvec_in_pm = np.array(sdmd.xyz2mpz(*uvecs_in_exact[ii]))

    blaze = sdmd.solve_blaze_output(uvecs_in_exact[ii], gamma_use, rot_axis_use)
    blaze_tm, blaze_tp = sdmd.uvector2tmtp(*blaze[0])
    blaze_deviations[ii] = np.arccos(np.clip(np.sum(blaze[0] * uvec_out), -1, 1))

    print(f"input data for {wlens[ii] * 1e3:.0f}nm, order {diff_orders[ii]:d}:")
    print(f"input angle (tx, ty) = ({tx_in * 180 / np.pi:.2f}, {ty_in * 180 / np.pi:.2f})deg")
    print(f"input angle (tm, tp) = ({tins_exact[ii] * 180 / np.pi:.2f}, {0:.2f})deg")
    print("(ax, ay, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvecs_in_exact[ii].squeeze()))
    print("(am, ap, az) = (%0.4f, %0.4f, %0.4f)" % tuple(uvec_in_pm))
    print(f"blaze angle (tm, tp) = ({blaze_tp * 180 / np.pi:.2f}, {blaze_tm * 180 / np.pi:.2f}) deg")
    print(f"blaze deviation = {blaze_deviations[ii] * 180 / np.pi:.2f} deg")

    # generate pattern for this wavelength
    periods[ii] = base_period * wlens[ii] / wlens[0]
    sinusoid = np.cos(2 * np.pi * (xx - yy) / np.sqrt(2) / periods[ii])
    pattern = np.zeros(((n_pattern, n_pattern)), dtype=bool)
    pattern[sinusoid >= 0] = 1

    # do 1D DMD simulations
    data_temp = sdmd.simulate_1d(pattern,
                                 wlens[ii],
                                 dmd,
                                 tins_exact[ii],
                                 tm_out_offsets=tout_offsets,
                                 inverted=inverted[ii])
    simulation_data.append(data_temp)

# ##################################
# plot results of 1D simulations
# ##################################
figh = plt.figure(figsize=(16, 8))
str = (r"Diffraction orders and blaze envelopes for beams diffracted along the "
       r"$\hat{e}_- = \frac{x - y}{\sqrt{2}}$ direction" + "\n"
       "output direction, " + r"($\theta_-$, $\theta_+$)"
       f" = ({tm_out * 180/np.pi:.2f}, {tp_out * 180/np.pi:.2f}) deg; "
       r"($\theta_x$, $\theta_y$)"
       f" = ({tx_out * 180/np.pi:.2f}, {ty_out * 180/np.pi:.2f}) deg;"
       f" $(b_x, b_y, b_z)$ = ({uvec_out[0]:.4f}, {uvec_out[1]:.4f}, {uvec_out[2]:.4f})")

figh.suptitle(str)
ax = figh.add_subplot(1, 1, 1)

for ii in range(len(wlens)):
    data = simulation_data[ii]

    # load data
    _, tout = sdmd.uvector2tmtp(*data["uvecs_out"][0].transpose())
    sinc = np.abs(data['sinc_efield_on'][0] / data['wx'] / data['wy']) ** 2
    sinc_off = np.abs(data['sinc_efield_off'][0] / data['wx'] / data['wy']) ** 2
    int = np.abs(data['efields'][0]) ** 2

    # normalize diffraction patterns
    if inverted[ii]:
        sinc_ref = sinc_off
    else:
        sinc_ref = sinc

    peak_angle = tm_out + wlens[0] / (dmd.dx * base_period)
    ind = np.argmin(np.abs(peak_angle - tout))
    int_check = np.array(int, copy=True)
    int_check[np.arange(int.size) < (ind - 10)] = 0
    imax = np.argmax(int_check)
    int = int / int[imax] * sinc_ref[imax]

    # plot
    label = (f"{wlens[ii]*1e3:.0f}nm, "
             f"$(n_x, n_y)$ = ({diff_orders[ii]:d}, {-diff_orders[ii]:d}); "
             r"$\theta_-$"
             f" = {tins_exact[ii] * 180/np.pi:.2f}deg\n"
             f"a=({uvecs_in_exact[ii][0]:.3f}, "
             f"{uvecs_in_exact[ii][1]:.3f}, "
             f"{uvecs_in_exact[ii][2]:.3f})\n"
             f"blaze deviation = {blaze_deviations[ii]*180/np.pi:.2f}deg")

    ax.plot(tout * 180/np.pi,
            int,
            color=display_colors[ii],
            label=label)

    ax.plot(tout * 180/np.pi,
            sinc,
            color=display_colors[ii])

    ax.plot(tout * 180 / np.pi,
            sinc_off,
            '--',
            color=display_colors[ii])
    ax.set_ylim([-0.05, 1.1])

ax.set_xlim([tm_out * 180/np.pi - 10,
             tm_out * 180/np.pi + 10])
ax.legend()
ax.set_xlabel(r"$\theta_-$ (degrees)")
ax.set_ylabel("intensity (arb)")

plt.show()
