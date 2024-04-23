"""
Simulate diffraction from a given pattern displayed on the DMD. Do simulations both in "1D", i.e. considering only
diffraction in the tx=-ty plane, and in "2D", i.e. considering the full output space.

Unlike simulate_multicolor_sim_patterns_1d.py, this file simulates the same input angles for all wavelength considered
"""
import numpy as np
import matplotlib.pyplot as plt
import mcsim.analysis.simulate_dmd as sdmd

# load DMD model, or define your own
dmd = sdmd.DLP6500()

wavelengths = [0.465, 0.532, 0.635]  # in um
colors = ["b", "g", "r"]

tx_ins = -np.array([35]) * np.pi / 180
ty_ins = np.array([30]) * np.pi / 180
_, tm_ins = sdmd.angle2pm(tx_ins, ty_ins)

# create pattern
nx = 50
ny = 50

# set pattern for DMD to be simulated
[xx, yy] = np.meshgrid(range(nx), range(ny))
theta_pattern = -np.pi / 4
period = 3 * np.sqrt(2)

pattern = np.cos(2 * np.pi / period * (xx * np.cos(theta_pattern) + yy * np.sin(theta_pattern)))
pattern[pattern <= 0] = 0
pattern[pattern > 0] = 1
pattern = pattern.astype(bool)

# sample 1D simulation
data1d = sdmd.simulate_1d(pattern,
                          wavelengths,
                          dmd,
                          tm_ins)
sdmd.plot_1d_sim(data1d, colors, save_dir=None)

# sample 2D simulation
data2d = sdmd.simulate_2d(pattern,
                          wavelengths,
                          dmd,
                          tx_ins,
                          ty_ins,
                          tout_offsets=np.linspace(-10, 10, 201) * np.pi / 180)
sdmd.plot_2d_sim(data2d, save_dir=None)

plt.show()
