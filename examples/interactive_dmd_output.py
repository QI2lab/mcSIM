"""
Produce interactive graphic showing DMD diffraction orders in multiple colors, allowing different input angles
 for different colors.
"""

import numpy as np
import simulate_dmd as dmd

wavelengths = [0.473, 0.532, 0.635]
colors = ['b', 'g', 'r']
gamma = 12 * np.pi / 180
dx = 7.56 # um

dmd.interactive_display_2d(wavelengths, gamma, dx, max_diff_order=15, colors=colors, angle_increment=0.1, figsize=(16, 8))