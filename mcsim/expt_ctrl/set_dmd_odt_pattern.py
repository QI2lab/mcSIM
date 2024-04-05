"""
Set a specific pattern to the DMD for testing
This script is typically run manually and not called from other code
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mcsim.expt_ctrl.dlp6500 as dlp6500

# ##################################
# prepare DMD for new patterns
# ##################################
dmd = dlp6500.dlp6500win(debug=False)

dmd.start_stop_sequence('stop')

# check DMD trigger state
delay1_us, mode_trig1 = dmd.get_trigger_in1()
print('trigger1 delay=%dus' % delay1_us)
print('trigger1 mode=%d' % mode_trig1)
mode_trig2 = dmd.get_trigger_in2()
print("trigger2 mode=%d" % mode_trig2)

# ##################################
# define patterns
# ##################################
dmd_size = np.array([1080, 1920], dtype=int)
ny, nx = dmd_size
cref = np.array([ny // 2, nx // 2])
xx, yy = np.meshgrid(range(nx), range(ny))

# pattern parameters
ang = -45 * np.pi/180
frq = np.array([np.sin(ang), np.cos(ang)]) * 1/4 * np.sqrt(2)
rad = 100
phase = 0

# base pattern
pattern_base = np.round(np.cos(2 * np.pi * (xx * frq[0] + yy * frq[1]) + phase), 12)
pattern_base[pattern_base <= 0] = 0
pattern_base[pattern_base > 0] = 1
pattern_base = 1 - pattern_base

# expected pupil info
dm = 7.56  # DMD mirror size
# fl_detection = 4e3  # focal length of mitutoya objective
# na_detection = 0.55
fl_detection = 3e3
na_detection = 1
fl_excitation = 1.8e3
# magnification between DMD and Mitutoyo BFP
mag_dmd2bfp = 100 / 200 * 300 / 400 * fl_detection / fl_excitation

pupil_rad_mirrors = fl_detection * na_detection / mag_dmd2bfp / dm

# ##################################
# set patterns
# ##################################
one_pattern = True
if one_pattern:
    # (y, x)
    # offset = np.array([0, 0])

    phi = -149 * np.pi/180
    fraction = 0.0
    offset = np.array([np.cos(phi) * pupil_rad_mirrors * fraction, np.sin(phi) * pupil_rad_mirrors * fraction])

    # centered pattern
    centered_pattern = np.array(pattern_base, copy=True)
    centered_pattern[np.sqrt((xx - cref[1] - offset[1])**2 + (yy - cref[0] - offset[0])**2) > rad] = 1

    # centered_pattern = np.ones(dmd_size)
    img_inds, bit_inds = dmd.upload_pattern_sequence(centered_pattern.astype(np.uint8), 105, 0)
else:
    # or, many patterns
    n_phis = 10
    phis = np.arange(n_phis) * 2 * np.pi / n_phis
    fractions = [0.97]
    n_thetas = len(fractions)

    xoffs = np.zeros((n_thetas, n_phis))
    yoffs = np.zeros((n_thetas, n_phis))
    for ii in range(n_thetas):
        for jj in range(n_phis):
            xoffs[ii, jj] = np.cos(phis[jj]) * pupil_rad_mirrors * fractions[ii]
            yoffs[ii, jj] = np.sin(phis[jj]) * pupil_rad_mirrors * fractions[ii]

    add_zero = True
    if add_zero:
        xoffs = np.concatenate((np.array([0]), xoffs.ravel()))
        yoffs = np.concatenate((np.array([0]), yoffs.ravel()))

    npatterns = xoffs.size

    patterns = np.ones((npatterns, ny, nx), dtype=np.uint8)
    for ii in range(npatterns):
        patterns[ii] = np.copy(pattern_base)
        patterns[ii, np.sqrt((xx - cref[1] - xoffs.ravel()[ii])**2 +
                             (yy - cref[0] - yoffs.ravel()[ii])**2) > rad] = 1

    # exp_time_us = 1000000 # 1s
    # exp_time_us = 100000 # 100ms
    exp_time_us = 105
    img_inds, bit_inds = dmd.upload_pattern_sequence(patterns, exp_time_us, 0)
