import numpy as np
import matplotlib.pyplot as plt
import mcsim.expt_ctrl.dlp6500 as dlp6500

dmd = dlp6500.dlp6500win(debug=False)

dmd.start_stop_sequence('stop')

# check DMD trigger state
delay1_us, mode_trig1 = dmd.get_trigger_in1()
print('trigger1 delay=%dus' % delay1_us)
print('trigger1 mode=%d' % mode_trig1)

# dmd.set_trigger_in2('rising')
mode_trig2 = dmd.get_trigger_in2()
print("trigger2 mode=%d" % mode_trig2)

# dmd.set_pattern_sequence([0, 0], [0, 1], 105, 0, triggered=True,
#                          clear_pattern_after_trigger=False, bit_depth=1, num_repeats=0, mode='pre-stored')

dmd_size = [1080, 1920]
ny, nx = dmd_size
cref = np.array([ny // 2, nx // 2])
xx, yy = np.meshgrid(range(nx), range(ny))

ang = -45 * np.pi/180
frq = np.array([np.sin(ang), np.cos(ang)]) * 1/4 * np.sqrt(2)
# frq = np.array([0, 0])
# rad = 5
rad = 0
phase = 0
# rad = np.inf
pattern_base = np.round(np.cos(2 * np.pi * (xx * frq[0] + yy * frq[1]) + phase), 12)
pattern_base[pattern_base <= 0] = 0
pattern_base[pattern_base > 0] = 1
pattern_base = 1 - pattern_base

# expected pupil info
na_mitutoyo = 0.55
dm = 7.56  # DMD mirror size
fl_mitutoyo = 4e3  # focal length of mitutoya objective
fl_olympus = 1.8e3
# magnification between DMD and Mitutoyo BFP
mag_dmd2bfp = 100 / 200 * 300 / 400 * fl_mitutoyo / fl_olympus

pupil_rad_mirrors = fl_mitutoyo * na_mitutoyo / mag_dmd2bfp / dm

if True:
    # (y, x)
    # offset = np.array([0, 0])
    offset = np.array([0, 0])

    # centered pattern
    centered_pattern = np.array(pattern_base, copy=True)
    centered_pattern[np.sqrt((xx - cref[1] - offset[1])**2 + (yy - cref[0] - offset[0])**2) > rad] = 1

    # centered_pattern = np.ones(dmd_size)
    img_inds, bit_inds = dmd.upload_pattern_sequence(centered_pattern.astype(np.uint8), 105, 0)
else:
    # or, many patterns

    # offs = np.arange(-45, 46, 5)
    # xoffs, yoffs = np.meshgrid(offs, 0)

    n_phis = 10
    phis = np.arange(n_phis) * 2 * np.pi / n_phis
    fractions = [0.5, 0.9]
    n_thetas = len(fractions)

    xoffs = np.zeros((n_thetas, n_phis))
    yoffs = np.zeros((n_thetas, n_phis))
    for ii in range(n_thetas):
        for jj in range(n_phis):
            xoffs[ii, jj] = np.cos(phis[jj]) * pupil_rad_mirrors * fractions[ii]
            yoffs[ii, jj] = np.sin(phis[jj]) * pupil_rad_mirrors * fractions[ii]

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
    exp_time_us = 200
    img_inds, bit_inds = dmd.upload_pattern_sequence(patterns, exp_time_us, 0)
