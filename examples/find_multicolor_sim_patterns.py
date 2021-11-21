"""
Generate SIM patterns for three wavelengths and DMD period of ~6 mirrors.
"""
import numpy as np
import os
import datetime
import dmd_patterns

# define DMD size
nx = 1920
ny = 1080

now = datetime.datetime.now()
save_dir = 'data/sim_patterns/%04d_%02d_%02d' % (now.year, now.month, now.day)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# define SIM patterns
data = dmd_patterns.export_all_pattern_sets([nx, ny], [6], nangles=3, nphases=3, avec_max_size=30, bvec_max_size=30,
                                            wavelengths=[465, 532, 635], invert=[False, True, False],
                                            ptol_relative=0.025, angle_sep_tol=5*np.pi/180, max_solutions_to_search=200,
                                            save_dir=save_dir, plot_results=True)