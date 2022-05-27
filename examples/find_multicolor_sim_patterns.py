"""
Generate multicolor SIM binary patterns for three wavelengths and period of ~6 pixels
"""
import numpy as np
import datetime
from pathlib import Path
import mcsim.analysis.dmd_patterns as dmd_patterns

# create save directory
save_dir = Path("data", f"{datetime.datetime.now().strftime('%Y_%d_%m_%H;%M;%S'):s}_sim_patterns")
save_dir.mkdir(parents=True, exist_ok=True)

# define DMD size
nx = 1920
ny = 1080

# pattern periods
periods = [6]

# define SIM patterns
data = dmd_patterns.export_all_pattern_sets([nx, ny], periods, nangles=3, nphases=3,
                                            wavelengths=[465, 532, 635], # wavelengths to find patterns
                                            invert=[False, True, False], # which patterns are inverted
                                            avec_max_size=30, # parameters controlling stringency of search
                                            bvec_max_size=30,
                                            ptol_relative=0.025,
                                            angle_sep_tol=5*np.pi/180,
                                            max_solutions_to_search=200,
                                            save_dir=save_dir,
                                            plot_results=True)