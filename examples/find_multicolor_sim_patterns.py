"""
Generate multicolor SIM binary patterns for three wavelengths and period of ~6 pixels
"""
import numpy as np
import datetime
from pathlib import Path
from mcsim.analysis import dmd_patterns

# create save directory
tstamp = datetime.datetime.now().strftime('%Y_%d_%m_%H;%M;%S')
save_dir = Path("data") / f"{tstamp:s}_sim_patterns"
save_dir.mkdir(parents=True, exist_ok=True)

# find SIM patterns
data = dmd_patterns.export_all_pattern_sets([1920, 1080],  # DMD size
                                            [6],  # approximate pattern period for first wavelength
                                            nangles=3,  # number of SIM pattern angles
                                            nphases=3,  # number of SIM pattern phases
                                            wavelengths=[465, 532, 635],  # wavelengths to find patterns
                                            invert=[False, True, False],  # which patterns are inverted
                                            avec_max_size=30,  # parameters controlling stringency of search
                                            bvec_max_size=30,
                                            ptol_relative=0.025,  # fractional pattern tolerance
                                            angle_sep_tol=5*np.pi/180,  # absolute angle tolerance
                                            max_solutions_to_search=200,
                                            save_dir=save_dir,
                                            plot_results=True
                                            )
