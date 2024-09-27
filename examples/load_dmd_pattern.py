"""
Command line call to load pattern to DMD in "pattern on the fly" mode
"""
import numpy as np
import sys
import os
from PIL import Image
import mcsim.expt_ctrl.dlp6500 as dlp6500

if __name__ == "__main__":
    pattern_dir = sys.argv[1]
    pattern_index = int(sys.argv[2])

    fname = os.path.join(pattern_dir, f"{pattern_index:d}.png")
    print("pattern %s" % fname)
    patterns = np.asarray(Image.open(fname)).astype(np.uint8)

    dmd = dlp6500.dlp6500win(debug=False)
    dmd.upload_pattern_sequence(patterns,
                                exp_times=dmd.min_time_us,
                                triggered=False,
                                clear_pattern_after_trigger=False
                                )
