"""
Command line call to load pattern to DMD in "pattern on the fly" mode
"""
import numpy as np
import sys
import os
import time
from PIL import Image
import mcsim.expt_ctrl.dlp6500 as dlp6500

if __name__ == "__main__":
    pattern_dir = sys.argv[1]
    pattern_index = int(sys.argv[2])

    fname = os.path.join(pattern_dir, "%d.png" % pattern_index)
    print("pattern %s" % fname)
    patterns = np.asarray(Image.open(fname)).astype(np.uint8)

    dmd = dlp6500.dlp6500win(debug=False)

    # upload sequences
    tstart = time.perf_counter()
    img_inds, bit_inds = dmd.upload_pattern_sequence(patterns,
                                                     exp_times=105,
                                                     dark_times=0,
                                                     triggered=False,
                                                     num_repeats=0,
                                                     compression_mode='erle')
    print(f"uploaded images in {time.perf_counter() - tstart:.2f}s")
