"""
Command line call to load pattern to DMD in "pattern on the fly" mode
"""
import numpy as np
import dlp6500
import sys
import os
import time
from PIL import Image

if __name__ == "__main__":
    pattern_dir = sys.argv[1]
    pattern_index = int(sys.argv[2])

    # files = glob.glob(os.path.join(pattern_dir, "*.png"))
    #
    fname = os.path.join(pattern_dir, "%d.png" % pattern_index)
    print("pattern %s" % fname)
    patterns = np.asarray(Image.open(fname))

    patterns = patterns.astype(np.uint8)

    dmd = dlp6500.dlp6500win(debug=False)
    exposure_t = 105
    dark_t = 0
    triggered = False

    # upload sequences
    tstart = time.perf_counter()
    img_inds, bit_inds = dmd.upload_pattern_sequence(patterns, exposure_t, dark_t, triggered=triggered,
                                                     num_repeats=0, compression_mode='erle')
    tend = time.perf_counter()
    print("uploaded images in %0.2fs" % (tend - tstart))

    # dmd.set_pattern_sequence(img_inds, bit_inds, exposure_t, dark_t, triggered=triggered,
    #                          clear_pattern_after_trigger=False, bit_depth=1, num_repeats=0, mode='on-the-fly')

    # dmd.start_stop_sequence("start")
