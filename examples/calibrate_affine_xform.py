"""
Example showing how affine calibration between DMD and image space is done
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
import mcsim.analysis.fit_dmd_affine as fit_dmd_affine
import mcsim.analysis.dmd_patterns as dmd
import mcsim.analysis.mm_io as mm_io

now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H;%M;%S")
channel_labels = ["blue", "red", "green"]

fname = "data/affine_calibration.tif"
# all colors are close enough can use same center guesses
centers_init = [[1039, 918], [982, 976], [1091, 979]]
indices_init = [[10, 16], [9, 16], [10, 15]] #[dmd short axis, dmd long axis]

roi_size = 25
options = {'cam_pix': 6.5e-6, 'dmd_pix': 7.56e-6,
           'dmd2cam_mag_expected': 180 / 300 * 400 / 200}
def sigma_pix(wl1, wl2, na, cam_mag):
    return np.sqrt(wl1**2 + wl2**2) / 2 / na / (2 * np.sqrt(2 * np.log(2))) / (options["cam_pix"] / cam_mag)

# load DMD pattern and dmd_centers
masks, radii, pattern_centers = dmd.get_affine_fit_pattern([1920, 1080])
mask = masks[1]


# read imges for each channel
for nc in range(len(channel_labels)):
        img, _ = mm_io.read_tiff(fname, slices=nc)
        img = img[0]
        affine_xform = fit_dmd_affine.estimate_xform(img, mask, pattern_centers, centers_init,
                                                     indices_init, options, roi_size=roi_size,
                                                     export_fname="%s_affine_xform_%s" % (now_str, channel_labels[nc]),
                                                     export_dir=None, chi_squared_relative_max=3)
plt.show()