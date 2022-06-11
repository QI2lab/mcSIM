"""
Example showing how to determine the affine calibration between the DMD and camera space
"""
import datetime
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mcsim.analysis.fit_dmd_affine as fit_dmd_affine
import mcsim.analysis.dmd_patterns as dmd
import mcsim.analysis.mm_io as mm_io

# ###########################
# set image data location
# ###########################
img_fname = Path("data", "affine_calibration.tif")
channel_labels = ["blue", "red", "green"]
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H;%M;%S")
save_dir = img_fname.parent / f"{time_stamp:s}_affine_calibration"

# ###########################
# set guesses for three "spots" from DMD pattern
# ###########################
# all colors are close enough can use same center guesses
centers_init = [[1039, 918], [982, 976], [1091, 979]]
indices_init = [[10, 16], [9, 16], [10, 15]] #[dmd short axis, dmd long axis]

# ###########################
# set other parameters for fitting
# ###########################

roi_size = 25
options = {'cam_pix': 6.5e-6,
           'dmd_pix': 7.56e-6,
           'dmd2cam_mag_expected': 180 / 300 * 400 / 200,
           'cam_mag': 100}

def sigma_pix(wl1, wl2, na, cam_mag):
    return np.sqrt(wl1**2 + wl2**2) / 2 / na / (2 * np.sqrt(2 * np.log(2))) / (options["cam_pix"] / cam_mag)

# load DMD pattern and dmd_centers
dmd_size = [1920, 1080]
masks, radii, pattern_centers = dmd.get_affine_fit_pattern(dmd_size)
mask = masks[1]

# ###########################
# perform affine calibration for each channel and plot/export results
# ###########################
affine_summary = {}
for nc in range(len(channel_labels)):
    img, _ = mm_io.read_tiff(img_fname, slices=nc)
    img = img[0]
    affine_xform_data, figh = fit_dmd_affine.estimate_xform(img, mask, pattern_centers, centers_init,
                                                            indices_init, options, roi_size=roi_size,
                                                            export_fname=f"affine_xform_{channel_labels[nc]:s}",
                                                            export_dir=save_dir, chi_squared_relative_max=3,
                                                            figsize=(20, 12))

    affine_summary[channel_labels[nc]] = affine_xform_data["affine_xform"]

# save summary results
affine_summary["transform_direction"] = "dmd to camera"
affine_summary["processing_time_stamp"] = time_stamp
affine_summary["data_time_stamp"] = ""
affine_summary["file_name"] = str(img_fname)

fname_summary = save_dir / "dmd_affine_transformations.json"
with open(fname_summary, "w") as f:
    json.dump(affine_summary, f, indent="\t")

plt.show()
