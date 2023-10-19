"""
Collect z-stacks of many isolated beads in single-field of view

(1) fit center position of beads
(2) plot bead focus positions in z and compute focus gradient across field of view and field curvature
(3) extract average PSF
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import numpy as np
import zarr
import tifffile
from localize_psf import localize, fit_psf, fit

# data path
fname = Path(r"I:\2023_08_23\001_psf_blue\sim_odt.zarr")
component = "cam1/widefield_blue"

# physical parameters
wavelength = 0.515
na = 1.3
ni = 1.51

# set filtering
filter_sigma_small = None
filter_sigma_large = (5., 3., 3.) # in um
threshold = 2000 # only consider points above this value in max filter. Threshold is applied after filtering
min_spot_sep = (1.0, 0.2) # in um. Regard spots closer than this as duplicates

# set fit rejection thresholds
amp_bounds = (1000., 10000.) # (min, max)
sxy_bounds = (0.05, 0.5) # (min, max) um
sz_bounds = (0.2, 0.5) # (min, max) um
fit_distance_max_err = (2., 0.2) # (sz, sxy) um
fit_roi_size = (1.75, 0.65, 0.65) # (sz, sy, sx) in um

# ########################################
# localization
# ########################################

# load data
z = zarr.open(fname, "r")
# this must be a 3D image
img = np.array(z[component]).squeeze()

dz = z.attrs["dz_um"]
dxy = z.cam1.attrs["dx_um"]

# save directory
tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
save_dir = fname.parent / f"{tstamp:s}_psf_calibration"
save_dir.mkdir(exist_ok=True)


# coordinates
coords_3d = localize.get_coords(img.shape, (dz, dxy, dxy))
coords_2d = localize.get_coords(img.shape[1:], (dxy, dxy))

# simple filter
model = fit_psf.gaussian3d_psf_model()
filter = localize.get_param_filter(coords_3d,
                                   fit_dist_max_err=fit_distance_max_err,
                                   min_spot_sep=min_spot_sep,
                                   amp_bounds=amp_bounds,
                                   sigma_bounds=((sz_bounds[0], sxy_bounds[0]),
                                                 (sz_bounds[1], sxy_bounds[1]))
                                   )

# todo: replace with this
# filter = localize.get_param_filter_model(model,
#                                          fit_dist_max_err=(2., 0.2),
#                                          min_spot_sep=(1., 0.2),
#                                          param_bounds=
#                                          )

# localize
_, r, img_filtered = localize.localize_beads_generic(img,
                                                     (dz, dxy, dxy),
                                                     threshold=threshold,
                                                     roi_size=fit_roi_size,
                                                     filter_sigma_small=filter_sigma_small,
                                                     filter_sigma_large=filter_sigma_large,
                                                     min_spot_sep=min_spot_sep,
                                                     model=model,
                                                     filter=filter,
                                                     use_gpu_fit=True,
                                                     use_gpu_filter=False,
                                                     return_filtered_images=True,
                                                     fit_filtered_images=False,
                                                     verbose=True)

to_keep = r["to_keep"]
centers = r["fit_params"][to_keep][:, (3, 2, 1)]
backgrounds = r["fit_params"][to_keep][:, -1]

# optionally plot some ROIs
# todo

# ###############################
# fit gradient in z-positions
# ###############################
cz = centers[:, 0]
cy = centers[:, 1]
cx = centers[:, 2]

A = np.concatenate((np.ones((len(centers), 1)), cx[:, None], cy[:, None]), axis=1)
B = cz
lsq_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
zf_fit = lsq_params[0] + lsq_params[1] * cx + lsq_params[2] * cy

grad_mag = np.sqrt(lsq_params[1]**2 + lsq_params[2]**2)
grad_angle = np.arctan2(lsq_params[2], lsq_params[1])

# ###############################
# fit field curvature after removing gradient
# ###############################
def curvature_model_fn(p):
    return p[2] * ((p[0] - cx)**2 + (p[1] - cy)**2)
#
init_params = np.array([np.mean(cx), np.mean(cy), 0])
results = fit.fit_model(cz - zf_fit, curvature_model_fn, init_params)
fp_curve = results["fit_params"]

# ###############################
# plot gradient in z-positions
# ###############################
width_ratios = [1, 0.05, 0.05, 1, 0.05, 0.05, 1, 0.05, 0.05]
wspace = 0.2 / np.mean(width_ratios)

figh_focus = plt.figure(figsize=(30, 10))
figh_focus.suptitle("Focus gradient and field curvature")
grid = figh_focus.add_gridspec(nrows=1,
                               ncols=len(width_ratios),
                               width_ratios=width_ratios,
                               wspace=wspace)
axes = []
for ii in range(len(width_ratios)):
    axes.append(figh_focus.add_subplot(grid[0, ii]))


title = f"grad={grad_mag * 1e2:.3g}um/ 100um, " \
        f"angle={grad_angle * 180/np.pi:.1f}deg, " \
        f"offset={lsq_params[0]:.2f}um"

# raw z-position
z_color_limits = [[np.percentile(cz, 1), np.percentile(cz, 99)]]
localize.plot_bead_locations(np.max(img, axis=0),
                             centers,
                             coords=coords_2d,
                             title=title,
                             color_lists=["hsv"],
                             weights=cz,
                             cbar_labels=[r"$c_z$ (um)"],
                             color_limits=z_color_limits,
                             gamma=0.5,
                             axes=axes[:3]
                             )

zres_color_limits = [[np.percentile(cz - zf_fit, 1),
                     np.percentile(cz - zf_fit, 99)]]
localize.plot_bead_locations(np.max(img, axis=0),
                             centers,
                             coords=coords_2d,
                             title=f"field curvature (after gradient removed)\n"
                                   f"curvature={fp_curve[2]:.3g} 1/um, "
                                   f"(cx, cy) = ({fp_curve[0]:.1f} um, {fp_curve[1]:.1f} um)",
                             color_lists=["hsv"],
                             weights=cz - zf_fit,
                             cbar_labels=[r"$c_z$ (um)"],
                             color_limits=zres_color_limits,
                             gamma=0.5,
                             axes=axes[3:6]
                             )

localize.plot_bead_locations(np.max(img, axis=0),
                             centers,
                             coords=coords_2d,
                             title=f"residual shift after removing gradient and curvature",
                             color_lists=["hsv"],
                             weights=cz - zf_fit - curvature_model_fn(fp_curve),
                             cbar_labels=[r"$c_z$ (um)"],
                             gamma=0.5,
                             axes=axes[6:]
                             )


figh_focus.savefig(save_dir / "focus_gradient_and_field_curvature.png")

# ###############################
# average PSF
# ###############################
psf_roi_size = (11, 21, 21)
psf = fit_psf.average_exp_psfs(img,
                               coords_3d,
                               centers,
                               psf_roi_size,
                               backgrounds=backgrounds)

tifffile.imwrite(save_dir / "psf.tif",
                 tifffile.transpose_axes(psf.astype(np.float32), "ZYX", asaxes="TZCYXS"),
                 imagej=True,
                 resolution=(1/dxy, 1/dxy),
                 metadata={"Info": f"PSF {tstamp:s}, wavelength={wavelength:.03f}",
                           "unit": "um",
                           'min': 0,
                           'max': 1,
                           'spacing': dz}
                 )

plt.show()





