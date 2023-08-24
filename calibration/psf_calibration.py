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
import napari
from pathlib import Path
import numpy as np
import dask.array as da
import zarr
from localize_psf import localize, affine, fit_psf
import tifffile

# data path
fname = Path(r"I:\2023_08_23\001_psf_blue\sim_odt.zarr")
component = "cam1/widefield_blue"

# load data
z = zarr.open(fname, "r")
# this must be a 3D image
img = np.array(z[component]).squeeze()

dz = z.attrs["dz_um"]
dxy = z.cam1.attrs["dx_um"]
wavelength = 0.515
na = 1.3
ni = 1.51


# save directory
tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
save_dir = fname.parent / f"{tstamp:s}_psf_calibration"
save_dir.mkdir(exist_ok=True)




# simple filter
# local coords
coords_3d = localize.get_coords(img.shape, (dz, dxy, dxy))
coords_2d = localize.get_coords(img.shape[1:], (dxy, dxy))
filter = localize.get_param_filter(coords_3d,
                                   fit_dist_max_err=(2., 0.2),
                                   min_spot_sep=(1., 0.2),
                                   amp_bounds=(1000., 10000),
                                   sigma_bounds=((0.2, 0.05), (0.5, 0.5))
                                   )

# localize
model = fit_psf.gaussian3d_psf_model()
_, r, img_filtered = localize.localize_beads_generic(img,
                                                     (dz, dxy, dxy),
                                                     threshold=2000,
                                                     roi_size=(1.75, 0.65, 0.65),
                                                     filter_sigma_small=None,
                                                     filter_sigma_large=(5., 3., 3.),
                                                     min_spot_sep=(1., 0.2),
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
# plot gradient in z-positions
# ###############################
width_ratios = [1, 0.05, 0.05, 1, 0.05, 0.05]
wspace = 0.2 / np.mean(width_ratios)

figh_focus = plt.figure(figsize=(20, 10))
figh_focus.suptitle("Focus gradient and field curvature")
grid = figh_focus.add_gridspec(nrows=1,
                               ncols=6,
                               width_ratios=width_ratios,
                               wspace=wspace)
axes = []
for ii in range(len(width_ratios)):
    axes.append(figh_focus.add_subplot(grid[0, ii]))


title = f"grad={grad_mag * 1e2:.3g}um/ 100um, " \
        f"angle={grad_angle * 180/np.pi:.1f}deg, " \
        f"offset={lsq_params[0]:.2f}um"

# raw z-position
localize.plot_bead_locations(np.max(img, axis=0),
                             centers,
                             coords=coords_2d,
                             title=title,
                             color_lists=["hsv"],
                             weights=cz,
                             cbar_labels=[r"$c_z$ (um)"],
                             gamma=0.5,
                             axes=axes[:3]
                             )

localize.plot_bead_locations(np.max(img, axis=0),
                             centers,
                             coords=coords_2d,
                             title="field curvature",
                             color_lists=["hsv"],
                             weights=cz - zf_fit,
                             cbar_labels=[r"$c_z$ (um)"],
                             gamma=0.5,
                             axes=axes[3:]
                             )

figh_focus.savefig(save_dir / "focus_gradient_and_field_curvature.png")

# ###############################
# average PSF
# ###############################
psf_roi_size = (11, 21, 21)
psf, psf_coords, otf, otf_coords = fit_psf.average_exp_psfs(img,
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





