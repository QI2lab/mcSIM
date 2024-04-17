"""
Test running SIM reconstruction at full speed on GPU for multiple frames
"""

import time
import numpy as np
import cupy as cp
from pathlib import Path
import tifffile
from mcsim.analysis.sim_reconstruction import SimImageSet

# load images (images can be downloaded from Zenodo repository as described in README)
fname_data = Path(r"data") / "synthetic_microtubules.tiff"
imgs = tifffile.imread(fname_data)

# imgs = imgs[..., :1450, :1450]

# physical parameters
na = 1.3  # numerical aperture
dxy = 0.065  # um
wavelength = 0.488  # um

# GPU memory usage
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
memory_start = mempool.used_bytes()

# set limit to test
# mempool.set_limit(size=8 * 1000**3)
# mempool.set_limit(size=14 * 1000**3)

# ############################
# for the first image, estimate the SIM parameters
# this step is slow, can take ~1-2 minutes
# ############################
print("running initial reconstruction with full parameter estimation")

tstart_estimate_params = time.perf_counter()

imgset = SimImageSet.initialize({"pixel_size": dxy,
                                 "na": na,
                                 "wavelength": wavelength},
                                imgs,
                                otf=None,
                                wiener_parameter=0.3,
                                frq_estimation_mode="band-correlation",
                                # frq_guess=frqs_gt, # todo: can add frequency guesses for more reliable fitting
                                phase_estimation_mode="wicker-iterative",
                                phases_guess=np.array([[0, 2*np.pi / 3, 4 * np.pi / 3],
                                                       [0, 2*np.pi / 3, 4 * np.pi / 3],
                                                       [0, 2*np.pi / 3, 4 * np.pi / 3]]),
                                combine_bands_mode="fairSIM",
                                fmax_exclude_band0=0.4,
                                normalize_histograms=False,
                                background=100,
                                gain=2,
                                use_gpu=True)

# this included parameter estimation
imgset.reconstruct()
# extract estimated parameters
frqs = imgset.frqs
phases = imgset.phases - np.expand_dims(imgset.phase_corrections, axis=1)
mod_depths = imgset.mod_depths
otf = imgset.otf

print(f"estimating parameters took = {time.perf_counter() - tstart_estimate_params:.2f}s,"
      f" used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB, "
      f"memory pool = {mempool.total_bytes() / 1e9:.3f}GB")

a = imgset.sim_sr.compute()

print(f" used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB, "
      f"memory pool = {mempool.total_bytes() / 1e9:.3f}GB")

# clear GPU memory
imgset.delete()

mempool.free_all_blocks()

# ############################
# subsequent images, use the parameters estimated from the first one
# each iteration should run in < 2s for a 2048 x 2048 SIM image
# ############################
for ii in range(30):
    tstart_next = time.perf_counter()

    imgs_next = imgs + np.random.rand(*imgs.shape)

    imgset_next = SimImageSet.initialize({"pixel_size": dxy,
                                          "na": na,
                                          "wavelength": wavelength},
                                         imgs_next,
                                         otf=otf,
                                         wiener_parameter=0.3,
                                         frq_estimation_mode="fixed",
                                         frq_guess=frqs,
                                         phase_estimation_mode="fixed",
                                         phases_guess=phases,
                                         combine_bands_mode="fairSIM",
                                         mod_depths_guess=mod_depths,
                                         use_fixed_mod_depths=True,
                                         fmax_exclude_band0=0.4,
                                         normalize_histograms=False,
                                         background=100,
                                         gain=2,
                                         use_gpu=True,
                                         print_to_terminal=False)

    imgset_next.reconstruct()
    c = imgset_next.sim_sr.compute()
    imgset_next.delete()

    print(f"iteration {ii + 1},"
          f" process time={time.perf_counter() - tstart_next:.2f}s,"
          f" used GPU memory = {(mempool.used_bytes() - memory_start) / 1e9:.3f}GB, "
          f"memory pool = {mempool.total_bytes() / 1e9:.3f}GB")

    mempool.free_all_blocks()
