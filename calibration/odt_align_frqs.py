"""
Find reference frequency by looking at circle in frequency space
Useful for aligning ODT frequencies so that center DMD mirror corresponds to beam passing vertically through sample
"""
import time
import os
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
import tifffile
import mcsim.analysis.sim_reconstruction as sim
import mcsim.analysis.tomography as tm

no = 1.474
wavelength = 0.785 # um
k = 2*np.pi / wavelength
dxy = 6.5 / 50
na = 0.55
fmax_int = 1 / (0.5 * wavelength / na)


# fname = r"F:\2021_12_08\37_odt_align\37_odt_align_MMStack_Pos0.ome.tif"
fname = r"F:\2021_12_08\41_1um_beads_odt\41_1um_beads_odt_MMStack.ome.tif"
img = tifffile.imread(fname)[0]
if img.ndim == 2:
    img = np.expand_dims(img, axis=0)

img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))

nimgs, ny, nx = img.shape
fxs = fft.fftshift(fft.fftfreq(nx, dxy))
dfx = fxs[1] - fxs[0]
fys = fft.fftshift(fft.fftfreq(ny, dxy))
dfy = fys[1] - fys[0]
fxfx, fyfy = np.meshgrid(fxs, fys)
ff_perp = np.sqrt(fxfx**2 + fyfy**2)
extent_fxy = [fxs[0] - 0.5 * dfx, fxs[-1] + 0.5 * dfx, fys[-1] + 0.5 * dfy, fys[0] - 0.5 * dfy]

tstart = time.perf_counter()
hologram_frqs = np.zeros((nimgs, 2))
frq_refs = np.zeros((nimgs, 2))
fbeams = np.zeros((nimgs, 2))
for ii in range(nimgs):
    print("%d/%d, elapsed t = %0.2fs" % (ii + 1, nimgs, time.perf_counter() - tstart), end="\r")
    if ii == (nimgs - 1):
        print("")

    # ######################
    # hologram interference frequency
    # ######################
    # ensure only looking in one half of image
    guess_ind_1d = np.argmax(np.abs(img_ft[ii]) * (fyfy <= 0) * (ff_perp > fmax_int))
    guess_ind = np.unravel_index(guess_ind_1d, img_ft[ii].shape)

    frq_guess = np.array([fxfx[guess_ind], fyfy[guess_ind]])

    hologram_frqs[ii], mask, _ = sim.fit_modulation_frq(img_ft[ii], img_ft[ii], dxy, fmax=np.inf, frq_guess=frq_guess, max_frq_shift=50 * dfx)
    # sim.plot_correlation_fit(img_ft, img_ft, hologram_frq, dxy, frqs_guess=frq_guess)

    # ######################
    # determine reference frequency
    # ######################
    results_ref, circ_dbl_fn = tm.fit_ref_frq(img[ii], dxy, fmax_int, search_rad_fraction=1, filter_size=1)
    fp_ref = results_ref["fit_params"]
    frq_refs[ii] = fp_ref[:2]

    if np.linalg.norm(hologram_frqs[ii] + frq_refs[ii]) <= np.linalg.norm(hologram_frqs[ii] - frq_refs[ii]):
        frq_refs[ii] = -frq_refs[ii]

    # ######################
    # determine reference frequency
    # ######################
    fbeams[ii] = hologram_frqs[ii] - frq_refs[ii]
    fz_beam = tm.get_fz(fbeams[ii, 0], fbeams[ii, 1], no, wavelength)
    theta, phi = tm.frqs2angles(np.array([fbeams[ii, 0], fbeams[ii, 1], fz_beam]), no, wavelength)

    # #######################
    # plot ref freq results
    # #######################
    figh = plt.figure(figsize=(16, 8))
    plt.suptitle("Reference frequency determined from image %d. Radius / expected = %0.3f\n"
                 "ref freq = (%0.3f, %0.3f)\n"
                 "beam freq = (%0.3f, %0.3f, %0.3f), theta = %0.2fdeg, phi = %0.2fdeg" %
                 (ii, fp_ref[2] / (fmax_int / 2),
                  frq_refs[ii, 0], frq_refs[ii, 1],
                  fbeams[ii, 0], fbeams[ii, 1], fz_beam, theta * 180/np.pi, phi * 180/np.pi))

    ax = plt.subplot(1, 1, 1)
    ax.imshow(np.abs(img_ft[ii]), norm=PowerNorm(gamma=0.1), extent=extent_fxy)
    ax.plot(frq_refs[ii,0], frq_refs[ii, 1], 'kx')
    ax.plot(-frq_refs[ii, 0], -frq_refs[ii, 1], 'kx')
    ax.plot(hologram_frqs[ii, 0], hologram_frqs[ii, 1], 'rx')
    ax.plot(-hologram_frqs[ii, 0], -hologram_frqs[ii, 1], 'rx')
    ax.plot(fbeams[ii, 0], fbeams[ii, 1], 'mx')
    ax.plot(-fbeams[ii, 0], -fbeams[ii, 1], 'mx')
    ax.add_artist(Circle((frq_refs[ii, 0], frq_refs[ii, 1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
    ax.add_artist(Circle((-frq_refs[ii, 0], -frq_refs[ii, 1]), radius=fp_ref[2], facecolor="none", edgecolor='k'))
    ax.add_artist(Circle((fbeams[ii, 0], fbeams[ii, 1]), radius=fp_ref[2], facecolor="none", edgecolor='m'))
    ax.add_artist(Circle((-fbeams[ii, 0], -fbeams[ii, 1]), radius=fp_ref[2], facecolor="none", edgecolor='m'))
    ax.add_artist(Circle((0, 0), radius=fmax_int, facecolor="none", edgecolor="k"))
    ax.set_title("efield ft")

# make reference frequencies agree on sign
frq_ds = np.linalg.norm(frq_refs - np.expand_dims(frq_refs[0], axis=0), axis=1)
frq_ds_neg = np.linalg.norm(frq_refs + np.expand_dims(frq_refs[0], axis=0), axis=1)
swap_sign = frq_ds_neg < frq_ds
frq_refs[swap_sign] = -frq_refs[swap_sign]

ref_frq = np.mean(frq_refs, axis=0)

# make hologram frequencies all have signs which minimize distances
frq_dists_ref = np.linalg.norm(hologram_frqs - np.expand_dims(hologram_frqs[0], axis=0), axis=1)
frq_dists_neg_ref = np.linalg.norm(hologram_frqs + np.expand_dims(hologram_frqs[0], axis=0), axis=1)
swap_sign = frq_dists_neg_ref < frq_dists_ref

hologram_frqs[swap_sign] = -hologram_frqs[swap_sign]
