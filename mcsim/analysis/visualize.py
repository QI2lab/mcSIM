"""
Tools for visualizing 3D volumes
"""
from typing import Union, Optional
from collections.abc import Sequence, Callable
import numpy as np
from functools import partial
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, writers
from localize_psf.affine import params2xform, xform_mat

try:
    import cupy as cp
except ImportError:
    cp = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


def assemble_2d_projections(p_xy: array,
                            p_xz: array,
                            p_yz: array,
                            z_to_xy_ratio: float = 1,
                            n_pix_sep: int = 5,
                            boundary_value: float = 0.,
                            ) -> array:
    """
    Assemble 2D projections into one image

    :param p_xy:
    :param p_xz:
    :param p_yz:
    :param z_to_xy_ratio: pixel size ratio dz/dxy
    :param n_pix_sep: number of blank pixels between projections
    :param boundary_value:
    :return img: 2D image showing projections
    """

    if cp and isinstance(p_xy, cp.ndarray):
        xp = cp
    else:
        xp = np

    ny, nx = p_xy.shape
    nz, _ = p_xz.shape

    ny_img = ny + n_pix_sep + int(np.ceil(nz * z_to_xy_ratio))
    nx_img = nx + n_pix_sep + int(np.ceil(nz * z_to_xy_ratio))

    # projected image
    img = xp.zeros((ny_img, nx_img))
    xx, yy = np.meshgrid(range(nx_img), range(ny_img))

    # xy slice (max-z)
    img[:ny, :nx] = p_xy

    # xz slice (max-y)
    xform_xz = params2xform([1, 0, 0, z_to_xy_ratio, 0, ny + n_pix_sep])
    n_xz = xform_mat(p_xz,
                     xform_xz,
                     (xx[ny + n_pix_sep:, :nx],
                     yy[ny + n_pix_sep:, :nx]))
    img[ny + n_pix_sep:, :nx] = n_xz

    # yz slice (max-x)
    xform_yz = params2xform([z_to_xy_ratio, 0, nx + n_pix_sep, 1, 0, 0])
    n_yz = xform_mat(p_yz.transpose(),
                            xform_yz,
                            (xx[:ny, nx + n_pix_sep:], yy[:ny, nx + n_pix_sep:]))
    img[:ny, nx + n_pix_sep:] = n_yz

    # remove NaNs
    img[xp.isnan(img)] = 0

    # set boundary
    img[ny:ny + n_pix_sep, :] = boundary_value
    img[:, nx:nx + n_pix_sep] = boundary_value

    return img


def get_2d_projections(n: array,
                       use_slice: bool = False,
                       **kwargs
                       ) -> array:
    """
    Generate an image showing 3 orthogonal projections from a 3D array.
    Additional keyword arguments are passed through to assemble_2d_projections()

    :param n: 3D array
    :param use_slice: use the central slice. If False, max project
    :return img: 2D image showing projections
    """
    nz, ny, nx = n.shape

    if use_slice:
        iz = nz // 2
        iy = ny // 2
        ix = nx // 2
        n_xy = n[iz]
        n_yz_before_xform = n[:, :, ix]
        n_xz_before_xform = n[:, iy, :]
    else:
        # max projection
        n_xy = n.max(axis=0)
        n_yz_before_xform = n.max(axis=2)
        n_xz_before_xform = n.max(axis=1)

    return assemble_2d_projections(n_xy, n_xz_before_xform, n_yz_before_xform, **kwargs)


def get_color_projection(n: np.ndarray,
                         contrast_limits=(0, 1),
                         mask: Optional[np.ndarray] = None,
                         cmap="turbo",
                         max_z: bool = False,
                         background_color: np.ndarray = np.array([0., 0., 0.])) -> (np.ndarray, np.ndarray):
    """
    Given a 3D refractive index distribution, take the max-z projection and color code the results
    by height. For each xy position, only consider the voxel along z with the maximum value.
    Display this in the final array in a color based on the height where that voxel was.

    :param n: refractive index array of size n0 x ... x nm x nz x ny x nx
    :param contrast_limits: (nmin, nmax)
    :param mask: only consider points where mask value is True
    :param cmap: matplotlib colormap
    :param max_z: whether to perform a max projection, or sum all slices
    :param background_color:
    :return: n_proj, colors
    """

    background_color = np.asarray(background_color)

    if mask is None:
        maxz_args = np.argmax(n, axis=-3)
    else:
        maxz_args = np.argmax(n * mask, axis=-3)

    nz, _, _ = n.shape[-3:]
    shape = list(n.shape + (3,))
    shape[-4] = 1

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, nz))

    n_proj = np.zeros(shape, dtype=float)
    for ii in range(nz):
        if max_z:
            to_use = maxz_args == ii
        else:
            to_use = np.ones(n[..., ii, :, :].shape, dtype=bool)

        intensity = np.clip((n[..., ii, :, :][to_use] - contrast_limits[0]) /
                            (contrast_limits[1] - contrast_limits[0]),
                            0, 1)

        n_proj[np.expand_dims(to_use, axis=-3), :] += np.expand_dims(intensity, axis=-1) * colors[ii, :3][None, :]

    # different background color
    is_bg = np.sum(n_proj, axis=-1) == 0
    n_proj[is_bg, :] = background_color

    return n_proj, colors


def export_mips_movie(mips: Sequence[np.ndarray],
                      dz: float,
                      dxy: float,
                      dt: float = 1.,
                      roi: Optional[Sequence[int, int, int, int, int, int]] = None,
                      video_length_s: float = 20.,
                      boundary_pix: int = 20,
                      lw: int = 5,
                      scale_bar_um: Optional[int] = None,
                      clims: Sequence[float, float] = (0.08, 0.019),
                      zlims: Sequence[float, float] = (0.0, 20.),
                      linked_centers: Optional = None,
                      trajectory_memory_frames: int = 100,
                      draw_calibration_bar: bool = True,
                      cbar_position: Sequence[float, float, float, float] = (0.05, 0.65, 0.025, 0.3),
                      out_fname: Optional[Union[str, Path]] = None,
                      fig=None,
                      ax=None,
                      **kwargs
                      ) -> Callable:
    """
    Display max projections along 3 orthogonal directions or export a time series of these as a movie

    :param mips: (maxz, maxy, maxx) where these are 3D arrays of sizes (nt, ny, nx), (nt, nz, nx), and (nt, nz, ny)
    :param dz:
    :param dxy:
    :param dt: time between images
    :param roi: [zstart, zend, ystart, yend, xstart, xend]. crop projections to this area before plotting.
    :param video_length_s: desired video length in seconds
    :param boundary_pix: pixels separating the different projections
    :param lw: rectangle boundary width
    :param scale_bar_um:
    :param clims: (cmin, cmax) color limits for displaying projections
    :param zlims: (zmin, zmax) color limits for displaying localizations
    :param linked_centers:
    :param trajectory_memory_frames:
    :param draw_calibration_bar: whether to draw the calibration bar
    :param cbar_position: (left, bottom, width, height) axes position on figure
    :param out_fname: full path ending in .mp4
    :param fig: figure to use
    :param ax: axis to use
    :param kwargs: passed through to writer
    :return animate_fn, fig, ax: A function which will draw the MIP's for a given frame on a figure and axis
    """

    plot_options = {"marker": 'o',
                    "s": np.pi * 7 ** 2,
                    "lw": 1.5,
                    "facecolors": "none"
                    }

    (n_maxz, n_maxy, n_maxx) = mips
    nt, ny_all, nx_all = n_maxz.shape
    _, nz_all, _ = n_maxy.shape

    if roi is not None:
        n_maxz = n_maxz[:, roi[2]:roi[3], roi[4]:roi[5]]
        n_maxy = n_maxy[:, roi[0]:roi[1], roi[4]:roi[5]]
        n_maxx = n_maxx[:, roi[0]:roi[1], roi[2]:roi[3]]

    # linked_centers = np.stack((linked['cz'].unstack().to_numpy(),
    #                            linked['cy'].unstack().to_numpy(),
    #                            linked['cx'].unstack().to_numpy()),
    #                           axis=-1)

    # ##################################
    # animate
    # ##################################
    def animate(frame,
                fig,
                ax,
                show_locs=False,
                show_tracks=False,
                set_text=True):
        print(f"animating frame = {frame + 1:.0f} / {nt:d}", end="\r")

        ax.clear()

        ax.set_xticks([])
        ax.set_yticks([])
        if set_text:
            time_text = ax.text(0.4,
                                0.95,
                                f"{frame * dt:.3f}s",
                                transform=ax.transAxes)

        projs = assemble_2d_projections(n_maxz[frame],
                                        n_maxy[frame],
                                        n_maxx[frame],
                                        dz / dxy,
                                        n_pix_sep=boundary_pix,
                                        boundary_value=np.nan)

        ny, nx = n_maxz.shape[1:]
        nz = n_maxy.shape[1]
        ny_proj, nx_proj = projs.shape

        im = ax.imshow(projs,
                       vmin=clims[0],
                       vmax=clims[1],
                       cmap="bone_r")

        # rectangles
        # note: linewidth is drawn symmetrically inside and outside of rectangle
        # note: linewidth set in pts
        patch = Rectangle((-0.5, -0.5),
                          width=nx,
                          height=ny,
                          facecolor="none",
                          edgecolor="gold",
                          lw=2*lw,
                          aa=True
                          )
        ax.add_artist(patch)
        patch.set_clip_path(patch)

        patch = Rectangle((-0.5, ny + boundary_pix - 0.5),
                          width=nx,
                          height=ny_proj - ny - boundary_pix,
                          facecolor="none",
                          edgecolor="purple",
                          lw=2 * lw,
                          aa=True
                          )
        ax.add_artist(patch)
        patch.set_clip_path(patch)

        patch = Rectangle((nx + boundary_pix - 0.5, -0.5),
                                width=nx_proj - nx - boundary_pix,
                                height=ny,
                                facecolor="none",
                                edgecolor="deepskyblue",
                                lw=2*lw,
                                aa=True
                                )
        ax.add_artist(patch)
        patch.set_clip_path(patch)

        # calibration bar
        if draw_calibration_bar:
            ax_cb = fig.add_axes(plt.axes(cbar_position, frameon=False))
            plt.colorbar(im, cax=ax_cb)
            ax_cb.set_yticklabels([])
            ax_cb.set_yticks([])

        # scale bar line
        if scale_bar_um is not None:
            line_middle = nx + (nx_proj - nx) / 2
            line_y = ny_proj - (ny_proj - ny) / 5
            ax.plot([line_middle - scale_bar_um / dxy / 2, line_middle + scale_bar_um / dxy / 2],
                    [line_y, line_y],
                    'k',
                    lw=10)

        if linked_centers is not None:
            fstart = max(0, frame - trajectory_memory_frames)
            cnow = linked_centers[fstart:frame + 1]

            to_plot = np.logical_and.reduce((cnow[..., 0] > roi[0] * dz,
                                             cnow[..., 0] < (roi[1] % nz_all) * dz,
                                             cnow[..., 1] > roi[2] * dxy,
                                             cnow[..., 1] < (roi[3] % ny_all) * dxy,
                                             cnow[..., 2] > roi[4] * dxy,
                                             cnow[..., 2] < (roi[5] % nx_all) * dxy
                                             )
                                            )

            cnow[np.logical_not(to_plot)] = np.nan

            z_cs = (plt.get_cmap("turbo")((cnow[..., 0] - zlims[0]) / (zlims[1] - zlims[0]))[..., :3])

            # tracks
            if show_tracks:
                tcolor = "mediumblue"
                talpha = 0.5

                # xy
                ax.plot((cnow[..., 2] - roi[4] * dxy) / dxy,
                        (cnow[..., 1] - roi[2] * dxy) / dxy,
                        ls="-",
                        color=tcolor,
                        alpha=talpha)
                # xz
                ax.plot((cnow[..., 2] - roi[4] * dxy) / dxy,
                        (cnow[..., 0] - roi[0] * dz) / dxy + ny + boundary_pix,
                        ls="-",
                        color=tcolor,
                        alpha=talpha)
                # yz
                ax.plot((cnow[..., 0] - roi[0] * dz) / dxy + nx + boundary_pix,
                        (cnow[..., 1] - roi[2] * dxy) / dxy,
                        ls="-",
                        color=tcolor,
                        alpha=talpha)

            if show_locs:
                ax.scatter((cnow[-1, ..., 2] - roi[4] * dxy) / dxy,
                           (cnow[-1, ..., 1] - roi[2] * dxy) / dxy,
                           edgecolors=z_cs[-1],
                           **plot_options)

                ax.scatter((cnow[-1, ..., 2] - roi[4] * dxy) / dxy,
                           (cnow[-1, ..., 0] - roi[0] * dz) / dxy + ny + boundary_pix,
                           edgecolors=z_cs[-1],
                           **plot_options)

                ax.scatter((cnow[-1, ..., 0] - roi[0] * dz) / dxy + nx + boundary_pix,
                           (cnow[-1, ..., 1] - roi[2] * dxy) / dxy,
                           edgecolors=z_cs[-1],
                           **plot_options)

        ax.axis('off')

    proj_sample = assemble_2d_projections(n_maxz[0],
                                          n_maxy[0],
                                          n_maxx[0],
                                          dz / dxy,
                                          n_pix_sep=boundary_pix,
                                          boundary_value=np.nan)

    # ############################
    # movie
    # ############################
    # save video results
    font = {'family': 'sans-serif',
            'size': 22}
    matplotlib.rc('font', **font)

    grid_settings = {"nrows": 1,
                     "ncols": 1,
                     "left": 0.01,
                     "right": 0.99,
                     "bottom": 0.01,
                     "top": 0.99
                     }

    if fig is None:
        fig = plt.figure(figsize=(9, 9 * proj_sample.shape[0] / proj_sample.shape[1]))

    if ax is None:
        grid = fig.add_gridspec(**grid_settings)
        ax = fig.add_subplot(grid[0, 0])

    if out_fname is not None:
        animation = FuncAnimation(fig,
                                  partial(animate,
                                          ax=ax,
                                          fig=fig,
                                          show_locs=True,
                                          show_tracks=True),
                                  frames=range(nt))

        Writer = writers['ffmpeg']
        animation.save(out_fname,
                       writer=Writer(fps=nt / video_length_s,
                                     metadata=dict(artist='qi2lab'),
                                     **kwargs
                                     )
                       )
        animation.event_source.stop()
        plt.close(fig)

    return animate, fig, ax

