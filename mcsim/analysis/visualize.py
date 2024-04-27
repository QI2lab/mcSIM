"""
Tools for visualizing 3D volumes
"""
from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt
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
