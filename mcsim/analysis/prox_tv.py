"""Parallel proximal operator for 3D total variation.

This code only works on 3D images and is optimized for use on the GPU using
cupy. You need to convert your array to a cupy array for use, e.g. using

`noisy_image_cp = cp.asarray(noisy_image)`

The only free parameter is `lambda_reg` as there is no iteration for this
algorithm.

Ulugbek S. Kamilov "A parallel proximal algorithm for anisotropic total
variation minimization."
IEEE Transactions on Image Processing 26.2 (2016): 539-548.

2024/11 - Shepherd. Initial commit.
"""

import numpy as np
from typing import Union
from collections.abc import Sequence

try:
    import cupy as cp
except ImportError:
    cp = None

# CPU/GPU arrays
if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray

DEBUG_PLOT = True


def shrinkage(y: array, tau: float, use_gpu: bool = True) -> array:
    """Shrinkage function.

    .. math::

        T(y; \\tau) = \\max(|y| - \\tau, 0) \\cdot \\frac{y}{|y|},

    where :math:`\\max` operates element-wise, and :math:`\\frac{y}{|y|}`
    represents the sign of :math:`y`.


    Parameters
    ----------
    y: array
        Array to shrink.
    tau: float
        Value to compare against.
    use_gpu : bool
        Use GPU. Default = True.

    Returns
    -------
    s: array
        Shrinkage operator applied to y
    """

    xp = cp if use_gpu and cp else np

    return xp.maximum(xp.abs(y) - tau, 0) * xp.sign(y)


def haar_wavelet_transform_3d(x: array, use_gpu: bool = True) -> Sequence[array]:
    """Compute the 3D Haar wavelet transform.

    Returns a tuple of (approximation, details along z, y, x, yz, xz, xy, xyz).

    Parameters
    ----------
    x: array
        3D image to decompose into haar wavelets.
    use_gpu : bool
        Use GPU. Default = True.

    Returns
    -------
    (avg, diff_z, diff_y, diff_x, diff_yz, diff_xz, diff_xy, diff_xyz): Sequence[array,...]
        Average and difference arrays of x.
    """

    xp = cp if use_gpu and cp else np

    scale = xp.sqrt(8)
    avg = (
        x[::2, ::2, ::2]
        + x[1::2, ::2, ::2]
        + x[::2, 1::2, ::2]
        + x[::2, ::2, 1::2]
        + x[1::2, 1::2, ::2]
        + x[1::2, ::2, 1::2]
        + x[::2, 1::2, 1::2]
        + x[1::2, 1::2, 1::2]
    ) / scale

    diff_z = (x[::2, ::2, ::2] - x[1::2, ::2, ::2]) / scale
    diff_y = (x[::2, ::2, ::2] - x[::2, 1::2, ::2]) / scale
    diff_x = (x[::2, ::2, ::2] - x[::2, ::2, 1::2]) / scale
    diff_yz = (x[::2, ::2, ::2] - x[1::2, 1::2, ::2]) / scale
    diff_xz = (x[::2, ::2, ::2] - x[1::2, ::2, 1::2]) / scale
    diff_xy = (x[::2, ::2, ::2] - x[::2, 1::2, 1::2]) / scale
    diff_xyz = (x[::2, ::2, ::2] - x[1::2, 1::2, 1::2]) / scale

    return avg, diff_z, diff_y, diff_x, diff_yz, diff_xz, diff_xy, diff_xyz


def inverse_haar_wavelet_transform_3d(
    avg: array,
    diff_z: array,
    diff_y: array,
    diff_x: array,
    diff_yz: array,
    diff_xz: array,
    diff_xy: array,
    diff_xyz: array,
    use_gpu: bool = True,
) -> array:
    """Compute the inverse 3D Haar wavelet transform.

    Reconstructs the original 3D signal from its wavelet coefficients.

    Parameters
    ----------
    avg: array
        Average term of Haar decomposition.
    diff_z: array
        difference in z term of Haar wavelet decomposition.
    diff_y: array
        difference in y term of Haar wavelet decomposition.
    diff_x: array
        difference in x term of Haar wavelet decomposition.
    diff_yz: array
        difference in yz term of Haar wavelet decomposition.
    diff_xz: array
        difference in xz term of Haar wavelet decomposition.
    diff_xy: array
        difference in xy term of Haar wavelet decomposition.
    diff_xyz: array
        difference in xyz term of Haar wavelet decomposition.
    use_gpu : bool
        Use GPU. Default = True.

    Returns
    -------
    x: array
        inverse Haar wavelet transform.
    """

    xp = cp if use_gpu and cp else np

    out_shape = (avg.shape[0] * 2, avg.shape[1] * 2, avg.shape[2] * 2)
    x = xp.zeros(out_shape, dtype=avg.dtype)

    x[::2, ::2, ::2] = (
        avg + diff_z + diff_y + diff_x + diff_yz + diff_xz + diff_xy + diff_xyz
    )
    x[1::2, ::2, ::2] = (
        avg - diff_z + diff_y + diff_x - diff_yz - diff_xz + diff_xy - diff_xyz
    )
    x[::2, 1::2, ::2] = (
        avg + diff_z - diff_y + diff_x - diff_yz + diff_xz - diff_xy - diff_xyz
    )
    x[1::2, 1::2, ::2] = (
        avg - diff_z - diff_y + diff_x + diff_yz - diff_xz - diff_xy + diff_xyz
    )
    x[::2, ::2, 1::2] = (
        avg + diff_z + diff_y - diff_x - diff_yz - diff_xz + diff_xy - diff_xyz
    )
    x[1::2, ::2, 1::2] = (
        avg - diff_z + diff_y - diff_x + diff_yz + diff_xz - diff_xy + diff_xyz
    )
    x[::2, 1::2, 1::2] = (
        avg + diff_z - diff_y - diff_x + diff_yz - diff_xz - diff_xy + diff_xyz
    )
    x[1::2, 1::2, 1::2] = (
        avg - diff_z - diff_y - diff_x - diff_yz + diff_xz + diff_xy - diff_xyz
    )

    return x


def parallel_proximal_tv(img: array, lambda_reg: float, use_gpu: bool = True) -> array:
    """Fast parallel proximal algorithm for anisotropic total variation.

    Parameters
    ----------
    img: array
        3D image to denoise.
    lambda_reg: float
        Regularization parameter.

    Returns
    -------
    x: array
        Denoised 3D image.
    """
    xp = cp if use_gpu and cp else np

    avg, diff_z, diff_y, diff_x, diff_yz, diff_xz, diff_xy, diff_xyz = (
        haar_wavelet_transform_3d(img)
    )

    # Shrinkage on the difference components
    diff_z = shrinkage(diff_z, xp.sqrt(2) * 8 * lambda_reg)
    diff_y = shrinkage(diff_y, xp.sqrt(2) * 8 * lambda_reg)
    diff_x = shrinkage(diff_x, xp.sqrt(2) * 8 * lambda_reg)
    diff_yz = shrinkage(diff_yz, xp.sqrt(2) * 8 * lambda_reg)
    diff_xz = shrinkage(diff_xz, xp.sqrt(2) * 8 * lambda_reg)
    diff_xy = shrinkage(diff_xy, xp.sqrt(2) * 8 * lambda_reg)
    diff_xyz = shrinkage(diff_xyz, xp.sqrt(2) * 8 * lambda_reg)

    # Inverse Haar transform
    x = inverse_haar_wavelet_transform_3d(
        avg, diff_z, diff_y, diff_x, diff_yz, diff_xz, diff_xy, diff_xyz
    )

    return x


# Example usage
if __name__ == "__main__":
    from skimage import data

    # generate noisy data
    image = np.squeeze(data.cells3d()[:, 0, :, :])
    image_max = np.max(image.astype(np.float32), axis=(0, 1, 2))
    image = image.astype(np.float32) / image_max
    noise_level = 0.2
    noisy_img = image + noise_level * np.random.rand(*image.shape)
    noisy_img[noisy_img < 0.0] = 0.0

    lambda_reg = 0.1
    max_iter = 1

    # Run simplified FPPA
    img_denoised = parallel_proximal_tv(cp.asarray(noisy_img), max_iter, lambda_reg)
    if DEBUG_PLOT:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(cp.asnumpy(img_denoised), name="fppa")
        viewer.add_image(noisy_img, name="noisy image")
        viewer.add_image(image, name="original image")
        napari.run()