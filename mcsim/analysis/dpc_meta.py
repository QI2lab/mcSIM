"""
Metadata containers for DPC inverse solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DPCMeta:
    """
    Metadata required for DPC reconstruction.

    :param wavelength_um: Vacuum wavelength (um).
    :param n_background: Background refractive index.
    :param NA_obj: Objective NA (also used for detection NA).
    :param magnification: Objective magnification.
    :param camera_pixel_pitch_um: Camera pixel pitch (um).
    :param volume_shape_zyx: Reconstruction volume shape (nz, ny, nx).
    :param voxel_size_um_zyx: Voxel size (dz, dy, dx) in um.
    :param z_planes_um: Object-space z planes in um, length nz.
    :param led_grid_shape: LED board grid shape (ny, nx).
    :param inner_na: Inner NA for annular LED mask.
    :param include_center_led: Include NA=0 LED if True.
    :param pattern_order: Pattern order matching measured data.
    """

    wavelength_um: float
    n_background: float
    NA_obj: float
    magnification: float
    camera_pixel_pitch_um: float
    volume_shape_zyx: tuple[int, int, int]
    voxel_size_um_zyx: tuple[float, float, float]
    z_planes_um: Sequence[float]
    led_grid_shape: tuple[int, int] = (64, 64)
    inner_na: float = 0.0
    include_center_led: bool = False
    pattern_order: tuple[str, str, str, str] = ("left", "right", "up", "down")

    def __post_init__(self) -> None:
        z_planes = np.asarray(self.z_planes_um, dtype=float)
        object.__setattr__(self, "z_planes_um", z_planes)
        if z_planes.ndim != 1:
            raise ValueError("z_planes_um must be 1D")
        if len(self.volume_shape_zyx) != 3:
            raise ValueError("volume_shape_zyx must have length 3")
        if len(self.voxel_size_um_zyx) != 3:
            raise ValueError("voxel_size_um_zyx must have length 3")
        if z_planes.size != int(self.volume_shape_zyx[0]):
            raise ValueError("z_planes_um length must match volume_shape_zyx[0]")
        allowed = ("left", "right", "up", "down")
        if len(self.pattern_order) != 4 or set(self.pattern_order) != set(allowed):
            raise ValueError(f"pattern_order must be a permutation of {allowed}")

    @property
    def dxy_um(self) -> float:
        """
        Object-space pixel size derived from the camera pitch and magnification.

        :return: Pixel size in um.
        """
        return float(self.camera_pixel_pitch_um) / float(self.magnification)

    def as_dict(self) -> dict[str, object]:
        """
        Export metadata as a serializable dictionary.

        :return: Metadata dictionary.
        """
        z_planes = np.asarray(self.z_planes_um, dtype=float).tolist()
        return {
            "wavelength_um": float(self.wavelength_um),
            "n_background": float(self.n_background),
            "NA_obj": float(self.NA_obj),
            "magnification": float(self.magnification),
            "camera_pixel_pitch_um": float(self.camera_pixel_pitch_um),
            "volume_shape_zyx": tuple(int(v) for v in self.volume_shape_zyx),
            "voxel_size_um_zyx": tuple(float(v) for v in self.voxel_size_um_zyx),
            "z_planes_um": z_planes,
            "led_grid_shape": tuple(int(v) for v in self.led_grid_shape),
            "inner_na": float(self.inner_na),
            "include_center_led": bool(self.include_center_led),
            "pattern_order": tuple(self.pattern_order),
        }
