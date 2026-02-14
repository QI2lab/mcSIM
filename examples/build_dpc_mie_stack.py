#!/usr/bin/env python3
"""
Generate the default DPC Mie simulation stack used by the example scripts.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

from simulate_dpc_microsphere import SphereSpec, simulate_dpc_images_sphere_to_zarr


@dataclass(frozen=True)
class SimParams:
    wavelength_um: float = 0.515
    na_obj: float = 0.8
    led_grid_shape: tuple[int, int] = (64, 64)
    pitch_mm: float = 2.5
    camera_pixel_um: float = 2.4
    magnification: float = 20.0
    camera_oversample: int = 1
    esize_camera: tuple[int, int] = (256, 256)
    z_plane_um: float = 0.0
    sphere: SphereSpec = SphereSpec(radius_um=1., n_sphere=1.59, n_medium=1.55)
    inner_na: float = 0.0
    include_center_led: bool = False
    pattern_order: tuple[str, str, str, str] = ("left", "right", "up", "down")
    normalize_by_led_count: bool = True
    led_subsample: int = 1
    mie_kwargs: dict | None = None
    camera_gains: float = 2.0
    camera_offsets: float = 100.0
    camera_readout_noise_sds: float = 3.0
    camera_photon_shot_noise: bool = True
    camera_saturation: float | None = None
    camera_image_is_integer: bool = True
    exposure_time_ms: float = 10.0
    illumination_photons_per_s_per_um2: float = 1e7
    focal_stack_planes: int = 31
    focal_stack_step_um: float = 0.325

    @property
    def z_planes_um(self) -> np.ndarray:
        return (
            np.arange(self.focal_stack_planes) - self.focal_stack_planes // 2
        ) * self.focal_stack_step_um


def ensure_dpc_mie_stack(zarr_path: Path, *, use_gpu: bool) -> None:
    """
    Ensure the default DPC Mie simulation stack exists at the provided path.
    """
    if zarr_path.exists():
        return
    params = SimParams()
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    simulate_dpc_images_sphere_to_zarr(
        zarr_path,
        wavelength_um=params.wavelength_um,
        na_obj=params.na_obj,
        led_grid_shape=params.led_grid_shape,
        pitch_mm=params.pitch_mm,
        camera_pixel_um=params.camera_pixel_um,
        magnification=params.magnification,
        camera_oversample=params.camera_oversample,
        esize_camera=params.esize_camera,
        z_plane_um=params.z_plane_um,
        sphere=params.sphere,
        inner_na=params.inner_na,
        include_center_led=params.include_center_led,
        pattern_order=params.pattern_order,
        normalize_by_led_count=params.normalize_by_led_count,
        led_subsample=params.led_subsample,
        use_gpu=use_gpu,
        mie_kwargs=params.mie_kwargs,
        camera_gains=params.camera_gains,
        camera_offsets=params.camera_offsets,
        camera_readout_noise_sds=params.camera_readout_noise_sds,
        camera_photon_shot_noise=params.camera_photon_shot_noise,
        camera_saturation=params.camera_saturation,
        camera_image_is_integer=params.camera_image_is_integer,
        exposure_time_ms=params.exposure_time_ms,
        illumination_photons_per_s_per_um2=params.illumination_photons_per_s_per_um2,
        focal_stack_planes=params.focal_stack_planes,
        focal_stack_step_um=params.focal_stack_step_um,
        overwrite=True,
        reuse_cache=True,
    )


def main() -> None:
    default_path = Path("build/dpc_mie_stack.zarr")
    use_gpu = False
    if cp is not None:
        try:
            use_gpu = bool(cp.is_available())
        except Exception:
            use_gpu = False
    ensure_dpc_mie_stack(default_path, use_gpu=use_gpu)
    print(f"Simulation written to {default_path}")


if __name__ == "__main__":
    main()
