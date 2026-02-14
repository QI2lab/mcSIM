#!/usr/bin/env python3
"""
Run WOTF FISTA reconstruction on a Mie-simulated DPC stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr  # type: ignore

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

from mcsim.analysis.dpc_meta import DPCMeta
from mcsim.analysis.fft import ft3, ift3
from mcsim.analysis.field_prop import get_n
from mcsim.analysis.optimize import to_cpu
from mcsim.analysis.wotf_fista import (
    WOTFParams,
    WOTFFISTAOptimizer,
    build_led_na_grid,
    build_led_pattern_membership,
    build_wotf_transfer,
    camera_to_photons,
)
from build_dpc_mie_stack import ensure_dpc_mie_stack


def _load_simulated_stack(zarr_path: Path) -> tuple[np.ndarray, dict, np.ndarray]:
    g = zarr.open_group(str(zarr_path), mode="r")
    if "dpc" not in g:
        raise ValueError("Missing 'dpc' dataset in simulation output.")
    dpc = np.asarray(g["dpc"], dtype=np.float32)

    meta_group = g.get("meta")
    if meta_group is None or "focal_offsets_um" not in meta_group:
        raise ValueError("Missing 'meta/focal_offsets_um' in simulation output.")
    z_planes_um = np.asarray(meta_group["focal_offsets_um"], dtype=float)

    if dpc.ndim == 3:
        if dpc.shape[0] != 4:
            raise ValueError(f"Expected dpc shape (4, ny, nx), got {dpc.shape}")
        dpc = dpc[None, ...]
    elif dpc.ndim == 4:
        if dpc.shape[1] != 4:
            raise ValueError(f"Expected dpc shape (nz, 4, ny, nx), got {dpc.shape}")
    else:
        raise ValueError(f"Unexpected dpc ndim: {dpc.ndim}")

    I_cam = np.moveaxis(dpc, 1, 0)
    return I_cam, dict(g.attrs), z_planes_um


def _build_meta(I_cam: np.ndarray, attrs: dict, z_planes_um: np.ndarray) -> DPCMeta:
    nz = int(I_cam.shape[1])
    ny = int(I_cam.shape[2])
    nx = int(I_cam.shape[3])
    if z_planes_um.size != nz:
        raise ValueError("z_planes_um length must match stack depth.")

    wavelength_um = float(attrs["wavelength_um"])
    na_obj = float(attrs["na_obj"])
    camera_pixel_um = float(attrs["camera_pixel_um"])
    magnification = float(attrs["magnification"])
    n_medium = float(attrs.get("n_medium", 1.0))
    led_grid_shape = tuple(int(v) for v in attrs.get("led_grid_shape", (64, 64)))
    pattern_order = tuple(attrs.get("pattern_order", ("left", "right", "up", "down")))
    inner_na = float(attrs.get("inner_na", 0.0))
    include_center_led = bool(attrs.get("include_center_led", False))

    dz = 0.0
    if z_planes_um.size > 1:
        dzs = np.diff(z_planes_um)
        if not np.allclose(dzs, dzs[0]):
            raise ValueError("z_planes_um must be evenly spaced.")
        dz = float(dzs[0])

    dxy_um = camera_pixel_um / magnification

    return DPCMeta(
        wavelength_um=wavelength_um,
        n_background=n_medium,
        NA_obj=na_obj,
        magnification=magnification,
        camera_pixel_pitch_um=camera_pixel_um,
        volume_shape_zyx=(nz, ny, nx),
        voxel_size_um_zyx=(dz, dxy_um, dxy_um),
        z_planes_um=z_planes_um,
        led_grid_shape=led_grid_shape,
        inner_na=inner_na,
        include_center_led=include_center_led,
        pattern_order=pattern_order,  # type: ignore[arg-type]
    )


def _tikhonov_wotf(
    contrast: np.ndarray,
    H_real: np.ndarray,
    *,
    reg: float,
) -> np.ndarray:
    contrast_ft = ft3(contrast, axes=(1, 2, 3), shift=True)
    num = np.sum(np.conj(H_real) * contrast_ft, axis=0)
    den = np.sum(np.abs(H_real) ** 2, axis=0) + float(reg)
    v_ft_real = num / den
    v_real = ift3(v_ft_real, axes=(0, 1, 2), shift=True).real
    return v_real.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-zarr", type=Path, default=Path("build/dpc_mie_stack.zarr"))
    parser.add_argument("--out", type=Path, default=Path("build/wotf_fista_mie.zarr"))
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--step", type=float, default=5.0e-5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--tv-max-num-iter", type=int, default=50)
    parser.add_argument("--tv-eps", type=float, default=2.0e-4)
    parser.add_argument("--tv-aniso-z", type=float, default=1.0)
    parser.add_argument("--z-taper", type=int, default=8)
    parser.add_argument("--xy-taper", type=int, default=None)
    parser.add_argument("--pupil-taper-na", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--line-search-factor", type=float, default=0.5)
    parser.add_argument("--restart-line-search", action="store_true")
    parser.add_argument("--tikhonov-reg", type=float, default=5.0e-4)
    parser.add_argument("--led-subsample", type=int, default=1)
    parser.add_argument("--pad-z", type=int, default=64)
    parser.add_argument("--pad-yx", type=int, default=64)
    args = parser.parse_args()

    use_gpu = bool(args.use_gpu and cp is not None)
    print(f"Using GPU: {use_gpu}")

    ensure_dpc_mie_stack(args.sim_zarr, use_gpu=use_gpu)
    I_cam, attrs, z_planes_um = _load_simulated_stack(args.sim_zarr)
    meta = _build_meta(I_cam, attrs, z_planes_um)

    camera_gain = attrs.get("camera_gains", 1.0)
    camera_offset = attrs.get("camera_offsets", 0.0)
    I_cam_xp = cp.asarray(I_cam) if use_gpu else np.asarray(I_cam)
    I_meas_phot = camera_to_photons(
        I_cam_xp,
        camera_offset_adu=camera_offset,
        camera_gain_photons_per_adu=camera_gain,
    )

    I_meas = np.asarray(to_cpu(I_meas_phot), dtype=np.float32)
    I0_pred = np.mean(I_meas, axis=(2, 3), keepdims=True).astype(np.float32, copy=False)

    led_na_xy = build_led_na_grid(
        meta.led_grid_shape[0],
        meta.led_grid_shape[1],
        na_obj=meta.NA_obj,
        na_in=meta.inner_na,
        include_center=meta.include_center_led,
        led_subsample=int(args.led_subsample),
    )
    membership = build_led_pattern_membership(
        led_na_xy,
        order=meta.pattern_order,
        include_center=meta.include_center_led,
    )

    params = WOTFParams(
        wavelength_um=meta.wavelength_um,
        na_obj=meta.NA_obj,
        na_in=meta.inner_na,
        n0=meta.n_background,
        dxy_um=meta.dxy_um,
        dz_um=float(meta.voxel_size_um_zyx[0]),
        nz=meta.volume_shape_zyx[0] + 2 * int(args.pad_z),
        ny=meta.volume_shape_zyx[1] + 2 * int(args.pad_yx),
        nx=meta.volume_shape_zyx[2] + 2 * int(args.pad_yx),
        pupil_taper_na=float(args.pupil_taper_na),
    )
    H_real, H_imag = build_wotf_transfer(led_na_xy, membership, params)

    xp = cp if use_gpu else np
    n0_vol = xp.full(meta.volume_shape_zyx, float(meta.n_background), dtype=xp.float32)

    contrast = (I_meas / I0_pred - 1.0) / -1.0
    pad_z = int(args.pad_z)
    pad_yx = int(args.pad_yx)
    if pad_z or pad_yx:
        contrast = np.pad(
            contrast,
            ((0, 0), (pad_z, pad_z), (pad_yx, pad_yx), (pad_yx, pad_yx)),
            mode="constant",
        )
    v_real = _tikhonov_wotf(contrast, H_real, reg=float(args.tikhonov_reg))
    if pad_z or pad_yx:
        nz, ny, nx = meta.volume_shape_zyx
        v_real = v_real[pad_z : pad_z + nz, pad_yx : pad_yx + ny, pad_yx : pad_yx + nx]
    n_tikh = get_n(v_real, meta.n_background, meta.wavelength_um).real.astype(np.float32, copy=False)

    optimizer = WOTFFISTAOptimizer(
        xp.asarray(I_meas),
        xp.asarray(I0_pred),
        xp.asarray(H_real),
        xp.asarray(H_imag),
        n0=meta.n_background,
        wavelength_um=meta.wavelength_um,
        eps=float(args.eps),
        tv_weight=float(args.tv_weight),
        tv_max_num_iter=int(args.tv_max_num_iter),
        tv_eps=float(args.tv_eps),
        tv_voxel_size_zyx=(params.dz_um, params.dxy_um, params.dxy_um),
        tv_weight_scale_zyx=(float(args.tv_aniso_z), 1.0, 1.0),
        pad_zyx=(int(args.pad_z), int(args.pad_yx), int(args.pad_yx)),
        z_taper=int(args.z_taper),
        xy_taper=args.xy_taper,
        use_real_constraint=True,
    )

    n_init = (cp.asarray if use_gpu else np.asarray)(
        np.full(meta.volume_shape_zyx, meta.n_background, dtype=np.float32)
    )
    line_search_iter_limit = None if args.line_search else 0
    result = optimizer.run(
        n_init,
        step=float(args.step),
        max_iterations=int(args.iters),
        use_fista=True,
        n_batch=1,
        compute_batch_grad_parallel=True,
        verbose=True,
        compute_cost=True,
        line_search_iter_limit=line_search_iter_limit,
        line_search_factor=float(args.line_search_factor),
        restart_line_search=bool(args.restart_line_search),
    )
    n_fista = np.asarray(to_cpu(result["x"]), dtype=np.float32)

    I_pred_fista = np.asarray(
        to_cpu(
            optimizer._forward_intensity(  # type: ignore[attr-defined]
                cp.asarray(n_fista) if use_gpu else n_fista
            )
        ),
        dtype=np.float32,
    )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(out_path), mode="w", zarr_format=3)
    g.attrs.update(meta.as_dict())
    g.attrs["use_gpu"] = bool(use_gpu)
    g.attrs["iters"] = int(args.iters)
    g.attrs["step"] = float(args.step)
    g.attrs["tv_weight"] = float(args.tv_weight)
    g.attrs["z_taper"] = int(args.z_taper)
    g.attrs["xy_taper"] = None if args.xy_taper is None else int(args.xy_taper)
    g.attrs["pupil_taper_na"] = float(args.pupil_taper_na)
    g.attrs["tikhonov_reg"] = float(args.tikhonov_reg)
    g.attrs["init_from_tikhonov"] = True
    g.create_array("I_meas", shape=I_meas.shape, dtype="float32")[...] = I_meas
    g.create_array("I0_pred", shape=I0_pred.shape, dtype="float32")[...] = I0_pred
    g.create_array("n_fista", shape=n_fista.shape, dtype="float32")[...] = n_fista
    g.create_array("n_tikh", shape=n_tikh.shape, dtype="float32")[...] = n_tikh
    g.create_array("I_pred_fista", shape=I_pred_fista.shape, dtype="float32")[...] = I_pred_fista
    g.create_array("led_na_xy", shape=led_na_xy.shape, dtype="float32")[...] = led_na_xy.astype(np.float32, copy=False)
    g.create_array("pattern_membership", shape=membership.shape, dtype="uint8")[...] = membership.astype(
        np.uint8, copy=False
    )
    print(f"Wrote outputs to {out_path}")


if __name__ == "__main__":
    main()
