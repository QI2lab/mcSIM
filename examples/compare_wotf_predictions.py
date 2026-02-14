#!/usr/bin/env python3
"""
Compare WOTF-predicted images for FISTA vs Tikhonov reconstructions.
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
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

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


def _report_z_sampling(meta: DPCMeta) -> tuple[float, float]:
    dzs = np.diff(np.asarray(meta.z_planes_um, dtype=float))
    dz_mean = float(np.mean(dzs)) if dzs.size else 0.0
    dz_std = float(np.std(dzs)) if dzs.size else 0.0
    dz_meta = float(meta.voxel_size_um_zyx[0])
    print(
        "Z sampling:",
        f"dz_meta={dz_meta:.6g} um",
        f"dz_mean={dz_mean:.6g} um",
        f"dz_std={dz_std:.3g} um",
        f"nz={meta.volume_shape_zyx[0]}",
    )
    if dzs.size and not np.isclose(dz_mean, dz_meta, rtol=1.0e-6, atol=1.0e-9):
        print("Warning: dz in meta does not match z_planes spacing.")
    return dz_mean, dz_meta


def _axial_transfer_summary(H: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(H), axis=(2, 3))


def _save_wotf_slice_plots(
    out_dir: Path,
    H_real: np.ndarray,
    H_imag: np.ndarray,
    *,
    suffix: str,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping WOTF plot images.")
        return

    n_patterns, nz, ny, nx = H_real.shape
    z_mid = nz // 2
    y_mid = ny // 2

    def _grid_plot(data: np.ndarray, title: str, out_name: str) -> None:
        log_data = np.log10(data + 1.0e-12)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        axes = axes.ravel()
        for pid in range(min(4, n_patterns)):
            axes[pid].imshow(log_data[pid], origin="lower", aspect="auto")
            axes[pid].set_title(f"pattern {pid}")
        for ax in axes[n_patterns:]:
            ax.axis("off")
        fig.suptitle(f"{title} (log10)")
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)

    h_real_kxy = np.abs(H_real[:, z_mid])
    h_imag_kxy = np.abs(H_imag[:, z_mid])
    h_real_kzx = np.abs(H_real[:, :, y_mid, :])
    h_imag_kzx = np.abs(H_imag[:, :, y_mid, :])

    _grid_plot(h_real_kxy, "H_real |kz mid| (kx-ky)", f"wotf_real_kxy_{suffix}.png")
    _grid_plot(h_imag_kxy, "H_imag |kz mid| (kx-ky)", f"wotf_imag_kxy_{suffix}.png")
    _grid_plot(h_real_kzx, "H_real |ky mid| (kz-kx)", f"wotf_real_kzx_{suffix}.png")
    _grid_plot(h_imag_kzx, "H_imag |ky mid| (kz-kx)", f"wotf_imag_kzx_{suffix}.png")


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
    parser.add_argument("--out", type=Path, default=Path("build/wotf_pred_compare.zarr"))
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
    parser.add_argument("--adjoint-check", action="store_true")
    parser.add_argument("--adjoint-crop-z", type=int, default=0)
    parser.add_argument("--adjoint-crop-yx", type=int, default=0)
    args = parser.parse_args()

    use_gpu = bool(args.use_gpu and cp is not None)
    print(f"Using GPU: {use_gpu}")

    ensure_dpc_mie_stack(args.sim_zarr, use_gpu=use_gpu)
    I_cam, attrs, z_planes_um = _load_simulated_stack(args.sim_zarr)
    meta = _build_meta(I_cam, attrs, z_planes_um)
    _report_z_sampling(meta)

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
    expected_shape = (
        meta.volume_shape_zyx[0] + 2 * int(args.pad_z),
        meta.volume_shape_zyx[1] + 2 * int(args.pad_yx),
        meta.volume_shape_zyx[2] + 2 * int(args.pad_yx),
    )
    if H_real.shape[1:] != expected_shape:
        print(f"Warning: H_real shape {H_real.shape} != expected {expected_shape}")

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

    if args.adjoint_check:
        nz, ny, nx = meta.volume_shape_zyx
        crop_z = int(args.adjoint_crop_z)
        crop_yx = int(args.adjoint_crop_yx)
        if crop_z > 0:
            crop_z = min(crop_z, nz)
        if crop_yx > 0:
            crop_yx = min(crop_yx, ny, nx)
        z0 = (nz - crop_z) // 2 if crop_z > 0 else 0
        y0 = (ny - crop_yx) // 2 if crop_yx > 0 else 0
        x0 = (nx - crop_yx) // 2 if crop_yx > 0 else 0
        z1 = z0 + (crop_z if crop_z > 0 else nz)
        y1 = y0 + (crop_yx if crop_yx > 0 else ny)
        x1 = x0 + (crop_yx if crop_yx > 0 else nx)
        v = xp.zeros((nz, ny, nx), dtype=xp.float32)
        g = xp.zeros((H_real.shape[0], nz, ny, nx), dtype=xp.float32)
        rng = np.random.default_rng(0)
        v_cpu = rng.standard_normal((z1 - z0, y1 - y0, x1 - x0)).astype(np.float32)
        g_cpu = rng.standard_normal((H_real.shape[0], z1 - z0, y1 - y0, x1 - x0)).astype(np.float32)
        v[z0:z1, y0:y1, x0:x1] = xp.asarray(v_cpu)
        g[:, z0:z1, y0:y1, x0:x1] = xp.asarray(g_cpu)
        pad_z = int(args.pad_z)
        pad_yx = int(args.pad_yx)
        if pad_z or pad_yx:
            v_pad = xp.pad(v, ((pad_z, pad_z), (pad_yx, pad_yx), (pad_yx, pad_yx)), mode="constant")
            g_pad = xp.pad(g, ((0, 0), (pad_z, pad_z), (pad_yx, pad_yx), (pad_yx, pad_yx)), mode="constant")
        else:
            v_pad = v
            g_pad = g
        v_ft = ft3(v_pad, axes=(0, 1, 2), shift=True)
        pred_ft = xp.asarray(H_real) * v_ft.real + xp.asarray(H_imag) * v_ft.imag
        contrast = ift3(pred_ft, axes=(1, 2, 3), shift=True).real
        if pad_z or pad_yx:
            contrast = contrast[:, pad_z : pad_z + nz, pad_yx : pad_yx + ny, pad_yx : pad_yx + nx]
        lhs = float(to_cpu(xp.sum(contrast * g)))
        g_ft = ift3(g_pad, axes=(1, 2, 3), shift=True, adjoint=True)
        grad_vft_real = xp.sum(xp.conj(xp.asarray(H_real)) * g_ft, axis=0)
        grad_vft_imag = xp.sum(xp.conj(xp.asarray(H_imag)) * g_ft, axis=0)
        grad_v = ft3(
            xp.real(grad_vft_real) + 1j * xp.real(grad_vft_imag),
            axes=(0, 1, 2),
            shift=True,
            adjoint=True,
        )
        if pad_z or pad_yx:
            grad_v = grad_v[pad_z : pad_z + nz, pad_yx : pad_yx + ny, pad_yx : pad_yx + nx]
        rhs = float(to_cpu(xp.sum(v * grad_v.real)))
        denom = max(1.0e-12, abs(lhs), abs(rhs))
        rel_err = abs(lhs - rhs) / denom
    n_init = (cp.asarray if use_gpu else np.asarray)(
        np.full(meta.volume_shape_zyx, meta.n_background, dtype=np.float32)
    )
    diag_optimizer = WOTFFISTAOptimizer(
        xp.asarray(I_meas),
        xp.asarray(I0_pred),
        xp.asarray(H_real),
        xp.asarray(H_imag),
        n0=meta.n_background,
        wavelength_um=meta.wavelength_um,
        eps=float(args.eps),
        tv_weight=0.0,
        tv_max_num_iter=0,
        tv_eps=float(args.tv_eps),
        tv_voxel_size_zyx=(params.dz_um, params.dxy_um, params.dxy_um),
        tv_weight_scale_zyx=(1.0, 1.0, 1.0),
        pad_zyx=(int(args.pad_z), int(args.pad_yx), int(args.pad_yx)),
        z_taper=int(args.z_taper),
        xy_taper=args.xy_taper,
        use_real_constraint=True,
    )
    cost_tikh = float(to_cpu(diag_optimizer.cost(n_init))[0])
    grad_tikh = diag_optimizer.gradient(n_init)[0]
    grad_norm = float(to_cpu(xp.sqrt(xp.sum(grad_tikh**2))))
    step_size = float(args.step)
    n_after_raw = n_init - step_size * grad_tikh
    n_after = diag_optimizer.prox(n_after_raw, step_size)
    cost_tikh_step = float(to_cpu(diag_optimizer.cost(n_after))[0])
    cost_tikh_step_raw = float(to_cpu(diag_optimizer.cost(n_after_raw))[0])
    dir_deriv = float(to_cpu(xp.sum(grad_tikh * grad_tikh)))
    n_min = float(to_cpu(xp.min(n_init)))
    n0 = float(meta.n_background)
    n_below = float(to_cpu(xp.mean(n_init < n0)))
    eps_candidates = [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]
    step_costs = []
    for eps_step in eps_candidates:
        n_pos = n_init - eps_step * grad_tikh
        n_neg = n_init + eps_step * grad_tikh
        cost_pos = float(to_cpu(diag_optimizer.cost(n_pos))[0])
        cost_neg = float(to_cpu(diag_optimizer.cost(n_neg))[0])
        step_costs.append((eps_step, cost_pos, cost_neg))
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
    I_pred_tikh = np.asarray(
        to_cpu(
            optimizer._forward_intensity(  # type: ignore[attr-defined]
                cp.asarray(n_tikh) if use_gpu else n_tikh
            )
        ),
        dtype=np.float32,
    )
    rmse_fista = np.sqrt(np.mean((I_pred_fista - I_meas) ** 2, axis=(1, 2, 3)))
    rmse_tikh = np.sqrt(np.mean((I_pred_tikh - I_meas) ** 2, axis=(1, 2, 3)))
    print("RMSE per pattern (FISTA):", rmse_fista)
    print("RMSE per pattern (Tikhonov):", rmse_tikh)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_wotf_slice_plots(out_path.parent, H_real, H_imag, suffix=f"{out_path.stem}_fista")
    _save_wotf_slice_plots(out_path.parent, H_real, H_imag, suffix=f"{out_path.stem}_tikh")
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
    g.attrs["diag_cost_tikh"] = float(cost_tikh)
    g.attrs["diag_cost_tikh_step"] = float(cost_tikh_step)
    g.attrs["diag_step_size"] = float(step_size)
    g.attrs["diag_grad_norm"] = float(grad_norm)
    g.attrs["diag_dir_deriv"] = float(dir_deriv)
    g.attrs["diag_step_costs"] = [(float(eps), float(cpos), float(cneg)) for eps, cpos, cneg in step_costs]
    g.create_array("I_meas", shape=I_meas.shape, dtype="float32")[...] = I_meas
    g.create_array("I0_pred", shape=I0_pred.shape, dtype="float32")[...] = I0_pred
    g.create_array("n_fista", shape=n_fista.shape, dtype="float32")[...] = n_fista
    g.create_array("n_tikh", shape=n_tikh.shape, dtype="float32")[...] = n_tikh
    g.create_array("rmse_fista", shape=rmse_fista.shape, dtype="float32")[...] = rmse_fista.astype(
        np.float32, copy=False
    )
    g.create_array("rmse_tikh", shape=rmse_tikh.shape, dtype="float32")[...] = rmse_tikh.astype(
        np.float32, copy=False
    )
    g.create_array("I_pred_fista", shape=I_pred_fista.shape, dtype="float32")[...] = I_pred_fista
    g.create_array("I_pred_tikh", shape=I_pred_tikh.shape, dtype="float32")[...] = I_pred_tikh
    h_real_abs = _axial_transfer_summary(H_real)
    h_imag_abs = _axial_transfer_summary(H_imag)
    g.create_array("H_real_abs_mean", shape=h_real_abs.shape, dtype="float32")[...] = h_real_abs.astype(
        np.float32, copy=False
    )
    g.create_array("H_imag_abs_mean", shape=h_imag_abs.shape, dtype="float32")[...] = h_imag_abs.astype(
        np.float32, copy=False
    )
    z_mid = int(meta.volume_shape_zyx[0]) // 2
    y_mid = int(meta.volume_shape_zyx[1]) // 2
    h_real_kxy = np.abs(H_real[:, z_mid])
    h_imag_kxy = np.abs(H_imag[:, z_mid])
    h_real_kzx = np.abs(H_real[:, :, y_mid, :])
    h_imag_kzx = np.abs(H_imag[:, :, y_mid, :])
    g.create_array("H_real_kxy_abs", shape=h_real_kxy.shape, dtype="float32")[...] = h_real_kxy.astype(
        np.float32, copy=False
    )
    g.create_array("H_imag_kxy_abs", shape=h_imag_kxy.shape, dtype="float32")[...] = h_imag_kxy.astype(
        np.float32, copy=False
    )
    g.create_array("H_real_kzx_abs", shape=h_real_kzx.shape, dtype="float32")[...] = h_real_kzx.astype(
        np.float32, copy=False
    )
    g.create_array("H_imag_kzx_abs", shape=h_imag_kzx.shape, dtype="float32")[...] = h_imag_kzx.astype(
        np.float32, copy=False
    )
    kz = np.fft.fftshift(np.fft.fftfreq(int(meta.volume_shape_zyx[0]), float(meta.voxel_size_um_zyx[0])))
    g.create_array("kz_cycles_per_um", shape=kz.shape, dtype="float32")[...] = kz.astype(np.float32, copy=False)
    g.create_array("led_na_xy", shape=led_na_xy.shape, dtype="float32")[...] = led_na_xy.astype(np.float32, copy=False)
    g.create_array("pattern_membership", shape=membership.shape, dtype="uint8")[...] = membership.astype(
        np.uint8, copy=False
    )
    print(f"Wrote outputs to {out_path}")


if __name__ == "__main__":
    main()
