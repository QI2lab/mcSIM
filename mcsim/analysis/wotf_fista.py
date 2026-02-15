"""
WOTF-based FISTA optimizer implemented without legacy inverse dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, Any

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from mcsim.analysis.fft import ft2, ift2, ft3, ift3, ftn
from mcsim.analysis.field_prop import get_v
from mcsim.analysis.optimize import Optimizer, to_cpu
from mcsim.analysis.tv_prox_fast import tv_prox_fast


if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


@dataclass(frozen=True)
class WOTFParams:
    wavelength_um: float
    na_obj: float
    na_in: float
    n0: float
    dxy_um: float
    dz_um: float
    nz: int
    ny: int
    nx: int
    pupil_taper_na: float = 0.0


def camera_to_photons(
    I_cam: array,
    *,
    camera_offset_adu: Union[float, array],
    camera_gain_photons_per_adu: Union[float, array],
) -> array:
    """
    Convert camera units to photons using a linear model.
    """
    xp = cp if cp and isinstance(I_cam, cp.ndarray) else np
    offset = xp.asarray(camera_offset_adu, dtype=xp.float32)
    gain = xp.asarray(camera_gain_photons_per_adu, dtype=xp.float32)
    I_phot = (I_cam - offset) * gain
    return xp.maximum(I_phot, xp.float32(0.0))


def _make_led_na_positions(
    ny_led: int,
    nx_led: int,
    *,
    na_obj: float,
    na_in: float,
    include_center: bool,
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(ny_led), np.arange(nx_led), indexing="ij")
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)
    cy = (ny_led - 1) / 2.0
    cx = (nx_led - 1) / 2.0
    x = xx - cx
    y = yy - cy
    r = np.sqrt(x * x + y * y)
    r_circle = float(min(cx, cy)) if (cx > 0 and cy > 0) else float(np.max(r))
    mask = r <= r_circle
    if not include_center:
        mask = mask & (r > 0)
    if not np.any(mask):
        raise ValueError("LED grid mask is empty; check LED geometry.")
    r_mask_max = float(np.max(r[mask]))
    scale = float(na_obj) / r_mask_max
    na_x = x[mask] * scale
    na_y = y[mask] * scale
    na_r = np.sqrt(na_x * na_x + na_y * na_y)
    if float(na_in) > 0:
        keep = na_r >= float(na_in)
    else:
        keep = np.ones_like(na_r, dtype=bool)
    if not include_center:
        keep &= na_r > 0
    na_x = na_x[keep]
    na_y = na_y[keep]
    na_xy = np.stack([na_x, na_y], axis=1)
    return na_xy.astype(np.float32, copy=False)


def build_led_na_grid(
    ny_led: int,
    nx_led: int,
    *,
    na_obj: float,
    na_in: float,
    include_center: bool,
    led_subsample: int = 1,
) -> np.ndarray:
    """
    Build an (N, 2) list of LED NA positions on the LED grid.

    led_subsample is the subsample factor for the LED grid.
    """
    if led_subsample < 1:
        raise ValueError("led_subsample must be >= 1.")
    if led_subsample == 1:
        return _make_led_na_positions(
            ny_led,
            nx_led,
            na_obj=na_obj,
            na_in=na_in,
            include_center=include_center,
        )
    cy = (ny_led - 1) / 2.0
    cx = (nx_led - 1) / 2.0
    step_y = int(max(1, np.floor(np.sqrt(float(led_subsample)))))
    step_x = int(max(1, np.ceil(float(led_subsample) / float(step_y))))

    yy_idx, xx_idx = np.meshgrid(
        np.arange(ny_led, dtype=int),
        np.arange(nx_led, dtype=int),
        indexing="ij",
    )
    yy = yy_idx.astype(np.float32)
    xx = xx_idx.astype(np.float32)
    x = xx - cx
    y = yy - cy
    r = np.sqrt(x * x + y * y)
    r_circle = float(min(cx, cy)) if (cx > 0 and cy > 0) else float(np.max(r))
    mask_circle = r <= r_circle
    if not include_center:
        mask_circle = mask_circle & (r > 0)
    if not np.any(mask_circle):
        raise ValueError("LED grid mask is empty; check LED geometry.")
    r_mask_max = float(np.max(r[mask_circle]))
    scale = float(na_obj) / r_mask_max

    best_key = None
    best_offsets = None
    for off_y in range(step_y):
        mask_y = (yy_idx - off_y) % step_y == 0
        for off_x in range(step_x):
            mask_sub = mask_y & ((xx_idx - off_x) % step_x == 0)
            mask = mask_circle & mask_sub
            if not np.any(mask):
                continue
            na_x = x[mask] * scale
            na_y = y[mask] * scale
            na_r = np.sqrt(na_x * na_x + na_y * na_y)
            if float(na_in) > 0:
                keep = na_r >= float(na_in)
            else:
                keep = np.ones_like(na_r, dtype=bool)
            if not include_center:
                keep &= na_r > 0
            if not np.any(keep):
                continue
            na_x = na_x[keep]
            na_y = na_y[keep]
            left = int(np.sum(na_x < 0))
            right = int(np.sum(na_x > 0))
            up = int(np.sum(na_y > 0))
            down = int(np.sum(na_y < 0))
            if include_center:
                center = int(np.sum((na_x == 0) & (na_y == 0)))
                left += center
                right += center
                up += center
                down += center
            min_count = min(left, right, up, down)
            total = int(na_x.size)
            key = (min_count, total)
            if best_key is None or key > best_key:
                best_key = key
                best_offsets = (off_y, off_x)

    if best_offsets is None:
        raise ValueError("Subsampling removed all LEDs from the board.")

    off_y, off_x = best_offsets
    mask_sub = ((yy_idx - off_y) % step_y == 0) & ((xx_idx - off_x) % step_x == 0)
    mask = mask_circle & mask_sub
    na_x = x[mask] * scale
    na_y = y[mask] * scale
    na_r = np.sqrt(na_x * na_x + na_y * na_y)
    if float(na_in) > 0:
        keep = na_r >= float(na_in)
    else:
        keep = np.ones_like(na_r, dtype=bool)
    if not include_center:
        keep &= na_r > 0
    na_x = na_x[keep]
    na_y = na_y[keep]
    na_xy = np.stack([na_x, na_y], axis=1)
    if na_xy.size == 0:
        raise ValueError("Subsampling removed all LEDs after inner NA filter.")
    return na_xy.astype(np.float32, copy=False)


def build_led_pattern_membership(
    led_na_xy: np.ndarray,
    *,
    order: Sequence[str],
    include_center: bool,
) -> np.ndarray:
    """
    Assign each LED to one of four patterns: left/right/up/down.
    """
    if len(order) != 4:
        raise ValueError("order must have 4 entries.")
    na = np.asarray(led_na_xy, dtype=float)
    membership = np.zeros((na.shape[0], 4), dtype=bool)
    for ii, name in enumerate(order):
        if name == "left":
            membership[:, ii] = na[:, 0] < 0
        elif name == "right":
            membership[:, ii] = na[:, 0] > 0
        elif name == "up":
            membership[:, ii] = na[:, 1] > 0
        elif name == "down":
            membership[:, ii] = na[:, 1] < 0
        else:
            raise ValueError(f"Unknown pattern name: {name}")
    if include_center:
        center = np.all(na == 0, axis=1)
        if np.any(center):
            membership[center] = True
    return membership


def _tukey_window(n: int, taper: int, xp: Any) -> Optional[array]:
    if taper <= 0 or n <= 0:
        return None
    taper = min(int(taper), n // 2)
    if taper <= 0:
        return None
    alpha = min(1.0, 2.0 * float(taper) / float(n))
    if xp is np:
        try:
            from scipy.signal.windows import tukey  # type: ignore

            return tukey(n, alpha=alpha).astype(np.float32, copy=False)
        except Exception:
            pass
    if cp and xp is cp:
        try:
            from cupyx.scipy.signal.windows import tukey  # type: ignore

            return tukey(n, alpha=alpha).astype(cp.float32, copy=False)
        except Exception:
            pass
    idx = xp.arange(n, dtype=xp.float32)
    w = xp.ones(n, dtype=xp.float32)
    edge = float(taper)
    left = idx < edge
    if xp.any(left):
        w[left] = 0.5 * (1.0 - xp.cos(np.pi * idx[left] / edge))
    right = idx >= (n - edge)
    if xp.any(right):
        tail = idx[right] - (n - edge)
        w[right] = 0.5 * (1.0 - xp.cos(np.pi * (edge - tail) / edge))
    return w



def _source_flip_unshifted(source: np.ndarray) -> np.ndarray:
    source_flip = np.fft.fftshift(source)
    source_flip = source_flip[::-1, ::-1]
    if source_flip.shape[0] % 2 == 0:
        source_flip = np.roll(source_flip, 1, axis=0)
    if source_flip.shape[1] % 2 == 0:
        source_flip = np.roll(source_flip, 1, axis=1)
    return np.fft.ifftshift(source_flip)


def _build_illumination_patterns_unshifted(
    led_na_xy: np.ndarray,
    led_pattern_membership: np.ndarray,
    *,
    ny: int,
    nx: int,
    dxy_um: float,
    wavelength_um: float,
    na_obj: float,
) -> np.ndarray:
    na_xy = np.asarray(led_na_xy, dtype=float)
    membership = np.asarray(led_pattern_membership, dtype=bool)
    fx = np.fft.fftfreq(int(nx), float(dxy_um))
    fy = np.fft.fftfreq(int(ny), float(dxy_um))
    fmax = float(na_obj) / float(wavelength_um)
    fx_led = na_xy[:, 0] / float(wavelength_um)
    fy_led = na_xy[:, 1] / float(wavelength_um)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="xy")
    mask = (fx_grid * fx_grid + fy_grid * fy_grid) <= (fmax * fmax + 1.0e-12)

    patterns = np.zeros((membership.shape[1], int(ny), int(nx)), dtype=float)
    for led_idx in range(na_xy.shape[0]):
        dist2 = (fx_grid - fx_led[led_idx]) ** 2 + (fy_grid - fy_led[led_idx]) ** 2
        dist2 = np.where(mask, dist2, np.inf)
        flat = int(np.argmin(dist2))
        iy, ix = np.unravel_index(flat, dist2.shape)
        for pid in range(membership.shape[1]):
            if membership[led_idx, pid]:
                patterns[pid, iy, ix] += 1.0
    return patterns


def build_wotf_transfer(
    led_na_xy: np.ndarray,
    led_pattern_membership: np.ndarray,
    params: WOTFParams,
    *,
    real_dtype: type[np.floating] = np.float32,
    complex_dtype: type[np.complexfloating] = np.complex64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build WOTF transfer functions for each illumination pattern.
    """
    fx = np.fft.fftfreq(int(params.nx), float(params.dxy_um)).astype(real_dtype, copy=False)
    fy = np.fft.fftfreq(int(params.ny), float(params.dxy_um)).astype(real_dtype, copy=False)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="xy")
    na_obj = float(params.na_obj)
    na_in = float(params.na_in)
    wavelength_um = float(params.wavelength_um)
    na_r = np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid) * wavelength_um
    pupil = (na_r <= na_obj).astype(real_dtype)
    if na_in != 0.0:
        pupil[na_r < na_in] = 0.0
    taper_na = max(float(params.pupil_taper_na), 0.0)
    if taper_na > 0.0:
        t_outer = np.clip((na_obj - na_r) / taper_na, 0.0, 1.0)
        outer_weight = 0.5 - 0.5 * np.cos(np.pi * t_outer)
        if na_in > 0.0:
            t_inner = np.clip((na_r - na_in) / taper_na, 0.0, 1.0)
            inner_weight = 0.5 - 0.5 * np.cos(np.pi * t_inner)
        else:
            inner_weight = 1.0
        pupil *= outer_weight * inner_weight

    term_defocus = (1.0 / wavelength_um) ** 2 - fx_grid * fx_grid - fy_grid * fy_grid
    term_oblique = (float(params.n0) / wavelength_um) ** 2 - fx_grid * fx_grid - fy_grid * fy_grid
    phase_defocus = pupil * (2.0 * np.pi) * np.sqrt(np.maximum(term_defocus, 0.0))
    oblique_factor = pupil / (4.0 * np.pi * np.sqrt(np.maximum(term_oblique, 1.0e-12)))

    z_lin = np.fft.ifftshift((np.arange(int(params.nz)) - int(params.nz) // 2) * float(params.dz_um)).astype(
        real_dtype, copy=False
    )
    prop_kernel = np.exp(1.0j * z_lin[None, None, :] * phase_defocus[:, :, None]).astype(
        complex_dtype, copy=False
    )
    window_z = np.fft.ifftshift(np.hamming(int(params.nz))).astype(real_dtype, copy=False)

    patterns = _build_illumination_patterns_unshifted(
        led_na_xy,
        led_pattern_membership,
        ny=int(params.ny),
        nx=int(params.nx),
        dxy_um=float(params.dxy_um),
        wavelength_um=float(params.wavelength_um),
        na_obj=float(params.na_obj),
    ).astype(real_dtype, copy=False)

    dfx = 1.0 / (float(params.nx) * float(params.dxy_um))
    dfy = 1.0 / (float(params.ny) * float(params.dxy_um))

    n_patterns = int(patterns.shape[0])
    H_real = np.zeros((n_patterns, int(params.nz), int(params.ny), int(params.nx)), dtype=complex_dtype)
    H_imag = np.zeros_like(H_real)

    for pid in range(n_patterns):
        source = patterns[pid]
        source_flip = _source_flip_unshifted(source)
        fsp = ft2(source_flip[:, :, None] * pupil[:, :, None] * prop_kernel, axes=(0, 1), shift=False)
        fpg = ft2(pupil[:, :, None] * prop_kernel * oblique_factor[:, :, None], axes=(0, 1), shift=False).conj()
        fsp_cfpg = fsp * fpg
        h_real = 2.0 * ift2(1.0j * fsp_cfpg.imag * dfx * dfy, axes=(0, 1), shift=False)
        h_imag = 2.0 * ift2(fsp_cfpg.real * dfx * dfy, axes=(0, 1), shift=False)
        h_real *= window_z[None, None, :]
        h_imag *= window_z[None, None, :]
        h_real = ftn(h_real, axes=(2,), shift=False).astype(complex_dtype, copy=False) * float(params.dz_um)
        h_imag = ftn(h_imag, axes=(2,), shift=False).astype(complex_dtype, copy=False) * float(params.dz_um)
        h_real = np.transpose(h_real, (2, 0, 1))
        h_imag = np.transpose(h_imag, (2, 0, 1))
        total_source = np.sum(source_flip * pupil * pupil.conj()) * dfx * dfy
        if total_source == 0:
            raise ValueError("Total source power is zero for a pattern.")
        H_real[pid] = h_real * (1.0j / total_source)
        H_imag[pid] = h_imag * (1.0 / total_source)

    H_real = np.fft.fftshift(H_real, axes=(1, 2, 3))
    H_imag = np.fft.fftshift(H_imag, axes=(1, 2, 3))
    return H_real, H_imag


class WOTFFISTAOptimizer(Optimizer):
    """
    FISTA optimizer for 3D WOTF pattern intensities (no DPC differencing).
    """

    def __init__(
        self,
        I_meas: array,
        I0_pred: array,
        H_real: array,
        H_imag: array,
        *,
        n0: float,
        wavelength_um: float,
        wotf_sign: float = -1.0,
        eps: float = 1.0e-8,
        tv_weight: float = 0.0,
        tv_max_num_iter: int = 50,
        tv_eps: float = 2.0e-4,
        tv_voxel_size_zyx: Optional[Sequence[float]] = None,
        tv_weight_scale_zyx: Optional[Sequence[float]] = None,
        pad_zyx: Optional[Sequence[int]] = None,
        z_taper: int = 0,
        xy_taper: Optional[int] = None,
        use_real_constraint: bool = True,
    ) -> None:
        super().__init__(n_samples=1, prox_parameters=None)
        self.n0 = float(n0)
        self.wavelength_um = float(wavelength_um)
        self.wotf_sign = float(wotf_sign)
        self.eps = float(eps)
        self.tv_weight = float(tv_weight)
        self.tv_max_num_iter = int(tv_max_num_iter)
        self.tv_eps = float(tv_eps)
        self.use_real_constraint = bool(use_real_constraint)
        if tv_voxel_size_zyx is None:
            self.tv_voxel_size_zyx = None
        else:
            if len(tv_voxel_size_zyx) != 3:
                raise ValueError("tv_voxel_size_zyx must be a 3-tuple (dz, dy, dx).")
            self.tv_voxel_size_zyx = tuple(float(v) for v in tv_voxel_size_zyx)
        if tv_weight_scale_zyx is None:
            self.tv_weight_scale_zyx = None
        else:
            if len(tv_weight_scale_zyx) != 3:
                raise ValueError("tv_weight_scale_zyx must be a 3-tuple (sz, sy, sx).")
            self.tv_weight_scale_zyx = tuple(float(v) for v in tv_weight_scale_zyx)
        self._I_meas_cpu = np.asarray(to_cpu(I_meas))
        self._I0_pred_cpu = np.asarray(to_cpu(I0_pred))
        self._H_real_cpu = np.asarray(to_cpu(H_real))
        self._H_imag_cpu = np.asarray(to_cpu(H_imag))
        self._shape_zyx = self._I_meas_cpu.shape[1:]
        self._z_taper = int(z_taper)
        self._xy_taper = int(z_taper if xy_taper is None else xy_taper)
        if pad_zyx is None:
            self.pad_zyx = (0, 0, 0)
        else:
            if len(pad_zyx) != 3:
                raise ValueError("pad_zyx must be a 3-tuple (pz, py, px).")
            self.pad_zyx = tuple(int(v) for v in pad_zyx)
            if any(v < 0 for v in self.pad_zyx):
                raise ValueError("pad_zyx entries must be >= 0.")
        if any(self.pad_zyx):
            nz, ny, nx = self._shape_zyx
            pz, py, px = self.pad_zyx
            expected = (nz + 2 * pz, ny + 2 * py, nx + 2 * px)
            if self._H_real_cpu.shape[1:] != expected:
                raise ValueError(
                    "H_real/H_imag must match padded shape "
                    f"{expected} when pad_zyx={self.pad_zyx}."
                )
        self._xp_cache = None

    def _pad_volume(self, xp: Any, vol: array) -> array:
        if not any(self.pad_zyx):
            return vol
        pz, py, px = self.pad_zyx
        return xp.pad(vol, ((pz, pz), (py, py), (px, px)), mode="constant")

    def _crop_volume(self, vol: array) -> array:
        if not any(self.pad_zyx):
            return vol
        pz, py, px = self.pad_zyx
        nz, ny, nx = self._shape_zyx
        return vol[pz : pz + nz, py : py + ny, px : px + nx]

    def _pad_contrast(self, xp: Any, contrast: array) -> array:
        if not any(self.pad_zyx):
            return contrast
        pz, py, px = self.pad_zyx
        return xp.pad(contrast, ((0, 0), (pz, pz), (py, py), (px, px)), mode="constant")

    def _crop_contrast(self, contrast: array) -> array:
        if not any(self.pad_zyx):
            return contrast
        pz, py, px = self.pad_zyx
        nz, ny, nx = self._shape_zyx
        return contrast[:, pz : pz + nz, py : py + ny, px : px + nx]

    def _ensure_backend(self, x: array) -> tuple:
        xp = cp if cp and isinstance(x, cp.ndarray) else np
        if self._xp_cache is None or self._xp_cache[0] is not xp:
            z_weight = None
            xy_weight = None
            if self._z_taper > 0:
                z_weight = _tukey_window(int(self._shape_zyx[0]), self._z_taper, xp)
                xy_weight = _tukey_window(int(self._shape_zyx[1]), self._xy_taper, xp)
            self._xp_cache = (
                xp,
                xp.asarray(self._I_meas_cpu),
                xp.asarray(self._I0_pred_cpu),
                xp.asarray(self._H_real_cpu),
                xp.asarray(self._H_imag_cpu),
                z_weight,
                xy_weight,
            )
        return self._xp_cache

    def _forward_contrast(self, n_now: array) -> array:
        xp, I_meas, I0_pred, H_real, H_imag, _, _ = self._ensure_backend(n_now)
        v_now = get_v(n_now, self.n0, self.wavelength_um)
        v_now = self._pad_volume(xp, v_now)
        v_ft = ft3(v_now, axes=(0, 1, 2), shift=True)
        pred_ft = H_real * v_ft.real + H_imag * v_ft.imag
        contrast = ift3(pred_ft, axes=(1, 2, 3), shift=True).real
        contrast = self._crop_contrast(contrast)
        return contrast

    def _forward_intensity(self, n_now: array) -> array:
        xp, I_meas, I0_pred, H_real, H_imag, _, _ = self._ensure_backend(n_now)
        contrast = self._forward_contrast(n_now)
        I_pred = I0_pred * (1.0 + self.wotf_sign * contrast)
        return xp.maximum(I_pred, xp.float32(0.0))

    def cost(self, x: array, inds: Optional[Sequence[int]] = None) -> array:
        xp, I_meas, I0_pred, H_real, H_imag, z_weight, xy_weight = self._ensure_backend(x)
        contrast_pred = self.wotf_sign * self._forward_contrast(x)
        contrast_meas = I_meas / xp.maximum(I0_pred, xp.float32(self.eps)) - xp.float32(1.0)
        residual = contrast_pred - contrast_meas
        if z_weight is not None:
            residual = residual * z_weight[None, :, None, None]
        if xy_weight is not None:
            residual = residual * xy_weight[None, None, :, None]
            residual = residual * xy_weight[None, None, None, :]
        cost = 0.5 * xp.sum(residual ** 2)
        return xp.asarray(cost)[None]

    def gradient(self, x: array, inds: Optional[Sequence[int]] = None) -> array:
        xp, I_meas, I0_pred, H_real, H_imag, z_weight, xy_weight = self._ensure_backend(x)
        contrast_pred = self.wotf_sign * self._forward_contrast(x)
        contrast_meas = I_meas / xp.maximum(I0_pred, xp.float32(self.eps)) - xp.float32(1.0)
        g_contrast = contrast_pred - contrast_meas
        if z_weight is not None:
            g_contrast = g_contrast * z_weight[None, :, None, None]
        if xy_weight is not None:
            g_contrast = g_contrast * xy_weight[None, None, :, None]
            g_contrast = g_contrast * xy_weight[None, None, None, :]
        g_contrast = self._pad_contrast(xp, g_contrast)
        g_contrast_ft = ift3(g_contrast, axes=(1, 2, 3), shift=True, adjoint=True)
        grad_vft_real = self.wotf_sign * xp.sum(xp.conj(H_real) * g_contrast_ft, axis=0)
        grad_vft_imag = self.wotf_sign * xp.sum(xp.conj(H_imag) * g_contrast_ft, axis=0)
        grad_v = ft3(
            xp.real(grad_vft_real) + 1j * xp.real(grad_vft_imag),
            axes=(0, 1, 2),
            shift=True,
            adjoint=True,
        )
        grad_v = self._crop_volume(grad_v)
        k2 = (2.0 * np.pi / float(self.wavelength_um)) ** 2
        grad = -2.0 * k2 * x * grad_v.real
        return grad[None, ...]

    def prox(self, x: array, step: float) -> array:
        xp = cp if cp and isinstance(x, cp.ndarray) else np
        n_out = x
        if self.use_real_constraint:
            n_out = xp.maximum(n_out, float(self.n0))
        if self.tv_weight != 0.0:
            if not (cp and isinstance(n_out, cp.ndarray)):
                raise ValueError("tv_prox_fast requires GPU/CuPy.")
            n_out = tv_prox_fast(
                cp.asarray(n_out, dtype=cp.float32),
                float(self.tv_weight),
                num_iter=int(self.tv_max_num_iter),
                eps=float(self.tv_eps),
                voxel_size_zyx=self.tv_voxel_size_zyx,
                weight_scale_zyx=self.tv_weight_scale_zyx,
            )
        return n_out
