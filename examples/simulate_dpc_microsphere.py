"""
Synthetic DPC image generation using Mie-theory scattered fields from a sphere,
with a simple synthetic imaging model (objective pupil + camera sampling).

Overview
--------
For each LED on a centered board, we:
1) map its board position to an incidence direction (NA_x, NA_y),
2) compute the complex vector electric field around/through a sphere using Mie theory
   (`mie_fields.mie_efield`),
3) apply a coherent imaging operator representing the objective/tube-lens relay
   (modeled as a circular pupil in spatial-frequency space with cutoff NA/lambda),
4) compute irradiance I(x,y) = sum_{c in {x,y,z}} |E_c(x,y)|^2,
5) optionally bin/average to camera pixels (camera integration / sampling),
6) incoherently sum irradiances for LEDs in left/right/up/down half-plane patterns.

Key implementation requirement
------------------------------
A 64 x 64 board has 4096 LEDs. This module simulates each active LED exactly once
(after circular/annular mask and optional subsampling), and accumulates that LED's
irradiance into any applicable DPC pattern sums (left/right/up/down). This avoids
redundant Mie computations across patterns.

Optical model included (requested)
----------------------------------
- Objective: circular pupil with coherent cutoff f_c = NA / lambda (cycles/µm).
- Tube lens: assumed ideal/infinity-corrected; its role is magnification.
- Camera: sampling and optional pixel integration via binning from an oversampled grid.

This is a first-order synthetic imaging model suitable for generating test data and
debugging inverse-model implementations.

Zarr output
-----------
Helpers are provided to write a Zarr dataset in the exact layout expected by the
unit tests in `test_dpc_idt_reconstruction.py`:
- root array: /dpc, shape (4, ny, nx), dtype float32
- root attrs: wavelength_um, na_obj, camera_pixel_um, magnification, n_medium,
  led_grid_shape, pattern_order, plus optional nz and z_span_um

Dependencies
------------
- mie_fields.py (user-provided): provides `mie_efield(...)`.

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import hashlib
import json
import os
from tqdm import trange

try:
    import cupy as cp  # Optional; mie_fields supports GPU mode
except ImportError:  # pragma: no cover
    cp = None

try:
    from localize_psf.camera import simulated_img
except ImportError:  # pragma: no cover
    simulated_img = None

from mcsim.analysis.mie_fields import mie_efield
from mcsim.analysis.fft import ft2, ift2
try:
    from mcsim.analysis.field_prop import propagate_homogeneous
except Exception as _e:  # pragma: no cover
    propagate_homogeneous = None


if cp:
    array = np.ndarray | cp.ndarray
else:
    array = np.ndarray


PatternName = Literal["left", "right", "up", "down"]


@dataclass(frozen=True, slots=True)
class SphereSpec:
    """
    Optical / geometric description of the sphere for Mie simulation.

    Parameters
    ----------
    radius_um : float
        Sphere radius (µm). Forwarded to `mie_fields.mie_efield` and used to pick
        a safe evaluation plane outside the sphere.
    n_sphere : complex or float
        Complex refractive index of the sphere (real part for phase, imaginary for
        absorption). Passed directly to `mie_fields.mie_efield`.
    n_medium : float
        Surrounding medium refractive index used by `mie_efield` and
        `na_to_incidence_angles`.
    """
    radius_um: float
    n_sphere: complex | float
    n_medium: float = 1.0


@dataclass(frozen=True, slots=True)
class SimulationSpace:
    """
    Grid used for field simulation (object space).

    Parameters
    ----------
    dxy_um : float
        Sampling pitch (µm) on the simulation grid passed to `mie_fields.mie_efield`
        and `field_prop.propagate_homogeneous`.
    esize : tuple[int, int]
        Grid shape (ny, nx) for the simulated field prior to camera binning.
    z_plane_um : float or None
        Axial distance from the sphere center for field evaluation before refocus.
        If None, a plane just outside the sphere is chosen for stability.
    """
    dxy_um: float
    esize: tuple[int, int]
    z_plane_um: float | None = None


@dataclass(frozen=True, slots=True)
class CameraSpace:
    """
    Description of the camera grid and noise model (image space).

    Parameters
    ----------
    pixel_um : float
        Camera pixel pitch (µm) in the camera plane; combined with `magnification`
        to compute the `bin_size` for `localize_psf.camera.simulated_img`.
    magnification : float
        Object-to-camera magnification; larger values increase the bin factor between
        simulation and camera grids.
    shape : tuple[int, int]
        Final camera image shape (ny, nx) returned by `simulated_img`.
    psf : array or None
        Optional PSF forwarded to `simulated_img` for blurring.
    apodization : array or int or float
        Apodization factor forwarded to `simulated_img` during PSF blurring (1 disables).
    gains : array or float
        Multiplicative conversion from photons to ADU (ADU/e) applied in `simulated_img`.
    offsets : array or float
        Additive camera offset (ADU) applied after gain and readout noise.
    readout_noise_sds : array or float
        Standard deviation (ADU) of Gaussian readout noise added in `simulated_img`.
    photon_shot_noise : bool
        If True, enable Poisson shot noise in `simulated_img`.
    saturation : int or None
        Clip simulated images above this value in `simulated_img`.
    image_is_integer : bool
        If True, round the final simulated image to integers in `simulated_img`.
    """
    pixel_um: float
    magnification: float
    shape: tuple[int, int]
    psf: array | None = None # type: ignore
    apodization: array | int | float = 1 # type: ignore
    gains: array | float = 1.0 # type: ignore
    offsets: array | float = 0.0 # type: ignore
    readout_noise_sds: array | float = 0.0 # type: ignore
    photon_shot_noise: bool = False
    saturation: int | None = None
    image_is_integer: bool = False

    @property
    def object_pixel_um(self) -> float:
        return float(self.pixel_um) / float(self.magnification)


def _get_xp(use_gpu: bool):
    """
    Select NumPy or CuPy module based on GPU usage.

    Parameters
    ----------
    use_gpu : bool
        If True and CuPy is available, return `cupy`; otherwise return `numpy`.

    Returns
    -------
    module
        Backend array module.
    """
    if use_gpu and (cp is not None):
        return cp
    return np


def _to_xp(x: array, *, use_gpu: bool) -> array: # type: ignore
    """
    Explicitly move arrays to the requested backend.

    Parameters
    ----------
    x : array
        Input array (NumPy, CuPy, or dask).
    use_gpu : bool
        If True, ensure the output is a CuPy array; otherwise ensure NumPy.

    Returns
    -------
    array
        Array on the requested backend, avoiding implicit host/device transfers.
    """
    if use_gpu:
        if cp is None:
            raise ImportError("use_gpu=True requested but CuPy is not available.")
        return cp.asarray(x)
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


def _nan_to_zero(x: array, *, use_gpu: bool) -> array: # type: ignore
    """
    Replace NaN/Inf with 0 on the requested backend.

    Parameters
    ----------
    x : array
        Input array possibly containing NaN/Inf.
    use_gpu : bool
        If True, operate with CuPy; otherwise NumPy.

    Returns
    -------
    array
        Cleaned array on the requested backend with NaN/Inf replaced by 0.
    """
    x = _to_xp(x, use_gpu=use_gpu)
    xp = cp if (use_gpu and cp is not None) else np
    # Both NumPy and CuPy implement nan_to_num (including for complex dtypes).
    return xp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _json_dumps_stable(obj) -> str:
    """
    Stable JSON dump for hashing parameters.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    str
        JSON string with deterministic key ordering.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _cache_key(params: dict) -> str:
    """
    Create a short stable cache key for a simulation run.

    Parameters
    ----------
    params : dict
        Parameter dictionary to hash.

    Returns
    -------
    str
        SHA1-based short key (first 16 hex chars).
    """
    import hashlib

    s = _json_dumps_stable(params).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:16]


def _cache_run_dir(cache_dir: str | Path, params: dict) -> Path:
    """
    Create (or reuse) the cache directory for a parameter set.

    Parameters
    ----------
    cache_dir : str or Path
        Base directory for simulation caches.
    params : dict
        Parameter dictionary hashed to form a unique run key.

    Returns
    -------
    Path
        Path to the run-specific cache directory containing metadata and Zarr stores.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(params)
    run_dir = cache_dir / key
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(_json_dumps_stable(params) + "\n")
    return run_dir


def _open_led_cache_zarr(
    run_dir: Path,
    *,
    n_led: int,
    n_planes: int,
    ny: int,
    nx: int,
    chunks: tuple[int, int, int, int],
):
    """
    Open (or create) a per-LED irradiance cache as a Zarr store.

    Parameters
    ----------
    run_dir : Path
        Directory in which to create the cache.
    n_led : int
        Number of LED entries to cache.
    n_planes : int
        Number of axial planes per LED (for focal stacks).
    ny, nx : int
        Camera image dimensions.
    chunks : tuple[int, int, int, int]
        Chunk shape for the cached array (n_led, n_planes, ny, nx).

    Returns
    -------
    group, I_arr, done
        Zarr group, cached camera irradiance array, and completion mask.

    Notes
    -----
    We cache the binned camera irradiance (float32) to reduce size and avoid recomputation.
    """
    try:
        import zarr  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("zarr is required for caching") from e

    store_path = run_dir / "led_cache.zarr"
    g = zarr.open_group(str(store_path), mode="a")

    g.attrs["n_led"] = int(n_led)
    g.attrs["n_planes"] = int(n_planes)
    g.attrs["ny"] = int(ny)
    g.attrs["nx"] = int(nx)

    if "I_cam" in g:
        I_arr = g["I_cam"]
        if tuple(I_arr.shape) != (int(n_led), int(n_planes), int(ny), int(nx)):
            raise ValueError(
                f"Existing LED cache shape {tuple(I_arr.shape)} != {(int(n_led), int(n_planes), int(ny), int(nx))}"
            )
    else:
        create_kwargs = {
            "shape": (int(n_led), int(n_planes), int(ny), int(nx)),
            "dtype": "float32",
        }
        # Use chunks kwarg for compatibility with current Zarr API
        I_arr = g.create_array("I_cam", chunks=chunks, **create_kwargs)  # type: ignore[call-arg]

    if "done" in g:
        done = g["done"]
        if tuple(done.shape) != (int(n_led),):
            raise ValueError(f"Existing LED cache done shape {tuple(done.shape)} != {(int(n_led),)}")
    else:
        try:
            done = g.create_array("done", shape=(int(n_led),), dtype="u1", overwrite=False)
        except (TypeError, AttributeError):
            done = g.create_array("done", shape=(int(n_led),), dtype="u1")
        done[...] = 0

    return g, I_arr, done


def _zarr_write_array(g, name: str, data: np.ndarray, *, dtype: str = "float32", chunks=None, overwrite: bool = True):
    """
    Write an array using Zarr v3 `Group.create_array`.

    Parameters
    ----------
    g : zarr.Group
        Target group.
    name : str
        Dataset name.
    data : np.ndarray
        Array to write (converted to dtype).
    dtype : str
        Target dtype for the stored array.
    chunks : tuple[int, ...] or None
        Chunk shape passed as `chunk_shape` to `create_array`. If None, let Zarr decide.
    overwrite : bool
        Included for API compatibility; Zarr v3 `create_array` overwrites only if allowed by mode.

    Returns
    -------
    zarr.Array
        Written array.
    """
    shape = tuple(int(s) for s in data.shape)
    create_kwargs = {
        "shape": shape,
        "dtype": dtype,
    }

    if chunks is not None:
        arr = g.create_array(name, chunks=chunks, **create_kwargs)  # type: ignore[call-arg]
    else:
        arr = g.create_array(name, **create_kwargs)
    arr[...] = data
    return arr

def sample_pixel_size_um(camera_pixel_um: float, magnification: float, *, oversample: int = 1) -> float:
    """
    Convert camera pixel size to object-plane pixel size, optionally oversampled.

    Parameters
    ----------
    camera_pixel_um : float
        Camera pixel pitch (micrometers). This matches `CameraSpace.pixel_um`.
    magnification : float
        System magnification (object→camera). Higher magnification reduces the
        effective object-plane sampling.
    oversample : int
        Oversampling factor. If >1, the object-plane sampling is divided by this
        factor (finer simulation grid), and images can be binned back to camera
        sampling using `bin2_average` or `simulated_img(bin_size=oversample)`.

    Returns
    -------
    dxy_um : float
        Object-plane pixel size (micrometers).
    """
    if magnification <= 0:
        raise ValueError("magnification must be > 0")
    if oversample < 1:
        raise ValueError("oversample must be >= 1")
    return float(camera_pixel_um) / float(magnification) / float(oversample)



def _default_z_plane_um(*, sphere_radius_um: float, dxy_um: float, wavelength_um: float) -> float:
    """
    Choose a stable observation plane for Mie evaluation.

    The Mie scattered-field expansion is unstable near r=0; this picks an exterior
    plane to evaluate fields.

    Parameters
    ----------
    sphere_radius_um : float
        Sphere radius (µm).
    dxy_um : float
        Object-plane sampling (µm).
    wavelength_um : float
        Vacuum wavelength (µm).

    Returns
    -------
    float
        Axial distance (µm) from sphere center to evaluation plane.
    """
    safety = float(max(float(dxy_um), 0.25 * float(wavelength_um)))
    return float(sphere_radius_um) + safety


def _safe_z_plane_um(
    z_plane_um: float,
    *,
    sphere_radius_um: float,
    dxy_um: float,
    wavelength_um: float,
) -> float:
    """Ensure the Mie evaluation plane lies outside the sphere.

    If |z_plane_um| is less than radius + safety_margin, we override it to
    sign(z_plane_um) * (radius + safety_margin). If z_plane_um==0, we choose +.

    This avoids evaluating the scattered-field expansion at/near r=0 and prevents
    overflow/NaN in irradiance formation.
    """
    safety = float(max(float(dxy_um), 0.01 * float(wavelength_um)))
    z_min = float(sphere_radius_um) + safety
    z = float(z_plane_um)
    if abs(z) < z_min:
        sgn = 1.0 if z == 0.0 else float(np.sign(z))
        return sgn * z_min
    return z



def make_led_na_positions(
    ny_led: int,
    nx_led: int,
    *,
    na_obj: float,
    inner_na: float = 0.0,
    include_center: bool = False,
) -> np.ndarray:
    """
    Generate LED positions in NA space for a centered rectangular board, masked to an inscribed circle.

    Assumptions
    ----------
    - Board is centered on the objective.
    - The outermost LED inside the circular mask maps to `na_obj`.
    - LEDs lie on a regular grid; positions are scaled to NA units.

    Parameters
    ----------
    ny_led, nx_led : int
        LED grid shape (64, 64 for the requested board).
    na_obj : float
        Objective NA; sets max LED radius in NA space. Passed later to
        `DPCMieSimulator` to form the pupil cutoff.
    inner_na : float
        Inner NA radius (>=0) to form an annulus. Default 0 (filled disk).
    include_center : bool
        If False, excludes the center LED (NA=0).

    Returns
    -------
    na_xy : np.ndarray
        array of shape (N, 2) with columns (NA_x, NA_y).
    """
    if ny_led < 1 or nx_led < 1:
        raise ValueError("ny_led and nx_led must be >= 1")
    if inner_na < 0 or inner_na >= na_obj:
        raise ValueError("inner_na must satisfy 0 <= inner_na < na_obj")

    cy = (ny_led - 1) / 2.0
    cx = (nx_led - 1) / 2.0

    yy, xx = np.meshgrid(
        np.arange(ny_led, dtype=np.float32),
        np.arange(nx_led, dtype=np.float32),
        indexing="ij",
    )
    x = xx - cx
    y = yy - cy
    r = np.sqrt(x * x + y * y)

    # Use inscribed circle to avoid corners.
    r_circle = float(min(cx, cy)) if (cx > 0 and cy > 0) else float(np.max(r))
    mask = r <= r_circle

    if not include_center:
        mask = mask & (r > 0)

    r_mask_max = float(np.max(r[mask])) if np.any(mask) else 1.0
    scale = float(na_obj) / r_mask_max

    na_x = x[mask] * scale
    na_y = y[mask] * scale
    na_r = np.sqrt(na_x * na_x + na_y * na_y)

    ann = na_r >= float(inner_na)
    na_xy = np.stack([na_x[ann], na_y[ann]], axis=1)
    return na_xy


def split_dpc_patterns(
    na_xy: np.ndarray,
    *,
    order: Sequence[PatternName] = ("left", "right", "up", "down"),
) -> dict[PatternName, np.ndarray]:
    """
    Split LED NA positions into canonical DPC half-circle patterns.

    Pattern membership is based on the sign of NA components:
    - left:  NA_x < 0
    - right: NA_x > 0
    - up:    NA_y > 0
    - down:  NA_y < 0

    Parameters
    ----------
    na_xy : np.ndarray
        LED positions in NA units, shape (N, 2).
    order : Sequence[PatternName]
        Must be a permutation of ("left","right","up","down"). Used for validation.

    Returns
    -------
    patterns : dict[PatternName, np.ndarray]
        Mapping pattern name -> LED NA positions in that pattern, each of shape (N_p, 2).
        These positions are later iterated in `DPCMieSimulator.simulate_patterns`
        when calling `mie_efield` for each LED.
    """
    allowed: tuple[PatternName, ...] = ("left", "right", "up", "down")
    if len(order) != 4 or set(order) != set(allowed):
        raise ValueError(f"order must be a permutation of {allowed}; got {tuple(order)}")

    na_x = na_xy[:, 0]
    na_y = na_xy[:, 1]

    masks = {
        "left": na_x < 0,
        "right": na_x > 0,
        "up": na_y > 0,
        "down": na_y < 0,
    }

    out: dict[PatternName, np.ndarray] = {}
    for name in allowed:
        pts = na_xy[masks[name]]
        if pts.shape[0] == 0:
            raise ValueError(f"pattern '{name}' has zero LEDs (check board geometry).")
        out[name] = pts

    return out


def na_to_incidence_angles(
    na_x: float,
    na_y: float,
    *,
    n_medium: float = 1.0,
) -> tuple[float, float]:
    """
    Convert (NA_x, NA_y) into incidence direction angles for `mie_efield`.

    We interpret NA components as:
        NA_x = n_medium * sin(theta) * cos(phi)
        NA_y = n_medium * sin(theta) * sin(phi)

    therefore:
        sin(theta) = sqrt(NA_x^2 + NA_y^2) / n_medium
        phi = atan2(NA_y, NA_x)

    Parameters
    ----------
    na_x, na_y : float
        NA components (dimensionless).
    n_medium : float
        Medium refractive index.

    Returns
    -------
    beam_theta, beam_phi : float
        Angles (radians) to pass as `beam_theta` and `beam_phi`.
        `beam_psi` can be set to 0 for x-polarized input in mie_fields.
        These are forwarded unmodified to `mie_fields.mie_efield`.
    """
    na_r = float(np.sqrt(na_x * na_x + na_y * na_y))
    s = na_r / float(n_medium)
    s = float(np.clip(s, 0.0, 1.0))
    theta = float(np.arcsin(s))
    phi = float(np.arctan2(na_y, na_x))
    return theta, phi


def _fftshift_freq_grid(n: int, d: float, xp) -> array: # type: ignore
    """
    Frequency samples (cycles/µm) aligned to fftshifted spectra.
    """
    f = xp.fft.fftfreq(n, d=d)
    return xp.fft.fftshift(f)


def make_pupil(
    ny: int,
    nx: int,
    *,
    dxy_um: float,
    wavelength_um: float,
    na_obj: float,
    xp,
) -> array: # type: ignore
    """
    Create a circular coherent pupil mask for the objective.

    Parameters
    ----------
    ny, nx : int
        Output mask dimensions.
    dxy_um : float
        Sampling pitch (µm) of the simulation grid.
    wavelength_um : float
        Vacuum wavelength (µm).
    na_obj : float
        Objective NA determining cutoff f_c = NA / wavelength.
    xp : module
        Backend array module (NumPy or CuPy).

    Returns
    -------
    array
        Pupil mask of shape (ny, nx) with ones inside the cutoff and zeros outside,
        used by `apply_objective_imaging`.
    """
    fx = _fftshift_freq_grid(nx, dxy_um, xp)  # (nx,)
    fy = _fftshift_freq_grid(ny, dxy_um, xp)  # (ny,)
    fxx, fyy = xp.meshgrid(fx, fy, indexing="xy")
    fr = xp.sqrt(fxx * fxx + fyy * fyy)

    fc = float(na_obj) / float(wavelength_um)
    return (fr <= fc).astype(xp.float32)


# -----------------------------------------------------------------------------#
# Coherent imaging helpers (pupil + optional apodization) with caching
# -----------------------------------------------------------------------------#

_APOD_CACHE: dict[tuple, array] = {} # type: ignore


def _tukey_1d(n: int, alpha: float, xp) -> array: # type: ignore
    """
    Create a 1D Tukey window on the requested backend.

    Parameters
    ----------
    n : int
        Number of samples (must be > 0).
    alpha : float
        Taper parameter in [0,1]; 0 yields a rectangular window, 1 yields Hann.
    xp : module
        Backend array module (NumPy or CuPy).

    Returns
    -------
    array
        Window of shape (n,) on the requested backend.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if alpha <= 0:
        return xp.ones((n,), dtype=xp.float32)
    if alpha >= 1:
        # Hann
        x = xp.arange(n, dtype=xp.float32)
        return (0.5 - 0.5 * xp.cos(2 * xp.pi * x / (n - 1))).astype(xp.float32)

    x = xp.linspace(0.0, 1.0, n, dtype=xp.float32)
    w = xp.ones((n,), dtype=xp.float32)
    edge = alpha / 2.0

    m1 = x < edge
    m2 = x > (1.0 - edge)
    # rising cosine
    w[m1] = 0.5 * (1.0 + xp.cos(xp.pi * ((2.0 * x[m1] / alpha) - 1.0)))
    # falling cosine
    w[m2] = 0.5 * (1.0 + xp.cos(xp.pi * ((2.0 * x[m2] / alpha) - (2.0 / alpha) + 1.0)))
    return w.astype(xp.float32)


def _get_cached_apodization(
    ny: int,
    nx: int,
    *,
    alpha: float,
    use_gpu: bool,
):
    """
    Retrieve or build a cached Tukey apodization window on the requested backend.

    Parameters
    ----------
    ny, nx : int
        Window dimensions.
    alpha : float
        Tukey alpha parameter passed to `_tukey_1d`.
    use_gpu : bool
        If True, cache CuPy arrays; otherwise NumPy.

    Returns
    -------
    array
        2D apodization window cached for reuse.
    """
    xp = cp if (use_gpu and cp is not None) else np
    key = (int(ny), int(nx), float(alpha), bool(use_gpu))
    apo = _APOD_CACHE.get(key)
    if apo is None:
        wy = _tukey_1d(int(ny), float(alpha), xp)
        wx = _tukey_1d(int(nx), float(alpha), xp)
        apo = (wy[:, None] * wx[None, :]).astype(xp.float32, copy=False)
        _APOD_CACHE[key] = apo
    return apo


def apply_objective_imaging(
    e_vec: array, # type: ignore
    *, 
    dxy_um: float,
    wavelength_um: float,
    na_obj: float,
    use_gpu: bool = False,
) -> array: # type: ignore
    """
    Apply a simple coherent imaging operator to a vector field.

    This models an ideal infinity-corrected objective + tube lens as a circular pupil
    (coherent transfer) in spatial-frequency space.
    `dxy_um`, `wavelength_um`, and `na_obj` jointly define the pupil generated by
    `make_pupil`, which is applied component-wise to `e_vec` before returning to
    the spatial domain. The resulting field is later converted to intensity and
    handed to `localize_psf.camera.simulated_img` for camera binning/noise.

    GPU/CPU behavior
    ----------------
    This function explicitly converts inputs to the requested backend to avoid
    implicit host<->device transfers.

    Parameters
    ----------
    e_vec : array
        Vector field, shape (3, ny, nx).
    dxy_um : float
        Sampling (µm) for e_vec grid.
    wavelength_um : float
        Vacuum wavelength (µm).
    na_obj : float
        Objective NA.
    use_gpu : bool
        If True, uses CuPy and runs FFTs on the GPU (requires CuPy).

    Returns
    -------
    e_img : array
        Filtered vector field, shape (3, ny, nx), on the requested backend.
    """
    if e_vec.shape[0] != 3:
        raise ValueError(f"e_vec must have shape (3, ny, nx), got {e_vec.shape}")

    # Ensure e_vec lives on the requested backend.
    e_vec = _to_xp(e_vec, use_gpu=use_gpu)
    xp = cp if (use_gpu and cp is not None) else np

    ny, nx = int(e_vec.shape[-2]), int(e_vec.shape[-1])
    P = make_pupil(
        ny,
        nx,
        dxy_um=float(dxy_um),
        wavelength_um=float(wavelength_um),
        na_obj=float(na_obj),
        xp=xp,
    )

    # Apply per-component in Fourier domain (centered FT).
    out = xp.empty_like(e_vec)
    for c in range(3):
        Ec = _nan_to_zero(e_vec[c], use_gpu=use_gpu)
        Ec_ft = ft2(Ec, axes=(-2, -1), shift=True, adjoint=False)
        Ec_ft = Ec_ft * P
        out[c] = ift2(Ec_ft, axes=(-2, -1), shift=True, adjoint=False)
    return out


def bin2_average(im: array, factor: int, *, use_gpu: bool = False) -> array: # type: ignore
    """
    Bin/average a 2D image by an integer factor.

    This approximates camera pixel integration when `factor` corresponds to an
    oversampling ratio (e.g., when simulation sampling is finer than camera pixels).
    It mirrors the binning performed inside `localize_psf.camera.simulated_img`
    when `bin_size` matches `factor`.

    Backend behavior
    ----------------
    This function explicitly converts `im` to the requested backend to avoid
    implicit host<->device transfers.

    Parameters
    ----------
    im : array
        2D image, shape (ny, nx).
    factor : int
        Binning factor (>=1).
    use_gpu : bool
        If True, operate on the GPU (requires CuPy).

    Returns
    -------
    binned : array
        Binned image, shape (ny//factor, nx//factor), on the requested backend.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return _to_xp(im, use_gpu=use_gpu)

    im = _to_xp(im, use_gpu=use_gpu)
    xp = cp if (use_gpu and cp is not None) else np

    ny, nx = int(im.shape[-2]), int(im.shape[-1])
    ny2 = (ny // factor) * factor
    nx2 = (nx // factor) * factor

    im2 = im[:ny2, :nx2]
    im2 = im2.reshape(ny2 // factor, factor, nx2 // factor, factor)
    return xp.mean(im2, axis=(1, 3))


class DPCMieSimulator:
    """
    End-to-end simulator that keeps a clear boundary between the simulation grid
    (Mie field generation + propagation) and the camera grid (pixel binning + noise).

    Parameters
    ----------
    wavelength_um : float
        Vacuum wavelength (µm) for field generation and pupil cutoff.
    na_obj : float
        Objective NA used in `apply_objective_imaging`.
    sphere : SphereSpec
        Sphere properties forwarded to `mie_fields.mie_efield`.
    simulation : SimulationSpace
        Simulation grid settings (sampling, size, z-plane).
    camera : CameraSpace
        Camera grid/noise settings forwarded to `localize_psf.camera.simulated_img`.
    led_grid_shape : tuple[int, int], optional
        LED board dimensions.
    inner_na : float, optional
        Inner NA for annular illumination mask.
    include_center_led : bool, optional
        Whether to include NA=0 LED.
    pattern_order : Sequence[str], optional
        Order of DPC patterns in outputs.
    normalize_by_led_count : bool, optional
        Normalize patterns by contributing LED count.
    led_subsample : int, optional
        Subsample factor for LEDs simulated.
    use_gpu : bool, optional
        If True, use CuPy-backed computations where available.
    mie_kwargs : dict or None, optional
        Extra kwargs passed to `mie_fields.mie_efield`.
    cache_dir : str or Path or None, optional
        Directory for Zarr per-LED camera irradiance cache.
    reuse_cache : bool, optional
        If True, reuse cached per-LED images when present.
    """

    def __init__(
        self,
        *,
        wavelength_um: float,
        na_obj: float,
        sphere: SphereSpec,
        simulation: SimulationSpace,
        camera: CameraSpace,
        led_grid_shape: tuple[int, int] = (64, 64),
        inner_na: float = 0.0,
        include_center_led: bool = False,
        pattern_order: Sequence[PatternName] = ("left", "right", "up", "down"),
        normalize_by_led_count: bool = True,
        led_subsample: int = 1,
        use_gpu: bool = False,
        mie_kwargs: dict | None = None,
        cache_dir: str | Path | None = None,
        reuse_cache: bool = True,
        exposure_time_ms: float = 1.0,
        illumination_photons_per_s_per_um2: float = 1.0,
        focal_stack_planes: int | None = None,
        focal_stack_step_um: float = 0.5,
    ):
        if simulated_img is None:
            raise ImportError("localize_psf.camera.simulated_img is required but not importable.")
        if propagate_homogeneous is None:
            raise ImportError("mcsim.analysis.field_prop.propagate_homogeneous is required but not importable.")

        allowed: tuple[PatternName, ...] = ("left", "right", "up", "down")
        if len(pattern_order) != 4 or set(pattern_order) != set(allowed):
            raise ValueError(f"pattern_order must be a permutation of {allowed}; got {tuple(pattern_order)}")
        if led_subsample < 1:
            raise ValueError("led_subsample must be >= 1")

        self.wavelength_um = float(wavelength_um)
        self.na_obj = float(na_obj)
        self.sphere = sphere
        self.simulation = simulation
        self.camera = camera
        self.led_grid_shape = (int(led_grid_shape[0]), int(led_grid_shape[1]))
        self.inner_na = float(inner_na)
        self.include_center_led = bool(include_center_led)
        self.pattern_order = pattern_order
        self.normalize_by_led_count = bool(normalize_by_led_count)
        self.led_subsample = int(led_subsample)
        self.use_gpu = bool(use_gpu)
        self.mie_kwargs = mie_kwargs or {}
        self.cache_dir = cache_dir
        self.reuse_cache = bool(reuse_cache)
        self.exposure_time_s = float(exposure_time_ms) / 1000.0
        self.illumination_photons_per_s_per_um2 = float(illumination_photons_per_s_per_um2)
        if focal_stack_planes is None or focal_stack_planes <= 1:
            self.focal_offsets_um = np.array([0.0], dtype=float)
        else:
            n = int(focal_stack_planes)
            offsets = (np.arange(n) - (n - 1) / 2.0) * float(focal_stack_step_um)
            self.focal_offsets_um = offsets.astype(float)

        self.xp = _get_xp(use_gpu)
        self.camera_bin_factor = self._camera_bin_factor()

    def _camera_bin_factor(self) -> int:
        """
        Integer ratio between camera sampling (object plane) and simulation sampling.

        Returns
        -------
        int
            Bin factor (`camera.object_pixel_um / simulation.dxy_um`) used as
            `bin_size` for `localize_psf.camera.simulated_img`.
        """
        ratio = float(self.camera.object_pixel_um) / float(self.simulation.dxy_um)
        ratio_round = int(round(ratio))
        if ratio_round < 1 or not np.isclose(ratio, ratio_round, rtol=1e-3, atol=1e-6):
            raise ValueError(
                f"Camera pixel size / simulation sampling must be a positive integer; got ratio={ratio:.4f}"
            )
        return ratio_round

    def _field_plane_z(self) -> float:
        """
        Choose a stable evaluation plane for the Mie field.

        This is the `dz` plane passed to `mie_fields.mie_efield`. If the user provides
        `simulation.z_plane_um`, it is validated with `_safe_z_plane_um`; otherwise a
        default just outside the sphere surface is selected.
        """
        if self.simulation.z_plane_um is not None:
            return float(_safe_z_plane_um(
                self.simulation.z_plane_um,
                sphere_radius_um=float(self.sphere.radius_um),
                dxy_um=float(self.simulation.dxy_um),
                wavelength_um=self.wavelength_um,
            ))
        return _safe_z_plane_um(
            _default_z_plane_um(
                sphere_radius_um=float(self.sphere.radius_um),
                dxy_um=float(self.simulation.dxy_um),
                wavelength_um=self.wavelength_um,
            ),
            sphere_radius_um=float(self.sphere.radius_um),
            dxy_um=float(self.simulation.dxy_um),
            wavelength_um=self.wavelength_um,
        )

    def _simulate_led_ground_truth(self, na_x: float, na_y: float) -> array: # type: ignore
        """
        Simulate pupil-filtered field on the simulation grid for a single LED (before camera).

        Parameters
        ----------
        na_x, na_y : float
            Illumination NA components converted to angles for `mie_efield`.

        Returns
        -------
        array
            Complex field on the simulation grid after refocus and pupil filtering.

        Notes
        -----
        Uses `mie_fields.mie_efield` with `simulation.dxy_um`, `simulation.esize`,
        and `_field_plane_z()`, refocuses via `propagate_homogeneous`, and applies
        `apply_objective_imaging`.
        """
        xp = self.xp
        theta, phi = na_to_incidence_angles(
            float(na_x),
            float(na_y),
            n_medium=float(self.sphere.n_medium),
        )

        z_plane = self._field_plane_z()
        ny_sim, nx_sim = self.simulation.esize

        e_scatt_vec, _, e_inc_vec, _ = mie_efield(
            float(self.wavelength_um),
            float(self.sphere.n_medium),
            float(self.sphere.radius_um),
            complex(self.sphere.n_sphere),
            float(self.simulation.dxy_um),
            (int(ny_sim), int(nx_sim)),
            dz=float(z_plane),
            beam_theta=float(theta),
            beam_phi=float(phi),
            use_gpu=bool(self.use_gpu),
            **self.mie_kwargs,
        )

        # Scalar field (x-component) and gentle apodization to suppress wrap-around.
        e0 = (e_scatt_vec[0] + e_inc_vec[0]).astype(xp.complex64, copy=False)
        apod = _get_cached_apodization(int(ny_sim), int(nx_sim), alpha=0.1, use_gpu=bool(self.use_gpu))
        e0 = (e0 * apod.astype(e0.real.dtype, copy=False)).astype(xp.complex64, copy=False)

        e_in = e0[None, None, :, :]  # shape (1, 1, ny, nx) for propagate_homogeneous
        e_prop = propagate_homogeneous(
            e_in,
            [-float(z_plane)],
            float(self.sphere.n_medium),
            (float(self.simulation.dxy_um), float(self.simulation.dxy_um)),
            float(self.wavelength_um),
        )

        if e_prop.ndim >= 5:
            e_foc = e_prop[..., 0, :, :][0, 0]
        elif e_prop.ndim == 4:
            e_foc = e_prop[0, 0]
        else:
            e_foc = e_prop

        # Reuse the existing objective pupil helper.
        e_vec = xp.zeros((3,) + e_foc.shape, dtype=e_foc.dtype)
        e_vec[0] = e_foc
        e_img_vec = apply_objective_imaging(
            e_vec,
            dxy_um=float(self.simulation.dxy_um),
            wavelength_um=float(self.wavelength_um),
            na_obj=float(self.na_obj),
            use_gpu=bool(self.use_gpu),
        )
        return e_img_vec[0]

    def _render_camera(self, irradiance: array) -> array: # type: ignore
        """
        Project simulation irradiance onto the camera grid (binning + optional noise).

        Parameters
        ----------
        irradiance : array
            Simulation-grid irradiance (pre-camera).

        Returns
        -------
        array
            Camera-grid image after binning/noise from `localize_psf.camera.simulated_img`.

        Notes
        -----
        `bin_size` is derived from `camera.object_pixel_um / simulation.dxy_um`; all
        camera noise parameters (gains, offsets, readout_noise_sds, photon_shot_noise,
        saturation, image_is_integer) are forwarded from `CameraSpace`. PSF input is
        not supported; the coherent transfer is modeled via the pupil internally.
        """
        xp = self.xp
        gt = xp.asarray(irradiance, dtype=xp.float32)
        gt = gt * self.exposure_time_s

        gains = xp.asarray(self.camera.gains, dtype=gt.dtype)
        offsets = xp.asarray(self.camera.offsets, dtype=gt.dtype)
        readout = xp.asarray(self.camera.readout_noise_sds, dtype=gt.dtype)

        psf = None
        if self.camera.psf is not None:
            psf = xp.asarray(self.camera.psf, dtype=gt.dtype)

        apo = self.camera.apodization
        if isinstance(apo, (np.ndarray,)) or (cp is not None and isinstance(apo, cp.ndarray)):
            apo = xp.asarray(apo, dtype=gt.dtype)

        img, _ = simulated_img(
            ground_truth=gt,
            gains=gains,
            offsets=offsets,
            readout_noise_sds=readout,
            psf=psf,
            photon_shot_noise=bool(self.camera.photon_shot_noise),
            bin_size=int(self.camera_bin_factor),
            apodization=apo,
            saturation=self.camera.saturation,
            image_is_integer=bool(self.camera.image_is_integer),
        )
        return xp.asarray(img, dtype=xp.float32)

    def _simulate_led(self, na_x: float, na_y: float) -> array: # type: ignore
        """
        Full pipeline for one LED: field generation → optional defocus → camera rendering.

        Returns a focal stack with shape (n_planes, ny_cam, nx_cam).
        """
        xp = self.xp
        field0 = self._simulate_led_ground_truth(na_x, na_y)

        flux_scale = float(self.illumination_photons_per_s_per_um2) * float(self.simulation.dxy_um) ** 2
        stack = []
        for dz in self.focal_offsets_um:
            if dz == 0:
                f_def = field0
            else:
                e_in = field0[None, None, :, :]
                e_prop = propagate_homogeneous(
                    e_in,
                    [float(dz)],
                    float(self.sphere.n_medium),
                    (float(self.simulation.dxy_um), float(self.simulation.dxy_um)),
                    float(self.wavelength_um),
                )
                if e_prop.ndim >= 5:
                    f_def = e_prop[..., 0, :, :][0, 0]
                elif e_prop.ndim == 4:
                    f_def = e_prop[0, 0]
                else:
                    f_def = e_prop

            I_sim = (xp.abs(f_def) ** 2).astype(xp.float32, copy=False) * flux_scale
            stack.append(self._render_camera(I_sim))

        return xp.stack(stack, axis=0)

    def simulate_patterns(self) -> tuple[array, dict[str, array]]: # type: ignore
        """
        Simulate all LEDs and accumulate into DPC patterns.

        Returns
        -------
        tuple[array, dict[str, array]]
            DPC stack ordered by `pattern_order` and metadata including NA positions,
            grid sizes, bin factor, and LED counts.

        Notes
        -----
        - LED NA grid and pattern membership come from `make_led_na_positions` and
          `split_dpc_patterns` using `led_grid_shape`, `inner_na`, `include_center_led`,
          and `pattern_order`.
        - Each LED image is produced by `_simulate_led` (internally
          `mie_fields.mie_efield` → `propagate_homogeneous` → `apply_objective_imaging`
          → `localize_psf.camera.simulated_img`).
        - Optional per-LED cache stored in Zarr (v3 only).
        """
        allowed: tuple[PatternName, ...] = ("left", "right", "up", "down")
        ny_led, nx_led = self.led_grid_shape
        na_xy_all = make_led_na_positions(
            ny_led,
            nx_led,
            na_obj=float(self.na_obj),
            inner_na=float(self.inner_na),
            include_center=bool(self.include_center_led),
        )
        patterns_full = split_dpc_patterns(na_xy_all, order=self.pattern_order)

        na_xy = na_xy_all[:: self.led_subsample]
        na_x = na_xy[:, 0]
        na_y = na_xy[:, 1]

        membership = {
            "left": na_x < 0,
            "right": na_x > 0,
            "up": na_y > 0,
            "down": na_y < 0,
        }

        if self.include_center_led:
            center_mask = (na_x == 0) & (na_y == 0)
            if np.any(center_mask):
                for k in membership:
                    membership[k] = membership[k] | center_mask

        xp = self.xp
        ny_cam, nx_cam = self.camera.shape
        n_planes = int(len(self.focal_offsets_um))
        acc = {k: xp.zeros((n_planes, ny_cam, nx_cam), dtype=xp.float32) for k in allowed}
        led_counts = {k: int(np.sum(np.asarray(membership[k], dtype=bool))) for k in allowed}

        run_dir = None
        I_cache = None
        done_cache = None

        if self.cache_dir is not None:
            cache_params = {
                "wavelength_um": float(self.wavelength_um),
                "na_obj": float(self.na_obj),
                "led_grid_shape": [int(self.led_grid_shape[0]), int(self.led_grid_shape[1])],
                "camera_pixel_um": float(self.camera.pixel_um),
                "magnification": float(self.camera.magnification),
                "camera_shape": [int(ny_cam), int(nx_cam)],
                "simulation_dxy_um": float(self.simulation.dxy_um),
                "simulation_esize": [int(self.simulation.esize[0]), int(self.simulation.esize[1])],
                "field_generation_z_um": float(self._field_plane_z()),
                "sphere": {
                    "radius_um": float(self.sphere.radius_um),
                    "n_sphere": str(self.sphere.n_sphere),
                    "n_medium": float(self.sphere.n_medium),
                },
                "inner_na": float(self.inner_na),
                "include_center_led": bool(self.include_center_led),
                "pattern_order": [str(p) for p in self.pattern_order],
                "normalize_by_led_count": bool(self.normalize_by_led_count),
                "led_subsample": int(self.led_subsample),
                "mie_kwargs": self.mie_kwargs,
                "camera_bin_factor": int(self.camera_bin_factor),
                "exposure_time_s": float(self.exposure_time_s),
                "illumination_photons_per_s_per_um2": float(self.illumination_photons_per_s_per_um2),
                "focal_offsets_um": self.focal_offsets_um.tolist(),
            }
            run_dir = _cache_run_dir(self.cache_dir, cache_params)
            chunks = (1, 1, min(256, int(ny_cam)), min(256, int(nx_cam)))
            _, I_cache, done_cache = _open_led_cache_zarr(
                Path(run_dir),
                n_led=int(na_xy.shape[0]),
                n_planes=int(n_planes),
                ny=int(ny_cam),
                nx=int(nx_cam),
                chunks=chunks,
            )

        for j in trange(int(na_xy.shape[0]), desc="Simulating LEDs"):
            nax = float(na_xy[j, 0])
            nay = float(na_xy[j, 1])

            I_cam = None
            if I_cache is not None and done_cache is not None and self.reuse_cache and int(done_cache[j]) == 1:
                I_cam_np = np.asarray(I_cache[j, ...], dtype=np.float32)
                I_cam = cp.asarray(I_cam_np) if (self.use_gpu and cp is not None) else I_cam_np
                I_cam = I_cam.astype(xp.float32, copy=False)

            if I_cam is None:
                I_cam = self._simulate_led(nax, nay).astype(xp.float32, copy=False)

                if I_cache is not None and done_cache is not None:
                    I_cam_save = cp.asnumpy(I_cam) if (cp is not None and hasattr(I_cam, "__cuda_array_interface__")) else np.asarray(I_cam)
                    I_cache[j, ...] = np.asarray(I_cam_save, dtype=np.float32)
                    done_cache[j] = 1

            if not (xp.isfinite(I_cam).all()) or xp.isinf(I_cam).any():
                print(f"isfinite check: {xp.isfinite(I_cam).all()}")
                print(f"isinf check: {xp.isinf(I_cam).any()}")

            if membership["left"][j]:
                acc["left"] += I_cam
            if membership["right"][j]:
                acc["right"] += I_cam
            if membership["up"][j]:
                acc["up"] += I_cam
            if membership["down"][j]:
                acc["down"] += I_cam

        if self.normalize_by_led_count:
            for k in allowed:
                if led_counts[k] <= 0:
                    raise RuntimeError(f"No LEDs accumulated for pattern '{k}' (after subsampling).")
                acc[k] = acc[k] / float(led_counts[k])

        # Sanity check: ensure no pattern is identically zero after accumulation
        for k in allowed:
            if float(xp.sum(acc[k])) == 0.0:
                raise RuntimeError(f"Accumulated pattern '{k}' is zero; check illumination masks and cache settings.")

        dpc = xp.stack([acc[p].copy() for p in self.pattern_order], axis=0)
        dpc = xp.moveaxis(dpc, 0, 1)  # (n_planes, 4, ny, nx)
        dpc_out: array
        if dpc.shape[0] == 1:
            dpc_out = dpc[0]
        else:
            dpc_out = dpc

        meta: dict[str, array] = { # type: ignore
            "na_xy": na_xy_all,
            "simulation_dxy_um": np.asarray(self.simulation.dxy_um, dtype=np.float32),
            "simulation_esize": (int(self.simulation.esize[0]), int(self.simulation.esize[1])),
            "camera_shape": (int(ny_cam), int(nx_cam)),
            "camera_bin_factor": int(self.camera_bin_factor),
            "led_counts": {k: int(led_counts[k]) for k in allowed},
            "z_plane_um": float(self._field_plane_z()),
            "exposure_time_s": float(self.exposure_time_s),
            "illumination_photons_per_s_per_um2": float(self.illumination_photons_per_s_per_um2),
            "pattern_sums": {k: float(xp.sum(acc[k])) for k in allowed},
            "focal_offsets_um": self.focal_offsets_um.tolist(),
        } 
        for k, v in patterns_full.items():
            meta[k] = v

        return dpc_out, meta



def simulate_led_irradiance(
    *,
    wavelength_um: float,
    sphere: SphereSpec,
    dxy_um: float,
    esize: tuple[int, int],
    z_plane_um: float,
    na_x: float,
    na_y: float,
    na_obj: float,
    camera_bin: int = 1,
    use_gpu: bool = False,
    mie_kwargs: dict | None = None,
) -> array: # type: ignore
    """
    Simulate the (binned) camera-plane irradiance for a single LED illumination.

    Parameters
    ----------
    wavelength_um : float
        Vacuum wavelength (µm) for `mie_efield` and the objective pupil.
    sphere : SphereSpec
        Sphere properties forwarded to `mie_efield`.
    dxy_um : float
        Simulation sampling (µm) passed to `mie_efield` and propagation.
    esize : tuple[int, int]
        Simulation grid size (ny, nx).
    z_plane_um : float
        Field evaluation plane for `mie_efield`; if inside the sphere, a safe plane is chosen.
    na_x, na_y : float
        Illumination NA components converted to angles for `mie_efield`.
    na_obj : float
        Objective NA for the coherent pupil.
    camera_bin : int, optional
        Bin size for `simulated_img`, derived from simulation vs camera sampling.
    use_gpu : bool, optional
        If True, use CuPy-backed operations when available.
    mie_kwargs : dict or None, optional
        Extra keyword arguments passed directly to `mie_fields.mie_efield`.

    Returns
    -------
    array
        Binned camera-plane irradiance for the single LED.

    Notes
    -----
    Internally builds a `DPCMieSimulator` with matching grids and calls its single-LED
    pipeline: `mie_efield` → `propagate_homogeneous` → `apply_objective_imaging` →
    `localize_psf.camera.simulated_img`.
    """
    if sphere is None:
        sphere = SphereSpec(radius_um=5.0, n_sphere=1.59 + 0.0j, n_medium=1.55)

    sim_space = SimulationSpace(
        dxy_um=float(dxy_um),
        esize=(int(esize[0]), int(esize[1])),
        z_plane_um=float(z_plane_um),
    )
    cam_shape = (int(esize[0] // camera_bin), int(esize[1] // camera_bin))
    cam_space = CameraSpace(
        pixel_um=float(dxy_um) * float(camera_bin),
        magnification=1.0,
        shape=cam_shape,
        photon_shot_noise=False,
        readout_noise_sds=0.0,
        gains=1.0,
        offsets=0.0,
        image_is_integer=False,
    )

    simulator = DPCMieSimulator(
        wavelength_um=float(wavelength_um),
        na_obj=float(na_obj),
        sphere=sphere,
        simulation=sim_space,
        camera=cam_space,
        led_grid_shape=(1, 1),
        normalize_by_led_count=False,
        led_subsample=1,
        use_gpu=use_gpu,
        mie_kwargs=mie_kwargs,
        cache_dir=None,
        reuse_cache=True,
    )
    return simulator._simulate_led(float(na_x), float(na_y))


def simulate_dpc_images_sphere(
    *,
    wavelength_um: float = 0.515,
    na_obj: float = 0.8,
    led_grid_shape: tuple[int, int] = (64, 64),
    camera_pixel_um: float = 2.4,
    magnification: float = 20.0,
    camera_oversample: int = 1,
    esize_camera: tuple[int, int] = (256, 256),
    sim_dxy_um: float | None = None,
    sim_esize: tuple[int, int] | None = None,
    z_plane_um: float = 0.0,
    sphere: SphereSpec | None = None,
    inner_na: float = 0.0,
    include_center_led: bool = False,
    pattern_order: Sequence[PatternName] = ("left", "right", "up", "down"),
    normalize_by_led_count: bool = True,
    led_subsample: int = 1,
    use_gpu: bool = False,
    mie_kwargs: dict | None = None,
    cache_dir: str | Path | None = None,
    reuse_cache: bool = True,
    camera_gains: array | float = 1.0, # type: ignore
    camera_offsets: array | float = 0.0, # type: ignore
    camera_readout_noise_sds: array | float = 0.0, # type: ignore
    camera_photon_shot_noise: bool = False,
    camera_saturation: int | None = None,
    camera_image_is_integer: bool = False,
    exposure_time_ms: float = 1.0,
    illumination_photons_per_s_per_um2: float = 1.0,
    focal_stack_planes: int | None = None,
    focal_stack_step_um: float = 0.5,
) -> tuple[array, dict[str, array]]: # type: ignore
    """
    Generate four synthetic DPC images by incoherently summing per-LED irradiance.

    Parameters (and where they go)
    ------------------------------
    wavelength_um : float
        Vacuum wavelength (µm). Passed to `mie_fields.mie_efield`, the objective pupil
        in `apply_objective_imaging`, and `propagate_homogeneous`.
    na_obj : float
        Objective NA. Sets the coherent cutoff in `apply_objective_imaging` via
        `make_pupil`.
    led_grid_shape : (int, int)
        LED board shape (ny_led, nx_led) for `make_led_na_positions`.
    camera_pixel_um : float
        Camera pixel pitch (µm) in the camera plane; combined with `magnification` to
        derive object-plane sampling and the bin size for `localize_psf.camera.simulated_img`.
    magnification : float
        Object-to-camera magnification. Higher values increase the bin factor between
        simulation grid and camera grid.
    camera_oversample : int
        Optional oversampling factor applied to the camera grid; influences the default
        `sim_dxy_um` and `sim_esize` when not provided.
    esize_camera : (int, int)
        Final camera image shape (ny, nx) produced by `simulated_img`.
    sim_dxy_um : float | None
        Simulation grid sampling (µm). If None, derived from `camera_pixel_um /
        magnification / camera_oversample` and forwarded to `mie_efield` and
        `propagate_homogeneous`.
    sim_esize : (int, int) | None
        Simulation grid size (ny, nx). If None, defaults to `esize_camera *
        camera_oversample`.
    z_plane_um : float
        Optional user-specified field evaluation plane for `mie_efield` (validated to
        stay outside the sphere). If 0, a safe exterior plane is chosen automatically.
    sphere : SphereSpec | None
        Sphere parameters (radius, n_sphere, n_medium) passed to `mie_efield`. Defaults
        to a polystyrene-like sphere in medium if None.
    inner_na : float
        Inner NA to form an annular LED mask in `make_led_na_positions`.
    include_center_led : bool
        Whether to include the origin LED (NA=0) in all patterns.
    camera_gains : array or float, optional
        Gains (ADU/e) for `simulated_img`.
    camera_offsets : array or float, optional
        Offsets (ADU) for `simulated_img`.
    camera_readout_noise_sds : array or float, optional
        Readout noise SD (ADU) for `simulated_img`.
    camera_photon_shot_noise : bool, optional
        Enable Poisson shot noise in `simulated_img`.
    camera_saturation : int or None, optional
        Saturation level passed to `simulated_img`.
    camera_image_is_integer : bool, optional
        If True, round simulated images to integers in `simulated_img`.
    exposure_time_ms : float, optional
        Exposure time (milliseconds). Multiplies simulated irradiance to convert to
        expected photon counts before camera noise is applied.
    illumination_photons_per_s_per_um2 : float, optional
        Incident photon flux density (photons/s/µm^2) used to scale the Mie-derived
        irradiance (relative units) into absolute photons/s per pixel area.
    focal_stack_planes : int or None, optional
        Number of focal planes to simulate. If None or <=1, a single plane is produced.
    focal_stack_step_um : float, optional
        Axial step (µm) between focal planes, centered on nominal focus.
    pattern_order : Sequence[str]
        Order of DPC patterns ("left","right","up","down") returned in the output stack.
    normalize_by_led_count : bool
        If True, divides each pattern sum by the number of LEDs contributing to it.
    led_subsample : int
        Subsample factor on the LED list before simulation (simulates every Nth LED).
    use_gpu : bool
        If True, uses CuPy-backed versions of `mie_efield`, FFTs, and `simulated_img`
        where available.
    mie_kwargs : dict | None
        Extra keyword arguments forwarded directly to `mie_fields.mie_efield`.
    cache_dir : str | Path | None
        Optional directory for Zarr-based per-LED camera irradiance caching
        (v3-compatible; no .npy files).
    reuse_cache : bool
        If True, reuses cached per-LED camera images when present.

    Returns
    -------
    dpc : array
        Stack of four DPC images ordered by `pattern_order` (shape (4, ny, nx)).
    meta : dict
        Metadata including NA positions, grid sizes, bin factor, and LED counts for
        downstream reconstruction/testing.
    """
    if led_subsample < 1:
        raise ValueError("led_subsample must be >= 1")
    if camera_oversample < 1:
        raise ValueError("camera_oversample must be >= 1")

    if sphere is None:
        sphere = SphereSpec(radius_um=5.0, n_sphere=1.59 + 0.0j, n_medium=1.55)

    sim_dxy_val = sim_dxy_um if sim_dxy_um is not None else sample_pixel_size_um(
        camera_pixel_um, magnification, oversample=int(camera_oversample)
    )
    sim_esize_val = sim_esize if sim_esize is not None else (
        int(esize_camera[0]) * int(camera_oversample),
        int(esize_camera[1]) * int(camera_oversample),
    )

    sim_space = SimulationSpace(
        dxy_um=float(sim_dxy_val),
        esize=(int(sim_esize_val[0]), int(sim_esize_val[1])),
        z_plane_um=float(z_plane_um),
    )
    cam_space = CameraSpace(
        pixel_um=float(camera_pixel_um),
        magnification=float(magnification),
        shape=(int(esize_camera[0]), int(esize_camera[1])),
        photon_shot_noise=bool(camera_photon_shot_noise),
        readout_noise_sds=camera_readout_noise_sds,
        gains=camera_gains,
        offsets=camera_offsets,
        image_is_integer=bool(camera_image_is_integer),
        psf=None,
        apodization=1,
        saturation=camera_saturation,
    )

    simulator = DPCMieSimulator(
        wavelength_um=float(wavelength_um),
        na_obj=float(na_obj),
        sphere=sphere,
        simulation=sim_space,
        camera=cam_space,
        led_grid_shape=led_grid_shape,
        inner_na=float(inner_na),
        include_center_led=bool(include_center_led),
        pattern_order=pattern_order,
        normalize_by_led_count=bool(normalize_by_led_count),
        led_subsample=int(led_subsample),
        use_gpu=bool(use_gpu),
        mie_kwargs=mie_kwargs,
        cache_dir=cache_dir,
        reuse_cache=bool(reuse_cache),
        exposure_time_ms=float(exposure_time_ms),
        illumination_photons_per_s_per_um2=float(illumination_photons_per_s_per_um2),
        focal_stack_planes=focal_stack_planes,
        focal_stack_step_um=float(focal_stack_step_um),
    )

    dpc, meta = simulator.simulate_patterns()

    # Compatibility metadata with previous helper
    meta.update({
        "dxy_um": np.asarray(sim_dxy_val, dtype=np.float32),
        "camera_oversample": int(simulator.camera_bin_factor),
        "esize_camera": (int(esize_camera[0]), int(esize_camera[1])),
        "esize_mie": (int(sim_space.esize[0]), int(sim_space.esize[1])),
    })

    # Crop 10% border from each side to keep central region
    def _crop_center(arr: array) -> array:
        if arr.ndim == 3:
            _, ny, nx = arr.shape
        elif arr.ndim == 4:
            _, _, ny, nx = arr.shape
        else:
            return arr
        ypad = max(0, int(round(0.1 * ny)))
        xpad = max(0, int(round(0.1 * nx)))
        y1, y2 = ypad, ny - ypad
        x1, x2 = xpad, nx - xpad
        if y2 <= y1 or x2 <= x1:
            return arr
        if arr.ndim == 3:
            return arr[:, y1:y2, x1:x2]
        else:
            return arr[:, :, y1:y2, x1:x2]

    dpc = _crop_center(dpc)
    if dpc.ndim == 3:
        _, ny_crop, nx_crop = dpc.shape
    else:
        _, _, ny_crop, nx_crop = dpc.shape
    meta["camera_shape"] = (int(ny_crop), int(nx_crop))
    meta["esize_camera"] = (int(ny_crop), int(nx_crop))

    return dpc, meta


def write_dpc_zarr(
    zarr_path: str | Path,
    dpc: array, # type: ignore
    meta: dict[str, array], # type: ignore
    *,
    wavelength_um: float,
    na_obj: float,
    camera_pixel_um: float,
    magnification: float,
    n_medium: float,
    led_grid_shape: tuple[int, int],
    pattern_order: Sequence[PatternName],
    sphere: SphereSpec,
    inner_na: float,
    include_center_led: bool,
    nz: int | None = None,
    z_span_um: float | None = None,
    overwrite: bool = True,
) -> None:
    """
    Write a DPC synthetic dataset to Zarr in the layout expected by tests.

    Parameters
    ----------
    zarr_path : str or Path
        Output Zarr group path.
    dpc : array
        DPC stack (4, ny, nx) to write.
    meta : dict[str, array]
        Metadata returned from `simulate_dpc_images_sphere` (includes NA positions).
    wavelength_um : float
        Vacuum wavelength (µm) for root attrs.
    na_obj : float
        Objective NA for root attrs.
    camera_pixel_um : float
        Camera pixel size (µm) for root attrs.
    magnification : float
        System magnification for root attrs.
    n_medium : float
        Medium index for root attrs.
    led_grid_shape : tuple[int, int]
        LED grid shape stored in root attrs.
    pattern_order : Sequence[str]
        Pattern order stored in root attrs.
    sphere : SphereSpec
        Sphere parameters; radius/index stored as extra attrs.
    inner_na : float
        Inner NA stored as extra attr.
    include_center_led : bool
        Whether center LED included; stored as extra attr.
    nz : int or None, optional
        Optional z-stacks count stored in attrs.
    z_span_um : float or None, optional
        Optional z-span stored in attrs.
    overwrite : bool, optional
        If True, overwrite existing Zarr group.

    Returns
    -------
    None
    """
    try:
        import zarr  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("zarr is required to write datasets") from e

    zarr_path = str(zarr_path)
    mode = "w" if overwrite else "w-"
    g = zarr.open_group(zarr_path, mode=mode)

    # --- Root attributes (exact keys expected by tests) ---
    g.attrs["wavelength_um"] = float(wavelength_um)
    g.attrs["na_obj"] = float(na_obj)
    g.attrs["camera_pixel_um"] = float(camera_pixel_um)
    g.attrs["magnification"] = float(magnification)
    g.attrs["n_medium"] = float(n_medium)
    g.attrs["led_grid_shape"] = [int(led_grid_shape[0]), int(led_grid_shape[1])]
    g.attrs["pattern_order"] = [str(p) for p in pattern_order]
    # Effective Mie evaluation plane used for simulation (µm).
    if "z_plane_um" in meta:
        g.attrs["field_generation_z_um"] = float(np.asarray(meta["z_plane_um"]))

    # Ensure CPU ndarray for Zarr
    if cp is not None and isinstance(dpc, cp.ndarray):
        dpc_np = cp.asnumpy(dpc).astype(np.float32, copy=False)
    else:
        dpc_np = np.asarray(dpc, dtype=np.float32)

    if nz is not None:
        g.attrs["nz"] = int(nz)
    elif dpc_np.ndim == 4:
        g.attrs["nz"] = int(dpc_np.shape[0])
    if z_span_um is not None:
        g.attrs["z_span_um"] = float(z_span_um)

    # Optional extras (not required by tests)
    g.attrs["sphere_radius_um"] = float(sphere.radius_um)
    g.attrs["sphere_n_sphere"] = complex(sphere.n_sphere).__repr__()
    g.attrs["sphere_n_medium"] = float(sphere.n_medium)
    g.attrs["inner_na"] = float(inner_na)
    g.attrs["include_center_led"] = bool(include_center_led)
    g.attrs["dxy_um"] = float(np.asarray(meta.get("dxy_um", camera_pixel_um / magnification)))

    if dpc_np.ndim == 3:
        if dpc_np.shape[0] != 4:
            raise ValueError(f"dpc must have shape (4, ny, nx), got {dpc_np.shape}")
        chunks = (1, min(256, dpc_np.shape[1]), min(256, dpc_np.shape[2]))
    elif dpc_np.ndim == 4:
        if dpc_np.shape[1] != 4:
            raise ValueError(f"dpc must have shape (nz, 4, ny, nx), got {dpc_np.shape}")
        chunks = (1, 1, min(256, dpc_np.shape[2]), min(256, dpc_np.shape[3]))
    else:
        raise ValueError(f"dpc must have ndim 3 or 4, got {dpc_np.ndim}")
    _zarr_write_array(g, "dpc", dpc_np, dtype="float32", chunks=chunks, overwrite=True)

    mg = g.require_group("meta")
    if "na_xy" in meta:
        na_xy_np = np.asarray(meta["na_xy"], dtype=np.float32)
        _zarr_write_array(mg, "na_xy", na_xy_np, dtype="float32", overwrite=True)

    for pname in ("left", "right", "up", "down"):
        if pname in meta:
            pts_np = np.asarray(meta[pname], dtype=np.float32)
            _zarr_write_array(mg, pname, pts_np, dtype="float32", overwrite=True)
    if "focal_offsets_um" in meta:
        offsets_np = np.asarray(meta["focal_offsets_um"], dtype=np.float32)
        _zarr_write_array(mg, "focal_offsets_um", offsets_np, dtype="float32", overwrite=True)


def simulate_dpc_images_sphere_to_zarr(
    zarr_path: str | Path,
    *,
    wavelength_um: float = 0.515,
    na_obj: float = 0.8,
    led_grid_shape: tuple[int, int] = (64, 64),
    camera_pixel_um: float = 2.4,
    magnification: float = 20.0,
    camera_oversample: int = 1,
    esize_camera: tuple[int, int] = (256, 256),
    sim_dxy_um: float | None = None,
    sim_esize: tuple[int, int] | None = None,
    z_plane_um: float = 0.0,
    sphere: SphereSpec | None = None,
    inner_na: float = 0.0,
    include_center_led: bool = False,
    pattern_order: Sequence[PatternName] = ("left", "right", "up", "down"),
    normalize_by_led_count: bool = True,
    led_subsample: int = 1,
    use_gpu: bool = False,
    mie_kwargs: dict | None = None,
    cache_dir: str | Path | None = None,
    reuse_cache: bool = True,
    nz: int | None = None,
    z_span_um: float | None = None,
    overwrite: bool = True,
    camera_gains: array | float = 1.0, # type: ignore
    camera_offsets: array | float = 0.0, # type: ignore
    camera_readout_noise_sds: array | float = 0.0, # type: ignore
    camera_photon_shot_noise: bool = False,
    camera_saturation: int | None = None,
    camera_image_is_integer: bool = False,
    exposure_time_ms: float = 1.0,
    illumination_photons_per_s_per_um2: float = 1.0,
    focal_stack_planes: int | None = None,
    focal_stack_step_um: float = 0.5,
) -> None:
    """
    Generate DPC images and write them to Zarr.

    Parameters
    ----------
    zarr_path : str or Path
        Output Zarr group path.
    wavelength_um : float, optional
        Vacuum wavelength (µm) passed to simulation and stored in attrs.
    na_obj : float, optional
        Objective NA for pupil cutoff and attrs.
    led_grid_shape : tuple[int, int], optional
        LED board shape for pattern generation.
    camera_pixel_um : float, optional
        Camera pixel size (µm) for attrs and camera sampling.
    magnification : float, optional
        Object-to-camera magnification used to derive camera binning.
    camera_oversample : int, optional
        Oversampling factor; affects default simulation sampling/size.
    esize_camera : tuple[int, int], optional
        Final camera image shape (ny, nx).
    sim_dxy_um : float or None, optional
        Simulation sampling (µm); defaults to camera_pixel_um / magnification / camera_oversample.
    sim_esize : tuple[int, int] or None, optional
        Simulation grid size; defaults to esize_camera * camera_oversample.
    z_plane_um : float, optional
        Field evaluation plane for `mie_efield`; validated to be outside the sphere.
    sphere : SphereSpec or None, optional
        Sphere parameters; defaults to a preset if None.
    inner_na : float, optional
        Inner NA for annular LED mask.
    include_center_led : bool, optional
        Whether to include the center LED.
    pattern_order : Sequence[str], optional
        Order of DPC patterns in output.
    normalize_by_led_count : bool, optional
        If True, normalize each pattern by its LED count.
    led_subsample : int, optional
        Subsample factor for LEDs.
    use_gpu : bool, optional
        If True, use CuPy-backed ops when available.
    mie_kwargs : dict or None, optional
        Extra kwargs passed to `mie_fields.mie_efield`.
    cache_dir : str or Path or None, optional
        Directory for Zarr per-LED camera irradiance cache.
    reuse_cache : bool, optional
        If True, reuse cached per-LED images.
    camera_gains : array or float, optional
        Gains (ADU/e) for `simulated_img`.
    camera_offsets : array or float, optional
        Offsets (ADU) for `simulated_img`.
    camera_readout_noise_sds : array or float, optional
        Readout noise SD (ADU) for `simulated_img`.
    camera_photon_shot_noise : bool, optional
        Enable Poisson shot noise in `simulated_img`.
    camera_saturation : int or None, optional
        Saturation level for `simulated_img`.
    camera_image_is_integer : bool, optional
        If True, round simulated images to integers in `simulated_img`.
    exposure_time_ms : float, optional
        Exposure time (milliseconds). Multiplies simulated irradiance before camera noise.
    illumination_photons_per_s_per_um2 : float, optional
        Incident photon flux density (photons/s/µm^2) used to scale irradiance to photons.
    focal_stack_planes : int or None, optional
        Number of focal planes to simulate. If None or <=1, a single plane is produced.
    focal_stack_step_um : float, optional
        Axial step (µm) between focal planes, centered on nominal focus.
    nz : int or None, optional
        Optional z-stack count stored in attrs.
    z_span_um : float or None, optional
        Optional z-span stored in attrs.
    overwrite : bool, optional
        If True, overwrite existing Zarr group.

    Returns
    -------
    None
    """
    if cache_dir is None:
        # Default cache directory adjacent to the output Zarr store.
        zp = Path(zarr_path)
        cache_dir = zp.with_name(zp.name + ".mie_cache")

    dpc, meta = simulate_dpc_images_sphere(
        wavelength_um=wavelength_um,
        na_obj=na_obj,
        led_grid_shape=led_grid_shape,
        camera_pixel_um=camera_pixel_um,
        magnification=magnification,
        camera_oversample=camera_oversample,
        esize_camera=esize_camera,
        sim_dxy_um=sim_dxy_um,
        sim_esize=sim_esize,
        z_plane_um=z_plane_um,
        sphere=sphere,
        inner_na=inner_na,
        include_center_led=include_center_led,
        pattern_order=pattern_order,
        normalize_by_led_count=normalize_by_led_count,
        led_subsample=led_subsample,
        use_gpu=use_gpu,
        mie_kwargs=mie_kwargs,
        cache_dir=cache_dir,
        reuse_cache=reuse_cache,
        camera_gains=camera_gains,
        camera_offsets=camera_offsets,
        camera_readout_noise_sds=camera_readout_noise_sds,
        camera_photon_shot_noise=camera_photon_shot_noise,
        camera_saturation=camera_saturation,
        camera_image_is_integer=camera_image_is_integer,
        exposure_time_ms=exposure_time_ms,
        illumination_photons_per_s_per_um2=illumination_photons_per_s_per_um2,
        focal_stack_planes=focal_stack_planes,
        focal_stack_step_um=focal_stack_step_um,
    )

    sphere_used = sphere if sphere is not None else SphereSpec(radius_um=5.0, n_sphere=1.59, n_medium=1.515)

    write_dpc_zarr(
        zarr_path,
        dpc,
        meta,
        wavelength_um=wavelength_um,
        na_obj=na_obj,
        camera_pixel_um=camera_pixel_um,
        magnification=magnification,
        n_medium=float(sphere_used.n_medium),
        led_grid_shape=led_grid_shape,
        pattern_order=pattern_order,
        sphere=sphere_used,
        inner_na=inner_na,
        include_center_led=include_center_led,
        nz=nz,
        z_span_um=z_span_um,
        overwrite=overwrite,
    )

if __name__ == "__main__":  # pragma: no cover
    # Example usage: generate a DPC dataset and write to Zarr.
    output_path = Path("/media/dps/data/synthetic_dpc_sphere.zarr")

    simulate_dpc_images_sphere_to_zarr(
        output_path,
        wavelength_um=0.515,
        na_obj=0.8,
        led_grid_shape=(64, 64),
        camera_pixel_um=2.4,
        magnification=20.0,
        camera_oversample=2,
        esize_camera=(512, 512),
        z_plane_um=0.0,
        sphere=SphereSpec(radius_um=2.5, n_sphere=1.59, n_medium=1.515),
        inner_na=0.0,
        include_center_led=True,
        pattern_order=("left", "right", "up", "down"),
        normalize_by_led_count=True,
        led_subsample=1,
        use_gpu=True,
        mie_kwargs=None,
        camera_gains=1.0,
        camera_offsets=100.0,
        camera_readout_noise_sds=3.31,
        camera_photon_shot_noise=True,
        camera_saturation=None,
        camera_image_is_integer=True,
        exposure_time_ms=100.0,
        illumination_photons_per_s_per_um2=1e6,
        focal_stack_planes=41,
        focal_stack_step_um=.325,
        overwrite=True,
    )
    print(f"Synthetic DPC dataset written to {output_path}")
