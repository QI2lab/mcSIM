from typing import Optional, Sequence, Union
import numpy as np
from mcsim.optimize import Optimizer
from mcsim.optimize.prox import tv_prox, soft_threshold, median_prox
try:
    import zarr  # must be >= 3
except Exception as _e:
    zarr = None
    _zarr_err = _e

try:
    import cupy as cp
except ImportError:
    cp = None

# -----------------------
# Alias for CPU/GPU arrays
# -----------------------
if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray

def _to_cpu(a):
    """Return NumPy array from NumPy/CuPy input without copying if possible."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)

class DPC3D(Optimizer):
    """
    3D Differential Phase-Contrast inverse problem with 4 half-circle source patterns.

    Forward model (Chen–Tian–Waller, BOE 2016, Eq. (6)):
        Ĩ_ℓ(kz, ky, kx) = H_Re,ℓ(kz, ky, kx)·V̂_Re(kz, ky, kx)
                        + H_Im,ℓ(kz, ky, kx)·V̂_Im(kz, ky, kx),

    with H_Re (Eq. (7)) and H_Im (Eq. (8)) the 3D phase/absorption WOTFs determined
    by the source S (left/right/top/bottom half-circles) and detection pupil P.
    Arrays are stored and operated in [pattern, z, y, x] order in *reciprocal space*.

    Parameters
    ----------
    shape_zyx : tuple[int, int, int]
        Spatial volume shape (Z, Y, X).
    voxel_size_zyx : tuple[float, float, float]
        Voxel size (dz, dy, dx) in meters.
    wavelength : float
        Illumination wavelength in meters.
    n_medium : float
        Refractive index of immersion medium.
    na_obj : float
        Objective NA (detection pupil).
    na_src : float
        Illumination NA (condenser/source).
    I_meas_4p : array | None
        Optional measured intensity stacks with shape (4, Z, Y, X) in *space*.
        If provided, they are normalized and FFT’d to produce I_hat_meas.
        If None, you must later set self.I_hat_meas.
    prox_parameters : dict | None
        Prox controls: {'tv_re', 'tv_im', 'soft_im', 'positivity_re', 'median'}.
    src_grid_N : int
        Discretization of the source pupil per axis (odd recommended).
    defocus_waves : float
        Optional *defocus* aberration in waves (adds quadratic phase in P).
        0.0 means ideal pupil amplitude-only.
    use_ortho_fft : bool
        Orthonormal FFTs (recommended -> clean Lipschitz scaling).
    use_gpu : bool
        Force GPU if CuPy is available and True; else CPU.
    """

    PATTERNS = ("left", "right", "top", "bottom")

    def __init__(self,
                 shape_zyx: tuple[int, int, int],
                 voxel_size_zyx: tuple[float, float, float],
                 wavelength: float,
                 n_medium: float,
                 na_obj: float,
                 na_src: float,
                 I_meas_4p: Optional[array] = None,
                 prox_parameters: Optional[dict] = None,
                 src_grid_N: int = 65,
                 defocus_waves: float = 0.0,
                 use_ortho_fft: bool = True,
                 use_gpu: bool = False):

        Z, Y, X = map(int, shape_zyx)
        dz, dy, dx = map(float, voxel_size_zyx)

        # ---------- backend ----------
        self.use_gpu = bool(use_gpu and (cp is not None))
        self.xp = cp if self.use_gpu else np

        # ---------- geometry ----------
        self.Z, self.Y, self.X = Z, Y, X
        self.dz, self.dy, self.dx = dz, dy, dx
        self.wavelength = float(wavelength)
        self.n = float(n_medium)
        self.na_obj = float(na_obj)
        self.na_src = float(na_src)
        self.k0_cyc = self.n / self.wavelength  # cycles per meter

        # ---------- FFT norms ----------
        self.use_ortho_fft = bool(use_ortho_fft)
        self._fft_norm = 'ortho' if self.use_ortho_fft else None

        # ---------- frequencies ----------
        xp = self.xp
        fz = xp.fft.fftfreq(Z, d=dz)  # cycles/m (unshifted)
        fy = xp.fft.fftfreq(Y, d=dy)
        fx = xp.fft.fftfreq(X, d=dx)
        self.fz = fz
        self.fy = fy
        self.fx = fx
        # Shifted z for splatting convenience
        self.fz_shift = xp.fft.fftshift(fz)
        self.dfz = float(self.fz_shift[1] - self.fz_shift[0])
        self.fz_min = float(self.fz_shift[0])
        self.fz_max = float(self.fz_shift[-1])

        # ---------- pupil (amplitude 1 inside NA, optional defocus phase) ----------
        self.P = self._make_pupil(defocus_waves=defocus_waves)  # (Y, X), complex

        # ---------- WOTFs for 4 patterns ----------
        self.H_re, self.H_im = self._build_wotf_4patterns(src_grid_N=src_grid_N)  # (4,Z,Y,X), real

        # ---------- measured spectra ----------
        self.I_hat_meas = None
        if I_meas_4p is not None:
            self.set_measurements(I_meas_4p)

        # ---------- base Optimizer ----------
        super().__init__(n_samples=4, prox_parameters=prox_parameters or {})
        self._L_est = None  # Lipschitz cache

    # ===========================================================
    # Eq. (10) helpers: synthesize BF and DPC stacks from raw 4 patterns
    # ===========================================================
    @staticmethod
    def synthesize_bf_dpc(
        I_lrbt: array,
        eps: float = 1e-12,
        pair_norm: bool = False,
    ) -> tuple[array, array, array]:
        """
        Build brightfield (BF) and two DPC stacks from raw LEFT/RIGHT/TOP/BOTTOM.

        Parameters
        ----------
        I_lrbt : array
            Raw intensity stacks in *space*, shape (4, Z, Y, X) ordered as:
            [left, right, top, bottom].
        eps : float
            Numerical floor to avoid divide-by-zero.
        pair_norm : bool
            If False (default), use the paper's BF normalization:
                I_BF = mean(L, R, T, B)
                DPC_x = (R - L) / mean(I_BF)
                DPC_y = (T - B) / mean(I_BF)
            If True, use pairwise normalization (sometimes used in 2D DPC):
                DPC_x = (R - L) / (R + L + eps)
                DPC_y = (T - B) / (T + B + eps)
                I_BF  = (L + R + T + B)/4  (returned for completeness)

        Returns
        -------
        I_BF : array
            Brightfield stack, (Z, Y, X).
        DPC_x : array
            Left–Right differential (X-direction), (Z, Y, X).
        DPC_y : array
            Top–Bottom differential (Y-direction), (Z, Y, X).

        Notes
        -----
        This follows the synthesis described around Eq. (10) of Chen, Tian & Waller (2016).
        They normalize by a DC/brightfield term; when a separate background BF is not
        available, they approximate it with the *average* brightfield intensity
        measured from the data (WOA regime), which is what the default branch implements. :contentReference[oaicite:0]{index=0}
        """
        L, R, T, B = I_lrbt[0], I_lrbt[1], I_lrbt[2], I_lrbt[3]
        I_BF = (L + R + T + B) / 4.0

        if pair_norm:
            DPC_x = (R - L) / (R + L + eps)
            DPC_y = (T - B) / (T + B + eps)
        else:
            # DC/brightfield normalization (Eq. 10 discussion)
            I_DC = I_BF.mean()  # scalar <I_BF>, cf. paper text
            DPC_x = (R - L) / (I_DC + eps)
            DPC_y = (T - B) / (I_DC + eps)

        return I_BF, DPC_x, DPC_y

    def set_measurements_from_raw_patterns(
        self,
        I_lrbt_space: array,
        normalize: bool = True,
        pair_norm: bool = False,
    ) -> None:
        """
        Normalize background (optional) and FFT the *raw* 4-pattern stacks.

        Parameters
        ----------
        I_lrbt_space : array
            Raw LEFT/RIGHT/TOP/BOTTOM volumes in *space*, (4, Z, Y, X).
        normalize : bool
            If True, do the same background/DC normalization used in set_measurements().
        pair_norm : bool
            If you plan to *also* compute and use BF/DPC stacks elsewhere, this lets
            you reproduce pairwise DPC for inspection. This flag does not affect the
            raw-pattern spectra used by the forward model here.
        """
        xp = self.xp
        assert I_lrbt_space.shape == (4, self.Z, self.Y, self.X), "Use [pattern,z,y,x] ordering"

        I = xp.array(I_lrbt_space, copy=True)

        if normalize:
            # simple per-pattern background removal
            bg = xp.min(I, axis=(1, 2, 3), keepdims=True)
            I = I - bg
            # global DC scale as in Eq. (10) paragraph (use BF mean as DC)
            I_BF, _, _ = self.synthesize_bf_dpc(I, pair_norm=pair_norm)
            dc = float(I_BF.mean())
            I = I / max(dc, 1e-12)

        # store Fourier-domain measurement spectra for the *4* raw patterns
        self.I_hat_meas = xp.fft.fftn(I, axes=(-3, -2, -1), norm=self._fft_norm)

    # (optional) convenience if you want BF/DPC spectra too (not used by current forward model)
    def bf_dpc_kspace(self, I_lrbt_space: array, pair_norm: bool = False) -> tuple[array, array, array]:
        """
        FFT of the synthesized BF/DPC stacks (space -> k-space).
        Returns (I_BF_hat, DPCx_hat, DPCy_hat), all (Z, Y, X).
        """
        xp = self.xp
        I_BF, DPCx, DPCy = self.synthesize_bf_dpc(I_lrbt_space, pair_norm=pair_norm)
        F = lambda v: xp.fft.fftn(v, axes=(-3, -2, -1), norm=self._fft_norm)
        return F(I_BF), F(DPCx), F(DPCy)


    # ===========================================================
    # Public API
    # ===========================================================
    def set_measurements(self, I_meas_4p: array, dc: Optional[float] = None) -> None:
        """
        Normalize and FFT the 4 raw pattern stacks (left/right/top/bottom).

        Normalization (Eq. (10) spirit): subtract background (min over z/y/x),
        divide by a DC scalar (provided or mean over brightfield proxy).
        """
        xp = self.xp
        assert I_meas_4p.shape == (4, self.Z, self.Y, self.X)
        I = xp.array(I_meas_4p, copy=True)

        # simple background remove (per-pattern) then single DC scale
        bg = xp.min(I, axis=(1, 2, 3), keepdims=True)
        I = I - bg

        if dc is None:
            # approximate DC as average “brightfield” (mean over all patterns)
            dc = float(xp.mean(I))
        I = I / max(dc, 1e-12)

        # 3D FFT per pattern to reciprocal space
        self.I_hat_meas = xp.fft.fftn(I, axes=(-3, -2, -1), norm=self._fft_norm)

    # ===========================================================
    # Optimizer interface: forward, adjoint, cost, grad, prox, step
    # ===========================================================
    def fwd_model(self, x: array, inds: Optional[Sequence[int]] = None) -> array:
        xp = self.xp
        if inds is None:
            inds = range(4)
        V_hat = xp.fft.fftn(x, axes=(-3, -2, -1), norm=self._fft_norm)  # (Z,Y,X), complex
        Vhat_re = V_hat.real
        Vhat_im = V_hat.imag
        Hre = self.H_re[xp.asarray(inds)]
        Him = self.H_im[xp.asarray(inds)]
        return Hre * Vhat_re + Him * Vhat_im  # (len(inds),Z,Y,X), real

    def fwd_model_adjoint(self, y_hat: array, inds: Optional[Sequence[int]] = None) -> array:
        xp = self.xp
        if inds is None:
            inds = range(4)
        Hre = self.H_re[xp.asarray(inds)]
        Him = self.H_im[xp.asarray(inds)]
        # per-pattern k-space gradient parts (sum over patterns)
        ghat_re = xp.sum(Hre * y_hat, axis=0)
        ghat_im = xp.sum(Him * y_hat, axis=0)
        g_hat = ghat_re + 1j * ghat_im
        return xp.fft.ifftn(g_hat, axes=(-3, -2, -1), norm=self._fft_norm)

    def cost(self, x: array, inds: Optional[Sequence[int]] = None) -> array:
        """
        Per-pattern data term; Optimizer takes the MEAN across patterns,
        so the scalar descent criterion is automatically satisfied.
        """
        xp = self.xp
        if inds is None:
            inds = range(4)
        assert self.I_hat_meas is not None, "Call set_measurements(...) first."
        pred = self.fwd_model(x, inds=inds)
        resid = pred - self.I_hat_meas[xp.asarray(inds)]
        vol = self.Z * self.Y * self.X
        return 0.5 * xp.sum(xp.abs(resid) ** 2, axis=(-3, -2, -1)) / vol  # (len(inds),)

    def gradient(self, x: array, inds: Optional[Sequence[int]] = None) -> array:
        xp = self.xp
        if inds is None:
            inds = range(4)
        assert self.I_hat_meas is not None, "Call set_measurements(...) first."
        pred = self.fwd_model(x, inds=inds)
        resid = pred - self.I_hat_meas[xp.asarray(inds)]
        Hre = self.H_re[xp.asarray(inds)]
        Him = self.H_im[xp.asarray(inds)]
        ghat_re = Hre * resid
        ghat_im = Him * resid
        g_hat = ghat_re + 1j * ghat_im
        return xp.fft.ifftn(ghat_re + 1j * ghat_im, axes=(-3, -2, -1), norm=self._fft_norm)

    def prox(self, x: array, step: float) -> array:
        # reuse your provided helpers (tv_prox, soft_threshold, median_prox)
        xp = self.xp
        v_re = x.real
        v_im = x.imag

        tv_re = self.prox_parameters.get('tv_re', 0.0) or 0.0
        tv_im = self.prox_parameters.get('tv_im', 0.0) or 0.0
        if tv_re > 0:
            v_re = tv_prox(v_re, tau=step * tv_re)
        if tv_im > 0:
            v_im = tv_prox(v_im, tau=step * tv_im)

        soft_im = self.prox_parameters.get('soft_im', 0.0) or 0.0
        if soft_im > 0:
            v_im = soft_threshold(step * soft_im, v_im)

        med = self.prox_parameters.get('median', None)
        if med is not None:
            v_re = median_prox(v_re, size=med)
            v_im = median_prox(v_im, size=med)

        if self.prox_parameters.get('positivity_re', False):
            v_re = xp.maximum(v_re, 0)

        return v_re + 1j * v_im

    def guess_step(self, x: Optional[array] = None) -> float:
        if self._L_est is None:
            xp = self.xp
            power = xp.abs(self.H_re) ** 2 + xp.abs(self.H_im) ** 2  # (4,Z,Y,X)
            self._L_est = float(xp.max(xp.sum(power, axis=0)))  # max_k sum over patterns
            self._L_est = max(self._L_est, 1e-8)
        return 0.9 / self._L_est

    # ===========================================================
    # Internals: Pupil, Source, WOTF construction
    # ===========================================================
    def _make_pupil(self, defocus_waves: float = 0.0) -> array:
        """
        Ideal circular pupil (amplitude 1 inside NA). Optional defocus phase.
        Returns complex array of shape (Y, X).
        """
        xp = self.xp
        fy, fx = xp.meshgrid(self.fy, self.fx, indexing="ij")
        rho = xp.sqrt(fx**2 + fy**2) / self.k0_cyc
        P_amp = (rho <= self.na_obj).astype(self.xp.float32)
        if defocus_waves == 0.0:
            return P_amp.astype(self.xp.complex64)
        # Defocus Zernike ~ 2*r^2 - 1 on unit disk (simple quadratic phase)
        r = xp.clip(rho / self.na_obj, 0.0, 1.0)
        phase = (2.0 * r**2 - 1.0) * (2.0 * np.pi * defocus_waves)
        return P_amp * xp.exp(1j * phase)

    def _build_wotf_4patterns(self, src_grid_N: int = 65) -> tuple[array, array]:
        """
        Numerically assemble H_Re/H_Im on the reciprocal grid for 4 DPC patterns.

        Implementation notes
        --------------------
        * Non-paraxial kinematics: fz = kz(q+u) - kz(u),   kz(w) = sqrt(f0^2 - |w|^2),
          in *cycles/m* (no 2π). We splat both +fz and -fz contributions.
        * Absorption TF is even in z (add both signs); Phase TF is odd (subtract).
        * Amplitude-only pupil assumed (|P| in {0,1}). Aberration phase can be
          added in P if desired; mixing terms then require complex products.
        """
        xp = self.xp
        Z, Y, X = self.Z, self.Y, self.X

        # allocate shifted-z buffers for easy splatting, then unshift
        Hre_s = xp.zeros((4, Z, Y, X), dtype=self.xp.float32)
        Him_s = xp.zeros_like(Hre_s)

        f0 = self.k0_cyc
        fy, fx = xp.meshgrid(self.fy, self.fx, indexing="ij")  # (Y,X)

        # precompute |P(q+u)| mask efficiently by evaluating radius threshold
        P_amp = xp.abs(self.P)  # (Y,X) in {0,1}

        # discretize source pupil on a square grid, mask to circle of radius na_src
        uu = xp.linspace(-self.na_src * f0, self.na_src * f0, src_grid_N)
        uy_grid, ux_grid = xp.meshgrid(uu, uu, indexing="ij")
        src_r = xp.sqrt(ux_grid**2 + uy_grid**2)
        in_src = src_r <= (self.na_src * f0)

        # four half-circles: left (ux<0), right (ux>0), top (uy>0), bottom (uy<0)
        half_masks = [
            in_src & (ux_grid < 0),   # left
            in_src & (ux_grid > 0),   # right
            in_src & (uy_grid > 0),   # top
            in_src & (uy_grid < 0),   # bottom
        ]

        # weights: uniform over active source samples per pattern
        for pidx, src_mask in enumerate(half_masks):
            ux_list = ux_grid[src_mask]
            uy_list = uy_grid[src_mask]
            if ux_list.size == 0:
                continue
            w = xp.ones_like(ux_list, dtype=self.xp.float32)
            w /= float(ux_list.size)

            # iterate source samples (vectorized over q=(fy,fx))
            for uxi, uyi, wi in zip(ux_list, uy_list, w):
                # unscattered ray must pass through pupil |P(u)|>0
                Pup_u = (xp.sqrt((uxi)**2 + (uyi)**2) <= (self.na_obj * f0))
                if not bool(Pup_u):
                    continue

                # lateral shift q+u
                fx_shift = fx + uxi
                fy_shift = fy + uyi
                rad_shift = xp.sqrt(fx_shift**2 + fy_shift**2)

                # scattered ray must pass through pupil
                pass_mask = (rad_shift <= (self.na_obj * f0)).astype(self.xp.float32)
                if not xp.any(pass_mask):
                    continue

                # axial frequency difference (non-paraxial)
                kz_u = xp.sqrt(xp.maximum(0.0, f0**2 - (uxi**2 + uyi**2)))
                kz_qu = xp.sqrt(xp.maximum(0.0, f0**2 - rad_shift**2))
                dfz = kz_qu - kz_u  # (Y,X) cycles/m

                # contributions only where valid (pass_mask)
                A = wi * pass_mask  # amplitude weight; |P(q+u)|*|P(u)|=1 here

                # splat +dfz and -dfz onto shifted fz grid
                self._splat_plane(Hre_s, Him_s, pidx, A, +dfz)
                self._splat_plane(Hre_s, Him_s, pidx, A, -dfz)

        # unshift z -> native FFT ordering
        Hre = xp.fft.ifftshift(Hre_s, axes=(-3))
        Him = xp.fft.ifftshift(Him_s, axes=(-3))
        return Hre, Him

    def _splat_plane(self,
                     Hre_s: array,
                     Him_s: array,
                     pidx: int,
                     A_yx: array,
                     dfz_yx: array) -> None:
        """
        Deposit weights at z-planes nearest to dfz(y,x) on *shifted* fz axis.

        H_Im: add  +A
        H_Re: add  +A for +dfz, and  -A for -dfz (caller passes sign via dfz)
        """
        xp = self.xp
        Z, Y, X = self.Z, self.Y, self.X

        # map dfz -> fractional index on shifted axis
        # clamp to Nyquist
        dfz_clamped = xp.clip(dfz_yx, self.fz_min, self.fz_max)
        zf = (dfz_clamped - self.fz_min) / self.dfz  # fractional in [0, Z-1]
        z0 = xp.floor(zf).astype(self.xp.int32)
        z1 = xp.clip(z0 + 1, 0, Z - 1)
        alpha = (zf - z0).astype(self.xp.float32)

        # odd/even combination for phase/absorption
        # H_Im receives +A; H_Re receives sign(dfz)*A (odd in z)
        sign = xp.where(dfz_yx >= 0, 1.0, -1.0).astype(self.xp.float32)

        # Flatten indexing for scatter-add
        yy, xx = xp.meshgrid(xp.arange(Y), xp.arange(X), indexing="ij")
        w0 = (1.0 - alpha) * A_yx
        w1 = alpha * A_yx

        # absorption (even)
        self._add_at(Him_s, (pidx, z0, yy, xx), w0)
        self._add_at(Him_s, (pidx, z1, yy, xx), w1)

        # phase (odd)
        self._add_at(Hre_s, (pidx, z0, yy, xx), sign * w0)
        self._add_at(Hre_s, (pidx, z1, yy, xx), sign * w1)

    def _add_at(self, arr4: array, idxs: tuple, weights: array) -> None:
        """backend-safe scatter add into 4D [pattern,z,y,x] (shifted-z buffers)."""
        xp = self.xp
        p, z, y, x = idxs
        if self.use_gpu:
            # CuPy supports add.at
            cp.add.at(arr4, (p, z, y, x), weights)
        else:
            np.add.at(arr4, (p, z, y, x), np.asarray(weights))

    # -------------------------
    # Zarr v3 save / load
    # -------------------------
    def save_wotf_zarr(
        self,
        store_path: str,
        *,
        overwrite: bool = False,
        chunks: Optional[tuple[int, int, int, int]] = None,
        include_pupil: bool = False,
        compressor: Optional[dict] = None,
    ) -> None:
        """
        Save WOTFs to a Zarr v3 store on disk.

        Parameters
        ----------
        store_path : str
            Directory path for the Zarr store (e.g., "./wotf_cache.zarr").
        overwrite : bool
            If True, remove any existing store_path first.
        chunks : tuple[int, int, int, int] | None
            Chunk shape for arrays in [pattern,z,y,x]. Default is (1, min(16,Z), min(128,Y), min(128,X)).
        include_pupil : bool
            If True, also save Pupil (as P_real/P_imag with shape (Y, X)).
        compressor : dict | None
            Optional Zarr v3 compressor config, e.g. {"id": "zstd", "level": 3}.
        """
        if zarr is None:
            raise RuntimeError(f"zarr>=3 required but not available: {_zarr_err}")

        Z, Y, X = self.Z, self.Y, self.X
        H_re_cpu = _to_cpu(self.H_re).astype(np.float32, copy=False)
        H_im_cpu = _to_cpu(self.H_im).astype(np.float32, copy=False)

        if chunks is None:
            chunks = (1, min(16, Z), min(128, Y), min(128, X))

        # Prepare store directory
        if os.path.exists(store_path):
            if overwrite:
                # safest removal: remove directory tree
                import shutil
                shutil.rmtree(store_path)
            else:
                raise FileExistsError(f"{store_path} exists. Use overwrite=True to replace.")

        root = zarr.open_group(store_path, mode="w", zarr_version=3)

        # Root metadata
        meta = dict(
            version=1,
            order="pattern,z,y,x",
            shape_zyx=(Z, Y, X),
            voxel_size_zyx=(self.dz, self.dy, self.dx),
            wavelength=self.wavelength,
            n_medium=self.n,
            na_obj=self.na_obj,
            na_src=self.na_src,
            use_ortho_fft=bool(self.use_ortho_fft),
            dtype="float32",
            notes="H_re/H_im are real-valued WOTFs; pupil saved as P_real/P_imag if requested.",
        )
        root.attrs["dpc3d_meta_json"] = json.dumps(meta)

        # Create arrays
        aHre = root.create_array(
            "H_re",
            shape=(4, Z, Y, X),
            chunks=chunks,
            dtype="f4",
            compressor=compressor,
        )
        aHim = root.create_array(
            "H_im",
            shape=(4, Z, Y, X),
            chunks=chunks,
            dtype="f4",
            compressor=compressor,
        )

        aHre[:] = H_re_cpu
        aHim[:] = H_im_cpu

        if include_pupil:
            P = _to_cpu(self.P)
            root.create_array("P_real", shape=(Y, X), chunks=(min(256, Y), min(256, X)),
                              dtype="f4", compressor=compressor)[:] = P.real.astype(np.float32, copy=False)
            root.create_array("P_imag", shape=(Y, X), chunks=(min(256, Y), min(256, X)),
                              dtype="f4", compressor=compressor)[:] = P.imag.astype(np.float32, copy=False)

    def load_wotf_from_zarr(
        self,
        store_path: str,
        *,
        strict_geometry: bool = True,
        map_to_gpu: Optional[bool] = None,
    ) -> None:
        """
        Load WOTFs from a Zarr v3 store and install into this instance.

        Parameters
        ----------
        store_path : str
            Directory path of the Zarr store (e.g., "./wotf_cache.zarr").
        strict_geometry : bool
            If True, validate Z,Y,X and optics (λ, n, NA) against this instance.
        map_to_gpu : bool | None
            If True (and CuPy available), map arrays to GPU. If None, follow current instance backend.
        """
        if zarr is None:
            raise RuntimeError(f"zarr>=3 required but not available: {_zarr_err}")

        root = zarr.open_group(store_path, mode="r", zarr_version=3)

        # Load metadata
        meta_json = root.attrs.get("dpc3d_meta_json", "{}")
        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = {}

        # Read arrays (NumPy on load)
        H_re_np = np.asarray(root["H_re"][:], dtype=np.float32)
        H_im_np = np.asarray(root["H_im"][:], dtype=np.float32)

        if strict_geometry:
            mZ, mY, mX = tuple(meta.get("shape_zyx", ()))
            if (mZ, mY, mX) != (self.Z, self.Y, self.X):
                raise ValueError(f"Cached WOTF shape ZYX {mZ,mY,mX} != current {(self.Z,self.Y,self.X)}")
            # optics check (tolerant)
            def _close(a, b, tol=1e-9) -> bool:
                return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
            if not (
                _close(meta.get("wavelength", self.wavelength), self.wavelength) and
                _close(meta.get("n_medium", self.n), self.n) and
                _close(meta.get("na_obj", self.na_obj), self.na_obj) and
                _close(meta.get("na_src", self.na_src), self.na_src)
            ):
                raise ValueError("Cached optics (λ, n, NA) differ from current instance.")

        # Map to GPU if requested
        use_gpu = self.use_gpu if map_to_gpu is None else (bool(map_to_gpu) and (cp is not None))
        if use_gpu:
            self.H_re = cp.asarray(H_re_np)
            self.H_im = cp.asarray(H_im_np)
            # Optional pupil
            if "P_real" in root and "P_imag" in root:
                Pre = cp.asarray(np.asarray(root["P_real"][:], dtype=np.float32))
                Pim = cp.asarray(np.asarray(root["P_imag"][:], dtype=np.float32))
                self.P = (Pre + 1j * Pim).astype(cp.complex64, copy=False)
        else:
            self.H_re = H_re_np
            self.H_im = H_im_np
            if "P_real" in root and "P_imag" in root:
                Pre = np.asarray(root["P_real"][:], dtype=np.float32)
                Pim = np.asarray(root["P_imag"][:], dtype=np.float32)
                self.P = (Pre + 1j * Pim).astype(np.complex64, copy=False)

        self._L_est = None  # recompute step bound on next guess_step()

    @classmethod
    def from_cached_wotf_zarr(
        cls,
        store_path: str,
        *,
        shape_zyx: tuple[int, int, int],
        voxel_size_zyx: tuple[float, float, float],
        wavelength: float,
        n_medium: float,
        na_obj: float,
        na_src: float,
        prox_parameters: Optional[dict] = None,
        use_ortho_fft: bool = True,
        use_gpu: bool = False,
    ):
        """
        Construct a DPC3D instance and populate WOTFs from a Zarr v3 cache.

        Notes
        -----
        Pupil is rebuilt analytically; if P_real/P_imag exist in the store, they
        overwrite the analytic P for exact reproducibility.
        """
        obj = cls(
            shape_zyx=shape_zyx,
            voxel_size_zyx=voxel_size_zyx,
            wavelength=wavelength,
            n_medium=n_medium,
            na_obj=na_obj,
            na_src=na_src,
            I_meas_4p=None,
            prox_parameters=prox_parameters,
            src_grid_N=3,            # placeholder; WOTF will be loaded
            defocus_waves=0.0,
            use_ortho_fft=use_ortho_fft,
            use_gpu=use_gpu,
        )
        obj.load_wotf_from_zarr(store_path, strict_geometry=True, map_to_gpu=use_gpu)
        return obj

if __name__ == "__main__":
    xp = cp if cp is not None else np  # choose GPU if available

    # Geometry (example)
    shape = (100, 256, 256)           # Z,Y,X voxels
    voxel = (1e-6, 0.2e-6, 0.2e-6)    # dz,dy,dx in meters
    λ = 520e-9                        # wavelength (m)
    n0 = 1.33                         # medium RI
    NA_obj = 0.65
    NA_src = 0.65

    # Build problem
    dpc = DPC3D(shape, voxel, λ, n0, NA_obj, NA_src,
                I_meas_4p=your_raw_4pattern_stacks,  # (4,Z,Y,X) in space
                prox_parameters=dict(tv_re=1e-3, positivity_re=True),
                src_grid_N=65,
                use_ortho_fft=True,
                use_gpu=(cp is not None))

    # Initial guess and step
    x0 = xp.zeros(shape, dtype=xp.complex64)
    step = dpc.guess_step()

    # Run APGD/FISTA
    res = dpc.run(x_start=x0,
                step=step,
                max_iterations=200,
                use_fista=True,
                n_batch=None,  # use all 4 patterns each iter
                compute_batch_grad_parallel=True,
                compute_cost=True,
                compute_all_costs=True,
                line_search_iter_limit=25,
                line_search_factor=0.5,
                xtol=1e-4,
                label="[DPC3D] ")

    V_est = res["x"]  # complex scattering potential (Z,Y,X)