"""
Fast GPU-only 3D TV proximal operator (Chambolle dual projection) implemented
with CuPy + CUDA RawKernel.

Import into optimize.py, e.g.:

    from tv_prox_fast import tv_prox_fast

Design constraints:
- 3D only
- GPU/CuPy only (raises if not CuPy array)
- No positivity constraint
- Fixed-iteration behavior consistent with the uploaded prox_fgp (default num_iter=10)
"""

from functools import lru_cache
from typing import Any, Optional, Tuple

import cupy as cp

__all__ = ["tv_prox_fast"]


@lru_cache(maxsize=None)
def _tv_fast_kernels(
    dtype_str: str,
) -> Tuple[cp.RawKernel, cp.RawKernel, cp.RawKernel, cp.RawKernel]:
    """
    Compile and cache RawKernels used by tv_prox_fast().

    :param dtype_str: Data type string ("float32" or "float64").
    :return: Tuple of (discontig_sub_kernel, tv_norm_kernel).
    """
    if dtype_str == "float32":
        ctype = "float"
        sqrt_fn = "sqrtf"
    elif dtype_str == "float64":
        ctype = "double"
        sqrt_fn = "sqrt"
    else:
        raise ValueError(f"Unsupported dtype for tv_prox_fast: {dtype_str}")

    code = rf"""
    extern "C" __global__
    void discontig_sub(
        const {ctype}* __restrict__ arr,
        {ctype}* __restrict__ out,
        unsigned long long step,
        unsigned long long length,
        int transpose,
        unsigned long long n_total
    ) {{
        unsigned long long i = (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                             + (unsigned long long)threadIdx.x;
        if (i >= n_total) return;

        unsigned long long pos = i % length;
        if (!transpose) {{
            if (pos + step < length) {{
                out[i] = arr[i + step] - arr[i];
            }} else {{
                out[i] = ({ctype})0;
            }}
        }} else {{
            out[i] -= arr[i];
            if (pos + step < length) {{
                out[i + step] += arr[i];
            }}
        }}
    }}

    extern "C" __global__
    void tv_norm3(
        {ctype}* __restrict__ out,      // length n
        const {ctype}* __restrict__ tv, // length 3*n, packed [x|y|z]
        unsigned long long n
    ) {{
        unsigned long long i = (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                             + (unsigned long long)threadIdx.x;
        if (i >= n) return;

        {ctype} a = tv[i];
        {ctype} b = tv[i + n];
        {ctype} c = tv[i + 2*n];

        out[i] = {sqrt_fn}(a*a + b*b + c*c);
    }}
    """

    code_grad = rf"""
    extern "C" __global__
    void tv_grad_update(
        const {ctype}* __restrict__ out,
        {ctype}* __restrict__ p,
        {ctype}* __restrict__ norms,
        {ctype} step,
        {ctype} weight,
        {ctype} w_z,
        {ctype} w_y,
        {ctype} w_x,
        unsigned long long nz,
        unsigned long long ny,
        unsigned long long nx,
        unsigned long long n_total,
        int write_norms
    ) {{
        unsigned long long i = (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                             + (unsigned long long)threadIdx.x;
        if (i >= n_total) return;

        unsigned long long plane = ny * nx;
        unsigned long long z = i / plane;
        unsigned long long rem = i - z * plane;
        unsigned long long y = rem / nx;
        unsigned long long x = rem - y * nx;

        {ctype} g0 = ({ctype})0;
        {ctype} g1 = ({ctype})0;
        {ctype} g2 = ({ctype})0;
        if (z + 1 < nz) {{
            g0 = (out[i + plane] - out[i]) * w_z;
        }}
        if (y + 1 < ny) {{
            g1 = (out[i + nx] - out[i]) * w_y;
        }}
        if (x + 1 < nx) {{
            g2 = (out[i + 1] - out[i]) * w_x;
        }}

        {ctype} norm = {sqrt_fn}(g0 * g0 + g1 * g1 + g2 * g2);
        {ctype} denom = ({ctype})1 + (step / weight) * norm;

        {ctype} p0 = p[i];
        {ctype} p1 = p[i + n_total];
        {ctype} p2 = p[i + 2 * n_total];

        p[i] = (p0 - step * g0) / denom;
        p[i + n_total] = (p1 - step * g1) / denom;
        p[i + 2 * n_total] = (p2 - step * g2) / denom;
        if (write_norms) {{
            norms[i] = norm;
        }}
    }}
    """

    code_div = rf"""
    extern "C" __global__
    void tv_divergence(
        const {ctype}* __restrict__ p,
        const {ctype}* __restrict__ x,
        {ctype}* __restrict__ out,
        {ctype} w_z,
        {ctype} w_y,
        {ctype} w_x,
        unsigned long long nz,
        unsigned long long ny,
        unsigned long long nx,
        unsigned long long n_total
    ) {{
        unsigned long long i = (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
                             + (unsigned long long)threadIdx.x;
        if (i >= n_total) return;

        unsigned long long plane = ny * nx;
        unsigned long long z = i / plane;
        unsigned long long rem = i - z * plane;
        unsigned long long y = rem / nx;
        unsigned long long xind = rem - y * nx;

        {ctype} p0 = p[i];
        {ctype} p1 = p[i + n_total];
        {ctype} p2 = p[i + 2 * n_total];

        {ctype} val = -(w_z * p0 + w_y * p1 + w_x * p2);
        if (z > 0) {{
            val += w_z * p[i - plane];
        }}
        if (y > 0) {{
            val += w_y * p[i + n_total - nx];
        }}
        if (xind > 0) {{
            val += w_x * p[i + 2 * n_total - 1];
        }}
        out[i] = x[i] + val;
    }}
    """

    return (
        cp.RawKernel(code, "discontig_sub"),
        cp.RawKernel(code, "tv_norm3"),
        cp.RawKernel(code_div, "tv_divergence"),
        cp.RawKernel(code_grad, "tv_grad_update"),
    )


def _launch_1d(kernel: cp.RawKernel, n: int, args: Tuple[Any, ...]) -> None:
    """
    Launch a 1D CUDA kernel with a fixed thread block size.

    :param kernel: RawKernel to launch.
    :param n: Total number of elements.
    :param args: Kernel argument tuple.
    :return: None.
    """
    threads = 256
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), args)


def _discontig_sub_cupy(
    arr: cp.ndarray,
    out: cp.ndarray,
    axis: int,
    transpose: bool = False,
) -> cp.ndarray:
    """
    Flattened-addressing difference/adjoint operator, equivalent to fista.py discontig_sub.

    :param arr: Input array.
    :param out: Output array (overwritten for forward, accumulated for adjoint).
    :param axis: Axis along which to compute the flattened-order difference.
    :param transpose: False for forward, True for adjoint.
    :return: Output array.
    """
    if arr.dtype != out.dtype:
        raise ValueError("arr and out must have the same dtype")

    if not arr.flags.c_contiguous:
        arr = cp.ascontiguousarray(arr)
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")

    axis = axis % arr.ndim
    shape = arr.shape

    step = 1
    for s in shape[axis + 1 :]:
        step *= int(s)

    length = step * int(shape[axis])
    n_total = int(arr.size)

    discontig, _, _, _ = _tv_fast_kernels(str(arr.dtype))
    _launch_1d(
        discontig,
        n_total,
        (
            arr,
            out,
            cp.uint64(step),
            cp.uint64(length),
            cp.int32(1 if transpose else 0),
            cp.uint64(n_total),
        ),
    )
    return out


def tv_prox_fast(
    x: cp.ndarray,
    tau: float,
    num_iter: int = 10,
    eps: float = 0.0,
    voxel_size_zyx: Optional[Tuple[float, float, float]] = None,
    weight_scale_zyx: Optional[Tuple[float, float, float]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Fast GPU-only TV proximal operator for 3D arrays using a Chambolle dual-projection loop.

    This implements a proximal map:

    .. math::
        \\text{prox}_{\\tau \\, TV}(x) = \\arg\\min_y \\; 0.5\\|y - x\\|_2^2 + \\tau \\, TV(y)

    matching skimage/cucim's Chambolle formulation (no positivity projection).

    :param x: CuPy ndarray, 3D.
    :param tau: TV proximal weight (must be >= 0). tau==0 returns identity.
    :param num_iter: Fixed number of FGP iterations.
    :param eps: Relative stopping tolerance, matching cucim TV when > 0.
    :param voxel_size_zyx: Physical voxel size (dz, dy, dx). Defaults to (1, 1, 1).
    :param weight_scale_zyx: Multiplicative scaling for (dz, dy, dx) weights.
    :param out: Optional CuPy array to write the result into (must match shape/dtype).
    :return: Proximal output (same shape/dtype as x, with float32/float64 enforced).
    """
    if not isinstance(x, cp.ndarray):
        raise TypeError("tv_prox_fast is GPU-only: x must be a CuPy ndarray.")
    if x.ndim != 3:
        raise ValueError(f"tv_prox_fast is 3D-only; got x.shape={x.shape}")
    if tau < 0:
        raise ValueError("tau must be >= 0")

    if tau == 0.0:
        if out is None:
            return x
        out[...] = x
        return out

    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)

    if x.dtype not in (cp.float32, cp.float64):
        x = x.astype(cp.float32, copy=False)

    dtype = x.dtype
    if voxel_size_zyx is None:
        voxel_size_zyx = (1.0, 1.0, 1.0)
    if len(voxel_size_zyx) != 3:
        raise ValueError("voxel_size_zyx must be a 3-tuple (dz, dy, dx).")
    dz, dy, dx = (float(v) for v in voxel_size_zyx)
    if dz <= 0 or dy <= 0 or dx <= 0:
        raise ValueError("voxel_size_zyx entries must be > 0.")
    if weight_scale_zyx is None:
        weight_scale_zyx = (1.0, 1.0, 1.0)
    if len(weight_scale_zyx) != 3:
        raise ValueError("weight_scale_zyx must be a 3-tuple (sz, sy, sx).")
    sz, sy, sx = (float(v) for v in weight_scale_zyx)
    if sz <= 0 or sy <= 0 or sx <= 0:
        raise ValueError("weight_scale_zyx entries must be > 0.")
    w_z = 1.0 / dz
    w_y = 1.0 / dy
    w_x = 1.0 / dx
    w_z *= sz
    w_y *= sy
    w_x *= sx
    step = 1.0 / (2.0 * (w_z * w_z + w_y * w_y + w_x * w_x))

    p = cp.zeros((3,) + x.shape, dtype=dtype)
    norms = cp.empty_like(x) if eps > 0 else None
    proj = cp.empty_like(x) if out is None else out

    _, _, div_kernel, grad_kernel = _tv_fast_kernels(str(dtype))
    n_vox = int(x.size)
    nz, ny, nx = (int(s) for s in x.shape)
    step_val = dtype.type(step)
    weight_val = dtype.type(tau)
    wz_val = dtype.type(w_z)
    wy_val = dtype.type(w_y)
    wx_val = dtype.type(w_x)
    write_norms = cp.int32(1 if eps > 0 else 0)
    eps_val = float(eps)
    e_init = None
    e_prev = None

    for ii in range(int(num_iter)):
        if ii > 0:
            _launch_1d(
                div_kernel,
                n_vox,
            (
                p,
                x,
                proj,
                wz_val,
                wy_val,
                wx_val,
                cp.uint64(nz),
                cp.uint64(ny),
                cp.uint64(nx),
                cp.uint64(n_vox),
            ),
            )
        else:
            proj[...] = x

        _launch_1d(
            grad_kernel,
            n_vox,
            (
                proj,
                p,
                norms if norms is not None else proj,
                step_val,
                weight_val,
                wz_val,
                wy_val,
                wx_val,
                cp.uint64(nz),
                cp.uint64(ny),
                cp.uint64(nx),
                cp.uint64(n_vox),
                write_norms,
            ),
        )

        if eps_val > 0.0:
            d = proj - x
            e = cp.sum(d * d)
            e += weight_val * cp.sum(norms)
            e /= float(n_vox)
            e_host = float(e)
            if ii == 0:
                e_init = e_host
                e_prev = e_host
            else:
                if e_init is not None and e_prev is not None:
                    if abs(e_prev - e_host) < eps_val * e_init:
                        break
                    e_prev = e_host

    return proj
