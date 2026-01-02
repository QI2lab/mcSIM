import numpy as np
import pytest

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None

from mcsim.analysis.dpc_inverse import DPCGeometry, DPCRytovInverse


def _make_simple_solver(ny=16, nx=16, n_planes=2, use_gpu: bool = False, delta: float = 1e-3):
    # Four LEDs, one per half-plane
    led_na = np.array(
        [
            [-0.1, 0.0],  # left
            [0.1, 0.0],   # right
            [0.0, 0.1],   # up
            [0.0, -0.1],  # down
        ],
        dtype=float,
    )

    geom = DPCGeometry(
        wavelength_um=0.5,
        n_medium=1.0,
        na_obj=0.8,
        camera_pixel_um=2.0,
        magnification=10.0,
        focal_offsets_um=[0.0] * n_planes,
        pattern_order=("left", "right", "up", "down"),
        normalize_by_led_count=True,
    )

    n_shape = (2, ny, nx)
    drs_n = (0.5, geom.dxy_um, geom.dxy_um)

    # simple ground truth RI
    n0 = np.full(n_shape, geom.n_medium, dtype=np.float32)
    n0[0, ny // 2, nx // 2] += float(delta)  # small perturbation
    if use_gpu and cp is not None:
        n0 = cp.asarray(n0)

    # simulate data from the forward model
    solver = DPCRytovInverse(
        np.zeros((n_planes, 4, ny, nx), dtype=np.float32),
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=use_gpu,
    )
    dpc_pred, _ = solver._predict_fields(n0)

    # re-instantiate with the simulated data
    solver = DPCRytovInverse(
        dpc_pred,
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=use_gpu,
    )
    return solver, n0


@pytest.mark.parametrize("use_gpu", [False] if cp is None else [False, True])
def test_forward_shape_and_values(use_gpu):
    solver, n0 = _make_simple_solver(use_gpu=use_gpu)
    pred, _ = solver._predict_fields(n0)
    assert pred.shape == solver.data.shape == (solver.n_planes, 4, solver.ny, solver.nx)
    # Forward evaluated at ground truth should match data
    np.testing.assert_allclose(
        cp.asnumpy(pred) if cp and use_gpu else pred,
        cp.asnumpy(solver.data) if cp and use_gpu else solver.data,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("use_gpu", [False] if cp is None else [False, True])
def test_gradient_matches_numeric(use_gpu):
    solver, n0 = _make_simple_solver(use_gpu=use_gpu)
    g, gn = solver.test_gradient(n0, jind=0, dx=1e-6)
    g_np = cp.asnumpy(g) if cp and use_gpu else g
    gn_np = cp.asnumpy(gn) if cp and use_gpu else gn
    np.testing.assert_allclose(g_np, gn_np, rtol=1e-3, atol=1e-5)


def _make_multiled_solver(ny=16, nx=16, n_planes=2, use_gpu: bool = False):
    # Multiple LEDs per half-plane
    led_na = np.array(
        [
            [-0.15, 0.0],
            [-0.05, 0.05],
            [0.15, 0.0],
            [0.05, -0.05],
            [0.0, 0.15],
            [-0.05, 0.1],
            [0.0, -0.15],
            [0.05, -0.1],
        ],
        dtype=float,
    )

    geom = DPCGeometry(
        wavelength_um=0.5,
        n_medium=1.0,
        na_obj=0.8,
        camera_pixel_um=2.0,
        magnification=10.0,
        focal_offsets_um=[0.0] * n_planes,
        pattern_order=("left", "right", "up", "down"),
        normalize_by_led_count=True,
    )

    n_shape = (2, ny, nx)
    drs_n = (0.5, geom.dxy_um, geom.dxy_um)

    n0 = np.full(n_shape, geom.n_medium, dtype=np.float32)
    n0[1, ny // 2, nx // 2] += 5e-4
    if use_gpu and cp is not None:
        n0 = cp.asarray(n0)

    solver = DPCRytovInverse(
        np.zeros((n_planes, 4, ny, nx), dtype=np.float32),
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=use_gpu,
    )
    dpc_pred, _ = solver._predict_fields(n0)

    solver = DPCRytovInverse(
        dpc_pred,
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=use_gpu,
    )
    return solver, n0


@pytest.mark.parametrize("use_gpu", [False] if cp is None else [False, True])
def test_forward_multiled_shape_and_values(use_gpu):
    solver, n0 = _make_multiled_solver(use_gpu=use_gpu)
    pred, _ = solver._predict_fields(n0)
    assert pred.shape == solver.data.shape == (solver.n_planes, 4, solver.ny, solver.nx)
    np.testing.assert_allclose(
        cp.asnumpy(pred) if cp and use_gpu else pred,
        cp.asnumpy(solver.data) if cp and use_gpu else solver.data,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("use_gpu", [False] if cp is None else [False, True])
def test_gradient_multiled_matches_numeric(use_gpu):
    solver, n0 = _make_multiled_solver(use_gpu=use_gpu)
    g, gn = solver.test_gradient(n0, jind=0, dx=1e-6)
    g_np = cp.asnumpy(g) if cp and use_gpu else g
    gn_np = cp.asnumpy(gn) if cp and use_gpu else gn
    np.testing.assert_allclose(g_np, gn_np, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("use_gpu", [False] if cp is None else [False, True])
def test_reconstruction_improves_mse(use_gpu):
    solver, n_true = _make_simple_solver(ny=12, nx=12, n_planes=1, use_gpu=use_gpu, delta=5e-3)
    xp = cp if (use_gpu and cp is not None) else np

    n_init = xp.full_like(n_true, solver.geom.n_medium)

    step = solver.guess_step()
    res = solver.run(
        n_init,
        step=step,
        max_iterations=80,
        use_fista=True,
        compute_cost=False,
        verbose=False,
        compute_all_costs=False,
        line_search_iter_limit=None,
        label="recon-test ",
    )
    n_rec = res["x"]

    mse_init = xp.mean(xp.abs(n_init - n_true) ** 2)
    mse_final = xp.mean(xp.abs(n_rec - n_true) ** 2)
    # Expect a clear reduction
    assert float(mse_final) < 0.5 * float(mse_init)
