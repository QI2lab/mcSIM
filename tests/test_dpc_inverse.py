import numpy as np
import pytest

from mcsim.analysis.dpc_inverse import DPCGeometry, DPCRytovInverse


def _make_simple_solver(ny=16, nx=16, n_planes=2):
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
    n0[0, ny // 2, nx // 2] += 1e-3  # small perturbation

    # simulate data from the forward model
    solver = DPCRytovInverse(
        np.zeros((n_planes, 4, ny, nx), dtype=np.float32),
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=False,
    )
    dpc_pred, _ = solver._predict_fields(n0)

    # re-instantiate with the simulated data
    solver = DPCRytovInverse(
        dpc_pred,
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=False,
    )
    return solver, n0


def test_forward_shape_and_values():
    solver, n0 = _make_simple_solver()
    pred, _ = solver._predict_fields(n0)
    assert pred.shape == solver.data.shape == (solver.n_planes, 4, solver.ny, solver.nx)
    # Forward evaluated at ground truth should match data
    np.testing.assert_allclose(pred, solver.data, rtol=1e-5, atol=1e-6)


def test_gradient_matches_numeric():
    solver, n0 = _make_simple_solver()
    g, gn = solver.test_gradient(n0, jind=0, dx=1e-6)
    np.testing.assert_allclose(g, gn, rtol=1e-3, atol=1e-5)


def _make_multiled_solver(ny=16, nx=16, n_planes=2):
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

    solver = DPCRytovInverse(
        np.zeros((n_planes, 4, ny, nx), dtype=np.float32),
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=False,
    )
    dpc_pred, _ = solver._predict_fields(n0)

    solver = DPCRytovInverse(
        dpc_pred,
        led_na,
        geom=geom,
        n_shape=n_shape,
        drs_n=drs_n,
        use_gpu=False,
    )
    return solver, n0


def test_forward_multiled_shape_and_values():
    solver, n0 = _make_multiled_solver()
    pred, _ = solver._predict_fields(n0)
    assert pred.shape == solver.data.shape == (solver.n_planes, 4, solver.ny, solver.nx)
    np.testing.assert_allclose(pred, solver.data, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_gradient_multiled_matches_numeric():
    solver, n0 = _make_multiled_solver()
    g, gn = solver.test_gradient(n0, jind=0, dx=1e-6)
    np.testing.assert_allclose(g, gn, rtol=1e-3, atol=1e-5)
