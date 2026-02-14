"""
Minimal LED board geometry helpers for DPC simulations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LEDBoard:
    """
    LED board definition for NA-space mapping.

    :param n_side: Number of LEDs per side (square board).
    :param pitch_mm: LED pitch in mm.
    :param na_obj: Objective NA.
    :param wavelength_um: Wavelength in um.
    :param n_medium: Medium refractive index.
    """

    n_side: int
    pitch_mm: float
    na_obj: float
    wavelength_um: float
    n_medium: float


@dataclass(frozen=True)
class LEDGeometry:
    """
    Geometry container for LED NA positions.

    :param na_components: NA positions, shape (N, 2).
    """

    na_components: np.ndarray


def _make_led_na_positions(
    ny_led: int,
    nx_led: int,
    *,
    na_obj: float,
    inner_na: float = 0.0,
    include_center: bool = True,
) -> np.ndarray:
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


def compute_led_geometry(board: LEDBoard) -> LEDGeometry:
    """
    Compute LED NA positions for a square board.

    :param board: LED board definition.
    :return: LEDGeometry with NA components.
    """
    na_xy = _make_led_na_positions(
        int(board.n_side),
        int(board.n_side),
        na_obj=float(board.na_obj),
        inner_na=0.0,
        include_center=True,
    )
    return LEDGeometry(na_components=na_xy)
