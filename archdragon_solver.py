"""High fidelity initial states for the Archdragon solver.

This module provides two velocity-field constructors that serve as
physically-motivated starting points for the Archdragon variational solver.
Both fields are intentionally designed to be topologically distinct so that
subsequent relaxation steps cannot collapse them into the same minimum-energy
configuration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["get_lepton_n1_ansatz", "get_baryon_n3_ansatz"]

VectorField = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _as_array(array: np.ndarray) -> np.ndarray:
    """Return *array* as a NumPy ndarray (without copying if unnecessary)."""
    return np.asarray(array, dtype=float)


def _validate_grid_components(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Ensure the three position grids share the same dimensionality."""
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            "Grid component shape mismatch: "
            f"x{ x.shape }, y{ y.shape }, z{ z.shape } must be identical"
        )


def _infer_length_scale(z: np.ndarray) -> float:
    """Infer a characteristic length scale from the z-grid."""
    span = float(np.max(z) - np.min(z))
    if span > 0:
        return span

    extent = float(np.max(np.abs(z)))
    if extent > 0:
        return 2.0 * extent

    # Degenerate grid (e.g. single plane); fall back to unit length.
    return 1.0


def _rankine_vortex(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    core_radius: float,
    length_scale: float,
) -> VectorField:
    """Construct a smooth Rankine vortex centred at the origin."""
    if core_radius <= 0:
        raise ValueError("core_radius must be positive")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")

    r = np.sqrt(x**2 + y**2)
    f = r / (core_radius**2 + r**2)
    g = np.exp(-(z**2) / (length_scale**2))

    u_x = -y * f * g
    u_y = x * f * g
    u_z = np.zeros_like(u_x)
    return u_x, u_y, u_z


def get_lepton_n1_ansatz(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    *,
    core_radius: float = 1.0,
    length_scale: float | None = None,
) -> VectorField:
    """Return a smooth n=1 vortex tube suitable for the reference lepton state.

    Parameters
    ----------
    grid_x, grid_y, grid_z
        Cartesian position grids with identical shapes.
    core_radius
        Rankine vortex core radius (:math:`R_c`).
    length_scale
        Characteristic decay length for the z-fade. If omitted, it is inferred
        from the supplied z-grid.
    """
    x = _as_array(grid_x)
    y = _as_array(grid_y)
    z = _as_array(grid_z)
    _validate_grid_components(x, y, z)

    z_scale = _infer_length_scale(z) if length_scale is None else float(length_scale)
    return _rankine_vortex(x, y, z, core_radius=float(core_radius), length_scale=z_scale)


def get_baryon_n3_ansatz(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    *,
    core_radius: float = 1.0,
    proton_radius: float = 2.0,
    length_scale: float | None = None,
    braid_amplitude: float = 1.0,
) -> VectorField:
    """Return the braided n=3 baryon ansatz.

    The result is the superposition of three displaced Rankine vortices
    (quark cores) and a torsional shear flow that induces a three-fold braid
    along the z-axis.
    """
    if proton_radius <= 0:
        raise ValueError("proton_radius must be positive")

    x = _as_array(grid_x)
    y = _as_array(grid_y)
    z = _as_array(grid_z)
    _validate_grid_components(x, y, z)

    z_scale = _infer_length_scale(z) if length_scale is None else float(length_scale)
    c_radius = float(core_radius)

    # Initialise accumulator for the superposed cores.
    u_x = np.zeros_like(x, dtype=float)
    u_y = np.zeros_like(y, dtype=float)
    u_z = np.zeros_like(z, dtype=float)

    # Equilateral triangle placement (centres on xy-plane).
    centres = (
        (proton_radius, 0.0, 0.0),
        (
            proton_radius * np.cos(2.0 * np.pi / 3.0),
            proton_radius * np.sin(2.0 * np.pi / 3.0),
            0.0,
        ),
        (
            proton_radius * np.cos(4.0 * np.pi / 3.0),
            proton_radius * np.sin(4.0 * np.pi / 3.0),
            0.0,
        ),
    )

    for cx, cy, cz in centres:
        vortex = _rankine_vortex(
            x - cx,
            y - cy,
            z - cz,
            core_radius=c_radius,
            length_scale=z_scale,
        )
        u_x += vortex[0]
        u_y += vortex[1]
        # u_z contribution is identically zero for the Rankine vortex.

    # Braid (flux tube) component.
    theta = np.arctan2(y, x)
    radial = np.sqrt(x**2 + y**2)
    g = np.exp(-(z**2) / (z_scale**2))
    confinement = np.exp(-(radial**2) / (proton_radius**2))
    u_z += braid_amplitude * np.sin(3.0 * theta) * confinement * g

    return u_x, u_y, u_z
