#!/usr/bin/env python3
"""
Archdragon solver ansatz generation.

This module provides two topologically distinct initial conditions for a
relaxation-based vortex solver:

1. `get_lepton_n1_ansatz` implements a single-core Rankine vortex extruded
   along the z-axis. This is the physically stable n=1 baseline.
2. `get_baryon_n3_ansatz` superposes three displaced cores with a braided
   axial flux tube to lock in the n=3 topology.

Both routines operate on structured Cartesian grids and can be invoked
from the command line to visualise or persist the fields.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

GridTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]
VectorField = np.ndarray

DEFAULT_GRID_SHAPE: Tuple[int, int, int] = (96, 96, 64)
DEFAULT_RADIAL_EXTENT: float = 6.0
DEFAULT_AXIAL_LENGTH: float = 8.0
_EPSILON: float = 1e-12


def build_cartesian_grid(
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
    radial_extent: float = DEFAULT_RADIAL_EXTENT,
    axial_length: float = DEFAULT_AXIAL_LENGTH,
    dtype: np.dtype | str = np.float64,
) -> GridTuple:
    """
    Construct a dense Cartesian meshgrid centred on the origin.

    Args:
        grid_shape: Number of sample points along (x, y, z).
        radial_extent: Extent for x and y axes (domain is [-extent, +extent]).
        axial_length: Total physical length along z (domain is [-L/2, +L/2]).
        dtype: Floating-point dtype for the grid.

    Returns:
        Tuple of arrays (X, Y, Z) with identical shapes.
    """
    if len(grid_shape) != 3:
        raise ValueError("grid_shape must contain exactly three integers (Nx, Ny, Nz).")
    nx, ny, nz = (int(n) for n in grid_shape)
    if nx <= 1 or ny <= 1 or nz <= 1:
        raise ValueError("grid_shape dimensions must be greater than one.")
    if radial_extent <= 0:
        raise ValueError("radial_extent must be positive.")
    if axial_length <= 0:
        raise ValueError("axial_length must be positive.")

    dtype = np.dtype(dtype)
    x = np.linspace(-radial_extent, radial_extent, nx, dtype=dtype)
    y = np.linspace(-radial_extent, radial_extent, ny, dtype=dtype)
    z = np.linspace(-axial_length / 2.0, axial_length / 2.0, nz, dtype=dtype)
    return np.meshgrid(x, y, z, indexing="ij")


def _prepare_grid(
    grid: Optional[GridTuple],
    dtype: Optional[np.dtype | str],
    grid_shape: Sequence[int],
    radial_extent: float,
    axial_length: float,
) -> Tuple[GridTuple, np.dtype]:
    if dtype is not None:
        dtype = np.dtype(dtype)

    if grid is None:
        dtype = np.dtype(np.float64 if dtype is None else dtype)
        grid_arrays = build_cartesian_grid(
            grid_shape=grid_shape,
            radial_extent=radial_extent,
            axial_length=axial_length,
            dtype=dtype,
        )
    else:
        if not isinstance(grid, (tuple, list)) or len(grid) != 3:
            raise TypeError("grid must be a tuple or list containing (X, Y, Z) arrays.")
        arrays = [
            np.asarray(component, dtype=dtype) if dtype is not None else np.asarray(component)
            for component in grid
        ]
        shape = arrays[0].shape
        if arrays[1].shape != shape or arrays[2].shape != shape:
            raise ValueError("All grid components must share the same shape.")

        if dtype is None:
            dtype = arrays[0].dtype
        else:
            arrays = [component.astype(dtype, copy=False) for component in arrays]

        grid_arrays = (arrays[0], arrays[1], arrays[2])

    return grid_arrays, np.dtype(dtype)


def _resolve_axial_scale(z: np.ndarray, axial_scale: Optional[float]) -> float:
    if axial_scale is not None:
        if axial_scale <= 0:
            raise ValueError("axial_scale must be strictly positive.")
        return float(axial_scale)

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    length = z_max - z_min
    if not np.isfinite(length) or length <= 0:
        fallback = float(np.mean(np.abs(z)))
        length = fallback if fallback > 0 else 1.0
    return length


def _rankine_vortex_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    core_radius: float,
    axial_scale: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    intensity: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if core_radius <= 0:
        raise ValueError("core_radius must be strictly positive.")

    cx, cy, cz = center
    dx = X - cx
    dy = Y - cy
    dz = Z - cz

    r_sq = dx * dx + dy * dy
    r = np.sqrt(r_sq)
    core_radius_sq = core_radius * core_radius
    fade = r / (core_radius_sq + r_sq + _EPSILON)
    z_profile = np.exp(-(dz * dz) / (axial_scale * axial_scale))

    u_x = -dy * fade * z_profile * intensity
    u_y = dx * fade * z_profile * intensity
    u_z = np.zeros_like(u_x)
    return u_x, u_y, u_z


def get_lepton_n1_ansatz(
    grid: Optional[GridTuple] = None,
    *,
    core_radius: float = 1.0,
    axial_scale: Optional[float] = None,
    dtype: Optional[np.dtype | str] = None,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
    radial_extent: float = DEFAULT_RADIAL_EXTENT,
    axial_length: float = DEFAULT_AXIAL_LENGTH,
) -> VectorField:
    """
    Generate the n=1 lepton ansatz (single Rankine vortex tube).

    Args:
        grid: Optional precomputed (X, Y, Z) meshgrid.
        core_radius: Rankine vortex core radius.
        axial_scale: Characteristic decay length along z. If None, the span of
            the supplied grid is used.
        dtype: Desired floating-point dtype. Uses grid dtype if supplied.
        grid_shape, radial_extent, axial_length: Parameters for grid creation
            when `grid` is not provided.

    Returns:
        Array of shape (*grid_shape, 3) containing velocity components.
    """
    (X, Y, Z), dtype = _prepare_grid(grid, dtype, grid_shape, radial_extent, axial_length)
    axial_scale_value = _resolve_axial_scale(Z, axial_scale)

    u_x, u_y, u_z = _rankine_vortex_field(
        X,
        Y,
        Z,
        core_radius=core_radius,
        axial_scale=axial_scale_value,
        center=(0.0, 0.0, 0.0),
        intensity=1.0,
    )
    return np.stack(
        (
            u_x.astype(dtype, copy=False),
            u_y.astype(dtype, copy=False),
            u_z.astype(dtype, copy=False),
        ),
        axis=-1,
    )


def get_baryon_n3_ansatz(
    grid: Optional[GridTuple] = None,
    *,
    core_radius: float = 1.0,
    proton_radius: float = 2.0,
    braid_amplitude: float = 1.0,
    axial_scale: Optional[float] = None,
    dtype: Optional[np.dtype | str] = None,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
    radial_extent: float = DEFAULT_RADIAL_EXTENT,
    axial_length: float = DEFAULT_AXIAL_LENGTH,
) -> VectorField:
    """
    Generate the n=3 baryon ansatz featuring three bound vortex cores and a braid.

    Args:
        grid: Optional precomputed (X, Y, Z) meshgrid.
        core_radius: Core radius for each constituent vortex.
        proton_radius: Distance from the origin to each vortex core centre.
        braid_amplitude: Scalar multiplier for the torsional flux tube.
        axial_scale: Characteristic decay length along z.
        dtype: Desired floating-point dtype. Uses grid dtype if supplied.
        grid_shape, radial_extent, axial_length: Grid parameters when creating
            a mesh internally.

    Returns:
        Array of shape (*grid_shape, 3) containing the composite velocity field.
    """
    if proton_radius <= 0:
        raise ValueError("proton_radius must be strictly positive.")

    (X, Y, Z), dtype = _prepare_grid(grid, dtype, grid_shape, radial_extent, axial_length)
    axial_scale_value = _resolve_axial_scale(Z, axial_scale)

    u_x_total = np.zeros_like(X)
    u_y_total = np.zeros_like(Y)
    u_z_total = np.zeros_like(Z)

    angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    centers = tuple(
        (proton_radius * np.cos(theta), proton_radius * np.sin(theta), 0.0)
        for theta in angles
    )

    for center in centers:
        u_x_core, u_y_core, _ = _rankine_vortex_field(
            X,
            Y,
            Z,
            core_radius=core_radius,
            axial_scale=axial_scale_value,
            center=center,
            intensity=1.0,
        )
        u_x_total += u_x_core
        u_y_total += u_y_core

    theta = np.arctan2(Y, X)
    r_sq = X * X + Y * Y
    radial_envelope = np.exp(-r_sq / (proton_radius * proton_radius))
    axial_envelope = np.exp(-(Z * Z) / (axial_scale_value * axial_scale_value))
    u_z_total = braid_amplitude * np.sin(3.0 * theta) * radial_envelope * axial_envelope

    return np.stack(
        (
            u_x_total.astype(dtype, copy=False),
            u_y_total.astype(dtype, copy=False),
            u_z_total.astype(dtype, copy=False),
        ),
        axis=-1,
    )


def _summarise_field(field: VectorField) -> Dict[str, float]:
    speeds = np.linalg.norm(field, axis=-1)
    summary = {
        "points": float(field.shape[0] * field.shape[1] * field.shape[2]),
        "speed_min": float(np.min(speeds)),
        "speed_max": float(np.max(speeds)),
        "speed_mean": float(np.mean(speeds)),
        "speed_rms": float(np.sqrt(np.mean(speeds * speeds))),
        "mean_u_z": float(np.mean(field[..., 2])),
        "mean_abs_u_z": float(np.mean(np.abs(field[..., 2]))),
    }
    return summary


def _print_summary(ansatz_name: str, field: VectorField) -> Dict[str, float]:
    summary = _summarise_field(field)
    print(f"Generated {ansatz_name} ansatz on grid {field.shape[:-1]}")
    for key in sorted(summary.keys()):
        value = summary[key]
        print(f"  {key:>12}: {value: .6e}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Archdragon vortex ansatz fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ansatz",
        choices=("lepton_n1", "baryon_n3"),
        default="lepton_n1",
        help="Which ansatz to generate.",
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=DEFAULT_GRID_SHAPE,
        help="Number of grid samples along each axis.",
    )
    parser.add_argument(
        "--radial-extent",
        type=float,
        default=DEFAULT_RADIAL_EXTENT,
        help="Half-width of the domain in x and y.",
    )
    parser.add_argument(
        "--axial-length",
        type=float,
        default=DEFAULT_AXIAL_LENGTH,
        help="Full length of the domain in z.",
    )
    parser.add_argument(
        "--core-radius",
        type=float,
        default=1.0,
        help="Core radius for constituent vortices.",
    )
    parser.add_argument(
        "--proton-radius",
        type=float,
        default=2.0,
        help="Radial displacement of baryon cores (baryon_n3 only).",
    )
    parser.add_argument(
        "--braid-amplitude",
        type=float,
        default=1.0,
        help="Amplitude of the torsional braid component (baryon_n3 only).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Floating-point precision for generated arrays.",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Optional path to save the field and metadata as a compressed .npz file.",
    )

    args = parser.parse_args()

    dtype = np.float64 if args.dtype == "float64" else np.float32
    grid = build_cartesian_grid(
        grid_shape=args.grid_shape,
        radial_extent=args.radial_extent,
        axial_length=args.axial_length,
        dtype=dtype,
    )

    if args.ansatz == "lepton_n1":
        field = get_lepton_n1_ansatz(
            grid=grid,
            core_radius=args.core_radius,
            axial_scale=None,
            dtype=dtype,
        )
    else:
        field = get_baryon_n3_ansatz(
            grid=grid,
            core_radius=args.core_radius,
            proton_radius=args.proton_radius,
            braid_amplitude=args.braid_amplitude,
            axial_scale=None,
            dtype=dtype,
        )

    summary = _print_summary(args.ansatz, field)

    if args.save:
        metadata = {
            "ansatz": args.ansatz,
            "grid_shape": tuple(int(dim) for dim in args.grid_shape),
            "radial_extent": float(args.radial_extent),
            "axial_length": float(args.axial_length),
            "core_radius": float(args.core_radius),
            "proton_radius": float(args.proton_radius),
            "braid_amplitude": float(args.braid_amplitude),
            "summary": summary,
        }
        np.savez_compressed(
            args.save,
            field=field,
            x=grid[0],
            y=grid[1],
            z=grid[2],
            metadata=json.dumps(metadata),
        )
        print(f"Saved field and grid to {args.save}")


if __name__ == "__main__":
    main()
