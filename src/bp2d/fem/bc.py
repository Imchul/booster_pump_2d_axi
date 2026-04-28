"""Boundary conditions: pressure traction and Dirichlet displacement."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# 3-point 1D Gauss for edge integration
_GP_EDGE = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GW_EDGE = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def _edge_shape(xi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D quadratic shape functions over a Q8 edge with 3 nodes (xi=-1, 0, +1).

    Node order on the edge: (corner_a, mid, corner_c).
    Returns (L, dL).
    """
    L = np.array(
        [
            0.5 * xi * (xi - 1.0),
            1.0 - xi * xi,
            0.5 * xi * (xi + 1.0),
        ]
    )
    dL = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
    return L, dL


def pressure_force(
    nodes: NDArray[np.float64],
    edges: NDArray[np.int64],
    pressure: float,
) -> NDArray[np.float64]:
    """Assemble equivalent nodal force vector from internal pressure on edges.

    Convention: edges traverse the *solid* boundary CCW so the solid is on the
    left of the tangent direction. Outward normal then points right of the
    tangent, and pressure traction = -pressure * n_outward (compressive into
    solid). Sign of `pressure` is positive for compressive (internal) pressure.

    Args:
        nodes: (n_nodes, 2) array of (r, z).
        edges: (n_edges, 3) array of node indices in CCW order
            [corner_a, mid, corner_c].
        pressure: scalar pressure value (Pa). Positive = compressive on solid.

    Returns:
        F: (2*n_nodes,) global force vector contribution.
    """
    n_dof = 2 * nodes.shape[0]
    F = np.zeros(n_dof)
    two_pi = 2.0 * np.pi

    for edge in edges:
        coords = nodes[edge]  # (3, 2)
        f_edge = np.zeros((3, 2))
        for g in range(_GP_EDGE.size):
            xi = _GP_EDGE[g]
            wg = _GW_EDGE[g]
            L, dL = _edge_shape(xi)
            r = float(L @ coords[:, 0])
            dr_dxi = float(dL @ coords[:, 0])
            dz_dxi = float(dL @ coords[:, 1])
            # Outward normal (CW rotation of tangent) times |t|: (dz/dxi, -dr/dxi)
            # Traction times dA: (-p) * (dz/dxi, -dr/dxi) * 2*pi*r * dxi
            tract_r = -pressure * dz_dxi
            tract_z = -pressure * (-dr_dxi)
            factor = two_pi * r * wg
            f_edge[:, 0] += L * tract_r * factor
            f_edge[:, 1] += L * tract_z * factor

        for k in range(3):
            n_idx = edge[k]
            F[2 * n_idx] += f_edge[k, 0]
            F[2 * n_idx + 1] += f_edge[k, 1]
    return F


def apply_dirichlet(
    K,
    F: NDArray[np.float64],
    fixed_dofs: NDArray[np.int64],
    fixed_values: NDArray[np.float64] | None = None,
):
    """Apply homogeneous or non-homogeneous Dirichlet BC by partitioning.

    Args:
        K: CSR sparse global stiffness.
        F: (n_dof,) RHS vector.
        fixed_dofs: indices of constrained DOFs.
        fixed_values: prescribed displacement values (defaults to zeros).

    Returns:
        K_ff: reduced stiffness on free DOFs (CSR).
        F_f: reduced RHS on free DOFs.
        free_dofs: indices of free DOFs (sorted).
        u_c: prescribed values aligned with fixed_dofs.
    """
    n_dof = F.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=np.int64)
    if fixed_values is None:
        u_c = np.zeros(fixed_dofs.size)
    else:
        u_c = np.asarray(fixed_values, dtype=np.float64)
        if u_c.shape != fixed_dofs.shape:
            raise ValueError("fixed_values must match fixed_dofs in shape.")

    mask = np.ones(n_dof, dtype=bool)
    mask[fixed_dofs] = False
    free_dofs = np.where(mask)[0]

    K_ff = K[free_dofs][:, free_dofs]
    K_fc = K[free_dofs][:, fixed_dofs]
    F_f = F[free_dofs] - K_fc @ u_c
    return K_ff, F_f, free_dofs, u_c
