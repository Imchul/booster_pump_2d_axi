"""Q8 (8-node serendipity) axisymmetric element.

Reference plane: meridional (r, z), axis of symmetry at r = 0.
Strain vector (4 components): [eps_r, eps_z, eps_theta, gamma_rz].
DOF order per node: (u_r, u_z). 8 nodes -> 16 DOF per element.

Node numbering (counter-clockwise, corners first):
    4 --- 7 --- 3
    |           |
    8           6
    |           |
    1 --- 5 --- 2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# 3-point 1D Gauss
_GP_1D = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GW_1D = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

# 3x3 tensor product Gauss in (xi, eta)
_GP_2D = np.array([(xi, eta) for xi in _GP_1D for eta in _GP_1D])
_GW_2D = np.array([wxi * weta for wxi in _GW_1D for weta in _GW_1D])

# Reference natural coordinates of the 8 nodes
NODE_NAT = np.array(
    [
        [-1.0, -1.0],  # 1
        [1.0, -1.0],   # 2
        [1.0, 1.0],    # 3
        [-1.0, 1.0],   # 4
        [0.0, -1.0],   # 5
        [1.0, 0.0],    # 6
        [0.0, 1.0],    # 7
        [-1.0, 0.0],   # 8
    ]
)


def shape_functions(xi: float, eta: float) -> NDArray[np.float64]:
    """Q8 serendipity shape functions evaluated at (xi, eta). Shape: (8,)."""
    N = np.empty(8)
    N[0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
    N[1] = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
    N[2] = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
    N[3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
    N[4] = 0.5 * (1 - xi * xi) * (1 - eta)
    N[5] = 0.5 * (1 + xi) * (1 - eta * eta)
    N[6] = 0.5 * (1 - xi * xi) * (1 + eta)
    N[7] = 0.5 * (1 - xi) * (1 - eta * eta)
    return N


def shape_function_grads(xi: float, eta: float) -> NDArray[np.float64]:
    """Gradients of Q8 shape functions in natural coords. Shape: (8, 2) = [dN/dxi, dN/deta]."""
    dN = np.empty((8, 2))
    # Corner nodes
    dN[0, 0] = 0.25 * (1 - eta) * (2 * xi + eta)
    dN[0, 1] = 0.25 * (1 - xi) * (xi + 2 * eta)
    dN[1, 0] = 0.25 * (1 - eta) * (2 * xi - eta)
    dN[1, 1] = 0.25 * (1 + xi) * (-xi + 2 * eta)
    dN[2, 0] = 0.25 * (1 + eta) * (2 * xi + eta)
    dN[2, 1] = 0.25 * (1 + xi) * (xi + 2 * eta)
    dN[3, 0] = 0.25 * (1 + eta) * (2 * xi - eta)
    dN[3, 1] = 0.25 * (1 - xi) * (-xi + 2 * eta)
    # Mid-edge nodes
    dN[4, 0] = -xi * (1 - eta)
    dN[4, 1] = -0.5 * (1 - xi * xi)
    dN[5, 0] = 0.5 * (1 - eta * eta)
    dN[5, 1] = -eta * (1 + xi)
    dN[6, 0] = -xi * (1 + eta)
    dN[6, 1] = 0.5 * (1 - xi * xi)
    dN[7, 0] = -0.5 * (1 - eta * eta)
    dN[7, 1] = -eta * (1 - xi)
    return dN


def jacobian(coords: NDArray[np.float64], dN_nat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map natural -> physical Jacobian.

    coords: (8, 2) [r, z]; dN_nat: (8, 2) [dN/dxi, dN/deta].
    Returns J (2, 2) = [[dr/dxi, dz/dxi], [dr/deta, dz/deta]].
    """
    return dN_nat.T @ coords


def physical_grads(
    dN_nat: NDArray[np.float64], J: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute dN/dr, dN/dz from dN/dxi, dN/deta. Returns (8, 2)."""
    return np.linalg.solve(J, dN_nat.T).T


def b_matrix(N: NDArray[np.float64], dN_xy: NDArray[np.float64], r: float) -> NDArray[np.float64]:
    """Strain-displacement matrix B for axisymmetric Q8. Shape: (4, 16).

    Strain order: [eps_r, eps_z, eps_theta, gamma_rz].
    """
    if r <= 0.0:
        raise ValueError(f"Gauss point at r={r} is on or beyond the axis; mesh must lie at r > 0.")
    B = np.zeros((4, 16))
    for i in range(8):
        dNr = dN_xy[i, 0]
        dNz = dN_xy[i, 1]
        Ni = N[i]
        cr = 2 * i
        cz = 2 * i + 1
        B[0, cr] = dNr
        B[1, cz] = dNz
        B[2, cr] = Ni / r
        B[3, cr] = dNz
        B[3, cz] = dNr
    return B


def element_stiffness(coords: NDArray[np.float64], D: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Q8 axisymmetric element stiffness (16, 16).

    Args:
        coords: (8, 2) nodal (r, z) coordinates.
        D: (4, 4) constitutive matrix.
    """
    Ke = np.zeros((16, 16))
    for g, (xi, eta) in enumerate(_GP_2D):
        wg = _GW_2D[g]
        N = shape_functions(xi, eta)
        dN_nat = shape_function_grads(xi, eta)
        J = jacobian(coords, dN_nat)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(f"Non-positive Jacobian at gauss point {g}: det(J)={detJ}")
        dN_xy = physical_grads(dN_nat, J)
        r_g = float(N @ coords[:, 0])
        B = b_matrix(N, dN_xy, r_g)
        Ke += (B.T @ D @ B) * (2.0 * np.pi * r_g * detJ * wg)
    return Ke


def gauss_points_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the 3x3 Gauss point natural coordinates (9, 2) and weights (9,)."""
    return _GP_2D.copy(), _GW_2D.copy()


def evaluate_strain_at_gauss(
    coords: NDArray[np.float64],
    u_elem: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate strain at the 9 Gauss points of a Q8 element.

    Args:
        coords: (8, 2) nodal (r, z) coordinates.
        u_elem: (16,) element displacement vector, ordered (u_r, u_z) per node.

    Returns:
        gp_xyz: (9, 2) physical (r, z) of each gauss point.
        strains: (9, 4) [eps_r, eps_z, eps_theta, gamma_rz] at each gauss point.
    """
    gp_xyz = np.zeros((9, 2))
    strains = np.zeros((9, 4))
    for g, (xi, eta) in enumerate(_GP_2D):
        N = shape_functions(xi, eta)
        dN_nat = shape_function_grads(xi, eta)
        J = jacobian(coords, dN_nat)
        dN_xy = physical_grads(dN_nat, J)
        r_g = float(N @ coords[:, 0])
        z_g = float(N @ coords[:, 1])
        B = b_matrix(N, dN_xy, r_g)
        gp_xyz[g] = (r_g, z_g)
        strains[g] = B @ u_elem
    return gp_xyz, strains
