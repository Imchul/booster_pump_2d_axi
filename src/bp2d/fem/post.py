"""Post-processing: stress recovery and failure metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bp2d.fem.element import evaluate_strain_at_gauss


def element_stress_at_gauss(
    coords: NDArray[np.float64],
    D: NDArray[np.float64],
    u_elem: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Element-level stress at Gauss points.

    Returns:
        gp_xyz: (9, 2) gauss point physical (r, z).
        stress: (9, 4) [sigma_r, sigma_z, sigma_theta, tau_rz].
    """
    gp_xyz, strain = evaluate_strain_at_gauss(coords, u_elem)
    stress = strain @ D.T
    return gp_xyz, stress


def stress_at_gauss(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    D: NDArray[np.float64],
    u: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Whole-mesh stress at all Gauss points.

    Returns:
        gp_xyz: (n_elem * 9, 2) gauss point coordinates.
        stress: (n_elem * 9, 4) stress components.
    """
    n_elem = elements.shape[0]
    gp_all = np.empty((n_elem * 9, 2))
    s_all = np.empty((n_elem * 9, 4))
    for e in range(n_elem):
        conn = elements[e]
        coords = nodes[conn]
        dof = np.empty(16, dtype=np.int64)
        dof[0::2] = 2 * conn
        dof[1::2] = 2 * conn + 1
        u_elem = u[dof]
        gp, s = element_stress_at_gauss(coords, D, u_elem)
        gp_all[e * 9 : (e + 1) * 9] = gp
        s_all[e * 9 : (e + 1) * 9] = s
    return gp_all, s_all


def von_mises(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute von Mises stress from axisymmetric stress (..., 4).

    Stress order: [sigma_r, sigma_z, sigma_theta, tau_rz].
    """
    s_r = stress[..., 0]
    s_z = stress[..., 1]
    s_t = stress[..., 2]
    t_rz = stress[..., 3]
    diff = (
        (s_r - s_z) ** 2
        + (s_z - s_t) ** 2
        + (s_t - s_r) ** 2
        + 6.0 * t_rz ** 2
    )
    return np.sqrt(0.5 * diff)


def safety_factor(uts: float, vm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Safety factor = UTS / sigma_vm (elementwise). Returns inf where vm=0."""
    if uts <= 0.0:
        raise ValueError("UTS must be positive.")
    with np.errstate(divide="ignore"):
        sf = np.where(vm > 0.0, uts / vm, np.inf)
    return sf
