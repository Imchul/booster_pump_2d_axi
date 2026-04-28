"""Lame thick-cylinder verification (plane-strain analog).

A finite-length axisymmetric annulus with uz = 0 on both axial faces is
plane-strain. The closed-form solution is::

    sigma_r(r)     = K * (1 - b^2 / r^2)
    sigma_theta(r) = K * (1 + b^2 / r^2)
    sigma_z(r)     = nu * (sigma_r + sigma_theta) = 2 * nu * K
    u_r(r)         = (1 + nu) * pi * a^2 / (E * (b^2 - a^2))
                     * ((1 - 2*nu) * r + b^2 / r)

with K = pi * a^2 / (b^2 - a^2) and pi the internal pressure.

Pass criterion: max relative error < 1% across all nodes (u_r) and across all
Gauss points (sigma_r, sigma_theta, sigma_z).
"""

from __future__ import annotations

import numpy as np

from bp2d.fem.assembly import assemble_stiffness
from bp2d.fem.bc import pressure_force
from bp2d.fem.material import IsotropicMaterial
from bp2d.fem.post import stress_at_gauss
from bp2d.fem.solver import solve_static
from bp2d.mesh.structured import structured_q8_rect


def _lame_solution(
    r: np.ndarray, a: float, b: float, p: float, E: float, nu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = p * a * a / (b * b - a * a)
    sig_r = K * (1.0 - (b * b) / (r * r))
    sig_th = K * (1.0 + (b * b) / (r * r))
    sig_z = nu * (sig_r + sig_th)
    u_r = (1.0 + nu) * p * a * a / (E * (b * b - a * a)) * ((1.0 - 2.0 * nu) * r + (b * b) / r)
    return sig_r, sig_th, sig_z, u_r


def test_lame_thick_cylinder_plane_strain() -> None:
    a = 0.050
    b = 0.100
    h = 0.020
    nr, nz = 16, 2
    p_int = 50e6  # 50 MPa

    mesh = structured_q8_rect(a, b, 0.0, h, nr, nz)
    mat = IsotropicMaterial(E=200e9, nu=0.30, uts=900e6)
    D = mat.constitutive_matrix()

    K = assemble_stiffness(mesh.nodes, mesh.elements, D)
    F = pressure_force(mesh.nodes, mesh.edges_left, p_int)

    # Plane strain: uz = 0 on top and bottom faces.
    nodes = mesh.nodes
    tol = 1e-12
    on_top = np.isclose(nodes[:, 1], h, atol=tol)
    on_bot = np.isclose(nodes[:, 1], 0.0, atol=tol)
    z_constrained = np.where(on_top | on_bot)[0]
    fixed_dofs = np.array([2 * i + 1 for i in z_constrained], dtype=np.int64)

    u = solve_static(K, F, fixed_dofs)

    # Nodal u_r vs analytical
    r_nodes = nodes[:, 0]
    _, _, _, u_r_exact = _lame_solution(r_nodes, a, b, p_int, mat.E, mat.nu)
    u_r_fem = u[0::2]
    rel_err_u = np.abs(u_r_fem - u_r_exact) / np.abs(u_r_exact).max()
    assert rel_err_u.max() < 0.01, (
        f"u_r max relative error {rel_err_u.max():.4%} exceeds 1%."
    )

    # Gauss-point stresses vs analytical
    gp_xyz, stress = stress_at_gauss(mesh.nodes, mesh.elements, D, u)
    r_gp = gp_xyz[:, 0]
    sr_ex, st_ex, sz_ex, _ = _lame_solution(r_gp, a, b, p_int, mat.E, mat.nu)

    K_const = p_int * a * a / (b * b - a * a)
    err_sr = np.abs(stress[:, 0] - sr_ex) / K_const
    err_st = np.abs(stress[:, 2] - st_ex) / K_const
    err_sz = np.abs(stress[:, 1] - sz_ex) / K_const

    assert err_sr.max() < 0.01, f"sigma_r max relative error {err_sr.max():.4%} exceeds 1%."
    assert err_st.max() < 0.01, f"sigma_theta max relative error {err_st.max():.4%} exceeds 1%."
    assert err_sz.max() < 0.01, f"sigma_z max relative error {err_sz.max():.4%} exceeds 1%."

    # tau_rz must be ~ 0
    assert np.abs(stress[:, 3]).max() < 1e-3 * K_const
