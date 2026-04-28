"""Axisymmetric patch test: uniform radial expansion u_r = c*r, u_z = 0.

Strain field: eps_r = eps_theta = c, eps_z = 0, gamma_rz = 0 (constant).
Stress: sigma_r = sigma_theta = (2*lambda + 2*G)*c, sigma_z = 2*lambda*c.

Q8 must reproduce this exactly (to round-off) regardless of mesh refinement.
"""

from __future__ import annotations

import numpy as np

from bp2d.fem.assembly import assemble_stiffness
from bp2d.fem.material import IsotropicMaterial
from bp2d.fem.post import stress_at_gauss
from bp2d.fem.solver import solve_static
from bp2d.mesh.structured import structured_q8_rect


def test_uniform_radial_expansion_patch() -> None:
    a, b = 0.05, 0.10
    h = 0.02
    nr, nz = 2, 2
    mesh = structured_q8_rect(a, b, 0.0, h, nr, nz)

    mat = IsotropicMaterial(E=200e9, nu=0.30, uts=400e6)
    D = mat.constitutive_matrix()

    K = assemble_stiffness(mesh.nodes, mesh.elements, D)

    # Identify boundary nodes via geometric position.
    nodes = mesh.nodes
    tol = 1e-12
    on_boundary = (
        (np.isclose(nodes[:, 0], a, atol=tol))
        | (np.isclose(nodes[:, 0], b, atol=tol))
        | (np.isclose(nodes[:, 1], 0.0, atol=tol))
        | (np.isclose(nodes[:, 1], h, atol=tol))
    )
    bnd_idx = np.where(on_boundary)[0]

    c = 1e-4
    fixed_dofs: list[int] = []
    fixed_values: list[float] = []
    for i in bnd_idx:
        r_i = float(nodes[i, 0])
        fixed_dofs.append(2 * i)
        fixed_values.append(c * r_i)
        fixed_dofs.append(2 * i + 1)
        fixed_values.append(0.0)

    F = np.zeros(2 * nodes.shape[0])
    u = solve_static(
        K,
        F,
        np.array(fixed_dofs, dtype=np.int64),
        np.array(fixed_values, dtype=np.float64),
    )

    # Interior nodes must satisfy u_r = c*r, u_z = 0 exactly.
    interior = np.where(~on_boundary)[0]
    for i in interior:
        r_i = float(nodes[i, 0])
        u_r = u[2 * i]
        u_z = u[2 * i + 1]
        assert np.isclose(u_r, c * r_i, atol=1e-10, rtol=1e-8), (
            f"u_r mismatch at node {i} (r={r_i}): got {u_r}, expected {c * r_i}"
        )
        assert np.isclose(u_z, 0.0, atol=1e-10), f"u_z != 0 at node {i}: {u_z}"

    # Gauss point stresses must be constant.
    lam = mat.lame_lambda
    G = mat.shear_modulus
    sigma_r_expected = (2.0 * lam + 2.0 * G) * c
    sigma_z_expected = 2.0 * lam * c
    sigma_theta_expected = sigma_r_expected
    tau_rz_expected = 0.0

    _gp, stress = stress_at_gauss(mesh.nodes, mesh.elements, D, u)
    np.testing.assert_allclose(stress[:, 0], sigma_r_expected, rtol=1e-8, atol=1.0)
    np.testing.assert_allclose(stress[:, 1], sigma_z_expected, rtol=1e-8, atol=1.0)
    np.testing.assert_allclose(stress[:, 2], sigma_theta_expected, rtol=1e-8, atol=1.0)
    np.testing.assert_allclose(stress[:, 3], tau_rz_expected, atol=1.0)
