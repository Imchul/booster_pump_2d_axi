"""End-to-end smoke test: Practice.dxf -> Q8 mesh -> solve -> nodal vM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bp2d.fem.assembly import assemble_stiffness
from bp2d.fem.bc import pressure_force
from bp2d.fem.material import IsotropicMaterial
from bp2d.fem.solver import solve_static
from bp2d.io.dxf import extract_half_section, parse_dxf
from bp2d.mesh.gmsh_q8 import (
    collect_fixed_dofs,
    collect_pressure_segments,
    mesh_half_section,
)
from bp2d.viz.post import compute_nodal_fields

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.skipif(
    not (EXAMPLES / "Practice.dxf").exists(),
    reason="Practice.dxf not available",
)
def test_practice_end_to_end() -> None:
    poly = parse_dxf(EXAMPLES / "Practice.dxf")
    miny, maxy = poly.bounds[1], poly.bounds[3]
    half = extract_half_section(poly, (miny + maxy) / 2.0)

    mesh = mesh_half_section(half, mesh_size=30.0)
    assert mesh.elements.shape[1] == 8
    assert mesh.nodes.shape[1] == 2
    assert mesh.elements.shape[0] > 50

    mat = IsotropicMaterial(E=200e9, nu=0.30, uts=900e6)
    D = mat.constitutive_matrix()
    nodes_m = mesh.nodes * 1e-3

    K = assemble_stiffness(nodes_m, mesh.elements, D)

    # Use top-of-geometry edges as pressure, end-flange verticals as fixed.
    # By construction of Practice.dxf the half-section has 6 edges; edge IDs
    # 0-2 are top, 3 and 5 are the vertical flange ends, 4 is axis.
    seg = collect_pressure_segments(mesh, [0, 1, 2])
    F = pressure_force(nodes_m, seg, pressure=10e6)
    assert np.linalg.norm(F) > 0.0

    fixed = collect_fixed_dofs(mesh, [3, 5])
    u = solve_static(K, F, fixed)
    assert np.isfinite(u).all()
    assert np.abs(u).max() < 1e-2  # < 1 cm displacement

    fields = compute_nodal_fields(nodes_m, mesh.elements, D, u, mat.uts)
    vm_max = fields.sigma_vm.max()
    finite_sf = fields.safety_factor[np.isfinite(fields.safety_factor)]
    assert 1e6 < vm_max < 1e9, f"vM out of expected range: {vm_max}"
    assert finite_sf.min() > 0.0
