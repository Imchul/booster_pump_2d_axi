"""gmsh-driven Q8 (8-node serendipity) quadrilateral meshing.

Takes a meridional half-section (closed polygon) and produces a
quad-dominant Q8 mesh in (r, z) coordinates. Boundary edges retain their
original IDs so that the caller can recover which mesh edges correspond to
each user-tagged DXF edge (pressure / fixed / axis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gmsh
import numpy as np
from numpy.typing import NDArray

from bp2d.io.dxf import BoundaryEdge, HalfSection


@dataclass
class Q8Mesh:
    """Output of `mesh_half_section`.

    Coordinate convention: nodes are stored as (r, z) where r = y_dxf -
    axis_y and z = x_dxf.
    """

    nodes: NDArray[np.float64]            # (n_nodes, 2)
    elements: NDArray[np.int64]           # (n_elem, 8) Q8 connectivity (0-based)
    edge_to_segments: dict[int, NDArray[np.int64]]
    """edge_id -> (n_seg, 3) array of (corner_a, mid, corner_c) node indices."""
    fixed_dofs_axis: NDArray[np.int64]
    """DOF indices of u_r on nodes that lie on the axis (must be zero for
    axisymmetry)."""


def mesh_half_section(
    section: HalfSection,
    mesh_size: float,
    constrain_axis: bool = True,
) -> Q8Mesh:
    """Mesh the meridional half-section with Q8 quad-dominant elements.

    Args:
        section: HalfSection from `extract_half_section`.
        mesh_size: target characteristic element size in DXF units.
        constrain_axis: if True, nodes on the symmetry axis get u_r = 0.

    Returns:
        Q8Mesh with nodes in (r, z) coordinates.
    """
    if mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive.")

    edges = section.edges
    if not edges:
        raise ValueError("HalfSection has no boundary edges.")

    # gmsh wants unique points; build a {coord -> point_tag} map by snapping.
    # Use a tolerance scaled with bounding box.
    bbox = section.bounds
    diag = ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5
    snap_tol = max(1e-9, diag * 1e-9)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bp2d")

        pt_cache: dict[tuple[int, int], int] = {}

        def add_point(p: tuple[float, float]) -> int:
            key = (round(p[0] / snap_tol), round(p[1] / snap_tol))
            if key in pt_cache:
                return pt_cache[key]
            tag = gmsh.model.geo.addPoint(p[0], p[1], 0.0, mesh_size)
            pt_cache[key] = tag
            return tag

        line_tags: list[int] = []
        edge_to_line: dict[int, int] = {}
        for edge in edges:
            p0_tag = add_point(edge.p_start)
            p1_tag = add_point(edge.p_end)
            line_tag = gmsh.model.geo.addLine(p0_tag, p1_tag)
            line_tags.append(line_tag)
            edge_to_line[edge.edge_id] = line_tag

        loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
        surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])

        gmsh.model.geo.synchronize()

        # All-quad Q8 mesh. Strategy:
        #  1. Generate first-order quad mesh via Recombine + Blossom-full-quad
        #     (Algorithm 8 + RecombinationAlgorithm 3).
        #  2. SubdivisionAlgorithm 1 splits any leftover triangle into 3 quads,
        #     guaranteeing a pure-quad mesh.
        #  3. ElementOrder 2 + SecondOrderIncomplete 1 promotes to Q8.
        # Calling setOrder(2) after generate(2) corrupts node bookkeeping, so
        # we let generate(2) build second-order elements directly.
        gmsh.model.mesh.setRecombine(2, surface_tag)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.25)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.5)

        gmsh.model.mesh.generate(2)

        # Extract nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)[:, :2]
        node_tags = np.asarray(node_tags, dtype=np.int64)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Convert (x_dxf, y_dxf) -> (r, z): r = y_dxf - axis_y, z = x_dxf.
        r_coords = node_coords[:, 1] - section.axis_y
        z_coords = node_coords[:, 0]
        nodes_rz = np.column_stack([r_coords, z_coords])

        # Extract Q8 elements (gmsh element type 16 = 8-node quad).
        # gmsh produces CCW corners in (x_dxf, y_dxf). The transform
        # (x, y) -> (r, z) = (y - axis, x) is a reflection, so corners must be
        # permuted to remain CCW in (r, z). The permutation
        # [0, 3, 2, 1, 7, 6, 5, 4] reverses the corner traversal direction
        # while keeping each mid-edge between its (new) adjacent corners.
        _Q8_RZ_PERM = np.array([0, 3, 2, 1, 7, 6, 5, 4], dtype=np.int64)

        e_types, e_tags, e_nodes = gmsh.model.mesh.getElements(dim=2)
        elements: NDArray[np.int64] | None = None
        n_other = 0
        for et, _etags, ents in zip(e_types, e_tags, e_nodes):
            if et == 16:
                arr = np.asarray(ents, dtype=np.int64).reshape(-1, 8)
                arr_local = np.vectorize(tag_to_idx.get)(arr).astype(np.int64)
                elements = arr_local[:, _Q8_RZ_PERM]
            else:
                n_other += int(np.asarray(ents).size)
        if elements is None:
            raise RuntimeError("gmsh did not produce any 8-node quadrilaterals.")
        if n_other > 0:
            raise RuntimeError(
                f"Mesh contains non-Q8 2D elements ({n_other} extra node-refs). "
                "Lower mesh_size or simplify geometry to obtain a pure Q8 mesh."
            )

        # gmsh's SubdivisionAlgorithm can flip some sub-elements; correct on a
        # per-element basis using signed area of the corner ring.
        _flip_negative_elements_inplace(nodes_rz, elements)
        _validate_ccw(nodes_rz, elements)

        # Boundary edges per user-tagged curve. gmsh element type 8 = 3-node
        # line ordered [corner_a, corner_c, mid] in (x, y) CCW. After the
        # reflection to (r, z) the corners must swap to remain CCW around the
        # solid boundary; final order [a_rz, mid, c_rz] = raw[:, [1, 2, 0]].
        edge_to_segments: dict[int, NDArray[np.int64]] = {}
        for edge_id, line_tag in edge_to_line.items():
            be_types, _be_tags, be_nodes = gmsh.model.mesh.getElements(dim=1, tag=line_tag)
            for et, _t, ents in zip(be_types, _be_tags, be_nodes):
                if et == 8:
                    raw = np.asarray(ents, dtype=np.int64).reshape(-1, 3)
                    reordered = raw[:, [1, 2, 0]]
                    seg_idx = np.vectorize(tag_to_idx.get)(reordered).astype(np.int64)
                    edge_to_segments[edge_id] = seg_idx
                    break
            edge_to_segments.setdefault(edge_id, np.empty((0, 3), dtype=np.int64))

        # Axis DOFs (u_r = 0 on r = 0)
        axis_node_idx = np.where(np.abs(nodes_rz[:, 0]) < 1e-6 * max(1.0, diag))[0]
        if constrain_axis:
            fixed_axis = (2 * axis_node_idx).astype(np.int64)
        else:
            fixed_axis = np.empty(0, dtype=np.int64)

        return Q8Mesh(
            nodes=nodes_rz,
            elements=elements,
            edge_to_segments=edge_to_segments,
            fixed_dofs_axis=fixed_axis,
        )
    finally:
        gmsh.finalize()


def _signed_area_q8(nodes: NDArray[np.float64], elem: NDArray[np.int64]) -> float:
    """Signed area of the corner ring (positive if CCW)."""
    c = nodes[elem[:4]]
    s = 0.0
    for i in range(4):
        x0, y0 = c[i]
        x1, y1 = c[(i + 1) % 4]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def _flip_negative_elements_inplace(
    nodes: NDArray[np.float64], elements: NDArray[np.int64]
) -> int:
    """Flip CW elements to CCW in (r, z) using [0, 3, 2, 1, 7, 6, 5, 4]."""
    flip = np.array([0, 3, 2, 1, 7, 6, 5, 4], dtype=np.int64)
    n_flipped = 0
    for i in range(elements.shape[0]):
        if _signed_area_q8(nodes, elements[i]) <= 0.0:
            elements[i] = elements[i, flip]
            n_flipped += 1
    return n_flipped


def _validate_ccw(nodes: NDArray[np.float64], elements: NDArray[np.int64]) -> None:
    """Sanity check: corner nodes of each element must be CCW in (r, z)."""
    bad = 0
    for elem in elements:
        if _signed_area_q8(nodes, elem) <= 0.0:
            bad += 1
    if bad:
        raise RuntimeError(
            f"{bad} element(s) still non-CCW after orientation correction."
        )


def collect_pressure_segments(
    mesh: Q8Mesh, edge_ids: Sequence[int]
) -> NDArray[np.int64]:
    """Concatenate all segments belonging to the given edge IDs into a single
    (n, 3) array suitable for `pressure_force`."""
    parts: list[NDArray[np.int64]] = []
    for eid in edge_ids:
        if eid in mesh.edge_to_segments:
            parts.append(mesh.edge_to_segments[eid])
    if not parts:
        return np.empty((0, 3), dtype=np.int64)
    return np.vstack(parts)


def collect_fixed_dofs(
    mesh: Q8Mesh,
    edge_ids: Sequence[int],
    fix_r: bool = True,
    fix_z: bool = True,
    extra_axis: bool = True,
) -> NDArray[np.int64]:
    """Return DOF indices that should be Dirichlet-zero, based on fixed edges
    plus (optionally) all axis nodes."""
    dofs: set[int] = set()
    for eid in edge_ids:
        seg = mesh.edge_to_segments.get(eid)
        if seg is None or seg.size == 0:
            continue
        for nid in np.unique(seg):
            n = int(nid)
            if fix_r:
                dofs.add(2 * n)
            if fix_z:
                dofs.add(2 * n + 1)
    if extra_axis:
        for d in mesh.fixed_dofs_axis:
            dofs.add(int(d))
    return np.array(sorted(dofs), dtype=np.int64)
