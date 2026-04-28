"""Structured Q8 mesh generator for rectangular meridional domains.

Used for verification problems (thick cylinder Lame, patch tests). Real
geometries go through gmsh/DXF in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class StructuredQ8Mesh:
    """Output of `structured_q8_rect`.

    Attributes:
        nodes: (n_nodes, 2) array of (r, z) coordinates.
        elements: (nr * nz, 8) connectivity using bp2d Q8 node order.
        edges_left: (nz, 3) node indices on r=r0, ordered CCW around solid
            (z decreasing from z1 to z0).
        edges_right: (nz, 3) node indices on r=r1, ordered CCW (z increasing).
        edges_bottom: (nr, 3) node indices on z=z0, ordered CCW (r increasing).
        edges_top: (nr, 3) node indices on z=z1, ordered CCW (r decreasing).
        nr: elements in r direction.
        nz: elements in z direction.
    """

    nodes: NDArray[np.float64]
    elements: NDArray[np.int64]
    edges_left: NDArray[np.int64]
    edges_right: NDArray[np.int64]
    edges_bottom: NDArray[np.int64]
    edges_top: NDArray[np.int64]
    nr: int
    nz: int


def structured_q8_rect(
    r0: float, r1: float, z0: float, z1: float, nr: int, nz: int
) -> StructuredQ8Mesh:
    """Generate a structured Q8 mesh on the rectangle [r0, r1] x [z0, z1].

    Args:
        r0, r1: radial bounds with r1 > r0 >= 0. If r0 == 0, the axis-side
            nodes are placed at r=0 and elements adjacent to the axis must be
            handled carefully (avoid for axisymmetric solid).
        z0, z1: axial bounds with z1 > z0.
        nr, nz: positive integer element counts.
    """
    if not (r1 > r0 and r0 >= 0.0):
        raise ValueError("Require r1 > r0 >= 0.")
    if not (z1 > z0):
        raise ValueError("Require z1 > z0.")
    if nr < 1 or nz < 1:
        raise ValueError("nr and nz must be >= 1.")

    # Doubled-resolution grid: skip (odd, odd) positions (no center nodes in Q8).
    nri = 2 * nr
    nzi = 2 * nz
    r_lin = np.linspace(r0, r1, nri + 1)
    z_lin = np.linspace(z0, z1, nzi + 1)

    node_id: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []
    for j in range(nzi + 1):
        for i in range(nri + 1):
            if (i % 2 == 1) and (j % 2 == 1):
                continue
            node_id[(i, j)] = len(coords)
            coords.append((r_lin[i], z_lin[j]))
    nodes = np.array(coords, dtype=np.float64)

    elements = np.empty((nr * nz, 8), dtype=np.int64)
    e = 0
    for je in range(nz):
        for ie in range(nr):
            i0, j0 = 2 * ie, 2 * je
            n1 = node_id[(i0, j0)]
            n2 = node_id[(i0 + 2, j0)]
            n3 = node_id[(i0 + 2, j0 + 2)]
            n4 = node_id[(i0, j0 + 2)]
            n5 = node_id[(i0 + 1, j0)]
            n6 = node_id[(i0 + 2, j0 + 1)]
            n7 = node_id[(i0 + 1, j0 + 2)]
            n8 = node_id[(i0, j0 + 1)]
            elements[e] = (n1, n2, n3, n4, n5, n6, n7, n8)
            e += 1

    # Boundary edges. Each edge = (corner, mid, corner) ordered so that the
    # solid is on the left of the traversal direction (CCW around solid).
    # For a solid filling [r0, r1] x [z0, z1] the CCW boundary is:
    #   bottom: r increasing,  right: z increasing,  top: r decreasing,
    #   left:   z decreasing.
    edges_bottom = np.empty((nr, 3), dtype=np.int64)
    for ie in range(nr):
        i0 = 2 * ie
        edges_bottom[ie] = (
            node_id[(i0, 0)],
            node_id[(i0 + 1, 0)],
            node_id[(i0 + 2, 0)],
        )

    edges_right = np.empty((nz, 3), dtype=np.int64)
    for je in range(nz):
        j0 = 2 * je
        edges_right[je] = (
            node_id[(nri, j0)],
            node_id[(nri, j0 + 1)],
            node_id[(nri, j0 + 2)],
        )

    edges_top = np.empty((nr, 3), dtype=np.int64)
    for ie in range(nr):
        i_end = nri - 2 * ie
        edges_top[ie] = (
            node_id[(i_end, nzi)],
            node_id[(i_end - 1, nzi)],
            node_id[(i_end - 2, nzi)],
        )

    edges_left = np.empty((nz, 3), dtype=np.int64)
    for je in range(nz):
        j_end = nzi - 2 * je
        edges_left[je] = (
            node_id[(0, j_end)],
            node_id[(0, j_end - 1)],
            node_id[(0, j_end - 2)],
        )

    return StructuredQ8Mesh(
        nodes=nodes,
        elements=elements,
        edges_left=edges_left,
        edges_right=edges_right,
        edges_bottom=edges_bottom,
        edges_top=edges_top,
        nr=nr,
        nz=nz,
    )
