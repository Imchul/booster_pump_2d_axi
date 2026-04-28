"""Global stiffness assembly for Q8 axisymmetric meshes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from bp2d.fem.element import element_stiffness


def _element_dof_indices(connectivity: NDArray[np.int64]) -> NDArray[np.int64]:
    """Convert (8,) node indices into (16,) global DOF indices, ordered (u_r, u_z) per node."""
    idx = np.empty(16, dtype=np.int64)
    idx[0::2] = 2 * connectivity
    idx[1::2] = 2 * connectivity + 1
    return idx


def assemble_stiffness(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    D: NDArray[np.float64],
) -> csr_matrix:
    """Assemble global stiffness matrix.

    Args:
        nodes: (n_nodes, 2) array of (r, z) coordinates.
        elements: (n_elem, 8) connectivity (node indices, 0-based).
        D: (4, 4) constitutive matrix.

    Returns:
        K: CSR sparse stiffness, size (2*n_nodes, 2*n_nodes).
    """
    n_nodes = nodes.shape[0]
    n_dof = 2 * n_nodes
    n_elem = elements.shape[0]

    rows = np.empty(n_elem * 16 * 16, dtype=np.int64)
    cols = np.empty(n_elem * 16 * 16, dtype=np.int64)
    vals = np.empty(n_elem * 16 * 16, dtype=np.float64)

    cursor = 0
    for e in range(n_elem):
        conn = elements[e]
        coords = nodes[conn]
        Ke = element_stiffness(coords, D)
        dofs = _element_dof_indices(conn)
        ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
        block = 16 * 16
        rows[cursor : cursor + block] = ii.ravel()
        cols[cursor : cursor + block] = jj.ravel()
        vals[cursor : cursor + block] = Ke.ravel()
        cursor += block

    K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()
