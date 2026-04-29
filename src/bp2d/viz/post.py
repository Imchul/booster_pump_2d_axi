"""Matplotlib post-processing figure (von Mises contour + deformed shape).

The Q8 mesh is converted to a triangulation by subdividing each element into
6 sub-triangles built only from existing nodes (4 corner-quadrants plus 2
central triangles spanning the four mid-edge nodes). No virtual centroid is
introduced, so all nodal scalar fields can be interpolated directly with
matplotlib's TriContourf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from numpy.typing import NDArray

from bp2d.fem.element import (
    NODE_NAT,
    b_matrix,
    jacobian,
    physical_grads,
    shape_function_grads,
    shape_functions,
)
from bp2d.fem.post import von_mises

# Q8 -> 6 sub-triangle template (indices into the 8-node connectivity row).
_Q8_SUBTRI = np.array(
    [
        [0, 4, 7],  # corner 1 quadrant
        [4, 1, 5],  # corner 2 quadrant
        [5, 2, 6],  # corner 3 quadrant
        [6, 3, 7],  # corner 4 quadrant
        [4, 5, 7],  # central upper
        [5, 6, 7],  # central lower
    ],
    dtype=np.int64,
)


@dataclass
class NodalFields:
    """Scalar/vector fields evaluated at mesh nodes."""

    sigma_vm: NDArray[np.float64]    # (n_nodes,)
    safety_factor: NDArray[np.float64]   # (n_nodes,)
    u_r: NDArray[np.float64]         # (n_nodes,)
    u_z: NDArray[np.float64]         # (n_nodes,)


def _subdivide_q8(elements: NDArray[np.int64]) -> NDArray[np.int64]:
    """Build a (6 * n_elem, 3) triangulation array from Q8 connectivity."""
    return elements[:, _Q8_SUBTRI].reshape(-1, 3)


def _stress_at_nodes(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    D: NDArray[np.float64],
    u: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project element stress onto nodes by evaluating each element at its
    centroid (xi=eta=0) and assigning that element-constant stress to all 8
    of its nodes; nodes that touch multiple elements get the average. This
    avoids the r=0 singularity in B (no axis-node has a centroid at r=0
    unless the element is degenerate)."""
    n_nodes = nodes.shape[0]
    accum = np.zeros((n_nodes, 4))
    count = np.zeros(n_nodes, dtype=np.int64)

    N0 = shape_functions(0.0, 0.0)
    dN0 = shape_function_grads(0.0, 0.0)

    for e in range(elements.shape[0]):
        conn = elements[e]
        coords = nodes[conn]
        dof = np.empty(16, dtype=np.int64)
        dof[0::2] = 2 * conn
        dof[1::2] = 2 * conn + 1
        u_elem = u[dof]
        J = jacobian(coords, dN0)
        dN_xy = physical_grads(dN0, J)
        r_c = float(N0 @ coords[:, 0])
        if r_c <= 1e-14:
            continue
        B = b_matrix(N0, dN_xy, r_c)
        sigma = D @ (B @ u_elem)
        for k in range(8):
            accum[conn[k]] += sigma
            count[conn[k]] += 1

    safe = count > 0
    out = np.zeros_like(accum)
    out[safe] = accum[safe] / count[safe, None]
    return out


def compute_nodal_fields(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    D: NDArray[np.float64],
    u: NDArray[np.float64],
    uts: float,
) -> NodalFields:
    """Evaluate von Mises and SF at every mesh node."""
    stress_n = _stress_at_nodes(nodes, elements, D, u)
    vm = von_mises(stress_n)
    sf = np.where(vm > 0.0, uts / np.maximum(vm, 1e-30), np.inf)
    return NodalFields(
        sigma_vm=vm,
        safety_factor=sf,
        u_r=u[0::2],
        u_z=u[1::2],
    )


def _set_aspect_equal(ax: matplotlib.axes.Axes) -> None:
    ax.set_aspect("equal", adjustable="datalim")


def build_post_figure(
    nodes_rz: NDArray[np.float64],
    elements: NDArray[np.int64],
    fields: NodalFields,
    *,
    deform_scale: float | None = None,
    show_axis: bool = True,
    units_length: str = "m",
    title_suffix: str = "",
) -> Figure:
    """Three-panel post figure: vM contour, SF contour, deformed shape.

    Args:
        nodes_rz: (n_nodes, 2) node coordinates (r, z).
        elements: (n_elem, 8) Q8 connectivity.
        fields: nodal scalar fields produced by `compute_nodal_fields`.
        deform_scale: displacement scale factor for the deformed shape panel.
            If None, auto-scaled to ~5% of bounding-box diagonal.
        show_axis: include a dashed line at r=0.
        units_length: label for the coordinate axes.
        title_suffix: appended to subplot titles.
    """
    triangles = _subdivide_q8(elements)
    triang = Triangulation(nodes_rz[:, 1], nodes_rz[:, 0], triangles)
    # We plot z horizontally and r vertically (typical meridional view).

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    # vM
    ax = axes[0]
    vm = fields.sigma_vm / 1e6  # MPa
    levels = np.linspace(0.0, max(vm.max(), 1e-9), 30)
    cf = ax.tricontourf(triang, vm, levels=levels, cmap="viridis")
    ax.tricontour(triang, vm, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    cb = fig.colorbar(cf, ax=ax, shrink=0.9, format="%.1f")
    cb.set_label("σ_vm (MPa)")
    ax.set_title(f"Von Mises stress {title_suffix}".strip())
    _set_aspect_equal(ax)

    # SF (clip inf and large values to keep the colormap usable)
    ax = axes[1]
    sf = fields.safety_factor.copy()
    finite_sf = sf[np.isfinite(sf)]
    if finite_sf.size:
        sf_cap = min(float(np.percentile(finite_sf, 99)), 50.0)
    else:
        sf_cap = 10.0
    sf_cap = max(sf_cap, 1.0)
    sf_plot = np.where(np.isfinite(sf), np.minimum(sf, sf_cap), sf_cap)
    levels = np.linspace(0.0, sf_cap, 30)
    cf = ax.tricontourf(triang, sf_plot, levels=levels, cmap="RdYlGn")
    cb = fig.colorbar(cf, ax=ax, shrink=0.9, format="%.1f")
    cb.set_label(f"SF = UTS / σ_vm  (capped at {sf_cap:.1f})")
    ax.set_title(f"Safety factor {title_suffix}".strip())
    _set_aspect_equal(ax)

    # Deformed shape (linear-scale displacement overlaid on undeformed mesh)
    ax = axes[2]
    bbox_diag = float(
        np.linalg.norm(nodes_rz.max(axis=0) - nodes_rz.min(axis=0))
    )
    u_max = float(np.linalg.norm(np.column_stack([fields.u_r, fields.u_z]), axis=1).max())
    if deform_scale is None:
        deform_scale = (
            (0.05 * bbox_diag) / max(u_max, 1e-30) if u_max > 0 else 1.0
        )
    deformed = nodes_rz + deform_scale * np.column_stack([fields.u_r, fields.u_z])

    triang_def = Triangulation(deformed[:, 1], deformed[:, 0], triangles)
    ax.triplot(triang, color="lightgray", lw=0.4, alpha=0.7)
    ax.triplot(triang_def, color="C3", lw=0.5)
    ax.set_title(
        f"Deformed shape (×{deform_scale:.3g}) {title_suffix}".strip()
    )
    _set_aspect_equal(ax)

    for ax in axes:
        if show_axis:
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel(f"z ({units_length})")
        ax.set_ylabel(f"r ({units_length})")

    return fig
