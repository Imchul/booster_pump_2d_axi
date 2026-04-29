"""Streamlit UI for bp2d.

Workflow:
    1. Upload a DXF and pick the rotation-axis y coordinate.
    2. Click edge midpoints to assign Pressure / Fixed boundary conditions.
    3. Enter material, pressure, and mesh-size values.
    4. Run -> gmsh meshing + FEM solve + post-processing contours.

Run with:
    uv run streamlit run src/bp2d/app.py
"""

from __future__ import annotations

import io
import time

import numpy as np
import streamlit as st

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
from bp2d.viz.geometry import build_geometry_figure
from bp2d.viz.post import build_post_figure, compute_nodal_fields

st.set_page_config(page_title="bp2d", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, object] = {
    "dxf_polygon": None,
    "dxf_filename": None,
    "axis_y": None,
    "half_section": None,
    "pressure_ids": set(),
    "fixed_ids": set(),
    "selection_mode": "Pressure",
    "results": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Sidebar — upload + axis + material + run
# ---------------------------------------------------------------------------
st.title("bp2d — 2D Axisymmetric Static Analysis")
st.caption("Booster pump casing FEM (Q8 axisymmetric, gmsh-meshed)")

with st.sidebar:
    st.header("1) Geometry")
    upload = st.file_uploader("DXF file", type=["dxf"])
    if upload is not None and upload.name != st.session_state.dxf_filename:
        try:
            poly = parse_dxf(io.BytesIO(upload.getvalue()))
            st.session_state.dxf_polygon = poly
            st.session_state.dxf_filename = upload.name
            ymin, ymax = poly.bounds[1], poly.bounds[3]
            st.session_state.axis_y = (ymin + ymax) / 2.0
            st.session_state.half_section = None
            st.session_state.pressure_ids = set()
            st.session_state.fixed_ids = set()
            st.session_state.results = None
            st.success(f"Loaded {upload.name} (area={poly.area:.1f})")
        except Exception as exc:
            st.error(f"DXF parse failed: {exc}")

    poly = st.session_state.dxf_polygon
    if poly is not None:
        ymin, ymax = poly.bounds[1], poly.bounds[3]
        axis_y = st.number_input(
            "Rotation axis y",
            min_value=float(ymin),
            max_value=float(ymax),
            value=float(st.session_state.axis_y),
            step=float((ymax - ymin) / 100.0),
            format="%.3f",
        )
        if axis_y != st.session_state.axis_y:
            st.session_state.axis_y = axis_y
            st.session_state.half_section = None
            st.session_state.pressure_ids = set()
            st.session_state.fixed_ids = set()
            st.session_state.results = None

        if st.session_state.half_section is None:
            try:
                st.session_state.half_section = extract_half_section(poly, axis_y)
            except Exception as exc:
                st.error(f"Half-section extraction failed: {exc}")

    st.header("2) Material")
    E_GPa = st.number_input("Young's modulus E (GPa)", min_value=1.0, value=200.0, step=10.0)
    nu = st.number_input("Poisson ratio ν", min_value=0.0, max_value=0.49, value=0.30, step=0.01)
    uts_MPa = st.number_input("Ultimate tensile strength (MPa)", min_value=1.0, value=900.0, step=50.0)

    st.header("3) Loads & Mesh")
    pressure_MPa = st.number_input("Internal pressure (MPa)", min_value=0.0, value=10.0, step=1.0)
    mesh_size = st.number_input("Mesh size (DXF units)", min_value=0.1, value=20.0, step=1.0)
    units = st.selectbox(
        "DXF length unit", options=["mm", "m"], index=0, help="Used to scale into SI before solve."
    )

    st.header("4) Run")
    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main panel — geometry view with click-to-select
# ---------------------------------------------------------------------------
half = st.session_state.half_section
if half is None:
    st.info("Upload a DXF in the sidebar to begin.")
    st.stop()

st.subheader("Edge BC selection")
mode_col, info_col = st.columns([1, 3])
with mode_col:
    mode = st.radio(
        "Click mode",
        options=["Pressure", "Fixed"],
        index=0 if st.session_state.selection_mode == "Pressure" else 1,
        horizontal=True,
    )
    st.session_state.selection_mode = mode
    st.markdown(
        """
        **How to use**
        - Click a numbered marker to assign / un-assign the edge to the
          current mode.
        - Black-dashed edges are auto-tagged as the symmetry axis (no
          click needed).
        - Use the lasso tool to clear selection if it gets sticky.
        """
    )
with info_col:
    auto_axis = [e.edge_id for e in half.edges if e.is_axis]
    st.markdown(f"**Axis edges:** {auto_axis}")
    st.markdown(f"**Pressure edges:** {sorted(st.session_state.pressure_ids)}")
    st.markdown(f"**Fixed edges:** {sorted(st.session_state.fixed_ids)}")

fig = build_geometry_figure(
    half,
    pressure_ids=st.session_state.pressure_ids,
    fixed_ids=st.session_state.fixed_ids,
)

ev = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode=("points",),
    key="geom_chart",
)

if ev and ev.get("selection") and ev["selection"].get("points"):
    changed = False
    for p in ev["selection"]["points"]:
        cd = p.get("customdata")
        if cd is None:
            continue
        eid = int(cd[0]) if isinstance(cd, (list, tuple, np.ndarray)) else int(cd)
        target = (
            st.session_state.pressure_ids
            if st.session_state.selection_mode == "Pressure"
            else st.session_state.fixed_ids
        )
        if eid in target:
            target.remove(eid)
        else:
            other = (
                st.session_state.fixed_ids
                if st.session_state.selection_mode == "Pressure"
                else st.session_state.pressure_ids
            )
            other.discard(eid)  # an edge cannot be both pressure and fixed
            target.add(eid)
        changed = True
    if changed:
        st.rerun()


# ---------------------------------------------------------------------------
# Run button: mesh + assemble + solve + post
# ---------------------------------------------------------------------------
if run_clicked:
    if not st.session_state.pressure_ids:
        st.warning("Select at least one Pressure edge before running.")
        st.stop()
    if not st.session_state.fixed_ids:
        st.warning("Select at least one Fixed edge to remove rigid-body axial mode.")
        st.stop()

    progress = st.progress(0.0, text="Meshing…")
    t0 = time.perf_counter()
    try:
        mesh = mesh_half_section(half, mesh_size=mesh_size)
    except Exception as exc:
        st.error(f"Meshing failed: {exc}")
        st.stop()
    t_mesh = time.perf_counter() - t0
    progress.progress(0.33, text=f"Meshed: {len(mesh.elements)} Q8 / {len(mesh.nodes)} nodes")

    mat = IsotropicMaterial(E=E_GPa * 1e9, nu=float(nu), uts=uts_MPa * 1e6)
    D = mat.constitutive_matrix()

    # Convert nodes to meters for SI consistency
    if units == "mm":
        nodes_m = mesh.nodes * 1e-3
    else:
        nodes_m = mesh.nodes

    t0 = time.perf_counter()
    try:
        K = assemble_stiffness(nodes_m, mesh.elements, D)
        seg = collect_pressure_segments(mesh, list(st.session_state.pressure_ids))
        F = pressure_force(nodes_m, seg, pressure=pressure_MPa * 1e6)
        fixed = collect_fixed_dofs(
            mesh,
            list(st.session_state.fixed_ids),
            fix_r=True,
            fix_z=True,
            extra_axis=True,
        )
        u = solve_static(K, F, fixed)
    except Exception as exc:
        st.error(f"Solve failed: {exc}")
        st.stop()
    t_solve = time.perf_counter() - t0
    progress.progress(0.75, text=f"Solved (DOFs: {K.shape[0]}, fixed: {len(fixed)})")

    fields = compute_nodal_fields(nodes_m, mesh.elements, D, u, mat.uts)
    progress.progress(1.0, text="Done")

    st.session_state.results = {
        "mesh_nodes": nodes_m,
        "elements": mesh.elements,
        "fields": fields,
        "t_mesh": t_mesh,
        "t_solve": t_solve,
        "n_dof": int(K.shape[0]),
    }


# ---------------------------------------------------------------------------
# Post-processing display
# ---------------------------------------------------------------------------
res = st.session_state.results
if res is not None:
    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    fields = res["fields"]
    finite_sf = fields.safety_factor[np.isfinite(fields.safety_factor)]
    sf_min = float(finite_sf.min()) if finite_sf.size else float("nan")
    c1.metric("max σ_vm (MPa)", f"{fields.sigma_vm.max() / 1e6:.2f}")
    c2.metric("min SF", f"{sf_min:.2f}")
    c3.metric("max |u| (mm)", f"{np.linalg.norm(np.column_stack([fields.u_r, fields.u_z]), axis=1).max() * 1e3:.4f}")
    c4.metric("DOFs", f"{res['n_dof']:,}")

    fig_post = build_post_figure(
        res["mesh_nodes"],
        res["elements"],
        fields,
        units_length="m",
    )
    st.pyplot(fig_post, clear_figure=True)
    st.caption(
        f"Mesh: {res['t_mesh']:.2f}s · Solve: {res['t_solve']:.2f}s"
    )
