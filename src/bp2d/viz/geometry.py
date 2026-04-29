"""Plotly figure for the meridional half-section, with clickable edge markers.

Each `BoundaryEdge` is rendered as:
  - one line trace (color reflects current assignment), plus
  - one circular marker at the edge midpoint with `customdata = [edge_id]`,
    used as the click target.
"""

from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go

from bp2d.io.dxf import BoundaryEdge, HalfSection

EDGE_COLORS = {
    "free": "#888888",
    "axis": "#000000",
    "pressure": "#d62728",
    "fixed": "#1f77b4",
}

LINE_WIDTH = {
    "free": 2,
    "axis": 2,
    "pressure": 4,
    "fixed": 4,
}


def _classify(
    edge: BoundaryEdge,
    pressure_ids: set[int],
    fixed_ids: set[int],
) -> str:
    if edge.is_axis:
        return "axis"
    if edge.edge_id in pressure_ids:
        return "pressure"
    if edge.edge_id in fixed_ids:
        return "fixed"
    return "free"


def build_geometry_figure(
    section: HalfSection,
    pressure_ids: Iterable[int] = (),
    fixed_ids: Iterable[int] = (),
    *,
    title: str = "Meridional half-section",
) -> go.Figure:
    """Return a Plotly figure suitable for `st.plotly_chart(on_select=...)`.

    Args:
        section: HalfSection produced by `extract_half_section`.
        pressure_ids: edge IDs currently assigned as pressure surfaces.
        fixed_ids: edge IDs currently assigned as fixed (Dirichlet) surfaces.
        title: figure title.
    """
    pressure_ids = set(pressure_ids)
    fixed_ids = set(fixed_ids)

    fig = go.Figure()

    # Lines first (drawn behind markers)
    for edge in section.edges:
        kind = _classify(edge, pressure_ids, fixed_ids)
        line_kwargs = dict(
            color=EDGE_COLORS[kind],
            width=LINE_WIDTH[kind],
        )
        if kind == "axis":
            line_kwargs["dash"] = "dash"
        fig.add_trace(
            go.Scatter(
                x=[edge.p_start[0], edge.p_end[0]],
                y=[edge.p_start[1], edge.p_end[1]],
                mode="lines",
                line=line_kwargs,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Click-target markers at each midpoint, in one trace so the selection
    # event returns a single point_index per click.
    mids_x: list[float] = []
    mids_y: list[float] = []
    custom: list[list[int]] = []
    colors: list[str] = []
    sizes: list[int] = []
    labels: list[str] = []
    for edge in section.edges:
        kind = _classify(edge, pressure_ids, fixed_ids)
        mx, my = edge.midpoint
        mids_x.append(mx)
        mids_y.append(my)
        custom.append([edge.edge_id])
        colors.append(EDGE_COLORS[kind])
        sizes.append(20 if kind in {"pressure", "fixed"} else 14)
        labels.append(str(edge.edge_id))

    fig.add_trace(
        go.Scatter(
            x=mids_x,
            y=mids_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#222"),
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color="black", width=1),
                symbol="circle",
            ),
            customdata=custom,
            hovertemplate="edge %{customdata[0]}<extra></extra>",
            name="edges",
            showlegend=False,
        )
    )

    # Legend proxies
    for kind, color in EDGE_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=LINE_WIDTH[kind],
                          dash="dash" if kind == "axis" else "solid"),
                name=kind,
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="x (DXF)",
        yaxis_title="y (DXF)",
        yaxis=dict(scaleanchor="x", scaleratio=1.0),
        margin=dict(l=40, r=40, t=50, b=40),
        clickmode="event+select",
        dragmode="pan",
        height=420,
    )
    return fig
