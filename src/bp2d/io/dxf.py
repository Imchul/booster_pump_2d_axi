"""DXF parsing and meridional half-section extraction.

Supports LINE, LWPOLYLINE, ARC, and SPLINE entities. Internally:
1. Convert all entities to short line segments.
2. Use shapely to stitch them into closed polygons; the largest polygon by
   area is taken as the part outline.
3. Intersect with the upper half-plane (y >= axis_y) to obtain the meridional
   half-section. The cut line at y = axis_y becomes the symmetry-axis edge.
4. Decompose the half-section boundary into a list of "edges" (corner-to-
   corner straight segments) that the user can click in the UI.

Coordinate convention in the DXF: x is axial (z), y is radial. After axis
extraction the radial coordinate r = y - axis_y >= 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Sequence

import ezdxf
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

# Maximum chord-to-arc distance (relative to arc radius) when discretizing arcs
ARC_CHORD_TOL = 0.01
# Number of segments to discretize SPLINE entities
SPLINE_SEGMENTS = 64
# Tolerance for endpoint matching when stitching polygons
SNAP_TOL = 1e-6


@dataclass
class BoundaryEdge:
    """A single straight segment of the half-section boundary."""

    edge_id: int
    p_start: tuple[float, float]
    p_end: tuple[float, float]
    is_axis: bool = False

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.p_start[0] + self.p_end[0]) / 2.0,
                (self.p_start[1] + self.p_end[1]) / 2.0)

    @property
    def length(self) -> float:
        dx = self.p_end[0] - self.p_start[0]
        dy = self.p_end[1] - self.p_start[1]
        return math.hypot(dx, dy)


@dataclass
class HalfSection:
    """Meridional half-section ready for meshing."""

    polygon: Polygon
    axis_y: float
    edges: list[BoundaryEdge] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Bounding box (x_min, y_min, x_max, y_max) in DXF coordinates."""

    @property
    def axis_edge_ids(self) -> list[int]:
        return [e.edge_id for e in self.edges if e.is_axis]


def _arc_to_segments(
    cx: float, cy: float, radius: float, start_angle: float, end_angle: float
) -> list[LineString]:
    """Discretize an arc into chord segments based on ARC_CHORD_TOL."""
    if end_angle < start_angle:
        end_angle += 360.0
    sweep = end_angle - start_angle
    sagitta_ratio = ARC_CHORD_TOL
    delta = math.degrees(2.0 * math.acos(max(0.0, 1.0 - sagitta_ratio)))
    delta = max(2.0, min(delta, 30.0))
    n = max(2, int(math.ceil(sweep / delta)))
    angles = np.linspace(start_angle, end_angle, n + 1)
    segs = []
    for i in range(n):
        a0 = math.radians(angles[i])
        a1 = math.radians(angles[i + 1])
        p0 = (cx + radius * math.cos(a0), cy + radius * math.sin(a0))
        p1 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1))
        segs.append(LineString([p0, p1]))
    return segs


def _entity_to_segments(entity) -> list[LineString]:
    """Convert a single DXF entity to one or more LineString segments."""
    t = entity.dxftype()
    segs: list[LineString] = []
    if t == "LINE":
        s = entity.dxf.start
        e = entity.dxf.end
        segs.append(LineString([(s.x, s.y), (e.x, e.y)]))
    elif t == "LWPOLYLINE":
        pts = list(entity.get_points("xyseb"))
        n = len(pts)
        for i in range(n - 1 + (1 if entity.closed else 0)):
            j = (i + 1) % n
            x0, y0 = pts[i][0], pts[i][1]
            x1, y1 = pts[j][0], pts[j][1]
            bulge = pts[i][4] if len(pts[i]) > 4 else 0.0
            if abs(bulge) < 1e-9:
                segs.append(LineString([(x0, y0), (x1, y1)]))
            else:
                # Bulge -> arc
                chord = math.hypot(x1 - x0, y1 - y0)
                sagitta = abs(bulge) * chord / 2.0
                if sagitta < 1e-12:
                    segs.append(LineString([(x0, y0), (x1, y1)]))
                    continue
                radius = (sagitta * sagitta + (chord / 2.0) ** 2) / (2.0 * sagitta)
                # Center
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                # Perpendicular unit vector
                dx, dy = (x1 - x0) / chord, (y1 - y0) / chord
                nx, ny = -dy, dx
                offset = math.copysign(radius - sagitta, bulge)
                cx, cy = mx + nx * offset, my + ny * offset
                a0 = math.degrees(math.atan2(y0 - cy, x0 - cx))
                a1 = math.degrees(math.atan2(y1 - cy, x1 - cx))
                if bulge > 0:
                    segs.extend(_arc_to_segments(cx, cy, radius, a0, a1))
                else:
                    segs.extend(_arc_to_segments(cx, cy, radius, a1, a0))
    elif t == "ARC":
        cx = entity.dxf.center.x
        cy = entity.dxf.center.y
        r = entity.dxf.radius
        sa = entity.dxf.start_angle
        ea = entity.dxf.end_angle
        segs.extend(_arc_to_segments(cx, cy, r, sa, ea))
    elif t == "SPLINE":
        pts = np.array([(p.x, p.y) for p in entity.flattening(0.5)])
        for i in range(len(pts) - 1):
            segs.append(LineString([(pts[i, 0], pts[i, 1]), (pts[i + 1, 0], pts[i + 1, 1])]))
    elif t == "CIRCLE":
        cx = entity.dxf.center.x
        cy = entity.dxf.center.y
        r = entity.dxf.radius
        segs.extend(_arc_to_segments(cx, cy, r, 0.0, 360.0))
    return segs


def parse_dxf(source: str | Path | IO) -> Polygon:
    """Parse a DXF file into a single shapely Polygon (largest closed area).

    Args:
        source: file path or file-like object containing DXF data.

    Returns:
        Polygon representing the part outline in DXF coordinates.
    """
    if hasattr(source, "read"):
        # ezdxf needs a path or stream that yields strings
        import io
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("cp949", errors="replace")
        doc = ezdxf.read(io.StringIO(text))
    else:
        doc = ezdxf.readfile(str(source))
    msp = doc.modelspace()

    segments: list[LineString] = []
    for entity in msp:
        segments.extend(_entity_to_segments(entity))

    if not segments:
        raise ValueError("No supported entities found in DXF.")

    merged = unary_union(segments)
    polygons = list(polygonize(merged))
    if not polygons:
        raise ValueError(
            "DXF entities do not form any closed polygon. Check for gaps "
            "between curves."
        )
    # Many CAD drawings split a part into adjacent regions (body + flanges).
    # Take the union of all closed faces to get the unified outer outline.
    union = unary_union(polygons)
    if union.geom_type == "MultiPolygon":
        # Disconnected pieces -> keep the largest, but warn at parse time.
        union = max(union.geoms, key=lambda p: p.area)
    if not isinstance(union, Polygon):
        raise ValueError(
            f"Polygon union produced unexpected geometry: {union.geom_type}."
        )
    if union.area <= 0.0:
        raise ValueError("Polygon union has zero area.")
    return union


def _ring_to_edges(coords: Sequence[tuple[float, float]], axis_y: float, start_id: int = 0) -> list[BoundaryEdge]:
    """Convert a closed ring's vertex sequence into boundary edges, tagging
    edges that lie on y = axis_y as the symmetry axis."""
    edges: list[BoundaryEdge] = []
    n = len(coords)
    eid = start_id
    for i in range(n - 1):  # last vertex == first for closed ring
        p0 = (float(coords[i][0]), float(coords[i][1]))
        p1 = (float(coords[i + 1][0]), float(coords[i + 1][1]))
        is_axis = (
            abs(p0[1] - axis_y) < 1e-6 * max(1.0, abs(axis_y))
            and abs(p1[1] - axis_y) < 1e-6 * max(1.0, abs(axis_y))
        )
        edges.append(BoundaryEdge(edge_id=eid, p_start=p0, p_end=p1, is_axis=is_axis))
        eid += 1
    return edges


def extract_half_section(polygon: Polygon, axis_y: float) -> HalfSection:
    """Clip the polygon by y >= axis_y and decompose its boundary into edges.

    The polygon's exterior ring is traversed in CCW order; consecutive
    co-linear segments are merged into single edges. Any edge lying on the
    cut line y = axis_y is flagged as the symmetry axis.
    """
    minx, miny, maxx, maxy = polygon.bounds
    if not (miny <= axis_y <= maxy):
        raise ValueError(
            f"axis_y={axis_y} outside polygon y-range [{miny}, {maxy}]."
        )
    upper = box(minx - 1.0, axis_y, maxx + 1.0, maxy + 1.0)
    half = polygon.intersection(upper)
    if half.is_empty:
        raise ValueError("Half-section is empty after clipping.")
    if half.geom_type == "MultiPolygon":
        half = max(half.geoms, key=lambda p: p.area)
    if not isinstance(half, Polygon):
        raise ValueError(f"Unexpected geometry type after clipping: {half.geom_type}.")

    coords = list(half.exterior.coords)
    if not _is_ccw(coords):
        coords = list(reversed(coords))

    coords = _merge_collinear(coords, tol=1e-6)
    edges = _ring_to_edges(coords, axis_y)

    return HalfSection(
        polygon=half,
        axis_y=axis_y,
        edges=edges,
        bounds=half.bounds,
    )


def _is_ccw(coords: Sequence[tuple[float, float]]) -> bool:
    """Shoelace test: positive area => CCW."""
    s = 0.0
    n = len(coords)
    for i in range(n - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        s += (x1 - x0) * (y1 + y0)
    return s < 0.0


def _merge_collinear(
    coords: Sequence[tuple[float, float]], tol: float = 1e-6
) -> list[tuple[float, float]]:
    """Drop intermediate vertices that sit on the line through their neighbors."""
    if len(coords) <= 3:
        return list(coords)
    out = [coords[0]]
    for i in range(1, len(coords) - 1):
        a = out[-1]
        b = coords[i]
        c = coords[i + 1]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        seg_len = math.hypot(c[0] - a[0], c[1] - a[1])
        if seg_len > 0.0 and abs(cross) / seg_len > tol:
            out.append(b)
    out.append(coords[-1])
    if out[0] != out[-1]:
        out.append(out[0])
    return out


def edges_to_arrays(
    edges: Sequence[BoundaryEdge],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Pack edge endpoints into numpy arrays for plotting / meshing.

    Returns:
        starts: (n_edges, 2)
        ends: (n_edges, 2)
        ids: (n_edges,)
    """
    n = len(edges)
    starts = np.empty((n, 2))
    ends = np.empty((n, 2))
    ids = np.empty(n, dtype=np.int64)
    for i, e in enumerate(edges):
        starts[i] = e.p_start
        ends[i] = e.p_end
        ids[i] = e.edge_id
    return starts, ends, ids
