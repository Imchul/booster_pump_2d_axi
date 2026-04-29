"""DXF parser + half-section extraction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bp2d.io.dxf import extract_half_section, parse_dxf

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def practice_polygon():
    fp = EXAMPLES / "Practice.dxf"
    if not fp.exists():
        pytest.skip("Practice.dxf not present; skipping DXF integration tests.")
    return parse_dxf(fp)


def test_practice_polygon_unified(practice_polygon) -> None:
    poly = practice_polygon
    # Body + 2 flanges should merge into one polygon spanning x ~[984, 4242].
    minx, miny, maxx, maxy = poly.bounds
    assert maxx - minx > 3000.0
    assert maxy - miny > 200.0
    # Sanity: polygon must be the union (area > body alone).
    assert poly.area > 5e5  # m^2 in DXF units (mm^2)


def test_half_section_axis_tag(practice_polygon) -> None:
    poly = practice_polygon
    miny, maxy = poly.bounds[1], poly.bounds[3]
    axis = (miny + maxy) / 2.0
    half = extract_half_section(poly, axis)
    assert len(half.edges) >= 4
    axis_edges = [e for e in half.edges if e.is_axis]
    assert len(axis_edges) == 1, f"Expected exactly one axis edge, got {len(axis_edges)}"
    e = axis_edges[0]
    # Axis edge must lie at y=axis (in DXF coords) on both endpoints.
    assert abs(e.p_start[1] - axis) < 1e-3
    assert abs(e.p_end[1] - axis) < 1e-3


def test_half_section_polygon_above_axis(practice_polygon) -> None:
    poly = practice_polygon
    miny, maxy = poly.bounds[1], poly.bounds[3]
    axis = (miny + maxy) / 2.0
    half = extract_half_section(poly, axis)
    coords = np.array(half.polygon.exterior.coords)
    assert (coords[:, 1] >= axis - 1e-6).all(), "Half-section should be entirely above axis."
