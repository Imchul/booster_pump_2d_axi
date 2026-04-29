# bp2d — 2D Axisymmetric Structural Analysis for Launch Vehicle Booster Pump Casings

A finite element tool for static structural analysis of axisymmetric pressure vessels — targeted at oxidizer booster pump casings on space launch vehicles.

## Status

**Phase 1: FEM core** ✅
- Q8 (8-node serendipity) axisymmetric element, 3×3 Gauss
- Linear elastic isotropic material (E, ν, UTS)
- Internal pressure traction + Dirichlet BC, sparse direct solver
- von Mises stress + safety factor (UTS basis)
- Verified against Lamé thick-cylinder analytical solution (< 1% relative error)

**Phase 2: DXF in → analysis → contour out** ✅
- DXF parser (`ezdxf` + `shapely`): LINE, LWPOLYLINE (with bulge), ARC, SPLINE, CIRCLE
- Auto-merge of disconnected closed regions (body + flanges) into a single outer polygon
- Meridional half-section extraction by clipping at user-supplied axis y-value
- gmsh-driven Q8 mesher (Algorithm 8 + Blossom-full-quad + SubdivisionAlgorithm 1)
- Streamlit GUI with click-to-select edge BC assignment
- matplotlib post-processing: σ_vm contour, SF contour, deformed shape

**Phase 3 (planned)**: Bolt / nozzle / flange modeling, multi-region materials, more advanced stress-recovery (SPR), thermal coupling.

## Scope

| Item | Spec |
|------|------|
| Analysis type | 2D axisymmetric, static, linear elastic |
| Element | Q8 (8-node serendipity), 3×3 Gauss |
| Mesh | Pure-quad Q8 via gmsh Recombine + Subdivision |
| Material | Single isotropic (E, ν, UTS) |
| Loads | Internal pressure on user-selected edges |
| BC | Fixed (u_r=u_z=0) on user-selected edges + axis u_r=0 |
| Failure metric | von Mises σ vs UTS, SF = UTS / σ_vm |

In-house launch vehicle strength evaluation criteria are applied externally to the SF output.

## Quickstart

```bash
uv sync
uv run pytest                              # run unit + integration tests
uv run streamlit run src/bp2d/app.py       # launch GUI
```

### GUI workflow

1. Upload a DXF in the sidebar.
2. Set the rotation-axis y-coordinate (default = bbox midline). The half-section is auto-extracted.
3. Choose `Pressure` or `Fixed` mode and click the numbered markers on the geometry plot to assign edges. Black-dashed edges are auto-tagged as the symmetry axis.
4. Enter material (E, ν, UTS), pressure, and target mesh size.
5. Click **Run analysis**.
6. Inspect σ_vm contour, SF contour, and deformed shape.

## Verification

- `tests/test_shape_functions.py` — Q8 partition of unity, Kronecker, gradient FD, linear field reproduction.
- `tests/test_patch.py` — axisymmetric uniform-radial-expansion patch test.
- `tests/test_lame.py` — Lamé thick-cylinder plane-strain (< 1% rel. error in σ_r, σ_θ, σ_z, u_r).
- `tests/test_dxf.py` — DXF parsing + half-section extraction on `examples/Practice.dxf`.
- `tests/test_practice_pipeline.py` — end-to-end smoke test (parse → mesh → solve → nodal vM).

## License

MIT
