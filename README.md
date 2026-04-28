# bp2d — 2D Axisymmetric Structural Analysis for Launch Vehicle Booster Pump Casings

A finite element tool for static structural analysis of axisymmetric pressure vessels — targeted at oxidizer booster pump casings on space launch vehicles.

## Status

**Phase 1 (current): FEM core.**
- Q8 (8-node serendipity) axisymmetric element
- Linear elastic isotropic material
- Internal pressure & displacement boundary conditions
- von Mises stress + safety factor (UTS basis)
- Verified against Lamé thick-cylinder analytical solution

**Phase 2 (planned):** DXF input, gmsh-driven quad-dominant meshing, Streamlit UI.

**Phase 3 (planned):** Bolt / nozzle / flange modeling, multi-region materials.

## Scope

| Item | Spec |
|------|------|
| Analysis type | 2D axisymmetric, static, linear elastic |
| Element | Q8 (8-node serendipity), 3×3 Gauss |
| Mesh | Quad-dominant (gmsh `Recombine`) |
| Material | Single isotropic (E, ν, UTS) |
| Loads | Internal pressure |
| Failure metric | von Mises σ vs UTS, SF = UTS / σ_vm |

In-house launch vehicle strength evaluation criteria are applied externally to the SF output.

## Quickstart

```bash
uv sync
uv run pytest
```

## Verification

`tests/test_lame.py` validates the FEM core against the closed-form Lamé thick-cylinder solution under internal pressure (plane-strain analog). Pass criterion: relative error < 1% in σ_r, σ_θ, σ_z, u_r at all nodes.

## License

MIT
