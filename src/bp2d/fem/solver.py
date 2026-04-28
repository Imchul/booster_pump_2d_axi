"""Linear static solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from bp2d.fem.bc import apply_dirichlet


def solve_static(
    K,
    F: NDArray[np.float64],
    fixed_dofs: NDArray[np.int64],
    fixed_values: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Solve K u = F subject to Dirichlet BCs. Returns full displacement (n_dof,)."""
    n_dof = F.shape[0]
    K_ff, F_f, free_dofs, u_c = apply_dirichlet(K, F, fixed_dofs, fixed_values)
    u = np.zeros(n_dof)
    u[fixed_dofs] = u_c
    u_f = spsolve(K_ff.tocsc(), F_f)
    u[free_dofs] = u_f
    return u
