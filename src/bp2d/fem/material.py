"""Linear elastic isotropic material for axisymmetric problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IsotropicMaterial:
    """Linear elastic isotropic material.

    Attributes:
        E: Young's modulus (Pa).
        nu: Poisson's ratio (-).
        uts: Ultimate tensile strength (Pa). Used downstream for safety factor.
        name: optional human-readable label.
    """

    E: float
    nu: float
    uts: float
    name: str = "isotropic"

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("E must be positive.")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"nu must be in (-1, 0.5); got {self.nu}.")
        if self.uts <= 0.0:
            raise ValueError("UTS must be positive.")

    @property
    def lame_lambda(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def shear_modulus(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    def constitutive_matrix(self) -> NDArray[np.float64]:
        """4x4 D for axisymmetric isotropic (strain order: r, z, theta, rz)."""
        lam = self.lame_lambda
        G = self.shear_modulus
        D = np.zeros((4, 4))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * G
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = G
        return D
