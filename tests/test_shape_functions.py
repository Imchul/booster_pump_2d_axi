"""Sanity checks for Q8 serendipity shape functions."""

from __future__ import annotations

import numpy as np
import pytest

from bp2d.fem.element import (
    NODE_NAT,
    shape_function_grads,
    shape_functions,
)


@pytest.mark.parametrize(
    "xi, eta",
    [
        (0.0, 0.0),
        (0.5, -0.3),
        (-0.7, 0.9),
        (-1.0, -1.0),
        (1.0, 1.0),
        (0.3, 0.6),
    ],
)
def test_partition_of_unity(xi: float, eta: float) -> None:
    N = shape_functions(xi, eta)
    assert N.shape == (8,)
    assert np.isclose(N.sum(), 1.0, atol=1e-12)


def test_kronecker_at_nodes() -> None:
    for i, (xi, eta) in enumerate(NODE_NAT):
        N = shape_functions(float(xi), float(eta))
        expected = np.zeros(8)
        expected[i] = 1.0
        np.testing.assert_allclose(N, expected, atol=1e-12)


@pytest.mark.parametrize(
    "xi, eta",
    [
        (0.0, 0.0),
        (0.4, -0.2),
        (-0.6, 0.7),
    ],
)
def test_grad_sum_zero(xi: float, eta: float) -> None:
    """Sum of shape function gradients must be zero (constant must reproduce)."""
    dN = shape_function_grads(xi, eta)
    np.testing.assert_allclose(dN.sum(axis=0), np.zeros(2), atol=1e-12)


@pytest.mark.parametrize(
    "xi, eta",
    [
        (0.1, 0.2),
        (-0.4, 0.5),
        (0.7, -0.6),
    ],
)
def test_grad_finite_difference(xi: float, eta: float) -> None:
    """Verify analytical gradients against centered finite differences."""
    dN = shape_function_grads(xi, eta)
    h = 1e-6
    dN_fd = np.empty_like(dN)
    dN_fd[:, 0] = (shape_functions(xi + h, eta) - shape_functions(xi - h, eta)) / (2 * h)
    dN_fd[:, 1] = (shape_functions(xi, eta + h) - shape_functions(xi, eta - h)) / (2 * h)
    np.testing.assert_allclose(dN, dN_fd, atol=1e-7)


def test_linear_field_reproduction() -> None:
    """Q8 must exactly reproduce a linear field f(xi, eta) = a + b*xi + c*eta."""
    a, b, c = 2.0, -1.5, 0.7
    nodal = a + b * NODE_NAT[:, 0] + c * NODE_NAT[:, 1]
    rng = np.random.default_rng(0)
    for _ in range(20):
        xi, eta = rng.uniform(-1, 1, 2)
        N = shape_functions(float(xi), float(eta))
        recovered = float(N @ nodal)
        exact = a + b * xi + c * eta
        assert np.isclose(recovered, exact, atol=1e-12)
