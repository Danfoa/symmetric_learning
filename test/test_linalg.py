# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, IrreducibleRepresentation
from escnn.nn import FieldType

from symm_learning.linalg import _project_to_irrep_endomorphism_basis, irrep_radii, lstsq
from symm_learning.representation_theory import direct_sum
from symm_learning.utils import check_equivariance


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 5])
@pytest.mark.parametrize("my", [3, 5])
def test_lstsq(group: Group, mx: int, my: int):  # noqa: D103
    import escnn
    from escnn.group import directsum

    # Icosahedral group has irreps of dimensions [1, ... 5]. Good test case.
    G = group
    rep_x = direct_sum([G.regular_representation] * mx)
    rep_y = direct_sum([G.regular_representation] * my)

    x_field = FieldType(escnn.gspaces.no_base_space(G), representations=[rep_x])
    y_field = FieldType(escnn.gspaces.no_base_space(G), representations=[rep_y])
    lin_map = escnn.nn.Linear(x_field, y_field, bias=False)
    A_gt, _ = lin_map.expand_parameters()
    A_gt = A_gt

    batch_size = 1000

    # Generate random X and and compute Y = A_gt @ X
    x = torch.randn(batch_size, rep_x.size)
    y = torch.einsum("ij,nj->ni", A_gt, x)
    # Use G-equivariant least-squares to recover A_gt
    A = lstsq(x, y, rep_x, rep_y)

    assert A.shape == (rep_y.size, rep_x.size), f"Expected A shape {(rep_y.size, rep_x.size)}, got {A.shape}"
    assert torch.allclose(A_gt, A, atol=1e-3, rtol=1e-3)

    # print("Symmetric Least Squares error test passed.")


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(4), id="dihedral4"),
    ],
)
def test_irrep_radii(group: Group):  # noqa: D103
    rep = direct_sum([group.regular_representation] * 2)
    out_rep = direct_sum([group.trivial_representation] * len(rep.irreps))

    # Invariant output check via equivariance helper with trivial output representation.
    check_equivariance(
        lambda t: irrep_radii(t, rep),
        in_rep=rep,
        out_rep=out_rep,
        module_name="irrep_radii",
        atol=1e-5,
        rtol=1e-5,
    )

    # First-order gradient check in the smooth regime (away from exact zero).
    x = (torch.randn(2, rep.size, dtype=torch.float64) + 0.1).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: irrep_radii(t, rep), (x,), eps=1e-6, atol=1e-4, rtol=1e-4)

    # Regression: exact-zero inputs should still produce finite gradients.
    x0 = torch.zeros(8, rep.size, dtype=torch.float64, requires_grad=True)
    loss = irrep_radii(x0, rep).sum()
    loss.backward()

    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()
    assert torch.allclose(x0.grad, torch.zeros_like(x0.grad))
