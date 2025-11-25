# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, IrreducibleRepresentation
from escnn.nn import FieldType

from symm_learning.linalg import _project_to_irrep_endomorphism_basis, lstsq
from symm_learning.representation_theory import direct_sum


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
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
