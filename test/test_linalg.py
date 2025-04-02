# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
import numpy as np
import torch
from escnn.nn import FieldType

from symm_learning.linalg import lstsq


def test_lstsq():  # noqa: D103
    import escnn
    from escnn.group import directsum

    # Icosahedral group has irreps of dimensions [1, ... 5]. Good test case.
    G = escnn.group.Icosahedral()
    mx, my = 2, 1
    # G = escnn.group.DihedralGroup(10)
    # mx, my = 6, 5
    rep_x = directsum([G.regular_representation] * mx)
    rep_y = directsum([G.regular_representation] * my)

    # Test in the isotypic basis
    # rep_x = isotypic_decomp_rep(rep_x)
    # rep_x = change_basis(rep_x, rep_x.change_of_basis_inv, f"{rep_x.name}-iso")
    # rep_y = isotypic_decomp_rep(rep_y)
    # rep_y = change_basis(rep_y, rep_y.change_of_basis_inv, f"{rep_y.name}-iso")

    x_field = FieldType(escnn.gspaces.no_base_space(G), representations=[rep_x])
    y_field = FieldType(escnn.gspaces.no_base_space(G), representations=[rep_y])
    lin_map = escnn.nn.Linear(x_field, y_field, bias=False)
    A_gt, _ = lin_map.expand_parameters()
    A_gt = A_gt.detach().cpu().numpy()

    batch_size = 512

    X = torch.randn(batch_size, rep_x.size)
    GX = [torch.einsum("ij,nj->ni", torch.tensor(rep_x(g), dtype=X.dtype), X) for g in G.elements]
    X = torch.cat(GX, dim=0)

    Y = torch.einsum("ij,nj->ni", torch.tensor(A_gt, dtype=X.dtype), X)
    A = lstsq(X, Y, rep_x, rep_y).numpy()

    assert np.allclose(A_gt, A, atol=1e-3, rtol=1e-3)

    print("Symmetric Least Squares error test passed.")
