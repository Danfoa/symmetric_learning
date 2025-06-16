# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, Representation
from torch import Tensor

from symm_learning.representation_theory import isotypic_decomp_rep
from symm_learning.stats import cov, isotypic_cov, var_mean


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_symmetric_moments(group: Group):  # noqa: D103
    import escnn
    from escnn.group import directsum

    G = group

    def compute_moments_for_rep(rep: Representation, batch_size=1000):
        rep = isotypic_decomp_rep(rep)
        x = torch.randn(batch_size, rep.size)
        p = torch.randn(batch_size, rep.size)
        p = torch.sin(p) + torch.cos(p) ** 2 + torch.cos(p) ** 3
        x = x + p
        var, mean = var_mean(x, rep)
        return x, mean, var

    # Test that G-invariant random variables should have equivalent mean and var as standard computation
    mx = 10
    rep_x = directsum([G.trivial_representation] * mx)
    x, mean, var = compute_moments_for_rep(rep_x)
    mean_gt = torch.mean(x, dim=0)
    var_gt = torch.var(x, dim=0)
    assert torch.allclose(mean, mean_gt, atol=1e-6, rtol=1e-4), f"Mean {mean} != {mean_gt}"
    assert torch.allclose(var, var_gt, atol=1e-4, rtol=1e-4), f"Var {var} != {var_gt}"

    # Ensure that computing the mean and variance on the Group orbit is equivalent to the standard computation
    mx = 1
    rep_x = directsum([G.regular_representation] * mx)  # 2D irrep * mx
    x, mean, var = compute_moments_for_rep(rep_x)

    def data_orbit(x, rep_x):
        G_x = []
        for g in rep_x.group.elements:
            g_x = torch.einsum("...ij,...j->...i", torch.tensor(rep_x(g), dtype=x.dtype, device=x.device), x)
            G_x.append(g_x)
        G_x = torch.cat(G_x, dim=0)
        return G_x

    G_x = data_orbit(x, rep_x)
    mean_gt = torch.mean(G_x, dim=0)
    x_c_gt = G_x - mean_gt
    var_gt = torch.sum(x_c_gt**2, dim=0) / (G_x.shape[0] - 1)

    rel_var_err = (var - var_gt) / var
    assert torch.allclose(mean, mean_gt, atol=1e-4, rtol=1e-4), f"Mean {mean} != {mean_gt}"
    assert torch.allclose(rel_var_err, torch.zeros_like(var), atol=1e-1, rtol=1e-1), (
        f"Var rel error {rel_var_err * 100}%"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_cross_cov(group: Group):  # noqa: D103
    import escnn
    from escnn.group import change_basis, directsum

    # Icosahedral group has irreps of dimensions [1, ... 5]. Good test case.
    G = group
    # G = escnn.group.CyclicGroup(3)
    mx, my = 1, 2
    x_rep = directsum([G.regular_representation] * mx)
    y_rep = directsum([G.regular_representation] * my)

    # G = escnn.group.CyclicGroup(3)

    x_rep = isotypic_decomp_rep(x_rep)
    y_rep = isotypic_decomp_rep(y_rep)
    Qx, Qy = x_rep.change_of_basis, y_rep.change_of_basis
    x_rep_iso = change_basis(x_rep, Qx.T, name=f"{x_rep.name}_iso")  # ρ_Χ_p = Q_Χ ρ_Χ Q_Χ^T
    y_rep_iso = change_basis(y_rep, Qy.T, name=f"{y_rep.name}_iso")  # ρ_Y_p = Q_Y ρ_Y Q_Y^T

    batch_size = 500
    # Isotypic basis computation
    X_iso = torch.randn(batch_size, x_rep.size)
    Y_iso = torch.randn(batch_size, y_rep.size)
    Cxy_iso = cov(X_iso, Y_iso, x_rep_iso, y_rep_iso).cpu().numpy()

    # Regular basis computation
    Qx = torch.tensor(x_rep.change_of_basis, dtype=X_iso.dtype)
    Qy = torch.tensor(y_rep.change_of_basis, dtype=Y_iso.dtype)
    X = torch.einsum("ij,...j->...i", Qx, X_iso)
    Y = torch.einsum("ij,...j->...i", Qy, Y_iso)
    Cxy = cov(X, Y, x_rep, y_rep).cpu().numpy()

    assert np.allclose(Cxy, Qy.T @ Cxy_iso @ Qx, atol=1e-6, rtol=1e-4), (
        f"Expected Cxy - Q_y.T Cxy_iso Q_x = 0. Got \n {Cxy - Qy.T @ Cxy_iso @ Qx}"
    )

    # Test that r.v with different irrep types have no covariance. ===========================================
    irrep_id1, irrep_id2 = list(G._irreps.keys())[:2]
    x_rep = directsum([G._irreps[irrep_id1]] * mx)
    y_rep = directsum([G._irreps[irrep_id2]] * my)
    X = torch.randn(batch_size, x_rep.size)
    Y = torch.randn(batch_size, y_rep.size)
    Cxy = cov(X, Y, x_rep, y_rep).cpu().numpy()
    assert np.allclose(Cxy, 0), f"Expected Cxy = 0, got {Cxy}"


# @pytest.mark.parametrize(
#     "group",
#     [
#         pytest.param(CyclicGroup(5), id="cyclic5"),
#         pytest.param(DihedralGroup(10), id="dihedral10"),
#         pytest.param(Icosahedral(), id="icosahedral"),
#     ],
# )
# def test_isotypic_cross_cov(group: Group):  # noqa: D103
#     import escnn
#     from escnn.group import IrreducibleRepresentation, change_basis, directsum

#     G = group

#     for irrep in G.representations.values():
#         if not isinstance(irrep, IrreducibleRepresentation):
#             continue
#         mx, my = 2, 3
#         x_rep_iso = directsum([irrep] * mx)  # ρ_Χ
#         y_rep_iso = directsum([irrep] * my)  # ρ_Y

#         batch_size = 500
#         #  Simulate symmetric random variables
#         X_iso = torch.randn(batch_size, x_rep_iso.size)
#         Y_iso = torch.randn(batch_size, y_rep_iso.size)

#         Cxy_iso, Dxy = isotypic_cov(X_iso, Y_iso, x_rep_iso, y_rep_iso)
#         Cxy_iso = Cxy_iso.numpy()

#         assert Cxy_iso.shape == (my * irrep.size, mx * irrep.size), (
#             f"Expected Cxy_iso to have shape ({my * irrep.size}, {mx * irrep.size}), got {Cxy_iso.shape}"
#         )

#         # Test change of basis is handled appropriately, using random change of basis.
#         Qx, _ = np.linalg.qr(np.random.randn(x_rep_iso.size, x_rep_iso.size))
#         Qy, _ = np.linalg.qr(np.random.randn(y_rep_iso.size, y_rep_iso.size))
#         x_rep = change_basis(x_rep_iso, Qx, name=f"{x_rep_iso.name}_p")  # ρ_Χ_p = Q_Χ ρ_Χ Q_Χ^T
#         y_rep = change_basis(y_rep_iso, Qy, name=f"{y_rep_iso.name}_p")  # ρ_Y_p = Q_Y ρ_Y Q_Y^T
#         # Random variables NOT in irrep-spectral basis.
#         X = Tensor(np.einsum("...ij,...j->...i", Qx, X_iso.numpy()))  # X_p = Q_x X
#         Y = Tensor(np.einsum("...ij,...j->...i", Qy, Y_iso.numpy()))  # Y_p = Q_y Y
#         Cxy_p, Dxy = isotypic_cov(X, Y, x_rep, y_rep)
#         Cxy_p = Cxy_p.numpy()

#         assert np.allclose(Cxy_p, Qy @ Cxy_iso @ Qx.T, atol=1e-6, rtol=1e-4), (
#             f"Expected Cxy_p - Q_y Cxy_iso Q_x^T = 0. Got \n {Cxy_p - Qy @ Cxy_iso @ Qx.T}"
#         )

#         # Test that computing Cxy_iso is equivalent to computing standard cross covariance using data augmentation.
#         GX_iso, GY_iso = [X_iso], [Y_iso]
#         for g in G.elements[1:]:
#             X_g = Tensor(np.einsum("...ij,...j->...i", x_rep(g), X_iso.numpy()))
#             Y_g = Tensor(np.einsum("...ij,...j->...i", y_rep(g), Y_iso.numpy()))
#             GX_iso.append(X_g)
#             GY_iso.append(Y_g)
#         GX_iso = torch.cat(GX_iso, dim=0)

#         Cx_iso, _ = isotypic_cov(x=GX_iso, y=GX_iso, rep_x=x_rep_iso, rep_y=x_rep_iso)
#         Cx_iso = Cx_iso.numpy()
#         # Compute the covariance in standard way doing data augmentation.
#         Cx_iso_orbit = (GX_iso.T @ GX_iso / (GX_iso.shape[0])).numpy()
#         # Project each empirical Cov to the subspace of G-equivariant linear maps, and average across orbit
#         Cx_iso_orbit = np.mean(
#             [np.einsum("ij,jk,kl->il", x_rep_iso(g), Cx_iso_orbit, x_rep_iso(~g)) for g in G.elements], axis=0
#         )
#         # Numerical error occurs for small sample sizes
#         assert np.allclose(Cx_iso, Cx_iso_orbit, atol=1e-2, rtol=1e-2), (
#             "isotypic_cross_cov is not equivalent to computing the covariance using data-augmentation"
#         )
