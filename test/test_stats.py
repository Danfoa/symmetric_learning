# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

import re

import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, Representation
from torch import Tensor

from symm_learning.representation_theory import direct_sum, isotypic_decomp_rep
from symm_learning.stats import cov, mean, var_mean


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_var_mean(group: Group):  # noqa: D103
    import escnn

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
    rep_x = direct_sum([G.trivial_representation] * mx)
    x, mean, var = compute_moments_for_rep(rep_x)
    mean_gt = torch.mean(x, dim=0)
    var_gt = torch.var(x, dim=0)
    assert torch.allclose(mean, mean_gt, atol=1e-6, rtol=1e-4), f"Mean {mean} != {mean_gt}"
    assert torch.allclose(var, var_gt, atol=1e-4, rtol=1e-4), f"Var {var} != {var_gt}"

    # Ensure that computing the mean and variance on the Group orbit is equivalent to the standard computation
    mx = 1
    rep_x = direct_sum([G.regular_representation] * mx)  # 2D irrep * mx
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

    # Ensure that the estimated mean is invariant under the group action
    for g in G.elements:
        g_x = torch.einsum("ij,...j->...i", torch.tensor(rep_x(g), dtype=x.dtype, device=x.device), x)
        g_var, g_mean = var_mean(g_x, rep_x)
        assert torch.allclose(g_mean, mean, atol=1e-4, rtol=1e-4), f"Mean {g_mean} != {mean}"
        assert torch.allclose(g_var, var, atol=1e-4, rtol=1e-4), f"Var {g_var} != {var}"


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="C2"),
        pytest.param(DihedralGroup(5), id="D5"),
        pytest.param(Icosahedral(), id="Ico"),
    ],
)
def test_cov(group: Group):  # noqa: D103
    from escnn.group import change_basis

    # Icosahedral group has irreps of dimensions [1, ... 5]. Good test case.
    G = group
    mx, my = 2, 5
    x_rep = isotypic_decomp_rep(direct_sum([G.regular_representation] * mx))
    y_rep = isotypic_decomp_rep(direct_sum([G.regular_representation] * my))

    def reynolds_covariance(
        X: Tensor, Y: Tensor, rep_X: Representation, rep_Y: Representation, uncentered: bool = False
    ) -> Tensor:
        if uncentered:
            X_eff, Y_eff = X, Y
        else:
            # Center with invariant means to match cov semantics used for symmetric statistics.
            mu_x = mean(X, rep_X)
            mu_y = mean(Y, rep_Y)
            X_eff = X - mu_x
            Y_eff = Y - mu_y
        C_raw = torch.einsum("...y,...x->yx", Y_eff, X_eff) / (X_eff.shape[0] - 1)
        C_orbit = torch.stack(
            [
                torch.tensor(rep_Y(g), dtype=C_raw.dtype, device=C_raw.device)
                @ C_raw
                @ torch.tensor(rep_X(~g), dtype=C_raw.dtype, device=C_raw.device)
                for g in G.elements
            ],
            dim=0,
        ).mean(dim=0)
        return C_orbit

    batch_size = 500
    dtype = torch.float64
    torch.manual_seed(0)

    # Check in an arbitrary (random orthogonal) basis.
    Qx_rand, _ = torch.linalg.qr(torch.randn(x_rep.size, x_rep.size, dtype=dtype))
    Qy_rand, _ = torch.linalg.qr(torch.randn(y_rep.size, y_rep.size, dtype=dtype))
    if torch.det(Qx_rand) < 0:
        Qx_rand[:, 0] = -Qx_rand[:, 0]
    if torch.det(Qy_rand) < 0:
        Qy_rand[:, 0] = -Qy_rand[:, 0]

    x_rep_rand = change_basis(x_rep, Qx_rand.cpu().numpy(), name=f"{x_rep.name}_rand")
    y_rep_rand = change_basis(y_rep, Qy_rand.cpu().numpy(), name=f"{y_rep.name}_rand")
    X_base = torch.randn(batch_size, x_rep.size, dtype=dtype)
    Y_base = torch.randn(batch_size, y_rep.size, dtype=dtype)
    X_rand = torch.einsum("ij,...j->...i", Qx_rand, X_base)
    Y_rand = torch.einsum("ij,...j->...i", Qy_rand, Y_base)

    Cxy_rand = cov(X_rand, Y_rand, x_rep_rand, y_rep_rand, uncentered=False)
    Cxy_rand_unc = cov(X_rand, Y_rand, x_rep_rand, y_rep_rand, uncentered=True)
    Cxy_rand_reynolds = reynolds_covariance(X_rand, Y_rand, x_rep_rand, y_rep_rand, uncentered=False)
    Cxy_rand_reynolds_unc = reynolds_covariance(X_rand, Y_rand, x_rep_rand, y_rep_rand, uncentered=True)

    assert torch.allclose(Cxy_rand, Cxy_rand_reynolds, atol=1e-6, rtol=1e-5), (
        f"cov != Reynolds in random basis, max error {(Cxy_rand - Cxy_rand_reynolds).abs().max().item():.3e}"
    )
    assert torch.allclose(Cxy_rand_unc, Cxy_rand_reynolds_unc, atol=1e-6, rtol=1e-5), (
        f"uncentered cov != Reynolds in random basis, max error {(Cxy_rand_unc - Cxy_rand_reynolds_unc).abs().max().item():.3e}"
    )

    # Test that r.v with different irrep types have no covariance. ===========================================
    irrep_id1, irrep_id2 = list(G._irreps.keys())[:2]
    x_rep = direct_sum([G._irreps[irrep_id1]] * mx)
    y_rep = direct_sum([G._irreps[irrep_id2]] * my)
    X = torch.randn(batch_size, x_rep.size, dtype=dtype)
    Y = torch.randn(batch_size, y_rep.size, dtype=dtype)
    Cxy = cov(X, Y, x_rep, y_rep, uncentered=False).cpu().numpy()
    Cxy_unc = cov(X, Y, x_rep, y_rep, uncentered=True).cpu().numpy()
    assert np.allclose(Cxy, 0), f"Expected centered Cxy = 0, got {Cxy}"
    assert np.allclose(Cxy_unc, 0), f"Expected uncentered Cxy = 0, got {Cxy_unc}"
