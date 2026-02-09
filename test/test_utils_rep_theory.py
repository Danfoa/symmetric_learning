from __future__ import annotations

import escnn
import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral

from symm_learning.representation_theory import GroupHomomorphismBasis, direct_sum, isotypic_decomp_rep


@pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10), Icosahedral()])
def test_rep_decomposition(group: Group):
    """Check that the disentangled representation is equivalent to the original representation."""
    rep = direct_sum([group.regular_representation] * 10)
    # Random change of basis.
    P, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, P, name="test_rep")

    rep_iso = isotypic_decomp_rep(rep)

    # Check the two representations are equivalent
    test_elements = [group.sample() for _ in range(min(10, group.order()))]

    for g in test_elements:
        assert np.allclose(rep(g), rep_iso(g), atol=1e-5, rtol=1e-5), (
            f"Representations are not equivalent for element {g}"
        )

    # Check that decomposing the representation twice returns the cached representation
    rep_iso2 = isotypic_decomp_rep(rep)
    assert rep_iso == rep_iso2, "Cached representation is not returned"

    # Check that iso_decomposing twice returns the same representation
    # rep_iso3 = isotypic_decomp_rep(rep_iso)


@pytest.mark.parametrize(
    "group", [pytest.param(CyclicGroup(5), id="cyclic5"), pytest.param(Icosahedral(), id="icosahedral")]
)
@pytest.mark.parametrize("basis_expansion", ["memory_heavy", "isotypic_expansion"])
def test_hom_basis(group: Group, basis_expansion: str, atol: float = 1e-4, rtol: float = 1e-4):
    """Basis expansion and projection behave consistently for batched inputs."""
    G = group
    in_rep = direct_sum([G.regular_representation])
    out_rep = direct_sum([G.regular_representation] * 2)
    basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion=basis_expansion)

    B = 4
    w_single = torch.randn(basis.dim)
    w_batch = torch.randn(B, basis.dim)

    W_single = basis(w_single)
    W_batch = basis(w_batch)

    # Test shapes.
    assert W_single.shape == (out_rep.size, in_rep.size), f"{W_single.shape}!= {(out_rep.size, in_rep.size)}"
    assert W_batch.shape == (B, out_rep.size, in_rep.size), f"{W_batch.shape}!= {(B, out_rep.size, in_rep.size)}"

    # Basis expansions: batched vs sequential
    W_batch_seq = torch.stack([basis(w_batch[i]) for i in range(B)], dim=0)
    err = W_batch - W_batch_seq
    assert torch.allclose(err, torch.zeros_like(err), atol=atol, rtol=rtol), (
        f"Basis expansion mismatch between batched and sequential with max error {err.abs().max()}"
    )

    # Random projection: batched vs sequential
    W_rand = torch.randn(B, out_rep.size, in_rep.size)
    W_proj_batch = basis.orthogonal_projection(W_rand)
    W_proj_seq = torch.stack([basis.orthogonal_projection(W_rand[i]) for i in range(B)], dim=0)
    err = W_proj_batch - W_proj_seq
    assert torch.allclose(err, torch.zeros_like(err), atol=atol, rtol=rtol), (
        f"Projection mismatch between batched and sequential with max error {err.abs().max()}"
    )

    # Check sampled elements belong to the homomorphism space
    for g in G.elements:
        rho_out = torch.tensor(out_rep(g), dtype=W_single.dtype)
        rho_in = torch.tensor(in_rep(g), dtype=W_single.dtype)
        left_transform = torch.matmul(rho_out, W_single)
        right_transform = torch.matmul(W_single, rho_in)
        err = left_transform - right_transform
        assert torch.allclose(err, torch.zeros_like(err), atol=atol, rtol=rtol), (
            f"Homomorphism condition failed for group element {g} with max error {err.abs().max()}"
        )

    # Elements from the basis stay invariant under projection (batch and sequential)
    W_batch_proj = basis.orthogonal_projection(W_batch_seq)
    err = W_batch_proj - W_batch_seq
    assert torch.allclose(err, torch.zeros_like(err), atol=atol, rtol=rtol), (
        f"Projection invariance (batched) failed with max error {err.abs().max()}"
    )
    W_batch_proj_seq = torch.stack([basis.orthogonal_projection(W_batch_seq[i]) for i in range(B)], dim=0)
    err = W_batch_proj_seq - W_batch_seq
    assert torch.allclose(err, torch.zeros_like(err), atol=atol, rtol=rtol), (
        f"Projection invariance (sequential) failed with max error {err.abs().max()}"
    )

    # initialize_params: non-dense DoF (batched) -> dense via forward, projection is identity
    B_init = 2
    w_init = basis.initialize_params(leading_shape=B_init)  # [B_init, dim]
    W_init = basis(w_init)  # [B_init, out, in]
    assert W_init.shape == (B_init, out_rep.size, in_rep.size)
    W_init_proj = basis.orthogonal_projection(W_init)
    err = W_init_proj - W_init
    assert torch.allclose(err, torch.zeros_like(err), atol=atol, rtol=rtol), (
        f"Dense params from initialize_params moved under projection; max error {err.abs().max()}"
    )

    # initialize_params: dense initialization (batched) stays in the subspace
    W_dense = basis.initialize_params(return_dense=True, leading_shape=B_init)  # [B_init, out, in]
    assert W_dense.shape == (B_init, out_rep.size, in_rep.size)
    W_dense_proj = basis.orthogonal_projection(W_dense)
    err = W_dense_proj - W_dense
    assert torch.allclose(err, torch.zeros_like(err), atol=atol, rtol=rtol), (
        f"Dense initialize_params output not invariant to projection; max error {err.abs().max()}"
    )
