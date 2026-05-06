# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, IrreducibleRepresentation
from escnn.nn import FieldType

from symm_learning.linalg import (
    IsotypicTensorCache,
    _project_to_irrep_endomorphism_basis,
    equiv_linear_map,
    equiv_orthogonal_projection,
    equiv_orthogonal_projection_coefficients,
    invariant_orthogonal_projector,
    irrep_radii,
    isotypic_signal2irreducible_subspaces,
    lstsq,
    project_in_isobasis,
)
from symm_learning.representation_theory import GroupHomomorphismBasis, direct_sum, isotypic_decomp_rep
from symm_learning.utils import check_equivariance


def _device_params():
    params = [pytest.param("cpu", id="cpu")]
    if torch.cuda.is_available():
        params.append(pytest.param("cuda", id="cuda"))
    return params


def _assert_meta(tensor: torch.Tensor, device: str, dtype: torch.dtype):
    assert tensor.device.type == torch.device(device).type
    assert tensor.dtype == dtype


def _make_hom_rep_pair(group: Group, rep_case: str):
    if rep_case == "full_overlap":
        return direct_sum([group.regular_representation]), direct_sum([group.regular_representation] * 2)
    if rep_case == "missing_irreps":
        irreps = group.irreps()
        return direct_sum([irreps[0], irreps[1], irreps[1]]), direct_sum([irreps[1], irreps[2], irreps[1]])
    raise ValueError(f"Unknown rep_case: {rep_case}")


def _assert_missing_irrep_blocks_zero(W: torch.Tensor, rep_x: Group, rep_y: Group, dtype: torch.dtype):
    tol = 1e-5 if dtype == torch.float32 else 1e-8
    Q_out, Q_in_inv, projection_iso_spaces = project_in_isobasis(W, rep_x, rep_y)
    W_iso = (Q_out.mT @ W) @ Q_in_inv.mT
    rep_x_iso = isotypic_decomp_rep(rep_x)
    rep_y_iso = isotypic_decomp_rep(rep_y)
    shared_irreps = set(rep_x_iso.irreps).intersection(set(rep_y_iso.irreps))
    x_only_irreps = set(rep_x_iso.irreps).difference(shared_irreps)
    y_only_irreps = set(rep_y_iso.irreps).difference(shared_irreps)

    assert set(projection_iso_spaces.keys()) == shared_irreps

    x_slices = rep_x_iso.attributes["isotypic_subspace_dims"]
    y_slices = rep_y_iso.attributes["isotypic_subspace_dims"]
    for irrep_id in x_only_irreps:
        block = W_iso[..., :, x_slices[irrep_id]]
        assert torch.allclose(block, torch.zeros_like(block), atol=tol, rtol=tol)
    for irrep_id in y_only_irreps:
        block = W_iso[..., y_slices[irrep_id], :]
        assert torch.allclose(block, torch.zeros_like(block), atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(4), id="dihedral4"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_isotypic_signal2irreducible_subspaces(group: Group, dtype: torch.dtype, device: str):  # noqa: D103
    irrep_id = group.regular_representation.irreps[0]
    irrep = group.irrep(*irrep_id)
    mk = 3
    rep = direct_sum([irrep] * mk)

    x = torch.randn(5, rep.size, device=device, dtype=dtype)
    z = isotypic_signal2irreducible_subspaces(x, rep)
    _assert_meta(z, device=device, dtype=dtype)
    assert z.shape == (x.shape[0] * irrep.size, mk)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(4), id="dihedral4"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_invariant_orthogonal_projector(group: Group, dtype: torch.dtype, device: str):  # noqa: D103
    rep = direct_sum([group.regular_representation] * 2)

    P_default = invariant_orthogonal_projector(rep)
    assert P_default.device.type == "cpu"
    assert P_default.dtype == torch.get_default_dtype()

    P = invariant_orthogonal_projector(rep, device=device, dtype=dtype)
    _assert_meta(P, device=device, dtype=dtype)

    tol = 1e-5 if dtype == torch.float32 else 1e-8
    assert torch.allclose(P @ P, P, atol=tol, rtol=tol)
    assert torch.allclose(P.T, P, atol=tol, rtol=tol)
    for i, g in enumerate(group.elements):
        if i == 4:
            break
        g_mat = torch.tensor(rep(g), device=device, dtype=dtype)
        assert torch.allclose(g_mat @ P, P, atol=tol, rtol=tol)
        assert torch.allclose(P @ g_mat, P, atol=tol, rtol=tol)


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_lstsq_dtype_device(group: Group, dtype: torch.dtype, device: str):  # noqa: D103
    from symm_learning.representation_theory import isotypic_decomp_rep

    rep_x = direct_sum([group.regular_representation] * 2)
    rep_y = direct_sum([group.regular_representation] * 2)

    warm_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    x_warm = torch.randn(128, rep_x.size, device="cpu", dtype=warm_dtype)
    _ = lstsq(x_warm, x_warm, rep_x, rep_y)

    rep_x_iso = isotypic_decomp_rep(rep_x)
    rep_y_iso = isotypic_decomp_rep(rep_y)
    assert rep_x_iso.attributes["Q"].device.type == "cpu"
    assert rep_x_iso.attributes["Q"].dtype == warm_dtype
    assert rep_y_iso.attributes["Q"].device.type == "cpu"
    assert rep_y_iso.attributes["Q"].dtype == warm_dtype

    x = torch.randn(128, rep_x.size, device=device, dtype=dtype)
    A = lstsq(x, x, rep_x, rep_y)

    _assert_meta(A, device=device, dtype=dtype)
    assert A.shape == (rep_y.size, rep_x.size)
    assert torch.isfinite(A).all()
    assert rep_x_iso.attributes["Q"].device.type == torch.device(device).type
    assert rep_x_iso.attributes["Q"].dtype == dtype
    assert rep_y_iso.attributes["Q"].device.type == torch.device(device).type
    assert rep_y_iso.attributes["Q"].dtype == dtype

    identity = torch.eye(rep_x.size, device=device, dtype=dtype)
    tol = 5e-3 if dtype == torch.float32 else 1e-6
    assert torch.allclose(A, identity, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(4), id="dihedral4"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_irrep_radii(group: Group, dtype: torch.dtype, device: str):  # noqa: D103
    rep = direct_sum([group.regular_representation] * 2)
    warm_dtype = torch.float64 if dtype == torch.float32 else torch.float32

    # Warm cache in a different dtype/device and ensure it updates on latest call.
    _ = irrep_radii(torch.randn(4, rep.size, device="cpu", dtype=warm_dtype), rep)
    assert rep.attributes["Q_inv"].device.type == "cpu"
    assert rep.attributes["Q_inv"].dtype == warm_dtype

    if device == "cpu" and dtype == torch.float32:
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

    if device == "cpu" and dtype == torch.float64:
        # First-order gradient check in the smooth regime (away from exact zero).
        x_gc = (torch.randn(2, rep.size, device=device, dtype=dtype) + 0.1).requires_grad_(True)
        assert torch.autograd.gradcheck(lambda t: irrep_radii(t, rep), (x_gc,), eps=1e-6, atol=1e-4, rtol=1e-4)

    x = torch.randn(8, rep.size, device=device, dtype=dtype, requires_grad=True)
    radii = irrep_radii(x, rep)
    _assert_meta(radii, device=device, dtype=dtype)
    assert rep.attributes["Q_inv"].device.type == torch.device(device).type
    assert rep.attributes["Q_inv"].dtype == dtype
    (radii.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Regression: exact-zero inputs should still produce finite gradients.
    x0 = torch.zeros(8, rep.size, device=device, dtype=dtype, requires_grad=True)
    loss = irrep_radii(x0, rep).sum()
    loss.backward()

    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()
    assert torch.allclose(x0.grad, torch.zeros_like(x0.grad))


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(4), id="dihedral4"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_project_in_isobasis(group: Group, dtype: torch.dtype, device: str):
    """Project-in-isobasis should expose only the shared-irrep metadata contract."""
    in_rep, out_rep = _make_hom_rep_pair(group, rep_case="missing_irreps")
    W = torch.randn(3, out_rep.size, in_rep.size, device=device, dtype=dtype)

    iso_output = project_in_isobasis(W, in_rep, out_rep)

    assert isinstance(iso_output, tuple)
    assert len(iso_output) == 3
    Q_out, Q_in_inv, projection_iso_spaces = iso_output
    _assert_meta(Q_out, device=device, dtype=dtype)
    _assert_meta(Q_in_inv, device=device, dtype=dtype)

    rep_x_iso = isotypic_decomp_rep(in_rep)
    rep_y_iso = isotypic_decomp_rep(out_rep)
    shared_irreps = set(rep_x_iso.irreps).intersection(set(rep_y_iso.irreps))
    assert set(projection_iso_spaces.keys()) == shared_irreps
    for irrep_id, iso_space in projection_iso_spaces.items():
        assert irrep_id in shared_irreps
        assert iso_space["coeff"].shape[-1] == iso_space["endo_basis_flat"].shape[0]
        assert iso_space["endo_basis_flat"].shape[-1] == iso_space["d_k"] * iso_space["d_k"]


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("rep_case", [pytest.param("full_overlap"), pytest.param("missing_irreps")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_equiv_orthogonal_projection(group: Group, rep_case: str, dtype: torch.dtype, device: str):
    """Projection kernel should satisfy core projection properties on Hom_G."""
    G = group  # Select the symmetry group instance under test (e.g., C5 or Icosahedral).
    in_rep, out_rep = _make_hom_rep_pair(G, rep_case=rep_case)
    basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion="isotypic_expansion").to(device=device, dtype=dtype)
    assert isinstance(basis.tensor_cache, IsotypicTensorCache)

    B = 4  # Number of random linear maps tested in one batched call.
    W_rand = torch.randn(B, out_rep.size, in_rep.size, device=device, dtype=dtype)  # Raw unconstrained maps.

    # Compute the projection in batched mode.
    W_proj_batch = equiv_orthogonal_projection(W_rand, in_rep, out_rep)
    W_proj_cached = equiv_orthogonal_projection(W_rand, in_rep, out_rep, tensor_cache=basis.tensor_cache)
    # Compute the same projection map-by-map to verify batching does not change results.
    W_proj_seq = torch.stack(
        [equiv_orthogonal_projection(W_rand[i], in_rep, out_rep) for i in range(B)],
        dim=0,
    )
    _assert_meta(W_proj_batch, device=device, dtype=dtype)  # Projection preserves requested device and dtype.
    assert W_proj_batch.shape == (B, out_rep.size, in_rep.size)  # Projection preserves input matrix shape.
    # Core check: batched and sequential projection paths are numerically equivalent.
    assert torch.allclose(W_proj_batch, W_proj_seq, atol=1e-5, rtol=1e-5), (
        f"Batched/seq projection mismatch, max error {(W_proj_batch - W_proj_seq).abs().max().item():.3e}"
    )
    assert torch.allclose(W_proj_batch, W_proj_cached, atol=1e-5, rtol=1e-5), (
        "Rep-cache and tensor-cache projection mismatch, "
        f"max error {(W_proj_batch - W_proj_cached).abs().max().item():.3e}"
    )

    # Idempotence property of orthogonal projections: P(P(W)) = P(W).
    W_proj_twice = equiv_orthogonal_projection(W_proj_batch, in_rep, out_rep)
    assert torch.allclose(W_proj_batch, W_proj_twice, atol=1e-5, rtol=1e-5), (
        f"Projection is not idempotent, max error {(W_proj_batch - W_proj_twice).abs().max().item():.3e}"
    )

    # Build an explicitly equivariant map via group-average (Reynolds operator).
    W0 = W_rand[0]  # Seed matrix to average over its group orbit.
    W_equiv = torch.stack(
        [
            # Conjugation action on maps: ρ_out(g) W ρ_in(g^{-1}).
            # Averaging this orbit projects onto Hom_G.
            torch.tensor(out_rep(g), device=device, dtype=dtype)
            @ W0
            @ torch.tensor(in_rep(~g), device=device, dtype=dtype)
            for g in G.elements
        ],
        dim=0,
    ).mean(dim=0)
    # Explicit Reynolds-operator equivalence on an arbitrary map W0:
    # P(W0) must equal the group-average conjugation of W0.
    W0_proj = equiv_orthogonal_projection(W0, in_rep, out_rep)
    assert torch.allclose(W0_proj, W_equiv, atol=1e-5, rtol=1e-5), (
        f"Projection != Reynolds average, max error {(W0_proj - W_equiv).abs().max().item():.3e}"
    )
    # Project an already-equivariant map.
    W_equiv_proj = equiv_orthogonal_projection(W_equiv, in_rep, out_rep)
    # Fixed-point check: true equivariant maps should be unchanged by the projection.
    assert torch.allclose(W_equiv, W_equiv_proj, atol=1e-5, rtol=1e-5), (
        f"Projection changed an already-equivariant map, max error {(W_equiv - W_equiv_proj).abs().max().item():.3e}"
    )

    # Direct homomorphism constraint check on one projected sample:
    # ρ_out(g) W = W ρ_in(g) for all g.
    W_single = W_proj_batch[0]  # Any projected sample should satisfy the constraint.
    for g in G.elements:
        rho_out = torch.tensor(out_rep(g), device=device, dtype=dtype)  # Matrix of ρ_out(g).
        rho_in = torch.tensor(in_rep(g), device=device, dtype=dtype)  # Matrix of ρ_in(g).
        err = rho_out @ W_single - W_single @ rho_in  # Zero iff W_single is equivariant for this g.
        assert torch.allclose(err, torch.zeros_like(err), atol=1e-5, rtol=1e-5), (
            f"Projected map violates hom condition for {g}, max error {err.abs().max().item():.3e}"
        )
    if rep_case == "missing_irreps":
        _assert_missing_irrep_blocks_zero(W_proj_batch, in_rep, out_rep, dtype=dtype)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("rep_case", [pytest.param("full_overlap"), pytest.param("missing_irreps")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_equiv_orthogonal_projection_coefficients(group: Group, rep_case: str, dtype: torch.dtype, device: str):
    """Projected homomorphism coefficients should reconstruct the dense projected map."""
    in_rep, out_rep = _make_hom_rep_pair(group, rep_case=rep_case)
    basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion="isotypic_expansion").to(device=device, dtype=dtype)

    batch_size = 4
    W_rand = torch.randn(batch_size, out_rep.size, in_rep.size, device=device, dtype=dtype)

    theta_batch = equiv_orthogonal_projection_coefficients(W_rand, in_rep, out_rep)
    theta_cached = equiv_orthogonal_projection_coefficients(W_rand, in_rep, out_rep, tensor_cache=basis.tensor_cache)
    theta_seq = torch.stack(
        [equiv_orthogonal_projection_coefficients(W_rand[i], in_rep, out_rep) for i in range(batch_size)],
        dim=0,
    )
    W_proj = equiv_orthogonal_projection(W_rand, in_rep, out_rep)
    W_recon = basis(theta_batch)
    theta_basis = basis.projection_coefficients(W_rand)

    _assert_meta(theta_batch, device=device, dtype=dtype)
    assert theta_batch.shape == (batch_size, basis.dim)
    assert torch.allclose(theta_batch, theta_seq, atol=1e-5, rtol=1e-5), (
        f"Batched/seq coefficient mismatch, max error {(theta_batch - theta_seq).abs().max().item():.3e}"
    )
    assert torch.allclose(theta_batch, theta_cached, atol=1e-5, rtol=1e-5), (
        "Rep-cache and tensor-cache coefficient mismatch, "
        f"max error {(theta_batch - theta_cached).abs().max().item():.3e}"
    )
    assert torch.allclose(theta_batch, theta_basis, atol=1e-5, rtol=1e-5), (
        f"Function/module coefficient mismatch, max error {(theta_batch - theta_basis).abs().max().item():.3e}"
    )
    assert torch.allclose(W_recon, W_proj, atol=1e-5, rtol=1e-5), (
        f"Coefficient reconstruction mismatch, max error {(W_recon - W_proj).abs().max().item():.3e}"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("rep_case", [pytest.param("full_overlap"), pytest.param("missing_irreps")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("device", _device_params())
def test_equiv_linear_map(group: Group, rep_case: str, dtype: torch.dtype, device: str):
    """Coefficient expansion should match the isotypic Hom-basis forward path."""
    in_rep, out_rep = _make_hom_rep_pair(group, rep_case=rep_case)
    basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion="isotypic_expansion").to(device=device, dtype=dtype)

    batch_size = 4
    theta_batch = torch.randn(batch_size, basis.dim, device=device, dtype=dtype)

    W_batch = equiv_linear_map(theta_batch, in_rep, out_rep)
    W_cached = equiv_linear_map(theta_batch, in_rep, out_rep, tensor_cache=basis.tensor_cache)
    W_seq = torch.stack([equiv_linear_map(theta_batch[i], in_rep, out_rep) for i in range(batch_size)], dim=0)
    W_basis = basis(theta_batch)

    _assert_meta(W_batch, device=device, dtype=dtype)
    assert W_batch.shape == (batch_size, out_rep.size, in_rep.size)
    assert torch.allclose(W_batch, W_seq, atol=1e-5, rtol=1e-5), (
        f"Batched/seq synthesis mismatch, max error {(W_batch - W_seq).abs().max().item():.3e}"
    )
    assert torch.allclose(W_batch, W_cached, atol=1e-5, rtol=1e-5), (
        f"Rep-cache and tensor-cache synthesis mismatch, max error {(W_batch - W_cached).abs().max().item():.3e}"
    )
    assert torch.allclose(W_batch, W_basis, atol=1e-5, rtol=1e-5), (
        "Standalone synthesis mismatch against GroupHomomorphismBasis, "
        f"max error {(W_batch - W_basis).abs().max().item():.3e}"
    )

    W_single = W_batch[0]
    for g in group.elements:
        rho_out = torch.tensor(out_rep(g), device=device, dtype=dtype)
        rho_in = torch.tensor(in_rep(g), device=device, dtype=dtype)
        err = rho_out @ W_single - W_single @ rho_in
        assert torch.allclose(err, torch.zeros_like(err), atol=1e-5, rtol=1e-5), (
            f"Expanded map violates hom condition for {g}, max error {err.abs().max().item():.3e}"
        )
    if rep_case == "missing_irreps":
        _assert_missing_irrep_blocks_zero(W_batch, in_rep, out_rep, dtype=dtype)
