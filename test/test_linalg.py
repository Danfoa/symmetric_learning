# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, IrreducibleRepresentation
from escnn.nn import FieldType

from symm_learning.linalg import (
    _project_to_irrep_endomorphism_basis,
    invariant_orthogonal_projector,
    irrep_radii,
    isotypic_signal2irreducible_subspaces,
    lstsq,
)
from symm_learning.representation_theory import direct_sum
from symm_learning.utils import check_equivariance


def _device_params():
    params = [pytest.param("cpu", id="cpu")]
    if torch.cuda.is_available():
        params.append(pytest.param("cuda", id="cuda"))
    return params


def _assert_meta(tensor: torch.Tensor, device: str, dtype: torch.dtype):
    assert tensor.device.type == torch.device(device).type
    assert tensor.dtype == dtype


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
