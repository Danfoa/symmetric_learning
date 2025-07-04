# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from copy import deepcopy

import escnn
import pytest
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, directsum
from escnn.nn import FieldType


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_irrep_pooling_equivariance(group: Group):
    """Check the IrrepSubspaceNormPooling layer is G-invariant."""
    import torch

    from symm_learning.nn import IrrepSubspaceNormPooling

    y_rep = directsum([group.regular_representation] * 10)  # ρ_Y = ρ_Χ ⊕ ρ_Χ
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[y_rep])
    pooling_layer = IrrepSubspaceNormPooling(in_type=type_Y)
    pooling_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    t_pooling_layer = pooling_layer.export()
    batch_size = 10
    y = type_Y(torch.randn(batch_size, type_Y.size, dtype=torch.float32))
    y_iso = pooling_layer(y).tensor
    y_iso_torch = t_pooling_layer(y.tensor)

    assert torch.allclose(y_iso, y_iso_torch, atol=1e-5, rtol=1e-5), (
        f"Max error: {torch.max(torch.abs(y_iso - y_iso_torch)):.5f}"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_change2disentangled_basis_equivariance(group: Group):  # noqa: D103
    import torch

    from symm_learning.nn import Change2DisentangledBasis
    from symm_learning.representation_theory import isotypic_decomp_rep

    y_rep = directsum([group.regular_representation] * 10)  # ρ_Y = ρ_Χ ⊕ ρ_Χ
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[y_rep])
    change_layer = Change2DisentangledBasis(in_type=type_Y)
    change_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    t_change_layter = change_layer.export()
    batch_size = 10
    y = torch.randn(batch_size, type_Y.size, dtype=torch.float32)
    rep_y = isotypic_decomp_rep(y_rep)
    Q_inv = torch.tensor(rep_y.change_of_basis_inv, dtype=torch.float32)
    y_iso = torch.einsum("ij,...j->...i", Q_inv, y)
    y_iso_nn = change_layer(type_Y(y)).tensor
    y_iso_torch = t_change_layter(y)

    assert torch.allclose(y_iso_nn, y_iso_torch, atol=1e-5, rtol=1e-5), (
        f"Max error: {torch.max(torch.abs(y_iso_nn - y_iso_torch)):.5f}"
    )
    assert torch.allclose(y_iso_nn, y_iso, atol=1e-5, rtol=1e-5), (
        f"Max error: {torch.max(torch.abs(y_iso_nn - y_iso)):.5f}"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [3])
@pytest.mark.parametrize("my", [3, 5])
def test_equiv_multivariate_normal(group: Group, mx: int, my: int):
    """Check the EquivMultivariateNormal layer is G-invariant."""
    import torch

    from symm_learning.nn import EquivMultivariateNormal, _EquivMultivariateNormal

    G = group
    x_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation] * mx)
    y_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation] * my)

    rep_x = x_type.representation
    G = rep_x.group

    e_normal = EquivMultivariateNormal(y_type, diagonal=True)

    e_normal.check_equivariance(atol=1e-6, rtol=1e-6)

    # Test that the exported torch module is also equivariant
    torch_e_normal: _EquivMultivariateNormal = e_normal.export()
    torch_e_normal.check_equivariance(in_type=e_normal.in_type, y_type=y_type, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1])
@pytest.mark.parametrize("my", [2])
@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [0])
def test_conv1d(group: Group, mx: int, my: int, kernel_size: int, stride: int, padding: int, bias: bool):
    """Check the eConv1D layer is G-invariant."""
    import torch

    from symm_learning.nn import GSpace1D, eConv1D, eConvTranspose1D

    G = group
    gspace = GSpace1D(G)
    in_type = FieldType(gspace, [G.regular_representation] * mx)
    out_type = FieldType(gspace, [G.regular_representation] * my)

    time = 10
    batch_size = 3
    x = torch.randn(batch_size, in_type.size, time)
    x = in_type(x)

    conv_layer = eConv1D(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    print(conv_layer)
    print("Weights shape:", conv_layer.weights.shape)
    print("Kernel shape:", conv_layer.kernel.shape)

    conv_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    y = conv_layer(x).tensor
    y_torch = conv_layer.export()(x.tensor)

    assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5)

    conv_transpose_layer = eConvTranspose1D(
        in_type=out_type,
        out_type=in_type,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    conv_transpose_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    x_trans = conv_transpose_layer(out_type(y)).tensor
    x_torch_trans = conv_transpose_layer.export()(y)
    assert torch.allclose(x_trans, x_torch_trans, atol=1e-5, rtol=1e-5)

    assert x_trans.shape == x.shape

    # conv1_layer = eConv1D(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    # conv2_layer = eConv1D(out_type, out_type, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    # cnn = escnn.nn.SequentialModule(conv1_layer, conv2_layer)
    # cnn.check_equivariance(atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_activations(group: Group):
    """Test custom activation functions for equivariance."""
    import torch
    from escnn.gspaces import no_base_space

    from symm_learning.nn import Mish

    gspace = no_base_space(group)
    in_type = FieldType(gspace, [group.regular_representation] * 3)

    mish_layer = Mish(in_type)
    mish_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    t_mish_layer = mish_layer.export()

    batch_size = 10
    x = torch.randn(batch_size, in_type.size, dtype=torch.float32)
    y_mish = mish_layer(in_type(x)).tensor
    y_mish_torch = t_mish_layer(x)

    assert torch.allclose(y_mish, y_mish_torch, atol=1e-5, rtol=1e-5), (
        f"Max error: {torch.max(torch.abs(y_mish - y_mish_torch)):.5f}"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [4])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("running_stats", [True, False])
def test_batchnorm1d(group: Group, mx: int, affine: bool, running_stats: bool):
    """Check the eBatchNorm1d layer is G-invariant."""
    import torch

    from symm_learning.nn import GSpace1D, eBatchNorm1d

    G = group
    gspace = GSpace1D(G)

    in_type = FieldType(gspace, [G.regular_representation] * mx)

    time = 2
    batch_size = 100
    x = torch.randn(batch_size, in_type.size, time)
    x = in_type(x)

    batchnorm_layer = eBatchNorm1d(in_type, affine=affine, track_running_stats=running_stats)

    if hasattr(batchnorm_layer, "affine_transform"):
        # Randomize the scale and bias DoFs
        batchnorm_layer.affine_transform.scale_dof.data.uniform_(-1, 1)
        if batchnorm_layer.affine_transform.bias:
            batchnorm_layer.affine_transform.bias_dof.data.uniform_(-1, 1)

    batchnorm_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    # y = batchnorm_layer(x).tensor
    # y_torch = batchnorm_layer.export()(x.tensor)

    # assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 10])
@pytest.mark.parametrize("bias", [True, False])
def test_affine(group: Group, mx: int, bias: bool):
    """Check the eBatchNorm1d layer is G-invariant."""
    import numpy as np
    import torch
    from escnn.gspaces import no_base_space

    from symm_learning.nn import GSpace1D, eAffine

    G = group
    rep = directsum([G.regular_representation] * mx)
    # Random orthogonal matrix for change of basis, using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, Q, name="test_rep")

    in_type = FieldType(no_base_space(G), representations=[rep])

    batch_size = 100
    x = torch.randn(batch_size, in_type.size)
    x = in_type(x)

    affine = eAffine(in_type, bias=bias)

    # Randomize the scale and bias DoFs
    affine.scale_dof.data.uniform_(-1, 1)
    if affine.bias:
        affine.bias_dof.data.uniform_(-1, 1)

    affine.check_equivariance(atol=1e-5, rtol=1e-5)

    in_type = FieldType(GSpace1D(G), [rep])
    time = 40
    x = torch.randn(batch_size, in_type.size, time)
    x = in_type(x)
    affine = eAffine(in_type, bias=bias)
    affine.check_equivariance(atol=1e-5, rtol=1e-5)
