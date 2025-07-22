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


@pytest.mark.parametrize("num_features", [10, 50])
@pytest.mark.parametrize("momentum", [None, 0.1, 0.9])
@pytest.mark.parametrize("ndim", [2, 3])
def test_data_norm(num_features: int, momentum: float, ndim: int):
    """Test DataNorm layer for basic functionality and running statistics."""
    import torch

    from symm_learning.nn import DataNorm

    # Test with online statistics (no mean/std provided)
    data_norm = DataNorm(num_features=num_features, momentum=momentum)

    # Create test data
    batch_size = 20
    if ndim == 2:
        x = torch.randn(batch_size, num_features)
    else:  # ndim == 3
        time_steps = 15
        x = torch.randn(batch_size, num_features, time_steps)

    # Test training mode - should update running stats
    data_norm.train()

    # First forward pass should initialize running stats
    y1 = data_norm(x)
    assert y1.shape == x.shape, "Output shape should match input shape"
    assert data_norm.num_batches_tracked == 1, "Should track one batch"

    # Second forward pass should update running stats
    y2 = data_norm(x)
    assert data_norm.num_batches_tracked == 2, "Should track two batches"

    # Test eval mode - should use running stats
    data_norm.eval()
    y3 = data_norm(x)
    assert data_norm.num_batches_tracked == 2, "Should not update in eval mode"

    # Test with pre-computed stats
    dims = [0] if ndim == 2 else [0, 2]
    mean = x.mean(dim=dims)
    std = torch.sqrt(x.var(dim=dims, unbiased=False) + 1e-5)

    data_norm_fixed = DataNorm(num_features=num_features, mean=mean, std=std)
    y_fixed = data_norm_fixed(x)
    assert y_fixed.shape == x.shape, "Output shape should match input shape"
    assert not hasattr(data_norm_fixed, "num_batches_tracked"), "Should not track stats with fixed mean/std"

    # Test normalization properties (approximately zero mean, unit std)
    y_mean = y_fixed.mean(dim=dims)
    y_std = y_fixed.std(dim=dims, unbiased=False)
    assert torch.allclose(y_mean, torch.zeros_like(y_mean), atol=1e-4), "Mean should be close to zero"
    assert torch.allclose(y_std, torch.ones_like(y_std), atol=1e-3), "Std should be close to one"


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
    ],
)
@pytest.mark.parametrize("mx", [2, 4])
@pytest.mark.parametrize("momentum", [None, 0.1])
@pytest.mark.parametrize("ndim", [2, 3])
def test_edata_norm(group: Group, mx: int, momentum: float, ndim: int):
    """Test eDataNorm layer for equivariance and basic functionality."""
    import torch
    from escnn.gspaces import no_base_space

    from symm_learning.nn import GSpace1D, eDataNorm

    G = group
    if ndim == 2:
        gspace = no_base_space(G)
    else:  # ndim == 3
        gspace = GSpace1D(G)

    in_type = FieldType(gspace, [G.regular_representation] * mx)

    # Test with online statistics (no mean/std provided)
    edata_norm = eDataNorm(in_type=in_type, momentum=momentum)

    # Create test data
    batch_size = 20
    if ndim == 2:
        x = torch.randn(batch_size, in_type.size)
    else:  # ndim == 3
        time_steps = 15
        x = torch.randn(batch_size, in_type.size, time_steps)
    x = in_type(x)

    # Test training mode - should update running stats
    edata_norm.train()

    # First forward pass should initialize running stats
    y1 = edata_norm(x)
    assert y1.shape == x.shape, "Output shape should match input shape"
    assert y1.type == in_type, "Output type should match input type"
    assert edata_norm.num_batches_tracked == 1, "Should track one batch"

    # Second forward pass should update running stats
    y2 = edata_norm(x)
    assert edata_norm.num_batches_tracked == 2, "Should track two batches"

    # Test eval mode - should use running stats
    edata_norm.eval()
    y3 = edata_norm(x)
    assert edata_norm.num_batches_tracked == 2, "Should not update in eval mode"

    # Test equivariance
    edata_norm.check_equivariance(atol=1e-4, rtol=1e-4)

    # Test with pre-computed stats using symmetry-aware computation
    from symm_learning.stats import var_mean

    var_batch, mean_batch = var_mean(x.tensor, rep_x=in_type.representation)
    std_batch = torch.sqrt(var_batch + 1e-5)

    edata_norm_fixed = eDataNorm(in_type=in_type, mean=mean_batch, std=std_batch)
    y_fixed = edata_norm_fixed(x)
    assert y_fixed.shape == x.shape, "Output shape should match input shape"
    assert y_fixed.type == in_type, "Output type should match input type"
    assert not hasattr(edata_norm_fixed, "num_batches_tracked"), "Should not track stats with fixed mean/std"

    # Test that fixed stats version is also equivariant
    edata_norm_fixed.check_equivariance(atol=1e-4, rtol=1e-4)
