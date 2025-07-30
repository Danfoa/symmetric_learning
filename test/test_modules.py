# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from copy import deepcopy

import escnn
import pytest
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, directsum
from escnn.nn import FieldType

from symm_learning.nn.normalization import DataNorm


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

    y = batchnorm_layer(x).tensor
    y_torch = batchnorm_layer.export()(x.tensor)

    assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5)


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


import torch
import symm_learning


def _test_datanorm_layer(datanorm: DataNorm):
    # Test training mode
    datanorm.train()

    def input_type(x):
        if hasattr(datanorm, "in_type"):
            return datanorm.in_type(x)
        return x

    num_features = datanorm.num_features
    only_centering = datanorm.only_centering
    compute_cov = datanorm.compute_cov

    batch_size = 50

    # Store initial running stats
    initial_mean = datanorm.running_mean.clone()
    initial_var = datanorm.running_var.clone()
    initial_batches_tracked = datanorm.num_batches_tracked.clone()

    # Forward pass in training mode
    for _ in range(5):  # Process several batches to stabilize running stats
        x = torch.randn(batch_size, num_features)
        y = datanorm(input_type(x))

    # Check that running stats were updated
    assert not torch.allclose(datanorm.running_mean, initial_mean), "Running mean should be updated during training"
    if not only_centering:
        assert not torch.allclose(datanorm.running_var, initial_var), "Running var should be updated during training"
    assert datanorm.num_batches_tracked > initial_batches_tracked, "Batch counter should be incremented"

    # Test evaluation mode - stats should not update
    datanorm.eval()
    prev_mean = datanorm.running_mean.clone()
    prev_var = datanorm.running_var.clone()
    prev_batches_tracked = datanorm.num_batches_tracked.clone()

    x_eval = torch.randn(batch_size, num_features)
    y_eval = datanorm(input_type(x_eval))

    # Check that running stats were NOT updated in eval mode
    assert torch.allclose(datanorm.running_mean, prev_mean), "Running mean should not be updated during evaluation"
    assert torch.allclose(datanorm.running_var, prev_var), "Running var should not be updated during evaluation"
    assert torch.equal(datanorm.num_batches_tracked, prev_batches_tracked), (
        "Batch counter should not change during evaluation"
    )

    # Test only_centering behavior
    if only_centering:
        # When only_centering=True, the layer should track running_var as all ones
        datanorm.train()
        # Process several batches to verify variance stays as ones
        for _ in range(5):
            x_batch = torch.randn(batch_size, num_features)
            y_batch = datanorm(input_type(x_batch))

        # Variance should remain all ones (since batch_var is set to ones)
        assert torch.allclose(datanorm.running_var, torch.ones_like(datanorm.running_var)), (
            "When only_centering=True, running_var should remain all ones"
        )
    else:
        # When only_centering=False, output should be normalized and running_var should be tracked
        datanorm.train()
        # Process several batches to get stable running stats
        for _ in range(10):
            x_batch = torch.randn(batch_size, num_features)
            _ = datanorm(input_type(x_batch))

        # Check that running_var is NOT all ones (it should be tracking actual variance)
        assert not torch.allclose(datanorm.running_var, torch.ones_like(datanorm.running_var)), (
            "When only_centering=False, running_var should track actual variance, not remain all ones"
        )

    # Test covariance computation
    if compute_cov:
        assert hasattr(datanorm, "running_cov"), "Should have running_cov attribute when compute_cov=True"
        assert datanorm.running_cov.shape == (num_features, num_features), "Covariance should be square matrix"
        cov = datanorm.cov  # Test property access
        assert cov.shape == (num_features, num_features), "Covariance property should return correct shape"
    else:
        with pytest.raises(RuntimeError, match="Covariance computation is disabled"):
            _ = datanorm.cov  # Should raise error when compute_cov=False


@pytest.mark.parametrize("num_features", [1, 10, 50])
@pytest.mark.parametrize("only_centering", [True, False])
@pytest.mark.parametrize("compute_cov", [True, False])
@pytest.mark.parametrize("momentum", [0.1, 1.0])
def test_datanorm(num_features: int, only_centering: bool, compute_cov: bool, momentum: float):
    """Test DataNorm layer functionality."""
    import torch

    from symm_learning.nn import DataNorm

    batch_size = 50
    eps = 1e-6

    # Test 2D input
    datanorm = DataNorm(
        num_features=num_features,
        eps=eps,
        only_centering=only_centering,
        compute_cov=compute_cov,
        momentum=momentum,
    )

    _test_datanorm_layer(datanorm)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [4])
@pytest.mark.parametrize("only_centering", [True, False])
@pytest.mark.parametrize("compute_cov", [True, False])
@pytest.mark.parametrize("momentum", [0.1, 1.0])
def test_edatanorm(group: Group, mx: int, only_centering: bool, compute_cov: bool, momentum: float):
    """Test eDataNorm layer functionality and equivariance."""
    import torch

    from symm_learning.nn import eDataNorm

    G = group
    gspace = escnn.gspaces.no_base_space(G)
    in_type = FieldType(gspace, [G.regular_representation] * mx)

    eps = 1e-6

    # Create eDataNorm layer
    edatanorm = eDataNorm(
        in_type=in_type,
        eps=eps,
        only_centering=only_centering,
        compute_cov=compute_cov,
        momentum=momentum,
    )

    _test_datanorm_layer(edatanorm)

    # Test export functionality
    edatanorm.eval()
    datanorm = edatanorm.export()
    datanorm.eval()

    # Test on 2D input
    x_test_2d = torch.randn(100, in_type.size)
    x_test_2d_geom = in_type(x_test_2d)

    # Get outputs from both layers
    y_edatanorm = edatanorm(x_test_2d_geom).tensor
    y_exported = datanorm(x_test_2d)

    # They should be exactly the same
    assert torch.allclose(y_edatanorm, y_exported, atol=1e-6, rtol=1e-6), (
        f"eDataNorm and exported DataNorm should produce identical results for 2D input. "
        f"Max diff: {torch.max(torch.abs(y_edatanorm - y_exported)):.6f}"
    )
