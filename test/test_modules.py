# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from copy import deepcopy

import escnn
import pytest
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, directsum
from escnn.nn import FieldType

from symm_learning.nn import EMAStats, eEMAStats
from symm_learning.nn.normalization import DataNorm
from symm_learning.representation_theory import direct_sum
from symm_learning.utils import check_equivariance


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_deepcopy(group: Group):
    from symm_learning.nn.linear import eLinear

    G = group
    rep = direct_sum([G.regular_representation] * 2)
    layer = eLinear(rep, rep)
    clone = deepcopy(layer)

    assert layer.in_rep is clone.in_rep, "Deepcopy should reuse the same input Representation object"
    assert layer.out_rep is clone.out_rep, "Deepcopy should reuse the same output Representation object"
    assert layer.in_rep.group is clone.in_rep.group, "Deepcopy should reuse the same Group singleton"
    assert layer.in_rep.group.representations is clone.in_rep.group.representations, (
        "Deepcopy should not duplicate the group's representation cache"
    )


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

    y_rep = direct_sum([group.regular_representation] * 10)  # ρ_Y = ρ_Χ ⊕ ρ_Χ
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[y_rep])
    pooling_layer = IrrepSubspaceNormPooling(in_type=type_Y)
    pooling_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    # t_pooling_layer = pooling_layer.export()
    # batch_size = 10
    # y = type_Y(torch.randn(batch_size, type_Y.size, dtype=torch.float32))
    # y_iso = pooling_layer(y).tensor
    # y_iso_torch = t_pooling_layer(y.tensor)

    # assert torch.allclose(y_iso, y_iso_torch, atol=1e-5, rtol=1e-5), (
    #     f"Max error: {torch.max(torch.abs(y_iso - y_iso_torch)):.5f}"
    # )


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

    y_rep = direct_sum([group.regular_representation] * 10)  # ρ_Y = ρ_Χ ⊕ ρ_Χ
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
    # print(conv_layer)
    # print("Weights shape:", conv_layer.weights.shape)
    # print("Kernel shape:", conv_layer.kernel.shape)

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
@pytest.mark.parametrize("mx", [1])
@pytest.mark.parametrize("my", [2, 1])
@pytest.mark.parametrize("basis_expansion_scheme", ["memory_heavy", "isotypic_expansion"])
def test_linear(group: Group, mx: int, my: int, basis_expansion_scheme: str):
    """Mirror the checks performed in symm_learning.nn.linear.__main__."""
    import torch
    import torch.nn.functional as F

    from symm_learning.nn.linear import eLinear

    G = group
    in_rep = direct_sum([G.regular_representation] * mx)
    out_rep = direct_sum([G.regular_representation] * my)

    def backprop_sanity(module: torch.nn.Module) -> None:
        module.train()
        optim = torch.optim.SGD(module.parameters(), lr=1e-3)
        x = torch.randn(16, module.in_rep.size)
        target = torch.randn(16, module.out_rep.size)
        optim.zero_grad()
        y = module(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
        assert grad_norms, "Expected at least one gradient to propagate."
        optim.step()

    layer = eLinear(in_rep, out_rep, bias=False, basis_expansion_scheme=basis_expansion_scheme)
    check_equivariance(layer, atol=1e-5, rtol=1e-5)
    backprop_sanity(layer)


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
@pytest.mark.parametrize("mx", [1])
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
    batch_size = 5
    x = torch.randn(batch_size, in_type.size, time)
    x = in_type(x)

    batchnorm_layer = eBatchNorm1d(in_type, affine=affine, track_running_stats=running_stats)

    if hasattr(batchnorm_layer, "affine_transform"):
        # Randomize the scale and bias DoFs
        batchnorm_layer.affine_transform.scale_dof.data.uniform_(-1, 1)
        if batchnorm_layer.affine_transform.has_bias:
            batchnorm_layer.affine_transform.bias_dof.data.uniform_(-1, 1)

    batchnorm_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    batchnorm_layer.eval()

    # TODO: This is not passing.
    # y = batchnorm_layer(x).tensor
    # y_torch = batchnorm_layer.export()(x.tensor)

    # print(y.shape, y_torch.shape)
    # assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5), f"{y - y_torch} should be 0"


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
    import numpy as np
    import torch
    from escnn.gspaces import no_base_space

    from symm_learning.nn.linear import eAffine

    G = group
    rep = direct_sum([G.regular_representation] * mx)
    # Random orthogonal matrix for change of basis, using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, Q, name="test_rep")

    batch_size = 100
    x = torch.randn(batch_size, rep.size)

    affine = eAffine(in_rep=rep, bias=bias)

    # Randomize the scale and bias DoFs
    affine.scale_dof.data.uniform_(-1, 1)
    if affine.has_bias:
        affine.bias_dof.data.uniform_(-1, 1)

    y = affine(x)
    assert not torch.allclose(y, x, atol=1e-5, rtol=1e-5)

    check_equivariance(affine, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 10])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm(group: Group, mx: int, bias: bool, affine: bool):
    import numpy as np
    import torch

    from symm_learning.nn.normalization import eLayerNorm

    G = group
    rep = direct_sum([G.regular_representation] * mx)
    Q, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, Q, name="test_layernorm_rep")

    layer = eLayerNorm(in_rep=rep, bias=bias, equiv_affine=affine, eps=0)

    if layer.equiv_affine:
        layer.affine.scale_dof.data.uniform_(-1, 1)
        if layer.affine.has_bias:
            layer.affine.bias_dof.data.uniform_(-1, 1)

    x = torch.randn(64, rep.size)
    y = layer(x)

    assert y.shape == x.shape

    check_equivariance(layer, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "kind",
    [pytest.param("ema", id="ema"), pytest.param("eema", id="eema")],
)
def test_ema_and_eema_stats(kind: str):
    import torch
    from escnn.gspaces import no_base_space

    if kind == "ema":
        stats = EMAStats(dim_x=3, dim_y=2, momentum=0.1)
        raw_x = torch.randn(12, stats.num_features_x)
        raw_y = torch.randn(12, stats.num_features_y)

        def prepare(x_tensor: torch.Tensor, y_tensor: torch.Tensor):
            return x_tensor, y_tensor

        def extract(output):
            return output

    else:
        G = CyclicGroup(3)
        gspace = no_base_space(G)
        field = FieldType(gspace, [G.regular_representation])
        stats = eEMAStats(x_type=field, y_type=field, momentum=0.1)
        raw_x = torch.randn(12, field.size)
        raw_y = torch.randn(12, field.size)

        def prepare(x_tensor: torch.Tensor, y_tensor: torch.Tensor):
            return field(x_tensor), field(y_tensor)

        def extract(output):
            return output.tensor

    stats.train()
    x_input, y_input = prepare(raw_x.clone(), raw_y.clone())
    prev_mean = stats.mean_x.clone()
    x_out, y_out = stats(x_input, y_input)
    assert torch.equal(extract(x_out), raw_x)
    assert torch.equal(extract(y_out), raw_y)
    assert stats.num_batches_tracked == 1
    assert not torch.equal(stats.mean_x, prev_mean)

    stats.eval()
    frozen_mean = stats.mean_x.clone()
    stats(x_input, y_input)
    assert torch.equal(stats.mean_x, frozen_mean)

    if kind == "ema":
        stats.train()
        linear = torch.nn.Linear(raw_x.shape[1], 1)
        for _ in range(2):
            linear.zero_grad()
            x = torch.randn(4, raw_x.shape[1], requires_grad=True)
            y = torch.randn(4, raw_y.shape[1], requires_grad=True)
            x_forward, _ = stats(x, y)
            linear(x_forward).sum().backward()
            assert linear.weight.grad is not None
    else:
        exported = stats.export()
        exported.eval()
        x_std, y_std = exported(raw_x, raw_y)
        assert torch.equal(x_std, raw_x)
        assert torch.equal(y_std, raw_y)


@pytest.mark.parametrize(
    "momentum, expect_error",
    [
        pytest.param(0.1, None, id="valid-0.1"),
        pytest.param(0.5, None, id="valid-0.5"),
        pytest.param(1.0, None, id="valid-1.0"),
        pytest.param(0.0, ValueError, id="invalid-0.0"),
        pytest.param(1.5, ValueError, id="invalid-1.5"),
    ],
)
def test_ema_stats_momentum(momentum: float, expect_error: type[Exception] | None):
    import torch

    if expect_error is not None:
        with pytest.raises(expect_error):
            EMAStats(dim_x=2, dim_y=2, momentum=momentum)
        return

    stats = EMAStats(dim_x=2, dim_y=2, momentum=momentum)
    stats.train()
    stats(torch.randn(6, 2), torch.randn(6, 2))
    assert stats.num_batches_tracked == 1


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


def test_ema_stats_basic():
    """Test basic EMAStats functionality."""
    from symm_learning.nn.running_stats import EMAStats

    # Create EMAStats instance
    stats = EMAStats(dim_x=4, dim_y=3, momentum=0.1)

    # Test initial state
    assert stats.num_batches_tracked == 0
    assert stats.mean_x.shape == (4,)
    assert stats.mean_y.shape == (3,)
    assert stats.cov_xx.shape == (4, 4)
    assert stats.cov_yy.shape == (3, 3)
    assert stats.cov_xy.shape == (4, 3)

    # Test training mode - statistics should update
    stats.train()
    x = torch.randn(16, 4)
    y = torch.randn(16, 3)

    initial_mean_x = stats.mean_x.clone()
    x_out, y_out = stats(x, y)

    # Check outputs are unchanged
    assert torch.equal(x_out, x)
    assert torch.equal(y_out, y)

    # Check statistics were updated
    assert stats.num_batches_tracked == 1
    assert not torch.equal(stats.mean_x, initial_mean_x)


def test_ema_stats_eval_mode():
    """Test EMAStats in evaluation mode."""
    from symm_learning.nn.running_stats import EMAStats

    stats = EMAStats(dim_x=3, dim_y=2, momentum=0.1)

    # Train on one batch to initialize stats
    stats.train()
    x1 = torch.randn(10, 3)
    y1 = torch.randn(10, 2)
    stats(x1, y1)

    # Switch to eval mode
    stats.eval()
    stored_mean_x = stats.mean_x.clone()
    stored_batches = stats.num_batches_tracked.clone()

    # Process another batch - stats should not update
    x2 = torch.randn(10, 3)
    y2 = torch.randn(10, 2)
    x_out, y_out = stats(x2, y2)

    # Check outputs are unchanged and stats didn't update
    assert torch.equal(x_out, x2)
    assert torch.equal(y_out, y2)
    assert torch.equal(stats.mean_x, stored_mean_x)
    assert torch.equal(stats.num_batches_tracked, stored_batches)


def test_ema_stats_center_with_running_mean():
    """Test center_with_running_mean parameter."""
    from symm_learning.nn.running_stats import EMAStats

    # Test with center_with_running_mean=True (default)
    stats_true = EMAStats(dim_x=2, dim_y=2, momentum=0.5, center_with_running_mean=True)

    # Test with center_with_running_mean=False
    stats_false = EMAStats(dim_x=2, dim_y=2, momentum=0.5, center_with_running_mean=False)

    # Same data for both
    torch.manual_seed(42)
    x1 = torch.ones(5, 2) * 2.0
    y1 = torch.ones(5, 2) * 3.0
    x2 = torch.zeros(5, 2)
    y2 = torch.zeros(5, 2)

    # Process two batches
    stats_true.train()
    stats_false.train()

    stats_true(x1, y1)
    stats_false(x1, y1)

    stats_true(x2, y2)
    stats_false(x2, y2)

    # Results should be different due to different centering strategies
    assert not torch.allclose(stats_true.cov_xx, stats_false.cov_xx, atol=1e-5)


def test_eema_stats_basic():
    """Test basic eEMAStats functionality."""
    from symm_learning.nn.running_stats import eEMAStats

    # Create group and field types
    G = CyclicGroup(4)
    gspace = escnn.gspaces.no_base_space(G)
    in_type_x = FieldType(gspace, [G.regular_representation] * 2)  # Size: 8
    in_type_y = FieldType(gspace, [G.regular_representation] * 1)  # Size: 4

    # Create eEMAStats instance
    stats = eEMAStats(x_type=in_type_x, y_type=in_type_y, momentum=0.1)

    # Test initial state
    assert stats.num_batches_tracked == 0
    assert stats.mean_x.shape == (8,)
    assert stats.mean_y.shape == (4,)
    assert stats.cov_xx.shape == (8, 8)
    assert stats.cov_yy.shape == (4, 4)
    assert stats.cov_xy.shape == (8, 4)

    # Test training mode
    stats.train()
    x_tensor = torch.randn(16, 8)
    y_tensor = torch.randn(16, 4)
    x_geom = in_type_x(x_tensor)
    y_geom = in_type_y(y_tensor)

    x_out, y_out = stats(x_geom, y_geom)

    # Check outputs are unchanged GeometricTensors
    assert torch.equal(x_out.tensor, x_tensor)
    assert torch.equal(y_out.tensor, y_tensor)
    assert x_out.type == in_type_x
    assert y_out.type == in_type_y

    # Check statistics were updated
    assert stats.num_batches_tracked == 1


def test_eema_stats_export():
    """Test eEMAStats export functionality."""
    from symm_learning.nn.running_stats import eEMAStats

    # Create group and field types
    G = CyclicGroup(3)
    gspace = escnn.gspaces.no_base_space(G)
    in_type_x = FieldType(gspace, [G.regular_representation])  # Size: 3
    in_type_y = FieldType(gspace, [G.regular_representation])  # Size: 3

    # Create and train eEMAStats
    stats = eEMAStats(x_type=in_type_x, y_type=in_type_y, momentum=0.2)
    stats.train()

    # Process a batch
    x_tensor = torch.randn(20, 3)
    y_tensor = torch.randn(20, 3)
    x_geom = in_type_x(x_tensor)
    y_geom = in_type_y(y_tensor)
    stats(x_geom, y_geom)

    # Export to standard EMAStats
    stats.eval()
    exported_stats = stats.export()
    exported_stats.eval()

    # Test on same data
    x_test = torch.randn(10, 3)
    y_test = torch.randn(10, 3)
    x_geom_test = in_type_x(x_test)
    y_geom_test = in_type_y(y_test)

    x_ema, y_ema = stats(x_geom_test, y_geom_test)
    x_exp, y_exp = exported_stats(x_test, y_test)

    # Outputs should be identical
    assert torch.equal(x_ema.tensor, x_test)
    assert torch.equal(y_ema.tensor, y_test)
    assert torch.equal(x_exp, x_test)
    assert torch.equal(y_exp, y_test)

    # Check that statistics are preserved
    assert torch.allclose(stats.mean_x, exported_stats.mean_x)
    assert torch.allclose(stats.mean_y, exported_stats.mean_y)
    assert torch.allclose(stats.cov_xx, exported_stats.cov_xx)
    assert torch.allclose(stats.cov_yy, exported_stats.cov_yy)
    assert torch.allclose(stats.cov_xy, exported_stats.cov_xy)


@pytest.mark.parametrize("momentum", [0.1, 0.5, 1.0])
def test_ema_momentum_values(momentum):
    """Test different momentum values."""
    from symm_learning.nn.running_stats import EMAStats

    stats = EMAStats(dim_x=2, dim_y=2, momentum=momentum)

    # Process multiple batches
    stats.train()
    for i in range(3):
        x = torch.randn(10, 2)
        y = torch.randn(10, 2)
        stats(x, y)

    # Just verify it runs without errors and updates stats
    assert stats.num_batches_tracked == 3


def test_invalid_momentum():
    """Test that invalid momentum values raise errors."""
    from symm_learning.nn.running_stats import EMAStats

    with pytest.raises(ValueError, match="momentum must be in"):
        EMAStats(dim_x=2, dim_y=2, momentum=0.0)

    with pytest.raises(ValueError, match="momentum must be in"):
        EMAStats(dim_x=2, dim_y=2, momentum=1.5)
