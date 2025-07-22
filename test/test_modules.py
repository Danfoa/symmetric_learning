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


@pytest.mark.parametrize("running_stats", [True, False])
@pytest.mark.parametrize("only_centering", [True, False])
@pytest.mark.parametrize("compute_cov", [True, False])
@pytest.mark.parametrize("momentum", [0.1, None])
def test_data_norm(running_stats: bool, only_centering: bool, compute_cov: bool, momentum: float | None):
    """Test DataNorm layer functionality."""
    import torch

    from symm_learning.nn import DataNorm

    num_features = 10
    batch_size = 20

    # Create layer
    norm_layer = DataNorm(
        num_features=num_features,
        running_stats=running_stats,
        only_centering=only_centering,
        compute_cov=compute_cov,
        momentum=momentum,
    )

    # Test 2D input
    x_2d = torch.randn(batch_size, num_features)

    # Test 3D input
    time = 15
    x_3d = torch.randn(batch_size, num_features, time)

    # Training mode - should update running stats
    norm_layer.train()

    if running_stats:
        initial_mean = norm_layer.running_mean.clone()
        initial_std = norm_layer.running_std.clone()
        if compute_cov and hasattr(norm_layer, "running_cov"):
            initial_cov = norm_layer.running_cov.clone()

    # Forward pass in training
    y_2d = norm_layer(x_2d)
    y_3d = norm_layer(x_3d)

    # Check output shapes
    assert y_2d.shape == x_2d.shape
    assert y_3d.shape == x_3d.shape

    # Check stats were updated in training mode (if running_stats=True)
    if running_stats:
        assert not torch.allclose(norm_layer.running_mean, initial_mean), "Running mean should have updated"
        assert not torch.allclose(norm_layer.running_std, initial_std), "Running std should have updated"
        if compute_cov and hasattr(norm_layer, "running_cov"):
            assert not torch.allclose(norm_layer.running_cov, initial_cov), "Running cov should have updated"

    # Eval mode - should NOT update running stats
    norm_layer.eval()

    if running_stats:
        eval_mean = norm_layer.running_mean.clone()
        eval_std = norm_layer.running_std.clone()
        if compute_cov and hasattr(norm_layer, "running_cov"):
            eval_cov = norm_layer.running_cov.clone()

    # Forward pass in eval
    _ = norm_layer(x_2d)
    _ = norm_layer(x_3d)

    # Check stats were NOT updated in eval mode
    if running_stats:
        assert torch.allclose(norm_layer.running_mean, eval_mean), "Running mean should not update in eval"
        assert torch.allclose(norm_layer.running_std, eval_std), "Running std should not update in eval"
        if compute_cov and hasattr(norm_layer, "running_cov"):
            assert torch.allclose(norm_layer.running_cov, eval_cov), "Running cov should not update in eval"

    # Test covariance property
    if compute_cov:
        cov_matrix = norm_layer.cov
        assert cov_matrix.shape == (num_features, num_features)
    else:
        with pytest.raises(RuntimeError, match="Covariance computation is disabled"):
            _ = norm_layer.cov

    # Test mean/std properties
    mean_val = norm_layer.mean
    std_val = norm_layer.std
    assert mean_val.shape == (num_features,)
    assert std_val.shape == (num_features,)

    # Test cumulative averaging when momentum=None
    if running_stats and momentum is None:
        # Create a fresh layer for cumulative averaging test
        cum_layer = DataNorm(
            num_features=num_features,
            running_stats=True,
            momentum=None,
            compute_cov=compute_cov,
        )
        cum_layer.train()

        # Process multiple batches and manually compute expected cumulative averages
        batch_means = []
        batch_stds = []
        if compute_cov:
            batch_covs = []

        for i in range(3):  # Process 3 batches
            x_batch = torch.randn(batch_size, num_features)
            batch_mean = x_batch.mean(dim=0)
            batch_std = torch.sqrt(x_batch.var(dim=0, unbiased=False))
            batch_means.append(batch_mean)
            batch_stds.append(batch_std)

            if compute_cov:
                x_centered = x_batch - x_batch.mean(dim=0, keepdim=True)
                batch_cov = torch.mm(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
                batch_covs.append(batch_cov)

            _ = cum_layer(x_batch)

            # Expected cumulative average after (i+1) batches
            expected_mean = torch.stack(batch_means[: i + 1]).mean(dim=0)
            expected_std = torch.stack(batch_stds[: i + 1]).mean(dim=0)

            assert torch.allclose(cum_layer.running_mean, expected_mean, atol=1e-5), (
                f"Cumulative mean incorrect after batch {i + 1}"
            )
            assert torch.allclose(cum_layer.running_std, expected_std, atol=1e-5), (
                f"Cumulative std incorrect after batch {i + 1}"
            )

            if compute_cov and hasattr(cum_layer, "running_cov"):
                expected_cov = torch.stack(batch_covs[: i + 1]).mean(dim=0)
                assert torch.allclose(cum_layer.running_cov, expected_cov, atol=1e-5), (
                    f"Cumulative cov incorrect after batch {i + 1}"
                )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("running_stats", [True, False])
@pytest.mark.parametrize("only_centering", [True, False])
@pytest.mark.parametrize("compute_cov", [True, False])
def test_edata_norm(group: Group, running_stats: bool, only_centering: bool, compute_cov: bool):
    """Test eDataNorm layer functionality and equivariance."""
    import torch
    from escnn.gspaces import no_base_space

    from symm_learning.nn import eDataNorm
    from symm_learning.stats import var_mean, cov

    G = group
    gspace = no_base_space(G)
    in_type = FieldType(gspace, [G.regular_representation] * 2)

    batch_size = 50
    x = torch.randn(batch_size, in_type.size)
    x_geom = in_type(x)

    # Create layer
    norm_layer = eDataNorm(
        in_type=in_type,
        running_stats=running_stats,
        only_centering=only_centering,
        compute_cov=compute_cov,
    )

    # Test that it uses equivariant statistics
    norm_layer.train()

    # Forward pass
    y = norm_layer(x_geom)

    # Check output type and shape
    assert y.type == in_type
    assert y.tensor.shape == x.shape

    # Test that the statistics computed are equivariant
    if running_stats and norm_layer.training:
        # Get the statistics from the layer
        layer_mean = norm_layer.running_mean
        layer_std = norm_layer.running_std

        # Compute equivariant statistics directly
        expected_var, expected_mean = var_mean(x, rep_x=in_type.representation)
        expected_std = torch.sqrt(expected_var)

        # Should match (approximately, due to running average)
        assert torch.allclose(layer_mean, expected_mean, atol=1e-4), (
            f"Layer mean {layer_mean} doesn't match equivariant mean {expected_mean}"
        )
        assert torch.allclose(layer_std, expected_std, atol=1e-4), (
            f"Layer std {layer_std} doesn't match equivariant std {expected_std}"
        )

    # Test covariance computation if enabled
    if compute_cov:
        cov_matrix = norm_layer.cov
        assert cov_matrix.shape == (in_type.size, in_type.size)

        # Check that it matches equivariant covariance
        expected_cov = cov(x, x, rep_x=in_type.representation, rep_y=in_type.representation)
        assert torch.allclose(cov_matrix, expected_cov, atol=1e-4), (
            f"Layer covariance doesn't match equivariant covariance"
        )

    # Test export to DataNorm
    exported = norm_layer.export()
    from symm_learning.nn import DataNorm

    assert isinstance(exported, DataNorm)
    assert exported.num_features == in_type.size
    assert exported.running_stats == running_stats
    assert exported.only_centering == only_centering
    assert exported.compute_cov == compute_cov

    # Test that exported layer produces same output on tensor data
    # Both layers should be in eval mode for consistent comparison
    norm_layer.eval()
    exported.eval()
    y_exported = exported(x)
    y_eval = norm_layer(x_geom)
    assert torch.allclose(y_eval.tensor, y_exported, atol=1e-6), "Exported layer should produce same output as original"

    # Test equivariance: E[g·x] = g·E[x] and Var[g·x] = Var[x]
    # Sample a group element
    g = G.sample()
    if g == G.identity:
        return  # Skip identity

    gx = x_geom.transform(g)

    # Compute statistics for transformed input
    g_var, g_mean = var_mean(gx.tensor, rep_x=in_type.representation)
    orig_var, orig_mean = var_mean(x, rep_x=in_type.representation)

    # Mean should transform: g·mean = mean_transformed
    expected_g_mean = in_type(orig_mean.unsqueeze(0)).transform(g).tensor.squeeze(0)

    assert torch.allclose(g_mean, expected_g_mean, atol=1e-4), (
        f"Equivariant mean property violated: {g_mean} vs {expected_g_mean}"
    )

    # Variance should be invariant: Var[g·x] = Var[x]
    assert torch.allclose(g_var, orig_var, atol=1e-4), f"Equivariant variance property violated: {g_var} vs {orig_var}"
