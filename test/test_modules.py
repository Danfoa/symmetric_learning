from __future__ import annotations

# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from copy import deepcopy

import escnn
import pytest
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, Representation, directsum
import torch

from symm_learning.representation_theory import direct_sum
from symm_learning.utils import backprop_sanity, check_equivariance
from utils import assert_module_save_load_consistency


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_deepcopy(group: Group):
    import torch

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

    assert_module_save_load_consistency(layer, torch.randn(6, rep.size))


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
def test_change2disentangled(group: Group):  # noqa: D103
    import torch

    from symm_learning.nn import Change2DisentangledBasis
    from symm_learning.representation_theory import isotypic_decomp_rep

    y_rep = direct_sum([group.regular_representation] * 10)  # ρ_Y = ρ_Χ ⊕ ρ_Χ
    change_layer = Change2DisentangledBasis(in_rep=y_rep)
    check_equivariance(change_layer, atol=1e-5, rtol=1e-5)

    batch_size = 10
    y = torch.randn(batch_size, y_rep.size, dtype=torch.float32)
    rep_y = isotypic_decomp_rep(y_rep)
    Q_inv = torch.tensor(rep_y.change_of_basis_inv, dtype=torch.float32)
    y_iso = torch.einsum("ij,...j->...i", Q_inv, y)
    y_iso_nn = change_layer(y)
    assert torch.allclose(y_iso_nn, y_iso, atol=1e-5, rtol=1e-5), (
        f"Max error: {torch.max(torch.abs(y_iso_nn - y_iso)):.5f}"
    )

    assert_module_save_load_consistency(change_layer, y)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [2])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("num_layers", [1, 5])
def test_etransformer_decoder(group: Group, mx: int, num_heads: int, num_layers: int):
    """Check equivariance and fast inference consistency of eTransformerDecoderLayer."""
    from symm_learning.nn import eTransformerDecoderLayer

    G = group
    rep = direct_sum([G.regular_representation] * mx)

    decoder_kwargs = dict(
        in_rep=rep,
        nhead=num_heads,
        dim_feedforward=rep.size * 4,
        dropout=0.0,  # dropout=0 for train/eval consistency
        activation="relu",
        norm_first=True,
        batch_first=True,
        norm_module="rmsnorm",
        bias=True,
    )

    # Create single layer or stacked layers
    if num_layers == 1:
        decoder = eTransformerDecoderLayer(**decoder_kwargs)
    else:
        base_layer = eTransformerDecoderLayer(**decoder_kwargs)
        decoder = torch.nn.TransformerDecoder(decoder_layer=base_layer, num_layers=num_layers)

    # Equivariance check
    decoder.eval()
    if num_layers == 1:
        decoder.check_equivariance(atol=1e-4, rtol=1e-4)
    else:
        # For stacked layers, use manual equivariance check
        def act(rep_local, g, tensor):
            mat = torch.tensor(rep_local(g), dtype=tensor.dtype, device=tensor.device)
            return torch.einsum("ij,...j->...i", mat, tensor)

        B, tgt_len, mem_len = 3, 2, 3
        for _ in range(5):
            g = G.sample()
            tgt = torch.randn(B, tgt_len, rep.size)
            mem = torch.randn(B, mem_len, rep.size)
            out = decoder(tgt=tgt, memory=mem)
            g_out = decoder(tgt=act(rep, g, tgt), memory=act(rep, g, mem))
            g_out_exp = act(rep, g, out)
            assert torch.allclose(g_out, g_out_exp, atol=1e-3, rtol=1e-3), (
                f"Decoder stack equivariance failed, max err {(g_out - g_out_exp).abs().max().item():.3e}"
            )

    # Fast inference consistency test
    B, tgt_len, mem_len = 4, 3, 5
    tgt = torch.randn(B, tgt_len, rep.size)
    mem = torch.randn(B, mem_len, rep.size)

    # 1. Update weights with some arbitrary loss
    decoder.train()
    decoder.zero_grad()
    y = decoder(tgt, mem) if num_layers == 1 else decoder(tgt=tgt, memory=mem)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    with torch.no_grad():
        for p in decoder.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad

    # 2. Forward in train mode with updated weights
    decoder.zero_grad()
    y_train = decoder(tgt, mem) if num_layers == 1 else decoder(tgt=tgt, memory=mem)

    # 3. Forward in eval mode
    decoder.eval()
    y_eval = decoder(tgt, mem) if num_layers == 1 else decoder(tgt=tgt, memory=mem)

    assert torch.allclose(y_train, y_eval, atol=1e-5, rtol=1e-5), (
        f"eTransformerDecoder output in eval mode must match train mode (layers={num_layers})"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [2])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("num_layers", [1, 5])
def test_etransformer_encoder(group: Group, mx: int, num_heads: int, num_layers: int):
    """Check equivariance and fast inference consistency of eTransformerEncoderLayer."""
    from symm_learning.nn.transformer.etransformer import eTransformerEncoderLayer

    G = group
    rep = direct_sum([G.regular_representation] * mx)

    encoder_kwargs = dict(
        in_rep=rep,
        nhead=num_heads,
        dim_feedforward=rep.size * 4,
        dropout=0.0,  # dropout=0 for train/eval consistency
        activation="relu",
        norm_first=True,
        batch_first=True,
        norm_module="rmsnorm",
        bias=True,
    )

    # Create single layer or stacked layers
    if num_layers == 1:
        encoder = eTransformerEncoderLayer(**encoder_kwargs)
    else:
        base_layer = eTransformerEncoderLayer(**encoder_kwargs)
        encoder = torch.nn.TransformerEncoder(
            encoder_layer=base_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    # Equivariance check
    encoder.eval()
    check_equivariance(
        encoder,
        input_dim=3,
        module_name=f"eTransformerEncoder(layers={num_layers})",
        in_rep=rep,
        out_rep=rep,
        atol=1e-4,
        rtol=1e-4,
    )

    # Fast inference consistency test
    B, L = 4, 5
    x = torch.randn(B, L, rep.size)

    # 1. Update weights with some arbitrary loss
    encoder.train()
    encoder.zero_grad()
    y = encoder(x)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    with torch.no_grad():
        for p in encoder.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad

    # 2. Forward in train mode with updated weights
    encoder.zero_grad()
    y_train = encoder(x)

    # 3. Forward in eval mode
    encoder.eval()
    y_eval = encoder(x)

    assert torch.allclose(y_train, y_eval, atol=1e-5, rtol=1e-5), (
        f"eTransformerEncoder output in eval mode must match train mode (layers={num_layers})"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [3])
@pytest.mark.parametrize("my", [3, 5])
def test_equiv_multivariate_normal(group: Group, mx: int, my: int):
    """Check the eMultivariateNormal layer is G-invariant."""
    import torch

    from symm_learning.nn import eMultivariateNormal
    from symm_learning.representation_theory import direct_sum

    G = group
    rep_x = direct_sum([G.regular_representation] * mx)
    rep_y = direct_sum([G.regular_representation] * my)

    e_normal = eMultivariateNormal(out_rep=rep_y, diagonal=True)

    e_normal.check_equivariance(atol=1e-6, rtol=1e-6)

    gaussian_params = torch.randn(12, e_normal.in_rep.size)
    assert_module_save_load_consistency(
        e_normal,
        gaussian_params,
        output_transform=lambda dist: (dist.mean, dist.covariance_matrix),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [2])
@pytest.mark.parametrize("my", [5])
def test_conv1d(group: Group, mx: int, my: int):  # noqa: D103
    import torch
    from symm_learning.nn.conv import eConv1d, eConvTranspose1d

    G = group
    in_rep = direct_sum([G.regular_representation] * mx)
    out_rep = direct_sum([G.regular_representation] * my)

    layer = eConv1d(in_rep, out_rep, kernel_size=3, padding=1, bias=True)
    layer.eval()

    B, L = 10, 30
    x = torch.randn(B, in_rep.size, L)
    y = layer(x)
    assert y.shape == (B, out_rep.size, L), f"Expected output shape {(B, out_rep.size, L)} got {y.shape}"

    layer.check_equivariance(atol=1e-5, rtol=1e-5)

    # Gradient sanity check
    layer.train()
    layer.zero_grad()
    out = layer(x)
    loss = (out - torch.randn_like(out)).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert grads, "Expected gradients to propagate through eConv1D_"

    # Transposed variant: equivariance and backprop
    t_layer = eConvTranspose1d(out_rep, in_rep, kernel_size=3, padding=1, bias=True)
    t_layer.eval()
    t_layer.check_equivariance(atol=1e-5, rtol=1e-5)

    t_layer.train()
    t_layer.zero_grad()
    x_t = torch.randn(B, out_rep.size, L)
    out_t = t_layer(x_t)
    loss_t = (out_t - torch.randn_like(out_t)).pow(2).mean()
    loss_t.backward()
    grads_t = [p.grad for p in t_layer.parameters() if p.grad is not None]
    assert grads_t, "Expected gradients to propagate through eConvTranspose1d"

    assert_module_save_load_consistency(layer, x)
    assert_module_save_load_consistency(t_layer, x_t)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [3])
@pytest.mark.parametrize("my", [2])
@pytest.mark.parametrize("basis_expansion_scheme", ["memory_heavy", "isotypic_expansion"])
def test_linear(group: Group, mx: int, my: int, basis_expansion_scheme: str):
    import torch
    from symm_learning.nn.linear import eLinear

    G = group
    in_rep = direct_sum([G.regular_representation] * mx)
    out_rep = direct_sum([G.regular_representation] * my)

    layer = eLinear(in_rep, out_rep, bias=True, basis_expansion_scheme=basis_expansion_scheme)
    check_equivariance(layer, atol=1e-5, rtol=1e-5)
    backprop_sanity(layer)

    # Eval cache: mutating DoFs should not change returned weight while cache is valid.
    layer.eval()
    w_cached = layer.weight.clone()
    layer.weight_dof.data.add_(torch.randn_like(layer.weight_dof))
    assert torch.allclose(layer.weight, w_cached), "Eval should reuse cached weight even if DoFs change"

    # Train cache invalidation: training should recompute weight.
    layer.train()
    w_train = layer.weight
    assert not torch.allclose(w_train, w_cached), "Train should recompute weight and differ from cached eval weight"

    # Consistency check: output in eval mode (fast inference) should match output in train
    # mode with same weights.
    # Create a random input
    x_input = torch.randn(10, in_rep.size)
    # Forward in train mode
    y_train = layer(x_input)
    # Switch to eval mode -> should cache the current training weight
    layer.eval()
    y_eval = layer(x_input)
    assert torch.allclose(y_train, y_eval, atol=1e-5, rtol=1e-5), (
        "Output in eval mode (fast inference) must match output in train mode with updated weights"
    )

    # Eval refresh: after training, eval should cache the latest weight.
    # layer.eval() # Already in eval
    w_refreshed = layer.weight
    assert torch.allclose(w_refreshed, w_train), "Eval should cache the latest training weight"

    # Manual expansion: explicit expand_weight should refresh cache and update value.
    layer.weight_dof.data.add_(torch.randn_like(layer.weight_dof))
    layer.expand_weight()
    w_expanded = layer.weight
    assert not torch.allclose(w_expanded, w_refreshed), "Explicit expand_weight should refresh cache and change value"

    # Double backward safety: separate backward passes should work without retaining the graph.
    layer.train()
    for _ in range(2):
        layer.zero_grad(set_to_none=True)
        fx = layer(torch.randn(3, in_rep.size))
        loss = fx.pow(2).mean()
        loss.backward()
        assert layer.weight_dof.grad is not None, "Grad should populate on each backward pass"

    # Dtype move: moving to float64 should invalidate cache and refresh on access.
    layer_double = layer.to(dtype=torch.float64)
    layer_double.eval()
    w_double = layer_double.weight
    assert w_double.dtype == torch.float64, "Weight cache should follow dtype changes"

    # Backward hook: gradients should mark cache dirty and eval should recompute after an optimizer-like step.
    layer_double.train()
    for _ in range(2):
        layer_double.zero_grad(set_to_none=True)
        x = torch.randn(2, in_rep.size, dtype=torch.float64)
        out = layer_double(x)
        out.sum().backward()
        assert layer_double._weight_cache_dirty is True, "Backward hook should mark weight cache dirty"
    with torch.no_grad():
        layer_double.weight_dof.add_(layer_double.weight_dof.grad, alpha=-0.1)
    layer_double.eval()
    w_after_step = layer_double.weight
    assert w_after_step.dtype == torch.float64, "Eval weight should stay in float64 after recompute"
    assert not torch.allclose(w_after_step, w_double), "Weight should change after applying gradient step"

    if torch.cuda.is_available():
        layer_cuda = layer.to("cuda")
        layer_cuda.eval()
        w_cuda = layer_cuda.weight
        assert w_cuda.device.type == "cuda", "Weight cache should move to CUDA device"

        layer_cuda.train()
        for _ in range(2):
            layer_cuda.zero_grad(set_to_none=True)
            x_cuda = torch.randn(2, in_rep.size, device=w_cuda.device, dtype=w_cuda.dtype)
            out = layer_cuda(x_cuda)
            out.sum().backward()
            assert layer_cuda._weight_cache_dirty is True, "Backward on CUDA should mark cache dirty"
        with torch.no_grad():
            layer_cuda.weight_dof.add_(layer_cuda.weight_dof.grad, alpha=-0.1)
        layer_cuda.eval()
        w_cuda_refreshed = layer_cuda.weight
        assert w_cuda_refreshed.device.type == "cuda", "Refreshed weight should remain on CUDA"
        assert not torch.allclose(w_cuda_refreshed.cpu(), w_cuda.cpu()), (
            "CUDA cache refresh should update value after gradient step"
        )

    x_roundtrip = torch.randn(8, in_rep.size, device=layer.weight_dof.device, dtype=layer.weight_dof.dtype)
    assert_module_save_load_consistency(layer, x_roundtrip)

    # Parent-level load_state_dict path: cached dense weight must match loaded DoFs.
    class _Parent(torch.nn.Module):
        def __init__(self, rep_in: Representation, rep_out: Representation, scheme: str):
            super().__init__()
            self.Dr = eLinear(rep_in, rep_out, bias=True, basis_expansion_scheme=scheme)

        def forward(self, x):
            return self.Dr(x)

    source = _Parent(in_rep, out_rep, basis_expansion_scheme).eval()
    target = _Parent(in_rep, out_rep, basis_expansion_scheme).eval()
    _ = source(torch.randn(2, in_rep.size))
    _ = target(torch.randn(2, in_rep.size))
    target.load_state_dict(source.state_dict())

    w_from_dof = target.Dr.homo_basis(target.Dr.weight_dof)
    assert torch.allclose(target.Dr.weight, w_from_dof, atol=1e-6, rtol=1e-6), (
        "eLinear cached dense weight is stale after parent load_state_dict"
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [3])
@pytest.mark.parametrize("my", [2])
@pytest.mark.parametrize("basis_expansion_scheme", ["memory_heavy", "isotypic_expansion"])
@pytest.mark.parametrize("bias", [True, False])
def test_parametrizations(group: Group, mx: int, my: int, basis_expansion_scheme: str, bias: bool):
    import torch

    from symm_learning.nn.linear import impose_linear_equivariance

    G = group
    in_rep = direct_sum([G.regular_representation] * mx)
    out_rep = direct_sum([G.regular_representation] * my)

    layer = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
    impose_linear_equivariance(lin=layer, in_rep=in_rep, out_rep=out_rep, basis_expansion_scheme=basis_expansion_scheme)

    check_equivariance(layer, atol=1e-5, rtol=1e-5)
    backprop_sanity(layer)

    assert_module_save_load_consistency(layer, torch.randn(7, in_rep.size))


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [2])
def test_bias(group: Group, mx: int):
    import torch

    from symm_learning.nn.linear import InvariantBias

    G = group
    in_rep = direct_sum([G.regular_representation] * mx)

    bias_layer = InvariantBias(in_rep)

    check_equivariance(bias_layer, atol=1e-5, rtol=1e-5)
    backprop_sanity(bias_layer)

    # Eval cache: refreshed on eval and reused while staying in eval mode.
    x = torch.randn(4, in_rep.size)
    bias_layer.bias_dof.data.fill_(1.0)
    bias_layer.eval()  # cache is populated lazily on first access in eval
    _ = bias_layer.bias
    cached_bias = bias_layer._bias.clone()
    y_eval = bias_layer(x)
    assert torch.allclose(y_eval, x + cached_bias), "Eval output should include cached bias"
    bias_layer.bias_dof.data.fill_(3.0)
    y_eval_cached = bias_layer(x)
    assert torch.allclose(y_eval_cached, y_eval), "Eval should reuse cached bias despite DoF change"
    assert torch.allclose(bias_layer._bias, cached_bias), "Cached bias tensor should stay unchanged in eval"

    # Train recompute: training should use latest DoFs, then eval should cache that bias.
    bias_layer.train()
    bias_layer.bias_dof.data.fill_(2.0)
    y_train = bias_layer(x)
    assert not torch.allclose(y_train, y_eval), "Train output should reflect updated DoFs"

    bias_layer.eval()  # cache current bias (lazy in eval)
    _ = bias_layer.bias
    updated_cached = bias_layer._bias.clone()
    bias_layer.bias_dof.data.fill_(4.0)
    y_eval_after = bias_layer(x)
    assert torch.allclose(y_eval_after, x + updated_cached), "Eval should reuse freshly cached bias"
    assert torch.allclose(bias_layer._bias, updated_cached), "Cached bias should match latest eval expansion"

    # Double backward safety: two passes should succeed without graph retention errors.
    bias_layer.train()
    for _ in range(2):
        bias_layer.zero_grad(set_to_none=True)
        x_fw = torch.randn(3, in_rep.size)
        out = bias_layer(x_fw)
        out.sum().backward()
        if bias_layer.has_bias:
            assert bias_layer.bias_dof.grad is not None, "Bias DoF grad should populate each backward"

    # Dtype move: moving to float64 should invalidate cache and refresh on access.
    bias_layer_double = bias_layer.to(dtype=torch.float64)
    bias_layer_double.eval()
    bias_double = bias_layer_double.bias
    assert bias_double is not None, "Bias tensor should exist after dtype move"
    assert bias_double.dtype == torch.float64, "Bias cache should follow dtype changes"

    # Backward hook dirties cache; eval recomputes after parameter updates.
    bias_layer_double.train()
    for _ in range(2):
        bias_layer_double.zero_grad(set_to_none=True)
        x_double = torch.randn(2, in_rep.size, dtype=torch.float64)
        out = bias_layer_double(x_double)
        out.sum().backward()
        assert bias_layer_double._bias_cache_dirty is True, "Backward hook should mark bias cache dirty"
    with torch.no_grad():
        bias_layer_double.bias_dof.add_(bias_layer_double.bias_dof.grad, alpha=-0.2)
    bias_layer_double.eval()
    refreshed_bias = bias_layer_double.bias
    assert refreshed_bias.dtype == torch.float64, "Refreshed bias should respect dtype move"
    assert not torch.allclose(refreshed_bias, bias_double), "Bias should update after gradient step"

    if torch.cuda.is_available():
        bias_layer_cuda = bias_layer.to("cuda")
        bias_layer_cuda.eval()
        bias_cuda = bias_layer_cuda.bias
        assert bias_cuda is not None, "Bias tensor should exist on CUDA"
        assert bias_cuda.device.type == "cuda", "Bias cache should move to CUDA"

        bias_layer_cuda.train()
        for _ in range(2):
            bias_layer_cuda.zero_grad(set_to_none=True)
            x_cuda = torch.randn(2, in_rep.size, device=bias_cuda.device, dtype=bias_cuda.dtype)
            out = bias_layer_cuda(x_cuda)
            out.sum().backward()
            assert bias_layer_cuda._bias_cache_dirty is True, "Backward on CUDA should mark bias cache dirty"
        with torch.no_grad():
            bias_layer_cuda.bias_dof.add_(bias_layer_cuda.bias_dof.grad, alpha=-0.2)
        bias_layer_cuda.eval()
        refreshed_bias_cuda = bias_layer_cuda.bias
        assert refreshed_bias_cuda.device.type == "cuda", "Refreshed bias should remain on CUDA"
        assert not torch.allclose(refreshed_bias_cuda.cpu(), bias_cuda.cpu()), (
            "CUDA cache refresh should update bias value after gradient step"
        )

    x_roundtrip = torch.randn(8, in_rep.size, device=bias_layer.bias_dof.device, dtype=bias_layer.bias_dof.dtype)
    assert_module_save_load_consistency(bias_layer, x_roundtrip)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 10])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("learnable", [True, False])
def test_affine(group: Group, mx: int, bias: bool, learnable: bool):
    import numpy as np
    import torch

    from symm_learning.nn.linear import eAffine

    G = group
    rep = direct_sum([G.regular_representation] * mx)
    # Random orthogonal matrix for change of basis, using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, Q, name="test_rep")

    batch_size = 20
    x = torch.randn(batch_size, rep.size)
    save_load_kwargs = {}

    affine = eAffine(in_rep=rep, bias=bias, learnable=learnable, init_scheme="random" if learnable else None)
    if learnable:
        y = affine(x)
        check_equivariance(affine, atol=1e-5, rtol=1e-5)

        # Consistency check for fast inference
        # 1. Update weights with some arbitrary loss
        affine.train()
        target = torch.randn_like(y)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        with torch.no_grad():
            for p in affine.parameters():
                p -= 0.1 * p.grad

        # 2. Forward in train mode with updated weights
        affine.zero_grad()
        y_train = affine(x)

        # 3. Forward in eval mode -> should use cached expanded parameters equivalent to updated weights
        affine.eval()
        y_eval = affine(x)

        assert torch.allclose(y_train, y_eval, atol=1e-5, rtol=1e-5), (
            "eAffine output in eval mode (fast inference) must match output in train mode with updated weights"
        )

    else:
        scale = torch.full((batch_size, affine.num_scale_dof), 2.0)
        bias_dof = torch.full((batch_size, affine.num_bias_dof), 0.25) if bias and affine.num_bias_dof > 0 else None
        save_load_kwargs = {"scale_dof": scale, "bias_dof": bias_dof}
        y = affine(x, scale_dof=scale, bias_dof=bias_dof)

        class _AffineWithExternal(torch.nn.Module):
            def __init__(self, base, scale_dof, bias_dof):
                super().__init__()
                self.base = base
                self.scale = scale_dof
                self.bias = bias_dof
                self.in_rep = base.in_rep
                self.out_rep = base.out_rep

            def forward(self, inp):
                scale_arg = self.scale.to(device=inp.device, dtype=inp.dtype)
                bias_arg = None if self.bias is None else self.bias.to(device=inp.device, dtype=inp.dtype)
                return self.base(inp, scale_dof=scale_arg, bias_dof=bias_arg)

        wrapped = _AffineWithExternal(affine, scale, bias_dof)
        check_equivariance(wrapped, atol=1e-5, rtol=1e-5)

    assert y.shape == x.shape
    assert not torch.allclose(y, x, atol=1e-5, rtol=1e-5)
    assert_module_save_load_consistency(affine, x, forward_kwargs=save_load_kwargs)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [2, 4])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_multihead_attention(group: Group, mx: int, num_heads: int, bias: bool):
    """Check equivariance and fast inference consistency of eMultiheadAttention."""
    import torch

    from symm_learning.nn.activation import eMultiheadAttention
    from symm_learning.utils import check_equivariance

    G = group
    rep = direct_sum([G.regular_representation] * mx)

    # Ensure mx is divisible by num_heads for this test
    if mx % num_heads != 0:
        pytest.skip(f"mx={mx} not divisible by num_heads={num_heads}")

    attn = eMultiheadAttention(in_rep=rep, num_heads=num_heads, bias=bias, batch_first=True, dropout=0.0)

    # Wrapper for check_equivariance: self-attention expects (query, key, value) but we test with q=k=v=x
    class SelfAttentionWrapper(torch.nn.Module):
        def __init__(self, attn_module):
            super().__init__()
            self.attn = attn_module
            self.in_rep = attn_module.in_rep
            self.out_rep = attn_module.out_rep

        def forward(self, x):
            # x shape: (batch, seq, embed)
            out, _ = self.attn(x, x, x, need_weights=False)
            return out

    wrapper = SelfAttentionWrapper(attn)
    wrapper.eval()
    check_equivariance(wrapper, input_dim=3, atol=1e-5, rtol=1e-5)

    # Fast inference consistency test
    B, L = 4, 5
    x = torch.randn(B, L, rep.size)

    # 1. Update weights with some arbitrary loss
    attn.train()
    attn.zero_grad()
    y, _ = attn(x, x, x, need_weights=False)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    with torch.no_grad():
        for p in attn.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad

    # 2. Forward in train mode with updated weights
    attn.zero_grad()
    y_train, _ = attn(x, x, x, need_weights=False)

    # 3. Forward in eval mode -> should use cached expanded parameters equivalent to updated weights
    attn.eval()
    y_eval, _ = attn(x, x, x, need_weights=False)

    assert torch.allclose(y_train, y_eval, atol=1e-5, rtol=1e-5), (
        "eMultiheadAttention output in eval mode (fast inference) must match output in train mode with updated weights"
    )

    assert_module_save_load_consistency(
        attn,
        x,
        x,
        x,
        forward_kwargs={"need_weights": False},
        output_transform=lambda out: out[0],
    )


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("num_heads", [1, 2, 3])
def test_rope_multihead_attention(bias: bool, num_heads: int):
    """Check that RoPE attention is invariant to a global shift of positions."""
    from symm_learning.nn.activation import RoPEMultiheadAttention, RotaryEmbedding
    import torch

    torch.manual_seed(0)
    model = RoPEMultiheadAttention(embed_dim=12, num_heads=num_heads, dropout=0.0, bias=bias, batch_first=True)
    model.eval()

    # Keep the support away from the borders so the signal is not affected by truncation.
    batch_size, seq_len, embed_dim = 1, 30, 12
    x = torch.zeros(batch_size, seq_len, embed_dim)
    x[0, 13:17] = torch.stack(
        (
            torch.arange(1, embed_dim + 1, dtype=x.dtype),
            torch.arange(embed_dim, 0, -1, dtype=x.dtype),
            torch.arange(1, embed_dim + 1, dtype=x.dtype) * 0.5,
            torch.arange(embed_dim, 0, -1, dtype=x.dtype) * 0.5,
        )
    )

    positions = torch.arange(seq_len)
    y, _ = model(x, x, x, q_positions=positions, k_positions=positions)

    shifted_positions = positions + 1
    y_shifted, _ = model(x, x, x, q_positions=shifted_positions, k_positions=shifted_positions)

    torch.testing.assert_close(y_shifted, y, atol=1e-5, rtol=1e-5, msg="RoPE not invariant to a global shift")

    rope = RotaryEmbedding(dim=model.head_dim)
    rope_input = torch.randn(2, num_heads, seq_len, model.head_dim)
    rope_mask = torch.ones(seq_len, dtype=torch.bool)
    rope_mask[7] = False
    rope_output = rope.apply_rope(rope_input, positions=positions, position_mask=rope_mask)

    torch.testing.assert_close(
        rope_output[:, :, 7],
        rope_input[:, :, 7],
        atol=0.0,
        rtol=0.0,
        msg="Masked RoPE positions should remain unchanged",
    )


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

    layer = eLayerNorm(in_rep=rep, bias=bias, equiv_affine=affine, eps=0, init_scheme="random")

    x = torch.randn(64, rep.size)
    y = layer(x)

    assert y.shape == x.shape

    check_equivariance(layer, atol=1e-4, rtol=1e-4)
    assert_module_save_load_consistency(layer, x, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 10])
@pytest.mark.parametrize("affine", [True, False])
def test_rms_norm(group: Group, mx: int, affine: bool):
    import numpy as np
    import torch

    from symm_learning.nn.normalization import eRMSNorm

    G = group
    rep = direct_sum([G.regular_representation] * mx)
    Q, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, Q, name="test_rmsnorm_rep")

    layer = eRMSNorm(in_rep=rep, equiv_affine=affine, eps=0, init_scheme="random")

    x = torch.randn(64, rep.size)
    y = layer(x)

    assert y.shape == x.shape

    check_equivariance(layer, atol=1e-5, rtol=1e-5)
    assert_module_save_load_consistency(layer, x, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", [pytest.param("ema", id="ema"), pytest.param("eema", id="eema")])
def test_ema_stats(kind: str):
    """Minimal smoke test for EMAStats and eEMAStats."""
    from symm_learning.nn import EMAStats, eEMAStats

    import torch

    if kind == "ema":
        stats = EMAStats(dim_x=3, dim_y=2, momentum=0.2)
    else:
        G = CyclicGroup(3)
        rep = G.regular_representation
        stats = eEMAStats(x_rep=rep, y_rep=rep, momentum=0.2)

    raw_x = torch.randn(8, stats.num_features_x)
    raw_y = torch.randn(8, stats.num_features_y)

    # Training path is identity on the activations, but it should still update the tracked EMA state.
    stats.train()
    x_input = raw_x.clone().requires_grad_(True)
    y_input = raw_y.clone().requires_grad_(True)
    prev_mean = stats.mean_x.clone()
    x_out, y_out = stats(x_input, y_input)
    assert torch.equal(x_out, x_input)
    assert torch.equal(y_out, y_input)
    assert stats.num_batches_tracked == 1
    assert not torch.equal(stats.mean_x, prev_mean)
    train_loss = x_out.square().mean() + y_out.square().mean()
    train_loss.backward()
    assert x_input.grad is not None and torch.isfinite(x_input.grad).all()
    assert y_input.grad is not None and torch.isfinite(y_input.grad).all()

    # The exposed statistics are part of the training API: downstream modules can consume
    # `mean_*` / `cov_*` inside the same optimization step. This regression check ensures the
    # train-time statistics still backpropagate to the current batch instead of reading only
    # detached running buffers.
    x_stats = torch.randn_like(raw_x, requires_grad=True)
    y_stats = torch.randn_like(raw_y, requires_grad=True)
    stats(x_stats, y_stats)
    stat_loss = (
        stats.mean_x.square().mean()
        + stats.mean_y.square().mean()
        + stats.cov_xx.square().mean()
        + stats.cov_yy.square().mean()
        + stats.cov_xy.square().mean()
    )
    stat_loss.backward()
    assert x_stats.grad is not None and torch.isfinite(x_stats.grad).all()
    assert y_stats.grad is not None and torch.isfinite(y_stats.grad).all()

    # Evaluation must freeze the tracked state while still letting gradients flow through the
    # identity outputs, since the module itself does not transform the activations.
    stats.eval()
    frozen_mean = stats.mean_x.clone()
    x_eval = raw_x.clone().requires_grad_(True)
    y_eval = raw_y.clone().requires_grad_(True)
    x_eval_out, y_eval_out = stats(x_eval, y_eval)
    assert torch.equal(stats.mean_x, frozen_mean)
    eval_loss = x_eval_out.square().mean() + y_eval_out.square().mean()
    eval_loss.backward()
    assert x_eval.grad is not None and torch.isfinite(x_eval.grad).all()
    assert y_eval.grad is not None and torch.isfinite(y_eval.grad).all()

    if kind == "eema":
        import symm_learning.stats as symm_stats

        # Build a dense reference implementation of the old equivariant EMA update:
        # 1. compute invariant means,
        # 2. center with the current EMA mean (or batch mean on the first step),
        # 3. compute projected dense covariances,
        # 4. apply the EMA update in dense matrix form.
        #
        # The new implementation tracks the same quantities in Hom_G degrees of freedom, so the
        # exposed dense covariances should match this oracle step by step.
        ref_stats = eEMAStats(x_rep=rep, y_rep=rep, momentum=0.2)
        ref_stats.train()
        manual_mean_x = None
        manual_mean_y = None
        manual_cov_xx = None
        manual_cov_yy = None
        manual_cov_xy = None
        batch_generator = torch.Generator().manual_seed(1234)
        for _ in range(3):
            x_batch = torch.randn(8, rep.size, generator=batch_generator)
            y_batch = torch.randn(8, rep.size, generator=batch_generator)

            # Invariant means used by both the old dense path and the current DoF path.
            batch_mean_x = symm_stats.mean(x_batch, rep_x=rep)
            batch_mean_y = symm_stats.mean(y_batch, rep_x=rep)
            if manual_mean_x is None:
                center_x = batch_mean_x
                center_y = batch_mean_y
            else:
                # After the first batch, EMA covariances are centered with the previously tracked mean.
                center_x = manual_mean_x
                center_y = manual_mean_y
            x_centered = x_batch - center_x.unsqueeze(0)
            y_centered = y_batch - center_y.unsqueeze(0)
            # `uncentered=True` treats the already-centered samples as second-moment inputs, which
            # reproduces the old dense covariance computation exactly, including the 1/N scaling.
            batch_cov_xx = symm_stats.cov(x_centered, x_centered, rep_x=rep, rep_y=rep, uncentered=True)
            batch_cov_yy = symm_stats.cov(y_centered, y_centered, rep_x=rep, rep_y=rep, uncentered=True)
            batch_cov_xy = symm_stats.cov(x_centered, y_centered, rep_x=rep, rep_y=rep, uncentered=True).T

            if manual_mean_x is None:
                manual_mean_x = batch_mean_x
                manual_mean_y = batch_mean_y
                manual_cov_xx = batch_cov_xx
                manual_cov_yy = batch_cov_yy
                manual_cov_xy = batch_cov_xy
            else:
                alpha = ref_stats.momentum
                manual_mean_x = manual_mean_x * (1 - alpha) + batch_mean_x * alpha
                manual_mean_y = manual_mean_y * (1 - alpha) + batch_mean_y * alpha
                manual_cov_xx = manual_cov_xx * (1 - alpha) + batch_cov_xx * alpha
                manual_cov_yy = manual_cov_yy * (1 - alpha) + batch_cov_yy * alpha
                manual_cov_xy = manual_cov_xy * (1 - alpha) + batch_cov_xy * alpha

            ref_stats(x_batch, y_batch)
            # The tracked dense quantities exposed by the DoF implementation must agree with the
            # dense oracle after every update, not just at the end of the sequence.
            assert torch.allclose(ref_stats.mean_x, manual_mean_x, atol=1e-6, rtol=1e-6)
            assert torch.allclose(ref_stats.mean_y, manual_mean_y, atol=1e-6, rtol=1e-6)
            assert torch.allclose(ref_stats.cov_xx, manual_cov_xx, atol=1e-6, rtol=1e-6)
            assert torch.allclose(ref_stats.cov_yy, manual_cov_yy, atol=1e-6, rtol=1e-6)
            assert torch.allclose(ref_stats.cov_xy, manual_cov_xy, atol=1e-6, rtol=1e-6)

        # In eval mode, the dense expansion is intentionally cached, so mutating the DoF buffer
        # alone should not change the exposed dense covariance until the cache is invalidated.
        cov_xx_cached = stats.cov_xx.clone()
        stats.running_cov_xx_dof.add_(torch.randn_like(stats.running_cov_xx_dof))
        assert torch.allclose(stats.cov_xx, cov_xx_cached), "Eval should reuse cached dense covariance expansion"

        # Returning to training invalidates the eval cache and recomputes the dense covariance
        # from the current DoF state.
        stats.train()
        cov_xx_train = stats.cov_xx
        assert not torch.allclose(cov_xx_train, cov_xx_cached), "Train should recompute covariance from DoFs"

        # Switching back to eval should freeze the latest training-time dense covariance.
        stats.eval()
        cov_xx_eval = stats.cov_xx
        assert torch.allclose(cov_xx_eval, cov_xx_train), "Eval should cache the latest training covariance"

        # Only the detached DoF buffers are serialized. The dense cache is derived state and should
        # be rebuilt on demand after load.
        state = stats.state_dict()
        assert "running_cov_xx_dof" in state and "running_cov_yy_dof" in state and "running_cov_xy_dof" in state
        assert "running_cov_xx" not in state and "running_cov_yy" not in state and "running_cov_xy" not in state
        assert "_cov_xx" not in state and "_cov_yy" not in state and "_cov_xy" not in state

    assert_module_save_load_consistency(stats, raw_x, raw_y)
