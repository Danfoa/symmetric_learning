from __future__ import annotations

# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from copy import deepcopy

import escnn
import pytest
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral, Representation, directsum
from escnn.nn import FieldType

from symm_learning.nn import EMAStats, eEMAStats
from symm_learning.representation_theory import direct_sum
from symm_learning.utils import backprop_sanity, check_equivariance


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
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
    out_t = t_layer(torch.randn(B, out_rep.size, L))
    loss_t = (out_t - torch.randn_like(out_t)).pow(2).mean()
    loss_t.backward()
    grads_t = [p.grad for p in t_layer.parameters() if p.grad is not None]
    assert grads_t, "Expected gradients to propagate through eConvTranspose1d"


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

    # Eval: mutate params but cached weight should be returned.
    layer.eval()
    w_cached = layer.weight.clone()
    layer.weight_dof.data.add_(torch.randn_like(layer.weight_dof))
    assert torch.allclose(layer.weight, w_cached), "Eval should reuse cached weight"

    # Train: recompute weight, so cache and output should change.
    layer.train()
    w_train = layer.weight
    assert not torch.allclose(w_train, w_cached), "Train should recompute weight"

    # Back to eval: new cache should match training weight.
    layer.eval()
    w_refreshed = layer.weight
    assert torch.allclose(w_refreshed, w_train), "Eval after train should cache refreshed weight"

    # Explicit call to expand_weight should refresh cache.
    layer.weight_dof.data.add_(torch.randn_like(layer.weight_dof))
    layer.expand_weight()
    w_expanded = layer.weight
    assert not torch.allclose(w_expanded, w_refreshed), "Explicit expand_weight should refresh cache"


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

    # Cache is refreshed on eval and reused while staying in eval mode.
    x = torch.randn(4, in_rep.size)
    bias_layer.bias_dof.data.fill_(1.0)
    bias_layer.eval()  # recompute cache after modifying bias_dof
    cached_bias = bias_layer._bias.clone()
    y_eval = bias_layer(x)
    assert torch.allclose(y_eval, x + cached_bias)
    bias_layer.bias_dof.data.fill_(3.0)
    y_eval_cached = bias_layer(x)
    assert torch.allclose(y_eval_cached, y_eval)  # should reuse cached bias
    assert torch.allclose(bias_layer._bias, cached_bias)

    # Switching to train recomputes with the latest bias_dof, then eval caches again.
    bias_layer.train()
    bias_layer.bias_dof.data.fill_(2.0)
    y_train = bias_layer(x)
    assert not torch.allclose(y_train, y_eval)

    bias_layer.eval()  # cache current bias
    updated_cached = bias_layer._bias.clone()
    bias_layer.bias_dof.data.fill_(4.0)
    y_eval_after = bias_layer(x)
    assert torch.allclose(y_eval_after, x + updated_cached)
    assert torch.allclose(bias_layer._bias, updated_cached)


# @pytest.mark.parametrize(
#     "group",
#     [
#         pytest.param(CyclicGroup(5), id="cyclic5"),
#         pytest.param(Icosahedral(), id="icosahedral"),
#     ],
# )
# @pytest.mark.parametrize("mx", [1])
# @pytest.mark.parametrize("affine", [True, False])
# @pytest.mark.parametrize("running_stats", [True, False])
# def test_batchnorm1d(group: Group, mx: int, affine: bool, running_stats: bool):
#     """Check the eBatchNorm1d layer is G-invariant."""
#     import torch

#     from symm_learning.nn import GSpace1D, eBatchNorm1d

#     G = group
#     gspace = GSpace1D(G)

#     in_type = FieldType(gspace, [G.regular_representation] * mx)

#     time = 2
#     batch_size = 5
#     x = torch.randn(batch_size, in_type.size, time)
#     x = in_type(x)

#     batchnorm_layer = eBatchNorm1d(in_type, affine=affine, track_running_stats=running_stats)

#     if hasattr(batchnorm_layer, "affine_transform"):
#         # Randomize the scale and bias DoFs
#         batchnorm_layer.affine_transform.scale_dof.data.uniform_(-1, 1)
#         if batchnorm_layer.affine_transform.has_bias:
#             batchnorm_layer.affine_transform.bias_dof.data.uniform_(-1, 1)

#     batchnorm_layer.check_equivariance(atol=1e-5, rtol=1e-5)

#     batchnorm_layer.eval()

#     # TODO: This is not passing.
#     # y = batchnorm_layer(x).tensor
#     # y_torch = batchnorm_layer.export()(x.tensor)

#     # print(y.shape, y_torch.shape)
#     # assert torch.allclose(y, y_torch, atol=1e-5, rtol=1e-5), f"{y - y_torch} should be 0"


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

    affine = eAffine(in_rep=rep, bias=bias, learnable=learnable, init_scheme="random" if learnable else None)
    if learnable:
        y = affine(x)
        check_equivariance(affine, atol=1e-5, rtol=1e-5)
    else:
        scale = torch.full((batch_size, affine.num_scale_dof), 2.0)
        bias_dof = torch.full((batch_size, affine.num_bias_dof), 0.25) if bias and affine.num_bias_dof > 0 else None
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


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 10])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("affine", [True, False])
def test_rms_norm(group: Group, mx: int, bias: bool, affine: bool):
    import numpy as np
    import torch

    from symm_learning.nn.normalization import eRMSNorm

    G = group
    rep = direct_sum([G.regular_representation] * mx)
    Q, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, Q, name="test_rmsnorm_rep")

    layer = eRMSNorm(in_rep=rep, bias=bias, equiv_affine=affine, eps=0, init_scheme="random")

    x = torch.randn(64, rep.size)
    y = layer(x)

    assert y.shape == x.shape

    check_equivariance(layer, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 2])
def test_etransformer_encoder(group: Group, mx: int):
    """Check equivariance of single-layer and 5-layer encoder stacks."""
    import torch

    from symm_learning.models.transformer.etransformer import eTransformerEncoderLayer

    G = group
    rep = direct_sum([G.regular_representation] * mx)

    encoder_kwargs = dict(
        in_rep=rep,
        nhead=1,
        dim_feedforward=rep.size * 4,
        dropout=0.1,
        activation="relu",
        norm_first=True,
        batch_first=True,
    )

    encoder = eTransformerEncoderLayer(**encoder_kwargs)
    encoder.eval()
    check_equivariance(
        encoder,
        input_dim=3,
        module_name="encoder layer",
        in_rep=rep,
        out_rep=rep,
        atol=1e-4,
        rtol=1e-4,
    )

    base_layer = eTransformerEncoderLayer(**encoder_kwargs)
    base_layer.reset_parameters()
    base_layer.eval()
    encoder_stack = torch.nn.TransformerEncoder(encoder_layer=base_layer, num_layers=5, enable_nested_tensor=False)
    for layer in encoder_stack.layers:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    encoder_stack.eval()
    check_equivariance(
        encoder_stack,
        input_dim=3,
        module_name="encoder stack depth=5",
        in_rep=rep,
        out_rep=rep,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1, 2])
def test_etransformer_decoder(group: Group, mx: int):
    """Check equivariance of single-layer and 5-layer decoder stacks."""
    import torch

    from symm_learning.models.transformer.etransformer import eTransformerDecoderLayer

    G = group
    rep = direct_sum([G.regular_representation] * mx)

    decoder_kwargs = dict(
        in_rep=rep,
        nhead=1,
        dim_feedforward=rep.size * 4,
        dropout=0.1,
        activation="relu",
        norm_first=True,
        batch_first=True,
    )

    decoder = eTransformerDecoderLayer(**decoder_kwargs)
    decoder.eval()
    decoder.check_equivariance(batch_size=3, tgt_len=2, mem_len=3, samples=5, atol=1e-3, rtol=1e-3)

    def check_decoder_stack(
        module: torch.nn.Module, rep: Representation, samples: int = 5, atol: float = 1e-3, rtol: float = 1e-3
    ):
        G_local = rep.group

        def act(rep_local: Representation, g, tensor: torch.Tensor) -> torch.Tensor:
            mat = torch.tensor(rep_local(g), dtype=tensor.dtype, device=tensor.device)
            return torch.einsum("ij,...j->...i", mat, tensor)

        B, tgt_len, mem_len = 3, 2, 3
        module.eval()
        for _ in range(samples):
            g = G_local.sample()
            tgt = torch.randn(B, tgt_len, rep.size)
            mem = torch.randn(B, mem_len, rep.size)
            out = module(tgt=tgt, memory=mem)
            g_tgt = act(rep, g, tgt)
            g_mem = act(rep, g, mem)
            g_out = module(tgt=g_tgt, memory=g_mem)
            g_out_exp = act(rep, g, out)
            assert torch.allclose(g_out, g_out_exp, atol=atol, rtol=rtol), (
                f"Decoder stack equivariance failed, max err {(g_out - g_out_exp).abs().max().item():.3e}"
            )

    base_layer = eTransformerDecoderLayer(**decoder_kwargs)
    base_layer.reset_parameters()
    base_layer.eval()
    decoder_stack = torch.nn.TransformerDecoder(decoder_layer=base_layer, num_layers=5)
    for layer in decoder_stack.layers:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    decoder_stack.eval()
    check_decoder_stack(decoder_stack, rep, samples=5, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("kind", [pytest.param("ema", id="ema"), pytest.param("eema", id="eema")])
def test_ema_stats_core(kind: str):
    """Minimal smoke test for EMAStats and eEMAStats."""
    import torch
    from escnn.gspaces import no_base_space

    if kind == "ema":
        stats = EMAStats(dim_x=3, dim_y=2, momentum=0.2)
        raw_x = torch.randn(8, stats.num_features_x)
        raw_y = torch.randn(8, stats.num_features_y)

        def prepare(x_tensor: torch.Tensor, y_tensor: torch.Tensor):
            return x_tensor, y_tensor

        def extract(output):
            return output

    else:
        G = CyclicGroup(3)
        gspace = no_base_space(G)
        field = FieldType(gspace, [G.regular_representation])
        stats = eEMAStats(x_type=field, y_type=field, momentum=0.2)
        raw_x = torch.randn(8, field.size)
        raw_y = torch.randn(8, field.size)

        def prepare(x_tensor: torch.Tensor, y_tensor: torch.Tensor):
            return field(x_tensor), field(y_tensor)

        def extract(output):
            return output.tensor

    # Train: outputs unchanged, stats update once
    stats.train()
    x_input, y_input = prepare(raw_x, raw_y)
    prev_mean = stats.mean_x.clone()
    x_out, y_out = stats(x_input, y_input)
    assert torch.equal(extract(x_out), raw_x)
    assert torch.equal(extract(y_out), raw_y)
    assert stats.num_batches_tracked == 1
    assert not torch.equal(stats.mean_x, prev_mean)

    # Eval: stats freeze
    stats.eval()
    frozen_mean = stats.mean_x.clone()
    stats(*prepare(raw_x, raw_y))
    assert torch.equal(stats.mean_x, frozen_mean)

    # Export round-trip for equivariant version
    if kind == "eema":
        exported = stats.export()
        exported.eval()
        x_std, y_std = exported(raw_x, raw_y)
        assert torch.equal(x_std, raw_x)
        assert torch.equal(y_std, raw_y)
