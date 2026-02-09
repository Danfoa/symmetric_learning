"""Model integration tests."""

from __future__ import annotations

import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral

from symm_learning.representation_theory import direct_sum
from symm_learning.utils import backprop_sanity, check_equivariance


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("hidden_units", [[64, 64]])
@pytest.mark.parametrize("activation", [torch.nn.ReLU()])
@pytest.mark.parametrize("bias", [True])
def test_emlp(group: Group, hidden_units: int, activation: str, bias: bool):  # noqa: D103
    from symm_learning.models import eMLP

    x_rep = group.regular_representation  # ρ_Χ
    y_rep = direct_sum([group.regular_representation] * 2)  # ρ_Y = ρ_Χ ⊕ ρ_Χ

    emlp = eMLP(in_rep=x_rep, out_rep=y_rep, hidden_units=hidden_units, activation=activation, bias=bias)

    check_equivariance(emlp, atol=1e-4, rtol=1e-4)
    backprop_sanity(emlp)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("hidden_units", [[32, 32]])
def test_imlp(group: Group, hidden_units: int):  # noqa: D103
    from symm_learning.models import iMLP

    x_rep = group.regular_representation  # ρ_Χ

    imlp = iMLP(in_rep=x_rep, out_dim=x_rep.group.order() * 2, hidden_units=hidden_units)

    check_equivariance(imlp, atol=1e-4, rtol=1e-4)
    backprop_sanity(imlp)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [2])
@pytest.mark.parametrize("my", [3])
def test_cond_res_block(group: Group, mx: int, my: int):  # noqa: D103
    import torch
    from symm_learning.models.diffusion.cond_eunet1d import eConditionalResidualBlock1D, eConditionalUnet1D

    G = group
    in_rep = direct_sum([G.regular_representation] * mx)
    out_rep = direct_sum([G.regular_representation] * my)
    cond_rep = direct_sum([G.regular_representation] * 2 * my)
    layer = eConditionalResidualBlock1D(in_rep=in_rep, out_rep=out_rep, cond_rep=cond_rep)
    layer.eval()

    layer.check_equivariance(atol=1e-5, rtol=1e-5)

    # Test U-Net variants (stride and pooling downsampling), with/without local conditioning
    # local_rep = direct_sum([G.regular_representation] * mx)
    for downsample, length in (("stride", 5), ("pooling", 4)):
        unet = eConditionalUnet1D(
            in_rep=in_rep,
            local_cond_rep=None,
            global_cond_rep=cond_rep,
            diffusion_step_embed_dim=8,
            down_dims=[in_rep.size, in_rep.size],
            kernel_size=3,
            cond_predict_scale=True,
            activation=torch.nn.ReLU(),
            normalize=True,
            downsample=downsample,
            init_scheme="xavier_uniform",
        )
        unet.eval()
        unet.check_equivariance(batch_size=2, length=length, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("mx", [1])
@pytest.mark.parametrize("hidden_channels", [[64]])
@pytest.mark.parametrize("mlp_hidden", [[32, 32]])
def test_time_ecnn(group: Group, mx: int, hidden_channels: list[int], mlp_hidden: list[int]):  # noqa: D103
    from symm_learning.models.time_cnn.ecnn_encoder import eTimeCNNEncoder

    G = group
    in_rep = direct_sum([G.regular_representation] * mx)
    out_rep = direct_sum([G.regular_representation] * mx)

    model = eTimeCNNEncoder(
        in_rep=in_rep,
        out_rep=out_rep,
        hidden_channels=hidden_channels,
        time_horizon=16,
        activation=torch.nn.ReLU(),
        batch_norm=True,
        bias=True,
        mlp_hidden=mlp_hidden,
        downsample="stride",
        append_last_frame=True,
        init_scheme="xavier_normal",
    )
    model.eval()

    # Equivariance check: act on channel dimension
    model.check_equivariance(atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("m", [2])
@pytest.mark.parametrize("horizons", [(10, 5)])
@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("num_cond_layers", [2])
@pytest.mark.parametrize("num_attention_heads", [0, 4])
def test_econd_transformer_regressor(
    group: Group,
    m: int,
    horizons: tuple[int, int],
    num_layers: int,
    num_cond_layers: int,
    num_attention_heads: int,
):
    """Port equivariance checks from eCondTransformerRegressor __main__ into pytest."""
    from symm_learning.models.diffusion.econd_transformer_regressor import eCondTransformerRegressor

    G = group
    in_rep = direct_sum([G.regular_representation] * m)
    cond_rep = in_rep
    out_rep = in_rep

    in_horizon, cond_horizon = horizons
    embedding_dim = G.order() * m * 4
    regular_copies = embedding_dim // G.order()

    kwargs = dict(
        in_rep=in_rep,
        cond_rep=cond_rep,
        out_rep=out_rep,
        in_horizon=in_horizon,
        cond_horizon=cond_horizon,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        embedding_dim=embedding_dim,
        num_cond_layers=num_cond_layers,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=False,
    )

    if num_attention_heads < 1 or regular_copies % num_attention_heads != 0:
        with pytest.raises(ValueError):
            eCondTransformerRegressor(**kwargs)
        return

    model = eCondTransformerRegressor(**kwargs)
    model.eval()
    model.check_equivariance(
        batch_size=2,
        in_len=min(3, in_horizon),
        cond_len=min(2, cond_horizon),
        atol=1e-3,
        rtol=1e-3,
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
    from symm_learning.models.transformer.etransformer import eTransformerEncoderLayer

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
@pytest.mark.parametrize("mx", [2])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("num_layers", [1, 5])
def test_etransformer_decoder(group: Group, mx: int, num_heads: int, num_layers: int):
    """Check equivariance and fast inference consistency of eTransformerDecoderLayer."""
    from symm_learning.models.transformer.etransformer import eTransformerDecoderLayer

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
        pytest.param(CyclicGroup(2), id="cyclic2"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("m", [2])
@pytest.mark.parametrize("num_attention_heads", [1, 2])
@pytest.mark.parametrize("cond_layers", [0, 1])
def test_econd_transformer_regressor(group: Group, m: int, num_attention_heads: int, cond_layers: int):
    """Check fast inference consistency of eCondTransformerRegressor."""
    from symm_learning.models.diffusion.econd_transformer_regressor import eCondTransformerRegressor

    G = group
    in_rep = direct_sum([G.regular_representation] * m)
    cond_rep = in_rep
    out_rep = in_rep

    in_horizon, cond_horizon = 5, 4
    embedding_dim = G.order() * m * 4
    regular_copies = embedding_dim // G.order()

    # Skip if num_attention_heads doesn't divide regular_copies
    if regular_copies % num_attention_heads != 0:
        pytest.skip(f"regular_copies={regular_copies} not divisible by num_attention_heads={num_attention_heads}")

    model = eCondTransformerRegressor(
        in_rep=in_rep,
        cond_rep=cond_rep,
        out_rep=out_rep,
        in_horizon=in_horizon,
        cond_horizon=cond_horizon,
        num_layers=3,
        num_attention_heads=num_attention_heads,
        embedding_dim=embedding_dim,
        num_cond_layers=cond_layers,
        p_drop_emb=0.0,  # dropout=0 for train/eval consistency
        p_drop_attn=0.0,
        causal_attn=False,
        norm_module="rmsnorm",
    )

    # Equivariance check
    model.eval()
    model.check_equivariance(batch_size=50, in_len=10, cond_len=5, atol=1e-4, rtol=1e-4)

    # Fast inference consistency test
    B = 4
    X = torch.randn(B, in_horizon, in_rep.size)
    Z = torch.randn(B, cond_horizon, cond_rep.size)
    k = torch.randn(B)

    # 1. Update weights with some arbitrary loss
    model.train()
    model.zero_grad()
    y = model(X=X, opt_step=k, Z=Z)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad

    # 2. Forward in train mode with updated weights
    model.zero_grad()
    y_train = model(X=X, opt_step=k, Z=Z).detach()

    # 3. Forward in eval mode
    model.eval()
    y_eval = model(X=X, opt_step=k, Z=Z).detach()

    # Use looser tolerance for complex composite model (precision accumulates through layers)
    assert torch.allclose(y_train, y_eval, atol=1e-3, rtol=1e-3), (
        f"y_train != y_eval.Max diff: {(y_train - y_eval).abs().max().item():.6f}",
        f"eCondTransformerRegressor output in eval mode must match output in train mode with updated weights. ",
    )
