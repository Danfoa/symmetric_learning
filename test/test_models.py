"""Model integration tests."""

from __future__ import annotations

import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral

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
    assert_module_save_load_consistency(emlp, torch.randn(8, x_rep.size), atol=1e-5, rtol=1e-5)


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
    assert_module_save_load_consistency(imlp, torch.randn(8, x_rep.size), atol=1e-5, rtol=1e-5)


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
        pytest.param(CyclicGroup(2), id="cyclic2"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("m", [2])
@pytest.mark.parametrize("num_attention_heads", [1, 2])
@pytest.mark.parametrize("cond_layers", [0, 1])
@pytest.mark.parametrize("pos_encoding", ["additive_absolute", "additive_relative", "none"])
def test_econd_transformer_regressor(
    group: Group, m: int, num_attention_heads: int, cond_layers: int, pos_encoding: str
):
    """Check fast inference consistency of eCondTransformerRegressor."""
    from symm_learning.models.control.econd_transformer import eCondTransformer

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

    model = eCondTransformer(
        in_rep=in_rep,
        cond_rep=cond_rep,
        out_rep=out_rep,
        in_horizon=in_horizon,
        cond_horizon=cond_horizon,
        num_layers=3,
        num_attention_heads=num_attention_heads,
        embedding_dim=embedding_dim,
        num_cond_layers=cond_layers,
        pos_encoding=pos_encoding,
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


@pytest.mark.parametrize("pos_encoding", ["additive_absolute", "additive_relative", "rope"])
@pytest.mark.parametrize("num_cond_layers", [0, 1])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("norm_module", ["layernorm", "rmsnorm"])
def test_cond_transformer_regressor(pos_encoding: str, num_cond_layers: int, norm_first: bool, norm_module: str):
    """Check forward and backprop pass for the baseline CondTransformerRegressor."""
    from symm_learning.models.control.cond_transformer import CondTransformer

    in_dim, out_dim, cond_dim = 4, 4, 3
    in_horizon, cond_horizon = 5, 4
    embedding_dim = 16
    num_attention_heads = 2
    batch_size = 2

    model = CondTransformer(
        in_dim=in_dim,
        out_dim=out_dim,
        cond_dim=cond_dim,
        in_horizon=in_horizon,
        cond_horizon=cond_horizon,
        pos_encoding=pos_encoding,
        num_layers=2,
        num_attention_heads=num_attention_heads,
        embedding_dim=embedding_dim,
        num_cond_layers=num_cond_layers,
        norm_first=norm_first,
        norm_module=norm_module,
    )

    model.train()
    optimizer = model.configure_optimizers()

    X = torch.randn(batch_size, in_horizon, in_dim)
    Z = torch.randn(batch_size, cond_horizon, cond_dim)
    opt_step = torch.randn(batch_size)

    # Forward pass
    optimizer.zero_grad()
    out = model(X=X, Z=Z, opt_step=opt_step)

    assert out.shape == (batch_size, in_horizon, out_dim), (
        f"Expected shape {(batch_size, in_horizon, out_dim)} got {out.shape}"
    )

    # Backward pass
    loss = out.mean()
    loss.backward()

    # Check for valid gradients
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            assert not torch.isnan(p.grad).any(), "NaN in gradients"

    assert has_grad, "No gradients were computed"

    # Optimizer step
    optimizer.step()


@pytest.mark.parametrize("pos_encoding", ["additive_absolute", "additive_relative", "none"])
@pytest.mark.parametrize("num_cond_layers", [0, 1])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("norm_module", ["layernorm", "rmsnorm"])
def test_econd_transformer(pos_encoding: str, num_cond_layers: int, norm_first: bool, norm_module: str):
    """Check forward, backward, and equivariance for the control-side eCondTransformer."""
    from symm_learning.models.control.econd_transformer import eCondTransformer

    G = CyclicGroup(2)
    m = 2
    in_rep = direct_sum([G.regular_representation] * m)
    cond_rep = in_rep
    out_rep = in_rep

    in_horizon, cond_horizon = 5, 4
    embedding_dim = G.order() * m * 4
    batch_size = 2

    model = eCondTransformer(
        in_rep=in_rep,
        cond_rep=cond_rep,
        out_rep=out_rep,
        in_horizon=in_horizon,
        cond_horizon=cond_horizon,
        num_layers=2,
        num_attention_heads=2,
        embedding_dim=embedding_dim,
        num_cond_layers=num_cond_layers,
        pos_encoding=pos_encoding,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        norm_first=norm_first,
        norm_module=norm_module,
    )

    model.eval()
    model.check_equivariance(batch_size=2, in_len=3, cond_len=2, atol=1e-4, rtol=1e-4)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.randn(batch_size, in_horizon, in_rep.size)
    Z = torch.randn(batch_size, cond_horizon, cond_rep.size)
    opt_step = torch.randn(batch_size)

    optimizer.zero_grad()
    out = model(X=X, Z=Z, opt_step=opt_step)
    assert out.shape == (batch_size, in_horizon, out_rep.size), (
        f"Expected shape {(batch_size, in_horizon, out_rep.size)} got {out.shape}"
    )

    loss = out.mean()
    loss.backward()
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            assert not torch.isnan(p.grad).any(), "NaN in gradients"
    assert has_grad, "No gradients were computed"

    optimizer.step()
