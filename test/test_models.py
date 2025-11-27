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
    from symm_learning.models.difussion.cond_eunet1d import eConditionalResidualBlock1D, eConditionalUnet1D

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
    from symm_learning.models.difussion.econd_transformer_regressor import eCondTransformerRegressor

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
