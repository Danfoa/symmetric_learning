# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
import escnn
import pytest
import torch
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
@pytest.mark.parametrize("n_hidden_layers", [3])
@pytest.mark.parametrize("hidden_units", [128, [64, 128, 64]])
@pytest.mark.parametrize("activation", ["ReLU"])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("pointwise_activation", [True, False])
def test_emlp(  # noqa: D103
    group: Group, n_hidden_layers: int, hidden_units: int, activation: str, bias: bool, pointwise_activation: bool
):  # noqa: D103
    x_rep = group.regular_representation  # ρ_Χ
    y_rep = directsum([group.regular_representation] * 2)  # ρ_Y = ρ_Χ ⊕ ρ_Χ

    x_rep.name = "x_rep"
    y_rep.name = "y_rep"

    type_X = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[x_rep])
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[y_rep])

    from symm_learning.models import EMLP

    if isinstance(group, Icosahedral) and not pointwise_activation:
        return

    emlp = EMLP(
        in_type=type_X,
        out_type=type_Y,
        hidden_layers=n_hidden_layers,
        hidden_units=hidden_units,
        activation=activation,
        bias=bias,
        pointwise_activation=pointwise_activation,
    )

    emlp.check_equivariance(atol=1e-5, rtol=1e-5)

    if pointwise_activation:
        temlp = emlp.export()

        batch_dim = 50
        x = type_X(torch.rand(batch_dim, type_X.size))
        y = emlp(x).tensor
        y_t = temlp(x.tensor)

        assert torch.allclose(y, y_t, atol=1e-5, rtol=1e-5)

    irreps = list(set(group.regular_representation.irreps))
    irreps.pop()
    hidden_rep = group.spectral_regular_representation(*irreps, name="Test hidden rep")

    try:
        emlp = EMLP(
            in_type=type_X,
            out_type=type_Y,
            hidden_layers=n_hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
            bias=bias,
            pointwise_activation=pointwise_activation,
            hidden_rep=hidden_rep,
        )
    except ValueError as e:
        print(e)
        # raise e
    except Exception as e:
        raise e


@pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10)])
@pytest.mark.parametrize("n_inv_features", [1, 10])
@pytest.mark.parametrize("n_hidden_layers", [5])
@pytest.mark.parametrize("hidden_units", [128])
@pytest.mark.parametrize("activation", ["ReLU"])
@pytest.mark.parametrize("bias", [True])
def test_imlp_invariance(  # noqa: D103
    group: Group, n_inv_features: int, n_hidden_layers: int, hidden_units: int, activation: str, bias: bool
):
    x_rep = group.regular_representation  # ρ_Χ
    x_rep.name = "x_rep"
    type_X = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[x_rep] * 5)

    from symm_learning.models import IMLP

    imlp = IMLP(
        in_type=type_X,
        out_dim=n_inv_features,
        hidden_layers=n_hidden_layers,
        hidden_units=hidden_units,
        activation=activation,
        bias=bias,
    )

    imlp.check_equivariance(atol=1e-5, rtol=1e-5)

    timlp = imlp.export()

    batch_dim = 50
    x = type_X(torch.rand(batch_dim, type_X.size))
    y = imlp(x).tensor
    y_t = timlp(x.tensor)

    assert torch.allclose(y, y_t, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10)])
def test_residual_encoder_equivariance(group: Group):  # noqa: D103
    x_rep = group.regular_representation  # ρ_Χ
    y_rep = directsum([group.regular_representation] * 2)  # ρ_Y = ρ_Χ ⊕ ρ_Χ

    x_rep.name = "x_rep"
    y_rep.name = "y_rep"

    type_X = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[x_rep])
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[y_rep])

    from symm_learning.models.emlp import EMLP

    emlp = EMLP(in_type=type_X, out_type=type_Y, hidden_layers=3, hidden_units=128)

    from symm_learning.models.residual_encoder import ResidualLayer

    res_encoder = ResidualLayer(nn=emlp)
    res_encoder.check_equivariance(atol=1e-5, rtol=1e-5)
    x_test = torch.randn(40, x_rep.size)
    x = type_X(x_test)  # Wrap in GeometricTensor
    y = res_encoder(x)  # Forward pass
    x_rec = y.tensor[..., res_encoder.residual_dims]  # Decode

    assert torch.allclose(x.tensor, x_rec, atol=1e-5, rtol=1e-5), (
        f"Error in obtaining the input tensor from the encoded tensor. {x.tensor} != {x_rec}"
    )
