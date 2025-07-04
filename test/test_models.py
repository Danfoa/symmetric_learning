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
@pytest.mark.parametrize("hidden_units", [[128], [64, 128, 64]])
@pytest.mark.parametrize("activation", ["ReLU"])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("pointwise_activation", [True, False])
def test_emlp(  # noqa: D103
    group: Group, hidden_units: int, activation: str, bias: bool, pointwise_activation: bool
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

    # --- Additional forward and backward pass ---
    batch_dim = 50
    # Ensure input tensor requires gradients
    x_grad = type_X(torch.rand(batch_dim, type_X.size, requires_grad=True))
    y_grad = emlp(x_grad)
    # Use a dummy loss (sum of squares of the output tensor)
    dummy_loss = (y_grad.tensor**2).sum()
    dummy_loss.backward()


@pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10)])
@pytest.mark.parametrize("n_inv_features", [10])
@pytest.mark.parametrize("hidden_units", [[128], [128, 64]])
@pytest.mark.parametrize("activation", ["ReLU"])
@pytest.mark.parametrize("bias", [True])
def test_imlp(  # noqa: D103
    group: Group, n_inv_features: int, hidden_units: int, activation: str, bias: bool
):
    x_rep = group.regular_representation  # ρ_Χ
    x_rep.name = "x_rep"
    type_X = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[x_rep] * 5)

    from symm_learning.models import IMLP

    imlp = IMLP(
        in_type=type_X,
        out_dim=n_inv_features,
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

    # --- Additional forward and backward pass ---
    # Make sure the input tensor requires gradients for the backward pass.
    x_grad = type_X(torch.rand(batch_dim, type_X.size, requires_grad=True))
    y_grad = imlp(x_grad)
    # Create a dummy loss and backpropagate.
    dummy_loss = (y_grad.tensor**2).sum()
    dummy_loss.backward()


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(CyclicGroup(5), id="cyclic5"),
        pytest.param(DihedralGroup(10), id="dihedral10"),
        pytest.param(Icosahedral(), id="icosahedral"),
    ],
)
@pytest.mark.parametrize("cond_predict_scale", [False, True])
def test_cond_res_conv_block(group: Group, cond_predict_scale: bool):  # noqa: D103
    from symm_learning.models.difussion.cond_eunet1d import eConditionalResidualBlock1D
    from symm_learning.nn import GSpace1D

    G = group
    gspace = GSpace1D(G)
    mx, my, mc = 2, 3, 5
    in_type = FieldType(gspace, [G.regular_representation] * mx)
    out_type = FieldType(gspace, [G.regular_representation] * my)
    cond_type = FieldType(escnn.gspaces.no_base_space(G), [G.regular_representation] * mc)

    block = eConditionalResidualBlock1D(
        in_type, out_type, cond_type, kernel_size=3, cond_predict_scale=cond_predict_scale
    )
    print(block)
    block.check_equivariance()
