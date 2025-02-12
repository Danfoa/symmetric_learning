# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
import escnn
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, directsum
from escnn.nn import FieldType


@pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10)])
def test_irrep_pooling_equivariance(group: Group):
    """Check the IrrepSubspaceNormPooling layer is G-invariant."""
    from symm_torch.nn.irrep_pooling import IrrepSubspaceNormPooling

    y_rep = directsum([group.regular_representation] * 10)  # ρ_Y = ρ_Χ ⊕ ρ_Χ
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[y_rep])
    pooling_layer = IrrepSubspaceNormPooling(in_type=type_Y)
    pooling_layer.check_equivariance(atol=1e-5, rtol=1e-5)


#
# @pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10)])
# def test_disentangled_layer(group: Group):  # noqa: D103
#     from symm_torch.nn.disentangled import Change2DisentangledBasis
#
#     x_rep = directsum([group.regular_representation] * 10)  # ρ_Y = ρ_Χ ⊕ ρ_Χ
#     type_X = FieldType(gspace=escnn.gspaces.no_base_space(group), representations=[x_rep])
#     pooling_layer = IrrepSubspaceNormPooling(in_type=type_X)
#     pooling_layer.check_equivariance(atol=1e-5, rtol=1e-5)
