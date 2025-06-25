# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from __future__ import annotations

from math import ceil

import escnn
import torch
from escnn.group import Representation
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from symm_learning.models.emlp import EMLP
from symm_learning.nn import IrrepSubspaceNormPooling


class IMLP(EquivariantModule):
    """G-Invariant Multi-Layer Perceptron.

    This module is a G-invariant MLP that extracts G-invariant features from the input tensor. The input tensor
    is first processed by an EMLP module that extracts G-equivariant features. The output of the EMLP module is
    then processed by an IrrepSubspaceNormPooling module that computes the norm of the features in each G-stable
    subspace associated to individual irreducible representations. The output of the IrrepSubspaceNormPooling module
    is a tensor with G-invariant features that can be processed with any NN architecture.
    Default implementation is to add a single linear layer projecting the invariant features to the desired
    output dimension.
    """

    def __init__(
        self,
        in_type: FieldType,
        out_dim: int,  # Number of G-invariant features to extract.
        hidden_units: list[int] = [128, 128, 128],
        activation: str = "ReLU",
        bias: bool = False,
        hidden_rep: Representation = None,
    ):
        super(IMLP, self).__init__()
        assert hasattr(hidden_units, "__iter__") and hasattr(hidden_units, "__len__"), (
            "hidden_units must be a list of integers"
        )
        assert len(hidden_units) > 0, "At least one equivariant layer is required"

        self.G = in_type.fibergroup
        self.in_type = in_type

        equiv_out_type = FieldType(
            gspace=in_type.gspace,
            representations=[self.G.regular_representation] * max(1, ceil(hidden_units[-1] / self.G.order())),
        )

        self.equiv_feature_extractor = EMLP(
            in_type=in_type,
            out_type=equiv_out_type,
            hidden_units=hidden_units,
            activation=activation,
            bias=bias,
            hidden_rep=hidden_rep,
        )
        self.inv_feature_extractor = IrrepSubspaceNormPooling(in_type=self.equiv_feature_extractor.out_type)
        self.head = torch.nn.Linear(
            in_features=self.inv_feature_extractor.out_type.size, out_features=out_dim, bias=bias
        )
        self.out_type = FieldType(gspace=in_type.gspace, representations=[self.G.trivial_representation] * out_dim)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Forward pass of the G-invariant MLP."""
        z = self.equiv_feature_extractor(x)  # Compute the equivariant features using the EMLP module.
        z_inv = self.inv_feature_extractor(z)  # Get G-invariant features
        out = self.head(z_inv.tensor)  # Unconstrained head linear layer.
        return self.out_type(out)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return input_shape[:-1] + (len(self.out_type.size),)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-4) -> list[tuple[any, float]]:  # noqa: D102
        self.equiv_feature_extractor.check_equivariance(atol=atol, rtol=rtol)
        self.inv_feature_extractor.check_equivariance(atol=atol, rtol=rtol)
        return super(IMLP, self).check_equivariance(atol=atol, rtol=rtol)

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        imlp: torch.nn.Sequential = escnn.nn.SequentialModule(
            self.equiv_feature_extractor,
            self.inv_feature_extractor,
        ).export()
        print(self.head)
        imlp.add_module("head", self.head)
        imlp.eval()

        return imlp
