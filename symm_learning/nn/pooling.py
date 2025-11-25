# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from __future__ import annotations

import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from symm_learning.linalg import irrep_radii


class IrrepSubspaceNormPooling(EquivariantModule):
    r"""Compute per-irrep radii (Euclidean norms) of an input field.

    For a FieldType whose representation decomposes as :math:`\bigoplus_i \rho_i`, this pooling layer
    returns :math:`\left(\|x_{i,1}\|_2, \ldots, \|x_{i,m_i}\|_2\right)` where each :math:`x_{i,j}` is one copy
    of an irreducible representation in the spectral basis. The computation is handled by
    :func:`symm_learning.linalg.irrep_radii`, which moves the coordinates to the spectral basis via the cached
    change-of-basis matrix and performs a parallel scatter-reduction to accumulate squared norms per irrep copy.

    Args:
        in_type: Input FieldType. The output size equals the total number of irreps in this type.
    """

    def __init__(self, in_type: FieldType):
        super(IrrepSubspaceNormPooling, self).__init__()
        self.G = in_type.fibergroup
        self.in_type = in_type
        # The number of features is equal to the number of irreducible representations
        n_inv_features = len(in_type.representation.irreps)
        self.out_type = FieldType(
            gspace=in_type.gspace, representations=[self.G.trivial_representation] * n_inv_features
        )

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Computes the norm of each G-irreducible subspace of the input GeometricTensor.

        The input_type representation in the spectral basis is composed of direct sum of N irreducible representations.
        This function computes the norms of the vectors on each G-irreducible subspace associated with each irrep.

        Args:
            x: Input GeometricTensor.

        Returns:
            GeometricTensor: G-Invariant tensor of shape (..., N) where N is the number of irreps in the input type.
        """
        x_tensor = x.tensor
        inv_features = irrep_radii(x_tensor, self.in_type.representation)
        assert inv_features.shape[-1] == self.out_type.size, f"{self.out_type.size} != {inv_features.shape[-1]}"
        return self.out_type(inv_features)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return input_shape[:-1] + (self.out_type.size,)

    def extra_repr(self) -> str:  # noqa: D102
        return f"{self.G}-Irrep Norm Pooling: in={self.in_type} -> out={self.out_type}"

    def export(self) -> torch.nn.Module:
        """Exporting to a torch.nn.Module"""
        return NotImplementedError("Exporting IrrepSubspaceNormPooling is not implemented yet.")
