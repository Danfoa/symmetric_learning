# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from __future__ import annotations

import numpy as np
import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from symm_learning.nn import Change2DisentangledBasis


class IrrepSubspaceNormPooling(EquivariantModule):
    """Module that outputs the norm of the features in each G-irreducible subspace of the input tensor.

    Args:
        in_type: Input FieldType. The dimension of the output tensors will be equal to the number of irreps in this type

    """

    def __init__(self, in_type: FieldType):
        super(IrrepSubspaceNormPooling, self).__init__()
        self.G = in_type.fibergroup
        self.in_type = in_type
        self.in2iso = Change2DisentangledBasis(in_type)
        self.in_type_iso = self.in2iso.out_type
        # The number of features is equal to the number of irreducible representations
        n_inv_features = sum(len(rep.irreps) for rep in self.in_type_iso.representations)
        self.out_type = FieldType(
            gspace=in_type.gspace, representations=[self.G.trivial_representation] * n_inv_features
        )
        # Store isotypic subspaces start/end indices for efficient slicing
        self.register_buffer(
            "iso_start_dims", torch.tensor(self.in2iso.out_type.fields_start.astype(np.int64), dtype=torch.int64)
        )
        self.register_buffer(
            "iso_end_dims", torch.tensor(self.in2iso.out_type.fields_end.astype(np.int64), dtype=torch.int64)
        )
        irreps_ids = [rep.irreps[0] for rep in self.in2iso.out_type.representations]
        self.register_buffer("irreps_dims", torch.tensor([self.G.irrep(*irreps_id).size for irreps_id in irreps_ids]))

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Computes the norm of each G-irreducible subspace of the input GeometricTensor.

        The input_type representation in the spectral basis is composed of direct sum of N irreducible representations.
        This function computes the norms of the vectors on each G-irreducible subspace associated with each irrep.

        Args:
            x: Input GeometricTensor.

        Returns:
            GeometricTensor: G-Invariant tensor of shape (..., N) where N is the number of irreps in the input type.
        """
        x_ = self.in2iso(x)
        # Decompose the input tensor into isotypic subspaces
        x_iso = [x_.tensor[..., s:e] for s, e in zip(self.iso_start_dims, self.iso_end_dims)]

        inv_features_iso = []
        for x_k, s, e, irrep_dim in zip(x_iso, self.iso_start_dims, self.iso_end_dims, self.irreps_dims):
            n_irrep_G_stable_spaces = int((e - s) / irrep_dim)  # Number of G-invariant features = multiplicity of irrep
            # This basis is useful because we can apply the norm in a vectorized way
            # Reshape features to [batch, n_irrep_G_stable_spaces, num_features_per_G_stable_space]
            x_field_p = torch.reshape(x_k, (x_k.shape[0], n_irrep_G_stable_spaces, -1))
            # Compute G-invariant measures as the norm of the features in each G-stable space
            inv_field_features = torch.norm(x_field_p, dim=-1)
            # Append to the list of inv features
            inv_features_iso.append(inv_field_features)

        inv_features = torch.cat(inv_features_iso, dim=-1)
        assert inv_features.shape[-1] == self.out_type.size, (
            f"Expected {self.out_type.size} features, got {inv_features.shape[-1]}"
        )
        return self.out_type(inv_features)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return input_shape[:-1] + (self.out_type.size,)

    def extra_repr(self) -> str:  # noqa: D102
        return f"{self.G}-Irrep Norm Pooling: in={self.in_type} -> out={self.out_type}"

    def export(self) -> torch.nn.Module:
        """Exporting to a torch.nn.Module"""
        return _IrrepSubspaceNormPooling(
            in2iso=self.in2iso.export(),
            iso_start_dims=self.iso_start_dims,
            iso_end_dims=self.iso_end_dims,
            irreps_dims=self.irreps_dims,
        )


class _IrrepSubspaceNormPooling(torch.nn.Module):
    """Torch module result of exporting the IrrepSubspaceNormPooling layer to a standard PyTorch module."""

    def __init__(
        self,
        in2iso: torch.nn.Module,
        iso_start_dims: torch.Tensor,
        iso_end_dims: torch.Tensor,
        irreps_dims: torch.Tensor,
    ):
        super(_IrrepSubspaceNormPooling, self).__init__()
        self.in2iso = in2iso
        self.iso_start_dims = iso_start_dims
        self.iso_end_dims = iso_end_dims
        self.irreps_dims = irreps_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the norm of each G-irreducible subspace of the input tensor."""
        x_ = self.in2iso(x)
        # Decompose the input tensor into isotypic subspaces
        x_iso = [x_[..., s:e] for s, e in zip(self.iso_start_dims, self.iso_end_dims)]

        inv_features_iso = []
        for x_k, s, e, irrep_dim in zip(x_iso, self.iso_start_dims, self.iso_end_dims, self.irreps_dims):
            n_irrep_G_stable_spaces = int((e - s) / irrep_dim)  # Number of G-invariant features = multiplicity of irrep
            # This basis is useful because we can apply the norm in a vectorized way
            # Reshape features to [batch, n_irrep_G_stable_spaces, num_features_per_G_stable_space]
            x_field_p = torch.reshape(x_k, (x_k.shape[0], n_irrep_G_stable_spaces, -1))
            # Compute G-invariant measures as the norm of the features in each G-stable space
            inv_field_features = torch.norm(x_field_p, dim=-1)
            # Append to the list of inv features
            inv_features_iso.append(inv_field_features)
        inv_features = torch.cat(inv_features_iso, dim=-1)
        return inv_features
