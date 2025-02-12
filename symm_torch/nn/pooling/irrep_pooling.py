# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from symm_torch.nn.disentangled import Change2IsotypicBasis


class IrrepSubspaceNormPooling(EquivariantModule):
    """Module that outputs the norm of the features in each G-irreducible subspace of the input tensor."""

    def __init__(self, in_type: FieldType):
        super(IrrepSubspaceNormPooling, self).__init__()
        self.G = in_type.fibergroup
        self.in_type = in_type
        self.in2iso = Change2IsotypicBasis(in_type)
        self.in_type_iso = self.in2iso.out_type
        # The number of features is equal to the number of irreducible representations
        n_inv_features = sum(len(rep.irreps) for rep in self.in_type_iso.representations)
        self.out_type = FieldType(
            gspace=in_type.gspace, representations=[self.G.trivial_representation] * n_inv_features
        )

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Computes the norm of each G-irreducible subspace of the input GeometricTensor."""
        x_ = self.in2iso(x)
        x_iso = self._orth_proj_isotypic_subspaces(x_)

        inv_features_iso = []
        for x_k, rep_k in zip(x_iso, self.in_type_iso.representations):
            n_irrep_G_stable_spaces = len(rep_k.irreps)  # Number of G-invariant features = multiplicity of irrep
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

    def _orth_proj_isotypic_subspaces(self, z: GeometricTensor) -> [torch.Tensor]:
        """Compute the orthogonal projection of the input tensor into the isotypic subspaces."""
        assert z.type == self.in_type_iso, f"Expected input tensor of type {self.in_type_iso}, got {z.type}"
        z_iso = [z.tensor[..., s:e] for s, e in zip(z.type.fields_start, z.type.fields_end)]
        return z_iso

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return input_shape[:-1] + (len(self.out_type.size),)

    def extra_repr(self) -> str:  # noqa: D102
        return f"{self.G}-Irrep Norm Pooling: in={self.in_type} -> out={self.out_type}"
