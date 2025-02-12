from __future__ import annotations

import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor


class ResidualEncoder(EquivariantModule):
    """Simple residual encoder that concatenates the input with the output of an encoder."""

    def __init__(self, encoder: EquivariantModule):
        """Initializes the ResidualEncoder.

        Args:
            encoder: Encoder to be used.
        """
        super(ResidualEncoder, self).__init__()
        self.encoder = encoder
        self.in_type = encoder.in_type
        self.out_type = FieldType(
            gspace=encoder.out_type.gspace,
            representations=self.in_type.representations + encoder.out_type.representations,
        )

    def forward(self, input: GeometricTensor):
        """Computes the output of the encoder and concatenates it with the input."""
        embedding = self.encoder(input)
        out = torch.cat([input.tensor, embedding.tensor], dim=-1)
        return self.out_type(out)

    def decode(self, encoded_x: torch.Tensor):
        """Decode the encoded tensor back to the input tensor."""
        x = encoded_x[..., self.residual_dims]
        return x

    @property
    def residual_dims(self):
        """Slice of dimensions of the output vector containing the input vector."""
        return slice(0, self.in_type.size)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return input_shape[:-1] + (len(self.out_type.size),)
