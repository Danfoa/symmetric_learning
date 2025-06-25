import torch
import torch.nn.functional as F
from escnn.nn import EquivariantModule


class Mish(EquivariantModule):
    """Mish activation function."""

    def __init__(self, in_type):
        super().__init__()

        self.in_type = in_type
        self.out_type = in_type
        for r in in_type.representations:
            assert "pointwise" in r.supported_nonlinearities, (
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)
            )

    def forward(self, x):  # noqa: D102
        return self.out_type(F.mish(x.tensor))

    def export(self) -> torch.nn.Mish:  # noqa: D102
        """Export the layer to a PyTorch module."""
        return torch.nn.Mish()

    def evaluate_output_shape(self, input_shape):  # noqa: D102
        return input_shape
