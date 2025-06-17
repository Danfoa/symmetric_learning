from __future__ import annotations

from math import ceil

import escnn
from escnn.nn import EquivariantModule, FieldType, FourierPointwise, GeometricTensor


class EMLP(EquivariantModule):
    """G-Equivariant Multi-Layer Perceptron."""

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        hidden_layers: int = 1,
        hidden_units: int = 128,
        activation: str = "ReLU",
        bias: bool = True,
        hidden_irreps: list | tuple = None,
    ):
        """EMLP constructor.

        Args:
            in_type: Input field type.
            out_type: Output field type.
            hidden_layers: Number of hidden layers.
            hidden_units: Number of units in the hidden layers.
            activation: Name of the class of activation function.
            bias: Whether to include a bias term in the linear layers.
            hidden_irreps: (Optional) List of irreps to use in the hidden layers. If None, the latent representation
            is constructed from multiplicities of the regular representation.
        """
        super(EMLP, self).__init__()
        assert hidden_layers > 0, "A MLP with 0 hidden layers is equivalent to a linear layer"
        self.G = in_type.fibergroup
        self.in_type, self.out_type = in_type, out_type

        hidden_irreps = hidden_irreps or self.G.regular_representation.irreps
        hidden_irreps = tuple(set(hidden_irreps))
        # Number of multiplicities / signals in the hidden layers
        signal_dim = sum(self.G.irrep(*id).size for id in hidden_irreps)

        if isinstance(hidden_units, int):
            hidden_units = [hidden_units] * hidden_layers
        elif isinstance(hidden_units, list):
            assert len(hidden_units) == hidden_layers, "Number of hidden units must match the number of hidden layers"

        layers = []
        layer_in_type = in_type
        for i, units in enumerate(hidden_units):
            channels = int(ceil(units // signal_dim))
            layer = FourierBlock(
                in_type=layer_in_type, irreps=hidden_irreps, channels=channels, activation=activation, bias=bias
            )
            layer_in_type = layer.out_type
            layers.append(layer)

        # Head layer
        layers.append(escnn.nn.Linear(in_type=layer_in_type, out_type=out_type, bias=bias))
        self.net = escnn.nn.SequentialModule(*layers)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Forward pass of the EMLP."""
        return self.net(x)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return self.net.evaluate_output_shape(input_shape)

    def extra_repr(self) -> str:  # noqa: D102
        return f"{self.G}-equivariant MLP: in={self.in_type}, out={self.out_type}"

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        return self.net.export()


class FourierBlock(EquivariantModule):
    """Module applying a linear layer followed by a escnn.nn.FourierPointwise activation."""

    def __init__(
        self,
        in_type: FieldType,
        irreps: tuple | list,
        channels: int,
        activation: str,
        bias: bool = True,
        grid_kwargs: dict = None,
    ):
        super(FourierBlock, self).__init__()
        self.G = in_type.fibergroup
        self._activation = activation
        gspace = in_type.gspace
        grid_kwargs = grid_kwargs or self.get_group_kwargs(self.G)

        self.act = FourierPointwise(
            gspace,
            channels=channels,
            irreps=list(irreps),
            function=f"p_{activation.lower()}",
            inplace=True,
            **grid_kwargs,
        )

        self.in_type = in_type
        self.out_type = self.act.in_type
        self.linear = escnn.nn.Linear(in_type=in_type, out_type=self.act.in_type, bias=bias)

    def forward(self, *input):
        """Forward pass of linear layer followed by activation function."""
        return self.act(self.linear(*input))

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return self.linear.evaluate_output_shape(input_shape)

    @staticmethod
    def get_group_kwargs(group: escnn.group.Group):
        """Configuration for sampling elements of the group to achieve equivariance."""
        grid_type = "regular" if not group.continuous else "rand"
        N = group.order() if not group.continuous else 10
        kwargs = dict()

        if isinstance(group, escnn.group.DihedralGroup):
            N = N // 2
        elif isinstance(group, escnn.group.DirectProductGroup):
            G1_args = FourierBlock.get_group_kwargs(group.G1)
            G2_args = FourierBlock.get_group_kwargs(group.G2)
            kwargs.update({f"G1_{k}": v for k, v in G1_args.items()})
            kwargs.update({f"G2_{k}": v for k, v in G2_args.items()})

        return dict(N=N, type=grid_type, **kwargs)

    def extra_repr(self) -> str:  # noqa: D102
        return f"{self.G}-FourierBlock {self._activation}: in={self.in_type.size}, out={self.out_type.size}"

    def export(self):
        """Exports the module to a torch.nn.Sequential instance."""
        return escnn.nn.SequentialModule(self.linear, self.act).export()
