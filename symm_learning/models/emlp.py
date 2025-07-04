from __future__ import annotations

from math import ceil

import escnn
from escnn.group import Representation, directsum
from escnn.nn import EquivariantModule, FieldType, FourierPointwise, GeometricTensor, PointwiseNonLinearity


def _get_group_kwargs(group: escnn.group.Group):
    """Configuration for sampling elements of the group to achieve equivariance."""
    grid_type = "regular" if not group.continuous else "rand"
    N = group.order() if not group.continuous else 10
    kwargs = dict()

    if isinstance(group, escnn.group.DihedralGroup):
        N = N // 2
    elif isinstance(group, escnn.group.DirectProductGroup):
        G1_args = _get_group_kwargs(group.G1)
        G2_args = _get_group_kwargs(group.G2)
        kwargs.update({f"G1_{k}": v for k, v in G1_args.items()})
        kwargs.update({f"G2_{k}": v for k, v in G2_args.items()})

    return dict(N=N, type=grid_type, **kwargs)


class EMLP(EquivariantModule):
    """G-Equivariant Multi-Layer Perceptron."""

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        hidden_units: int = 128,
        activation: str | list[str] = "ReLU",
        pointwise_activation: bool = True,
        bias: bool = True,
        hidden_rep: Representation = None,
    ):
        """EMLP constructor.

        Args:
            in_type: Input field type.
            out_type: Output field type.
            hidden_units: (list[int]) List of number of units in each hidden layer.
            activation: Name of the class of activation function.
            bias: Whether to include a bias term in the linear layers.
            hidden_rep: Representation used (up to multiplicity) to construct the hidden layer `FieldType`. If None,
                it defaults to the regular representation.
            pointwise_activation: Whether to use a pointwise activation function (e.g., ReLU, ELU, LeakyReLU). This
                only works for latent representations build in regular (permutation) basis. If False, a
                `FourierPointwise` activation is used, and the latent representations are build in the irrep spectral
                basis.
        """
        super(EMLP, self).__init__()
        assert hasattr(hidden_units, "__iter__") and hasattr(hidden_units, "__len__"), (
            "hidden_units must be a list of integers"
        )
        assert len(hidden_units) > 0, "A MLP with 0 hidden layers is equivalent to a linear layer"

        self.G = in_type.fibergroup
        self.in_type, self.out_type = in_type, out_type
        self.pointwise_activation = pointwise_activation

        hidden_rep = hidden_rep or self.G.regular_representation
        self._check_for_shur_blocking(hidden_rep)

        if isinstance(activation, str):
            activations = [activation] * len(hidden_units)
        else:
            assert isinstance(activation, list) and len(activation) == len(hidden_units), (
                "List of activation names must have the same length as the number of hidden layers"
            )
            activations = activation

        layers = []
        layer_in_type = in_type
        for units, act_name in zip(hidden_units, activations):
            act = self._get_activation(act_name, hidden_rep, units)
            linear = escnn.nn.Linear(in_type=layer_in_type, out_type=act.in_type, bias=bias)
            layer_in_type = act.out_type
            layers.extend([linear, act])

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
        """Exporting to a torch.nn.Sequential"""
        if not self.pointwise_activation:
            raise RuntimeError(
                "`FourierPointwise` activation has no `export` method. Only EMLP with `pointwise_activation=True` "
                "can be exported at the moment"
            )
        return self.net.export()

    def _get_activation(self, activation: str, hidden_rep: Representation, n_units: int) -> EquivariantModule:
        """Gets a representation action on the output of a linear layer with n_units /neurons"""
        channels = max(1, ceil(n_units / hidden_rep.size))
        if self.pointwise_activation:
            in_type = FieldType(self.in_type.gspace, representations=[hidden_rep] * channels)
            if activation.lower() == "elu":
                act = escnn.nn.ELU(in_type=in_type)
            elif activation.lower() == "relu":
                act = escnn.nn.ReLU(in_type=in_type)
            elif activation.lower() == "leakyrelu":
                act = escnn.nn.LeakyReLU(in_type=in_type)
            elif activation.lower() == "mish":
                import symm_learning.nn

                act = symm_learning.nn.Mish(in_type=in_type)
            else:
                act = escnn.nn.PointwiseNonLinearity(in_type=in_type, function=f"p_{activation.lower()}")
        else:
            grid_kwargs = _get_group_kwargs(self.G)
            act = FourierPointwise(
                self.in_type.gspace,
                channels=channels,
                irreps=list(set(hidden_rep.irreps)),
                function=f"p_{activation.lower()}",
                inplace=True,
                **grid_kwargs,
            )
        return act

    def _check_for_shur_blocking(self, hidden_rep: Representation):
        """Check if large portions of the network will be zeroed due to Shur's orthogonality relations."""
        if self.pointwise_activation:
            out_irreps = set(self.out_type.representation.irreps)
            in_irreps = set(self.in_type.representation.irreps)
            hidden_irreps = set(hidden_rep.irreps)

            # Get the set of irreps in the output not present in the hidden representation:
            out_missing_irreps = out_irreps - hidden_irreps
            in_missing_irreps = in_irreps - hidden_irreps
            msg = (
                "\n\tUsing `pointwise_activation` the dimensions associated to the missing irreps will be zeroes out"
                " (by Shur's orthogonality). "
                "\n\tEither set `pointwise_activation=False` or pass a different `hidden_rep`"
            )
            if len(out_missing_irreps) > 0:
                raise ValueError(
                    f"Output irreps {out_missing_irreps} not present in the hidden layers irreps {hidden_irreps}.{msg}"
                )
            if len(in_missing_irreps) > 0:
                raise ValueError(
                    f"Input irreps {in_missing_irreps} not present in the hidden layers irreps {hidden_irreps}.{msg}"
                )
        else:
            return


# class FourierBlock(EquivariantModule):
#     """Module applying a linear layer followed by a escnn.nn.FourierPointwise activation."""

#     def __init__(
#         self,
#         in_type: FieldType,
#         irreps: tuple | list,
#         channels: int,
#         activation: str,
#         bias: bool = True,
#         grid_kwargs: dict = None,
#     ):
#         super(FourierBlock, self).__init__()
#         self.G = in_type.fibergroup
#         self._activation = activation
#         gspace = in_type.gspace
#         grid_kwargs = grid_kwargs or _get_group_kwargs(self.G)

#         self.act = FourierPointwise(
#             gspace,
#             channels=channels,
#             irreps=list(irreps),
#             function=f"p_{activation.lower()}",
#             inplace=True,
#             **grid_kwargs,
#         )

#         self.in_type = in_type
#         self.out_type = self.act.in_type
#         self.linear = escnn.nn.Linear(in_type=in_type, out_type=self.act.in_type, bias=bias)

#     def forward(self, *input):
#         """Forward pass of linear layer followed by activation function."""
#         return self.act(self.linear(*input))

#     def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
#         return self.linear.evaluate_output_shape(input_shape)

#     def extra_repr(self) -> str:  # noqa: D102
#         return f"{self.G}-FourierBlock {self._activation}: in={self.in_type.size}, out={self.out_type.size}"

#     def export(self):
#         """Exports the module to a torch.nn.Sequential instance."""
#         return escnn.nn.SequentialModule(self.linear, self.act).export()
