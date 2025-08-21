from __future__ import annotations

import logging
from math import ceil

import escnn
import torch
from escnn.group import Representation, directsum
from escnn.nn import EquivariantModule, FieldType, FourierPointwise, GeometricTensor, PointwiseNonLinearity

import symm_learning


def _get_group_kwargs(group: escnn.group.Group) -> dict:
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
        hidden_units: list[int],
        activation: str | list[str] = "ReLU",
        batch_norm: bool = False,
        pointwise_activation: bool = True,
        bias: bool = True,
        hidden_rep: Representation | None = None,
    ):
        """EMLP constructor.

        Args:
            in_type: Input field type.
            out_type: Output field type.
            hidden_units: List of number of units in each hidden layer.
            activation: Name of the class of activation function or list of activation names.
            bias: Whether to include a bias term in the linear layers.
            batch_norm: Whether to include equivariant batch normalization before activations.
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
            if batch_norm:
                layers.append(symm_learning.nn.eBatchNorm1d(in_type=act.in_type))
            layer_in_type = act.out_type
            layers.extend([linear, act])

        # Head layer
        layers.append(escnn.nn.Linear(in_type=layer_in_type, out_type=out_type, bias=bias))
        self.net = escnn.nn.SequentialModule(*layers)

    def forward(self, x: GeometricTensor | torch.Tensor) -> GeometricTensor:
        """Forward pass of the EMLP."""
        if not isinstance(x, GeometricTensor):
            x = self.in_type(x)
        return self.net(x)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:  # noqa: D102
        return self.net.evaluate_output_shape(input_shape)

    def extra_repr(self) -> str:  # noqa: D102
        return f"{self.G}-equivariant MLP: in={self.in_type}, out={self.out_type}"

    def export(self) -> torch.nn.Sequential:
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

    def _check_for_shur_blocking(self, hidden_rep: Representation) -> None:
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


class MLP(torch.nn.Module):
    """Standard baseline MLP. Symmetry-related parameters are ignored."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_units: list[int],
        activation: torch.nn.Module | list[torch.nn.Module] = torch.nn.ReLU(),
        batch_norm: bool = False,
        bias: bool = True,
    ):
        """Constructor of a Multi-Layer Perceptron (MLP) model.

        Args:
            in_dim: Dimension of the input space.
            out_dim: Dimension of the output space.
            hidden_units: List of number of units in each hidden layer.
            activation: Activation module or list of activation modules.
            batch_norm: Whether to include batch normalization.
            bias: Whether to include a bias term in the linear layers.
        """
        super().__init__()
        logging.info("Instantiating MLP (PyTorch)")
        self.in_dim, self.out_dim = in_dim, out_dim

        assert hasattr(hidden_units, "__iter__") and hasattr(hidden_units, "__len__"), (
            "hidden_units must be a list of integers"
        )
        assert len(hidden_units) > 0, "A MLP with 0 hidden layers is equivalent to a linear layer"

        # Handle activation modules
        if isinstance(activation, list):
            assert len(activation) == len(hidden_units), (
                "List of activation modules must have the same length as the number of hidden layers"
            )
            activations = activation
        else:
            activations = [activation] * len(hidden_units)

        layers = []
        dim_in = in_dim

        for units, act_module in zip(hidden_units, activations):
            layers.append(torch.nn.Linear(dim_in, units, bias=bias))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(units))
            layers.append(act_module)
            dim_in = units

        # Head layer (output layer)
        layers.append(torch.nn.Linear(dim_in, out_dim, bias=bias))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model."""
        output = self.net(input)
        return output
