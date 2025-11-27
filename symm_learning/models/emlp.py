from __future__ import annotations

import logging
from math import ceil

import torch
from escnn.group import Representation

from symm_learning.nn.linear import eLinear
from symm_learning.nn.pooling import IrrepSubspaceNormPooling
from symm_learning.representation_theory import direct_sum

logger = logging.getLogger(__name__)


class eMLP(torch.nn.Module):
    """Equivariant MLP composed of :class:`~symm_learning.nn.linear.eLinear` layers.

    The network preserves the action of the underlying group on every layer by
    constructing hidden representations from the group regular representation
    (or a user-provided base representation) repeated as needed to reach the
    requested width.
    """

    def __init__(
        self,
        in_rep: Representation,
        out_rep: Representation,
        hidden_units: list[int],
        activation: torch.nn.Module = torch.nn.ReLU(),
        dropout: float = 0.0,
        bias: bool = True,
        hidden_rep: Representation | None = None,
        init_scheme: str | None = "xavier_normal",
    ) -> None:
        """Create an equivariant MLP.

        Args:
            in_rep: Input representation defining the group action on the input.
            out_rep: Output representation; must belong to the same group as ``in_rep``.
            hidden_units: Width of each hidden layer (number of representation copies).
            activation: Non-linearity inserted after every hidden layer.
            dropout: Dropout probability applied after activations; ``0.0`` disables it.
            bias: Whether to include a bias term in equivariant linear layers.
            hidden_rep: Base representation used to build hidden layers. Defaults to the
                regular representation when ``None``.
            init_scheme: Parameter initialization scheme passed to :class:`eLinear`.
        """
        super().__init__()
        if len(hidden_units) == 0:
            raise ValueError("hidden_units must contain at least one layer")
        if in_rep.group != out_rep.group:
            raise ValueError("Input and output representations must belong to the same group")

        G = in_rep.group
        self.in_rep, self.out_rep = in_rep, out_rep

        assert isinstance(activation, torch.nn.Module), f"activation must be a torch.nn.Module, got {type(activation)}"

        drop_value = float(dropout)
        assert 0.0 <= drop_value <= 1.0, f"dropout must be within [0, 1], got {drop_value}"

        base_hidden_rep = hidden_rep or G.regular_representation
        assert base_hidden_rep.group == G, "hidden_rep must belong to the same group as in_rep"

        self.hidden_specs = []

        layers: list[torch.nn.Module] = []
        prev_rep = in_rep

        for idx, requested_dim in enumerate(hidden_units):
            target_rep = _hidden_representation(base_hidden_rep, requested_dim)
            linear = eLinear(prev_rep, target_rep, bias=bias, init_scheme=init_scheme)
            layers.append(linear)
            layers.append(activation)
            if drop_value > 0:
                layers.append(torch.nn.Dropout(drop_value))
            prev_rep = target_rep
            self.hidden_specs.append(target_rep)

        layers.append(eLinear(prev_rep, out_rep, bias=bias, init_scheme=init_scheme))
        self.net = torch.nn.Sequential(*layers)

        if init_scheme is not None:
            self.reset_parameters(init_scheme)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the equivariant MLP to ``x`` preserving group structure.

        Args:
            x: Tensor with trailing dimension matching ``in_rep.size``.

        Returns:
            Tensor with trailing dimension ``out_rep.size``.
        """
        assert x.shape[-1] == self.in_rep.size, f"Expected (..., {self.in_rep.size}), got {x.shape}"
        return self.net(x)

    @torch.no_grad()
    def reset_parameters(self, scheme: str = "xavier_normal") -> None:
        """Reinitialize all :class:`eLinear` layers with the provided scheme."""
        for module in self.net:
            if isinstance(module, eLinear):
                module.reset_parameters(scheme)
        logger.debug(f"Initialized eMLP with scheme '{scheme}'")


class iMLP(torch.nn.Module):
    """G-invariant MLP built from an equivariant backbone and invariant head."""

    def __init__(
        self,
        in_rep: Representation,
        out_dim: int,
        hidden_units: list[int],
        activation: torch.nn.Module = torch.nn.ReLU(),
        dropout: float = 0.0,
        bias: bool = True,
        hidden_rep: Representation | None = None,
        init_scheme: str | None = "xavier_normal",
    ):
        """Create a group-invariant MLP.

        The model first applies an equivariant MLP to extract group-aware
        features, pools them into the trivial representation, and finishes with
        an unconstrained linear head to produce invariant outputs.

        Args:
            in_rep: Input representation defining the group action on the input.
            out_dim: Dimension of the invariant output vector.
            hidden_units: Width of each hidden layer in the equivariant backbone.
            activation: Non-linearity inserted after every hidden layer and after the backbone.
            dropout: Dropout probability applied after backbone activations.
            bias: Whether to include biases in the backbone and head.
            hidden_rep: Base representation used to build hidden layers. Defaults to the
                regular representation when ``None``.
            init_scheme: Parameter initialization scheme passed to :class:`eLinear`.
        """
        super().__init__()
        assert isinstance(hidden_units, list) and len(hidden_units) > 0, (
            f"hidden_units must be a non-empty list, got {hidden_units}"
        )
        self.in_rep = in_rep
        self.out_rep = direct_sum([in_rep.group.trivial_representation] * out_dim)
        G = in_rep.group

        # Build the equivariant feature extractor (eMLP)
        last_dim = hidden_units[-1]
        base_hidden_rep = hidden_rep or G.regular_representation
        out_rep = _hidden_representation(base_hidden_rep, last_dim)
        self.emlp_backbone = eMLP(
            in_rep=in_rep,
            out_rep=out_rep,
            hidden_units=hidden_units,
            activation=activation,
            dropout=dropout,
            bias=bias,
            hidden_rep=hidden_rep,
            init_scheme=None,
        )
        # G-invariant pooling
        inv_pooling = IrrepSubspaceNormPooling(in_rep=out_rep)
        # Unconstrained head
        self.head = torch.nn.Linear(
            in_features=inv_pooling.out_rep.size,
            out_features=out_dim,
            bias=bias,
        )
        # Network: [emlp -> activation -> inv pooling -> head]
        self.net = torch.nn.Sequential(*self.emlp_backbone.net, activation, inv_pooling, self.head)

        if init_scheme is not None:
            self.reset_parameters(init_scheme)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute invariant outputs from the input representation values."""
        assert x.shape[-1] == self.in_rep.size, f"Expected (..., {self.in_rep.size}), got {x.shape}"
        return self.net(x)

    @torch.no_grad()
    def reset_parameters(self, scheme: str = "xavier_normal") -> None:
        """Reinitialize all :class:`eLinear` layers with the provided scheme."""
        self.emlp_backbone.reset_parameters(scheme)
        # Initialize the unconstraine head
        if scheme == "xavier_normal":
            torch.nn.init.xavier_normal_(self.head.weight)
        elif scheme == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.head.weight)
        elif scheme == "kaiming_normal":
            torch.nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        elif scheme == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.head.weight, nonlinearity="linear")
        logger.debug(f"Initialized iMLP head with scheme '{scheme}'")


def _hidden_representation(base: Representation, target_dim: int) -> Representation:
    repeats = max(1, ceil(target_dim / base.size))
    return direct_sum([base] * repeats)


class MLP(torch.nn.Module):
    """Standard baseline MLP."""

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
