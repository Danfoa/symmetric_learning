from __future__ import annotations

import math

import escnn
import escnn.nn as escnn_nn
import torch
from escnn.nn import FieldType, GeometricTensor

import symm_learning
from symm_learning.nn import eBatchNorm1d
from symm_learning.nn.conv import GSpace1D, eConv1D

from .cnn_encoder import TimeCNNEncoder


class eTimeCNNEncoder(torch.nn.Module):
    """Equivariant 1D CNN encoder using eConv1D and eBatchNorm1d.

    Processes inputs of shape (N, in_dim, H) through L equivariant conv blocks (stride-2 only).
    The flattened feature map is fed to an equivariant (EMLP) or invariant (IMLP) head depending
    on the requested out_type. Pooling is not supported in the equivariant blocks.

    Args:
      in_type: Input FieldType defining channel representation structure.
      out_type: Desired output field type. If it contains only trivial irreps, an invariant
        head (IMLP) is used; otherwise an equivariant head (EMLP) is used.
      hidden_channels: Channels per conv block; depth equals len(hidden_channels).
      horizon: Input sequence length H.
      activation: Activation module or list (one per block). Only pointwise activations are
        supported equivariantly in the blocks. ReLU/ELU/LeakyReLU/Mish are mapped; others fall
        back to PointwiseNonLinearity with a matching name.
      batch_norm: If True, add eBatchNorm1d after each convolution.
      bias: Use bias in conv/linear layers.
      mlp_hidden: Hidden units of the head MLP as a list (single hidden layer by default).
      downsample: Must be "stride". If "pooling" is requested, NotImplementedError is raised.
      append_last_frame: If True, concatenate the last input frame (N, in_type.size) to the flattened
        conv features before the head. This can break output equivariance.

    Returns:
      GeometricTensor with field type out_type.
    """

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        hidden_channels: list[int],
        time_horizon: int,
        activation: torch.nn.Module | list[torch.nn.Module] = torch.nn.ReLU(),
        batch_norm: bool = False,
        bias: bool = True,
        mlp_hidden: list[int] = [128],
        downsample: str = "stride",
        append_last_frame: bool = False,
    ) -> None:
        super().__init__()

        assert len(hidden_channels) > 0, "At least one conv block is required"
        if downsample != "stride":
            raise NotImplementedError("Equivariant encoder does not support pooling; use downsample='stride'.")

        self.in_type = in_type
        self.out_type = out_type
        self.time_horizon = int(time_horizon)
        self.append_last_frame = append_last_frame

        gspace_1d = in_type.gspace
        G = gspace_1d.fibergroup
        gspace_0d = escnn.gspaces.no_base_space(G)
        reg_rep = G.regular_representation

        if not isinstance(activation, list):
            activation = [activation] * len(hidden_channels)
        else:
            assert len(activation) == len(hidden_channels)
            activation = activation

        # 1D CNN Feature extractor: stride-2 equivariant conv blocks
        layers = []
        cnn_in_type = self.in_type
        h = self.time_horizon

        for c_out, act in zip(hidden_channels, activation):
            multiplicity = max(1, math.ceil(c_out // reg_rep.size))
            cnn_out_type = FieldType(gspace_1d, [reg_rep] * multiplicity)
            # 1D Conv Layer
            conv = eConv1D(in_type=cnn_in_type, out_type=cnn_out_type, kernel_size=3, stride=2, padding=1, bias=bias)
            layers.append(conv)
            h = (h + 1) // 2
            # BatchNorm1d
            if batch_norm:
                layers.append(eBatchNorm1d(cnn_out_type))
            # Activation
            layers.append(self._get_activation(act, cnn_out_type))

            cnn_in_type = cnn_out_type

        self.feature_layers = escnn.nn.SequentialModule(*layers)
        assert h > 0, f"Horizon {self.time_horizon} too short for {len(hidden_channels)} blocks"
        self.time_horizon_out = h

        # Head input FieldType: repeat across time, then optionally append last frame (with same rep as in_type)
        head_reps = cnn_in_type.representations * self.time_horizon_out
        if self.append_last_frame:
            head_reps = head_reps + self.in_type.representations
        head_in_type = FieldType(gspace_0d, head_reps)

        # Choose head based on out_type (trivial-only -> IMLP, else EMLP)
        rep_out = out_type.representation
        invariant_head = set(rep_out.irreps) == {rep_out.group.trivial_representation.id}
        if invariant_head:
            self.head = symm_learning.nn.iMLP(
                in_type=head_in_type,
                out_dim=out_type.size,
                hidden_units=mlp_hidden,
                activation=activation,
                bias=bias,
            )
        else:
            self.head = symm_learning.nn.eMLP(
                in_rep=head_in_type,
                out_rep=FieldType(gspace_1d, out_type.representations),
                hidden_units=mlp_hidden,
                activation=activation,
                batch_norm=batch_norm,
                bias=bias,
            )
        self.out_type = self.head.out_type

    def forward(self, x: torch.Tensor | GeometricTensor) -> GeometricTensor:
        """Forward pass.

        Args:
          x: Input of shape (N, in_type.size, H). Can be a raw tensor or a GeometricTensor.

        Returns:
          GeometricTensor with field type out_type.
        """
        gx = x if isinstance(x, GeometricTensor) else self.in_type(x)

        feats = self.feature_layers(gx)  # Apply equivariant conv blocks
        # Flatten preserving fibers (B, C, H) -> (B, H, C) -> (B, C*H)
        z = feats.tensor.permute(0, 2, 1).contiguous().view(feats.tensor.size(0), -1)
        if self.append_last_frame:
            z = torch.cat([z, gx.tensor[:, :, -1]], dim=1)

        out = self.head(self.head.in_type(z))  # Apply the head
        return out

    def export(self) -> TimeCNNEncoder:  # noqa: D102
        backbone = self.feature_layers.export()
        head = self.head.export()
        return torch.nn.Sequential(backbone, head)

    @staticmethod
    def _get_activation(name: str, in_ft: FieldType) -> escnn_nn.EquivariantModule:
        if name == "relu":
            return escnn_nn.ReLU(in_type=in_ft)
        if name == "elu":
            return escnn_nn.ELU(in_type=in_ft)
        if name == "leakyrelu":
            return escnn_nn.LeakyReLU(in_type=in_ft)
        if name == "mish":
            try:
                import symm_learning.nn as sl_nn

                return sl_nn.Mish(in_type=in_ft)
            except Exception:
                pass
        return escnn_nn.PointwiseNonLinearity(in_type=in_ft, function=f"p_{name}")


if __name__ == "__main__":
    from escnn.group import CyclicGroup

    # Simple sanity check
    torch.manual_seed(0)
    for H in [11]:
        B, in_dim = 128, 64
        out_dim = 10
        # Stride downsampling
        model_s = TimeCNNEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_channels=[16, 32, 64],
            horizon=H,
            activation=torch.nn.ReLU(),
            batch_norm=True,
            downsample="stride",
            append_last_frame=True,
        )
        x = torch.randn(B, in_dim, H)
        print("input: (batch, in_dim, time_horizon) = ", x.shape)
        y = model_s(x)
        print("TimeCNNEncoder out", y.shape)

        G = CyclicGroup(4)
        gspace = GSpace1D(G)
        in_type = FieldType(gspace, [G.regular_representation])

        x = torch.randn(B, in_type.size, H)

        out_type_inv = FieldType(gspace, [G.trivial_representation] * out_dim)
        emodel_inv = eTimeCNNEncoder(
            in_type=in_type,
            out_type=out_type_inv,
            hidden_channels=[32, 64],
            time_horizon=H,
            activation="elu",
            mlp_hidden=[128],
            downsample="stride",
            append_last_frame=True,
        )
        with torch.no_grad():
            y_inv_geom = emodel_inv(x)

        print("eTimeCNNEncoder (IMLP head) output:", y_inv_geom.tensor.shape)
