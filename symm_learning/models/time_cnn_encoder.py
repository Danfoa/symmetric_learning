from __future__ import annotations

import math

import escnn
import escnn.nn as escnn_nn
import torch
from escnn.nn import FieldType, GeometricTensor

from symm_learning.models.emlp import EMLP
from symm_learning.models.imlp import IMLP
from symm_learning.nn import eBatchNorm1d
from symm_learning.nn.conv import GSpace1D, eConv1D


class TimeCNNEncoder(torch.nn.Module):
    """1D CNN encoder for inputs of shape (N, in_dim, H).

    The model applies a stack of 1D conv blocks over time. Each block halves the time
    horizon using either stride-2 convolution or max-pooling. The final feature map of
    shape (N, C_L, H_out) is flattened and passed to a 1-hidden-layer MLP head.

    Convolutional features are time-translation equivariant; flattening preserves this
    up to a fixed permutation under shifts. The optional concatenation of the last input
    frame and the final MLP can break output equivariance.

    Args:
      in_dim: Input channel dimension.
      out_dim: Output feature dimension.
      hidden_channels: Channels per conv block; depth equals len(hidden_channels).
      horizon: Input sequence length H.
      activation: Activation module or list (one per block). If a single module is given,
        it is replicated for all blocks.
      batch_norm: If True, add BatchNorm1d after each convolution.
      bias: Use bias in conv/linear layers.
      mlp_hidden: Hidden units of the final MLP head (list for deeper heads).
      downsample: "stride" (default) or "pooling"; each block halves H.
      append_last_frame: If True, concatenate x[:, :, -1] (N, in_dim) to flattened conv
        features before the MLP.

    Returns:
      Tensor of shape (N, out_dim).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_channels: list[int],
        horizon: int,
        activation: torch.nn.Module | list[torch.nn.Module] = torch.nn.ReLU(),
        batch_norm: bool = False,
        bias: bool = True,
        mlp_hidden: list[int] = [128],
        downsample: str = "stride",
        append_last_frame: bool = False,
    ) -> None:
        super().__init__()
        assert hasattr(hidden_channels, "__iter__") and hasattr(hidden_channels, "__len__"), (
            "hidden_channels must be a list of integers"
        )
        assert len(hidden_channels) > 0, "At least one conv block is required"
        assert downsample in {"stride", "pooling"}, "downsample must be 'stride' or 'pooling'"

        self.in_dim, self.out_dim = in_dim, out_dim
        self.horizon_in = int(horizon)
        self.batch_norm = batch_norm
        self.bias = bias
        self.downsample = downsample
        self.append_last_frame = append_last_frame

        # Prepare per-block activations (minimal logic)
        if not isinstance(activation, list):
            conv_acts = [activation] * len(hidden_channels)
        else:
            assert len(activation) == len(hidden_channels), f"{len(activation)} != {len(hidden_channels)}: "
            conv_acts = activation
        head_act = conv_acts[-1]

        # Build conv feature extractor; kernel=3, padding=1 (time length preserved before downsampling)
        conv_layers = []
        cin = in_dim
        h = self.horizon_in

        for cout, act in zip(hidden_channels, conv_acts):
            if self.downsample == "stride":
                # Conv1d with stride=2 halves time: L_out = floor((L+1)/2) for k=3,p=1
                conv_layers.append(torch.nn.Conv1d(cin, cout, kernel_size=3, stride=2, padding=1, bias=bias))
                if batch_norm:
                    conv_layers.append(torch.nn.BatchNorm1d(cout))
                conv_layers.append(act)
                h = (h + 1) // 2
            else:  # pooling
                # Conv stride=1 keeps time, then MaxPool1d(2) halves: floor(L/2)
                conv_layers.append(torch.nn.Conv1d(cin, cout, kernel_size=3, stride=1, padding=1, bias=bias))
                if batch_norm:
                    conv_layers.append(torch.nn.BatchNorm1d(cout))
                conv_layers.append(act)
                conv_layers.append(torch.nn.MaxPool1d(kernel_size=2, stride=2))
                h = h // 2

            cin = cout

        self.feature_extractor = torch.nn.Sequential(*conv_layers)
        self.horizon_out = max(1, h)
        flat_dim = cin * self.horizon_out
        if self.append_last_frame:
            flat_dim += in_dim

        # MLP head (supports multiple hidden layers)
        head_layers = []
        prev = flat_dim
        for hidden in mlp_hidden:
            head_layers.append(torch.nn.Linear(prev, hidden, bias=bias))
            # replicate a fresh activation instance of the same type as head_act
            head_layers.append(type(head_act)())
            prev = hidden
        head_layers.append(torch.nn.Linear(prev, out_dim, bias=bias))

        self.head = torch.nn.Sequential(*head_layers)
        torch.nn.init.orthogonal_(self.head[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape ``(N, in_dim, H)``.
        :type x: torch.Tensor
        :returns: Encoded tensor of shape ``(N, out_dim)``.
        :rtype: torch.Tensor
        """
        z = self.feature_extractor(x)
        z = z.view(z.size(0), -1)
        if self.append_last_frame:
            last = x[:, :, -1]
            z = torch.cat([z, last], dim=1)
        return self.head(z)


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
            self.head = IMLP(
                in_type=head_in_type,
                out_dim=out_type.size,
                hidden_units=mlp_hidden,
                activation="ReLU",
                bias=bias,
            )
        else:
            self.head = EMLP(
                in_type=head_in_type,
                out_type=FieldType(gspace_1d, out_type.representations),
                hidden_units=mlp_hidden,
                activation="ReLU",
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
