from __future__ import annotations

import torch


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
