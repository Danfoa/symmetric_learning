from __future__ import annotations

import math
from typing import Iterable

import torch
from escnn.group import Representation

from symm_learning.models.emlp import eMLP, iMLP
from symm_learning.nn import eConv1d, eRMSNorm
from symm_learning.representation_theory import direct_sum


class _eChannelRMSNorm(torch.nn.Module):
    """Apply eRMSNorm over the channel dimension for tensors shaped (B, C, L)."""

    def __init__(self, rep: Representation, eps: float = 1e-6):
        super().__init__()
        self.rep = rep
        self.norm = eRMSNorm(rep, eps=eps, equiv_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over channels; treat time steps as leading dimensions.
        y = x.permute(0, 2, 1)  # (B, L, C)
        y = self.norm(y)
        return y.permute(0, 2, 1)  # (B, C, L)


class eTimeCNNEncoder(torch.nn.Module):
    r"""Equivariant 1D CNN encoder built from channel-equivariant blocks.

    Inputs are plain tensors of shape ``(N, in_rep.size, H)``. Each conv block halves the
    time horizon via stride-2 convolution; optional eRMSNorm and pointwise activation follow.
    The flattened feature map feeds either an equivariant head (:class:`eMLP`) or an invariant
    head (:class:`iMLP`) depending on whether ``out_rep`` contains only the trivial irrep.

    The encoder defines:

    .. math::
        \mathbf{f}_{\mathbf{\theta}}: \mathcal{X}^{H} \to \mathcal{Y},

    where :math:`H` is the input horizon, :math:`\mathcal{X}` is the channel feature
    space transforming by :math:`\rho_{\mathcal{X}}`, and :math:`\mathcal{Y}` transforms
    by :math:`\rho_{\mathcal{Y}}`.

    Functional constraint (equivariant head):

    .. math::
        \mathbf{f}_{\mathbf{\theta}}(\rho_{\mathcal{X}}(g)\mathbf{x})
        = \rho_{\mathcal{Y}}(g)\mathbf{f}_{\mathbf{\theta}}(\mathbf{x})
        \quad \forall g\in\mathbb{G}.

    If ``out_rep`` is trivial-only, the head is invariant:

    .. math::
        \mathbf{f}_{\mathbf{\theta}}(\rho_{\mathcal{X}}(g)\mathbf{x})
        = \mathbf{f}_{\mathbf{\theta}}(\mathbf{x})
        \quad \forall g\in\mathbb{G}.
    """

    def __init__(
        self,
        in_rep: Representation,
        out_rep: Representation,
        hidden_channels: list[int],
        time_horizon: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        batch_norm: bool = False,
        bias: bool = True,
        mlp_hidden: list[int] = (128,),
        downsample: str = "stride",
        append_last_frame: bool = False,
        init_scheme: str | None = "xavier_uniform",
    ) -> None:
        r"""Create an equivariant time-series CNN encoder.

        Args:
            in_rep (:class:`~escnn.group.Representation`): Input representation :math:`\rho_{\text{in}}` defining the
                group action on the input channels.
            out_rep (:class:`~escnn.group.Representation`): Output representation :math:`\rho_{\text{out}}`. If it
                contains only trivial irreps, an :class:`iMLP` head is used; otherwise an :class:`eMLP` head is used.
            hidden_channels: List of output channel counts for each convolution block.
            time_horizon: Length of the input time series (number of frames).
            activation: Non-linearity applied after every convolution block.
            batch_norm: Whether to include channel-wise RMS normalization.
            bias: Whether to include bias in convolutions and linear heads.
            mlp_hidden: Hidden layer widths for the final MLP head.
            downsample: Downsampling strategy, either ``'stride'`` (stride-2 conv) or ``'pooling'`` (max pool).
            append_last_frame: Whether to concatenate the last frame of the input to the encoding before the head.
            init_scheme: Initialization scheme for equivariant layers.
        """
        super().__init__()
        assert len(hidden_channels) > 0, "At least one conv block is required"
        assert downsample in {"stride", "pooling"}, "downsample must be 'stride' or 'pooling'"

        self.in_rep = in_rep
        self.out_rep = out_rep
        self.time_horizon = int(time_horizon)
        self.append_last_frame = append_last_frame
        self.downsample = downsample

        G = in_rep.group
        reg_rep = G.regular_representation

        layers: list[torch.nn.Module] = []
        cnn_in_rep = in_rep
        h = self.time_horizon

        for c_out in hidden_channels:
            multiplicity = max(1, math.ceil(c_out / reg_rep.size))
            cnn_out_rep = direct_sum([reg_rep] * multiplicity)

            if self.downsample == "stride":
                layers.append(
                    eConv1d(
                        cnn_in_rep, cnn_out_rep, kernel_size=3, stride=2, padding=1, bias=bias, init_scheme=init_scheme
                    )
                )
                if batch_norm:
                    layers.append(_eChannelRMSNorm(cnn_out_rep))
                layers.append(activation)
                h = (h + 1) // 2
            else:  # pooling
                layers.append(
                    eConv1d(
                        cnn_in_rep, cnn_out_rep, kernel_size=3, stride=1, padding=1, bias=bias, init_scheme=init_scheme
                    )
                )
                if batch_norm:
                    layers.append(_eChannelRMSNorm(cnn_out_rep))
                layers.append(activation)
                layers.append(torch.nn.MaxPool1d(kernel_size=2, stride=2))
                h = h // 2

            cnn_in_rep = cnn_out_rep

        self.feature_layers = torch.nn.Sequential(*layers)
        assert h > 0, f"Horizon {self.time_horizon} too short for {len(hidden_channels)} blocks"
        self.time_horizon_out = h

        # Head input representation: repeat conv out_rep for each remaining time step
        head_rep = direct_sum([cnn_in_rep] * self.time_horizon_out)
        if self.append_last_frame:
            head_rep = direct_sum([head_rep, in_rep])
        self.head_in_rep = head_rep

        # Choose head: invariant if out_rep is trivial-only, else equivariant
        trivial_id = G.trivial_representation.id
        invariant_head = set(out_rep.irreps) == {trivial_id}
        if invariant_head:
            self.head = iMLP(
                in_rep=head_rep,
                out_dim=out_rep.size,
                hidden_units=list(mlp_hidden),
                activation=activation,
                bias=bias,
                init_scheme=init_scheme,
            )
        else:
            self.head = eMLP(
                in_rep=head_rep,
                out_rep=out_rep,
                hidden_units=list(mlp_hidden),
                activation=activation,
                bias=bias,
                init_scheme=init_scheme,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input of shape ``(N, in_rep.size, H)`` into ``(N, out_rep.size)``."""
        assert x.shape[-2:] == (self.in_rep.size, self.time_horizon), (
            f"Expected input shape (..., {self.in_rep.size}, {self.time_horizon}), got {x.shape}"
        )
        feats = self.feature_layers(x)
        z = feats.permute(0, 2, 1).reshape(feats.size(0), -1)
        if self.append_last_frame:
            z = torch.cat([z, x[:, :, -1]], dim=1)
        return self.head(z)

    @torch.no_grad()
    def check_equivariance(self, atol: float = 1e-5, rtol: float = 1e-5):
        """Check equivariance under channel actions of the underlying group."""
        import random

        G = self.in_rep.group
        B, L = 10, self.time_horizon
        dtype, device = next(self.head.parameters()).dtype, next(self.head.parameters()).device

        x = torch.randn(B, self.in_rep.size, L, device=device, dtype=dtype)
        y = self(x)

        elements = set(G.elements)
        for _ in range(10):
            g = random.choice(tuple(elements))
            rho_in = torch.tensor(self.in_rep(g), dtype=x.dtype, device=x.device)
            rho_out = torch.tensor(self.out_rep(g), dtype=y.dtype, device=y.device)
            gx = torch.einsum("ij,bjl->bil", rho_in, x)
            gy = self(gx)
            gy_exp = torch.einsum("ij,bj->bi", rho_out, y)
            assert torch.allclose(gy_exp, gy, atol=atol, rtol=rtol), (
                f"Equivariance failed for group element {g} with max error {(gy_exp - gy).abs().max().item():.3e}"
            )

            elements.remove(g)
            if len(elements) == 0:
                break
