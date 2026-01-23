#  Code Taken from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py

from __future__ import annotations

import logging
import math
from typing import Union

import torch
import torch.nn as nn


class UnsqueezeLast(nn.Module):
    """Append a singleton channel dimension."""

    def forward(self, x):  # noqa: D102
        return x.unsqueeze(-1)


logger = logging.getLogger(__name__)


class ConditionalUnet1D(nn.Module):
    """A 1D/Time Unet architecture for predicting the score vector: `∇log(P(y | x))`.

    This model is intended to take a given sample y and to predict the conditional
    probability P(y | x) score functional `∇log(P(y | x))`, which can be used in score-based diffusion process
    to compute a new sample `y' = y + ∇log(P(y | x))` featuring higher likelihood given the conditional probability
    distribution.

    The influence of x in the diffusion process is captured via `local` and `global` conditioning of the Unet
    architecture.

    Local conditioning: Provided a local conditioning encoder `z(x)`, the output of the encoder is
        concatenated to the input of the Unet architecture.
    Global conditioning: Provided a global conditioning vector `c = b(x)`, the output of the encoder is
        used to modulate the convolutional layers of the Unet architecture via Feature-Wise Linear Modulation (FiLM)
        modulation.

    Args:
        input_dim (int): The dimension of the input data.
        local_cond_dim (int, optional): The dimension of the local conditioning vector. Defaults to None.
        global_cond_dim (int, optional): The dimension of the global conditioning vector. Defaults to None.
        diffusion_step_embed_dim (int, optional): The dimension of the diffusion step embedding. Defaults to 256.
        down_dims (list, optional): A list of dimensions for the downsampling path. Defaults to [256, 512, 1024].
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
        n_groups (int, optional): The number of groups for GroupNorm. Defaults to 8.
        cond_predict_scale (bool, optional): Whether to predict the scale for conditioning. Defaults to False.
    """

    def __init__(
        self,
        input_dim,
        local_cond_dim: int = None,
        global_cond_dim: int = None,
        diffusion_step_embed_dim=16,
        down_dims=[32, 64],
        kernel_size=3,
        n_groups=1,
        cond_predict_scale=True,
    ):
        super().__init__()

        # Calculate effective input dimension considering local conditioning concatenation
        effective_input_dim = input_dim
        if local_cond_dim is not None:
            effective_input_dim = input_dim + local_cond_dim

        all_dims = [effective_input_dim] + list(down_dims)

        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        cond_dim = diffusion_step_embed_dim
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(effective_input_dim, effective_input_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(effective_input_dim, input_dim, 1),
        )

        self.input_dim = input_dim
        self.local_cond_dim = local_cond_dim
        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: torch.Tensor = None,
        film_cond: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass of the Conditional Unet 1D model.

        Args:
            sample (torch.Tensor): The input tensor of shape (B, input_dim, T).
            timestep (Union[torch.Tensor, float, int]): The diffusion timestep.
            local_cond (torch.Tensor, optional): The local conditioning tensor of shape (B, local_cond_dim).
                Defaults to None.
            film_cond (torch.Tensor, optional): The global conditioning tensor of shape (B, film_cond_dim).
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor of shape (B, input_dim, T).
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        film_diff_step_features = self.diffusion_step_encoder(timesteps)

        if film_cond is not None:
            film_features = torch.cat([film_cond, film_diff_step_features], axis=-1)
        else:
            film_features = film_diff_step_features

        # Handle local conditioning by concatenation at the input of the UNet
        if self.local_cond_dim is not None:
            assert local_cond is not None and local_cond.shape == (sample.shape[0], self.local_cond_dim), (
                f"local_cond does not match expected {(sample.shape[0], self.local_cond_dim)}"
            )
            # Expand local_cond to match time dimension of sample (B, local_cond_dim) -> (B, local_cond_dim, T)
            local_cond_expanded = local_cond.unsqueeze(-1).expand(-1, -1, sample.shape[-1])
            x = torch.cat([sample, local_cond_expanded], dim=1)
        else:
            x = sample

        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, film_features)
            x = resnet2(x, film_features)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, film_features)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, film_features)
            x = resnet2(x, film_features)
            x = upsample(x)

        x = self.final_conv(x)

        return x


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding layer.

    This layer encodes a scalar input (e.g., a diffusion timestep) into a high-dimensional
    vector using a combination of sine and cosine functions of varying frequencies. This technique,
    introduced in the "Attention Is All You Need" paper, allows the model to easily attend
    to relative positions and is effective for representing periodic or sequential data.

    The embedding is calculated as follows:
        emb(x, 2i) = sin(x / 10000^(2i/dim))
        emb(x, 2i+1) = cos(x / 10000^(2i/dim))
    where `x` is the input scalar, `dim` is the embedding dimension, and `i` is the channel index.

    The `forward` method implements this by first calculating the frequency term `1 / 10000^(2i/dim)`
    and then multiplying the input `x` by these frequencies. This creates the argument for the
    sine and cosine functions, effectively encoding the position `x` across the embedding dimension.

    Args:
        dim (int): The dimension of the embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):  # noqa: D102
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max((half_dim - 1), 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """Downsampling layer for 1D data.

    Args:
        dim (int): The number of input and output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):  # noqa: D102
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsampling layer for 1D data.

    Args:
        dim (int): The number of input and output channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):  # noqa: D102
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """A 1D convolutional block with GroupNorm and Mish activation.

    Args:
        inp_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        n_groups (int, optional): The number of groups for GroupNorm. Defaults to 8.
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):  # noqa: D102
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Conditional residual block for 1D convolution with FiLM modulation.

    This block applies two 1D convolutional blocks with conditional modulation between them.
    The conditioning is applied via Feature-wise Linear Modulation (FiLM) which can either
    predict scale and bias parameters or just bias, depending on the `cond_predict_scale` flag.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_dim (int): Dimension of the conditioning vector.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        n_groups (int, optional): Number of groups for GroupNorm. Defaults to 8.
        cond_predict_scale (bool, optional): If True, conditioning predicts both scale and bias.
            If False, conditioning only predicts bias. Defaults to False.
    """

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Linear(cond_dim, cond_channels), nn.Mish(), UnsqueezeLast())

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """Forward pass of the conditional residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, horizon).
            cond (torch.Tensor): Conditioning vector of shape (batch_size, cond_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, horizon).
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test parameters
    batch_size = 512
    input_dim = 99
    local_cond_dim = 10
    film_cond_dim = 5
    time_steps = 200

    # Create model
    model = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=local_cond_dim,
        global_cond_dim=film_cond_dim,
        diffusion_step_embed_dim=16,
        down_dims=[32, 64],
        kernel_size=3,
        n_groups=1,
        cond_predict_scale=True,
    ).to(device)

    # Create test inputs
    sample = torch.randn(batch_size, input_dim, time_steps, device=device, requires_grad=True)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    local_cond = torch.randn(batch_size, local_cond_dim, device=device)
    film_cond = torch.randn(batch_size, film_cond_dim, device=device)

    print("\n" + "=" * 60)
    print("TEST 1: Model with LOCAL + GLOBAL conditioning")
    print("=" * 60)
    # Create model
    model = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=local_cond_dim,
        global_cond_dim=film_cond_dim,
        diffusion_step_embed_dim=16,
        down_dims=[32, 64],
        kernel_size=3,
        n_groups=1,
        cond_predict_scale=True,
    ).to(device)

    # Forward pass
    output = model(sample=sample, timestep=timestep, local_cond=local_cond, film_cond=film_cond)
    print(f"Output shape: {output.shape}")
