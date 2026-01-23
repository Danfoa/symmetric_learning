from __future__ import annotations

from math import ceil
from typing import Iterable

import torch
from escnn.group import Representation

from symm_learning.models.diffusion.cond_unet1d import SinusoidalPosEmb
from symm_learning.nn import IrrepSubspaceNormPooling, eAffine, eConv1d, eConvTranspose1d, eRMSNorm
from symm_learning.representation_theory import direct_sum


class _eChannelRMSNorm(torch.nn.Module):
    """Apply eRMSNorm over channels for tensors shaped (B, C, L)."""

    def __init__(self, rep: Representation, eps: float = 1e-6):
        super().__init__()
        self.norm = eRMSNorm(rep, eps=eps, equiv_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 1)  # (B, L, C)
        y = self.norm(y)
        return y.permute(0, 2, 1)


class eConditionalResidualBlock1D(torch.nn.Module):
    """Channel-equivariant conditional residual block with FiLM modulation.

    This block applies two equivariant convolutional layers with a residual
    connection. The output of the first convolutional block is modulated by a
    conditioning tensor using an equivariant FiLM (Feature-wise Linear Modulation)
    layer. The FiLM layer applies an equivariant affine transformation, where the
    scale and bias parameters are predicted by an invariant neural network from the
    conditioning tensor.

    The scale transformation is applied in the spectral domain, with a separate
    learnable parameter for each irreducible representation (irrep). The bias
    is applied only to the invariant (trivial) subspaces of the representation.
    See details in :class:`~symm_learning.nn.eAffine`.
    """

    def __init__(
        self,
        in_rep: Representation,
        out_rep: Representation,
        cond_rep: Representation,
        kernel_size: int = 3,
        cond_predict_scale: bool = True,
        activation: torch.nn.Module = torch.nn.ReLU(),
        normalize: bool = True,
        init_scheme: str | None = "xavier_uniform",
    ):
        r"""Initialize the conditional residual block.

        Args:
            in_rep (Representation): Input representation acting on the channel axis.
            out_rep (Representation): Output representation for the convolutions and residual.
            cond_rep (Representation): Representation of the conditioning vector used to predict FiLM parameters.
            kernel_size (int, optional): Spatial kernel size for the equivariant convolutions. Defaults to 3.
            cond_predict_scale (bool, optional): Whether the FiLM encoder predicts scale parameters.
                Currently must be ``True``. Defaults to True.
            activation (torch.nn.Module, optional): Non-linearity applied after each normalization. Defaults to ``ReLU``
            normalize (bool, optional): If ``True``, apply channel-wise equivariant RMS normalization. Defaults to True.
            init_scheme (str | None, optional): Weight initialization scheme for convolutions.
                Defaults to ``\"xavier_uniform\"``.
        """
        super().__init__()
        self.in_rep, self.out_rep, self.cond_rep = in_rep, out_rep, cond_rep
        if not cond_predict_scale:
            raise NotImplementedError("Currently only cond_predict_scale=True is supported.")

        self.conv1 = eConv1d(in_rep, out_rep, kernel_size, padding=kernel_size // 2, init_scheme=init_scheme)
        self.conv2 = eConv1d(out_rep, out_rep, kernel_size, padding=kernel_size // 2, init_scheme=init_scheme)

        self.norm1 = _eChannelRMSNorm(out_rep) if normalize else torch.nn.Identity()
        self.norm2 = _eChannelRMSNorm(out_rep) if normalize else torch.nn.Identity()
        self.act = activation

        self.affine = eAffine(in_rep=out_rep, learnable=False)
        self.film_dims = self.affine.num_scale_dof + self.affine.num_bias_dof

        # Invariant encoder of FiLM modulation parameters
        self.cond_encoder = torch.nn.Sequential(
            IrrepSubspaceNormPooling(in_rep=cond_rep),
            torch.nn.Linear(in_features=len(cond_rep.irreps), out_features=self.film_dims),
            torch.nn.Mish(),
        )

        self.residual_conv = (  # Final conv for residual addition if needed.
            eConv1d(in_rep, out_rep, 1, init_scheme=init_scheme) if in_rep != out_rep else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply two equivariant conv blocks with FiLM modulation and a residual connection.

        Args:
            x (torch.Tensor): Input of shape ``(B, in_rep.size, L)`` where ``B`` is batch and ``L`` is sequence length.
            cond (torch.Tensor): Conditioning tensor of shape ``(B, cond_rep.size)`` used to predict FiLM parameters.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, out_rep.size, L)`` after modulation and residual addition.
        """
        assert x.shape[1] == self.in_rep.size, f"Expected channel dim {self.in_rep.size}, got {x.shape}"
        assert cond.shape[-1] == self.cond_rep.size, f"Cond dim mismatch {cond.shape}"

        B, _, L = x.shape

        out = self.conv1(x)  # (B, C, L)
        out = self.act(self.norm1(out))

        if self.film_dims > 0:
            film_params = self.cond_encoder(cond)  # (B, film_dims)
            scale_dof = torch.broadcast_to(
                film_params[:, None, : self.affine.num_scale_dof], (B, L, self.affine.num_scale_dof)
            )  # (B, num_scale_dof) -> (B, L, num_scale_dof)
            bias_dof = torch.broadcast_to(
                film_params[:, None, self.affine.num_scale_dof :], (B, L, self.affine.num_bias_dof)
            )  # (B, num_bias_dof) -> (B, L, num_bias_dof)
            out = self.affine(
                out.permute(0, 2, 1),  # (B, C, L) -> (B, L, C)
                scale_dof=scale_dof,
                bias_dof=bias_dof,
            ).permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        out = self.conv2(out)
        out = self.act(self.norm2(out))

        res = self.residual_conv(x) if self.residual_conv is not None else x
        return out + res

    @torch.no_grad()
    def check_equivariance(self, atol=1e-5, rtol=1e-5):  # noqa: D102
        G = self.in_rep.group
        B, L = 10, 30
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype
        x = torch.randn(B, self.in_rep.size, L, device=device, dtype=dtype)
        z = torch.randn(B, self.cond_rep.size, device=device, dtype=dtype)
        y = self(x, z)

        for _ in range(10):
            g = G.sample()
            rho_in = torch.tensor(self.in_rep(g), dtype=x.dtype, device=x.device)
            rho_out = torch.tensor(self.out_rep(g), dtype=y.dtype, device=y.device)
            rho_cond = torch.tensor(self.cond_rep(g), dtype=z.dtype, device=z.device)
            gx = torch.einsum("ij,bjl->bil", rho_in, x)
            gz = torch.einsum("ij,bj->bi", rho_cond, z)
            y_expected = self(gx, gz)
            gy = torch.einsum("ij,bjl->bil", rho_out, y)
            assert torch.allclose(gy, y_expected, atol=atol, rtol=rtol), (
                f"Equivariance failed for group element {g} with max error {(gy - y_expected).abs().max().item():.3e}"
            )


class eConditionalUnet1D(torch.nn.Module):
    """Equivariant U-Net for 1D signals with global conditioning and FiLM."""

    def __init__(
        self,
        in_rep: Representation,
        local_cond_rep: Representation | None,
        global_cond_rep: Representation | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Iterable[int] = (256, 512, 1024),
        kernel_size: int = 3,
        cond_predict_scale: bool = True,
        activation: torch.nn.Module = torch.nn.ReLU(),
        normalize: bool = True,
        downsample: str = "stride",
        init_scheme: str | None = "xavier_uniform",
    ):
        super().__init__()
        assert downsample in {"stride", "pooling"}, "downsample must be 'stride' or 'pooling'"
        self.in_rep = self.out_rep = in_rep
        self.global_cond_rep = global_cond_rep
        self.local_cond_rep = local_cond_rep
        self.downsample = downsample

        G = in_rep.group
        reg_rep = G.regular_representation
        trivial_rep = G.trivial_representation

        down_dims = list(down_dims)
        # all_dims = [in_rep.size] + down_dims
        reps = [in_rep] + [direct_sum([reg_rep] * ceil(d / reg_rep.size)) for d in down_dims]

        diffusion_step_encoder = torch.nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            torch.nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            torch.nn.Mish(),
            torch.nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        if global_cond_rep is not None:
            cond_rep = direct_sum([global_cond_rep, direct_sum([trivial_rep] * diffusion_step_embed_dim)])
        else:
            cond_rep = direct_sum([trivial_rep] * diffusion_step_embed_dim)
        self.cond_rep = cond_rep
        self.diffusion_step_encoder = diffusion_step_encoder

        block_kwargs = dict(
            kernel_size=kernel_size,
            cond_predict_scale=cond_predict_scale,
            activation=activation,
            normalize=normalize,
            init_scheme=init_scheme,
        )
        conv_kwargs = dict(bias=True, init_scheme=init_scheme)

        in_out_reps = list(zip(reps[:-1], reps[1:]))

        local_cond_encoder = None
        if local_cond_rep is not None:
            first_out_rep = in_out_reps[0][1]
            local_cond_encoder = torch.nn.ModuleList(
                [
                    eConditionalResidualBlock1D(
                        in_rep=local_cond_rep,
                        out_rep=first_out_rep,
                        cond_rep=cond_rep,
                        **block_kwargs,
                    ),
                    eConditionalResidualBlock1D(
                        in_rep=local_cond_rep,
                        out_rep=first_out_rep,
                        cond_rep=cond_rep,
                        **block_kwargs,
                    ),
                ]
            )
        self.local_cond_encoder = local_cond_encoder

        mid_rep = reps[-1]
        self.mid_modules = torch.nn.ModuleList(
            [
                eConditionalResidualBlock1D(
                    mid_rep,
                    mid_rep,
                    cond_rep=cond_rep,
                    **block_kwargs,
                ),
                eConditionalResidualBlock1D(
                    mid_rep,
                    mid_rep,
                    cond_rep=cond_rep,
                    **block_kwargs,
                ),
            ]
        )

        down_modules = torch.nn.ModuleList()
        for idx, (rep_in, rep_out) in enumerate(in_out_reps):
            is_last = idx == len(in_out_reps) - 1
            down_modules.append(
                torch.nn.ModuleList(
                    [
                        eConditionalResidualBlock1D(
                            rep_in,
                            rep_out,
                            cond_rep=cond_rep,
                            **block_kwargs,
                        ),
                        eConditionalResidualBlock1D(
                            rep_out,
                            rep_out,
                            cond_rep=cond_rep,
                            **block_kwargs,
                        ),
                        eConv1d(
                            rep_out,
                            rep_out,
                            kernel_size=3,
                            stride=2 if not is_last and downsample == "stride" else 1,
                            padding=1,
                            **conv_kwargs,
                        )
                        if (downsample == "stride" and not is_last)
                        else (
                            torch.nn.MaxPool1d(kernel_size=2, stride=2)
                            if downsample == "pooling" and not is_last
                            else torch.nn.Identity()
                        ),
                    ]
                )
            )
        self.down_modules = down_modules

        up_modules = torch.nn.ModuleList()
        current_rep = in_out_reps[-1][1]  # deepest representation
        up_pairs = list(reversed(in_out_reps[1:]))  # skip shallowest pair
        for idx, (target_rep, skip_rep) in enumerate((pair[0], pair[1]) for pair in up_pairs):
            is_last = idx == len(up_pairs) - 1
            upsample_in_rep = direct_sum([current_rep, skip_rep])  # concat current + skip
            out_rep = target_rep
            up_modules.append(
                torch.nn.ModuleList(
                    [
                        eConditionalResidualBlock1D(
                            upsample_in_rep,
                            out_rep,
                            cond_rep=cond_rep,
                            **block_kwargs,
                        ),
                        eConditionalResidualBlock1D(
                            out_rep,
                            out_rep,
                            cond_rep=cond_rep,
                            **block_kwargs,
                        ),
                        eConvTranspose1d(
                            out_rep,
                            out_rep,
                            kernel_size=3,
                            stride=2 if not is_last and downsample == "stride" else 1,
                            padding=1,
                            **conv_kwargs,
                        )
                        if (downsample == "stride" and not is_last)
                        else (
                            torch.nn.Upsample(scale_factor=2, mode="nearest")
                            if downsample == "pooling" and not is_last
                            else torch.nn.Identity()
                        ),
                    ]
                )
            )
            current_rep = out_rep
        self.up_modules = up_modules

        first_rep = in_out_reps[0][1]
        final_norm = _eChannelRMSNorm(first_rep) if normalize else torch.nn.Identity()
        self.final_conv = torch.nn.Sequential(
            eConv1d(first_rep, first_rep, kernel_size=kernel_size, padding=kernel_size // 2, **conv_kwargs),
            final_norm,
            activation,
            eConv1d(first_rep, in_rep, kernel_size=1, padding=0, **conv_kwargs),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        film_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass of the equivariant U-Net.

        Args:
            sample (torch.Tensor): Input signal shaped ``(B, in_rep.size, L)``.
            timestep (torch.Tensor | float | int): Diffusion step; scalar or batch, broadcast to ``B``.
            local_cond (torch.Tensor | None, optional): Local conditioning signal shaped
                ``(B, local_cond_rep.size, L)`` when provided.
            film_cond (torch.Tensor | None, optional): Global conditioning vector shaped
                ``(B, global_cond_rep.size)`` to drive FiLM.

        Returns:
            torch.Tensor: Output tensor shaped ``(B, in_rep.size, L)``.
        """
        assert sample.shape[1] == self.in_rep.size, f"Expected channels {self.in_rep.size}, got {sample.shape}"
        device = sample.device

        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([timestep], dtype=torch.long, device=device)
        elif t.ndim == 0:
            t = t[None].to(device)
        t = t.expand(sample.shape[0])

        film_diff = self.diffusion_step_encoder(t)
        film_feature = torch.cat([film_cond, film_diff], dim=-1) if film_cond is not None else film_diff
        assert film_feature.shape[-1] == self.cond_rep.size

        h_local = []
        if self.local_cond_encoder is not None and local_cond is not None:
            resnet_a, resnet_b = self.local_cond_encoder
            x_loc = resnet_a(local_cond, film_feature)
            h_local.append(x_loc)
            x_loc = resnet_b(local_cond, film_feature)
            h_local.append(x_loc)

        x = sample
        skips = []
        for idx, (res1, res2, down) in enumerate(self.down_modules):
            x = res1(x, film_feature)
            if idx == 0 and h_local:
                x = x + h_local[0]
            x = res2(x, film_feature)
            skips.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, film_feature)

        for idx, (res1, res2, up) in enumerate(self.up_modules):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = res1(x, film_feature)
            if idx == len(self.up_modules) - 1 and h_local:
                x = x + h_local[1]
            x = res2(x, film_feature)
            x = up(x)

        x = self.final_conv(x)
        return x

    @torch.no_grad()
    def check_equivariance(self, batch_size=3, length=5, atol=1e-5, rtol=1e-5):  # noqa: D102
        """Check equivariance under channel actions of the underlying fiber group."""
        G = self.in_rep.group
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        x = torch.randn(batch_size, self.in_rep.size, length, device=device, dtype=dtype)
        local_cond = (
            torch.randn(batch_size, self.local_cond_rep.size, length, device=device, dtype=dtype)
            if self.local_cond_rep is not None
            else None
        )
        global_cond = (
            torch.randn(batch_size, self.global_cond_rep.size, device=device, dtype=dtype)
            if self.global_cond_rep is not None
            else None
        )
        t = torch.tensor(0, dtype=torch.long, device=device)

        y = self(x, timestep=t, local_cond=local_cond, film_cond=global_cond)

        for _ in range(10):
            g = G.sample()
            rho_in = torch.tensor(self.in_rep(g), dtype=dtype, device=device)
            gx = torch.einsum("ij,bjl->bil", rho_in, x)

            g_local = None
            if local_cond is not None:
                rho_local = torch.tensor(self.local_cond_rep(g), dtype=dtype, device=device)
                g_local = torch.einsum("ij,bjl->bil", rho_local, local_cond)

            g_global = None
            if global_cond is not None:
                rho_global = torch.tensor(self.global_cond_rep(g), dtype=dtype, device=device)
                g_global = torch.einsum("ij,bj->bi", rho_global, global_cond)

            y_expected = self(gx, timestep=t, local_cond=g_local, film_cond=g_global)

            rho_out = torch.tensor(self.out_rep(g), dtype=y.dtype, device=y.device)
            gy = torch.einsum("ij,bjl->bil", rho_out, y)
            assert torch.allclose(gy, y_expected, atol=atol, rtol=rtol), (
                f"Equivariance failed for group element {g} with max error {(gy - y_expected).abs().max().item():.3e}"
            )


if __name__ == "__main__":
    # Example usage
    from escnn.group import CyclicGroup

    diffusion_step_embed_dim = 32

    G = CyclicGroup(2)
    mx, my = 1, 2
    in_rep = direct_sum([G.regular_representation] * mx)
    out_rep = direct_sum([G.regular_representation] * my)
    cond_rep = direct_sum([G.regular_representation] * 2)

    print("Testing eConditionalResidualBlock1D ------------------------------------------")
    res_block = eConditionalResidualBlock1D(
        in_rep=in_rep,
        out_rep=out_rep,
        cond_rep=cond_rep,
        kernel_size=3,
        cond_predict_scale=True,
    )
    res_block.check_equivariance(atol=1e-5, rtol=1e-5)

    print("\nTesting eConditionalUnet1D ----------------------------------------------------")
    model = eConditionalUnet1D(
        in_rep=in_rep,
        local_cond_rep=None,
        global_cond_rep=cond_rep,
        diffusion_step_embed_dim=16,
        down_dims=[64, 128, 256],
        kernel_size=3,
        cond_predict_scale=True,
    )

    model.check_equivariance(atol=1e-5, rtol=1e-5)
