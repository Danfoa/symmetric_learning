from __future__ import annotations

import logging
from math import ceil
from typing import Union

import escnn
import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor, IdentityModule

from symm_learning.models import EMLP, IMLP
from symm_learning.models.difussion.cond_unet1d import SinusoidalPosEmb
from symm_learning.nn import Mish, eBatchNorm1d, eConv1D, eConvTranspose1D
from symm_learning.representation_theory import isotypic_decomp_rep

logger = logging.getLogger(__name__)


class eConditionalResidualBlock1D(EquivariantModule):
    """Equivariant conditional residual block for symmetric signals in time/sequence.

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

    Args:
        in_type (FieldType): The type of the input field.
        out_type (FieldType): The type of the output field.
        cond_type (FieldType): The type of the conditioning field, which must be
            invariant.
        kernel_size (int, optional): The size of the convolutional kernel.
            Defaults to 3.
        cond_predict_scale (bool, optional): Whether to predict the scale
            parameter in the FiLM layer. If False, only bias is used.
            Defaults to False.
    """

    def __init__(
        self, in_type: FieldType, out_type: FieldType, cond_type: FieldType, kernel_size=3, cond_predict_scale=False
    ):
        super().__init__()
        assert in_type.gspace == out_type.gspace, "Input and output types must have the same G-space"

        self.in_type = in_type
        self.out_type = out_type
        self.cond_type = cond_type
        self.film_with_scale = cond_predict_scale

        self.conv1 = self._conv_block(in_type, out_type, kernel_size)
        self.conv2 = self._conv_block(out_type, out_type, kernel_size)

        # Equivariant version of the FiLM modulation https://arxiv.org/abs/1709.07871. ================================
        # Similar to the concept presented in vector neurons, the FiLM modulation applies an affine transformation to
        # the output of the first convolution block (batch, C1_out, horizon), computed from the conditioning tensor.
        # (batch, cond_dim). Given that we aim to preserve equivariance, this affine transformation has to be
        # equivariant, meaning that the scale and bias vectors are symmetry constrained.
        self.rep_out = isotypic_decomp_rep(out_type.representation)
        G = self.rep_out.group
        self.film_with_bias = G.trivial_representation.id in self.rep_out.attributes["isotypic_reps"]
        self.n_bias_params, self.n_scale_params = 0, 0
        if self.film_with_bias:  # Bias DoF config __________________________________________________________________
            inv_subspace_dims: slice = self.rep_out.attributes["isotypic_subspace_dims"][G.trivial_representation.id]
            self.n_bias_params = inv_subspace_dims.stop - inv_subspace_dims.start
            inv_projector = torch.tensor(self.rep_out.change_of_basis[:, inv_subspace_dims]).to(
                dtype=torch.get_default_dtype()
            )
            self.register_buffer("inv_projector", inv_projector)

        if self.film_with_scale:  # Scale DoF config __________________________________________________________________
            self.n_scale_params = len(self.rep_out.irreps)
            # Configure multiplicities of scale parameters for each irrep subspace.
            irrep_dims = torch.tensor([G.irrep(*irrep_id).size for irrep_id in self.rep_out.irreps])
            self.register_buffer("irrep_indices", torch.repeat_interleave(torch.arange(len(irrep_dims)), irrep_dims))
            # Change of basis to irrep spectral basis.
            dtype = torch.get_default_dtype()
            self.register_buffer("Q", torch.tensor(self.rep_out.change_of_basis, dtype=dtype))
            self.register_buffer("Q_inv", torch.tensor(self.rep_out.change_of_basis_inv, dtype=dtype))

        # Invariant NN parameterizing the DoF of the scale and bias of the affine transformation. ======================
        self.cond_encoder = IMLP(
            in_type=cond_type,
            out_dim=self.n_scale_params + self.n_bias_params,
            hidden_units=[cond_type.size],
            activation="Mish",
            bias=True,
        )

        # make sure dimensions compatible
        self.residual_conv = (
            eConv1D(in_type, out_type, kernel_size=1) if in_type != out_type else escnn.nn.IdentityModule(out_type)
        )

    def forward(self, x: GeometricTensor, cond: GeometricTensor) -> GeometricTensor:
        """Forward pass through the block.

        Args:
            x (GeometricTensor): The input tensor.
            cond (GeometricTensor): The global conditioning tensor (Assumed invariant to group action).

        Returns:
            GeometricTensor
        """
        assert cond.type == self.cond_type, f"Expected conditioning type {self.cond_type}, got {cond.type}"
        assert x.type == self.in_type, f"Expected input type {self.in_type}, got {x.type}"

        # First convolution block
        out = self.conv1(x)
        # Compute the conditioning which will modulate linearly the first convolution output
        dof = self.cond_encoder(cond)
        assert dof.shape[1] == self.n_scale_params + self.n_bias_params
        film_scale_dof, film_bias_dof = dof.tensor[:, : self.n_scale_params], dof.tensor[:, self.n_scale_params :]

        if self.film_with_scale:
            # Reshape film_scale for proper broadcasting: (B, C) -> (B, C, 1)
            film_scale_spectral = film_scale_dof[:, self.irrep_indices, None]
            # This is computationally expensive, but for now easiest way to implement equivariant scaling.
            out_spectral = torch.einsum("ij,bj...->bi...", self.Q_inv, out.tensor)
            out_spectral_scaled = out_spectral * film_scale_spectral
            out_scaled = torch.einsum("ij,bj...->bi...", self.Q, out_spectral_scaled)
            out = self.out_type(out_scaled)
        if self.film_with_bias:
            bias = torch.einsum("ij,bj->bi", self.inv_projector, film_bias_dof)
            out_biased = out.tensor + bias[:, :, None]  # Broadcasting bias to match output shape
            out = self.out_type(out_biased)

        out = self.conv2(out)
        out = self.out_type(out.tensor + self.residual_conv(x).tensor)
        return out

    def evaluate_output_shape(self, input_shape):  # noqa: D102
        s1 = self.blocks[0].evaluate_output_shape(input_shape)
        s2 = self.blocks[1].evaluate_output_shape(s1)
        return s2

    @staticmethod
    def _conv_block(in_type: FieldType, out_type: FieldType, kernel_size: int):  # noqa: D102
        return escnn.nn.SequentialModule(
            # Equivariant Time-Convolution
            eConv1D(in_type, out_type, kernel_size, padding=kernel_size // 2),
            # Use eBatchNorm instead of GroupNorm to keep equivariance
            eBatchNorm1d(out_type, affine=True),
            # Use Mish activation function as in the original code
            Mish(out_type),
        )

    def check_equivariance(self, atol=1e-5, rtol=1e-5):  # noqa: D102
        import numpy as np

        B, T = 3, 5
        x = torch.randn(B, self.in_type.size, *[T] * self.in_type.gspace.dimensionality)
        x = GeometricTensor(x, self.in_type)

        cond = torch.randn(B, self.cond_type.size)
        cond = self.cond_type(cond)

        was_training = self.training
        self.eval()  # Set the module to evaluation mode to disable batchnorm statistics updates

        # for el in self.out_type.testing_elements:
        for _ in range(20):
            g = self.in_type.gspace.fibergroup.sample()
            rep_X_g = torch.tensor(self.in_type.representation(g), dtype=x.tensor.dtype)
            rep_Y_g = torch.tensor(self.out_type.representation(g), dtype=x.tensor.dtype)
            rep_cond_g = torch.tensor(self.cond_type.representation(g), dtype=cond.tensor.dtype)

            gx = self.in_type(torch.einsum("ij,bjt->bit", rep_X_g, x.tensor))
            gcond = self.cond_type(torch.einsum("ij,bj->bi", rep_cond_g, cond.tensor))

            y = self(x, cond).tensor

            gy = self(gx, gcond).tensor
            gy_gt = torch.einsum("ij,bjt->bit", rep_Y_g, y)

            errs = (gy_gt - gy).detach().numpy()
            errs = np.abs(errs).reshape(-1)

            assert torch.allclose(gy_gt, gy, atol=atol, rtol=rtol), (
                'The error found during equivariance check with element "{}" is too high: '
                "max = {}, mean = {} var ={}".format(g, errs.max(), errs.mean(), errs.var())
            )

        self.train(was_training)  # Restore the training mode if it was previously set


class eConditionalUnet1D(EquivariantModule):
    """Equivariant U-Net model for 1D signals, conditioned on diffusion timestep and other context.

    This model adapts the ConditionalU-Net for symmetric signals on time/sequence.
    The U-net is composed of equivariant down and up blocks built from residual CNNs.
    Each residual block has a FiLM (Feature-wise Linear Modulation) layer that applies
    the global conditioning information to the intermediate features. Crucially,
    this conditioning if invariant to the group action.

    Local equivariant conditioning is supported.

    Args:
        input_type (FieldType): The type of the input field.
        local_cond_type (FieldType): The type of the local conditioning field.
        global_cond_type (FieldType, optional): The type of the global
            conditioning field. Defaults to None.
        diffusion_step_embed_dim (int, optional): The dimension of the diffusion
            step embedding. Defaults to 256.
        down_dims (list, optional): A list of channel dimensions for the
            downsampling path. Defaults to [256, 512, 1024].
        kernel_size (int, optional): The size of the convolutional kernel.
            Defaults to 3.
        cond_predict_scale (bool, optional): Whether to predict the scale
            parameter in the FiLM layer. Defaults to False.
    """

    def __init__(
        self,
        input_type: FieldType,
        local_cond_type: FieldType,
        global_cond_type: FieldType = None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        # n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        self.in_type = self.out_type = input_type
        self.global_cond_type = global_cond_type
        self.local_cond_type = local_cond_type

        all_dims = [input_type.size] + list(down_dims)
        reg_rep = input_type.fibergroup.regular_representation
        trivial_rep = input_type.gspace.fibergroup.trivial_representation

        all_types = [input_type] + [
            FieldType(input_type.gspace, representations=[reg_rep] * ceil(dim / reg_rep.size)) for dim in down_dims
        ]

        # start_dim = down_dims[0]
        start_type = all_types[1]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = torch.nn.Sequential(
            SinusoidalPosEmb(dsed),
            torch.nn.Linear(dsed, dsed * 4),
            torch.nn.Mish(),
            torch.nn.Linear(dsed * 4, dsed),
        )
        print(f"Diffusion encoder {sum(p.numel() for p in diffusion_step_encoder.parameters()) / 1000} [k] params")
        self.dsed = dsed

        cond_dim = dsed
        if global_cond_type is not None:
            cond_dim += global_cond_type.size
        # Global conditioning is assumed to be invariant to group actions.
        gs_global = escnn.gspaces.no_base_space(input_type.fibergroup)
        self.cond_type = FieldType(gs_global, representations=[trivial_rep] * cond_dim)

        # all_dims = [64, 256, 512, 1024, 2048] -> in_out = [(64, 256), (256, 512), (512, 1024), (1024, 2048)]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        in_out_types = list(zip(all_types[:-1], all_types[1:]))

        local_cond_encoder = None
        if local_cond_type is not None:
            # _, dim_out = in_out[0]  # dim_out = 256 given comments up
            _, out_type = in_out_types[0]

            # dim_in = local_cond_type.size
            in_type = local_cond_type

            local_cond_encoder = torch.nn.ModuleList(
                [
                    # down encoder -> Local context added to the start of the U-Net
                    eConditionalResidualBlock1D(
                        in_type=in_type,
                        out_type=out_type,
                        # Global invariant conditioning. TODO: This means we can use iMLP instead of a MLP.
                        cond_type=self.cond_type,
                        kernel_size=kernel_size,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    # up encoder -> Local context added to the end of the U-Net
                    eConditionalResidualBlock1D(
                        in_type=in_type,
                        out_type=out_type,
                        # Global invariant conditioning. TODO: This means we can use iMLP instead of a MLP.
                        cond_type=self.cond_type,
                        kernel_size=kernel_size,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )
            print(f"local encoder {sum(p.numel() for p in local_cond_encoder.parameters()) / 1000} [k] params")
        # mid_dim = all_dims[-1]
        mid_type = all_types[-1]  # Core dimension of the U-Net (lowest time resolution, highest feature dimension)
        self.mid_modules = torch.nn.ModuleList(
            [
                eConditionalResidualBlock1D(
                    mid_type,
                    mid_type,
                    cond_type=self.cond_type,
                    kernel_size=kernel_size,
                    cond_predict_scale=cond_predict_scale,
                ),
                eConditionalResidualBlock1D(
                    mid_type,
                    mid_type,
                    cond_type=self.cond_type,
                    kernel_size=kernel_size,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )
        print(f"Mid CNN {sum(p.numel() for p in self.mid_modules.parameters()) / 1000} [k] params")

        down_modules = torch.nn.ModuleList([])
        for ind, (in_type, out_type) in enumerate(in_out_types):
            is_last = ind >= (len(in_out) - 1)
            id = IdentityModule(out_type)
            down_modules.append(
                torch.nn.ModuleList(  # Level blocks: ResNet Block 1 + ResNet Block 2 + Downsampling Conv
                    [
                        eConditionalResidualBlock1D(  # ResNet Block 1
                            in_type=in_type,
                            out_type=out_type,
                            cond_type=self.cond_type,
                            kernel_size=kernel_size,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        eConditionalResidualBlock1D(  # ResNet Block 2
                            in_type=out_type,
                            out_type=out_type,
                            cond_type=self.cond_type,
                            kernel_size=kernel_size,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        # Add downsampling conv layer
                        eConv1D(out_type, out_type, kernel_size=3, stride=2, padding=1) if not is_last else id,
                    ]
                )
            )
        print(f"Down CNN {sum(p.numel() for p in down_modules.parameters()) / 1000} [k] params")

        up_modules = torch.nn.ModuleList([])
        for ind, (in_type, out_type) in enumerate(reversed(in_out_types[1:])):
            is_last = ind >= (len(in_out) - 1)
            # Given the skip connections, we double the input_type
            upsample_in_type = FieldType(out_type.gspace, representations=out_type.representations * 2)
            id = IdentityModule(in_type)
            up_modules.append(
                torch.nn.ModuleList(
                    [
                        eConditionalResidualBlock1D(  # ResNet Block 1
                            in_type=upsample_in_type,
                            out_type=in_type,
                            cond_type=self.cond_type,
                            kernel_size=kernel_size,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        eConditionalResidualBlock1D(  # ResNet Block 2
                            in_type=in_type,
                            out_type=in_type,
                            cond_type=self.cond_type,
                            kernel_size=kernel_size,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        eConvTranspose1D(in_type, in_type, kernel_size=3, stride=2, padding=1) if not is_last else id,
                    ]
                )
            )
        print(f"Down CNN {sum(p.numel() for p in down_modules.parameters()) / 1000} [k] params")

        final_conv = escnn.nn.SequentialModule(
            eConditionalResidualBlock1D._conv_block(start_type, start_type, kernel_size=kernel_size),
            # Equivalent to a linear layer applied timewise
            eConv1D(in_type=start_type, out_type=self.in_type, kernel_size=1),
        )

        print(f"Final CNN {sum(p.numel() for p in final_conv.parameters()) / 1000} [k] params")

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        sample: GeometricTensor,
        timestep: torch.Tensor | float | int,
        local_cond: GeometricTensor = None,
        global_cond: GeometricTensor = None,
    ):
        """Forward pass of the eConditionalUnet1D model.

        Args:
            sample (GeometricTensor): The input tensor of shape (B, in_type.size, T).
            timestep (Union[torch.Tensor, float, int]): The diffusion timestep.
            local_cond (GeometricTensor, optional): The local conditioning tensor.
                Defaults to None.
            global_cond (GeometricTensor, optional): The global conditioning tensor.
                Defaults to None.

        Returns:
            GeometricTensor: The output tensor of the U-Net.
        """
        assert sample.type == self.in_type, f"Expected input type {self.in_type}, got {sample.type}"
        device = sample.tensor.device
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # raise ValueError("timestep must be a tensor, float, or int.")
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature_time = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature_time, global_cond.tensor], axis=-1)

        global_feature = self.cond_type(global_feature)

        # encode local features
        h_local = list()
        if local_cond is not None:
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)  # Local context added to the start of the U-Net
            x = resnet2(local_cond, global_feature)
            h_local.append(x)  # Local context added to the end of the U-Net

        x = sample
        h = []  # Intermediate features for skip connections
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]  # Local context added to the start of the U-Net
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = resnet.in_type(torch.cat((x.tensor, h.pop().tensor), dim=1))  # Concatenate skip connection features
            x = resnet(x, global_feature)
            if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
                x = x + h_local[1]  # Local context added to the end of the U-Net
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        return x

    def check_equivariance(self, atol=1e-5, rtol=1e-5):  # noqa: D102
        """Test the equivariance of the module.

        This method tests the equivariance of the module by applying a random
        group element to the input and checking if the output transforms
        correctly.

        Args:
            atol (float, optional): The absolute tolerance for the check.
                Defaults to 1e-5.
            rtol (float, optional): The relative tolerance for the check.
                Defaults to 1e-5.

        Raises:
            AssertionError: If the equivariance check fails.
        """
        import numpy as np

        was_training = self.training
        self.eval()  # Set the module to evaluation mode to disable batchnorm statistics updates

        B, T = 3, 5
        x = torch.randn(B, self.in_type.size, *[T] * self.in_type.gspace.dimensionality)
        x = GeometricTensor(x, self.in_type)

        if self.local_cond_encoder is not None:
            cond = torch.randn(B, self.cond_type.size)
            cond = self.cond_type(cond)
        else:
            cond = None

        global_cond = torch.randn(B, self.global_cond_type.size)
        global_cond = self.global_cond_type(global_cond)

        t = torch.tensor(0, dtype=torch.long, device=x.tensor.device)  # Assuming a single timestep for testing

        for _ in range(20):
            g = self.in_type.gspace.fibergroup.sample()

            g_cond = cond.transform(g) if self.local_cond_encoder is not None else None
            g_local_cond = global_cond.transform(g)
            g_x = x.transform(g)

            y = self(sample=x, timestep=t, local_cond=g_cond, global_cond=g_local_cond)
            gy = self(sample=g_x, timestep=t, local_cond=g_cond, global_cond=g_local_cond).tensor
            gy_gt = y.transform(g).tensor

            errs = (gy_gt - gy).detach().numpy()
            errs = np.abs(errs).reshape(-1)

            t = t + 1
            assert torch.allclose(gy_gt, gy, atol=atol, rtol=rtol), (
                'The error found during equivariance check with element "{}" is too high: '
                "max = {}, mean = {} var ={}".format(g, errs.max(), errs.mean(), errs.var())
            )

        self.train(was_training)  # Restore the training mode if it was previously set

    def evaluate_output_shape(self, input_shape):  # noqa: D102
        raise NotImplementedError("Fuck off")


if __name__ == "__main__":
    # Example usage
    import escnn

    from symm_learning.nn import GSpace1D

    diffusion_step_embed_dim = 32

    G = escnn.group.CyclicGroup(2)
    gspace = GSpace1D(G)
    mx, my, mc = 1, 1, 1
    in_type = FieldType(gspace, [G.regular_representation] * mx)
    out_type = FieldType(gspace, [G.regular_representation] * my)
    cond_type = FieldType(escnn.gspaces.no_base_space(G), [G.regular_representation] * mc)
    global_cond_type = FieldType(
        escnn.gspaces.no_base_space(G), [G.trivial_representation] * (mc + diffusion_step_embed_dim)
    )

    model = eConditionalUnet1D(
        input_type=in_type,
        # local_cond_type=out_type,
        local_cond_type=None,
        global_cond_type=global_cond_type,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=[64, 64],
        kernel_size=3,
        cond_predict_scale=True,
    )

    model.check_equivariance(atol=1e-5, rtol=1e-5)
