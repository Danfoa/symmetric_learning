import escnn
import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from symm_learning.models import EMLP, IMLP
from symm_learning.nn import Mish, eBatchNorm1d, eConv1D, eConvTranspose1D
from symm_learning.representation_theory import isotypic_decomp_rep


class eConditionalResidualBlock1D(EquivariantModule):
    """Equivariant conditional residual block for 1D signals.

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
            cond (GeometricTensor): The conditioning tensor.

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


# if __name__ == "__main__":
#     # Example usage
#     import escnn

#     from symm_learning.nn import GSpace1D

#     G = escnn.group.CyclicGroup(5)
#     gspace = GSpace1D(G)
#     mx, my, mc = 1, 2, 5
#     in_type = FieldType(gspace, [G.regular_representation] * mx)
#     out_type = FieldType(gspace, [G.regular_representation] * my)
#     cond_type = FieldType(escnn.gspaces.no_base_space(G), [G.regular_representation] * mc)

#     block = eConditionalResidualBlock1D(in_type, out_type, cond_type, kernel_size=3, cond_predict_scale=True)
#     print(block)
#     block.check_equivariance()
