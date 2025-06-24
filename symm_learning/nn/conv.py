from __future__ import annotations

from typing import Literal

import escnn
import escnn.nn.init
import numpy as np
import torch
import torch.nn.functional as F
from escnn.gspaces import no_base_space
from escnn.nn import EquivariantModule, FieldType, GeometricTensor, Linear
from escnn.nn.modules.basismanager import BasisManager, BlocksBasisExpansion

from symm_learning.representation_theory import isotypic_decomp_rep


class eConv1D(EquivariantModule):
    """1D Equivariant convolution layer."""

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        kernel_size: int = 3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        # ESCNN-specific parameters
        basisexpansion: Literal["blocks"] = "blocks",
        recompute: bool = False,
        initialize: bool = True,
        # PyTorch-specific parameters
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert in_type.gspace == out_type.gspace and isinstance(in_type.gspace, GSpace1D)
        # Assert that hyperparameters are valid for a 1D convolution layer
        assert isinstance(kernel_size, int) and kernel_size > 0, "kernel_size must be a positive integer"
        assert isinstance(stride, int) and stride > 0, "stride must be a positive integer"
        assert isinstance(padding, int) and padding >= 0, "padding must be a non-negative integer"
        assert isinstance(dilation, int) and dilation > 0, "dilation must be a positive integer"
        self.in_type = in_type
        self.out_type = out_type

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.bias = bias
        self.basisexpansion_type = basisexpansion
        self.device, self.dtype = device, dtype
        if self.basisexpansion_type == "blocks":  # Inefficient but easy implementation as reuses ESCNN code
            space = no_base_space(in_type.fibergroup)
            self._basisexpansion = BlocksBasisExpansion(
                in_type.representations,
                out_type.representations,
                basis_generator=space.build_fiber_intertwiner_basis,
                points=np.zeros((1, 1)),  # Not used
                recompute=recompute,
            )
            # Free parameters are `kernel_size` * `dim(End_G(in_space, out_space))`
            print("Intertwiner basis dimension:", self._basisexpansion.dimension())
            self.weights = torch.nn.Parameter(
                torch.zeros(self._basisexpansion.dimension() * kernel_size), requires_grad=True
            ).to(
                device=device,
                dtype=dtype,
            )

            filter_size = (out_type.size, in_type.size, kernel_size)
            self.register_buffer("kernel", torch.zeros(*filter_size))
            if initialize:
                # by default, the weights are initialized with a generalized form of He's weight initialization
                for i in range(kernel_size):
                    escnn.nn.init.generalized_he_init(
                        self.weights.data[i * self._dim_interwiner_basis : (i + 1) * self._dim_interwiner_basis],
                        self._basisexpansion,
                    )

        else:
            raise ValueError('Basis Expansion algorithm "%s" not recognized' % basisexpansion)

        if self.bias:
            rep_out = isotypic_decomp_rep(self.out_type.representation)
            G = rep_out.group
            has_trivial_irrep = G.trivial_representation.id in rep_out.attributes["isotypic_reps"]

            if not has_trivial_irrep:
                self.bias = False

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Forward pass of the 1D convolution layer."""
        assert input.type == self.in_type, "Input type does not match the layer's input type"
        assert len(input.shape) == 3, "Input tensor must be 3D (batch, channels, time)"

        # Shape: (out_channels, in_channels, kernel_size)
        kernel = self.expand_kernel()

        x = input.tensor
        y = F.conv1d(
            input=x,
            weight=kernel,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,  # No groups supported.
        )
        return self.out_type(y)

    def expand_kernel(self) -> torch.Tensor:
        """Kernel of the convolution layer of shape (out_channels, in_channels, kernel_size)."""
        kernel = []
        for i in range(self.kernel_size):
            # Extract the weights for the current kernel
            kernel.append(
                self._basisexpansion(
                    self.weights[i * self._dim_interwiner_basis : (i + 1) * self._dim_interwiner_basis]
                )
            )
        self.kernel.data = torch.cat(kernel, dim=-1)
        return self.kernel

    def evaluate_output_shape(self, input_shape) -> tuple[int, ...]:
        """Calculate the output shape of the convolution layer."""
        b, _, t = input_shape
        return (b, self.out_type.size, self.dim_after_conv(t))

    def dim_after_conv(self, input_dim: int) -> int:
        """Calculate the output dimension after the convolution."""
        return (input_dim + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

    def check_equivariance(self, atol=1e-5, rtol=1e-5):
        """Check the equivariance of the convolution layer."""
        c = self.in_type.size
        B, H = 10, 50
        x = torch.randn(B, c, H)
        x = GeometricTensor(x, self.in_type)

        errors = []

        # for el in self.out_type.testing_elements:
        rep_Y = self.out_type.representation
        for _ in range(20):
            g = self.in_type.gspace.fibergroup.sample()

            gx = x.transform(g)
            y = self(x).tensor.detach().numpy()
            gy = self(gx).tensor.detach().numpy()

            gy_gt = np.einsum("ij,bjt->bit", rep_Y(g), y)
            errs = gy - gy_gt
            errs = np.abs(errs).reshape(-1)

            assert np.allclose(gy, gy_gt, atol=atol, rtol=rtol), (
                'The error found during equivariance check with element "{}" is too high: '
                "max = {}, mean = {} var ={}".format(g, errs.max(), errs.mean(), errs.var())
            )

            errors.append((g, errs.mean()))

        return errors

    @property
    def _dim_interwiner_basis(self):
        """Dimension of the fiber intertwiner basis."""
        return self._basisexpansion.dimension()

    def export(self) -> torch.nn.Module:
        """Exports the module to a standard PyTorch module."""
        conv1D = torch.nn.Conv1d(
            in_channels=self.in_type.size,
            out_channels=self.out_type.size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.bias,
            padding_mode=self.padding_mode,
        ).to(device=self.device, dtype=self.dtype)

        conv1D.weight.data = self.kernel.data

        if self.bias:
            raise NotImplementedError("Bias export not implemented yet")


class GSpace1D(escnn.gspaces.GSpace):
    """Hacky solution to use GeometricTensor with time as a homogenous space."""

    def __init__(self, fibergroup: escnn.group.Group, name: str = "Time"):
        super().__init__(fibergroup=fibergroup, name=name, dimensionality=1)

    def _basis_generator(self, in_repr, out_repr, **kwargs):
        raise NotImplementedError("Sines and cosines")

    @property
    def basespace_action(self) -> escnn.group.Representation:  # noqa: D102
        return self.fibergroup.trivial_representation

    def restrict(self, id):  # noqa: D102
        raise NotImplementedError("Cannot restrict a 1D GSpace")


if __name__ == "__main__":
    # Example usage
    from escnn.group import DihedralGroup

    G = DihedralGroup(10)
    gspace = GSpace1D(G)
    in_type = FieldType(gspace, [G.regular_representation])
    out_type = FieldType(gspace, [G.regular_representation] * 2)

    time = 10
    kernel_size = 3
    batch_size = 30
    x = torch.randn(batch_size, in_type.size, time)
    x = in_type(x)

    conv_layer = eConv1D(in_type, out_type, kernel_size=3, stride=1, padding=0, bias=True)
    print(conv_layer)
    print("Weights shape:", conv_layer.weights.shape)
    print("Kernel shape:", conv_layer.kernel.shape)

    conv_layer.check_equivariance(atol=1e-5, rtol=1e-5)
