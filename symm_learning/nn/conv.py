from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


class eConv1D(EquivariantModule):
    r"""One-dimensional **G-equivariant** convolution.

    This layer applies a standard 1D convolution (see
    [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html))
    to geometric tensors by ensuring the convolution kernl $K$ of shape
    `(out_type.size, in_type.size, kernel_size)` is constrained to be constructed
    from interwiners between the input and output representations, such that
    :math:`K[:, :, i] \in \text{Hom}_{\mathbb{G}}(\mathcal{V}_{\text{in}}), \mathcal{V}_{\text{out}})`

    For the usual convolution hyper-parameters (stride, padding,
    dilation, etc.) this class follows exactly the semantics of
    :class:`torch.nn.Conv1d`; please refer to the PyTorch docs for
    details.

    Parameters
    ----------
    in_type : :class:`escnn.nn.FieldType`
        Field type of the input tensor. Must have :class:`GSpace1D` as its `gspace`.
        Input tensors should be of shape `(batch_dim, in_type.size, H)`, where `H`
        is the 1D/time dimension.
    out_type : :class:`escnn.nn.FieldType`
        Field type of the output tensor. Must have the same `gspace` as `in_type`.
        Output tensors will be of shape `(batch_dim, out_type.size, H_out)`,
    kernel_size : int, default=3
        Temporal receptive field :math:`h`.
    stride : int, default=1
    padding : int, default=0
    dilation : int, default=1
    bias : bool, default=True
    padding_mode : str, default="zeros"
        Passed through to :func:`torch.nn.functional.conv1d`.
    basisexpansion : Literal["blocks"], default="blocks"
        Basis-construction strategy. Currently only ``"blocks"`` (ESCNN's
        block-matrix algorithm) is implemented.
    recompute : bool, default=False
        Whether to rebuild the kernel basis at every forward pass
        (useful for debugging; slow).
    initialize : bool, default=True
        If *True*, the free parameters are initialised with the
        generalised He scheme implemented in ``escnn.nn.init``.
    device : torch.device, optional
    dtype : torch.dtype, optional

    Shape
    -----
    * **Input:** ``(B, in_type.size, H)``
    * **Output:** ``(B, out_type.size, H_out)``, where `H_out` is computed as in
        [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).

    Examples:
    --------
    >>> from escnn.group import DihedralGroup
    >>> from escnn.nn import FieldType
    >>> from symm_learning.nn import eConv1D, GSpace1D
    >>> G = DihedralGroup(10)
    >>> # Custom (hacky) 1D G-space needed to use `GeometricTensor`
    >>> gspace = GSpace1D(G)  # Note G does not act on points in the 1D space.
    >>> in_type = FieldType(gspace, [G.regular_representation])
    >>> out_type = FieldType(gspace, [G.regular_representation] * 2)
    >>> H, kernel_size, batch_size = 10, 3, 5
    >>> # Inputs to Conv1D/eConv1D are of shape (B, in_type.size, T) where B is the batch size, C is the number of channels and T is the time dimension.
    >>> x = in_type(torch.randn(batch_size, in_type.size, H))
    >>> # Instance of eConv1D
    >>> conv_layer = eConv1D(in_type, out_type, kernel_size=3, stride=1, padding=0, bias=True)
    >>> # Forward pass
    >>> y = conv_layer(x)  # (B, out_type.size, H_out)
    >>> # After training you can export this `EquivariantModule` to a `torch.nn.Module` by:
    >>> conv1D = conv_layer.export()
    torch.Size([8, 2*G.order(), 100])

    See Also:
    --------
    * :class:`torch.nn.Conv1d` - reference implementation for ordinary
      convolutions (definitions of *stride*, *padding*, *dilation*, …).
    """  # noqa: E501

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

        # Compute the basis of equivariant kernels. Given that the convolution kernel if of shape:
        # K: (out_channels, in_channels, kernel_size), we exploit the fact that each K[:, :, i] is
        # constrained to be a group homophorphism between the input and output representations.
        # Hence we use escnn to compute the basis of homomorphisms and use this for each i.
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
                self.bias = False  # Bias only lives in the Invariant isotypic subspace.
                log.info(
                    f"Conv1D layer {self} initiated with bias=True, but the output type {self.out_type} is centered"
                    "by construction. Setting bias=False."
                )
            else:
                inv_subspace_rep = rep_out.attributes["isotypic_reps"][G.trivial_representation.id]
                # Free DoF of the bias vector.
                self.bias_weights = torch.nn.Parameter(
                    torch.zeros(inv_subspace_rep.size, dtype=torch.float32), requires_grad=True
                )
                inv_subspace_dims = rep_out.attributes["isotypic_subspace_dims"][G.trivial_representation.id]
                # Compute the map from the free DoF of the bias vector to the full bias vector.
                Q_invariant = torch.tensor(rep_out.change_of_basis[:, inv_subspace_dims], dtype=torch.float32)
                assert Q_invariant.shape == (self.out_type.size, inv_subspace_rep.size)
                self.register_buffer("Q_invariant", Q_invariant)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """Forward pass of the 1D convolution layer."""
        assert input.type == self.in_type, "Input type does not match the layer's input type"
        assert len(input.shape) == 3, "Input tensor must be 3D (batch, channels, time)"

        # Shape: (out_channels, in_channels, kernel_size)
        kernel = self.expand_kernel()
        bias = self.expand_bias() if self.bias else None

        x = input.tensor
        y = F.conv1d(
            input=x,
            weight=kernel,
            bias=bias,
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

    def expand_bias(self) -> torch.Tensor:
        """Expand the bias vector to the full bias vector."""
        if not self.bias:
            raise ValueError("Bias is not enabled for this layer.")
        # Shape: (out_channels,)
        return self.Q_invariant @ self.bias_weights

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

    def extra_repr(self):  # noqa: D102
        H = 100
        _, _, H_out = self.evaluate_output_shape((1, self.in_type.size, H))
        diff = H - H_out
        return (
            f"{self.in_type.fibergroup}-equivariant Conv1D layer\n"
            f"in_type={self.in_type}, in_shape=(B, {self.in_type.size}, H) \n"
            f"out_type={self.out_type}, out_shape=(B, {self.out_type.size}, H - {diff:d}) \n"
            f"kernel_size={self.kernel_size} stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, bias={self.bias}, "
        )

    def export(self) -> torch.nn.Conv1d:
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

        conv1D.weight.data = self.expand_kernel().data
        if self.bias:
            conv1D.bias.data = self.expand_bias()
        conv1D.eval()

        return conv1D


class GSpace1D(escnn.gspaces.GSpace):
    """Hacky solution to use GeometricTensor with time as a homogenous space."""

    def __init__(self, fibergroup: escnn.group.Group, name: str = "GSpace1D"):
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
