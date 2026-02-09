from __future__ import annotations

import logging
from typing import Literal

import torch
from escnn.group import Representation

from symm_learning.nn.linear import eINIT_SCHEMES
from symm_learning.representation_theory import GroupHomomorphismBasis

log = logging.getLogger(__name__)


class eConv1d(torch.nn.Conv1d):
    r"""Channel-equivariant 1D convolution.

    Matches :class:`torch.nn.Conv1d`—inputs ``(B, in_rep.size, L)`` to outputs ``(B, out_rep.size, L_out)``—while
    constraining each kernel slice to lie in :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\text{in}}, \rho_{\text{out}})`.

    The layer satisfies the equivariance constraint:

    .. math::
        \rho_{\text{out}}(g) \mathbf{y}_t = \mathbf{W} * (\rho_{\text{in}}(g) \mathbf{x})_t + \mathbf{b}

    where :math:`*` denotes convolution, and :math:`\mathbf{b}` is an invariant bias.

    Kernel DoF are stored as ``(kernel_size, dim(Hom_G))`` and expanded via
    :class:`~symm_learning.representation_theory.GroupHomomorphismBasis`; bias exists only if the trivial irrep appears
    in ``out_rep``.
    """

    def __init__(
        self,
        in_rep: Representation,
        out_rep: Representation,
        kernel_size: int = 3,
        basis_expansion: Literal["isotypic_expansion", "memory_heavy"] = "isotypic_expansion",
        init_scheme: eINIT_SCHEMES = "xavier_uniform",
        **conv1d_kwargs,
    ):
        r"""Initialize the constrained convolution.

        Args:
            in_rep (:class:`~escnn.group.Representation`): Channel representation :math:`\rho_{\text{in}}` describing
                input transformation.
            out_rep (:class:`~escnn.group.Representation`): Channel representation :math:`\rho_{\text{out}}` describing
                output transformation.
            kernel_size (:class:`int`, optional): Spatial kernel size. Defaults to 3.
            basis_expansion (:class:`typing.Literal`, optional): Basis realization strategy for
                :class:`~symm_learning.representation_theory.GroupHomomorphismBasis`.
            init_scheme (``eINIT_SCHEMES``, optional): Initialization passed to
                :meth:`~symm_learning.representation_theory.GroupHomomorphismBasis.initialize_params`. Defaults to
                ``"xavier_uniform"``.
            **conv1d_kwargs: Standard :class:`torch.nn.Conv1d` arguments (stride, padding, bias, etc.).
        """
        assert in_rep.group == out_rep.group, f"Incompatible group: {in_rep.group} and {out_rep.group}"

        if "groups" in conv1d_kwargs:
            assert conv1d_kwargs["groups"] == 1, "`groups`>1 are not supported in eConv1D"

        super().__init__(in_channels=in_rep.size, out_channels=out_rep.size, kernel_size=kernel_size, **conv1d_kwargs)
        dtype = conv1d_kwargs.get("dtype", torch.get_default_dtype())
        # Delete linear unconstrained module parameters
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        # Instanciate the handler of the basis of Hom_G(in_rep, out_rep)
        self.homo_basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion)
        self.in_rep, self.out_rep = self.homo_basis.in_rep, self.homo_basis.out_rep
        if self.homo_basis.dim == 0:
            raise ValueError(
                f"No equivariant linear maps exist between {in_rep} and {out_rep}.\n dim(Hom_G(in_rep, out_rep))=0"
            )
        # Weight is a tensor of shape (out_rep.size, in_rep.size, kernel_size), hence:
        self.register_parameter(
            "weight_dof",
            torch.nn.Parameter(torch.zeros(kernel_size, self.homo_basis.dim, dtype=dtype), requires_grad=True),
        )

        # Assert bias vector is feasible given out_rep symmetries
        bias = conv1d_kwargs.get("bias", True)
        trivial_id = self.homo_basis.G.trivial_representation.id
        can_have_bias = out_rep._irreps_multiplicities.get(trivial_id, 0) > 0
        self.has_bias = bias and can_have_bias
        if self.has_bias:  # Register bias parameters
            # Number of bias trainable parameters are equal to the output multiplicity of the trivial irrep
            m_out_trivial = out_rep._irreps_multiplicities[trivial_id]
            self.register_parameter(
                "bias_dof", torch.nn.Parameter(torch.zeros(m_out_trivial, dtype=dtype), requires_grad=True)
            )
            self.register_buffer("Qout", torch.tensor(self.out_rep.change_of_basis, dtype=dtype))

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        """Apply the constrained 1D convolution on channel-last dimension."""
        assert input.shape[-2] == self.in_rep.size, (
            f"Expected input of shape (..., {self.in_rep.size}, H) got {input.shape}"
        )
        return super().forward(input)

    @property
    def weight(self) -> torch.Tensor:  # noqa: D102
        """Dense kernel of shape ``(out_channels, in_channels, kernel_size)``."""
        W = self.homo_basis(self.weight_dof)  # (kernel_size, out_rep.size, in_rep.size)
        return W.permute(1, 2, 0)  # (out_rep.size, in_rep.size, kernel_size)

    @property
    def bias(self) -> torch.Tensor | None:  # noqa: D102
        """Expanded invariant bias or ``None`` when not admissible."""
        return self._expand_bias() if self.has_bias else None

    @torch.no_grad()
    def reset_parameters(self, scheme: eINIT_SCHEMES = "xavier_normal"):
        """Reset trainable parameters using the chosen initialization scheme."""
        if not hasattr(self, "homo_basis"):  # First call on torch.nn.Conv1d init
            return super().reset_parameters()
        new_params = self.homo_basis.initialize_params(scheme=scheme, leading_shape=self.kernel_size)
        self.weight_dof.copy_(new_params)

        if self.has_bias:
            trivial_id = self.out_rep.group.trivial_representation.id
            m_in_inv = self.in_rep._irreps_multiplicities[trivial_id]
            m_out_inv = self.out_rep._irreps_multiplicities[trivial_id]
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(m_out_inv, m_in_inv))
            bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_dof, -bound, bound)

    @torch.no_grad()
    def check_equivariance(self, atol=1e-5, rtol=1e-5):
        """Check equivariance under channel actions of the underlying fiber group."""
        G = self.in_rep.group
        B, L = 10, 30
        x = torch.randn(B, self.in_rep.size, L, device=self.weight_dof.device, dtype=self.weight_dof.dtype)
        y = self(x)

        for _ in range(10):
            g = G.sample()
            rho_in = torch.tensor(self.in_rep(g), dtype=x.dtype, device=x.device)
            rho_out = torch.tensor(self.out_rep(g), dtype=y.dtype, device=y.device)
            gx = torch.einsum("ij,bjl->bil", rho_in, x)
            y_expected = self(gx)
            gy = torch.einsum("ij,bjl->bil", rho_out, y)
            assert torch.allclose(gy, y_expected, atol=atol, rtol=rtol), (
                f"Equivariance failed for group element {g} with max error {(gy - y_expected).abs().max().item():.3e}"
            )

    def _expand_bias(self):
        """Expand bias degrees of freedom into the original basis."""
        trivial_id = self.out_rep.group.trivial_representation.id
        trivial_indices = self.homo_basis.iso_blocks[trivial_id]["out_slice"]
        bias = torch.mv(self.Qout[:, trivial_indices], self.bias_dof)
        return bias


class eConvTranspose1d(torch.nn.ConvTranspose1d):
    r"""Channel-equivariant transposed 1D convolution.

    Matches :class:`torch.nn.ConvTranspose1d`—inputs ``(B, in_rep.size, L)`` to outputs ``(B, out_rep.size, L_out)``—
    while constraining each kernel slice to lie in
    :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\text{in}}, \rho_{\text{out}})`.

    The layer satisfies the equivariance constraint:

    .. math::
        \rho_{\text{out}}(g) \mathbf{y}_t = \mathbf{W}^T * (\rho_{\text{in}}(g) \mathbf{x})_t + \mathbf{b}

    where :math:`*` denotes transposed convolution.

    Kernel DoF are stored as ``(kernel_size, dim(Hom_G))`` and expanded via
    :class:`~symm_learning.representation_theory.GroupHomomorphismBasis`; bias exists only if the trivial irrep appears
    in ``out_rep``.
    """

    def __init__(
        self,
        in_rep: Representation,
        out_rep: Representation,
        kernel_size: int = 3,
        basis_expansion: Literal["isotypic_expansion", "memory_heavy"] = "isotypic_expansion",
        init_scheme: eINIT_SCHEMES = "xavier_uniform",
        **conv1d_kwargs,
    ):
        r"""Initialize the constrained transposed convolution.

        Args:
            in_rep (:class:`~escnn.group.Representation`): Channel representation :math:`\rho_{\text{in}}` describing
                input transformation.
            out_rep (:class:`~escnn.group.Representation`): Channel representation :math:`\rho_{\text{out}}` describing
                output transformation.
            kernel_size (:class:`int`, optional): Spatial kernel size. Defaults to 3.
            basis_expansion (:class:`typing.Literal`, optional): Basis realization strategy for
                :class:`~symm_learning.representation_theory.GroupHomomorphismBasis`.
            init_scheme (``eINIT_SCHEMES``, optional): Initialization passed to
                :meth:`~symm_learning.representation_theory.GroupHomomorphismBasis.initialize_params`. Defaults to
                ``"xavier_uniform"``.
            **conv1d_kwargs: Standard :class:`torch.nn.ConvTranspose1d` arguments (stride, padding, bias, etc.).
        """
        assert in_rep.group == out_rep.group, f"Incompatible group: {in_rep.group} and {out_rep.group}"
        if "groups" in conv1d_kwargs:
            assert conv1d_kwargs["groups"] == 1, "`groups`>1 are not supported in eConvTranspose1d"

        super().__init__(in_channels=in_rep.size, out_channels=out_rep.size, kernel_size=kernel_size, **conv1d_kwargs)
        dtype = conv1d_kwargs.get("dtype", torch.get_default_dtype())
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

        self.homo_basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion)
        self.in_rep, self.out_rep = self.homo_basis.in_rep, self.homo_basis.out_rep
        if self.homo_basis.dim == 0:
            raise ValueError(
                f"No equivariant linear maps exist between {in_rep} and {out_rep}.\n dim(Hom_G(in_rep, out_rep))=0"
            )
        self.register_parameter(
            "weight_dof",
            torch.nn.Parameter(torch.zeros(kernel_size, self.homo_basis.dim, dtype=dtype), requires_grad=True),
        )

        bias = conv1d_kwargs.get("bias", True)
        trivial_id = self.homo_basis.G.trivial_representation.id
        can_have_bias = out_rep._irreps_multiplicities.get(trivial_id, 0) > 0
        self.has_bias = bias and can_have_bias
        if self.has_bias:
            m_out_trivial = out_rep._irreps_multiplicities[trivial_id]
            self.register_parameter(
                "bias_dof", torch.nn.Parameter(torch.zeros(m_out_trivial, dtype=dtype), requires_grad=True)
            )
            self.register_buffer("Qout", torch.tensor(self.out_rep.change_of_basis, dtype=dtype))

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        """Apply the constrained transposed 1D convolution on channel dimension."""
        assert input.shape[-2] == self.in_rep.size, (
            f"Expected input of shape (..., {self.in_rep.size}, H > 0) got {input.shape}"
        )
        return super().forward(input)

    @property
    def weight(self) -> torch.Tensor:  # noqa: D102
        """Dense kernel of shape ``(in_channels, out_channels, kernel_size)``."""
        W = self.homo_basis(self.weight_dof)  # (kernel_size, out_rep.size, in_rep.size)
        return W.permute(2, 1, 0)  # (in_rep.size, out_rep.size, kernel_size)

    @property
    def bias(self) -> torch.Tensor | None:  # noqa: D102
        """Expanded invariant bias or ``None`` when not admissible."""
        return self._expand_bias() if self.has_bias else None

    def _expand_bias(self):
        """Expand bias degrees of freedom into the original basis."""
        trivial_id = self.out_rep.group.trivial_representation.id
        trivial_indices = self.homo_basis.iso_blocks[trivial_id]["out_slice"]
        bias = torch.mv(self.Qout[:, trivial_indices], self.bias_dof)
        return bias

    @torch.no_grad()
    def reset_parameters(self, scheme: eINIT_SCHEMES = "xavier_normal"):
        """Reset trainable parameters using the chosen initialization scheme."""
        if not hasattr(self, "homo_basis"):  # First call on torch.nn.ConvTranspose1d init
            return super().reset_parameters()
        new_params = self.homo_basis.initialize_params(scheme=scheme, leading_shape=self.kernel_size)
        self.weight_dof.copy_(new_params)

        if self.has_bias:
            trivial_id = self.out_rep.group.trivial_representation.id
            m_in_inv = self.in_rep._irreps_multiplicities[trivial_id]
            m_out_inv = self.out_rep._irreps_multiplicities[trivial_id]
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(m_out_inv, m_in_inv))
            bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_dof, -bound, bound)

    @torch.no_grad()
    def check_equivariance(self, atol: float = 1e-5, rtol: float = 1e-5):
        """Check equivariance under channel actions of the underlying group."""
        G = self.in_rep.group
        B, L = 10, 30
        x = torch.randn(B, self.in_rep.size, L, device=self.weight_dof.device, dtype=self.weight_dof.dtype)
        y = self(x)

        for _ in range(10):
            g = G.sample()
            rho_in = torch.tensor(self.in_rep(g), dtype=x.dtype, device=x.device)
            rho_out = torch.tensor(self.out_rep(g), dtype=y.dtype, device=y.device)
            gx = torch.einsum("ij,bjl->bil", rho_in, x)
            y_expected = self(gx)
            gy = torch.einsum("ij,bjl->bil", rho_out, y)
            assert torch.allclose(gy, y_expected, atol=atol, rtol=rtol), (
                f"Equivariance failed for group element {g} with max error {(gy - y_expected).abs().max().item():.3e}"
            )
