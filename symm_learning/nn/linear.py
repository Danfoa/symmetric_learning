from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from escnn.group import Representation
from torch.nn.utils import parametrize

from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint
from symm_learning.representation_theory import GroupHomomorphismBasis, direct_sum, isotypic_decomp_rep

logger = logging.getLogger(__name__)


def impose_linear_equivariance(lin: torch.nn.Linear, in_rep: Representation, out_rep: Representation) -> None:
    r"""Impose equivariance constraints on a given torch.nn.Linear layer using torch parametrizations.

    Impose via torch parametrizations (hard constraints on trainable parameters ) that the weight matrix of
    the given linear layer commutes with the group actions of the input and output representations. That is:

    .. math::
        \rho_{\text{out}}(g) W = W \rho_{\text{in}}(g) \quad \forall g \in G

    If the layer has a bias term, it is constrained to be invariant:
    .. math::
        \rho_{\text{out}}(g) b = b \quad \forall g \in G

    Parameters
    ----------
    lin : torch.nn.Module
        The linear layer to impose equivariance on. Must have 'weight' and optionally 'bias' attributes.
    in_rep : escnn.group.Representation
        The input representation of the layer.
    out_rep : escnn.group.Representation
        The output representation of the layer.
    """
    assert isinstance(lin, torch.nn.Module), f"lin must be a torch.nn.Module, got {type(lin)}"
    # Add attributes to the layer for later reference
    lin.in_rep = in_rep
    lin.out_rep = out_rep
    # Register parametrizations enforcing equivariance
    parametrize.register_parametrization(lin, "weight", CommutingConstraint(in_rep, out_rep))
    if lin.bias is not None:
        parametrize.register_parametrization(lin, "bias", InvariantConstraint(out_rep))


class eLinear(torch.nn.Linear):
    r"""Parameterize a :math:`\mathbb{G}`-equivariant linear map with optional invariant bias.

    The layer learns coefficients over :math:`\operatorname{Hom}_G(\text{in}, \text{out})`, synthesizing a dense weight
    commuting with the supplied representations and, when admissible, a bias in :math:`\mathrm{Fix}_G(\text{out})`.
    Training rebuilds these tensors every call; evaluation reuses the cached expansion.

    Attributes:
        homo_basis (GroupHomomorphismBasis): Handler exposing the equivariant basis and metadata.
    """

    def __init__(
        self,
        in_rep: Representation,
        out_rep: Representation,
        bias: bool = True,
        init_scheme: str | None = "xavier_normal",
        basis_expansion_scheme: str = "isotypic_expansion",
    ):
        r"""Initialize the equivariant layer.

        Args:
            in_rep (Representation): Representation describing how inputs transform.
            out_rep (Representation): Representation describing how outputs transform.
            bias (bool, optional): Enables the invariant bias if the trivial irrep is present in ``out_rep``.
                Default: ``True``.
            init_scheme (str | None, optional): Initialization method passed to
                :meth:`GroupHomomorphismBasis.initialize_params`. Use ``None`` to skip initialization. Default:
                ``"xavier_normal"``.
            basis_expansion_scheme (str, optional): Strategy for materializing the basis (``"isotypic_expansion"`` or
                ``"memory_heavy"``). Default: ``"isotypic_expansion"``.

        Raises:
            ValueError: If :math:`\dim(\mathrm{Hom}_G(\text{in}, \text{out})) = 0`.
        """
        super().__init__(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
        # Delete linear unconstrained module parameters
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        # Instanciate the handler of the basis of Hom_G(in_rep, out_rep)
        self.homo_basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion=basis_expansion_scheme)
        self.in_rep, self.out_rep = self.homo_basis.in_rep, self.homo_basis.out_rep
        if self.homo_basis.dim == 0:
            raise ValueError(
                f"No equivariant linear maps exist between {in_rep} and {out_rep}.\n dim(Hom_G(in_rep, out_rep))=0"
            )
        # Assert bias vector is feasible given out_rep symmetries
        trivial_id = self.homo_basis.G.trivial_representation.id
        can_have_bias = out_rep._irreps_multiplicities.get(trivial_id, 0) > 0
        self.has_bias = bias and can_have_bias
        # Register change of basis matrices as buffers
        dtype = torch.get_default_dtype()

        # Register weight parameters (degrees of freedom: dof) and buffers
        self.register_parameter(
            "weight_dof", torch.nn.Parameter(torch.zeros(self.homo_basis.dim, dtype=dtype), requires_grad=True)
        )

        if self.has_bias:  # Register bias parameters
            # Number of bias trainable parameters are equal to the output multiplicity of the trivial irrep
            m_out_trivial = out_rep._irreps_multiplicities[trivial_id]
            self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(m_out_trivial), requires_grad=True))
            self.register_buffer("Qout", torch.tensor(self.out_rep.change_of_basis, dtype=dtype))

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)
        self._weight, self._bias = None, None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the equivariant map.

        Args:
            input (torch.Tensor): Tensor whose last dimension equals ``in_rep.size``.

        Returns:
            torch.Tensor: Output tensor with the same leading shape as ``input`` and last dimension ``out_rep.size``.
        """
        W, b = self.expand_weight_and_bias()

        return F.linear(input, W, b)

    def expand_weight_and_bias(self):
        r"""Return the dense equivariant weight and optional invariant bias.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Dense matrix of shape ``(out_rep.size, in_rep.size)`` and the
            invariant bias (or ``None`` if the trivial irrep is absent).
        """
        W = self.homo_basis(self.weight_dof)  # Recompute linear map
        bias = None
        if self.has_bias:  # Recompute bias
            bias = self._expand_bias()
        self._weight = W
        self._bias = bias
        return W, bias

    def _expand_bias(self):
        trivial_id = self.out_rep.group.trivial_representation.id
        trivial_indices = self.homo_basis.iso_blocks[trivial_id]["out_slice"]
        bias = torch.mv(self.Qout[:, trivial_indices], self.bias_dof)
        return bias

    @property
    def weight(self) -> torch.Tensor:  # noqa: D102
        return self.homo_basis(self.weight_dof)

    @property
    def bias(self) -> torch.Tensor | None:  # noqa: D102
        return self._expand_bias() if self.has_bias else None

    @torch.no_grad()
    def reset_parameters(self, scheme="xavier_normal"):
        """Reset all trainable parameters.

        Args:
            scheme (str): Initialization scheme (``"xavier_normal"``, ``"xavier_uniform"``, ``"kaiming_normal"``, or
                ``"kaiming_uniform"``).
        """
        if not hasattr(self, "homo_basis"):  # First call on torch.nn.Linear init
            return super().reset_parameters()
        new_params = self.homo_basis.initialize_params(scheme)
        self.weight_dof.copy_(new_params)

        if self.has_bias:
            trivial_id = self.out_rep.group.trivial_representation.id
            m_in_inv = self.in_rep._irreps_multiplicities[trivial_id]
            m_out_inv = self.out_rep._irreps_multiplicities[trivial_id]
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(m_out_inv, m_in_inv))
            bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_dof, -bound, bound)

        logger.debug(f"Reset parameters of linear layer to {scheme}")


class eAffine(torch.nn.Module):
    r"""Applies a symmetry-preserving affine transformation y = x * alpha + beta to the input x.

    The affine transformation for a given input :math:`x \in \mathcal{X} \subseteq \mathbb{R}^{D_x}` is defined as:

    .. math::
        \mathbf{y} = \mathbf{x} \cdot \alpha + \beta

    such that

    .. math::
        \rho_{\mathcal{X}}(g) \mathbf{y} = (\rho_{\mathcal{X}}(g) \mathbf{x}) \cdot \alpha + \beta \quad \forall g \in G

    Where :math:`\mathcal{X}` is a symmetric vector space with group representation
    :math:`\rho_{\mathcal{X}}: G \to \mathbb{GL}(D_x)`, and :math:`\alpha \in \mathbb{R}^{D_x}`,
    :math:`\beta \in \mathbb{R}^{D_x}` are symmetry constrained learnable vectors.

    Args:
        in_rep: the :class:`escnn.group.Representation` group representation of the input feature space.
        bias: a boolean value that when set to ``True``, this module has a learnable bias vector
            in the invariant subspace of the input representation Default: ``True``

    Shape:
        - Input: of shape `(..., D)` where :math:`D` is the dimension of the input type.
        - Output: of shape `(..., D)`
    """

    def __init__(self, in_rep: Representation, bias: bool = True):
        super().__init__()
        self.in_rep, self.out_rep = in_rep, in_rep

        self.rep_x = in_rep
        G = self.rep_x.group
        default_dtype = torch.get_default_dtype()
        self.register_buffer("Q", torch.tensor(self.rep_x.change_of_basis, dtype=default_dtype))
        self.register_buffer("Q_inv", torch.tensor(self.rep_x.change_of_basis_inv, dtype=default_dtype))

        # Symmetry-preserving scaling implies scaling each irreducible subspace uniformly.
        irrep_dims_list = [G.irrep(*irrep_id).size for irrep_id in self.rep_x.irreps]
        irrep_dims = torch.tensor(irrep_dims_list, dtype=torch.long)
        n_scale_params = len(irrep_dims_list)
        self.register_parameter("scale_dof", torch.nn.Parameter(torch.ones(n_scale_params, dtype=default_dtype)))
        self.register_buffer(
            "irrep_indices", torch.repeat_interleave(torch.arange(len(irrep_dims), dtype=torch.long), irrep_dims)
        )

        has_invariant_subspace = G.trivial_representation.id in self.rep_x.irreps
        self.has_bias = bias and has_invariant_subspace
        if self.has_bias:
            is_trivial_irrep = torch.tensor(
                [irrep_id == G.trivial_representation.id for irrep_id in self.rep_x.irreps], dtype=torch.bool
            )
            n_bias_params = int(is_trivial_irrep.sum().item())
            self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(n_bias_params, dtype=default_dtype)))
            dim_to_param = torch.full((self.rep_x.size,), -1, dtype=torch.long)
            offset = 0
            bias_idx = 0
            for is_trivial, dim in zip(is_trivial_irrep.tolist(), irrep_dims_list):
                if is_trivial:
                    dim_to_param[offset : offset + dim] = bias_idx
                    bias_idx += 1
                offset += dim
            self.register_buffer("bias_dim_to_param", dim_to_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the affine transformation; works for any input with last dim ``D``."""
        if x.shape[-1] != self.rep_x.size:
            raise ValueError(f"Expected last dimension {self.rep_x.size}, got {x.shape[-1]}")

        x_spectral = torch.einsum("ij,...j->...i", self.Q_inv, x)

        scale_spec, bias_spec = self.spectral_parameters(device=x_spectral.device, dtype=x_spectral.dtype)
        x_spectral = x_spectral * scale_spec.view(*([1] * (x_spectral.ndim - 1)), -1)

        if bias_spec is not None:
            x_spectral = x_spectral + bias_spec.view(*([1] * (x_spectral.ndim - 1)), -1)

        y = torch.einsum("ij,...j->...i", self.Q, x_spectral)
        return y

    def spectral_parameters(self, device=None, dtype=None):
        """Return per-dimension spectral scale and bias vectors."""
        scale = self.scale_dof[self.irrep_indices].to(device=device, dtype=dtype or self.scale_dof.dtype)
        bias = None
        if self.has_bias and self.bias_dof.numel() > 0:
            bias_index = self.bias_dim_to_param
            valid = bias_index >= 0
            if valid.any():
                bias = torch.zeros(
                    self.rep_x.size, device=scale.device if device is None else device, dtype=dtype or scale.dtype
                )
                bias_vals = self.bias_dof.to(device=bias.device, dtype=bias.dtype)
                bias_index = bias_index.to(bias.device)
                bias[valid] = bias_vals[bias_index[valid]]
        return scale, bias

    def reset_parameters(self) -> None:  # noqa: D102
        torch.nn.init.ones_(self.scale_dof)
        if self.has_bias and self.bias_dof is not None:
            torch.nn.init.zeros_(self.bias_dof)

    def extra_repr(self) -> str:  # noqa: D102
        return f"in_rep{self.in_rep} bias={self.has_bias}"


if __name__ == "__main__":
    from escnn.group import CyclicGroup, DihedralGroup, Icosahedral
    from escnn.nn import Linear
    from numpy import set_printoptions

    from symm_learning.utils import describe_memory

    set_printoptions(precision=2, suppress=True)

    from symm_learning.utils import bytes_to_mb, check_equivariance, module_memory_breakdown

    G = CyclicGroup(10)
    # G = Icosahedral()
    m_in, m_out = 5, 1
    bias = False
    in_rep = direct_sum([G.regular_representation] * m_in)
    out_rep = direct_sum([G.regular_representation] * m_out)

    baseline_layer = torch.nn.Linear(in_rep.size, out_rep.size, bias=bias)
    describe_memory("torch.nn.Linear baseline", baseline_layer)

    def backprop_sanity(module: torch.nn.Module, label: str):  # noqa: D103
        module.train()
        optim = torch.optim.SGD(module.parameters(), lr=1e-3)
        x = torch.randn(64, module.in_rep.size)
        target = torch.randn(64, module.out_rep.size)
        optim.zero_grad()
        y = module(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        optim.step()
        grad_norm = sum((p.grad.norm().item() for p in module.parameters() if p.grad is not None))
        print(f"{label} backprop test passed (grad norm {grad_norm:.4f})")

    for basis_expansion in ["isotypic_expansion", "memory_heavy"]:
        layer = eLinear(in_rep, out_rep, bias=bias, basis_expansion_scheme=basis_expansion)
        describe_memory(f"eLinear ({basis_expansion})", layer)
        check_equivariance(layer, atol=1e-5, rtol=1e-5)
        print(f"eLinear with basis expansion '{basis_expansion}' equivariant test passed.")
        backprop_sanity(layer, f"eLinear with basis expansion '{basis_expansion}'")
