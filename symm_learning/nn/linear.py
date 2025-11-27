from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from escnn.group import Representation
from torch.nn.utils import parametrize

from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint
from symm_learning.representation_theory import GroupHomomorphismBasis
from symm_learning.utils import get_spectral_trivial_mask

logger = logging.getLogger(__name__)

eINIT_SCHEMES = Literal["xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform"]


def impose_linear_equivariance(
    lin: torch.nn.Linear,
    in_rep: Representation,
    out_rep: Representation,
    basis_expansion_scheme: str = "isotypic_expansion",
) -> None:
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
    basis_expansion_scheme : str
        Basis expansion strategy for the commuting constraint (``\"memory_heavy\"`` or ``\"isotypic_expansion\"``).
    """
    assert isinstance(lin, torch.nn.Module), f"lin must be a torch.nn.Module, got {type(lin)}"
    # Add attributes to the layer for later reference
    lin.in_rep = in_rep
    lin.out_rep = out_rep
    # Register parametrizations enforcing equivariance
    parametrize.register_parametrization(
        lin, "weight", CommutingConstraint(in_rep, out_rep, basis_expansion=basis_expansion_scheme)
    )
    if lin.bias is not None:
        parametrize.register_parametrization(lin, "bias", InvariantConstraint(out_rep))


class eLinear(torch.nn.Linear):
    r"""Parameterize a :math:`\mathbb{G}`-equivariant linear map with optional invariant bias.

    The layer learns coefficients over :math:`\operatorname{Hom}_G(\text{in}, \text{out})`, synthesizing a dense weight
    commuting with the supplied representations. The optional bias is delegated to :class:`InvariantBias`, which caches
    its expansion in eval mode. The weight is cached after each expansion in eval mode to avoid redundant synthesis.

    Note:
        This layer can be used as a drop-in replacement for ``torch.nn.Linear``.

    Attributes:
        homo_basis (GroupHomomorphismBasis): Handler exposing the equivariant basis and metadata.
        bias_module (InvariantBias | None): Optional module handling the invariant bias.
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
        # Register weight parameters (degrees of freedom: dof) and buffers
        self.register_parameter(
            "weight_dof",
            torch.nn.Parameter(torch.zeros(self.homo_basis.dim, dtype=torch.get_default_dtype()), requires_grad=True),
        )

        self.bias_module = InvariantBias(out_rep) if bias else None

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def expand_weight(self):
        r"""Return the dense equivariant weight, caching it outside training.

        Returns:
            torch.Tensor: Dense matrix of shape ``(out_rep.size, in_rep.size)``.
        """
        W = self.homo_basis(self.weight_dof)  # Recompute linear map
        self._weight = W
        return W

    @property
    def weight(self) -> torch.Tensor:
        """Dense equivariant weight; recomputed in train, cached in eval."""
        if self.training or self._weight is None:
            return self.expand_weight()
        return self._weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Invariant bias from :class:`InvariantBias` (``None`` if disabled)."""
        return self.bias_module.bias if self.bias_module is not None else None

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
        # Update cache
        self.expand_weight()
        if self.bias_module is not None:
            self.bias_module.reset_parameters(scheme=scheme)

        logger.debug(f"Reset parameters of linear layer to {scheme}")


class InvariantBias(torch.nn.Module):
    r"""Module parameterizing a learnable :math:`G`-invariant bias.

    For point-group symmetries the bias is constrained to live in the `trivial/invariant` subspace of the input vector
    space. Therefore this module allocates a trainable parameter of size equal to the multiplicity of the trivial irrep
    in the input representation.

    If the input representation does not contain the trivial irrep (no trivial/invariant subspace), the module behaves
    as the identity function.

    Note:
        The bias is cached after switching to evaluation mode for efficiency.
    """

    def __init__(self, in_rep: Representation):
        """Construct the invariant bias module.

        Args:
            in_rep: Representation of the input space (same as output space).
        """
        super().__init__()
        self.in_rep, self.out_rep = in_rep, in_rep

        G = self.in_rep.group
        trivial_id = G.trivial_representation.id
        # Assert invariant vector is possible.
        self.has_bias = in_rep._irreps_multiplicities.get(trivial_id, 0) > 0

        if not self.has_bias:  # No bias -> No buffer memory consumption
            return

        dtype = torch.get_default_dtype()
        # Number of bias trainable parameters are equal to the output multiplicity of the trivial irrep
        m_out_trivial = self.in_rep._irreps_multiplicities[trivial_id]
        self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(m_out_trivial), requires_grad=True))
        self.register_buffer("Qout", torch.tensor(self.in_rep.change_of_basis, dtype=dtype))
        # Save mask of trivial dimensions in the irrep-spectral basis
        self.register_buffer("spectral_trivial_mask", get_spectral_trivial_mask(self.in_rep))

        # Cache reference of last computed bias
        self._bias: torch.Tensor | None = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the invariant bias.

        Args:
            input (torch.Tensor): Tensor whose last dimension equals ``in_rep.size``.

        Returns:
            torch.Tensor: Output tensor with the same shape as ``input``.
        """
        if not self.has_bias:
            return input
        return input + self.bias

    @property
    def bias(self):
        """Invariant bias; recomputed in training, cached otherwise."""
        # If training, recompute bias; else use cached version
        if self.training or self._bias is None:
            return self.expand_bias()
        return self._bias

    def expand_bias(self):
        """Expand the learnable parameters into the invariant bias in the original basis."""
        bias = torch.mv(self.Qout[:, self.spectral_trivial_mask], self.bias_dof)
        # Update cache
        self._bias = bias
        return bias

    def expand_bias_spectral_basis(self):
        """Return the invariant bias expressed in the irrep-spectral basis."""
        spectral_bias = torch.zeros(self.in_rep.size, dtype=self.bias_dof.dtype, device=self.bias_dof.device)
        spectral_bias[self.spectral_trivial_mask] = self.bias_dof
        return spectral_bias

    def reset_parameters(self, scheme="kaiming_uniform"):
        """Initialize the invariant bias degrees of freedom."""
        if not self.has_bias:
            return
        if scheme == "zeros":
            torch.nn.init.zeros_(self.bias_dof)

        trivial_id = self.in_rep.group.trivial_representation.id
        m = self.in_rep._irreps_multiplicities[trivial_id]
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(m, m))
        bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias_dof, -bound, bound)

        self._bias = None  # Clear cache

    def eval(self):
        """Refresh the cached bias and switch to evaluation mode."""
        # Update cache.
        self.expand_bias()
        return super().eval()


class eAffine(torch.nn.Module):
    r"""Symmetry-preserving affine map :math:`y = \alpha \odot x + \beta`.

    The scale :math:`\alpha` is constant within each irrep block and the bias :math:`\beta`
    lives only in the invariant subspace. When ``learnable=True`` these degrees of freedom are
    stored as trainable parameters. When ``learnable=False`` they must be provided to
    :meth:`forward` as ``scale_dof`` (length ``n_irreps``) and, if enabled, ``bias_dof``
    (length ``#trivial``), allowing external FiLM-style modulation.

    Note:
        TODO: This module can be implemented without transitioning to the spectral basis, which can improve efficiency
        dramatically.

    Args:
        in_rep: :class:`escnn.group.Representation` describing the input/output space.
        bias: include invariant biases when the trivial irrep is present. Default: ``True``.
        learnable: if ``False``, no parameters are registered and ``scale_dof``/``bias_dof`` must
            be passed at call time. Default: ``True``.
        init_scheme: initialization for the learnable DoFs (``"identity"`` or ``"random"``). Set
            to ``None`` to skip init (e.g. when loading weights). Ignored when ``learnable=False``.

    Shape:
        - Input: ``(..., D)`` with ``D = in_rep.size``.
        - ``scale_dof`` (optional): ``(..., num_scale_dof)`` where ``n_irreps`` is the number of irreps in ``in_rep``.
        - ``bias_dof`` (optional): ``(..., num_bias_dof)`` when ``bias=True``.
        - Output: ``(..., D)``.

    Attributes:
        num_scale_dof (int): Expected length of ``scale_dof`` (one per irrep).
        num_bias_dof (int): Expected length of ``bias_dof`` when an invariant subspace is present.
    """

    def __init__(
        self,
        in_rep: Representation,
        bias: bool = True,
        learnable: bool = True,
        init_scheme: Literal["identity", "random"] | None = "identity",
    ):
        super().__init__()
        self.in_rep, self.out_rep = in_rep, in_rep
        self.learnable = learnable

        self.rep_x = in_rep
        G = self.rep_x.group
        dtype = torch.get_default_dtype()

        # Common metadata --------------------------------------------------------------------------
        self.register_buffer("Q", torch.tensor(self.rep_x.change_of_basis, dtype=dtype))
        self.register_buffer("Q_inv", torch.tensor(self.rep_x.change_of_basis_inv, dtype=dtype))
        # Buffers needed to map per-irrep scale to full spectral scale
        irrep_dims_list = [G.irrep(*irrep_id).size for irrep_id in self.rep_x.irreps]
        irrep_dims = torch.tensor(irrep_dims_list, dtype=torch.long)
        self._num_scale_dof = len(irrep_dims_list)
        self.register_buffer(
            "irrep_indices", torch.repeat_interleave(torch.arange(len(irrep_dims), dtype=torch.long), irrep_dims)
        )

        trivial_id = G.trivial_representation.id
        self.has_bias = bias and trivial_id in self.rep_x._irreps_multiplicities
        self._num_bias_dof = 0
        self.bias_module = None
        if self.has_bias and self.learnable:
            # Reuse invariant-bias helper (stores bias_dof and spectral mask)
            self.bias_module = InvariantBias(self.rep_x)
            self._num_bias_dof = self.bias_module.bias_dof.numel()
            # Convenience handle so callers expecting ``bias_dof`` still find it.
            self.bias_dof = self.bias_module.bias_dof
        elif self.has_bias:
            trivial_mask = torch.zeros(self.rep_x.size, dtype=torch.bool)
            offset = 0
            for irrep_id in self.rep_x.irreps:
                if irrep_id == trivial_id:
                    trivial_mask[offset] = 1
                    self._num_bias_dof += 1
                offset += G.irrep(*irrep_id).size
            self.register_buffer("trivial_subspace_mask", trivial_mask)

        # Mode-specific parameters -----------------------------------------------------------------
        if self.learnable:
            self.register_parameter("scale_dof", torch.nn.Parameter(torch.ones(self.num_scale_dof, dtype=dtype)))
            if self.has_bias and self.bias_module is None:
                self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(self.num_bias_dof, dtype=dtype)))
            elif self.has_bias:
                # bias handled by bias_module; keep attribute for API compatibility
                self.bias_dof = self.bias_module.bias_dof
            else:
                self.register_parameter("bias_dof", None)

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def forward(
        self,
        input: torch.Tensor,
        scale_dof: torch.Tensor | None = None,
        bias_dof: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the equivariant affine transform.

        When ``learnable=False`` ``scale_dof`` (and ``bias_dof`` if ``bias=True``) must be provided.

        Args:
            input: tensor whose last dimension matches ``in_rep.size``.
            scale_dof: optional per-irrep scaling degrees of freedom (length ``num_scale_dof``). Required when
                ``learnable=False`` with leading dims matching ``input.shape[:-1]``.
            bias_dof: optional bias degrees of freedom for the invariant irreps (length ``num_bias_dof``). Required
                when ``learnable=False`` and ``bias=True`` with leading dims matching ``input.shape[:-1]``.
        """
        if input.shape[-1] != self.rep_x.size:
            raise ValueError(f"Expected last dimension {self.rep_x.size}, got {input.shape[-1]}")

        # Obtain per-dimension spectral scale; reuse learnable bias directly in original basis.
        if self.learnable:
            scale_spec = self.scale_dof[self.irrep_indices]  # (D,)
            bias_orig = self.bias_module.bias if self.has_bias and self.bias_module is not None else None
        else:
            scale_spec, spectral_bias = self.get_spectral_scale_and_bias(scale_dof, bias_dof, input_shape=input.shape)
            bias_orig = None
            if spectral_bias is not None:
                # Map spectral bias back to original basis: (..., D)
                bias_orig = torch.einsum("ij,...j->...i", self.Q, spectral_bias)

        # Apply scaling in original basis via Q * diag(scale_spec) * Q_inv. Output shape matches input (..., D).
        y = torch.einsum("ij,...j,jk,...k->...i", self.Q, scale_spec, self.Q_inv, input)
        if bias_orig is not None:
            y = y + bias_orig
        return y

    def get_spectral_scale_and_bias(
        self,
        scale_dof: torch.Tensor,
        bias_dof: torch.Tensor | None = None,
        input_shape: torch.Size | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return spectral scale and bias from provided DoFs.

        Args:
            scale_dof: Per-irrep scale coefficients shaped ``(..., num_scale_dof)``.
            bias_dof: Invariant-bias coefficients shaped ``(..., num_bias_dof)`` when ``bias=True``.
            input_shape: Shape of the input tensor when ``learnable=False``. Used to validate DoF shapes.

        Returns:
            spectral_scale: Spectral scale tensor shaped ``(..., rep_x.size)``.
            spectral_bias: Spectral bias tensor shaped ``(..., rep_x.size)`` or ``None`` when ``bias=False``.
        """
        input_shape = () if input_shape is None else input_shape[:-1]

        if scale_dof is None or scale_dof.shape != (*input_shape, self._num_scale_dof):
            raise ValueError(
                f"Expected scale_dof shape {(*input_shape, self._num_scale_dof)}, got "
                f"{scale_dof.shape if scale_dof is not None else None}"
            )

        # Broadcast scale per irrep subspace to each irrep subspace dimension.
        spectral_scale = scale_dof[..., self.irrep_indices]
        spectral_bias = None

        if self.has_bias:
            # Use provided bias DoFs when passed (external control), otherwise fall back to learnable helper.
            if bias_dof is None and self.bias_module is not None:
                bias_dof = self.bias_module.bias_dof
            if bias_dof is None or bias_dof.shape != (*input_shape, self._num_bias_dof):
                raise ValueError(
                    f"Expected bias_dof shape {(*input_shape, self._num_bias_dof)}, got "
                    f"{bias_dof.shape if bias_dof is not None else None}"
                )

            if self.bias_module is not None:
                # Learnable bias: use helper to expand into spectral basis
                spectral_bias = self.bias_module.expand_bias_spectral_basis()
            else:
                spectral_bias = bias_dof.new_zeros(*bias_dof.shape[:-1], self.rep_x.size)
                spectral_bias[..., self.trivial_subspace_mask] = bias_dof

        return spectral_scale, spectral_bias

    def reset_parameters(self, scheme: Literal["identity", "random"] = "identity") -> None:
        """Initialize spectral scale/bias DoFs.

        Args:
            scheme: ``"identity"`` sets all scales to one and bias to zero; ``"random"`` samples both
                uniformly in ``[-1, 1]``. Set to ``None`` when loading checkpoints to skip reinit.
        """
        if not self.learnable:
            return
        if scheme == "identity":
            torch.nn.init.ones_(self.scale_dof)
            if self.has_bias and self.bias_module is not None and self.bias_module.bias_dof is not None:
                torch.nn.init.zeros_(self.bias_module.bias_dof)
            elif self.has_bias and self.bias_dof is not None:
                torch.nn.init.zeros_(self.bias_dof)
        elif scheme == "random":
            torch.nn.init.uniform_(self.scale_dof, -1, 1)
            if self.has_bias and self.bias_module is not None and self.bias_module.bias_dof is not None:
                torch.nn.init.uniform_(self.bias_module.bias_dof, -1, 1)
            elif self.has_bias and self.bias_dof is not None:
                torch.nn.init.uniform_(self.bias_dof, -1, 1)
        else:
            raise NotImplementedError(f"Init scheme {scheme} not implemented")

    def extra_repr(self) -> str:  # noqa: D102
        return f"bias={self.has_bias} learnable={self.learnable} \nin_rep={self.in_rep}"

    @property
    def num_scale_dof(self) -> int:
        """Number of per-irrep scaling degrees of freedom (length of ``scale_dof``)."""
        return self._num_scale_dof

    @property
    def num_bias_dof(self) -> int:
        """Number of bias degrees of freedom (length of ``bias_dof``) for invariant irreps."""
        return self._num_bias_dof
