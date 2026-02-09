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
    lin : :class:`~torch.nn.Module`
        The linear layer to impose equivariance on. Must have 'weight' and optionally 'bias' attributes.
    in_rep : :class:`~escnn.group.Representation`
        The input representation :math:`\rho_{\text{in}}` of the layer.
    out_rep : :class:`~escnn.group.Representation`
        The output representation :math:`\rho_{\text{out}}` of the layer.
    basis_expansion_scheme : str
        Basis expansion strategy for the commuting constraint (``"memory_heavy"`` or ``"isotypic_expansion"``).
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

    The layer learns coefficients over :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\text{in}}, \rho_{\text{out}})`,
    synthesizing a dense weight matrix :math:`\mathbf{W}` satisfying:

    .. math::
        \rho_{\text{out}}(g) \mathbf{W} = \mathbf{W} \rho_{\text{in}}(g) \quad \forall g \in \mathbb{G}

    If ``bias=True``, the bias vector :math:`\mathbf{b}` is constrained to the invariant subspace:

    .. math::
        \rho_{\text{out}}(g) \mathbf{b} = \mathbf{b} \quad \forall g \in \mathbb{G}

    Note:
        Runtime behavior depends on mode.
        In training mode (``model.train()``), the constrained dense tensors are recomputed every forward pass, which
        is correct for gradient updates but slower.
        In inference mode (``model.eval()``), the expanded dense weight (and optional invariant bias) are cached and
        reused until parameters change or :meth:`invalidate_cache` is called, which is faster.
        With the cache active, :meth:`forward` is computationally equivalent to a symmetry-agnostic
        :class:`~torch.nn.Linear` with fixed dense ``weight`` and ``bias``.

    Attributes:
        homo_basis (:class:`~symm_learning.representation_theory.GroupHomomorphismBasis`): Handler exposing the
            equivariant basis and metadata.
        bias_module (:class:`~symm_learning.nn.linear.InvariantBias` | None): Optional module handling the invariant
            bias.
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
            in_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{in}}` describing how inputs
                transform.
            out_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{out}}` describing how
                outputs transform.
            bias (:class:`bool`, optional): Enables the invariant bias if the trivial irrep is present in ``out_rep``.
                Default: ``True``.
            init_scheme (:class:`str` | :class:`None`, optional): Initialization method passed to
                :meth:`~symm_learning.representation_theory.GroupHomomorphismBasis.initialize_params`. Use ``None``
                to skip initialization. Default: ``"xavier_normal"``.
            basis_expansion_scheme (:class:`str`, optional): Strategy for materializing the basis
                (``"isotypic_expansion"`` or ``"memory_heavy"``). Default: ``"isotypic_expansion"``.

        Raises:
            ValueError: If :math:`\dim(\mathrm{Hom}_{\mathbb{G}}(\rho_{\text{in}}, \rho_{\text{out}})) = 0`.
        """
        super().__init__(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
        # Delete linear unconstrained module parameters
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        self.register_buffer("_weight", None, persistent=False)
        self._weight_cache_dirty = True
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

        # Register backward hook to flag caches stale/invalid whenever grads are produced.
        self.weight_dof.register_hook(self._mark_weight_cache_dirty)

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def expand_weight(self):
        r"""Return the dense equivariant weight, caching it outside training.

        Returns:
            torch.Tensor: Dense matrix of shape ``(out_rep.size, in_rep.size)``.
        """
        W = self.homo_basis(self.weight_dof)  # Recompute linear map
        self._weight = W
        self._weight_cache_dirty = False
        return W

    @property
    def weight(self) -> torch.Tensor:
        """Dense equivariant weight; recomputed in train, cached in eval."""
        if self.training or self._weight is None or self._weight_cache_dirty:
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
            scheme (:class:`str`): Initialization scheme (``"xavier_normal"``, ``"xavier_uniform"``,
                ``"kaiming_normal"``, or ``"kaiming_uniform"``).
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

    def _mark_weight_cache_dirty(self, grad: torch.Tensor) -> torch.Tensor:
        self._weight_cache_dirty = True
        return grad

    def invalidate_cache(self) -> None:
        """Clear cached expansions and mark them stale."""
        self._weight = None
        self._weight_cache_dirty = True
        if self.bias_module is not None:
            self.bias_module.invalidate_cache()

    def _refresh_eval_cache(self) -> None:
        """Ensure eval-mode caches are materialized."""
        if self._weight is None or self._weight_cache_dirty:
            self.expand_weight()
        if self.bias_module is not None:
            self.bias_module.refresh_eval_cache()

    def _apply(self, fn):
        super()._apply(fn)
        self.invalidate_cache()
        return self

    def train(self, mode: bool = True):  # noqa: D102
        """Switch mode and keep cached expanded tensors consistent."""
        result = super().train(mode)
        if mode:  # Switching to train mode - invalidate cache
            self.invalidate_cache()
        else:  # Switching to eval mode - refresh cache
            self._refresh_eval_cache()
        return result

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D102
        """Load parameters and invalidate cached expanded tensors."""
        result = super().load_state_dict(state_dict, strict)
        self.invalidate_cache()
        return result


class InvariantBias(torch.nn.Module):
    r"""Module parameterizing a learnable :math:`\mathbb{G}`-invariant bias.

    For representation space :math:`\mathcal{X}`, this module enforces
    :math:`\rho_{\mathcal{X}}(g)\mathbf{b}=\mathbf{b}` for all :math:`g\in\mathbb{G}`. Hence only trivial-irrep
    coordinates in the irrep-spectral basis carry free parameters.

    If the input representation does not contain the trivial irrep (no trivial/invariant subspace), the module behaves
    as the identity function.

    Note:
        Runtime behavior depends on mode.
        In training mode (``model.train()``), the invariant bias is recomputed each forward pass.
        In inference mode (``model.eval()``), the expanded invariant bias is cached and reused until ``bias_dof``
        changes or :meth:`invalidate_cache` is called, which is faster.
        With the cache active, the forward path is the same computation as the standard symmetry-agnostic bias add
        ``input + b`` with fixed ``b``.

    Attributes:
        in_rep (:class:`~escnn.group.Representation`): Representation defining the symmetry action on
            :math:`\mathcal{X}`.
        out_rep (:class:`~escnn.group.Representation`): Same as ``in_rep`` (bias acts in the same space).
        has_bias (:class:`bool`): ``True`` iff the trivial irrep is present and a learnable invariant bias exists.
        bias_dof (:class:`~torch.nn.Parameter`): Learnable trivial-subspace coefficients (present only if
            ``has_bias=True``).
    """

    def __init__(self, in_rep: Representation):
        r"""Construct the invariant bias module.

        Args:
            in_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{in}}` of the input space
                (same as output space).
        """
        super().__init__()
        self.in_rep, self.out_rep = in_rep, in_rep

        G = self.in_rep.group
        trivial_id = G.trivial_representation.id
        # Assert invariant vector is possible.
        self.has_bias = in_rep._irreps_multiplicities.get(trivial_id, 0) > 0

        self.register_buffer("_bias", None, persistent=False)
        self._bias_cache_dirty = False

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
        self._bias_cache_dirty = True
        self.bias_dof.register_hook(self._mark_bias_cache_dirty)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the invariant bias.

        Args:
            input (:class:`~torch.Tensor`): Tensor whose last dimension equals ``in_rep.size``.

        Returns:
            :class:`~torch.Tensor`: Output tensor with the same shape as ``input``.
        """
        if not self.has_bias:
            return input
        return input + self.bias

    @property
    def bias(self):
        """Invariant bias; recomputed in training, cached otherwise."""
        if not self.has_bias:
            return None
        # If training, recompute bias; else use cached version
        if self.training or self._bias is None or self._bias_cache_dirty:
            return self.expand_bias()
        return self._bias

    def expand_bias(self):
        """Expand the learnable parameters into the invariant bias in the original basis."""
        bias = torch.mv(self.Qout[:, self.spectral_trivial_mask], self.bias_dof)
        # Update cache
        self._bias = bias
        self._bias_cache_dirty = False
        return bias

    def expand_bias_spectral_basis(self):
        """Return the invariant bias expressed in the irrep-spectral basis."""
        spectral_bias = torch.zeros(self.in_rep.size, dtype=self.bias_dof.dtype, device=self.bias_dof.device)
        spectral_bias[self.spectral_trivial_mask] = self.bias_dof
        return spectral_bias

    def reset_parameters(self, scheme="zeros"):
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

        self.invalidate_cache()

    def train(self, mode: bool = True):
        """Switch between training and evaluation modes, managing cache appropriately."""
        result = super().train(mode)
        if mode:  # Switching to train mode - invalidate cache
            self.invalidate_cache()
        else:  # Switching to eval mode - refresh cache
            self.refresh_eval_cache()
        return result

    def _mark_bias_cache_dirty(self, grad: torch.Tensor) -> torch.Tensor:
        self._bias_cache_dirty = True
        return grad

    def invalidate_cache(self) -> None:
        """Clear cached bias so it is recomputed on next use."""
        if not self.has_bias:
            return
        self._bias = None
        self._bias_cache_dirty = True

    def refresh_eval_cache(self) -> None:
        """Ensure eval-mode cache is populated."""
        if not self.has_bias:
            return
        if self._bias is None or self._bias_cache_dirty:
            self.expand_bias()

    def _apply(self, fn):
        super()._apply(fn)
        self.invalidate_cache()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D102
        """Load parameters and invalidate cached expanded bias."""
        result = super().load_state_dict(state_dict, strict)
        self.invalidate_cache()
        return result


class eAffine(torch.nn.Module):
    r"""Equivariant affine map with per-irrep scales and invariant bias.

    Let :math:`\mathbf{x}\in\mathcal{X}` with representation

    .. math::
        \rho_{\mathcal{X}} = \mathbf{Q}\left(
        \bigoplus_{k\in[1,n_{\text{iso}}]}
        \bigoplus_{i\in[1,n_k]}
        \hat{\rho}_k
        \right)\mathbf{Q}^T.

    This module applies

    .. math::
        \mathbf{y} = \mathbf{Q}\,\mathbf{D}_{\alpha}\,\mathbf{Q}^T\mathbf{x} + \mathbf{b},

    where :math:`\mathbf{D}_{\alpha}` is diagonal in irrep-spectral basis and constant over dimensions of each irrep
    copy (:math:`\alpha_{k,i}`), while :math:`\mathbf{b}\in\mathrm{Fix}(\rho_{\mathcal{X}})` (trivial block only).
    Therefore:

    .. math::
        \rho_{\mathcal{X}}(g)\mathbf{y}
        = \operatorname{eAffine}\!\left(\rho_{\mathcal{X}}(g)\mathbf{x}\right)
        \quad \forall g\in\mathbb{G}.

    When ``learnable=True`` these DoFs are trainable parameters. When ``learnable=False``,
    ``scale_dof`` and ``bias_dof`` are provided at call-time (FiLM style).

    Args:
        in_rep: :class:`~escnn.group.Representation` describing the input/output space :math:`\rho_{\text{in}}`.
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

    Note:
        Runtime behavior depends on mode.
        In training mode (``model.train()``), the affine map is recomputed each forward pass.
        In inference mode (``model.eval()``) and with ``learnable=True``, the dense affine map
        :math:`\mathbf{Q}\mathbf{D}_{\alpha}\mathbf{Q}^T` (and optional invariant bias) is cached and reused until
        parameters change or :meth:`invalidate_cache` is called, which is faster.
        Unlike :class:`~symm_learning.nn.linear.eLinear`, this module is not a strict symmetry-agnostic drop-in
        affine block, because parameters are irrep-structured and may also be provided externally
        (FiLM-style via ``scale_dof``/``bias_dof``).

    Attributes:
        rep_x (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\mathcal{X}}` of the feature space.
        Q (:class:`~torch.Tensor`): Change-of-basis matrix to the irrep-spectral basis.
        Q_inv (:class:`~torch.Tensor`): Inverse change-of-basis matrix from irrep-spectral basis.
        bias_module (:class:`~symm_learning.nn.linear.InvariantBias` | None): Optional module handling the invariant
            bias.
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

        self.register_buffer("_affine", None, persistent=False)
        self._affine_cache_dirty = True
        if self.learnable:
            self.scale_dof.register_hook(self._mark_affine_cache_dirty)

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def forward(
        self,
        input: torch.Tensor,
        scale_dof: torch.Tensor | None = None,
        bias_dof: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Apply the equivariant affine transform.

        When ``learnable=False`` ``scale_dof`` (and ``bias_dof`` if ``bias=True``) must be provided.

        Args:
            input: Tensor :math:`\mathbf{x}` whose last dimension matches ``in_rep.size``.
            scale_dof: Optional per-irrep scaling DoFs (length ``num_scale_dof``). Required when
                ``learnable=False`` with leading dims matching ``input.shape[:-1]``.
            bias_dof: Optional invariant-bias DoFs (length ``num_bias_dof``). Required
                when ``learnable=False`` and ``bias=True`` with leading dims matching ``input.shape[:-1]``.
        """
        if input.shape[-1] != self.rep_x.size:
            raise ValueError(f"Expected last dimension {self.rep_x.size}, got {input.shape[-1]}")

        # Obtain per-dimension spectral scale; reuse learnable bias directly in original basis.
        if self.learnable:
            bias_orig = self.bias_module.bias if self.has_bias and self.bias_module is not None else None
            if not self.training:
                affine = self._affine
                if affine is None or self._affine_cache_dirty:
                    affine = self.expand_affine()
                y = torch.einsum("ij,...j->...i", affine, input)
                if bias_orig is not None:
                    y = y + bias_orig
                return y  # Use cached matrix.
            scale_spec = self.scale_dof[self.irrep_indices]  # (D,)
        else:
            scale_spec, spectral_bias = self.broadcast_spectral_scale_and_bias(
                scale_dof, bias_dof, input_shape=input.shape
            )
            bias_orig = None
            if spectral_bias is not None:
                # Map spectral bias back to original basis: (..., D)
                bias_orig = torch.einsum("ij,...j->...i", self.Q, spectral_bias)

        # Apply scaling in original basis via Q * diag(scale_spec) * Q_inv. Output shape matches input (..., D).
        y = torch.einsum("ij,...j,jk,...k->...i", self.Q, scale_spec, self.Q_inv, input)
        if bias_orig is not None:
            y = y + bias_orig
        return y

    def expand_affine(self) -> torch.Tensor:
        """Expand the per-irrep scales into an affine matrix in the original basis."""
        scale_spec = self.scale_dof[self.irrep_indices]
        affine = (self.Q * scale_spec) @ self.Q_inv
        self._affine = affine
        self._affine_cache_dirty = False
        return affine

    def broadcast_spectral_scale_and_bias(
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
            tuple[torch.Tensor, torch.Tensor | None]: Pair ``(spectral_scale, spectral_bias)`` where
            ``spectral_scale`` is shaped ``(..., rep_x.size)`` and ``spectral_bias`` is shaped
            ``(..., rep_x.size)`` (or ``None`` when ``bias=False``).
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
        self.invalidate_cache()

    def extra_repr(self) -> str:  # noqa: D102
        return f"bias={self.has_bias} learnable={self.learnable} \nin_rep={self.in_rep}"

    def _mark_affine_cache_dirty(self, grad: torch.Tensor) -> torch.Tensor:
        self._affine_cache_dirty = True
        return grad

    def invalidate_cache(self) -> None:
        """Clear cached affine map so it is recomputed on next use."""
        self._affine = None
        self._affine_cache_dirty = True
        if self.bias_module is not None:
            self.bias_module.invalidate_cache()

    def _refresh_eval_cache(self) -> None:
        if not self.learnable:
            return
        if self._affine is None or self._affine_cache_dirty:
            self.expand_affine()
        if self.bias_module is not None:
            self.bias_module.refresh_eval_cache()

    def _apply(self, fn):
        super()._apply(fn)
        self.invalidate_cache()
        return self

    def train(self, mode: bool = True):  # noqa: D102
        """Switch mode and keep cached affine expansion consistent."""
        result = super().train(mode)
        if mode:  # Switching to train mode - invalidate cache
            self.invalidate_cache()
        else:  # Switching to eval mode - refresh cache
            self._refresh_eval_cache()
        return result

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D102
        """Load parameters and invalidate cached affine expansion."""
        result = super().load_state_dict(state_dict, strict)
        self.invalidate_cache()
        return result

    @property
    def num_scale_dof(self) -> int:
        """Number of per-irrep scaling degrees of freedom (length of ``scale_dof``)."""
        return self._num_scale_dof

    @property
    def num_bias_dof(self) -> int:
        """Number of bias degrees of freedom (length of ``bias_dof``) for invariant irreps."""
        return self._num_bias_dof
