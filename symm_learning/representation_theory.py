"""Symmetric Learning - Representation Theory Utilities.

Tools for working with group representations, homomorphism bases, and irreducible
decompositions. These utilities support the construction of equivariant neural network
layers and the analysis of symmetric vector spaces.

Classes
-------
GroupHomomorphismBasis
    Handle bases of equivariant linear maps between representation spaces.

Functions
---------
isotypic_decomp_rep
    Decompose a representation into isotypic components.
direct_sum
    Direct sum of representations.
irreps_stats
    Statistics about irreducible representations in a representation.
escnn_representation_form_mapping
    Map between ESCNN representation forms.
is_complex_irreducible
    Check if a representation is complex irreducible.
decompose_representation
    Full decomposition of a representation into irreducibles.
"""

from __future__ import annotations

import functools
import itertools
import logging
from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
from escnn.group import Group, GroupElement, Representation
from scipy.linalg import block_diag

from symm_learning.utils import CallableDict

logger = logging.getLogger(__name__)

# Global cache for basis elements storage.
_cache_ = {}


class GroupHomomorphismBasis(torch.nn.Module):
    r"""Basis handler for :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`.

    This module provides:

    1. Synthesis of equivariant maps from basis coefficients.
    2. Orthogonal projection of dense matrices onto the equivariant subspace.

    Using the :ref:`isotypic decomposition <isotypic-decomposition-example>`
    (:func:`~symm_learning.representation_theory.isotypic_decomp_rep`) of
    :math:`\rho_{\mathcal{X}}` and :math:`\rho_{\mathcal{Y}}`, each shared irrep type :math:`k` contributes a block
    with structure

    .. math::
        \mathbf{A}^{(k)}_{o,i} = \sum_{s=1}^{S_k}\theta^{(k)}_{s,o,i}\,\mathbf{\Psi}^{(k)}_s,

    where :math:`\{\mathbf{\Psi}^{(k)}_s\}_{s=1}^{S_k}` is a basis of
    :math:`\operatorname{End}_\mathbb{G}(\hat{\rho}_k)`. This induces
    :math:`m^{\text{out}}_k m^{\text{in}}_k S_k` degrees of freedom for type :math:`k`.

    Core utilities:

    - :meth:`forward`: map basis coefficients to a dense equivariant matrix
      :math:`\mathbf{W}\in\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}},\rho_{\mathcal{Y}})`.
    - :meth:`orthogonal_projection`: project any dense matrix onto
      :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}},\rho_{\mathcal{Y}})` in Frobenius norm.
    - :meth:`initialize_params`: sample valid initialization either in coefficient space or as a dense equivariant
      matrix.

    .. rubric:: Examples

    Build and synthesize an equivariant matrix with :meth:`forward`.

    .. code-block:: pycon

        >>> from escnn.group import CyclicGroup
        >>> G = CyclicGroup(4)
        >>> rep = direct_sum([G.regular_representation] * 2)
        >>> basis = GroupHomomorphismBasis(rep, rep)
        >>> w_dof = torch.randn(basis.dim)
        >>> W = basis(w_dof)
        >>> # W satisfies rho(g) @ W == W @ rho(g) for all g in G

    Project an unconstrained matrix with :meth:`orthogonal_projection`.

    .. code-block:: pycon

        >>> W0 = torch.randn(rep.size, rep.size)
        >>> W_proj = basis.orthogonal_projection(W0)
        >>> # W_proj is the closest equivariant matrix to W0 in Frobenius norm

    Sample initialization with :meth:`initialize_params`.

    .. code-block:: pycon

        >>> w0 = basis.initialize_params("xavier_normal", return_dense=False)
        >>> W0 = basis.initialize_params("xavier_normal", return_dense=True)
        >>> # both parameterize equivariant maps

    Attributes:
        G (:class:`~escnn.group.group.Group`): Symmetry group shared by in_rep and out_rep.
        in_rep (:class:`~escnn.group.representation.Representation`): Input representation
            :math:`\rho_{\mathcal{X}}` rewritten in an isotypic basis.
        out_rep (:class:`~escnn.group.representation.Representation`): Output representation
            :math:`\rho_{\mathcal{Y}}` rewritten in an isotypic basis.
        basis_expansion (:class:`str`): Strategy ("memory_heavy" or "isotypic_expansion") controlling
            storage/perf trade-offs.
        common_irreps (list[tuple]): Irrep identifiers present in both in_rep and out_rep.
        iso_blocks (`dict <https://docs.python.org/3/library/stdtypes.html#dict>`_): Per-irrep metadata for
            irreps shared by in_rep/out_rep, keys:

            - out_slice / in_slice: slice selecting the isotypic coordinates in out/in reps.
            - mul_out / mul_in: multiplicities of the irrep in out/in reps.
            - irrep_dim: Dimension of the irreducible representation :math:`d_k`.
            - endomorphism_basis: basis of :math:`\operatorname{End}_\mathbb{G}(\hat{\rho}_k)` with shape
              :math:`(S_k, d_k, d_k)`.
            - dim_hom_basis: Dimension of the homomorphism between isotypic spaces of type :math:`k`, i.e.
              :math:`\dim(\operatorname{Hom}_\mathbb{G}(\rho_k^{\mathcal{X}}, \rho_k^{\mathcal{Y}})) = m_{out}
              \cdot m_{in} \cdot S_k`.
            - hom_basis_slice: slice containing the degrees of freedom associated to
              :math:`\dim(\operatorname{Hom}_\mathbb{G}(\rho_k^{\mathcal{X}}, \rho_k^{\mathcal{Y}}))` from a vector of
              shape :math:`(\dim(\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})),)`.
        basis_elements (:class:`~torch.Tensor`): Full dense basis stack
            ``(dim, out_rep.size, in_rep.size)`` when ``basis_expansion="memory_heavy"``.
        basis_norm_sq (:class:`~torch.Tensor`): Squared Frobenius norms of basis elements when
            ``basis_expansion="memory_heavy"``.
        endo_basis_flat_<irrep_id> (:class:`~torch.Tensor`): Per-irrep flattened endomorphism bases
            ``(S_k, d_k*d_k)`` when ``basis_expansion="isotypic_expansion"``.
        endo_basis_norm_sq_<irrep_id> (:class:`~torch.Tensor`): Per-irrep squared norms of the flattened
            endomorphism bases with isotypic expansion.
        Q_in_inv (:class:`~torch.Tensor`): Change-of-basis matrix :math:`Q_{in}^{-1}` cached as buffer for
            isotypic expansion.
        Q_out (:class:`~torch.Tensor`): Change-of-basis matrix :math:`Q_{out}` cached as buffer for
            isotypic expansion.
    """

    def __init__(
        self, in_rep: Representation, out_rep: Representation, basis_expansion: str = "isotypic_expansion"
    ) -> None:
        r"""Construct the equivariant basis and cache block-wise metadata.

        Args:
            in_rep: Input representation :math:`\rho_{\mathcal{X}}` whose
                :ref:`isotypic decomposition <isotypic-decomposition-example>` has size `in_rep.size`.
                The change of basis is applied in-place and used to define the rows of the basis matrices.
            out_rep: Output representation :math:`\rho_{\mathcal{Y}}`, treated analogously to `in_rep` with total size
                `out_rep.size`.
            basis_expansion: Strategy for realizing the basis. ``"memory_heavy"`` stores the full stack of basis
              elements with shape `(dim(Hom_G(in_rep, out_rep)), out_rep.size, in_rep.size)`, maximizing speed at the
              cost of one dense matrix per basis element. ``"isotypic_expansion"`` keeps only the tiny irrep
              endomorphism bases `(S_k, d_k, d_k)` and reconstructs on the fly—much lighter on memory, moderately
              slower.

        Raises:
            AssertionError: If `in_rep` and `out_rep` do not belong to the same symmetry group.
        """
        super().__init__()
        assert in_rep.group == out_rep.group, f"in group: {in_rep.group} != out group: {out_rep.group}"
        self.G = in_rep.group
        self.basis_expansion = basis_expansion
        self.in_rep = isotypic_decomp_rep(in_rep)
        self.out_rep = isotypic_decomp_rep(out_rep)
        dtype = torch.get_default_dtype()

        # Common irreps defining the only non-zero blocks in isotypic basis of Hom_G(in_rep, out_rep)
        self.common_irreps = sorted(set(self.in_rep.irreps).intersection(set(self.out_rep.irreps)))

        self.iso_blocks = {}

        dof_offset = 0
        for irrep_id in self.common_irreps:
            irrep = self.G.irrep(*irrep_id)
            # Slices defining the location of the isotypic subspace block in Hom_G(in_rep, out_rep)
            out_slice = self.out_rep.attributes["isotypic_subspace_dims"][irrep_id]
            in_slice = self.in_rep.attributes["isotypic_subspace_dims"][irrep_id]
            # Multiplicities of the irrep of type k=irrep_id in in_rep and out_rep
            mul_out = self.out_rep._irreps_multiplicities[irrep_id]
            mul_in = self.in_rep._irreps_multiplicities[irrep_id]
            # Endomorphism basis of the irrep End_G(k=irrep_id)
            irrep_end_basis = torch.tensor(irrep.endomorphism_basis(), dtype=dtype)  # [D_k, d_k, d_k]
            # Dimension of basis of homomorphisms between the two isotypic subspaces of type k=irrep_id
            dim_irrep_hom_basis = mul_out * mul_in * irrep_end_basis.size(0)  # dim(Hom_G(V_k^{in}, V_k^{out}))
            # Store block info for inference time use.
            self.iso_blocks[irrep_id] = dict(
                out_slice=out_slice,
                in_slice=in_slice,
                endomorphism_basis=irrep_end_basis,
                mul_out=mul_out,
                mul_in=mul_in,
                irrep_dim=irrep.size,
                dim_hom_basis=dim_irrep_hom_basis,
                hom_basis_slice=slice(dof_offset, dof_offset + dim_irrep_hom_basis),
            )
            dof_offset += dim_irrep_hom_basis
        # Total dimension of the homomorphism space Hom_G(in_rep, out_rep)
        self._dim_homomorphism = dof_offset

        if self.basis_expansion == "memory_heavy":
            # Construct full basis of shape (dim(Hom_G(in_rep, out_rep)), out_rep.size, in_rep.size)
            basis_elements = self._build_fullsize_homomorphism_basis()
            # Register contiguous buffer such that flattening returns a view.
            # This register is memory heavy as fuck, but enables ultra fast forward/backward passes.
            self.register_buffer("basis_elements", basis_elements.contiguous())
            basis_norm_sq = torch.einsum("sab,sab->s", self.basis_elements, self.basis_elements)
            self.register_buffer("basis_norm_sq", basis_norm_sq)
        elif self.basis_expansion == "isotypic_expansion":
            # We store as buffers only each irrep endomorphism basis.
            # This is ultra memory efficient, but mildly slower at runtime.
            for irrep_id, irrep_metadata in self.iso_blocks.items():
                endo_basis = irrep_metadata["endomorphism_basis"].contiguous()  # [S_k, d_k, d_k]
                endo_basis_flat = endo_basis.view(endo_basis.size(0), -1)
                self.register_buffer(f"endo_basis_flat_{irrep_id}", endo_basis_flat)
                endo_basis_norm_sq = torch.einsum("sd,sd->s", endo_basis_flat, endo_basis_flat)
                self.register_buffer(f"endo_basis_norm_sq_{irrep_id}", endo_basis_norm_sq)
            self.register_buffer("Q_in_inv", torch.tensor(self.in_rep.change_of_basis_inv, dtype=dtype))
            self.register_buffer("Q_out", torch.tensor(self.out_rep.change_of_basis, dtype=dtype))
        else:
            raise NotImplementedError(f"Basis expansion '{self.basis_expansion}' not implemented yet.")

    def forward(self, w_dof: torch.Tensor) -> torch.Tensor:
        r"""Return equivariant linear map from coefficients.

        The returned matrix satisfies

        .. math::
            \rho_{\mathcal{Y}}(g)\mathbf{W} = \mathbf{W}\rho_{\mathcal{X}}(g), \quad \forall g\in\mathbb{G}.

        Args:
            w_dof (:class:`~torch.Tensor`): Basis expansion coefficients of shape ``(D,)`` or ``(..., D)``, where
                ``D = dim(Hom_G(in_rep, out_rep))``.

        Returns:
            :class:`~torch.Tensor`: Dense matrix of shape
            ``(out_rep.size, in_rep.size)`` or ``(B, out_rep.size, in_rep.size)``.

        Example:
            >>> W = basis(w_dof)
            >>> # equivariance constraint:
            >>> # rho_out(g) @ W == W @ rho_in(g) for all g
        """
        assert w_dof.shape[-1] == (self.dim), f"Expected w_dof shape (B,{self.dim}), but got {w_dof.shape}"
        leading_shape = w_dof.shape[:-1]

        if self.basis_expansion == "memory_heavy":
            W_flat = torch.matmul(w_dof, self.basis_elements.view(self.dim, -1))  # [out_rep.size * in_rep.size]
            W = W_flat.view(*leading_shape, self.out_rep.size, self.in_rep.size)
        elif self.basis_expansion == "isotypic_expansion":
            W = w_dof.new_zeros(*leading_shape, self.out_rep.size, self.in_rep.size)
            for irrep_id, irrep_metadata in self.iso_blocks.items():
                m_out, m_in = irrep_metadata["mul_out"], irrep_metadata["mul_in"]
                out_slice, in_slice = irrep_metadata["out_slice"], irrep_metadata["in_slice"]
                d_k = irrep_metadata["irrep_dim"]
                hom_basis_slice = irrep_metadata["hom_basis_slice"]

                endo_basis_flat = getattr(self, f"endo_basis_flat_{irrep_id}")  # [S_k, d_k * d_k]
                theta_k = w_dof[..., hom_basis_slice].view(
                    *leading_shape, m_out * m_in, endo_basis_flat.size(0)
                )  # [*, m_out*m_in, S_k]
                # Compute basis expansion for this irrep block
                coeffs = torch.matmul(theta_k, endo_basis_flat)  # [*, m_out*m_in, d_k*d_k]
                # [m_out*m_in, d_k*d_k] -> [m_out, m_in, d_k, d_k] -> [m_out, d_k, m_in, d_k] -> [m_out*d_k, m_in*d_k]
                block = coeffs.view(*leading_shape, m_out, m_in, d_k, d_k)
                block = block.permute(*range(len(leading_shape)), -4, -2, -3, -1).reshape(
                    *leading_shape, m_out * d_k, m_in * d_k
                )
                W[..., out_slice, in_slice] = block
            W = self.Q_out @ (W @ self.Q_in_inv)

        return W

    def orthogonal_projection(self, W: torch.Tensor) -> torch.Tensor:
        r"""Project a dense matrix onto :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`.

        The projection :math:`\Pi_{\mathrm{Hom}}(\mathbf{W})` is orthogonal under the Frobenius inner product and
        enforces

        .. math::
            \rho_{\mathcal{Y}}(g)\,\Pi_{\mathrm{Hom}}(\mathbf{W})
            = \Pi_{\mathrm{Hom}}(\mathbf{W})\,\rho_{\mathcal{X}}(g),
            \quad \forall g\in\mathbb{G}.

        Args:
            W (:class:`~torch.Tensor`): Weight matrix of shape ``(..., out_rep.size, in_rep.size)`` in the
                original basis.

        Returns:
            :class:`~torch.Tensor`: Projection of ``W`` onto the equivariant subspace, matching the input shape.

        Note:
            The projection is orthogonal with respect to the Frobenius inner product. The selected
            ``basis_expansion`` controls how the projection is computed (speed vs. memory) but not the result.

        Example:
            >>> W_proj = basis.orthogonal_projection(W)
            >>> # W_proj satisfies:
            >>> # rho_out(g) @ W_proj == W_proj @ rho_in(g) for all g
            >>> # and is idempotent under projection:
            >>> # basis.orthogonal_projection(W_proj) == W_proj
        """
        assert W.shape[-2:] == (self.out_rep.size, self.in_rep.size), (
            f"Expected weight shape (..., {self.out_rep.size}, {self.in_rep.size}), got {W.shape}"
        )

        if self.basis_expansion == "memory_heavy":
            basis_flat = self.basis_elements.view(self.dim, -1)  # [S, d_out*d_in]
            W_flat = W.view(*W.shape[:-2], -1)  # [..., d_out*d_in]
            # coeff = <W, B_s> / ||B_s||^2, yields [..., S]
            coeff = W_flat.matmul(basis_flat.mT) / self.basis_norm_sq
            # Recompose: sum_s coeff_s * B_s, returning [..., d_out*d_in]
            W_proj_flat = coeff.matmul(basis_flat)
            return W_proj_flat.view(*W.shape)

        if self.basis_expansion == "isotypic_expansion":
            Q_out_inv = self.Q_out.mT
            Q_in = self.Q_in_inv.mT
            W_iso_in = (Q_out_inv @ W) @ Q_in  # [..., d_out, d_in] in isotypic basis
            W_iso = torch.zeros_like(W_iso_in)  # accumulator in isotypic basis
            leading_shape = W_iso_in.shape[:-2]  # batch (possibly empty) dims

            for irrep_id, irrep_metadata in self.iso_blocks.items():
                m_out, m_in = irrep_metadata["mul_out"], irrep_metadata["mul_in"]
                out_slice, in_slice = irrep_metadata["out_slice"], irrep_metadata["in_slice"]
                d_k = irrep_metadata["irrep_dim"]
                endo_basis_flat = getattr(self, f"endo_basis_flat_{irrep_id}")
                endo_norm_sq = getattr(self, f"endo_basis_norm_sq_{irrep_id}")

                block = W_iso_in[..., out_slice, in_slice]
                block = block.view(*leading_shape, m_out, d_k, m_in, d_k)
                block = block.permute(*range(len(leading_shape)), -4, -2, -3, -1)  # [..., m_out, m_in, d_k, d_k]
                block_flat = block.reshape(*leading_shape, m_out * m_in, d_k * d_k)  # [..., m_out*m_in, d_k^2]

                # coeff[..., (o,i), s] = <block_{o,i}, E_s> / ||E_s||^2
                coeff = block_flat.matmul(endo_basis_flat.mT)
                coeff = coeff / endo_norm_sq
                block_proj_flat = coeff.matmul(endo_basis_flat)  # [..., m_out*m_in, d_k^2]

                block_proj = block_proj_flat.view(*leading_shape, m_out, m_in, d_k, d_k)
                block_proj = block_proj.permute(*range(len(leading_shape)), -4, -2, -3, -1)
                block_proj = block_proj.reshape(*leading_shape, m_out * d_k, m_in * d_k)
                W_iso_block = W_iso[..., out_slice, in_slice]
                W_iso_block.copy_(block_proj)

            W_proj = (self.Q_out @ W_iso) @ self.Q_in_inv  # back to original basis
            return W_proj

        raise NotImplementedError(f"Basis expansion '{self.basis_expansion}' not implemented yet.")

    @property
    def dim(self) -> int:
        r"""Dimension :math:`\dim(\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}}))`."""
        return self._dim_homomorphism

    @torch.no_grad()
    def _build_fullsize_homomorphism_basis(self):
        r"""Construct the basis of :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`.

        Returns:
            ``basis_elements`` (:class:`~torch.Tensor`): Stack of homomorphism basis elements of shape
            `(dim(Hom_G(in_rep, out_rep)), out_rep.size, in_rep.size)`.

        Note:
            The dense basis is cached in module-level ``_cache_`` keyed by
            ``(self.in_rep, self.out_rep)``. Repeated calls with the same representation pair reuse the cached basis.
        """
        global _cache_
        cache_key = (self.in_rep, self.out_rep)
        if cache_key in _cache_:
            logger.debug(f"Using cached basis for Hom_G({self.in_rep}, {self.out_rep})")
            basis_elements = _cache_[cache_key]["basis_elements"]
        else:
            logger.debug(f"Building full-size basis for Hom_G({self.in_rep}, {self.out_rep})")
            basis_elements_iso_basis = []
            for _, irrep_metadata in self.iso_blocks.items():
                irrep_end_basis = irrep_metadata["endomorphism_basis"]
                dim_end_basis = irrep_end_basis.size(0)
                mul_out, mul_in = irrep_metadata["mul_out"], irrep_metadata["mul_in"]
                out_slice, in_slice = irrep_metadata["out_slice"], irrep_metadata["in_slice"]
                irrep_dim = irrep_metadata["irrep_dim"]
                # Construct the basis of Hom_G(V_k^{in}, V_k^{out}) of shape [S_k * m_out * m_in, d_out * d_in]
                # This is memory-intensive but the fastest way for forward/backward passes.
                irrep_basis_elements = []  # (S_k,  d_out, d_in)
                for s in range(dim_end_basis):
                    endo_basis_s = irrep_end_basis[s]  # [d_k, d_k]
                    for o_mul in range(mul_out):
                        for i_mul in range(mul_in):
                            basis_block = torch.zeros(self.out_rep.size, self.in_rep.size)  # [d_out, d_in]
                            row_start = out_slice.start + o_mul * irrep_dim
                            col_start = in_slice.start + i_mul * irrep_dim
                            basis_block[row_start : row_start + irrep_dim, col_start : col_start + irrep_dim] = (
                                endo_basis_s
                            )
                            irrep_basis_elements.append(basis_block)
                # Every block contributes exactly S_k * m_out * m_in rows; no need for an emptiness check.
                basis_elements_iso_basis.append(
                    torch.stack(irrep_basis_elements, dim=0)
                )  # [S_k*m_out_k*m_in_k, d_out, d_in]

            # Change basis elements from isotypic basis to original input/output basis
            basis_elements_iso_basis = torch.cat(
                basis_elements_iso_basis, dim=0
            )  # [dim(Hom_G(in_rep, out_rep)), d_out, d_in]
            if self.basis_expansion == "memory_heavy":
                Q_out = torch.tensor(self.out_rep.change_of_basis, dtype=basis_elements_iso_basis.dtype)
                Q_in_inv = torch.tensor(self.in_rep.change_of_basis_inv, dtype=basis_elements_iso_basis.dtype)
            else:
                Q_out, Q_in_inv = self.Q_out, self.Q_in_inv
            # basis elements of shape (dim(Hom_G(in_rep, out_rep)), d_out, d_in)
            basis_elements = torch.einsum("ab,sbc,cd->sad", Q_out, basis_elements_iso_basis, Q_in_inv).contiguous()
            # Store in global cache
            _cache_[cache_key] = {"basis_elements": basis_elements.cpu()}

        return basis_elements

    @torch.no_grad()
    def initialize_params(
        self, scheme: str = "kaiming_uniform", return_dense: bool = False, leading_shape: int | tuple | None = None
    ) -> torch.Tensor:
        r"""Sample valid parameters in :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\mathcal{X}},\rho_{\mathcal{Y}})`.

        Args:
            scheme (:class:`str`): Initialization scheme (``"xavier_normal"``, ``"xavier_uniform"``,
                ``"kaiming_normal"``, or ``"kaiming_uniform"``).
            return_dense: If ``True``, return dense weights in the original basis; otherwise return basis expansion
                coefficients.
            leading_shape: Optional leading dimensions (e.g., batch size or a tuple of dims). ``None`` yields no leading
                dims. Examples: ``None`` → ``(dim,)`` / ``(d_out, d_in)``; ``B`` → ``(B, dim)`` / ``(B, d_out, d_in)``.

        Shapes:
            - return_dense=False → ``(*leading_shape, dim(Hom_G(out_rep, in_rep)))``
            - return_dense=True  → ``(*leading_shape, out_rep.size, in_rep.size)``

        Returns:
            :class:`~torch.Tensor`: Initialized parameters with the shapes above.

        Notes:
            - If ``return_dense=False``, output lives in coefficient space and can be passed to :meth:`forward`.
            - If ``return_dense=True``, returned dense matrices already satisfy
              :math:`\rho_{\mathcal{Y}}(g)\mathbf{W}=\mathbf{W}\rho_{\mathcal{X}}(g)` for all :math:`g`.

        Example:
            >>> w_dof = basis.initialize_params("xavier_uniform")
            >>> W = basis(w_dof)
            >>> W2 = basis.initialize_params("xavier_uniform", return_dense=True)
        """
        if leading_shape is None:
            leading_shape = ()
        elif isinstance(leading_shape, int):
            leading_shape = (leading_shape,)
        else:
            leading_shape = tuple(leading_shape)

        buffer = next(self.buffers(), None)
        device = buffer.device if buffer is not None else None
        dtype = buffer.dtype if buffer is not None else torch.get_default_dtype()

        if return_dense:
            W_iso = torch.zeros((*leading_shape, self.out_rep.size, self.in_rep.size), dtype=dtype, device=device)
        else:
            w_dof = torch.zeros((*leading_shape, self.dim), dtype=dtype, device=device)

        for _, irrep_metadata in self.iso_blocks.items():
            # for out_slice, in_slice, endo_basis, m_out, m_in, irrep_dim in self.irreps_meta:
            endo_basis = irrep_metadata["endomorphism_basis"]
            m_out, m_in = irrep_metadata["mul_out"], irrep_metadata["mul_in"]
            out_slice, in_slice = irrep_metadata["out_slice"], irrep_metadata["in_slice"]
            irrep_dim = irrep_metadata["irrep_dim"]
            hom_basis_slice = irrep_metadata["hom_basis_slice"]

            dim_endo_basis = endo_basis.size(0)
            dtype, device = endo_basis.dtype, endo_basis.device
            # fans for this irrep
            fan_in = dim_endo_basis * m_in
            fan_out = dim_endo_basis * m_out
            # isotypic_param_shape := (dim_irrep_endomorphism, irep_multiplicity_out, irrep_multiplicity_in)
            isotypic_param_shape = (*leading_shape, dim_endo_basis, m_out, m_in)
            if scheme == "xavier_uniform":
                bound = (6.0 / (fan_in + fan_out)) ** 0.5
                theta = torch.empty(isotypic_param_shape, device=device, dtype=dtype).uniform_(-bound, bound)
            elif scheme == "xavier_normal":
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                theta = torch.empty(isotypic_param_shape, device=device, dtype=dtype).normal_(0.0, std)
            elif scheme in {"kaiming_normal", "he_normal"}:
                std = (2.0 / fan_in) ** 0.5
                theta = torch.empty(isotypic_param_shape, device=device, dtype=dtype).normal_(0.0, std)
            elif scheme in {"kaiming_uniform", "he_uniform"}:
                bound = (6.0 / fan_in) ** 0.5
                theta = torch.empty(isotypic_param_shape, device=device, dtype=dtype).uniform_(-bound, bound)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            if return_dense:
                # Expand irrep block (m_out * irrep_dim, m_in * irrep_dim)
                block = torch.einsum("...soi,sab->...oaib", theta, endo_basis).reshape(
                    *leading_shape, m_out * irrep_dim, m_in * irrep_dim
                )
                W_iso[..., out_slice, in_slice] = block
            else:
                w_dof[..., hom_basis_slice] = theta.reshape(*leading_shape, -1)

        if return_dense:  # Change to original coordinates
            Q_in_inv = torch.tensor(self.in_rep.change_of_basis_inv, dtype=W_iso.dtype, device=W_iso.device)
            Q_out = torch.tensor(self.out_rep.change_of_basis, dtype=W_iso.dtype, device=W_iso.device)
            return torch.einsum("ab,...bc,cd->...ad", Q_out, W_iso, Q_in_inv)
        else:
            return w_dof

    def extra_repr(self):  # noqa: D102
        return f"basis_expansion={self.basis_expansion}\nin_rep={self.in_rep}\nout_rep={self.out_rep}"


def isotypic_decomp_rep(rep: Representation) -> Representation:
    r"""Return an equivalent representation disentangled into isotypic subspaces.

    Given an input :class:`~escnn.group.representation.Representation`, this function computes an equivalent
    representation by
    updating the change of basis (and its inverse) and reordering the irreducible representations. The returned
    representation is guaranteed to be disentangled into its isotypic subspaces.

    A representation is considered disentangled if, in its spectral basis, the irreducible representations (irreps) are
    clustered by type, i.e., all irreps of the same type are consecutive:

    .. math::
        \rho_{\mathcal{X}} = \mathbf{Q}\left(
        \bigoplus_{k\in[1,n_{\text{iso}}]}
        \bigoplus_{i\in[1,n_k]}
        \hat{\rho}_k
        \right)\mathbf{Q}^T

    where :math:`\hat{\rho}_k` is the irreducible representation of type :math:`k`,
    and :math:`n_k` is its multiplicity.

    The change of basis decomposes the representation space into orthogonal isotypic subspaces:

    .. math::
        \mathcal{X} = \bigoplus_{k\in[1,n_{\text{iso}}]} \mathcal{X}^{(k)}.

    Ordering convention:

    - If present, the trivial-irrep isotypic block is placed first and denoted :math:`\mathcal{X}^{\text{inv}}`.
    - Remaining isotypic blocks are sorted by subspace dimension.

    Output metadata:

    - ``attributes["isotypic_reps"]``: ordered mapping ``irrep_id -> isotypic representation``.
    - ``attributes["isotypic_subspace_dims"]``: slice map locating each isotypic block in the disentangled basis.
    - ``attributes["in_isotypic_basis"]``: boolean flag set to ``True``.

    Args:
        rep (:class:`~escnn.group.representation.Representation`): The input representation :math:`\rho`.

    Returns:
        :class:`~escnn.group.representation.Representation`: An equivalent, disentangled representation.

    Note:
        The decomposition is cached per symmetry group under ``rep.group.representations`` with key
        ``rep.name + "-Iso"``. Repeated calls with the same representation/group reuse the cached result.
    """
    # If group representation is already disentangled (in isotypic basis), return it without changes.
    if rep.attributes.get("in_isotypic_basis", False):
        logger.debug(f"Representation {rep.name} is already in isotypic basis, returning as is.")
        return rep

    symm_group = rep.group
    iso_rep_name = rep.name + "-Iso"
    if iso_rep_name in symm_group.representations:
        logger.debug(f"Returning cached {iso_rep_name}")
        return symm_group.representations[iso_rep_name]

    logger.debug(f"Computing isotypic decomposition of representation {rep.name}")
    potential_irreps = rep.group.irreps()
    isotypic_subspaces_indices = {irrep.id: [] for irrep in potential_irreps}

    for pot_irrep in potential_irreps:
        cur_dim = 0
        for rep_irrep_id in rep.irreps:
            rep_irrep = symm_group.irrep(*rep_irrep_id)
            if rep_irrep == pot_irrep:
                isotypic_subspaces_indices[rep_irrep_id].append(list(range(cur_dim, cur_dim + rep_irrep.size)))
            cur_dim += rep_irrep.size

    # Remove inactive Isotypic Spaces
    for irrep in potential_irreps:
        if len(isotypic_subspaces_indices[irrep.id]) == 0:
            del isotypic_subspaces_indices[irrep.id]

    # Each Isotypic Space will be indexed by the irrep it is associated with.
    active_isotypic_reps = {}
    for irrep_id, indices in isotypic_subspaces_indices.items():
        irrep = symm_group.irrep(*irrep_id)
        multiplicities = len(indices)
        active_isotypic_reps[irrep_id] = Representation(
            group=rep.group,
            irreps=[irrep_id] * multiplicities,
            name=f"{rep.name}-IsoSubspace{irrep_id}",
            change_of_basis=np.identity(irrep.size * multiplicities),
            supported_nonlinearities=irrep.supported_nonlinearities,
        )
        # Store attributes used in stats/linalg operations.
        active_isotypic_reps[irrep_id].attributes["is_isotypic_rep"] = True
        active_isotypic_reps[irrep_id].attributes["irrep_multiplicity"] = multiplicities
        active_isotypic_reps[irrep_id].attributes["irrep_dim"] = irrep.size
        active_isotypic_reps[irrep_id].attributes["irrep_endomorphism_basis"] = torch.tensor(irrep.endomorphism_basis())

    # Impose canonical order on the Isotypic Subspaces.
    # If the trivial representation is active it will be the first Isotypic Subspace.
    # Then sort by dimension of the space from smallest to largest.
    ordered_isotypic_reps = OrderedDict(sorted(active_isotypic_reps.items(), key=lambda item: item[1].size))
    if symm_group.trivial_representation.id in ordered_isotypic_reps:
        ordered_isotypic_reps.move_to_end(symm_group.trivial_representation.id, last=False)

    # Required permutation to change the order of the irreps. So we obtain irreps of the same type consecutively.
    oneline_permutation = []
    for irrep_id, iso_rep in ordered_isotypic_reps.items():
        idx = isotypic_subspaces_indices[irrep_id]
        oneline_permutation.extend(idx)
    oneline_permutation = np.concatenate(oneline_permutation)
    expected = np.arange(rep.size)
    if oneline_permutation.shape[0] != rep.size or not np.array_equal(np.sort(oneline_permutation), expected):
        raise RuntimeError(
            "Invalid isotypic permutation: indices must be a permutation of [0, ..., rep.size-1], "
            f"got size={oneline_permutation.shape[0]} and unique={np.unique(oneline_permutation).size}"
        )
    P_in2iso = permutation_matrix(oneline_permutation)

    Q_iso = rep.change_of_basis @ P_in2iso.T
    rep_iso_basis = direct_sum(list(ordered_isotypic_reps.values()), name=iso_rep_name, change_of_basis=Q_iso)

    # Get variable of indices of isotypic subspaces in the disentangled representation.
    d = 0
    iso_subspace_dims = {}
    for irrep_id, iso_rep in ordered_isotypic_reps.items():
        iso_subspace_dims[irrep_id] = slice(d, d + iso_rep.size)
        d += iso_rep.size
    if d != rep.size:
        raise RuntimeError(f"Isotypic slices cover {d} dimensions, expected {rep.size}")

    iso_supported_nonlinearities = [iso_rep.supported_nonlinearities for iso_rep in ordered_isotypic_reps.values()]
    rep_iso_basis.supported_nonlinearities = functools.reduce(set.intersection, iso_supported_nonlinearities)
    rep_iso_basis.attributes["isotypic_reps"] = ordered_isotypic_reps
    rep_iso_basis.attributes["isotypic_subspace_dims"] = iso_subspace_dims
    rep_iso_basis.attributes["in_isotypic_basis"] = True  # Boolean flag useful

    # Store representation in symmetry group cache:
    symm_group.representations[rep_iso_basis.name] = rep_iso_basis
    logger.debug(f"Stored isotypic decomposition representation {rep_iso_basis.name} in cache.")
    return rep_iso_basis


def direct_sum(
    reps: list[Representation], name: str | None = None, change_of_basis: np.ndarray | None = None
) -> Representation:
    r"""Return the direct sum :math:`\rho=\bigoplus_i \rho_i` of representations.

    If ``name`` is not provided, the function builds one by run-length encoding consecutive repeated names:

    - ``[rep_1, rep_1, rep_2, rep_1]`` -> ``"(rep_1 x 2)⊕rep_2⊕rep_1"``

    This preserves order and makes repeated contiguous blocks explicit.

    Args:
        reps (list[:class:`~escnn.group.representation.Representation`]): Summands in the direct sum.
        name (:class:`str`, optional): Name of the resulting representation. Auto-generated when ``None``.
        change_of_basis (:class:`~numpy.ndarray`, optional): Optional change of basis for the resulting representation.

    Returns:
        :class:`~escnn.group.representation.Representation`: Direct-sum representation with
        ``attributes["direct_sum_reps"]`` set.
    """
    from escnn.group import directsum

    if len(reps) == 1:
        return reps[0]

    # Name is a string of original representation names, where continuous multiplicities of a
    # representaton are groups [rep_1, rep_1, rep_2, rep_1] -> [rep_1 x 2, rep_2, rep_1]
    if name is None:
        rep_names = []
        for rep_id, multiplicities in itertools.groupby(reps, lambda r: r.name):
            m_list = list(multiplicities)
            if len(m_list) > 1:
                rep_names.append(f"({rep_id} x {len(m_list)})")
            else:
                rep_names.append(rep_id)
        name = "⊕".join(rep_names)

    out_rep = directsum(reps, name=name, change_of_basis=change_of_basis)
    out_rep.attributes["direct_sum_reps"] = reps
    return out_rep


def permutation_matrix(oneline_notation):
    r"""Generate a permutation matrix from one-line notation.

    If ``oneline_notation = [p_0,\dots,p_{d-1}]``, the returned matrix :math:`\mathbf{P}` satisfies
    :math:`(\mathbf{P}\mathbf{x})_i = x_{p_i}`.

    Example:
        >>> permutation_matrix([2, 0, 1])
        array([[0, 0, 1],
               [1, 0, 0],
               [0, 1, 0]])
    """
    d = len(oneline_notation)
    oneline_notation = np.asarray(oneline_notation, dtype=int)
    if d != np.unique(oneline_notation).size:
        raise ValueError("oneline_notation must describe a non-defective permutation")
    if oneline_notation.min() < 0 or oneline_notation.max() >= d:
        raise ValueError(f"oneline_notation entries must be in [0, {d - 1}]")
    P = np.zeros((d, d), dtype=int)
    P[np.arange(d), oneline_notation] = 1
    return P


def irreps_stats(irreps_ids):
    r"""Compute summary statistics for a sequence of irrep identifiers.

    Args:
        irreps_ids (:class:`typing.Iterable`): Sequence of ESCNN irrep IDs.

    Returns:
        tuple:
            - unique_ids: unique irrep IDs (sorted by numpy lexical order over their string representation).
            - counts: multiplicity of each unique irrep.
            - indices: first position where each unique irrep appears.
    """
    str_ids = [str(irrep_id) for irrep_id in irreps_ids]
    unique_str_ids, counts, indices = np.unique(str_ids, return_counts=True, return_index=True)
    unique_ids = [eval(s) for s in unique_str_ids]
    return unique_ids, counts, indices


def escnn_representation_form_mapping(
    group: Group,
    rep: dict[GroupElement, np.ndarray] | Callable[[GroupElement], np.ndarray],
    name: str = "reconstructed",
):
    r"""Reconstruct a representation from a map :math:`\mathbb{G}\to\mathrm{GL}(\mathcal{X})`.

    Given a representation map :math:`\rho: \mathbb{G}\to\mathrm{GL}(\mathcal{X})`, this function identifies the
    irreducible ESCNN decomposition and returns an equivalent
    :class:`~escnn.group.representation.Representation` object.

    Args:
        group (:class:`~escnn.group.group.Group`): Symmetry group of the representation.
        rep (dict | collections.abc.Callable): A :class:`~typing.Union`-style input.
            Either a `dict <https://docs.python.org/3/library/stdtypes.html#dict>`_ or a
            :class:`collections.abc.Callable` returning the matrix :math:`\rho(g)` for each
            :class:`~escnn.group.group.GroupElement` ``g``.
        name (:class:`str`, optional): Name of the representation. Defaults to 'reconstructed'.

    Returns:
        representation (:class:`~escnn.group.representation.Representation`): Reconstructed ESCNN representation
            instance.

    Note:
        Matrices must define a valid representation (invertible and group-consistent).
        If a dictionary is provided, keys must be ESCNN's group elements from ``group.elements``.

    Example:
        >>> from escnn.group import CyclicGroup
        >>> G = CyclicGroup(4)
        >>> rep_map = {g: G.regular_representation(g) for g in G.elements}
        >>> rep = escnn_representation_form_mapping(G, rep_map, name="C4-regular-rec")
    """
    if isinstance(rep, dict):
        from symm_learning.utils import CallableDict

        rep = CallableDict(rep)
    else:
        rep = rep

    # Find Q such that `iso_cplx(g) = Q @ rep(g) @ Q^-1` is block diagonal with blocks being complex irreps.
    cplx_irreps, Q = cplx_isotypic_decomposition(group, rep)
    # Get the size and location of each cplx irrep in `iso_cplx(g)`
    cplx_irreps_size = [irrep(group.sample()).shape[0] for irrep in cplx_irreps]
    irrep_dim_start = np.cumsum([0] + cplx_irreps_size[:-1])
    # Compute the character table of the found complex irreps and of all complex irreps of G
    irreps_char_table = compute_character_table(group, cplx_irreps)

    # We need to identify which real ESCNN irreps are present in rep(g).
    # First, we decompose the Group's ESCNN real irreps into complex irreps.
    escnn_cplx_irreps_data = {}
    for re_irrep in group.irreps():
        # Find Q_sub s.t. `block_diag([cplx_irrep_i1(g), cplx_irrep_i2(g)...]) = Q @ re_irrep_i(g) @ Q^-1`
        irreps, Q_sub = cplx_isotypic_decomposition(group, re_irrep)
        char_table = compute_character_table(group, irreps)
        escnn_cplx_irreps_data[re_irrep] = dict(subreps=irreps, Q=Q_sub, char_table=char_table)

    # Then, we find which of the Group complex irreps are present in the input representation, and determine
    # each group complex irrep multiplicity. As the complex irreps forming a real irrep can be spread over the
    # dimensions of the input rep, we find a permutation matrix P such that all complex irreps associated with a real
    # irrep are contiguous in dimensions.trifinger
    oneline_perm, Q_isore2isoimg = [], []
    escnn_real_irreps = []
    for escnn_irrep, data in escnn_cplx_irreps_data.items():
        # Match complex irreps by their character tables.
        multiplicities, irrep_locs = map_character_tables(data["char_table"], irreps_char_table)
        subreps_start_dims = [irrep_dim_start[i] for i in irrep_locs]  # Identify start of blocks in `rep(g)`
        data.update(multiplicities=multiplicities, subrep_start_dims=subreps_start_dims)
        assert np.unique(multiplicities).size == 1, "Multiplicities error"
        multiplicity = multiplicities[0]
        for m in range(multiplicity):
            Q_isore2isoimg.append(data["Q"])  # Add transformation from Real irrep to complex irrep
            escnn_real_irreps.append(escnn_irrep)  # Add escnn irrep to the list for instanciation
            for subrep, rep_start_dims in zip(data["subreps"], subreps_start_dims):
                rep_size = subrep[group.sample()].shape[0] if isinstance(subrep, dict) else subrep.size
                oneline_perm += list(range(rep_start_dims[m], rep_start_dims[m] + rep_size))
    # As the complex irreps forming a real irrep can be spread over the dimensions of the input rep, we find a
    # permutation matrix P such that all complex irreps of a real irrep are contiguous in dimensions / in the same block
    P = permutation_matrix(oneline_notation=oneline_perm)
    # Then we use the known transformations `Q_sub` for each real irrep, to create a mapping from cplx to real irreps.
    # s.t. `iso_re(g) = Q_iso_cplx2iso_re @ block_diag([cplx_irrep_11(g),...,cplx_irrep_ij(g)]) @ Q_iso_cplx2iso_re^-1`
    Q_iso_cplx2iso_re = block_diag(*[Q_sub.conj().T for Q_sub in Q_isore2isoimg])

    # Assert the matrix `P` and `Q_iso_cplx2iso_re` turn complex irreps into real irreps
    # `iso_re(g) = (Q_iso_cplx2iso_re @ P) @ iso_cplx(g) @ (Q_iso_cplx2iso_re @ P)^-1`,
    for g in group.elements:
        iso_re_g = block_diag(*[irrep(g) for irrep in escnn_real_irreps])
        iso_cplx_g = block_diag(*[cplx_irrep(g) for cplx_irrep in cplx_irreps])
        rec_iso_re_g = (Q_iso_cplx2iso_re @ P) @ iso_cplx_g @ (Q_iso_cplx2iso_re @ P).conj().T
        error = np.abs(iso_re_g - rec_iso_re_g)
        assert np.isclose(error, 0).all(), "Error in the conversion of Real irreps to Complex irreps"

    # Now we have an orthogonal transformation between the input `rep` and `iso_re`.
    #                        |     iso_cplx(g)     |
    # (Q_iso_cplx2iso_re @ P @ Q) @ rep(g) @ (Q^-1 @ P^-1 @ Q_iso_cplx2iso_re^-1) = Q_re @ rep(g) @ Q_re^-1 = iso_re(g)
    Q_re = Q_iso_cplx2iso_re @ P @ Q

    assert np.allclose(Q_re @ Q_re.conj().T, np.eye(Q_re.shape[0])), "Q_re is not an orthogonal transformation"
    if np.allclose(np.imag(Q_re), 0):
        Q_re = np.real(Q_re)  # Remove numerical noise and ensure rep(g) is of dtype: float instead of cfloat

    # Then we have that `Q_re^-1 @ iso_re(g) @ Q_re = rep(g)`
    reconstructed_rep = Representation(
        group, name=name, irreps=[irrep.id for irrep in escnn_real_irreps], change_of_basis=Q_re.conj().T
    )

    # Test ESCNN reconstruction
    for g in group.elements:
        g_true, g_rec = rep(g), reconstructed_rep(g)
        error = np.abs(g_true - g_rec)
        error[error < 1e-10] = 0
        assert np.allclose(error, 0), f"Reconstructed rep do not match input rep. g={g}, error:\n{error}"
        assert np.allclose(np.imag(g_rec), 0), f"Reconstructed rep not real for g={g}: \n{g_rec}"

    return reconstructed_rep


def is_complex_irreducible(group: Group, rep: dict[GroupElement, np.ndarray] | Callable[[GroupElement], np.ndarray]):
    r"""Check complex irreducibility of :math:`\rho:\mathbb{G}\to\mathrm{GL}(\mathcal{X})`.

    By Schur's lemma, :math:`\rho` is complex-irreducible iff every Hermitian matrix commuting with all
    :math:`\rho(g)` is scalar. The routine searches for a non-scalar commuting Hermitian witness :math:`\mathbf{H}`.

    Args:
        group (:class:`~escnn.group.group.Group`): Symmetry group of the representation.
        rep (dict | collections.abc.Callable): A :class:`~typing.Union`-style input.
            It must be either a `dict <https://docs.python.org/3/library/stdtypes.html#dict>`_ or a
            :class:`collections.abc.Callable`.
            It must map each :class:`~escnn.group.group.GroupElement` to a matrix :class:`~numpy.ndarray`.

    Returns:
        tuple[:class:`bool`, :class:`~numpy.ndarray`]:
            - ``(True, I)`` if irreducible (identity witness),
            - ``(False, H)`` if reducible with a non-scalar commuting Hermitian witness.
    """
    if isinstance(rep, dict):

        def rep(g):
            return rep[g]

    else:
        rep = rep

    # Compute the dimension of the representation
    n = rep(group.sample()).shape[0]

    # Run through all r,s = 1,2,...,n
    for r in range(n):
        for s in range(n):
            # Define H_rs
            H_rs = np.zeros((n, n), dtype=complex)
            if r == s:
                H_rs[r, s] = 1
            elif r > s:
                H_rs[r, s] = 1
                H_rs[s, r] = 1
            else:  # r < s
                H_rs[r, s] = 1j
                H_rs[s, r] = -1j

            # Compute H
            H = sum([rep(g).conj().T @ H_rs @ rep(g) for g in group.elements]) / group.order()

            # If H is not a scalar matrix, then it is a matrix that commutes with all group actions.
            if not np.allclose(H[0, 0] * np.eye(H.shape[0]), H):
                return False, H
    # No Hermitian matrix was found to commute with all group actions. This is an irreducible rep
    return True, np.eye(n)


def decompose_representation(G: Group, rep: dict[GroupElement, np.ndarray] | Callable[[GroupElement], np.ndarray]):
    r"""Block-diagonalize :math:`\rho:\mathbb{G}\to\mathrm{GL}(\mathcal{X})` into invariant subspaces.

    Finds a unitary matrix :math:`\mathbf{Q}` such that

    .. math::
        \mathbf{Q}\rho(g)\mathbf{Q}^H = \operatorname{blockdiag}(\rho_1(g), \ldots, \rho_m(g)),
        \quad \forall g\in\mathbb{G},

    where each :math:`\rho_i` is an invariant subrepresentation. The algorithm recursively uses commuting-Hermitian
    witnesses and graph connected-components to obtain contiguous block structure.

    Args:
        G (:class:`~escnn.group.group.Group`): The symmetry group.
        rep (dict | collections.abc.Callable): A :class:`~typing.Union`-style input.
            It must be either a `dict <https://docs.python.org/3/library/stdtypes.html#dict>`_ or a
            :class:`collections.abc.Callable`.
            It must map each :class:`~escnn.group.group.GroupElement` to a matrix :class:`~numpy.ndarray`.

    Returns:
        tuple:
            - subreps (list[Callable]): Decomposed subrepresentations (not guaranteed sorted by dimension).
            - Q (:class:`~numpy.ndarray`): Unitary change-of-basis matrix.

    """
    import networkx as nx
    from networkx import Graph

    eps = 1e-12
    if isinstance(rep, dict):

        def rep(g):
            return rep[g]

    else:
        rep = rep
    # Compute the dimension of the representation
    n = rep(G.sample()).shape[0]

    for g in G.elements:  # Ensure the representation is unitary/orthogonal
        error = np.abs((rep(g) @ rep(g).conj().T) - np.eye(n))
        assert np.allclose(error, 0), f"Rep {rep} is not unitary: rep(g)@rep(g)^H=\n{(rep(g) @ rep(g).conj().T)}"

    # Find Hermitian matrix non-scalar `H` that commutes with all group actions
    is_irred, H = is_complex_irreducible(G, rep)
    if is_irred:
        return [rep], np.eye(n)

    # Eigen-decomposition of matrix `H = P·A·P^-1` reveals the G-invariant subspaces/eigenspaces of the representations.
    eivals, eigvects = np.linalg.eigh(H, UPLO="L")
    P = eigvects.conj().T
    assert np.allclose(P.conj().T @ np.diag(eivals) @ P, H)

    # Eigendcomposition is not guaranteed to block_diagonalize the representation. An additional permutation of the
    # rows and columns od the representation might be needed to produce a Jordan block canonical form.
    # First: We want to identify the diagonal blocks. To find them we use the trick of thinking of the representation
    # as an adjacency matrix of a graph. The non-zero entries of the adjacency matrix are the edges of the graph.
    edges = set()
    decomposed_reps = {}
    for g in G.elements:
        diag_rep = P @ rep(g) @ P.conj().T  # Obtain block diagonal representation
        diag_rep[np.abs(diag_rep) < eps] = 0  # Remove rounding errors.
        non_zero_idx = np.nonzero(diag_rep)
        edges.update([(x_idx, y_idx) for x_idx, y_idx in zip(*non_zero_idx)])
        decomposed_reps[g] = diag_rep

    # Each connected component of the graph is equivalent to the rows and columns determining a block in the diagonal
    graph = Graph()
    graph.add_edges_from(set(edges))
    connected_components = [sorted(list(comp)) for comp in nx.connected_components(graph)]
    connected_components = sorted(connected_components, key=lambda x: (len(x), min(x)))  # Impose a canonical order
    # If connected components are not adjacent dimensions, say subrep_1_dims = [0,2] and subrep_2_dims = [1,3] then
    # We permute them to get a jordan block canonical form. I.e. subrep_1_dims = [0,1] and subrep_2_dims = [2,3].
    oneline_notation = list(itertools.chain.from_iterable([list(comp) for comp in connected_components]))
    PJ = permutation_matrix(oneline_notation=oneline_notation)
    # After permuting the dimensions, we can assume the components are ordered in dimension
    ordered_connected_components = []
    idx = 0
    for comp in connected_components:
        ordered_connected_components.append(tuple(range(idx, idx + len(comp))))
        idx += len(comp)
    connected_components = ordered_connected_components

    # The output of connected components is the set of nodes/row-indices of the rep.
    subreps = [CallableDict() for _ in connected_components]
    for g in G.elements:
        for comp_id, comp in enumerate(connected_components):
            block_start, block_end = comp[0], comp[-1] + 1
            # Transform the decomposed representation into the Jordan Cannonical Form (jcf)
            jcf_rep = PJ @ decomposed_reps[g] @ PJ.T
            # Check Jordan Cannonical Form TODO: Extract this to a utils. function
            above_block = jcf_rep[0:block_start, block_start:block_end]
            below_block = jcf_rep[block_end:, block_start:block_end]
            left_block = jcf_rep[block_start:block_end, 0:block_start]
            right_block = jcf_rep[block_start:block_end, block_end:]

            assert np.allclose(above_block, 0) or above_block.size == 0, "Non zero elements above block"
            assert np.allclose(below_block, 0) or below_block.size == 0, "Non zero elements below block"
            assert np.allclose(left_block, 0) or left_block.size == 0, "Non zero elements left of block"
            assert np.allclose(right_block, 0) or right_block.size == 0, "Non zero elements right of block"
            sub_g = jcf_rep[block_start:block_end, block_start:block_end]
            subreps[comp_id][g] = sub_g

    # Decomposition to Jordan Canonical form is accomplished by (PJ @ P) @ rep @ (PJ @ P)^-1
    Q = PJ @ P

    # Test decomposition.
    for g in G.elements:
        jcf_rep = block_diag(*[subrep[g] for subrep in subreps])
        error = np.abs(jcf_rep - (Q @ rep(g) @ Q.conj().T))
        assert np.allclose(error, 0), f"Q @ rep[g] @ Q^-1 != block_diag[{[f'rep{i},' for i in range(len(subreps))]}]"

    return subreps, Q


def compute_character_table(G: Group, reps: list[dict[GroupElement, np.ndarray] | Representation]):
    """Computes the character table of a group for a given set of representations.

    Args:
        G (:class:`~escnn.group.group.Group`): Symmetry group.
        reps (:class:`list`): Representations (or representation mappings) for which characters are evaluated on all
            group elements. Each entry is a :class:`~typing.Union` of:
            - a mapping from :class:`~escnn.group.group.GroupElement` to :class:`~numpy.ndarray`, or
            - an :class:`~escnn.group.representation.Representation`.

    Returns:
        :class:`~numpy.ndarray`: The character table of shape `(n_reps, G.order())`.
    """
    n_reps = len(reps)
    table = np.zeros((n_reps, G.order()), dtype=complex)
    for i, rep in enumerate(reps):
        for j, g in enumerate(G.elements):
            table[i, j] = rep.character(g) if isinstance(rep, Representation) else np.trace(rep(g))
    return table


def map_character_tables(in_table: np.ndarray, reference_table: np.ndarray):
    """Find a representation of a group in the set of irreducible representations."""
    n_in_reps = in_table.shape[0]
    out_ids, multiplicities = [], []
    for in_id in range(n_in_reps):
        character_orbit = in_table[in_id, :]
        orbit_error = np.isclose(np.abs(reference_table - character_orbit), 0)
        match_idx = np.argwhere(np.all(orbit_error, axis=1)).flatten()
        multiplicity = len(match_idx)
        out_ids.append(match_idx), multiplicities.append(multiplicity)
    return multiplicities, out_ids


def cplx_isotypic_decomposition(G: Group, rep: Callable[[GroupElement], np.ndarray]):
    """Perform the :ref:`isotypic decomposition <isotypic-decomposition-example>` of a unitary representation.

    This routine decomposes the representation into complex irreducibles.

    Args:
        G (:class:`~escnn.group.group.Group`): Symmetry group of the representation.
        rep (dict | collections.abc.Callable): A :class:`~typing.Union`-style input.
            It must be either a `dict <https://docs.python.org/3/library/stdtypes.html#dict>`_ or a
            :class:`collections.abc.Callable`.
            It must map each :class:`~escnn.group.group.GroupElement` to a matrix :class:`~numpy.ndarray`.

    Returns:
        sorted_irreps (list[dict]): List of complex irreducible representations, sorted in ascending order of
            dimension.
        Q (:class:`~numpy.ndarray`): Hermitian matrix such that ``Q @ rep[g] @ Q^-1`` is block diagonal, with blocks
            ``sorted_irreps``.

    """
    if isinstance(rep, dict):

        def rep(g):
            return rep[g]

    else:
        rep = rep

    n = rep(G.sample()).shape[0]
    subreps, Q_internal = decompose_representation(G, rep)

    found_irreps = []
    Qs = []

    # Check if each subrepresentation can be further decomposed.
    for subrep in subreps:
        n_sub = subrep(G.sample()).shape[0]  # Dimension of sub representation
        is_irred, _ = is_complex_irreducible(G, subrep)
        if is_irred:
            found_irreps.append(subrep)
            Qs.append(np.eye(n_sub))
        else:
            # Find Q_sub such that Q_sub @ subrep[g] @ Q_sub^-1 is block diagonal, with blocks `sub_subrep`
            sub_subreps, Q_sub = cplx_isotypic_decomposition(G, subrep)
            found_irreps += sub_subreps
            Qs.append(Q_sub)

    # Sort irreps by dimension.
    P, sorted_irreps = sorted_jordan_canonical_form(G, found_irreps)

    # If subreps were decomposable, then these get further decomposed with an additional Hermitian matrix such that:
    # Q @ rep[g] @ Q^-1 = block_diag[irreps] | Q = (Q_external @ Q_internal)
    Q_external = block_diag(*Qs)
    Q = P @ Q_external @ Q_internal

    # Test isotypic decomposition.
    assert np.allclose(Q @ Q.conj().T, np.eye(n)), "Q is not unitary."
    for g in G.elements:
        g_iso = block_diag(*[irrep[g] if isinstance(irrep, dict) else irrep(g) for irrep in sorted_irreps])
        error = np.abs(g_iso - (Q @ rep(g) @ Q.conj().T))
        assert np.allclose(error, 0), f"Q @ rep[g] @ Q^-1 != block_diag[irreps[g]], for g={g}. Error \n:{error}"

    return sorted_irreps, Q


def sorted_jordan_canonical_form(group: Group, reps: list[Callable[[GroupElement], np.ndarray]]):
    """Sorts a list of representations in ascending order of dimension, and returns a permutation matrix P such that.

    Args:
        group (:class:`~escnn.group.group.Group`): Symmetry group of the representation.
        reps (:class:`list`[:class:`collections.abc.Callable`]): List of representations to sort by dimension.

    Returns:
        P (:class:`~numpy.ndarray`): Permutation matrix sorting the input reps.
        reps (:class:`list`[:class:`collections.abc.Callable`]): Sorted list of representations.
    """
    reps_idx = range(len(reps))
    reps_size = [rep(group.sample()).shape[0] for rep in reps]
    sort_order = sorted(reps_idx, key=lambda idx: reps_size[idx])
    if sort_order == list(reps_idx):
        return np.eye(sum(reps_size)), reps
    irrep_dim_start = np.cumsum([0] + reps_size[:-1])
    oneline_perm = []
    for idx in sort_order:
        rep_size = reps_size[idx]
        oneline_perm += list(range(irrep_dim_start[idx], irrep_dim_start[idx] + rep_size))
    P = permutation_matrix(oneline_perm)

    return P, [reps[idx] for idx in sort_order]
