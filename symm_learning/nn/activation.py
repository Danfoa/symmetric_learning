from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from escnn.group import Representation
from torch.nn.utils import parametrize

from symm_learning.nn.linear import eLinear
from symm_learning.nn.module import eModule
from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint
from symm_learning.representation_theory import direct_sum

logger = logging.getLogger(__name__)


class eMultiheadAttention(eModule, torch.nn.MultiheadAttention):
    """Drop-in replacement for :class:`torch.nn.MultiheadAttention` that preserves G-equivariance.

    This module keeps the runtime logic of PyTorch’s implementation untouched: we still rely on
    the packed ``in_proj_weight`` / ``in_proj_bias`` for computing queries, keys, and values,
    and the internal attention kernel (including mask handling, dropouts, and softmax) is exactly
    the stock MultiheadAttention behavior.

    Equivariance is achieved by constraining every linear projection involved in the attention block:

    * the input projection ``[Q; K; V] = W_in @ x`` is treated as a single map from the input
      representation to three stacked copies of a regular-representation block that
      aligns with the requested ``num_heads`` (enforced via
      :class:`~symm_learning.nn.parametrizations.CommutingConstraint`);
    * the optional stacked bias is projected onto the invariant subspace of that same block via
      :class:`~symm_learning.nn.parametrizations.InvariantConstraint`;
    * the output projection ``out_proj`` is constrained to commute with the group action so that
      the concatenated value vectors are mapped back into the original feature space equivariantly.

    Additionally, we restrict ``num_heads`` to divide the number of regular-representation copies
    present in the input feature space to avoid splitting irreducible subspaces across heads.
    """

    def __init__(
        self,
        in_rep: Representation,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        device=None,
        dtype=None,
        init_scheme: str | None = "xavier_normal",
    ) -> None:
        r"""Initialize the equivariant multihead attention.

        Args:
            in_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{in}}` of the input/output
                space.
            num_heads (:class:`int`): Number of parallel attention heads.
            dropout (:class:`float`): Dropout probability on attention weights. Default: 0.0.
            bias (:class:`bool`): If ``True``, adds learnable input and output projection biases. Default: ``True``.
            add_bias_kv (:class:`bool`): **Not supported**. Must be ``False``.
            add_zero_attn (:class:`bool`): **Not supported**. Must be ``False``.
            device (:class:`torch.device`, optional): Parameter factory options.
            dtype (:class:`torch.dtype`, optional): Parameter factory options.
            init_scheme (:class:`str` | :class:`None`, optional): Initialization scheme for the equivariant linear
                layers. Default: ``"xavier_normal"``.
        """
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if add_bias_kv:
            raise NotImplementedError("Equivariant attention does not support add_bias_kv.")
        if add_zero_attn:
            raise NotImplementedError("Equivariant attention does not support add_zero_attn.")

        G = in_rep.group
        if in_rep.size % G.order() != 0:
            raise ValueError(f"Input rep dim ({in_rep.size}) must be divisible of the group order ({G.order()}).")

        regular_copies = in_rep.size // G.order()
        if regular_copies % num_heads != 0:
            raise ValueError(f"For input dim {in_rep.size} `num_heads` must divide {in_rep.size}/|G|={regular_copies}")

        super().__init__(
            embed_dim=in_rep.size,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.in_rep, self.out_rep = in_rep, in_rep
        self._regular_stack_rep = direct_sum([G.regular_representation] * regular_copies)
        if not self._qkv_same_embed_dim:
            raise ValueError("eMultiheadAttention requires kdim == vdim == embed_dim.")

        stacked_qkv_rep = direct_sum([G.regular_representation] * regular_copies * 3)
        parametrize.register_parametrization(self, "in_proj_weight", CommutingConstraint(in_rep, stacked_qkv_rep))
        if bias and self.in_proj_bias is not None:
            parametrize.register_parametrization(self, "in_proj_bias", InvariantConstraint(stacked_qkv_rep))

        # Replace output projection linear layer.
        self.out_proj = eLinear(in_rep, in_rep, bias=bias, init_scheme=init_scheme).to(device=device, dtype=dtype)

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    @torch.no_grad()
    def reset_parameters(self, scheme="xavier_uniform") -> None:
        """Overload parent method to take into account equivariance constraints."""
        if not hasattr(self, "parametrizations"):
            return super()._reset_parameters()
        logger.debug(f"Resetting parameters of {self.__class__.__name__} with scheme: {scheme}")
        # Reset equivariant linear layers (symm_learning.nn.eLinear)
        self.out_proj.reset_parameters(scheme=scheme)

        for param_name, constaint_list in self.parametrizations.items():
            param = getattr(self, param_name)
            if param.dim() == 2:
                commuting_constraint: CommutingConstraint = constaint_list[0]
                W = commuting_constraint.homo_basis.initialize_params(scheme=scheme, return_dense=True)
                param = W
                logger.debug(f"Initialized {param_name} with scheme {scheme}")
            elif param.dim() == 1:
                # invariant_constraint: InvariantConstraint = constaint_list[0]
                param = torch.zeros_like(param)
                logger.debug(f"Initialized {param_name} with zeros")

        # if self._qkv_same_embed_dim:
        #     xavier_uniform_(self.in_proj_weight)
        # if self.in_proj_bias is not None:
        #     constant_(self.in_proj_bias, 0.0)
        #     constant_(self.out_proj.bias, 0.0)


class PositionalAttentionBase(torch.nn.Module, ABC):
    r"""Abstract interface for attention blocks with explicit positional branches.

    The convention in this module is that positional information is provided as a
    function, defined by a submodule, acting on the query and key streams only.
    Values are left unchanged:

    .. math::

                ilde{\mathbf{q}} = \mathbf{q} + \phi_\theta(\mathbf{P}_Q),
            \qquad
                ilde{\mathbf{k}} = \mathbf{k} + \phi_\theta(\mathbf{P}_K),
            \qquad
                ilde{\mathbf{v}} = \mathbf{v}.

    A standard multi-head attention operator is then applied to
    :math:`(\tilde{\mathbf{q}}, \tilde{\mathbf{k}}, \tilde{\mathbf{v}})`.
    If the positional branch is the identity/no-op map, the module reduces to
    :class:`torch.nn.MultiheadAttention`.

    Concrete implementations may ignore any positional arguments they do not use,
    but the forward signature stays stable so encoder and decoder layers can call
    all attention backends uniformly.

    Shape
    -----
    - ``query``, ``key``, ``value``: ``(B, P, D)``.
    - ``q_positions``, ``k_positions``: ``(P,)`` or ``(B, P)``.
      When omitted, they default to ``torch.arange(P)`` for the corresponding sequence.
    - ``q_position_mask``, ``k_position_mask``: ``(P,)`` or ``(B, P)`` boolean masks.
    - ``attn_mask``: any attention mask layout accepted by
        :class:`torch.nn.MultiheadAttention`.
    - ``key_padding_mask``: ``(B, S)`` boolean or additive padding mask.
    - Returns: attention output with the same leading layout as the input, plus
        optional attention weights.

    Attributes:
    ----------
    This base class does not define any storage beyond the standard
    :class:`torch.nn.Module` state. Concrete subclasses define their own
    positional encoder and attention parameters.
    """

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        q_positions: torch.Tensor | None = None,
        k_positions: torch.Tensor | None = None,
        q_position_mask: torch.Tensor | None = None,
        k_position_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply attention with optional explicit position metadata.

        Shape
        -----
        - ``query``, ``key``, ``value``: see :class:`PositionalAttentionBase`.
        - ``q_positions``, ``k_positions``: position coordinates for the query and
            key tokens. They may differ in cross-attention.
        - Returns: ``(output, attn_weights)`` where ``attn_weights`` is ``None``
            unless ``need_weights=True``.
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Initialize parameters to match torch.nn.MultiheadAttention."""
        if hasattr(self, "attn") and isinstance(self.attn, torch.nn.MultiheadAttention):
            self.attn._reset_parameters()
            return

        weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
        for name in weight_names:
            weight = getattr(self, name, None)
            if weight is not None:
                torch.nn.init.xavier_uniform_(weight)
            elif hasattr(self, name.replace("_weight", "")) and isinstance(
                getattr(self, name.replace("_weight", "")), torch.nn.Linear
            ):
                torch.nn.init.xavier_uniform_(getattr(self, name.replace("_weight", "")).weight)

        bias_names = ["in_proj_bias", "bias_k", "bias_v"]
        for name in bias_names:
            bias = getattr(self, name, None)
            if bias is not None:
                torch.nn.init.constant_(bias, 0.0)
            elif (
                hasattr(self, name.replace("_bias", "").replace("bias_", "") + "_proj")
                and isinstance(getattr(self, name.replace("_bias", "").replace("bias_", "") + "_proj"), torch.nn.Linear)
                and getattr(self, name.replace("_bias", "").replace("bias_", "") + "_proj").bias is not None
            ):
                torch.nn.init.constant_(
                    getattr(self, name.replace("_bias", "").replace("bias_", "") + "_proj").bias, 0.0
                )

        out_proj = getattr(self, "out_proj", None)
        if out_proj is not None:
            if getattr(out_proj, "weight", None) is not None:
                torch.nn.init.xavier_uniform_(out_proj.weight)
            if getattr(out_proj, "bias", None) is not None:
                torch.nn.init.constant_(out_proj.bias, 0.0)

    @staticmethod
    def _positions_or_arange(
        positions: torch.Tensor | None,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        r"""Use explicit positions when provided, otherwise default to ``0, \ldots, P-1``."""
        if positions is None:
            return torch.arange(seq_len, device=device)
        return positions

    @staticmethod
    def _normalize_positions(
        positions: torch.Tensor,
        position_mask: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize position tensors and validate their layout.

        The helper accepts positions with shape ``(P,)`` or ``(B, P)`` and checks
        that the sequence length matches ``seq_len`` and the batch dimension is
        either ``1`` or ``batch_size``. If a mask is provided, it must have the
        same rank and shape as ``positions`` before any singleton batch expansion.
        The returned mask is expanded together with ``positions`` when needed.
        """
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        elif positions.ndim != 2:
            raise ValueError(f"Expected positions tensor with shape (P,) or (B, P), got {tuple(positions.shape)}")

        if positions.shape[-1] != seq_len:
            raise ValueError(f"Position length {positions.shape[-1]} does not match sequence length {seq_len}")
        if positions.shape[0] not in (1, batch_size):
            raise ValueError(f"Position batch size {positions.shape[0]} must be 1 or match batch size {batch_size}")
        if positions.shape[0] == 1 and batch_size != 1:
            positions = positions.expand(batch_size, -1)

        if position_mask is None:
            return positions, None

        if position_mask.ndim == 1:
            position_mask = position_mask.unsqueeze(0)
        elif position_mask.ndim != 2:
            raise ValueError(f"Expected position_mask with shape (P,) or (B, P), got {tuple(position_mask.shape)}")
        if position_mask.shape[-1] != seq_len:
            raise ValueError(f"Position mask length {position_mask.shape[-1]} does not match sequence length {seq_len}")
        if position_mask.shape[0] not in (1, batch_size):
            raise ValueError(
                f"Position mask batch size {position_mask.shape[0]} must be 1 or match batch size {batch_size}"
            )
        if position_mask.shape[0] == 1 and batch_size != 1:
            position_mask = position_mask.expand(batch_size, -1)

        return positions, position_mask


class AdditivePosMultiheadAttention(PositionalAttentionBase):
    r"""Wrap :class:`torch.nn.MultiheadAttention` and add positional features to Q/K only.

    A learned table maps coordinates to an additive update of the query and key
    streams,

    .. math::

            \mathbf{q}' = \mathbf{q} + E_\theta(\mathbf{P}_Q),
            \qquad
            \mathbf{k}' = \mathbf{k} + E_\theta(\mathbf{P}_K),
            \qquad
            \mathbf{v}' = \mathbf{v}.

    The values are never position-modulated. When ``E_\theta`` is the identity/
    no-op branch, this is exactly ordinary multi-head attention.

    Shape
    -----
    - ``query``, ``key``, ``value``: ``(B, T, D)``.
    - ``q_positions``, ``k_positions``: ``(P,)`` or ``(B, P)``.
    - ``q_position_mask``, ``k_position_mask``: boolean masks with the same shape
        as the corresponding position tensor.
    - Returns: the attention output and, optionally, attention weights.

    Attributes:
    ----------
    pos_emb:
        Learnable table with shape ``(max_len, D)`` storing the absolute
        positional embeddings.
    attn:
        Internal :class:`torch.nn.MultiheadAttention` backend.
    embed_dim:
        Model width ``D``.
    num_heads:
        Number of attention heads.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        max_len: int,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Initialize additive positional attention with a wrapped multi-head backend.

        Args:
            embed_dim (:class:`int`): Model width ``D``.
            num_heads (:class:`int`): Number of attention heads.
            max_len (:class:`int`): Maximum supported sequence length.
            dropout (:class:`float`): Dropout probability on attention weights. Default: 0.0.
            bias (:class:`bool`): If ``True``, adds learnable input and output projection biases. Default: ``True``.
            device (:class:`torch.device`, optional): Parameter factory options.
            dtype (:class:`torch.dtype`, optional): Parameter factory options.
        """
        super().__init__()
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_heads = num_heads
        self.pos_emb = torch.nn.Parameter(torch.zeros(max_len, embed_dim, device=device, dtype=dtype))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        q_positions: torch.Tensor | None = None,
        k_positions: torch.Tensor | None = None,
        q_position_mask: torch.Tensor | None = None,
        k_position_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Add the positional update to query and key before attention.

        The values are passed through unchanged.

        Shape
        -----
        - ``query``, ``key``, ``value``: see :class:`PositionalAttentionBase`.
        - Returns: ``(output, attn_weights)`` from the wrapped
          :class:`torch.nn.MultiheadAttention`.
        """
        query = query + self._position_update(query, q_positions, q_position_mask)
        key = key + self._position_update(key, k_positions, k_position_mask)
        return self.attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )

    def _position_update(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None,
        position_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = self._positions_or_arange(positions, seq_len=seq_len, device=x.device)

        positions, position_mask = self._normalize_positions(
            positions,
            position_mask,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        encoded_positions = positions.masked_fill(~position_mask, 0) if position_mask is not None else positions
        pos_emb = self.pos_emb[encoded_positions.long()]
        if pos_emb.ndim == 2:
            expected_shape = (seq_len, self.embed_dim)
            if pos_emb.shape != expected_shape:
                raise ValueError(f"Expected positional embedding shape {expected_shape}, got {tuple(pos_emb.shape)}")
            pos_emb = pos_emb.unsqueeze(0)
            if batch_size != 1:
                pos_emb = pos_emb.expand(batch_size, -1, -1)
        else:
            expected_shape = (batch_size, seq_len, self.embed_dim)
            if pos_emb.ndim != 3 or pos_emb.shape != expected_shape:
                raise ValueError(f"Expected positional embedding shape {expected_shape}, got {tuple(pos_emb.shape)}")

        if position_mask is not None:
            pos_emb = pos_emb * position_mask.unsqueeze(-1)

        return pos_emb


class AdditiveRelMultiheadAttention(PositionalAttentionBase):
    r"""Wrap :class:`torch.nn.MultiheadAttention` and subtract a relative-position bias from the logits.

    A learned table maps pairwise relative distances to an additive correction
    of the attention scores,

    .. math::

            \mathbf{A}_{ij} =
            \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_h}}
            - \phi_\theta(\mathbf{P}_{Q, i} - \mathbf{P}_{K, j}),

    while the value stream remains unchanged. Since the correction depends only
    on relative offsets, the attention module is time-translation equivariant.

    Shape
    -----
    - ``query``, ``key``, ``value``: ``(B, T, D)``.
    - ``q_positions``, ``k_positions``: ``(P,)`` or ``(B, P)``.
    - ``q_position_mask``, ``k_position_mask``: boolean masks with the same shape
        as the corresponding position tensor.
    - Returns: the attention output and, optionally, attention weights.

    Attributes:
    ----------
    rel_bias:
        Learnable table with shape ``(2 * max_distance + 1,)`` storing the
        scalar bias for each clipped relative offset.
    attn:
        Internal :class:`torch.nn.MultiheadAttention` backend.
    embed_dim:
        Model width ``D``.
    num_heads:
        Number of attention heads.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        max_distance: int,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Initialize relative-bias attention with a wrapped multi-head backend.

        Args:
            embed_dim (:class:`int`): Model width ``D``.
            num_heads (:class:`int`): Number of attention heads.
            max_distance (:class:`int`): Maximum relative distance represented
                explicitly before clipping.
            dropout (:class:`float`): Dropout probability on attention weights. Default: 0.0.
            bias (:class:`bool`): If ``True``, adds learnable input and output projection biases. Default: ``True``.
            device (:class:`torch.device`, optional): Parameter factory options.
            dtype (:class:`torch.dtype`, optional): Parameter factory options.

        """
        super().__init__()
        if max_distance < 0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        self.attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.embed_dim = embed_dim
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.rel_bias = torch.nn.Parameter(torch.zeros(2 * max_distance + 1, device=device, dtype=dtype))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        q_positions: torch.Tensor | None = None,
        k_positions: torch.Tensor | None = None,
        q_position_mask: torch.Tensor | None = None,
        k_position_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Subtract the relative-position bias from the attention logits.

        Shape
        -----
        - ``query``, ``key``, ``value``: see :class:`PositionalAttentionBase`.
        - Returns: ``(output, attn_weights)`` from the wrapped
          :class:`torch.nn.MultiheadAttention`.
        """
        batch_size = query.shape[0]
        tgt_len = query.shape[1]
        src_len = key.shape[1]
        q_positions = self._positions_or_arange(q_positions, seq_len=tgt_len, device=query.device)
        k_positions = self._positions_or_arange(k_positions, seq_len=src_len, device=key.device)

        rel_bias = self._relative_bias(
            q_positions,
            k_positions,
            q_position_mask,
            k_position_mask,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
            target_dtype=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if rel_bias is not None:
            attn_mask = -rel_bias if attn_mask is None else attn_mask - rel_bias

        return self.attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )

    def _relative_bias(
        self,
        q_positions: torch.Tensor | None,
        k_positions: torch.Tensor | None,
        q_position_mask: torch.Tensor | None,
        k_position_mask: torch.Tensor | None,
        *,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if q_positions is None or k_positions is None:
            return None

        positions_shared_across_batch = self._positions_are_shared_across_batch(
            q_positions, q_position_mask
        ) and self._positions_are_shared_across_batch(k_positions, k_position_mask)
        q_positions, q_position_mask = self._normalize_positions(
            q_positions,
            q_position_mask,
            batch_size=batch_size,
            seq_len=tgt_len,
        )
        k_positions, k_position_mask = self._normalize_positions(
            k_positions,
            k_position_mask,
            batch_size=batch_size,
            seq_len=src_len,
        )

        if positions_shared_across_batch:
            q_positions = q_positions[0]
            k_positions = k_positions[0]
            rel_positions = q_positions.unsqueeze(-1) - k_positions.unsqueeze(-2)
            pair_mask = None
            if q_position_mask is not None or k_position_mask is not None:
                q_valid = (
                    torch.ones(tgt_len, device=rel_positions.device, dtype=torch.bool)
                    if q_position_mask is None
                    else q_position_mask[0]
                )
                k_valid = (
                    torch.ones(src_len, device=rel_positions.device, dtype=torch.bool)
                    if k_position_mask is None
                    else k_position_mask[0]
                )
                pair_mask = q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)
        else:
            rel_positions = q_positions.unsqueeze(-1) - k_positions.unsqueeze(-2)
            pair_mask = None
            if q_position_mask is not None or k_position_mask is not None:
                q_valid = (
                    torch.ones(batch_size, tgt_len, device=rel_positions.device, dtype=torch.bool)
                    if q_position_mask is None
                    else q_position_mask
                )
                k_valid = (
                    torch.ones(batch_size, src_len, device=rel_positions.device, dtype=torch.bool)
                    if k_position_mask is None
                    else k_position_mask
                )
                pair_mask = q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)

        rel_bias = self._relative_bias_values(rel_positions).to(dtype=target_dtype)
        if rel_bias.ndim == 3:
            rel_bias = rel_bias.repeat_interleave(self.num_heads, dim=0)
        elif not positions_shared_across_batch:
            rel_bias = rel_bias.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)

        if pair_mask is None:
            return rel_bias
        if rel_bias.ndim == 2:
            return rel_bias * pair_mask.to(dtype=rel_bias.dtype)
        if pair_mask.ndim == 2:
            pair_mask = pair_mask.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)
        else:
            pair_mask = pair_mask.repeat_interleave(self.num_heads, dim=0)
        return rel_bias * pair_mask.to(dtype=rel_bias.dtype)

    @staticmethod
    def _positions_are_shared_across_batch(
        positions: torch.Tensor,
        position_mask: torch.Tensor | None,
    ) -> bool:
        shared_positions = positions.ndim == 1 or (positions.ndim == 2 and positions.shape[0] == 1)
        if position_mask is None:
            return shared_positions
        shared_mask = position_mask.ndim == 1 or (position_mask.ndim == 2 and position_mask.shape[0] == 1)
        return shared_positions and shared_mask

    def _relative_bias_values(self, rel_positions: torch.Tensor) -> torch.Tensor:
        clipped_positions = rel_positions.clamp(-self.max_distance, self.max_distance).long() + self.max_distance
        return self.rel_bias[clipped_positions]


class RoPEMultiheadAttention(PositionalAttentionBase):
    r"""Multi-head attention with rotary position embeddings applied to Q and K.

    Query, key, and value tensors are first projected into per-head features. The
    entire head embedding is then rotated in position space, block by block in 2D
    pairs:

    .. math::

            \mathbf{q}_r' = R(\mathbf{P}_Q)\,\mathbf{q}_r,
            \qquad
            \mathbf{k}_r' = R(\mathbf{P}_K)\,\mathbf{k}_r,
            \qquad
            \mathbf{v}' = \mathbf{v},

    where ``R`` is the block-wise 2D rotation induced by the sine/cosine tables.
    The attention scores are then

    .. math::

            \mathbf{A} = \operatorname{softmax}\left(
            \frac{\mathbf{q}'\mathbf{k}'^\top}{\sqrt{d_h}} + \mathbf{M}\right),
            \qquad
            \mathbf{O} = \mathbf{A}\mathbf{v}'.

    Shape
    -----
    - ``query``, ``key``, ``value``: ``(B, T, D)``.
    - ``q_positions``, ``k_positions``: ``(P,)`` or ``(B, P)``.
    - ``q_position_mask``, ``k_position_mask``: boolean masks with the same shape
        as the corresponding position tensor.
    - Returns: ``(output, attn_weights)`` where ``output`` has the same leading
        layout as the input and ``attn_weights`` is ``(B, T_q, T_k)`` when requested.

    Attributes:
    ----------
    embed_dim:
        Total feature width ``D``.
    num_heads:
        Number of attention heads.
    head_dim:
        Width of each head, ``head_dim = embed_dim / num_heads``.
    dropout:
        Dropout probability applied to attention weights during training.
    q_proj, k_proj, v_proj, out_proj:
        Learnable linear projections used to form queries, keys, values, and the
        final output.
    rotary_emb:
        Helper module that builds the sine and cosine tables used by RoPE.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope_base: float = 10000.0,
        device=None,
        dtype=None,
    ) -> None:
        r"""Initialize the rotary-position attention block.

        Args:
            embed_dim (:class:`int`): Model width ``D``.
            num_heads (:class:`int`): Number of attention heads.
            dropout (:class:`float`): Dropout probability on attention weights. Default: 0.0.
            bias (:class:`bool`): If ``True``, adds learnable input and output projection biases. Default: ``True``.
            rope_base (:class:`float`): Frequency base used to build the rotary spectrum. Default: ``10000.0``.
            device (:class:`torch.device`, optional): Parameter factory options.
            dtype (:class:`torch.dtype`, optional): Parameter factory options.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim={embed_dim} must be divisible by num_heads={num_heads} "
                "so each head has an integer dimension".format(embed_dim=embed_dim, num_heads=num_heads)
            )

        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be even because RoPE rotates the entire head embedding in 2D pairs. "
                f"Choose embed_dim and num_heads so embed_dim / num_heads is even."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.rotary_emb = RotaryEmbedding(head_dim, base=rope_base, device=device, dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        q_positions: torch.Tensor | None = None,
        k_positions: torch.Tensor | None = None,
        q_position_mask: torch.Tensor | None = None,
        k_position_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Project inputs, apply RoPE to each head, then compute attention.

        Shape
        -----
        - ``query``, ``key``, ``value``: see :class:`PositionalAttentionBase`.
        - Returns: ``(output, attn_weights)`` with the same leading layout as the
          input and optional attention weights when requested.
        """
        q_positions = self._positions_or_arange(q_positions, seq_len=query.shape[1], device=query.device)
        k_positions = self._positions_or_arange(k_positions, seq_len=key.shape[1], device=key.device)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        q = self.rotary_emb.apply_rope(q, positions=q_positions, position_mask=q_position_mask)
        k = self.rotary_emb.apply_rope(k, positions=k_positions, position_mask=k_position_mask)

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        merged_mask = self._merge_masks(attn_mask, key_padding_mask, query.shape[0], k.shape[2])

        if need_weights:
            output, attn_weights = self._attention_with_weights(q, k, v, merged_mask, is_causal=is_causal)
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=merged_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_weights = None

        output = self.out_proj(self._merge_heads(output))
        return output, attn_weights

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    def _merge_masks(
        self,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        batch_size: int,
        src_len: int,
    ) -> torch.Tensor | None:
        merged_mask = attn_mask
        if merged_mask is not None:
            if merged_mask.ndim == 2:
                merged_mask = merged_mask.unsqueeze(0).unsqueeze(0)
            elif merged_mask.ndim == 3:
                if merged_mask.shape[0] == batch_size * self.num_heads:
                    merged_mask = merged_mask.view(batch_size, self.num_heads, merged_mask.shape[-2], src_len)
                else:
                    merged_mask = merged_mask.unsqueeze(1)
            elif merged_mask.ndim != 4:
                raise ValueError(f"Unsupported attn_mask shape {tuple(merged_mask.shape)}")

        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :]
            merged_mask = padding_mask if merged_mask is None else merged_mask + padding_mask

        return merged_mask

    def _attention_with_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)

        if is_causal:
            tgt_len, src_len = attn_scores.shape[-2:]
            causal_mask = torch.ones(tgt_len, src_len, device=attn_scores.device, dtype=torch.bool).triu(1)
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = torch.dropout(attn_weights, self.dropout, train=True)
        return torch.matmul(attn_weights, v), attn_weights.mean(dim=1)


class RotaryEmbedding(torch.nn.Module):
    r"""Precompute the cosine and sine tables used by rotary embeddings.

    Shape
    -----
    - ``positions``: ``(P,)`` or ``(B, P)``.
    - Returns: ``(cos, sin)`` with shape ``(P, dim / 2)`` or ``(B, P, dim / 2)``.

    Attributes:
    ----------
    dim:
        Number of channels rotated by RoPE.
    base:
        Frequency base used to build the inverse frequency spectrum.
    inv_freq:
        Buffer containing the inverse frequencies used to generate the tables.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        device=None,
        dtype=None,
    ) -> None:
        r"""Initialize the RoPE table builder.

        Args:
            dim (:class:`int`): Number of channels rotated by RoPE.
            base (:class:`float`): Frequency base used to build the inverse frequency spectrum.
            device (:class:`torch.device`, optional): Buffer factory options.
            dtype (:class:`torch.dtype`, optional): Buffer factory options.
        """
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"Rotary dim must be a positive even integer, got {dim}")

        factory_kwargs = {"device": device, "dtype": dtype}
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, **factory_kwargs) / dim))
        self.dim = dim
        self.base = base
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the RoPE cosine and sine tables for the provided positions."""
        angles = positions.unsqueeze(-1) * self.inv_freq
        return angles.cos(), angles.sin()

    def apply_rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Apply rotary position embeddings to the entire head embedding of ``x``.

        Shape
        -----
        - ``x``: ``(B, H, T, D)``.
        - ``positions``: ``(P,)`` or ``(B, P)``.
        - ``position_mask``: optional boolean mask with the same layout as
          ``positions``.
        - Returns: ``x`` with every head channel rotated in place.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, H, T, D), got {tuple(x.shape)}")

        batch_size, _, seq_len, head_dim = x.shape
        positions, position_mask = PositionalAttentionBase._normalize_positions(
            positions,
            position_mask,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        if position_mask is not None:
            positions = positions.masked_fill(~position_mask, 0)

        if self.dim != head_dim:
            raise ValueError(f"RotaryEmbedding.dim={self.dim} must match the head dimension {head_dim}")

        cos, sin = self(positions)
        cos = cos.unsqueeze(1).repeat_interleave(2, dim=-1).to(dtype=x.dtype)
        sin = sin.unsqueeze(1).repeat_interleave(2, dim=-1).to(dtype=x.dtype)

        # Apply the 2D rotation block by block across the entire head embedding.
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rotated = x * cos + torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2) * sin

        if position_mask is not None:
            # Leave padded positions unchanged so masking does not inject a phase rotation.
            keep_mask = position_mask.unsqueeze(1).unsqueeze(-1)
            x = torch.where(keep_mask, x_rotated, x)
        else:
            x = x_rotated

        return x
