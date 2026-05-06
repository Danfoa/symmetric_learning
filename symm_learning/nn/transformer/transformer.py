from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

from symm_learning.nn.activation import PositionalAttentionBase

__all__ = [
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]


class TransformerEncoderLayer(torch.nn.Module):
    r"""Transformer encoder layer with optional positional attention.

    Given an input sequence :math:`\mathbf{X} \in \mathbb{R}^{B \times T \times D}`, the layer computes:

    .. math::
        \mathbf{X}' = \mathbf{X} + \operatorname{Dropout}\!\left(
        \operatorname{Attn}(\mathbf{X}, \mathbf{X}, \mathbf{X})\right),
        \qquad
        \mathbf{Y} = \mathbf{X}' + \operatorname{FFN}(\mathbf{X}'),

    with layer normalization applied either before each residual branch (``norm_first=True``, pre-norm)
    or after (``norm_first=False``, post-norm). The attention operator :math:`\operatorname{Attn}` is either
    :class:`torch.nn.MultiheadAttention` or any :class:`~symm_learning.nn.activation.PositionalAttentionBase`
    subclass, the latter injecting positional information into query and key streams.

    Attributes:
    ----------
    self_attn:
        Attention module used for the source self-attention block.
    feed_forward_block:
        Sequential feed-forward network applied after self-attention.
    norm1, norm2:
        Normalization layers (``RMSNorm`` or ``LayerNorm``) applied around the residual branches.
    norm_first:
        Whether normalization is applied before each residual branch.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        self_attn: torch.nn.MultiheadAttention | PositionalAttentionBase,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: torch.nn.Module = torch.nn.GELU(),
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        norm_module: Literal["layernorm", "rmsnorm"] = "rmsnorm",
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Initialize the encoder layer.

        Args:
            d_model (:class:`int`): Model width ``D``.
            self_attn (:class:`torch.nn.MultiheadAttention` | :class:`~symm_learning.nn.activation.PositionalAttentionBase`): Self-attention module.
            dim_feedforward (:class:`int`): Width of the feed-forward hidden layer. Default: ``2048``.
            dropout (:class:`float`): Dropout probability. Default: ``0.1``.
            activation (:class:`str` | callable): Feed-forward activation. Default: :func:`torch.nn.functional.gelu`.
            layer_norm_eps (:class:`float`): LayerNorm epsilon. Default: ``1e-5``.
            norm_first (:class:`bool`): If ``True``, apply LayerNorm before each residual branch. Default: ``False``.
            norm_module (:class:`str`): Normalization layer type (``'layernorm'`` or ``'rmsnorm'``). Default: ``'rmsnorm'``.
            bias (:class:`bool`): If ``True``, use learnable normalization biases. Default: ``True``.
            device (:class:`torch.device`, optional): Parameter factory options.
            dtype (:class:`torch.dtype`, optional): Parameter factory options.
        """  # noqa: E501
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = self_attn
        assert isinstance(activation, torch.nn.Module), f"activation must be a torch.nn.Module got {type(activation)}"
        self.feed_forward_block = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs),
            activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs),
            torch.nn.Dropout(dropout),
        )

        self.norm_first = norm_first
        if norm_module == "layernorm":
            self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        elif norm_module == "rmsnorm":
            self.norm1 = torch.nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = torch.nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            raise ValueError(f"norm_module must be 'layernorm' or 'rmsnorm', got {norm_module}")
        self.attn_dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        *,
        src_positions: torch.Tensor | None = None,
        src_position_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Apply the encoder layer to a batch-first source sequence.

        Args:
            src (:class:`torch.Tensor`): Input sequence.
            src_mask (:class:`torch.Tensor`, optional): Additive attention mask for the source sequence.
            src_key_padding_mask (:class:`torch.Tensor`, optional): Boolean mask for padded source elements.
            is_causal (:class:`bool`): Whether to apply a causal attention mask. Default: ``False``.
            src_positions (:class:`torch.Tensor`, optional): Absolute positions for the source sequence,
              used by positional attention backends.
            src_position_mask (:class:`torch.Tensor`, optional): Boolean mask for padded source positions,
              used by positional attention backends.

        Returns:
            :class:`torch.Tensor`: The encoded source sequence.

        Shape
        -----
        - ``src``: ``(B, T, D)``.
        - ``src_positions``: ``(T,)`` or ``(B, T)``.
        - ``src_position_mask``: boolean mask with the same layout as
          ``src_positions``._ff_block
        - Returns: encoded source with shape ``(B, T, D)``.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal, src_positions, src_position_mask
            )
            x = x + self.feed_forward_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal, src_positions, src_position_mask)
            )
            x = self.norm2(x + self.feed_forward_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        positions: torch.Tensor | None,
        position_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply self-attention with optional positional encoding."""
        pos_enc_kwargs = {}
        if isinstance(self.self_attn, PositionalAttentionBase):
            pos_enc_kwargs.update(
                q_positions=positions,
                k_positions=positions,
                q_position_mask=position_mask,
                k_position_mask=position_mask,
            )
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
            **pos_enc_kwargs,
        )[0]
        return self.attn_dropout(x)


class TransformerDecoderLayer(torch.nn.Module):
    r"""Transformer decoder layer with optional positional self- and cross-attention.

    Given a target sequence :math:`\mathbf{X} \in \mathbb{R}^{B \times T_t \times D}` and a memory sequence
    :math:`\mathbf{M} \in \mathbb{R}^{B \times T_m \times D}`, the layer computes:

    .. math::
        \mathbf{X}'  &= \mathbf{X}  + \operatorname{Dropout}\!\left(
        \operatorname{SelfAttn}(\mathbf{X}, \mathbf{X}, \mathbf{X})\right), \\
        \mathbf{X}'' &= \mathbf{X}' + \operatorname{Dropout}\!\left(
        \operatorname{CrossAttn}(\mathbf{X}', \mathbf{M}, \mathbf{M})\right), \\
        \mathbf{Y}   &= \mathbf{X}'' + \operatorname{FFN}(\mathbf{X}''),

    with layer normalization applied either before each residual branch (``norm_first=True``, pre-norm)
    or after (``norm_first=False``, post-norm). Each attention operator is either
    :class:`torch.nn.MultiheadAttention` or any :class:`~symm_learning.nn.activation.PositionalAttentionBase`
    subclass.

    Attributes:
    ----------
    self_attn:
        Attention module used for masked target self-attention.
    multihead_attn:
        Attention module used for target-to-memory cross-attention.
    feed_forward_block:
        Sequential feed-forward network applied after cross-attention.
    norm1, norm2, norm3:
        Normalization layers (``RMSNorm`` or ``LayerNorm``) applied around the residual branches.
    norm_first:
        Whether normalization is applied before each residual branch.
    """

    __constants__ = ["norm_first"]  # For jit compilation

    def __init__(
        self,
        d_model: int,
        self_attn: torch.nn.MultiheadAttention | PositionalAttentionBase,
        multihead_attn: torch.nn.MultiheadAttention | PositionalAttentionBase,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: torch.nn.Module = torch.nn.GELU(),
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        norm_module: Literal["layernorm", "rmsnorm"] = "rmsnorm",
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Initialize the decoder layer.

        Args:
            d_model (:class:`int`): Model width ``D``.
            self_attn (:class:`torch.nn.MultiheadAttention` | :class:`~symm_learning.nn.activation.PositionalAttentionBase`):
                Masked self-attention module.
            multihead_attn (:class:`torch.nn.MultiheadAttention` | :class:`~symm_learning.nn.activation.PositionalAttentionBase`):
                Cross-attention module.
            dim_feedforward (:class:`int`): Width of the feed-forward hidden layer. Default: ``2048``.
            dropout (:class:`float`): Dropout probability. Default: ``0.1``.
            activation (:class:`str` | callable): Feed-forward activation. Default: :func:`torch.nn.functional.gelu`.
            layer_norm_eps (:class:`float`): LayerNorm epsilon. Default: ``1e-5``.
            norm_first (:class:`bool`): If ``True``, apply LayerNorm before each residual branch. Default: ``False``.
            norm_module (:class:`str`): Normalization layer type (``'layernorm'`` or ``'rmsnorm'``). Default: ``'rmsnorm'``.
            bias (:class:`bool`): If ``True``, use learnable normalization biases. Default: ``True``.
            device (:class:`torch.device`, optional): Parameter factory options.
            dtype (:class:`torch.dtype`, optional): Parameter factory options.
        """  # noqa: E501
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = self_attn
        self.multihead_attn = multihead_attn

        assert isinstance(activation, torch.nn.Module), f"activation must be a torch.nn.Module got {type(activation)}"
        self.feed_forward_block = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs),
            activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs),
            torch.nn.Dropout(dropout),
        )

        self.norm_first = norm_first
        if norm_module == "layernorm":
            self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.norm3 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        elif norm_module == "rmsnorm":
            self.norm1 = torch.nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = torch.nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = torch.nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            raise ValueError(f"norm_module must be 'layernorm' or 'rmsnorm', got {norm_module}")
        self.attn_dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        *,
        tgt_positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
        tgt_position_mask: torch.Tensor | None = None,
        memory_position_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Apply the decoder layer to target and memory sequences.

        Args:
            tgt (:class:`torch.Tensor`): Target sequence.
            memory (:class:`torch.Tensor`): Memory sequence from the encoder.
            tgt_mask (:class:`torch.Tensor`, optional): Additive attention mask for the target sequence.
            memory_mask (:class:`torch.Tensor`, optional): Additive attention mask for the memory sequence.
            tgt_key_padding_mask (:class:`torch.Tensor`, optional): Boolean mask for padded target elements.
            memory_key_padding_mask (:class:`torch.Tensor`, optional): Boolean mask for padded memory elements.
            tgt_is_causal (:class:`bool`): Whether to apply a causal attention mask to the target. Default: ``False``.
            memory_is_causal (:class:`bool`): Whether to apply a causal attention mask to the memory. Default:``False``.
            tgt_positions (:class:`torch.Tensor`, optional): Absolute positions for the target sequence.
            memory_positions (:class:`torch.Tensor`, optional): Absolute positions for the memory sequence.
            tgt_position_mask (:class:`torch.Tensor`, optional): Boolean mask for padded target positions.
            memory_position_mask (:class:`torch.Tensor`, optional): Boolean mask for padded memory positions.

        Returns:
            :class:`torch.Tensor`: The decoded target sequence.

        Shape
        -----
        - ``tgt``: ``(B, T_t, D)``.
        - ``memory``: ``(B, T_m, D)``.
        - ``tgt_positions``, ``memory_positions``: ``(T,)`` or ``(B, T)``.
        - ``tgt_position_mask``, ``memory_position_mask``: boolean masks with
          the same layout as their corresponding positions.
        - Returns: decoded target with shape ``(B, T_t, D)``.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal, tgt_positions, tgt_position_mask
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
                tgt_positions,
                memory_positions,
                tgt_position_mask,
                memory_position_mask,
            )
            x = x + self.feed_forward_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal, tgt_positions, tgt_position_mask)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                    tgt_positions,
                    memory_positions,
                    tgt_position_mask,
                    memory_position_mask,
                )
            )
            x = self.norm3(x + self.feed_forward_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        positions: torch.Tensor | None,
        position_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply masked self-attention with optional positional encoding."""
        kwargs = {}
        if isinstance(self.self_attn, PositionalAttentionBase):
            kwargs.update(
                q_positions=positions,
                k_positions=positions,
                q_position_mask=position_mask,
                k_position_mask=position_mask,
            )
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
            **kwargs,
        )[0]
        return self.attn_dropout(x)

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        q_positions: torch.Tensor | None,
        k_positions: torch.Tensor | None,
        q_position_mask: torch.Tensor | None,
        k_position_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply cross-attention with optional positional encoding."""
        kwargs = {}
        if isinstance(self.multihead_attn, PositionalAttentionBase):
            kwargs.update(
                q_positions=q_positions,
                k_positions=k_positions,
                q_position_mask=q_position_mask,
                k_position_mask=k_position_mask,
            )
        x = self.multihead_attn(
            query=x,
            key=mem,
            value=mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
            **kwargs,
        )[0]
        return self.attn_dropout(x)


class TransformerEncoder(torch.nn.Module):
    r"""Stack encoder layers and apply an optional final normalization.

    Attributes:
    ----------
    layers:
        Sequential copies of the encoder layer.
    norm:
        Optional normalization applied after the final layer.
    num_layers:
        Number of stacked encoder layers.
    """

    def __init__(
        self,
        encoder_layer: torch.nn.Module,
        num_layers: int,
        norm: torch.nn.Module | None = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
        r"""Initialize the encoder stack.

        Args:
            encoder_layer (:class:`torch.nn.Module`): Base layer to replicate.
            num_layers (:class:`int`): Number of stacked encoder layers.
            norm (:class:`torch.nn.Module`, optional): Final normalization layer.
            enable_nested_tensor (:class:`bool`): Preserved for API compatibility. Default: ``True``.
            mask_check (:class:`bool`): Preserved for API compatibility. Default: ``True``.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.layers = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.use_nested_tensor = False
        self.mask_check = mask_check

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool | None = None,
        **layer_kwargs,
    ) -> torch.Tensor:
        r"""Apply the encoder stack to a batch-first source sequence.

        Shape
        -----
        - ``src``: ``(B, T, D)``.
        - Returns: encoded source with shape ``(B, T, D)``.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        seq_len = src.shape[-2]
        # A square subsequent mask means the source is being decoded autoregressively.
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                **layer_kwargs,
            )

        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(torch.nn.Module):
    r"""Stack decoder layers and apply an optional final normalization.

    Attributes:
    ----------
    layers:
        Sequential copies of the decoder layer.
    norm:
        Optional normalization applied after the final layer.
    num_layers:
        Number of stacked decoder layers.
    """

    def __init__(
        self,
        decoder_layer: torch.nn.Module,
        num_layers: int,
        norm: torch.nn.Module | None = None,
    ) -> None:
        r"""Initialize the decoder stack.

        Args:
            decoder_layer (:class:`torch.nn.Module`): Base layer to replicate.
            num_layers (:class:`int`): Number of stacked decoder layers.
            norm (:class:`torch.nn.Module`, optional): Final normalization layer.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.layers = torch.nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
        **layer_kwargs,
    ) -> torch.Tensor:
        r"""Apply the decoder stack to target and memory sequences.

        Shape
        -----
        - ``tgt``: ``(B, T_t, D)``.
        - ``memory``: ``(B, T_m, D)``.
        - Returns: decoded target with shape ``(B, T_t, D)``.
        """
        output = tgt
        seq_len = tgt.shape[-2]
        # Only the target mask can imply autoregressive decoding.
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
                **layer_kwargs,
            )

        if self.norm is not None:
            output = self.norm(output)
        return output


def _detect_is_causal_mask(
    mask: torch.Tensor | None,
    is_causal: bool | None = None,
    size: int | None = None,
) -> bool:
    """Infer whether a square attention mask represents causal decoding."""
    make_causal = is_causal is True
    if is_causal is None and mask is not None:
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            size if size is not None else mask.shape[-2],
            device=mask.device,
            dtype=mask.dtype,
        )
        if mask.size() == causal_mask.size():
            make_causal = bool((mask == causal_mask).all())
    return make_causal
