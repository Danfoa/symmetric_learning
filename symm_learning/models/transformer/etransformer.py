from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import Optional, Union

import torch
import torch.nn.functional as F
from escnn.group import Representation, directsum
from torch import Tensor

from symm_learning.nn.activation import eMultiheadAttention
from symm_learning.nn.linear import eLinear
from symm_learning.nn.normalization import eLayerNorm


class eTransformerEncoderLayer(torch.nn.Module):
    """Equivariant Transformer encoder layer built from eLinear and eLayerNorm blocks."""

    __constants__ = ["norm_first"]

    def __init__(
        self,
        in_rep: Representation,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if dim_feedforward <= 0:
            raise ValueError(f"dim_feedforward must be positive, got {dim_feedforward}")

        self.in_rep, self.out_rep = in_rep, in_rep
        factory_kwargs = {"device": device, "dtype": dtype or torch.get_default_dtype()}

        G = in_rep.group
        num_hidden_reps = max(1, ceil(dim_feedforward / G.order()))
        self.embedding_rep = directsum([G.regular_representation] * num_hidden_reps)
        self.hidden_dim = self.embedding_rep.size
        self.requested_dim_feedforward = dim_feedforward

        self.self_attn = eMultiheadAttention(
            in_rep=self.in_rep,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        self.linear1 = eLinear(self.in_rep, self.embedding_rep, bias=bias).to(**factory_kwargs)
        self.linear2 = eLinear(self.embedding_rep, self.out_rep, bias=bias).to(**factory_kwargs)

        self.dropout = torch.nn.Dropout(dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.norm1 = eLayerNorm(self.in_rep, eps=layer_norm_eps, equiv_affine=True, bias=bias, **factory_kwargs)
        self.norm2 = eLayerNorm(self.out_rep, eps=layer_norm_eps, equiv_affine=True, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.batch_first = batch_first

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the equivariant encoder layer."""
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
            x = x + self._self_attention_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._feed_forward_block(self.norm2(x))
        else:
            x = self.norm1(x + self._self_attention_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._feed_forward_block(x))

        return x

    def _self_attention_block(
        self,
        x: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _feed_forward_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


if __name__ == "__main__":
    from escnn.group import CyclicGroup, DihedralGroup

    from symm_learning.models.transformer.etransformer import eTransformerEncoderLayer
    from symm_learning.utils import check_equivariance

    G = CyclicGroup(10)
    m = 2
    in_rep = directsum([G.regular_representation] * m)

    etransformer = eTransformerEncoderLayer(
        in_rep,
        nhead=1,
        dim_feedforward=in_rep.size * 4,
        dropout=0.1,
        activation="relu",
        norm_first=True,
        batch_first=True,
    )
    etransformer.eval()  # disable dropout for the test
    check_equivariance(etransformer, input_dim=3, atol=1e-4, rtol=1e-4)

    print("All tests passed!")
