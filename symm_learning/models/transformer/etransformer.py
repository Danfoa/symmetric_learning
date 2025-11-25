from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from math import ceil
from typing import Optional, Union

import torch
import torch.nn.functional as F
from escnn.group import Representation
from torch import Tensor

# from torch.nn import Transformer
from symm_learning.nn.activation import eMultiheadAttention
from symm_learning.nn.linear import eLinear
from symm_learning.nn.normalization import eLayerNorm
from symm_learning.representation_theory import direct_sum

logger = logging.getLogger(__name__)


class eTransformerEncoderLayer(torch.nn.Module):
    """Equivariant Transformer encoder layer with the same API as ``torch.nn.TransformerEncoderLayer``.

    Applies :class:`eMultiheadAttention` followed by an equivariant feed-forward block
    built from :class:`eLinear` layers and :class:`eLayerNorm`, mirroring PyTorch’s ordering
    (pre- or post-norm) while constraining every linear map to commute with the group action.
    """

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
        init_scheme: str | None = "xavier_normal",
    ) -> None:
        super().__init__()
        if dim_feedforward <= 0:
            raise ValueError(f"dim_feedforward must be positive, got {dim_feedforward}")

        self.in_rep, self.out_rep = in_rep, in_rep
        factory_kwargs = {"device": device, "dtype": dtype or torch.get_default_dtype()}

        G = in_rep.group
        num_hidden_reps = max(1, ceil(dim_feedforward / G.order()))
        self.embedding_rep = direct_sum([G.regular_representation] * num_hidden_reps)
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
            init_scheme=init_scheme,
        )

        self.linear1 = eLinear(self.in_rep, self.embedding_rep, bias, init_scheme=init_scheme).to(**factory_kwargs)
        self.linear2 = eLinear(self.embedding_rep, self.out_rep, bias, init_scheme=init_scheme).to(**factory_kwargs)

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

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the equivariant encoder layer.

        Args:
            src: input sequence of shape ``(T, B, D)`` or ``(B, T, D)`` depending on ``batch_first``,
                with last dimension equal to ``in_rep.size``.
            src_mask: optional attention mask for the input sequence.
            src_key_padding_mask: optional padding mask for the batch.
            is_causal: if ``True``, applies a causal mask to the self-attention block.
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
            x = x + self._self_attention_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._feed_forward_block(self.norm2(x))
        else:
            x = self.norm1(x + self._self_attention_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._feed_forward_block(x))

        return x

    def _self_attention_block(
        self, x: Tensor, attn_mask: Tensor = None, key_padding_mask: Tensor = None, is_causal: bool = False
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

    @torch.no_grad()
    def reset_parameters(self, scheme="xavier_uniform") -> None:  # noqa: D102
        logger.debug(f"Resetting parameters of {self.__class__.__name__} with scheme: {scheme}")
        self.linear1.reset_parameters(scheme)
        self.linear2.reset_parameters(scheme)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        # Reset attention layers:
        self.self_attn.reset_parameters(scheme)


class eTransformerDecoderLayer(torch.nn.Module):
    """Equivariant Transformer decoder layer mirroring :class:`torch.nn.TransformerDecoderLayer`.

    Combines an equivariant self-attention block, an equivariant cross-attention block,
    and the same eLinear/eLayerNorm feed-forward structure used by the encoder so every
    submodule commutes with the group action while keeping PyTorch’s runtime logic intact.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        in_rep: Representation,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        init_scheme: str | None = "xavier_uniform",
    ) -> None:
        super().__init__()
        if dim_feedforward <= 0:
            raise ValueError(f"dim_feedforward must be positive, got {dim_feedforward}")

        self.in_rep, self.out_rep = in_rep, in_rep
        factory_kwargs = {"device": device, "dtype": dtype or torch.get_default_dtype()}

        G = in_rep.group
        num_hidden_reps = max(1, ceil(dim_feedforward / G.order()))
        self.embedding_rep = direct_sum([G.regular_representation] * num_hidden_reps)
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
            init_scheme=init_scheme,
        )
        self.cross_attn = eMultiheadAttention(
            in_rep=self.in_rep,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
            init_scheme=init_scheme,
        )

        self.linear1 = eLinear(self.in_rep, self.embedding_rep, bias, init_scheme=init_scheme).to(**factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = eLinear(self.embedding_rep, self.out_rep, bias, init_scheme=init_scheme).to(**factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = eLayerNorm(self.in_rep, eps=layer_norm_eps, equiv_affine=True, bias=bias, **factory_kwargs)
        self.norm2 = eLayerNorm(self.in_rep, eps=layer_norm_eps, equiv_affine=True, bias=bias, **factory_kwargs)
        self.norm3 = eLayerNorm(self.out_rep, eps=layer_norm_eps, equiv_affine=True, bias=bias, **factory_kwargs)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation
        self.batch_first = batch_first

        if init_scheme is not None:
            self.reset_parameters(scheme=init_scheme)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the equivariant decoder layer.

        Args:
            tgt: target/query tensor of shape ``(T, B, D)`` or ``(B, T, D)`` matching
                ``batch_first``. The last dimension must equal ``in_rep.size``.
            memory: encoder memory tensor of shape ``(S, B, D)`` or ``(B, S, D)``
                (same ``batch_first``). We assume this tensor transforms under the
                *same representation* as ``tgt``; i.e., it is typically the output
                of an equivariant encoder with representation ``in_rep``.
            tgt_mask: optional target attention mask (same semantics as PyTorch’s API).
            memory_mask: optional memory attention mask.
            tgt_key_padding_mask: optional padding mask for the target batch.
            memory_key_padding_mask: optional padding mask for the memory batch.
            tgt_is_causal: if ``True``, applies a causal mask to the target self-attention.
            memory_is_causal: if ``True``, applies a causal mask to the cross-attention.
        """
        tgt_key_padding_mask = F._canonical_mask(
            mask=tgt_key_padding_mask,
            mask_name="tgt_key_padding_mask",
            other_type=F._none_or_dtype(tgt_mask),
            other_name="tgt_mask",
            target_type=tgt.dtype,
        )
        tgt_mask = F._canonical_mask(
            mask=tgt_mask,
            mask_name="tgt_mask",
            other_type=None,
            other_name="",
            target_type=tgt.dtype,
            check_other=False,
        )

        memory_key_padding_mask = F._canonical_mask(
            mask=memory_key_padding_mask,
            mask_name="memory_key_padding_mask",
            other_type=F._none_or_dtype(memory_mask),
            other_name="memory_mask",
            target_type=memory.dtype,
        )
        memory_mask = F._canonical_mask(
            mask=memory_mask,
            mask_name="memory_mask",
            other_type=None,
            other_name="",
            target_type=memory.dtype,
            check_other=False,
        )

        x = tgt
        if self.norm_first:
            x = x + self._self_attention_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._multihead_attention_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )
            x = x + self._feed_forward_block(self.norm3(x))
        else:
            x = self.norm1(x + self._self_attention_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(
                x + self._multihead_attention_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            )
            x = self.norm3(x + self._feed_forward_block(x))

        return x

    def _self_attention_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
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

    def _multihead_attention_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.cross_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout2(x)

    def _feed_forward_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    @torch.no_grad()
    def reset_parameters(self, scheme="xavier_uniform") -> None:  # noqa: D102
        logger.debug(f"Resetting parameters of {self.__class__.__name__} with scheme: {scheme}")
        # Reset equivariant linear layers (symm_learning.nn.eLinear)
        self.linear1.reset_parameters(scheme)
        self.linear2.reset_parameters(scheme)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        # Reset attention layers:
        self.self_attn.reset_parameters(scheme)
        self.cross_attn.reset_parameters(scheme)

    @torch.no_grad()
    def check_equivariance(
        self,
        batch_size: int = 4,
        tgt_len: int = 3,
        mem_len: int = 5,
        samples: int = 20,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        """Quick sanity check ensuring both attention blocks and the full layer are equivariant."""
        G = self.in_rep.group

        def act(rep: Representation, g, tensor: Tensor) -> Tensor:
            mat = torch.tensor(rep(g), dtype=tensor.dtype, device=tensor.device)
            return torch.einsum("ij,...j->...i", mat, tensor)

        for _ in range(samples):
            g = G.sample()
            tgt = torch.randn(batch_size, tgt_len, self.in_rep.size, device=self.norm1.Q.device)
            mem = torch.randn(batch_size, mem_len, self.in_rep.size, device=self.norm1.Q.device)
            g_tgt = act(self.in_rep, g, tgt)
            g_mem = act(self.in_rep, g, mem)

            sa = self._self_attention_block(tgt)
            g_sa = act(self.in_rep, g, sa)
            g_sa_exp = self._self_attention_block(g_tgt)
            assert torch.allclose(g_sa, g_sa_exp, atol=atol, rtol=rtol), (
                f"Self-attention equivarinace failed max error: {torch.max(g_sa - g_sa_exp).item():.3e}"
            )

            ca = self._multihead_attention_block(tgt, mem)
            g_ca = act(self.in_rep, g, ca)
            g_ca_exp = self._multihead_attention_block(g_tgt, g_mem)
            assert torch.allclose(g_ca, g_ca_exp, atol=atol, rtol=rtol), (
                f"Cross-attention equivarinace failed max error: {torch.max(g_ca - g_ca_exp).item():.3e}"
            )

            out = self(tgt, mem)
            g_out = act(self.in_rep, g, out)
            g_out_exp = self(g_tgt, g_mem)
            assert torch.allclose(g_out, g_out_exp, atol=atol, rtol=rtol), (
                f"Transormer decoder equivarinace failed max error: {torch.max(g_ca - g_ca_exp).item():.3e}"
            )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


if __name__ == "__main__":
    import logging

    # logging.basicConfig(level=logging.DEBUG)
    from escnn.group import CyclicGroup, DihedralGroup, Icosahedral

    from symm_learning.models.transformer.etransformer import eTransformerEncoderLayer
    from symm_learning.utils import check_equivariance, describe_memory

    G = CyclicGroup(2)
    m = 2
    in_rep = direct_sum([G.regular_representation] * m)

    encoder_kwargs = dict(
        in_rep=in_rep,
        nhead=1,
        dim_feedforward=in_rep.size * 4,
        dropout=0.1,
        activation="relu",
        norm_first=True,
        batch_first=True,
    )
    etransformer = eTransformerEncoderLayer(**encoder_kwargs)
    etransformer.eval()  # disable dropout for the test
    # describe_memory("transformer encoder", etransformer)
    check_equivariance(
        lambda x: etransformer._feed_forward_block(x),
        in_rep=etransformer.in_rep,
        out_rep=etransformer.out_rep,
        module_name="feed forward",
    )
    check_equivariance(
        lambda x: etransformer._self_attention_block(x),
        in_rep=etransformer.in_rep,
        out_rep=etransformer.out_rep,
        module_name="self_attention",
    )

    for depth in [1, 3, 5, 10]:
        base_layer = eTransformerEncoderLayer(**encoder_kwargs)
        base_layer.reset_parameters()
        base_layer.eval()
        encoder_stack = torch.nn.TransformerEncoder(
            encoder_layer=base_layer, num_layers=depth, enable_nested_tensor=False
        )
        for layer in encoder_stack.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        encoder_stack.eval()
        print(f"\n Testing encoder stack depth={depth} equivariance...")
        check_equivariance(
            encoder_stack,
            input_dim=3,
            module_name=f"encoder stack depth={depth}",
            atol=1e-4,
            rtol=1e-4,
            in_rep=in_rep,
            out_rep=in_rep,
        )
        print(f"Encoder stack depth={depth} equivariance test passed")

    print("\n\n\nTesting decoder layer equivariance...")

    decoder_kwargs = dict(
        in_rep=in_rep,
        nhead=1,
        dim_feedforward=in_rep.size * 2,
        dropout=0.0,
        activation="relu",
        norm_first=True,
        batch_first=True,
    )
    tdecoder = eTransformerDecoderLayer(**decoder_kwargs)
    tdecoder.eval()
    tdecoder.check_equivariance()

    def check_decoder_stack(module: torch.nn.Module, rep: Representation, depth: int, atol=1e-4, rtol=1e-4):  # noqa: D103
        G = rep.group

        def act(rep: Representation, g, tensor: Tensor) -> Tensor:
            mat = torch.tensor(rep(g), dtype=tensor.dtype, device=tensor.device)
            return torch.einsum("ij,...j->...i", mat, tensor)

        B, tgt_len, mem_len = 11, 3, 5
        module.eval()
        for _ in range(10):
            g = G.sample()
            tgt = torch.randn(B, tgt_len, rep.size)
            mem = torch.randn(B, mem_len, rep.size)
            out = module(tgt=tgt, memory=mem)
            g_tgt = act(rep, g, tgt)
            g_mem = act(rep, g, mem)
            g_out = module(tgt=g_tgt, memory=g_mem)
            g_out_exp = act(rep, g, out)
            assert torch.allclose(g_out, g_out_exp, atol=atol, rtol=rtol), (
                f"Decoder stack depth={depth} equivariance failed, max err {(g_out - g_out_exp).abs().max().item():.3e}"
            )

    for depth in (1, 3, 5, 10):
        base_layer = eTransformerDecoderLayer(**decoder_kwargs)
        base_layer.reset_parameters()
        base_layer.eval()
        decoder_stack = torch.nn.TransformerDecoder(decoder_layer=base_layer, num_layers=depth)
        for layer in decoder_stack.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        decoder_stack.eval()
        print(f"\n Testing decoder stack depth={depth} equivariance...")
        check_decoder_stack(decoder_stack, in_rep, depth=depth, atol=1e-3, rtol=1e-3)
        print(f"Decoder stack depth={depth} equivariance test passed")
    print("Decoder equivariance test passed")
