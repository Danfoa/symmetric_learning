from __future__ import annotations

import torch
from escnn.group import Representation, directsum
from torch.nn.utils import parametrize

from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint


class eMultiheadAttention(torch.nn.MultiheadAttention):
    """Equivariant Multi-head attention built by constraining PyTorch's implementation."""

    def __init__(
        self,
        in_rep: Representation,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        device=None,
        dtype=None,
    ) -> None:
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
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.in_rep, self.out_rep = in_rep, in_rep
        self._regular_stack_rep = directsum([G.regular_representation] * regular_copies)

        if not self._qkv_same_embed_dim:
            raise ValueError("eMultiheadAttention requires kdim == vdim == embed_dim.")

        stacked_qkv_rep = directsum([self._regular_stack_rep] * 3)
        parametrize.register_parametrization(
            self,
            "in_proj_weight",
            CommutingConstraint(in_rep, stacked_qkv_rep),
        )
        if bias and self.in_proj_bias is not None:
            parametrize.register_parametrization(self, "in_proj_bias", InvariantConstraint(stacked_qkv_rep))

        parametrize.register_parametrization(self.out_proj, "weight", CommutingConstraint(in_rep, in_rep))
        if bias and self.out_proj.bias is not None:
            parametrize.register_parametrization(self.out_proj, "bias", InvariantConstraint(in_rep))


if __name__ == "__main__":
    from escnn.group import CyclicGroup

    from symm_learning.models.transformer.etransformer import eTransformerEncoderLayer
    from symm_learning.utils import check_equivariance

    G = CyclicGroup(4)
    m = 10
    in_rep = directsum([G.regular_representation] * m)

    for n_heads in [1, 2, 5, 10]:
        eattention = eMultiheadAttention(in_rep=in_rep, num_heads=n_heads, bias=True, batch_first=True)

        check_equivariance(
            lambda x: eattention(x, x, x, need_weights=False)[0],
            input_dim=3,
            in_rep=eattention.in_rep,
            out_rep=eattention.out_rep,
            atol=1e-3,
            rtol=1e-3,
        )

        print(f"Equivariance test passed for eMultiheadAttention with {n_heads} heads!")
