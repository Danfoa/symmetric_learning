from __future__ import annotations

import logging

import torch
from escnn.group import Representation, directsum
from torch.nn.utils import parametrize

from symm_learning.nn.linear import eLinear
from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint

logger = logging.getLogger(__name__)


class eMultiheadAttention(torch.nn.MultiheadAttention):
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
        parametrize.register_parametrization(self, "in_proj_weight", CommutingConstraint(in_rep, stacked_qkv_rep))
        if bias and self.in_proj_bias is not None:
            parametrize.register_parametrization(self, "in_proj_bias", InvariantConstraint(stacked_qkv_rep))

        # Replace output projection linear layer.
        self.out_proj = eLinear(in_rep, in_rep, bias=bias).to(device=device, dtype=dtype)

        self.reset_parameters(scheme="xavier_uniform")

    @torch.no_grad()
    def reset_parameters(self, scheme="xavier_uniform") -> None:
        """Overload parent method to take into account equivariance constraints."""
        if not hasattr(self, "parametrizations"):
            return super()._reset_parameters()

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


if __name__ == "__main__":
    from escnn.group import CyclicGroup, DihedralGroup

    from symm_learning.models.transformer.etransformer import eTransformerEncoderLayer
    from symm_learning.utils import check_equivariance

    G = DihedralGroup(10)
    m = 6
    in_rep = directsum([G.regular_representation] * m)

    for n_heads in [1, 2, 3]:
        eattention = eMultiheadAttention(in_rep=in_rep, num_heads=n_heads, bias=True, batch_first=True, dropout=0.1)
        eattention.eval()  # disable dropout for the test
        check_equivariance(
            lambda x: eattention(x, x, x, need_weights=False)[0],
            input_dim=3,
            in_rep=eattention.in_rep,
            out_rep=eattention.out_rep,
            atol=1e-3,
            rtol=1e-3,
        )

        print(f"Equivariance test passed for eMultiheadAttention with {n_heads} heads!")
