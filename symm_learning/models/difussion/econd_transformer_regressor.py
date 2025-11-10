from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
from escnn.group import Representation, directsum
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList

from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.models.difussion.cond_transformer_regressor import GenCondRegressor
from symm_learning.models.difussion.cond_unet1d import SinusoidalPosEmb
from symm_learning.models.transformer.etransformer import eTransformerDecoderLayer, eTransformerEncoderLayer
from symm_learning.nn.activation import eMultiheadAttention
from symm_learning.nn.linear import eAffine, eLinear
from symm_learning.nn.normalization import eLayerNorm
from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint

logger = logging.getLogger(__name__)


class eCondTransformerRegressor(GenCondRegressor):
    r"""Equivariant analogue of :class:`CondTransformerRegressor`.

    Tokens transforming according to ``in_rep`` are embedded into an ``embedding_rep`` space built from copies of the
    regular representation so that :class:`eTransformerEncoderLayer`/:class:`eTransformerDecoderLayer` can be used
    directly. Positional encodings and timestep embeddings are projected onto the invariant subspace so they can be
    added to equivariant tokens without breaking symmetry.
    """

    def __init__(
        self,
        in_rep: Representation,
        cond_rep: Representation,
        out_rep: Optional[Representation],
        in_horizon: int,
        cond_horizon: int,
        num_layers: int,
        num_attention_heads: int,
        embedding_dim: int,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        num_cond_layers: int = 0,
    ) -> None:
        out_rep = out_rep or in_rep
        super().__init__(in_rep.size, out_rep.size, cond_rep.size)

        self.in_rep = in_rep
        self.out_rep = out_rep
        self.cond_rep = cond_rep
        self.in_horizon = in_horizon
        self.cond_horizon = cond_horizon
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.dropout = torch.nn.Dropout(p_drop_emb)

        G = in_rep.group
        assert cond_rep.group == G == out_rep.group, "All representations must belong to the same group"
        if embedding_dim % G.order() != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of the group order ({G.order()})")
        regular_copies = embedding_dim // G.order()
        self.embedding_rep = directsum([G.regular_representation] * regular_copies)

        self.register_buffer("invariant_projector", invariant_orthogonal_projector(self.embedding_rep))

        self.input_emb = eLinear(in_rep, self.embedding_rep, bias=True)
        self.cond_emb = eLinear(cond_rep, self.embedding_rep, bias=True)
        self.opt_time_emb = SinusoidalPosEmb(embedding_dim)

        self.pos_emb = torch.nn.Parameter(torch.zeros(1, in_horizon, embedding_dim))
        self.cond_pos_emb = torch.nn.Parameter(torch.zeros(1, cond_horizon, embedding_dim))

        if num_cond_layers > 0:
            self.encoder_layers = torch.nn.ModuleList(
                [
                    eTransformerEncoderLayer(
                        in_rep=self.embedding_rep,
                        nhead=num_attention_heads,
                        dim_feedforward=4 * embedding_dim,
                        dropout=p_drop_attn,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(num_cond_layers)
                ]
            )
            self.cond_mlp = None
        else:
            self.encoder_layers = None
            self.cond_mlp = torch.nn.Sequential(
                eLayerNorm(self.embedding_rep, eps=1e-5, equiv_affine=True, bias=True),
                eLinear(self.embedding_rep, self.embedding_rep, bias=True),
            )

        self.decoder_layers = torch.nn.ModuleList(
            [
                eTransformerDecoderLayer(
                    in_rep=self.embedding_rep,
                    nhead=num_attention_heads,
                    dim_feedforward=4 * embedding_dim,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        if causal_attn:
            mask = (torch.triu(torch.ones(in_horizon, in_horizon)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("self_att_mask", mask)

            t, s = torch.meshgrid(torch.arange(in_horizon), torch.arange(cond_horizon), indexing="ij")
            mask = t >= (s - 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("cross_att_mask", mask)
        else:
            self.self_att_mask = None
            self.cross_att_mask = None

        self.ln_f = eLayerNorm(self.embedding_rep, eps=1e-5, equiv_affine=True, bias=True)
        self.head = eLinear(self.embedding_rep, out_rep, bias=True)

        self.apply(self._init_weights)
        with torch.no_grad():
            projector = self.invariant_projector.to(self.pos_emb.device, self.pos_emb.dtype)
            self.pos_emb.copy_(torch.matmul(self.pos_emb, projector.T))
            self.cond_pos_emb.copy_(torch.matmul(self.cond_pos_emb, projector.T))
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    @staticmethod
    def _init_param(module: torch.nn.Module, name: str, init_fn: Callable[[torch.Tensor], None]) -> None:
        """Safely initialize a (possibly parametrized) parameter."""
        if not hasattr(module, name):
            return
        param = getattr(module, name)
        if param is None:
            return
        with torch.no_grad():
            if parametrize.is_parametrized(module, name):
                init_fn(getattr(module.parametrizations, name).original)
            else:
                init_fn(param)

    def _init_weights(self, module):
        class_name = module.__class__.__name__
        ignore_types = (
            torch.nn.Dropout,
            torch.nn.Sequential,
            SinusoidalPosEmb,
            eTransformerEncoderLayer,
            eTransformerDecoderLayer,
            CommutingConstraint,
            InvariantConstraint,
            ParametrizationList,
            torch.nn.ModuleList,
            torch.nn.ModuleDict,
            torch.nn.Identity,
            eAffine,
            self.__class__,
        )
        if isinstance(module, (eLinear, torch.nn.Linear)):
            self._init_param(module, "weight", lambda t: torch.nn.init.normal_(t, mean=0.0, std=0.02))
            self._init_param(module, "bias", torch.nn.init.zeros_)
        elif isinstance(module, eMultiheadAttention):
            self._init_param(module, "in_proj_weight", lambda t: torch.nn.init.normal_(t, mean=0.0, std=0.02))
            self._init_param(module, "in_proj_bias", torch.nn.init.zeros_)
            self._init_param(module.out_proj, "weight", lambda t: torch.nn.init.normal_(t, mean=0.0, std=0.02))
            self._init_param(module.out_proj, "bias", torch.nn.init.zeros_)
        elif isinstance(module, eLayerNorm):
            module.affine.reset_parameters()
        elif isinstance(module, ignore_types) or class_name.startswith("Parametrized"):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {class_name}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Todo."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (eLinear, eMultiheadAttention)
        blacklist_weight_modules = (eLayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} in both decay/no_decay"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params} not separated into decay/no_decay"
        )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(  # noqa: D102
        self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95)
    ):  # noqa: D102
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def _apply_cond_encoder(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.encoder_layers is not None:
            out = tokens
            for layer in self.encoder_layers:
                out = layer(out)
            return out
        return self.cond_mlp(tokens)

    def forward(self, X: torch.Tensor, Z: torch.Tensor, opt_step: torch.Tensor | float | int):
        r"""Forward pass approximating :math:`V_k = f(X_k, Z, k)`."""
        assert X.shape[1] <= self.in_horizon, f"Input horizon {X.shape[1]} larger than {self.in_horizon}"
        assert Z.shape[1] <= self.cond_horizon - 1, f"Cond horizon {Z.shape[1]} larger than {self.cond_horizon - 1}"

        projector = self.invariant_projector.to(X.device, X.dtype)

        if isinstance(opt_step, torch.Tensor):
            opt_steps = opt_step.to(device=X.device, dtype=torch.float32)
        else:
            opt_steps = torch.scalar_tensor(opt_step, device=X.device, dtype=torch.float32)
        opt_steps = opt_steps.reshape(-1).expand(X.shape[0])
        opt_time_emb = self.opt_time_emb(opt_steps).unsqueeze(1).to(dtype=X.dtype, device=X.device)
        opt_time_emb = torch.matmul(opt_time_emb, projector.T)

        cond_emb = self.cond_emb(Z)
        cond_embeddings = torch.cat([opt_time_emb, cond_emb], dim=1)
        cond_horizon = cond_embeddings.shape[1]
        cond_pos = torch.matmul(self.cond_pos_emb[:, :cond_horizon, :], projector.T)
        cond_tokens = self.dropout(cond_embeddings + cond_pos)
        cond_tokens = self._apply_cond_encoder(cond_tokens)

        input_tokens = self.input_emb(X)
        input_horizon = input_tokens.shape[1]
        pos = torch.matmul(self.pos_emb[:, :input_horizon, :], projector.T)
        input_tokens = self.dropout(input_tokens + pos)

        out_tokens = input_tokens
        for layer in self.decoder_layers:
            out_tokens = layer(
                tgt=out_tokens,
                memory=cond_tokens,
                tgt_mask=self.self_att_mask,
                memory_mask=self.cross_att_mask,
            )

        out_tokens = self.ln_f(out_tokens)
        return self.head(out_tokens)

    @torch.no_grad()
    def check_equivariance(  # noqa: D102
        self,
        batch_size: int = 3,
        in_len: int = 4,
        cond_len: int = 3,
        samples: int = 10,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        G = self.in_rep.group
        in_len = min(in_len, self.in_horizon)
        cond_len = min(cond_len, self.cond_horizon - 1)
        training_mode = self.training
        self.eval()

        def act(rep: Representation, g, tensor: torch.Tensor) -> torch.Tensor:
            mat = torch.tensor(rep(g), dtype=tensor.dtype, device=tensor.device)
            return torch.einsum("ij,...j->...i", mat, tensor)

        device = self.pos_emb.device
        dtype = self.pos_emb.dtype

        for _ in range(samples):
            g = G.sample()
            X = torch.randn(batch_size, in_len, self.in_rep.size, device=device, dtype=dtype)
            Z = torch.randn(batch_size, cond_len, self.cond_rep.size, device=device, dtype=dtype)
            k = torch.randn(batch_size, device=device, dtype=dtype)

            out = self(X, Z, k)
            out_g = act(self.out_rep, g, out)
            out_expected = self(act(self.in_rep, g, X), act(self.cond_rep, g, Z), k)
            torch.testing.assert_close(out_g, out_expected, atol=atol, rtol=rtol)
        if training_mode:
            self.train()


def test():  # noqa: D103
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    from escnn.group import CyclicGroup

    G = CyclicGroup(4)
    in_rep = directsum([G.regular_representation] * 2)  # dim 8
    cond_rep = in_rep
    out_rep = in_rep

    Tx, Tz = 8, 6
    model = eCondTransformerRegressor(
        in_rep=in_rep,
        cond_rep=cond_rep,
        out_rep=out_rep,
        in_horizon=Tx,
        cond_horizon=Tz,
        num_layers=3,
        num_attention_heads=2,
        embedding_dim=16,
        num_cond_layers=3,
    ).to(device=device, dtype=dtype)

    B = 4
    X = torch.randn(B, Tx, in_rep.size, device=device, dtype=dtype)
    Z = torch.randn(B, Tz - 1, cond_rep.size, device=device, dtype=dtype)
    k = torch.arange(B, device=device, dtype=dtype)
    out = model(X, Z, k)
    assert out.shape == (B, Tx, out_rep.size)
    print("eCondTransformerRegressor forward pass succeeded.")
    model.eval()
    model.check_equivariance()
    print("Equivariance check passed.")


if __name__ == "__main__":
    test()
