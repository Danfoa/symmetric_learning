from __future__ import annotations

import logging
import random
import time
from typing import Optional, Tuple

import torch
from escnn.group import Representation

import symm_learning
from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.models.difussion.cond_transformer_regressor import GenCondRegressor
from symm_learning.models.difussion.cond_unet1d import SinusoidalPosEmb
from symm_learning.representation_theory import direct_sum
from symm_learning.utils import module_memory

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
        init_scheme: str = "xavier_uniform",
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
        regular_copies = max(1, embedding_dim // G.order())
        self.embedding_rep = direct_sum([G.regular_representation] * regular_copies)

        self.register_buffer("invariant_projector", invariant_orthogonal_projector(self.embedding_rep))

        self.input_emb = symm_learning.nn.eLinear(in_rep, self.embedding_rep, bias=True, init_scheme=None)
        self.cond_emb = symm_learning.nn.eLinear(cond_rep, self.embedding_rep, bias=True, init_scheme=None)
        self.opt_time_emb = SinusoidalPosEmb(embedding_dim)

        self.pos_emb = torch.nn.Parameter(torch.zeros(1, in_horizon, embedding_dim))
        self.cond_pos_emb = torch.nn.Parameter(torch.zeros(1, cond_horizon + 1, embedding_dim))

        # Encoder parameterized as an equivariant MLP or a Transformer
        if num_cond_layers > 0:
            encoder_layer = symm_learning.models.eTransformerEncoderLayer(
                in_rep=self.embedding_rep,
                nhead=num_attention_heads,
                dim_feedforward=4 * embedding_dim,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # important for stability.
                norm_module="rmsnorm",  # important for stability.
                init_scheme=None,
            )
            logger.debug(
                f"Initializing {num_cond_layers} layers of eTransformerEncoderLayer of "
                f"{sum(p.numel() for p in encoder_layer.parameters()) / 1e6:.2f}M parameters each"
            )
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=num_cond_layers, enable_nested_tensor=False
            )
        else:
            hidden_rep = direct_sum([self.embedding_rep] * 4)
            logger.debug(f"Initializing eMLP encoder with hidden representation {hidden_rep}")
            self.encoder = torch.nn.Sequential(
                symm_learning.nn.eLinear(in_rep=self.embedding_rep, out_rep=hidden_rep, bias=True, init_scheme=None),
                torch.nn.Mish(),
                symm_learning.nn.eLinear(in_rep=hidden_rep, out_rep=self.embedding_rep, bias=True, init_scheme=None),
            )

        decoder_layer = symm_learning.models.eTransformerDecoderLayer(
            in_rep=self.embedding_rep,
            nhead=num_attention_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # important for stability.
            norm_module="rmsnorm",  # important for stability.
            init_scheme=None,
        )
        logger.debug(f"Initializing {num_layers} layers of eTransformerDecoderLayer")
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        # Self-Attention and Cross-Attention mask.
        # Cross-attention is used to compute updates to the action vector based on the conditioing tokens
        # composed of a inference-time optimization step token and the observation conditioning tokens.
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
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

        self.layer_norm = symm_learning.nn.eLayerNorm(self.embedding_rep, eps=1e-5, equiv_affine=True, bias=True)
        self.head = symm_learning.nn.eLinear(self.embedding_rep, out_rep, bias=True, init_scheme=None)

        self.reset_parameters(scheme=init_scheme)
        trainable_mem, non_trainable_mem = module_memory(self, units="MiB")
        logger.info(
            f"[{self.__class__.__name__}]: {trainable_mem:.3f} MiB trainable, {non_trainable_mem:.3f} "
            f"MiB non-trainable parameters."
        )

    @torch.no_grad()
    def reset_parameters(self, scheme="xavier_uniform") -> None:
        """Re-initialize all parameters."""
        logger.debug(f"Resetting parameters of {self.__class__.__name__} with scheme: {scheme}")
        # Initialize eLinear layers.
        self.input_emb.reset_parameters(scheme=scheme)
        self.cond_emb.reset_parameters(scheme=scheme)
        self.head.reset_parameters(scheme=scheme)
        # Initialize final layer norm and head.
        self.layer_norm.reset_parameters()
        # Initalize conditional encoder layers.
        if isinstance(self.encoder, torch.nn.TransformerEncoder):
            for i, layer in enumerate(self.encoder.layers, start=1):
                assert isinstance(layer, symm_learning.models.eTransformerEncoderLayer)
                logger.debug(f"Resetting encoder layer {i}:[{layer.__class__.__name__}] with scheme: {scheme}")
                layer.reset_parameters(scheme=scheme)
        else:  # eMLP.
            for i, module in enumerate(self.encoder, start=1):
                if isinstance(module, symm_learning.nn.eLinear):
                    logger.debug(f"Resetting encoder module {i}:[{module.__class__.__name__}] with scheme: {scheme}")
                    module.reset_parameters(scheme=scheme)
        # Initialize decoder layers.
        for i, layer in enumerate(self.decoder.layers, start=1):
            assert isinstance(layer, symm_learning.models.eTransformerDecoderLayer)
            logger.debug(f"Resetting decoder layer {i}:[{layer.__class__.__name__}] with scheme: {scheme}")
            layer.reset_parameters(scheme=scheme)

        logger.info(f"[{self.__class__.__name__}]: parameters initialized with `{scheme}` scheme.")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Todo."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (symm_learning.nn.eLinear, symm_learning.nn.eMultiheadAttention)
        blacklist_weight_modules = (symm_learning.nn.eLayerNorm, torch.nn.Embedding)

        for module_name, m in self.named_modules():
            for param_name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                fpn = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias") or param_name.startswith("bias"):
                    no_decay.add(fpn)
                elif param_name.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif param_name.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                else:
                    raise ValueError(f"Unrecognized parameter {fpn} in module {module_name}")

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

    def forward(
        self,
        X: torch.Tensor,
        opt_step: torch.Tensor | float | int,
        Z: torch.Tensor,
    ):
        r"""Forward pass approximating :math:`V_k = f(X_k, Z, k)`."""
        assert X.dim() == 3 and X.shape[-1] == self.in_rep.size and X.shape[1] <= self.in_horizon, (
            f"Expected X shape (B, Tx<={self.in_horizon}, {self.in_rep.size}) got {X.shape}"
        )
        assert Z.dim() == 3 and Z.shape[-1] == self.cond_rep.size and Z.shape[1] <= self.cond_horizon, (
            f"Expected Z shape (B, Tz<={self.cond_horizon}, {self.cond_rep.size}) got {Z.shape}"
        )
        batch_size = X.shape[0]

        # 1. Inference-time optimization step embedding (k). First conditioning token.
        if isinstance(opt_step, torch.Tensor):
            opt_steps = opt_step.to(device=X.device, dtype=torch.float32)
        else:
            opt_steps = torch.scalar_tensor(opt_step, device=X.device, dtype=torch.float32)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        opt_steps = opt_steps.reshape(-1).expand(batch_size)
        opt_time_emb = self.opt_time_emb(opt_steps).unsqueeze(1)  # (B, 1, D)
        # Project time embedding onto embedding space's invariant subspace
        opt_time_emb = torch.einsum("ij,...j->...i", self.invariant_projector, opt_time_emb)

        # 2. Conditioning variable Z embedding/tokenization
        z_cond_emb = self.cond_emb(Z)  # (B, Tz-1, D)
        cond_embeddings = torch.cat([opt_time_emb, z_cond_emb], dim=1)  # (B, Tz, D)
        cond_horizon = z_cond_emb.shape[1]  # (Tz)
        # Project time embedding onto embedding space's invariant subspace
        cond_pos_emb = torch.einsum(
            "ij,...j->...i", self.invariant_projector, self.cond_pos_emb[:, : cond_horizon + 1, :]
        )
        # Transformer encoder of conditing tokens
        cond_tokens = self.dropout(cond_embeddings + cond_pos_emb)  # (B, Tz, D)
        cond_tokens = self.encoder(cond_tokens)  # (B, Tz, D)

        # 3. Input embedding/tokenization
        input_tokens = self.input_emb(X)  # (B, Tx, D)

        # 4. Transformer encoder of input tokens with self-attention and cross-attention to cond tokens
        input_horizon = input_tokens.shape[1]  # (Tx)
        # Project time embedding onto embedding space's invariant subspace
        pos_emb = torch.einsum("ij,...j->...i", self.invariant_projector, self.pos_emb[:, :input_horizon, :])
        input_tokens = self.dropout(input_tokens + pos_emb)  # (B, Tx, D)

        out_tokens = self.decoder(
            tgt=input_tokens, memory=cond_tokens, tgt_mask=self.self_att_mask, memory_mask=self.cross_att_mask
        )  # (B, Tx, D)
        # 5. Regression head projecting to output dimension.
        out_tokens = self.layer_norm(out_tokens)
        out = self.head(out_tokens)  # (B, Tx, out_dim)
        return out

    @torch.no_grad()
    def check_equivariance(  # noqa: D102
        self,
        batch_size: int = 10,
        in_len: int = 10,
        cond_len: int = 5,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        import escnn

        G = self.in_rep.group
        in_len = min(in_len, self.in_horizon)
        cond_len = min(cond_len, self.cond_horizon)
        training_mode = self.training
        self.eval()

        def act(rep: Representation, g: escnn.group.GroupElement, x: torch.Tensor) -> torch.Tensor:
            mat = torch.tensor(rep(g), dtype=x.dtype, device=x.device)
            return torch.einsum("ij,...j->...i", mat, x)

        device = self.pos_emb.device
        dtype = self.pos_emb.dtype

        for _ in range(min(10, G.order())):
            g = random.choice(list(G.elements[1:]))  # skip identity
            X = torch.randn(batch_size, in_len, self.in_rep.size, device=device, dtype=dtype)
            Z = torch.randn(batch_size, cond_len, self.cond_rep.size, device=device, dtype=dtype)
            k = torch.randn(batch_size, device=device, dtype=dtype)

            Y = self(X=X, Z=Z, opt_step=k)
            # Evaluate on symmetric points.
            gX = act(self.in_rep, g, X)
            gZ = act(self.cond_rep, g, Z)
            gY = self(X=gX, Z=gZ, opt_step=k)

            gY_expected = act(self.out_rep, g, Y)

            assert torch.allclose(gY, gY_expected, atol=atol, rtol=rtol), (
                f"Equivariance test failed for group element {g}.\n"
                f"Max absolute difference: {torch.max(torch.abs(gY - gY_expected))}\n"
            )

        if training_mode:
            self.train()


if __name__ == "__main__":
    from symm_learning.utils import describe_memory

    # Set logging to debug
    logging.basicConfig(level=logging.DEBUG)
    # Set debug message to [name][level][time][message]
    formatter = logging.Formatter("[%(name)s][%(levelname)s][%(asctime)s]: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    from escnn.group import CyclicGroup, Icosahedral

    # G = Icosahedral()
    # # G = CyclicGroup(2)
    # d = 70
    # # m = int(70 // G.order())
    # m = 1
    # in_rep = directsum([G.regular_representation] * m)  # dim 8
    # cond_rep = in_rep
    # out_rep = in_rep

    # Tx, Tz = 8, 6
    # model = eCondTransformerRegressor(
    #     in_rep=in_rep,
    #     cond_rep=cond_rep,
    #     out_rep=out_rep,
    #     in_horizon=Tx,
    #     cond_horizon=Tz,
    #     num_layers=8,
    #     num_attention_heads=1,
    #     embedding_dim=G.order() * m,
    #     num_cond_layers=0,
    # ).to(device=device, dtype=dtype)

    # model.check_equivariance()
    # G = Icosahedral()
    G = CyclicGroup(2)
    m = 5
    embedding_dim = G.order() * m * 4
    in_rep = direct_sum([G.regular_representation] * m)  # dim 8
    print(in_rep.size)
    cond_rep = in_rep
    out_rep = in_rep

    Tx, Tz = 8, 6

    start_time = time.time()
    model = eCondTransformerRegressor(
        in_rep=in_rep,
        cond_rep=cond_rep,
        out_rep=out_rep,
        in_horizon=Tx,
        cond_horizon=Tz,
        num_layers=5,
        num_attention_heads=1,
        embedding_dim=embedding_dim,
        num_cond_layers=0,
    )
    # print(describe_memory("eCondTransformer", model))
    print(f"Model initialized in {time.time() - start_time:.2f} seconds")

    model.to(device=device, dtype=dtype)
    model.eval()
    model.check_equivariance(atol=1e-2, rtol=1e-2)

    print("Equivariance test passed!")
