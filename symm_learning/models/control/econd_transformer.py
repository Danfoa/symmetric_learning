from __future__ import annotations

import logging
import random
import time
from typing import Literal, Optional, Tuple

import torch
from escnn.group import Representation

import symm_learning
from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.models.control.cond_transformer import (
    GenCondRegressor,
    SinusoidalPosEmb,
    build_causal_attention_masks,
    build_cond_positions,
    build_input_positions,
)
from symm_learning.nn.module import eModule
from symm_learning.representation_theory import direct_sum
from symm_learning.utils import module_memory

logger = logging.getLogger(__name__)


class eCondTransformer(eModule, GenCondRegressor):
    r"""Equivariant encoder/decoder Transformer with configurable positional attention.

    Let :math:`A := \texttt{num\_cond\_layers}` and :math:`B := \texttt{num\_layers}`. This module is the
    equivariant counterpart of :class:`~symm_learning.models.control.cond_transformer.CondTransformer`: an
    encoder/decoder Transformer with :math:`A` conditioning encoder layers and :math:`B` decoder layers,
    following the architecture introduced in *Attention Is All You Need* by Vaswani, Shazeer, Parmar,
    Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin (NeurIPS 2017), while constraining every learnable map
    to respect the prescribed group actions.

    The conditioning stream is assembled as

    .. math::
        [k, \mathbf{z}_{-(T_z - 1)}, \ldots, \mathbf{z}_{0}],

    where :math:`k` is the inference-time optimisation step token and
    :math:`\mathbf{z}_{-(T_z - 1)}, \ldots, \mathbf{z}_{0}` are the conditioning tokens ordered from oldest
    to most recent observation. The decoder predicts the target sequence

    .. math::
        [\mathbf{x}_{0}, \ldots, \mathbf{x}_{T_x - 1}],

    ordered from the first action to the last action in the predicted horizon.

    Supported positional encodings are:

    ``"additive_absolute"``
        Uses :class:`~symm_learning.nn.activation.eAdditivePosMultiheadAttention`. Learned absolute
        positions are added in the equivariant embedding space before self-attention or cross-attention.

    ``"additive_relative"``
        Uses :class:`~symm_learning.nn.activation.eAdditiveRelMultiheadAttention`. Learned relative
        distance biases are injected into attention logits, preserving the time-translation structure of the
        sequence while keeping the feature maps equivariant.

    ``"none"``
        Uses :class:`~symm_learning.nn.activation.eMultiheadAttention` with no explicit positional encoding.

    Temporal assumptions:

    * :math:`Z` must already be ordered in time from past to present.
    * :math:`X` must already be ordered from the first predicted action to the last predicted action.
    * For ``"additive_relative"``, the last conditioning token :math:`\mathbf{z}_{0}` and the first action
      token :math:`\mathbf{x}_{0}` are both placed at time index :math:`0`, so cross-attention is anchored
      at the present time.
    * The optimisation-step token :math:`k` is prepended to the conditioning memory, but it is not treated as
      part of the observation timeline.

    Equivariance is enforced by embedding tokens into a representation space built from copies of the regular
    representation, projecting the scalar step embedding onto the invariant subspace, and using equivariant
    encoder, decoder, normalization, and head layers throughout. The resulting conditional map satisfies

    .. math::
        \mathbf{f}_{\mathbf{\theta}}(\rho_{\mathcal{X}}(g)\mathbf{X}_k,\,
        \rho_{\mathcal{Z}}(g)\mathbf{Z},\, k)
        = \rho_{\mathcal{Y}}(g)\,\mathbf{f}_{\mathbf{\theta}}(\mathbf{X}_k,\mathbf{Z},k),
        \qquad \forall g \in \mathbb{G}.
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
        pos_encoding: Literal["additive_absolute", "additive_relative", "none"] = "additive_absolute",
        norm_first: bool = True,
        norm_module: Literal["layernorm", "rmsnorm"] = "rmsnorm",
        init_scheme: str = "xavier_uniform",
    ) -> None:
        r"""Create an equivariant conditional transformer regressor.

        Args:
            in_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{in}}` of the input tokens.
            cond_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{cond}}` of the
                conditioning tokens.
            out_rep (:class:`~escnn.group.Representation`, optional): Output representation :math:`\rho_{\text{out}}`.
                Defaults to ``in_rep`` if ``None``.
            in_horizon: Maximum length of the input sequence.
            cond_horizon: Maximum length of the conditioning sequence.
            num_layers: Number of transformer decoder layers in the main generation trunk.
            num_attention_heads: Number of attention heads in transformer layers.
            embedding_dim: Dimension of the regular representation embedding space.
                Must be a multiple of the group order.
            p_drop_emb: Dropout probability for embeddings.
            p_drop_attn: Dropout probability inside attention blocks.
            causal_attn: Whether to mask future tokens (causal masking).
            num_cond_layers: Number of transformer encoder layers for processing conditioning tokens.
                If 0, an eMLP is used instead.
            pos_encoding: Positional attention backend (``"additive_absolute"``,
                ``"additive_relative"``, or ``"none"``).
            norm_first: Whether to apply normalization before each residual branch.
            norm_module: Normalization layer type (``'layernorm'`` or ``'rmsnorm'``).
            init_scheme: Initialization scheme for equivariant layers.
        """
        out_rep = out_rep or in_rep
        super().__init__(in_rep.size, out_rep.size, cond_rep.size)

        self.in_rep = in_rep
        self.out_rep = out_rep
        self.cond_rep = cond_rep
        self.in_horizon = in_horizon
        self.cond_horizon = cond_horizon
        self.cond_token_horizon = cond_horizon + 1
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.pos_encoding = pos_encoding
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
        max_pos_len = max(self.in_horizon, self.cond_token_horizon)
        max_rel_distance = self.in_horizon + self.cond_token_horizon - 2

        def _build_attn():
            if pos_encoding == "additive_absolute":
                return symm_learning.nn.eAdditivePosMultiheadAttention(
                    in_rep=self.embedding_rep,
                    num_heads=num_attention_heads,
                    max_len=max_pos_len,
                    dropout=p_drop_attn,
                    bias=True,
                    init_scheme=None,
                )
            elif pos_encoding == "additive_relative":
                return symm_learning.nn.eAdditiveRelMultiheadAttention(
                    in_rep=self.embedding_rep,
                    num_heads=num_attention_heads,
                    max_distance=max_rel_distance,
                    dropout=p_drop_attn,
                    bias=True,
                    init_scheme=None,
                )
            elif pos_encoding == "none":
                return symm_learning.nn.eMultiheadAttention(
                    in_rep=self.embedding_rep,
                    num_heads=num_attention_heads,
                    dropout=p_drop_attn,
                    bias=True,
                    init_scheme=None,
                )
            else:
                raise ValueError(
                    f"Unknown pos_encoding={pos_encoding!r}. Expected "
                    "'additive_absolute', 'additive_relative', or 'none'."
                )

        # Conditioning encoder
        if num_cond_layers > 0:
            encoder_layer = symm_learning.nn.eTransformerEncoderLayer(
                in_rep=self.embedding_rep,
                self_attn=_build_attn(),
                dim_feedforward=4 * embedding_dim,
                dropout=p_drop_attn,
                activation=torch.nn.GELU(),
                norm_first=norm_first,
                norm_module=norm_module,
                init_scheme=None,
            )
            logger.debug(
                f"Initializing {num_cond_layers} layers of eTransformerEncoderLayer of "
                f"{sum(p.numel() for p in encoder_layer.parameters()) / 1e6:.2f}M parameters each"
            )
            self.encoder = symm_learning.nn.TransformerEncoder(
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

        # Decoder
        decoder_layer = symm_learning.nn.eTransformerDecoderLayer(
            in_rep=self.embedding_rep,
            self_attn=_build_attn(),
            multihead_attn=_build_attn(),
            dim_feedforward=4 * embedding_dim,
            dropout=p_drop_attn,
            activation=torch.nn.GELU(),
            norm_first=norm_first,
            norm_module=norm_module,
            init_scheme=None,
        )
        logger.debug(f"Initializing {num_layers} layers of eTransformerDecoderLayer")
        self.decoder = symm_learning.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        # Self-Attention and Cross-Attention mask.
        # Cross-attention is used to compute updates to the action vector based on the conditioing tokens
        # composed of a inference-time optimization step token and the observation conditioning tokens.
        if causal_attn:
            self_att_mask, cross_att_mask = build_causal_attention_masks(in_horizon, self.cond_token_horizon)
            self.register_buffer("self_att_mask", self_att_mask)
            self.register_buffer("cross_att_mask", cross_att_mask)
        else:
            self.self_att_mask = None
            self.cross_att_mask = None

        # Decoder head
        if norm_module == "layernorm":
            self.layer_norm = symm_learning.nn.eLayerNorm(self.embedding_rep, eps=1e-5, equiv_affine=True, bias=True)
        else:  # rmsnorm
            self.layer_norm = symm_learning.nn.eRMSNorm(self.embedding_rep, eps=1e-5, equiv_affine=True)
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
        self.input_emb.reset_parameters(scheme=scheme)
        self.cond_emb.reset_parameters(scheme=scheme)
        self.layer_norm.reset_parameters()
        self.head.reset_parameters(scheme=scheme)

        # Initalize conditional encoder layers.
        if isinstance(self.encoder, symm_learning.nn.TransformerEncoder):
            for i, layer in enumerate(self.encoder.layers, start=1):
                if not isinstance(layer, symm_learning.nn.eTransformerEncoderLayer):
                    raise RuntimeError(f"Unaccounted encoder layer {i}: {layer}")
                logger.debug(f"Resetting encoder layer {i}:[{layer.__class__.__name__}] with scheme: {scheme}")
                layer.reset_parameters(scheme=scheme)
        elif isinstance(self.encoder, torch.nn.Sequential):  # eMLP.
            for i, module in enumerate(self.encoder, start=1):
                if isinstance(module, symm_learning.nn.eLinear):
                    logger.debug(f"Resetting encoder module {i}:[{module.__class__.__name__}] with scheme: {scheme}")
                    module.reset_parameters(scheme=scheme)
                elif any(param.requires_grad for param in module.parameters(recurse=False)):
                    raise RuntimeError(f"Unaccounted encoder module {i}: {module}")
        else:
            raise RuntimeError(f"Unaccounted encoder module {self.encoder}")
        # Initialize decoder layers.
        for i, layer in enumerate(self.decoder.layers, start=1):
            if not isinstance(layer, symm_learning.nn.eTransformerDecoderLayer):
                raise RuntimeError(f"Unaccounted decoder layer {i}: {layer}")
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
                elif (
                    param_name.endswith("weight")
                    and isinstance(m, blacklist_weight_modules)
                    or param_name in {"pos_emb", "rel_bias"}
                ):
                    no_decay.add(fpn)
                else:
                    raise ValueError(f"Unrecognized parameter {fpn} in module {module_name}")

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
        self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95)
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
        z_cond_emb = self.cond_emb(Z)
        cond_embeddings = torch.cat([opt_time_emb, z_cond_emb], dim=1)
        cond_tokens = self.dropout(cond_embeddings)
        cond_token_horizon = cond_embeddings.shape[1]
        cond_positions, cond_position_mask = build_cond_positions(
            self.pos_encoding, cond_token_horizon, device=X.device
        )

        if isinstance(self.encoder, symm_learning.nn.TransformerEncoder):
            cond_tokens = self.encoder(
                cond_tokens,
                src_positions=cond_positions,
                src_position_mask=cond_position_mask,
            )
        else:
            cond_tokens = self.encoder(cond_tokens)

        # 3. Input embedding/tokenization
        input_tokens = self.dropout(self.input_emb(X))

        # 4. Transformer encoder of input tokens with self-attention and cross-attention to cond tokens
        input_horizon = input_tokens.shape[1]
        input_positions = torch.arange(input_horizon, device=X.device)
        input_position_mask = torch.ones(input_horizon, device=X.device, dtype=torch.bool)
        out_tokens = self.decoder(
            tgt=input_tokens,
            memory=cond_tokens,
            tgt_mask=self.self_att_mask,
            memory_mask=self.cross_att_mask,
            tgt_positions=input_positions,
            tgt_position_mask=input_position_mask,
            memory_positions=cond_positions,
            memory_position_mask=cond_position_mask,
        )
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

        device = self.invariant_projector.device
        dtype = self.invariant_projector.dtype

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
    model = eCondTransformer(
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
