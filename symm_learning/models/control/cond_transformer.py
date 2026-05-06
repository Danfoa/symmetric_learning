from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Literal

import torch

from symm_learning.nn.activation import (
    AdditivePosMultiheadAttention,
    AdditiveRelMultiheadAttention,
    PositionalAttentionBase,
    RoPEMultiheadAttention,
    RotaryEmbedding,
)
from symm_learning.nn.transformer.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)

NormModule = Literal["layernorm", "rmsnorm"]
PosEncoding = Literal["additive_absolute", "additive_relative", "rope", "none"]


class GenCondRegressor(torch.nn.Module, ABC):
    r"""Generative Conditional Regressor module.

    This is an abstract module inteded to be used as the backbone of a conditional flow-matching/diffusion process which
    enables sampling from the conditional probability distribution:

    .. math::
        \mathbb{P}(X \mid Z)

    Let :math:`\mathcal{X}=\mathbb{R}^{d_x}`, :math:`\mathcal{Z}=\mathbb{R}^{d_z}`, and
    :math:`\mathcaleTransformerEncoderLayer{Y}=\mathbb{R}^{d_v}`.
    Where :math:`X = [x_0,\ldots,x_{T_x}] \in \mathcal{X}^{T_x}` is the input/data sample composed of a
    trajectory of :math:`T_x` points, and :math:`Z = [z_0,\ldots,z_{T_z}] \in \mathcal{Z}^{T_z}` is the
    conditioning/observation variable composed of :math:`T_z` points.

    The module parameterizes a conditional vector-valued regression map:

    .. math::
        \mathbf{f}_{\mathbf{\theta}}: \mathcal{X}^{T_x} \times \mathcal{Z}^{T_z} \times \mathbb{R}
        \to \mathcal{Y}^{T_x},

    with

    .. math::
        V_k = \mathbf{f}_{\mathbf{\theta}}(X_k, Z, k).

    Where :math:`k` denotes the inference-time optimization timestep (i.e., the step of the flow-matching/diffusion)
    process, :math:`X_k` is the noisy version of the data sample at step `k`, and
    :math:`V_k \in (\mathbb{R}^{d_v})^{T_x}` is the target regression vector-valued variable.
    For diffusion models :math:`V_k` typically corresponds to the score functional of
    :math:`\mathbb{P}_k(X \mid Z)`, while for flow-matching models it typically corresponds
    to the flow-matching velocity vector field.

    This abstract base class does not impose equivariance/invariance constraints by itself.
    """

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim

    @abstractmethod
    def forward(self, X: torch.Tensor, opt_step: torch.Tensor | float | int, Z: torch.Tensor):
        r"""Forward pass of the generative conditional regressor.

        Args:
            X (:class:`~torch.Tensor`): The input/data sample composed of a trajectory of `T_x` points in a
                `d_x`-dimensional space. Shape: `(B, T_x, d_x)`, where `B` is the batch size.
            opt_step (:class:`~torch.Tensor` | :class:`float` | :class:`int`): The optimization step(s) `k` at which to
                evaluate the regressor. Can be a single scalar or a tensor of shape `(B,)`.
            Z (:class:`~torch.Tensor`): The conditioning/observation variable composed of `T_z` points in a
                `d_z`-dimensional space. Shape: `(B, T_z, d_z)`, where `B` is the batch size.

        Returns:
            :class:`~torch.Tensor`: The output regression variable of shape `(B, T_x, d_v)`.
        """
        pass


class SinusoidalPosEmb(torch.nn.Module):
    """Sinusoidal positional embedding layer.

    This layer encodes a scalar input (e.g., a diffusion/transport timestep) into a high-dimensional
    vector using a combination of sine and cosine functions of varying frequencies. This technique,
    introduced in the "Attention Is All You Need" paper, allows the model to easily attend
    to relative positions and is effective for representing periodic or sequential data.

    The embedding is calculated as follows:
        emb(x, 2i) = sin(x / 10000^(2i/dim))
        emb(x, 2i+1) = cos(x / 10000^(2i/dim))
    where `x` is the input scalar, `dim` is the embedding dimension, and `i` is the channel index.

    The `forward` method implements this by first calculating the frequency term `1 / 10000^(2i/dim)`
    and then multiplying the input `x` by these frequencies. This creates the argument for the
    sine and cosine functions, effectively encoding the position `x` across the embedding dimension.

    Args:
        dim (:class:`int`): The dimension of the embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):  # noqa: D102
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max((half_dim - 1), 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def build_cond_positions(
    pos_encoding: PosEncoding,
    cond_horizon: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build conditioning-token positions and a mask selecting timeline tokens."""
    cond_position_mask = torch.cat([torch.tensor([False]), torch.ones(cond_horizon - 1)])
    if pos_encoding in {"rope", "additive_relative"}:
        cond_positions = torch.cat([torch.zeros(1), torch.arange(-(cond_horizon - 2), 1)])
    else:
        cond_positions = torch.arange(cond_horizon, device=device)

    cond_positions = cond_positions.to(device=device, dtype=torch.long)
    cond_position_mask = cond_position_mask.to(device=device, dtype=torch.bool)
    return cond_positions, cond_position_mask.to(device=device, dtype=torch.bool)


def build_input_positions(input_horizon: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build decoder target positions and the corresponding validity mask."""
    input_positions = torch.arange(input_horizon, device=device)
    input_position_mask = torch.ones(input_horizon, device=device, dtype=torch.bool)
    return input_positions, input_position_mask


def build_causal_attention_masks(in_horizon: int, cond_horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build additive causal masks for decoder self-attention and cross-attention."""
    self_att_mask = (torch.triu(torch.ones(in_horizon, in_horizon)) == 1).transpose(0, 1)
    self_att_mask = (
        self_att_mask.float().masked_fill(self_att_mask == 0, float("-inf")).masked_fill(self_att_mask == 1, float(0.0))
    )

    t, s = torch.meshgrid(torch.arange(in_horizon), torch.arange(cond_horizon), indexing="ij")
    cross_att_mask = t >= (s - 1)
    cross_att_mask = (
        cross_att_mask.float()
        .masked_fill(cross_att_mask == 0, float("-inf"))
        .masked_fill(cross_att_mask == 1, float(0.0))
    )
    return self_att_mask, cross_att_mask


class CondTransformer(GenCondRegressor):
    r"""Encoder/decoder Transformer with configurable positional attention.

    This module is an encoder/decoder Transformer with `num_cond_layers` conditioning encoder layers and
    `num_layers` decoder layers, following the architecture introduced in *Attention Is All You Need* by
    Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin (NeurIPS 2017),
    with two task-specific changes:

    1. the conditioning memory is built from an optimisation/transport-step token together with the conditioning
       sequence, and
    2. the attention blocks support several positional encoding schemes.

    The conditioning stream is assembled as

    .. math::
        [k, \mathbf{z}_{-(T_z - 1)}, \ldots, \mathbf{z}_{0}],

    where :math:`k` is the inference-time optimisation/transport step token and
    :math:`\mathbf{z}_{-(T_z - 1)}, \ldots, \mathbf{z}_{0}` are the conditioning tokens ordered from oldest
    to most recent observation. The decoder predicts the target sequence

    .. math::
        [\mathbf{x}_{0}, \ldots, \mathbf{x}_{T_x - 1}],

    ordered from the first action to the last action in the predicted horizon.

    Supported positional encodings are:

    ``"additive_absolute"``
        Uses :class:`~symm_learning.nn.activation.AdditivePosMultiheadAttention`. A learned
        table maps integer positions to additive updates that are added to the
        query and key streams before standard multi-head attention.

    ``"additive_relative"``
        Uses :class:`~symm_learning.nn.activation.AdditiveRelMultiheadAttention`. A learned
        table maps relative token distances to additive score biases, yielding a
        time-translation-equivariant attention rule.

    ``"rope"``
        Uses :class:`~symm_learning.nn.activation.RoPEMultiheadAttention`. Rotary position
        embeddings are applied per-head to the query and key projections, leaving values
        untouched.

    ``"none"``
        Uses :class:`~torch.nn.MultiheadAttention` with no explicit positional encoding.

    Temporal assumptions:

    * :math:`Z` must already be ordered in time from past to present.
    * :math:`X` must already be ordered from the first predicted action to the last predicted action.
    * For ``"additive_relative"`` and ``"rope"``, the last conditioning token
      :math:`\mathbf{z}_{0}` and the first action token :math:`\mathbf{x}_{0}` are both placed at time
      index :math:`0`, so cross-attention is anchored at the present time.
    * The optimisation-step token :math:`k` is prepended to the conditioning memory, but it is not treated as
      part of the observation timeline.

    The architecture is implemented as follows:

    * The inference-time optimisation step ``k`` is sinusoidally embedded and prepended as the
      first conditioning token.
    * The observed sequence :math:`Z` is linearly embedded and concatenated after the step token.
    * When ``num_cond_layers > 0`` the conditioning tokens pass through a positional-attention
      encoder; otherwise a lightweight MLP refines them.
    * A positional-attention decoder attends from the input trajectory :math:`X_k` to the
      conditioning memory.

    Args:
        in_dim (:class:`int`): Dimensionality of each element in :math:`X`.
        out_dim (:class:`int`): Dimensionality of the regressed vector field.
        cond_dim (:class:`int`): Dimensionality of each conditioning element in :math:`Z`.
        in_horizon (:class:`int`): Maximum length of :math:`X`.
        cond_horizon (:class:`int`): Maximum length of :math:`Z` (excluding the optimisation-step token).
        pos_encoding (:class:`str`): Positional encoding strategy: ``"additive_absolute"``,
            ``"additive_relative"``, ``"rope"``, or ``"none"``.
        num_layers (:class:`int`): Number of Transformer decoder layers.
        num_attention_heads (:class:`int`): Number of attention heads in Multi-Head Attention blocks.
        embedding_dim (:class:`int`): Dimensionality of token embeddings.
        p_drop_emb (:class:`float`): Dropout applied to embeddings.
        p_drop_attn (:class:`float`): Dropout applied inside attention blocks.
        causal_attn (:class:`bool`): Whether to use causal attention in self-attention and cross-attention layers.
        num_cond_layers (:class:`int`): Number of encoder layers dedicated to conditioning tokens.
        norm_first (:class:`bool`): Whether to apply normalization before each residual branch.
        norm_module (:class:`str`): Final and per-layer normalization type: ``"layernorm"`` or ``"rmsnorm"``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        in_horizon: int,
        cond_horizon: int,
        pos_encoding: PosEncoding = "additive_absolute",
        num_layers: int = 6,
        num_attention_heads: int = 6,
        embedding_dim: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        num_cond_layers: int = 0,
        norm_first: bool = True,
        norm_module: NormModule = "rmsnorm",
        **pos_encoding_kwargs,
    ) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim, cond_dim=cond_dim)

        assert cond_horizon > 0, f"{cond_horizon} !> 0"
        assert in_horizon > 0, f"{in_horizon} !> 0"

        self.in_horizon = in_horizon
        self.cond_horizon = cond_horizon + 1  # Inference-time opt step is another token
        self.pos_encoding = pos_encoding

        # Input embedding stem
        self.input_emb = torch.nn.Linear(in_dim, embedding_dim)
        self.drop = torch.nn.Dropout(p_drop_emb)

        # Conditioning variables z and k embedding stem
        self.cond_emb = torch.nn.Linear(cond_dim, embedding_dim)
        self.opt_time_emb = SinusoidalPosEmb(embedding_dim)

        max_pos_len = max(self.in_horizon, self.cond_horizon)
        max_rel_distance = self.in_horizon + self.cond_horizon - 2

        def _build_attn() -> PositionalAttentionBase | torch.nn.MultiheadAttention:
            if pos_encoding == "additive_absolute":
                return AdditivePosMultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    max_len=max_pos_len,
                    dropout=p_drop_attn,
                )
            elif pos_encoding == "additive_relative":
                return AdditiveRelMultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    max_distance=max_rel_distance,
                    dropout=p_drop_attn,
                )
            elif pos_encoding == "rope":
                return RoPEMultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=p_drop_attn,
                    rope_base=pos_encoding_kwargs.get("rope_base", 100),
                )
            elif pos_encoding == "none":
                return torch.nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=p_drop_attn,
                    batch_first=True,
                )
            else:
                raise ValueError(
                    f"Unknown pos_encoding={pos_encoding!r}. Expected "
                    "'additive_absolute', 'additive_relative', 'rope', or 'none'."
                )

        # Conditioning encoder
        self.encoder = None
        if num_cond_layers > 0:
            enc_layer = TransformerEncoderLayer(
                d_model=embedding_dim,
                self_attn=_build_attn(),
                dim_feedforward=4 * embedding_dim,
                dropout=p_drop_attn,
                activation=torch.nn.GELU(),
                norm_first=norm_first,
                norm_module=norm_module,
            )
            self.encoder = TransformerEncoder(encoder_layer=enc_layer, num_layers=num_cond_layers)
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, 4 * embedding_dim),
                torch.nn.Mish(),
                torch.nn.Linear(4 * embedding_dim, embedding_dim),
            )

        # Decoder
        dec_layer = TransformerDecoderLayer(
            d_model=embedding_dim,
            self_attn=_build_attn(),
            multihead_attn=_build_attn(),
            dim_feedforward=4 * embedding_dim,
            dropout=p_drop_attn,
            activation=torch.nn.GELU(),
            norm_first=norm_first,
            norm_module=norm_module,
        )
        self.decoder = TransformerDecoder(decoder_layer=dec_layer, num_layers=num_layers)

        # Self-Attention and Cross-Attention mask.
        if causal_attn:
            self_att_mask, cross_att_mask = build_causal_attention_masks(self.in_horizon, self.cond_horizon)
            self.register_buffer("self_att_mask", self_att_mask)
            self.register_buffer("cross_att_mask", cross_att_mask)
        else:
            self.self_att_mask = None
            self.cross_att_mask = None

        # Decoder head
        if norm_module == "layernorm":
            self.layer_norm = torch.nn.LayerNorm(embedding_dim, eps=1e-5, bias=True)
        elif norm_module == "rmsnorm":
            self.layer_norm = torch.nn.RMSNorm(embedding_dim, eps=1e-5)
        else:
            raise ValueError(f"norm_module must be 'layernorm' or 'rmsnorm', got {norm_module}")
        self.head = torch.nn.Linear(embedding_dim, out_dim)

        # init
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    # -- helpers --

    def _init_weights(self, module):
        ignore_types = (
            torch.nn.Dropout,
            SinusoidalPosEmb,
            TransformerEncoderLayer,
            TransformerDecoderLayer,
            TransformerEncoder,
            TransformerDecoder,
            AdditivePosMultiheadAttention,
            RoPEMultiheadAttention,
            torch.nn.TransformerEncoderLayer,
            torch.nn.TransformerDecoderLayer,
            torch.nn.TransformerEncoder,
            torch.nn.TransformerDecoder,
            torch.nn.ModuleList,
            torch.nn.Mish,
            torch.nn.Sequential,
            torch.nn.GELU,
        )
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (torch.nn.MultiheadAttention, PositionalAttentionBase)):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name, None)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
                elif hasattr(module, name.replace("_weight", "")) and isinstance(
                    getattr(module, name.replace("_weight", "")), torch.nn.Linear
                ):
                    torch.nn.init.normal_(getattr(module, name.replace("_weight", "")).weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name, None)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
                elif (
                    hasattr(module, name.replace("_bias", "").replace("bias_", "") + "_proj")
                    and isinstance(
                        getattr(module, name.replace("_bias", "").replace("bias_", "") + "_proj"), torch.nn.Linear
                    )
                    and getattr(module, name.replace("_bias", "").replace("bias_", "") + "_proj").bias is not None
                ):
                    torch.nn.init.zeros_(getattr(module, name.replace("_bias", "").replace("bias_", "") + "_proj").bias)

            if hasattr(module, "out_proj") and isinstance(module.out_proj, torch.nn.Linear):
                torch.nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
                if module.out_proj.bias is not None:
                    torch.nn.init.zeros_(module.out_proj.bias)
            if hasattr(module, "pos_emb"):
                torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if hasattr(module, "rel_bias"):
                torch.nn.init.normal_(module.rel_bias, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, torch.nn.RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, CondTransformer):
            pass  # No standalone pos_emb parameter to init in this variant
        elif isinstance(module, RotaryEmbedding):
            pass  # Rotary embeddings are deterministic and have no learnable parameters
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Create optimizer groups separating parameters that receive weight decay from those that don't."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules) or pn in {"pos_emb", "rel_bias"}:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95)
    ):
        """Create optimizer groups separating parameters that receive weight decay from those that don't."""
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, X: torch.Tensor, opt_step: torch.Tensor | float | int, Z: torch.Tensor):
        r"""Forward pass of the conditional transformer regressor, approximating V_k = f(X_k, Z, k).

        Args:
            X (:class:`~torch.Tensor`): The input/data sample composed of a trajectory of `T_x` points in a
                `d_x`-dimensional space. Shape: `(B, T_x, d_x)`, where `B` is the batch size.
            opt_step (:class:`~torch.Tensor` | :class:`float` | :class:`int`): The optimisation timestep(s) `k` at which
                to evaluate the regressor. Can be a single scalar or a tensor of shape `(B,)`.
            Z (:class:`~torch.Tensor`): The conditioning/observation variable composed of `T_z` points in a
                `d_z`-dimensional space. Shape: `(B, T_z, d_z)`, where `B` is the batch size.

        Returns:
            :class:`~torch.Tensor`: The output regression variable of shape `(B, T_x, d_v)`.

        """
        # 1. Inference-time optimisation step embedding (k). First conditioning token.
        if isinstance(opt_step, torch.Tensor):
            opt_steps = opt_step.to(device=X.device, dtype=torch.float32)
        else:
            opt_steps = torch.scalar_tensor(opt_step, device=X.device, dtype=torch.float32)
        opt_steps = opt_steps.reshape(-1).expand(X.shape[0])
        opt_time_emb = self.opt_time_emb(opt_steps).unsqueeze(1)  # (B,1,n_emb)

        # 2. Conditioning variable Z embedding/tokenization
        z_cond_emb = self.cond_emb(Z)  # (B,Tz,n_emb)
        cond_embeddings = torch.cat([opt_time_emb, z_cond_emb], dim=1)  # (B,Tz + 1,n_emb)
        cond_horizon = cond_embeddings.shape[1]
        cond_tokens = self.drop(cond_embeddings)

        # Build integer position indices for the conditioning and input sequences.
        # Under RoPE, observation history lives on the past-to-present timeline while the action
        # trajectory starts at the present, so the last observation and first action both sit at time 0.
        cond_positions, cond_position_mask = build_cond_positions(self.pos_encoding, cond_horizon, device=X.device)

        input_horizon = X.shape[1]
        input_positions, input_position_mask = build_input_positions(input_horizon, device=X.device)

        # Transformer encoder of conditioning tokens
        if isinstance(self.encoder, TransformerEncoder):
            cond_tokens = self.encoder(cond_tokens, src_positions=cond_positions, src_position_mask=cond_position_mask)
        else:
            cond_tokens = self.encoder(cond_tokens)  # MLP fallback

        # 3. Input embedding/tokenization
        input_tokens = self.drop(self.input_emb(X))  # (B,Tx,n_emb)

        # 4. Transformer decoder with positional attention
        out_tokens = self.decoder(
            tgt=input_tokens,
            memory=cond_tokens,
            tgt_mask=self.self_att_mask,
            memory_mask=self.cross_att_mask,
            tgt_positions=input_positions,
            tgt_position_mask=input_position_mask,
            memory_positions=cond_positions,
            memory_position_mask=cond_position_mask,
        )  # (B,Tx,n_emb)

        # 5. Regression head projecting to output dimension.
        out_tokens = self.layer_norm(out_tokens)
        out = self.head(out_tokens)  # (B,Tx, out_dim := d_v)
        return out


if __name__ == "__main__":  # noqa: D103
    torch.manual_seed(0)
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running test on device: {device}")
    dtype = torch.float32
    torch.set_float32_matmul_precision("high")

    dx, dz, dv = 30, 10, 30
    Tx, Tz = 15, 5
    batch_size = 512
    num_batches = 30

    for pos_enc in ("additive_absolute", "additive_relative", "rope"):
        print(f"\n{'=' * 60}")
        print(f"Testing pos_encoding={pos_enc!r}")
        print(f"{'=' * 60}")

        def build_model():  # noqa: D103
            model = CondTransformer(
                in_dim=dx,
                out_dim=dv,
                cond_dim=dz,
                in_horizon=Tx,
                cond_horizon=Tz,
                pos_encoding=pos_enc,
                num_layers=3,
                num_attention_heads=6,
                num_cond_layers=0,
            )
            return model.to(device=device, dtype=dtype).train()

        X_batches = [torch.randn(batch_size, Tx, dx, device=device, dtype=dtype) for _ in range(num_batches)]
        Z_batches = [torch.randn(batch_size, Tz, dz, device=device, dtype=dtype) for _ in range(num_batches)]
        opt_steps = [torch.tensor(float(i % Tx), device=device, dtype=dtype) for i in range(num_batches)]

        model = build_model()
        optimizer = model.configure_optimizers()
        for idx, (x, z, step) in tqdm(enumerate(zip(X_batches, Z_batches, opt_steps))):
            optimizer.zero_grad(set_to_none=True)
            out = model(X=x, Z=z, opt_step=step)
            assert out.shape == (batch_size, Tx, dv), f"out shape {out.shape}!= {(batch_size, Tx, dv)}"
            loss = out.mean()
            loss.backward()
            optimizer.step()
        print(f"  pos_encoding={pos_enc!r}: forward+backward OK")
