from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Optional, Tuple, Union

import torch

from symm_learning.models.difussion.cond_unet1d import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class GenCondRegressor(torch.nn.Module, ABC):
    r"""Generative Conditional Regressor module.

    This is an abstract module inteded to be used as the backbone of a conditional flow-matching/diffusion process which
    enables sampling from the conditional probability distribution:

    .. math::
        \\mathbb{P}(X | Z)

    Where :math:`X = [x_0,...,x_{T_x}] \\in (\R^{d_x})^{T_x}` is the input/data sample composed of a
    trajectory of :math:`T_x` points in a `d_x`-dimensional space, and
    :math:`Z = [z_0,...,z_{T_z}] \\in (\R^{d_z})^{T_z}` is the conditioning/observation variable composed
    of `T_z` points in a `d_z`-dimensional space.

    The module parameterizes a conditional vector-valued regression function:

    .. math::
        V_k = f\_\\theta(X_k, Z, k)

    Where :math:`k` denotes the inference-time optimization timestep (i.e., the step of the flow-matching/diffusion)
    process, :math:`X_k` is the noisy version of the data sample at step `k`, and :math:`V_k \in (\R^{d_v})^{T_x}` is
    the target regression vector-valued variable. For diffusion models :math:`V_k` typically corresponds to the score
    functional of :math:`\\mathbb{P}_k( X | Z )`, while for flow-matching models it typically corresponds to the
    flow-matching velocity vector field.
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
            X (torch.Tensor): The input/data sample composed of a trajectory of `T_x` points in a `d_x`-dimensional
                space. Shape: `(B, T_x, d_x)`, where `B` is the batch size.
            opt_step (Union[torch.Tensor, float, int]): The optimization step(s) `k` at which to evaluate the
                regressor. Can be a single scalar or a tensor of shape `(B,)`.
            Z (torch.Tensor): The conditioning/observation variable composed of `T_z` points in a `d_z`-dimensional
            space. Shape: `(B, T_z, d_z)`, where `B` is the batch size.


        Returns:
            torch.Tensor: The output regression variable of shape `(B, T_x, d_v)`.
        """
        pass


class CondTransformerRegressor(GenCondRegressor):
    r"""Transformer-based generative conditional regressor.

    The module parameterizes :math:`f_\theta(X_k, Z, k)` with a stack of Transformer blocks. The input trajectory
    :math:`X_k` is first projected into an embedding space and interpreted as the target (`tgt`) sequence of a standard
    :class:`torch.nn.TransformerDecoder`. Conditioning information is packed into the decoder `memory` stream:

    * The inference-time step `k` is mapped with a sinusoidal embedding and inserted as the first conditioning token.
    * The observed sequence :math:`Z` is linearly embedded, receives learned positional encodings, and is appended after
      the step token.
    * When ``n_cond_layers > 0`` the conditioning tokens are processed by a Transformer encoder so that the decoder
      attends to context-aware features; otherwise a lightweight MLP refines the embeddings.

    During decoding, self-attention layers refine :math:`X_k` internally while cross-attention layers pull information
    from the conditioning memory, enabling the model to fuse optimisation step, observations, and trajectory features at
    every layer.

    Args:
        in_dim (int): Dimensionality of each element in :math:`X`.
        out_dim (int): Dimensionality of the regressed vector field.
        cond_dim (int): Dimensionality of each conditioning element in :math:`Z`.
        in_horizon (int): Maximum length of :math:`X`.
        cond_horizon (int): Maximum length of :math:`Z` (excluding the optimization-step token).
        num_layers (int): Number of Transformer decoder layers.
        num_attention_heads (int): Number of attention heads in Multi-Head Attention blocks.
        embedding_dim (int): Dimensionality of token embeddings.
        p_drop_emb (float): Dropout applied to embeddings.
        p_drop_attn (float): Dropout applied inside attention blocks.
        causal_attn (bool): Whether to use causal attention in self-attention and cross-attention layers.
        num_cond_layers (int): Number of encoder layers dedicated to conditioning tokens.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        in_horizon: int,
        cond_horizon: int,
        num_layers: int = 6,
        num_attention_heads: int = 6,
        embedding_dim: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        num_cond_layers: int = 0,
    ) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim, cond_dim=cond_dim)

        assert cond_horizon > 0, f"{cond_horizon} !> 0"
        assert in_horizon > 0, f"{in_horizon} !> 0"

        self.in_horizon = in_horizon
        self.cond_horizon = cond_horizon + 1  # Inference-time opt step is another token

        # Input embedding stem
        self.input_emb = torch.nn.Linear(in_dim, embedding_dim)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, self.in_horizon, embedding_dim))
        self.drop = torch.nn.Dropout(p_drop_emb)

        # Conditioning variables z and k embedding stem
        self.cond_emb = torch.nn.Linear(cond_dim, embedding_dim)
        self.cond_pos_emb = torch.nn.Parameter(torch.zeros(1, self.cond_horizon, embedding_dim))
        self.opt_time_emb = SinusoidalPosEmb(embedding_dim)  # Inference-time optimization step embedding

        self.encoder = None
        self.decoder = None

        if num_cond_layers > 0:
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                dim_feedforward=4 * embedding_dim,
                nhead=num_attention_heads,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_cond_layers)
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, 4 * embedding_dim),
                torch.nn.Mish(),
                torch.nn.Linear(4 * embedding_dim, embedding_dim),
            )

        # decoder
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # important for stability
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        # Self-Attention and Cross-Attention mask.
        # Cross-attention is used to compute updates to the action vector based on the conditioing tokens
        # composed of a inference-time optimization step token and the observation conditioning tokens.
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = (torch.triu(torch.ones(self.in_horizon, self.in_horizon)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("self_att_mask", mask)

            t, s = torch.meshgrid(torch.arange(self.in_horizon), torch.arange(self.cond_horizon), indexing="ij")
            mask = t >= (s - 1)  # add one dimension since opt-time  is the first token in cond
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("cross_att_mask", mask)
        else:
            self.self_att_mask = None
            self.cross_att_mask = None

        # Decoder head
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.head = torch.nn.Linear(embedding_dim, out_dim)

        # init
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            torch.nn.Dropout,
            SinusoidalPosEmb,
            torch.nn.TransformerEncoderLayer,
            torch.nn.TransformerDecoderLayer,
            torch.nn.TransformerEncoder,
            torch.nn.TransformerDecoder,
            torch.nn.ModuleList,
            torch.nn.Mish,
            torch.nn.Sequential,
        )
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, CondTransformerRegressor):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Creates optimizer groups separating out parameters to apply weight decay to and those that don't.

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        # no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
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
        self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95)
    ):
        """Creates optimizer groups separating out parameters to apply weight decay to and those that don't."""
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, X: torch.Tensor, opt_step: torch.Tensor | float | int, Z: torch.Tensor):
        r"""Forward pass of the conditional transformer regressor, approximating V_k = f(X_k, Z, k).

        Args:
            X (torch.Tensor): The input/data sample composed of a trajectory of `T_x` points in a `d_x`-dimensional
                space. Shape: `(B, T_x, d_x)`, where `B` is the batch size.
            opt_step (Union[torch.Tensor, float, int]): The optimization timestep(s) `k` at which to evaluate the
                regressor. Can be a single scalar or a tensor of shape `(B,)`.
            Z (torch.Tensor): The conditioning/observation variable composed of `T_z` points in a `d_z`-dimensional
            space. Shape: `(B, T_z, d_z)`, where `B` is the batch size.

        Returns:
            torch.Tensor: The output regression variable of shape `(B, T_x, d_v)`.
        """
        # assert X.shape[1] <= self.in_horizon, f"Input horizon {X.shape[1]} larger than {self.in_horizon}"
        # assert Z.shape[1] <= self.cond_horizon - 1, f"Cond horizon {Z.shape[1]} larger than {self.cond_horizon - 1}"

        # 1. Inference-time optimization step embedding (k). First conditioning token.
        if isinstance(opt_step, torch.Tensor):
            opt_steps = opt_step.to(device=X.device, dtype=torch.float32)
        else:
            opt_steps = torch.scalar_tensor(opt_step, device=X.device, dtype=torch.float32)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        opt_steps = opt_steps.reshape(-1).expand(X.shape[0])
        opt_time_emb = self.opt_time_emb(opt_steps).unsqueeze(1)  # (B,1,n_emb)

        # 2. Conditioning variable Z embedding/tokenization
        z_cond_emb = self.cond_emb(Z)  # (B,Tz,n_emb)
        cond_embeddings = torch.cat([opt_time_emb, z_cond_emb], dim=1)  # (B,Tz + 1,n_emb)
        cond_horizon = cond_embeddings.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :cond_horizon, :]  # each position maps to a (learnable) vector
        # Transformer encoder of conditing tokens
        cond_tokens = self.drop(cond_embeddings + cond_pos_emb)
        cond_tokens = self.encoder(cond_tokens)  # (B,T_cond,n_emb)

        # 3. Input embedding/tokenization
        input_tokens = self.input_emb(X)

        # 4. Transformer encoder of input tokens with self-attention and cross-attention to cond tokens
        input_horizon = input_tokens.shape[1]
        pos_emb = self.pos_emb[:, :input_horizon, :]  # each position maps to a (learnable) vector
        input_tokens = self.drop(input_tokens + pos_emb)  # (B,Tx,n_emb)
        out_tokens = self.decoder(
            tgt=input_tokens, memory=cond_tokens, tgt_mask=self.self_att_mask, memory_mask=self.cross_att_mask
        )  # (B,Tx,n_emb)

        # 5. Regression head projecting to output dimension.
        out_tokens = self.layer_norm(out_tokens)
        out = self.head(out_tokens)  # (B,Tx, out_dim := d_v)
        return out


def test():  # noqa: D103
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

    def build_model():
        model = CondTransformerRegressor(
            in_dim=dx,
            out_dim=dv,
            cond_dim=dz,
            in_horizon=Tx,
            cond_horizon=Tz,
            num_layers=3,
            num_attention_heads=6,
            num_cond_layers=0,
        )
        return model.to(device=device, dtype=dtype).train()

    X_batches = [torch.randn(batch_size, Tx, dx, device=device, dtype=dtype) for _ in range(num_batches)]
    Z_batches = [torch.randn(batch_size, Tz, dz, device=device, dtype=dtype) for _ in range(num_batches)]
    opt_steps = [torch.tensor(float(i % Tx), device=device, dtype=dtype) for i in range(num_batches)]

    def benchmark(model, skip_first: bool = False):
        optimizer = model.configure_optimizers()
        measurements: list[float] = []
        for idx, (x, z, step) in tqdm(enumerate(zip(X_batches, Z_batches, opt_steps))):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            optimizer.zero_grad(set_to_none=True)
            start.record()
            out = model(X=x, Z=z, opt_step=step)
            assert out.shape == (batch_size, Tx, dv), f"out shape {out.shape}!= {(batch_size, Tx, dv)}"
            loss = out.mean()
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            end.record()
            torch.cuda.synchronize()
            if skip_first and idx == 0:
                continue
            measurements.append(start.elapsed_time(end) * 1e-3)
        return sum(measurements) / len(measurements)

    eager_model = build_model()
    eager_time = benchmark(eager_model)
    print(f"Eager avg forward+backward step time:    {eager_time:.3f} [s]")

    compiled_model = torch.compile(build_model())
    compiled_time = benchmark(compiled_model, skip_first=True)
    print(f"Compiled avg forward+backward step time: {compiled_time:.3f} [s] (excluding first batch)")


if __name__ == "__main__":
    test()
