"""Symmetric Learning - Neural Network Architectures.

Ready-to-use equivariant and standard neural network architectures for structured data.

Architectures
-------------
eMLP, iMLP, MLP
    Equivariant and invariant multi-layer perceptrons for vector-valued data.
eTimeCNNEncoder, TimeCNNEncoder
    1D CNN encoders for time-series data with optional equivariance constraints.
eTransformerEncoderLayer, eTransformerDecoderLayer
    Equivariant Transformer layers preserving group symmetries in attention.
eCondTransformerRegressor, CondTransformerRegressor
    Conditional Transformer regressors for sequence-to-sequence tasks (e.g., diffusion).

Examples:
--------
>>> from symm_learning.models import eMLP
>>> model = eMLP(in_rep, out_rep, hidden_reps=[hidden_rep] * 3)
"""

from .diffusion.cond_transformer_regressor import CondTransformerRegressor, GenCondRegressor
from .diffusion.econd_transformer_regressor import eCondTransformerRegressor
from .emlp import MLP, eMLP, iMLP
from .time_cnn.cnn_encoder import TimeCNNEncoder
from .time_cnn.ecnn_encoder import eTimeCNNEncoder
from .transformer.etransformer import eTransformerDecoderLayer, eTransformerEncoderLayer

__all__ = [
    "eMLP",
    "iMLP",
    "MLP",
    "eTimeCNNEncoder",
    "TimeCNNEncoder",
    "eTransformerEncoderLayer",
    "eTransformerDecoderLayer",
    "eCondTransformerRegressor",
    "CondTransformerRegressor",
    "GenCondRegressor",
]
