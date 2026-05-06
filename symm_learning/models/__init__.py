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

from importlib import import_module

__all__ = [
    "eMLP",
    "iMLP",
    "MLP",
    "eTimeCNNEncoder",
    "TimeCNNEncoder",
    "eCondTransformerRegressor",
    "CondTransformerRegressor",
    "GenCondRegressor",
    "CondTransformer",
]  # noqa: F822


_MODULE_MAP = {
    "CondTransformer": "symm_learning.models.control.cond_transformer",
    "CondTransformerRegressor": "symm_learning.models.diffusion.cond_transformer_regressor",
    "eCondTransformerRegressor": "symm_learning.models.diffusion.econd_transformer_regressor",
    "GenCondRegressor": "symm_learning.models.diffusion.cond_transformer_regressor",
    "MLP": "symm_learning.models.emlp",
    "eMLP": "symm_learning.models.emlp",
    "iMLP": "symm_learning.models.emlp",
    "TimeCNNEncoder": "symm_learning.models.time_cnn.cnn_encoder",
    "eTimeCNNEncoder": "symm_learning.models.time_cnn.ecnn_encoder",
}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_MAP[name])
    return getattr(module, name)
