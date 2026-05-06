"""Symmetric Learning - Neural Network Modules.

Equivariant neural network layers that respect group symmetries. These modules are
designed for processing vector-valued data and time series where symmetry constraints
should be preserved throughout the network.

Submodules
----------
activation
    Equivariant attention mechanisms (eMultiheadAttention).
conv
    Equivariant 1D convolutions (eConv1d, eConvTranspose1d).
disentangled
    Change of basis to disentangled/isotypic representations.
distributions
    Equivariant multivariate distributions.
linear
    Equivariant linear and affine layers (eLinear, eAffine).
normalization
    Equivariant normalization layers (eBatchNorm1d, eLayerNorm, eRMSNorm).
pooling
    Invariant pooling based on irreducible subspace norms.
running_stats
    Exponential moving average statistics modules.

Examples:
--------
>>> from symm_learning.nn import eLinear
>>> layer = eLinear(in_rep, out_rep)  # Equivariant linear layer
"""

from importlib import import_module

__all__ = [
    "Change2DisentangledBasis",
    "eMultivariateNormal",
    "IrrepSubspaceNormPooling",
    "AdditivePosMultiheadAttention",
    "AdditiveRelMultiheadAttention",
    "eAdditivePosMultiheadAttention",
    "eAdditiveRelMultiheadAttention",
    "eMultiheadAttention",
    "eBatchNorm1d",
    "eAffine",
    "eConv1d",
    "eConvTranspose1d",
    "eModule",
    "EMAStats",
    "eEMAStats",
    "eLinear",
    "eLayerNorm",
    "eRMSNorm",
    "eTransformerDecoderLayer",
    "eTransformerEncoderLayer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]  # noqa: F822


_MODULE_MAP = {
    "eAdditivePosMultiheadAttention": "symm_learning.nn.activation",
    "eAdditiveRelMultiheadAttention": "symm_learning.nn.activation",
    "eMultiheadAttention": "symm_learning.nn.activation",
    "AdditivePosMultiheadAttention": "symm_learning.nn.activation",
    "AdditiveRelMultiheadAttention": "symm_learning.nn.activation",
    "eConv1d": "symm_learning.nn.conv",
    "eConvTranspose1d": "symm_learning.nn.conv",
    "Change2DisentangledBasis": "symm_learning.nn.disentangled",
    "eMultivariateNormal": "symm_learning.nn.distributions",
    "eAffine": "symm_learning.nn.linear",
    "eLinear": "symm_learning.nn.linear",
    "eModule": "symm_learning.nn.module",
    "eBatchNorm1d": "symm_learning.nn.normalization",
    "eLayerNorm": "symm_learning.nn.normalization",
    "eRMSNorm": "symm_learning.nn.normalization",
    "IrrepSubspaceNormPooling": "symm_learning.nn.pooling",
    "EMAStats": "symm_learning.nn.running_stats",
    "eEMAStats": "symm_learning.nn.running_stats",
    "eTransformerDecoderLayer": "symm_learning.nn.transformer.etransformer",
    "eTransformerEncoderLayer": "symm_learning.nn.transformer.etransformer",
    "TransformerDecoder": "symm_learning.nn.transformer.transformer",
    "TransformerDecoderLayer": "symm_learning.nn.transformer.transformer",
    "TransformerEncoder": "symm_learning.nn.transformer.transformer",
    "TransformerEncoderLayer": "symm_learning.nn.transformer.transformer",
}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_MAP[name])
    return getattr(module, name)
