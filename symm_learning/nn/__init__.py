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

from .activation import eMultiheadAttention
from .conv import eConv1d, eConvTranspose1d
from .disentangled import Change2DisentangledBasis
from .distributions import eMultivariateNormal
from .linear import eAffine, eLinear
from .normalization import eBatchNorm1d, eLayerNorm, eRMSNorm
from .pooling import IrrepSubspaceNormPooling
from .running_stats import EMAStats, eEMAStats

__all__ = [
    "Change2DisentangledBasis",
    "eMultivariateNormal",
    "IrrepSubspaceNormPooling",
    "eMultiheadAttention",
    "eBatchNorm1d",
    "eAffine",
    "eConv1d",
    "eConvTranspose1d",
    "EMAStats",
    "eEMAStats",
    "eLinear",
    "eLayerNorm",
    "eRMSNorm",
]
