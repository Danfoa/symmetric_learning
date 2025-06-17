# Symmetric Learning 

[![PyPI version](https://img.shields.io/pypi/v/symm-learning.svg)](https://pypi.org/project/morpho-symm/) [![Python Version](https://img.shields.io/badge/python-3.8%20--%203.12-blue)](https://github.com/Danfoa/MorphoSymm/actions/workflows/tests.yaml)

Lightweight python package for doing geometric deep learning using ESCNN. This package simply holds:
 - Generic equivariant torch models and modules that are not present in ESCNN.
 - Linear algebra utilities when working with symmetric vector spaces.
 - Statistics utilities for symmetric random variables.
 
## Installation

```bash
pip install symm-learning
# or
git clone https://github.com/Danfoa/symmetric_learning
cd symmetric_learning
pip install -e .
```

## Structure:
### [Linear Algebra](/symm_learning/linalg.py)
- [lstsq](/symm_learning/linalg.py): Symmetry-aware computation of the least-squares solution to a linear system of equations with symmetric input-output data.
- [invariant_orthogonal_projector](/symm_learning/linalg.py): Computes the orthogonal projection to the invariant subspace of a symmetric vector space.


### [Statistics](/symm_learning/stats.py)

- [var_mean](/symm_learning/stats.py): Symmetry-aware computation of the variance and mean of a symmetric random variable.
- [cov](/symm_learning/stats.py): Symmetry-aware computation of the covariance / cross-covariance of two symmetric random variables.

### [Models](/symm_learning/models/)

- [iMLP](/symm_learning/models/imlp.py): Invariant MLP for learning invariant functions.
- [eMLP](/symm_learning/models/emlp.py): Equivariant MLP for learning equivariant functions.

### [Torch Modules](/symm_learning/nn/)

- [Change2DisentangledBasis](/symm_learning/nn/disentangled.py): Module for changing the basis of a tensor to a disentangled / isotypic basis.
- [IrrepSubspaceNormPooling](/symm_learning/nn/irrep_pooling.py): Module for extracting invariant features from a geometric tensor, giving one feature per irreducible subspace/representation.
