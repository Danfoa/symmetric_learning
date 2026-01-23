# Symmetric Learning

[![PyPI version](https://img.shields.io/pypi/v/symm-learning.svg?logo=pypi)](https://pypi.org/project/symm-learning/)
[![Python Version](https://img.shields.io/badge/python-3.8%20--%203.12-blue?logo=pypy&logoColor=white)](https://github.com/Danfoa/symmetric_learning/actions/workflows/tests.yaml)
[![Docs](https://img.shields.io/github/actions/workflow/status/Danfoa/symmetric_learning/docs.yaml?branch=main&logo=readthedocs&logoColor=white&label=Docs)](https://danfoa.github.io/symmetric_learning/index.html)

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

## Documentation

Updated documentation can be found in the [official documentation](https://danfoa.github.io/symmetric_learning/index.html). The repository is structured as follows:

- [Linear Algebra](https://danfoa.github.io/symmetric_learning/linalg.html)
    Utility functions for doing linear algebra on symmetric vector spaces, including: least squares solutions, group invariant projections, disentangled/isotypic decompositions, and more.
- [Statistics](https://danfoa.github.io/symmetric_learning/stats.html)
    Functions for computing statistics (e.g., mean, variance, covariance) of symmetric random variables.
- [Models](https://danfoa.github.io/symmetric_learning/models.html)
- [Models](https://danfoa.github.io/symmetric_learning/models.html)
    A collection of equivariant neural network architectures, including MLPs, CNNs, Transformers, and Diffusion models.
- [Neural Network Equivariant Modules](https://danfoa.github.io/symmetric_learning/nn.html)
    A collection of neural network modules that are equivariant to group actions, specifically targeted at processing vector-valued data and time series data:
  - [Activations](https://danfoa.github.io/symmetric_learning/nn/activation.html)
  - [Pooling](https://danfoa.github.io/symmetric_learning/nn/pooling.html)
  - [Normalization](https://danfoa.github.io/symmetric_learning/nn/normalization.html)
  - [Distributions](https://danfoa.github.io/symmetric_learning/nn/distributions.html)
  - [Convolution](https://danfoa.github.io/symmetric_learning/nn/convolution.html)
  - [Disentangled](https://danfoa.github.io/symmetric_learning/nn/disentangled.html)
  - [Linear](https://danfoa.github.io/symmetric_learning/nn/linear.html)
  - [Parametrizations](https://danfoa.github.io/symmetric_learning/nn/parametrizations.html)
  - [Running Stats](https://danfoa.github.io/symmetric_learning/nn/running_stats.html)
