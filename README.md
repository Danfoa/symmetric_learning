# Symmetric Learning

[![PyPI version](https://img.shields.io/pypi/v/symm-learning.svg?logo=pypi)](https://pypi.org/project/symm-learning/)
[![Python Version](https://img.shields.io/badge/python-3.8%20--%203.12-blue?logo=pypy&logoColor=white)](https://github.com/Danfoa/symmetric_learning/actions/workflows/tests.yaml)
[![Docs](https://img.shields.io/github/actions/workflow/status/Danfoa/symmetric_learning/docs.yaml?branch=main&logo=readthedocs&logoColor=white&label=Docs)](https://danfoa.github.io/symmetric_learning/index.html)

**Symmetric Learning** is a machine learning library tailored to optimization problems featuring symmetry priors. It provides equivariant neural network modules, models, and utilities for leveraging group symmetries in data.

## Key Features

- **Neural Network Modules (`nn`)**: Equivariant layers including linear, convolutional, normalization, and attention modules that respect group symmetries.
- **Models (`models`)**: Ready-to-use architectures like equivariant MLPs, Transformers, and CNN encoders for time-series and structured data.
- **Linear Algebra (`linalg`)**: Utilities for symmetric vector spaces—least squares, invariant projections, and isotypic decompositions.
- **Statistics (`stats`)**: Functions for computing statistics (mean, variance, covariance) of symmetric random variables.
- **Representation Theory (`representation_theory`)**: Tools for working with group representations, homomorphism bases, and irreducible decompositions.

## Installation

```bash
pip install symm-learning
# or
git clone https://github.com/Danfoa/symmetric_learning
cd symmetric_learning
pip install -e .
```

## Documentation

Full documentation is available at the [official documentation site](https://danfoa.github.io/symmetric_learning/index.html).

### Modules

| Module | Description |
|--------|-------------|
| [nn](https://danfoa.github.io/symmetric_learning/nn.html) | Equivariant neural network layers (linear, conv, normalization, attention) |
| [models](https://danfoa.github.io/symmetric_learning/models.html) | Complete architectures (eMLP, eTransformer, eTimeCNNEncoder) |
| [linalg](https://danfoa.github.io/symmetric_learning/linalg.html) | Linear algebra for symmetric vector spaces |
| [stats](https://danfoa.github.io/symmetric_learning/stats.html) | Statistics for symmetric random variables |
| [representation_theory](https://danfoa.github.io/symmetric_learning/representation_theory.html) | Group representation utilities |
