<!-- # Symmetric Learning -->

<p align="center">
  <img src="docs/media/logo_v1_without_text_dark_background.svg" alt="Symmetric Learning logo" width="520">
</p>

<p align="center">
  <a href="https://pypi.org/project/symm-learning/">
    <img src="https://img.shields.io/pypi/v/symm-learning.svg?logo=pypi" alt="PyPI version">
  </a>
  <a href="https://github.com/Danfoa/symmetric_learning/actions/workflows/tests.yaml">
    <img src="https://img.shields.io/badge/python-3.8%20--%203.12-blue?logo=pypy&logoColor=white" alt="Python Version">
  </a>
  <a href="https://danfoa.github.io/symmetric_learning/">
    <img src="https://img.shields.io/github/actions/workflow/status/Danfoa/symmetric_learning/docs.yaml?branch=main&logo=readthedocs&logoColor=white&label=Docs" alt="Docs">
  </a>
</p>

_________________________

**Symmetric Learning** is a torch-based machine learning library tailored to optimization problems featuring symmetry priors. It provides equivariant neural network modules, models, and utilities for leveraging group symmetries in data.

## Package Structure

- [Neural Networks (nn)](https://danfoa.github.io/symmetric_learning/nn.html): Equivariant neural network layers including linear, convolutional, normalization, and attention modules.
- [Models (models)](https://danfoa.github.io/symmetric_learning/models.html): Ready-to-use equivarint architectures such as equivariant MLPs, Transformers, and CNN encoders.
- [Linear Algebra (linalg)](https://danfoa.github.io/symmetric_learning/linalg.html): Linear algebra utilities for symmetric vector spaces, including equivariant least squares solutions, projections to invariant subspaces, and more.
- [Symmetry-aware Statistics (stats)](https://danfoa.github.io/symmetric_learning/stats.html): Mean, variance, and covariance for symmetric random variables.
- [Representation Theory (representation_theory)](https://danfoa.github.io/symmetric_learning/representation_theory.html): Representation theory utils, enabling de isotypic decomposition of group representations, intuitive management of the degrees of freedom of equivariant linear maps, orthogonal projections to the space of equivariant linear maps, and more.

## Installation

```bash
pip install symm-learning
# or
git clone https://github.com/Danfoa/symmetric_learning
cd symmetric_learning
pip install -e .
```

## Documentation

Documentation is published per branch:

- `main` (stable): [https://danfoa.github.io/symmetric_learning/](https://danfoa.github.io/symmetric_learning/)
- `devel`: [https://danfoa.github.io/symmetric_learning/devel/](https://danfoa.github.io/symmetric_learning/devel/)

## Citation

If you use `symm-learning` in research, please cite:

```bibtex
@software{ordonez_apraez_symmetric_learning,
  author  = {Ordonez Apraez, Daniel Felipe},
  title   = {Symmetric Learning},
  year    = {2026},
  url     = {https://github.com/Danfoa/symmetric_learning}
}
```

## License

This project is released under the MIT License. See `LICENSE`.
