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

## Structure

_______________

### [Linear Algebra](/symm_learning/linalg.py)

- [lstsq](/symm_learning/linalg.py): Symmetry-aware computation of the least-squares solution to a linear system of equations with symmetric input-output data.
- [invariant_orthogonal_projector](/symm_learning/linalg.py): Computes the orthogonal projection to the invariant subspace of a symmetric vector space.

_______________

### [Statistics](/symm_learning/stats.py)

- [var_mean](/symm_learning/stats.py): Symmetry-aware computation of the variance and mean of a symmetric random variable.
- [cov](/symm_learning/stats.py): Symmetry-aware computation of the covariance / cross-covariance of two symmetric random variables.

_______________

### [Models](/symm_learning/models/)

- [iMLP](/symm_learning/models/imlp.py): Invariant MLP for learning invariant functions.
- [eMLP](/symm_learning/models/emlp.py): Equivariant MLP for learning equivariant functions.

_______________

### [Torch Modules](/symm_learning/nn/)

#### [Change2DisentangledBasis](/symm_learning/nn/disentangled.py)

Module for changing the basis of a tensor to a disentangled / isotypic basis.

#### [IrrepSubspaceNormPooling](/symm_learning/nn/irrep_pooling.py)

Module for extracting invariant features from a geometric tensor, giving one feature per irreducible subspace/representation.%

#### [EquivMultivariateNormal](/symm_learning/nn/equiv_multivariate_normal.py)

Utility layer to parameterize a G-equivariant multivariate Gaussian/Normal distribution:

```math
\begin{aligned}
y &\sim \mathcal{N} \bigl(\mu(x), \Sigma(x)\bigr)& \\
\text{s.t.}
&\rho_Y(g)\mu(x) = \mu \bigl(\rho_X(g)\cdot x\bigr) \\
&\rho_Y(g)\Sigma(x)\rho_Y(g)^{\top} = \Sigma\bigl(\rho_X(g)\cdot x\bigr),
\quad \forall\, g \in G.
\end{aligned}
```

Such that the conditional probability distribution of `y` given `x` is $\mathbb{G}$-invariant to the simultaneous group action on $\mathcal{X}$ and $\mathcal{Y}$:

$$
P(y \mid x) = P(\rho_Y(g) y \mid \rho_X(g) x) \quad \forall g \in \mathbb{G}.
$$

This means that if you want to parameterize a $\mathbb{G}$-equivariant stochastic function $y = f(x)$ using neural networks, you can use any backbone architecture whose output are the input parameters of a `EquivMultivariateNormal` distribution, as shown below:
```python
from escnn.group import CyclicGroup
from escnn.nn import FieldType
from symm_learning.models.emlp import EMLP
from symm_learning.nn.equiv_multivariate_gaussian import EquivMultivariateNormal

G = CyclicGroup(3)
x_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation])
y_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation] * 1)
# Instanciate the output equivariant multivariate normal distribution in order to get the NN output type
e_normal = EquivMultivariateNormal(y_type, diagonal=True)
# Instanciate your NN model to output the parameters of the distribution
nn = EMLP(in_type=x_type, out_type=e_normal.in_type)
# Sample from the distribution
x = x_type(torch.randn(10, x_type.size))
z = nn(x) # (B, dim_y + n_dof_cov)
dist = e_normal.get_distribution(z) # instance of  torch.distributions.MultivariateNormal
y = dist.sample()  # (B, n)
```
Here, $z$ is a `(batch_size, dim_y + n_dof_cov)` input tensor with the first `dim_y` entries defining the mean of the distribution $\mu(x)$ and the next `n_dof_cov` entries define the free degrees of freedom from the symmetry constrained covariance matrix. See below
