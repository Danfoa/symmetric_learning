# Math Notation Standard

Use this notation consistently across docstrings and docs in this repository.

## Core Symbols

- Group: :math:`\mathbb{G}`.
- Vector spaces: :math:`\mathcal{X}, \mathcal{Y}, \ldots`.
- Vectors (including batched vectors): lowercase bold, e.g. :math:`\mathbf{x}`.
- Matrices/tensors: uppercase bold, e.g. :math:`\mathbf{X}, \mathbf{A}, \mathbf{Q}`.

## Symmetric Vector Spaces

- A symmetric vector space is a pair :math:`(\mathcal{X}, \rho_{\mathcal{X}})` with
  :math:`\rho_{\mathcal{X}}:\mathbb{G}\to \mathrm{GL}(\mathcal{X})`.
- When using coordinates, :math:`\mathbf{x}` and :math:`\tilde{\mathbf{x}}` denote
  vectors in the original and isotypic bases, respectively.
- The space of G-homomorphims between two symmetric vector spaces is denoted as :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`.

## Representations

- Representation of a group on vector space :math:`\mathcal{X}`: :math:`\rho_{\mathcal{X}}:\mathbb{G}\to \mathbb{GL}(\mathcal{X})`.
- Irreducible representations: :math:`\hat{\rho}_k`.
- Decomposable representations on named spaces: :math:`\rho_{\mathcal{X}}, \rho_{\mathcal{Y}}, \ldots`.

## Isotypic Decomposition

Denote the isotypic decomposition of a representation as follows:

.. math::
    \rho_{\mathcal{X}}(g) = \mathbf{Q}\left(
    \bigoplus_{k\in[1,n_{\text{iso}}]}
    \bigoplus_{i\in[1,m_k]}
    \hat{\rho}_k(g)
    \right)\mathbf{Q}^T

With :math:`\mathbf{Q}` the change-of-basis from isotypic to original coordinates.
Equivalent vector-space decomposition:

.. math::
    \mathcal{X} = \bigoplus_{k\in[1,n_{\text{iso}}]} \mathcal{X}^{(k)},
    \qquad
    \mathcal{X}^{(k)} = \bigoplus_{i\in[1,m_k]} \hat{\mathcal{X}}_k^{(i)}.

Change of basis to the isotypic coordinates:

.. math::
    \tilde{\mathbf{x}} = \mathbf{Q}^T\mathbf{x},
    \qquad
    \mathbf{x} = \mathbf{Q}\tilde{\mathbf{x}},
    \qquad
    \tilde{\rho}_{\mathcal{X}}(g)=\mathbf{Q}^T\rho_{\mathcal{X}}(g)\mathbf{Q}.

An equivariant linear map :math:`\mathbf{W} in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})` in the isotypic basis should be denoted as follows:

.. math::
    \mathbf{W} = \mathbf{Q}_{\mathcal{Y}}^\top
    \left(
    \bigoplus_{k\in[1, n_{\text{iso}}]} \mathbf{W}^{(k)}
    \right) \mathbf{Q}_{\mathcal{X}}
    \qquad
    \text{with }
    \begin{aligned}
    \mathbf{W}^{(k)} &=
    \sum_{s=1}^{
    \mathrm{dim}(\mathrm{End}_{\mathbb{G}}(\hat{\rho}_k))
    }\mathbf{\Theta}^{(k)}_s \otimes \mathbf{\Psi}^{(k)}_s,
    \\
    \mathbf{\Theta}^{(k)}_s &\in \mathbb{R}^{m_k^{\mathcal{Y}} \times m_k^{\mathcal{X}}},
    \\
    \mathbf{\Psi}^{(k)}_s &\in \mathbb{R}^{d_k \times d_k},
    \\
    d_k &= \dim(\hat{\rho}_k)
    \end{aligned}
    .

where :math:`n_{\text{iso}}` denotes the number of unique irreducible representations present in :math:`\rho_{\mathcal{Y}}`, and the blocks :math:`\mathbf{W}^{(k)}` are non-zero only if the irrep type :math:`k` is also present in :math:`\rho_{\mathcal{X}}`. These blocks are constrained to commute with the group representations on each isotypic subspace, such that :math:`\mathbf{W}^{(k)} \in \mathrm{Hom}_{\mathbb{G}}(\bigoplus_{i\in[1,m_k^{\mathcal{X}}]} \hat{\rho}_k, \bigoplus_{i\in[1,m_k^{\mathcal{Y}}]} \hat{\rho}_k)`. Consequently, they can be furthere decomposed into a sum of Kronecker products between the free degrees of freedom of the homomorphism space, :math:`\mathbf{\Theta}^{(k)}_s \in \mathbb{R}^{m_k^{\mathcal{Y}} \times m_k^{\mathcal{X}}}`, and the elements of the basis of endomorphisms of the irreducible representation :math:`\hat{\rho}_k`, i.e., :math:`\mathrm{End}_{\mathbb{G}}(\hat{\rho}_k)`, denoted above by the set of matrices :math:`\{\mathbf{\Psi}^{(k)}_s\}_{s \in [1, \mathrm{dim}(\mathrm{End}_{\mathbb{G}}(\hat{\rho}_k))]}`.

Indexing convention:

- :math:`k`: isotypic-subspace / irrep-type index.
- :math:`i`: multiplicity index of irrep type :math:`k` within its isotypic subspace.
- :math:`n_{\text{iso}}`: number of isotypic subspaces (number of distinct irrep types).
- :math:`m_k`: multiplicity of irrep type :math:`k` (number of copies of :math:`\hat{\rho}_k` in the decomposition).
- :math:`d_k := \dim(\hat{\rho}_k)`: dimension of irrep type :math:`k`.

Ordering convention:

- In the canonical isotypic ordering used in this repository, the trivial irrep block
  (invariant subspace) is placed first when present.

Isotypic-subspace notation:

- :math:`\mathcal{X}^{(k)}` denotes the isotypic subspace of type :math:`k`.
- :math:`\mathcal{X}^{\text{inv}}` denotes the invariant subspace (the first block when present).
