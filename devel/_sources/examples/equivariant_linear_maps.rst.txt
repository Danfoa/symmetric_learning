.. _equivariant-linear-maps-example:

Leveraging the structure of Equivariant Linear maps
===================================================

This example tutorial shows how to use `symm_learning` to leverage the structure of equivariant linear maps in linear algebraic and statistic operations. 

Whats the structure of equivariant linear maps?
-----------------------------------------------

Let :math:`(\mathcal{X}, \rho_{\mathcal{X}})` and :math:`(\mathcal{Y}, \rho_{\mathcal{Y}})` be two :math:`\mathbb{G}`-symmetric vector spaces, with 
group representations :math:`\rho_{\mathcal{X}} : \mathbb{G} \to \mathrm{GL}(\mathcal{X})` and :math:`\rho_{\mathcal{Y}} : \mathbb{G} \to \mathrm{GL}(\mathcal{Y})`, and :math:`\mathbb{G}` a compact symmetry group. Due to the isotypic decomposition of symmetric vector spaces, the group representations can be decomposed, via respective change-of-basis matrices :math:`\mathbf{Q}_{\mathcal{X}}` and :math:`\mathbf{Q}_{\mathcal{Y}}`, into a block-diagonal form where each block is composed of the direct sum of :math:`m_k` copies/multiplicities of the same irrep :math:`\hat{\rho}_k`:

.. math::
    \rho_{\mathcal{X}}(g) = \mathbf{Q}_{\mathcal{X}}\left(
    \bigoplus_{k\in[1,n_{\text{iso},x}]}
    \bigoplus_{i\in[1,m_k^{\mathcal{X}}]}
    \hat{\rho}_k(g)
    \right)\mathbf{Q}_{\mathcal{X}}^T,
    \qquad
    \rho_{\mathcal{Y}}(g) = \mathbf{Q}_{\mathcal{Y}}\left(
    \bigoplus_{k\in[1,n_{\text{iso},y}]}
    \bigoplus_{i\in[1,m_k^{\mathcal{Y}}]}
    \hat{\rho}_k(g)
    \right)\mathbf{Q}_{\mathcal{Y}}^T.

In this context, any linear map, :math:`\mathbf{W}: \mathcal{X} \to \mathcal{Y}`, that belongs to the group-homomorphism space between these two symmetric vector spaces, :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`, that is, any linear map satisfying the following equivariance/commutativity condition:

.. math::
    \mathbf{W}\rho_{\mathcal{X}}(g) = \rho_{\mathcal{Y}}(g)\mathbf{W},
    \qquad\forall g\in\mathbb{G}, 
    \quad \iff \quad 
    \mathbf{W} \in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}}).

admits a block-diagonal decomposition analog to the one of the representations, given by:

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

How to leverage linear equivariance in code?
~~~~~~~~~~~~~~~~~~~~~~

The `symm_learning.linalg` module exposes a set of standalone utility functions to leverage these structural constraints directly from the representations :math:`\rho_{\mathcal{X}}` and :math:`\rho_{\mathcal{Y}}`. In practice, the most useful ones are:

1. **Projection to space of equivariant maps**: Having a dense linear map :math:`\mathbf{A} \in \mathbb{R}^{\mathrm{dim}(\mathcal{Y}) \times \mathrm{dim}(\mathcal{X})}`, you can project it onto the homomorphism space, :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`, to get the closest equivariant map in Frobenius norm. To do this, you use `symm_learning.linalg.equiv_orthogonal_projection`:

.. code-block:: python
   
   import torch
   from symm_learning.linalg import equiv_orthogonal_projection
   A = torch.randn(rep_y.size, rep_x.size)  # Example dense linear map.
   A_proj = equiv_orthogonal_projection(A, rep_x, rep_y)
   
   # This map will commute with the group representations:
   G = rep_x.group
   for g in G.elements:
       assert torch.allclose(A_proj @ rep_x(g), rep_y(g) @ A_proj, atol=1e-5)

2. **Extracting degrees-of-freedom of an equivariant map**: In many linear-algebraic and statistical operations, it is useful not to work with the dense :math:`\mathbb{G}`-equivariant matrix :math:`\mathbf{W} \in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})` directly, but rather with its free parameters, i.e., the basis expansion coefficients :math:`\{\mathbf{\Theta}^{(k)}_s\}_{k\in[1,n_{\text{iso}}], s \in [1, \mathrm{dim}(\mathrm{End}_{\mathbb{G}}(\hat{\rho}_k))]}`. For this, the most informative access point is :func:`~symm_learning.linalg.project_in_isobasis`, which exposes the isotypic decomposition used internally by the projection routines:

.. code-block:: python

   import torch
   from symm_learning.linalg import (
       equiv_orthogonal_projection,
       equiv_orthogonal_projection_coefficients,
       project_in_isobasis,
   )

   A = torch.randn(rep_y.size, rep_x.size)
   W = equiv_orthogonal_projection(A, rep_x, rep_y)

   Q_out, Q_in_inv, projection_iso_spaces = project_in_isobasis(W, rep_x, rep_y)
   w_dof = equiv_orthogonal_projection_coefficients(W, rep_x, rep_y)

`project_in_isobasis` returns the following objects, in the notation introduced above:

- ``Q_out`` is :math:`\mathbf{Q}_{\mathcal{Y}}`, the change of basis from isotypic coordinates back to the original coordinates of :math:`\mathcal{Y}`.
- ``Q_in_inv`` is :math:`\mathbf{Q}_{\mathcal{X}}^T`, the change of basis from the original coordinates of :math:`\mathcal{X}` to isotypic coordinates.
- ``projection_iso_spaces`` is a dictionary keyed by the shared irrep type :math:`k`. For each key :math:`k`, the entry ``iso_space = projection_iso_spaces[k]`` stores:

  - ``iso_space["coeff"]``: the coefficients :math:`\{\mathbf{\Theta}^{(k)}_s\}_{s \in [1,S_k]}` arranged as a tensor of shape :math:`(..., m_k^{\mathcal{Y}} m_k^{\mathcal{X}}, S_k)`.
  - ``iso_space["endo_basis_flat"]``: the flattened endomorphism basis :math:`\operatorname{flat}(\mathbf{\Psi}^{(k)}) = [\operatorname{flat}(\mathbf{\Psi}^{(k)}_1), \ldots, \operatorname{flat}(\mathbf{\Psi}^{(k)}_{S_k})]^\top \in \mathbb{R}^{S_k \times d_k^2}`.
  - ``iso_space["m_out"]``: the output multiplicity :math:`m_k^{\mathcal{Y}}`.
  - ``iso_space["m_in"]``: the input multiplicity :math:`m_k^{\mathcal{X}}`.
  - ``iso_space["d_k"]``: the irrep dimension :math:`d_k = \dim(\hat{\rho}_k)`.
  - ``iso_space["out_slice"]``: the slice locating the output coordinates of the block :math:`\mathbf{W}^{(k)}` inside :math:`\mathbf{Q}_{\mathcal{Y}}^T \mathbf{W} \mathbf{Q}_{\mathcal{X}}`.
  - ``iso_space["in_slice"]``: the slice locating the input coordinates of the block :math:`\mathbf{W}^{(k)}` inside :math:`\mathbf{Q}_{\mathcal{Y}}^T \mathbf{W} \mathbf{Q}_{\mathcal{X}}`.

The tensor ``iso_space["coeff"]`` stores the coefficients of the projection of each :math:`d_k \times d_k` sub-block onto the basis :math:`\{\mathbf{\Psi}^{(k)}_s\}`. If you only need the flattened coefficients, then :func:`~symm_learning.linalg.equiv_orthogonal_projection_coefficients` returns them directly as ``w_dof``. This vector is the compact representation of the same degrees of freedom encoded blockwise in ``projection_iso_spaces``.
