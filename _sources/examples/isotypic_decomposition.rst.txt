.. _isotypic-decomposition-example:

Isotypic Decomposition
======================

What It Is
----------

In this library, a **symmetric vector space** is a pair :math:`(\mathcal{X}, \rho_{\mathcal{X}})`, where :math:`\rho_{\mathcal{X}}:\mathbb{G}\to \mathbb{GL}(\mathcal{X})` is a linear group representation of the vector space symmetry group :math:`\mathbb{G}`, and :math:`\mathbb{GL}(\mathcal{X})` is the group of invertible linear transformations on :math:`\mathcal{X}`. 

To perform most relevant statistical and linear algebraic operations on vector spaces efficiently, we leverage the **isotypic decomposition** of :math:`\mathcal{X}`. Intuitively, this decomposition is an ordered breakdown of the vector space into a direct sum of (potentially many copies of) its irreducible building-block subspaces. Similarly to how ordinary vector spaces can be decomposed into a set of one-dimensional (irreducible) subspaces, symmetric vector spaces can also be decomposed; however, their irreducible subspaces are not always one-dimensional, and they come in different types (associated to different irreducible representations of the group).

The isotypic decomposition can be applied to any **compact symmetry group** (e.g., finite groups, compact Lie groups, and their products), and is directly linked to the decomposition of the vector space representation into an ordered direct sum of irreducible representations (irreps) :math:`\{ \hat{\rho}_k: \mathbb{G} \to \mathrm{GL}(\mathcal{H}_k) \}_{k\in[1,n_{\text{iso}}]}`. These are the irreducible linear group representations describing the symmetries of the different types of irreducible constituents of the symmetric vector space, namely :math:`\{\mathcal{H}_k\}_{k\in[1,n_{\text{iso}}]}`. In particular, this means that any :math:`\mathbb{G}`-symmetric vector space can be decomposed into copies of these irreducible building-block vector spaces. Here, :math:`\hat{\rho}_k` denotes the :math:`k`-th irrep type, and :math:`n_{\text{iso}}` denotes the number of (real) irreducible representations of the group. Having this in mind, the isotypic decomposition can be written as:

.. math::
   \begin{aligned}
   \rho_{\mathcal{X}}(g)
   &= \mathbf{Q}\left(
   \bigoplus_{k\in[1,n_{\text{iso}}]}
   \bigoplus_{i\in[1,n_k]}
   \hat{\rho}_k(g)
   \right)\mathbf{Q}^T
   \quad\iff \quad
   &
   \mathcal{X}
   &= \bigoplus_{k\in[1,n_{\text{iso}}]}^{\perp} \mathcal{X}^{(k)} 
   = 
    \bigoplus_{k\in[1,n_{\text{iso}}]}^{\perp} 
    \bigoplus_{i\in[1,n_k]} \hat{\mathcal{X}}_i^{(k)}.
   \end{aligned}

The decomposition of the representation is directly linked to the orthogonal decomposition of the vector space into **isotypic subspaces** :math:`\mathcal{X}^{(k)}`. Each isotypic subspace is composed of :math:`n_k` subspaces :math:`\hat{\mathcal{X}}_i^{(k)}` that are all isomorphic to the irreducible representation space of the irrep of type :math:`k`, i.e., :math:`\hat{\mathcal{X}}_i^{(k)} \cong \mathcal{H}_{k} \quad \forall i \in [1,n_k]`. `

The orthogonality between isotypic subspaces follows from `Schur's orthogonality relations <https://en.wikipedia.org/wiki/Schur_orthogonality_relations>`_ and is a key property we exploit throughout the library. In practice, to reach this decomposition we use the change-of-basis :math:`\mathbf{Q}`. Applying this change of basis to the data allows us to work in an isotypic-irrep-spectral basis, enabling more efficient and more interpretable operations per irrep type / isotypic subspace.

Notation Convention
-------------------

- :math:`\mathbb{G}` denotes the symmetry group.
- :math:`(\mathcal{X}, \rho_{\mathcal{X}})` denotes a :math:`\mathbb{G}`-symmetric vector space, where :math:`\rho_{\mathcal{X}}:\mathbb{G}\to \mathrm{GL}(\mathcal{X})` is a (possibly reducible) linear representation.
- :math:`k` indexes **irrep types** / **isotypic components**, i.e., :math:`k\in[1,n_{\text{iso}}]`.
- :math:`i` indexes the **multiplicity copies** within type :math:`k`, i.e., :math:`i\in[1,n_k]`.
- :math:`\hat{\rho}_k:\mathbb{G}\to \mathrm{GL}(\mathcal{H}_k)` denotes the irrep of type :math:`k` with the associated irreducible vector space :math:`\mathcal{H}_k`.
- :math:`\mathcal{X}^{(k)}` denotes the isotypic subspace of type :math:`k`, composed of (potentially many) copies of spaces isomorphic to :math:`\mathcal{H}_k`.
- :math:`\hat{\mathcal{X}}_i^{(k)}` denotes the :math:`i`-th irreducible copy of type :math:`k`, with :math:`\hat{\mathcal{X}}_i^{(k)} \cong \mathcal{H}_k`.
- By convention, when present, the first isotypic block corresponds to the trivial (invariant) representation: :math:`\mathcal{X}^{\text{inv}}=\mathcal{X}^{(1)}` (equivalently, :math:`\hat{\rho}_1 = \hat{\rho}_{\text{tr}}` is the trivial irrep).

Change Of Basis
---------------

Taking data representations to the isotypic-irrep-spectral basis is done via the orthogonal change-of-basis operator :math:`\mathbf{Q}`:

.. math::
   \tilde{\mathbf{x}} = \mathbf{Q}^T\mathbf{x},
   \qquad
   \mathbf{x} = \mathbf{Q}\tilde{\mathbf{x}},
   \qquad
   \tilde{\rho}_{\mathcal{X}}(g)
   = \mathbf{Q}^T\rho_{\mathcal{X}}(g)\mathbf{Q}
   = \bigoplus_{k\in[1,n_{\text{iso}}]}\bigoplus_{i\in[1,n_k]}\hat{\rho}_k(g).

In practice, given any linear group representation, you can use :func:`symm_learning.representation_theory.isotypic_decomp_rep`, to obtain an equivalent representation exposing the change-of-basis operator :math:`\mathbf{Q}`. 

Practical Example (Icosahedral Group)
-------------------------------------

.. code-block:: python

   import torch
   from escnn.group import Icosahedral
   from symm_learning.representation_theory import direct_sum, isotypic_decomp_rep

   # 1) Build a decomposable representation of the Icosahedral group
   G = Icosahedral()
   rep_x = direct_sum([G.regular_representation] * 2)

   # 2) Compute equivalent representation in canonical isotypic ordering
   rep_x_iso = isotypic_decomp_rep(rep_x)

   # 3) Access change-of-basis operators
   Q = torch.tensor(rep_x_iso.change_of_basis, dtype=torch.float32)
   Q_inv = torch.tensor(rep_x_iso.change_of_basis_inv, dtype=torch.float32)

   # 4) Transform data to spectral/isotypic coordinates and back
   x = torch.randn(8, rep_x.size)              # batch of vectors in original basis
   x_iso = torch.einsum("ij,...j->...i", Q_inv, x)
   x_rec = torch.einsum("ij,...j->...i", Q, x_iso)
   assert torch.allclose(x, x_rec, atol=1e-5)

   # 5) Inspect contiguous slices for each isotypic subspace
   # keys are irrep identifiers; values are slices in the isotypic basis coordinates
   print(rep_x_iso.attributes["isotypic_subspace_dims"])

Use these slices to apply block-wise operations per irrep type
(e.g., constrained statistics, equivariant least squares, and equivariant
parametrizations).
