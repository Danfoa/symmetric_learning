.. _isotypic-decomposition-example:

Isotypic Decomposition
======================

What It Is
----------

In this library, a **symmetric vector space** is a pair
:math:`(\mathcal{X}, \rho_{\mathcal{X}})` where
:math:`\rho_{\mathcal{X}}:\mathbb{G}\to \mathrm{GL}(\mathcal{X})` is a representation.

The isotypic decomposition can be written equivalently at the representation level
and at the vector-space level:

.. math::
   \begin{aligned}
   \rho_{\mathcal{X}}(g)
   &= \mathbf{Q}\left(
   \bigoplus_{k\in[1,n_{\text{iso}}]}
   \bigoplus_{i\in[1,n_k]}
   \hat{\rho}_k(g)
   \right)\mathbf{Q}^T
   \qquad
   &
   \mathcal{X}
   &= \bigoplus_{k\in[1,n_{\text{iso}}]} \mathcal{X}^{(k)},
   \\
   &&
   \mathcal{X}^{(k)}
   &= \bigoplus_{i\in[1,n_k]} \hat{\mathcal{X}}_k^{(i)}.
   \end{aligned}

These two views are equivalent: block-diagonalizing :math:`\rho_{\mathcal{X}}`
with :math:`\mathbf{Q}` is exactly the same operation as decomposing
:math:`\mathcal{X}` into isotypic subspaces and irreducible copies.

Notation Convention
-------------------

- :math:`\mathbb{G}` denotes the group.
- :math:`k` indexes isotypic subspaces / irrep types.
- :math:`i` indexes multiplicity within type :math:`k`.
- :math:`\mathcal{X}^{(k)}` denotes the isotypic subspace of type :math:`k`.
- :math:`\hat{\mathcal{X}}_k^{(i)}` denotes the :math:`i`-th copy of irrep type :math:`k`.
- :math:`\hat{\rho}_k` denotes an irreducible representation.
- :math:`\rho_{\mathcal{X}}` denotes a (possibly decomposable) representation on :math:`\mathcal{X}`.
- The first block is the invariant subspace when present:
  :math:`\mathcal{X}^{\text{inv}}=\mathcal{X}^{(1)}`.

Change Of Basis
---------------

The isotypic/spectral coordinates are defined by:

.. math::
   \tilde{\mathbf{x}} = \mathbf{Q}^T\mathbf{x},
   \qquad
   \mathbf{x} = \mathbf{Q}\tilde{\mathbf{x}},
   \qquad
   \tilde{\rho}_{\mathcal{X}}(g)
   = \mathbf{Q}^T\rho_{\mathcal{X}}(g)\mathbf{Q}
   = \bigoplus_{k\in[1,n_{\text{iso}}]}\bigoplus_{i\in[1,n_k]}\hat{\rho}_k(g).

In practice, use :func:`symm_learning.representation_theory.isotypic_decomp_rep`,
which returns an equivalent representation object carrying this change of basis.

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
