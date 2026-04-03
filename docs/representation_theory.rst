representation_theory
=====================

.. module:: symm_learning.representation_theory

Representation-theoretic utilities for symmetric vector spaces, including
:ref:`isotypic decomposition <isotypic-decomposition-example>`, homomorphism-basis parameterizations of
:math:`\mathbb{G}`-equivariant linear maps, and irreducible decomposition helpers.

For the canonical isotypic decomposition and a practical Icosahedral example, see
:ref:`Isotypic Decomposition <isotypic-decomposition-example>`.

For the structure of equivariant linear maps and how
:class:`~symm_learning.representation_theory.GroupHomomorphismBasis` implements Proposition I.13 and Eq. (40) from
``main_vk.pdf``, see
:ref:`Leveraging the structure of Equivariant Linear maps <equivariant-linear-maps-example>`.


.. currentmodule:: symm_learning.representation_theory


.. autosummary::
   :toctree: generated/
   :recursive:

   GroupHomomorphismBasis
   isotypic_decomp_rep
   direct_sum

   permutation_matrix
   irreps_stats
   escnn_representation_form_mapping
   is_complex_irreducible
   decompose_representation
