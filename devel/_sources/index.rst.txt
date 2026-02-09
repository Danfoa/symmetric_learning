.. Symmetric Learning documentation master file

Symmetric Learning
==================

**Symmetric Learning** is a machine learning library tailored to optimization problems featuring symmetry priors. It provides equivariant neural network modules, models, and utilities for leveraging group symmetries in data.

Installation
------------

.. code-block:: bash

    pip install symm-learning

Key Features
------------

- **Neural Network Modules** (:mod:`~symm_learning.nn`): Equivariant layers including linear, convolutional, normalization, and attention modules that respect group symmetries.
- **Models** (:mod:`~symm_learning.models`): Ready-to-use architectures like equivariant MLPs, Transformers, and CNN encoders for time-series and structured data.
- **Linear Algebra** (:mod:`~symm_learning.linalg`): Utilities for symmetric vector spaces—least squares, invariant projections, and :ref:`isotypic decomposition <isotypic-decomposition-example>`.
- **Statistics** (:mod:`~symm_learning.stats`): Functions for computing statistics (mean, variance, covariance) of symmetric random variables.
- **Representation Theory** (:mod:`~symm_learning.representation_theory`): Tools for working with group representations, homomorphism bases, and irreducible decompositions.

Citation
--------

If you use ``symm-learning`` in research, please cite:

.. code-block:: bibtex

    @software{ordonez_apraez_symmetric_learning,
      author  = {Ordonez Apraez, Daniel Felipe},
      title   = {Symmetric Learning},
      year    = {2026},
      url     = {https://github.com/Danfoa/symmetric_learning}
    }

License
-------

This project is released under the MIT License. See ``LICENSE`` in the repository root.

Resources
---------

* :doc:`Reference API <reference>`
* :doc:`Examples <examples>`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Navigation

   Home <self>
   examples
   reference
