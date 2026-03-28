.. Symmetric Learning documentation master file
.. raw:: html

   <div style="text-align:center; margin: 0.5rem 0 1rem 0;">
     <img src="_static/logo_v1_with_text.svg?v=5" alt="Symmetric Learning logo" class="sl-home-logo sl-home-logo-light">
     <img src="_static/logo_v1_without_text_dark_background.svg?v=5" alt="Symmetric Learning logo" class="sl-home-logo sl-home-logo-dark">
   </div>
   <p align="center">
     <a href="https://pypi.org/project/symm-learning/">
       <img src="https://img.shields.io/pypi/v/symm-learning.svg?logo=pypi" alt="PyPI version">
     </a>
     <a href="https://github.com/Danfoa/symmetric_learning">
       <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub repository">
     </a>
     <a href="https://github.com/Danfoa/symmetric_learning/actions/workflows/tests.yaml">
       <img src="https://img.shields.io/badge/python-3.8%20--%203.12-blue?logo=pypy&logoColor=white" alt="Python Version">
     </a>
     <a href="https://danfoa.github.io/symmetric_learning/">
       <img src="https://img.shields.io/github/actions/workflow/status/Danfoa/symmetric_learning/docs.yaml?branch=main&logo=readthedocs&logoColor=white&label=Docs" alt="Docs">
     </a>
   </p>

**Symmetric Learning** is a torch-based machine learning library tailored to optimization problems featuring symmetry priors. It provides equivariant neural network modules, models, and utilities for leveraging group symmetries in data.


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
