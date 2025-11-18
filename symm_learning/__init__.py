"""Top-level package for symm_learning.

This file makes the package a regular (non-namespace) package so editors like
VS Code / Pylance can statically resolve symbols for go-to-definition.
"""

from symm_learning import linalg, models, nn, representation_theory, stats, utils  # noqa: F401

__all__ = ["linalg", "models", "nn", "representation_theory", "stats", "utils"]
