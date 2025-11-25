"""Top-level package for symm_learning.

This file makes the package a regular (non-namespace) package so editors like
VS Code / Pylance can statically resolve symbols for go-to-definition.
"""

import logging

logger = logging.getLogger(__name__)

_deepcopy_patched = False


def _ensure_group_deepcopy_singleton() -> None:
    """Keep escnn Group/Representation singletons when modules are deep-copied.

    PyTorch clones prototype modules (e.g., in Transformer stacks) via ``copy.deepcopy``.
    escnn’s ``Group``/``Representation`` objects carry large caches (irreps, change-of-basis
    matrices) and are intended to be unique per group. Without overriding ``__deepcopy__``,
    every clone would duplicate those caches, ballooning memory and breaking the implicit
    singleton assumption. This helper installs a no-op ``__deepcopy__`` on both classes,
    ensuring all clones reuse the same underlying instances and shared caches.
    """
    global _deepcopy_patched
    if _deepcopy_patched:
        return

    from escnn.group import Group, GroupElement, HomSpace, IrreducibleRepresentation, Representation

    def _identity_deepcopy(self, memo):
        """No-op deepcopy to keep singleton instances."""
        logger.debug(f"Reusing singleton {self.__class__.__name__} during deepcopy")
        memo[id(self)] = self
        return self

    for _cls in (Group, Representation, IrreducibleRepresentation, GroupElement, HomSpace):
        if not hasattr(_cls, "__deepcopy__"):
            _cls.__deepcopy__ = _identity_deepcopy  # type: ignore[attr-defined]
    _deepcopy_patched = True


# Patch eagerly on package import so any module using escnn reps shares the singleton caches.
_ensure_group_deepcopy_singleton()

from symm_learning import linalg, models, nn, representation_theory, stats, utils  # noqa: F401

__all__ = [
    "linalg",
    "models",
    "nn",
    "representation_theory",
    "stats",
    "utils",
]
