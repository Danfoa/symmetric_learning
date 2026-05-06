"""Symmetric Learning - Machine Learning with Symmetry Priors.

**Symmetric Learning** is a machine learning library tailored to optimization problems
featuring symmetry priors. It provides equivariant neural network modules, models, and
utilities for leveraging group symmetries in data.

Modules
-------
nn
    Equivariant neural network layers (linear, conv, normalization, attention).
models
    Complete architectures (eMLP, eTransformer, eTimeCNNEncoder).
linalg
    Linear algebra utilities for symmetric vector spaces.
stats
    Statistics for symmetric random variables.
representation_theory
    Tools for group representations and irreducible decompositions.
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

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


def _version_from_pyproject() -> Optional[str]:
    """Read the package version from pyproject.toml in source checkouts."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        return None

    content = pyproject.read_text(encoding="utf-8")
    if "[project]" not in content:
        return None

    project_block = content.split("[project]", maxsplit=1)[1]
    project_block = re.split(r"^\[", project_block, maxsplit=1, flags=re.MULTILINE)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project_block, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1)


def _resolve_version() -> str:
    """Resolve package version from installed metadata or source pyproject."""
    source_version = _version_from_pyproject()
    if source_version is not None:
        return source_version

    try:
        return version("symm-learning")
    except PackageNotFoundError:
        return "0+unknown"


# Patch eagerly on package import so any module using escnn reps shares the singleton caches.
_ensure_group_deepcopy_singleton()

from symm_learning import linalg, models, nn, representation_theory, stats, utils  # noqa: F401

__version__ = _resolve_version()

__all__ = [
    "__version__",
    "linalg",
    "models",
    "nn",
    "representation_theory",
    "stats",
    "utils",
]
