"""Control models subpackage."""

from importlib import import_module

__all__ = ["CondTransformer"]  # noqa: F822


_MODULE_MAP = {
    "CondTransformer": "symm_learning.models.control.cond_transformer",
}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_MAP[name])
    return getattr(module, name)
