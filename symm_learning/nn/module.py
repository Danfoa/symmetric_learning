from __future__ import annotations

import torch


class eModule(torch.nn.Module):
    """Lightweight base class for equivariant modules.

    This base centralizes lifecycle behavior related to cache validity and mode transitions.
    Subclasses are expected to:
    - define ``in_rep`` and ``out_rep`` when ``requires_reps=True``
    - optionally override ``invalidate_cache``
    """

    requires_reps: bool = True

    def __getattribute__(self, name):  # noqa: D105
        if name in ("in_rep", "out_rep"):
            try:
                return super().__getattribute__(name)
            except AttributeError as exc:
                try:
                    requires_reps = super().__getattribute__("requires_reps")
                except AttributeError:
                    requires_reps = True
                if requires_reps:
                    raise AttributeError(
                        f"{self.__class__.__name__} did not define `{name}`. "
                        "Equivariant modules are expected to define `in_rep` and `out_rep` in the main constructor "
                        "(`__init__`)."
                    ) from exc
                raise
        return super().__getattribute__(name)

    def invalidate_cache(self) -> None:
        """Clear derived cached tensors so they are recomputed on next use."""

    def train(self, mode: bool = True):  # noqa: D102
        result = super().train(mode)
        self.invalidate_cache()
        return result

    def _apply(self, fn):
        result = super()._apply(fn)
        self.invalidate_cache()
        return result

    def _load_from_state_dict(self, *args, **kwargs):  # noqa: D102
        # Use the recursive load path itself so submodules are covered when a parent
        # calls `load_state_dict(...)`.
        result = super()._load_from_state_dict(*args, **kwargs)
        self.invalidate_cache()
        return result
