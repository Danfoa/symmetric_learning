from __future__ import annotations

import torch


class eModule(torch.nn.Module):
    """Lightweight base class for equivariant modules.

    This base centralizes lifecycle behavior related to cache validity and mode transitions.
    Subclasses are expected to:
    - define ``in_rep`` and ``out_rep`` when ``requires_reps=True``
    - optionally override ``invalidate_cache`` and ``_refresh_eval_cache``
    """

    requires_reps: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def _refresh_eval_cache(self) -> None:
        """Materialize eval-time caches after switching to eval mode."""

    def train(self, mode: bool = True):  # noqa: D102
        result = super().train(mode)
        if mode:
            self.invalidate_cache()
        else:
            self._refresh_eval_cache()
        return result

    def _apply(self, fn):
        result = super()._apply(fn)
        self.invalidate_cache()
        return result

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D102
        result = super().load_state_dict(state_dict, strict)
        self.invalidate_cache()
        if not self.training:
            self._refresh_eval_cache()
        return result
