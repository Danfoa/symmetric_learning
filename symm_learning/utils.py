# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from collections.abc import Callable

import torch


class CallableDict(dict, Callable):
    """Dictionary that can be called as a function."""

    def __call__(self, key):
        """Return the value of the key."""
        return self[key]
