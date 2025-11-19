# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from escnn.group import Representation


class CallableDict(dict, Callable):
    """Dictionary that can be called as a function."""

    def __call__(self, key):
        """Return the value of the key."""
        return self[key]


def module_memory(module: torch.nn.Module):  # noqa: D103
    trainable = 0
    frozen = 0
    for p in module.parameters():
        nbytes = p.numel() * p.element_size()
        if p.requires_grad:
            trainable += nbytes
        else:
            frozen += nbytes
    buffer_bytes = sum(buf.numel() * buf.element_size() for buf in module.buffers())
    non_trainable = frozen + buffer_bytes
    return trainable, non_trainable


def bytes_to_mb(num_bytes: int) -> float:  # noqa: D103
    return num_bytes / (1024**2)


def check_equivariance(
    e_module,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    input_dim: int | tuple[int] = 2,
    in_rep: Representation | tuple[Representation, ...] = None,
    out_rep: Representation | tuple[Representation, ...] = None,
):
    """Method that automatically tests the equivariance of the current module."""
    in_rep = e_module.in_rep if hasattr(e_module, "in_rep") else in_rep
    out_rep = e_module.out_rep if hasattr(e_module, "out_rep") else out_rep
    assert in_rep is not None, f"in_rep must be provided or be an attribute of the module {e_module}"
    assert out_rep is not None, f"out_rep must be provided or be an attribute of the module {e_module}"
    G = in_rep.group

    batch_size = 11
    spatial_dims = 9

    if input_dim == 2:
        x = torch.randn(batch_size, in_rep.size)
    elif input_dim == 3:
        x = torch.randn(batch_size, spatial_dims, in_rep.size)
    else:
        raise ValueError(f"Input dimension {input_dim} not supported.")

    dtype, device = x.dtype, x.device

    errors = []

    if isinstance(e_module, torch.nn.Module):
        e_module.eval()  # Disable dropout, batchnorm, etc.

    # for el in self.out_type.testing_elements:
    for _ in range(20):
        g = G.sample()
        gx = torch.einsum("ij,...j->...i", torch.tensor(in_rep(g), dtype=dtype, device=device), x)
        y = e_module(x)
        gy = torch.einsum("ij,...j->...i", torch.tensor(out_rep(g)).to(dtype=dtype, device=device), y)
        gy = gy.detach().cpu().numpy()

        gy_expected = e_module(gx).detach().cpu().numpy()

        errs = gy - gy_expected
        errs = np.abs(errs).reshape(-1)

        assert gy.shape == gy_expected.shape, f"Shape mismatch: got {gy.shape}, expected {gy_expected.shape}"
        assert np.allclose(gy, gy_expected, atol=atol, rtol=rtol), (
            f"Equivariance test failed for group element {g} max scalar error {errs.max()}"
        )

        errors.append((g, errs.mean()))

    print(f"Equivariant check passed for module {e_module}")
