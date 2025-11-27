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


def _tensor_bytes(t: torch.Tensor) -> int:
    if t.layout == torch.sparse_coo:
        vals = t._values()
        idx = t._indices()
        return vals.numel() * vals.element_size() + idx.numel() * idx.element_size()
    if t.layout == torch.sparse_csr or t.layout == torch.sparse_bsr:
        total = t.values().numel() * t.values().element_size()
        total += t.crow_indices().numel() * t.crow_indices().element_size()
        total += t.col_indices().numel() * t.col_indices().element_size()
        return total
    if t.layout == torch.sparse_csc:
        total = t.values().numel() * t.values().element_size()
        total += t.ccol_indices().numel() * t.ccol_indices().element_size()
        total += t.row_indices().numel() * t.row_indices().element_size()
        return total
    return t.numel() * t.element_size()


def module_memory(module: torch.nn.Module, units: str = "bytes"):
    """Return trainable/non-trainable parameter memory for ``module``.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameter memory footprint is inspected.
    units : {"bytes", "kib", "mib", "gib"}, optional
        Output units. ``kib``/``mib``/``gib`` are binary kilo/mega/gigabytes (powers of 1024).
        Default is raw bytes.

    Returns:
    -------
    Tuple[float, float]
        Memory in the selected units stored in parameters that require gradients, and frozen parameters plus buffers.
    """
    unit_scale = {
        "bytes": 1,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    if units.lower() not in unit_scale:
        raise ValueError(f"Unsupported units '{units}'. Expected one of {tuple(unit_scale.keys())}.")
    scale = unit_scale[units.lower()]

    trainable = 0
    frozen = 0
    for p in module.parameters():
        nbytes = _tensor_bytes(p)
        if p.requires_grad:
            trainable += nbytes
        else:
            frozen += nbytes
    buffer_bytes = sum(_tensor_bytes(buf) for buf in module.buffers())
    non_trainable = frozen + buffer_bytes
    return trainable / scale, non_trainable / scale


def backprop_sanity(module: torch.nn.Module) -> None:
    """Simple backpropagation sanity check for ``module``."""
    module.train()
    optim = torch.optim.SGD(module.parameters(), lr=1e-3)
    x = torch.randn(16, module.in_rep.size)
    target = torch.randn(16, module.out_rep.size)
    optim.zero_grad()
    y = module(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
    assert grad_norms, "Expected at least one gradient to propagate."
    optim.step()


def module_memory_breakdown(module: torch.nn.Module) -> list[dict]:
    """Return a per-tensor memory breakdown for ``module`` (sizes in bytes)."""

    def _entry(name: str, tensor: torch.Tensor, kind: str, trainable: bool) -> dict:
        return {
            "name": name,
            "kind": kind,
            "trainable": bool(trainable),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": tensor.numel(),
            "bytes": _tensor_bytes(tensor),
        }

    details: list[dict] = []
    for name, param in module.named_parameters():
        details.append(_entry(name, param, "parameter", param.requires_grad))
    for name, buf in module.named_buffers():
        details.append(_entry(name, buf, "buffer", False))

    return details


def describe_memory(label: str, module: torch.nn.Module) -> None:
    """Pretty-print parameter/buffer memory consumption for ``module``."""
    entries = module_memory_breakdown(module)
    if not entries:
        print(f"\n{label}: no parameters or buffers.")
        return
    entries.sort(key=lambda item: item["bytes"], reverse=True)

    print(f"\n{label} memory breakdown")
    header = f"{'Kind':<12}{'Name':<45}{'Trainable':<11}{'Shape':<25}{'DType':<12}{'MB':>10}"
    print(header)
    print("-" * len(header))
    for entry in entries:
        shape = "×".join(str(dim) for dim in entry["shape"]) if entry["shape"] else "scalar"
        mb = bytes_to_mb(entry["bytes"])
        print(
            f"{entry['kind']:<12}{entry['name']:<45}{str(entry['trainable']):<11}"
            f"{shape:<25}{entry['dtype']:<12}{mb:>10.3f}"
        )
    total_trainable = sum(bytes_to_mb(e["bytes"]) for e in entries if e["kind"] == "parameter" and e["trainable"])
    total_frozen = sum(bytes_to_mb(e["bytes"]) for e in entries if e["kind"] == "parameter" and not e["trainable"])
    total_buffers = sum(bytes_to_mb(e["bytes"]) for e in entries if e["kind"] == "buffer")
    print("-" * len(header))
    print(f"{'Trainable params MB:':<68}{total_trainable:>10.3f}")
    print(f"{'Frozen params MB:':<68}{total_frozen:>10.3f}")
    print(f"{'Buffers MB:':<68}{total_buffers:>10.3f}")


def module_device_memory(module: torch.nn.Module, device: str | torch.device | None = None) -> tuple[int, int]:
    """Measure CUDA memory footprint (bytes allocated/peak) of ``module`` cloned onto ``device``."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type != "cuda":
        return 0, 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    module.to("cpu")
    before = torch.cuda.memory_allocated(device)
    module_clone = module.to(device)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) - before
    peak = torch.cuda.max_memory_allocated(device)

    del module_clone
    torch.cuda.empty_cache()
    return allocated, peak


def bytes_to_mb(num_bytes: int) -> float:  # noqa: D103
    return num_bytes / (1024**2)


def get_spectral_trivial_dims(rep: Representation) -> list[int]:
    """Return the list of indices of the trivial irreps in the spectral decomposition of ``rep``."""
    G = rep.group
    trivial_id = G.trivial_representation.id
    trivial_spectral_dims = []
    offset = 0
    for irrep_id in rep.irreps:
        if irrep_id == trivial_id:
            trivial_spectral_dims.append(offset)
        offset += G.irrep(*irrep_id).size

    return trivial_spectral_dims


def get_spectral_trivial_mask(rep: Representation) -> torch.Tensor:
    """Return a boolean mask selecting the trivial irreps in the spectral decomposition of ``rep``."""
    trivial_dims = get_spectral_trivial_dims(rep)
    mask = torch.zeros(rep.size, dtype=torch.bool)
    mask[trivial_dims] = 1
    return mask


def check_equivariance(
    e_module,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    input_dim: int | tuple[int] = 2,
    in_rep: Representation | tuple[Representation, ...] = None,
    out_rep: Representation | tuple[Representation, ...] = None,
    module_name: str = None,
):
    """Method that automatically tests the equivariance of the current module."""
    in_rep = e_module.in_rep if hasattr(e_module, "in_rep") else in_rep
    out_rep = e_module.out_rep if hasattr(e_module, "out_rep") else out_rep
    assert in_rep is not None, f"in_rep must be provided or be an attribute of the module {e_module}"
    assert out_rep is not None, f"out_rep must be provided or be an attribute of the module {e_module}"
    G = in_rep.group
    module_name = e_module.__class__.__name__ if module_name is None else module_name
    batch_size = 20
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

    print(f"Equivariant check passed for module {module_name} with max error {max(e[1] for e in errors)} ")
