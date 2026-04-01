# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
from __future__ import annotations

import time
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

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter memory footprint is inspected.
        units (:class:`str`, optional): Output units. One of ``"bytes"``, ``"kib"``, ``"mib"``, ``"gib"``.
             Default is raw bytes.

    Returns:
        :class:`tuple` [:class:`float`, :class:`float`]: Memory in the selected units stored in parameters that
            require gradients, and frozen parameters plus buffers.
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


# ____________________________ PROFILING UTILITIES _____________________________________________


def _profile_stats(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def _n_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _prime_eval_cache(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> None:
    """Populate eval caches once outside measured/profiled loops."""
    model.eval()
    with torch.no_grad():
        _ = model(x)
    _sync(device)


def _run_timing(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    n_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    model.train()
    train_forward_ms: list[float] = []
    backward_ms: list[float] = []
    for i in range(warmup_steps + n_steps):
        model.zero_grad(set_to_none=True)

        _sync(device)
        t0 = time.perf_counter()
        out = model(x)
        _sync(device)
        t1 = time.perf_counter()

        loss = out.square().mean()
        _sync(device)
        t2 = time.perf_counter()
        loss.backward()
        _sync(device)
        t3 = time.perf_counter()

        if i >= warmup_steps:
            train_forward_ms.append((t1 - t0) * 1e3)
            backward_ms.append((t3 - t2) * 1e3)

    model.eval()
    _prime_eval_cache(model, x, device)
    val_forward_ms: list[float] = []
    with torch.no_grad():
        for i in range(warmup_steps + n_steps):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(x)
            _sync(device)
            t1 = time.perf_counter()
            if i >= warmup_steps:
                val_forward_ms.append((t1 - t0) * 1e3)

    f_m, f_s = _profile_stats(train_forward_ms)
    b_m, b_s = _profile_stats(backward_ms)
    v_m, v_s = _profile_stats(val_forward_ms)
    return {
        "train_forward_ms_mean": f_m,
        "train_forward_ms_std": f_s,
        "backward_ms_mean": b_m,
        "backward_ms_std": b_s,
        "val_forward_ms_mean": v_m,
        "val_forward_ms_std": v_s,
    }


def _profile_ops(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    mode: str,
    active_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    use_cuda = device.type == "cuda"
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model.train(mode == "train")
    if mode == "eval":
        _prime_eval_cache(model, x, device)
    prof_warmup_steps = warmup_steps if mode == "train" else 0
    with torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        for _ in range(prof_warmup_steps + active_steps):
            if mode == "train":
                model.zero_grad(set_to_none=True)
                out = model(x)
                loss = out.square().mean()
                loss.backward()
            else:
                with torch.no_grad():
                    _ = model(x)
            prof.step()

    op_to_ms_per_step: dict[str, float] = {}
    for item in prof.key_averages():
        op_name = item.key
        if not op_name.startswith("aten::"):
            continue
        if use_cuda:
            self_time_us = getattr(item, "self_cuda_time_total", None)
            if self_time_us is None:
                self_time_us = getattr(item, "self_device_time_total", 0.0)
        else:
            self_time_us = getattr(item, "self_cpu_time_total", 0.0)
        self_time_us = float(self_time_us)
        if self_time_us <= 0:
            continue
        op_to_ms_per_step[op_name] = self_time_us / 1000.0 / active_steps
    return op_to_ms_per_step


def _fmt_ms(mean_ms: float, std_ms: float) -> str:
    return f"{mean_ms:7.3f}±{std_ms:6.3f} ms"


def _print_top_ops(title: str, ops: dict[str, float], top_k: int) -> None:
    print(title)
    print(f"{'Op':<30} {'ms/step':>12}")
    print("-" * 44)
    for op_name, ms in sorted(ops.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        print(f"{op_name:<30.30} {ms:10.3f}")


def run_module_pair_profile(
    lhs_name: str,
    lhs: torch.nn.Module,
    rhs_name: str,
    rhs: torch.nn.Module,
    x: torch.Tensor,
    *,
    group_name: str | None = None,
    profile_active_steps: int = 25,
    profile_warmup_steps: int = 5,
    mode: str = "eval",
    top_k: int = 10,
) -> dict[str, object]:
    """Run timing + op profiling for two already-instantiated modules on the same input."""
    if mode not in {"eval", "train", "both"}:
        raise ValueError("mode must be one of: eval, train, both")

    n_steps = profile_active_steps
    warmup_steps = profile_warmup_steps

    device = x.device
    lhs = lhs.to(device=device, dtype=x.dtype)
    rhs = rhs.to(device=device, dtype=x.dtype)

    lhs_timing = _run_timing(lhs, x, device=device, n_steps=n_steps, warmup_steps=warmup_steps)
    rhs_timing = _run_timing(rhs, x, device=device, n_steps=n_steps, warmup_steps=warmup_steps)

    print(f"Device: {device} | dtype: {x.dtype}")
    if group_name is not None:
        print(f"Group: {group_name} | input dim: {x.shape[-1]} | batch: {x.shape[0]}")
    else:
        print(f"Input dim: {x.shape[-1]} | batch: {x.shape[0]}")
    print(f"Compare: {lhs_name} vs {rhs_name}")
    print(f"Timing steps: warmup={warmup_steps}, measured={n_steps}")
    print(f"Profiler steps: warmup={profile_warmup_steps}, active={profile_active_steps}")
    print()
    batch_size = int(x.shape[0])
    input_dim = int(x.shape[-1])
    print("Average Runtime (ms, mean±std)")
    print(
        f"{'Module':<12} {'#params':>10} {'Batch':>8} {'InputDim':>10} {'TrainFwd':>18} {'Backward':>18} {'ValFwd':>18}"
    )
    print("-" * 108)
    print(
        f"{lhs_name:<12} {_n_parameters(lhs):10d} {batch_size:8d} {input_dim:10d} "
        f"{_fmt_ms(lhs_timing['train_forward_ms_mean'], lhs_timing['train_forward_ms_std']):>18} "
        f"{_fmt_ms(lhs_timing['backward_ms_mean'], lhs_timing['backward_ms_std']):>18} "
        f"{_fmt_ms(lhs_timing['val_forward_ms_mean'], lhs_timing['val_forward_ms_std']):>18}"
    )
    print(
        f"{rhs_name:<12} {_n_parameters(rhs):10d} {batch_size:8d} {input_dim:10d} "
        f"{_fmt_ms(rhs_timing['train_forward_ms_mean'], rhs_timing['train_forward_ms_std']):>18} "
        f"{_fmt_ms(rhs_timing['backward_ms_mean'], rhs_timing['backward_ms_std']):>18} "
        f"{_fmt_ms(rhs_timing['val_forward_ms_mean'], rhs_timing['val_forward_ms_std']):>18}"
    )

    profile_modes = ["eval", "train"] if mode == "both" else [mode]
    all_mode_results: dict[str, dict[str, object]] = {}
    for profile_mode in profile_modes:
        lhs_ops = _profile_ops(
            lhs,
            x,
            device=device,
            mode=profile_mode,
            active_steps=profile_active_steps,
            warmup_steps=profile_warmup_steps,
        )
        rhs_ops = _profile_ops(
            rhs,
            x,
            device=device,
            mode=profile_mode,
            active_steps=profile_active_steps,
            warmup_steps=profile_warmup_steps,
        )

        keys = set(lhs_ops) | set(rhs_ops)
        deltas: list[tuple[float, str, float, float, float]] = []
        for op_name in keys:
            lhs_ms = lhs_ops.get(op_name, 0.0)
            rhs_ms = rhs_ops.get(op_name, 0.0)
            delta = lhs_ms - rhs_ms
            if delta <= 1e-3:
                continue
            ratio = lhs_ms / rhs_ms if rhs_ms > 1e-12 else float("inf")
            deltas.append((delta, op_name, lhs_ms, rhs_ms, ratio))
        deltas.sort(reverse=True)

        print()
        print(f"[{profile_mode}] Where {lhs_name} is slower than {rhs_name} (top aten ops, ms/step)")
        print(f"{'Op':<30} {lhs_name + ' (ms)':>12} {rhs_name + ' (ms)':>12} {'Delta':>10} {'Ratio':>8}")
        print("-" * 86)
        for delta, op_name, lhs_ms, rhs_ms, ratio in deltas[:top_k]:
            ratio_str = "inf" if ratio == float("inf") else f"{ratio:.2f}x"
            print(f"{op_name:<30.30} {lhs_ms:10.3f} {rhs_ms:10.3f} {delta:8.3f} {ratio_str:>8}")

        print()
        _print_top_ops(f"[{profile_mode}] Top ops for {lhs_name}", lhs_ops, top_k=top_k)
        print()
        _print_top_ops(f"[{profile_mode}] Top ops for {rhs_name}", rhs_ops, top_k=top_k)

        all_mode_results[profile_mode] = {"lhs_ops": lhs_ops, "rhs_ops": rhs_ops, "slowdowns": deltas}

    return {
        "lhs_timing": lhs_timing,
        "rhs_timing": rhs_timing,
        "modes": all_mode_results,
    }
