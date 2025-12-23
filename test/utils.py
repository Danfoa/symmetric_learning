from __future__ import annotations

from collections.abc import Callable
import time

import torch


def _infer_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    module: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    iters: int = 1000,
    warmup: int = 50,
    lr: float = 1e-5,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Benchmark a module and return independent forward/backward timings (in ms)."""
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")

    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        raise ValueError("benchmark requires parameters that require gradients.")

    device = _infer_device(module)
    use_cuda = device.type == "cuda"
    was_training = module.training
    module.train()

    optim = torch.optim.SGD(params, lr=lr)
    forward_times, backward_times = [], []

    if use_cuda:
        _sync_if_cuda(device)
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

    for i in range(iters + warmup):
        optim.zero_grad()
        if use_cuda:
            fwd_start.record()
            y = forward_fn()
            fwd_end.record()

            loss = y.pow(2).mean()
            bwd_start.record()
            loss.backward()
            bwd_end.record()
            optim.step()

            _sync_if_cuda(device)
            if i >= warmup:
                forward_times.append(fwd_start.elapsed_time(fwd_end))
                backward_times.append(bwd_start.elapsed_time(bwd_end))
        else:
            fwd_start_t = time.perf_counter()
            y = forward_fn()
            fwd_end_t = time.perf_counter()

            loss = y.pow(2).mean()
            bwd_start_t = time.perf_counter()
            loss.backward()
            bwd_end_t = time.perf_counter()
            optim.step()

            if i >= warmup:
                forward_times.append((fwd_end_t - fwd_start_t) * 1000.0)
                backward_times.append((bwd_end_t - bwd_start_t) * 1000.0)

    if not was_training:
        module.eval()

    fwd_stats = torch.tensor(forward_times)
    bwd_stats = torch.tensor(backward_times)
    forward_mean = float(fwd_stats.mean().item())
    forward_std = float(fwd_stats.std(unbiased=False).item())
    backward_mean = float(bwd_stats.mean().item())
    backward_std = float(bwd_stats.std(unbiased=False).item())
    return (forward_mean, forward_std), (backward_mean, backward_std)


def benchmark_eval_forward(
    module: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    iters: int = 1000,
    warmup: int = 50,
) -> tuple[float, float]:
    """Benchmark eval-mode forward pass timings (in ms)."""
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")

    device = _infer_device(module)
    use_cuda = device.type == "cuda"
    was_training = module.training
    module.eval()

    forward_times = []
    if use_cuda:
        _sync_if_cuda(device)
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for i in range(iters + warmup):
            if use_cuda:
                fwd_start.record()
                _ = forward_fn()
                fwd_end.record()

                _sync_if_cuda(device)
                if i >= warmup:
                    forward_times.append(fwd_start.elapsed_time(fwd_end))
            else:
                fwd_start_t = time.perf_counter()
                _ = forward_fn()
                fwd_end_t = time.perf_counter()
                if i >= warmup:
                    forward_times.append((fwd_end_t - fwd_start_t) * 1000.0)

    if was_training:
        module.train()

    fwd_stats = torch.tensor(forward_times)
    forward_mean = float(fwd_stats.mean().item())
    forward_std = float(fwd_stats.std(unbiased=False).item())
    return forward_mean, forward_std
