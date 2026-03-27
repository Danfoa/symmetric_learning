from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import io
import time
from typing import Any

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

    with torch.no_grad():
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


def _assert_output_close(expected: Any, actual: Any, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    from torch.distributions import MultivariateNormal

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if isinstance(expected, MultivariateNormal):
        assert isinstance(actual, MultivariateNormal), f"Expected MultivariateNormal, got {type(actual)}"
        torch.testing.assert_close(actual.mean, expected.mean, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix, atol=atol, rtol=rtol)
        return
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple), f"Expected tuple output, got {type(actual)}"
        assert len(actual) == len(expected), f"Tuple lengths differ: {len(actual)} != {len(expected)}"
        for got_item, exp_item in zip(actual, expected):
            _assert_output_close(exp_item, got_item, atol=atol, rtol=rtol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"Expected list output, got {type(actual)}"
        assert len(actual) == len(expected), f"List lengths differ: {len(actual)} != {len(expected)}"
        for got_item, exp_item in zip(actual, expected):
            _assert_output_close(exp_item, got_item, atol=atol, rtol=rtol)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected dict output, got {type(actual)}"
        assert actual.keys() == expected.keys(), f"Dict keys differ: {actual.keys()} != {expected.keys()}"
        for key in expected:
            _assert_output_close(expected[key], actual[key], atol=atol, rtol=rtol)
        return
    assert actual == expected, f"Outputs differ: {actual} != {expected}"


def _assert_state_dict_close(
    expected_state: dict[str, Any],
    actual_state: dict[str, Any],
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    assert actual_state.keys() == expected_state.keys(), (
        f"State dict keys differ: {actual_state.keys()} != {expected_state.keys()}"
    )
    for key in expected_state:
        expected_val = expected_state[key]
        actual_val = actual_state[key]
        if isinstance(expected_val, torch.Tensor):
            assert isinstance(actual_val, torch.Tensor), f"State key '{key}' expected Tensor, got {type(actual_val)}"
            torch.testing.assert_close(actual_val.detach().cpu(), expected_val.detach().cpu(), atol=atol, rtol=rtol)
        else:
            assert actual_val == expected_val, f"State key '{key}' differs: {actual_val} != {expected_val}"


def _clone_module_for_reload(module: torch.nn.Module) -> torch.nn.Module:
    was_training = module.training
    module.train()
    for submodule in module.modules():
        invalidate_cache = getattr(submodule, "invalidate_cache", None)
        if callable(invalidate_cache):
            invalidate_cache()
    clone = deepcopy(module)
    module.train(was_training)
    return clone


class _ParentWrapper(torch.nn.Module):
    def __init__(self, child: torch.nn.Module):
        super().__init__()
        self.child = child

    def forward(self, *args, **kwargs):
        return self.child(*args, **kwargs)


def assert_module_save_load_consistency(
    module: torch.nn.Module,
    *forward_args,
    forward_kwargs: dict[str, Any] | None = None,
    output_transform: Callable[[Any], Any] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    mode_pairs: tuple[tuple[bool, bool], ...] = ((True, True), (False, False), (True, False), (False, True)),
) -> None:
    """Assert that save/load round-trips preserve outputs across save/load modes.

    Args:
        module: Module under test.
        *forward_args: Positional args passed to forward.
        forward_kwargs: Optional forward kwargs.
        output_transform: Optional projection from module output to comparable values.
        atol: Absolute tolerance for numerical comparisons.
        rtol: Relative tolerance for numerical comparisons.
        mode_pairs: Tuples of ``(save_training, load_training)`` to test.
    """
    kwargs = {} if forward_kwargs is None else forward_kwargs
    transform = (lambda out: out) if output_transform is None else output_transform

    original_mode = module.training

    try:
        for idx, (save_training, load_training) in enumerate(mode_pairs):
            module.train(save_training)

            # Prime module once in the save mode to emulate arbitrary user save points.
            with torch.no_grad():
                _ = module(*forward_args, **kwargs)

            state_buffer = io.BytesIO()
            torch.save(module.state_dict(), state_buffer)
            state_buffer.seek(0)
            loaded_state = torch.load(state_buffer, map_location="cpu", weights_only=False)

            module.train(load_training)
            seed = 1234 + idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                expected = transform(module(*forward_args, **kwargs))

            # Path 1: direct module load_state_dict(...)
            reloaded_direct = _clone_module_for_reload(module)
            reloaded_direct.load_state_dict(loaded_state)
            reloaded_direct.train(load_training)
            _assert_state_dict_close(loaded_state, reloaded_direct.state_dict(), atol=atol, rtol=rtol)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                actual_direct = transform(reloaded_direct(*forward_args, **kwargs))
            _assert_output_close(expected, actual_direct, atol=atol, rtol=rtol)

            # Path 2: parent-recursive load_state_dict(...) into submodule
            reloaded_parent = _clone_module_for_reload(module)
            parent = _ParentWrapper(reloaded_parent)
            parent_state = {f"child.{k}": v for k, v in loaded_state.items()}
            parent.load_state_dict(parent_state)
            reloaded_parent.train(load_training)
            _assert_state_dict_close(loaded_state, reloaded_parent.state_dict(), atol=atol, rtol=rtol)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                actual_parent = transform(reloaded_parent(*forward_args, **kwargs))
            _assert_output_close(expected, actual_parent, atol=atol, rtol=rtol)
    finally:
        module.train(original_mode)
