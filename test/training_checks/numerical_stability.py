from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn.group import CyclicGroup, DihedralGroup, directsum
from symm_learning.nn.linear import eAffine, eLinear
from symm_learning.nn.normalization import eLayerNorm
from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint
from symm_learning.utils import check_equivariance

GROUP_BUILDERS: Dict[str, Callable[[int], object]] = {
    "cyclic": CyclicGroup,
    "dihedral": DihedralGroup,
}

ACTIVATION_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
}

ELINEAR_INIT_SCHEMES = (
    "kaiming_uniform",
    "kaiming_normal",
    "he_uniform",
    "he_normal",
    "xavier_uniform",
    "xavier_normal",
)


def _kaiming_normal(weight: torch.Tensor) -> None:
    nn.init.kaiming_normal_(weight, nonlinearity="linear")


def _kaiming_uniform(weight: torch.Tensor) -> None:
    nn.init.kaiming_uniform_(weight, nonlinearity="linear")


def _no_init(_tensor: torch.Tensor) -> None:
    """Placeholder initializer that leaves the tensor untouched."""
    return None


ModuleBlueprint = Union[nn.Module, Callable[[], nn.Module]]
ModuleEntry = Union[ModuleBlueprint, Tuple[ModuleBlueprint, Callable[[nn.Module], None]]]


def _init_hook_factory(weight_init: Callable[[torch.Tensor], None]) -> Callable[[nn.Module], None]:
    """Create a recursive initializer mirroring eCondTransformer init logic."""

    def _apply_init(tensor: torch.Tensor, init_fn: Callable[[torch.Tensor], None]) -> torch.Tensor:
        """Run ``init_fn`` and return the tensor to support assignment semantics."""
        result = init_fn(tensor)
        return tensor if result is None else result

    @torch.no_grad()
    def _init_weights(module: nn.Module) -> None:
        class_name = module.__class__.__name__
        ignore_types = (
            torch.nn.Dropout,
            torch.nn.Sequential,
            torch.nn.ModuleList,
            torch.nn.ModuleDict,
            torch.nn.Identity,
            torch.nn.ReLU,
            torch.nn.GELU,
            torch.nn.SiLU,
            torch.nn.Mish,
            torch.nn.Tanh,
            CommutingConstraint,
            InvariantConstraint,
            escnn.nn.Linear,
        )
        if isinstance(module, eLinear):
            module.reset_parameters(scheme="xavier_normal")
            # module.reset_parameters(scheme="kaiming_uniform")
        elif isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight = _apply_init(module.weight, weight_init)
                if module.bias is not None:
                    module.bias = _apply_init(module.bias, nn.init.zeros_)

        elif isinstance(module, torch.nn.LayerNorm):
            module.reset_parameters()
        elif isinstance(module, eLayerNorm):
            pass
        elif isinstance(module, eAffine):
            with torch.no_grad():
                module.scale_dof = _apply_init(module.scale_dof, nn.init.uniform_)
                if getattr(module, "bias_dof", None) is not None:
                    module.bias_dof = _apply_init(module.bias_dof, nn.init.zeros_)
        elif (
            isinstance(module, ignore_types)
            or class_name.startswith("Parametrized")
            or class_name in ["BlocksBasisExpansion", "SingleBlockBasisExpansion"]
        ):
            return
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    return _init_weights


class LayerRecorder:
    """Tracks activations and their backward signals."""

    def __init__(self) -> None:
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}

    def capture(self, name: str, tensor: torch.Tensor) -> None:
        self.activations[name] = tensor.detach().flatten().cpu()

        def _hook(grad: torch.Tensor, key: str = name) -> torch.Tensor:
            self.gradients[key] = grad.detach().flatten().cpu()
            return grad

        tensor.register_hook(_hook)


def _build_stack(
    blueprint: ModuleBlueprint,
    depth: int,
    device: str,
    dtype: torch.dtype,
    init_hook: Optional[Callable[[nn.Module], None]] = None,
) -> nn.Sequential:
    layers = []
    for _ in range(depth):
        layer = copy.deepcopy(blueprint)
        layer.to(device=device, dtype=dtype)
        layers.append(layer)

    net = nn.Sequential(*layers)
    hook = init_hook or _init_hook_factory(INIT_REGISTRY["kaiming_normal"])
    net.apply(hook)
    # print(net)
    return net


def _run_stack(stack: nn.Sequential, inputs: torch.Tensor, target: torch.Tensor, prefix: str) -> dict:
    recorder = LayerRecorder()
    x = inputs
    for idx, module in enumerate(stack, start=1):
        try:
            x = module(x)
        except Exception as e:
            x = module(module[0].in_type(x)).tensor
        recorder.capture(f"L{idx:02d}", x)
    loss = F.mse_loss(x, target)
    loss.backward()
    stack.zero_grad(set_to_none=True)
    return {"activations": recorder.activations, "gradients": recorder.gradients, "loss": loss.item(), "label": prefix}


def _stats_to_frame(stats: Dict[str, dict], metric: str) -> pd.DataFrame:
    rows = []
    for stack_name, data in stats.items():
        tensors = data[metric]
        for layer_name in sorted(tensors):
            values = tensors[layer_name]
            rows.append(
                pd.DataFrame(
                    {
                        "value": values.numpy(),
                        "layer": layer_name,
                        "stack": stack_name,
                        "metric": metric,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def plot_violin_distributions(stats: Dict[str, dict], title: str | None = None) -> None:
    sns.set_theme(style="whitegrid")
    metrics = ["activations", "gradients"]
    stacks = list(stats.keys())
    fig, axes = plt.subplots(
        len(metrics),
        len(stacks),
        figsize=(5 * len(stacks), 4 * len(metrics)),
        squeeze=False,
        tight_layout=True,
    )

    for row, metric in enumerate(metrics):
        df = _stats_to_frame(stats, metric)
        for col, stack_name in enumerate(stacks):
            ax = axes[row][col]
            subset = df[df["stack"] == stack_name]
            sns.violinplot(
                data=subset,
                x="layer",
                y="value",
                inner="quartile",
                density_norm="width",
                cut=0,
                color=sns.color_palette("Set2")[row % len(sns.color_palette("Set2"))],
                ax=ax,
            )
            ax.set_title(f"{stack_name} – {metric}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Value")

    if title:
        fig.suptitle(title)
    plt.show()


def run_experiment(
    modules: Dict[str, ModuleEntry], input_shape: Tuple[int, ...], cfg=None, title: str = None
) -> Dict[str, dict]:
    cfg = cfg or DemoBlockConfig()
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(cfg.dtype)

    base_inputs = torch.randn(cfg.batch_size, *input_shape, device=cfg.device, dtype=cfg.dtype)
    target = torch.zeros_like(base_inputs)

    stats = {}
    for name, entry in modules.items():
        # Allow passing either the module blueprint directly or a (blueprint, init_hook) tuple.
        if isinstance(entry, tuple):
            if len(entry) != 2:
                raise ValueError(f"Module entry for '{name}' must be (blueprint, init_hook).")
            blueprint, init_hook = entry
        else:
            blueprint, init_hook = entry, None

        stack = _build_stack(blueprint, cfg.depth, cfg.device, cfg.dtype, init_hook=init_hook)
        inputs = base_inputs.clone().requires_grad_(True)
        stats[name] = _run_stack(stack, inputs, target, prefix=name)

    plot_violin_distributions(stats, title=title)
    return stats


def collect_initialization_stats(
    rep, cfg: DemoBlockConfig, schemes: Optional[Tuple[str, ...]] = None
) -> Dict[str, dict]:
    """Sample eLinear parameters for every supported initialization scheme."""

    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(cfg.dtype)
    schemes = schemes or cfg.init_schemes
    stats: Dict[str, dict] = {}
    for scheme in schemes:
        layer = eLinear(rep, rep, bias=cfg.bias).to(device=cfg.device, dtype=cfg.dtype)
        layer.reset_parameters(scheme=scheme)
        stats[scheme] = {
            "weight_dof": layer.weight_dof.detach().flatten().cpu(),
            "weight": layer.weight.detach().flatten().cpu(),
        }
    return stats


def summarize_initialization_stats(stats: Dict[str, dict]) -> None:
    print("=== eLinear initialization summary ===")
    print(f"{'scheme':>16} | {'dof μ±σ':>20} | {'weight μ±σ':>20}")
    for scheme, tensors in stats.items():
        dof = tensors["weight_dof"].float()
        weight = tensors["weight"].float()
        dof_mean, dof_std = dof.mean().item(), dof.std(unbiased=False).item()
        w_mean, w_std = weight.mean().item(), weight.std(unbiased=False).item()
        print(f"{scheme:>16} | {dof_mean:+.3e} ± {dof_std:.3e} | {w_mean:+.3e} ± {w_std:.3e}")


def plot_initialization_histograms(
    stats: Dict[str, dict], cfg: DemoBlockConfig, bins: Optional[int] = None, title: Optional[str] = None
) -> None:
    bins = bins or cfg.hist_bins
    schemes = list(stats.keys())
    fig, axes = plt.subplots(
        2,
        len(schemes),
        figsize=(3.6 * len(schemes), 6),
        sharey="row",
        tight_layout=True,
    )
    if len(schemes) == 1:
        axes = axes.reshape(2, 1)

    for col, scheme in enumerate(schemes):
        dof_vals = stats[scheme]["weight_dof"].numpy()
        weight_vals = stats[scheme]["weight"].numpy()
        axes[0, col].hist(dof_vals, bins=bins, density=True, color="tab:blue", alpha=0.8)
        axes[0, col].set_title(f"{scheme} – DoF")
        axes[0, col].set_xlabel("value")
        axes[0, col].set_ylabel("density" if col == 0 else "")

        axes[1, col].hist(weight_vals, bins=bins, density=True, color="tab:orange", alpha=0.8)
        axes[1, col].set_title(f"{scheme} – Dense weight")
        axes[1, col].set_xlabel("value")
        axes[1, col].set_ylabel("density" if col == 0 else "")

    overall_title = title or f"eLinear initializations ({cfg.symmetry.title()} – depth {cfg.depth})"
    fig.suptitle(overall_title)
    plt.show()


INIT_REGISTRY: Dict[str, Callable[[torch.Tensor], None]] = {
    "kaiming_normal": _kaiming_normal,
    "kaiming_uniform": _kaiming_uniform,
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "orthogonal": nn.init.orthogonal_,
    "uniform": nn.init.uniform_,
    "no_init": _no_init,
}


@dataclass
class DemoBlockConfig:
    depth: int = 8
    batch_size: int = 512
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    #
    symmetry: str = "cyclic"
    symmetry_order: int = 9
    regular_copies: int = 2  # neurons = order * regular_copies
    activation: str = "gelu"
    init: str = "kaiming_uniform"
    bias: bool = False
    init_schemes: Tuple[str, ...] = ELINEAR_INIT_SCHEMES
    hist_bins: int = 60


if __name__ == "__main__":
    import argparse
    import escnn

    parser = argparse.ArgumentParser(description="Visual tests for eLinear stacks and initializations.")
    parser.add_argument(
        "--experiment",
        choices=("stack", "inits"),
        default="stack",
        help="stack: reproduce activation/gradient violin plots; inits: plot eLinear init distributions.",
    )
    args = parser.parse_args()

    cfg = DemoBlockConfig()

    builder = GROUP_BUILDERS.get(cfg.symmetry.lower())
    if builder is None:
        raise ValueError(f"Unsupported symmetry '{cfg.symmetry}'. Choose from {list(GROUP_BUILDERS)}.")
    # G = builder(cfg.symmetry_order)
    G = escnn.group.Icosahedral()
    rep = directsum([G.regular_representation] * cfg.regular_copies)
    dim = rep.size

    if args.experiment == "stack":
        dense_block = torch.nn.Sequential(
            nn.Linear(dim, dim, bias=cfg.bias),
            # _activation_factory(cfg.activation)(),
            # nn.LayerNorm(dim),
        )
        dense_init_hook = _init_hook_factory(INIT_REGISTRY[cfg.init.lower()])

        equiv_block = torch.nn.Sequential(
            eLinear(rep, rep, bias=cfg.bias),
            # _activation_factory(cfg.activation)(),
            # eLayerNorm(rep, bias=cfg.bias),
        )
        equiv_init_hook = _init_hook_factory(INIT_REGISTRY[cfg.init.lower()])

        in_type = escnn.nn.FieldType(escnn.gspaces.no_base_space(G), [rep])
        escnn_block = torch.nn.Sequential(
            escnn.nn.Linear(in_type, in_type, bias=cfg.bias),
            # _activation_factory(cfg.activation)(),
            # eLayerNorm(rep, bias=cfg.bias),
        )

        escnnlin = escnn.nn.Linear(in_type, in_type, bias=cfg.bias)

        run_experiment(
            modules={
                "dense": (dense_block, dense_init_hook),
                "equivariant": (equiv_block, equiv_init_hook),
                "escnn": (escnn_block, _init_hook_factory(INIT_REGISTRY["no_init"])),
            },
            input_shape=(dim,),
            cfg=cfg,
            title=(
                f"{cfg.symmetry.title()}(|G|={dim // cfg.regular_copies}) with depth={cfg.depth}, "
                f"dim={dim}, activation={cfg.activation} init={cfg.init}"
            ),
        )
    else:
        init_stats = collect_initialization_stats(rep, cfg, schemes=cfg.init_schemes)
        summarize_initialization_stats(init_stats)
        plot_initialization_histograms(
            init_stats,
            cfg,
            title=f"Initialization gallery (dim={dim}, copies={cfg.regular_copies})",
        )
