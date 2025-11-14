from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from escnn import gspaces, nn as escnn_nn
from escnn.group import CyclicGroup, DihedralGroup, Representation, directsum
from torch.utils.data import DataLoader, TensorDataset

from symm_learning.nn.linear import eLinear

GROUP_BUILDERS: Dict[str, Callable[[int], object]] = {
    "cyclic": CyclicGroup,
    "dihedral": DihedralGroup,
}

ACTIVATION_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
}

ELINEAR_INIT_SCHEMES: Tuple[str, ...] = (
    "kaiming_uniform",
    "kaiming_normal",
    "he_uniform",
    "he_normal",
    "xavier_uniform",
    "xavier_normal",
)


def _build_activation(name: str) -> nn.Module:
    try:
        return ACTIVATION_REGISTRY[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation '{name}'. Options: {list(ACTIVATION_REGISTRY)}") from exc


@dataclass
class TrainingDynamicsConfig:
    symmetry: str = "cyclic"
    symmetry_order: int = 8
    regular_copies: int = 2
    depth: int = 3
    teacher_depth: int = 8
    activation: str = "gelu"
    bias: bool = False
    init: str = "kaiming_uniform"
    teacher_init: str | None = None
    num_samples: int = 2_000
    train_split: float = 0.8
    batch_size: int = 128
    epochs: int = 200
    lr: float = 2e-3
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class ELinearMLP(nn.Module):
    """Stack of eLinear layers with pointwise activations."""

    def __init__(self, rep: Representation, depth: int, activation: str, bias: bool, init: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList([eLinear(rep, rep, bias=bias) for _ in range(depth)])
        for layer in self.layers:
            layer.reset_parameters(scheme=init)
        self.activation = _build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.activation(x)
        return x


class EscnnLinearMLP(nn.Module):
    """Stack of escnn Linear layers operating on GeometricTensors."""

    def __init__(self, rep: Representation, depth: int, activation: str, bias: bool) -> None:
        super().__init__()
        gspace = gspaces.no_base_space(rep.group)
        self.field_type = escnn_nn.FieldType(gspace, [rep])
        self.layers = nn.ModuleList(
            [escnn_nn.Linear(self.field_type, self.field_type, bias=bias) for _ in range(depth)]
        )
        self.activation = _build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gt = escnn_nn.GeometricTensor(x, self.field_type)
        for idx, layer in enumerate(self.layers):
            gt = layer(gt)
            if idx < len(self.layers) - 1:
                activated = self.activation(gt.tensor)
                gt = escnn_nn.GeometricTensor(activated, self.field_type)
        return gt.tensor


def _build_group_rep(cfg: TrainingDynamicsConfig) -> Tuple[object, Representation]:
    builder = GROUP_BUILDERS.get(cfg.symmetry.lower())
    if builder is None:
        raise ValueError(f"Symmetry '{cfg.symmetry}' not supported. Choose from {list(GROUP_BUILDERS)}.")
    group = builder(cfg.symmetry_order)
    rep = directsum([group.regular_representation] * cfg.regular_copies)
    return group, rep


def _build_teacher(rep: Representation, cfg: TrainingDynamicsConfig) -> nn.Module:
    torch.manual_seed(cfg.seed + 1)
    init_scheme = cfg.teacher_init or cfg.init
    teacher = ELinearMLP(rep, depth=cfg.teacher_depth, activation=cfg.activation, bias=cfg.bias, init=init_scheme)
    teacher.to(device=cfg.device, dtype=cfg.dtype)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _sample_dataset(
    rep: Representation, teacher: nn.Module, cfg: TrainingDynamicsConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(cfg.seed + 2)
    inputs = torch.randn(cfg.num_samples, rep.size, device=cfg.device, dtype=cfg.dtype)
    with torch.no_grad():
        targets = teacher(inputs)
    return inputs.cpu(), targets.cpu()


def _build_dataloaders(
    inputs: torch.Tensor, targets: torch.Tensor, cfg: TrainingDynamicsConfig
) -> Tuple[DataLoader, DataLoader]:
    num_train = int(cfg.num_samples * cfg.train_split)
    indices = torch.randperm(cfg.num_samples, generator=torch.Generator().manual_seed(cfg.seed + 3))
    inputs, targets = inputs[indices], targets[indices]
    train_ds = TensorDataset(inputs[:num_train], targets[:num_train])
    val_ds = TensorDataset(inputs[num_train:], targets[num_train:])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def _train_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainingDynamicsConfig, label: str
) -> Dict[str, List[float]]:
    model = model.to(device=cfg.device, dtype=cfg.dtype)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = {"label": label, "train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device, dtype=cfg.dtype)
            yb = yb.to(cfg.device, dtype=cfg.dtype)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        history["train"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device, dtype=cfg.dtype)
                yb = yb.to(cfg.device, dtype=cfg.dtype)
                preds = model(xb)
                batch_loss = criterion(preds, yb)
                val_loss += batch_loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        history["val"].append(val_loss)

    history["final_train"] = history["train"][-1]
    history["final_val"] = history["val"][-1]
    return history


def _training_dynamics_rmse(curve_a: Iterable[float], curve_b: Iterable[float]) -> float:
    a = torch.tensor(list(curve_a), dtype=torch.float64)
    b = torch.tensor(list(curve_b), dtype=torch.float64)
    if a.numel() != b.numel():
        raise ValueError("Loss curves must have the same length to compute the dynamics distance.")
    return float(torch.sqrt(torch.mean((a - b) ** 2)))


def _plot_training_curves(results: Dict[str, Dict[str, List[float]]], cfg: TrainingDynamicsConfig) -> None:
    init_labels = [label for label in results if label.startswith("eLinear-")]
    extra_col = 1 if len(init_labels) > 1 else 0
    fig_cols = 2 + extra_col
    fig, axes = plt.subplots(1, fig_cols, figsize=(5 * fig_cols, 4), tight_layout=True)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    elif not isinstance(axes, list):
        axes = [axes]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label_colors: Dict[str, str] = {}

    train_ax, val_ax = axes[0], axes[1]
    for idx, (label, history) in enumerate(results.items()):
        color = color_cycle[idx % len(color_cycle)]
        label_colors[label] = color
        train_ax.plot(history["train"], label=label, color=color)
        val_ax.plot(history["val"], label=label, color=color)

    train_ax.set_title("Training Loss")
    train_ax.set_xlabel("Epoch")
    train_ax.set_ylabel("MSE")
    train_ax.legend()

    val_ax.set_title("Validation Loss")
    val_ax.set_xlabel("Epoch")
    val_ax.set_ylabel("MSE")
    val_ax.legend()

    if extra_col:
        init_ax = axes[2]
        for label in init_labels:
            color = label_colors.get(label, color_cycle[init_labels.index(label) % len(color_cycle)])
            init_ax.plot(results[label]["train"], label=label, color=color)
        init_ax.set_title("eLinear init comparison (train)")
        init_ax.set_xlabel("Epoch")
        init_ax.set_ylabel("MSE")
        init_ax.legend()

    title = (
        f"{cfg.symmetry.title()}(|G|={cfg.symmetry_order}) regular^{cfg.regular_copies} · depth={cfg.depth} "
        f"(teacher depth={cfg.teacher_depth})"
    )
    fig.suptitle(title)
    plt.show()


def run_visual_training_test(
    cfg: TrainingDynamicsConfig | None = None,
    init_schemes: Sequence[str] | None = None,
    include_escnn: bool = True,
) -> Dict[str, Dict[str, List[float]]]:
    cfg = cfg or TrainingDynamicsConfig()
    torch.manual_seed(cfg.seed)

    _, rep = _build_group_rep(cfg)
    teacher = _build_teacher(rep, cfg)
    inputs, targets = _sample_dataset(rep, teacher, cfg)
    train_loader, val_loader = _build_dataloaders(inputs, targets, cfg)

    schemes = tuple(init_schemes) if init_schemes is not None else ELINEAR_INIT_SCHEMES
    results: Dict[str, Dict[str, List[float]]] = {}

    for scheme in schemes:
        if scheme not in ELINEAR_INIT_SCHEMES:
            raise ValueError(f"Initialization '{scheme}' is not supported. Choose from {ELINEAR_INIT_SCHEMES}.")
        label = f"eLinear-{scheme}"
        model = ELinearMLP(rep, depth=cfg.depth, activation=cfg.activation, bias=cfg.bias, init=scheme)
        history = _train_model(model, train_loader, val_loader, cfg, label=label)
        results[label] = history

    if include_escnn:
        escnn_model = EscnnLinearMLP(rep, depth=cfg.depth, activation=cfg.activation, bias=cfg.bias)
        results["escnn"] = _train_model(escnn_model, train_loader, val_loader, cfg, label="escnn")

    print("=== Training Summary ===")
    for label, history in results.items():
        print(f"{label:12s} | train MSE: {history['final_train']:.4e} | val MSE: {history['final_val']:.4e}")

    baseline_label = next(iter(results))
    for label, history in list(results.items())[1:]:
        rmse = _training_dynamics_rmse(history["train"], results[baseline_label]["train"])
        print(f"RMSE(train_loss {label} vs {baseline_label}): {rmse:.4e}")

    _plot_training_curves(results, cfg)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare eLinear vs escnn training dynamics.")
    parser.add_argument(
        "--init-schemes",
        nargs="+",
        choices=ELINEAR_INIT_SCHEMES,
        help="List of eLinear init schemes to compare. Defaults to all schemes if omitted.",
    )
    parser.add_argument(
        "--all-inits",
        action="store_true",
        help="Shortcut to compare every supported eLinear initialization scheme.",
    )
    parser.add_argument(
        "--skip-escnn",
        action="store_true",
        help="Skip training the escnn baseline and only sweep eLinear initializations.",
    )
    args = parser.parse_args()

    schemes: Sequence[str] | None
    if args.init_schemes:
        schemes = tuple(args.init_schemes)
    elif args.all_inits:
        schemes = ELINEAR_INIT_SCHEMES
    else:
        schemes = None

    run_visual_training_test(
        init_schemes=schemes,
        include_escnn=not args.skip_escnn,
    )
