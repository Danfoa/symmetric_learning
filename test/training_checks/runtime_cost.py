from __future__ import annotations

import argparse
from typing import Callable

import escnn
import torch

from symm_learning.models import MLP, eMLP
from symm_learning.nn.linear import eLinear
from symm_learning.representation_theory import direct_sum
from symm_learning.utils import run_module_pair_profile

MODULE_BUILDERS: dict[str, Callable[..., torch.nn.Module]] = {
    "elinear": lambda rep, hidden_units, bias: eLinear(in_rep=rep, out_rep=rep, bias=bias),
    "linear": lambda rep, hidden_units, bias: torch.nn.Linear(in_features=rep.size, out_features=rep.size, bias=bias),
    "emlp": lambda rep, hidden_units, bias: eMLP(in_rep=rep, out_rep=rep, hidden_units=hidden_units, bias=bias),
    "mlp": lambda rep, hidden_units, bias: MLP(in_dim=rep.size, out_dim=rep.size, hidden_units=hidden_units, bias=bias),
}


def _parse_hidden_units(value: str) -> list[int]:
    units = [v.strip() for v in value.split(",") if v.strip()]
    if not units:
        raise ValueError("hidden_units cannot be empty.")
    return [int(v) for v in units]


def _parse_compare(value: str) -> tuple[str, str]:
    names = [v.strip().lower() for v in value.split(",") if v.strip()]
    if len(names) != 2:
        raise ValueError("--compare must contain exactly two comma-separated module names.")
    unsupported = [name for name in names if name not in MODULE_BUILDERS]
    if unsupported:
        raise ValueError(f"Unsupported modules {unsupported}. Available: {list(MODULE_BUILDERS)}")
    return names[0], names[1]


def _build_group(name: str, order: int) -> escnn.group.Group:
    key = name.lower()
    if key == "cyclic":
        return escnn.group.CyclicGroup(order)
    if key == "dihedral":
        return escnn.group.DihedralGroup(order)
    if key == "icosahedral":
        return escnn.group.Icosahedral()
    raise ValueError(f"Unsupported group '{name}'. Available: cyclic, dihedral, icosahedral")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile runtime costs of selected modules.")
    parser.add_argument(
        "--compare",
        type=str,
        default="elinear,linear",
        help=f"Two module names to compare, comma-separated. Available: {list(MODULE_BUILDERS)}",
    )
    parser.add_argument("--group", type=str, default="cyclic", help="Group type: cyclic, dihedral, icosahedral")
    parser.add_argument("--group-order", type=int, default=5, help="Order for cyclic/dihedral groups")
    parser.add_argument("--regular-copies", type=int, default=2, help="How many regular reps to direct-sum")
    parser.add_argument("--hidden-units", type=str, default="128,128", help="Hidden units for (e)MLP modules")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--profile-active-steps", type=int, default=25)
    parser.add_argument("--profile-warmup-steps", type=int, default=5)
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "train", "both"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bias", action="store_true", help="Enable module bias (disabled by default)")
    parser.add_argument("--top-k", type=int, default=10, help="Top ops to print")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    lhs_name, rhs_name = _parse_compare(args.compare)
    hidden_units = _parse_hidden_units(args.hidden_units)

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)

    G = _build_group(args.group, args.group_order)
    rep = direct_sum([G.regular_representation] * args.regular_copies)

    lhs = MODULE_BUILDERS[lhs_name](rep=rep, hidden_units=hidden_units, bias=args.bias).to(device=device, dtype=dtype)
    rhs = MODULE_BUILDERS[rhs_name](rep=rep, hidden_units=hidden_units, bias=args.bias).to(device=device, dtype=dtype)
    x = torch.randn(args.batch_size, rep.size, device=device, dtype=dtype)

    run_module_pair_profile(
        lhs_name=lhs_name,
        lhs=lhs,
        rhs_name=rhs_name,
        rhs=rhs,
        x=x,
        group_name=G.name,
        profile_active_steps=args.profile_active_steps,
        profile_warmup_steps=args.profile_warmup_steps,
        mode=args.mode,
        top_k=args.top_k,
    )
