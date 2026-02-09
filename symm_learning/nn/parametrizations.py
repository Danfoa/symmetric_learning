import copy
import logging

import torch
from escnn.group import Representation

from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.representation_theory import GroupHomomorphismBasis, direct_sum
from symm_learning.utils import bytes_to_mb, check_equivariance, module_device_memory, module_memory

logger = logging.getLogger(__name__)


class InvariantConstraint(torch.nn.Module):
    r"""Orthogonally project vectors onto :math:`\mathrm{Fix}(\rho)`.

    For representation :math:`\rho_{\mathcal{X}}`, this parametrization enforces

    .. math::
        \mathbf{b} \in \mathrm{Fix}(\rho_{\mathcal{X}})
        = \{\mathbf{v}\in\mathcal{X}: \rho_{\mathcal{X}}(g)\mathbf{v}=\mathbf{v},\ \forall g\in\mathbb{G}\},

    by applying :math:`\mathbf{P}_{\mathrm{inv}}` from
    :func:`~symm_learning.linalg.invariant_orthogonal_projector`.

    Note:
        Runtime behavior depends on mode.
        In training mode (``model.train()``), the projection is recomputed each forward pass.
        In inference mode (``model.eval()``), the projected output is cached for the same unchanged input tensor
        (same object identity and version counter), which is faster.
        With the cache active, the operation is equivalent to a symmetry-agnostic fixed linear map
        :math:`\mathbf{b}\mapsto\mathbf{P}_{\mathrm{inv}}\mathbf{b}`.

    Attributes:
        rep (:class:`~escnn.group.Representation`): Representation :math:`\rho` whose action defines invariance,
            dimension ``rep.size``.
        inv_projector (:class:`~torch.Tensor`): Orthogonal projector of shape
            ``(rep.size, rep.size)`` onto the fixed subspace
            :math:`\mathrm{Fix}(\rho)`.
    """

    def __init__(self, rep: Representation):
        """Precompute the invariant projector for the supplied representation."""
        super().__init__()
        self.rep = rep
        self.register_buffer("inv_projector", invariant_orthogonal_projector(rep))
        self.register_buffer("_bias", None, persistent=False)
        self._cached_input_id = None
        self._cached_input_version = None

    def _cache_is_valid(self, b: torch.Tensor) -> bool:
        if self._bias is None:
            return False
        if id(b) != self._cached_input_id:
            return False
        version = getattr(b, "_version", None)
        return version is not None and version == self._cached_input_version

    def _update_cache(self, b: torch.Tensor, bias: torch.Tensor) -> None:
        version = getattr(b, "_version", None)
        if version is None:
            self.invalidate_cache()
            return
        self._bias = bias
        self._cached_input_id = id(b)
        self._cached_input_version = version

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """Project b onto the invariant subspace using the precomputed projector."""
        if not self.training and self._cache_is_valid(b):
            return self._bias
        bias = torch.mv(self.inv_projector, b)
        if not self.training:
            self._update_cache(b, bias)
        return bias

    def right_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a parameter tensor whose projection equals ``tensor``."""
        return tensor

    def invalidate_cache(self) -> None:
        """Clear cached projection so it is recomputed on next use."""
        self._bias = None
        self._cached_input_id = None
        self._cached_input_version = None

    def _apply(self, fn):
        super()._apply(fn)
        self.invalidate_cache()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D102
        """Load parameters and clear cached projected bias."""
        result = super().load_state_dict(state_dict, strict)
        self.invalidate_cache()
        return result


class CommutingConstraint(torch.nn.Module):
    r"""Orthogonal projection onto :math:`\operatorname{Hom}_{\mathbb{G}}(\rho_{\text{in}},\rho_{\text{out}})`.

    For a dense weight :math:`\mathbf{W}\in\mathbb{R}^{D_{\text{out}}\times D_{\text{in}}}`, this module returns
    :math:`\Pi_{\mathrm{Hom}}(\mathbf{W})`, the Frobenius-orthogonal projection onto

    .. math::
        \operatorname{Hom}_{\mathbb{G}}(\rho_{\text{in}},\rho_{\text{out}})
        = \{\mathbf{A}: \rho_{\text{out}}(g)\mathbf{A}=\mathbf{A}\rho_{\text{in}}(g),\ \forall g\in\mathbb{G}\}.

    The basis and projection are handled by
    :class:`~symm_learning.representation_theory.GroupHomomorphismBasis`, using the
    :ref:`isotypic decomposition <isotypic-decomposition-example>`
    (:func:`~symm_learning.representation_theory.isotypic_decomp_rep`) blockwise.

    Args:
        in_rep (:class:`~escnn.group.Representation`): Input representation :math:`\rho_{\text{in}}` of
            size ``in_rep.size``.
        out_rep (:class:`~escnn.group.Representation`): Output representation :math:`\rho_{\text{out}}` of
            size ``out_rep.size``.
        basis_expansion (:class:`str`, optional): Strategy used to realize the basis
            (``"memory_heavy"`` or ``"isotypic_expansion"``).

    Note:
        Runtime behavior depends on mode.
        In training mode (``model.train()``), the projection is recomputed each forward pass.
        In inference mode (``model.eval()``), the projected matrix is cached for the same unchanged input tensor
        (same object identity and version counter), which is faster.
        With the cache active, as a parametrization of :class:`~torch.nn.Linear`, the forward path is equivalent to
        a symmetry-agnostic standard linear layer with a fixed projected dense weight.

    Attributes:
        homo_basis (:class:`~symm_learning.representation_theory.GroupHomomorphismBasis`): Basis generator carrying the
            :ref:`isotypic decomposition <isotypic-decomposition-example>` and block metadata for
            :math:`\operatorname{Hom}_\mathbb{G}(\rho_{\text{in}}, \rho_{\text{out}})`.
        in_rep / out_rep (:class:`~escnn.group.Representation`): Cached references to the isotypic
            versions of the supplied representations.
    """

    def __init__(self, in_rep: Representation, out_rep: Representation, basis_expansion: str = "isotypic_expansion"):
        super().__init__()
        self.homo_basis = GroupHomomorphismBasis(in_rep, out_rep, basis_expansion=basis_expansion)
        self.in_rep = self.homo_basis.in_rep
        self.out_rep = self.homo_basis.out_rep
        self.register_buffer("_weight", None, persistent=False)
        self._cached_input_id = None
        self._cached_input_version = None

    def _cache_is_valid(self, W: torch.Tensor) -> bool:
        if self._weight is None:
            return False
        if id(W) != self._cached_input_id:
            return False
        version = getattr(W, "_version", None)
        return version is not None and version == self._cached_input_version

    def _update_cache(self, W: torch.Tensor, W_proj: torch.Tensor) -> None:
        version = getattr(W, "_version", None)
        if version is None:
            self.invalidate_cache()
            return
        self._weight = W_proj
        self._cached_input_id = id(W)
        self._cached_input_version = version

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        r"""Project :math:`\mathbf{W}` onto the space of equivariant linear maps.

        Args:
            W (:class:`~torch.Tensor`): Dense matrix
                :math:`\mathbf{W}\in\mathbb{R}^{D_{\mathrm{out}}\times D_{\mathrm{in}}}`.

        Returns:
            :class:`~torch.Tensor`: Frobenius-orthogonal projection
            :math:`\Pi_{\mathrm{Hom}}(\mathbf{W})`, which satisfies
            :math:`\rho_{\mathrm{out}}(g)\Pi_{\mathrm{Hom}}(\mathbf{W})=\Pi_{\mathrm{Hom}}(\mathbf{W})\rho_{\mathrm{in}}(g)`
            for all :math:`g\in\mathbb{G}`.
        """
        if not self.training and self._cache_is_valid(W):
            return self._weight
        W_proj = self.homo_basis.orthogonal_projection(W)
        if not self.training:
            self._update_cache(W, W_proj)
        return W_proj

    def right_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a pre-image for the parametrization (identity for now)."""
        return tensor

    def invalidate_cache(self) -> None:
        """Clear cached projection so it is recomputed on next use."""
        self._weight = None
        self._cached_input_id = None
        self._cached_input_version = None

    def _apply(self, fn):
        super()._apply(fn)
        self.invalidate_cache()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D102
        """Load parameters and clear cached projected weight."""
        result = super().load_state_dict(state_dict, strict)
        self.invalidate_cache()
        return result


if __name__ == "__main__":
    import sys
    import types
    from pathlib import Path

    import escnn
    from escnn.group import CyclicGroup, Icosahedral
    from escnn.gspaces import no_base_space
    from escnn.nn import FieldType

    repo_root = Path(__file__).resolve().parents[2]
    test_dir = repo_root / "test"
    sys.path.insert(0, str(repo_root))
    test_pkg = sys.modules.get("test")
    test_paths = [str(path) for path in getattr(test_pkg, "__path__", [])] if test_pkg else []
    if str(test_dir) not in test_paths:
        test_pkg = types.ModuleType("test")
        test_pkg.__path__ = [str(test_dir)]
        sys.modules["test"] = test_pkg

    from symm_learning.nn.linear import eAffine, eLinear, impose_linear_equivariance
    from test.utils import benchmark, benchmark_eval_forward

    # G = CyclicGroup(2)
    G = Icosahedral()
    m = 2
    bias = True
    in_rep = direct_sum([G.regular_representation] * m)
    out_rep = direct_sum([G.regular_representation] * m * 2)
    eq_layer_proj_heavy = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
    impose_linear_equivariance(
        lin=eq_layer_proj_heavy, in_rep=in_rep, out_rep=out_rep, basis_expansion_scheme="memory_heavy"
    )
    eq_layer_proj_iso = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
    impose_linear_equivariance(
        lin=eq_layer_proj_iso, in_rep=in_rep, out_rep=out_rep, basis_expansion_scheme="isotypic_expansion"
    )

    # Check the orthogonal projection to Hom_G(rep, rep) works as expected
    in_type = escnn.nn.FieldType(escnn.gspaces.no_base_space(G), [G.regular_representation] * m)
    out_type = escnn.nn.FieldType(escnn.gspaces.no_base_space(G), [G.regular_representation] * m * 2)
    escnn_layer = escnn.nn.Linear(in_type, out_type, bias=bias)
    W, b = escnn_layer.expand_parameters()

    def check_projection(layer, label):  # noqa: D103
        layer.weight = W
        W_projected = layer.weight
        assert torch.allclose(W, W_projected, atol=1e-5, rtol=1e-5), f"Max err: {(W - W_projected).abs().max()}"
        check_equivariance(layer, atol=1e-5, rtol=1e-5, in_rep=in_rep, out_rep=out_rep)
        print(f"{label} projection equivariance test passed.")

        W_random = torch.randn_like(W)
        layer.weight = W_random
        check_equivariance(layer, atol=1e-5, rtol=1e-5)

        W_proj = layer.weight
        layer.weight = W_proj
        W_proj2 = layer.weight
        assert torch.allclose(W_proj, W_proj2, atol=1e-5, rtol=1e-5), f"Max err: {(W_proj - W_proj2).abs().max()}"
        print(f"{label} projection idempotence test passed.")

    check_projection(eq_layer_proj_heavy, "Memory-heavy")
    check_projection(eq_layer_proj_iso, "Isotypic-expansion")

    # ____________________________________________________________________________________
    standad_layer = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
    eq_layer_iso = eLinear(in_rep, out_rep, bias=bias, basis_expansion_scheme="isotypic_expansion")
    eq_layer_heavy = eLinear(in_rep, out_rep, bias=bias, basis_expansion_scheme="memory_heavy")
    e_affine = eAffine(in_rep, bias=bias)

    in_type = FieldType(no_base_space(G), [in_rep])
    out_type = FieldType(no_base_space(G), [out_rep])
    escnn_layer = escnn.nn.Linear(in_type, out_type, bias=bias)

    batch_size = 1024
    device = torch.device("cuda")
    standad_layer = standad_layer.to(device)
    eq_layer_iso = eq_layer_iso.to(device)
    eq_layer_heavy = eq_layer_heavy.to(device)
    eq_layer_proj_heavy = eq_layer_proj_heavy.to(device)
    eq_layer_proj_iso = eq_layer_proj_iso.to(device)
    e_affine = e_affine.to(device)
    escnn_layer = escnn_layer.to(device)
    print(f"Device: {device}")

    x = torch.randn(batch_size, in_rep.size, device=device)

    def run_forward(mod):  # noqa: D103
        return mod(in_type(x)).tensor if isinstance(mod, escnn.nn.Linear) else mod(x)

    modules_to_benchmark = [
        ("Standard", standad_layer),
        ("eLinear (iso)", eq_layer_iso),
        ("eLinear (heavy)", eq_layer_heavy),
        ("Linear Proj (iso)", eq_layer_proj_iso),
        ("Linear Proj (heavy)", eq_layer_proj_heavy),
        ("eAffine", e_affine),
        ("escnn", escnn_layer),
    ]

    results = []
    for name, module in modules_to_benchmark:

        def forward_fn(mod=module):  # noqa: D103
            return run_forward(mod)

        train_mem, non_train_mem = module_memory(module)
        gpu_alloc, gpu_peak = module_device_memory(module)
        eval_fwd_mean, eval_fwd_std = benchmark_eval_forward(module, forward_fn)
        (fwd_mean, fwd_std), (bwd_mean, bwd_std) = benchmark(module, forward_fn)
        results.append(
            {
                "name": name,
                "fwd_eval_mean": eval_fwd_mean,
                "fwd_eval_std": eval_fwd_std,
                "fwd_mean": fwd_mean,
                "fwd_std": fwd_std,
                "bwd_mean": bwd_mean,
                "bwd_std": bwd_std,
                "total_time": fwd_mean + bwd_mean,
                "train_mem": train_mem,
                "non_train_mem": non_train_mem,
                "gpu_mem": gpu_alloc,
                "gpu_peak": gpu_peak,
            }
        )

    name_width = 20
    header = (
        f"{'Layer':<{name_width}} {'Forward eval (ms)':>18} {'Forward (ms)':>18} {'Backward (ms)':>18} "
        f"{'Total (ms)':>15} "
        f"{'Trainable MB':>15} {'Non-train MB':>15} {'Total MB':>12} {'GPU Alloc MB':>15} {'GPU Peak MB':>15}"
    )
    separator = "-" * len(header)
    print(f"\nBenchmark results per {batch_size}-sample batch")
    print(separator)
    print(header)
    print(separator)
    for res in results:
        fwd_eval_str = f"{res['fwd_eval_mean']:.3f} ± {res['fwd_eval_std']:.3f}"
        fwd_str = f"{res['fwd_mean']:.3f} ± {res['fwd_std']:.3f}"
        bwd_str = f"{res['bwd_mean']:.3f} ± {res['bwd_std']:.3f}"
        total_mb = res["train_mem"] + res["non_train_mem"]
        gpu_alloc_mb = bytes_to_mb(res["gpu_mem"])
        gpu_peak_mb = bytes_to_mb(res["gpu_peak"])
        print(
            f"{res['name']:<{name_width}} {fwd_eval_str:>18} {fwd_str:>18} {bwd_str:>18} "
            f"{res['total_time']:>15.3f} {bytes_to_mb(res['train_mem']):>15.3f} "
            f"{bytes_to_mb(res['non_train_mem']):>15.3f} {bytes_to_mb(total_mb):>12.3f} "
            f"{gpu_alloc_mb:>15.3f} {gpu_peak_mb:>15.3f}"
        )
    print(separator)
