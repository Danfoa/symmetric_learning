from __future__ import annotations

import logging
from typing import Literal

import torch
from escnn.group import Representation

import symm_learning.stats
from symm_learning.linalg import irrep_radii
from symm_learning.nn.linear import eAffine
from symm_learning.representation_theory import direct_sum


class eRMSNorm(torch.nn.Module):
    r"""Root-mean-square normalization with :math:`\mathbb{G}`-equivariant affine map.

    For :math:`\mathbf{x}\in\mathcal{X}` with :math:`D=\dim(\rho_{\mathcal{X}})`, define

    .. math::
        \hat{\mathbf{x}} = \frac{\mathbf{x}}{\sqrt{\frac{1}{D} \|\mathbf{x}\|^2 + \varepsilon}}, \qquad
        \mathbf{y} = \alpha \odot \hat{\mathbf{x}} + \mathbf{\beta}

    where the second step is implemented by :class:`~symm_learning.nn.linear.eAffine`
    (per-irrep scaling and optional invariant bias).

    Equivariance:

    .. math::
        \operatorname{eRMSNorm}(\rho_{\mathcal{X}}(g)\mathbf{x})
        = \rho_{\mathcal{X}}(g)\operatorname{eRMSNorm}(\mathbf{x}),
        \quad \forall g\in\mathbb{G},

    because the RMS factor is invariant and :class:`eAffine` commutes with :math:`\rho_{\mathcal{X}}`.

    Args:
        in_rep (:class:`~escnn.group.Representation`): Description of the feature space :math:`\rho_{\text{in}}`.
        eps (:class:`float`): Numerical stabilizer added inside the RMS computation.
        equiv_affine (:class:`bool`): If ``True``, apply a symmetry-preserving :class:`eAffine` after normalization.
        device, dtype: Optional tensor factory kwargs passed to the affine parameters.
        init_scheme (:class:`typing.Literal`["identity", "random"] | :class:`None`): Initialization scheme forwarded to
            :meth:`eAffine.reset_parameters`. Set to ``None`` to skip initialization (useful when loading checkpoints).

    Shape:
        - Input: ``(..., in_rep.size)``
        - Output: same shape

    Note:
        The normalization factor is a single scalar per sample, so the operation commutes with any
        matrix representing the group action defined by ``in_rep``.
    """

    def __init__(
        self,
        in_rep: Representation,
        eps: float = 1e-6,
        equiv_affine: bool = True,
        device=None,
        dtype=None,
        init_scheme: Literal["identity", "random"] | None = "identity",
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_rep, self.out_rep = in_rep, in_rep
        if equiv_affine:
            self.affine = eAffine(in_rep, bias=False).to(**factory_kwargs)
            if init_scheme is not None:
                self.affine.reset_parameters(init_scheme)
        self.eps = eps
        self.normalized_shape = (in_rep.size,)

        if init_scheme is not None:
            self.reset_parameters(init_scheme)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize by a single RMS scalar and (optionally) apply equivariant affine.

        Args:
            input: Tensor shaped ``(..., in_rep.size)``.

        Returns:
            Tensor with identical shape, RMS-normalized and possibly transformed by :class:`eAffine`.
        """
        assert input.shape[-1] == self.in_rep.size, f"Expected (...,{self.in_rep.size}), got {input.shape}"
        rms_input = torch.sqrt(self.eps + torch.mean(input.pow(2), dim=-1, keepdim=True))
        normalized = input / rms_input
        if hasattr(self, "affine"):
            normalized = self.affine(normalized)
        return normalized

    def reset_parameters(self, scheme: Literal["identity", "random"] = "identity") -> None:
        """(Re)initialize the optional affine transform using the provided scheme."""
        if hasattr(self, "affine"):
            self.affine.reset_parameters(scheme)


class eLayerNorm(torch.nn.Module):
    r"""Equivariant Layer Normalization.

    Given :math:`\mathbf{x}\in\mathcal{X}`, we first move to the irrep-spectral basis
    :math:`\hat{\mathbf{x}} = \mathbf{Q}^{-1}\mathbf{x}`, compute one variance scalar per irreducible block
    via :func:`~symm_learning.linalg.irrep_radii`,
    and normalize each block uniformly:

    .. math::
        \hat{\mathbf{y}} = \frac{\hat{\mathbf{x}}}{\sqrt{\mathbf{\sigma}^{2} + \varepsilon}}, \qquad
        \mathbf{y} = Q\hat{\mathbf{y}}.

    The layer is equivariant:

    .. math::
        \rho_{\text{in}}(g) \mathbf{y} = \text{LayerNorm}(\rho_{\text{in}}(g) \mathbf{x})

    since the statistics are computed per irreducible subspace (which are preserved by the group action).

    When ``equiv_affine=True`` the learnable affine step is performed directly in the spectral basis
    using the per-irrep scale/bias provided by :class:`~symm_learning.nn.linear.eAffine`.

    Args:
        in_rep (:class:`~escnn.group.Representation`): description of the feature space :math:`\rho_{\text{in}}`.
        eps (:class:`float`): numerical stabilizer added to each variance.
        equiv_affine (:class:`bool`): if ``True``, applies an :class:`eAffine` in spectral space.
        bias (:class:`bool`): whether the affine term includes invariant biases (only used if ``equiv_affine``).
        device, dtype: optional tensor factory kwargs.

    Note:
        This layer appears to generate numerical instability when used in equivariant transformer blocks.
        Use eRMSNorm instead in such cases.
    """

    r"""Symmetry-preserving LayerNorm:

    .. math:: \mathbf{y} = Q ( (Q^{-1} \mathbf{x}) \odot \alpha + \mathbf{\beta} )
    """

    def __init__(
        self,
        in_rep: Representation,
        eps: float = 1e-6,
        equiv_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        init_scheme: Literal["identity", "random"] | None = "identity",
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_rep, self.out_rep = in_rep, in_rep

        # We require to transition to/from the irrep-spectral basis to compute the normalization and affine transform
        self.register_buffer("Q", torch.tensor(self.in_rep.change_of_basis, dtype=torch.get_default_dtype()))
        self.register_buffer("Q_inv", torch.tensor(self.in_rep.change_of_basis_inv, dtype=torch.get_default_dtype()))

        # Only works for (..., in_rep.size) inputs with normalization over the last dimension
        self.normalized_shape = (in_rep.size,)
        self.eps = eps
        self.equiv_affine = equiv_affine
        dims = torch.tensor(
            [self.in_rep.group.irrep(*irrep_id).size for irrep_id in self.in_rep.irreps],
            dtype=torch.long,
        )
        self.register_buffer("irrep_dims", dims)
        self.register_buffer("irrep_indices", torch.repeat_interleave(torch.arange(len(dims), dtype=torch.long), dims))
        if self.equiv_affine:
            self.affine = eAffine(in_rep, bias=bias).to(**factory_kwargs)

        self.reset_parameters(init_scheme)

    def reset_parameters(self, scheme: Literal["identity", "random"] = "identity") -> None:  # noqa: D102
        if self.equiv_affine:
            self.affine.reset_parameters(scheme)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Normalize per irreducible block and (optionally) apply the spectral affine transform."""
        assert input.shape[-1] == self.in_rep.size, f"Expected (...,{self.in_rep.size}), got {input.shape}"

        radii = irrep_radii(input, rep=self.in_rep)  # (..., num_irreps)
        dims = self.irrep_dims.to(radii.device, radii.dtype)
        var_irreps = radii.pow(2) / dims
        var_broadcasted = var_irreps[..., self.irrep_indices.to(var_irreps.device)]

        x_spec = torch.einsum("ij,...j->...i", self.Q_inv, input)
        x_spec = x_spec / torch.sqrt(var_broadcasted + self.eps)

        if self.equiv_affine:
            spectral_scale, spectral_bias = self.affine.broadcast_spectral_scale_and_bias(
                self.affine.scale_dof, self.affine.bias_dof
            )
            x_spec = x_spec * spectral_scale.view(*([1] * (x_spec.ndim - 1)), -1)
            if spectral_bias is not None:
                x_spec = x_spec + spectral_bias.view(*([1] * (x_spec.ndim - 1)), -1)

        normalized = torch.einsum("ij,...j->...i", self.Q, x_spec)
        return normalized

    def extra_repr(self) -> str:  # noqa: D102
        return "{normalized_shape}, eps={eps}, affine={equiv_affine}".format(**self.__dict__)


class eBatchNorm1d(torch.nn.Module):
    r"""Symmetry-aware Batch Normalization over the representation dimension.

    The mean and variance are computed with :func:`~symm_learning.stats.var_mean`,
    enforcing that each irreducible subspace shares a single variance scalar. The
    optional affine parameters are implemented via :class:`eAffine` to preserve
    equivariance.

    The layer satisfies:

    .. math::
        \rho_{\text{in}}(g) \mathbf{y} = \text{BatchNorm}(\rho_{\text{in}}(g) \mathbf{x})

    Args:
        in_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{in}}` describing the feature
            space.
        eps: Numerical stabilizer added to the variance.
        momentum: Momentum for exponential moving averages.
        affine: If ``True``, apply a symmetry-preserving affine transform.
        track_running_stats: If ``True``, keep running mean/variance buffers.

    Shape:
        - Input: ``(..., in_rep.size)``
        - Output: same shape
    """

    def __init__(
        self,
        in_rep: Representation,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        if not isinstance(in_rep, Representation):
            raise TypeError(f"in_rep must be a Representation, got {type(in_rep)}")
        self.in_rep = in_rep
        self.out_rep = in_rep
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._rep_x = in_rep

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(in_rep.size))
            self.register_buffer("running_var", torch.ones(in_rep.size))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        if self.affine:
            self.affine_transform = eAffine(in_rep, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        assert x.shape[-1] == self.in_rep.size, f"Expected (..., {self.in_rep.size}), got {x.shape}"

        x_flat = x.reshape(-1, self.in_rep.size)
        var_batch, mean_batch = symm_learning.stats.var_mean(x_flat, rep_x=self._rep_x)

        if self.track_running_stats:
            if self.training:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(mean_batch, alpha=self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(var_batch, alpha=self.momentum)
                    self.num_batches_tracked += 1
            mean, var = self.running_mean, self.running_var
        else:
            mean, var = mean_batch, var_batch

        view_shape = [1] * (x.ndim - 1) + [-1]
        mean = mean.view(*view_shape)
        var = var.view(*view_shape)
        y = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            y = self.affine_transform(y)
        return y

    def evaluate_output_shape(self, input_shape):  # noqa: D102
        return input_shape

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in_rep: {self.in_rep}, affine: {self.affine}, track_running_stats: {self.track_running_stats} "
            f"eps: {self.eps}, momentum: {self.momentum}"
        )

    def check_equivariance(self, atol=1e-5, rtol=1e-5):
        """Check equivariance using random group elements."""
        was_training = self.training
        batch_size = 50

        self.train()
        # Warm up running statistics
        for _ in range(5):
            x = torch.randn(batch_size, self.in_rep.size)
            _ = self(x)

        self.eval()

        x_batch = torch.randn(batch_size, self.in_rep.size)
        G = self.in_rep.group
        for _ in range(10):
            g = G.sample()
            if g == G.identity:
                continue
            rho_g = torch.tensor(self.in_rep(g), dtype=x_batch.dtype, device=x_batch.device)
            gx_batch = x_batch @ rho_g.T

            var, mean = symm_learning.stats.var_mean(x_batch, rep_x=self.in_rep)
            g_var, g_mean = symm_learning.stats.var_mean(gx_batch, rep_x=self.in_rep)

            assert torch.allclose(mean, g_mean, atol=1e-4, rtol=1e-4), f"Mean {mean} != {g_mean}"
            assert torch.allclose(var, g_var, atol=1e-4, rtol=1e-4), f"Var {var} != {g_var}"

            y = self(x_batch)
            g_y = self(gx_batch)
            g_y_gt = y @ rho_g.T

            assert torch.allclose(g_y, g_y_gt, atol=1e-5, rtol=1e-5), (
                f"Output {g_y} does not match the expected output {g_y_gt} for group element {g}"
            )

        self.train(was_training)

        return None


if __name__ == "__main__":
    import sys
    import types
    from pathlib import Path

    from escnn.group import CyclicGroup, Icosahedral

    repo_root = Path(__file__).resolve().parents[2]
    test_dir = repo_root / "test"
    sys.path.insert(0, str(repo_root))
    test_pkg = sys.modules.get("test")
    test_paths = [str(path) for path in getattr(test_pkg, "__path__", [])] if test_pkg else []
    if str(test_dir) not in test_paths:
        test_pkg = types.ModuleType("test")
        test_pkg.__path__ = [str(test_dir)]
        sys.modules["test"] = test_pkg

    from symm_learning.utils import bytes_to_mb, module_device_memory, module_memory
    from test.utils import benchmark, benchmark_eval_forward

    # G = CyclicGroup(2)
    G = Icosahedral()
    m = 2
    eps = 1e-6
    in_rep = direct_sum([G.regular_representation] * m)

    rms_norm = torch.nn.RMSNorm(in_rep.size, eps=eps, elementwise_affine=True)
    eq_rms_norm = eRMSNorm(in_rep, eps=eps, equiv_affine=True)

    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rms_norm = rms_norm.to(device)
    eq_rms_norm = eq_rms_norm.to(device)
    print(f"Device: {device}")

    x = torch.randn(batch_size, in_rep.size, device=device)

    def run_forward(mod):  # noqa: D103
        return mod(x)

    modules_to_benchmark = [
        ("RMSNorm", rms_norm),
        ("eRMSNorm", eq_rms_norm),
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
        fwd_eval_str = f"{res['fwd_eval_mean']:.3f} +/- {res['fwd_eval_std']:.3f}"
        fwd_str = f"{res['fwd_mean']:.3f} +/- {res['fwd_std']:.3f}"
        bwd_str = f"{res['bwd_mean']:.3f} +/- {res['bwd_std']:.3f}"
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
