from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from torch import batch_norm

import symm_learning
import symm_learning.stats
from symm_learning.linalg import irrep_radii
from symm_learning.nn.linear import eAffine
from symm_learning.representation_theory import direct_sum, isotypic_decomp_rep


class eRMSNorm(torch.nn.Module):
    r"""Equivariant Root-Mean-Square Normalization.

    This layer mirrors :class:`torch.nn.RMSNorm` while keeping the affine step symmetry-preserving.
    For an input :math:`x \in \mathbb{R}^{D}` with :math:`D = \texttt{in_rep.size}`, it shares a
    single normalization factor across all channels:

    .. math::
        \operatorname{rms}(x) = \sqrt{\tfrac{1}{D}\langle x, x\rangle + \varepsilon}, \qquad
        y = \frac{x}{\operatorname{rms}(x)}.

    When ``equiv_affine=True`` a learnable :class:`~symm_learning.nn.linear.eAffine` is applied
    after normalization, providing per-irrep scales (and optional invariant biases) that commute
    with the group action and therefore preserve equivariance.

    Args:
        in_rep (escnn.group.Representation): Description of the feature space.
        eps (float): Numerical stabilizer added inside the RMS computation.
        equiv_affine (bool): If ``True``, apply a symmetry-preserving :class:`eAffine` after normalization.
        bias (bool): Include invariant biases in the affine term (only used if ``equiv_affine``).
        device, dtype: Optional tensor factory kwargs passed to the affine parameters.
        init_scheme (Literal["identity", "random"] | None): Initialization scheme forwarded to
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

    Given an input :math:`x \in \mathbb{R}^{D}`, we first move to the irrep-spectral basis
    :math:`\hat{x} = Q^{-1}x`, compute one variance scalar per irreducible block,
    and normalize each block uniformly:

    .. math::
        \hat{y} = \frac{\hat{x}}{\sqrt{\sigma^{2} + \varepsilon}}, \qquad
        y = Q\hat{y}.

    When ``equiv_affine=True`` the learnable affine step is performed directly in the spectral basis
    using the per-irrep scale/bias provided by :class:`~symm_learning.nn.linear.eAffine`.

    Args:
        in_rep: (:class:`escnn.group.Representation`) description of the feature space.
        eps: numerical stabilizer added to each variance.
        equiv_affine: if ``True``, applies an :class:`eAffine` in spectral space.
        bias: whether the affine term includes invariant biases (only used if ``equiv_affine``).
        device, dtype: optional tensor factory kwargs.

    Note:
        This layer appears to generate numerical instability when used in equivariant transformer blocks.
        Use eRMSNorm instead in such cases.
    """

    r"""Symmetry-preserving LayerNorm:

    .. math:: y = Q ( (Q^{-1} x) \odot \alpha + \beta )
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


class eBatchNorm1d(EquivariantModule):
    r"""Applies Batch Normalization over a 2D or 3D symmetric input :class:`escnn.nn.GeometricTensor`.

    Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated **using symmetry-aware estimates** (see
    :func:`~symm_learning.stats.var_mean`) over the mini-batches and :math:`\gamma` and :math:`\beta` are
    the scale and bias vectors of a :class:`eAffine`, which ensures that the affine transformation is
    symmetry-preserving. By default, the elements of :math:`\gamma` are initialized to 1 and the elements
    of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        If input tensor is of shape :math:`(N, C, L)`, the implementation of this module
        computes a unique mean and variance for each feature or channel :math:`C` and applies it to
        all the elements in the sequence length :math:`L`.

    Args:
        input_type: the :class:`escnn.nn.FieldType` of the input geometric tensor.
            The output type is the same as the input type.
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`, where :math:`N` is the batch size,
          :math:`C` is the number of features or channels, and :math:`L` is the sequence length
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        in_type: FieldType,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._rep_x = in_type.representation

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(in_type.size))
            self.register_buffer("running_var", torch.ones(in_type.size))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        if self.affine:
            self.affine_transform = eAffine(
                in_type=in_type,
                bias=True,
            )

    def forward(self, x: GeometricTensor):  # noqa: D102
        assert x.type == self.in_type, "Input type does not match the expected input type."

        var_batch, mean_batch = symm_learning.stats.var_mean(x.tensor, rep_x=self._rep_x)

        if self.track_running_stats:
            if self.training:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_batch
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
                self.num_batches_tracked += 1
            mean, var = self.running_mean, self.running_var
        else:
            mean, var = mean_batch, var_batch

        mean, var = mean[..., None], var[..., None] if x.tensor.ndim == 3 else (mean, var)
        y = (x.tensor - mean) / torch.sqrt(var + self.eps)

        y = self.affine_transform(self.out_type(y)) if self.affine else self.out_type(y)
        return y

    def evaluate_output_shape(self, input_shape):  # noqa: D102
        return input_shape

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in type: {self.in_type}, affine: {self.affine}, track_running_stats: {self.track_running_stats}"
            f" eps: {self.eps}, momentum: {self.momentum}  "
        )

    def check_equivariance(self, atol=1e-5, rtol=1e-5):
        """Check the equivariance of the convolution layer."""
        was_training = self.training
        time = 1
        batch_size = 50

        self.train()
        # Compute some empirical statistics
        for _ in range(5):
            x = torch.randn(batch_size, self.in_type.size, time)
            x = self.in_type(x)
            _ = self(x)

        self.eval()

        x_batch = torch.randn(batch_size, self.in_type.size, time)
        x_batch = self.in_type(x_batch)

        for i in range(10):
            g = self.in_type.representation.group.sample()
            if g == self.in_type.representation.group.identity:
                i -= 1
                continue
            gx_batch = x_batch.transform(g)

            var, mean = symm_learning.stats.var_mean(x_batch.tensor, rep_x=self.in_type.representation)
            g_var, g_mean = symm_learning.stats.var_mean(gx_batch.tensor, rep_x=self.in_type.representation)

            assert torch.allclose(mean, g_mean, atol=1e-4, rtol=1e-4), f"Mean {mean} != {g_mean}"
            assert torch.allclose(var, g_var, atol=1e-4, rtol=1e-4), f"Var {var} != {g_var}"

            y = self(x_batch)
            g_y = self(gx_batch)
            g_y_gt = y.transform(g)

            assert torch.allclose(g_y.tensor, g_y_gt.tensor, atol=1e-5, rtol=1e-5), (
                f"Output {g_y.tensor} does not match the expected output {g_y_gt.tensor} for group element {g}"
            )

        self.train(was_training)

        return None

    def export(self) -> torch.nn.BatchNorm1d:
        """Export the layer to a standard PyTorch BatchNorm1d layer."""
        bn = torch.nn.BatchNorm1d(
            num_features=self.in_type.size,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )

        if self.affine:
            bn.weight.data = self.affine_transform.scale.clone()
            bn.bias.data = self.affine_transform.bias.clone()

        if self.track_running_stats:
            bn.running_mean.data = self.running_mean.clone()
            bn.running_var.data = self.running_var.clone()
            bn.num_batches_tracked.data = self.num_batches_tracked.clone()
        else:
            bn.running_mean = None
            bn.running_var = None

        bn.train(False)
        bn.eval()
        return bn


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
