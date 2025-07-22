from __future__ import annotations

import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

import symm_learning
import symm_learning.stats
from symm_learning.nn import eAffine


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
        print("mean", mean_batch)
        print("var", var_batch)

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
        import numpy as np

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
            bn.weight.data = self.affine_transform.expand_scale().clone()
            bn.bias.data = self.affine_transform.expand_bias().clone()

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


class DataNorm(torch.nn.Module):
    r"""Applies data normalization to a 2D or 3D tensor.

    Standardizes the data to have zero mean and optionally unit variance.

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}  \text{ or }  y = x - \\mu

    Args:
        num_features (int): The number of features in the input tensor.
        mean (torch.Tensor, optional): Fixed mean. If provided with running_stats=True,
            used as initial value. Defaults to None.
        std (torch.Tensor, optional): Fixed std. If provided with running_stats=True,
            used as initial value. Defaults to None.
        eps (float, optional): A value added to the denominator for numerical
            stability. Defaults to 1e-6.
        only_centering (bool, optional): If True, only center data (don't scale).
            Defaults to False.
        compute_cov (bool, optional): Whether to compute covariance matrix.
            Defaults to False.
        running_stats (bool, optional): Whether to use running statistics.
            Defaults to True.
        momentum (float, optional): Momentum for exponential moving average.
            If None, uses cumulative averaging. Defaults to None.
    """

    def __init__(
        self,
        num_features: int,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        eps: float = 1e-6,
        only_centering: bool = False,
        compute_cov: bool = False,
        running_stats: bool = True,
        momentum: float = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.only_centering = only_centering
        self.compute_cov = compute_cov
        self.running_stats = running_stats
        self.momentum = momentum

        # Initialize statistics
        init_mean = mean if mean is not None else torch.zeros(num_features)
        init_std = std if std is not None else torch.ones(num_features)

        if running_stats:
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
            self.register_buffer("running_mean", init_mean.clone())
            self.register_buffer("running_std", init_std.clone())
            if compute_cov:
                self.register_buffer("running_cov", torch.eye(num_features))
        else:
            # Fixed stats mode - these act as our "fixed" values
            self.register_buffer("_mean", init_mean)
            self.register_buffer("_std", init_std)  # For batch covariance tracking
        self._last_cov = None

    def _compute_batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batch statistics (mean and std). Can be overridden for equivariant versions."""
        dims = [0] + ([2] if x.ndim > 2 else [])
        batch_mean = x.mean(dim=dims)
        batch_var = x.var(dim=dims, unbiased=False)
        batch_std = torch.sqrt(batch_var)
        return batch_mean, batch_std

    def _compute_batch_cov(self, x: torch.Tensor) -> torch.Tensor:
        """Compute batch covariance. Can be overridden for equivariant versions."""
        x_flat = x.permute(0, 2, 1).reshape(-1, self.num_features) if x.ndim == 3 else x
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        batch_cov = torch.mm(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
        return batch_cov

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalization to the input tensor."""
        assert x.shape[1] == self.num_features

        # Update running statistics if needed
        if self.running_stats and self.training:
            # Compute batch statistics
            batch_mean, batch_std = self._compute_batch_stats(x)

            # Update running statistics
            if self.num_batches_tracked == 0 and self.momentum is None:
                self.running_mean.copy_(batch_mean)
                self.running_std.copy_(batch_std)
            else:
                momentum = self.momentum if self.momentum is not None else 1.0 / (self.num_batches_tracked.item() + 1)
                self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_std.mul_(1 - momentum).add_(batch_std, alpha=momentum)

            self.num_batches_tracked += 1

        # Compute covariance if needed
        if self.compute_cov:
            batch_cov = self._compute_batch_cov(x)
            self._last_cov = batch_cov

            if self.running_stats and hasattr(self, "running_cov") and self.training:
                if self.num_batches_tracked == 1 and self.momentum is None:
                    self.running_cov.copy_(batch_cov)
                else:
                    momentum = self.momentum if self.momentum is not None else 1.0 / self.num_batches_tracked.item()
                    self.running_cov.mul_(1 - momentum).add_(batch_cov, alpha=momentum)

        # Get current statistics
        mean = self.mean
        std = self.std if not self.only_centering else torch.ones_like(self.mean)

        # Reshape for broadcasting
        if x.ndim == 3:
            mean = mean.view(1, self.num_features, 1)
            std = std.view(1, self.num_features, 1)
        else:
            mean = mean.view(1, self.num_features)
            std = std.view(1, self.num_features)

        # Apply normalization
        return x - mean if self.only_centering else (x - mean) / (std + self.eps)

    @property
    def mean(self) -> torch.Tensor:
        """Return the current mean estimate."""
        if self.running_stats:
            return self.running_mean
        else:
            return self._mean

    @property
    def std(self) -> torch.Tensor:
        """Return the current std estimate."""
        if self.running_stats:
            return self.running_std
        else:
            return self._std

    @property
    def cov(self) -> torch.Tensor:
        """Return the current covariance matrix estimate."""
        if not self.compute_cov:
            raise RuntimeError("Covariance computation is disabled. Set compute_cov=True to enable.")

        if self.running_stats and hasattr(self, "running_cov"):
            return self.running_cov
        elif self._last_cov is not None:
            return self._last_cov
        else:
            raise RuntimeError("No covariance available. Ensure at least one forward pass has been completed.")

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"{self.num_features}, eps={self.eps}, only_centering={self.only_centering}, "
            f"compute_cov={self.compute_cov}, running_stats={self.running_stats}, "
            f"momentum={self.momentum}"
        )


class eDataNorm(DataNorm, EquivariantModule):
    r"""Equivariant version of DataNorm using symmetry-aware statistics.

    Applies data normalization to a 2D or 3D equivariant tensor using the same
    API as DataNorm but with equivariant statistics from symm_learning.stats.

    Args:
        in_type (FieldType): The input field type.
        mean (torch.Tensor, optional): Fixed mean. If provided with running_stats=True,
            used as initial value. Defaults to None.
        std (torch.Tensor, optional): Fixed std. If provided with running_stats=True,
            used as initial value. Defaults to None.
        eps (float, optional): A value added to the denominator for numerical
            stability. Defaults to 1e-6.
        only_centering (bool, optional): If True, only center data (don't scale).
            Defaults to False.
        compute_cov (bool, optional): Whether to compute covariance matrix.
            Defaults to False.
        running_stats (bool, optional): Whether to use running statistics.
            Defaults to True.
        momentum (float, optional): Momentum for exponential moving average.
            If None, uses cumulative averaging. Defaults to None.
    """

    def __init__(
        self,
        in_type: FieldType,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        eps: float = 1e-6,
        only_centering: bool = False,
        compute_cov: bool = False,
        running_stats: bool = True,
        momentum: float = None,
    ):
        # Initialize DataNorm with the field type size
        super().__init__(
            num_features=in_type.size,
            mean=mean,
            std=std,
            eps=eps,
            only_centering=only_centering,
            compute_cov=compute_cov,
            running_stats=running_stats,
            momentum=momentum,
        )

        # Store EquivariantModule-specific attributes first
        self.in_type = in_type
        self.out_type = in_type
        self._rep_x = in_type.representation

    def _compute_batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivariant batch statistics using symm_learning.stats."""
        batch_var, batch_mean = symm_learning.stats.var_mean(x, rep_x=self._rep_x)
        batch_std = torch.sqrt(batch_var)
        return batch_mean, batch_std

    def _compute_batch_cov(self, x: torch.Tensor) -> torch.Tensor:
        """Compute equivariant batch covariance using symm_learning.stats."""
        return symm_learning.stats.cov(x=x, y=x, rep_x=self._rep_x, rep_y=self._rep_x)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Apply equivariant normalization to the input GeometricTensor."""
        assert x.type == self.in_type, f"Input type {x.type} does not match expected type {self.in_type}."

        # Apply DataNorm forward to the tensor data
        normalized_tensor = super().forward(x.tensor)

        # Return as GeometricTensor
        return self.out_type(normalized_tensor)

    def evaluate_output_shape(self, input_shape):
        """Return the same shape as input for EquivariantModule compatibility."""
        return input_shape

    def check_equivariance(self, atol=1e-5, rtol=1e-5):
        """Check the equivariance of the normalization layer."""
        was_training = self.training
        batch_size = 50

        self.train()

        # Process a few batches to get some running statistics
        for _ in range(3):
            x = torch.randn(batch_size, self.in_type.size)
            x_geom = self.in_type(x)
            _ = self(x_geom)

        self.eval()

        # Test equivariance
        x = torch.randn(batch_size, self.in_type.size)
        x_geom = self.in_type(x)

        for _ in range(5):
            g = self.in_type.representation.group.sample()
            if g == self.in_type.representation.group.identity:
                continue

            gx_geom = x_geom.transform(g)

            y = self(x_geom)
            gy = self(gx_geom)
            gy_expected = y.transform(g)

            assert torch.allclose(gy.tensor, gy_expected.tensor, atol=atol, rtol=rtol), (
                f"Equivariance check failed for group element {g}"
            )

        self.train(was_training)

    def export(self) -> DataNorm:
        """Export to a standard DataNorm layer."""
        exported = DataNorm(
            num_features=self.num_features,
            eps=self.eps,
            only_centering=self.only_centering,
            compute_cov=self.compute_cov,
            running_stats=self.running_stats,
            momentum=self.momentum,
        )

        # Transfer state
        if self.running_stats:
            exported.running_mean.data = self.running_mean.clone()
            exported.running_std.data = self.running_std.clone()
            exported.num_batches_tracked.data = self.num_batches_tracked.clone()
            if self.compute_cov and hasattr(self, "running_cov"):
                exported.running_cov.data = self.running_cov.clone()
        else:
            exported._mean.data = self._mean.clone()
            exported._std.data = self._std.clone()

        exported._last_cov = self._last_cov

        # Set to same training mode as original
        exported.train(self.training)

        return exported
