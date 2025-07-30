from __future__ import annotations

import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from torch import batch_norm

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

    This module standardizes input data by centering (subtracting the mean) and optionally
    scaling (dividing by the standard deviation). The module supports multiple modes of
    operation controlled by its configuration parameters.

    **Mathematical Formulation:**

    The normalization is applied element-wise as:

    .. math::
        y = \begin{cases}
            x - \mu & \text{if } \texttt{only centering} = \text{True} \\
            \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} & \text{otherwise}
        \end{cases}

    where :math:`\mu` is the mean, :math:`\sigma^2` is the variance, and :math:`\epsilon`
    is a small constant for numerical stability.

    **Mode of Operation:**

    This layer features a non-standard behavior during training. Unlike typical
    normalization layers (e.g., ``torch.nn.BatchNorm1d``) that normalize using
    batch statistics, this layer normalizes the data using the running statistics
    that have been updated with the current batch's information.

    - **During training**:
      1. Batch statistics (:math:`\mu_{\text{batch}}`, :math:`\sigma^2_{\text{batch}}`) are computed from the input.
      2. Running statistics (:math:`\mu_{\text{run}}`, :math:`\sigma^2_{\text{run}}`) are updated using exponential
        moving average:
         :math:`\text{running stat} = (1-\alpha) \cdot \text{running stat} + \alpha \cdot \text{batch stat}`
      3. The input data is then normalized using these newly updated running statistics.
      This allows the loss to be dependent on the running statistics, with gradients flowing
      back through the batch statistics component of the update, but not into the historical
      state of the running statistics from previous steps.

    - **During evaluation**: Uses the final stored running statistics for normalization.

    **Special case**: When ``momentum=1.0``, the layer effectively uses batch statistics for 
    normalization, becoming equivalent to a `torch.nn.BatchNorm1d` layer with `track_running_stats=False`.

    Args:
        num_features (int): Number of features or channels in the input tensor.
        eps (float, optional): Small constant added to the denominator for numerical
            stability. Only used when ``only_centering=False``. Default: ``1e-6``.
        only_centering (bool, optional): If ``True``, only centers the data (subtracts mean)
            without scaling by standard deviation. Default: ``False``.
        compute_cov (bool, optional): If ``True``, computes and tracks the full covariance
            matrix in addition to mean and variance. Accessible via the ``cov`` property.
            Default: ``False``.
        momentum (float, optional): Momentum factor for exponential moving average of
            running statistics. Must be greater than 0. Setting to ``1.0`` effectively 
            uses only batch statistics. Default: ``1.0``.

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where:
          - :math:`N` is the batch size
          - :math:`C` is the number of features (must equal ``num_features``)
          - :math:`L` is the sequence length (optional, for 3D inputs)
        - Output: Same shape as input

    Attributes:
        running_mean (torch.Tensor): Running average of input means. Shape: ``(num_features,)``.
        running_var (torch.Tensor): Running average of input variances. Shape: ``(num_features,)``.
        running_cov (torch.Tensor): Running average of input covariance matrix.
            Shape: ``(num_features, num_features)``. Only available when ``compute_cov=True``.
        num_batches_tracked (torch.Tensor): Number of batches processed during training.

    Note:
        When using 3D inputs :math:`(N, C, L)`, statistics are computed over both the batch
        dimension :math:`N` and sequence dimension :math:`L`, treating each feature channel
        independently.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        only_centering: bool = False,
        compute_cov: bool = False,
        momentum: float = 1.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.only_centering = only_centering
        self.compute_cov = compute_cov

        if momentum <= 0.0:
            raise ValueError(f"momentum must be greater than 0, got {momentum}")
        self.momentum = momentum

        # Initialize running statistics buffers
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        if compute_cov:
            self.register_buffer("running_cov", torch.eye(num_features))
        self._last_cov = None

    def _compute_batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batch statistics (mean and var). Can be overridden for equivariant versions."""
        dims = [0] + ([2] if x.ndim > 2 else [])
        batch_mean = x.mean(dim=dims)
        batch_var = x.var(dim=dims, unbiased=False)
        return batch_mean, batch_var

    def _compute_batch_cov(self, x: torch.Tensor) -> torch.Tensor:
        """Compute batch covariance. Can be overridden for equivariant versions."""
        x_flat = x.permute(0, 2, 1).reshape(-1, self.num_features) if x.ndim == 3 else x
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        batch_cov = torch.mm(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
        return batch_cov

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalization to the input tensor."""
        assert x.shape[1] == self.num_features

        # --- Determine statistics for normalization ---
        if self.training:
            # Training mode: Use running stats for normalization, but in a way
            # that gradients flow back through the current batch's statistics.
            batch_mean, batch_var = self._compute_batch_stats(x)
            # 2. Calculate the new running statistics for the CURRENT step.
            if self.num_batches_tracked == 0:
                # For the very first batch, initialize running stats with batch stats
                self.running_mean = batch_mean
                self.running_var = batch_var
            else:
                # Detach prev running stats to prevent gradients from flowing into the previous iteration's state.
                self.running_mean = (1 - self.momentum) * self.running_mean.detach() + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var.detach() + self.momentum * batch_var
            self.num_batches_tracked += 1

        # --- Covariance Computation (if enabled) ---
        if self.compute_cov:
            batch_cov = self._compute_batch_cov(x)
            if self.training:
                if self.num_batches_tracked == 0:
                    self.running_cov = batch_cov
                else:
                    self.running_cov = (1 - self.momentum) * self.running_cov.detach() + self.momentum * batch_cov

        # --- Apply Normalization ---
        mean = self.running_mean
        std = torch.sqrt(self.running_var + self.eps) if not self.only_centering else torch.ones_like(self.running_mean)

        # Reshape for broadcasting
        if x.ndim == 3:
            mean = mean.view(1, self.num_features, 1)
            std = std.view(1, self.num_features, 1)
        else:
            mean = mean.view(1, self.num_features)
            std = std.view(1, self.num_features)

        return (x - mean) / std

    @property
    def mean(self) -> torch.Tensor:
        """Return the current mean estimate."""
        return self.running_mean

    @property
    def var(self) -> torch.Tensor:
        """Return the current variance estimate."""
        return self.running_var

    @property
    def std(self) -> torch.Tensor:
        """Return the current std estimate (computed from variance)."""
        return torch.sqrt(self.running_var)

    @property
    def cov(self) -> torch.Tensor:
        """Return the current covariance matrix estimate."""
        if not self.compute_cov:
            raise RuntimeError("Covariance computation is disabled. Set compute_cov=True to enable.")
        return self.running_cov

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"{self.num_features}, eps={self.eps}, only_centering={self.only_centering}, "
            f"compute_cov={self.compute_cov}, momentum={self.momentum}"
        )


class eDataNorm(DataNorm, EquivariantModule):
    r"""Equivariant version of DataNorm using group-theoretic symmetry-aware statistics.

    This module extends :class:`DataNorm` to work with equivariant data by computing
    statistics that respect the symmetry structure defined by a group representation.
    It maintains the same API and modes of operation as ``DataNorm`` while using
    symmetry-aware mean, variance, and covariance computations from
    :mod:`symm_learning.stats`.

    **Mathematical Formulation:**

    The equivariant normalization follows the same mathematical form as ``DataNorm``:

    .. math::
        y = \begin{cases}
            x - \mu_{\text{equiv}} & \text{if } \texttt{only\_centering} = \text{True} \\
            \frac{x - \mu_{\text{equiv}}}{\sqrt{\sigma^2_{\text{equiv}} + \epsilon}} & \text{otherwise}
        \end{cases}

    However, the statistics :math:`\mu_{\text{equiv}}` and :math:`\sigma^2_{\text{equiv}}`
    are computed using symmetry-aware estimators:

    - **Mean**: Projected onto the :math:`G`-invariant subspace
    - **Variance**: Constrained to be constant within each irreducible subspace
    - **Covariance**: Respects the block-diagonal structure imposed by the representation

    **Symmetry Properties:**

    The computed statistics satisfy equivariance and invariance properties:

    - :math:`\mathbb{E}[g \cdot x] = g \cdot \mathbb{E}[x]` (mean equivariance)
    - :math:`\text{Var}[g \cdot x] = \text{Var}[x]` (variance invariance)
    - :math:`\text{Cov}[g \cdot x, g \cdot y] = g \cdot \text{Cov}[x, y] \cdot g^T` (covariance equivariance)

    **Input/Output Types:**

    Unlike ``DataNorm`` which operates on raw tensors, ``eDataNorm`` processes
    :class:`escnn.nn.GeometricTensor` objects that encode the group representation
    information along with the tensor data.

    Args:
        in_type (escnn.nn.FieldType): The field type defining the input's group
            representation structure. The output type will be the same as the input type.
        mean (torch.Tensor, optional): Initial mean values. If provided with
            ``running_stats=True``, used as initialization for running mean.
            Shape: ``(in_type.size,)``. Defaults to zeros.
        std (torch.Tensor, optional): Initial standard deviation values. If provided with
            ``running_stats=True``, used as initialization for running std.
            Shape: ``(in_type.size,)``. Defaults to ones.
        eps (float, optional): Small constant added to the denominator for numerical
            stability. Default: ``1e-6``.
        only_centering (bool, optional): If ``True``, only centers the data using
            equivariant mean without scaling. Default: ``False``.
        compute_cov (bool, optional): If ``True``, computes and tracks the equivariant
            covariance matrix. Default: ``False``.
        running_stats (bool, optional): If ``True``, maintains running estimates of
            equivariant statistics. Default: ``True``.
        momentum (float, optional): Momentum factor for running statistics updates.
            If ``None``, uses cumulative averaging. Default: ``None``.

    Shape:
        - Input: :class:`escnn.nn.GeometricTensor` with tensor shape :math:`(N, D)` or :math:`(N, D, L)` where:
          - :math:`N` is the batch size
          - :math:`D` is ``in_type.size`` (total representation dimension)
          - :math:`L` is the sequence length (optional, for 3D inputs)
        - Output: :class:`escnn.nn.GeometricTensor` with the same type and shape as input

    Methods:
        export() -> DataNorm: Exports the current state to a standard ``DataNorm`` layer
            that can operate on raw tensors, transferring all learned statistics.

    Examples:
        >>> from escnn import gspaces, nn as escnn_nn
        >>> from escnn.group import CyclicGroup
        >>> 
        >>> # Define group and representation
        >>> G = CyclicGroup(4)
        >>> gspace = gspaces.no_base_space(G)
        >>> in_type = escnn_nn.FieldType(gspace, [G.regular_representation] * 2)
        >>> 
        >>> # Create equivariant normalization layer
        >>> norm = eDataNorm(in_type=in_type, compute_cov=True)
        >>> 
        >>> # Process equivariant data
        >>> x_tensor = torch.randn(16, in_type.size)  # Raw tensor data
        >>> x_geom = in_type(x_tensor)  # Wrap in GeometricTensor
        >>> y_geom = norm(x_geom)  # Normalized GeometricTensor
        >>> 
        >>> # Export to standard DataNorm
        >>> standard_norm = norm.export()
        >>> y_tensor = standard_norm(x_tensor)  # Same result on raw tensor

    Note:
        This layer inherits all modes of operation from :class:`DataNorm` (running statistics,
        fixed statistics, centering-only, covariance computation) while computing all
        statistics using group-theoretic constraints. The statistics respect the irreducible
        decomposition of the input representation, ensuring that symmetries are preserved
        throughout the normalization process.

    See Also:
        :class:`DataNorm`: The base normalization layer for standard (non-equivariant) data.
        :func:`symm_learning.stats.var_mean`: Equivariant mean and variance computation.
        :func:`symm_learning.stats.cov`: Equivariant covariance computation.
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
