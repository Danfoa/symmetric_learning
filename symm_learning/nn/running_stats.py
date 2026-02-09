from __future__ import annotations

import torch
from escnn.group import Representation

import symm_learning.stats


class EMAStats(torch.nn.Module):
    r"""Exponential Moving Average (EMA) statistics tracker for paired data.

    This module tracks running statistics of two input tensors using exponential moving
    averages without transforming the data. It computes and maintains estimates of:

    - :math:`\\mu_x`: Mean of input tensor x
    - :math:`\\mu_y`: Mean of input tensor y
    - :math:`\\Sigma_{xx}`: Covariance matrix of x
    - :math:`\\Sigma_{yy}`: Covariance matrix of y
    - :math:`\\Sigma_{xy}`: Cross-covariance matrix between x and y

    **Mathematical Formulation:**

    The exponential moving average update rule for any statistic :math:`S` is:

    .. math::
        S_{\\text{running}} = (1 - \\alpha) \\cdot S_{\\text{running}} + \\alpha \\cdot S_{\\text{batch}}

    where :math:`\\alpha` is the momentum parameter and :math:`S_{\\text{batch}}` is the
    statistic computed from the current batch.

    **Covariance Computation:**

    For tensors of shape :math:`(N, D)`:

    - Mean: :math:`\\mu = \\mathbb{E}[x]` computed over batch dimension
    - Covariance: :math:`\\Sigma = \\mathbb{E}[(x - \\mu)(x - \\mu)^T]`
    - Cross-covariance: :math:`\\Sigma_{xy} = \\mathbb{E}[(x - \\mu_x)(y - \\mu_y)^T]`

    Args:
        num_features_x: Number of features in input tensor x.
        num_features_y: Number of features in input tensor y. If None, uses same as x.
        momentum: Momentum factor for exponential moving average. Must be in (0, 1].
            Higher values give more weight to recent batches. Default: 0.1.
        eps: Small constant for numerical stability. Default: 1e-6.
        center_with_running_mean: If True, center covariance computation using running means
            instead of batch means (except for first batch). Default: True.

    Shape:
        - Input x: :math:`(N, D_x)` where N is batch size and :math:`D_x` is num_features_x.
        - Input y: :math:`(N, D_y)` where :math:`D_y` is num_features_y.
        - Output: Same as inputs (data is not transformed).

    Attributes:
        running_mean_x (:class:`~torch.Tensor`): Running mean of x. Shape: :math:`(D_x,)`.
        running_mean_y (:class:`~torch.Tensor`): Running mean of y. Shape: :math:`(D_y,)`.
        running_cov_xx (:class:`~torch.Tensor`): Running covariance of x. Shape: :math:`(D_x, D_x)`.
        running_cov_yy (:class:`~torch.Tensor`): Running covariance of y. Shape: :math:`(D_y, D_y)`.
        running_cov_xy (:class:`~torch.Tensor`): Running cross-covariance. Shape: :math:`(D_x, D_y)`.
        num_batches_tracked (:class:`~torch.Tensor`): Number of batches processed.

    Example:
        >>> stats = EMAStats(num_features_x=10, num_features_y=5, momentum=0.1)
        >>> x = torch.randn(32, 10)  # Batch of 32 samples, 10 features
        >>> y = torch.randn(32, 5)  # Batch of 32 samples, 5 features
        >>> x_out, y_out = stats(x, y)  # x_out == x, y_out == y (no transformation)
        >>> print(stats.mean_x.shape)  # torch.Size([10])
        >>> print(stats.cov_xy.shape)  # torch.Size([10, 5])
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int | None = None,
        momentum: float = 0.1,
        eps: float = 1e-6,
        center_with_running_mean: bool = True,
    ):
        super().__init__()

        self.num_features_x = dim_x
        self.num_features_y = dim_y if dim_y is not None else dim_x
        self.eps = eps
        self.center_with_running_mean = center_with_running_mean

        if not (0 < momentum <= 1):
            raise ValueError(f"momentum must be in (0, 1], got {momentum}")
        self.momentum = momentum

        # Initialize running statistics buffers
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("running_mean_x", torch.zeros(self.num_features_x))
        self.register_buffer("running_mean_y", torch.zeros(self.num_features_y))
        self.register_buffer("running_cov_xx", torch.eye(self.num_features_x))
        self.register_buffer("running_cov_yy", torch.eye(self.num_features_y))
        self.register_buffer("running_cov_xy", torch.zeros(self.num_features_x, self.num_features_y))

    def _compute_batch_stats(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute batch statistics. Can be overridden for equivariant versions.

        Args:
            x: Input tensor x of shape (N, D_x).
            y: Input tensor y of shape (N, D_y).

        Returns:
            Tuple of (mean_x, mean_y, cov_xx, cov_yy, cov_xy).
        """
        # Compute batch means
        mean_x = x.mean(dim=0)
        mean_y = y.mean(dim=0)

        # For covariance computation, use running means if available and enabled, otherwise batch means
        if self.center_with_running_mean and self.num_batches_tracked > 0:
            # Use running means for centering to maintain consistency with EMA
            # Detach to prevent gradients from flowing through previous iterations
            center_x = self.running_mean_x.detach()
            center_y = self.running_mean_y.detach()
        else:
            # First batch or when center_with_running_mean=False: use batch means
            center_x = mean_x
            center_y = mean_y

        # Center the data using the appropriate means
        x_centered = x - center_x.unsqueeze(0)
        y_centered = y - center_y.unsqueeze(0)

        # Compute covariances
        n_samples = x.shape[0]
        cov_xx = torch.mm(x_centered.T, x_centered) / (n_samples - 1)
        cov_yy = torch.mm(y_centered.T, y_centered) / (n_samples - 1)
        cov_xy = torch.mm(x_centered.T, y_centered) / (n_samples - 1)

        return mean_x, mean_y, cov_xx, cov_yy, cov_xy

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update running statistics and return inputs unchanged.

        Args:
            x: Input tensor x of shape (N, num_features_x).
            y: Input tensor y of shape (N, num_features_y).

        Returns:
            Tuple (x, y) - inputs are returned unchanged.
        """
        assert x.ndim == 2, f"Expected 2D tensor for x, got {x.ndim}D"
        assert y.ndim == 2, f"Expected 2D tensor for y, got {y.ndim}D"
        assert x.shape[1] == self.num_features_x, f"Expected x.shape[1]={self.num_features_x}, got {x.shape[1]}"
        assert y.shape[1] == self.num_features_y, f"Expected y.shape[1]={self.num_features_y}, got {y.shape[1]}"
        assert x.shape[0] == y.shape[0], f"Batch sizes must match: x={x.shape[0]}, y={y.shape[0]}"

        if self.training:
            # Compute batch statistics
            batch_mean_x, batch_mean_y, batch_cov_xx, batch_cov_yy, batch_cov_xy = self._compute_batch_stats(x, y)

            # Update running statistics with EMA
            if self.num_batches_tracked == 0:
                # First batch: initialize with batch statistics
                self.running_mean_x.copy_(batch_mean_x)
                self.running_mean_y.copy_(batch_mean_y)
                self.running_cov_xx.copy_(batch_cov_xx)
                self.running_cov_yy.copy_(batch_cov_yy)
                self.running_cov_xy.copy_(batch_cov_xy)
            else:
                # EMA update: detach previous running stats to prevent gradient flow into history
                alpha = self.momentum
                self.running_mean_x = self.running_mean_x.detach() * (1 - alpha) + batch_mean_x * alpha
                self.running_mean_y = self.running_mean_y.detach() * (1 - alpha) + batch_mean_y * alpha
                self.running_cov_xx = self.running_cov_xx.detach() * (1 - alpha) + batch_cov_xx * alpha
                self.running_cov_yy = self.running_cov_yy.detach() * (1 - alpha) + batch_cov_yy * alpha
                self.running_cov_xy = self.running_cov_xy.detach() * (1 - alpha) + batch_cov_xy * alpha

            self.num_batches_tracked += 1

        # Return inputs unchanged
        return x, y

    @property
    def mean_x(self) -> torch.Tensor:
        """Running mean of input x."""
        return self.running_mean_x

    @property
    def mean_y(self) -> torch.Tensor:
        """Running mean of input y."""
        return self.running_mean_y

    @property
    def cov_xx(self) -> torch.Tensor:
        """Running covariance matrix of x."""
        return self.running_cov_xx

    @property
    def cov_yy(self) -> torch.Tensor:
        """Running covariance matrix of y."""
        return self.running_cov_yy

    @property
    def cov_xy(self) -> torch.Tensor:
        """Running cross-covariance matrix between x and y."""
        return self.running_cov_xy

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f"num_features_x={self.num_features_x}, num_features_y={self.num_features_y}, "
            f"momentum={self.momentum}, eps={self.eps}, center_with_running_mean={self.center_with_running_mean}"
        )


class eEMAStats(EMAStats):
    r"""Equivariant version of EMAStats using group-theoretic symmetry-aware statistics.

    This module extends :class:`EMAStats` to work with equivariant data by computing
    statistics that respect the symmetry structure defined by group representations.
    It uses symmetry-aware mean and covariance computations from :mod:`symm_learning.stats`.

    Args:
        x_rep (:class:`~escnn.group.Representation`): Representation defining input x's group action.
        y_rep (:class:`~escnn.group.Representation`): Representation defining input y's group action.
            If None, uses ``x_rep``.
        momentum (float, optional): Momentum factor for exponential moving average. Default: 0.1.
        eps (float, optional): Small constant for numerical stability. Default: 1e-6.
        center_with_running_mean (bool, optional): If True, center covariance computation
            using running means instead of batch means (except for first batch). Default: True.

    Shape:
        - Input x: ``(N, D_x)``
        - Input y: ``(N, D_y)``
        - Output: Same as inputs (data is not transformed)

    Example:
        >>> stats = eEMAStats(x_rep=rep_x, y_rep=rep_y, momentum=0.1)
        >>> x_out, y_out = stats(x, y)  # Same tensors, updated statistics
        >>> standard_stats = stats.export()  # Export to standard EMAStats
    """

    def __init__(
        self,
        x_rep: Representation,
        y_rep: Representation | None = None,
        momentum: float = 0.1,
        eps: float = 1e-6,
        center_with_running_mean: bool = True,
    ):
        if not isinstance(x_rep, Representation):
            raise TypeError(f"x_rep must be a Representation, got {type(x_rep)}")
        if y_rep is not None and not isinstance(y_rep, Representation):
            raise TypeError(f"y_rep must be a Representation, got {type(y_rep)}")
        # Store representations
        self.x_rep = x_rep
        self.y_rep = y_rep if y_rep is not None else x_rep

        # Ensure groups match
        assert self.x_rep.group == self.y_rep.group, "x_rep and y_rep must share the same group"

        # Store representations for stats computation
        self._rep_x = self.x_rep
        self._rep_y = self.y_rep

        # Initialize EMAStats with the representation sizes
        super().__init__(
            dim_x=self.x_rep.size,
            dim_y=self.y_rep.size,
            momentum=momentum,
            eps=eps,
            center_with_running_mean=center_with_running_mean,
        )

    def _compute_batch_stats(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute equivariant batch statistics using symm_learning.stats.

        Args:
            x: Input tensor x of shape (N, D_x).
            y: Input tensor y of shape (N, D_y).

        Returns:
            Tuple of (mean_x, mean_y, cov_xx, cov_yy, cov_xy) computed using
            symmetry-aware estimators.
        """
        # For means, always compute fresh batch means using group-aware method
        mean_x = symm_learning.stats.mean(x, rep_x=self._rep_x)
        mean_y = symm_learning.stats.mean(y, rep_x=self._rep_y)

        # For covariances, we need to center using EMA means for consistency (if enabled)
        if self.center_with_running_mean and self.num_batches_tracked > 0:
            # Use running means for centering to maintain EMA consistency
            # Detach to prevent gradients from flowing through previous iterations
            center_x = self.running_mean_x.detach()
            center_y = self.running_mean_y.detach()
        else:
            # First batch or when center_with_running_mean=False: use batch means
            center_x = mean_x
            center_y = mean_y

        # Center the data manually since we can't pass custom means to cov function
        x_centered = x - center_x.unsqueeze(0)
        y_centered = y - center_y.unsqueeze(0)

        # Compute covariances using group-aware method on centered data
        cov_xx = symm_learning.stats.cov(x=x_centered, y=x_centered, rep_x=self._rep_x, rep_y=self._rep_x)
        cov_yy = symm_learning.stats.cov(x=y_centered, y=y_centered, rep_x=self._rep_y, rep_y=self._rep_y)
        # Transpose to match expected shape (D_x, D_y)
        cov_xy = symm_learning.stats.cov(x=x_centered, y=y_centered, rep_x=self._rep_x, rep_y=self._rep_y).T

        return mean_x, mean_y, cov_xx, cov_yy, cov_xy

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update running statistics and return inputs unchanged.

        Args:
            x: Input tensor x with representation ``x_rep``.
            y: Input tensor y with representation ``y_rep``.

        Returns:
            Tuple (x, y) - inputs are returned unchanged.
        """
        assert x.shape[-1] == self.x_rep.size, f"Expected x.shape[-1]={self.x_rep.size}, got {x.shape}"
        assert y.shape[-1] == self.y_rep.size, f"Expected y.shape[-1]={self.y_rep.size}, got {y.shape}"

        # Apply EMAStats forward to the tensor data
        x_out_tensor, y_out_tensor = super().forward(x, y)
        return x_out_tensor, y_out_tensor

    def export(self) -> EMAStats:
        """Export to a standard EMAStats layer."""
        exported = EMAStats(
            dim_x=self.num_features_x,
            dim_y=self.num_features_y,
            momentum=self.momentum,
            eps=self.eps,
        )

        # Transfer state
        exported.running_mean_x.data = self.running_mean_x.clone()
        exported.running_mean_y.data = self.running_mean_y.clone()
        exported.running_cov_xx.data = self.running_cov_xx.clone()
        exported.running_cov_yy.data = self.running_cov_yy.clone()
        exported.running_cov_xy.data = self.running_cov_xy.clone()
        exported.num_batches_tracked.data = self.num_batches_tracked.clone()

        exported.eval()

        return exported
