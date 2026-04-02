from __future__ import annotations

import torch
from escnn.group import Representation

import symm_learning.stats
from symm_learning.linalg import equiv_orthogonal_projection_coefficients
from symm_learning.nn.module import eModule
from symm_learning.representation_theory import GroupHomomorphismBasis


class EMAStats(eModule):
    r"""Exponential moving averages of first and second moments.

    Let :math:`\mathbf{X}:\Omega\to\mathcal{X}` and :math:`\mathbf{Y}:\Omega\to\mathcal{Y}` be two random variables.
    This module tracks exponential moving averages of their batch means
    :math:`\boldsymbol{\mu}_x`, :math:`\boldsymbol{\mu}_y`, self-covariances
    :math:`\mathbf{C}_{xx}`, :math:`\mathbf{C}_{yy}`, and cross-covariance :math:`\mathbf{C}_{xy}`.

    For any tracked statistic :math:`\mathbf{S}`, the running update is

    .. math::
        \mathbf{S}_{t} = (1-\alpha)\mathbf{S}_{t-1} + \alpha\,\mathbf{S}^{\text{batch}}_{t},

    where :math:`\alpha\in(0,1]` is the momentum and :math:`\mathbf{S}^{\text{batch}}_{t}` is the statistic computed
    from the current batch.

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

    Example:
        >>> stats = EMAStats(dim_x=10, dim_y=5, momentum=0.1)
        >>> x = torch.randn(32, 10)
        >>> y = torch.randn(32, 5)
        >>> x_out, y_out = stats(x, y)
        >>> mu_x = stats.mean_x
        >>> C_xy = stats.cov_xy
        >>> print(mu_x.shape, C_xy.shape)

    Note:
        Whenever previously tracked statistics are reused, the historical means
        :math:`\boldsymbol{\mu}_{x,t-1}`, :math:`\boldsymbol{\mu}_{y,t-1}` and covariance operators
        :math:`\mathbf{C}_{xx,t-1}`, :math:`\mathbf{C}_{yy,t-1}`, :math:`\mathbf{C}_{xy,t-1}` are first detached from
        the autograd graph. In particular, the centering mean and the previous EMA state in the update

        .. math::
            \mathbf{S}_{t} = (1-\alpha)\,\operatorname{detach}(\mathbf{S}_{t-1}) + \alpha\,\mathbf{S}^{\text{batch}}_{t}

        are treated as constants with respect to the current optimization step. This truncates the computation graph
        across batches, preventing gradients from propagating through historical running state while keeping the
        current batch contribution differentiable.
    """

    requires_reps = False

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
        r"""Compute batch statistics. Can be overridden for equivariant versions.

        Args:
            x: Input tensor x of shape (N, D_x).
            y: Input tensor y of shape (N, D_y).

        Returns:
            Tuple of (mean_x, mean_y, cov_xx, cov_yy, cov_xy).

        Note:
            If covariance centering reuses the running means :math:`\boldsymbol{\mu}_{x,t-1}` and
            :math:`\boldsymbol{\mu}_{y,t-1}`, they are replaced by
            :math:`\operatorname{detach}(\boldsymbol{\mu}_{x,t-1})` and
            :math:`\operatorname{detach}(\boldsymbol{\mu}_{y,t-1})`. This keeps the current batch covariance
            differentiable only with respect to the current inputs.
        """
        # Compute batch means
        mean_x = x.mean(dim=0)
        mean_y = y.mean(dim=0)

        # For covariance computation, use running means if available and enabled, otherwise batch means
        if self.center_with_running_mean and self.num_batches_tracked > 0:
            # Use running means for centering to maintain consistency with EMA
            # Detach here so covariance gradients do not backpropagate through old EMA state.
            # Only the current batch should contribute gradient information in this step.
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
        r"""Update running statistics and return inputs unchanged.

        Args:
            x: Input tensor x of shape (N, num_features_x).
            y: Input tensor y of shape (N, num_features_y).

        Returns:
            Tuple (x, y) - inputs are returned unchanged.

        Note:
            For batches after the first one, the EMA update uses
            :math:`\operatorname{detach}(\boldsymbol{\mu}_{x,t-1})`,
            :math:`\operatorname{detach}(\boldsymbol{\mu}_{y,t-1})`,
            :math:`\operatorname{detach}(\mathbf{C}_{xx,t-1})`,
            :math:`\operatorname{detach}(\mathbf{C}_{yy,t-1})`, and
            :math:`\operatorname{detach}(\mathbf{C}_{xy,t-1})` before mixing them with the current batch statistics.
            This preserves differentiability with respect to the current batch while preventing the running buffers
            from carrying a growing cross-batch autograd graph.
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
                # First batch: initialize from the current batch directly. `copy_` preserves the
                # current-step gradient path without mixing in the arbitrary initialization values.
                self.running_mean_x.copy_(batch_mean_x)
                self.running_mean_y.copy_(batch_mean_y)
                self.running_cov_xx.copy_(batch_cov_xx)
                self.running_cov_yy.copy_(batch_cov_yy)
                self.running_cov_xy.copy_(batch_cov_xy)
            else:
                # Detach the previous EMA state so autograd sees it as a constant for this step.
                # This avoids backpropagating through the full batch history while preserving the
                # gradient contribution of the current batch statistic.
                alpha = self.momentum
                self.running_mean_x = self.running_mean_x.detach() * (1 - alpha) + batch_mean_x * alpha
                self.running_mean_y = self.running_mean_y.detach() * (1 - alpha) + batch_mean_y * alpha
                self.running_cov_xx = self.running_cov_xx.detach() * (1 - alpha) + batch_cov_xx * alpha
                self.running_cov_yy = self.running_cov_yy.detach() * (1 - alpha) + batch_cov_yy * alpha
                self.running_cov_xy = self.running_cov_xy.detach() * (1 - alpha) + batch_cov_xy * alpha

            self.num_batches_tracked += 1

        # Return inputs unchanged
        return x, y

    def invalidate_cache(self) -> None:
        """Standard EMA stats keep no derived cache."""

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
    r"""Equivariant EMA statistics on symmetric vector spaces.

    Let :math:`\mathbf{X}:\Omega\to\mathcal{X}` and :math:`\mathbf{Y}:\Omega\to\mathcal{Y}` be two
    :math:`\mathbb{G}`-invariant random variables taking values in the symmetric vector spaces
    :math:`(\mathcal{X}, \rho_{\mathcal{X}})` and :math:`(\mathcal{Y}, \rho_{\mathcal{Y}})`. This module tracks
    exponential moving averages of their symmetry-constrained first and second moments.

    The running means are constrained to be :math:`\mathbb{G}`-invariant,

    .. math::
        \rho_{\mathcal{X}}(g)\boldsymbol{\mu}_x = \boldsymbol{\mu}_x,
        \qquad
        \rho_{\mathcal{Y}}(g)\boldsymbol{\mu}_y = \boldsymbol{\mu}_y,
        \qquad \forall g\in\mathbb{G},

    and the running covariance operators commute with the corresponding group actions,

    .. math::
        \rho_{\mathcal{X}}(g)\mathbf{C}_{xx} = \mathbf{C}_{xx}\rho_{\mathcal{X}}(g),
        \qquad
        \rho_{\mathcal{Y}}(g)\mathbf{C}_{yy} = \mathbf{C}_{yy}\rho_{\mathcal{Y}}(g),
        \qquad
        \rho_{\mathcal{X}}(g)\mathbf{C}_{xy} = \mathbf{C}_{xy}\rho_{\mathcal{Y}}(g),
        \qquad \forall g\in\mathbb{G}.

    Equivalently,

    .. math::
        \mathbf{C}_{xx} \in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{X}}),
        \qquad
        \mathbf{C}_{yy} \in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{Y}}),
        \qquad
        \mathbf{C}_{xy} \in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{X}}).

    For any tracked statistic :math:`\mathbf{S}`, the running update is

    .. math::
        \mathbf{S}_{t} = (1-\alpha)\mathbf{S}_{t-1} + \alpha\,\mathbf{S}^{\text{batch}}_{t},

    where :math:`\alpha\in(0,1]` is the momentum and :math:`\mathbf{S}^{\text{batch}}_{t}` is the statistic computed
    from the current batch.

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
        >>> mu_x = stats.mean_x
        >>> mu_y = stats.mean_y
        >>> C_xx = stats.cov_xx
        >>> C_xy = stats.cov_xy
        >>> print(mu_x.shape, mu_y.shape, C_xx.shape, C_xy.shape)

    Note:
        Running covariance buffers are stored internally in the degrees of freedom of
        :math:`\mathrm{Hom}_{\mathbb{G}}` rather than as dense matrices. As in :class:`EMAStats`, previously tracked
        means :math:`\boldsymbol{\mu}_{x,t-1}`, :math:`\boldsymbol{\mu}_{y,t-1}` and covariance coefficients
        :math:`\boldsymbol{\theta}_{xx,t-1}`, :math:`\boldsymbol{\theta}_{yy,t-1}`,
        :math:`\boldsymbol{\theta}_{xy,t-1}` are always replaced by their detached counterparts before reuse in
        covariance centering or in the EMA update. In training mode, the DoF statistics are updated directly using

        .. math::
            \boldsymbol{\theta}_{t}
            = (1-\alpha)\,\operatorname{detach}(\boldsymbol{\theta}_{t-1})
            + \alpha\,\boldsymbol{\theta}^{\text{batch}}_{t},

        so gradients never propagate through older batches. The dense covariance matrices exposed by :attr:`cov_xx`,
        :attr:`cov_yy`, and :attr:`cov_xy` are expanded from the current DoF coefficients on demand. In eval mode,
        they are expanded lazily and cached until the module changes mode, device, dtype, or reloads from a
        checkpoint.
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
        self._buffers.pop("running_cov_xx", None)
        self._buffers.pop("running_cov_yy", None)
        self._buffers.pop("running_cov_xy", None)
        self._cov_xx = None
        self._cov_yy = None
        self._cov_xy = None
        self._cov_cache_dirty = True

        self.cov_xx_basis = GroupHomomorphismBasis(self._rep_x, self._rep_x, basis_expansion="isotypic_expansion")
        self.cov_yy_basis = GroupHomomorphismBasis(self._rep_y, self._rep_y, basis_expansion="isotypic_expansion")
        self.cov_xy_basis = GroupHomomorphismBasis(self._rep_y, self._rep_x, basis_expansion="isotypic_expansion")

        dtype = self.running_mean_x.dtype
        self.register_buffer(
            "running_cov_xx_dof",
            equiv_orthogonal_projection_coefficients(
                torch.eye(self.num_features_x, dtype=dtype),
                rep_x=self._rep_x,
                rep_y=self._rep_x,
            ),
        )
        self.register_buffer(
            "running_cov_yy_dof",
            equiv_orthogonal_projection_coefficients(
                torch.eye(self.num_features_y, dtype=dtype),
                rep_x=self._rep_y,
                rep_y=self._rep_y,
            ),
        )
        self.register_buffer(
            "running_cov_xy_dof",
            equiv_orthogonal_projection_coefficients(
                torch.zeros(self.num_features_x, self.num_features_y, dtype=dtype),
                rep_x=self._rep_y,
                rep_y=self._rep_x,
            ),
        )

    def _compute_batch_stats(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute invariant batch means and equivariant covariance coefficients.

        Args:
            x: Samples of :math:`\mathbf{X}` with shape :math:`(N, D_x)`.
            y: Samples of :math:`\mathbf{Y}` with shape :math:`(N, D_y)`.

        Returns:
            Tuple ``(\boldsymbol{\mu}_x, \boldsymbol{\mu}_y, \boldsymbol{\theta}_{xx},
            \boldsymbol{\theta}_{yy}, \boldsymbol{\theta}_{xy})`` where

            - :math:`\boldsymbol{\mu}_x \in \mathcal{X}^{\text{inv}}`
            - :math:`\boldsymbol{\mu}_y \in \mathcal{Y}^{\text{inv}}`
            - :math:`\boldsymbol{\theta}_{xx}` parameterizes an operator in
              :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{X}})`
            - :math:`\boldsymbol{\theta}_{yy}` parameterizes an operator in
              :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{Y}})`
            - :math:`\boldsymbol{\theta}_{xy}` parameterizes an operator in
              :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{X}})`

            with the covariance quantities expressed in the flattened homomorphism basis.

        Shape:
            - **x**: :math:`(N, D_x)`.
            - **y**: :math:`(N, D_y)`.
            - **Output**: ``((D_x,), (D_y,), (H_{xx},), (H_{yy},), (H_{xy},))`` where
              :math:`H_{xx}=\dim(\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{X}}))`,
              :math:`H_{yy}=\dim(\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{Y}}))`, and
              :math:`H_{xy}=\dim(\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{X}}))`.

        Note:
            If covariance centering reuses the running means :math:`\boldsymbol{\mu}_{x,t-1}` and
            :math:`\boldsymbol{\mu}_{y,t-1}`, it uses
            :math:`\operatorname{detach}(\boldsymbol{\mu}_{x,t-1})` and
            :math:`\operatorname{detach}(\boldsymbol{\mu}_{y,t-1})` so the current covariance coefficients remain
            differentiable only with respect to the current batch.
        """
        # For means, always compute fresh batch means using group-aware method
        mean_x = symm_learning.stats.mean(x, rep_x=self._rep_x)
        mean_y = symm_learning.stats.mean(y, rep_x=self._rep_y)

        # For covariances, we need to center using EMA means for consistency (if enabled)
        if self.center_with_running_mean and self.num_batches_tracked > 0:
            # Use running means for centering to maintain EMA consistency
            # Detach here so the current covariance DoFs do not inherit an autograd path through
            # all previously tracked EMA means.
            center_x = self.running_mean_x.detach()
            center_y = self.running_mean_y.detach()
        else:
            # First batch or when center_with_running_mean=False: use batch means
            center_x = mean_x
            center_y = mean_y

        # Center the data manually since we can't pass custom means to cov function
        x_centered = x - center_x.unsqueeze(0)
        y_centered = y - center_y.unsqueeze(0)

        # Match symm_learning.stats.cov(..., uncentered=True): centered inputs are treated
        # as already-prepared second-moment samples and normalized by N.
        n_samples = x_centered.shape[0]
        cov_xx_dof = equiv_orthogonal_projection_coefficients(
            W=torch.mm(x_centered.T, x_centered) / n_samples,
            rep_x=self._rep_x,
            rep_y=self._rep_x,
        )
        cov_yy_dof = equiv_orthogonal_projection_coefficients(
            W=torch.mm(y_centered.T, y_centered) / n_samples,
            rep_x=self._rep_y,
            rep_y=self._rep_y,
        )
        cov_xy_dof = equiv_orthogonal_projection_coefficients(
            W=torch.mm(x_centered.T, y_centered) / n_samples,
            rep_x=self._rep_y,
            rep_y=self._rep_x,
        )

        return mean_x, mean_y, cov_xx_dof, cov_yy_dof, cov_xy_dof

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Update equivariant running statistics and return the inputs unchanged.

        Args:
            x: Samples in :math:`\mathcal{X}` transforming according to :math:`\rho_{\mathcal{X}}`.
            y: Samples in :math:`\mathcal{Y}` transforming according to :math:`\rho_{\mathcal{Y}}`.

        Returns:
            Tuple ``(x, y)``. The activations are not transformed; only the running invariant means
            and equivariant covariance operators are updated.

        Shape:
            - **x**: :math:`(N, D_x)`.
            - **y**: :math:`(N, D_y)`.
            - **Output**: ``((N, D_x), (N, D_y))``.

        """
        assert x.shape[-1] == self.x_rep.size, f"Expected x.shape[-1]={self.x_rep.size}, got {x.shape}"
        assert y.shape[-1] == self.y_rep.size, f"Expected y.shape[-1]={self.y_rep.size}, got {y.shape}"
        assert x.ndim == 2, f"Expected 2D tensor for x, got {x.ndim}D"
        assert y.ndim == 2, f"Expected 2D tensor for y, got {y.ndim}D"
        assert x.shape[0] == y.shape[0], f"Batch sizes must match: x={x.shape[0]}, y={y.shape[0]}"

        if self.training:
            batch_mean_x, batch_mean_y, batch_cov_xx_dof, batch_cov_yy_dof, batch_cov_xy_dof = (
                self._compute_batch_stats(x, y)
            )
            if self.num_batches_tracked == 0:
                # First batch: expose the current batch statistics directly, without mixing in the
                # initialization buffers. `copy_` keeps the current-step gradient path intact.
                self.running_mean_x.copy_(batch_mean_x)
                self.running_mean_y.copy_(batch_mean_y)
                self.running_cov_xx_dof.copy_(batch_cov_xx_dof)
                self.running_cov_yy_dof.copy_(batch_cov_yy_dof)
                self.running_cov_xy_dof.copy_(batch_cov_xy_dof)
            else:
                # Detach the previous EMA state before the update so the running DoF buffers expose
                # the latest differentiable statistic without backpropagating through older batches.
                alpha = self.momentum
                self.running_mean_x = self.running_mean_x.detach() * (1 - alpha) + batch_mean_x * alpha
                self.running_mean_y = self.running_mean_y.detach() * (1 - alpha) + batch_mean_y * alpha
                self.running_cov_xx_dof = self.running_cov_xx_dof.detach() * (1 - alpha) + batch_cov_xx_dof * alpha
                self.running_cov_yy_dof = self.running_cov_yy_dof.detach() * (1 - alpha) + batch_cov_yy_dof * alpha
                self.running_cov_xy_dof = self.running_cov_xy_dof.detach() * (1 - alpha) + batch_cov_xy_dof * alpha

            self._mark_cov_cache_dirty()
            self.num_batches_tracked += 1

        return x, y

    def _mark_cov_cache_dirty(self) -> None:
        self._cov_cache_dirty = True

    def _expand_covariances(self) -> None:
        self._cov_xx = self.cov_xx_basis(self.running_cov_xx_dof)
        self._cov_yy = self.cov_yy_basis(self.running_cov_yy_dof)
        self._cov_xy = self.cov_xy_basis(self.running_cov_xy_dof)
        self._cov_cache_dirty = False

    def invalidate_cache(self) -> None:
        """Clear cached dense covariance expansions."""
        self._cov_xx = None
        self._cov_yy = None
        self._cov_xy = None
        self._mark_cov_cache_dirty()

    def _ensure_cov_cache(self) -> None:
        if self._cov_cache_dirty or self._cov_xx is None or self._cov_yy is None or self._cov_xy is None:
            self._expand_covariances()

    @property
    def mean_x(self) -> torch.Tensor:
        r"""Running invariant mean in :math:`\mathcal{X}^{\text{inv}}`.

        This vector satisfies

        .. math::
            \rho_{\mathcal{X}}(g)\boldsymbol{\mu}_x = \boldsymbol{\mu}_x,
            \qquad \forall g\in\mathbb{G}.
        """
        return self.running_mean_x

    @property
    def mean_y(self) -> torch.Tensor:
        r"""Running invariant mean in :math:`\mathcal{Y}^{\text{inv}}`.

        This vector satisfies

        .. math::
            \rho_{\mathcal{Y}}(g)\boldsymbol{\mu}_y = \boldsymbol{\mu}_y,
            \qquad \forall g\in\mathbb{G}.
        """
        return self.running_mean_y

    @property
    def cov_xx(self) -> torch.Tensor:
        r"""Running covariance operator in :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{X}})`.

        Equivalently, the returned matrix :math:`\mathbf{C}_{xx}` satisfies

        .. math::
            \rho_{\mathcal{X}}(g)\mathbf{C}_{xx} = \mathbf{C}_{xx}\rho_{\mathcal{X}}(g),
            \qquad \forall g\in\mathbb{G}.

        Shape:
            :math:`(D_x, D_x)`.
        """
        if self.training:
            return self.cov_xx_basis(self.running_cov_xx_dof)
        self._ensure_cov_cache()
        return self._cov_xx

    @property
    def cov_yy(self) -> torch.Tensor:
        r"""Running covariance operator in :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{Y}})`.

        Equivalently, the returned matrix :math:`\mathbf{C}_{yy}` satisfies

        .. math::
            \rho_{\mathcal{Y}}(g)\mathbf{C}_{yy} = \mathbf{C}_{yy}\rho_{\mathcal{Y}}(g),
            \qquad \forall g\in\mathbb{G}.

        Shape:
            :math:`(D_y, D_y)`.
        """
        if self.training:
            return self.cov_yy_basis(self.running_cov_yy_dof)
        self._ensure_cov_cache()
        return self._cov_yy

    @property
    def cov_xy(self) -> torch.Tensor:
        r"""Running cross-covariance operator.

        The returned matrix :math:`\mathbf{C}_{xy}` maps coordinates in :math:`\mathcal{Y}` to coordinates in
        :math:`\mathcal{X}`. It belongs to
        :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{Y}}, \rho_{\mathcal{X}})` and satisfies the intertwining
        constraint

        .. math::
            \rho_{\mathcal{X}}(g)\mathbf{C}_{xy} = \mathbf{C}_{xy}\rho_{\mathcal{Y}}(g),
            \qquad \forall g\in\mathbb{G}.

        Shape:
            :math:`(D_x, D_y)`.
        """
        if self.training:
            return self.cov_xy_basis(self.running_cov_xy_dof)
        self._ensure_cov_cache()
        return self._cov_xy

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        legacy_cov_keys = {
            "running_cov_xx": (self._rep_x, self._rep_x),
            "running_cov_yy": (self._rep_y, self._rep_y),
            "running_cov_xy": (self._rep_y, self._rep_x),
        }
        for legacy_key, (rep_in, rep_out) in legacy_cov_keys.items():
            legacy_full_key = prefix + legacy_key
            dof_full_key = prefix + f"{legacy_key}_dof"
            legacy_value = state_dict.pop(legacy_full_key, None)
            if legacy_value is not None and dof_full_key not in state_dict:
                state_dict[dof_full_key] = equiv_orthogonal_projection_coefficients(
                    W=legacy_value,
                    rep_x=rep_in,
                    rep_y=rep_out,
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


if __name__ == "__main__":
    import escnn

    from symm_learning.representation_theory import direct_sum
    from symm_learning.utils import run_module_pair_profile

    SEED = 123
    BATCH_SIZE = 1024
    REGULAR_COPIES = 2
    MODE = "train"  # options: eval, train, both

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    G = escnn.group.Icosahedral()
    rep = direct_sum([G.regular_representation] * REGULAR_COPIES)

    estats = eEMAStats(x_rep=rep, y_rep=rep, momentum=0.1).to(device)
    stats = EMAStats(dim_x=rep.size, dim_y=rep.size, momentum=0.1).to(device)

    x = torch.randn(BATCH_SIZE, rep.size, device=device)
    y = torch.randn(BATCH_SIZE, rep.size, device=device)

    run_module_pair_profile(
        lhs_name="eEMAStats",
        lhs=estats,
        rhs_name="EMAStats",
        rhs=stats,
        x=(x, y),
        group_name=G.name,
        mode=MODE,
        profile_active_steps=200,
        profile_warmup_steps=10,
    )
