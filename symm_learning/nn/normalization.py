from __future__ import annotations

import logging

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
        init_scheme="identity",
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

    def reset_parameters(self, scheme="identity") -> None:  # noqa: D102
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
            scale_spec, bias_spec = self.affine.spectral_parameters(device=x_spec.device, dtype=x_spec.dtype)
            x_spec = x_spec * scale_spec.view(*([1] * (x_spec.ndim - 1)), -1)
            if bias_spec is not None:
                x_spec = x_spec + bias_spec.view(*([1] * (x_spec.ndim - 1)), -1)

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
        batch_var = torch.ones_like(batch_mean) if self.only_centering else x.var(dim=dims, unbiased=False)
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
        eps (float, optional): Small constant added to the denominator for numerical
            stability. Only used when ``only_centering=False``. Default: ``1e-6``.
        only_centering (bool, optional): If ``True``, only centers the data using
            equivariant mean without scaling. Default: ``False``.
        compute_cov (bool, optional): If ``True``, computes and tracks the equivariant
            covariance matrix. Default: ``False``.
        momentum (float, optional): Momentum factor for exponential moving average of
            running statistics. Must be greater than 0. Setting to ``1.0`` effectively 
            uses only batch statistics. Default: ``1.0``.

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
        eps: float = 1e-6,
        only_centering: bool = False,
        compute_cov: bool = False,
        momentum: float = 1.0,
    ):
        # Initialize DataNorm with the field type size
        super().__init__(
            num_features=in_type.size,
            eps=eps,
            only_centering=only_centering,
            compute_cov=compute_cov,
            momentum=momentum,
        )

        # Store EquivariantModule-specific attributes first
        self.in_type = in_type
        self.out_type = in_type
        self._rep_x = in_type.representation

    def _compute_batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivariant batch statistics using symm_learning.stats."""
        batch_var, batch_mean = symm_learning.stats.var_mean(x, rep_x=self._rep_x)
        if self.only_centering:
            batch_var = torch.ones_like(batch_mean)
        return batch_mean, batch_var

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
            momentum=self.momentum,
        )

        # Transfer state
        exported.running_mean.data = self.running_mean.clone()
        exported.running_var.data = self.running_var.clone()
        exported.num_batches_tracked.data = self.num_batches_tracked.clone()
        if self.compute_cov and hasattr(self, "running_cov"):
            exported.running_cov.data = self.running_cov.clone()

        exported._last_cov = self._last_cov

        exported.eval()

        return exported


if __name__ == "__main__":
    # logger.setLevel(logging.DEBUG)
    from escnn.group import CyclicGroup, DihedralGroup

    from symm_learning.models.transformer.etransformer import eTransformerEncoderLayer
    from symm_learning.nn.linear import eLinear
    from symm_learning.utils import check_equivariance

    torch.manual_seed(0)
    np.random.seed(0)

    G = CyclicGroup(10)
    m = 10
    in_rep = direct_sum([G.regular_representation] * m)

    class ModuleStack(torch.nn.Module):  # noqa: D101
        def __init__(self, module, iters: int = 1):
            super().__init__()
            self.module = module
            self.iters = iters

        def forward(self, x: torch.Tensor):  # noqa: D102
            for _ in range(self.iters):
                y = x + self.module(x)
                x = y
            return x

    def check_equivariance_local(  # noqa: D103
        module,
        module_name: str,
        in_rep,
        out_rep=None,
        batch_size: int = 64,
        samples: int = 20,
        input_dim: int = 3,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        dtype=torch.float64,
        raise_on_fail: bool = True,
    ):
        out_rep = out_rep or in_rep
        spatial_dims = 9
        module = module.to(dtype=dtype)
        if input_dim == 2:
            x = torch.randn(batch_size, in_rep.size, dtype=dtype)
        elif input_dim == 3:
            x = torch.randn(batch_size, spatial_dims, in_rep.size, dtype=dtype)
        else:
            raise ValueError(f"Unsupported input_dim {input_dim}")

        module.eval()
        max_err = 0.0
        success = True
        for g in G.elements:
            # g = in_rep.group.sample()
            gx = torch.einsum("ij,...j->...i", torch.tensor(in_rep(g), dtype=dtype), x)
            y = module(x)
            gy = torch.einsum("ij,...j->...i", torch.tensor(out_rep(g), dtype=dtype), y)
            gy_expected = module(gx)
            err = (gy - gy_expected).abs().max().item()
            max_err = max(max_err, err)
            if not torch.allclose(gy, gy_expected, atol=atol, rtol=rtol):
                success = False
                if raise_on_fail:
                    raise AssertionError(f"Equivariance test failed for {module_name} and g={g}, max err {err}")
                break
        if success:
            print(f"Equivariant check passed for module {module_name} with max error {max_err}")
        return success, max_err

    class ResidualStack(torch.nn.Module):  # noqa: D101
        def __init__(self, lin, norm, iters: int = 1, normalize: bool = True, residual: bool = True):
            super().__init__()
            self.lin = lin
            self.norm = norm
            self.act = torch.nn.ReLU()
            self.iters = iters
            self.normalize = normalize
            self.residual = residual

        def forward(self, x: torch.Tensor):  # noqa: D102
            for _ in range(self.iters):
                xp = x
                if self.normalize:
                    xp = self.norm(xp)
                xp = self.lin(xp)
                xp = self.act(xp)
                y = x + xp if self.residual else xp
                x = y
            return x

    ln_results = []
    res_results = []
    # Test LayerNorm with a residual add to see if the add alone causes drift.
    print("\nLayerNorm residual chains (bias=False, equiv_affine=False)")
    for n in [1, 10, 50]:
        norm = eLayerNorm(in_rep=in_rep, bias=False, equiv_affine=False, init_scheme="random")

        class LNResidual(torch.nn.Module):  # noqa: D101
            def __init__(self, norm_layer, iters):
                super().__init__()
                self.norm_layer = norm_layer
                self.iters = iters

            def forward(self, x):  # noqa: D102
                for _ in range(self.iters):
                    x = x + self.norm_layer(x)
                return x

        stack = LNResidual(norm, n)
        ok, err = check_equivariance_local(
            stack,
            module_name=f"{norm}+res x {n}",
            in_rep=norm.in_rep,
            out_rep=norm.out_rep,
            atol=1e-4,
            rtol=1e-4,
            raise_on_fail=False,
        )
        ln_results.append({"depth": n, "ok": ok, "err": err, "normalize": True, "residual": True})

    print("\nResidual stacks (norm ? -> linear -> act -> residual ?)")
    init_schemes = ["xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform"]
    configs = [
        {"normalize": True, "residual": True},
        {"normalize": True, "residual": False},
        {"normalize": False, "residual": True},
        {"normalize": False, "residual": False},
    ]
    for cfg in configs:
        print(f"\nConfig normalize={cfg['normalize']}, residual={cfg['residual']}")
        for scheme in init_schemes:
            print(f"  Init scheme: {scheme}")
            for n in [1, 5, 10]:
                lin = eLinear(in_rep=in_rep, out_rep=in_rep, bias=False, init_scheme=scheme)
                norm = eLayerNorm(in_rep=in_rep, bias=False, equiv_affine=False, init_scheme="random")
                stack = ResidualStack(lin=lin, norm=norm, iters=n, normalize=cfg["normalize"], residual=cfg["residual"])
                ok, err = check_equivariance_local(
                    stack,
                    module_name=f"[res_stack init={scheme} norm={cfg['normalize']} res={cfg['residual']}] x {n}",
                    in_rep=norm.in_rep,
                    out_rep=norm.out_rep,
                    atol=1e-4,
                    rtol=1e-4,
                    raise_on_fail=False,
                )
                res_results.append(
                    {
                        "scheme": scheme,
                        "depth": n,
                        "ok": ok,
                        "err": err,
                        "normalize": cfg["normalize"],
                        "residual": cfg["residual"],
                    }
                )

    def print_table(title, rows, headers):  # noqa: D103
        print(f"\n{title}")
        print(f"{'norm':<6} {'res':<5} {headers[0]:<15} {headers[1]:<8} {headers[2]:<6} {'max_err':>12}")
        for r in rows:
            print(
                f"{str(r['normalize']):<6} {str(r['residual']):<5} {str(r[headers[0]]):<15} "
                f"{str(r[headers[1]]):<8} {str(r[headers[2]]):<6} {r['err']:.3e}"
            )

    print_table("LayerNorm residual summary", ln_results, headers=("depth", "ok", "err"))
    print_table("Residual stack summary", res_results, headers=("scheme", "depth", "ok"))
