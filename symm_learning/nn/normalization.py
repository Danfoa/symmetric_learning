import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from symm_learning.representation_theory import isotypic_decomp_rep
from symm_learning.stats import var_mean


class eAffine(EquivariantModule):
    r"""Applies a symmetry-preserving affine transformation to the input :class:`escnn.nn.GeometricTensor`.

    The affine transformation for a given input :math:`x \in \mathcal{X} \subseteq \mathbb{R}^{D_x}` is defined as:

    .. math::
        y = x \cdot \alpha + \beta

    Where :math:`\mathcal{X}` is a symmetric vector space with group representation
    :math:`\rho_{\mathcal{X}}: G \to \mathbb{GL}(D_x)`, :math:`\alpha \in \mathbb{R}^{D_x}` is a learnable
    scale vector that scales uniformly the irreducible subspaces of the feature space, and
    :math:`\beta \in \mathbb{R}^{D_x}` is a bias vector living in the invariant subspace of :math:`\mathcal{X}`.
    """

    def __init__(self, in_type: FieldType, bias: bool = True):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self._rep_in = isotypic_decomp_rep(in_type.representation)
        G = self._rep_in.group

        has_invariant_subspace = G.trivial_representation.id in self._rep_in.attributes["isotypic_reps"]
        self.bias = bias and has_invariant_subspace
        if self.bias:
            inv_subspace_dims: slice = self._rep_in.attributes["isotypic_subspace_dims"][G.trivial_representation.id]
            n_bias_params = inv_subspace_dims.stop - inv_subspace_dims.start
            self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(n_bias_params)))
            inv_projector = torch.tensor(self._rep_in.change_of_basis[:, inv_subspace_dims]).to(self.bias_dof.dtype)
            self.register_buffer("inv_projector", inv_projector)

        # Symmetry-preserving scaling implies scaling each irreducible subspace uniformly.
        # Hence we have only num_irreps free learnable parameters.
        n_scale_params = len(self._rep_in.irreps)
        self.register_parameter("scale_dof", torch.nn.Parameter(torch.ones(n_scale_params)))
        irrep_dims = [G.irrep(*irrep_id).size for irrep_id in self._rep_in.irreps]
        self.register_buffer("_irrep_dims", torch.tensor(irrep_dims, dtype=torch.long))

    def forward(self, x: GeometricTensor):
        """Applies the affine transformation to the input geometric tensor."""
        assert x.type == self.in_type, "Input type does not match the expected input type."

        scale, bias = self.expand_scale(), self.expand_bias() if self.bias else 0
        if x.tensor.ndim == 2:
            scale, bias = scale[None, ...], (bias[None, ...] if self.bias else 0)
        elif x.tensor.ndim == 3:
            scale, bias = scale[None, ..., None], (bias[None, ..., None] if self.bias else 0)
        y = x.tensor * scale + bias
        return self.out_type(y)

    def expand_scale(self):
        """Returns the scale parameter which uniformly scales each irreducible subspace."""
        return torch.repeat_interleave(self.scale_dof, self._irrep_dims, dim=0)

    def expand_bias(self):
        """Returns the bias vector in the invariant subspace of the input type."""
        if self.bias:
            return torch.einsum("ij,j->i", self.inv_projector, self.bias_dof)
        return 0

    def evaluate_output_shape(self, input_shape):  # noqa: D102
        return input_shape

    def extra_repr(self) -> str:  # noqa: D102
        return f"in type: {self.in_type}, bias: {self.bias}"


class eBatchNorm1d(EquivariantModule):
    r"""Applies Batch Normalization over a 2D or 3D symmetric input :class:`escnn.nn.GeometricTensor`.

    Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated **using symmetry-aware estimates** per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.
    At train time in the forward pass, the variance is calculated via the biased estimator,
    equivalent to ``torch.var(input, unbiased=False)``. However, the value stored in the
    moving average of the variance is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

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

        var_batch, mean_batch = var_mean(x.tensor, rep_x=self._rep_x)
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

            var, mean = var_mean(x_batch.tensor, rep_x=self.in_type.representation)
            g_var, g_mean = var_mean(gx_batch.tensor, rep_x=self.in_type.representation)

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
