import torch
from escnn.group import Representation
from torch.nn.utils import parametrize

from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint


class eLinear(torch.nn.Linear):
    """Equivariant Linear layer between two representations."""

    def __init__(self, in_rep: Representation, out_rep: Representation, bias: bool = True):
        super().__init__(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
        self.in_rep = in_rep
        self.out_rep = out_rep
        # Register parametrizations enforcing equivariance
        parametrize.register_parametrization(self, "weight", CommutingConstraint(in_rep, out_rep))
        if bias:
            parametrize.register_parametrization(self, "bias", InvariantConstraint(out_rep))


class eAffine(torch.nn.Module):
    r"""Applies a symmetry-preserving affine transformation y = x * alpha + beta to the input x.

    The affine transformation for a given input :math:`x \in \mathcal{X} \subseteq \mathbb{R}^{D_x}` is defined as:

    .. math::
        \mathbf{y} = \mathbf{x} \cdot \alpha + \beta

    such that

    .. math::
        \rho_{\mathcal{X}}(g) \mathbf{y} = (\rho_{\mathcal{X}}(g) \mathbf{x}) \cdot \alpha + \beta \quad \forall g \in G

    Where :math:`\mathcal{X}` is a symmetric vector space with group representation
    :math:`\rho_{\mathcal{X}}: G \to \mathbb{GL}(D_x)`, and :math:`\alpha \in \mathbb{R}^{D_x}`,
    :math:`\beta \in \mathbb{R}^{D_x}` are symmetry constrained learnable vectors.

    Args:
        in_rep: the :class:`escnn.group.Representation` group representation of the input feature space.
        bias: a boolean value that when set to ``True``, this module has a learnable bias vector
            in the invariant subspace of the input representation Default: ``True``

    Shape:
        - Input: of shape `(..., D)` where :math:`D` is the dimension of the input type.
        - Output: of shape `(..., D)`
    """

    def __init__(self, in_rep: Representation, bias: bool = True):
        super().__init__()
        self.in_rep, self.out_rep = in_rep, in_rep

        self.rep_x = in_rep
        G = self.rep_x.group
        default_dtype = torch.get_default_dtype()
        self.register_buffer("Q", torch.tensor(self.rep_x.change_of_basis, dtype=default_dtype))
        self.register_buffer("Q_inv", torch.tensor(self.rep_x.change_of_basis_inv, dtype=default_dtype))

        # Symmetry-preserving scaling implies scaling each irreducible subspace uniformly.
        irrep_dims_list = [G.irrep(*irrep_id).size for irrep_id in self.rep_x.irreps]
        irrep_dims = torch.tensor(irrep_dims_list, dtype=torch.long)
        n_scale_params = len(irrep_dims_list)
        self.register_parameter("scale_dof", torch.nn.Parameter(torch.ones(n_scale_params, dtype=default_dtype)))
        self.register_buffer(
            "irrep_indices", torch.repeat_interleave(torch.arange(len(irrep_dims), dtype=torch.long), irrep_dims)
        )

        has_invariant_subspace = G.trivial_representation.id in self.rep_x.irreps
        self.has_bias = bias and has_invariant_subspace
        if self.has_bias:
            is_trivial_irrep = torch.tensor(
                [irrep_id == G.trivial_representation.id for irrep_id in self.rep_x.irreps], dtype=torch.bool
            )
            n_bias_params = int(is_trivial_irrep.sum().item())
            self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(n_bias_params, dtype=default_dtype)))
            dim_to_param = torch.full((self.rep_x.size,), -1, dtype=torch.long)
            offset = 0
            bias_idx = 0
            for is_trivial, dim in zip(is_trivial_irrep.tolist(), irrep_dims_list):
                if is_trivial:
                    dim_to_param[offset : offset + dim] = bias_idx
                    bias_idx += 1
                offset += dim
            self.register_buffer("bias_dim_to_param", dim_to_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the affine transformation; works for any input with last dim ``D``."""
        if x.shape[-1] != self.rep_x.size:
            raise ValueError(f"Expected last dimension {self.rep_x.size}, got {x.shape[-1]}")

        x_spectral = torch.einsum("ij,...j->...i", self.Q_inv, x)

        scale = self.scale_dof[self.irrep_indices].to(x_spectral.device, x_spectral.dtype)
        scale = scale.view(*([1] * (x_spectral.ndim - 1)), -1)
        x_spectral = x_spectral * scale

        if self.has_bias and self.bias_dof.numel() > 0:
            bias_index = self.bias_dim_to_param.to(x_spectral.device)
            valid = bias_index >= 0
            if valid.any():
                bias_vals = self.bias_dof.to(x_spectral.device, x_spectral.dtype)
                bias_flat = torch.zeros(self.rep_x.size, device=x_spectral.device, dtype=x_spectral.dtype)
                bias_flat[valid] = bias_vals[bias_index[valid]]
                bias_flat = bias_flat.view(*([1] * (x_spectral.ndim - 1)), -1)
                x_spectral = x_spectral + bias_flat

        y = torch.einsum("ij,...j->...i", self.Q, x_spectral)
        return y

    def extra_repr(self) -> str:  # noqa: D102
        return f"in_rep{self.in_rep} bias={self.has_bias}"
