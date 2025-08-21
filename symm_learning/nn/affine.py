import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from symm_learning.stats import var_mean


class eAffine(EquivariantModule):
    r"""Applies a symmetry-preserving affine transformation to the input :class:`escnn.nn.GeometricTensor`.

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
        in_type: the :class:`escnn.nn.FieldType` of the input geometric tensor.
            The output type is the same as the input type.
        bias: a boolean value that when set to ``True``, this module has a learnable bias vector
            in the invariant subspace of the input type. Default: ``True``

    Shape:
        - Input: :math:`(N, D_x)` or :math:`(N, D_x, L)`, where :math:`N` is the batch size,
          :math:`D_x` is the dimension of the input type, and :math:`L` is the sequence length.
        - Output: :math:`(N, D_x)` or :math:`(N, D_x, L)` (same shape as input)
    """

    def __init__(self, in_type: FieldType, bias: bool = True):
        super().__init__()
        self.in_type, self.out_type = in_type, in_type

        self.rep_x = in_type.representation
        G = self.rep_x.group
        self.register_buffer("Q", torch.tensor(self.rep_x.change_of_basis, dtype=torch.get_default_dtype()))
        self.register_buffer("Q_inv", torch.tensor(self.rep_x.change_of_basis_inv, dtype=torch.get_default_dtype()))

        # Symmetry-preserving scaling implies scaling each irreducible subspace uniformly.
        n_scale_params = len(self.rep_x.irreps)
        self.register_parameter("scale_dof", torch.nn.Parameter(torch.ones(n_scale_params)))
        irrep_dims = torch.tensor([G.irrep(*irrep_id).size for irrep_id in self.rep_x.irreps])
        self.register_buffer("irrep_indices", torch.repeat_interleave(torch.arange(len(irrep_dims)), irrep_dims))

        has_invariant_subspace = G.trivial_representation.id in self.rep_x.irreps
        self.has_bias = bias and has_invariant_subspace
        if self.has_bias:
            is_trivial_irrep = torch.tensor([irrep_id == G.trivial_representation.id for irrep_id in self.rep_x.irreps])
            self.register_buffer("inv_dims", torch.repeat_interleave(is_trivial_irrep, irrep_dims))
            n_bias_params = is_trivial_irrep.sum()
            self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(n_bias_params)))

    def forward(self, x: GeometricTensor):
        """Applies the affine transformation to the input geometric tensor."""
        assert x.type == self.in_type, "Input type does not match the expected input type."

        # Handle x: (in_type.size,)
        if x.tensor.ndim == 1:
            x.tensor = x.tensor.unsqueeze(0)  # Add batch dimension

        x_spectral = torch.einsum("ij,bj...->bi...", self.Q_inv, x.tensor)
        scale_spectral = self.scale_dof[self.irrep_indices]
        # Reshape for broadcasting
        scale_spectral = scale_spectral.view(1, -1, *([1] * (x_spectral.ndim - 2)))
        # Apply the scale to the spectral components
        x_spectral = x_spectral * scale_spectral

        y = torch.einsum("ij,bj...->bi...", self.Q, x_spectral)

        y = y + self.bias.view(1, -1, *([1] * (x_spectral.ndim - 2)))

        return self.out_type(y)

    @property
    def scale(self) -> torch.Tensor:
        """Affine scaling that independently uniformly scales each irreducible subspace."""
        return self.scale_dof[self.irrep_indices]

    @property
    def bias(self) -> torch.Tensor:
        """Bias constrained to the invariant subspace of the output type"""
        if self.has_bias:
            bias = torch.einsum("ij,j->i", self.Q[:, self.inv_dims], self.bias_dof)
            return bias
        else:
            return torch.zeros(self.out_type.size, dtype=torch.get_default_dtype())

    def evaluate_output_shape(self, input_shape):  # noqa: D102
        return input_shape

    def extra_repr(self) -> str:  # noqa: D102
        return f"in type: {self.in_type}, bias: {self.has_bias}"
