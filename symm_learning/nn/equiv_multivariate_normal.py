from __future__ import annotations

import escnn
import torch
import torch.nn.functional as F
from escnn.group import directsum
from escnn.nn import EquivariantModule, FieldType


def _equiv_mean_var_from_input(
    input: torch.Tensor,
    idx: torch.Tensor,
    Q2_T: torch.Tensor,
    dim_y: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract mean and variance from the input tensor."""
    mu = input[..., :dim_y]  # (B, n)
    log_eigvals = input[..., dim_y:]  # (B, n_irreps)
    var_irrep_spectral_basis = torch.exp(log_eigvals[..., idx]) + 1e-6  # (B, n)
    var = var_irrep_spectral_basis @ Q2_T  # (B, n)
    return mu, var


class EquivMultivariateNormal(EquivariantModule):
    r"""G-equivariant multivariate normal.

    Utility layer to parameterize a G-equivariant multivariate gaussian/normal distribution
    .. math::
        y \sim \mathcalN \bigl( \mu(x), \Sigma(x) \bigr),
    Where x is the input to the layer parameterizing the mean and (free) degrees of freedom
    of the covariance matrix, constrained to satisfy:
    .. math::
        \rho_Y(g) \mu(x) = \mu(\rho_X(g) \cdot x)
        \rho_Y(g) \Sigma(x) \rho_Y(g)^{\top}= \Sigma(\rho_X(g) x)
        \\quad \forall g \in G.
    Such that:
    .. math::
        P(y \mid x) = P(\rho_Y(g) y \mid \rho_X(g) x) \quad \forall g \in G.
    The input of the layer is composed of the desired mean of the distribution and the
    log-variances of each irreducible subspace of the representation :math:`\rho_Y` of the output.
    The number of log-variances varies with the number of irreducible subspaces of the representation,
    hence this layer is meant to be instantiated before the `EquivaraintModule` that will be used to
    parameterize the multivariate normal distribution. See the example below.
    ------------------------------------------------------------------
    Example - how to use
    ------------------------------------------------------------------
    >>> G = CyclicGroup(3)
    >>> x_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation])
    >>> y_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation] * 1)
    >>> e_normal = EquivMultivariateNormal(y_type, diagonal=True)
    >>> nn = EMLP(in_type=x_type, out_type=e_normal.in_type)
    >>> dist = e_normal.get_distribution(nn(x))
    >>> # Sample from the distribution
    >>> y = dist.sample()  # (B, n)
    ------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------
    y_type : FieldType
        Field/feature type of *X* (the mean).
    diagonal : bool, default ``True``
        Only diagonal covariance matrices are implemented. Note these are not necessarily constant multiples of the
        identity.
    """

    def __init__(self, y_type: FieldType, diagonal=True):
        super().__init__()
        self.y_type = y_type
        self.diagonal = diagonal

        rep_y = y_type.representation
        G = rep_y.group

        if not diagonal:
            raise NotImplementedError("Full covariance matrices are not implemented yet.")

        # ----- irrep metadata ------------------------------------------------
        self.irrep_dims = torch.tensor([G.irrep(*irr).size for irr in rep_y.irreps], dtype=torch.long)
        # index vector that broadcasts irrep-scalars to component level
        idx = [i for i, d in enumerate(self.irrep_dims) for _ in range(d)]
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))
        self.n_cov_params = len(rep_y.irreps)  # Number of params for the covariance matrix
        # ----- change-of-basis (irrep_spectral → user) -----------------------------
        Q = torch.tensor(rep_y.change_of_basis, dtype=torch.get_default_dtype())
        self.register_buffer("Q2_T", (Q.pow(2)).t())  # (n, n) transposed
        # ----- Group action on the degrees of freedom of the Cov matrix ------------
        self.rep_cov_dof = directsum([G.trivial_representation] * len(rep_y.irreps))

        self.in_type = FieldType(y_type.gspace, [rep_y, self.rep_cov_dof])
        self.out_type = self.y_type  # Not used.

    def forward(self, input):
        """Compute the mean and variance of a equivariant multivariate normal distribution

        Args:
            input (FieldType): Input tensor of shape (B, n + n_irreps) where: `B` is the batch size, `n` is the size of
            the output type (mean), `n_irreps` is the number of irreducible representations in the output type
            (covariance degrees of freedom)

        Returns:
            mu (torch.Tensor): Mean of the distribution of shape (B, n).
            var (torch.Tensor): Variance of the distribution of shape (B, n).
        """
        assert input.type == self.in_type, "Input type does not match the expected input type."

        # Extract the mean and covariance from the input
        if self.diagonal:
            mu, var = _equiv_mean_var_from_input(input.tensor, self.idx, self.Q2_T, self.y_type.size)
        else:
            raise NotImplementedError("Full covariance matrices are not implemented yet.")

        return mu, var

    def get_distribution(self, input):
        """Returns the MultivariateNormal distribution."""
        mu, var = self(input)
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(var))

    def evaluate_output_shape(self, input_shape):
        """Output shape are vector of samples from the normal distribution"""
        return input_shape[0], self.y_type.size

    def check_equivariance(self, atol=1e-5, rtol=1e-5):
        """Check equivariance of the module."""
        B = 50
        # Generate random input
        input = torch.randn(B, self.in_type.size)
        y = torch.randn(B, self.y_type.size)
        prob_Gy = []
        for g in self.y_type.fibergroup.elements:
            # Transform the input
            g_input = self.in_type.transform_fibers(input, g)
            gy = self.y_type.transform_fibers(y, g)

            normal = self.get_distribution(self.in_type(g_input))
            prob_Gy.append(normal.log_prob(gy))

        prob_Gy = torch.stack(prob_Gy, dim=1)
        # Check that all probabilities are equal on group orbits
        assert torch.allclose(prob_Gy, prob_Gy.mean(dim=1, keepdim=True), atol=atol, rtol=rtol), (
            "Probabilities are not invariant on group orbits"
        )

    def export(self):
        r"""Export recursively each submodule to a normal PyTorch module and set to "eval" mode."""
        torch_e_normal = tEquivMultivariateNormal(
            idx=self.idx,
            Q2_T=self.Q2_T,
            dim_y=self.y_type.size,
            diagonal=self.diagonal,
        )
        torch_e_normal.eval()  # Set to eval mode
        return torch_e_normal


class tEquivMultivariateNormal(torch.nn.Module):
    """Utility class to export `EquivMultivariateNormal` to a standard PyTorch module."""

    def __init__(
        self,
        idx: torch.Tensor,  # shape (n,)   –  broadcast map irrep→component
        Q2_T: torch.Tensor,  # shape (n, n) –  (Q ** 2).T from escnn
        dim_y: int,  # dim of the output space (= y_type.size)
        diagonal: bool = True,  # only diagonal covariance matrices are implemented
    ):
        super().__init__()
        self.register_buffer("idx", idx.clone())
        self.register_buffer("Q2_T", Q2_T.clone())
        self.dim_y = dim_y
        self.diagonal = diagonal

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of a multivariate normal distribution."""
        if self.diagonal:
            mu, var = _equiv_mean_var_from_input(x, self.idx, self.Q2_T, self.dim_y)
        else:
            raise NotImplementedError("Full covariance matrices are not implemented yet.")
        return mu, var

    def get_distribution(self, x: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Returns the MultivariateNormal distribution."""
        mu, var = self(x)
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(var))

    def check_equivariance(self, in_type, y_type, atol=1e-5, rtol=1e-5):
        """Check equivariance of the module."""
        B = 50
        # Generate random input
        input = torch.randn(B, in_type.size)
        y = torch.randn(B, y_type.size)
        prob_Gy = []
        for g in y_type.fibergroup.elements:
            # Transform the input
            g_input = in_type.transform_fibers(input, g)
            gy = y_type.transform_fibers(y, g)

            normal = self.get_distribution(g_input)
            prob_Gy.append(normal.log_prob(gy))

        prob_Gy = torch.stack(prob_Gy, dim=1)
        # Check that all probabilities are equal on group orbits
        assert torch.allclose(prob_Gy, prob_Gy.mean(dim=1, keepdim=True), atol=atol, rtol=rtol), (
            "Probabilities are not invariant on group orbits"
        )


if __name__ == "__main__":
    # Example usage

    from escnn.group import CyclicGroup, DihedralGroup, Icosahedral
    from torch.distributions import MultivariateNormal

    from symm_learning.models.emlp import EMLP

    G = CyclicGroup(3)
    x_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation])
    y_type = FieldType(escnn.gspaces.no_base_space(G), representations=[G.regular_representation] * 1)

    rep_x = x_type.representation
    G = rep_x.group
    rep_var = directsum([G.trivial_representation] * rep_x.size)

    e_normal = EquivMultivariateNormal(y_type, diagonal=True)

    nn = EMLP(in_type=x_type, out_type=e_normal.in_type)

    batch_size = 1
    x = torch.randn(batch_size, x_type.size)
    y = torch.randn(batch_size, y_type.size)
    n_params = nn(x_type(x))

    prob_Gx = []
    for g in G.elements:
        gx = x_type.transform_fibers(x, g)
        gy = y_type.transform_fibers(y, g)
        out = nn(x_type(gx))
        normal = e_normal.get_distribution(out)
        prob_Gx.append(normal.log_prob(gy))

    prob_Gx = torch.stack(prob_Gx, dim=1)
    # Check that all probabilities are equal on group orbits
    assert torch.allclose(prob_Gx, prob_Gx.mean(dim=1, keepdim=True)), "Probabilities are not equal on group orbits"

    e_normal.check_equivariance(atol=1e-5, rtol=1e-5)

    torch_e_normal: tEquivMultivariateNormal = e_normal.export()
    torch_e_normal.check_equivariance(in_type=e_normal.in_type, y_type=y_type, atol=1e-5, rtol=1e-5)
