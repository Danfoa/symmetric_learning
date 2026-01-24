from __future__ import annotations

import torch
from escnn.group import Representation
from torch.distributions import MultivariateNormal

from symm_learning.representation_theory import direct_sum


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


class eMultivariateNormal(torch.nn.Module):
    r"""G-equivariant multivariate normal distribution layer.

    A pure PyTorch module parameterizing a :math:`G`-equivariant multivariate Gaussian distribution:

    .. math::

        y \sim \mathcal{N} \bigl( \mu(x), \Sigma(x) \bigr),

    Where the input :math:`x` parameterizes the mean and the (free) degrees of freedom of the covariance matrix,
    constrained to satisfy:

    .. math::

        \rho_Y(g) \mu(x) = \mu(\rho_X(g) \cdot x)

        \rho_Y(g) \Sigma(x) \rho_Y(g)^{\top}= \Sigma(\rho_X(g) x)
        \quad \forall g \in G.

    Such that the distribution is bi-invariant:

    .. math::

        P(y \mid x) = P(\rho_Y(g) y \mid \rho_X(g) x) \quad \forall g \in G.

    The input is composed of the desired mean of the distribution and the log-variances of each irreducible subspace
    of the representation :math:`\rho_Y`. The number of log-variances equals the number of irreducible subspaces
    in the representation.

    Args:
        out_rep: :class:`escnn.group.Representation` describing the distribution's output space (i.e., the
            representation of the mean).
        diagonal: Only diagonal covariance matrices are implemented. These are not necessarily constant multiples of
            identity. Default: ``True``.

    Attributes:
        in_rep: The input representation, a direct sum of ``out_rep`` and ``n_irreps`` trivial representations
            (for the log-variance parameters).
        out_rep: The output representation (same as the input ``out_rep``).
        n_cov_params: Number of independent covariance parameters (equals the number of irreps in ``out_rep``).

    Example:
        >>> from escnn.group import CyclicGroup
        >>> from symm_learning.models.emlp import eMLP
        >>> G = CyclicGroup(3)
        >>> rep_x = G.regular_representation
        >>> rep_y = G.regular_representation
        >>> e_normal = eMultivariateNormal(out_rep=rep_y, diagonal=True)
        >>> # Create an eMLP that outputs mean + cov params
        >>> nn = eMLP(in_rep=rep_x, out_rep=e_normal.in_rep, hidden_units=[32])
        >>> x = torch.randn(1, rep_x.size)
        >>> dist = e_normal(nn(x))  # Returns torch.distributions.MultivariateNormal
        >>> y = dist.sample()  # Sample from the distribution
    """

    def __init__(self, out_rep: Representation, diagonal: bool = True):
        super().__init__()
        if not diagonal:
            raise NotImplementedError("Full covariance matrices are not implemented yet.")
        self.diagonal = diagonal
        self.out_rep = out_rep
        G = out_rep.group

        # ----- irrep metadata ------------------------------------------------
        self.irrep_dims = torch.tensor([G.irrep(*irr).size for irr in out_rep.irreps], dtype=torch.long)
        # index vector that broadcasts irrep-scalars to component level
        idx = [i for i, d in enumerate(self.irrep_dims) for _ in range(d)]
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))
        self.n_cov_params = len(out_rep.irreps)  # Number of params for the covariance matrix

        # ----- change-of-basis (irrep_spectral → user) -----------------------------
        Q = torch.tensor(out_rep.change_of_basis, dtype=torch.get_default_dtype())
        self.register_buffer("Q2_T", (Q.pow(2)).t())  # (n, n) transposed

        # ----- Group action on the degrees of freedom of the Cov matrix ------------
        rep_cov_dof = direct_sum([G.trivial_representation] * len(out_rep.irreps))
        self.in_rep = direct_sum([out_rep, rep_cov_dof])

    def forward(self, input: torch.Tensor) -> MultivariateNormal:
        """Compute the mean and variance and return the equivariant multivariate normal distribution.

        Args:
            input: Tensor of shape ``(..., in_rep.size)`` containing the mean and log-variance parameters.
                The first ``out_rep.size`` elements are the mean, and the remaining ``n_cov_params`` elements are the
                log-variances.

        Returns:
            A :class:`torch.distributions.MultivariateNormal` distribution with diagonal covariance.
        """
        if input.shape[-1] != self.in_rep.size:
            raise ValueError(f"Expected last dimension {self.in_rep.size}, got {input.shape[-1]}")

        if self.diagonal:
            mu, var = _equiv_mean_var_from_input(input, self.idx, self.Q2_T, self.out_rep.size)
        else:
            raise NotImplementedError("Full covariance matrices are not implemented yet.")

        return MultivariateNormal(mu, torch.diag_embed(var))

    def check_equivariance(self, atol: float = 1e-5, rtol: float = 1e-5) -> None:  # noqa: D301
        r"""Verify that the distribution satisfies the equivariance constraint.

        Checks that :math:`P(y | x) = P(\rho_Y(g) y | \rho_X(g) x)` for all group elements.

        Args:
            atol: Absolute tolerance for the equivariance check.
            rtol: Relative tolerance for the equivariance check.

        Raises:
            AssertionError: If the distribution is not equivariant within the given tolerances.
        """
        B = 50
        G = self.out_rep.group

        # Generate random input
        input = torch.randn(B, self.in_rep.size)
        y = torch.randn(B, self.out_rep.size)

        prob_Gy = []
        for g in G.elements:
            # Transform input: x -> rho_in(g) x
            rho_in_g = torch.tensor(self.in_rep(g), dtype=torch.get_default_dtype())
            g_input = input @ rho_in_g.T

            # Transform output: y -> rho_out(g) y
            rho_out_g = torch.tensor(self.out_rep(g), dtype=torch.get_default_dtype())
            gy = y @ rho_out_g.T

            normal = self(g_input)
            prob_Gy.append(normal.log_prob(gy))

        prob_Gy = torch.stack(prob_Gy, dim=1)
        # Check that all probabilities are equal on group orbits
        assert torch.allclose(prob_Gy, prob_Gy.mean(dim=1, keepdim=True), atol=atol, rtol=rtol), (
            "Probabilities are not invariant on group orbits"
        )


if __name__ == "__main__":
    # Example usage

    from escnn.group import CyclicGroup

    from symm_learning.models.emlp import eMLP

    G = CyclicGroup(3)
    rep_x = G.regular_representation
    rep_y = G.regular_representation

    e_normal = eMultivariateNormal(out_rep=rep_y, diagonal=True)

    nn = eMLP(in_rep=rep_x, out_rep=e_normal.in_rep, hidden_units=[32])

    batch_size = 1
    x = torch.randn(batch_size, rep_x.size)
    y = torch.randn(batch_size, rep_y.size)
    params = nn(x)

    prob_Gx = []
    for g in G.elements:
        # Transform input: x -> rho_x(g) x
        rho_x_g = torch.tensor(rep_x(g), dtype=torch.get_default_dtype())
        gx = x @ rho_x_g.T

        # Transform output: y -> rho_y(g) y
        rho_y_g = torch.tensor(rep_y(g), dtype=torch.get_default_dtype())
        gy = y @ rho_y_g.T

        out = nn(gx)
        normal = e_normal(out)
        prob_Gx.append(normal.log_prob(gy))

    prob_Gx = torch.stack(prob_Gx, dim=1)
    # Check that all probabilities are equal on group orbits
    assert torch.allclose(prob_Gx, prob_Gx.mean(dim=1, keepdim=True)), "Probabilities are not equal on group orbits"

    e_normal.check_equivariance(atol=1e-5, rtol=1e-5)
    print("Equivariance check passed!")
