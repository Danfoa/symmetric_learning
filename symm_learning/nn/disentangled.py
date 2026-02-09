# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from __future__ import annotations

import torch
from escnn.group import Representation

from symm_learning.representation_theory import direct_sum, isotypic_decomp_rep


class Change2DisentangledBasis(torch.nn.Module):
    r"""Map features to the isotypic/irrep-spectral basis.

    For :math:`\mathbf{x}\in\mathcal{X}` with representation :math:`\rho_{\mathcal{X}}`, this module applies
    :math:`\mathbf{Q}^{-1}` from :func:`~symm_learning.representation_theory.isotypic_decomp_rep`:

    .. math::
        \hat{\mathbf{x}} = \mathbf{Q}^{-1}\mathbf{x},
        \qquad
        \rho_{\mathcal{X}} = \mathbf{Q}\left(
        \bigoplus_{k\in[1,n_{\text{iso}}]}
        \bigoplus_{i\in[1,n_k]}
        \hat{\rho}_k
        \right)\mathbf{Q}^T.

    Hence, coordinates in ``out_rep`` are grouped by isotypic subspace (same irrep type contiguous).
    The map is linear and :math:`\mathbb{G}`-equivariant:

    .. math::
        \hat{\rho}_{\mathcal{X}}(g)\,\hat{\mathbf{x}}
        = \mathbf{Q}^{-1}\rho_{\mathcal{X}}(g)\mathbf{x},
        \quad
        \hat{\rho}_{\mathcal{X}}(g) = \mathbf{Q}^{-1}\rho_{\mathcal{X}}(g)\mathbf{Q}.

    Args:
        in_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{in}}` describing the input
            feature space.
        learnable (:class:`bool`, optional): If ``True``, the change-of-basis matrix is a trainable parameter.
            Defaults to ``False``.
    """

    def __init__(self, in_rep: Representation, learnable: bool = False):
        super().__init__()

        in_rep_iso_basis = isotypic_decomp_rep(in_rep)
        iso_subspaces_reps = in_rep_iso_basis.attributes["isotypic_reps"]
        self.in_rep, self.out_rep = in_rep, direct_sum(list(iso_subspaces_reps.values()))

        Qin2iso = torch.as_tensor(in_rep_iso_basis.change_of_basis_inv, dtype=torch.get_default_dtype())
        identity = torch.eye(Qin2iso.shape[-1], device=Qin2iso.device, dtype=Qin2iso.dtype)
        self._is_in_iso_basis = torch.allclose(Qin2iso, identity, atol=1e-5, rtol=1e-5)

        self._learnable = learnable
        if self._learnable:
            self.Qin2iso = torch.nn.Linear(in_features=self.in_rep.size, out_features=self.out_rep.size, bias=False)
            with torch.no_grad():
                self.Qin2iso.weight.copy_(Qin2iso)
        else:
            self.register_buffer("Qin2iso", Qin2iso)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the basis change to isotypic coordinates.

        Args:
            x (:class:`torch.Tensor`): Input whose last dimension equals ``in_rep.size``; arbitrary leading dimensions
                allowed.

        Returns:
            :class:`torch.Tensor`: Tensor with the same leading shape and last dimension ``out_rep.size``
            (same as ``in_rep``), expressed in the isotypic basis. If the input is already in that basis,
            the tensor is returned unchanged.
        """
        assert x.shape[-1] == self.in_rep.size, f"Expected input shape (..., {self.in_rep.size}), got {x.shape}"
        if self._is_in_iso_basis:
            return x

        if self._learnable:
            return self.Qin2iso(x)

        Q = self.Qin2iso.to(dtype=x.dtype, device=x.device)
        return torch.matmul(x, Q.t())

    def extra_repr(self) -> str:  # noqa: D102
        return f"learnable: {self._learnable}"
