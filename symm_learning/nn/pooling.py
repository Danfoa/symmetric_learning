# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from __future__ import annotations

import torch
from escnn.group import Representation

from symm_learning.linalg import irrep_radii
from symm_learning.representation_theory import direct_sum


class IrrepSubspaceNormPooling(torch.nn.Module):
    r"""Pool irrep features into :math:`\mathbb{G}`-invariant radii.

    Given :math:`\mathbf{x}\in\mathcal{X}` with representation :math:`\rho_{\mathcal{X}}`, the module computes one
    scalar per irreducible copy in the isotypic/irrep-spectral basis:

    .. math::
        r_{k,i} = \lVert \hat{\mathbf{x}}_{k,i} \rVert_2,
        \qquad
        \hat{\mathbf{x}}=\mathbf{Q}^T\mathbf{x}.

    This is exactly :func:`~symm_learning.linalg.irrep_radii`, exposed as a module. The output transforms under a
    direct sum of trivial representations, hence is invariant.

    Args:
        in_rep (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\text{in}}` describing how the input
            last dimension transforms.
    """

    def __init__(self, in_rep: Representation):
        super().__init__()
        G = in_rep.group
        n_inv_features = len(in_rep.irreps)

        self.in_rep = in_rep
        self.out_rep = direct_sum([G.trivial_representation] * n_inv_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute one invariant radius per irreducible copy.

        Args:
            x (:class:`~torch.Tensor`): Input with trailing dimension ``in_rep.size``; any leading batch/time dims are
                accepted.

        Returns:
            :class:`~torch.Tensor`: Tensor with same leading shape as ``x`` and last dim ``out_rep.size`` containing one
                Euclidean norm per irrep block (trivial features).
        """
        assert x.shape[-1] == self.in_rep.size, f"Expected input shape (..., {self.in_rep.size}), but got {x.shape}"
        inv_features = irrep_radii(x, self.in_rep)

        assert inv_features.shape[-1] == self.out_rep.size, f"{self.out_rep.size} != {inv_features.shape[-1]}"
        return inv_features

    def extra_repr(self) -> str:  # noqa: D102
        return f"Irrep Norm Pooling output={self.out_rep.size} {self.in_rep.group}-invariant features"
