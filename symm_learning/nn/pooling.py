# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from __future__ import annotations

import torch
from escnn.group import Representation

from symm_learning.linalg import irrep_radii
from symm_learning.representation_theory import direct_sum


class IrrepSubspaceNormPooling(torch.nn.Module):
    """Pool an irrep-structured feature into invariant magnitudes per isotypic block.

    For an input transforming under ``in_rep``, the layer computes one invariant feature per irrep block by taking its
    Euclidean norm. The output representation is a direct sum of trivial irreps with length equal to the number of
    irreps in ``in_rep``.

    Args:
        in_rep (Representation): Representation describing how the input last dimension transforms.
    """

    def __init__(self, in_rep: Representation):
        super().__init__()
        G = in_rep.group
        n_inv_features = len(in_rep.irreps)

        self.in_rep = in_rep
        self.out_rep = direct_sum([G.trivial_representation] * n_inv_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute invariant norms per irrep block.

        Args:
            x (torch.Tensor): Input with trailing dimension ``in_rep.size``; any leading batch/time dims are accepted.

        Returns:
            torch.Tensor: Tensor with same leading shape as ``x`` and last dim ``out_rep.size`` containing one
                Euclidean norm per irrep block (trivial features).
        """
        assert x.shape[-1] == self.in_rep.size, f"Expected input shape (..., {self.in_rep.size}), but got {x.shape}"
        inv_features = irrep_radii(x, self.in_rep)

        assert inv_features.shape[-1] == self.out_rep.size, f"{self.out_rep.size} != {inv_features.shape[-1]}"
        return inv_features

    def extra_repr(self) -> str:  # noqa: D102
        return f"Irrep Norm Pooling output={self.out_rep.size} {self.in_rep.group}-invariant features"
