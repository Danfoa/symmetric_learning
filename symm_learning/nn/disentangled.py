# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
from __future__ import annotations

import torch
from escnn.group import Representation

from symm_learning.representation_theory import direct_sum, isotypic_decomp_rep


class Change2DisentangledBasis(torch.nn.Module):
    """Change the basis of a representation to its isotypic (disentangled) form.

    Given an input representation, this layer applies the pre-computed change of basis that groups irreducible
    components of the same type together. The transformation is linear and can optionally be made learnable.

    Args:
        in_rep (Representation): Representation describing the input feature space.
        learnable (bool, optional): If ``True``, the change-of-basis matrix is a trainable parameter. Defaults to
            ``False``.
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
        """Apply the change of basis to the disentangled (isotypic) basis.

        Args:
            x (torch.Tensor): Input whose last dimension equals ``in_rep.size``; arbitrary leading dimensions allowed.

        Returns:
            torch.Tensor: Tensor with the same leading shape and last dimension ``out_rep.size`` (same as ``in_rep``)
                expressed in the isotypic basis. If the input is already in that basis, the tensor is returned
                unchanged.
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
