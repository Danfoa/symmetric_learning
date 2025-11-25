from __future__ import annotations

# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
import escnn
import numpy as np
import pytest
from escnn.group import CyclicGroup, DihedralGroup, Group, Icosahedral

from symm_learning.representation_theory import direct_sum, isotypic_decomp_rep


@pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10), Icosahedral()])
def test_rep_decomposition(group: Group):
    """Check that the disentangled representation is equivalent to the original representation."""
    rep = direct_sum([group.regular_representation] * 10)
    # Random change of basis.
    P, _ = np.linalg.qr(np.random.randn(rep.size, rep.size).astype(np.float64))
    rep = escnn.group.change_basis(rep, P, name="test_rep")

    rep_iso = isotypic_decomp_rep(rep)

    # Check the two representations are equivalent
    test_elements = [group.sample() for _ in range(min(10, group.order()))]

    for g in test_elements:
        assert np.allclose(rep(g), rep_iso(g), atol=1e-5, rtol=1e-5), (
            f"Representations are not equivalent for element {g}"
        )

    # Check that decomposing the representation twice returns the cached representation
    rep_iso2 = isotypic_decomp_rep(rep)
    assert rep_iso == rep_iso2, "Cached representation is not returned"

    # Check that iso_decomposing twice returns the same representation
    # rep_iso3 = isotypic_decomp_rep(rep_iso)
