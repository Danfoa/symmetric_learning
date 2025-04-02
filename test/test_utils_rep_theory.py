# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 12/02/25
import escnn
import numpy as np
import pytest
import torch
from escnn.group import CyclicGroup, DihedralGroup, Group, directsum
from escnn.nn import FieldType


@pytest.mark.parametrize("group", [CyclicGroup(5), DihedralGroup(10)])
def test_rep_decomposition(group: Group):
    """Check that the disentangled representation is equivalent to the original representation."""
    from symm_learning.utils.rep_theory import isotypic_decomp_rep

    rep = directsum([group.regular_representation] * 10)
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
    rep_iso3 = isotypic_decomp_rep(rep_iso)
