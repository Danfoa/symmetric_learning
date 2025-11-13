from __future__ import annotations

import functools
import itertools
from collections import OrderedDict
from typing import Callable, Dict, List, Union

import numpy as np
import torch
from escnn.group import Group, GroupElement, Representation, directsum
from escnn.nn import FieldType
from scipy.linalg import block_diag

from symm_learning.utils import CallableDict


class GroupHomomorphismBasis:
    """TODO"""

    def __init__(self, in_rep: Representation, out_rep: Representation, basis_expansion: str = "memory_heavy") -> None:
        assert in_rep.group == out_rep.group, f"in group: {in_rep.group} != out group: {out_rep.group}"
        self.G = in_rep.group
        self.in_rep = isotypic_decomp_rep(in_rep)
        self.out_rep = isotypic_decomp_rep(out_rep)
        dtype = torch.get_default_dtype()

        if basis_expansion != "memory_heavy":
            raise NotImplementedError(f"Basis expansion '{basis_expansion}' not implemented yet.")

        # Common irreps defining the only non-zero blocks in isotypic basis of Hom_G(in_rep, out_rep)
        self.common_irreps = sorted(set(self.in_rep.irreps).intersection(set(self.out_rep.irreps)))

        self.iso_blocks = {}
        self.basis_elements = []
        dof_offset = 0
        for irrep_id in self.common_irreps:
            irrep = self.G.irrep(*irrep_id)
            irrep_dim = irrep.size
            # Slices defining the location of the isotypic subspace block in Hom_G(in_rep, out_rep)
            out_slice = self.out_rep.attributes["isotypic_subspace_dims"][irrep_id]
            in_slice = self.in_rep.attributes["isotypic_subspace_dims"][irrep_id]
            # Multiplicities of the irrep of type k=irrep_id in in_rep and out_rep
            mul_out = self.out_rep._irreps_multiplicities[irrep_id]
            mul_in = self.in_rep._irreps_multiplicities[irrep_id]
            # Endomorphism basis of the irrep End_G(k=irrep_id)
            irrep_end_basis = torch.tensor(irrep.endomorphism_basis(), dtype=dtype)  # [D_k, d_k, d_k]
            dim_end_basis = irrep_end_basis.size(0)
            # Normalize basis elements for orthogonal projections.
            irrep_end_basis = (
                irrep_end_basis / torch.einsum("dij,dij->d", irrep_end_basis, irrep_end_basis).sqrt()[:, None, None]
            )
            dim_hom_basis = mul_out * mul_in * irrep_end_basis.size(0)  # dim(Hom_G(V_k^{in}, V_k^{out}))
            # Store block info for inference time use.
            self.iso_blocks[irrep_id] = dict(
                out_slice=out_slice,
                in_slice=in_slice,
                endomorphism_basis=irrep_end_basis,
                mul_out=mul_out,
                mul_in=mul_in,
                irrep_dim=irrep.size,
                dim_hom_basis=dim_hom_basis,
                hom_basis_slice=slice(dof_offset, dof_offset + dim_hom_basis),
            )
            dof_offset += dim_hom_basis
            # Construct the basis of Hom_G(V_k^{in}, V_k^{out}) of shape [S_k * m_out * m_in, Ny * Nx]
            # This is memory-intensive but the fastest way for forward/backward passes.
            irrep_basis_elements = []
            for s in range(dim_end_basis):
                irrep_end_basis = irrep_end_basis[s]  # [d_k, d_k]
                for o_mul in range(mul_out):
                    for i_mul in range(mul_in):
                        basis_block = torch.zeros(self.out_rep.size, self.in_rep.size, dtype=dtype)  # [Ny, Nx]
                        row_start = out_slice.start + o_mul * irrep_dim
                        col_start = in_slice.start + i_mul * irrep_dim
                        basis_block[row_start : row_start + irrep_dim, col_start : col_start + irrep_dim] = (
                            irrep_end_basis
                        )
                        irrep_basis_elements.append(basis_block.reshape(-1))
            # Every block contributes exactly S_k * m_out * m_in rows; no need for an emptiness check.
            self.basis_elements.append(torch.stack(irrep_basis_elements, dim=0))

        self.basis_elements = torch.cat(self.basis_elements, dim=0)  # [dim(Hom_G(in_rep, out_rep)), Ny * Nx]

    @property
    def dimension(self) -> int:
        """Return the total number of degrees of freedom of Hom_G(out_rep, in_rep)."""
        return self.basis_elements.size(0)


def isotypic_decomp_rep(rep: Representation) -> Representation:
    r"""Return an equivalent representation disentangled into isotypic subspaces.

    Given an input :class:`~escnn.group.Representation`, this function computes an
    equivalent representation by updating the change of basis (and its inverse)
    and reordering the irreducible representations. The returned representation
    is guaranteed to be disentangled into its isotypic subspaces.

    A representation is considered disentangled if, in its spectral basis, the
    irreducible representations (irreps) are clustered by type, i.e., all
    irreps of the same type are consecutive:

    .. math::
        \rho_{\mathcal{X}} = \mathbf{Q} \left( \bigoplus_{k\in[1,n_{\text{iso}}]} (\mathbf{I}_{m_{k}} \otimes
        \hat{\rho}_k) \right) \mathbf{Q}^T

    where :math:`\hat{\rho}_k` is the irreducible representation of type :math:`k`,
    and :math:`m_{k}` is its multiplicity.

    The change of basis of the representation returned by this function can be
    used to decompose the representation space :math:`\mathcal{X}` into its
    orthogonal isotypic subspaces:

    .. math::
        \mathcal{X} = \bigoplus_{k\in[1,n]} \mathcal{X}^{(k)}

    Args:
        rep (escnn.group.Representation): The input representation.

    Returns:
        escnn.group.Representation: An equivalent, disentangled representation.
    """
    # If group representation is already disentangled (in isotypic basis), return it without changes.
    if rep.attributes.get("in_isotypic_basis", False):
        return rep

    symm_group = rep.group

    if rep.name + "-Iso" in symm_group.representations:
        print(f"Returning cached {rep.name}-Iso")
        return symm_group.representations[rep.name + "-Iso"]

    potential_irreps = rep.group.irreps()
    isotypic_subspaces_indices = {irrep.id: [] for irrep in potential_irreps}

    for pot_irrep in potential_irreps:
        cur_dim = 0
        for rep_irrep_id in rep.irreps:
            rep_irrep = symm_group.irrep(*rep_irrep_id)
            if rep_irrep == pot_irrep:
                isotypic_subspaces_indices[rep_irrep_id].append(list(range(cur_dim, cur_dim + rep_irrep.size)))
            cur_dim += rep_irrep.size

    # Remove inactive Isotypic Spaces
    for irrep in potential_irreps:
        if len(isotypic_subspaces_indices[irrep.id]) == 0:
            del isotypic_subspaces_indices[irrep.id]

    # Each Isotypic Space will be indexed by the irrep it is associated with.
    active_isotypic_reps = {}
    for irrep_id, indices in isotypic_subspaces_indices.items():
        irrep = symm_group.irrep(*irrep_id)
        multiplicities = len(indices)
        active_isotypic_reps[irrep_id] = Representation(
            group=rep.group,
            irreps=[irrep_id] * multiplicities,
            name=f"IsoSubspace {irrep_id}",
            change_of_basis=np.identity(irrep.size * multiplicities),
            supported_nonlinearities=irrep.supported_nonlinearities,
        )
        # Store attributes used in stats/linalg operations.
        active_isotypic_reps[irrep_id].attributes["is_isotypic_rep"] = True
        active_isotypic_reps[irrep_id].attributes["irrep_multiplicity"] = multiplicities
        active_isotypic_reps[irrep_id].attributes["irrep_dim"] = irrep.size
        active_isotypic_reps[irrep_id].attributes["irrep_endomorphism_basis"] = torch.tensor(irrep.endomorphism_basis())

    # Impose canonical order on the Isotypic Subspaces.
    # If the trivial representation is active it will be the first Isotypic Subspace.
    # Then sort by dimension of the space from smallest to largest.
    ordered_isotypic_reps = OrderedDict(sorted(active_isotypic_reps.items(), key=lambda item: item[1].size))
    if symm_group.trivial_representation.id in ordered_isotypic_reps:
        ordered_isotypic_reps.move_to_end(symm_group.trivial_representation.id, last=False)

    # Required permutation to change the order of the irreps. So we obtain irreps of the same type consecutively.
    oneline_permutation = []
    for irrep_id, iso_rep in ordered_isotypic_reps.items():
        idx = isotypic_subspaces_indices[irrep_id]
        oneline_permutation.extend(idx)
    oneline_permutation = np.concatenate(oneline_permutation)
    P_in2iso = permutation_matrix(oneline_permutation)

    Q_iso = rep.change_of_basis @ P_in2iso.T
    rep_iso_basis = directsum(list(ordered_isotypic_reps.values()), name=rep.name + "-Iso", change_of_basis=Q_iso)

    # Get variable of indices of isotypic subspaces in the disentangled representation.
    d = 0
    iso_subspace_dims = {}
    for irrep_id, iso_rep in ordered_isotypic_reps.items():
        iso_subspace_dims[irrep_id] = slice(d, d + iso_rep.size)
        d += iso_rep.size

    iso_supported_nonlinearities = [iso_rep.supported_nonlinearities for iso_rep in ordered_isotypic_reps.values()]
    rep_iso_basis.supported_nonlinearities = functools.reduce(set.intersection, iso_supported_nonlinearities)
    rep_iso_basis.attributes["isotypic_reps"] = ordered_isotypic_reps
    rep_iso_basis.attributes["isotypic_subspace_dims"] = iso_subspace_dims
    rep_iso_basis.attributes["in_isotypic_basis"] = True  # Boolean flag useful

    return rep_iso_basis


def field_type_to_isotypic_basis(field_type: FieldType):
    """Returns a new field type in a disentangled basis ignoring change of basis."""
    rep = field_type.representation
    # Organize the irreps such that we get: rep_ordered_irreps := Q (⊕_k (⊕_i^mk irrep_k)) Q^T
    rep_ordered_irreps = isotypic_decomp_rep(rep)
    # Get dictionary of irrep_id: (⊕_i^mk irrep_k)
    iso_subspaces_reps = rep_ordered_irreps.attributes["isotypic_reps"]
    # Define a field type composed of the representations of each isotypic subspace
    new_field_type = FieldType(gspace=field_type.gspace, representations=list(iso_subspaces_reps.values()))
    return new_field_type


def permutation_matrix(oneline_notation):
    """Generate a permutation matrix from its oneline notation."""
    d = len(oneline_notation)
    assert d == np.unique(oneline_notation).size, "oneline_notation must describe a non-defective permutation"
    P = np.zeros((d, d), dtype=int)
    P[range(d), np.abs(oneline_notation)] = 1
    return P


def irreps_stats(irreps_ids):
    """Compute the unique irreps, their counts and indices in the input list of irreps ids."""
    str_ids = [str(irrep_id) for irrep_id in irreps_ids]
    unique_str_ids, counts, indices = np.unique(str_ids, return_counts=True, return_index=True)
    unique_ids = [eval(s) for s in unique_str_ids]
    return unique_ids, counts, indices


def escnn_representation_form_mapping(
    group: Group,
    rep: Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]],
    name: str = "reconstructed",
):
    r"""Get a ESCNN representation instance from a mapping from group elements to unitary matrices.

    Args:
        group (:class:`escnn.group.Group`): Symmetry group of the representation.
        rep (Union[Dict[escnn.group.GroupElement, np.ndarray], Callable[[escnn.group.GroupElement], np.ndarray]]):
            Mapping from group elements to unitary matrices.
        name (str, optional): Name of the representation. Defaults to 'reconstructed'.

    Returns:
        representation (escnn.group.Representation): ESCNN representation instance.
    """
    if isinstance(rep, dict):
        from symm_learning.utils import CallableDict

        rep = CallableDict(rep)
    else:
        rep = rep

    # Find Q such that `iso_cplx(g) = Q @ rep(g) @ Q^-1` is block diagonal with blocks being complex irreps.
    cplx_irreps, Q = cplx_isotypic_decomposition(group, rep)
    # Get the size and location of each cplx irrep in `iso_cplx(g)`
    cplx_irreps_size = [irrep(group.sample()).shape[0] for irrep in cplx_irreps]
    irrep_dim_start = np.cumsum([0] + cplx_irreps_size[:-1])
    # Compute the character table of the found complex irreps and of all complex irreps of G
    irreps_char_table = compute_character_table(group, cplx_irreps)

    # We need to identify which real ESCNN irreps are present in rep(g).
    # First, we decompose the Group's ESCNN real irreps into complex irreps.
    escnn_cplx_irreps_data = {}
    for re_irrep in group.irreps():
        # Find Q_sub s.t. `block_diag([cplx_irrep_i1(g), cplx_irrep_i2(g)...]) = Q @ re_irrep_i(g) @ Q^-1`
        irreps, Q_sub = cplx_isotypic_decomposition(group, re_irrep)
        char_table = compute_character_table(group, irreps)
        escnn_cplx_irreps_data[re_irrep] = dict(subreps=irreps, Q=Q_sub, char_table=char_table)

    # Then, we find which of the Group complex irreps are present in the input representation, and determine
    # each group complex irrep multiplicity. As the complex irreps forming a real irrep can be spread over the
    # dimensions of the input rep, we find a permutation matrix P such that all complex irreps associated with a real
    # irrep are contiguous in dimensions.trifinger
    oneline_perm, Q_isore2isoimg = [], []
    escnn_real_irreps = []
    for escnn_irrep, data in escnn_cplx_irreps_data.items():
        # Match complex irreps by their character tables.
        multiplicities, irrep_locs = map_character_tables(data["char_table"], irreps_char_table)
        subreps_start_dims = [irrep_dim_start[i] for i in irrep_locs]  # Identify start of blocks in `rep(g)`
        data.update(multiplicities=multiplicities, subrep_start_dims=subreps_start_dims)
        assert np.unique(multiplicities).size == 1, "Multiplicities error"
        multiplicity = multiplicities[0]
        for m in range(multiplicity):
            Q_isore2isoimg.append(data["Q"])  # Add transformation from Real irrep to complex irrep
            escnn_real_irreps.append(escnn_irrep)  # Add escnn irrep to the list for instanciation
            for subrep, rep_start_dims in zip(data["subreps"], subreps_start_dims):
                rep_size = subrep[group.sample()].shape[0] if isinstance(subrep, dict) else subrep.size
                oneline_perm += list(range(rep_start_dims[m], rep_start_dims[m] + rep_size))
    # As the complex irreps forming a real irrep can be spread over the dimensions of the input rep, we find a
    # permutation matrix P such that all complex irreps of a real irrep are contiguous in dimensions / in the same block
    P = permutation_matrix(oneline_notation=oneline_perm)
    # Then we use the known transformations `Q_sub` for each real irrep, to create a mapping from cplx to real irreps.
    # s.t. `iso_re(g) = Q_iso_cplx2iso_re @ block_diag([cplx_irrep_11(g),...,cplx_irrep_ij(g)]) @ Q_iso_cplx2iso_re^-1`
    Q_iso_cplx2iso_re = block_diag(*[Q_sub.conj().T for Q_sub in Q_isore2isoimg])

    # Assert the matrix `P` and `Q_iso_cplx2iso_re` turn complex irreps into real irreps
    # `iso_re(g) = (Q_iso_cplx2iso_re @ P) @ iso_cplx(g) @ (Q_iso_cplx2iso_re @ P)^-1`,
    for g in group.elements:
        iso_re_g = block_diag(*[irrep(g) for irrep in escnn_real_irreps])
        iso_cplx_g = block_diag(*[cplx_irrep(g) for cplx_irrep in cplx_irreps])
        rec_iso_re_g = (Q_iso_cplx2iso_re @ P) @ iso_cplx_g @ (Q_iso_cplx2iso_re @ P).conj().T
        error = np.abs(iso_re_g - rec_iso_re_g)
        assert np.isclose(error, 0).all(), "Error in the conversion of Real irreps to Complex irreps"

    # Now we have an orthogonal transformation between the input `rep` and `iso_re`.
    #                        |     iso_cplx(g)     |
    # (Q_iso_cplx2iso_re @ P @ Q) @ rep(g) @ (Q^-1 @ P^-1 @ Q_iso_cplx2iso_re^-1) = Q_re @ rep(g) @ Q_re^-1 = iso_re(g)
    Q_re = Q_iso_cplx2iso_re @ P @ Q

    assert np.allclose(Q_re @ Q_re.conj().T, np.eye(Q_re.shape[0])), "Q_re is not an orthogonal transformation"
    if np.allclose(np.imag(Q_re), 0):
        Q_re = np.real(Q_re)  # Remove numerical noise and ensure rep(g) is of dtype: float instead of cfloat

    # Then we have that `Q_re^-1 @ iso_re(g) @ Q_re = rep(g)`
    reconstructed_rep = Representation(
        group, name=name, irreps=[irrep.id for irrep in escnn_real_irreps], change_of_basis=Q_re.conj().T
    )

    # Test ESCNN reconstruction
    for g in group.elements:
        g_true, g_rec = rep(g), reconstructed_rep(g)
        error = np.abs(g_true - g_rec)
        error[error < 1e-10] = 0
        assert np.allclose(error, 0), f"Reconstructed rep do not match input rep. g={g}, error:\n{error}"
        assert np.allclose(np.imag(g_rec), 0), f"Reconstructed rep not real for g={g}: \n{g_rec}"

    return reconstructed_rep


def is_complex_irreducible(
    group: Group, rep: Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]]
):
    """Check if a representation is complex irreducible.

    We check this by asserting weather non-scalar (no multiple of
    identity ) Hermitian matrix `H` exists, such that `H` commutes with all group elements' representation.
    If rho is irreducible, this function returns (True, H=I)  where I is the identity matrix.
    Otherwise, returns (False, H) where H is a non-scalar matrix that commutes with all elements' representation.

    Args:
        group (:class:`escnn.group.Group`): Symmetry group of the representation.
        rep (Union[Dict[escnn.group.GroupElement, np.ndarray], Callable[[escnn.group.GroupElement], np.ndarray]]):
            Mapping from group elements to their representation matrices.

    """
    if isinstance(rep, dict):

        def rep(g):
            return rep[g]

    else:
        rep = rep

    # Compute the dimension of the representation
    n = rep(group.sample()).shape[0]

    # Run through all r,s = 1,2,...,n
    for r in range(n):
        for s in range(n):
            # Define H_rs
            H_rs = np.zeros((n, n), dtype=complex)
            if r == s:
                H_rs[r, s] = 1
            elif r > s:
                H_rs[r, s] = 1
                H_rs[s, r] = 1
            else:  # r < s
                H_rs[r, s] = 1j
                H_rs[s, r] = -1j

            # Compute H
            H = sum([rep(g).conj().T @ H_rs @ rep(g) for g in group.elements]) / group.order()

            # If H is not a scalar matrix, then it is a matrix that commutes with all group actions.
            if not np.allclose(H[0, 0] * np.eye(H.shape[0]), H):
                return False, H
    # No Hermitian matrix was found to commute with all group actions. This is an irreducible rep
    return True, np.eye(n)


def decompose_representation(
    G: Group, rep: Union[Dict[GroupElement, np.ndarray], Callable[[GroupElement], np.ndarray]]
):
    r"""Find the Hermitian matrix `Q` that block-diagonalizes the representation `rep` of group `G`.

    Such that

    .. math::
        \mathbf{Q} \rho(g) \mathbf{Q}^\top = \operatorname{block\_ diag}(\rho_1(g), ..., \rho_m(g))
        \quad \forall g \in G

    Args:
        G (:class:`escnn.group.Group`): The symmetry group.
        rep (Union[Dict[escnn.group.GroupElement, np.ndarray], Callable[[escnn.group.GroupElement], np.ndarray]]):
            The representation to decompose.

    Returns:
        tuple: A tuple containing:
            - subreps (List[CallableDict]): A list of irreducible representations.
            - Q (np.ndarray): The change of basis matrix.

    """
    import networkx as nx
    from networkx import Graph

    eps = 1e-12
    if isinstance(rep, dict):

        def rep(g):
            return rep[g]

    else:
        rep = rep
    # Compute the dimension of the representation
    n = rep(G.sample()).shape[0]

    for g in G.elements:  # Ensure the representation is unitary/orthogonal
        error = np.abs((rep(g) @ rep(g).conj().T) - np.eye(n))
        assert np.allclose(error, 0), f"Rep {rep} is not unitary: rep(g)@rep(g)^H=\n{(rep(g) @ rep(g).conj().T)}"

    # Find Hermitian matrix non-scalar `H` that commutes with all group actions
    is_irred, H = is_complex_irreducible(G, rep)
    if is_irred:
        return [rep], np.eye(n)

    # Eigen-decomposition of matrix `H = P·A·P^-1` reveals the G-invariant subspaces/eigenspaces of the representations.
    eivals, eigvects = np.linalg.eigh(H, UPLO="L")
    P = eigvects.conj().T
    assert np.allclose(P.conj().T @ np.diag(eivals) @ P, H)

    # Eigendcomposition is not guaranteed to block_diagonalize the representation. An additional permutation of the
    # rows and columns od the representation might be needed to produce a Jordan block canonical form.
    # First: We want to identify the diagonal blocks. To find them we use the trick of thinking of the representation
    # as an adjacency matrix of a graph. The non-zero entries of the adjacency matrix are the edges of the graph.
    edges = set()
    decomposed_reps = {}
    for g in G.elements:
        diag_rep = P @ rep(g) @ P.conj().T  # Obtain block diagonal representation
        diag_rep[np.abs(diag_rep) < eps] = 0  # Remove rounding errors.
        non_zero_idx = np.nonzero(diag_rep)
        edges.update([(x_idx, y_idx) for x_idx, y_idx in zip(*non_zero_idx)])
        decomposed_reps[g] = diag_rep

    # Each connected component of the graph is equivalent to the rows and columns determining a block in the diagonal
    graph = Graph()
    graph.add_edges_from(set(edges))
    connected_components = [sorted(list(comp)) for comp in nx.connected_components(graph)]
    connected_components = sorted(connected_components, key=lambda x: (len(x), min(x)))  # Impose a canonical order
    # If connected components are not adjacent dimensions, say subrep_1_dims = [0,2] and subrep_2_dims = [1,3] then
    # We permute them to get a jordan block canonical form. I.e. subrep_1_dims = [0,1] and subrep_2_dims = [2,3].
    oneline_notation = list(itertools.chain.from_iterable([list(comp) for comp in connected_components]))
    PJ = permutation_matrix(oneline_notation=oneline_notation)
    # After permuting the dimensions, we can assume the components are ordered in dimension
    ordered_connected_components = []
    idx = 0
    for comp in connected_components:
        ordered_connected_components.append(tuple(range(idx, idx + len(comp))))
        idx += len(comp)
    connected_components = ordered_connected_components

    # The output of connected components is the set of nodes/row-indices of the rep.
    subreps = [CallableDict() for _ in connected_components]
    for g in G.elements:
        for comp_id, comp in enumerate(connected_components):
            block_start, block_end = comp[0], comp[-1] + 1
            # Transform the decomposed representation into the Jordan Cannonical Form (jcf)
            jcf_rep = PJ @ decomposed_reps[g] @ PJ.T
            # Check Jordan Cannonical Form TODO: Extract this to a utils. function
            above_block = jcf_rep[0:block_start, block_start:block_end]
            below_block = jcf_rep[block_end:, block_start:block_end]
            left_block = jcf_rep[block_start:block_end, 0:block_start]
            right_block = jcf_rep[block_start:block_end, block_end:]

            assert np.allclose(above_block, 0) or above_block.size == 0, "Non zero elements above block"
            assert np.allclose(below_block, 0) or below_block.size == 0, "Non zero elements below block"
            assert np.allclose(left_block, 0) or left_block.size == 0, "Non zero elements left of block"
            assert np.allclose(right_block, 0) or right_block.size == 0, "Non zero elements right of block"
            sub_g = jcf_rep[block_start:block_end, block_start:block_end]
            subreps[comp_id][g] = sub_g

    # Decomposition to Jordan Canonical form is accomplished by (PJ @ P) @ rep @ (PJ @ P)^-1
    Q = PJ @ P

    # Test decomposition.
    for g in G.elements:
        jcf_rep = block_diag(*[subrep[g] for subrep in subreps])
        error = np.abs(jcf_rep - (Q @ rep(g) @ Q.conj().T))
        assert np.allclose(error, 0), f"Q @ rep[g] @ Q^-1 != block_diag[{[f'rep{i},' for i in range(len(subreps))]}]"

    return subreps, Q


def compute_character_table(G: Group, reps: List[Union[Dict[GroupElement, np.ndarray], Representation]]):
    """Computes the character table of a group for a given set of representations.

    Args:
        G (:class:`escnn.group.Group`): The symmetry group.
        reps (List[Union[Dict[escnn.group.GroupElement, np.ndarray], escnn.group.Representation]]): The representations
          to compute the character table for.

    """
    n_reps = len(reps)
    table = np.zeros((n_reps, G.order()), dtype=complex)
    for i, rep in enumerate(reps):
        for j, g in enumerate(G.elements):
            table[i, j] = rep.character(g) if isinstance(rep, Representation) else np.trace(rep(g))
    return table


def map_character_tables(in_table: np.ndarray, reference_table: np.ndarray):
    """Find a representation of a group in the set of irreducible representations."""
    n_in_reps = in_table.shape[0]
    out_ids, multiplicities = [], []
    for in_id in range(n_in_reps):
        character_orbit = in_table[in_id, :]
        orbit_error = np.isclose(np.abs(reference_table - character_orbit), 0)
        match_idx = np.argwhere(np.all(orbit_error, axis=1)).flatten()
        multiplicity = len(match_idx)
        out_ids.append(match_idx), multiplicities.append(multiplicity)
    return multiplicities, out_ids


def cplx_isotypic_decomposition(G: Group, rep: Callable[[GroupElement], np.ndarray]):
    """Perform the isotypic decomposition of unitary representation, decomposing the rep into complex irreps.

    Args:
        G (:class:`escnn.group.Group`): Symmetry group of the representation.
        rep (Union[Dict[escnn.group.GroupElement, np.ndarray], Callable[[escnn.group.GroupElement, np.ndarray]]]):
            dict/mapping from group elements to matrices, or a function that takes a group element and returns a matrix.

    Returns:
        sorted_irreps (List[Dict[escnn.group.GroupElement, np.ndarray]]): List of complex irreducible representations,
        sorted in ascending order of dimension.
        Q (:class:`numpy.ndarray`): Hermitian matrix such that Q @ rep[g] @ Q^-1 is block diagonal, with blocks
        `sorted_irreps`.

    """
    if isinstance(rep, dict):

        def rep(g):
            return rep[g]

    else:
        rep = rep

    n = rep(G.sample()).shape[0]
    subreps, Q_internal = decompose_representation(G, rep)

    found_irreps = []
    Qs = []

    # Check if each subrepresentation can be further decomposed.
    for subrep in subreps:
        n_sub = subrep(G.sample()).shape[0]  # Dimension of sub representation
        is_irred, _ = is_complex_irreducible(G, subrep)
        if is_irred:
            found_irreps.append(subrep)
            Qs.append(np.eye(n_sub))
        else:
            # Find Q_sub such that Q_sub @ subrep[g] @ Q_sub^-1 is block diagonal, with blocks `sub_subrep`
            sub_subreps, Q_sub = cplx_isotypic_decomposition(G, subrep)
            found_irreps += sub_subreps
            Qs.append(Q_sub)

    # Sort irreps by dimension.
    P, sorted_irreps = sorted_jordan_canonical_form(G, found_irreps)

    # If subreps were decomposable, then these get further decomposed with an additional Hermitian matrix such that:
    # Q @ rep[g] @ Q^-1 = block_diag[irreps] | Q = (Q_external @ Q_internal)
    Q_external = block_diag(*Qs)
    Q = P @ Q_external @ Q_internal

    # Test isotypic decomposition.
    assert np.allclose(Q @ Q.conj().T, np.eye(n)), "Q is not unitary."
    for g in G.elements:
        g_iso = block_diag(*[irrep[g] if isinstance(irrep, dict) else irrep(g) for irrep in sorted_irreps])
        error = np.abs(g_iso - (Q @ rep(g) @ Q.conj().T))
        assert np.allclose(error, 0), f"Q @ rep[g] @ Q^-1 != block_diag[irreps[g]], for g={g}. Error \n:{error}"

    return sorted_irreps, Q


def sorted_jordan_canonical_form(group: Group, reps: List[Callable[[GroupElement], np.ndarray]]):
    """Sorts a list of representations in ascending order of dimension, and returns a permutation matrix P such that.

    Args:
        group (:class:`escnn.group.Group`): Symmetry group of the representation.
        reps (List[Union[Callable[[escnn.group.GroupElement], np.ndarray], escnn.group.Representation]]):
            List of representations to sort by dimension.

    Returns:
        P (:class:`numpy.ndarray`): Permutation matrix sorting the input reps.
        reps (List[Callable[:class:`escnn.group.GroupElement`, :class:`numpy.ndarray`]]): Sorted list of
        representations.
    """
    reps_idx = range(len(reps))
    reps_size = [rep(group.sample()).shape[0] for rep in reps]
    sort_order = sorted(reps_idx, key=lambda idx: reps_size[idx])
    if sort_order == list(reps_idx):
        return np.eye(sum(reps_size)), reps
    irrep_dim_start = np.cumsum([0] + reps_size[:-1])
    oneline_perm = []
    for idx in sort_order:
        rep_size = reps_size[idx]
        oneline_perm += list(range(irrep_dim_start[idx], irrep_dim_start[idx] + rep_size))
    P = permutation_matrix(oneline_perm)

    return P, [reps[idx] for idx in sort_order]
