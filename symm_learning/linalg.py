"""Symmetric Learning - Linear Algebra Utilities.

Utility functions for linear algebra operations on symmetric vector spaces with known
group representations.

Functions
---------
lstsq
    Least squares solution constrained to equivariant linear maps.
invariant_orthogonal_projector
    Orthogonal projection onto the G-invariant subspace.
irrep_radii
    Compute Euclidean radius of each irreducible subspace.
isotypic_signal2irreducible_subspaces
    Flatten isotypic signals into irreducible subspace components.
"""

import numpy as np
import torch
from escnn.group import Representation
from torch import Tensor


def isotypic_signal2irreducible_subspaces(x: Tensor, rep_x: Representation):
    r"""Flatten an isotypic signal into its irreducible-subspace coordinates.

    This function assumes :math:`\mathcal{X}` is a single isotypic subspace of type :math:`k`, i.e.

    .. math::
        \rho_{\mathcal{X}} = \bigoplus_{i\in[1,n_k]} \hat{\rho}_k.

    For an input :math:`\mathbf{x}` of shape :math:`(n, n_k \cdot d_k)`, where :math:`d_k=\dim(\hat{\rho}_k)`, the
    output rearranges coordinates to shape :math:`(n \cdot d_k, n_k)` so each column stores one irrep copy across the
    sample axis.

    .. math::
        \mathbf{z}[:, i] = [x_{1,i,1}, \ldots, x_{1,i,d_k}, x_{2,i,1}, \ldots, x_{n,i,d_k}]^\top.

    Args:
        x (:class:`torch.Tensor`): Shape :math:`(n, n_k \cdot d_k)`.
        rep_x (:class:`~escnn.group.Representation`): Representation in isotypic basis with a single active irrep type.

    Returns:
        Tensor: Flattened irreducible-subspace signal of shape :math:`(n \cdot d_k, n_k)`.

    Shape:
        :math:`(n \cdot d_k, n_k)`.
    """
    assert len(rep_x._irreps_multiplicities) == 1, "Random variable is assumed to be in a single isotypic subspace."
    irrep_id = rep_x.irreps[0]
    irrep_dim = rep_x.group.irrep(*irrep_id).size
    mk = rep_x._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X

    Z = x.view(-1, mk, irrep_dim).permute(0, 2, 1).reshape(-1, mk)

    assert Z.shape == (x.shape[0] * irrep_dim, mk)
    return Z


def irrep_radii(x: Tensor, rep: Representation) -> Tensor:
    r"""Compute Euclidean radii for all irreducible-subspace features.

    Let :math:`\rho_{\mathcal{X}}` be the (possibly decomposable) representation of a vector space
    :math:`\mathcal{X}`:

    .. math::
        \rho_{\mathcal{X}} = \mathbf{Q}\left(
        \bigoplus_{k\in[1,n_{\text{iso}}]}
        \bigoplus_{i\in[1,n_k]}
        \hat{\rho}_k
        \right)\mathbf{Q}^T.

    We first change to the irrep-spectral basis induced by this
    :ref:`isotypic decomposition <isotypic-decomposition-example>`
    (as returned by :func:`~symm_learning.representation_theory.isotypic_decomp_rep`),
    :math:`\hat{\mathbf{x}}=\mathbf{Q}^T\mathbf{x}`, and then compute the radius of each irrep copy:

    .. math::
        r_{k,i} = \lVert \hat{\mathbf{x}}_{k,i} \rVert_2.

    Args:
        x: (:class:`torch.Tensor`) of shape :math:`(..., D)` describing vectors transforming according to ``rep``.
        rep: (:class:`~escnn.group.Representation`) acting on the last dimension of ``x``.

    Returns:
        (:class:`torch.Tensor`): Radii of shape :math:`(..., N)` where :math:`N=\texttt{len(rep.irreps)}`. The output
        order follows :attr:`rep.irreps` (one radius per irreducible copy in the decomposition).

    Shape:
        - **Input** ``x``: :math:`(..., D)` with :math:`D=\dim(\rho_{\mathcal{X}})`.
        - **Output**: :math:`(..., N)` containing the per-irrep Euclidean norms.

    Note:
        For repeated calls with the same representation object ``rep``, the matrix
        :math:`\mathbf{Q}^{-1}` is cached in ``rep.attributes["Q_inv"]`` and reused.
    """
    if x.shape[-1] != rep.size:
        raise ValueError(f"Expected last dimension {rep.size}, got {x.shape[-1]}")

    if "Q_inv" not in rep.attributes:  # cache inverse change-of-basis for reuse
        Q_inv = torch.tensor(rep.change_of_basis_inv, device=x.device, dtype=x.dtype)
        rep.attributes["Q_inv"] = Q_inv
    else:
        Q_inv = rep.attributes["Q_inv"].to(x.device, x.dtype)

    # Change to irrep-spectral basis
    x_spectral = torch.einsum("ij,...j->...i", Q_inv, x)
    # Compute a mask for each irreducible subspace (subspace associated to an irrep)
    n_subspaces = len(rep.irreps)
    subspace_ids = _get_irrep_subspace_index(rep).to(x.device)

    flat = x_spectral.reshape(-1, rep.size)
    flat_sq = flat.pow(2)  # squared magnitudes per coordinate

    scatter_idx = subspace_ids.view(1, -1).expand(flat_sq.size(0), -1)  # broadcast labels across batch
    radii_sq = torch.zeros(flat_sq.size(0), n_subspaces, device=x.device, dtype=x.dtype)
    radii_sq.scatter_add_(1, scatter_idx, flat_sq)  # sum squares inside each irrep block

    radii = radii_sq.sqrt().reshape(*x_spectral.shape[:-1], n_subspaces)
    return radii


def lstsq(x: Tensor, y: Tensor, rep_x: Representation, rep_y: Representation):
    r"""Computes a solution to the least squares problem of a system of linear equations with equivariance constraints.

    The :math:`\mathbb{G}`-equivariant least squares problem to the linear system of equations
    :math:`\mathbf{Y} = \mathbf{A}\,\mathbf{X}`, is defined as:

    .. math::
        \begin{align}
            &\operatorname{argmin}_{\mathbf{A}} \| \mathbf{Y} - \mathbf{A}\,\mathbf{X} \|_F \\
            & \text{s.t.} \quad \rho_{\mathcal{Y}}(g) \mathbf{A} = \mathbf{A}\rho_{\mathcal{X}}(g) \quad \forall g
            \in \mathbb{G},
        \end{align}

    where :math:`\mathbf{X}: \Omega \to \mathcal{X}` and :math:`\mathbf{Y}: \Omega \to \mathcal{Y}` are random
    variables taking values in representation spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`, and
    :math:`\rho_{\mathcal{X}}`, :math:`\rho_{\mathcal{Y}}` are the corresponding (possibly decomposable)
    representations of :math:`\mathbb{G}`.

    Args:
        x (:class:`torch.Tensor`): Realizations of the random variable :math:`\mathbf{X}` with shape
            :math:`(N, D_x)`, where :math:`N` is the number of samples.
        y (:class:`torch.Tensor`):
            Realizations of the random variable :math:`\mathbf{Y}` with shape :math:`(N, D_y)`.
        rep_x (:class:`~escnn.group.Representation`):
            Representation :math:`\rho_{\mathcal{X}}` acting on the vector space :math:`\mathcal{X}`.
        rep_y (:class:`~escnn.group.Representation`):
            Representation :math:`\rho_{\mathcal{Y}}` acting on the vector space :math:`\mathcal{Y}`.

    Returns:
        :class:`torch.Tensor`:
            A :math:`(D_y \times D_x)` matrix :math:`\mathbf{A}` satisfying the :math:`\mathbb{G}`-equivariance
            constraint and minimizing :math:`\|\mathbf{Y} - \mathbf{A}\,\mathbf{X}\|^2`.

    Shape:
        - X: :math:`(N, D_x)`
        - Y: :math:`(N, D_y)`
        - Output: :math:`(D_y, D_x)`

    Note:
        This function calls :func:`~symm_learning.representation_theory.isotypic_decomp_rep`, which caches
        decompositions in the group representation registry. Repeated calls with the same representations reuse
        cached decompositions.
    """
    from symm_learning.representation_theory import isotypic_decomp_rep

    #   assert X.shape[0] == Y.shape[0], "Expected equal number of samples in X and Y"
    assert x.shape[1] == rep_x.size, f"Expected X shape (N, {rep_x.size}), got {x.shape}"
    assert y.shape[1] == rep_y.size, f"Expected Y shape (N, {rep_y.size}), got {y.shape}"
    assert x.shape[-1] == rep_x.size, f"Expected X shape (..., {rep_x.size}), got {x.shape}"
    assert y.shape[-1] == rep_y.size, f"Expected Y shape (..., {rep_y.size}), got {y.shape}"

    rep_X_iso = isotypic_decomp_rep(rep_x)
    rep_Y_iso = isotypic_decomp_rep(rep_y)
    # Changes of basis from the Disentangled/Isotypic-basis of X, and Y to the original basis.
    Qx = torch.tensor(rep_X_iso.change_of_basis, device=x.device, dtype=x.dtype)
    Qy = torch.tensor(rep_Y_iso.change_of_basis, device=y.device, dtype=y.dtype)
    rep_X_iso_subspaces = rep_X_iso.attributes["isotypic_reps"]
    rep_Y_iso_subspaces = rep_Y_iso.attributes["isotypic_reps"]

    # Get the dimensions of the isotypic subspaces of the same type in the input/output representations.
    iso_idx_X, iso_idx_Y = {}, {}
    x_dim = 0
    for iso_id, rep_k in rep_X_iso_subspaces.items():
        iso_idx_X[iso_id] = slice(x_dim, x_dim + rep_k.size)
        x_dim += rep_k.size
    y_dim = 0
    for iso_id, rep_k in rep_Y_iso_subspaces.items():
        iso_idx_Y[iso_id] = slice(y_dim, y_dim + rep_k.size)
        y_dim += rep_k.size

    x_iso = torch.einsum("ij,...j->...i", Qx.T, x)
    y_iso = torch.einsum("ij,...j->...i", Qy.T, y)
    A_iso = torch.zeros((rep_y.size, rep_x.size), dtype=x.dtype, device=x.device)
    for iso_id in rep_Y_iso_subspaces:
        if iso_id not in rep_X_iso_subspaces:
            continue  # No covariance between the isotypic subspaces of different types.
        x_k = x_iso[..., iso_idx_X[iso_id]]
        y_k = y_iso[..., iso_idx_Y[iso_id]]
        rep_X_k = rep_X_iso_subspaces[iso_id]
        rep_Y_k = rep_Y_iso_subspaces[iso_id]
        # Compute empirical least-squares.
        A_k_emp = torch.linalg.lstsq(x_k, y_k).solution.T
        A_k = _project_to_irrep_endomorphism_basis(A_k_emp, rep_X_k, rep_Y_k)
        A_iso[iso_idx_Y[iso_id], iso_idx_X[iso_id]] = A_k

    # Change back to the original input output basis sets
    A = Qy @ A_iso @ Qx.T
    return A


def invariant_orthogonal_projector(rep_x: Representation) -> Tensor:
    r"""Computes the orthogonal projection to the invariant subspace.

    The input representation :math:`\rho_{\mathcal{X}}: \mathbb{G} \mapsto \mathbb{G}\mathbb{L}(\mathcal{X})` is
    transformed to the spectral basis given by:

    .. math::
        \rho_{\mathcal{X}} = \mathbf{Q}\left(
        \bigoplus_{k\in[1,n_{\text{iso}}]}
        \bigoplus_{i\in[1,n_k]}
        \hat{\rho}_k
        \right)\mathbf{Q}^T

    where :math:`\hat{\rho}_k` are irreducible representations of :math:`\mathbb{G}`, :math:`n_k` is the multiplicity
    of type :math:`k`, and :math:`\mathbf{Q}: \mathcal{X}\to\mathcal{X}` is the orthogonal change of basis from the
    irrep-spectral basis to the original basis.

    Define the diagonal selector :math:`\mathbf{S}\in\mathbb{R}^{D\times D}` in irrep-spectral coordinates by

    .. math::
        S_{jj} = \begin{cases}
        1, & \text{if coordinate } j \text{ belongs to a trivial irrep copy}, \\
        0, & \text{otherwise}.
        \end{cases}

    Then the orthogonal projector onto the invariant subspace
    :math:`\mathcal{X}^{\text{inv}}=\{\mathbf{x}\in\mathcal{X}: \rho_{\mathcal{X}}(g)\mathbf{x}=\mathbf{x},
    \forall g\in\mathbb{G}\}` is

    .. math::
        \mathbf{P}_{\mathrm{inv}} = \mathbf{Q}\,\mathbf{S}\,\mathbf{Q}^T.

    This projector enforces the invariance constraint:

    .. math::
        \rho_{\mathcal{X}}(g)\,\mathbf{P}_{\mathrm{inv}}
        = \mathbf{P}_{\mathrm{inv}}\,\rho_{\mathcal{X}}(g)
        = \mathbf{P}_{\mathrm{inv}} \quad \forall g\in\mathbb{G}.

    Args:
        rep_x (:class:`~escnn.group.Representation`): The representation for which the orthogonal projection to the
        invariant subspace is computed.

    Returns:
        :class:`torch.Tensor`: The orthogonal projection matrix to the invariant subspace,
        :math:`\mathbf{Q} \mathbf{S} \mathbf{Q}^T`.
    """
    Qx_T, Qx = Tensor(rep_x.change_of_basis_inv), Tensor(rep_x.change_of_basis)

    # S is an indicator of which dimension (in the irrep-spectral basis) is associated with a trivial irrep
    S = torch.zeros((rep_x.size, rep_x.size))
    irreps_dimension = []
    cum_dim = 0
    for irrep_id in rep_x.irreps:
        irrep = rep_x.group.irrep(*irrep_id)
        # Get dimensions of the irrep in the original basis
        irrep_dims = range(cum_dim, cum_dim + irrep.size)
        irreps_dimension.append(irrep_dims)
        if irrep_id == rep_x.group.trivial_representation.id:
            # this dimension is associated with a trivial irrep
            S[irrep_dims, irrep_dims] = 1
        cum_dim += irrep.size

    inv_projector = Qx @ S @ Qx_T
    return inv_projector


def _project_to_irrep_endomorphism_basis(A: Tensor, rep_x: Representation, rep_y: Representation) -> Tensor:
    r"""Projects a linear map A: X -> Y between two isotypic spaces to the space of equivariant linear maps.

    Given a linear map :math:`A: X \to Y`, where :math:`X` and :math:`Y` are isotypic vector spaces of the same type,
    that is, their representations are built from :math:`m_x` and :math:`m_y` copies of the same irrep, this
    function projects the linear map to the space of equivariant linear maps between the two isotypic spaces.

    Args:
        A (:class:`torch.Tensor`): The linear map to be projected, of shape :math:`(m_y \cdot d, m_x \cdot d)`,
            where :math:`d` is the dimension of the irreducible representation, and :math:`m_x` and :math:`m_y`
            are the multiplicities of the irreducible representation in :math:`X` and :math:`Y`, respectively.
        rep_x (:class:`~escnn.group.Representation`): The representation of the isotypic space :math:`X`.
        rep_y (:class:`~escnn.group.Representation`): The representation of the isotypic space :math:`Y`.

    Returns:
        A_equiv (:class:`torch.Tensor`): A projected linear map of shape :math:`(m_y \cdot d, m_x \cdot d)` which
            commutes with the action of the group on the isotypic spaces :math:`X` and :math:`Y`. That is:
            :math:`A_{equiv} \circ \rho_X(g) = \rho_Y(g) \circ A_{equiv}` for all :math:`g \in \mathbb{G}`.
    """
    irrep_id = rep_x.irreps[0]
    irrep = rep_x.group.irrep(*irrep_id)
    assert A.shape == (rep_y.size, rep_x.size), "Expected A: X -> Y"
    assert len(rep_x._irreps_multiplicities) == 1, f"Expected rep with a single irrep type, got {rep_x.irreps}"
    assert len(rep_y._irreps_multiplicities) == 1, f"Expected rep with a single irrep type, got {rep_y.irreps}"
    assert irrep_id == rep_y.irreps[0], f"Irreps {irrep_id} != {rep_y.irreps[0]}. Hence A=0"
    x_in_iso_basis = np.allclose(rep_x.change_of_basis_inv, np.eye(rep_x.size), atol=1e-6, rtol=1e-4)
    assert x_in_iso_basis, "Expected X to be in spectral/isotypic basis"
    y_in_iso_basis = np.allclose(rep_y.change_of_basis_inv, np.eye(rep_y.size), atol=1e-6, rtol=1e-4)
    assert y_in_iso_basis, "Expected Y to be in spectral/isotypic basis"

    m_x = rep_x._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X
    m_y = rep_y._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in Y

    # Get the basis of endomorphisms of the irrep (B, d, d)  B = 1 | 2 | 4
    irrep_end_basis = torch.tensor(irrep.endomorphism_basis(), device=A.device, dtype=A.dtype)
    A_irreps = A.view(m_y, irrep.size, m_x, irrep.size).permute(0, 2, 1, 3).contiguous()
    # Compute basis expansion coefficients of each irrep cross-covariance in basis of End(irrep) ========
    # Frobenius inner product  <C , Ψ_b>  =  Σ_{i,j} C_{ij} Ψ_b,ij
    A_irreps_basis_coeff = torch.einsum("mnij,bij->mnb", A_irreps, irrep_end_basis)  # (m_y , m_x , B)
    # squared norms ‖Ψ_b‖² (only once, very small)
    basis_coeff_norms = torch.einsum("bij,bij->b", irrep_end_basis, irrep_end_basis)  # (B,)
    A_irreps_basis_coeff = A_irreps_basis_coeff / basis_coeff_norms[None, None]

    A_irreps = torch.einsum("...b,bij->...ij", A_irreps_basis_coeff, irrep_end_basis)  # (m_y , m_x , d , d)
    # Reshape to (my * d, mx * d)
    A_equiv = A_irreps.permute(0, 2, 1, 3).reshape(m_y * irrep.size, m_x * irrep.size)
    return A_equiv


def _get_irrep_subspace_index(rep: Representation):
    labels = torch.empty(rep.size, dtype=torch.long)
    irreps_dims = {id: rep.group.irrep(*id).size for id in rep._irreps_multiplicities}
    pos = 0
    for count, irrep_id in enumerate(rep.irreps):
        labels[pos : pos + irreps_dims[irrep_id]] = count  # fill contiguous block for this copy
        pos += irreps_dims[irrep_id]

    return labels
