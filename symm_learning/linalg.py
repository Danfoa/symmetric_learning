"""Symmetric Learning - Linear Algebra Utilities.

Utility functions for linear algebra operations on symmetric vector spaces with known
group representations.

Functions
---------
lstsq
    Least squares solution constrained to equivariant linear maps.
invariant_orthogonal_projector
    Orthogonal projection onto the G-invariant subspace.
equiv_orthogonal_projection_coefficients
    Orthogonal projection onto Hom_G returned in the flattened homomorphism basis.
equiv_linear_map
    Construct a dense equivariant linear map from flattened homomorphism-basis coefficients.
equiv_orthogonal_projection
    Orthogonal projection onto Hom_G using precomputed isotypic-basis tensors.
irrep_radii
    Compute Euclidean radius of each irreducible subspace.
isotypic_signal2irreducible_subspaces
    Flatten isotypic signals into irreducible subspace components.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, TypedDict

import numpy as np
import torch
from escnn.group import Representation

from symm_learning.utils import _cached_rep_matrix


class IsoSpaceProjection(TypedDict):
    """Per-irrep projection metadata returned by :func:`project_in_isobasis`."""

    coeff: torch.Tensor
    endo_basis_flat: torch.Tensor
    m_out: int
    m_in: int
    d_k: int
    out_slice: slice
    in_slice: slice


class IsotypicTensorCache(NamedTuple):
    """Reusable tensors for repeated isotypic-basis projection and synthesis calls."""

    Q_out: torch.Tensor
    Q_in_inv: torch.Tensor
    endo_basis_flat: dict[tuple[int, ...], torch.Tensor]
    endo_norm_sq: dict[tuple[int, ...], torch.Tensor] | None = None


def _validate_tensor_cache(
    tensor_cache: IsotypicTensorCache,
    rep_X_iso: Representation,
    rep_Y_iso: Representation,
    require_endo_norm_sq: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, ...], torch.Tensor], dict[tuple[int, ...], torch.Tensor] | None]:
    Q_out = tensor_cache.Q_out
    Q_in_inv = tensor_cache.Q_in_inv
    endo_basis_flat_cache = tensor_cache.endo_basis_flat
    endo_norm_sq_cache = tensor_cache.endo_norm_sq

    if not isinstance(Q_out, torch.Tensor) or not isinstance(Q_in_inv, torch.Tensor):
        raise TypeError("tensor_cache.Q_out and tensor_cache.Q_in_inv must be torch.Tensor instances")
    if Q_out.shape != (rep_Y_iso.size, rep_Y_iso.size):
        raise ValueError(f"Expected tensor_cache.Q_out shape {(rep_Y_iso.size, rep_Y_iso.size)}, got {Q_out.shape}")
    if Q_in_inv.shape != (rep_X_iso.size, rep_X_iso.size):
        raise ValueError(
            f"Expected tensor_cache.Q_in_inv shape {(rep_X_iso.size, rep_X_iso.size)}, got {Q_in_inv.shape}"
        )
    if not isinstance(endo_basis_flat_cache, Mapping):
        raise TypeError("tensor_cache.endo_basis_flat must be a mapping keyed by irrep id")
    if require_endo_norm_sq:
        if endo_norm_sq_cache is None:
            raise ValueError("tensor_cache.endo_norm_sq is required for this operation")
        if not isinstance(endo_norm_sq_cache, Mapping):
            raise TypeError("tensor_cache.endo_norm_sq must be a mapping keyed by irrep id")

    return Q_out, Q_in_inv, endo_basis_flat_cache, endo_norm_sq_cache


def isotypic_signal2irreducible_subspaces(x: torch.Tensor, rep_x: Representation):
    r"""Flatten an isotypic signal into its irreducible-subspace coordinates.

    This function assumes :math:`\mathcal{X}` is a single isotypic subspace of type :math:`k`, i.e.

    .. math::
        \rho_{\mathcal{X}} = \bigoplus_{i\in[1,m_k]} \hat{\rho}_k.

    For an input :math:`\mathbf{x}` of shape :math:`(n, m_k \cdot d_k)`, where :math:`d_k=\dim(\hat{\rho}_k)`, the
    output rearranges coordinates to shape :math:`(n \cdot d_k, m_k)` so each column stores one irrep copy across the
    sample axis.

    .. math::
        \mathbf{z}[:, i] = [x_{1,i,1}, \ldots, x_{1,i,d_k}, x_{2,i,1}, \ldots, x_{n,i,d_k}]^\top.

    Args:
        x (:class:`~torch.Tensor`): Shape :math:`(n, m_k \cdot d_k)`.
        rep_x (:class:`~escnn.group.Representation`): Representation in isotypic basis with a single active irrep type.

    Returns:
        :class:`~torch.Tensor`: Flattened irreducible-subspace signal of shape :math:`(n \cdot d_k, m_k)`.

    Shape:
        :math:`(n \cdot d_k, m_k)`.
    """
    assert len(rep_x._irreps_multiplicities) == 1, "Random variable is assumed to be in a single isotypic subspace."
    irrep_id = rep_x.irreps[0]
    irrep_dim = rep_x.group.irrep(*irrep_id).size
    mk = rep_x._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X

    Z = x.view(-1, mk, irrep_dim).permute(0, 2, 1).reshape(-1, mk)

    assert Z.shape == (x.shape[0] * irrep_dim, mk)
    return Z


def irrep_radii(x: torch.Tensor, rep: Representation) -> torch.Tensor:
    r"""Compute Euclidean radii for all irreducible-subspace features.

    Let :math:`\rho_{\mathcal{X}}` be the (possibly decomposable) representation of a vector space
    :math:`\mathcal{X}`:

    .. math::
        \rho_{\mathcal{X}} = \mathbf{Q}\left(
        \bigoplus_{k\in[1,n_{\text{iso}}]}
        \bigoplus_{i\in[1,m_k]}
        \hat{\rho}_k
        \right)\mathbf{Q}^T.

    We first change to the irrep-spectral basis induced by this
    :ref:`isotypic decomposition <isotypic-decomposition-example>`
    (as returned by :func:`~symm_learning.representation_theory.isotypic_decomp_rep`),
    :math:`\hat{\mathbf{x}}=\mathbf{Q}^T\mathbf{x}`, and then compute the radius of each irrep copy:

    .. math::
        r_{k,i} = \lVert \hat{\mathbf{x}}_{k,i} \rVert_2.

    Args:
        x: (:class:`~torch.Tensor`) of shape :math:`(..., D)` describing vectors transforming according to ``rep``.
        rep: (:class:`~escnn.group.Representation`) acting on the last dimension of ``x``.

    Returns:
        (:class:`~torch.Tensor`): Radii of shape :math:`(..., N)` where :math:`N=\texttt{len(rep.irreps)}`. The output
        order follows ``rep.irreps`` (one radius per irreducible copy in the decomposition).

    Shape:
        - **Input** ``x``: :math:`(..., D)` with :math:`D=\dim(\rho_{\mathcal{X}})`.
        - **Output**: :math:`(..., N)` containing the per-irrep Euclidean norms.

    Note:
        For repeated calls with the same representation object ``rep``, the matrix
        :math:`\mathbf{Q}^{-1}` is cached in ``rep.attributes["Q_inv"]`` and reused.
    """
    if x.shape[-1] != rep.size:
        raise ValueError(f"Expected last dimension {rep.size}, got {x.shape[-1]}")

    Q_inv = _cached_rep_matrix(rep=rep, key="Q_inv", matrix=rep.change_of_basis_inv, like=x)

    # Change to irrep-spectral basis
    x_spectral = torch.einsum("ij,...j->...i", Q_inv, x)
    n_subspaces = len(rep.irreps)
    subspace_dims = [rep.group.irrep(*irrep_id).size for irrep_id in rep.irreps]

    flat = x_spectral.reshape(-1, rep.size)
    # vector_norm has a stable subgradient at zero, unlike manual sqrt(sum(x^2))
    flat_blocks = torch.split(flat, subspace_dims, dim=-1)
    radii = torch.stack([torch.linalg.vector_norm(block, ord=2, dim=-1) for block in flat_blocks], dim=-1)
    radii = radii.reshape(*x_spectral.shape[:-1], n_subspaces)
    return radii


def lstsq(x: torch.Tensor, y: torch.Tensor, rep_x: Representation, rep_y: Representation):
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
        x (:class:`~torch.Tensor`): Realizations of the random variable :math:`\mathbf{X}` with shape
            :math:`(N, D_x)`, where :math:`N` is the number of samples.
        y (:class:`~torch.Tensor`):
            Realizations of the random variable :math:`\mathbf{Y}` with shape :math:`(N, D_y)`.
        rep_x (:class:`~escnn.group.Representation`):
            Representation :math:`\rho_{\mathcal{X}}` acting on the vector space :math:`\mathcal{X}`.
        rep_y (:class:`~escnn.group.Representation`):
            Representation :math:`\rho_{\mathcal{Y}}` acting on the vector space :math:`\mathcal{Y}`.

    Returns:
        :class:`~torch.Tensor`:
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

    if x.device != y.device:
        raise ValueError(f"Expected x and y on same device, got {x.device} and {y.device}")

    work_dtype = torch.promote_types(x.dtype, y.dtype)
    x_work = x if x.dtype == work_dtype else x.to(dtype=work_dtype)
    y_work = y if y.dtype == work_dtype else y.to(dtype=work_dtype)

    rep_X_iso = isotypic_decomp_rep(rep_x)
    rep_Y_iso = isotypic_decomp_rep(rep_y)
    # Changes of basis from the Disentangled/Isotypic-basis of X, and Y to the original basis.
    Qx = _cached_rep_matrix(rep=rep_X_iso, key="Q", matrix=rep_X_iso.change_of_basis, like=x_work)
    Qy = _cached_rep_matrix(rep=rep_Y_iso, key="Q", matrix=rep_Y_iso.change_of_basis, like=x_work)
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

    x_iso = torch.einsum("ij,...j->...i", Qx.T, x_work)
    y_iso = torch.einsum("ij,...j->...i", Qy.T, y_work)
    A_iso = torch.zeros((rep_y.size, rep_x.size), dtype=work_dtype, device=x_work.device)
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


def invariant_orthogonal_projector(
    rep_x: Representation, device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    r"""Computes the orthogonal projection to the invariant subspace.

    The input representation :math:`\rho_{\mathcal{X}}: \mathbb{G} \mapsto \mathbb{G}\mathbb{L}(\mathcal{X})` is
    transformed to the spectral basis given by:

    .. math::
        \rho_{\mathcal{X}} = \mathbf{Q}\left(
        \bigoplus_{k\in[1,n_{\text{iso}}]}
        \bigoplus_{i\in[1,m_k]}
        \hat{\rho}_k
        \right)\mathbf{Q}^T

    where :math:`\hat{\rho}_k` are irreducible representations of :math:`\mathbb{G}`, :math:`m_k` is the multiplicity
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
        rep_x (:class:`~escnn.group.Representation`): The representation for which the orthogonal projection
            to the invariant subspace is computed.
        device (:class:`~torch.device`, optional): Device for the returned projector. If ``None``, uses CPU.
        dtype (:class:`~torch.dtype`, optional): Data type for the returned projector. If ``None``, uses
            ``torch.get_default_dtype()``.

    Returns:
        :class:`~torch.Tensor`: The orthogonal projection matrix to the invariant subspace,
        :math:`\mathbf{Q} \mathbf{S} \mathbf{Q}^T`.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    Qx_T = torch.as_tensor(rep_x.change_of_basis_inv, device=device, dtype=dtype)
    Qx = torch.as_tensor(rep_x.change_of_basis, device=device, dtype=dtype)

    # S is an indicator of which dimension (in the irrep-spectral basis) is associated with a trivial irrep
    S = torch.zeros((rep_x.size, rep_x.size), device=device, dtype=dtype)
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


def equiv_orthogonal_projection(
    W: torch.Tensor,
    rep_x: Representation,
    rep_y: Representation,
    tensor_cache: IsotypicTensorCache | None = None,
) -> torch.Tensor:
    r"""Orthogonally project a linear map onto :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}},\rho_{\mathcal{Y}})`.

    Let :math:`(\mathcal{X}, \rho_{\mathcal{X}})` and :math:`(\mathcal{Y}, \rho_{\mathcal{Y}})` be two
    :math:`\mathbb{G}`-symmetric vector spaces. Given any dense linear map
    :math:`\mathbf{W}\in\mathbb{R}^{\dim(\mathcal{Y})\times\dim(\mathcal{X})}`, this function returns its
    Frobenius-orthogonal projection onto
    :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`:

    .. math::
        \Pi_{\mathrm{Hom}_{\mathbb{G}}}(\mathbf{W})
        = \operatorname*{argmin}_{\mathbf{A}\in\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}},\rho_{\mathcal{Y}})}
        \|\mathbf{W}-\mathbf{A}\|_F.

    The computation uses the isotypic decomposition of both representations and projects each shared-irrep block
    independently.

    This projection is equivalent to the Reynolds/group-average operator, but more computational and memory
    efficient when the order of the group is large.

    .. math::
        \Pi_{\mathrm{Hom}_{\mathbb{G}}}(\mathbf{A})
        = \frac{1}{|\mathbb{G}|}\sum_{g\in\mathbb{G}}
        \rho_{\mathcal{Y}}(g)\,\mathbf{A}\,\rho_{\mathcal{X}}(g^{-1}).

    Args:
        W (:class:`~torch.Tensor`): Dense map (or batch of maps) of shape
            :math:`(..., D_y, D_x)`.
        rep_x (:class:`~escnn.group.Representation`): Input representation :math:`\rho_{\mathcal{X}}`.
        rep_y (:class:`~escnn.group.Representation`): Output representation :math:`\rho_{\mathcal{Y}}`.
        tensor_cache (:class:`IsotypicTensorCache`, optional): Optional override containing the tensor cache
            required by :func:`project_in_isobasis`. When provided, all required tensors must be present.

    Returns:
        :class:`~torch.Tensor`: Projected map(s) with same shape, dtype, and device as ``W``.

    Shape:
        - **W**: :math:`(..., D_y, D_x)`.
        - **Output**: :math:`(..., D_y, D_x)`.
    """
    Q_out, Q_in_inv, projection_iso_spaces = project_in_isobasis(
        W=W, rep_x=rep_x, rep_y=rep_y, tensor_cache=tensor_cache
    )
    leading_shape = W.shape[:-2]
    batch_axes = tuple(range(len(leading_shape)))
    W_iso = W.new_zeros(*leading_shape, rep_y.size, rep_x.size)

    for iso_space in projection_iso_spaces.values():
        coeff = iso_space["coeff"]
        endo_basis_flat = iso_space["endo_basis_flat"]
        m_out = iso_space["m_out"]
        m_in = iso_space["m_in"]
        d_k = iso_space["d_k"]
        out_slice = iso_space["out_slice"]
        in_slice = iso_space["in_slice"]
        block_proj_flat = coeff.matmul(endo_basis_flat)

        # Undo flattening and permutation to recover block matrix in isotypic coordinates:
        #   [..., m_out, m_in, d_k, d_k] -> [..., m_out*d_k, m_in*d_k]
        block_proj = block_proj_flat.view(*leading_shape, m_out, m_in, d_k, d_k)
        block_proj = block_proj.permute(*batch_axes, -4, -2, -3, -1)
        block_proj = block_proj.reshape(*leading_shape, m_out * d_k, m_in * d_k)
        W_iso[..., out_slice, in_slice] = block_proj

    # Return projected operator in original coordinates.
    W_proj = (Q_out @ W_iso) @ Q_in_inv
    return W_proj


def equiv_orthogonal_projection_coefficients(
    W: torch.Tensor,
    rep_x: Representation,
    rep_y: Representation,
    tensor_cache: IsotypicTensorCache | None = None,
) -> torch.Tensor:
    r"""Return the flattened homomorphism-basis coefficients of :math:`\Pi_{\mathrm{Hom}_{\mathbb{G}}}(\mathbf{W})`.

    Let :math:`\mathbf{W}\in\mathbb{R}^{D_y\times D_x}` be any dense linear map between
    :math:`(\mathcal{X}, \rho_{\mathcal{X}})` and :math:`(\mathcal{Y}, \rho_{\mathcal{Y}})`. This function computes
    the orthogonal projection

    .. math::
        \Pi_{\mathrm{Hom}_{\mathbb{G}}}(\mathbf{W})
        \in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})

    and returns its coefficients in the blockwise isotypic basis described in
    :ref:`Leveraging the structure of Equivariant Linear maps <equivariant-linear-maps-example>`.

    In isotypic coordinates, the projected operator decomposes as

    .. math::
        \Pi_{\mathrm{Hom}_{\mathbb{G}}}(\mathbf{W})
        = \mathbf{Q}_{\mathcal{Y}}
        \left(
        \bigoplus_{k\in[1,n_{\text{iso}}]} \mathbf{W}^{(k)}
        \right)
        \mathbf{Q}_{\mathcal{X}}^T.

    For each common irrep type :math:`k`, the corresponding block is written as

    .. math::
        \mathbf{W}^{(k)}
        = \sum_{s=1}^{S_k}\mathbf{\Theta}^{(k)}_s \otimes \mathbf{\Psi}^{(k)}_s,

    where :math:`\mathbf{\Theta}^{(k)}_s \in \mathbb{R}^{m_k^{\mathcal{Y}} \times m_k^{\mathcal{X}}}`, and
    :math:`\{\mathbf{\Psi}^{(k)}_s\}_{s=1}^{S_k}` is a basis of
    :math:`\mathrm{End}_{\mathbb{G}}(\hat{\rho}_k)`. The output concatenates all coefficient blocks after flattening
    each tensor of shape :math:`(m_k^{\mathcal{Y}} m_k^{\mathcal{X}}, S_k)`, with the endomorphism-basis index
    varying fastest.

    Args:
        W (:class:`~torch.Tensor`): Dense map (or batch of maps) of shape
            :math:`(..., D_y, D_x)`.
        rep_x (:class:`~escnn.group.Representation`): Input representation :math:`\rho_{\mathcal{X}}`.
        rep_y (:class:`~escnn.group.Representation`): Output representation :math:`\rho_{\mathcal{Y}}`.
        tensor_cache (:class:`IsotypicTensorCache`, optional): Optional override containing the tensor cache
            required by :func:`project_in_isobasis`. When provided, all required tensors must be present.

    Returns:
        :class:`~torch.Tensor`: Flattened coefficient vector(s) of shape
        :math:`(..., \dim(\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})))`.

    Shape:
        - **W**: :math:`(..., D_y, D_x)`.
        - **Output**: :math:`(..., |\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})|)`.
    """
    _, _, projection_iso_spaces = project_in_isobasis(W=W, rep_x=rep_x, rep_y=rep_y, tensor_cache=tensor_cache)
    leading_shape = W.shape[:-2]
    if not projection_iso_spaces:
        return W.new_zeros(*leading_shape, 0)
    coeff = [iso_space["coeff"].reshape(*leading_shape, -1) for iso_space in projection_iso_spaces.values()]
    return torch.cat(coeff, dim=-1)


def equiv_linear_map(
    w_dof: torch.Tensor,
    rep_x: Representation,
    rep_y: Representation,
    tensor_cache: IsotypicTensorCache | None = None,
) -> torch.Tensor:
    r"""Expand flattened homomorphism-basis coefficients into a dense equivariant linear map.

    Let :math:`\boldsymbol{\theta}` denote flattened coefficients in the blockwise basis of
    :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})` used by
    :func:`equiv_orthogonal_projection_coefficients`. In isotypic coordinates, the resulting equivariant map has the
    form

    .. math::
        \mathbf{W}
        = \mathbf{Q}_{\mathcal{Y}}
        \left(
        \bigoplus_{k\in[1,n_{\text{iso}}]} \mathbf{W}^{(k)}
        \right)
        \mathbf{Q}_{\mathcal{X}}^T.

    For each common irrep type :math:`k`, the corresponding block is synthesized as

    .. math::
        \mathbf{W}^{(k)}
        = \sum_{s=1}^{S_k}\mathbf{\Theta}^{(k)}_s \otimes \mathbf{\Psi}^{(k)}_s,

    where :math:`\mathbf{\Theta}^{(k)}_s \in \mathbb{R}^{m_k^{\mathcal{Y}} \times m_k^{\mathcal{X}}}` and
    :math:`\{\mathbf{\Psi}^{(k)}_s\}_{s=1}^{S_k}` is a basis of
    :math:`\mathrm{End}_{\mathbb{G}}(\hat{\rho}_k)`.

    The flattened input ordering matches :func:`equiv_orthogonal_projection_coefficients`: within each common irrep
    type :math:`k`, the endomorphism-basis index :math:`s` varies fastest, then the input multiplicity index, and
    finally the output multiplicity index.

    Args:
        w_dof (:class:`~torch.Tensor`): Flattened homomorphism-basis coefficients of shape
            :math:`(..., \dim(\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})))`.
        rep_x (:class:`~escnn.group.Representation`): Input representation :math:`\rho_{\mathcal{X}}`.
        rep_y (:class:`~escnn.group.Representation`): Output representation :math:`\rho_{\mathcal{Y}}`.
        tensor_cache (:class:`IsotypicTensorCache`, optional): Optional tensor cache override containing
            ``Q_out``, ``Q_in_inv``, and ``endo_basis_flat`` keyed by irrep id. When provided, all required tensors
            must be present.

    Returns:
        :class:`~torch.Tensor`: Dense equivariant linear map(s) of shape :math:`(..., D_y, D_x)`.

    Shape:
        - **w_dof**: :math:`(..., |\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})|)`.
        - **Output**: :math:`(..., D_y, D_x)`.

    Note:
        Repeated calls with the same representations reuse cached isotypic decompositions,
        change-of-basis matrices, and flattened irrep endomorphism bases.
    """
    if w_dof.ndim == 0:
        raise ValueError("Expected w_dof to have at least one dimension")

    rep_X_iso, rep_Y_iso, hom_iso_spaces, hom_dim = _hom_irrep_iso_spaces(rep_x=rep_x, rep_y=rep_y)
    if w_dof.shape[-1] != hom_dim:
        raise ValueError(f"Expected w_dof shape (..., {hom_dim}), got {tuple(w_dof.shape)}")

    if tensor_cache is None:
        Q_out = _cached_rep_matrix(rep=rep_Y_iso, key="Q", matrix=rep_Y_iso.change_of_basis, like=w_dof)
        Q_in_inv = _cached_rep_matrix(rep=rep_X_iso, key="Q_inv", matrix=rep_X_iso.change_of_basis_inv, like=w_dof)
        endo_basis_flat_cache = None
    else:
        Q_out, Q_in_inv, endo_basis_flat_cache, _ = _validate_tensor_cache(
            tensor_cache=tensor_cache,
            rep_X_iso=rep_X_iso,
            rep_Y_iso=rep_Y_iso,
            require_endo_norm_sq=False,
        )

    leading_shape = w_dof.shape[:-1]
    batch_axes = tuple(range(len(leading_shape)))
    W_iso = w_dof.new_zeros(*leading_shape, rep_Y_iso.size, rep_X_iso.size)

    for iso_space in hom_iso_spaces:
        irrep_id = iso_space["irrep_id"]
        if tensor_cache is None:
            irrep = rep_X_iso.group.irrep(*irrep_id)
            endo_basis_flat, _ = _cached_irrep_endomorphism_basis(irrep=irrep, like=w_dof)
        else:
            if irrep_id not in endo_basis_flat_cache:
                raise ValueError(f"tensor_cache['endo_basis_flat'] missing tensor for irrep {irrep_id}")
            endo_basis_flat = endo_basis_flat_cache[irrep_id]
            if not isinstance(endo_basis_flat, torch.Tensor):
                raise TypeError(f"tensor_cache['endo_basis_flat'][{irrep_id}] must be a torch.Tensor")
            if endo_basis_flat.ndim != 2 or endo_basis_flat.shape[1] != iso_space["d_k"] * iso_space["d_k"]:
                raise ValueError(
                    "Expected tensor_cache['endo_basis_flat']["
                    f"{irrep_id}] shape (S_k, {iso_space['d_k'] * iso_space['d_k']}), "
                    f"got {tuple(endo_basis_flat.shape)}"
                )
        m_out = iso_space["m_out"]
        m_in = iso_space["m_in"]
        d_k = iso_space["d_k"]
        out_slice = iso_space["out_slice"]
        in_slice = iso_space["in_slice"]
        hom_basis_slice = iso_space["hom_basis_slice"]

        theta_k = w_dof[..., hom_basis_slice].view(*leading_shape, m_out * m_in, endo_basis_flat.size(0))
        block_flat = theta_k.matmul(endo_basis_flat)
        block = block_flat.view(*leading_shape, m_out, m_in, d_k, d_k)
        block = block.permute(*batch_axes, -4, -2, -3, -1)
        block = block.reshape(*leading_shape, m_out * d_k, m_in * d_k)
        W_iso[..., out_slice, in_slice] = block

    return (Q_out @ W_iso) @ Q_in_inv


def project_in_isobasis(
    W: torch.Tensor,
    rep_x: Representation,
    rep_y: Representation,
    tensor_cache: IsotypicTensorCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, ...], IsoSpaceProjection]]:
    r"""Project W onto :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})` in isotypic coordinates

    This is an utility function handling the projection of a dense linear map
    :math:`\mathbf{W}:\mathcal{X}\to\mathcal{Y}` onto the space of :math:`\mathbb{G}`-equivariant linear maps,
    :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`. Returning useful intermediate tensors,
    for algebraic and statistical computations.

    Let :math:`\mathbf{Q}_{\mathcal{X}}` and :math:`\mathbf{Q}_{\mathcal{Y}}` be the orthogonal change-of-basis exposing
    the isotypic decomposition of the symmetric vector spaces :math:`(\mathcal{X}, \rho_{\mathcal{X}})` and
    :math:`(\mathcal{Y}, \rho_{\mathcal{Y}})` (see :ref:`Isotypic Decomposition <isotypic-decomposition-example>`).
    In this basis, any
    :math:`\mathbb{G}`-equivariant linear map
    :math:`\mathbf{A} \in \mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})` decomposes in block
    diagonal form as described in
    :ref:`Leveraging the structure of Equivariant Linear maps <equivariant-linear-maps-example>`:

    .. math::
        \mathbf{A}_{\mathrm{iso}}
        =
        \mathbf{Q}_{\mathcal{Y}}^\top \mathbf{A} \mathbf{Q}_{\mathcal{X}}
        =
        \bigoplus_{k\in[1,n_{\text{iso}}]} \mathbf{A}^{(k)},
        \qquad
        \mathbf{A}^{(k)}
        = \sum_{s=1}^{S_k}\mathbf{\Theta}^{(k)}_s \otimes \mathbf{\Psi}^{(k)}_s.

    Where :math:`\{\mathbf{\Psi}^{(k)}_s \in \mathbb{R}^{d_k \times d_k}\}_{s=1}^{S_k}` is a fixed basis of the
    endomorphism :math:`\mathrm{End}_{\mathbb{G}}(\hat{\rho}_k)`, and
    :math:`\mathbf{\Theta}^{(k)}_s \in \mathbb{R}^{m_k^{\mathcal{Y}} \times m_k^{\mathcal{X}}}` are the free parameters
    (or degrees of freedom) of the equivariant map, serving as basis expandion coefficients.

    Consequently, this function projects the input map :math:`\mathbf{W}` to isotypic coordinates:
    :math:`\mathbf{W}_{\mathrm{iso}} = \mathbf{Q}_{\mathcal{Y}}^T \mathbf{W} \mathbf{Q}_{\mathcal{X}}`, and
    computes the coefficients :math:`\mathbf{\Theta}^{(k)}` of the projected map in each isotypic block:

    .. math::
        \mathbf{\Theta}^{(k)}_{o,i,s}
        = \frac{\langle \mathbf{W}^{(k)}_{o,i}, \mathbf{\Psi}^{(k)}_s\rangle_F}
        {\lVert \mathbf{\Psi}^{(k)}_s \rVert_F^2},
        \qquad \forall \;
        k \in [1, n_{\text{iso}}], o \in [1, m_k^{\mathcal{Y}}], i \in [1, m_k^{\mathcal{X}}], s \in [1, S_k].

    Args:
        W (:class:`~torch.Tensor`): Dense linear map (or batch of maps) from
            :math:`\mathcal{X}` to :math:`\mathcal{Y}` with shape :math:`(..., D_y, D_x)`.
        rep_x (:class:`~escnn.group.Representation`): Input representation
            :math:`\rho_{\mathcal{X}}`.
        rep_y (:class:`~escnn.group.Representation`): Output representation
            :math:`\rho_{\mathcal{Y}}`.
        tensor_cache (:class:`IsotypicTensorCache`, optional): Optional tensor cache override containing
            ``Q_out``, ``Q_in_inv``, ``endo_basis_flat``, and ``endo_norm_sq`` keyed by irrep id. When provided,
            these tensors are reused directly, without copying, and therefore must already have dtype and device
            compatible with ``W``. In particular, the function reads ``tensor_cache.Q_out``,
            ``tensor_cache.Q_in_inv``, ``tensor_cache.endo_basis_flat[irrep_id]``, and
            ``tensor_cache.endo_norm_sq[irrep_id]`` for each shared irrep.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[tuple[int, ...], IsoSpaceProjection]]:
            Tuple with entries:

            - ``Q_out``: matrix :math:`\mathbf{Q}_{\mathcal{Y}}`. If ``tensor_cache`` is ``None``, this is a tensor
              copy with dtype and device matching ``W``. Otherwise, ``tensor_cache.Q_out`` is returned directly.
            - ``Q_in_inv``: matrix :math:`\mathbf{Q}_{\mathcal{X}}^{\top}`. If ``tensor_cache`` is ``None``, this is
              a tensor copy with dtype and device matching ``W``. Otherwise, ``tensor_cache.Q_in_inv`` is returned
              directly.
            - ``projection_iso_spaces``: dictionary keyed by irrep identifier ``irrep_id``. Each
              ``projection_iso_spaces[irrep_id]`` is an :class:`IsoSpaceProjection` containing:

              - ``coeff`` (:class:`~torch.Tensor`): projection coefficients :math:`\mathbf{\Theta}^{(k)}` of shape
                :math:`(..., m_k^{\mathcal{Y}} m_k^{\mathcal{X}}, S_k)`.
              - ``endo_basis_flat`` (:class:`~torch.Tensor`): stacked flattened endomorphism basis
                :math:`\operatorname{flat}(\mathbf{\Psi}^{(k)}) := [\operatorname{flat}(\mathbf{\Psi}^{(k)}_1), \ldots,
                \operatorname{flat}(\mathbf{\Psi}^{(k)}_{S_k})]^\top \in \mathbb{R}^{S_k \times d_k^2}`. If
                ``tensor_cache`` is provided, this entry is ``tensor_cache.endo_basis_flat[irrep_id]`` returned
                directly.
              - ``m_out`` (``int``): output multiplicity :math:`m_k^{\mathcal{Y}}`.
              - ``m_in`` (``int``): input multiplicity :math:`m_k^{\mathcal{X}}`.
              - ``d_k`` (``int``): irrep dimension :math:`d_k = \dim(\hat{\rho}_k)`.
              - ``out_slice`` (``slice``): slice locating :math:`\mathbf{W}^{(k)}` inside
                :math:`\mathbf{W}_{\mathrm{iso}}`.
              - ``in_slice`` (``slice``): slice locating :math:`\mathbf{W}^{(k)}` inside
                :math:`\mathbf{W}_{\mathrm{iso}}`.

    Shape:
        - **W**: :math:`(..., D_y, D_x)`.
        - **Q_out**: :math:`(D_y, D_y)`.
        - **Q_in_inv**: :math:`(D_x, D_x)`.
    """
    if W.shape[-2:] != (rep_y.size, rep_x.size):
        raise ValueError(f"Expected W shape (..., {rep_y.size}, {rep_x.size}), got {tuple(W.shape)}")

    rep_X_iso, rep_Y_iso, hom_iso_spaces, _ = _hom_irrep_iso_spaces(rep_x=rep_x, rep_y=rep_y)
    if tensor_cache is None:
        Q_out = _cached_rep_matrix(rep=rep_Y_iso, key="Q", matrix=rep_Y_iso.change_of_basis, like=W)
        Q_in_inv = _cached_rep_matrix(rep=rep_X_iso, key="Q_inv", matrix=rep_X_iso.change_of_basis_inv, like=W)
        endo_basis_flat_cache = None
        endo_norm_sq_cache = None
    else:
        Q_out, Q_in_inv, endo_basis_flat_cache, endo_norm_sq_cache = _validate_tensor_cache(
            tensor_cache=tensor_cache,
            rep_X_iso=rep_X_iso,
            rep_Y_iso=rep_Y_iso,
            require_endo_norm_sq=True,
        )
    Q_out_inv = Q_out.mT
    Q_in = Q_in_inv.mT

    # Move dense map to isotypic coordinates:
    #   W_iso = Q_out^{-1} W Q_in
    # Shape stays [..., d_out, d_in].
    W_iso = (Q_out_inv @ W) @ Q_in
    leading_shape = W_iso.shape[:-2]
    batch_axes = tuple(range(len(leading_shape)))
    iso_space_metadata = {}

    for iso_space in hom_iso_spaces:
        irrep_id = iso_space["irrep_id"]
        out_slice = iso_space["out_slice"]
        in_slice = iso_space["in_slice"]
        m_out = iso_space["m_out"]
        m_in = iso_space["m_in"]
        d_k = iso_space["d_k"]
        if tensor_cache is None:
            irrep = rep_X_iso.group.irrep(*irrep_id)
            endo_basis_flat, endo_norm_sq = _cached_irrep_endomorphism_basis(irrep=irrep, like=W)
        else:
            if irrep_id not in endo_basis_flat_cache or irrep_id not in endo_norm_sq_cache:
                raise ValueError(f"tensor_cache missing endomorphism tensors for irrep {irrep_id}")
            endo_basis_flat = endo_basis_flat_cache[irrep_id]
            endo_norm_sq = endo_norm_sq_cache[irrep_id]
            if not isinstance(endo_basis_flat, torch.Tensor) or not isinstance(endo_norm_sq, torch.Tensor):
                raise TypeError(f"tensor_cache entries for irrep {irrep_id} must be torch.Tensor instances")
            if endo_basis_flat.ndim != 2 or endo_basis_flat.shape[1] != d_k * d_k:
                raise ValueError(
                    f"Expected tensor_cache['endo_basis_flat'][{irrep_id}] shape (S_k, {d_k * d_k}), "
                    f"got {tuple(endo_basis_flat.shape)}"
                )
            if endo_norm_sq.shape != (endo_basis_flat.shape[0],):
                raise ValueError(
                    f"Expected tensor_cache['endo_norm_sq'][{irrep_id}] shape {(endo_basis_flat.shape[0],)}, "
                    f"got {tuple(endo_norm_sq.shape)}"
                )
        block = W_iso[..., out_slice, in_slice]
        block = block.view(*leading_shape, m_out, d_k, m_in, d_k)
        block = block.permute(*batch_axes, -4, -2, -3, -1)  # [..., m_out, m_in, d_k, d_k]
        block_flat = block.reshape(*leading_shape, m_out * m_in, d_k * d_k)  # [..., m_out*m_in, d_k^2]

        # coeff[..., p, s] = <block_{p}, E_s> / ||E_s||^2
        coeff = block_flat.matmul(endo_basis_flat.mT)
        coeff = coeff / endo_norm_sq
        iso_space_metadata[irrep_id] = dict(
            coeff=coeff,
            endo_basis_flat=endo_basis_flat,
            m_out=m_out,
            m_in=m_in,
            d_k=d_k,
            out_slice=out_slice,
            in_slice=in_slice,
        )

    return Q_out, Q_in_inv, iso_space_metadata


def _project_to_irrep_endomorphism_basis(
    A: torch.Tensor,
    rep_x: Representation,
    rep_y: Representation,
) -> torch.Tensor:
    r"""Project a dense linear map between two isotypic spaces onto the equivariant subspace.

    Let :math:`(\mathcal{X}, \rho_{\mathcal{X}})` and :math:`(\mathcal{Y}, \rho_{\mathcal{Y}})` be two isotypic
    vector spaces of the same irrep type :math:`k`, with multiplicities :math:`m_k^{\mathcal{X}}` and
    :math:`m_k^{\mathcal{Y}}`. This function projects a dense linear map
    :math:`\mathbf{A}:\mathcal{X}\to\mathcal{Y}` onto
    :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}}, \rho_{\mathcal{Y}})`.

    Args:
        A (:class:`~torch.Tensor`): Dense linear map of shape
            :math:`(m_k^{\mathcal{Y}} d_k,\; m_k^{\mathcal{X}} d_k)`.
        rep_x (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\mathcal{X}}` of the isotypic
            space :math:`\mathcal{X}`.
        rep_y (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\mathcal{Y}}` of the isotypic
            space :math:`\mathcal{Y}`.

    Returns:
        A_equiv (:class:`~torch.Tensor`): Projected map of shape
            :math:`(m_k^{\mathcal{Y}} d_k,\; m_k^{\mathcal{X}} d_k)` satisfying
            :math:`\rho_{\mathcal{Y}}(g)\mathbf{A}_{\mathrm{equiv}}
            = \mathbf{A}_{\mathrm{equiv}}\rho_{\mathcal{X}}(g)` for all :math:`g \in \mathbb{G}`.
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


def _cached_irrep_endomorphism_basis(
    irrep: Representation,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = irrep.attributes.setdefault("_endo_basis_flat_cache", {})
    cache_key = (like.device.type, like.device.index, like.dtype)
    if cache_key not in cache:
        endo_basis = torch.as_tensor(irrep.endomorphism_basis(), device=like.device, dtype=like.dtype).contiguous()
        endo_basis_flat = endo_basis.view(endo_basis.size(0), -1)
        endo_norm_sq = torch.einsum("sd,sd->s", endo_basis_flat, endo_basis_flat)
        cache[cache_key] = (endo_basis_flat, endo_norm_sq)
    return cache[cache_key]


def _hom_irrep_iso_spaces(
    rep_x: Representation,
    rep_y: Representation,
) -> tuple[Representation, Representation, list[dict], int]:
    from symm_learning.representation_theory import isotypic_decomp_rep

    if rep_x.group != rep_y.group:
        raise ValueError(f"Expected same group, got {rep_x.group} and {rep_y.group}")

    rep_X_iso = isotypic_decomp_rep(rep_x)
    rep_Y_iso = isotypic_decomp_rep(rep_y)
    iso_idx_X = rep_X_iso.attributes["isotypic_subspace_dims"]
    iso_idx_Y = rep_Y_iso.attributes["isotypic_subspace_dims"]
    common_irreps = sorted(set(rep_X_iso.irreps).intersection(set(rep_Y_iso.irreps)))

    hom_iso_spaces = []
    dof_offset = 0
    for irrep_id in common_irreps:
        out_slice = iso_idx_Y[irrep_id]
        in_slice = iso_idx_X[irrep_id]
        m_out = rep_Y_iso._irreps_multiplicities[irrep_id]
        m_in = rep_X_iso._irreps_multiplicities[irrep_id]
        irrep = rep_X_iso.group.irrep(*irrep_id)
        d_k = irrep.size
        dim_endo_basis = len(irrep.endomorphism_basis())
        irrep_dim = m_out * m_in * dim_endo_basis
        hom_iso_spaces.append(
            dict(
                irrep_id=irrep_id,
                dim_endo_basis=dim_endo_basis,
                m_out=m_out,
                m_in=m_in,
                d_k=d_k,
                out_slice=out_slice,
                in_slice=in_slice,
                hom_basis_slice=slice(dof_offset, dof_offset + irrep_dim),
            )
        )
        dof_offset += irrep_dim

    return rep_X_iso, rep_Y_iso, hom_iso_spaces, dof_offset
