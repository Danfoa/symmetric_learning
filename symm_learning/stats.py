"""Statistics utilities for symmetric random variables with known group representations."""

from __future__ import annotations

import numpy as np
import torch
from escnn.group import Representation
from torch import Tensor

from symm_learning.representation_theory import isotypic_decomp_rep


def var_mean(x: Tensor, rep_x: Representation):
    """Compute the mean and variance of a symmetric random variable.

    Args:
        x: (Tensor) of shape :math:`(N, Dx)` containing the observations of the symmetric random variable
        rep_x: (escnn.group.Representation) representation of the symmetric random variable.

    Returns:
        (Tensor, Tensor): Mean and variance of the symmetric random variable. The mean is redistricted to be in the
        trivial/G-invariant subspace of the symmetric vector space. The variance is constrained to be the same for all
        dimensions of each G-irreducible subspace (i.e., each subspace associated with an irrep).

    Shape:
        :code:`x`: :math:`(N, Dx)` where N is the number of samples and Dx is the dimension of the symmetric random
         variable.

        Output: :math:`(Dx, Dx)`
    """
    assert len(x.shape) == 2, f"Expected x to have shape (N, n_features), got {x.shape}"
    # Compute the mean of the observation.
    mean_empirical = torch.mean(x, dim=0)
    # Project to the inv-subspace and map back to the original basis
    P_inv = invariant_orthogonal_projector(rep_x)
    # print(P_inv.dtype, x.dtype)
    mean = torch.einsum("ij,...j->...i", P_inv, mean_empirical)

    # Symmetry constrained variance computation.
    Cx = cov(x, x, rep_x, rep_x)
    var = torch.diag(Cx)
    return var, mean


def isotypic_covariance(x: Tensor, y: Tensor, rep_x: Representation, rep_y: Representation, center=True):
    r"""Cross covariance of signals between isotypic subspaces of the same type.

    This function exploits the fact that the covariance of signals between isotypic subspaces of the same type
    is constrained to be of the block form:

    .. math::
        \mathbf{C}_{xy} = \text{Cov}(X, Y) = \mathbf{Z}_{xy} \otimes \mathbf{I}_d,

    where :math:`d = \text{dim(irrep)}` and :math:`\mathbf{Z}_{xy} \in \mathbb{R}^{m_x \times m_y}` and
    :math:`\mathbf{C}_{xy} \in \mathbb{R}^{(m_x \cdot d) \times (m_y \cdot d)}`.

    Being :math:`m_x` and :math:`m_y` the multiplicities of the irrep in X and Y respectively. This implies that the
    matrix :math:`\mathbf{Z}_{xy}` represents the free parameters of the covariance we are required to estimate.
    To do so we reshape the signals :math:`X \in \mathbb{R}^{N \times (m_x \cdot d)}` and
    :math:`Y \in \mathbb{R}^{N \times (m_y \cdot d)}` to :math:`X_{\text{sing}} \in \mathbb{R}^{(d \cdot N) \times m_x}`
    and :math:`Y_{\text{sing}} \in \mathbb{R}^{(d \cdot N) \times m_y}` respectively. Ensuring all dimensions of the
    irreducible subspaces associated to each multiplicity of the irrep are considered as a single dimension for
    estimating :math:`\mathbf{Z}_{xy} = \frac{1}{n \cdot d} X_{\text{sing}}^T Y_{\text{sing}}`.

    Args:
        x (Tensor): Realizations of the random variable X.
        y (Tensor): Realizations of the random variable Y.
        rep_x (escnn.nn.Representation): composed of :math:`m_x` copies of a single irrep:
            :math:`\rho_X = \otimes_i^{m_x} \rho_k`
        rep_y (escnn.nn.Representation): composed of :math:`m_y` copies of a single irrep:
            :math:`\rho_Y = \otimes_i^{m_y} \rho_k`
        center (bool): whether to center the signals before computing the covariance.

    Returns:
        (Tensor, Tensor): :math:`\mathbf{C}_{xy}`, (:math:`m_y \cdot d, m_x \cdot d`) the covariance matrix between the
         isotypic subspaces of :code:`x` and :code:`y`, and :math:`\mathbf{Z}_{xy}`, (:math:`m_y, m_x`) the free
         parameters of the covariance matrix in the isotypic basis.

    Shape:
        :code:`x`: :math:`(..., N, m_x * d)` where N is the number of samples, :math:`d` is the dimension of the only
        irrep in :math:`rep_X` and :math:`m_x` is the multiplicity of the irrep in X.

        :code:`y`: :math:`(..., N, m_y * d)` where N is the number of samples, :math:`d` is the dimension of the only
        irrep in :math:`rep_Y` and :math:`m_y` is the multiplicity of the irrep in Y.

        Output: :math:`(m_y * d, m_x * d)`.
    """
    assert len(rep_x._irreps_multiplicities) == len(rep_y._irreps_multiplicities) == 1, (
        f"Expected group representation of an isotypic subspace.I.e., with only one type of irrep. \nFound: "
        f"{list(rep_x._irreps_multiplicities.keys())} in rep_X, {list(rep_y._irreps_multiplicities.keys())} in rep_Y."
    )
    assert rep_x.group == rep_y.group, f"{rep_x.group} != {rep_y.group}"
    irrep_id = rep_x.irreps[0]  # Irrep id of the isotypic subspace
    assert irrep_id == rep_y.irreps[0], (
        f"Irreps {irrep_id} != {rep_y.irreps[0]}. Hence signals are orthogonal and Cxy=0."
    )
    assert rep_x.size == x.shape[-1], f"Expected signal shape to be (..., {rep_x.size}) got {x.shape}"
    assert rep_y.size == y.shape[-1], f"Expected signal shape to be (..., {rep_y.size}) got {y.shape}"

    # Get information about the irreducible representation present in the isotypic subspace
    irrep_dim = rep_x.group.irrep(*irrep_id).size
    mk_X = rep_x._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X
    mk_Y = rep_y._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in Y

    # If required we must change bases to the isotypic bases.
    Qx_T, Qx = rep_x.change_of_basis_inv, rep_x.change_of_basis
    Qy_T, Qy = rep_y.change_of_basis_inv, rep_y.change_of_basis
    x_in_iso_basis = np.allclose(Qx_T, np.eye(Qx_T.shape[0]), atol=1e-6, rtol=1e-4)
    y_in_iso_basis = np.allclose(Qy_T, np.eye(Qy_T.shape[0]), atol=1e-6, rtol=1e-4)
    if x_in_iso_basis:
        x_iso = x
    else:
        Qx_T = Tensor(Qx_T).to(device=x.device, dtype=x.dtype)
        Qx = Tensor(Qx).to(device=x.device, dtype=x.dtype)
        x_iso = torch.einsum("...ij,...j->...i", Qx_T, x)  # x_iso = Q_x2iso @ x
    if np.allclose(Qy_T, np.eye(Qy_T.shape[0]), atol=1e-6, rtol=1e-4):
        y_iso = y
    else:
        Qy_T = Tensor(Qy_T).to(device=y.device, dtype=y.dtype)
        Qy = Tensor(Qy).to(device=y.device, dtype=y.dtype)
        y_iso = torch.einsum("...ij,...j->...i", Qy_T, y)  # y_iso = Q_y2iso @ y

    if irrep_dim > 1:
        # Since Cxy = Dxy ⊗ I_d  , d = dim(irrep) and D_χy ∈ R^{mχ x my}
        # We compute the constrained covariance, by estimating the matrix D_χy
        # This requires reshape X_iso ∈ R^{n x p} to X_sing ∈ R^{nd x mχ} and Y_iso ∈ R^{n x q} to Y_sing ∈ R^{nd x my}
        # Ensuring all samples from dimensions of a single irrep are flattened into a row of X_sing and Y_sing
        x_sing = x_iso.view(-1, mk_X, irrep_dim).permute(0, 2, 1).reshape(-1, mk_X)
        y_sing = y_iso.view(-1, mk_Y, irrep_dim).permute(0, 2, 1).reshape(-1, mk_Y)
    else:  # For one dimensional (real) irreps, this defaults to the standard covariance
        x_sing, y_sing = x_iso, y_iso

    is_inv_subspace = irrep_id == rep_x.group.trivial_representation.id
    if center and is_inv_subspace:  # Non-trivial isotypic subspace are centered
        x_sing = x_sing - torch.mean(x_sing, dim=0, keepdim=True)
        y_sing = y_sing - torch.mean(y_sing, dim=0, keepdim=True)

    N = x_sing.shape[0]
    assert x.shape[0] * irrep_dim == N

    c = 1 if center and is_inv_subspace else 0
    Dxy = torch.einsum("...y,...x->yx", y_sing, x_sing) / (N - c)
    if irrep_dim > 1:  # Broadcast the estimates according to Cxy = Dxy ⊗ I_d.
        I_d = torch.eye(irrep_dim, device=Dxy.device, dtype=Dxy.dtype)
        Cxy_iso = torch.kron(Dxy, I_d)
    else:
        Cxy_iso = Dxy

    # Change back to original basis if needed _______________________
    Cxy = Qy @ Cxy_iso if not x_in_iso_basis else Cxy_iso

    if not y_in_iso_basis:
        Cxy = Cxy @ Qx_T

    return Cxy, Dxy


def cov(x: Tensor, y: Tensor, rep_x: Representation, rep_y: Representation):
    r"""Compute the covariance between two symmetric random variables.

    The covariance of r.v. can be computed from the orthogonal projections of the r.v. to each isotypic subspace.
    Hence, in the disentangled/isotypic basis the covariance can be computed in block-diagonal form:

    .. math::
        \begin{align}
            \mathbf{C}_{xy} &= \mathbf{Q}_y^T (\bigoplus_{k} \mathbf{C}_{xy}^{(k)} )\mathbf{Q}_x \\
            &= \mathbf{Q}_y^T (\bigoplus_{k} \mathbf{Z}_{xy}^{(k)}  \otimes \mathbf{I}_{d_k} )\mathbf{Q}_x \\
        \end{align}
    Where :math:`\mathbf{Q}_x^T` and :math:`\mathbf{Q}_y^T` are the change of basis matrices to the isotypic basis of
    X and Y respectively,
    :math:`\mathbf{C}_{xy}^{(k)}` is the covariance between the isotypic subspaces of type k,
    :math:`\mathbf{Z}_{xy}^{(k)}` is the free parameters of the covariance matrix in the isotypic basis,
    and :math:`d_k` is the dimension of the irrep associated with the isotypic subspace of type k.

    Args:
        x (Tensor): Realizations of a random variable x.
        y (Tensor): Realizations of a random variable y.
        rep_x (Representation): The representation acting on the variables X.
        rep_y (Representation): The representation acting on the variables Y.

    Returns:
        Tensor: The covariance matrix between the two random variables, of shape :math:`(Dy, Dx)`.

    Shape:
        X: :math:`(N, Dx)` where :math:`Dx` is the dimension of the random variable X.
        Y: :math:`(N, Dy)` where :math:`Dy` is the dimension of the random variable Y.

        Output: :math:`(Dy, Dx)`
    """
    # assert X.shape[0] == Y.shape[0], "Expected equal number of samples in X and Y"
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

    X_iso = torch.einsum("ij,...j->...i", Qx.T, x)
    Y_iso = torch.einsum("ij,...j->...i", Qy.T, y)
    Cxy_iso = torch.zeros((rep_y.size, rep_x.size), dtype=x.dtype, device=x.device)
    for iso_id in rep_Y_iso_subspaces:
        if iso_id not in rep_X_iso_subspaces:
            continue  # No covariance between the isotypic subspaces of different types.
        X_k = X_iso[..., iso_idx_X[iso_id]]
        Y_k = Y_iso[..., iso_idx_Y[iso_id]]
        rep_X_k = rep_X_iso_subspaces[iso_id]
        rep_Y_k = rep_Y_iso_subspaces[iso_id]
        # Cxy_k = Dxy_k ⊗ I_d [my * d x mx * d]
        Cxy_k, _ = isotypic_covariance(X_k, Y_k, rep_X_k, rep_Y_k, center=True)
        Cxy_iso[iso_idx_Y[iso_id], iso_idx_X[iso_id]] = Cxy_k

    # Change to the original basis
    Cxy = Qy.T @ Cxy_iso @ Qx
    return Cxy


def invariant_orthogonal_projector(rep_x: Representation) -> Tensor:
    r"""Computes the orthogonal projection to the invariant subspace.

    The input representation :math:`\rho_{\mathcal{X}}: \mathbb{G} \mapsto \mathbb{G}\mathbb{L}(\mathcal{X})` is
    transformed to the spectral basis given by:

    .. math::
        \rho_\mathcal{X} = \mathbf{Q} \left( \bigoplus_{i\in[1,n]} \hat{\rho}_i \right) \mathbf{Q}^T

    where :math:`\hat{\rho}_i` denotes an instance of one of the irreducible representations of the group, and
    :math:`\mathbf{Q}: \mathcal{X} \mapsto \mathcal{X}` is the orthogonal change of basis from the spectral basis to
    the original basis.

    The projection is performed by:
        1. Changing the basis to the representation spectral basis (exposing signals per irrep).
        2. Zeroing out all signals on irreps that are not trivial.
        3. Mapping back to the original basis set.

    Args:
        rep_x (Representation): The representation for which the orthogonal projection to the invariant subspace is
        computed.

    Returns:
        Tensor: The orthogonal projection matrix to the invariant subspace, :math:`\mathbf{Q} \mathbf{S} \mathbf{Q}^T`.
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
