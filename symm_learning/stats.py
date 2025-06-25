"""Statistics utilities for symmetric random variables with known group representations."""

from __future__ import annotations

import numpy as np
import torch
from escnn.group import Representation
from torch import Tensor

from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.representation_theory import isotypic_decomp_rep


def var_mean(x: Tensor, rep_x: Representation):
    """Compute the mean and variance of a symmetric random variable.

    Args:
        x: (Tensor) of shape :math:`(N, D_x)` containing the observations of the symmetric random variable
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
    if "invariant_orthogonal_projector" not in rep_x.attributes:
        P_inv = invariant_orthogonal_projector(rep_x)
        rep_x.attributes["invariant_orthogonal_projector"] = P_inv
    else:
        P_inv = rep_x.attributes["invariant_orthogonal_projector"]
    # print(P_inv.dtype, x.dtype)
    mean = torch.einsum("ij,...j->...i", P_inv, mean_empirical)

    # Symmetry constrained variance computation.
    # The variance is constraint to be a single constant per each irreducible subspace.
    # Hence, we compute the empirical variance, and average within each irreducible subspace.
    n_samples = x.shape[0]
    if "Q_inv" not in rep_x.attributes:  # Use cache Tensor if available.
        Q_inv = torch.tensor(rep_x.change_of_basis_inv, device=x.device, dtype=x.dtype)
        rep_x.attributes["Q_inv"] = Q_inv
    else:
        Q_inv = rep_x.attributes["Q_inv"]

    if "Q" not in rep_x.attributes:  # Use cache Tensor if available.
        Q = torch.tensor(rep_x.change_of_basis, device=x.device, dtype=x.dtype)
        rep_x.attributes["Q"] = Q
    else:
        Q = rep_x.attributes["Q"]

    x_c_irrep_spectral = torch.einsum("ij,...j->...i", Q_inv.to(device=x.device), x - mean)
    var_spectral = torch.sum(x_c_irrep_spectral**2, dim=0) / (n_samples - 1)

    d = 0
    for irrep_id in rep_x.irreps:
        irrep_dim = rep_x.group.irrep(*irrep_id).size
        avg_irrep_var = torch.mean(var_spectral[d : d + irrep_dim])
        var_spectral[d : d + irrep_dim] = avg_irrep_var
        d += irrep_dim

    var = torch.einsum("ij,...j->...i", Q.pow(2).to(device=x.device), var_spectral)
    return var, mean


def _isotypic_cov(x: Tensor, rep_x: Representation, y: Tensor = None, rep_y: Representation = None):
    r"""Cross-covariance between two **isotypic sub-spaces that share the same irrep**.

    If both signals live in
    :math:`\rho_X=\bigoplus_{i=1}^{m_x}\rho_k` and
    :math:`\rho_Y=\bigoplus_{i=1}^{m_y}\rho_k`
    (with :math:`\rho_k` of dimension *d*), every
    :math:`G`-equivariant linear map factorises as

    .. math::
        \operatorname{Cov}(X,Y)
        \;=\;\mathbf Z_{XY}\otimes \mathbf I_d,  \qquad
        \mathbf Z_{XY}\in\mathbb R^{m_y\times m_x}.

    We estimate the free matrix :math:`\mathbf Z_{XY}` by

    1. **centering** (skipped if the irrep is trivial);
    2. **reshaping** the data so that each copy of the irrep becomes one
       “channel” of length *d·N*;
    3. **projecting** every :math:`d\times d` block onto the orthogonal basis
       of :math:`\mathrm{End}_G(\rho_k)` via Frobenius inner products
       (see <https://arxiv.org/abs/2505.19809>`_);
    4. rebuilding the block matrix that respects the constraint above.

    When ``y is None`` the routine reduces to an **auto-covariance** and only
    the symmetric (identity) basis element is kept.


    Args:
        x (Tensor): shape :math:`(N,\; m_x d)` — samples drawn from ``rep_x``.
        rep_x (escnn.group.Representation): isotypic representation
            containing exactly one irrep type.
        y (Tensor, optional): shape :math:`(N,\; m_y d)` —
            samples drawn from ``rep_y``.  If *None*, computes the
            auto-covariance of *x*.
        rep_y (escnn.group.Representation, optional): isotypic
            representation matching the irrep of ``rep_x``; ignored when
            *y* is *None*.

    Returns:
        (Tensor, Tensor):
            C_xy: math:`(m_y d,\; m_x d)` projected covariance.
            Z_xy:math:`(m_y,\; m_x,\; B)`, free coefficients of each cross-covariance between irrep subspaces,
              representing basis expansion coefficients in the basis of endomorphisms of the irrep subspaces.
              Where :math:`B = 1, 2, 4` for real, complex, quaternionic irreps, respectively.
    """
    irrep_id = rep_x.irreps[0]  # Irrep id of the isotypic subspace
    assert rep_x.size == x.shape[-1], f"Expected signal shape to be (..., {rep_x.size}) got {x.shape}"
    assert len(rep_x._irreps_multiplicities) == 1, f"Expected rep with a single irrep type, got {rep_x.irreps}"
    x_in_iso_basis = np.allclose(rep_x.change_of_basis_inv, np.eye(rep_x.size), atol=1e-6, rtol=1e-4)
    assert x_in_iso_basis, "Expected X to be in spectral/isotypic basis"
    if y is not None:
        assert len(rep_y._irreps_multiplicities) == 1, f"Expected rep with a single irrep type, got {rep_y.irreps}"
        assert rep_x.group == rep_y.group, f"{rep_x.group} != {rep_y.group}"
        assert irrep_id == rep_y.irreps[0], f"Irreps {irrep_id} != {rep_y.irreps[0]}. Hence Cxy=0"
        assert rep_y.size == y.shape[-1], f"Expected signal shape to be (..., {rep_y.size}) got {y.shape}"
        y_in_iso_basis = np.allclose(rep_y.change_of_basis_inv, np.eye(rep_y.size), atol=1e-6, rtol=1e-4)
        assert y_in_iso_basis, "Expected Y to be in spectral/isotypic basis"

    # Get information about the irreducible representation present in the isotypic subspace
    irrep_dim = rep_x.group.irrep(*irrep_id).size
    # irrep_end_basis := (dim(End(irrep)), dim(irrep), dim(irrep))
    irrep_end_basis = torch.tensor(rep_x.group.irrep(*irrep_id).endomorphism_basis(), device=x.device, dtype=x.dtype)

    # if y is None Cxy = Cx is symmetric matrix. Hence it has non-zero entries only in the diagonal.
    if y is None:
        irrep_end_basis = irrep_end_basis[[0]]
        rep_y = rep_x  # Use the same representation for Y
        y = x

    m_x = rep_x._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X
    m_y = rep_y._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in Y

    x_iso, y_iso = x, y

    is_inv_subspace = irrep_id == rep_x.group.trivial_representation.id
    if is_inv_subspace:  # Nothing to do, return empirical covariance.
        x_iso = x - torch.mean(x, dim=0, keepdim=True)
        y_iso = y - torch.mean(y, dim=0, keepdim=True)
        Cxy_iso = torch.einsum("...y,...x->yx", y_iso, x_iso) / (x_iso.shape[0] - 1)
        return Cxy_iso, Cxy_iso  # Invariant subspace covariance is the same as the covariance matrix.

    # Compute empirical cross-covariance
    Cxy_iso = torch.einsum("...y,...x->yx", y_iso, x_iso) / (x_iso.shape[0] - 1)
    # ReshapCxy_isoe from (my * d, mx * d) to (my, mx, d, d)
    Cxy_irreps = Cxy_iso.view(m_y, irrep_dim, m_x, irrep_dim).permute(0, 2, 1, 3).contiguous()
    # Compute basis expansion coefficients of each irrep cross-covariance in basis of End(irrep) ========
    # Frobenius inner product  <C , Ψ_b>  =  Σ_{i,j} C_{ij} Ψ_b,ij
    Cxy_irreps_basis_coeff = torch.einsum("mnij,bij->mnb", Cxy_irreps, irrep_end_basis)  # (m_y , m_x , B)
    # squared norms ‖Ψ_b‖² (only once, very small)
    basis_coeff_norms = torch.einsum("bij,bij->b", irrep_end_basis, irrep_end_basis)  # (B,)
    Cxy_irreps_basis_coeff = Cxy_irreps_basis_coeff / basis_coeff_norms[None, None]

    Cxy_irreps = torch.einsum("...b,bij->...ij", Cxy_irreps_basis_coeff, irrep_end_basis)  # (m_y , m_x , d , d)
    # Reshape to (my * d, mx * d)
    Cxy_iso = Cxy_irreps.permute(0, 2, 1, 3).reshape(m_y * irrep_dim, m_x * irrep_dim)

    return Cxy_iso, Cxy_irreps_basis_coeff


def cov(x: Tensor, y: Tensor, rep_x: Representation, rep_y: Representation):
    r"""Compute the covariance between two symmetric random variables.

    The covariance of r.v. can be computed from the orthogonal projections of the r.v. to each isotypic subspace.
    Hence, in the disentangled/isotypic basis the covariance can be computed in block-diagonal form:

    .. math::
        \begin{align}
            \mathbf{C}_{xy} &= \mathbf{Q}_y^T (\bigoplus_{k} \mathbf{C}_{xy}^{(k)} )\mathbf{Q}_x
            \\
            &= \mathbf{Q}_y^T (
            \bigoplus_{k} \sum_{b\in \mathbb{B}_k} \mathbf{Z}_b^{(k)} \otimes \mathbf{b}
            ) \mathbf{Q}_x
            \\
        \end{align}

    Where :math:`\mathbf{Q}_x^{\mathsf T}` and :math:`\mathbf{Q}_y^{\mathsf T}`
    are the change-of-basis matrices to the isotypic bases of *X* and *Y*,
    respectively; :math:`\mathbf{C}_{xy}^{(k)}` is the covariance restricted to the
    isotypic subspaces of type *k*; and :math:`\mathbf{Z}_b^{(k)}` are the free
    parameters—i.e. the expansion coefficients in the endomorphism basis
    :math:`\mathbb{B}_k` of the irreducible representation of type *k*.

    Args:
        x (Tensor): Realizations of a random variable x.
        y (Tensor): Realizations of a random variable y.
        rep_x (Representation): The representation acting on the variables X.
        rep_y (Representation): The representation acting on the variables Y.

    Returns:
        Tensor: The covariance matrix between the two random variables, of shape :math:`(D_y, D_x)`.

    Shape:
        X: :math:`(N, D_x)` where :math:`D_x` is the dimension of the random variable X.
        Y: :math:`(N, D_y)` where :math:`D_y` is the dimension of the random variable Y.

        Output: :math:`(D_y, D_x)`
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
        # Cxy_k = D_xy_k ⊗ I_d [my * d x mx * d]
        Cxy_k, _ = _isotypic_cov(x=X_k, y=Y_k, rep_x=rep_X_k, rep_y=rep_Y_k)
        Cxy_iso[iso_idx_Y[iso_id], iso_idx_X[iso_id]] = Cxy_k

    # Change to the original basis
    Cxy = Qy.T @ Cxy_iso @ Qx
    return Cxy
