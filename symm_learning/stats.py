"""Statistics utilities for symmetric random variables with known group representations."""

from __future__ import annotations

import numpy as np
import torch
from escnn.group import Representation
from torch import Tensor

from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.representation_theory import isotypic_decomp_rep


def mean(x: Tensor, rep_x: Representation) -> Tensor:
    """Compute the mean of a symmetric random variable.

    Args:
        x: (:class:`torch.Tensor`) of shape :math:`(N, D_x)` containing the observations of the symmetric random
        variable
        rep_x: (:class:`~escnn.group.Representation`) representation of the symmetric random variable.

    Returns:
        (:class:`torch.Tensor`): Mean of the symmetric random variable.
        The mean is restricted to be in the trivial/G-invariant subspace of the symmetric vector space.

    Shape:
        - **x**: :math:`(N, D_x)` or :math:`(N, D_x, T)` where N is the number of samples, D_x is the dimension of
          the symmetric random variable, and T is the sequence length (if applicable).
        - **Output**: The mean has shape :math:`(D_x,)`. If a sequence is provided (T dimension),
          the shape is :math:`(D_x, T)`.
    """
    assert x.ndim in (2, 3), f"Expected x to be a 2D or 3D tensor, got {x.ndim}D tensor"

    if "invariant_orthogonal_projector" not in rep_x.attributes:
        P_inv = invariant_orthogonal_projector(rep_x)
        rep_x.attributes["invariant_orthogonal_projector"] = P_inv
    else:
        P_inv = rep_x.attributes["invariant_orthogonal_projector"]

    x_flat = x if x.ndim == 2 else x.reshape(-1, x.shape[1])

    mean_empirical = torch.mean(x_flat, dim=0)  # Mean over batch as sequence length.
    # Project to the inv-subspace and map back to the original basis
    mean_result = torch.einsum("ij,j->i...", P_inv.to(device=x_flat.device), mean_empirical)
    return mean_result


def var(x: Tensor, rep_x: Representation, center: Tensor = None) -> Tensor:
    """Compute the variance of a symmetric random variable.

    Args:
        x: (:class:`torch.Tensor`) of shape :math:`(N, D_x)` containing the observations of the symmetric random
        variable
        rep_x: (:class:`~escnn.group.Representation`) representation of the symmetric random variable.
        center: (:class:`torch.Tensor`, optional) Center for variance computation. If None, computes the mean.

    Returns:
        (:class:`torch.Tensor`): Variance of the symmetric random variable.
        The variance is constrained such that in the irrep-spectral basis, each G-irreducible subspace
        (i.e., each subspace associated with an irrep) has the same variance in all dimensions of that subspace.

    Shape:
        - **x**: :math:`(N, D_x)` or :math:`(N, D_x, T)` where N is the number of samples, D_x is the dimension of
          the symmetric random variable, and T is the sequence length (if applicable).
        - **center**: :math:`(D_x,)` or :math:`(D_x, T)` if provided.
        - **Output**: The variance has shape :math:`(D_x,)`. If a sequence is provided (T dimension),
          the shape is :math:`(D_x, T)`.
    """
    assert x.ndim in (2, 3), f"Expected x to be a 2D or 3D tensor, got {x.ndim}D tensor"

    if "Q_inv" not in rep_x.attributes:  # Use cache Tensor if available.
        Q_inv = torch.tensor(rep_x.change_of_basis_inv, device=x.device, dtype=x.dtype)
        rep_x.attributes["Q_inv"] = Q_inv
    else:
        Q_inv = rep_x.attributes["Q_inv"]

    if "Q_squared" not in rep_x.attributes:  # Use cache Tensor if available.
        Q = torch.tensor(rep_x.change_of_basis, device=x.device, dtype=x.dtype)
        Q_squared = Q.pow(2)
        rep_x.attributes["Q_squared"] = Q_squared
    else:
        Q_squared = rep_x.attributes["Q_squared"]

    x_flat = x if x.ndim == 2 else x.reshape(-1, x.shape[1])

    # Use provided center or compute mean
    if center is None:
        center = mean(x, rep_x)

    # Symmetry constrained variance computation.
    # The variance is constraint to be a single constant per each irreducible subspace.
    # Hence, we compute the empirical variance, and average within each irreducible subspace.
    n_samples = x_flat.shape[0]

    x_c_irrep_spectral = torch.einsum("ij,...j->...i", Q_inv.to(device=x_flat.device), x_flat - center)
    var_spectral = torch.sum(x_c_irrep_spectral**2, dim=0) / (n_samples - 1)

    # Vectorized averaging over irreducible subspace dimensions
    if "irrep_dims" not in rep_x.attributes:
        irrep_dims = torch.tensor([rep_x.group.irrep(*irrep_id).size for irrep_id in rep_x.irreps], device=x.device)
        rep_x.attributes["irrep_dims"] = irrep_dims
    else:
        irrep_dims = rep_x.attributes["irrep_dims"].to(device=x.device)

    if "irrep_indices" not in rep_x.attributes:
        # Create indices for each irrep subspace: [0,0,0,1,1,2,2,2,2,...] for irrep dims [3,2,4,...]
        indices = torch.arange(len(irrep_dims)).to(device=irrep_dims.device)
        irrep_indices = torch.repeat_interleave(indices, irrep_dims)
        rep_x.attributes["irrep_indices"] = irrep_indices
    else:
        irrep_indices = rep_x.attributes["irrep_indices"].to(device=x.device)

    # Compute average variance for each irrep subspace using scatter operations
    avg_vars = torch.zeros(len(irrep_dims), device=x.device, dtype=var_spectral.dtype)

    # Sum variances within each irrep subspace using scatter_add_:
    # For irrep_indices = [0,0,0,1,1,2,2,2,2] and var_spectral = [v0,v1,v2,v3,v4,v5,v6,v7,v8]
    # This computes: avg_vars[0] = v0+v1+v2, avg_vars[1] = v3+v4, avg_vars[2] = v5+v6+v7+v8
    avg_vars.scatter_add_(0, irrep_indices, var_spectral)
    avg_vars = avg_vars / irrep_dims

    # Broadcast back to full spectral dimensions
    var_spectral = avg_vars[irrep_indices]

    var_result = torch.einsum("ij,...j->...i", Q_squared.to(device=x.device), var_spectral)
    return var_result


def var_mean(x: Tensor, rep_x: Representation):
    """Compute the mean and variance of a symmetric random variable.

    Args:
        x: (:class:`torch.Tensor`) of shape :math:`(N, D_x)` containing the observations of the symmetric random
        variable
        rep_x: (:class:`~escnn.group.Representation`) representation of the symmetric random variable.

    Returns:
        (:class:`torch.Tensor`, :class:`torch.Tensor`): Mean and variance of the symmetric random variable.
        The mean is restricted to be in the trivial/G-invariant subspace of the symmetric vector space.
        The variance is constrained such that in the irrep-spectral basis, each G-irreducible subspace
        (i.e., each subspace associated with an irrep) has the same variance in all dimensions of that subspace.

    Shape:
        - **x**: :math:`(N, D_x)` or :math:`(N, D_x, T)` where N is the number of samples, D_x is the dimension of
          the symmetric random variable, and T is the sequence length (if applicable).
        - **Output**: A tuple containing the variance and the mean. The variance has shape :math:`(D_x,)` and the mean
          has shape :math:`(D_x,)`. If a sequence is provided (T dimension), the shapes are :math:`(D_x, T)`.
    """
    # Compute mean first
    mean_result = mean(x, rep_x)
    # Compute variance using the computed mean
    var_result = var(x, rep_x, center=mean_result)
    return var_result, mean_result


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
       of :math:`\mathrm{End}_G(\rho_k)` via Frobenius inner products (see
       `arXiv:2505.19809 <https://arxiv.org/abs/2505.19809>`_);
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
            - C_xy: :math:`(m_y d,\; m_x d)` projected covariance.
            - Z_xy: :math:`(m_y,\; m_x,\; B)`, free coefficients of each cross-covariance between irrep subspaces,
              representing basis expansion coefficients in the basis of endomorphisms of the irrep subspaces.
              Where :math:`B = 1, 2, 4` for real, complex, quaternionic irreps, respectively.
    """
    irrep_id = rep_x.irreps[0]  # Irrep id of the isotypic subspace
    assert rep_x.size == x.shape[-1], f"Expected signal shape to be (..., {rep_x.size}) got {x.shape}"
    assert rep_x.attributes.get("is_isotypic_rep", False), f"Expected rep of a single type, got {rep_x.irreps}"

    if y is not None:
        assert rep_y.size == y.shape[-1], f"Expected signal shape to be (..., {rep_y.size}) got {y.shape}"
        assert rep_y.attributes.get("is_isotypic_rep", False), f"Expected rep of a single type, got {rep_y.irreps}"

    # Get information about the irreducible representation present in the isotypic subspace
    irrep_dim = rep_x.attributes["irrep_dim"]
    # irrep_end_basis := (dim(End(irrep)), dim(irrep), dim(irrep))
    irrep_end_basis = rep_x.attributes["irrep_endomorphism_basis"].to(device=x.device, dtype=x.dtype)

    if y is None:
        rep_y = rep_x  # Use the same representation for Y
        y = x

    m_x = rep_x.attributes["irrep_multiplicity"]  # Multiplicity of the irrep in X
    m_y = rep_y.attributes["irrep_multiplicity"]  # Multiplicity of the irrep in Y

    x_iso, y_iso = x, y

    is_inv_subspace = irrep_id == rep_x.group.trivial_representation.id
    if is_inv_subspace:  # Nothing to do, return empirical covariance.
        x_iso = x - torch.mean(x, dim=0, keepdim=True)
        y_iso = y - torch.mean(y, dim=0, keepdim=True)
        Cxy_iso = torch.einsum("...y,...x->yx", y_iso, x_iso) / (x_iso.shape[0] - 1)
        return Cxy_iso, Cxy_iso  # Invariant subspace covariance is the same as the covariance matrix.

    # Compute empirical cross-covariance
    Cxy_iso = torch.einsum("...y,...x->yx", y_iso, x_iso) / (x_iso.shape[0] - 1)
    # Reshape from (my * d, mx * d) to (my, mx, d, d)
    Cxy_irreps = Cxy_iso.view(m_y, irrep_dim, m_x, irrep_dim).permute(0, 2, 1, 3).contiguous()
    # Compute basis expansion coefficients of each irrep cross-covariance in basis of End(irrep) ========
    # Frobenius inner product  <C , Ψ_b>  =  Σ_{i,j} C_{ij} Ψ_b,ij
    Cxy_irreps_basis_coeff = torch.einsum("mnij,bij->mnb", Cxy_irreps, irrep_end_basis)  # (m_y , m_x , B)
    # squared norms ‖Ψ_b‖²
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
    are the change-of-basis matrices to the isotypic bases of :math:`\mathcal{X}` and :math:`\mathcal{Y}`,
    respectively; :math:`\mathbf{C}_{xy}^{(k)}` is the covariance restricted to the
    isotypic subspaces of type *k*; and :math:`\mathbf{Z}_b^{(k)}` are the free
    parameters—i.e. the expansion coefficients in the endomorphism basis
    :math:`\mathbb{B}_k` of the irreducible representation of type *k*.

    Args:
        x (Tensor): Realizations of a random variable :math:`X`.
        y (Tensor): Realizations of a random variable :math:`Y`.
        rep_x (Representation): The representation acting on the symmetric vector spaces :math:`\mathcal{X}`.
        rep_y (Representation): The representation acting on the symmetric vector spaces :math:`\mathcal{Y}`.

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
    rep_X_iso_subspaces = rep_X_iso.attributes["isotypic_reps"]
    rep_Y_iso_subspaces = rep_Y_iso.attributes["isotypic_reps"]
    iso_idx_X = rep_X_iso.attributes["isotypic_subspace_dims"]
    iso_idx_Y = rep_Y_iso.attributes["isotypic_subspace_dims"]

    # Changes of basis from the Disentangled/Isotypic-basis of X, and Y to the original basis.
    Qx = torch.tensor(rep_X_iso.change_of_basis, device=x.device, dtype=x.dtype)
    Qy = torch.tensor(rep_Y_iso.change_of_basis, device=y.device, dtype=y.dtype)

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
