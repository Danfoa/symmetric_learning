"""Symmetric Learning - Statistics Utilities.

Functions for computing statistics of symmetric random variables that respect group
symmetry constraints.

Functions
---------
mean
    Compute the mean projected onto the G-invariant subspace.
var
    Compute variance respecting symmetry structure.
var_mean
    Compute variance and mean together efficiently.
cov
    Compute covariance between symmetric random variables.
"""

from __future__ import annotations

import torch
from escnn.group import Representation
from torch import Tensor

from symm_learning.linalg import equiv_orthogonal_projection, invariant_orthogonal_projector


def mean(x: Tensor, rep_x: Representation) -> Tensor:
    r"""Estimate the :math:`\mathbb{G}`-invariant mean of a random variable.

    Let :math:`\mathbf{X}: \Omega \to \mathcal{X}` be a random variable taking values in the symmetric vector space
    :math:`\mathcal{X}`, with group representation :math:`\rho_{\mathcal{X}}:\mathbb{G}\to\mathrm{GL}(\mathcal{X})`,
    and marginal density :math:`\mathbb{P}_{\mathbf{X}}`.
    Under the assumption that this marginal is invariant under the group action
    (i.e., a point and all its symmetric points have equal likelihood under the marginal), formally:

    .. math::
        \mathbb{P}_{\mathbf{X}}(\mathbf{x})
        = \mathbb{P}_{\mathbf{X}}\!\left(\rho_{\mathcal{X}}(g)\mathbf{x}\right),
        \quad \forall \mathbf{x}\in\mathcal{X},\ \forall g\in\mathbb{G},

    the true mean satisfies

    .. math::
        \mathbb{E}[\mathbf{X}] = \rho_{\mathcal{X}}(g)\,\mathbb{E}[\mathbf{X}], \quad \forall g\in\mathbb{G},

    hence :math:`\mathbb{E}[\mathbf{X}] \in \mathcal{X}^{\text{inv}}`.

    Implementation:
    from samples :math:`\{\mathbf{x}^{(n)}\}_{n=1}^N`, we first compute the empirical mean

    .. math::
        \widehat{\mathbb{E}}[\mathbf{X}] = \frac{1}{N}\sum_{n=1}^N \mathbf{x}^{(n)},

    and then project it onto the invariant subspace:

    .. math::
        \widehat{\mathbb{E}}_{\mathbb{G}}[\mathbf{X}]
        = \mathbf{P}_{\mathrm{inv}}\,\widehat{\mathbb{E}}[\mathbf{X}],
        \quad
        \mathbf{P}_{\mathrm{inv}} = \mathbf{Q}\mathbf{S}\mathbf{Q}^T,

    where :math:`\mathbf{S}` selects trivial-irrep coordinates in the irrep-spectral basis.
    (see :func:`~symm_learning.linalg.invariant_orthogonal_projector`).
    Under the repository's canonical isotypic ordering, this corresponds to the first isotypic block when present.

    Args:
        x: (:class:`~torch.Tensor`) samples of :math:`\mathbf{X}` with shape :math:`(N,D_x)` or
            :math:`(N,D_x,T)`; when a time axis is present it is folded into the sample axis.
        rep_x: (:class:`~escnn.group.Representation`) representation :math:`\rho_{\mathcal{X}}` on :math:`\mathcal{X}`.

    Returns:
        (:class:`~torch.Tensor`): Invariant mean vector in :math:`\mathcal{X}`.

    Shape:
        - **x**: :math:`(N,D_x)` or :math:`(N,D_x,T)`.
        - **Output**: :math:`(D_x,)`.

    Note:
        For repeated calls with the same representation object ``rep_x``, this function caches
        :math:`\mathbf{P}_{\mathrm{inv}}` in ``rep_x.attributes["invariant_orthogonal_projector"]``.
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
    mean_result = torch.einsum("ij,j->i...", P_inv.to(device=x_flat.device, dtype=mean_empirical.dtype), mean_empirical)
    return mean_result


def var(x: Tensor, rep_x: Representation, center: Tensor = None) -> Tensor:
    r"""Estimate the symmetry-constrained variance of :math:`\mathbf{X}:\Omega\to\mathcal{X}`.

    Let :math:`\mathbf{X}: \Omega \to \mathcal{X}` be a random variable taking values in the symmetric vector space
    :math:`\mathcal{X}`, with group representation :math:`\rho_{\mathcal{X}}:\mathbb{G}\to\mathrm{GL}(\mathcal{X})`,
    and marginal density :math:`\mathbb{P}_{\mathbf{X}}`.
    Under the assumption that this marginal is invariant under the group action
    (i.e., a point and all its symmetric points have equal likelihood under the marginal), formally:

    .. math::
        \mathbb{P}_{\mathbf{X}}(\mathbf{x})
        = \mathbb{P}_{\mathbf{X}}\!\left(\rho_{\mathcal{X}}(g)\mathbf{x}\right),
        \quad \forall \mathbf{x}\in\mathcal{X},\ \forall g\in\mathbb{G},

    the true variance in the irrep-spectral basis
    (:func:`~symm_learning.representation_theory.isotypic_decomp_rep`) is constant within each irreducible copy:

    .. math::
        \operatorname{Var}(\hat{\mathbf{X}}_{k,i,1})
        = \cdots =
        \operatorname{Var}(\hat{\mathbf{X}}_{k,i,d_k})
        = \sigma^2_{k,i}.

    Implementation:
    given samples :math:`\{\mathbf{x}^{(n)}\}_{n=1}^{N}`, we compute:

    1. Centering (using provided center or :func:`mean`):

    .. math::
        \widehat{\boldsymbol{\mu}} =
        \begin{cases}
        \texttt{center}, & \text{if provided} \\
        \widehat{\mathbb{E}}_{\mathbb{G}}[\mathbf{X}], & \text{otherwise}
        \end{cases}

    2. Empirical spectral variance:

    .. math::
        \hat{\mathbf{x}}^{(n)} = \mathbf{Q}^{T}(\mathbf{x}^{(n)}-\widehat{\boldsymbol{\mu}}),\qquad
        \widehat{v}_{j} = \frac{1}{N-1}\sum_{n=1}^{N}\left(\hat{x}^{(n)}_{j}\right)^2.

    3. Irrep-wise averaging for each copy :math:`(k,i)`:

    .. math::
        \widehat{\sigma}^{2}_{k,i}
        = \frac{1}{d_k}\sum_{r=1}^{d_k}\widehat{v}_{k,i,r},
        \qquad
        \widehat{v}_{k,i,1}=\cdots=\widehat{v}_{k,i,d_k}:=\widehat{\sigma}^{2}_{k,i}.

    4. Mapping back to the original basis:

    .. math::
        \widehat{\operatorname{Var}}(\mathbf{X}) = \mathbf{Q}^{\odot 2}\,\widehat{\mathbf{v}},

    where :math:`\mathbf{Q}^{\odot 2}` is the elementwise square of :math:`\mathbf{Q}` and
    :math:`\widehat{\mathbf{v}}` denotes the broadcast spectral variance vector after step 3.

    Args:
        x: (:class:`~torch.Tensor`) samples with shape :math:`(N,D_x)` or :math:`(N,D_x,T)`;
            the optional time axis is folded into samples.
        rep_x: (:class:`~escnn.group.Representation`) representation :math:`\rho_{\mathcal{X}}`.
        center: (:class:`~torch.Tensor`, optional) Center for variance computation. If None, computes the mean.

    Returns:
        (:class:`~torch.Tensor`): Variance vector in the original basis, consistent with
            the irrep-wise constraint above.

    Shape:
        - **x**: :math:`(N,D_x)` or :math:`(N,D_x,T)`.
        - **center**: :math:`(D_x,)` if provided.
        - **Output**: :math:`(D_x,)`.

    Note:
        For repeated calls with the same representation object ``rep_x``, this function caches and reuses:
        ``Q_inv``, ``Q_squared``, ``irrep_dims``, and ``irrep_indices`` in ``rep_x.attributes``.
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
    r"""Compute :func:`var` and :func:`mean` under symmetry constraints.

    Args:
        x: (:class:`~torch.Tensor`) samples with shape :math:`(N,D_x)` or :math:`(N,D_x,T)`.
        rep_x: (:class:`~escnn.group.Representation`) representation :math:`\rho_{\mathcal{X}}`.

    Returns:
        (:class:`~torch.Tensor`, :class:`~torch.Tensor`): Tuple ``(var, mean)`` where mean is projected to
        :math:`\mathcal{X}^{\text{inv}}` and variance satisfies irrep-wise isotropy in spectral basis.

    Shape:
        - **x**: :math:`(N,D_x)` or :math:`(N,D_x,T)`.
        - **Output**: ``(var, mean)`` both with shape :math:`(D_x,)`.

    Note:
        This function reuses the same caches as :func:`mean` and :func:`var` when called repeatedly with the
        same representation object ``rep_x``.
    """
    # Compute mean first
    mean_result = mean(x, rep_x)
    # Compute variance using the computed mean
    var_result = var(x, rep_x, center=mean_result)
    return var_result, mean_result


def cov(
    x: Tensor,
    y: Tensor,
    rep_x: Representation,
    rep_y: Representation,
    uncentered: bool = False,
):
    r"""Compute symmetry-aware cross-covariance.

    Let :math:`\mathbf{X}:\Omega\to\mathcal{X}` and :math:`\mathbf{Y}:\Omega\to\mathcal{Y}` be two
    :math:`\mathbb{G}`-invariant random variables endowed with the :math:`\mathbb{G}` representations
    :math:`\rho_{\mathcal{X}}` and :math:`\rho_{\mathcal{Y}}` respectively. This function computes the symmetry-aware
    cross-covariance which by construction is a :math:`\mathbb{G}`-equivariant linear map in
    :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}},\rho_{\mathcal{Y}})`.

    Implementation: To achieve this we first compute the symmetry-agnostic empirical covariance
    :math:`\mathbf{C}^{\text{raw}}_{yx} = \frac{1}{N-1}\sum_{n=1}^{N}\mathbf{y}^{\star}_n (\mathbf{x}^{\star}_n)^\top`.
    By default (:attr:`uncentered=False`), centered variables use invariant means from :func:`mean`:
    :math:`\mathbf{x}^{\star}_n = \mathbf{x}_n - \boldsymbol{\mu}_x`,
    :math:`\mathbf{y}^{\star}_n = \mathbf{y}_n - \boldsymbol{\mu}_y`,
    with
    :math:`\boldsymbol{\mu}_x = \widehat{\mathbb{E}}_{\mathbb{G}}[\mathbf{X}]`,
    :math:`\boldsymbol{\mu}_y = \widehat{\mathbb{E}}_{\mathbb{G}}[\mathbf{Y}]`.
    If :attr:`uncentered=True`, :math:`\mathbf{x}^{\star}_n=\mathbf{x}_n` and
    :math:`\mathbf{y}^{\star}_n=\mathbf{y}_n`.

    The returned covariance is the orthogonal projection of :math:`\mathbf{C}^{\text{raw}}_{yx}` onto
    :math:`\mathrm{Hom}_{\mathbb{G}}(\rho_{\mathcal{X}},\rho_{\mathcal{Y}})` via
    :func:`~symm_learning.linalg.equiv_orthogonal_projection`:

    .. math::
        \mathbf{C}_{yx}
        = \Pi_{\mathrm{Hom}_{\mathbb{G}}}(\mathbf{C}^{\text{raw}}_{yx}).

    This orthogonal projector is equivalent to the Reynolds/group-average operator:

    .. math::
        \Pi_{\mathrm{Hom}_{\mathbb{G}}}(\mathbf{A})
        = \frac{1}{|\mathbb{G}|}\sum_{g\in\mathbb{G}}
        \rho_{\mathcal{Y}}(g)\,\mathbf{A}\,\rho_{\mathcal{X}}(g^{-1}).

    Args:
        x (:class:`~torch.Tensor`): Samples of :math:`\mathbf{X}`.
        y (:class:`~torch.Tensor`): Samples of :math:`\mathbf{Y}`.
        rep_x (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\mathcal{X}}`.
        rep_y (:class:`~escnn.group.Representation`): Representation :math:`\rho_{\mathcal{Y}}`.
        uncentered (:class:`bool`): If ``False`` (default), subtract invariant means before covariance
            computation. If ``True``, compute the uncentered second moment and project it.

    Returns:
        :class:`~torch.Tensor`: Projected cross-covariance :math:`\mathbf{C}_{yx}` with shape
        :math:`(D_y, D_x)`.

    Shape:
        - **x**: :math:`(N, D_x)`. With `N` denoting the number of samples and `D_x` the dimension of the
            representation space of `x`.
        - **y**: :math:`(N, D_y)`. With `D_y` the dimension of the representation space of `y`.
        - **Output**: :math:`(D_y, D_x)`.
    """
    assert x.shape[1] == rep_x.size, f"Expected X shape (N, {rep_x.size}), got {x.shape}"
    assert y.shape[1] == rep_y.size, f"Expected Y shape (N, {rep_y.size}), got {y.shape}"
    assert x.shape[-1] == rep_x.size, f"Expected X shape (..., {rep_x.size}), got {x.shape}"
    assert y.shape[-1] == rep_y.size, f"Expected Y shape (..., {rep_y.size}), got {y.shape}"

    if uncentered:
        x_eff, y_eff = x, y
    else:
        mu_x = mean(x, rep_x)
        mu_y = mean(y, rep_y)
        x_eff = x - mu_x
        y_eff = y - mu_y

    # Empirical cross-covariance (centered or uncentered) in original coordinates: [D_y, D_x]
    Cxy_raw = torch.einsum("...y,...x->yx", y_eff, x_eff) / (x_eff.shape[0] - 1)
    # Orthogonal projection to Hom_G(rep_x, rep_y), preserving shape [D_y, D_x].
    Cxy_proj = equiv_orthogonal_projection(W=Cxy_raw, rep_x=rep_x, rep_y=rep_y)
    return Cxy_proj
