import torch
from escnn.group import Representation
from torch.nn.utils import parametrize

from symm_learning.nn.parametrizations import CommutingConstraint, InvariantConstraint
from symm_learning.representation_theory import GroupHomomorphismBasis, isotypic_decomp_rep


class eLinear(torch.nn.Linear):
    """Equivariant Linear layer between two representations using torch parametrizations."""

    def __init__(self, in_rep: Representation, out_rep: Representation, bias: bool = True):
        super().__init__(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
        self.in_rep = in_rep
        self.out_rep = out_rep
        # Register parametrizations enforcing equivariance
        parametrize.register_parametrization(self, "weight", CommutingConstraint(in_rep, out_rep))
        if bias:
            parametrize.register_parametrization(self, "bias", InvariantConstraint(out_rep))


class eLinear2(torch.nn.Linear):
    r"""Equivariant linear map `A in Hom_G(in_rep, out_rep)` parameterized in the isotypic basis.

    **Structure.** Decompose input/output into isotypic components for irreps
    :math:`\{\bar\rho_k\}` with multiplicities :math:`m_k^{\text{in}}, m_k^{\text{out}}`
    and carrier dimension :math:`r_k`. Then
    :math:`\mathrm{Hom}_G \cong \bigoplus_k \mathbb{R}^{m_k^{\text{out}}\times m_k^{\text{in}}}
    \otimes \mathrm{End}_G(\bar V_k)`. Each block uses basis
    :math:`\{E_{pq} \otimes \Phi_k(d)\}`, where :math:`E_{pq}` are multiplicity units and
    :math:`\{\Phi_k(d)\}_{d=1}^{D_k}` spans :math:`\mathrm{End}_G(\bar V_k)` with
    :math:`D_k \in \{1,2,4\}` for real/complex/quaternionic types.

    **Parameterization.**
    For each common irrep :math:`k`, learn
    :math:`\Theta_k \in \mathbb{R}^{D_k \times m_k^{\text{out}} \times m_k^{\text{in}}}` and synthesize
    :math:`A_k = \sum_{d=1}^{D_k} \Theta_k[d]\otimes \Phi_k(d)
      \in \mathbb{R}^{(m_k^{\text{out}} r_k)\times(m_k^{\text{in}} r_k)}`.
    Assemble blocks in isotypic coordinates and conjugate with
    :math:`Q_{\text{out}}` and :math:`Q_{\text{in}}^{-1}`.

    **Bias.**
    Allowed iff the trivial irrep appears in ``out_rep``; the bias lives in the invariant
    subspace and is parameterized in its multiplicity coordinates.

    Notes:
    -----
    * :math:`\Phi_k(d)` are pre-normalized to unit Frobenius norm.
    * Uses cached :math:`Q_{\text{out}}` and :math:`Q_{\text{in}}^{-1}` buffers.
    * In eval mode, expanded tensors are cached and refreshed on DoF version changes.

    Parameters
    ----------
    in_rep : escnn.group.Representation Input representation of :math:`G`.
    out_rep : escnn.group.Representation Output representation of :math:`G`.
    bias : bool, default=True. If True and the trivial irrep is present in ``out_rep``, enables an invariant bias.

    Attributes:
    ----------
    hom_basis : GroupHomomorphismBasis. Per-irrep slices, multiplicities, and unit-Frobenius endomorphism bases.
    Qin_inv : torch.Tensor. Inverse change-of-basis to isotypic coords for the input, shape ``[Nx, Nx]``.
    Qout : torch.Tensor. Change-of-basis from isotypic to original coords for the output, shape ``[Ny, Ny]``.
    """

    def __init__(self, in_rep: Representation, out_rep: Representation, bias: bool = True):
        super().__init__(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
        # Delete linear unconstrained module parameters
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        # Instanciate the handler of the basis of Hom_G(in_rep, out_rep)
        self.hom_basis = GroupHomomorphismBasis(in_rep, out_rep)
        self.in_rep, self.out_rep = self.hom_basis.in_rep, self.hom_basis.out_rep
        # Assert bias vector is feasible given out_rep symmetries
        trivial_id = self.hom_basis.G.trivial_representation.id
        can_have_bias = out_rep._irreps_multiplicities.get(trivial_id, 0) > 0
        self.has_bias = bias and can_have_bias
        # Register change of basis matrices as buffers
        dtype = torch.get_default_dtype()
        self.register_buffer("Qin_inv", torch.tensor(self.in_rep.change_of_basis_inv, dtype=dtype))
        self.register_buffer("Qout", torch.tensor(self.out_rep.change_of_basis, dtype=dtype))

        # Register parameters for each irrep block in Hom_G(out_rep, in_rep)
        for irrep_id, irrep_metadata in self.hom_basis.blocks.items():
            m_out, m_in = irrep_metadata["mul_out"], irrep_metadata["mul_in"]
            dk = irrep_metadata["irrep_dim"]
            end_basis = irrep_metadata["endomorphism_basis"]
            dim_end_basis = end_basis.shape[0]
            # Register flatted endomorphism basis (S_k, d_k, d_k) -> (S_k, d_k*d_k)
            self.register_buffer(f"{irrep_id}_end_basis_flat", end_basis.reshape(-1, dk * dk))
            block_basis = end_basis[:, :, None, :, None].expand(-1, -1, m_out, -1, m_in)
            block_basis = block_basis.permute(0, 2, 1, 3, 4).reshape(dim_end_basis, m_out * dk, m_in * dk)
            self.register_buffer(f"{irrep_id}_block_basis", block_basis)
            # Register parameters per irrep block.  # (S_k, m_out * m_in)
            irrep_block_param = torch.nn.Parameter(torch.zeros(dim_end_basis, m_out * m_in), requires_grad=True)
            self.register_parameter(f"{irrep_id}_DoF_flat", irrep_block_param)

            if self.has_bias and irrep_id == trivial_id:
                # Number of bias trainable parameters are equal to the output multiplicity of the trivial irrep
                bias_param = torch.nn.Parameter(torch.zeros(m_out), requires_grad=True)
                self.register_parameter("bias_DoF", bias_param)
        # Buffers to hold the expanded parameters
        self.register_buffer("_weight", torch.zeros((out_rep.size, in_rep.size), dtype=dtype))
        if self.has_bias:
            self.register_buffer("_bias", torch.zeros((out_rep.size,), dtype=dtype))

        self._expand_weight_iso_basis()
        self._expand_bias()

    @property
    def weight(self) -> torch.Tensor:
        r"""Dense equivariant weight :math:`W \in \mathbb{R}^{\dim(\mathrm{out}),\,\dim(\mathrm{in})}`.

        Computation
        -----------
        1. Build :math:`W_{\text{iso}}` blockwise via
           :math:`W_k = \sum_d \Theta_k[d]\otimes \Phi_k(d)`.
        2. Change of original input/output basis:
           :math:`W = Q_{\text{out}}\, W_{\text{iso}}\, Q_{\text{in}}^{-1}`.

        Caching
        -------
        * **Training:** recomputed for autograd.
        * **Eval:** memoized and refreshed when DoF versions change.

        Returns:
        -------
            torch.Tensor Equivariant linear map of shape `(out_rep.size, in_rep.size)`.
        """
        if self.training:
            W_iso = self._expand_weight_iso_basis()
            self._weight = self.Qout @ W_iso @ self.Qin_inv
        return self._weight

    @property
    def bias(self) -> torch.Tensor:
        r"""Invariant bias :math:`b \in \mathrm{Fix}_G(\mathrm{out\_rep})`.

        If the trivial irrep is present with multiplicity :math:`m_{\text{triv}}^{\text{out}}`,
        parameterize :math:`\beta \in \mathbb{R}^{m_{\text{triv}}^{\text{out}}}` and embed via

        .. math::
            b \;=\; Q_{\text{out}}[:,\,\mathcal{I}_{\text{triv}}]\;\beta,

        where :math:`\mathcal{I}_{\text{triv}}` selects the trivial isotypic columns
        (note :math:`r_{\text{triv}}=1`). Caching mirrors :meth:`weight`.

        Returns:
        -------
        torch.Tensor or None
            Invariant bias vector of shape `(out_rep.size,)` or ``None`` if not admissible.
        """
        if not self.has_bias:
            return None
        if self.training:
            self._bias = self._expand_bias()
        return self._bias

    def _expand_weight_iso_basis(self) -> torch.Tensor:
        """Expand the weight matrix in Hom_G(out_rep, in_rep) from the constrained trainable parameters."""
        W_iso = torch.zeros((self.out_rep.size, self.in_rep.size), dtype=self.Qin_inv.dtype, device=self.Qin_inv.device)
        for irrep_id, irrep_metadata in self.hom_basis.blocks.items():
            out_slice, in_slice = irrep_metadata["out_slice"], irrep_metadata["in_slice"]
            m_out, m_in = irrep_metadata["mul_out"], irrep_metadata["mul_in"]
            d_k = irrep_metadata["irrep_dim"]
            params_k_flat = getattr(self, f"{irrep_id}_DoF_flat")  # [S_k, m_out*m_in]

            # Option 1: memory heavy.
            # block_basis_k = getattr(self, f"{irrep_id}_block_basis")  # [S_k, m_out*d_k, m_in*d_k]
            # W_iso[out_slice, in_slice] = torch.einsum("boi,bk->oi", block_basis_k, params_k_flat)

            # Option 2: memory light.
            end_basis_k_flat = getattr(self, f"{irrep_id}_end_basis_flat")  # [S_k, d_k*d_k]
            blocks_flat = params_k_flat.transpose(0, 1) @ end_basis_k_flat  # [m_out*m_in, d_k*d_k]
            W_iso[out_slice, in_slice] = (
                blocks_flat.view(m_out, m_in, d_k, d_k).permute(0, 2, 1, 3).reshape(m_out * d_k, m_in * d_k)
            )

            # Reshape to stacked blocks (m_out*m_in, d_k*d_k) -> (m_out * d_k, m_in * d_k)
        return W_iso

    def _expand_bias(self) -> torch.Tensor:
        """Expand the bias vector in the invariant subspace of out_rep from the constrained trainable parameters."""
        if not self.has_bias:
            return None
        # Recompute bias from trainable parameters
        trivial_id = self.out_rep.group.trivial_representation.id
        trivial_indices = self.hom_basis.blocks[trivial_id]["out_slice"]
        bias_dof = getattr(self, "bias_DoF")
        return torch.mv(self.Qout[:, trivial_indices], bias_dof)


class eAffine(torch.nn.Module):
    r"""Applies a symmetry-preserving affine transformation y = x * alpha + beta to the input x.

    The affine transformation for a given input :math:`x \in \mathcal{X} \subseteq \mathbb{R}^{D_x}` is defined as:

    .. math::
        \mathbf{y} = \mathbf{x} \cdot \alpha + \beta

    such that

    .. math::
        \rho_{\mathcal{X}}(g) \mathbf{y} = (\rho_{\mathcal{X}}(g) \mathbf{x}) \cdot \alpha + \beta \quad \forall g \in G

    Where :math:`\mathcal{X}` is a symmetric vector space with group representation
    :math:`\rho_{\mathcal{X}}: G \to \mathbb{GL}(D_x)`, and :math:`\alpha \in \mathbb{R}^{D_x}`,
    :math:`\beta \in \mathbb{R}^{D_x}` are symmetry constrained learnable vectors.

    Args:
        in_rep: the :class:`escnn.group.Representation` group representation of the input feature space.
        bias: a boolean value that when set to ``True``, this module has a learnable bias vector
            in the invariant subspace of the input representation Default: ``True``

    Shape:
        - Input: of shape `(..., D)` where :math:`D` is the dimension of the input type.
        - Output: of shape `(..., D)`
    """

    def __init__(self, in_rep: Representation, bias: bool = True):
        super().__init__()
        self.in_rep, self.out_rep = in_rep, in_rep

        self.rep_x = in_rep
        G = self.rep_x.group
        default_dtype = torch.get_default_dtype()
        self.register_buffer("Q", torch.tensor(self.rep_x.change_of_basis, dtype=default_dtype))
        self.register_buffer("Q_inv", torch.tensor(self.rep_x.change_of_basis_inv, dtype=default_dtype))

        # Symmetry-preserving scaling implies scaling each irreducible subspace uniformly.
        irrep_dims_list = [G.irrep(*irrep_id).size for irrep_id in self.rep_x.irreps]
        irrep_dims = torch.tensor(irrep_dims_list, dtype=torch.long)
        n_scale_params = len(irrep_dims_list)
        self.register_parameter("scale_dof", torch.nn.Parameter(torch.ones(n_scale_params, dtype=default_dtype)))
        self.register_buffer(
            "irrep_indices", torch.repeat_interleave(torch.arange(len(irrep_dims), dtype=torch.long), irrep_dims)
        )

        has_invariant_subspace = G.trivial_representation.id in self.rep_x.irreps
        self.has_bias = bias and has_invariant_subspace
        if self.has_bias:
            is_trivial_irrep = torch.tensor(
                [irrep_id == G.trivial_representation.id for irrep_id in self.rep_x.irreps], dtype=torch.bool
            )
            n_bias_params = int(is_trivial_irrep.sum().item())
            self.register_parameter("bias_dof", torch.nn.Parameter(torch.zeros(n_bias_params, dtype=default_dtype)))
            dim_to_param = torch.full((self.rep_x.size,), -1, dtype=torch.long)
            offset = 0
            bias_idx = 0
            for is_trivial, dim in zip(is_trivial_irrep.tolist(), irrep_dims_list):
                if is_trivial:
                    dim_to_param[offset : offset + dim] = bias_idx
                    bias_idx += 1
                offset += dim
            self.register_buffer("bias_dim_to_param", dim_to_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the affine transformation; works for any input with last dim ``D``."""
        if x.shape[-1] != self.rep_x.size:
            raise ValueError(f"Expected last dimension {self.rep_x.size}, got {x.shape[-1]}")

        x_spectral = torch.einsum("ij,...j->...i", self.Q_inv, x)

        scale_spec, bias_spec = self.spectral_parameters(device=x_spectral.device, dtype=x_spectral.dtype)
        x_spectral = x_spectral * scale_spec.view(*([1] * (x_spectral.ndim - 1)), -1)

        if bias_spec is not None:
            x_spectral = x_spectral + bias_spec.view(*([1] * (x_spectral.ndim - 1)), -1)

        y = torch.einsum("ij,...j->...i", self.Q, x_spectral)
        return y

    def spectral_parameters(self, device=None, dtype=None):
        """Return per-dimension spectral scale and bias vectors."""
        scale = self.scale_dof[self.irrep_indices].to(device=device, dtype=dtype or self.scale_dof.dtype)
        bias = None
        if self.has_bias and self.bias_dof.numel() > 0:
            bias_index = self.bias_dim_to_param
            valid = bias_index >= 0
            if valid.any():
                bias = torch.zeros(
                    self.rep_x.size, device=scale.device if device is None else device, dtype=dtype or scale.dtype
                )
                bias_vals = self.bias_dof.to(device=bias.device, dtype=bias.dtype)
                bias_index = bias_index.to(bias.device)
                bias[valid] = bias_vals[bias_index[valid]]
        return scale, bias

    def reset_parameters(self) -> None:  # noqa: D102
        torch.nn.init.ones_(self.scale_dof)
        if self.has_bias and self.bias_dof is not None:
            torch.nn.init.zeros_(self.bias_dof)

    def extra_repr(self) -> str:  # noqa: D102
        return f"in_rep{self.in_rep} bias={self.has_bias}"


if __name__ == "__main__":
    import escnn
    from escnn.group import CyclicGroup, DihedralGroup, directsum
    from numpy import set_printoptions
    from torch.profiler import ProfilerActivity, profile, record_function

    set_printoptions(precision=2, suppress=True)

    from symm_learning.utils import check_equivariance

    G = DihedralGroup(3)
    m = 5
    rep = directsum([G.regular_representation] * m)
    rep = escnn.group.change_basis(rep, change_of_basis=rep.change_of_basis_inv, name="iso")

    layer2 = eLinear2(rep, rep, bias=True)
    layer = eLinear(rep, rep, bias=True)
    check_equivariance(layer, atol=1e-5, rtol=1e-5)
    check_equivariance(layer2, atol=1e-5, rtol=1e-5)

    # Identity should be preserved under projection to Hom_G(rep, rep)
    W_id = torch.eye(rep.size)
    # print(W_id.detach().cpu().numpy())
    layer.weight = W_id
    W_id_proj = layer.weight  # Trigger Hom_G(rep, rep) projection
    # print(W_id_proj.detach().cpu().numpy())
    assert torch.allclose(W_id, W_id_proj, atol=1e-5, rtol=1e-5), (
        f"Identity reprojection error max: {(W_id - W_id_proj).abs().max()}"
    )
    print("Identity reprojection test passed.")

    # Check the orthogonal projection to Hom_G(rep, rep) works as expected by
    # leving invariant a linear map already in Hom_G(rep, rep).
    in_type = escnn.nn.FieldType(escnn.gspaces.no_base_space(G), [rep])
    layer = eLinear(rep, rep, bias=True)
    escnn_layer = escnn.nn.Linear(in_type, in_type, bias=True)
    W, b = escnn_layer.expand_parameters()
    # Test that the projection of eLinear to the space of Hom_G(rep, rep) does not alter W,
    # which is already in Hom_G(rep, rep).
    layer.weight = W  # This trigg
    W_projected = layer.weight  # Parametrization projects to Hom_G(rep, rep)
    assert torch.allclose(W, W_projected, atol=1e-5, rtol=1e-5), f"Max err: {(W - W_projected).abs().max()}"
    check_equivariance(layer, atol=1e-5, rtol=1e-5)
    print("Projection invariance test passed.")

    # For any random linear map check the projected map is indeed in Hom_G(rep, rep)
    W_random = torch.randn_like(W)
    layer.weight = W_random
    check_equivariance(layer, atol=1e-5, rtol=1e-5)

    # Check projection is idempotent
    W = layer.weight
    layer.weight = W
    W_proj2 = layer.weight
    assert torch.allclose(W, W_proj2, atol=1e-5, rtol=1e-5), f"Max err: {(W - W_proj2).abs().max()}"
    print("Projection idempotence test passed.")

    def profile_forward_pass(module: torch.nn.Module, batch_size: int = 512, warmup: int = 10, iters: int = 100):
        """Profile the forward pass of ``module`` and print a summary table."""
        module_device = next(module.parameters()).device
        module.train()
        x = torch.randn(batch_size, module.in_rep.size, device=module_device)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        for _ in range(warmup):
            module(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            for _ in range(iters):
                with record_function("eLinear2_forward"):
                    module(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print("\nForward-pass profiler summary (sorted by self CPU time)")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        if torch.cuda.is_available():
            print("\nForward-pass profiler summary (sorted by self CUDA time)")
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    print("\nProfiling eLinear2 forward pass ...")
    profile_forward_pass(layer2.cuda())
