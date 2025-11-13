import escnn
import torch
from escnn.group import Representation
from escnn.gspaces import no_base_space
from escnn.nn import FieldType
from torch.nn.utils import parametrize

from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.representation_theory import isotypic_decomp_rep


class InvariantConstraint(torch.nn.Module):
    """Parametrization for a bias vector to be invariant under a group representation."""

    def __init__(self, rep: Representation):
        super().__init__()
        self.rep = rep
        self.inv_projector = invariant_orthogonal_projector(rep)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """Project b onto the invariant subspace using the precomputed projector."""
        self.inv_projector = self.inv_projector.to(b.device, b.dtype)
        return torch.mv(self.inv_projector, b)

    def right_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a parameter tensor whose projection equals ``tensor``."""
        print("[InvariantConstraint] right_inverse called")
        return tensor


class CommutingConstraint(torch.nn.Module):
    r"""Equivariant weight parametrization via isotypic-basis projection.

    Args:
        in_rep: (:class:`~escnn.group.Representation`) input representation.
        out_rep: (:class:`~escnn.group.Representation`) output representation.

    Every pair of matching isotypic subspaces (one from ``in_rep``, one from ``out_rep``) inherits the
    same irreducible type :math:`\bar\rho_k`. Inside that block, any equivariant map factors as

    .. math::
        A_{i,j}^{(k)} = \sum_{b \in \mathbb{B}_k} \Theta^{(k)}_{b,\,i,j} \,\Psi_k(b),

    where :math:`\Psi_k(b)` spans :math:`\mathrm{End}_G(\bar\rho_k)` and
    :math:`\Theta^{(k)}_{b,\,i,j} = \langle A_{i,j}^{(k)}, \Psi_k(b) \rangle / \|\Psi_k(b)\|^2`.
    The combinatorics over multiplicity indices is captured by Kronecker-factoring these basis
    endomorphisms with canonical ``E_{ij}`` selectors.

    Implementation detail: we prebuild all
    :math:`E_{ij} \otimes \Psi_k(b)` blocks once, stack them into a tall matrix, and perform the whole
    projection as two batched GEMV operations—one to gather all Frobenius inner products, one to
    reconstruct the projected weight. This keeps the projection GPU-friendly while matching the
    orthogonal projection defined by Proposition H.13.
    """

    def __init__(self, in_rep: Representation, out_rep: Representation):
        super().__init__()
        self.in_rep = isotypic_decomp_rep(in_rep)
        self.out_rep = isotypic_decomp_rep(out_rep)

        G = self.in_rep.group
        self.register_buffer("Qin", torch.tensor(self.in_rep.change_of_basis, dtype=torch.float))
        self.register_buffer("Qout", torch.tensor(self.out_rep.change_of_basis, dtype=torch.float))

        self.common_irreps = sorted(set(self.in_rep.irreps).intersection(set(self.out_rep.irreps)))
        dtype = torch.get_default_dtype()
        total_basis_dim = 0
        basis_blocks = []
        basis_norms_chunks = []
        self.irreps_meta = []
        for irrep_id in self.common_irreps:
            irrep_out_slice = self.out_rep.attributes["isotypic_subspace_dims"][irrep_id]
            irrep_in_slice = self.in_rep.attributes["isotypic_subspace_dims"][irrep_id]
            mul_out = self.out_rep._irreps_multiplicities[irrep_id]
            mul_in = self.in_rep._irreps_multiplicities[irrep_id]
            irrep = G.irrep(*irrep_id)
            endo_basis = torch.tensor(irrep.endomorphism_basis(), dtype=dtype)
            irrep_dim = irrep.size
            self.irreps_meta.append((irrep_out_slice, irrep_in_slice, endo_basis, mul_out, mul_in, irrep_dim))

            # Build Kron(E_{ij}, Psi_b) for every multiplicity pair (i, j).
            pair_blocks = []
            for i in range(mul_out):
                out_block = slice(i * irrep_dim, (i + 1) * irrep_dim)
                for j in range(mul_in):
                    in_block = slice(j * irrep_dim, (j + 1) * irrep_dim)
                    block = torch.zeros(
                        endo_basis.shape[0],
                        mul_out * irrep_dim,
                        mul_in * irrep_dim,
                        dtype=self.Qin.dtype,
                    )
                    block[:, out_block, in_block] = endo_basis.to(self.Qin.dtype)
                    pair_blocks.append(block)
            pair_blocks = (
                torch.cat(pair_blocks, dim=0)
                if pair_blocks
                else torch.empty(0, mul_out * irrep_dim, mul_in * irrep_dim, dtype=self.Qin.dtype)
            )

            rows = pair_blocks.shape[0]
            container = torch.zeros(rows, self.out_rep.size, self.in_rep.size, dtype=self.Qin.dtype)
            container[:, irrep_out_slice, irrep_in_slice] = pair_blocks
            basis_blocks.append(container)

            base_norms = torch.einsum("bij,bij->b", endo_basis, endo_basis).to(self.Qin.dtype)
            basis_norms_chunks.append(base_norms.repeat(mul_out * mul_in))
            total_basis_dim += rows

        if basis_blocks:
            basis_vectors = torch.cat(basis_blocks, dim=0)
            basis_vectors_flat = basis_vectors.reshape(total_basis_dim, -1)
            basis_norms = torch.cat(basis_norms_chunks, dim=0)
            basis_inv_norms = torch.zeros_like(basis_norms)
            mask = basis_norms > 0
            basis_inv_norms[mask] = basis_norms[mask].reciprocal()
        else:
            basis_vectors = torch.zeros(0, self.out_rep.size, self.in_rep.size, dtype=self.Qin.dtype)
            basis_vectors_flat = basis_vectors.reshape(0, -1)
            basis_inv_norms = torch.zeros(0, dtype=self.Qin.dtype)

        self.register_buffer("basis_vectors", basis_vectors)
        self.register_buffer("basis_vectors_flat", basis_vectors_flat)
        self.register_buffer("basis_inv_norms", basis_inv_norms)
        self.num_basis = total_basis_dim

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """Project W onto Hom_G(out_rep, in_rep) using a single GEMM-based contraction."""
        assert W.shape[-2:] == (self.out_rep.size, self.in_rep.size), (
            f"Expected weight shape (..., {self.out_rep.size}, {self.in_rep.size}), got {W.shape}"
        )

        W_iso = self.Qout.T @ W @ self.Qin  # [Ny, Nx]

        W_eq = self._project_gemm(W_iso) if self.num_basis > 0 else torch.zeros_like(W_iso)

        return self.Qout @ W_eq @ self.Qin.T  # [Ny, Nx]

    def _project_gemm(self, W_iso: torch.Tensor) -> torch.Tensor:
        r"""Project in isotypic coordinates using batched Frobenius inner products.

        Args:
            W_iso: (:class:`torch.Tensor`) weight in the isotypic basis with shape
                :math:`(\\dim \\rho_{out}, \\dim \\rho_{in})`.

        Returns:
            (:class:`torch.Tensor`): Equivariant projection of ``W_iso`` with the same shape.

        The method flattens ``W_iso`` and the stacked basis tensors once so both
        :math:`c_i = \\langle B_i, W \\rangle_F` and the synthesis
        :math:`\\sum_i c_i B_i` are computed by standard matrix-vector multiplications.
        """
        # Flatten once so that all Frobenius inner products <W_iso, B_i> become matrix-vector products.
        W_flat = W_iso.reshape(-1)
        coeff = torch.mv(self.basis_vectors_flat, W_flat)  #
        scaled_coeff = coeff * self.basis_inv_norms
        W_eq_flat = torch.mv(self.basis_vectors_flat.t(), scaled_coeff)  # [Ny * Nx]
        return W_eq_flat.reshape_as(W_iso)

    @torch.no_grad()
    def sample_equivariant_iso(self, scheme: str = "xavier_uniform"):
        """Return W_iso ~ Hom_G(out, in), initialized per-irrep with Xavier/He."""
        #
        device = self.Qin.device
        W_iso = torch.zeros(self.out_rep.size, self.in_rep.size, dtype=self.Qin.dtype, device=device)

        for out_slice, in_slice, endo_basis, m_out, m_in, irrep_dim in self.irreps_meta:
            dim_endo_basis, _, _ = endo_basis.shape  # number of basis elements
            dtype = endo_basis.dtype
            # fans for this irrep
            fan_in = dim_endo_basis * m_in
            fan_out = dim_endo_basis * m_out
            # isotypic_param_shape := (dim_irrep_endomorphism, irep_multiplicity_out, irrep_multiplicity_in)
            isotypic_param_shape = (dim_endo_basis, m_out, m_in)
            if scheme == "xavier_uniform":
                bound = (6.0 / (fan_in + fan_out)) ** 0.5
                theta = torch.empty(*isotypic_param_shape, device=device, dtype=dtype).uniform_(-bound, bound)
            elif scheme == "xavier_normal":
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                theta = torch.empty(*isotypic_param_shape, device=device, dtype=dtype).normal_(0.0, std)
            elif scheme in {"kaiming_normal", "he_normal"}:
                std = (2.0 / fan_in) ** 0.5
                theta = torch.empty(*isotypic_param_shape, device=device, dtype=dtype).normal_(0.0, std)
            elif scheme in {"kaiming_uniform", "he_uniform"}:
                bound = (6.0 / fan_in) ** 0.5
                theta = torch.empty(*isotypic_param_shape, device=device, dtype=dtype).uniform_(-bound, bound)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            # Synthesize Σ_b kron(Θ_b, Ψ_b)
            # basis[b] is [d, d], theta[b] is [m_out, m_in]
            block = torch.zeros(m_out * irrep_dim, m_in * irrep_dim, device=device, dtype=dtype)
            for b in range(dim_endo_basis):
                block.add_(torch.kron(theta[b], endo_basis[b]))
            W_iso[out_slice, in_slice] = block

        return W_iso

    def right_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a pre-image for the parametrization (identity for now)."""
        print("[CommutingConstraint] right_inverse called")
        return tensor


if __name__ == "__main__":
    from escnn.group import CyclicGroup, Icosahedral, directsum

    from symm_learning.nn.linear import eLinear, eLinear2

    G = Icosahedral()

    in_rep = directsum([G.regular_representation] * 2)
    out_rep = directsum([G.regular_representation] * 2)

    std_layer = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=False)
    eq_layer_param = eLinear(in_rep, out_rep, bias=False)
    eq_layer_basis = eLinear2(in_rep, out_rep, bias=False)

    in_type = FieldType(no_base_space(G), [in_rep])
    out_type = FieldType(no_base_space(G), [out_rep])
    escnn_layer = escnn.nn.Linear(in_type, out_type, bias=False)

    batch_size = 1024
    eq_layer_basis.cuda()
    device = eq_layer_basis.weight.device
    print(f"Device: {device}")

    def benchmark(module, x, iters=500, warmup=50, device="cuda"):
        """Benchmark a module and return independent forward/backward timings (in ms)."""
        torch.cuda.synchronize()
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        optim = torch.optim.SGD(module.parameters(), lr=0.00001)
        forward_times, backward_times = [], []

        for i in range(iters + warmup):
            optim.zero_grad()

            fwd_start.record()
            y = module(in_type(x)).tensor if isinstance(module, escnn.nn.Linear) else module(x)
            fwd_end.record()

            loss = y.pow(2).mean()

            bwd_start.record()
            loss.backward()
            bwd_end.record()
            optim.step()

            torch.cuda.synchronize()

            if i >= warmup:
                forward_times.append(fwd_start.elapsed_time(fwd_end))
                backward_times.append(bwd_start.elapsed_time(bwd_end))

        fwd_stats = torch.tensor(forward_times)
        bwd_stats = torch.tensor(backward_times)
        forward_mean = float(fwd_stats.mean().item())
        forward_std = float(fwd_stats.std(unbiased=False).item())
        backward_mean = float(bwd_stats.mean().item())
        backward_std = float(bwd_stats.std(unbiased=False).item())
        return (forward_mean, forward_std), (backward_mean, backward_std)

    x = torch.randn(batch_size, in_rep.size, device=device)

    modules_to_benchmark = [
        ("Standard", std_layer.cuda()),
        ("eLinear", eq_layer_param.cuda()),
        ("eLinear2", eq_layer_basis.cuda()),
        ("escnn", escnn_layer.cuda()),
    ]

    results = []
    for name, module in modules_to_benchmark:
        (fwd_mean, fwd_std), (bwd_mean, bwd_std) = benchmark(module, x)
        results.append(
            {
                "name": name,
                "fwd_mean": fwd_mean,
                "fwd_std": fwd_std,
                "bwd_mean": bwd_mean,
                "bwd_std": bwd_std,
            }
        )

    header = f"{'Layer':<12} {'Forward (ms)':>18} {'Backward (ms)':>18}"
    separator = "-" * len(header)
    print(f"\nBenchmark results per {batch_size}-sample batch")
    print(separator)
    print(header)
    print(separator)
    for res in results:
        fwd_str = f"{res['fwd_mean']:.3f} ± {res['fwd_std']:.3f}"
        bwd_str = f"{res['bwd_mean']:.3f} ± {res['bwd_std']:.3f}"
        print(f"{res['name']:<12} {fwd_str:>18} {bwd_str:>18}")
    print(separator)
