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


class CommutingConstraint(torch.nn.Module):
    r"""Equivariant weight parametrization using a precomputed orthonormal basis.

    Args:
        in_rep: (:class:`~escnn.group.Representation`) input representation.
        out_rep: (:class:`~escnn.group.Representation`) output representation.

    This parametrization expands all :math:`G`-equivariant linear maps in the isotypic basis
    :math:`\\{B_i\\}_{i=1}^K` returned by :func:`escnn.group.Representation.endomorphism_basis`.
    Each basis element is broadcast to the full multiplicity block once at initialization, so
    projecting a weight matrix boils down to two dense contractions:

    .. math::
        c_i = \\langle B_i, W \\rangle_F, \\quad
        \\Pi(W) = \\sum_i c_i B_i.

    Because the :math:`B_i` are orthonormal, no extra normalization is required and the projection can
    be implemented as a pair of GEMV calls.
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
        irrep_blocks = []
        for irrep_id in self.common_irreps:
            irrep_out_slice = self.out_rep.attributes["isotypic_subspace_dims"][irrep_id]
            irrep_in_slice = self.in_rep.attributes["isotypic_subspace_dims"][irrep_id]
            mul_out = self.out_rep._irreps_multiplicities[irrep_id]
            mul_in = self.in_rep._irreps_multiplicities[irrep_id]
            irrep = G.irrep(*irrep_id)
            basis = torch.tensor(irrep.endomorphism_basis(), dtype=dtype)
            irrep_dim = irrep.size
            irrep_blocks.append((irrep_out_slice, irrep_in_slice, basis, mul_out, mul_in, irrep_dim))
            total_basis_dim += basis.shape[0]

        basis_vectors = torch.zeros(
            total_basis_dim,
            self.out_rep.size,
            self.in_rep.size,
            dtype=self.Qin.dtype,
        )
        cursor = 0
        for out_slice, in_slice, basis, mul_out, mul_in, irrep_dim in irrep_blocks:
            block = (
                basis[:, None, :, None, :]
                .repeat(1, mul_out, 1, mul_in, 1)
                .reshape(basis.shape[0], mul_out * irrep_dim, mul_in * irrep_dim)
            )
            bdim = block.shape[0]
            basis_vectors[cursor : cursor + bdim, out_slice, in_slice] = block
            cursor += bdim

        self.register_buffer("basis_vectors", basis_vectors)
        self.register_buffer("basis_vectors_flat", basis_vectors.reshape(total_basis_dim, -1))
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
        W_flat = W_iso.reshape(-1)
        coeff = torch.mv(self.basis_vectors_flat, W_flat)  # [B]
        W_eq_flat = torch.mv(self.basis_vectors_flat.t(), coeff)  # [Ny * Nx]
        return W_eq_flat.reshape_as(W_iso)


if __name__ == "__main__":
    from escnn.group import CyclicGroup, DihedralGroup, Representation, directsum

    from symm_learning.nn.linear import eLinear

    G = DihedralGroup(6)

    in_rep = directsum([G.regular_representation] * 10)
    out_rep = directsum([G.regular_representation] * 11)

    layer = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=True)
    eq_layer = eLinear(in_rep, out_rep, bias=True)

    in_type = FieldType(no_base_space(G), [in_rep])
    out_type = FieldType(no_base_space(G), [out_rep])
    escnn_layer = escnn.nn.Linear(in_type, out_type, bias=True)

    batch_size = 512
    eq_layer.cuda()
    device = eq_layer.weight.device
    print(f"Device: {device}")
    for _ in range(30):
        x = torch.randn(batch_size, in_rep.size).to(device)
        for g in G.elements:
            g_x = torch.einsum("ij,bj->bi", torch.tensor(in_rep(g), dtype=torch.float).to(device), x)

            y = eq_layer(x)
            g_y = eq_layer(g_x)
            g_y_expected = torch.einsum("ij,bj->bi", torch.tensor(out_rep(g), dtype=torch.float).to(device), y)

            assert torch.allclose(g_y, g_y_expected, atol=1e-4), f"Equivariance test failed for group element {g}"

    print("Equivariance test passed!")

    def benchmark(module, x, iters=1000, warmup=50, device="cuda"):
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

    x = torch.randn(batch_size, in_rep.size).cuda()
    (time_std_fwd, time_std_fwddev), (time_std_bwd, time_std_bwddev) = benchmark(layer.cuda(), x)
    print(
        f"Standard linear layer — forward: {time_std_fwd:.3f} ms ± {time_std_fwddev:.3f} ms, "
        f"backward: {time_std_bwd:.3f} ms ± {time_std_bwddev:.3f} ms over {batch_size} samples"
    )
    (time_eq_fwd, time_eq_fwddev), (time_eq_bwd, time_eq_bwddev) = benchmark(eq_layer.cuda(), x)
    print(
        f"Equivariant linear layer — forward: {time_eq_fwd:.3f} ms ± {time_eq_fwddev:.3f} ms, "
        f"backward: {time_eq_bwd:.3f} ms ± {time_eq_bwddev:.3f} ms over {batch_size} samples"
    )
    (time_escnn_fwd, time_escnn_fwddev), (time_escnn_bwd, time_escnn_bwddev) = benchmark(escnn_layer.cuda(), x)
    print(
        f"e2cnn linear layer — forward: {time_escnn_fwd:.3f} ms ± {time_escnn_fwddev:.3f} ms, "
        f"backward: {time_escnn_bwd:.3f} ms ± {time_escnn_bwddev:.3f} ms over {batch_size} samples"
    )
    print("Done")
