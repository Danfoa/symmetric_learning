import copy
import logging

import escnn
import torch
from escnn.group import Representation
from escnn.gspaces import no_base_space
from escnn.nn import FieldType
from torch.nn.utils import parametrize

from symm_learning.linalg import invariant_orthogonal_projector
from symm_learning.representation_theory import GroupHomomorphismBasis
from symm_learning.utils import bytes_to_mb, check_equivariance, module_device_memory, module_memory

logger = logging.getLogger(__name__)


class InvariantConstraint(torch.nn.Module):
    r"""Project affine parameters onto the invariant subspace of a representation.

    Attributes:
        rep (Representation): Representation whose action defines invariance,
            dimension ``rep.size``.
        inv_projector (torch.Tensor): Orthogonal projector of shape
            ``(rep.size, rep.size)`` onto the fixed subspace
            :math:`\\mathrm{Fix}(\\rho)`.
    """

    def __init__(self, rep: Representation):
        """Precompute the invariant projector for the supplied representation."""
        super().__init__()
        self.rep = rep
        self.register_buffer("inv_projector", invariant_orthogonal_projector(rep))

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """Project b onto the invariant subspace using the precomputed projector."""
        return torch.mv(self.inv_projector, b)

    def right_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a parameter tensor whose projection equals ``tensor``."""
        return tensor

    def __deepcopy__(self, memo):  # noqa: D105
        memo = memo or {}
        memo[id(self.rep)] = self.rep
        memo[id(self.inv_projector)] = self.inv_projector
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for k, v in self.__dict__.items():
            setattr(clone, k, copy.deepcopy(v, memo))
        return clone


class CommutingConstraint(torch.nn.Module):
    r"""Equivariant weight parametrization via isotypic-basis projection.

    Args:
        in_rep (:class:`~escnn.group.Representation`): Input representation of
            size ``in_rep.size``.
        out_rep (:class:`~escnn.group.Representation`): Output representation of
            size ``out_rep.size``.

    Attributes:
        homo_basis (GroupHomomorphismBasis): Basis generator carrying the
            isotypic decomposition and block metadata for
            :math:`\\operatorname{Hom}_G(V_{\\text{in}}, V_{\\text{out}})`.
        in_rep / out_rep (Representation): Cached references to the isotypic
            versions of the supplied representations.
        _basis_vectors_normalized (torch.Tensor): Tensor of shape
            ``(dim, out_rep.size, in_rep.size)`` storing the orthonormal basis
            elements used to project weights into the equivariant subspace.

    Every pair of matching isotypic subspaces (one from ``in_rep``, one from
    ``out_rep``) inherits the same irreducible type :math:`\\bar\\rho_k`.
    Inside that block, any equivariant map factors as

    .. math::
        A_{i,j}^{(k)} = \\sum_{b \\in \\mathbb{B}_k} \\Theta^{(k)}_{b,\,i,j} \\,\\Psi_k(b),

    where :math:`\\Psi_k(b)` spans :math:`\\mathrm{End}_G(\\bar\\rho_k)` and
    :math:`\\Theta^{(k)}_{b,\,i,j} = \\langle A_{i,j}^{(k)}, \\Psi_k(b) \\rangle /
    \\|\\Psi_k(b)\\|^2`. The combinatorics over multiplicity indices is captured
    by Kronecker-factoring these basis endomorphisms with canonical ``E_{ij}``
    selectors. Implementation detail: we prebuild all
    :math:`E_{ij} \\otimes \\Psi_k(b)` blocks once, stack them into a tall
    matrix, and perform the whole projection as two batched GEMV operations—one
    to gather all Frobenius inner products, one to reconstruct the projected
    weight. This keeps the projection GPU-friendly while matching the orthogonal
    projection defined by Proposition H.13.
    """

    def __init__(self, in_rep: Representation, out_rep: Representation):
        super().__init__()
        self.homo_basis = GroupHomomorphismBasis(in_rep, out_rep)
        self.in_rep = self.homo_basis.in_rep
        self.out_rep = self.homo_basis.out_rep

        basis_elements = self.homo_basis.basis_elements
        basis_norms = torch.einsum("sab,sab->s", basis_elements, basis_elements).sqrt()
        self.register_buffer("_basis_vectors_normalized", basis_elements / basis_norms[:, None, None])

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """Project W onto Hom_G(out_rep, in_rep)."""
        assert W.shape[-2:] == (self.out_rep.size, self.in_rep.size), (
            f"Expected weight shape (..., {self.out_rep.size}, {self.in_rep.size}), got {W.shape}"
        )
        # Compute basis expansion coefficients
        coeff = torch.einsum("sab,...ab->...s", self._basis_vectors_normalized, W)
        # Expand the orthogonal projection in basis of Hom_G(out_rep, in_rep)
        W_proj = torch.einsum("sab,...s->...ab", self._basis_vectors_normalized, coeff)
        return W_proj

    def right_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a pre-image for the parametrization (identity for now)."""
        return tensor

    def __deepcopy__(self, memo):  # noqa: D105
        memo = memo or {}
        memo[id(self.in_rep)] = self.in_rep
        memo[id(self.out_rep)] = self.out_rep
        memo[id(self.homo_basis)] = self.homo_basis
        memo[id(self._basis_vectors_normalized)] = self._basis_vectors_normalized
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for k, v in self.__dict__.items():
            setattr(clone, k, copy.deepcopy(v, memo))
        return clone


if __name__ == "__main__":
    from escnn.group import CyclicGroup, Icosahedral, directsum

    from symm_learning.nn.linear import eLinear, impose_linear_equivariance

    # G = CyclicGroup(2)
    G = Icosahedral()
    m = 4
    bias = True
    in_rep = directsum([G.regular_representation] * m)
    out_rep = directsum([G.regular_representation] * m * 2)
    eq_layer_proj = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
    impose_linear_equivariance(lin=eq_layer_proj, in_rep=in_rep, out_rep=out_rep)

    # Check the orthogonal projection to Hom_G(rep, rep) works as expected
    in_type = escnn.nn.FieldType(escnn.gspaces.no_base_space(G), [G.regular_representation] * m)
    out_type = escnn.nn.FieldType(escnn.gspaces.no_base_space(G), [G.regular_representation] * m * 2)
    escnn_layer = escnn.nn.Linear(in_type, out_type, bias=bias)
    W, b = escnn_layer.expand_parameters()
    eq_layer_proj.weight = W
    W_projected = eq_layer_proj.weight
    assert torch.allclose(W, W_projected, atol=1e-5, rtol=1e-5), f"Max err: {(W - W_projected).abs().max()}"
    check_equivariance(eq_layer_proj, atol=1e-5, rtol=1e-5, in_rep=in_rep, out_rep=out_rep)
    print("Projection invariance test passed.")

    # For any random linear map check the projected map is indeed in Hom_G(rep, rep)
    W_random = torch.randn_like(W)
    eq_layer_proj.weight = W_random
    check_equivariance(eq_layer_proj, atol=1e-5, rtol=1e-5)

    # Check projection is idempotent
    W = eq_layer_proj.weight
    eq_layer_proj.weight = W
    W_proj2 = eq_layer_proj.weight
    assert torch.allclose(W, W_proj2, atol=1e-5, rtol=1e-5), f"Max err: {(W - W_proj2).abs().max()}"
    print("Projection idempotence test passed.")

    # ____________________________________________________________________________________
    standad_layer = torch.nn.Linear(in_features=in_rep.size, out_features=out_rep.size, bias=bias)
    eq_layer = eLinear(in_rep, out_rep, bias=bias)

    in_type = FieldType(no_base_space(G), [in_rep])
    out_type = FieldType(no_base_space(G), [out_rep])
    escnn_layer = escnn.nn.Linear(in_type, out_type, bias=bias)

    batch_size = 512
    device = torch.device("cuda")
    standad_layer = standad_layer.to(device)
    eq_layer = eq_layer.to(device)
    eq_layer_proj = eq_layer_proj.to(device)
    escnn_layer = escnn_layer.to(device)
    print(f"Device: {device}")

    import time

    def benchmark(module, x, iters=1000, warmup=50):
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
        ("Standard", standad_layer),
        ("eLinear", eq_layer),
        ("Linear (Proj)", eq_layer_proj),
        ("escnn", escnn_layer),
    ]

    results = []
    for name, module in modules_to_benchmark:
        train_mem, non_train_mem = module_memory(module)
        gpu_alloc, gpu_peak = module_device_memory(module)
        (fwd_mean, fwd_std), (bwd_mean, bwd_std) = benchmark(module, x)
        results.append(
            {
                "name": name,
                "fwd_mean": fwd_mean,
                "fwd_std": fwd_std,
                "bwd_mean": bwd_mean,
                "bwd_std": bwd_std,
                "total_time": fwd_mean + bwd_mean,
                "train_mem": train_mem,
                "non_train_mem": non_train_mem,
                "gpu_mem": gpu_alloc,
                "gpu_peak": gpu_peak,
            }
        )

    header = (
        f"{'Layer':<12} {'Forward (ms)':>18} {'Backward (ms)':>18} {'Total (ms)':>15} "
        f"{'Trainable MB':>15} {'Non-train MB':>15} {'Total MB':>12} {'GPU Alloc MB':>15} {'GPU Peak MB':>15}"
    )
    separator = "-" * len(header)
    print(f"\nBenchmark results per {batch_size}-sample batch")
    print(separator)
    print(header)
    print(separator)
    for res in results:
        fwd_str = f"{res['fwd_mean']:.3f} ± {res['fwd_std']:.3f}"
        bwd_str = f"{res['bwd_mean']:.3f} ± {res['bwd_std']:.3f}"
        total_mb = res["train_mem"] + res["non_train_mem"]
        gpu_alloc_mb = bytes_to_mb(res["gpu_mem"])
        gpu_peak_mb = bytes_to_mb(res["gpu_peak"])
        print(
            f"{res['name']:<12} {fwd_str:>18} {bwd_str:>18} "
            f"{res['total_time']:>15.3f} {bytes_to_mb(res['train_mem']):>15.3f} "
            f"{bytes_to_mb(res['non_train_mem']):>15.3f} {bytes_to_mb(total_mb):>12.3f} "
            f"{gpu_alloc_mb:>15.3f} {gpu_peak_mb:>15.3f}"
        )
    print(separator)
