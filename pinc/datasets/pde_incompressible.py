"""
Sampling for the incompressible-flow PDE-PINC nets, following Sec. 4.3 /
4.4.1 of arXiv:2506.06188: collocation points for the PDE loss and
boundary/initial condition points are all drawn with Latin Hypercube
Sampling (LHS), no measured/simulated data required (just like the
ODE-PINC's `PINCSampler`, this is "physics-only" training).
"""
import torch
from scipy.stats import qmc


def _lhs(n, dim, bounds, seed=None):
    """
    n      : number of samples
    dim    : number of dimensions
    bounds : list of (low, high) pairs, one per dimension
    returns (n, dim) tensor
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit = sampler.random(n=n)  # (n, dim) in [0,1)
    lows = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
    highs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
    return lows + torch.from_numpy(unit).float() * (highs - lows)


class SteadyStatePDESampler:
    """
    Sec. 4.3: (x, u) in [0,1]^2 (bidimensional LHS) for the PDE-residual
    collocation points; unidimensional LHS over u alone for boundary
    points (x fixed at 0 or 1 depending on which BC is being enforced).
    """

    def __init__(self, u_range=(0.0, 1.0), device="cpu"):
        self.u_range = u_range
        self.device = device

    def sample_collocation(self, n, seed=None):
        pts = _lhs(n, 2, [(0.0, 1.0), self.u_range], seed=seed).to(self.device)
        x, u = pts[:, 0:1], pts[:, 1:2]
        x.requires_grad_(True)
        return x, u

    def sample_boundary(self, n, seed=None):
        """returns (x0, u0, x1, u1), one independent LHS draw of u per boundary."""
        u0 = _lhs(n, 1, [self.u_range], seed=seed).to(self.device)
        u1 = _lhs(n, 1, [self.u_range], seed=None if seed is None else seed + 1).to(self.device)
        x0 = torch.zeros(n, 1, device=self.device, requires_grad=True)
        x1 = torch.ones(n, 1, device=self.device, requires_grad=True)
        return x0, u0, x1, u1


class TransientPDESampler:
    """
    Sec. 4.4.1: (x, t, u0, u) in [0,1]^4 (4-D LHS) for PDE-residual
    collocation points; 3-D LHS over (t, u0, u) for boundary points (x
    fixed); 3-D LHS over (x, u0, u) for initial-condition points (t=0).
    """

    def __init__(self, u_range=(0.0, 1.0), device="cpu"):
        self.u_range = u_range
        self.device = device

    def sample_collocation(self, n, seed=None):
        pts = _lhs(n, 4, [(0.0, 1.0), (0.0, 1.0), self.u_range, self.u_range], seed=seed).to(self.device)
        x, t, u0, u = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], pts[:, 3:4]
        x.requires_grad_(True)
        t.requires_grad_(True)
        return x, t, u0, u

    def sample_boundary(self, n, seed=None):
        """returns (x0, t0, u0_0, u_0, x1, t1, u0_1, u_1)."""
        pts0 = _lhs(n, 3, [(0.0, 1.0), self.u_range, self.u_range], seed=seed).to(self.device)
        seed1 = None if seed is None else seed + 1
        pts1 = _lhs(n, 3, [(0.0, 1.0), self.u_range, self.u_range], seed=seed1).to(self.device)

        x0 = torch.zeros(n, 1, device=self.device)
        x1 = torch.ones(n, 1, device=self.device)
        t0, u0_0, u_0 = pts0[:, 0:1], pts0[:, 1:2], pts0[:, 2:3]
        t1, u0_1, u_1 = pts1[:, 0:1], pts1[:, 1:2], pts1[:, 2:3]
        return x0, t0, u0_0, u_0, x1, t1, u0_1, u_1

    def sample_initial_condition(self, n, seed=None):
        """returns (x, u0, u) at t=0, for the IC loss (Eq. 34)."""
        pts = _lhs(n, 3, [(0.0, 1.0), self.u_range, self.u_range], seed=seed).to(self.device)
        x, u0, u = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]
        return x, u0, u
