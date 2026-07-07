import torch


class PINCSampler:
    """
    Generates the two kinds of training points used by PINC (Sec. 3.3.3):

    1) "Data" / boundary points, at t = 0, for the loss term MSE_y (Eq. 10):
           v_j = (0, y0_j, u_j)  ->  target = y0_j
       The network must learn to reproduce the initial state at t = 0.
       Crucially, no ODE simulation is required to build this dataset:
       y0 and u are simply drawn at random from the domain of interest.

    2) Collocation points, for the physics loss MSE_F (Eq. 11):
           v_k = (t_k, y0_k, u_k),  t_k ~ U(0, T], y0_k ~ U(y_range), u_k ~ U(u_range)
       These enforce the ODE residual F(y) = dy/dt - f(y, u) = 0 throughout
       the inner interval, via automatic differentiation (no simulator
       needed either).

    y_range / u_range: sequences of (low, high) pairs, one per state /
    control dimension.
    """

    def __init__(self, physics, T, y_range, u_range):
        self.physics = physics
        self.T = T
        self.y_range = y_range
        self.u_range = u_range

    def _sample_uniform(self, ranges, n):
        cols = []
        for (lo, hi) in ranges:
            cols.append(torch.empty(n, 1).uniform_(lo, hi))
        return torch.cat(cols, dim=-1)

    def sample_boundary(self, n):
        y0 = self._sample_uniform(self.y_range, n)
        u = self._sample_uniform(self.u_range, n)
        t = torch.zeros(n, 1)
        target = y0.clone()
        return t, y0, u, target

    def sample_collocation(self, n):
        y0 = self._sample_uniform(self.y_range, n)
        u = self._sample_uniform(self.u_range, n)
        t = torch.empty(n, 1).uniform_(1e-4, self.T)  # t in (0, T]
        return t, y0, u
