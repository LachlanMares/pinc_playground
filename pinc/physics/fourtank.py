import torch

from pinc.core.physics_model import PhysicsModel


class FourTank(PhysicsModel):
    """
    The quadruple-tank process (Johansson, 2000), Eq. (17)-(18) of the
    paper:

        h1' = (gamma1*k1*u1 + w3 - w1) / A1
        h2' = (gamma2*k2*u2 + w4 - w2) / A2
        h3' = ((1 - gamma2)*k2*u2 - w3) / A3
        h4' = ((1 - gamma1)*k1*u1 - w4) / A4

        w_i = a_i * sqrt(2 * g * h_i)     (Bernoulli orifice equation)

    Default parameters correspond to the classic non-minimum-phase
    operating point of Johansson (2000) (gamma1 + gamma2 < 1), the
    setting referenced in Sec. 4.2.1 of the paper. Adjust as needed to
    match a specific reproduction target.
    """

    def __init__(self,
                 A1=28.0, A2=32.0, A3=28.0, A4=32.0,
                 a1=0.071, a2=0.057, a3=0.071, a4=0.057,
                 k1=3.33, k2=3.35,
                 gamma1=0.43, gamma2=0.34,
                 g=981.0):
        self.A = torch.tensor([A1, A2, A3, A4])
        self.a = torch.tensor([a1, a2, a3, a4])
        self.k1 = k1
        self.k2 = k2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.g = g

    @property
    def state_dim(self):
        return 4

    @property
    def control_dim(self):
        return 2

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        h1, h2, h3, h4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        u1, u2 = u[..., 0], u[..., 1]

        # clamp to avoid negative levels producing NaNs in sqrt during
        # training/collocation sampling or transient NMPC overshoot
        h1c = torch.clamp(h1, min=0.0)
        h2c = torch.clamp(h2, min=0.0)
        h3c = torch.clamp(h3, min=0.0)
        h4c = torch.clamp(h4, min=0.0)

        w1 = self.a[0] * torch.sqrt(2 * self.g * h1c + 1e-9)
        w2 = self.a[1] * torch.sqrt(2 * self.g * h2c + 1e-9)
        w3 = self.a[2] * torch.sqrt(2 * self.g * h3c + 1e-9)
        w4 = self.a[3] * torch.sqrt(2 * self.g * h4c + 1e-9)

        dh1 = (self.gamma1 * self.k1 * u1 + w3 - w1) / self.A[0]
        dh2 = (self.gamma2 * self.k2 * u2 + w4 - w2) / self.A[1]
        dh3 = ((1 - self.gamma2) * self.k2 * u2 - w3) / self.A[2]
        dh4 = ((1 - self.gamma1) * self.k1 * u1 - w4) / self.A[3]

        return torch.stack([dh1, dh2, dh3, dh4], dim=-1)
