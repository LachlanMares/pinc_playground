import torch
from pinc.core.physics_model import PhysicsModel


class VanDerPol(PhysicsModel):
    """
    x1' = x2
    x2' = μ(1 - x1^2)x2 - x1 + u
    """

    def __init__(self, mu: float = 1.0):
        self.mu = mu

    @property
    def state_dim(self):
        return 2

    @property
    def control_dim(self):
        return 1

    def dynamics(self, x: torch.Tensor, u: torch.Tensor):
        x1 = x[..., 0]
        x2 = x[..., 1]
        u = u[..., 0]

        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1 + u

        return torch.stack([dx1, dx2], dim=-1)