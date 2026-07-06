import torch
import torch.nn as nn


class PINCModel(nn.Module):
    def __init__(self, backbone, physics, dt=0.05):
        super().__init__()
        self.backbone = backbone
        self.physics = physics
        self.dt = dt

    def forward(self, x, t, u):
        f_phys = self.physics.dynamics(x, u)
        correction = self.backbone(torch.cat([x, u], dim=-1))

        return x + self.dt * f_phys + correction

    def predict(self, x, t, u):
        return self.forward(x, t, u)