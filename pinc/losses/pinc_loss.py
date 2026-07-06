import torch
import torch.nn.functional as F


class PINCLoss:
    def __init__(self, physics, dt=0.05, lambda_phys=1e-2):
        self.physics = physics
        self.dt = dt
        self.lambda_phys = lambda_phys

    def __call__(self, model, batch):
        x, u, y = batch

        # ensure batch shape safety
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            y = y.unsqueeze(0)

        t = torch.zeros((x.shape[0], 1), device=x.device)

        pred_next = model(x, t, u)

        # physics consistency (Euler step constraint)
        f = self.physics.dynamics(x, u)
        physics_next = x + self.dt * f

        data_loss = F.mse_loss(pred_next, y)
        phys_loss = F.mse_loss(pred_next, physics_next)

        return data_loss + self.lambda_phys * phys_loss