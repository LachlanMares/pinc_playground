import torch
import torch.nn.functional as F

from pinc.utils.autodiff import time_derivative


class PINCLoss:
    """
    Loss described in Sec. 3.3.3 of the paper:

        MSE = MSE_y + lambda * MSE_F

    MSE_y (Eq. 10): standard regression loss at the boundary points t=0,
                    where the network must reproduce the given initial
                    state y(0).

    MSE_F (Eq. 11): physics-informed residual loss at randomly sampled
                    collocation points (t, y0, u) with t in (0, T]. The
                    network's output y(t) is differentiated w.r.t. t via
                    autograd, and the ODE residual

                        F(y) = dy/dt - f(y, u)

                    is penalized, where f(y, u) is the known ODE
                    right-hand side (physics.dynamics).
    """

    def __init__(self, physics, lambda_phys: float = 1.0):
        self.physics = physics
        self.lambda_phys = lambda_phys

    def __call__(self, model, boundary_batch, collocation_batch):
        t_b, y0_b, u_b, target_b = boundary_batch
        pred_b = model(t_b, y0_b, u_b)
        data_loss = F.mse_loss(pred_b, target_b)

        t_c, y0_c, u_c = collocation_batch
        t_c = t_c.clone().requires_grad_(True)

        y_c = model(t_c, y0_c, u_c)
        dydt = time_derivative(y_c, t_c)

        f = self.physics.dynamics(y_c, u_c)
        residual = dydt - f
        physics_loss = torch.mean(residual ** 2)

        total_loss = data_loss + self.lambda_phys * physics_loss

        return {
            "total": total_loss,
            "data": data_loss,
            "physics": physics_loss,
        }
