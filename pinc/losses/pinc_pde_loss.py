import torch
import torch.nn.functional as F

from pinc.utils.autodiff import derivative


class SteadyStatePDELoss:
    """
    Sec. 4.3, Eq. 29-30: MSE = lambda_F * MSE_F + lambda_B * MSE_B

    MSE_F : residuals of the (time-derivative-free) mass + momentum
            equations at LHS collocation points (x, u).
    MSE_B : residuals of the upstream IPR and downstream pressure
            boundary conditions.

    No data/IC term -- the steady-state regime is, by definition,
    independent of initial conditions (Sec. 4.3).
    """

    def __init__(self, physics, lambda_phys=1.0, lambda_bc=1.0):
        self.physics = physics
        self.lambda_phys = lambda_phys
        self.lambda_bc = lambda_bc

    def __call__(self, model, collocation_batch, boundary_batch):
        x_c, u_c = collocation_batch
        y_c = model(x_c, u_c)
        P_c, V_c = y_c[:, 0:1], y_c[:, 1:2]

        dPdx = derivative(P_c, x_c)
        dVdx = derivative(V_c, x_c)

        mass_res = self.physics.residual_mass(dVdx)
        mom_res = self.physics.residual_momentum_steady(V_c, dPdx)
        physics_loss = torch.mean(mass_res**2) + torch.mean(mom_res**2)

        x0, u0, x1, u1 = boundary_batch
        y0 = model(x0, u0)
        y1 = model(x1, u1)
        P0, V0 = y0[:, 0:1], y0[:, 1:2]
        P1 = y1[:, 0:1]

        bc_up = self.physics.residual_bc_upstream(V0, P0)
        bc_down = self.physics.residual_bc_downstream(P1, u1)
        bc_loss = torch.mean(bc_up**2) + torch.mean(bc_down**2)

        total = self.lambda_phys * physics_loss + self.lambda_bc * bc_loss

        return {
            "total": total,
            "physics": physics_loss,
            "bc": bc_loss,
            "mass": torch.mean(mass_res**2),
            "momentum": torch.mean(mom_res**2),
        }


class TransientPDELoss:
    """
    Sec. 4.4, Eq. 32-35: MSE = lambda_F*MSE_F + lambda_B*MSE_B + lambda_I*MSE_I

    MSE_F : residuals of the full (time-dependent) mass + momentum
            equations at 4-D LHS collocation points (x, t, u0, u).
    MSE_B : upstream/downstream BC residuals at 3-D LHS points
            (t, u0, u), x fixed at 0 or 1.
    MSE_I : the network's t=0 output must match the *frozen*, already
            trained steady-state net evaluated at the same (x, u0) --
            Eq. 34, the mechanism that lets the transient net inherit
            realistic initial conditions without ever seeing a
            spatially-resolved initial state as input (Sec. 4.4.2).
    """

    def __init__(self, physics, steady_state_model, lambda_phys=1.0,
                 lambda_bc=1.0, lambda_ic=1.0):
        self.physics = physics
        self.steady_state_model = steady_state_model
        for p in self.steady_state_model.parameters():
            p.requires_grad_(False)
        self.lambda_phys = lambda_phys
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

    def __call__(self, model, collocation_batch, boundary_batch, ic_batch):
        x_c, t_c, u0_c, u_c = collocation_batch
        y_c = model(x_c, t_c, u0_c, u_c)
        P_c, V_c = y_c[:, 0:1], y_c[:, 1:2]

        dPdx = derivative(P_c, x_c)
        dVdx = derivative(V_c, x_c)
        dVdt = derivative(V_c, t_c)

        mass_res = self.physics.residual_mass(dVdx)
        mom_res = self.physics.residual_momentum_transient(V_c, dVdt, dPdx)
        physics_loss = torch.mean(mass_res**2) + torch.mean(mom_res**2)

        x0, t0, u0_0, u_0, x1, t1, u0_1, u_1 = boundary_batch
        y0 = model(x0, t0, u0_0, u_0)
        y1 = model(x1, t1, u0_1, u_1)
        P0, V0 = y0[:, 0:1], y0[:, 1:2]
        P1 = y1[:, 0:1]

        bc_up = self.physics.residual_bc_upstream(V0, P0)
        bc_down = self.physics.residual_bc_downstream(P1, u_1)
        bc_loss = torch.mean(bc_up**2) + torch.mean(bc_down**2)

        x_i, u0_i, u_i = ic_batch
        n = x_i.shape[0]
        t_i = torch.zeros(n, 1, device=x_i.device)
        y_pred_i = model(x_i, t_i, u0_i, u_i)
        with torch.no_grad():
            y_target_i = self.steady_state_model(x_i, u0_i)
        ic_loss = F.mse_loss(y_pred_i, y_target_i)

        total = (self.lambda_phys * physics_loss
                 + self.lambda_bc * bc_loss
                 + self.lambda_ic * ic_loss)

        return {
            "total": total,
            "physics": physics_loss,
            "bc": bc_loss,
            "ic": ic_loss,
            "mass": torch.mean(mass_res**2),
            "momentum": torch.mean(mom_res**2),
        }
