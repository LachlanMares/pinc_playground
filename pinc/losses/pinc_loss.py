import torch
import torch.nn.functional as F

from pinc.utils.autodiff import time_derivative
from pinc.simulation.rk4 import rk4_control_interface


class PINCLoss:
    """
    Loss described in Sec. 3.3.3 of the paper, plus two additions aimed
    at what the network is actually evaluated on operationally (see
    below): MSE = MSE_y + lambda_phys * MSE_F
                        + lambda_endpoint  * MSE_endpoint
                        + lambda_multistep * MSE_multistep

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

    MSE_endpoint: neither term above ever directly supervises
                    model(T, y0, u) -- the one value `model.step` (and
                    therefore every downstream NMPC solve) actually
                    consumes. This term closes that gap: for the same
                    (y0, u) pairs used in the boundary batch, it
                    compares the network's t=T prediction against a
                    true RK4 integration of the real dynamics, held
                    under torch.no_grad() (it's a fixed regression
                    target, not something to differentiate through).

    MSE_multistep: MSE_endpoint alone only checks single-hop accuracy.
                    NMPC's own internal lookahead -- and the "long-range
                    self-loop" generalization metric -- both *chain*
                    model.step k times, feeding each prediction back in
                    as the next y0. This term explicitly trains that
                    chained behavior: it rolls the network forward k
                    steps (independently-sampled u at each hop, as NMPC
                    would apply) and penalizes divergence from the
                    equivalent k-step RK4 rollout at every hop, not just
                    the last one, so gradient signal doesn't vanish
                    early in training when the final-hop error would
                    otherwise be too large to be useful.
    """

    def __init__(self, physics, T, lambda_phys: float = 1.0,
                 lambda_endpoint: float = 1.0, lambda_multistep: float = 1.0,
                 endpoint_substeps: int = 20):
        self.physics = physics
        self.lambda_phys = lambda_phys
        self.lambda_endpoint = lambda_endpoint
        self.lambda_multistep = lambda_multistep

        # ground-truth T-second-ahead integrator used to build targets
        # for both the endpoint and multistep losses -- deliberately
        # more accurate (more substeps) than anything the network needs
        # to match at inference time, so it's a trustworthy target.
        self._true_step = rk4_control_interface(physics, T, substeps=endpoint_substeps)

    def __call__(self, model, boundary_batch, collocation_batch, multistep_batch=None):
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

        # --- #2: endpoint data-consistency loss ---
        # Reuses the same (y0, u) pairs already sampled for the boundary
        # batch -- no extra sampling call needed, since they're drawn
        # from the same distribution the endpoint target needs.
        pred_endpoint = model.step(y0_b, u_b)
        with torch.no_grad():
            target_endpoint = self._true_step(y0_b, u_b)
        endpoint_loss = F.mse_loss(pred_endpoint, target_endpoint)

        # --- #3: multi-step (chained rollout) consistency loss ---
        if multistep_batch is not None:
            y0_m, u_seq_m = multistep_batch  # y0_m: (n, state_dim), u_seq_m: (k, n, control_dim)
            y_pred = y0_m
            y_true = y0_m
            multistep_loss = 0.0
            k = u_seq_m.shape[0]
            for i in range(k):
                u_i = u_seq_m[i]
                y_pred = model.step(y_pred, u_i)
                with torch.no_grad():
                    y_true = self._true_step(y_true, u_i)
                multistep_loss = multistep_loss + F.mse_loss(y_pred, y_true)
            multistep_loss = multistep_loss / k
        else:
            multistep_loss = torch.zeros((), device=data_loss.device)

        total_loss = (data_loss
                      + self.lambda_phys * physics_loss
                      + self.lambda_endpoint * endpoint_loss
                      + self.lambda_multistep * multistep_loss)

        return {
            "total": total_loss,
            "data": data_loss,
            "physics": physics_loss,
            "endpoint": endpoint_loss,
            "multistep": multistep_loss,
        }