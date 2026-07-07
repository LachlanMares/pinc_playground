import torch
import torch.nn as nn

from pinc.core.base_model import BaseModel


class PINCModel(BaseModel):
    """
    Physics-Informed Neural Net for Control (PINC), following
    Antonelo et al., "Physics-Informed Neural Nets for Control of
    Dynamical Systems" (arXiv:2104.02556).

    The network is augmented with three groups of inputs (Eq. 6):

        y(t) = f_w(t, y(0), u),   t in [0, T]

    where:
        t    : continuous time scalar, always relative to the start
               of the current "inner interval" (i.e. t in [0, T])
        y(0) : initial state of the dynamical system for this interval
        u    : control input, held constant over the interval

    A single trained network can be *chained* in self-loop mode
    (Fig. 3a/4, Eq. 7-9) to simulate for an arbitrarily long horizon,
    by feeding the last predicted state back in as y(0) for the next
    interval and always evaluating at t = T.
    """

    def __init__(self, backbone: nn.Module, state_dim: int, control_dim: int, T: float):
        super().__init__()
        self.backbone = backbone
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.T = T

    def forward(self, t: torch.Tensor, y0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        t  : (N, 1)
        y0 : (N, state_dim)
        u  : (N, control_dim)
        returns y(t) : (N, state_dim)
        """
        network_input = torch.cat([t, y0, u], dim=-1)
        return self.backbone(network_input)

    def predict(self, t: torch.Tensor, y0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.forward(t, y0, u)

    def step(self, y_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Control interface f_hat_w from Eq. (8):

            y[k] = f_hat_w(y[k-1], u[k]) = f_w(T, y[k-1], u[k])

        Single forward pass that predicts the state T seconds ahead,
        i.e. exactly at the boundary of the inner continuous interval.
        This is what replaces a numerical ODE solver inside NMPC.
        """
        n = y_prev.shape[0]
        t_T = torch.full((n, 1), self.T, dtype=y_prev.dtype, device=y_prev.device)
        return self.forward(t_T, y_prev, u)
