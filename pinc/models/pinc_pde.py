import torch
import torch.nn as nn

from pinc.core.base_model import BaseModel


class PINCSteadyStatePDE(BaseModel):
    """
    Steady-state PINC net for PDEs (Sec. 4.3, Eq. 26-30 of arXiv:2506.06188):

        y(x, u) = f_w(x, u),   x in [0,1], u in R

    where y = (P, V) is the (normalized) pressure/velocity pair, x is the
    normalized spatial position, and u is the normalized control
    (downstream pressure) parameterizing a whole family of steady-state
    solutions. No time input: the network directly represents equilibrium
    profiles for a wide range of controls.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 1) normalized position
        u : (N, 1) normalized control
        returns (P, V) : (N, 2)
        """
        return self.backbone(torch.cat([x, u], dim=-1))


class PINCTransientPDE(BaseModel):
    """
    Transient PINC net for PDEs (Sec. 4.4, Eq. 31-36):

        y(x, t, u0, u) = f_w(x, t, u0, u),   x,t in [0,1]

    where u0 is the control applied in the *previous* window (used as a
    compact stand-in for the initial condition, since the true initial
    state is itself a function of x and would otherwise blow up the input
    dimensionality -- Sec. 4.4.2) and u is the control held constant over
    the current window. Because the network's output depends only on
    (u0, u) and not on any fed-back prediction, errors do not accumulate
    across windows during long forward simulation (Algorithm 1).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                u0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x, t, u0, u : (N, 1) each
        returns (P, V) : (N, 2)
        """
        return self.backbone(torch.cat([x, t, u0, u], dim=-1))

    def at_position(self, x_bar: float, u0: torch.Tensor, u: torch.Tensor,
                     t: float = 1.0) -> torch.Tensor:
        """
        Convenience wrapper for the common "control interface" use case:
        evaluate the network at a single fixed spatial position x_bar
        (e.g. the PDG sensor location) and at the end of the current
        window (t=1, i.e. Ts=tref seconds after the window started), for
        a batch of (u0, u) pairs.

        u0, u : (N, 1)
        returns (P, V) : (N, 2)
        """
        n = u0.shape[0]
        x = torch.full((n, 1), x_bar, dtype=u0.dtype, device=u0.device)
        t_in = torch.full((n, 1), t, dtype=u0.dtype, device=u0.device)
        return self.forward(x, t_in, u0, u)
