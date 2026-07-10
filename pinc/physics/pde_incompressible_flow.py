"""
Physics for the incompressible single-phase pipe flow system of Miyatake
et al., "Physics-Informed Neural Networks for Control of Single-Phase Flow
Systems Governed by Partial Differential Equations" (arXiv:2506.06188),
Sec. 2.3 / 2.3.1.

Governing PDEs (normalized, Eq. 17):

    mass:      dV/dx = 0,                                   x in [0,1]
    momentum:  dV/dt + c_p * dP/dx + c_g + c_f*f*|V|*V = 0,  x in [0,1], t in [0,1]

boundary conditions:

    upstream (x=0, IPR):   V(0,t) - k*(Preservoir - Pref*P(0,t))/Vref = 0
    downstream (x=1):      P(1,t) - u(t) = 0

where u(t) is the (normalized) control signal -- the downstream pressure.
Tildes are dropped in code/comments below; every quantity handled here is
already normalized unless a function is explicitly named `*_phys`.

Key physical fact this module leans on (Sec. 2.3, Eq. 12-13): for
*incompressible* flow, mass conservation forces dV/dx = 0 identically, not
just at steady state. Combined with the momentum equation, this means
dP/dx(t) cannot depend on x either, so the true pressure profile is exactly
linear in x at every instant:

    P(x,t) = P(0,t) + x * (u(t) - P(0,t))

and the whole PDE system collapses onto a single scalar ODE for V(t),
driven by the control u(t) (see `plant_dVdt` / `PipePlant` in
`pinc/simulation/pipe_flow_plant.py`). We use this reduction to build an
*exact* ground-truth "plant" for validation and MPC, exactly analogous to
how `pinc/simulation/rk4.py` integrates the true Van der Pol / four-tank
ODEs for the original (ODE) PINC.
"""
from dataclasses import dataclass

import torch


@dataclass
class IncompressibleFlowParams:
    """Table 1 of the paper (incompressible water flow)."""
    D: float = 0.1                 # pipe diameter [m]
    mu: float = 0.001              # dynamic viscosity [Pa.s]
    k_ipr: float = 1e-5            # IPR proportional constant
    P_reservoir: float = 2e5       # reservoir pressure [Pa]
    L: float = 100.0               # pipe length == xref [m]
    theta: float = 0.0             # inclination [rad]
    rho: float = 1000.0            # fluid density [kg/m^3]
    g: float = 9.81                # gravitational acceleration [m/s^2]

    Pref: float = 1e5              # pressure reference [Pa]
    Vref: float = 1.0              # velocity reference [m/s]
    tref: float = 10.0             # time reference [s]


class IncompressibleFlowPhysics:
    """
    Normalized residuals (Eq. 17), boundary conditions, and the exact
    scalar-ODE reduction of the plant, for a horizontal or inclined
    incompressible pipe flow segment.
    """

    def __init__(self, params: IncompressibleFlowParams = None):
        self.p = params or IncompressibleFlowParams()

        p = self.p
        self.xref = p.L

        # steady/transient momentum-equation normalization constants,
        # Eq. (17)/(38)
        self.c_press_steady = p.Pref / (p.rho * p.Vref * self.xref)
        self.c_grav_steady = p.g * torch.sin(torch.tensor(p.theta)).item() / p.Vref
        self.c_fric_steady = 0.5 * (p.Vref / p.D)

        self.c_press_trans = p.tref * self.c_press_steady
        self.c_grav_trans = p.tref * self.c_grav_steady
        self.c_fric_trans = p.tref * self.c_fric_steady

        # IPR boundary condition, restated in normalized velocity form
        # (Eq. 16): V(0,t) = k*(Preservoir - P(0,t))/Vref
        self.k_ipr_norm = p.k_ipr * p.Pref / p.Vref
        self.preservoir_norm = p.P_reservoir / p.Pref

    # ------------------------------------------------------------------
    # Friction (Blasius correlation, Eq. 4-5), with the Reynolds-number
    # clamp mentioned in Sec. 5.1.1 to keep training numerically stable
    # for V close to (or crossing) zero.
    # ------------------------------------------------------------------
    def reynolds(self, V_tilde: torch.Tensor) -> torch.Tensor:
        p = self.p
        V_phys = torch.abs(V_tilde) * p.Vref
        Re = p.rho * V_phys * p.D / p.mu
        return torch.clamp(Re, min=1.0)

    def friction_factor(self, V_tilde: torch.Tensor) -> torch.Tensor:
        Re = self.reynolds(V_tilde)
        return 0.316 / Re**0.25

    # ------------------------------------------------------------------
    # PDE residuals -- F(y) := ... = 0 (Eq. 17)
    # ------------------------------------------------------------------
    def residual_mass(self, dVdx: torch.Tensor) -> torch.Tensor:
        """dV/dx = 0, holds identically for incompressible flow."""
        return dVdx

    def residual_momentum_steady(self, V_tilde, dPdx) -> torch.Tensor:
        f = self.friction_factor(V_tilde)
        return (self.c_press_steady * dPdx
                + self.c_grav_steady
                + self.c_fric_steady * f * torch.abs(V_tilde) * V_tilde)

    def residual_momentum_transient(self, V_tilde, dVdt, dPdx) -> torch.Tensor:
        f = self.friction_factor(V_tilde)
        return (dVdt
                + self.c_press_trans * dPdx
                + self.c_grav_trans
                + self.c_fric_trans * f * torch.abs(V_tilde) * V_tilde)

    # ------------------------------------------------------------------
    # Boundary conditions -- B(y) := ... = 0 (Eq. 17)
    # ------------------------------------------------------------------
    def residual_bc_upstream(self, V_at_0, P_at_0) -> torch.Tensor:
        """IPR: V(0,t) - k*(Preservoir - P(0,t)) = 0 (normalized)."""
        return V_at_0 - self.k_ipr_norm * (self.preservoir_norm - P_at_0)

    def residual_bc_downstream(self, P_at_1, u_tilde) -> torch.Tensor:
        """P(1,t) - u(t) = 0."""
        return P_at_1 - u_tilde

    def p0_from_V(self, V_tilde: torch.Tensor) -> torch.Tensor:
        """
        Inverts the upstream IPR relation for P(0,t) given V(t) (exact,
        algebraic -- used to close the scalar-ODE plant reduction, and as
        a consistency check on trained networks).
        """
        return self.preservoir_norm - V_tilde / self.k_ipr_norm

    def pressure_profile(self, x_tilde: torch.Tensor, V_tilde: torch.Tensor,
                          u_tilde: torch.Tensor) -> torch.Tensor:
        """
        Exact pressure profile P(x,t), linear in x (see module docstring),
        given the (spatially uniform) velocity V(t) and control u(t).
        Broadcasts over batch dimensions.
        """
        p0 = self.p0_from_V(V_tilde)
        return p0 + x_tilde * (u_tilde - p0)

    # ------------------------------------------------------------------
    # Exact scalar-ODE reduction of the plant (ground truth), driven by
    # the control signal u_tilde(t) -- see module docstring.
    # ------------------------------------------------------------------
    def plant_dVdt(self, V_tilde: torch.Tensor, u_tilde: torch.Tensor) -> torch.Tensor:
        """
        dV/dt (normalized, w.r.t. real time t -- NOT tref-normalized
        t_tilde) for the true plant, obtained by substituting the linear
        pressure profile's slope (u - P0)/L into the momentum equation
        and P0 = Preservoir - V/k from the IPR relation.

        Returned in *tref-normalized* time (dV_tilde/dt_tilde), i.e.
        already multiplied by tref, so it plugs directly into
        `pinc.simulation.rk4.rk4_step` with a normalized step size
        (e.g. dt_tilde = 1/substeps for one full window of duration
        tref seconds).
        """
        f = self.friction_factor(V_tilde)
        dPdx = u_tilde - self.p0_from_V(V_tilde)  # x already normalized, L=xref folded into c_press
        return -(self.c_press_trans * dPdx
                  + self.c_grav_trans
                  + self.c_fric_trans * f * torch.abs(V_tilde) * V_tilde)
