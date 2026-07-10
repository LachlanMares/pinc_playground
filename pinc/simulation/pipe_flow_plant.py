"""
Ground-truth "plant" for the incompressible pipe flow PDE, exploiting the
exact scalar-ODE reduction described in
`pinc.physics.pde_incompressible_flow` (mass conservation forces V to be
spatially uniform, so the whole PDE collapses onto a 1-state ODE for V(t)
driven by the control u(t), with the full pressure profile recovered
algebraically afterwards).

Because it is exposed as a `PhysicsModel` (dynamics(x, u) -> dx/dt), it
plugs directly into the *existing* `pinc.simulation.rk4` machinery
(`simulate`, `rk4_control_interface`) used elsewhere in this repo for the
ODE-PINC ground truth -- no new numerical integrator needed.
"""
import torch

from pinc.core.physics_model import PhysicsModel
from pinc.physics.pde_incompressible_flow import IncompressibleFlowPhysics


class IncompressiblePipePlant(PhysicsModel):
    """
    state = [V_tilde]  (normalized velocity, spatially uniform)
    control = [u_tilde] (normalized downstream pressure)

    dynamics() returns dV_tilde/dt_tilde, i.e. already normalized so that
    `pinc.simulation.rk4.simulate(..., dt=1.0, substeps=...)` integrates
    exactly one PINC "window" (tref seconds of real time).
    """

    def __init__(self, physics: IncompressibleFlowPhysics):
        self.physics = physics

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def control_dim(self) -> int:
        return 1

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        V_tilde = x[..., 0]
        u_tilde = u[..., 0]
        dVdt = self.physics.plant_dVdt(V_tilde, u_tilde)
        return dVdt.unsqueeze(-1)


def pressure_trajectory(physics: IncompressibleFlowPhysics, x_tilde: float,
                         V_traj: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the true pressure P(x_tilde, t) at a fixed position
    x_tilde, given a velocity trajectory (n_steps+1,) from
    `pinc.simulation.rk4.simulate` and the control sequence (n_steps,)
    that produced it (control is held constant within each step, and
    V_traj[0] is the initial condition, i.e. one entry longer than u_seq).

    returns P_traj : (n_steps + 1,)
    """
    u_padded = torch.cat([u_seq[:1], u_seq], dim=0)  # pad so it aligns with V_traj[0] (uses first control as ~pre-window value)
    V_flat = V_traj[..., 0]
    return physics.pressure_profile(torch.tensor(x_tilde), V_flat, u_padded[..., 0])
