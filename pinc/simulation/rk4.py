import torch


def rk4_step(dynamics, x, u, dt):
    """One classic 4th-order Runge-Kutta integration step for dx/dt = dynamics(x, u)."""
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def simulate(physics, x0, u_sequence, dt, substeps=10):
    """
    Simulates the true plant (via RK4) applying one (possibly different)
    control value per outer timestep, used as ground truth ("the plant")
    for comparison against PINC and for closed-loop NMPC simulations.

    x0         : (state_dim,) initial state
    u_sequence : (n_steps, control_dim) control input held constant
                 within each outer step (mirrors the PINC/MPC sampling
                 period T = Ts)
    dt         : outer step size (T = Ts)
    substeps   : number of RK4 sub-steps used internally per outer step,
                 for numerical accuracy independent of the network's T.

    returns traj : (n_steps + 1, state_dim)
    """
    x = x0.clone()
    traj = [x.clone()]
    h = dt / substeps

    for u in u_sequence:
        for _ in range(substeps):
            x = rk4_step(physics.dynamics, x, u, h)
        traj.append(x.clone())

    return torch.stack(traj)


def rk4_control_interface(physics, T, substeps=20):
    """
    Builds a control-interface function f_hat_w(y_prev, u) -> y_next with
    the exact same signature as PINCModel.step, but backed by numerical
    RK4 integration instead of a trained network. This is the "ODE/RK"
    baseline predictive model used for NMPC in Table 1 / Fig. 10 (right)
    of the paper, letting the same NMPC code drive either predictive
    model interchangeably.
    """
    h = T / substeps

    def step(y_prev, u):
        # y_prev, u : (N, dim) batched, matches PINCModel.step's signature
        y = y_prev
        for _ in range(substeps):
            y = rk4_step(physics.dynamics, y, u, h)
        return y

    return step
