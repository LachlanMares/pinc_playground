import torch


@torch.no_grad()
def pinc_rollout(model, y0, u_sequence):
    """
    Self-loop ("free-run") simulation with the trained PINC net (Fig. 3a,
    Fig. 4, Eq. 7-8):

        y[k] = f_hat_w(y[k-1], u[k]) = f_w(T, y[k-1], u[k])

    The network's own prediction at t=T is fed back as the initial state
    y(0) for the next inner interval -- no numerical integration of
    intermediate points is needed, only a single forward pass per step.

    y0         : (state_dim,) initial state
    u_sequence : (n_steps, control_dim)

    returns traj : (n_steps + 1, state_dim)
    """
    y = y0.unsqueeze(0)
    traj = [y0]

    for u in u_sequence:
        u_in = u.unsqueeze(0)
        y = model.step(y, u_in)
        traj.append(y[0])

    return torch.stack(traj)


@torch.no_grad()
def mse_gen(model, y0, u_sequence, y_true):
    """
    Generalization MSE (Eq. 13), computed only at the discrete inner-interval
    boundaries (the vertical lines in Fig. 4), comparing the PINC self-loop
    rollout against the true (RK4) trajectory under the same control input.
    """
    y_pred = pinc_rollout(model, y0, u_sequence)
    return torch.mean((y_pred - y_true) ** 2).item()
