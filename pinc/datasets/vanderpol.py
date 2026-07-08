import torch

from pinc.datasets.pinc_dataset import PINCSampler


def make_vanderpol_sampler(physics, T, device="cpu"):
    """
    Sampling ranges used in Sec. 4.1 of the paper for the Van der Pol
    oscillator: u in [-1, 1], x1, x2 in [-3, 3].

    device : generate the (large) training batches directly on this
             device -- avoids sampling on CPU and re-copying every
             single training iteration.
    """
    y_range = [(-3.0, 3.0), (-3.0, 3.0)]
    u_range = [(-1.0, 1.0)]
    return PINCSampler(physics, T=T, y_range=y_range, u_range=u_range, device=device)


def random_control_signal(n_steps, control_dim=1, u_range=(-1.0, 1.0), hold=1, seed=None):
    """
    Generates a piecewise-constant random control sequence, used to
    excite/evaluate the trained network (as in Fig. 9) and as a plant
    input for open-loop long-range simulation (Fig. 6).
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_holds = n_steps // hold + 1
    values = torch.empty(n_holds, control_dim).uniform_(*u_range)
    u = values.repeat_interleave(hold, dim=0)[:n_steps]
    return u