import torch


def rollout(model, x0, t0, u_fn, steps, dt):
    traj = [x0]
    x = x0

    t = t0

    for i in range(steps):
        u = u_fn(i)

        x = model(x, t, u)
        t = t + dt

        traj.append(x)

    return torch.stack(traj)