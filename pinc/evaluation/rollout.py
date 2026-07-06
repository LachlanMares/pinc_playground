import torch


def rollout(model, physics, x0, steps=200, dt=0.05):
    traj = [x0]
    x = x0

    for i in range(steps):
        t = torch.zeros((1, 1))
        u = torch.zeros((1, 1))

        x_in = x.unsqueeze(0)
        x = model(x_in, t, u)[0]

        traj.append(x)

    return torch.stack(traj)