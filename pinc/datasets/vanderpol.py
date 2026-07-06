import torch


def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class VanDerPolDataset:
    def __init__(self, physics, T=50, dt=0.05, trajectories=200):
        self.physics = physics
        self.T = T
        self.dt = dt
        self.trajectories = trajectories

    def generate(self):
        X, U, Y = [], [], []

        for _ in range(self.trajectories):
            x = torch.randn(2) * 2.0

            for _ in range(self.T):
                u = torch.randn(1) * 0.5

                x_next = rk4(self.physics.dynamics, x, u, self.dt)

                X.append(x)
                U.append(u)
                Y.append(x_next)

                x = x_next

        return (
            torch.stack(X).float(),
            torch.stack(U).float(),
            torch.stack(Y).float(),
        )