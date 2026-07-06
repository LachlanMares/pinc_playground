import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, depth=4, activation=nn.Tanh):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden), activation()]

        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation()]

        layers += [nn.Linear(hidden, out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)