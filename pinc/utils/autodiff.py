import torch


def time_derivative(y, t):
    """
    dy/dt via autograd
    """
    return torch.autograd.grad(
        outputs=y,
        inputs=t,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]