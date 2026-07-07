import torch


def time_derivative(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes dy/dt via automatic differentiation, one column of y at a time
    so that it also works for vector-valued (multi-output) y.

    y : (N, D) network output, must have been produced from a graph that
        depends on t (t must have requires_grad=True and been part of the
        forward pass).
    t : (N, 1)

    returns dy/dt : (N, D)
    """
    grads = []
    for d in range(y.shape[-1]):
        g = torch.autograd.grad(
            outputs=y[..., d],
            inputs=t,
            grad_outputs=torch.ones_like(y[..., d]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(g)

    return torch.cat(grads, dim=-1)
