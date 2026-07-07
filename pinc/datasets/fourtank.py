from pinc.datasets.pinc_dataset import PINCSampler


def make_fourtank_sampler(physics, T):
    """
    Sampling ranges roughly matching Sec. 4.2.2 of the paper: tank levels
    initial conditions in [2, 20] cm, control voltages u1, u2 in a
    plausible operating range for this benchmark (Johansson, 2000).
    """
    y_range = [(2.0, 20.0)] * 4
    u_range = [(0.0, 5.0)] * 2
    return PINCSampler(physics, T=T, y_range=y_range, u_range=u_range)
