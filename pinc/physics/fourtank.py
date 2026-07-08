import torch

from pinc.core.physics_model import PhysicsModel


class FourTank(PhysicsModel):
    """
    The quadruple-tank process (Johansson, 2000), Eq. (17)-(18) of the
    paper:

        h1' = (gamma1*k1*u1 + w3 - w1) / A1
        h2' = (gamma2*k2*u2 + w4 - w2) / A2
        h3' = ((1 - gamma2)*k2*u2 - w3) / A3
        h4' = ((1 - gamma1)*k1*u1 - w4) / A4

        w_i = a_i * sqrt(2 * g * h_i)     (Bernoulli orifice equation)

    Default parameters correspond to the classic non-minimum-phase
    operating point of Johansson (2000) (gamma1 + gamma2 < 1), the
    setting referenced in Sec. 4.2.1 of the paper. Adjust as needed to
    match a specific reproduction target.

    A1, A2, A3, A4 = 28.0, 32.0, 28.0, 32.0 — cross-sectional area of each tank, in cm². This is the denominator in
    every dynamics equation — it converts a flow (volume/time) into a rate of change of level (height/time): a bigger
    tank means the same inflow/outflow moves the water level less. Note tanks 1&3 share one area (28) and 2&4 share
    another (32) — a deliberate symmetry from the original Johansson benchmark, not a coincidence.

    a1, a2, a3, a4 = 0.071, 0.057, 0.071, 0.057 — cross-sectional area of the outlet hole at the bottom of each tank,
    in cm². This is the a_i in the Bernoulli orifice equation w_i = a_i·sqrt(2·g·h_i) — bigger hole means faster
    drainage for the same water level. Same pairing pattern as A: tanks 1&3 share one hole size, 2&4 share another.

    k1, k2 = 3.33, 3.35 — pump gain: how much flow (cm³/s) each pump produces per volt of input. This is why k1*u1
    appears in the inflow terms — it converts the control signal (a voltage, u1 ∈ [0,5]) into an actual flow rate
    before it gets split and added to the mass balance.

    gamma1, gamma2 = 0.43, 0.34 — the split fraction for each pump's flow, and the single most important parameter for
    how this system behaves. Pump 1's total flow k1*u1 doesn't go entirely to tank 1 — a fraction gamma1 goes to tank 1,
    and the remaining (1-gamma1) is diverted to tank 4 instead (same idea for pump 2 → tanks 2 and 3, via gamma2).
    This is the physical mechanism behind the cross-coupling we discussed last message: turning up u1 feeds tank 1
    directly and indirectly refills tank 4, which then drains back into tank 1 later. Whether this system is "easy"
    (minimum-phase) or "hard" (non-minimum-phase) to control hinges entirely on whether gamma1 + gamma2 is greater or
    less than 1:

    gamma1 + gamma2 > 1: most of each pump's flow goes to its "own" tank — well-behaved, minimum-phase.
    gamma1 + gamma2 < 1 (your case: 0.43 + 0.34 = 0.77): most of each pump's flow is actually diverted to the other
    tank's pair — this is the deliberately harder, non-minimum-phase operating point from the paper, where increasing
    u1 can transiently make h1 dip before it rises, because the immediate direct contribution to tank 1 is smaller
    than the delayed indirect contribution arriving via tank 4's drainage.

    g = 981.0 — gravitational acceleration, but in cm/s² (not the more familiar 9.81 m/s²) — consistent with tank
    levels and areas being specified in centimeters throughout this model. It only ever appears inside the sqrt(2·g·h)
    orifice term, setting how fast water accelerates out through a hole at a given depth.

    """

    def __init__(self,
                 A1=28.0, A2=32.0, A3=28.0, A4=32.0,
                 a1=0.071, a2=0.057, a3=0.071, a4=0.057,
                 k1=3.33, k2=3.35,
                 gamma1=0.43, gamma2=0.34,
                 g=981.0):
        self.A = torch.tensor([A1, A2, A3, A4])
        self.a = torch.tensor([a1, a2, a3, a4])
        self.k1 = k1
        self.k2 = k2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.g = g

    @property
    def state_dim(self):
        return 4

    @property
    def control_dim(self):
        return 2

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        h1, h2, h3, h4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        u1, u2 = u[..., 0], u[..., 1]

        # clamp to avoid negative levels producing NaNs in sqrt during
        # training/collocation sampling or transient NMPC overshoot
        h1c = torch.clamp(h1, min=0.0)
        h2c = torch.clamp(h2, min=0.0)
        h3c = torch.clamp(h3, min=0.0)
        h4c = torch.clamp(h4, min=0.0)

        w1 = self.a[0] * torch.sqrt(2 * self.g * h1c + 1e-9)
        w2 = self.a[1] * torch.sqrt(2 * self.g * h2c + 1e-9)
        w3 = self.a[2] * torch.sqrt(2 * self.g * h3c + 1e-9)
        w4 = self.a[3] * torch.sqrt(2 * self.g * h4c + 1e-9)

        dh1 = (self.gamma1 * self.k1 * u1 + w3 - w1) / self.A[0]
        dh2 = (self.gamma2 * self.k2 * u2 + w4 - w2) / self.A[1]
        dh3 = ((1 - self.gamma2) * self.k2 * u2 - w3) / self.A[2]
        dh4 = ((1 - self.gamma1) * self.k1 * u1 - w4) / self.A[3]

        return torch.stack([dh1, dh2, dh3, dh4], dim=-1)
