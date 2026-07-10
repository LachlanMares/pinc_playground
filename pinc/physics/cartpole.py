import torch
from pinc.core.physics_model import PhysicsModel


class CartPole(PhysicsModel):
    """
    Classic cart-and-pole (inverted pendulum on a cart), full nonlinear
    swing-up dynamics -- not the small-angle-only balancing model.

    State y = [x, x_dot, theta, theta_dot]:
        x         : cart position (m)
        x_dot     : cart velocity (m/s)
        theta     : pole angle (rad), measured from the *upright*
                    vertical -- theta = 0 is the unstable upright
                    equilibrium, theta = +/-pi is the stable
                    hanging-down equilibrium. theta is NOT wrapped
                    anywhere in this codebase: it's a plain,
                    unbounded real number, so a swing-up trajectory
                    that starts at pi and ends at 0 (or 2*pi, -2*pi,
                    ...) is just an ordinary continuous trajectory,
                    with no modulo/discontinuity for autograd (used
                    by the physics-residual loss) to trip over.
    Control u = [F]: horizontal force applied to the cart (N).

    Equations of motion (Lagrangian mechanics, pole modeled as a
    uniform rigid rod of mass `m` and length `2*l` pivoting on the
    cart -- this is the same derivation used in e.g. Tedrake's
    "Underactuated Robotics" notes, adapted to the upright-referenced
    theta convention above):

        (M + m) * x_ddot     + m*l*cos(theta) * theta_ddot - m*l*sin(theta)*theta_dot^2 = F
        m*l*cos(theta)*x_ddot + (I + m*l^2)   * theta_ddot - m*g*l*sin(theta)            = 0

    Solved in closed form for [x_ddot, theta_ddot] via Cramer's rule
    (2x2 linear system, `dynamics()` below).

    Defaults (M=1.0 kg, m=0.1 kg, L=1.0 m, g=9.8) are a fairly
    "light pole, heavy cart" setup, chosen so a moderate force
    (single-digit to ~15 N) is enough to swing the pole up in a few
    seconds -- adjust if you want a harder-to-swing (heavier pole /
    longer rod) or more sluggish-cart variant.
    """

    def __init__(self, M: float = 1.0, m: float = 0.1, L: float = 1.0, g: float = 9.8):
        self.M = M
        self.m = m
        self.L = L
        self.l = L / 2.0  # distance from pivot to the rod's center of mass
        self.I = (1.0 / 12.0) * m * L ** 2  # rod moment of inertia about its own CoM
        self.g = g

    @property
    def state_dim(self):
        return 4

    @property
    def control_dim(self):
        return 1

    def dynamics(self, x: torch.Tensor, u: torch.Tensor):
        theta = x[..., 2]
        theta_dot = x[..., 3]
        F = u[..., 0]

        s = torch.sin(theta)
        c = torch.cos(theta)

        M, m, l, I, g = self.M, self.m, self.l, self.I, self.g

        a11 = M + m
        a12 = m * l * c
        # a21 == a12 (symmetric mass matrix)
        a22 = I + m * l ** 2

        b1 = F + m * l * s * theta_dot ** 2
        b2 = m * g * l * s

        det = a11 * a22 - a12 * a12

        x_ddot = (a22 * b1 - a12 * b2) / det
        theta_ddot = (a11 * b2 - a12 * b1) / det

        x_dot = x[..., 1]
        return torch.stack([x_dot, x_ddot, theta_dot, theta_ddot], dim=-1)

    def total_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mechanical energy relative to the upright-at-rest equilibrium
        (theta=0, theta_dot=0), i.e. E=0 there and E<0 everywhere else
        reachable without external work. Not used by the PINC/NMPC
        training or control machinery itself (the swing-up reference
        in `pinc.datasets.cartpole.generate_swingup_reference` is built
        by direct trajectory optimization rather than energy shaping)
        -- exposed here as a general-purpose physics-level diagnostic,
        e.g. for sanity-checking a trajectory's energy profile or
        plotting it alongside a swing-up demo.
        """
        theta = x[..., 2]
        theta_dot = x[..., 3]
        KE = 0.5 * (self.I + self.m * self.l ** 2) * theta_dot ** 2
        PE = self.m * self.g * self.l * torch.cos(theta)
        PE_upright = self.m * self.g * self.l
        return (KE + PE) - PE_upright