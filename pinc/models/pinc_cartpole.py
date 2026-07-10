import torch
import torch.nn as nn

from pinc.models.pinc import PINCModel


class CartPolePINCModel(PINCModel):
    """
    PINC net for the cart-pole, identical in spirit to the base
    `PINCModel` (Antonelo et al., Eq. 6-9) -- same three input groups
    (t, y0, u), same `.step` self-loop interface -- but with one
    addition: the *raw* physical state y0 = [x, x_dot, theta, theta_dot]
    is re-encoded as [x, x_dot, sin(theta), cos(theta), theta_dot]
    before being handed to the backbone MLP.

    Why: theta is an unbounded, unwrapped angle (see
    `pinc.physics.cartpole.CartPole`'s docstring) that can wrap around
    past +/-pi during a swing-up. Feeding raw theta into the network
    as a conditioning input means the network has to somehow learn
    that theta=3.14 and theta=-3.14 describe the same physical pole
    orientation -- an artificial discontinuity that raw theta forces
    onto an otherwise smooth function, and that the network has no way
    to know about a priori. sin/cos removes the discontinuity from the
    network's *input* features entirely (sin/cos of any real theta,
    however large, is a smooth, bounded, correctly-periodic pair).

    Crucially, this only changes what the backbone *sees*, not what it
    *predicts*: `forward` still outputs the raw 4-dim physical state
    y(t) = [x, x_dot, theta, theta_dot], so nothing downstream (the
    physics-residual loss's autograd w.r.t. t, the RK4 ground truth
    comparisons, NMPC's rollout, self-loop chaining) needs to know this
    encoding exists -- they all keep operating on plain, unwrapped
    theta exactly as they already do for every other physics/state
    quantity in this codebase.

    Backbone input dimensionality is therefore
        1 (t) + 5 (encoded state) + 1 (control) = 7
    instead of the base PINCModel's 1 + state_dim + control_dim = 6.

    One more wrinkle the sin/cos encoding introduces, and how it's
    handled: sin/cos of theta is a many-to-one map (theta and
    theta + 2*pi*k all encode identically). That's fine for the
    dynamics themselves -- the true physics genuinely only depends on
    sin(theta)/cos(theta), never on which "wrap count" theta is
    currently at (rotating the whole system by a multiple of 2*pi
    changes nothing physically) -- but it means the backbone's output
    for the theta channel *cannot* be an absolute angle: two y0's that
    differ only by theta -> theta + 2*pi would produce identical
    encoded features, yet (for the boundary loss, Eq. 10) need
    different target outputs at t=0 (y0 itself, wrap-count included).
    An absolute-output backbone has no way to satisfy both.

    The fix is to have the backbone predict a *delta* for the theta
    channel instead of an absolute value: `forward` adds the network's
    theta-channel output on top of y0's own (unencoded, wrap-count-
    intact) raw theta. This is well-posed precisely because the
    physical quantity that a smooth function of (sin(theta0),
    cos(theta0), theta_dot0, x0, x_dot0, u, t) *can* legitimately
    predict is "how far did the angle rotate over this window" -- a
    small, bounded quantity for any reasonably short T -- not "what is
    the absolute angle". At t=0 the network only has to learn to
    output a theta-delta of 0 (trivial, and a nice ResNet-style
    near-identity target besides); at t=T it predicts the actual
    rotation increment. x, x_dot, theta_dot have no such ambiguity
    (nothing about the encoding folds their range), so they're still
    predicted directly as absolute values, same as the base PINCModel.
    """

    def __init__(self, backbone: nn.Module, T: float):
        # state_dim is fixed at 4 (the physical state); control_dim at 1
        # (cart force) -- CartPole doesn't take these as constructor
        # args precisely because this class is only ever used with
        # pinc.physics.cartpole.CartPole, unlike the generic PINCModel.
        super().__init__(backbone=backbone, state_dim=4, control_dim=1, T=T)

    @staticmethod
    def encode_state(y0: torch.Tensor) -> torch.Tensor:
        """(N, 4) raw [x, x_dot, theta, theta_dot] -> (N, 5) network features."""
        x = y0[..., 0:1]
        x_dot = y0[..., 1:2]
        theta = y0[..., 2:3]
        theta_dot = y0[..., 3:4]
        return torch.cat([x, x_dot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

    def forward(self, t: torch.Tensor, y0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        features = self.encode_state(y0)
        network_input = torch.cat([t, features, u], dim=-1)
        raw_out = self.backbone(network_input)  # [x, x_dot, delta_theta, theta_dot]

        theta0 = y0[..., 2:3]
        theta_pred = theta0 + raw_out[..., 2:3]

        return torch.cat([raw_out[..., 0:2], theta_pred, raw_out[..., 3:4]], dim=-1)


def build_cartpole_pinc_model(meta: dict) -> "CartPolePINCModel":
    """
    Mirrors `pinc.utils.checkpoint.build_pinc_model`, but for the
    sin/cos-encoded cart-pole architecture (backbone in_dim=7, out_dim=4
    fixed, rather than derived from a generic state_dim/control_dim
    pair) -- kept as a separate function/file rather than folded into
    `pinc/utils/checkpoint.py` so that the base PINC checkpointing code
    stays architecture-agnostic.
    """
    from pinc.nn.mlp import MLP

    backbone = MLP(
        in_dim=1 + 5 + 1,
        out_dim=4,
        hidden=meta["hidden"],
        depth=meta["depth"],
    )
    return CartPolePINCModel(backbone=backbone, T=meta["T"])


def save_cartpole_checkpoint(path, model, meta, optimizer=None, extra=None):
    """Thin re-export of the generic `save_checkpoint` -- it's already
    architecture-agnostic (just calls `model.state_dict()`), so no
    cart-pole-specific logic is needed here; kept alongside
    `build_cartpole_pinc_model`/`load_cartpole_pinc_model` purely so
    callers only need to import from one place."""
    from pinc.utils.checkpoint import save_checkpoint
    save_checkpoint(path, model, meta, optimizer=optimizer, extra=extra)


def load_cartpole_pinc_model(path, map_location=None, device=None):
    """Cart-pole counterpart of `pinc.utils.checkpoint.load_pinc_model`,
    using `build_cartpole_pinc_model` to reconstruct the sin/cos-encoded
    architecture instead of the base MLP-only one."""
    from pinc.utils.checkpoint import load_checkpoint

    payload = load_checkpoint(path, map_location=map_location)
    model = build_cartpole_pinc_model(payload["meta"])
    model.load_state_dict(payload["model_state"])
    if device is not None:
        model = model.to(device)
    return model, payload