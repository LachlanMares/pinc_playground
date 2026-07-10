"""
Model checkpointing for PINC nets: saves/restores not just weights, but
enough architecture metadata (state_dim, control_dim, T, hidden, depth)
to rebuild the exact same MLP + PINCModel without the caller having to
remember the original construction arguments, plus optional optimizer
state and training progress (stage/iteration/history/best_val) so that
`Trainer.fit(..., resume=True)` can pick up where it left off.

Also covers the PDE-PINC nets (`PINCSteadyStatePDE`, `PINCTransientPDE`
from `pinc.models.pinc_pde`) via the analogous `build_pinc_*_pde_model` /
`load_pinc_*_pde_model` functions below -- `save_checkpoint` /
`load_checkpoint` themselves are architecture-agnostic (they just persist
`model.state_dict()` + an arbitrary `meta` dict), so no changes were
needed there.
"""
import os
import torch

from pinc.nn.mlp import MLP
from pinc.models.pinc import PINCModel
from pinc.models.pinc_pde import PINCSteadyStatePDE, PINCTransientPDE


def build_pinc_model(meta: dict) -> PINCModel:
    """Reconstructs a PINCModel from the architecture metadata saved in
    a checkpoint (see `save_checkpoint`)."""
    backbone = MLP(
        in_dim=1 + meta["state_dim"] + meta["control_dim"],
        out_dim=meta["state_dim"],
        hidden=meta["hidden"],
        depth=meta["depth"],
    )
    return PINCModel(
        backbone=backbone,
        state_dim=meta["state_dim"],
        control_dim=meta["control_dim"],
        T=meta["T"],
    )


def build_pinc_steady_state_pde_model(meta: dict) -> PINCSteadyStatePDE:
    """Reconstructs a PINCSteadyStatePDE (inputs: x, u) from checkpoint
    metadata {"hidden", "depth"}."""
    backbone = MLP(in_dim=2, out_dim=2, hidden=meta["hidden"], depth=meta["depth"])
    return PINCSteadyStatePDE(backbone)


def build_pinc_transient_pde_model(meta: dict) -> PINCTransientPDE:
    """Reconstructs a PINCTransientPDE (inputs: x, t, u0, u) from
    checkpoint metadata {"hidden", "depth"}."""
    backbone = MLP(in_dim=4, out_dim=2, hidden=meta["hidden"], depth=meta["depth"])
    return PINCTransientPDE(backbone)


def save_checkpoint(path, model, meta, optimizer=None, extra=None):
    """
    path      : file path to write to (parent directories are created
                automatically)
    model     : the model to save (PINCModel, PINCSteadyStatePDE, or
                PINCTransientPDE -- anything with a plain state_dict())
    meta      : dict of architecture args needed by the matching
                `build_*` function, e.g. {"state_dim", "control_dim",
                "T", "hidden", "depth"} for `build_pinc_model`, or just
                {"hidden", "depth"} for the PDE builders
    optimizer : optional optimizer whose state should be checkpointed
                too (useful for resuming ADAM mid-training; L-BFGS
                state is intentionally not checkpointed since it is
                cheap to rebuild and its internal history buffer isn't
                meaningful across a resume boundary)
    extra     : optional dict of additional data to persist, e.g.
                {"stage": "adam", "iter_in_stage": 120,
                 "history": {...}, "best_val": 0.0123}
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "meta": meta,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }

    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX/NTFS same-filesystem rename


def load_checkpoint(path, map_location=None):
    """
    Loads a checkpoint file and returns the raw payload dict
    (model_state, meta, optimizer_state, extra). Use `load_pinc_model`
    (or one of the PDE variants below) for the common case of just
    wanting a ready-to-use model.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def load_pinc_model(path, map_location=None, device=None):
    """
    Convenience one-shot loader: rebuilds the PINCModel architecture
    from the checkpoint's metadata, loads the weights into it, and
    (optionally) moves it to `device`.

    returns (model, payload) so callers can also inspect
    payload["extra"] (history, best_val, training stage, etc.) if
    needed.
    """
    payload = load_checkpoint(path, map_location=map_location)
    model = build_pinc_model(payload["meta"])
    model.load_state_dict(payload["model_state"])

    if device is not None:
        model = model.to(device)

    return model, payload


def load_pinc_steady_state_pde_model(path, map_location=None, device=None):
    """PDE analogue of `load_pinc_model`, for a `PINCSteadyStatePDE`
    checkpoint (see `pinc.training.train_incompressible_pde`)."""
    payload = load_checkpoint(path, map_location=map_location)
    model = build_pinc_steady_state_pde_model(payload["meta"])
    model.load_state_dict(payload["model_state"])

    if device is not None:
        model = model.to(device)

    return model, payload


def load_pinc_transient_pde_model(path, map_location=None, device=None):
    """PDE analogue of `load_pinc_model`, for a `PINCTransientPDE`
    checkpoint (see `pinc.training.train_incompressible_pde`)."""
    payload = load_checkpoint(path, map_location=map_location)
    model = build_pinc_transient_pde_model(payload["meta"])
    model.load_state_dict(payload["model_state"])

    if device is not None:
        model = model.to(device)

    return model, payload