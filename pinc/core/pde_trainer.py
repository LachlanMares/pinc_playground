"""
Generic two-stage (ADAM then L-BFGS) trainer, following the same
Algorithm-1-style procedure as `pinc.core.trainer.Trainer`, but
parameterized purely over a `sample_batches()` / `loss_fn(model, *batches)`
pair rather than baking in the ODE-PINC's specific (boundary, collocation,
multistep) batch shape. This lets the same loop drive both
`SteadyStatePDELoss` (2 batches) and `TransientPDELoss` (3 batches)
without duplicating the ADAM/L-BFGS bookkeeping.

Also supports periodic checkpointing and resuming a previously
interrupted run (see `fit(..., checkpoint_path=..., resume=True)`),
mirroring `Trainer.fit`'s behavior on the ODE-PINC side.
"""
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from pinc.utils.checkpoint import save_checkpoint, load_checkpoint


class PDETrainer:
    def __init__(self, model, sample_batches_fn, loss_fn, lr=1e-3, device="cpu"):
        """
        sample_batches_fn : callable() -> tuple of batches, forwarded
                             as *batches to loss_fn(model, *batches)
        loss_fn            : callable(model, *batches) -> dict with a
                             "total" key (and any other scalars to log)
        """
        self.device = device
        self.model = model.to(device)
        self.sample_batches_fn = sample_batches_fn
        self.loss_fn = loss_fn
        self.adam = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _closure_factory(self, batches):
        def closure():
            self.lbfgs.zero_grad()
            losses = self.loss_fn(self.model, *batches)
            losses["total"].backward()
            closure.last_losses = losses
            return losses["total"]
        closure.last_losses = None
        return closure

    def _save(self, checkpoint_path, meta, stage, iter_in_stage,
              history, best_val, best_state):
        save_checkpoint(
            checkpoint_path, self.model, meta or {},
            optimizer=self.adam,
            extra={
                "stage": stage,
                "iter_in_stage": iter_in_stage,
                "history": history,
                "best_val": best_val,
                "best_model_state": best_state,
            },
        )

    def fit(self, k1_epochs=500, k2_iters=2000, log_every=-1,
            refresh_every=200, desc="",
            validate_fn=None, checkpoint_path=None, meta=None,
            save_every=100, resume=False):
        """
        validate_fn     : optional callable(model) -> float, used to
                           track the best network seen during training
                           (mirrors `Trainer.fit`'s validation tracking
                           for Algorithm 1's "save network w with best
                           performance" step). If omitted, no
                           validation-based best-model tracking is done
                           (checkpoints still save the latest model).
        checkpoint_path : if given, periodically saves a checkpoint
                           here (model + ADAM optimizer state + training
                           progress), overwriting the same file, so a
                           killed/interrupted run can be resumed.
        meta            : architecture dict (e.g. {"hidden", "depth"})
                           stored in the checkpoint so the model can
                           later be rebuilt via
                           `pinc.utils.checkpoint.load_pinc_steady_state_pde_model`
                           / `load_pinc_transient_pde_model` without
                           needing to remember the construction args.
        save_every      : how often (in iterations) to write the
                           checkpoint. A checkpoint is also always
                           written whenever validation improves, and
                           once more at the very end of training.
        resume          : if True and `checkpoint_path` exists, restores
                           model weights, ADAM state, training history,
                           and the ADAM/L-BFGS progress counter, and
                           continues training from there instead of
                           starting over.
        """
        history = defaultdict(list)
        print_log = log_every != -1

        best_state = None
        best_val = float("inf")
        start_stage = "adam"
        start_iter = 0

        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            payload = load_checkpoint(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(payload["model_state"])
            if payload.get("optimizer_state") is not None:
                self.adam.load_state_dict(payload["optimizer_state"])

            extra = payload.get("extra", {})
            loaded_history = extra.get("history", {})
            for k, v in loaded_history.items():
                history[k] = list(v)
            best_val = extra.get("best_val", best_val)
            best_state = extra.get("best_model_state", best_state)
            start_stage = extra.get("stage", "adam")
            start_iter = extra.get("iter_in_stage", 0)

            print(f"Resumed from '{checkpoint_path}': stage={start_stage}, "
                  f"iter={start_iter}, best_val={best_val:.3e}"
                  if best_val != float("inf") else
                  f"Resumed from '{checkpoint_path}': stage={start_stage}, iter={start_iter}")

        # ---- Stage 1: ADAM ----
        adam_start = start_iter if start_stage == "adam" else k1_epochs

        adam_bar = tqdm(range(adam_start, k1_epochs), initial=adam_start, total=k1_epochs,
                         desc=f"{desc} ADAM", unit="epoch")
        for epoch in adam_bar:
            batches = self.sample_batches_fn()
            self.adam.zero_grad()
            losses = self.loss_fn(self.model, *batches)
            losses["total"].backward()
            self.adam.step()

            for key, val in losses.items():
                history[key].append(val.item())

            val = validate_fn(self.model) if validate_fn is not None else None
            if val is not None:
                history["val"].append(val)

            improved = val is not None and val < best_val
            if improved:
                best_val = val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if checkpoint_path and (improved or (epoch + 1) % save_every == 0 or epoch == k1_epochs - 1):
                self._save(checkpoint_path, meta, "adam", epoch + 1, dict(history), best_val, best_state)

            postfix = {k: f"{v.item():.3e}" for k, v in losses.items()}
            if val is not None:
                postfix["val"] = f"{val:.3e}"
            adam_bar.set_postfix(postfix)
            if print_log and (epoch % log_every == 0 or epoch == k1_epochs - 1):
                adam_bar.write(f"[ADAM {epoch:05d}] " +
                                " ".join(f"{k}={v.item():.3e}" for k, v in losses.items()) +
                                (f" val={val:.3e}" if val is not None else ""))

        # ---- Stage 2: L-BFGS ----
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(), lr=1.0, max_iter=1,
            history_size=10, line_search_fn="strong_wolfe",
        )

        batches = self.sample_batches_fn()
        closure = self._closure_factory(batches)

        lbfgs_start = start_iter if start_stage == "lbfgs" else 0

        lbfgs_bar = tqdm(range(lbfgs_start, k2_iters), initial=lbfgs_start, total=k2_iters,
                          desc=f"{desc} LBFGS", unit="iter")
        for it in lbfgs_bar:
            if it > lbfgs_start and it % refresh_every == 0:
                batches = self.sample_batches_fn()
                closure = self._closure_factory(batches)
                self.lbfgs.state = defaultdict(dict)  # drop stale curvature history

            self.lbfgs.step(closure)
            losses = closure.last_losses

            for key, val in losses.items():
                history[key].append(val.item())

            val = validate_fn(self.model) if validate_fn is not None else None
            if val is not None:
                history["val"].append(val)

            improved = val is not None and val < best_val
            if improved:
                best_val = val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if checkpoint_path and (improved or (it + 1) % save_every == 0 or it == k2_iters - 1):
                self._save(checkpoint_path, meta, "lbfgs", it + 1, dict(history), best_val, best_state)

            postfix = {k: f"{v.item():.3e}" for k, v in losses.items()}
            if val is not None:
                postfix["val"] = f"{val:.3e}"
            lbfgs_bar.set_postfix(postfix)
            if print_log and (it % log_every == 0 or it == k2_iters - 1):
                lbfgs_bar.write(f"[LBFGS {it:05d}] " +
                                 " ".join(f"{k}={v.item():.3e}" for k, v in losses.items()) +
                                 (f" val={val:.3e}" if val is not None else ""))

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model with val={best_val:.3e}")

        if checkpoint_path:
            final_stage = "lbfgs" if k2_iters > 0 else "adam"
            final_iter = k2_iters if k2_iters > 0 else k1_epochs
            self._save(checkpoint_path, meta, final_stage, final_iter, dict(history), best_val, best_state)

        return dict(history)