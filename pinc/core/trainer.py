import os

import torch

from pinc.utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """
    Implements Algorithm 1 of the paper: train with ADAM for K1 epochs,
    then refine with L-BFGS for K2 iterations, tracking the best model
    (lowest validation error) seen during training.

    Also supports periodic checkpointing and resuming a previously
    interrupted run (see `fit(..., checkpoint_path=..., resume=True)`).
    """

    def __init__(self, model, sampler, loss_fn,
                 n_boundary=1000, n_collocation=10000,
                 lr=1e-3, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.n_boundary = n_boundary
        self.n_collocation = n_collocation

        self.adam = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _sample_batches(self):
        t_b, y0_b, u_b, target_b = self.sampler.sample_boundary(self.n_boundary)
        t_c, y0_c, u_c = self.sampler.sample_collocation(self.n_collocation)

        device = self.device
        boundary = (t_b.to(device), y0_b.to(device), u_b.to(device), target_b.to(device))
        collocation = (t_c.to(device), y0_c.to(device), u_c.to(device))
        return boundary, collocation

    def _closure_factory(self, optimizer):
        def closure():
            optimizer.zero_grad()
            boundary, collocation = self._sample_batches()
            losses = self.loss_fn(self.model, boundary, collocation)
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

    def fit(self, k1_epochs=500, k2_iters=2000,
            validate_fn=None, log_every=50,
            checkpoint_path=None, meta=None, save_every=100, resume=False):
        """
        k1_epochs : number of ADAM epochs (each epoch = one fresh batch
                    of randomly sampled boundary/collocation points).
        k2_iters  : number of L-BFGS iterations.
        validate_fn : optional callable(model) -> float, used to track
                      the best network on a held-out validation trajectory
                      (Eq. 13), mirroring the "save network w with best
                      performance" step of Algorithm 1.
        checkpoint_path : if given, periodically saves a checkpoint here
                      (model + ADAM optimizer state + training progress),
                      overwriting the same file, so a killed/interrupted
                      run can be resumed.
        meta        : architecture dict (state_dim, control_dim, T,
                      hidden, depth) stored in the checkpoint so the
                      model can later be rebuilt via
                      `pinc.utils.checkpoint.load_pinc_model` without
                      needing to remember the construction arguments.
        save_every  : how often (in iterations) to write the checkpoint.
                      A checkpoint is also always written whenever the
                      validation error improves, and once more at the
                      very end of training.
        resume      : if True and `checkpoint_path` exists, restores
                      model weights, ADAM state, training history and
                      the ADAM/L-BFGS progress counter, and continues
                      training from there instead of starting over.
        """
        history = {"total": [], "data": [], "physics": [], "val": []}
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
            history = extra.get("history", history)
            best_val = extra.get("best_val", best_val)
            best_state = extra.get("best_model_state", best_state)
            start_stage = extra.get("stage", "adam")
            start_iter = extra.get("iter_in_stage", 0)

            print(f"Resumed from '{checkpoint_path}': stage={start_stage}, "
                  f"iter={start_iter}, best_val={best_val:.3e}")

        # ---- Stage 1: ADAM ----
        adam_start = start_iter if start_stage == "adam" else k1_epochs

        for epoch in range(adam_start, k1_epochs):
            boundary, collocation = self._sample_batches()

            self.adam.zero_grad()
            losses = self.loss_fn(self.model, boundary, collocation)
            losses["total"].backward()
            self.adam.step()

            history["total"].append(losses["total"].item())
            history["data"].append(losses["data"].item())
            history["physics"].append(losses["physics"].item())

            val = validate_fn(self.model) if validate_fn is not None else None
            history["val"].append(val)

            improved = val is not None and val < best_val
            if improved:
                best_val = val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if checkpoint_path and (improved or (epoch + 1) % save_every == 0 or epoch == k1_epochs - 1):
                self._save(checkpoint_path, meta, "adam", epoch + 1, history, best_val, best_state)

            if epoch % log_every == 0 or epoch == k1_epochs - 1:
                msg = (f"[ADAM {epoch:05d}] total={losses['total'].item():.3e} "
                       f"data={losses['data'].item():.3e} physics={losses['physics'].item():.3e}")
                if val is not None:
                    msg += f" val={val:.3e}"
                print(msg)

        # ---- Stage 2: L-BFGS ----
        lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=1,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
        closure = self._closure_factory(lbfgs)

        lbfgs_start = start_iter if start_stage == "lbfgs" else 0

        for it in range(lbfgs_start, k2_iters):
            lbfgs.step(closure)
            losses = closure.last_losses

            history["total"].append(losses["total"].item())
            history["data"].append(losses["data"].item())
            history["physics"].append(losses["physics"].item())

            val = validate_fn(self.model) if validate_fn is not None else None
            history["val"].append(val)

            improved = val is not None and val < best_val
            if improved:
                best_val = val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if checkpoint_path and (improved or (it + 1) % save_every == 0 or it == k2_iters - 1):
                self._save(checkpoint_path, meta, "lbfgs", it + 1, history, best_val, best_state)

            if it % log_every == 0 or it == k2_iters - 1:
                msg = (f"[LBFGS {it:05d}] total={losses['total'].item():.3e} "
                       f"data={losses['data'].item():.3e} physics={losses['physics'].item():.3e}")
                if val is not None:
                    msg += f" val={val:.3e}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model with val={best_val:.3e}")

        # Re-save once more so the checkpoint's model_state reflects the
        # best weights just loaded above (nice for later `--load-only`
        # style usage), while keeping the same terminal (stage, iter)
        # markers the per-iteration saves above already established --
        # this is what lets a *future* resume with a larger k1_epochs/
        # k2_iters correctly continue instead of redoing finished work.
        if checkpoint_path:
            final_stage = "lbfgs" if k2_iters > 0 else "adam"
            final_iter = k2_iters if k2_iters > 0 else k1_epochs
            self._save(checkpoint_path, meta, final_stage, final_iter, history, best_val, best_state)

        return history