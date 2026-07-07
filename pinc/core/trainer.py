import torch


class Trainer:
    """
    Implements Algorithm 1 of the paper: train with ADAM for K1 epochs,
    then refine with L-BFGS for K2 iterations, tracking the best model
    (lowest validation error) seen during the L-BFGS phase.
    """

    def __init__(self, model, sampler, loss_fn,
                 n_boundary=1000, n_collocation=10000,
                 lr=1e-3, device="cpu"):
        self.model = model.to(device)
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.n_boundary = n_boundary
        self.n_collocation = n_collocation
        self.device = device

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

    def fit(self, k1_epochs=500, k2_iters=2000, resample_every=1,
            validate_fn=None, log_every=50):
        """
        k1_epochs : number of ADAM epochs (each epoch = one fresh batch
                    of randomly sampled boundary/collocation points).
        k2_iters  : number of L-BFGS iterations.
        validate_fn : optional callable(model) -> float, used to track
                      the best network on a held-out validation trajectory
                      (Eq. 13), mirroring the "save network w with best
                      performance" step of Algorithm 1.
        """
        history = {"total": [], "data": [], "physics": [], "val": []}
        best_state = None
        best_val = float("inf")

        # ---- Stage 1: ADAM ----
        for epoch in range(k1_epochs):
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

            if val is not None and val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

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

        for it in range(k2_iters):
            lbfgs.step(closure)
            losses = closure.last_losses

            history["total"].append(losses["total"].item())
            history["data"].append(losses["data"].item())
            history["physics"].append(losses["physics"].item())

            val = validate_fn(self.model) if validate_fn is not None else None
            history["val"].append(val)

            if val is not None and val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if it % log_every == 0 or it == k2_iters - 1:
                msg = (f"[LBFGS {it:05d}] total={losses['total'].item():.3e} "
                       f"data={losses['data'].item():.3e} physics={losses['physics'].item():.3e}")
                if val is not None:
                    msg += f" val={val:.3e}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model with val={best_val:.3e}")

        return history
