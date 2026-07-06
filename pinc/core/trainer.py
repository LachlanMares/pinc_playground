

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, dataloader, loss_fn, epochs=10):

        for ep in range(epochs):
            total = 0.0

            for batch in dataloader:
                loss = loss_fn(self.model, batch)
                self.step(loss)
                total += loss.item()

            print(f"[Epoch {ep}] loss={total:.6f}")