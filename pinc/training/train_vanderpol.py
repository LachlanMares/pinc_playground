import torch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

from pinc.physics.vanderpol import VanDerPol
from pinc.datasets.vanderpol import VanDerPolDataset
from pinc.nn.mlp import MLP
from pinc.models.pinc import PINCModel
from pinc.core.trainer import Trainer
from pinc.losses.pinc_loss import PINCLoss
from pinc.evaluation.rollout import rollout


def simulate_true(physics, x0, steps=200, dt=0.05):
    traj = [x0]
    x = x0

    for _ in range(steps):
        u = torch.zeros(1)
        x = x + dt * physics.dynamics(x, u)
        traj.append(x)

    return torch.stack(traj)


def main():

    physics = VanDerPol()

    # -----------------------
    # Dataset
    # -----------------------
    dataset = VanDerPolDataset(physics)
    X, U, Y = dataset.generate()

    dataset = TensorDataset(X, U, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # -----------------------
    # Model
    # -----------------------
    model = PINCModel(
        backbone=MLP(in_dim=3, out_dim=2, hidden=64, depth=4),
        physics=physics,
        dt=0.05
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer)

    loss_fn = PINCLoss(physics)

    # -----------------------
    # Training loop
    # -----------------------
    loss_history = []

    for epoch in range(30):

        total_loss = 0.0

        for batch in dataloader:
            loss = loss_fn(model, batch)
            trainer.step(loss)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

    # -----------------------
    # Loss plot
    # -----------------------
    plt.figure()
    plt.plot(loss_history)
    plt.title("PINC Training Loss (Van der Pol)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    # -----------------------
    # Rollout evaluation
    # -----------------------
    x0 = X[0]

    pred_traj = rollout(model, physics, x0, steps=200, dt=0.05)
    true_traj = simulate_true(physics, x0, steps=200, dt=0.05)

    pred = pred_traj.detach().cpu().numpy()
    true = true_traj.detach().cpu().numpy()

    # -----------------------
    # Phase plot
    # -----------------------
    plt.figure()
    plt.plot(true[:, 0], true[:, 1], label="True")
    plt.plot(pred[:, 0], pred[:, 1], label="PINC")
    plt.legend()
    plt.title("Van der Pol Phase Space")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()