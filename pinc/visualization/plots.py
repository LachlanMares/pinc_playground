import matplotlib.pyplot as plt


def plot_trajectory(traj):
    traj = traj.detach().cpu().numpy()

    plt.plot(traj[:, 0], traj[:, 1])
    plt.title("Van der Pol Trajectory (PINC)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()