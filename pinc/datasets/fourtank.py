from pinc.datasets.pinc_dataset import PINCSampler


def make_fourtank_sampler(physics, T, device="cpu"):
    """
    Sampling ranges roughly matching Sec. 4.2.2 of the paper: tank levels
    initial conditions in [2, 20] cm, control voltages u1, u2 in a
    plausible operating range for this benchmark (Johansson, 2000).

    device : generate the (large) training batches directly on this
             device -- avoids sampling on CPU and re-copying every
             single training iteration, which matters once batch sizes
             are large enough to make that copy non-negligible.

    Every training step, instead of loading real recorded data, the code just rolls random numbers within physically
    sensible ranges (tank levels 2–20 cm, pump voltages 0–5V) and asks: "if the real physics equations were applied to
    this random starting point, what would happen?" That's the entire "dataset" — a random-number generator plus the
    known physics equations, not stored data.

    What each sample represents, in plain terms:
    There are three kinds, matching the three "did the network get it right" checks in PINCLoss:

    A boundary sample — one random tank state y0 (e.g. [8, 12, 4, 5] cm) and one random pump setting u
    (e.g. [3.2, 1.1] volts). This sample's job is trivial: it asks the network "if no time has passed yet (t=0),
    what should the level be?" — the answer is just y0 itself, unchanged. It's a sanity check, not really "data" in
    the usual sense.

    A collocation sample — same kind of random (y0, u) pair, but paired with a random amount of elapsed time t somewhere
    between 0 and T seconds. This sample doesn't come with a "correct answer" at all — instead, the code checks whether
    the network's own output obeys the physics equations at that instant (does its predicted rate of change match what
    the real tank equations say it should be). This is the "physics" loss.

    A multistep sample (the one added not in the paper) — one random starting state y0, plus a short sequence of k
    random pump settings, one after another. This one does get a real answer to check against: the code actually
    simulates the true physics forward through those k steps (using RK4, a numerical integrator) to get a ground-truth
    trajectory, then compares the network's own chained predictions against it.

    Why make up data instead of using real recordings? Because the "correct answer" is just the known laws of physics
    for this tank system — you don't need real sensor recordings when you already have the equations. The random samples
    force the network to learn the physics everywhere in the operating range, not just along one specific recorded
    trajectory, which is the whole point of this "physics-informed" approach.
    """
    y_range = [(2.0, 20.0)] * 4
    u_range = [(0.0, 5.0)] * 2

    return PINCSampler(physics, T=T, y_range=y_range, u_range=u_range, device=device)