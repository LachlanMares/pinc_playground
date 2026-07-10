"""
Animated cart-pole demo: a schematic (cart + pole) panel on top, and a
control-input u(t) trace on the bottom with a moving marker showing
"where we are" in the top panel at every frame.

Deliberately kept independent of `pinc/visualization/plots.py` (which
holds static, non-animated figures) since animation has a different
enough dependency (matplotlib.animation, a Writer backend) and update
pattern (a per-frame callback) that bundling it in felt like it'd
clutter the static-plot file rather than share meaningful code with it.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_cartpole(y_traj, u_traj, T, physics,
                      out_path="cartpole_swingup.gif",
                      title="Cart-pole swing-up",
                      fps=None, stride=1, dpi=100, trail_len=40):
    """
    y_traj : (n_steps+1, 4) [x, x_dot, theta, theta_dot] closed-loop
             (or reference) trajectory, T seconds apart.
    u_traj : (n_steps, control_dim) control sequence applied at each
             step (only the first component is plotted -- cart-pole
             here always has control_dim=1).
    T      : seconds between consecutive rows of y_traj/u_traj -- used
             both to get the playback speed right (fps = 1/T by
             default, i.e. real-time) and to build the time axis for
             the control-input panel.
    physics: a `pinc.physics.cartpole.CartPole` instance -- only its
             `.L` (pole length) is used, for drawing the pole at the
             right physical scale relative to the cart's travel range.
    stride : plot every `stride`-th frame (use >1 to shrink the
             resulting GIF/runtime for long trajectories; playback
             speed is kept correct since `fps` is scaled accordingly).
    trail_len : number of past pole-tip positions to draw as a fading
             trail, purely a visual aid for seeing the swing shape.

    Saves a GIF to `out_path` (via Pillow -- no ffmpeg dependency) and
    also returns the underlying `FuncAnimation` object, in case the
    caller wants to display it inline (e.g. in a notebook) instead of
    / in addition to saving it.
    """
    y = y_traj.detach().cpu().numpy() if torch.is_tensor(y_traj) else np.asarray(y_traj)
    u = u_traj.detach().cpu().numpy() if torch.is_tensor(u_traj) else np.asarray(u_traj)
    u = u.reshape(-1)  # (n_steps,) -- control_dim=1 for cart-pole

    y = y[::stride]
    n_frames = len(y)

    if fps is None:
        fps = max(1, round(1.0 / (T * stride)))

    L = physics.L
    cart_w, cart_h = 0.3, 0.18
    x_all = y[:, 0]
    x_lo, x_hi = x_all.min() - L - 0.5, x_all.max() + L + 0.5

    t_axis = np.arange(len(u)) * T

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [2.2, 1]}
    )

    # ---- top panel: cart-pole schematic ----
    ax_top.set_xlim(x_lo, x_hi)
    ax_top.set_ylim(-L - 0.4, L + 0.4)
    ax_top.set_aspect("equal")
    ax_top.axhline(-cart_h / 2, color="dimgray", linewidth=1.5, zorder=0)
    ax_top.set_title(title)
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    cart_patch = patches.Rectangle((0, -cart_h / 2), cart_w, cart_h,
                                    facecolor="steelblue", edgecolor="black", zorder=3)
    ax_top.add_patch(cart_patch)
    pole_line, = ax_top.plot([], [], "o-", color="firebrick", linewidth=3,
                              markersize=6, zorder=4)
    trail_line, = ax_top.plot([], [], "-", color="firebrick", alpha=0.25, linewidth=1.5, zorder=2)
    time_text = ax_top.text(0.02, 0.92, "", transform=ax_top.transAxes)

    # ---- bottom panel: control input trace ----
    ax_bottom.plot(t_axis, u, color="tab:purple", linewidth=1.2, alpha=0.8)
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("control u (N)")
    ax_bottom.grid(True, alpha=0.3)
    time_marker = ax_bottom.axvline(0, color="black", linestyle="--", linewidth=1)
    u_dot, = ax_bottom.plot([], [], "o", color="tab:purple", markersize=6)

    tip_history_x, tip_history_y = [], []

    def init():
        pole_line.set_data([], [])
        trail_line.set_data([], [])
        time_text.set_text("")
        time_marker.set_xdata([0, 0])
        u_dot.set_data([], [])
        return cart_patch, pole_line, trail_line, time_text, time_marker, u_dot

    def update(frame):
        x, _, theta, _ = y[frame]
        pivot_x, pivot_y = x, cart_h / 2
        tip_x = pivot_x + L * np.sin(theta)
        tip_y = pivot_y + L * np.cos(theta)

        cart_patch.set_xy((x - cart_w / 2, -cart_h / 2))
        pole_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])

        tip_history_x.append(tip_x)
        tip_history_y.append(tip_y)
        if len(tip_history_x) > trail_len:
            del tip_history_x[0]
            del tip_history_y[0]
        trail_line.set_data(tip_history_x, tip_history_y)

        t_now = frame * stride * T
        time_text.set_text(f"t = {t_now:5.2f}s")

        time_marker.set_xdata([t_now, t_now])
        u_idx = min(frame * stride, len(u) - 1)
        u_dot.set_data([t_now], [u[u_idx]])

        return cart_patch, pole_line, trail_line, time_text, time_marker, u_dot

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                          blit=False, interval=1000.0 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Saved animation to '{out_path}' ({n_frames} frames @ {fps} fps)")
    return anim