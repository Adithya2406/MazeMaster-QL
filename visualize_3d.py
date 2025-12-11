import os
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio


def _draw_3d_maze(ax, maze, visited, wall_alpha=0.15):
    """
    Draw a semi-transparent 3D maze with optionally highlighted visited cells.
    """
    H, W = maze.shape
    floor_h = 0.02
    wall_h = 0.4

    # Floor tiles
    for r in range(H):
        for c in range(W):
            if maze[r, c] == 0:
                if visited is not None and visited[r, c]:
                    color = (0.55, 0.85, 0.55, 1.0)  # soft green for explored
                else:
                    color = (0.9, 0.9, 0.9, 1.0)
                ax.bar3d(c, r, 0, 1, 1, floor_h, color=color, shade=False)

    # Walls (semi-transparent dark bars)
    for r in range(H):
        for c in range(W):
            if maze[r, c] == 1:
                color = (0.1, 0.1, 0.1, wall_alpha)
                ax.bar3d(c, r, 0, 1, 1, wall_h, color=color, shade=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, wall_h + 0.2)
    ax.view_init(elev=55, azim=135)
    ax.set_title("3D Maze and Explored Path")

def _draw_topdown_step(ax, maze, path, step_idx):
    """
    Top-down view: maze + full path + moving agent.
    """
    H, W = maze.shape
    ax.imshow(1 - maze, cmap="gray", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])

    if not path:
        return

    xs = [c for (r, c) in path]
    ys = [r for (r, c) in path]

    # Full path faint
    ax.plot(xs, ys, color="red", linewidth=1.5, alpha=0.25)

    # Path so far bright
    upto = step_idx + 1
    ax.plot(xs[:upto], ys[:upto], color="red", linewidth=2.5, alpha=1.0)

    # Start / goal
    ax.scatter([xs[0]], [ys[0]], c="lime", s=60)    # start
    ax.scatter([xs[-1]], [ys[-1]], c="cyan", s=60)  # goal

    # Agent
    ax.scatter([xs[step_idx]], [ys[step_idx]], c="yellow", s=60)

    ax.set_title(f"Top-down RL rollout — step {step_idx}")

# ================================================================
# TRUE FIRST-PERSON VIEW — FIXED
# ================================================================
def _draw_true_firstperson(ax, maze, path, step_idx):
    """
    First-person view from robot eye-level.
    Walls rendered only in front using perspective.
    """
    H, W = maze.shape
    r, c = path[step_idx]

    # Eye position
    cam_x = c + 0.5
    cam_y = H - (r + 0.5)
    cam_z = 0.35

    # Facing direction
    if step_idx == 0:
        ang = 0
    else:
        ang = _direction_angle(path[step_idx - 1], path[step_idx])

    ang_rad = math.radians(ang)
    dir_x = math.cos(ang_rad)
    dir_y = math.sin(ang_rad)

    wall_h = 1.2
    max_dist = 12  # render distance

    # Render walls in FOV
    for rr in range(H):
        for cc in range(W):
            if maze[rr, cc] != 1:
                continue

            wx0, wy0 = cc, H - rr
            wx1, wy1 = cc + 1, H - rr - 1

            # Check distance (skip far)
            cx = (wx0 + wx1) / 2
            cy = (wy0 + wy1) / 2
            if (cx - cam_x)**2 + (cy - cam_y)**2 > max_dist**2:
                continue

            # Quick check: wall in front?
            vx, vy = cx - cam_x, cy - cam_y
            dot = vx * dir_x + vy * dir_y
            if dot < 0:
                continue  # behind camera

            # Wall faces
            faces = [
                [(wx0, wy0, 0), (wx1, wy0, 0), (wx1, wy0, wall_h), (wx0, wy0, wall_h)],
                [(wx0, wy1, 0), (wx1, wy1, 0), (wx1, wy1, wall_h), (wx0, wy1, wall_h)],
                [(wx0, wy0, 0), (wx0, wy1, 0), (wx0, wy1, wall_h), (wx0, wy0, wall_h)],
                [(wx1, wy0, 0), (wx1, wy1, 0), (wx1, wy1, wall_h), (wx1, wy0, wall_h)],
            ]
            poly = Poly3DCollection(faces)
            poly.set_facecolor((0.15, 0.15, 0.15, 1))
            poly.set_edgecolor((0.3, 0.3, 0.3, 0.3))
            ax.add_collection3d(poly)

    # FLOOR
    ax.plot_surface(
        np.array([[0, W], [0, W]]),
        np.array([[0, 0], [H, H]]),
        np.zeros((2, 2)),
        color=(0.7, 0.7, 0.7, 1),
        alpha=1.0,
    )

    # Camera limits
    ax.set_xlim(cam_x - 4, cam_x + 4)
    ax.set_ylim(cam_y - 4, cam_y + 4)
    ax.set_zlim(0, wall_h)

    ax.view_init(elev=12, azim=ang)
    ax.set_axis_off()
    ax.set_title(f"First-Person — step {step_idx}")


# ================================================================
# THIRD-PERSON CHASE CAMERA (distance = 2.5)
# ================================================================
def _draw_chase_camera(ax, maze, path, step_idx, dist=2.5):
    """
    Third-person camera following behind the robot.
    """

    H, W = maze.shape
    r, c = path[step_idx]

    # Robot real coords
    rob_x = c + 0.5
    rob_y = H - (r + 0.5)
    rob_z = 0.5

    # Facing direction
    if step_idx == 0:
        ang = 0
    else:
        ang = _direction_angle(path[step_idx - 1], path[step_idx])

    ang_rad = math.radians(ang)
    dir_x = math.cos(ang_rad)
    dir_y = math.sin(ang_rad)

    # Camera is behind robot
    cam_x = rob_x - dist * dir_x
    cam_y = rob_y - dist * dir_y
    cam_z = 2.2

    # ========== DRAW MAZE WALLS ==========
    wall_h = 1.2
    for rr in range(H):
        for cc in range(W):
            if maze[rr, cc] != 1:
                continue

            wx0, wy0 = cc, H - rr
            wx1, wy1 = cc + 1, H - rr - 1

            faces = [
                [(wx0, wy0, 0), (wx1, wy0, 0), (wx1, wy0, wall_h), (wx0, wy0, wall_h)],
                [(wx0, wy1, 0), (wx1, wy1, 0), (wx1, wy1, wall_h), (wx0, wy1, wall_h)],
                [(wx0, wy0, 0), (wx0, wy1, 0), (wx0, wy1, wall_h), (wx0, wy0, wall_h)],
                [(wx1, wy0, 0), (wx1, wy1, 0), (wx1, wy1, wall_h), (wx1, wy0, wall_h)],
            ]
            poly = Poly3DCollection(faces)
            poly.set_facecolor((0.2, 0.2, 0.2, 1))
            poly.set_edgecolor((0.4, 0.4, 0.4, 0.3))
            ax.add_collection3d(poly)

    # DRAW PATH
    xs = [p[1] + 0.5 for p in path[:step_idx+1]]
    ys = [H - (p[0] + 0.5) for p in path[:step_idx+1]]
    zs = [0.51] * len(xs)
    ax.plot(xs, ys, zs, color="white", linewidth=3)

    # ROBOT (sphere)
    ax.scatter([rob_x], [rob_y], [rob_z], s=200, color="yellow")

    # Camera view
    ax.set_xlim(cam_x - 4, cam_x + 4)
    ax.set_ylim(cam_y - 4, cam_y + 4)
    ax.set_zlim(0, 2)

    ax.view_init(elev=22, azim=ang)
    ax.set_axis_off()
    ax.set_title(f"Third-Person — step {step_idx}")



def _figure_to_array(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.renderer.buffer_rgba())
    return buf[..., :3].copy()


# ================================================================
# Direction angle helper
# ================================================================
def _direction_angle(prev, curr):
    (r1, c1), (r2, c2) = prev, curr
    dr, dc = r2 - r1, c2 - c1
    if dc > 0:  # east
        return -90
    if dc < 0:  # west
        return 90
    if dr > 0:  # south
        return 180
    if dr < 0:  # north
        return 0
    return -60


def _draw_local_3d_firstperson(ax, maze, path, step_idx, radius=6):
    """
    Pseudo first-person 3D: crop a window around the agent and aim the camera
    along the direction of motion.
    """
    H, W = maze.shape
    r, c = path[step_idx]
    # crop window
    r0 = max(0, int(r - radius))
    r1 = min(H, int(r + radius + 1))
    c0 = max(0, int(c - radius))
    c1 = min(W, int(c + radius + 1))

    floor_h = 0.02
    wall_h = 0.4

    # draw local floor and walls
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if maze[rr, cc] == 0:
                color = (0.85, 0.85, 0.85, 1.0)
                ax.bar3d(cc, rr, 0, 1, 1, floor_h, color=color, shade=False)
            else:
                color = (0.1, 0.1, 0.1, 0.2)
                ax.bar3d(cc, rr, 0, 1, 1, wall_h, color=color, shade=False)

    # draw short path segment so far (only the points inside window)
    xs = []
    ys = []
    zs = []
    for (pr, pc) in path[: step_idx + 1]:
        if r0 <= pr < r1 and c0 <= pc < c1:
            xs.append(pc + 0.5)
            ys.append(pr + 0.5)
            zs.append(0.45)
    if xs:
        ax.plot(xs, ys, zs, color="red", linewidth=3)

    # agent marker
    ax.scatter([c + 0.5], [r + 0.5], [0.45], color="yellow", s=80)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(c0, c1)
    ax.set_ylim(r0, r1)
    ax.set_zlim(0, wall_h + 0.2)

    # Camera orientation
    if step_idx == 0:
        az = -60
    else:
        az = _direction_angle(path[step_idx - 1], path[step_idx])
    ax.view_init(elev=35, azim=az)
    ax.set_title(f"Third-person RL view — step {step_idx}")


def _figure_to_array(fig):
    """
    Convert a Matplotlib figure to an RGB numpy array in a backend-safe way.
    Works correctly on macOS/Retina (no manual reshape with width/height).
    """
    fig.canvas.draw()
    # renderer.buffer_rgba() already has shape (H, W, 4)
    buf = np.asarray(fig.canvas.renderer.buffer_rgba())
    # Drop alpha channel
    return buf[..., :3].copy()



def make_exploration_gif_3d(maze, path, visited_global, out_dir):
    """
    Generates two GIFs:
      1) exploration3d.gif  – side-by-side 3D maze + top-down rollout
      2) exploration3d_fp.gif – first-person pseudo 3D view
    """
    os.makedirs(out_dir, exist_ok=True)

    combo_path = os.path.join(out_dir, "exploration3d.gif")
    fp_path = os.path.join(out_dir, "exploration3d_fp.gif")

    H, W = maze.shape

    n_steps = len(path)
    if n_steps == 0:
        return

    # ------------------------------------------
    # 1) COMBINED GIF (3D maze + top-down)
    # ------------------------------------------
    combo_frames = []

    for step_idx in range(n_steps):
        fig = plt.figure(figsize=(12, 5))

        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)

        # --- 3D Maze ---
        _draw_3d_maze(ax3d, maze, visited_global)

        # --- WHITE PATH ---
        xs = [c + 0.5 for (r, c) in path[:step_idx+1]]
        ys = [r + 0.5 for (r, c) in path[:step_idx+1]]
        zs = [0.46] * len(xs)

        ax3d.plot(xs, ys, zs, color="red", linewidth=4.0)

        # Start & end markers
        ax3d.scatter(xs[0], ys[0], zs[0], color="yellow", s=70)
        ax3d.scatter(xs[-1], ys[-1], zs[-1], color="cyan", s=70)

        # --- Top-down ---
        _draw_topdown_step(ax2d, maze, path, step_idx)

        img = _figure_to_array(fig)
        combo_frames.append(img)
        plt.close(fig)

    imageio.mimsave(combo_path, combo_frames, fps=15)
    print("Saved:", combo_path)

    # ------------------------------------------
    # 2) FIRST-PERSON GIF
    # ------------------------------------------
    fp_frames = []

    for step_idx in range(n_steps):
        fig = plt.figure(figsize=(6, 6))
        ax_fp = fig.add_subplot(1, 1, 1, projection="3d")

        _draw_local_3d_firstperson(ax_fp, maze, path, step_idx)

        img = _figure_to_array(fig)
        fp_frames.append(img)
        plt.close(fig)

    imageio.mimsave(fp_path, fp_frames, fps=15)
    print("Saved:", fp_path)


def make_exploration_gif_fp_tp(maze, path, out_dir):
    out_path = os.path.join(out_dir, "exploration_fp_tp.gif")

    frames = []
    n = len(path)

    for i in range(n):
        fig = plt.figure(figsize=(12, 5))

        ax_fp = fig.add_subplot(1, 2, 1, projection="3d")
        ax_tp = fig.add_subplot(1, 2, 2, projection="3d")

        _draw_true_firstperson(ax_fp, maze, path, i)
        _draw_chase_camera(ax_tp, maze, path, i, dist=2.5)

        img = _figure_to_array(fig)
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=15)
    print("Saved:", out_path)
