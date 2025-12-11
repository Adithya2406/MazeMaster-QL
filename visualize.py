import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio


def plot_solution(maze, q, path, filename):
    H, W = maze.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(1 - maze, cmap="gray", origin="upper")
    xs = [c for r, c in path]
    ys = [r for r, c in path]
    ax.plot(xs, ys, c="red", lw=2)
    ax.scatter(1, 1, c="lime", s=80)          # start
    ax.scatter(W - 2, H - 2, c="blue", s=80)  # goal
    ax.set_title("Learned Path (Qλ)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def plot_value_heatmap(maze, q, filename):
    H, W = maze.shape
    vals = np.max(q, axis=2)
    masked = np.ma.masked_where(maze == 1, vals)
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")
    plt.imshow(masked, cmap=cmap, origin="upper")
    plt.title("State Value Heatmap (max Q)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_visitation_heatmap(maze, q, filename):
    H, W = maze.shape
    heat = np.abs(q).sum(axis=2)  # FIX HERE
    heat = np.log1p(heat)
    heat = np.ma.masked_where(maze == 1, heat)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heat, cmap="magma", origin="upper")
    fig.colorbar(im)
    ax.set_title("Visitation Heatmap (|Q| sum)")
    plt.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_training_curves(success, steps, returns, eps, filename):
    fig, axs = plt.subplots(3, 1, figsize=(7, 9))
    axs[0].plot(success)
    axs[0].set_title("Success rate")
    axs[1].plot(steps)
    axs[1].set_title("Steps")
    axs[2].plot(returns, label="Return")
    axs[2].plot(eps, label="Epsilon")
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def make_exploration_gif(maze, path, save_path):
    frames = []
    for step, (r, c) in enumerate(path):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(maze, cmap="gray_r")
        ax.scatter([c], [r], c="red", s=40)
        ax.set_title(f"Step {step}")
        ax.axis("off")
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf, dtype=np.uint8)
        frames.append(image)
        plt.close(fig)
    imageio.mimsave(save_path, frames, duration=0.05)


def plot_maze_3d(maze, path, filename, visited=None):
    """
    Render a static 3D view of the maze with walls, visited cells, and the learned path.
    - maze: 2D numpy array of the maze (1=wall, 0=free)
    - path: list of (r,c) coordinates representing the final path
    - visited: 2D boolean array of same shape as maze, True for visited cells
    """
    H, W = maze.shape
    if visited is None:
        visited = np.zeros_like(maze, dtype=bool)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # Draw floor as squares: visited vs unvisited
    floor_height = 0.01
    for r in range(H):
        for c in range(W):
            if maze[r, c] == 0:  # free cell
                color = (0.6, 0.8, 0.6) if visited[r, c] else (0.9, 0.9, 0.9)
                # Draw a thin column (using bar3d) to represent floor tile color
                ax.bar3d(c, r, 0, 1, 1, floor_height, color=color + (1.0,), shade=True)
    # Draw walls as 3D bars
    wall_height = 0.25
    rng = np.random.default_rng(0)
    for r in range(H):
        for c in range(W):
            if maze[r, c] == 1:
                wall_color = tuple(rng.uniform(0.2, 0.9, size=3)) + (1.0,)
                ax.bar3d(c, r, 0, 1, 1, wall_height, color=wall_color, shade=True)
    # Draw learned path as a red line (polyline through cell centers)
    if path is not None and len(path) > 1:
        xs = [c + 0.5 for (_, c) in path]
        ys = [r + 0.5 for (r, _) in path]
        zs = [floor_height + 0.02] * len(xs)
        ax.plot(xs, ys, zs, color="red", linewidth=3, label="RL Path")
    # Mark start and goal
    ax.scatter([1 + 0.5], [1 + 0.5], [wall_height + 0.05], color="lime", s=100, label="Start")
    ax.scatter([W - 2 + 0.5], [H - 2 + 0.5], [wall_height + 0.05], color="blue", s=100, label="Goal")
    ax.view_init(elev=60, azim=135)  # camera angle
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("3D Maze and Explored Path", fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)

def make_exploration_gif_3d(env, path, save_path):
    """
    Create an animation (GIF) of the agent's final path in the 3D maze.
    - env: MazeEnv (environment must be in a reset state at start)
    - path: list of (r,c) coordinates of the agent's trajectory (from start to goal)
    """
    frames = []
    # Reset environment to start position
    env.reset()
    # Action mapping for coordinate differences
    action_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    # Capture initial frame
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    # Step through the path
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        # Determine action from prev to curr
        move = (curr[0] - prev[0], curr[1] - prev[1])
        if move == (0, 0):
            # Agent hit a wall (state didn't change)
            # Just render again to show a pause
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            continue
        action = action_map.get(move)
        if action is None:
            continue  # skip if unexpected move
        env.step(action)
        frame = env.render(mode="rgb_array")
        frames.append(frame)
    # Save frames to GIF
    imageio.mimsave(save_path, frames, duration=0.1)
