# train.py

import argparse
import os
from collections import deque

import numpy as np

from maze_generator import generate_maze
from maze_env_3d_wrapper import MazeEnv3DWrapper
from qlambda_agent import QLambdaAgent
from visualize import (
    plot_solution,
    plot_value_heatmap,
    plot_visitation_heatmap,
    plot_training_curves,
    plot_maze_3d,
)
from visualize_3d import make_exploration_gif_3d, make_exploration_gif_fp_tp

OUTPUT_DIR = "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Q(λ) RL on a 2D → 3D maze")
    parser.add_argument("--size", type=int, default=15, help="Maze size (NxN)")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Maze difficulty (affects wall density)",
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20000,
        help="Max steps per episode before giving up",
    )
    return parser.parse_args()


# ----------------------------------------------------------
# Simple BFS for optimal length + path (independent of maze_utils)
# ----------------------------------------------------------
def bfs_shortest_path_local(maze, start, goal):
    """
    Returns (length, path) using 4-connected BFS on grid.
    - length = number of steps (edges) from start to goal
    - path = list[(r, c)] including start and goal
    """
    H, W = maze.shape
    sr, sc = start
    gr, gc = goal

    q = deque()
    q.append((sr, sc))
    visited = np.zeros((H, W), dtype=bool)
    visited[sr, sc] = True

    parent = {start: None}

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            # reconstruct path
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            length = len(path) - 1
            return length, path

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < H
                and 0 <= nc < W
                and not visited[nr, nc]
                and maze[nr, nc] == 0
            ):
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    return None, []  # no path


# ----------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Generate maze and compute BFS baseline
    print(f"Generating maze of size {args.size} (difficulty={args.difficulty})")
    maze = generate_maze(args.size, difficulty=args.difficulty, seed=42)

    H, W = maze.shape
    start = (1, 1)
    goal = (H - 2, W - 2)

    optimal_len, optimal_path = bfs_shortest_path_local(maze, start, goal)
    if optimal_len is None:
        raise RuntimeError("Generated maze has no valid path from start to goal!")
    optimal_steps = optimal_len
    print(f"Optimal shortest path length (BFS): {optimal_steps}")

    # 2) RL setup
    env = MazeEnv3DWrapper(maze)
    agent = QLambdaAgent(H, W, actions=4)

    success_hist = []
    steps_hist = []
    return_hist = []
    epsilon_hist = []

    visit_counts = np.zeros((H, W), dtype=np.int32)

    # ε schedule similar to your earlier logs
    eps0 = 1.0
    eps_min = 0.05
    eps_decay = 0.993

    # 3) Training loop
    for ep in range(1, args.episodes + 1):
        epsilon = max(eps_min, eps0 * (eps_decay ** ep))

        state = env.reset()  # (r, c)
        state = tuple(state)

        agent.start_episode()
        action = agent.choose_action(state, epsilon)

        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < args.max_steps:
            steps += 1

            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)
            total_reward += reward

            r, c = next_state
            visit_counts[r, c] += 1

            if done:
                # terminal: no bootstrap
                agent.update(state, action, reward, None, None, True)
                state = next_state
                break

            next_action = agent.choose_action(next_state, epsilon)
            agent.update(state, action, reward, next_state, next_action, False)

            state, action = next_state, next_action

        success = int(done and state == goal)
        success_hist.append(success)
        steps_hist.append(steps)
        return_hist.append(total_reward)
        epsilon_hist.append(epsilon)

        if ep % 50 == 0:
            recent = np.mean(success_hist[-50:]) * 100.0
            print(
                f"Episode {ep}/{args.episodes} | "
                f"Success rate (last 50) = {recent:.1f}% | epsilon = {epsilon:.3f}"
            )

    print("Training complete.")

    # 4) Greedy rollout
    state = env.reset()
    state = tuple(state)
    greedy_path = [state]

    done = False
    steps = 0
    max_eval_steps = 10 * optimal_steps

    while not done and steps < max_eval_steps:
        steps += 1
        r, c = state
        action = int(np.argmax(agent.q[r, c]))
        next_state, reward, done = env.step(action)
        next_state = tuple(next_state)
        greedy_path.append(next_state)
        state = next_state

    solved = (state == goal) and done
    print(f"Solved maze? {solved}")
    print(f"Greedy steps = {len(greedy_path) - 1}  (optimal shortest = {optimal_steps})")

    # visited cells over training + greedy path
    visited_global = visit_counts > 0
    for (r, c) in greedy_path:
        visited_global[r, c] = True

    # 5) Plots + 3D GIFs
    value_fig = os.path.join(OUTPUT_DIR, "value_heatmap.png")
    visit_fig = os.path.join(OUTPUT_DIR, "visitation_heatmap.png")
    sol_fig = os.path.join(OUTPUT_DIR, "rl_solution.png")
    curves_fig = os.path.join(OUTPUT_DIR, "training_curves.png")
    maze3d_fig = os.path.join(OUTPUT_DIR, "maze_3d.png")

    plot_solution(maze, agent.q, greedy_path, sol_fig)
    plot_value_heatmap(maze, agent.q, value_fig)
    plot_visitation_heatmap(maze, agent.q, visit_fig)
    plot_training_curves(success_hist, steps_hist, return_hist, epsilon_hist, curves_fig)
    plot_maze_3d(maze, greedy_path, maze3d_fig)

    # 3D GIFs: topdown + first-person + combined
    make_exploration_gif_3d(maze, greedy_path, visited_global, OUTPUT_DIR)
    make_exploration_gif_fp_tp(maze, greedy_path, OUTPUT_DIR)
    print("All visualizations and 3D GIFs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
