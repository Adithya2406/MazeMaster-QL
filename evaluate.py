import argparse
import os
import numpy as np
from multiprocessing import Pool

from maze_env import MazeEnv
from qlambda_agent import run_episode
from maze_generator import shortest_path_length

MODELS_DIR = "models"

def _worker(args):
    maze, q, seed = args
    env = MazeEnv(maze)
    result = run_episode(env, q, epsilon=0.0, seed=seed)
    env.close()
    return result.success, result.steps, result.total_reward, len(result.path)

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()
    # Load trained maze and Q-table
    maze = np.load(os.path.join(MODELS_DIR, "maze.npy"))
    q = np.load(os.path.join(MODELS_DIR, "qtable.npy"))
    start = (1, 1)
    goal = (maze.shape[0] - 2, maze.shape[1] - 2)
    opt_len = shortest_path_length(maze, start, goal)
    print("Loaded trained model.")
    print("Maze shape:", maze.shape)
    print("Optimal BFS path length:", opt_len)
    # Prepare evaluation episodes
    seeds = [1000 + i for i in range(args.episodes)]
    args_list = [(maze, q, s) for s in seeds]
    if args.procs > 1:
        with Pool(args.procs) as p:
            results = p.map(_worker, args_list)
    else:
        results = [_worker(a) for a in args_list]
    results = np.array(results)
    succ = results[:, 0].astype(bool)
    steps = results[:, 1]
    ratios = results[:, 3] / opt_len
    success_rate = 100 * np.mean(succ)
    print(f"Success rate: {success_rate:.2f}% ({succ.sum()}/{len(succ)})")
    if succ.any():
        print(f"Mean path length ratio (solved episodes): {ratios[succ].mean():.3f}")
    
if __name__ == "__main__":
    evaluate()
