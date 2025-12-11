# train_3d.py
import numpy as np
from maze_env_3d_wrapper import MazeEnv3DWrapper
from renderer_3d import MazeRenderer3D
from exploration_recorder import ExplorationRecorder

def run_greedy_episode(maze, qtable):
    env = MazeEnv3DWrapper(maze)
    state = env.reset()

    while True:
        r, c = state
        action = np.argmax(qtable[r, c])
        state, reward, done = env.step(action)
        if done:
            break

    return env.get_visited_path(), env.goal

# -------------------------------------------------------------------

def main():
    maze = np.load("models/maze.npy")
    q = np.load("models/qtable.npy")

    # Run deterministic greedy policy and collect visited cells
    visited, goal_pos = run_greedy_episode(maze, q)

    # Build 3D renderer
    renderer = MazeRenderer3D(maze, width=1280, height=720)

    # Create recorder instance
    recorder = ExplorationRecorder(renderer)

    # Record cinematic top-down replay
    recorder.record_topdown(
        env=MazeEnv3DWrapper(maze),
        visited=visited,
        outfile="output/exploration3d_topdown.gif"
    )

    # Record first-person replay
    recorder.record_firstperson(
        env=MazeEnv3DWrapper(maze),
        visited=visited,
        outfile="output/exploration3d_fpv.gif"
    )

    print("\n✔ 3D Exploration GIFs saved in output/")
    print("   - exploration3d_topdown.gif")
    print("   - exploration3d_fpv.gif")

    renderer.close()

# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
