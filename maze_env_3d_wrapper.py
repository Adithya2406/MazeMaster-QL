# maze_env_3d_wrapper.py

import numpy as np


class MazeEnv3DWrapper:
    """
    Very simple wrapper around the 2D maze grid for Q-learning.
    It **does not** manage rendering; it just exposes a tiny RL API
    and records visited cells for 3D visualization.
    """

    def __init__(self, maze):
        self.maze = np.array(maze, copy=True)
        self.H, self.W = self.maze.shape

        self.start = (1, 1)
        self.goal = (self.H - 2, self.W - 2)

        self.reset()

    def reset(self):
        """Reset to start state and clear visited path. Returns state (r, c)."""
        self.state = self.start
        self.visited = [self.start]
        return self.state

    def step(self, action):
        """
        action: 0=up, 1=down, 2=left, 3=right
        Returns: (next_state, reward, done)
        """
        r, c = self.state
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        # Default
        reward = -1.0
        done = False

        # Invalid move → stay, small penalty, not done
        if (
            nr < 0
            or nr >= self.H
            or nc < 0
            or nc >= self.W
            or self.maze[nr, nc] == 1
        ):
            next_state = self.state
        else:
            # Valid move
            self.state = (nr, nc)
            next_state = self.state
            self.visited.append(self.state)

            if self.state == self.goal:
                reward = 100.0
                done = True

        return next_state, reward, done

    def get_visited_path(self):
        """
        Return visited path as list of (x, z) positions for the 3D renderer.
        (row, col) → (x, z)
        """
        return [(c, r) for r, c in self.visited]
