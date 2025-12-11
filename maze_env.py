import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeEnv(gym.Env):
    """
    FAST + STABLE GRID-BASED TRAINING ENVIRONMENT
    ------------------------------------------------
    • Used only for RL (training + evaluation)
    • Produces rgb_array frames for heatmaps/GIFs
    • No OpenGL dependencies here
    • 3D renderer is called separately in visualize_3d.py
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # ----------------------------------------------------
    # Initialization
    # ----------------------------------------------------
    def __init__(self, maze, render_mode=None):
        super().__init__()
        self.maze = np.array(maze, copy=True)
        self.H, self.W = self.maze.shape

        self.start = (1, 1)
        self.goal = (self.H - 2, self.W - 2)

        self.state = None
        self.done = False
        self.render_mode = render_mode

        # RL Action Space (Up / Down / Left / Right)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.H, self.W])

        # movement deltas
        self._moves = [
            (-1, 0),  # up
            (+1, 0),  # down
            (0, -1),  # left
            (0, +1),  # right
        ]

        # Visitation heatmap (for exploration GIF)
        self.visited_global = np.zeros((self.H, self.W), dtype=bool)

    # ----------------------------------------------------
    # Reset
    # ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.done = False

        # reset visitation
        self.visited_global[:, :] = False
        self.visited_global[self.start] = True

        return np.array(self.state, dtype=np.int32), {}

    # ----------------------------------------------------
    # Step
    # ----------------------------------------------------
    def step(self, action):
        if self.done:
            return np.array(self.state), 0.0, True, False, {}

        r, c = self.state
        dr, dc = self._moves[action]
        nr, nc = r + dr, c + dc

        # Wall collision = negative reward
        if self.maze[nr, nc] == 1:
            reward = -1.0
        else:
            self.state = (nr, nc)
            reward = -1.0
            self.visited_global[nr, nc] = True

        # Goal reached
        if self.state == self.goal:
            reward = 100.0
            self.done = True

        return np.array(self.state), reward, self.done, False, {}

    # ----------------------------------------------------
    # Lightweight 2D RGB Render (for GIF generation)
    # ----------------------------------------------------
    def render(self, mode="rgb_array"):
        """
        This provides a SIMPLE 2D visualization used internally during training.
        The high-quality 3D rendering is done in renderer_3d.py after training.
        """
        cell = 12
        img = np.ones((self.H * cell, self.W * cell, 3), dtype=np.uint8) * 255

        # draw walls
        for r in range(self.H):
            for c in range(self.W):
                if self.maze[r, c] == 1:
                    img[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = (30, 30, 30)

        # draw visited cells
        for r in range(self.H):
            for c in range(self.W):
                if self.visited_global[r, c]:
                    img[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = (180, 230, 180)

        # draw agent
        ar, ac = self.state
        img[ar*cell:(ar+1)*cell, ac*cell:(ac+1)*cell] = (255, 50, 50)

        # draw goal
        gr, gc = self.goal
        img[gr*cell:(gr+1)*cell, gc*cell:(gc+1)*cell] = (50, 80, 255)

        return img

    # ----------------------------------------------------
    def close(self):
        pass
