# qlambda_agent.py

import numpy as np


class QLambdaAgent:
    """
    Q(λ) agent with eligibility traces.

    - Works with a grid of size H x W and a discrete action set of size A.
    - API is designed to match train.py:
        * QLambdaAgent(H, W, actions=4)
        * start_episode()
        * choose_action(state, epsilon)
        * update(state, action, reward, next_state, next_action, done)
        * .q holds the Q-table (H x W x A)
    """

    def __init__(
        self,
        H,
        W,
        actions,
        alpha=0.03,
        gamma=0.99,
        lam=0.80,
        beta=0.5,
        seed=None,
    ):
        """
        H, W   : maze height/width
        actions: number of discrete actions (e.g., 4)
        alpha  : learning rate
        gamma  : discount factor
        lam    : trace decay λ
        beta   : mix between Q-learning and SARSA for bootstrap target
                 (beta=1 → pure Q-learning, beta=0 → pure SARSA)
        """
        self.H = H
        self.W = W
        self.A = actions

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.beta = beta

        self.q = np.zeros((H, W, actions), dtype=np.float32)
        self.e = np.zeros_like(self.q)

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Episode lifecycle
    # --------------------------------------------------------
    def start_episode(self):
        """Reset eligibility traces at the start of each episode."""
        self.e.fill(0.0)

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------
    def choose_action(self, state, epsilon):
        """
        Epsilon-greedy over Q-table.

        state: (row, col)
        epsilon: exploration probability
        """
        r, c = state

        # random action with probability epsilon
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.A))

        # greedy with tie-breaking
        qs = self.q[r, c]
        max_q = np.max(qs)
        best_actions = np.flatnonzero(qs == max_q)
        return int(self.rng.choice(best_actions))

    # --------------------------------------------------------
    # Learning update
    # --------------------------------------------------------
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Perform a single Q(λ) update.

        - state, action, reward: s_t, a_t, r_{t+1}
        - next_state, next_action: s_{t+1}, a_{t+1} (for the SARSA component)
        - done: True if episode has terminated at s_{t+1} (no bootstrap)
        """
        r, c = state
        q_sa = self.q[r, c, action]

        if done or next_state is None:
            # terminal → no bootstrap
            td_target = reward
        else:
            nr, nc = next_state
            # SARSA and Q-learning components
            sarsa_val = self.q[nr, nc, next_action]
            qmax_val = np.max(self.q[nr, nc])

            # hybrid target
            td_target = reward + self.gamma * (
                self.beta * qmax_val + (1.0 - self.beta) * sarsa_val
            )

        delta = td_target - q_sa

        # eligibility trace update
        self.e *= (self.gamma * self.lam)
        self.e[r, c, action] += 1.0
        np.clip(self.e, -5.0, 5.0, out=self.e)

        # Q update with traces
        self.q += self.alpha * delta * self.e
        np.clip(self.q, -500.0, 500.0, out=self.q)


# ============================================================
# EpisodeResult + run_episode
# (used by evaluate.py)
# ============================================================

class EpisodeResult:
    def __init__(self, success, steps, total_reward, path):
        self.success = success
        self.steps = steps
        self.total_reward = total_reward
        self.path = path


def run_episode(env, q, epsilon=0.1, max_steps=20000, seed=None):
    """
    Run one episode in MazeEnv using a plain Q-table with Q(λ) updates.

    This matches the original evaluate.py expectations:

      from maze_env import MazeEnv
      from qlambda_agent import run_episode

      result = run_episode(env, q, epsilon=0.0, seed=seed)

    Parameters
    ----------
    env : MazeEnv-like environment
        Must implement:
          - obs, info = reset(seed=...)
          - obs2, reward, done, truncated, info = step(action)

    q : np.ndarray
        Q-table of shape (H, W, A)

    epsilon : float
        Exploration rate for epsilon-greedy policy.

    max_steps : int
        Safety cap on episode length.

    seed : int or None
        Random seed for action selection.

    Returns
    -------
    EpisodeResult
        success, steps, total_reward, path
    """
    rng = np.random.default_rng(seed)
    H, W, A = q.shape

    # ---- Episode init ----
    obs, _ = env.reset(seed=seed)
    # Ensure obs is a pair (row, col)
    state = tuple(obs)
    steps = 0
    total_reward = 0.0
    path = [state]

    alpha = 0.03
    gamma = 0.99
    lam = 0.80
    beta = 0.5

    elig = np.zeros_like(q)

    def choose_action(s):
        if rng.random() < epsilon:
            return int(rng.integers(A))
        r, c = s
        return int(np.argmax(q[r, c]))

    # initial action
    action = choose_action(state)

    while steps < max_steps:
        steps += 1

        obs2, reward, done, truncated, _ = env.step(action)
        next_state = tuple(obs2)
        total_reward += reward

        next_action = choose_action(next_state)

        # TD target (hybrid Q-learning + SARSA)
        r2, c2 = next_state
        sarsa_val = q[r2, c2, next_action]
        qmax_val = np.max(q[r2, c2])
        td_target = reward + gamma * (beta * qmax_val + (1.0 - beta) * sarsa_val)

        # TD error
        r, c = state
        td_err = td_target - q[r, c, action]

        # eligibility trace update
        elig *= (gamma * lam)
        elig[r, c, action] += 1.0
        np.clip(elig, -5, 5, out=elig)

        # Q update
        q += alpha * td_err * elig
        np.clip(q, -500, 500, out=q)

        path.append(next_state)

        if done:
            return EpisodeResult(True, steps, total_reward, path)

        state = next_state
        action = next_action

    # max_steps reached without success
    return EpisodeResult(False, steps, total_reward, path)
