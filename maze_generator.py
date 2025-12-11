import numpy as np
from collections import deque
from typing import Tuple, Optional


def _rng(seed=None):
    return np.random.default_rng(seed)


def generate_maze(side: int, difficulty="easy", seed=None):
    """
    Generate a square solvable maze using DFS + optional loops.

    difficulty:
        "easy" → loop_fraction=0.03
        "medium" → loop_fraction=0.08
    """

    loop_fraction = 0.03 if difficulty == "easy" else 0.08

    if side % 2 == 0:
        side += 1

    rng = _rng(seed)
    maze = np.ones((side, side), dtype=np.int8)

    stack = [(1, 1)]
    maze[1, 1] = 0
    dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

    while stack:
        r, c = stack[-1]
        candidates = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < side - 1 and 1 <= nc < side - 1:
                if maze[nr, nc] == 1:
                    candidates.append((nr, nc))

        if candidates:
            nr, nc = candidates[rng.integers(len(candidates))]
            maze[(r + nr)//2, (c + nc)//2] = 0
            maze[nr, nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    # Add loops
    for r in range(1, side - 1):
        for c in range(1, side - 1):
            if maze[r, c] == 1:
                if (maze[r - 1, c] == 0 and maze[r + 1, c] == 0) or \
                   (maze[r, c - 1] == 0 and maze[r, c + 1] == 0):
                    if rng.random() < loop_fraction:
                        maze[r, c] = 0

    maze[1, 1] = 0
    maze[-2, -2] = 0

    return maze


def shortest_path_length(maze, start, goal):
    from collections import deque
    R, C = maze.shape
    sr, sc = start
    gr, gc = goal
    if maze[sr, sc] == 1 or maze[gr, gc] == 1:
        return None

    q = deque([(sr, sc, 0)])
    vis = {(sr, sc)}
    moves = [(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        r, c, d = q.popleft()
        if (r, c) == (gr, gc):
            return d
        for dr, dc in moves:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C:
                if maze[nr, nc] == 0 and (nr, nc) not in vis:
                    vis.add((nr, nc))
                    q.append((nr, nc, d+1))
    return None
