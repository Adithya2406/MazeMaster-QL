import numpy as np
from collections import deque


# ============================================================
# BFS to compute shortest path length
# ============================================================
def bfs_shortest_path_length(maze, start, goal):
    """
    Return length of shortest path from start→goal using BFS.
    maze: 2D numpy array (0=free, 1=wall)
    start, goal: (r, c)
    """
    H, W = maze.shape
    sr, sc = start
    gr, gc = goal

    q = deque()
    q.append((sr, sc, 0))
    visited = set()
    visited.add((sr, sc))

    moves = [(-1,0),(1,0),(0,-1),(0,1)]

    while q:
        r, c, d = q.popleft()

        if (r, c) == (gr, gc):
            return d

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if maze[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, d + 1))

    return None   # No path found


# ============================================================
# BFS to return the actual shortest path (list of coordinates)
# ============================================================
def bfs_shortest_path(maze, start, goal):
    """
    Returns actual shortest path as a list of (r,c) coordinates.
    """
    H, W = maze.shape
    sr, sc = start
    gr, gc = goal

    q = deque()
    q.append((sr, sc))
    parent = { (sr, sc): None }

    moves = [(-1,0),(1,0),(0,-1),(0,1)]

    while q:
        r, c = q.popleft()

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = []
            cur = (gr, gc)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if maze[nr, nc] == 0 and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))

    return None   # no path
