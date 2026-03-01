# sim/routing.py
import numpy as np
from collections import deque

INF = 10**9
NEIGHBORS_8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)

def compute_dist_to_exit_bfs(walkable, blocked, exit_cell):
    H, W = walkable.shape
    dist = np.full((H, W), INF, dtype=np.int32)

    er, ec = exit_cell
    if not walkable[er, ec]:
        return dist

    dist[er, ec] = 0
    q = deque([(er, ec)])

    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1

        # 8-neighborhood (with no-corner-cutting on diagonals)
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not walkable[nr, nc] or blocked[nr, nc]:
                    continue
                if dr != 0 and dc != 0:
                    if blocked[r + dr, c] or blocked[r, c + dc]:
                        continue
                if dist[nr, nc] > d:
                    dist[nr, nc] = d
                    q.append((nr, nc))

    return dist
