from collections import deque
import numpy as np

INF = 10**9


def compute_dist_to_exit_bfs(walkable, blocked, exit_cell):
    H, W = walkable.shape
    dist = np.full((H, W), INF, dtype=np.int32)

    r0, c0 = exit_cell
    if blocked[r0, c0]:
        return dist

    dist[r0, c0] = 0
    q = deque([(r0, c0)])

    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if walkable[nr, nc] and not blocked[nr, nc]:
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
    return dist