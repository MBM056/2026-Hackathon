from collections import deque
import numpy as np


class FireModel:
    def __init__(self, walkable, start, spread_every=3):
        self.walkable = walkable
        self.blocked = np.zeros_like(walkable, dtype=bool)
        self.front = deque()
        self.spread_every = spread_every

        r, c = start
        if walkable[r, c]:
            self.blocked[r, c] = True
            self.front.append((r, c))

    def update(self, t):
        if t % self.spread_every != 0:
            return

        for _ in range(len(self.front)):
            r, c = self.front.popleft()
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.blocked.shape[0] and 0 <= nc < self.blocked.shape[1]:
                    if self.walkable[nr, nc] and not self.blocked[nr, nc]:
                        self.blocked[nr, nc] = True
                        self.front.append((nr, nc))
            self.front.append((r, c))