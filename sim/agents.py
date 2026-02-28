import numpy as np


def spawn_agents(walkable, blocked, exits, n):
    candidates = [(r,c) for r in range(walkable.shape[0])
                         for c in range(walkable.shape[1])
                         if walkable[r,c] and not blocked[r,c]]
    rng = np.random.default_rng(0)
    agents = []

    for pos in rng.choice(len(candidates), size=min(n, len(candidates)), replace=False):
        agents.append({
            "pos": candidates[pos],
            "exit": None
        })
    return agents


def step_agents(agents, walkable, blocked, exits, dist_maps):
    new_agents = []
    occupied = set()

    for a in agents:
        r, c = a["pos"]

        if a["exit"] is None:
            a["exit"] = min(
                exits,
                key=lambda e: dist_maps[e][r, c]
            )

        dist = dist_maps[a["exit"]]

        best = (r, c)
        best_d = dist[r, c]

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < walkable.shape[0] and 0 <= nc < walkable.shape[1]:
                if walkable[nr,nc] and not blocked[nr,nc]:
                    if dist[nr,nc] < best_d:
                        best = (nr,nc)
                        best_d = dist[nr,nc]

        if best in exits:
            continue

        if best not in occupied:
            occupied.add(best)
            a["pos"] = best

        new_agents.append(a)

    return new_agents