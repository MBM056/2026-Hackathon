import numpy as np
from sim.routing import INF

# Spawns in the agents at random walkable and unblocked locations on the map. Each agent is represented as a dictionary with its current position and assigned exit.
def spawn_agents(walkable, blocked, exits, n):
    # Generate a list of candidate positions for agents to spawn, which are walkable and not blocked by fire. We use a fixed random seed for reproducibility.
    candidates = [(r,c) for r in range(walkable.shape[0])
                         for c in range(walkable.shape[1])
                         if walkable[r,c] and not blocked[r,c]]
    # Randomly select agents positions from the candidates
    rng = np.random.default_rng(0)
    agents = []

    # We randomly select the posotions of the agents from the list of candidates 
    for pos in rng.choice(len(candidates), size=min(n, len(candidates)), replace=False):
        agents.append({
            "pos": candidates[pos],
            "exit": None
        })
    return agents

def step_agents(agents, walkable, blocked, exits, stairs, dist_maps, doors=None, open_doors=None):
    """
    doors: set[(r,c)] door cells
    open_doors: set[(r,c)] doors that have been opened
    blocked: numpy bool mask of blocked cells (fire + closed doors, etc.)
    """
    if doors is None:
        doors = set()
    if open_doors is None:
        open_doors = set()

    new_agents = []
    occupied = set()
    H, W = walkable.shape

    for a in agents:
        r, c = a["pos"]

        # --- NEW: open any adjacent door (even if the door cell itself is blocked/closed) ---
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in doors and (nr, nc) not in open_doors:
                open_doors.add((nr, nc))

        # Drop exit assignment if it became unreachable
        if a["exit"] is not None and dist_maps[a["exit"]][r, c] >= INF:
            a["exit"] = None

        # Assign nearest reachable exit
        if a["exit"] is None:
            reachable = [e for e in exits if dist_maps[e][r, c] < INF]
            if not reachable:
                # No reachable exit from here (for now) -> stay put
                new_agents.append(a)
                continue
            a["exit"] = min(reachable, key=lambda e: dist_maps[e][r, c])

        dist = dist_maps[a["exit"]]

        best = (r, c)
        best_score = dist[r, c]

        # Move to neighbor with strictly better distance (stairs penalized)
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if walkable[nr, nc] and not blocked[nr, nc]:
                    score = dist[nr, nc]
                    if (nr, nc) in stairs:
                        score += 6
                    if score < best_score:
                        best = (nr, nc)
                        best_score = score

        # Evacuated
        if best in exits:
            continue

        # Avoid collisions
        if best not in occupied:
            occupied.add(best)
            a["pos"] = best

        new_agents.append(a)

    # Stair cleanup: try snapping agents off stairs onto nearby floor
    cleaned = []
    for a in new_agents:
        if a["pos"] in stairs:
            snapped = snap_to_nearest_floor(a["pos"], walkable, blocked, stairs)
            if snapped is not None:
                a["pos"] = snapped
            cleaned.append(a)
        else:
            cleaned.append(a)

    return cleaned


def snap_to_nearest_floor(pos, walkable, blocked, stairs, max_radius=6):
    r, c = pos
    H, W = walkable.shape

    if walkable[r, c] and not blocked[r, c] and (r, c) not in stairs:
        return pos

    for d in range(1, max_radius + 1):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < H
                    and 0 <= nc < W
                    and walkable[nr, nc]
                    and not blocked[nr, nc]
                    and (nr, nc) not in stairs
                ):
                    return (nr, nc)
    return None
