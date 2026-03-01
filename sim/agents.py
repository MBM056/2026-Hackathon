import numpy as np
from sim.routing import INF

NEIGHBORS_8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)

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
            "exit": None,
            "alerted": False,
        })
    return agents

def step_agents(agents, walkable, blocked, exits, stairs, dist_maps,
                doors=None, open_doors=None, fire_smoke=None,
                fire_mask=None, alarm_active=False, awareness_radius=6,
                pre_alert_wander_prob=0.20, evac_exits=None,
                fire_avoid_radius=2, avoid_smoke_cells=True):
    if doors is None:
        doors = set()
    if open_doors is None:
        open_doors = set()
    if evac_exits is None:
        evac_exits = exits
    planner_exits = list(dist_maps.keys())

    # Convert smoke field to a boolean mask
    smoke_mask = None
    if fire_smoke is not None:
        try:
            smoke_mask = fire_smoke if fire_smoke.dtype == np.bool_ else (fire_smoke > 0)
        except Exception:
            smoke_mask = fire_smoke

    new_agents = []
    occupied = set()
    H, W = walkable.shape

    for a in agents:
        r, c = a["pos"]
        if "alerted" not in a:
            a["alerted"] = False

        if alarm_active:
            a["alerted"] = True
        elif _hazard_visible(r, c, smoke_mask, fire_mask, H, W, awareness_radius):
            a["alerted"] = True

        # Open adjacent doors
        for dr, dc in NEIGHBORS_8:
            rr, cc = r + dr, c + dc
            if (rr, cc) in doors and (rr, cc) not in open_doors:
                open_doors.add((rr, cc))

        # Stay in place until alerted by local smoke/fire or building alarm.
        if not a["alerted"]:
            # Slow random wandering before alert.
            if np.random.random() < pre_alert_wander_prob:
                choices = []
                for dr, dc in NEIGHBORS_8:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not walkable[nr, nc]:
                        continue
                    if blocked[nr, nc]:
                        continue
                    if dr != 0 and dc != 0:
                        if blocked[r + dr, c] or blocked[r, c + dc]:
                            continue
                    if _is_dangerous_cell(
                        nr, nc, smoke_mask, fire_mask, H, W,
                        fire_avoid_radius=fire_avoid_radius,
                        avoid_smoke_cells=avoid_smoke_cells,
                    ):
                        continue
                    if (nr, nc) in evac_exits:
                        continue
                    choices.append((nr, nc))
                if choices:
                    candidate = choices[int(np.random.randint(0, len(choices)))]
                    if candidate not in occupied:
                        occupied.add(candidate)
                        a["pos"] = candidate
                    else:
                        occupied.add((r, c))
                else:
                    occupied.add((r, c))
            else:
                occupied.add((r, c))
            new_agents.append(a)
            continue

        # Reset exit if unreachable
        if a.get("exit") is not None and dist_maps[a["exit"]][r, c] >= INF:
            a["exit"] = None

        # Choose a reachable exit
        if a.get("exit") is None:
            reachable = [e for e in planner_exits if dist_maps[e][r, c] < INF]
            if not reachable:
                new_agents.append(a)
                continue
            a["exit"] = min(reachable, key=lambda e: dist_maps[e][r, c])

        dist = dist_maps[a["exit"]]
        current_is_dangerous = _is_dangerous_cell(
            r, c, smoke_mask, fire_mask, H, W,
            fire_avoid_radius=fire_avoid_radius,
            avoid_smoke_cells=avoid_smoke_cells,
        )
        if current_is_dangerous:
            best = None
            best_score = INF
        else:
            best = (r, c)
            best_score = dist[r, c]

        # Evaluate neighbors
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not walkable[nr, nc]:
                continue
            if blocked[nr, nc]:
                continue
            # No corner cutting when moving diagonally.
            if dr != 0 and dc != 0:
                if blocked[r + dr, c] or blocked[r, c + dc]:
                    continue
            if _is_dangerous_cell(
                nr, nc, smoke_mask, fire_mask, H, W,
                fire_avoid_radius=fire_avoid_radius,
                avoid_smoke_cells=avoid_smoke_cells,
            ):
                continue

            score = dist[nr, nc]

            # Stairs penalty
            if (nr, nc) in stairs:
                score += 6

            # Smoke penalty (discourage but allow)
            if smoke_mask is not None and smoke_mask[nr, nc]:
                score += 4

            if score < best_score:
                best = (nr, nc)
                best_score = score

        # Evacuated
        if best in evac_exits:
            continue

        # Collision avoidance
        if best is None:
            occupied.add((r, c))
        elif best not in occupied:
            occupied.add(best)
            a["pos"] = best

        new_agents.append(a)

    # Snap off stairs
    cleaned = []
    for a in new_agents:
        if a["pos"] in stairs:
            snapped = snap_to_nearest_floor(a["pos"], walkable, blocked, stairs, max_radius=6)
            if snapped is not None:
                a["pos"] = snapped
        cleaned.append(a)

    return cleaned


def _hazard_visible(r, c, smoke_mask, fire_mask, H, W, radius):
    if radius <= 0:
        return False

    r0 = max(0, r - radius)
    r1 = min(H, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(W, c + radius + 1)

    if smoke_mask is not None and np.any(smoke_mask[r0:r1, c0:c1]):
        return True
    if fire_mask is not None and np.any(fire_mask[r0:r1, c0:c1]):
        return True
    return False


def _is_dangerous_cell(r, c, smoke_mask, fire_mask, H, W, fire_avoid_radius, avoid_smoke_cells):
    if avoid_smoke_cells and smoke_mask is not None and smoke_mask[r, c]:
        return True
    if fire_mask is None or fire_avoid_radius < 0:
        return False

    r0 = max(0, r - fire_avoid_radius)
    r1 = min(H, r + fire_avoid_radius + 1)
    c0 = max(0, c - fire_avoid_radius)
    c1 = min(W, c + fire_avoid_radius + 1)
    return bool(np.any(fire_mask[r0:r1, c0:c1]))


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
