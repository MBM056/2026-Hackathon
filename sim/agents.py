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
            "smoke_exposure_seconds": 0.0,
            "move_cooldown": 0,
        })
    return agents

def step_agents(agents, walkable, blocked, exits, stairs, dist_maps,
                doors=None, open_doors=None, fire_smoke=None,
                fire_mask=None, alarm_active=False, awareness_radius=6,
                pre_alert_wander_prob=0.20, evac_exits=None,
                fire_avoid_radius=2, avoid_smoke_cells=False,
                smoke_alert_threshold=0.12, smoke_lethal_time_seconds=35.0,
                smoke_harm_threshold=0.45, step_seconds=0.5,
                return_stats=False):
    if doors is None:
        doors = set()
    if open_doors is None:
        open_doors = set()
    if evac_exits is None:
        evac_exits = exits
    planner_exits = list(dist_maps.keys())

    # Smoke density field (float32 [0..1]) and derived visible mask.
    smoke_density = None
    smoke_mask = None
    if fire_smoke is not None:
        try:
            if fire_smoke.dtype == np.bool_:
                smoke_mask = fire_smoke
                smoke_density = fire_smoke.astype(np.float32)
            else:
                smoke_density = np.asarray(fire_smoke, dtype=np.float32)
                smoke_mask = smoke_density >= smoke_alert_threshold
        except Exception:
            smoke_mask = fire_smoke

    new_agents = []
    occupied = set()
    H, W = walkable.shape
    deaths_this_step = 0
    evacuated_this_step = 0

    for a in agents:
        r, c = a["pos"]
        if "alerted" not in a:
            a["alerted"] = False
        if "smoke_exposure_seconds" not in a:
            a["smoke_exposure_seconds"] = 0.0
        if "move_cooldown" not in a:
            a["move_cooldown"] = 0

        # Smoke can be deadly over prolonged exposure:
        # if inside smoke for >= smoke_lethal_time_seconds, agent dies.
        # Exposure timer resets immediately when agent escapes smoke.
        here_density = _smoke_density_at(smoke_density, r, c)
        if here_density >= smoke_harm_threshold:
            a["smoke_exposure_seconds"] += float(step_seconds)
        else:
            a["smoke_exposure_seconds"] = 0.0
        if a["smoke_exposure_seconds"] >= smoke_lethal_time_seconds:
            # Agent dies from prolonged smoke exposure.
            deaths_this_step += 1
            continue

        if alarm_active:
            a["alerted"] = True
        elif _hazard_visible(r, c, smoke_mask, fire_mask, H, W, awareness_radius):
            a["alerted"] = True

        # Open adjacent doors
        for dr, dc in NEIGHBORS_8:
            rr, cc = r + dr, c + dc
            if (rr, cc) in doors and (rr, cc) not in open_doors:
                open_doors.add((rr, cc))

        # Smoke slows movement while inside smoky air.
        if a["move_cooldown"] > 0:
            a["move_cooldown"] -= 1
            occupied.add((r, c))
            new_agents.append(a)
            continue

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

        # Continuously re-evaluate reachable exits so agents can adapt
        # instead of committing to one greedy choice.
        reachable = [e for e in planner_exits if dist_maps[e][r, c] < INF]
        if not reachable:
            closed_doors = list(set(doors) - set(open_doors))
            if closed_doors:
                best_local = None
                best_key = (INF, INF, INF)
                for dr, dc in NEIGHBORS_8:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not walkable[nr, nc] or blocked[nr, nc]:
                        continue
                    if dr != 0 and dc != 0:
                        if blocked[r + dr, c] or blocked[r, c + dc]:
                            continue

                    door_dist = min(abs(nr - dr0) + abs(nc - dc0) for dr0, dc0 in closed_doors)
                    smoke_pen = _smoke_density_at(smoke_density, nr, nc)
                    fire_pen = _fire_proximity_penalty(nr, nc, fire_mask, H, W)
                    key = (door_dist, smoke_pen, fire_pen)
                    if key < best_key:
                        best_local = (nr, nc)
                        best_key = key

                if best_local is not None and best_local not in occupied:
                    occupied.add(best_local)
                    a["pos"] = best_local
                else:
                    occupied.add((r, c))
            else:
                occupied.add((r, c))
            new_agents.append(a)
            continue

        current_dist = min(dist_maps[e][r, c] for e in reachable)

        # Distance-aware behavior tuning:
        # - far from exits: be greedier (distance dominates)
        # - mid-range: balanced risk-aware routing
        # - very close: commit hard toward exit unless heavily blocked
        if current_dist >= 35:
            risk_scale = 0.35
            dyn_fire_avoid_radius = max(0, fire_avoid_radius - 1)
            stay_penalty = 1.3
        elif current_dist <= 8:
            risk_scale = 0.2
            dyn_fire_avoid_radius = max(0, fire_avoid_radius - 2)
            stay_penalty = 0.0
        else:
            risk_scale = 1.0
            dyn_fire_avoid_radius = fire_avoid_radius
            stay_penalty = 0.2

        # Keep a preferred exit for traceability.
        a["exit"] = min(
            reachable,
            key=lambda e: dist_maps[e][r, c] + 0.4 * _fire_proximity_penalty(r, c, fire_mask, H, W),
        )

        # Very near an exit: prioritize aggressive progress toward any exit.
        if current_dist <= 8:
            best = (r, c)
            best_key = (current_dist, _smoke_density_at(smoke_density, r, c), 1 if (r, c) in stairs else 0)
            for dr, dc in NEIGHBORS_8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if not walkable[nr, nc] or blocked[nr, nc]:
                    continue
                if dr != 0 and dc != 0:
                    if blocked[r + dr, c] or blocked[r, c + dc]:
                        continue

                step_dist = min(dist_maps[e][nr, nc] for e in reachable)
                if step_dist >= INF:
                    continue

                smoke_here = _smoke_density_at(smoke_density, nr, nc)
                immensely_blocked = (
                    smoke_here >= 0.80
                    or _is_dangerous_cell(
                        nr, nc, smoke_mask, fire_mask, H, W,
                        fire_avoid_radius=max(3, fire_avoid_radius + 1),
                        avoid_smoke_cells=False,
                    )
                )
                if immensely_blocked and (nr, nc) not in evac_exits:
                    continue

                key = (step_dist, smoke_here, 1 if (nr, nc) in stairs else 0)
                if key < best_key:
                    best = (nr, nc)
                    best_key = key
        else:
            current_is_dangerous = _is_dangerous_cell(
                r, c, smoke_mask, fire_mask, H, W,
                fire_avoid_radius=dyn_fire_avoid_radius,
                avoid_smoke_cells=avoid_smoke_cells,
            )
            if current_is_dangerous:
                best = None
                best_score = INF
            else:
                best = (r, c)
                current_penalty = 0.0
                if smoke_density is not None:
                    current_penalty += 10.0 * float(smoke_density[r, c])
                current_penalty += _fire_proximity_penalty(r, c, fire_mask, H, W)
                best_score = current_dist + risk_scale * current_penalty + stay_penalty

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
                    fire_avoid_radius=dyn_fire_avoid_radius,
                    avoid_smoke_cells=avoid_smoke_cells,
                ):
                    continue

                step_dist = min(dist_maps[e][nr, nc] for e in reachable)
                if step_dist >= INF:
                    continue
                score = step_dist

                # Stairs penalty
                if (nr, nc) in stairs:
                    score += 6

                # Dense smoke is discouraged, but we still allow movement through it.
                if smoke_density is not None:
                    score += risk_scale * (10.0 * float(smoke_density[nr, nc]))

                # Penalty near active fire.
                score += risk_scale * _fire_proximity_penalty(nr, nc, fire_mask, H, W)

                if score < best_score:
                    best = (nr, nc)
                    best_score = score

        # Evacuated
        if best in evac_exits:
            evacuated_this_step += 1
            continue

        # Collision avoidance
        if best is None:
            occupied.add((r, c))
        elif best not in occupied:
            occupied.add(best)
            a["pos"] = best

        # Slowdown after standing/moving in smoky cells.
        rr, cc = a["pos"]
        dest_density = _smoke_density_at(smoke_density, rr, cc)
        if dest_density >= 0.15:
            # Slowdown, but never full immobilization.
            a["move_cooldown"] = max(a["move_cooldown"], int(dest_density >= 0.55))

        new_agents.append(a)

    # Snap off stairs
    cleaned = []
    for a in new_agents:
        if a["pos"] in stairs:
            snapped = snap_to_nearest_floor(a["pos"], walkable, blocked, stairs, max_radius=6)
            if snapped is not None:
                a["pos"] = snapped
        cleaned.append(a)

    if return_stats:
        return cleaned, {
            "deaths": deaths_this_step,
            "evacuated": evacuated_this_step,
        }
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


def _fire_proximity_penalty(r, c, fire_mask, H, W, max_radius=6):
    if fire_mask is None:
        return 0.0
    if fire_mask[r, c]:
        return 1000.0

    # Higher penalty when fire exists in tighter neighborhoods.
    for radius in range(1, max_radius + 1):
        r0 = max(0, r - radius)
        r1 = min(H, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(W, c + radius + 1)
        if np.any(fire_mask[r0:r1, c0:c1]):
            return 7.0 * (max_radius - radius + 1)
    return 0.0


def _smoke_density_at(smoke_density, r, c):
    if smoke_density is None:
        return 0.0
    return float(np.clip(smoke_density[r, c], 0.0, 1.0))


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
