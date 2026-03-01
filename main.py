import argparse
import numpy as np

from sim.map_loader import load_map
from sim.routing import compute_dist_to_exit_bfs
from sim.fire import FireModel
from sim.agents import spawn_agents, step_agents
from sim.renderer import VideoRenderer

SECONDS_PER_STEP = 0.5


def run_simulation(
    map_path,
    out_path,
    *,
    grid=140,
    people=200,
    steps=800,
    fps=12,
    render_every=2,
    alarm_at=None,
    awareness_radius=6,
    route_recompute_every=40,
    max_route_exits=12,
    fire=None,
    seed=None,
    doors_cli=None,
):
    if doors_cli is None:
        doors_cli = []
    # Load map (your loader returns 5 values)
    walkable, exits, stairs, doors_from_map, base_img = load_map(map_path, grid)
    if len(exits) == 0:
        raise SystemExit("No exits detected (check your exit color in the map).")

    H, W = walkable.shape

    # --- Fire start parsing (FIXED) ---
    rng = np.random.default_rng(seed)

    if fire is None:
        fire_start = (H // 2, W // 2)

    elif isinstance(fire, str) and fire.lower() == "random":
        # Pick a random walkable cell (not on exits)
        exits_set = set(exits)
        rs, cs = np.where(walkable)
        if len(rs) == 0:
            raise SystemExit("No walkable cells found in map.")
        for _ in range(5000):
            idx = int(rng.integers(0, len(rs)))
            r, c = int(rs[idx]), int(cs[idx])
            if (r, c) not in exits_set:
                fire_start = (r, c)
                break
        else:
            fire_start = (H // 2, W // 2)

    elif isinstance(fire, (list, tuple)) and len(fire) == 2:
        fire_start = (int(fire[0]), int(fire[1]))

    else:
        raise SystemExit("Use --fire random OR --fire <row> <col>")

    print("[FIRE] start:", fire_start)

    # Init fire
    fire = FireModel(walkable, fire_start)

    # --- Doors (map + CLI) ---
    doors = set(doors_from_map)
    for r, c in doors_cli:
        if 0 <= r < H and 0 <= c < W:
            doors.add((int(r), int(c)))
        else:
            print(f"[WARN] Ignoring door out of bounds: {(r, c)}")

    open_doors = set()

    # Agents
    agents = spawn_agents(walkable, fire.blocked, exits, people)
    initial_agents = len(agents)
    total_deaths = 0
    total_evacuated = 0

    # Renderer
    render_every = max(1, int(render_every))
    renderer = VideoRenderer(out_path, fps, base_img, scale=3, exits=exits)

    def build_blocked():
        blocked = fire.blocked.copy()  # burning OR collapsed (per your new FireModel)
        # Closed doors are blocked
        for d in doors:
            if d not in open_doors:
                blocked[d] = True
        return blocked

    # Initial routing
    blocked = build_blocked()
    route_exits = pick_route_exits(exits, max(1, int(max_route_exits)))
    print(f"[ROUTING] using {len(route_exits)} representative exits (from {len(exits)} total)")
    dist_maps = {e: compute_dist_to_exit_bfs(walkable, blocked, e) for e in route_exits}
    prev_open_doors_count = len(open_doors)
    alarm_active = False
    alarm_announced = False
    route_recompute_every = max(1, int(route_recompute_every))

    completed_t = None
    last_sim_time = 0.0
    survival_rate = 0.0
    try:
        for t in range(steps):
            fire.update(t)
            sim_time = t * SECONDS_PER_STEP
            last_sim_time = sim_time

            if alarm_at is not None and sim_time >= alarm_at:
                alarm_active = True
            if alarm_active and not alarm_announced:
                print(f"[ALARM] Fire alarm active at t={sim_time:.1f}s")
                alarm_announced = True

            blocked = build_blocked()

            # Recompute routing periodically (tune frequency)
            if t % route_recompute_every == 0:
                for e in route_exits:
                    dist_maps[e] = compute_dist_to_exit_bfs(walkable, blocked, e)

            # Step agents (smoke-aware)
            agents, step_stats = step_agents(
                agents,
                walkable,
                blocked,
                exits,
                stairs,
                dist_maps,
                doors,
                open_doors,
                fire_smoke=fire.smoke_field,
                fire_mask=fire.burning,
                alarm_active=alarm_active,
                awareness_radius=awareness_radius,
                evac_exits=exits,
                step_seconds=SECONDS_PER_STEP,
                return_stats=True,
            )
            total_deaths += int(step_stats.get("deaths", 0))
            total_evacuated += int(step_stats.get("evacuated", 0))

            # If any door opened, update routing immediately
            if len(open_doors) != prev_open_doors_count:
                prev_open_doors_count = len(open_doors)
                blocked = build_blocked()
                dist_maps = {e: compute_dist_to_exit_bfs(walkable, blocked, e) for e in route_exits}

            # Render (NOTE: this call assumes your renderer signature is still:
            # write_frame(agents, exits, fire_mask, time_seconds)
            # If you updated renderer to support smoke overlay, pass smoke_mask=fire.smoke instead.
            should_render = (t % render_every == 0)
            if should_render or not agents:
                renderer.write_frame(
                    agents,
                    exits,
                    fire.burning,
                    time_seconds=sim_time,
                    smoke_mask=fire.smoke_field,
                    alarm_active=alarm_active,
                    doors=doors,
                    open_doors=open_doors,
                )

            if not agents:
                print(f"Evacuation complete at t={t * SECONDS_PER_STEP:.1f}s")
                completed_t = sim_time
                break
        survivors = max(0, initial_agents - total_deaths)
        survival_rate = (100.0 * survivors / initial_agents) if initial_agents > 0 else 0.0
        # Show death/survival only at the end as a final summary frame.
        renderer.write_frame(
            agents,
            exits,
            fire.burning,
            time_seconds=last_sim_time,
            smoke_mask=fire.smoke_field,
            alarm_active=alarm_active,
            death_count=total_deaths,
            survival_rate=survival_rate,
            doors=doors,
            open_doors=open_doors,
        )
    finally:
        renderer.close()

    print(f"Saved video to {out_path}")
    return {
        "out": out_path,
        "completed_time_seconds": completed_t,
        "last_sim_time_seconds": last_sim_time,
        "alarm_at_seconds": alarm_at,
        "alarm_triggered": bool(alarm_announced),
        "initial_agents": initial_agents,
        "evacuated_count": int(total_evacuated),
        "death_count": int(total_deaths),
        "survival_rate": round(float(survival_rate), 2),
        "remaining_agents": len(agents),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=140)
    ap.add_argument("--people", type=int, default=200)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument(
        "--render-every",
        type=int,
        default=2,
        help="Render one frame every N simulation steps (higher = faster rendering)."
    )
    ap.add_argument(
        "--alarm-at",
        type=float,
        default=None,
        help="Activate building fire alarm at this simulation time in seconds."
    )
    ap.add_argument(
        "--awareness-radius",
        type=int,
        default=6,
        help="Cells around an agent used to detect nearby smoke/fire."
    )
    ap.add_argument(
        "--route-recompute-every",
        type=int,
        default=40,
        help="Recompute routing maps every N steps (higher = faster)."
    )
    ap.add_argument(
        "--max-route-exits",
        type=int,
        default=12,
        help="Max representative exits used for routing BFS (visual exits unchanged)."
    )
    ap.add_argument(
        "--fire",
        nargs="+",
        default=None,
        help="Fire start: 'random' OR two integers: row col"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (useful with --fire random for reproducible runs)"
    )
    ap.add_argument("--door", nargs=2, type=int, action="append", default=[])
    args = ap.parse_args()
    fire = None
    if args.fire is None:
        fire = None
    elif len(args.fire) == 1 and str(args.fire[0]).lower() == "random":
        fire = "random"
    else:
        fire = (int(args.fire[0]), int(args.fire[1]))

    run_simulation(
        args.map,
        args.out,
        grid=args.grid,
        people=args.people,
        steps=args.steps,
        fps=args.fps,
        render_every=args.render_every,
        alarm_at=args.alarm_at,
        awareness_radius=args.awareness_radius,
        route_recompute_every=args.route_recompute_every,
        max_route_exits=args.max_route_exits,
        fire=fire,
        seed=args.seed,
        doors_cli=args.door,
    )


def pick_route_exits(exits, max_route_exits):
    if len(exits) <= max_route_exits:
        return list(exits)

    # Deterministic downsampling across sorted exit coordinates.
    exits_sorted = sorted(exits)
    idx = np.linspace(0, len(exits_sorted) - 1, num=max_route_exits, dtype=int)
    return [exits_sorted[i] for i in idx]


if __name__ == "__main__":
    main()

