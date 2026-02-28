import argparse
import numpy as np

# This is the main entry point for the evacuation simulation.
# It loads the map, initializes the fire and agents, and runs the simulation loop,
# updating the fire and agents at each step, and rendering the video output.
from sim.map_loader import load_map_from_image
from sim.routing import compute_dist_to_exit_bfs
from sim.fire import FireModel
from sim.agents import spawn_agents, step_agents
from sim.renderer import VideoRenderer

# Constants
SECONDS_PER_STEP = 0.5


def main():
    # CLI argument parsing for map path, output video path, grid size, number of people, steps, fps, and fire start position.
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=140)
    ap.add_argument("--people", type=int, default=200)
    # We need to figure out the steps because peoples positions may vary and predefining steps may lead to incomplete evacuation in some cases. So we will run the simulation until all agents are evacuated or die.
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--fire", nargs=2, type=int, default=None)
    #NEW
    ap.add_argument("--door", nargs=2, type=int, action="append", default=[])
    args = ap.parse_args()

    # Load the map and detect walkable areas and exits.
    # walkable: where agents can move
    # exits: list of exit cell coordinates
    # stairs: set of stair cell coordinates (currently not used in the simulation logic but can be used for more complex pathfinding or fire spreading in the future)
    # base_img: the original image for rendering
    # args.map is the path which gets resized by args.grid x args.grid and then processed to determine walkable areas and exits.
    walkable, exits, stairs, doors_from_map, base_img = load_map_from_image(args.map, args.grid)
    # Check if any exits were detected, if not, exit with an error message.
    if len(exits) == 0:
        raise SystemExit("❌ No exits detected (use pure red pixels).")

    # H is the number of rows (height) and W is the number of columns (width) in the walkable grid.
    H, W = walkable.shape

    # This needs to be more randomized
    # Starts the fire at the specified location (default is the center of the map) and initializes the fire model. (We need to change this for the fire to start at a random location or multiple locations in the future.)
    fire_start = tuple(args.fire) if args.fire else (H // 2, W // 2)
    # Check if the fire start position is valid (walkable and not an exit).
    fire = FireModel(walkable, fire_start)

    # NEW
    doors = set(doors_from_map)
    for rc in args.door:
        r, c = int(rc[0]), int(rc[1])
        if 0 <= r < H and 0 <= c < W:
            doors.add((r, c))
        else:
            print(f"⚠️ Ignoring door out of bounds: {(r, c)}")

    open_doors = set()

    # Spawns agents at random positons on the map and not where the fire is. Each agent will have a position and an assigned exit that they will try to reach.
    agents = spawn_agents(walkable, fire.blocked, exits, args.people)

    # Initializes the video render to save the simulation. intakes the output path, fps, the original img, and a scale factor for rendering.
    renderer = VideoRenderer(args.out, args.fps, base_img, scale=4)
    print("[DEBUG] renderer created")

    #NEW
    def build_blocked():
        blocked = fire.blocked.copy()
        for d in doors:
            if d not in open_doors:
                blocked[d] = True
        return blocked

    blocked = build_blocked()

    dist_maps = {e: compute_dist_to_exit_bfs(walkable, blocked, e) for e in exits}

    prev_open_doors_count = len(open_doors)

    try:
        for t in range(args.steps):
            fire.update(t)

            # Rebuild blocked each step (fire grows)
            blocked = build_blocked()

            # IMPORTANT: Don't recompute every 2 steps for every exit until BFS is fast.
            # Recompute less often to avoid huge slowdown.
            if t % 10 == 0:
                for e in exits:
                    dist_maps[e] = compute_dist_to_exit_bfs(walkable, blocked, e)

            agents = step_agents(agents, walkable, blocked, exits, stairs, dist_maps, doors, open_doors)

            # If any door opened, recompute immediately (connectivity changed)
            if len(open_doors) != prev_open_doors_count:
                prev_open_doors_count = len(open_doors)
                blocked = build_blocked()
                dist_maps = {e: compute_dist_to_exit_bfs(walkable, blocked, e) for e in exits}

            renderer.write_frame(agents, exits, fire.blocked, time_seconds=t * SECONDS_PER_STEP)

            if not agents:
                print(f"✅ Evacuation complete at t={t * SECONDS_PER_STEP:.1f}s")
                break
    finally:
        renderer.close()

    print(f"🎥 Saved video to {args.out}")


if __name__ == "__main__":
    main()