import argparse
import numpy as np

from sim.map_loader import load_map_from_image
from sim.routing import compute_dist_to_exit_bfs
from sim.fire import FireModel
from sim.agents import spawn_agents, step_agents
from sim.renderer import VideoRenderer

SECONDS_PER_STEP = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=140)
    ap.add_argument("--people", type=int, default=200)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--fire", nargs=2, type=int, default=None)
    args = ap.parse_args()

    walkable, exits, base_img = load_map_from_image(args.map, args.grid)

    if len(exits) == 0:
        raise SystemExit("❌ No exits detected (use pure red pixels).")

    H, W = walkable.shape

    fire_start = tuple(args.fire) if args.fire else (H // 2, W // 2)
    fire = FireModel(walkable, fire_start)

    agents = spawn_agents(walkable, fire.blocked, exits, args.people)

    renderer = VideoRenderer(args.out, args.fps, base_img, scale=4)

    dist_maps = {}
    for e in exits:
        dist_maps[e] = compute_dist_to_exit_bfs(walkable, fire.blocked, e)

    for t in range(args.steps):
        fire.update(t)

        if t % 2 == 0:
            for e in exits:
                dist_maps[e] = compute_dist_to_exit_bfs(walkable, fire.blocked, e)

        agents = step_agents(agents, walkable, fire.blocked, exits, dist_maps)

        renderer.write_frame(
            agents,
            exits,
            fire.blocked,
            time_seconds=t * SECONDS_PER_STEP
        )

        if not agents:
            print(f"✅ Evacuation complete at t={t * SECONDS_PER_STEP:.1f}s")
            break

    renderer.close()
    print(f"🎥 Saved video to {args.out}")


if __name__ == "__main__":
    main()