import numpy as np

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

# Updates the agents position by stepping based on the input provided. Intakes the agents, the walkable paths, the block paths, the blocked exits and the distance maps to the exits. The agents will move towards the exit that is closest to them based on the distance maps, while avoiding blocked paths and fire.
def step_agents(agents, walkable, blocked, exits, dist_maps):
    new_agents = []
    occupied = set()

    # Assigns the nearest exit to each agent if they don't have one already. 
    # Loops through all agents
    for a in agents:
        # Gets agents current row and column position
        r, c = a["pos"]
        # If they aren't assigned a exit, we assign them the nearest exit based on the distance maps.
        if a["exit"] is None:
            # Checks every e exit and looks at the agents r and c position relative to the exits using the exits distance.
            a["exit"] = min(
                exits,
                key=lambda e: dist_maps[e][r, c]
            )
        # Gets agents current distance to their assigned exit from the distance maps.
        dist = dist_maps[a["exit"]]

        # calculates the best next position for the agent to move to by looking at the neighboring cells (up, down, left, right) and choosing the one with the smallest distance to the exit that is also walkable and not blocked by fire. 
        # If the best next position is an exit, we consider the agent evacuated and do not add it to the new_agents list. If the best next position is occupied by another agent, we keep the agent in its current position.
        best = (r, c)
        best_d = dist[r, c]

        # Looping over the 4 neighbors cells checking for next best move
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < walkable.shape[0] and 0 <= nc < walkable.shape[1]:
                if walkable[nr,nc] and not blocked[nr,nc]:
                    if dist[nr,nc] < best_d:
                        best = (nr,nc)
                        best_d = dist[nr,nc]
        # If agent reaches exit they are removed from the simulation and not added to the new_agents list.
        if best in exits:
            continue

        # To avoid agent collision, the agent reserves that cell in occupied and updates the agents position to the best cell. 
        if best not in occupied:
            occupied.add(best)
            a["pos"] = best
        # If not exited, not blocked, and not occupied, we add the agent to the new_agents list to be processed in the next step.
        new_agents.append(a)
    # Returns the list of the agents that have not exited after the movement. 
    return new_agents