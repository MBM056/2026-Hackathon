from collections import deque
import numpy as np

# This class models the fire spreading through the enviroment but this may need to be changed to account for different types of fires spreading.
# The fire currently starting at inputted positions is not realistic and needs to be changed to account for multiple starting points. 
class FireModel:
    # Initializes the fire model with the walkable grid, the current starting position, the rate is spreads at (we need to change this to account for different
    # types of fires spreading at different rates), and a queue to keep track of the fire front for spreading.
    def __init__(self, walkable, start, spread_every=3):
        # Creates a deep copy of the walkable grid to keep track of which cells are blocked by fire. Fire can only spread into walkable cells (which needs to be changed)
        self.walkable = walkable
        # Creates a boolean grid which is initially all False (which means its not blocked by fire)
        self.blocked = np.zeros_like(walkable, dtype=bool)
        # Current front of the fire to spread from (bfs style spreading)
        self.front = deque()
        # Spread speed of the fire (may change to be more stagnant based on the type of fire and realism)
        self.spread_every = spread_every
        
        # unpacks starting coordinates of the fire and checks if the starting point is good
        r, c = start
        # Checks if the path is walkable and not an exit (we need to change this to account for the fire starting in non walkable areas and spreading into walkable areas) and if it is we set it as blocked and add it to the fire front.
        if walkable[r, c]:
            self.blocked[r, c] = True
            self.front.append((r, c))

    # Updates the fire spreading
    def update(self, t):
        # Fire will only spread based off the tick speed. Using modulo ensures the fire only spreads on the correct tick speed.
        if t % self.spread_every != 0:
            return

        # Processes the frontier of the fire and its spread
        for _ in range(len(self.front)):
            r, c = self.front.popleft()
            # Look at 4 neighboring cells and if they are walkable and not already blocked by fire
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.blocked.shape[0] and 0 <= nc < self.blocked.shape[1]:
                    if self.walkable[nr, nc] and not self.blocked[nr, nc]:
                        # Mark as burnt and add to the frontier for the next spread
                        self.blocked[nr, nc] = True
                        self.front.append((nr, nc))
            # Append to the current list because we need the updated frontier for the next fire spread tick
            self.front.append((r, c))