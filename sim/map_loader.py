import cv2
import numpy as np

# Convert input image into a grid map (also returns simulation data)
def load_map_from_image(path, grid):
    # Loads the image from the specified path
    img = cv2.imread(path)
    # Resizes the image to the specified grid size
    img = cv2.resize(img, (grid, grid), interpolation=cv2.INTER_AREA)
    # Splits the image into colors
    b, g, r = cv2.split(img)
    # Convert to grayscale img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Walkable areas are determined by the brightness of the pixels 
    walkable = gray > 200
    # Exits are detected by the presence of red pixels (pure red pixels indicate exits)
    exits_mask = (r > 200) & (g < 80) & (b < 80)
    # Makes a 3x3 kernel to dilate the exits so they are more visible and easier for agents to pathfind to. This also makes the exits slightly larger which is more realistic and allows for more agents to exit at once.
    kernel = np.ones((3, 3), np.uint8)
    exits_mask = cv2.dilate(exits_mask.astype(np.uint8), kernel) > 0
    # Marks exits as walkable
    walkable |= exits_mask
    # Gets coordinates of the exits 
    exits = list(zip(*np.where(exits_mask)))
    # Returns where is walkable, the exits, and the original image for rendering. 
    return walkable, exits, img