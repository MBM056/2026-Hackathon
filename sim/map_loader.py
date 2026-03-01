"""
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
"""
import cv2
import json
import numpy as np
from pathlib import Path


def load_map(path, grid):
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return load_map_from_semantic_json(path, grid)
    return load_map_from_image(path, grid)


def load_map_from_image(path, grid):
    img = cv2.imread(path)
    img = cv2.resize(img, (grid, grid), interpolation=cv2.INTER_AREA)

    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    walkable = gray > 200

    exits_mask = (r > 200) & (g < 80) & (b < 80)
    stairs_mask = (b > 200) & (g < 80) & (r < 80)

    # Doors = magenta. Use slightly relaxed thresholds to tolerate anti-aliased PNG edges.
    doors_mask = (r > 170) & (b > 170) & (g < 120)

    kernel = np.ones((3, 3), np.uint8)
    exits_mask = cv2.dilate(exits_mask.astype(np.uint8), kernel) > 0

    # Expand door marks into short segments so they behave like real openings.
    # Combine horizontal and vertical dilation to avoid tiny 1-cell doors.
    door_u8 = doors_mask.astype(np.uint8)
    door_u8 = cv2.dilate(door_u8, np.ones((1, 5), np.uint8), iterations=1)
    door_u8 = cv2.dilate(door_u8, np.ones((5, 1), np.uint8), iterations=1)
    door_u8 = cv2.morphologyEx(door_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    doors_mask = door_u8 > 0

    walkable |= exits_mask
    walkable |= stairs_mask
    walkable |= doors_mask   # doors are "walkable" *in principle* (but start closed)

    exits = list(zip(*np.where(exits_mask)))
    stairs = set(zip(*np.where(stairs_mask)))
    doors = set(zip(*np.where(doors_mask)))

    return walkable, exits, stairs, doors, img


def load_map_from_semantic_json(path, grid):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g = int(data.get("grid", grid))
    g = max(40, min(400, g))
    H = W = g

    walkable = _coords_to_mask(data.get("walkable", []), H, W)
    exits_mask = _coords_to_mask(data.get("exits", []), H, W)
    stairs_mask = _coords_to_mask(data.get("stairs", []), H, W)
    doors_mask = _coords_to_mask(data.get("doors", []), H, W)

    # Safety: exits/stairs/doors should always be traversable in principle.
    walkable |= exits_mask | stairs_mask | doors_mask

    exits = list(zip(*np.where(exits_mask)))
    stairs = set(zip(*np.where(stairs_mask)))
    doors = set(zip(*np.where(doors_mask)))

    base_img = np.zeros((H, W, 3), dtype=np.uint8)
    base_img[walkable] = (235, 235, 235)       # floor
    base_img[~walkable] = (35, 35, 35)         # walls
    base_img[doors_mask] = (255, 0, 255)       # magenta doors
    base_img[stairs_mask] = (255, 0, 0)        # blue stairs (BGR)
    base_img[exits_mask] = (0, 0, 255)         # red exits

    return walkable, exits, stairs, doors, base_img


def _coords_to_mask(coords, H, W):
    mask = np.zeros((H, W), dtype=bool)
    if not isinstance(coords, list):
        return mask
    for item in coords:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            r, c = int(item[0]), int(item[1])
        except Exception:
            continue
        if 0 <= r < H and 0 <= c < W:
            mask[r, c] = True
    return mask
