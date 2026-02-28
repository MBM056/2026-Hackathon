import cv2
import numpy as np


def load_map_from_image(path, grid):
    img = cv2.imread(path)
    img = cv2.resize(img, (grid, grid), interpolation=cv2.INTER_AREA)

    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    walkable = gray > 200

    exits_mask = (r > 200) & (g < 80) & (b < 80)

    # 🔥 CRITICAL FIX: dilate exits
    kernel = np.ones((3, 3), np.uint8)
    exits_mask = cv2.dilate(exits_mask.astype(np.uint8), kernel) > 0

    walkable |= exits_mask

    exits = list(zip(*np.where(exits_mask)))
    return walkable, exits, img