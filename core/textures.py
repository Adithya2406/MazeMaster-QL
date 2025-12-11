# core/textures.py
import numpy as np
from PIL import Image
from noise import pnoise2


def make_stone_texture(size=128):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            n = pnoise2(x/40, y/40, octaves=4)
            g = int((n + 1) * 0.5 * 200)
            img[y, x] = [g, g, g]

    return Image.fromarray(img, "RGB")


def make_wood_texture(size=128):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        band = (y // 10) % 2
        color = 120 if band == 0 else 160
        img[y, :] = [color, color//2, 0]

    return Image.fromarray(img, "RGB")
