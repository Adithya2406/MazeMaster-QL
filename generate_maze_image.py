import matplotlib.pyplot as plt
import numpy as np

def save_maze_image(maze, filename):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(1-maze, cmap="gray")
    ax.set_title("Maze")
    fig.savefig(filename, dpi=200)
    plt.close(fig)
