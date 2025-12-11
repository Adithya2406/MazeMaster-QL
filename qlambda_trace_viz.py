# qlambda_trace_viz.py
import numpy as np
import matplotlib.pyplot as plt
import imageio

def animate_q_lambda_trace(path, filename="q_lambda_trace.gif"):
    frames = []
    H = len(path)
    elig = np.zeros(H)

    for t in range(H):
        elig *= 0.85  # decay
        elig[t] = 1.0  # assign credit

        fig, ax = plt.subplots(figsize=(6,2))
        ax.bar(range(H), elig, color="blue")
        ax.set_ylim(0, 1)
        ax.set_title("Eligibility Trace Propagation")
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(filename, frames, fps=8)
